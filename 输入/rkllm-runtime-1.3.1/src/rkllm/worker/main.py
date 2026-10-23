"""worker 进程主循环：单读线程分发 + 单推理线程串行执行。

并发模型（这是对的那个）：
- 一个 socket 只允许一个读线程（两个 makefile("rb") 会抢字节流——run 与 abort
  帧的归属无法预期，这是本文件初版踩过的坑）；
- 读线程把 run 帧派发到推理线程（队列深度 1，满了立即回 busy error），
  abort/ping/shutdown 由读线程直接处理；
- 写回统一走 _send()（锁保护）：推理线程发 token/end/error，读线程发 state，
  互不拆帧。

用法（由 supervisor 拉起）：
  python -m rkllm.worker.main --address /run/rkllm/worker.sock \
      --model /path/to.rkllm --engine real --context 4096 ...
"""

from __future__ import annotations

import argparse
import os
import queue
import socket
import sys
import threading

from .frames import (OP_ABORT, OP_END, OP_ERROR, OP_PING, OP_READY, OP_RUN,
                     OP_SHUTDOWN, OP_STATE, OP_TOKEN, PROTOCOL_VERSION,
                     FrameError, read_frame, write_frame)


class Worker:
    def __init__(self, engine):
        self.engine = engine
        self._abort_requested = threading.Event()
        self._write_lock = threading.Lock()
        self._run_queue: queue.Queue = queue.Queue(maxsize=1)
        self._shutdown = threading.Event()      # 进程级：仅 shutdown 帧触发
        self._conn_closed = threading.Event()   # 连接级：EOF 触发，停推理等待

    # -- 写回（锁保护） -----------------------------------------------------

    def _send(self, wfile, frame: dict) -> None:
        with self._write_lock:
            write_frame(wfile, frame)

    # -- 推理线程 -----------------------------------------------------------

    def _inference_loop(self, wfile) -> None:
        while not self._shutdown.is_set():
            try:
                # 0.02s 轮询：0.5s 的粒度会给每个请求叠加最多 0.5s
                # 首帧延迟（真机 TTFT 实测暴露），而空转成本可忽略。
                frame = self._run_queue.get(timeout=0.02)
            except queue.Empty:
                if self._conn_closed.is_set():
                    return
                continue
            req_id = frame.get("id", "")
            self._abort_requested.clear()
            try:
                params = frame.get("params") or {}
                for token in self.engine.stream(
                        frame.get("prompt", ""),
                        role=params.get("role", "user"),
                        enable_thinking=bool(params.get("enable_thinking", False)),
                        sampling=params.get("sampling"),
                        max_new_tokens=params.get("max_new_tokens"),
                        cache_hint=bool(params.get("cache_hint", True))):
                    if self._abort_requested.is_set():
                        self._send(wfile, {"op": OP_END, "id": req_id,
                                           "aborted": True})
                        break
                    self._send(wfile, {"op": OP_TOKEN, "id": req_id,
                                       "text": token})
                else:
                    perf = self.engine.last_perf()
                    body: dict = {"op": OP_END, "id": req_id}
                    if perf:
                        body["perf"] = {"prefill_tokens": perf[0],
                                        "generate_tokens": perf[1]}
                    self._send(wfile, body)
            except Exception as exc:             # 推理失败不拖垮 worker
                self._send(wfile, {"op": OP_ERROR, "id": req_id,
                                   "message": str(exc)[:500]})

    # -- 读线程（唯一读方） ---------------------------------------------------

    def _reader_loop(self, conn, wfile) -> None:
        rfile = conn.makefile("rb")
        try:
            while not self._shutdown.is_set():
                frame = read_frame(rfile)
                if not frame:
                    break                        # EOF
                op = frame.get("op")
                if op == OP_RUN:
                    try:
                        self._run_queue.put_nowait(frame)
                    except queue.Full:
                        self._send(wfile, {"op": OP_ERROR,
                                           "id": frame.get("id", ""),
                                           "message": "busy",
                                           "busy": True})
                elif op == OP_ABORT:
                    self._abort_requested.set()
                    try:
                        self.engine.abort()
                    except Exception:
                        pass
                elif op == OP_PING:
                    self._send(wfile, {"op": OP_STATE, "state": "ready"})
                elif op == OP_SHUTDOWN:
                    self._shutdown.set()
                    break
        except (FrameError, OSError, ValueError):
            pass                                 # 连接断开：正常退出路径
        finally:
            self._conn_closed.set()              # 仅停本连接的推理等待

    # -- 服务循环 -----------------------------------------------------------

    def serve_connection(self, conn) -> None:
        wfile = conn.makefile("wb")
        self._conn_closed.clear()
        inference = threading.Thread(target=self._inference_loop, args=(wfile,),
                                     daemon=True)
        inference.start()
        self._reader_loop(conn, wfile)           # 读循环跑在当前线程
        inference.join(timeout=5)
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.close()

    def serve_forever(self, address: str, *, tcp: bool = False) -> None:
        if tcp:
            host, _, port = address.rpartition(":")
            srv = socket.create_server((host or "127.0.0.1", int(port)))
        else:
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                os.unlink(address)
            except FileNotFoundError:
                pass
            srv.bind(address)
        srv.listen(1)
        first = True
        while not self._shutdown.is_set():
            conn, _ = srv.accept()
            wfile = conn.makefile("wb")
            if first:
                # ready 帧走 UDS 连接本身，不走 stdout：librkllmrt 的 C 库日志
                # 会污染 stdout 管道（真机首测：帧头被日志字节覆盖，supervisor
                # 解析出 1.2 GB 假长度后永久阻塞）。stdout 从此只承载 C 库日志，
                # 由 supervisor 的排水线程丢弃。
                ready = {"op": OP_READY, "v": PROTOCOL_VERSION}
                fingerprint = getattr(self.engine, "lib_fingerprint", None)
                if fingerprint:
                    # D10：库指纹由真正 dlopen 的 worker 提供，server 不重复加载
                    ready.update({"lib_sha256": fingerprint.get("sha256"),
                                  "lib_size": fingerprint.get("size"),
                                  "lib_path": fingerprint.get("path")})
                self._send(wfile, ready)
                first = False
            self.serve_connection(conn)


def build_engine(args):
    """按参数构造引擎；fake 引擎零依赖，real 引擎仅板端可加载。"""
    if args.engine == "fake":
        from rkllm.worker.fake import FakeEngine
        return FakeEngine(model_name=args.model or "fake",
                          context_len=args.context, delay=args.delay)
    from rkllm.worker.engine import RealEngine
    from rkllm.worker.promptcache import PromptCache
    cache = None
    if getattr(args, "cache_dir", "") and not getattr(args, "no_cache", False):
        cache = PromptCache(args.cache_dir,
                            max_mb=getattr(args, "cache_max_mb", 512),
                            ttl_s=getattr(args, "cache_ttl_s", 604800))
    return RealEngine(model_path=args.model, platform=args.platform,
                      context_len=args.context, max_new_tokens=args.max_tokens,
                      keep_history=not args.no_history, threads=args.threads,
                      prompt_cache=cache)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rkllm-worker")
    parser.add_argument("--address", required=True,
                        help="unix socket 路径或 host:port（--tcp）")
    parser.add_argument("--tcp", action="store_true",
                        help="用 TCP 而非 unix socket（测试/跨机）")
    parser.add_argument("--engine", choices=("real", "fake"), default="real")
    parser.add_argument("--model", default="")
    parser.add_argument("--platform", default="rk3588")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--cache-dir", default="",
                        help="prompt 前缀缓存目录；空=关闭（cache.enabled=false）")
    parser.add_argument("--cache-max-mb", type=int, default=512)
    parser.add_argument("--cache-ttl-s", type=int, default=604800)
    parser.add_argument("--delay", type=float, default=0.0,
                        help="fake 引擎每 token 延迟（压测/演示）")
    args = parser.parse_args(argv)
    # P2：命令行范围校验，避免 --context=-1 / --max-tokens=0 直达 C 结构体
    if args.context <= 0 or args.max_tokens <= 0 or args.threads <= 0:
        parser.error("--context/--max-tokens/--threads 必须为正整数")
    engine = build_engine(args)
    try:
        Worker(engine).serve_forever(args.address, tcp=args.tcp)
    finally:
        # P1-7：任何退出路径都释放 NPU / dma-buf（与 cli / LocalSupervisor 对齐）
        try:
            engine.destroy()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
