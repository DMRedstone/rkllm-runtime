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
import logging
import os
import queue
import socket
import threading
import time

from .frames import (OP_ABORT, OP_END, OP_ERROR, OP_HEARTBEAT, OP_PING,
                     OP_READY, OP_RUN, OP_SHUTDOWN, OP_STATE, OP_TOKEN,
                     PROTOCOL_VERSION, FrameError, read_frame, write_frame)


class Worker:
    def __init__(self, engine, heartbeat_interval_s: float = 5.0,
                 heartbeat_grace_s: float = 120.0):
        self.engine = engine
        self.heartbeat_interval_s = max(1.0, float(heartbeat_interval_s))
        # 心跳续命的「prefill 宽限期」：首 token 之前没有 token 进度可观测
        # （rkllm 同步 C API 不暴露 prefill 进度），而 27B 4k 前缀 prefill 实测
        # 20-60 s，需要留出足够余量；超过宽限期仍无进度就不再发心跳，让
        # supervisor 的 engine.heartbeat_timeout_s 生效并杀 worker（P1-1）。
        self.heartbeat_grace_s = max(0.0, float(heartbeat_grace_s))
        # 推理进度计数：每产出一个 token +1。心跳线程只据此续命——独立定时器
        # 的「假心跳」会让 C 调用卡死也看着健康（2026-10-27 审核 P1-1）。
        self._progress = 0
        self._run_started = 0.0
        self._abort_requested = threading.Event()
        # P2.2：abort 按请求 id 匹配。旧的 abort 帧只有 op、没有 id，且读线程
        # 无条件置位全局 event——上一条请求迟到的 abort 会误杀下一条在飞请求；
        # 反向地，abort 帧先于推理线程取到 run 帧时（prefill 尚未开始）又会
        # 被 run 起点 `clear()` 抹掉，导致 abort 丢失。这里按 id 记账：
        # - `_active_run_id`：推理线程正在执行的请求；
        # - `_pending_run_id`：读线程已入队、推理线程还没取的请求；
        # - `_aborted_ids`：已收到 abort 的请求 id（跨线程，锁保护）。
        self._active_run_id: str | None = None
        self._pending_run_id: str | None = None
        self._aborted_ids: set[str] = set()
        self._abort_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._run_queue: queue.Queue = queue.Queue(maxsize=1)
        self._shutdown = threading.Event()      # 进程级：仅 shutdown/写失败触发
        self._conn_closed = threading.Event()   # 连接级：EOF 触发，停推理等待
        self._infer_active = threading.Event()  # 推理中：心跳线程据此发帧
        self._conn = None                       # 写失败时主动关连接唤醒读线程

    # -- 写回（锁保护） -----------------------------------------------------

    def _send(self, wfile, frame: dict) -> bool:
        """写一帧；失败只标记连接死亡并返回 False，**绝不向推理线程抛异常**。

        P2.2：此前写失败（客户端断开/被杀）会把 OSError 抛进推理线程，
        线程直接死掉；worker 进程仍在 accept 下一个连接，但新连接不会重发
        ready（first=False），supervisor 只能等满 load_timeout。现在写失败 =
        本连接不可恢复：置 `_conn_closed`/`_shutdown` 并关掉 socket，让
        worker 干净退出，由 supervisor 重建进程（恢复可服务状态）。
        """
        try:
            with self._write_lock:
                write_frame(wfile, frame)
            return True
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).warning(
                "worker 写帧失败（连接已断）：%r", exc)
            self._fail_connection()
            return False

    def _fail_connection(self) -> None:
        """连接级致命错误：停推理/心跳，关闭 socket 唤醒读线程，进程退出。"""
        self._conn_closed.set()
        self._shutdown.set()
        conn = self._conn
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    # -- 心跳线程 -----------------------------------------------------------

    def _heartbeat_loop(self, wfile) -> None:
        """推理期间上报心跳，但**只在推理有进展时续命**。

        2026-10-27 审核 P1-1：原实现按独立定时器无条件发帧，导致 `rkllm_run`
        在 C 里卡死时 supervisor 仍能收到心跳，`heartbeat_timeout_s` 永远不触发，
        `/health` 也一直报 ready。现规则：

        - 每次心跳带上 `progress`（已产出 token 数，prefill 阶段为 0）；
        - 有进展 → 发帧续命；
        - 无进展但仍在本次推理的 prefill 宽限期内（`heartbeat_grace_s`，默认
          120 s，覆盖 27B 的长 prefill）→ 照常发帧；
        - 宽限期过后仍无进展 → **不发帧**，让 supervisor 的心跳超时逻辑杀
          worker 释放 NPU（下一次请求由 supervisor 重启恢复）。

        空闲时不发帧：supervisor 没有读者，发心跳只会堆在 socket 缓冲区里。
        C 调用期间 ctypes 释放 GIL，本线程仍可写帧。
        """
        while not (self._shutdown.is_set() or self._conn_closed.is_set()):
            if not self._infer_active.wait(0.5):
                continue
            seen = self._progress
            while (self._infer_active.is_set()
                   and not self._shutdown.is_set()
                   and not self._conn_closed.is_set()):
                # 首个心跳延后一个间隔：短请求不会多出 heartbeat 帧，
                # 长 prefill 超过 interval 后才开始上报。
                time.sleep(self.heartbeat_interval_s)
                if not self._infer_active.is_set():
                    break
                progress = self._progress
                in_grace = (time.monotonic() - self._run_started
                            < self.heartbeat_grace_s)
                if progress == seen and not in_grace:
                    continue          # 卡死/停滞：停止续命，交给 supervisor 超时
                seen = progress
                if not self._send(wfile, {"op": OP_HEARTBEAT,
                                          "ts": time.time(),
                                          "progress": progress}):
                    return

    # -- 推理线程 -----------------------------------------------------------

    def _abort_pending(self, req_id: str) -> bool:
        """本次 run 是否已被 abort（全局旧帧或按 id 匹配的新帧）。"""
        with self._abort_lock:
            return (self._abort_requested.is_set()
                    or req_id in self._aborted_ids)

    def _finish_run(self, req_id: str) -> None:
        """run 收尾：清当前 id / abort 记录，避免影响下一次 run。"""
        with self._abort_lock:
            self._active_run_id = None
            self._abort_requested.clear()
            self._aborted_ids.discard(req_id)

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
            with self._abort_lock:
                self._active_run_id = req_id
                self._pending_run_id = None
                self._abort_requested.clear()
                # abort 帧可能先于本线程取到 run 帧（排队窗口）：按 id 记账后
                # 这里必须立刻兑现，不能因为起点清标志而丢掉。
                pre_aborted = req_id in self._aborted_ids
            try:
                params = frame.get("params") or {}
                # 进度/起点必须在置位 _infer_active 之前写好：心跳线程已置位后
                # 立即可能读这两个字段（P1-1 的续命依据）。
                self._progress = 0
                self._run_started = time.monotonic()
                self._infer_active.set()
                try:
                    if pre_aborted:
                        # run 还没开始就被取消：不进入 engine（不触发 prefill），
                        # 直接回明确的 aborted end，避免"假完成"。
                        self._send(wfile, {"op": OP_END, "id": req_id,
                                           "aborted": True})
                        continue
                    try:
                        aborted = False
                        for token in self.engine.stream(
                                frame.get("prompt", ""),
                                role=params.get("role", "user"),
                                enable_thinking=bool(params.get("enable_thinking", False)),
                                sampling=params.get("sampling"),
                                max_new_tokens=params.get("max_new_tokens"),
                                cache_hint=bool(params.get("cache_hint", True)),
                                lora=params.get("lora")):
                            if self._abort_pending(req_id):
                                aborted = True
                                break
                            self._progress += 1
                            if not self._send(wfile, {"op": OP_TOKEN,
                                                      "id": req_id,
                                                      "text": token}):
                                return      # 写失败：连接已死，线程退出
                        # P2.2：for 正常结束（含 0 个 token 的 prefill 阶段被
                        # abort）也要看 abort——此前只有循环体内检查，prefill
                        # 期间 abort 会落到 try/else 发**正常 end**（假完成）。
                        if not aborted and self._abort_pending(req_id):
                            aborted = True
                    except Exception as exc:     # 推理失败不拖垮 worker
                        # abort 打断 C 调用后引擎常以异常收尾：这是取消，不是
                        # 请求失败，必须回 aborted end（否则 supervisor 会当
                        # RequestFailed 503）。
                        if self._abort_pending(req_id):
                            self._send(wfile, {"op": OP_END, "id": req_id,
                                               "aborted": True})
                        else:
                            self._send(wfile, {"op": OP_ERROR, "id": req_id,
                                               "message": str(exc)[:500]})
                        continue
                    if aborted:
                        self._send(wfile, {"op": OP_END, "id": req_id,
                                           "aborted": True})
                    else:
                        perf = self.engine.last_perf()
                        body: dict = {"op": OP_END, "id": req_id}
                        if perf:
                            body["perf"] = {"prefill_tokens": perf[0],
                                            "generate_tokens": perf[1]}
                        if not self._send(wfile, body):
                            return
                finally:
                    self._infer_active.clear()
                    self._finish_run(req_id)
            except Exception as exc:             # 兜底：绝不静默杀死推理线程
                self._send(wfile, {"op": OP_ERROR, "id": req_id,
                                   "message": str(exc)[:500]})
                self._finish_run(req_id)

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
                    with self._abort_lock:
                        self._pending_run_id = frame.get("id", "")
                    try:
                        self._run_queue.put_nowait(frame)
                    except queue.Full:
                        with self._abort_lock:
                            self._pending_run_id = None
                        self._send(wfile, {"op": OP_ERROR,
                                           "id": frame.get("id", ""),
                                           "message": "busy",
                                           "busy": True})
                elif op == OP_ABORT:
                    # P2.2：按 id 匹配。旧客户端不带 id → 作用于当前在飞请求；
                    # 带 id 时只认「正在推理」或「已入队待推理」的 id，迟到的
                    # 旧请求 abort 一律忽略，绝不误杀下一条请求。
                    fid = frame.get("id")
                    with self._abort_lock:
                        active = self._active_run_id
                        pending = self._pending_run_id
                    if fid:
                        if fid not in (active, pending):
                            logging.getLogger(__name__).info(
                                "忽略迟到的 abort（id=%s, active=%s）",
                                fid, active)
                            continue
                        with self._abort_lock:
                            self._aborted_ids.add(fid)
                        if fid == active:
                            # 只有真的在推理时才打断 C 调用；仅排队（pending）
                            # 时由推理线程在开跑前兑现，避免对空闲句柄 abort。
                            try:
                                self.engine.abort()
                            except Exception as exc:
                                logging.getLogger(__name__).warning(
                                    "engine.abort failed (cancel): %r", exc)
                    else:
                        if active is None and pending is None:
                            continue        # 空闲：没有可取消的请求
                        self._abort_requested.set()
                        try:
                            self.engine.abort()
                        except Exception as exc:
                            logging.getLogger(__name__).warning(
                                "engine.abort failed (cancel): %r", exc)
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
        self._conn = conn
        self._conn_closed.clear()
        inference = threading.Thread(target=self._inference_loop, args=(wfile,),
                                     daemon=True)
        heartbeat = threading.Thread(target=self._heartbeat_loop, args=(wfile,),
                                     daemon=True, name="rkllm-worker-heartbeat")
        inference.start()
        heartbeat.start()
        self._reader_loop(conn, wfile)           # 读循环跑在当前线程
        self._conn_closed.set()
        inference.join(timeout=5)
        heartbeat.join(timeout=self.heartbeat_interval_s + 1)
        self._conn = None
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
                            ttl_s=getattr(args, "cache_ttl_s", 604800),
                            similarity=getattr(args, "cache_similarity", 0.5))
    return RealEngine(model_path=args.model, platform=args.platform,
                      context_len=args.context, max_new_tokens=args.max_tokens,
                      keep_history=not args.no_history, threads=args.threads,
                      prompt_cache=cache,
                      lora_paths=getattr(args, "lora_path", []) or [])


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
    # P2.2：cache.similarity 此前只有 build_engine 的 getattr 默认值，CLI 从未
    # 定义该参数 → 配置项是死键。补上参数面，由 server 侧 worker_args 传入。
    parser.add_argument("--cache-similarity", type=float, default=0.5,
                        help="前缀缓存接管单槽的相似度阈值（cache.similarity）")
    parser.add_argument("--lora-path", action="append", default=[],
                        help="LoRA adapter 文件（可重复）；名字取文件名 stem")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="fake 引擎每 token 延迟（压测/演示）")
    parser.add_argument("--heartbeat-interval", type=float, default=5.0,
                        help="推理期间心跳帧间隔秒数")
    parser.add_argument("--heartbeat-grace-s", type=float, default=120.0,
                        help="prefill 宽限期：首 token 之前无进度仍续命的上限秒数；"
                             "超过后停止心跳，交给 supervisor 超时杀 worker")
    args = parser.parse_args(argv)
    # P2：命令行范围校验，避免 --context=-1 / --max-tokens=0 直达 C 结构体
    if args.context <= 0 or args.max_tokens <= 0 or args.threads <= 0:
        parser.error("--context/--max-tokens/--threads 必须为正整数")
    if not 0.0 <= args.cache_similarity <= 1.0:
        parser.error("--cache-similarity 必须在 [0, 1] 内")
    engine = build_engine(args)
    try:
        Worker(engine, heartbeat_interval_s=args.heartbeat_interval,
               heartbeat_grace_s=args.heartbeat_grace_s).serve_forever(
            args.address, tcp=args.tcp)
    finally:
        # P1-7：任何退出路径都释放 NPU / dma-buf（与 cli / LocalSupervisor 对齐）
        try:
            engine.destroy()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
