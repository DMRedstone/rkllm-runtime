"""supervisor：worker 子进程生命周期 + 帧客户端（改造设计 §6.1，DoD：
kill -9 worker 后 HTTP 存活并在飞请求 503，120 s 内恢复）。

- 启动：spawn `python -m rkllm.worker.main --engine ...`，等 ready 帧；
- 崩溃：检测到连接断 → 状态 error → 在飞请求拿到 503；指数退避重启
  （restart_backoff_s 起，封顶 restart_backoff_max_s），每小时超过
  restart_max_per_hour 次停止自愈；
- engine=fake 时同样走子进程（与生产同一条帧协议路径，测试更有意义）；
- Windows 无 AF_UNIX：transport 参数化（板端 unix，开发/测试 tcp）。
"""

from __future__ import annotations

import collections
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator

from ..worker.frames import (OP_ABORT, OP_END, OP_ERROR, OP_HEARTBEAT,
                             OP_READY, OP_RUN, OP_TOKEN, PROTOCOL_VERSION,
                             FrameError, read_frame, write_frame)

SRC_DIR = str(Path(__file__).resolve().parents[2])
log = logging.getLogger("rkllm.server.supervisor")


class WorkerDown(Exception):
    """worker 不可用（崩溃 / 加载中）：调用方应回 503 + Retry-After。"""


class RequestFailed(WorkerDown):
    """单次请求失败（op:"error" 帧 / 请求超时），worker 进程本身仍然健康。

    必须与 WorkerDown 区分：execute() 只在连接失效时重启 worker。把请求级
    错误也当连接死亡，代价是一次 rkllm_run -1 就触发 14 s 模型重载，还烧掉
    restart_max_per_hour 配额（2026-10-22 审计 P1-2）。
    """


class WorkerClient:
    """与单个 worker 连接的帧客户端：run 请求逐 token 产出。"""

    def __init__(self, conn):
        self.conn = conn
        self._rfile = conn.makefile("rb")
        self._wfile = conn.makefile("wb")
        self._write_lock = threading.Lock()
        # 心跳：仅在本次 run 进行期间要求 worker 上报（空闲时没有读者，
        # worker 也不会发帧，避免 socket 缓冲区堆积）。
        self.active = False
        self.last_heartbeat = time.monotonic()

    def run(self, prompt: str, *, role: str = "user",
            enable_thinking: bool = False, max_new_tokens: int | None = None,
            sampling: dict | None = None, cache_hint: bool = True,
            request_timeout_s: int = 900) -> Iterator[str]:
        req_id = uuid.uuid4().hex[:10]
        self.active = True
        self.last_heartbeat = time.monotonic()
        try:
            try:
                with self._write_lock:
                    write_frame(self._wfile, {"op": OP_RUN, "id": req_id,
                                              "prompt": prompt,
                                              "params": {"role": role,
                                                         "enable_thinking": enable_thinking,
                                                         "max_new_tokens": max_new_tokens,
                                                         "sampling": sampling,
                                                         "cache_hint": cache_hint}})
            except OSError as exc:
                # worker 被 kill -9 后第一次写帧是 BrokenPipeError，不是 EOF：
                # 不转成 WorkerDown 的话，自愈逻辑不会触发，路由层还会把它当 500。
                raise WorkerDown("worker write failed: %s" % exc)
            deadline = time.monotonic() + request_timeout_s
            while True:
                if time.monotonic() > deadline:
                    self.abort()
                    raise RequestFailed("request timeout")
                try:
                    frame = read_frame(self._rfile)
                except (FrameError, OSError) as exc:
                    raise WorkerDown("worker connection lost: %s" % exc)
                if not frame:
                    raise WorkerDown("worker closed connection")
                op = frame.get("op")
                fid = frame.get("id")
                # P1-9：单 socket FIFO 上可能残留上一请求的帧；只处理本请求的帧，
                # 否则会把残留 end 当成当前请求完成而立即返回空响应。
                if fid is not None and fid != req_id:
                    continue
                if op == OP_HEARTBEAT:
                    self.last_heartbeat = time.monotonic()
                    continue
                if op == OP_TOKEN:
                    yield frame["text"]
                elif op == OP_END:
                    # end 帧带 perf（prefill/generate tokens）——此前直接 return
                    # 把它丢了，usage 的 exact counts 从未生效（永远估算值），
                    # finish_reason 的 length 判定也没有依据。作为管线元数据
                    # dict 产出（与 tool_calls/parse_error 同一条路）。
                    perf = frame.get("perf")
                    if perf:
                        yield {"perf": perf}
                    return
                elif op == OP_ERROR:
                    # worker 明确报告本次请求失败：进程活着，绝不走重启路径
                    raise RequestFailed(frame.get("message", "worker error"))
        finally:
            self.active = False

    def abort(self) -> None:
        with self._write_lock:
            try:
                write_frame(self._wfile, {"op": OP_ABORT})
            except OSError:
                pass

    def close(self) -> None:
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.conn.close()
        except OSError:
            pass


class Supervisor:
    def __init__(self, worker_args: list[str], *, transport: str = "auto",
                 load_timeout_s: int = 300, restart_backoff_s: int = 5,
                 restart_backoff_max_s: int = 120, restart_max_per_hour: int = 3,
                 health_interval_s: int = 10, request_timeout_s: int = 900,
                 heartbeat_timeout_s: int = 20):
        self.worker_args = worker_args
        self.transport = transport        # auto | unix | tcp
        self.load_timeout_s = load_timeout_s
        # P1-10：推理期 socket 读超时与加载超时解耦（加载 300s，请求 900s）
        self.request_timeout_s = request_timeout_s
        self.backoff_s = restart_backoff_s
        self.backoff_max_s = restart_backoff_max_s
        self.restart_max_per_hour = restart_max_per_hour
        self.health_interval_s = health_interval_s
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self._health_stop = threading.Event()
        self.state = "loading"            # loading / ready / error / stopped
        self.restarts: list[float] = []
        self.metrics = {"worker_restarts_total": 0,
                        "worker_heartbeat_timeouts_total": 0}
        self._proc: subprocess.Popen | None = None
        self._client: WorkerClient | None = None
        self._lock = threading.Lock()
        self._current_backoff = self.backoff_s
        # D10：worker 在 ready 帧里报回真正 dlopen 的库指纹（server 不重复加载）
        self.lib_fingerprint: dict | None = None

    # -- 状态 ---------------------------------------------------------------

    def is_ready(self) -> bool:
        return self.state == "ready"

    # -- 子进程管理 -----------------------------------------------------------

    def _make_address(self) -> tuple[str, bool]:
        if self.transport == "unix" or (
                self.transport == "auto" and hasattr(socket, "AF_UNIX")):
            path = "/tmp/rkllm-worker-%d.sock" % id(self)
            return path, False
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        addr = "127.0.0.1:%d" % srv.getsockname()[1]
        srv.close()
        return addr, True

    def _spawn(self) -> None:
        address, tcp = self._make_address()
        cmd = [sys.executable, "-m", "rkllm.worker.main",
               "--address", address, "--engine"] + self.worker_args
        if tcp:
            cmd.insert(cmd.index("--address") + 2, "--tcp")
        self._proc = subprocess.Popen(cmd, cwd=SRC_DIR,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      # 禁止 worker 在包目录写 __pycache__：
                                      # 字节码不在 dpkg 清单里，卸载会留残留
                                      env=dict(os.environ,
                                               PYTHONDONTWRITEBYTECODE="1"))
        # stdout/stderr 都只承载 librkllmrt 的 C 库日志，起排水线程持续排空：
        # 管道缓冲仅 64KB，没人读的话 worker 会卡死在日志 write 上。
        # ready 帧不再走 stdout——真机首测中 C 库日志把 stdout 帧流污染，
        # 帧头被解析成 1.2 GB 的假长度后 supervisor 永久阻塞（FakeEngine
        # 日志少，pytest 全绿测不到）。ready 改走 UDS 连接本身，见下。
        self._drain("stdout")
        self._drain("stderr")
        # 轮询 connect worker 的 socket：连接成功即模型加载完成（worker 在
        # bind+listen 之后、accept 首连接时发 ready）。
        deadline = time.monotonic() + self.load_timeout_s
        conn = None
        while conn is None:
            if self._proc.poll() is not None:
                raise WorkerDown("worker exited during load: %s"
                                 % self._stderr_tail(4000))
            if time.monotonic() > deadline:
                raise WorkerDown("worker not ready within %ss"
                                 % self.load_timeout_s)
            try:
                if tcp:
                    host, _, port = address.rpartition(":")
                    conn = socket.create_connection((host, int(port)),
                                                    timeout=5)
                else:
                    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    conn.settimeout(5)
                    conn.connect(address)
            except OSError:
                if conn is not None:
                    conn.close()
                    conn = None
                time.sleep(0.5)
        conn.settimeout(self.load_timeout_s)
        self._client = WorkerClient(conn)
        # ready 帧从 UDS 连接收：read_frame 自带帧长上限校验，C 库日志再也无法
        # 伪装成协议帧。
        try:
            ready = read_frame(self._client._rfile)
        except (FrameError, OSError, ValueError) as exc:
            # P2：就绪帧失败路径必须关闭连接，否则 fd 泄漏、重试时放大
            self._client.close()
            self._client = None
            raise WorkerDown("worker ready frame failed: %s" % exc)
        if ready.get("op") != OP_READY:
            self._client.close()
            self._client = None
            raise WorkerDown("worker sent %r instead of ready" % ready)
        # A-8：帧协议版本必须一致（灰度/回滚时 worker 与 server 可能错配）。
        # 缺字段视为旧 worker，按兼容放行。
        worker_proto = ready.get("v")
        if worker_proto is not None and worker_proto != PROTOCOL_VERSION:
            self._client.close()
            self._client = None
            raise WorkerDown("worker frame protocol v%s != server v%s"
                             % (worker_proto, PROTOCOL_VERSION))
        if ready.get("lib_sha256"):
            self.lib_fingerprint = {
                "sha256": ready.get("lib_sha256"),
                "size": ready.get("lib_size"),
                "path": ready.get("lib_path"),
            }
        self.state = "ready"
        # P1-10：加载完成，把读超时从 load_timeout 提升到请求超时，避免长 prefill
        # （>300s 但 <900s 的合法请求）被误判 WorkerDown 而触发重启 + 503。
        conn.settimeout(self.request_timeout_s)

    def _drain(self, which: str) -> None:
        """后台线程持续排空 worker 的 stdout/stderr（C 库日志），防管道死锁。

        stderr 尾部片段存内存，WorkerDown 时作为诊断输出；stdout 的 C 库日志
        无协议价值，读后直接丢弃。"""
        if which == "stderr":
            self._stderr_tail_buf = collections.deque(maxlen=32)   # ~8KB 尾部
        proc = self._proc
        stream = proc.stdout if which == "stdout" else proc.stderr

        def _pump():
            try:
                for raw in iter(lambda: stream.read(256), b""):
                    if which == "stderr":
                        self._stderr_tail_buf.append(
                            raw.decode("utf-8", "replace"))
            except (ValueError, OSError):
                pass   # 进程退出 / 管道关闭：正常结束

        threading.Thread(target=_pump, daemon=True,
                         name="rkllm-worker-%s-drain" % which).start()

    def _stderr_tail(self, limit: int = 4000) -> str:
        """WorkerDown 时替代原来的 stderr.read()（那个会再次死锁）。"""
        return "".join(getattr(self, "_stderr_tail_buf", []))[-limit:]

    def _heartbeat_expired(self, now: float | None = None) -> bool:
        """推理中超过 heartbeat_timeout_s 没有心跳 → 认为 NPU 卡死。"""
        client = self._client
        if not (self.state == "ready" and client is not None
                and getattr(client, "active", False)):
            return False
        now = time.monotonic() if now is None else now
        return (now - getattr(client, "last_heartbeat", now)
                > self.heartbeat_timeout_s)

    def _start_health(self) -> None:
        """P2：健康监测线程——消费 health_interval_s / _health_stop。

        定期检查 worker 进程是否已死；发现后置 state=error，让下一次请求触发
        自愈重启。另检查推理期心跳：C 调用卡死时进程仍活着（poll 为 None），
        只能靠心跳缺失判定；发现后杀掉 worker 释放 NPU，下一次请求自动重启。
        """
        if getattr(self, "_health_thread", None) is not None:
            return

        def _loop() -> None:
            while not self._health_stop.wait(max(1, self.health_interval_s)):
                try:
                    proc = self._proc
                    if (self.state == "ready" and proc is not None
                            and proc.poll() is not None):
                        self.state = "error"
                        continue
                    if self._heartbeat_expired():
                        self.metrics["worker_heartbeat_timeouts_total"] += 1
                        log.error("worker heartbeat timeout after %ss "
                                  "(inference stuck); killing worker",
                                  self.heartbeat_timeout_s)
                        self.state = "error"
                        if proc is not None and proc.poll() is None:
                            proc.kill()
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                pass
                        if self._client is not None:
                            self._client.close()
                except Exception:
                    pass

        self._health_thread = threading.Thread(
            target=_loop, daemon=True, name="rkllm-health")
        self._health_thread.start()

    def start(self) -> None:
        with self._lock:
            try:
                self._spawn()
            except WorkerDown:
                self.state = "error"
                raise
            self._start_health()

    def restart(self) -> None:
        """指数退避重启；超过每小时上限停止自愈（DoD：120 s 内恢复）。"""
        now = time.monotonic()
        self.restarts = [t for t in self.restarts if now - t < 3600]
        if len(self.restarts) >= self.restart_max_per_hour:
            self.state = "error"
            raise WorkerDown("restart limit reached (%d/h)" % self.restart_max_per_hour)
        self.metrics["worker_restarts_total"] += 1
        if self._client:
            self._client.close()
        if self._proc and self._proc.poll() is None:
            # P1-8：terminate 后必须 wait，否则旧 worker 卡在 C 调用时会与新 worker
            # 并存并争抢同一 NPU，且已 terminate 的进程成僵尸；超时则 kill 再 wait。
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        delay = self._current_backoff
        self._current_backoff = min(self._current_backoff * 2, self.backoff_max_s)
        time.sleep(delay)
        with self._lock:
            self._spawn()
        self.restarts.append(time.monotonic())

    def execute(self, prompt: str, **kwargs) -> Iterator[str]:
        """带自愈的执行入口：连接失效时重启一次再抛（由路由转 503）。

        kwargs 原样透传（role / enable_thinking / max_new_tokens / sampling /
        request_timeout_s）——采样参数由 Pipeline 合成后经这里下发到 worker。
        """
        client = self._client
        if client is None or self.state != "ready":
            self._current_backoff = self.backoff_s
            try:
                self.restart()
            except WorkerDown:
                raise
            client = self._client
        try:
            yield from client.run(prompt, **kwargs)
        except RequestFailed:
            raise                          # worker 健康，路由层转 503 即可
        except WorkerDown:
            self.state = "loading"
            try:
                self.restart()
            except WorkerDown:
                self.state = "error"
            raise

    def abort(self) -> None:
        if self._client:
            self._client.abort()

    def shutdown(self) -> None:
        self._health_stop.set()
        self.state = "stopped"
        if self._client:
            self._client.close()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()


class LocalSupervisor:
    """`engine.mode=in_process`：进程内推理（S1 回滚开关 / 排障用）。

    与 Supervisor 同接口，但**不隔离崩溃**——NPU/库段错误会带走整个 HTTP
    进程。设计把 in_process 定位为回滚与排障通道，生产默认仍是 subprocess：

      engine.mode = "subprocess"（默认，崩溃隔离）
      engine.mode = "in_process"（回滚：省掉 IPC，换掉隔离性）

    abort() 从非推理线程调用 rkllm_abort，与 worker 内部的做法一致。
    """

    def __init__(self, engine, *, load_timeout_s: int = 300):
        self.engine = engine
        self.load_timeout_s = load_timeout_s
        # P1-12：与 Supervisor 接口对齐，避免 shutdown() 引用不存在的属性而
        # 在 engine.destroy() 之前 AttributeError 崩溃。
        self._health_stop = threading.Event()
        self.state = "ready"          # 引擎在构造时就已加载完成
        self.metrics = {"worker_restarts_total": 0}
        self.lib_fingerprint = getattr(engine, "lib_fingerprint", None)

    def is_ready(self) -> bool:
        return self.state == "ready"

    def start(self) -> None:
        self.state = "ready"

    def restart(self) -> None:        # 语义对齐：进程内无子进程可重启
        raise WorkerDown("in_process mode has no worker to restart")

    def execute(self, prompt: str, *, role: str = "user",
                enable_thinking: bool = False,
                max_new_tokens: int | None = None,
                sampling: dict | None = None,
                cache_hint: bool = True,
                request_timeout_s: int = 900) -> Iterator[str]:
        """与 `Supervisor.execute` 同语义：超过 `request_timeout_s` 中止推理。

        P2：此前该参数被完全忽略（进程内模式没有 socket 超时），长 prefill
        在 in_process 模式下没有任何上限。这里用定时器调 `abort()` 对齐
        supervisor 的「请求超时 → 打断推理」行为。
        """
        if self.state != "ready":
            raise WorkerDown("engine not ready")
        timer = None
        if request_timeout_s and request_timeout_s > 0:
            def _expire() -> None:
                # 从定时器线程打断：等价于 subprocess 模式的读超时
                try:
                    self.engine.abort()
                except Exception:
                    pass
            timer = threading.Timer(request_timeout_s, _expire)
            timer.daemon = True
            timer.start()
        try:
            yield from self.engine.stream(
                prompt, role=role, enable_thinking=enable_thinking,
                sampling=sampling, max_new_tokens=max_new_tokens,
                cache_hint=cache_hint)
            # 对齐 worker 的 end 帧：产出 perf 元数据（usage exact counts 与
            # finish_reason 的 length 判定的依据）。此前进程内模式没有这段，
            # 与 subprocess 路径行为不一致。
            perf = self.engine.last_perf()
            if perf:
                yield {"perf": {"prefill_tokens": perf[0],
                                "generate_tokens": perf[1]}}
        except WorkerDown:
            raise
        except Exception as exc:      # 与 worker 路径的错误语义对齐：结构 503；
            # 引擎进程仍在，按请求级失败处理，不把 supervisor 打进 error
            raise RequestFailed("engine failure: %s" % exc)
        finally:
            if timer is not None:
                timer.cancel()

    def abort(self) -> None:
        try:
            self.engine.abort()
        except Exception:
            pass

    def shutdown(self) -> None:
        self._health_stop.set()
        self.state = "stopped"
        try:
            self.engine.destroy()
        except Exception:
            pass

