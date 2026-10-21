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
import json
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator

from ..worker.frames import read_frame, write_frame, FrameError

SRC_DIR = str(Path(__file__).resolve().parents[2])


class WorkerDown(Exception):
    """worker 不可用（崩溃 / 加载中）：调用方应回 503 + Retry-After。"""


class WorkerClient:
    """与单个 worker 连接的帧客户端：run 请求逐 token 产出。"""

    def __init__(self, conn):
        self.conn = conn
        self._rfile = conn.makefile("rb")
        self._wfile = conn.makefile("wb")
        self._write_lock = threading.Lock()

    def run(self, prompt: str, *, role: str = "user",
            enable_thinking: bool = False, max_new_tokens: int | None = None,
            request_timeout_s: int = 900) -> Iterator[str]:
        req_id = uuid.uuid4().hex[:10]
        with self._write_lock:
            write_frame(self._wfile, {"op": "run", "id": req_id,
                                      "prompt": prompt,
                                      "params": {"role": role,
                                                 "enable_thinking": enable_thinking,
                                                 "max_new_tokens": max_new_tokens}})
        deadline = time.monotonic() + request_timeout_s
        while True:
            if time.monotonic() > deadline:
                self.abort()
                raise WorkerDown("request timeout")
            try:
                frame = read_frame(self._rfile)
            except (FrameError, OSError) as exc:
                raise WorkerDown("worker connection lost: %s" % exc)
            if not frame:
                raise WorkerDown("worker closed connection")
            op = frame.get("op")
            if op == "token":
                yield frame["text"]
            elif op == "end":
                return
            elif op == "error":
                raise WorkerDown(frame.get("message", "worker error"))

    def abort(self) -> None:
        with self._write_lock:
            try:
                write_frame(self._wfile, {"op": "abort"})
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
                 restart_backoff_max_s: int = 120, restart_max_per_hour: int = 3):
        self.worker_args = worker_args
        self.transport = transport        # auto | unix | tcp
        self.load_timeout_s = load_timeout_s
        self.backoff_s = restart_backoff_s
        self.backoff_max_s = restart_backoff_max_s
        self.restart_max_per_hour = restart_max_per_hour
        self.state = "loading"            # loading / ready / error / stopped
        self.restarts: list[float] = []
        self.metrics = {"worker_restarts_total": 0}
        self._proc: subprocess.Popen | None = None
        self._client: WorkerClient | None = None
        self._lock = threading.Lock()
        self._current_backoff = self.backoff_s

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
                                      stderr=subprocess.PIPE)
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
                    import os
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
        ready = read_frame(self._client._rfile)
        if ready.get("op") != "ready":
            raise WorkerDown("worker sent %r instead of ready" % ready)
        self.state = "ready"

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

    def start(self) -> None:
        with self._lock:
            try:
                self._spawn()
            except WorkerDown:
                self.state = "error"
                raise

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
            self._proc.terminate()
        delay = self._current_backoff
        self._current_backoff = min(self._current_backoff * 2, self.backoff_max_s)
        time.sleep(delay)
        with self._lock:
            self._spawn()
        self.restarts.append(time.monotonic())

    def execute(self, prompt: str, **kwargs) -> Iterator[str]:
        """带自愈的执行入口：连接失效时重启一次再抛（由路由转 503）。"""
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
        self.state = "stopped"
        if self._client:
            self._client.close()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()


def _unpack(head: bytes) -> tuple[int]:
    import struct
    return struct.unpack(">I", head)
