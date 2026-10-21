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
        # 读 ready 帧（worker 加载完模型才发）
        header = self._proc.stdout.read(4)
        if len(header) < 4:
            stderr = self._proc.stderr.read(4000).decode("utf-8", "replace")
            raise WorkerDown("worker failed to start: %s" % stderr[-800:])
        (length,) = _unpack(header)
        ready = json.loads(self._proc.stdout.read(length))
        if ready.get("op") != "ready":
            raise WorkerDown("worker sent %r instead of ready" % ready)
        if tcp:
            host, _, port = address.rpartition(":")
            conn = socket.create_connection((host, int(port)),
                                            timeout=self.load_timeout_s)
        else:
            import os
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(self.load_timeout_s)
            conn.connect(address)
        self._client = WorkerClient(conn)
        self.state = "ready"

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
