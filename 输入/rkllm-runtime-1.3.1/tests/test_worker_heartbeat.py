"""worker 心跳 / supervisor 卡死判定回归。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.server.supervisor import Supervisor, WorkerClient


class _FakeClient:
    def __init__(self, active=True, age=0.0):
        self.active = active
        self.last_heartbeat = time.monotonic() - age


def test_heartbeat_expired_only_while_inference_active():
    sup = Supervisor(["fake"], heartbeat_timeout_s=20)
    sup.state = "ready"
    sup._client = _FakeClient(active=True, age=30)
    assert sup._heartbeat_expired() is True

    sup._client = _FakeClient(active=False, age=30)
    assert sup._heartbeat_expired() is False

    sup._client = _FakeClient(active=True, age=1)
    assert sup._heartbeat_expired() is False


def test_heartbeat_expired_requires_ready_state():
    sup = Supervisor(["fake"], heartbeat_timeout_s=20)
    sup.state = "error"
    sup._client = _FakeClient(active=True, age=30)
    assert sup._heartbeat_expired() is False


def test_worker_client_tracks_heartbeat_and_active_flag():
    # 用 socketpair 模拟 worker 端；直接验证 OP_HEARTBEAT 被消费且不产出 token。
    import socket
    import threading

    from rkllm.worker.frames import OP_END, OP_HEARTBEAT, OP_TOKEN, write_frame

    left, right = socket.socketpair()
    client = WorkerClient(left)
    rfile = right.makefile("wb")

    def worker():
        write_frame(rfile, {"op": OP_HEARTBEAT})
        write_frame(rfile, {"op": OP_TOKEN, "text": "hi"})
        write_frame(rfile, {"op": OP_END})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    items = list(client.run("prompt", request_timeout_s=5))
    assert items == ["hi"]
    assert client.active is False
    assert client.last_heartbeat > 0
    client.close()
    right.close()
