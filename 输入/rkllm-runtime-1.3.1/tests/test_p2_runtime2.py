"""P2.2 剩余项回归：非流式断连取消 + `heartbeat_grace_s` 配置化。

1. 非流式（`/v1/chat/completions`、`/v1/completions`、`/v1/responses`）在
   200 之前断开：真实 uvicorn + 原始 socket 关闭，断言请求被取消、worker 收到
   abort、执行槽归还、随后请求仍可服务（不让请求算到 900s）。
2. 排队中的非流式请求断开：不得 abort 无关的在飞请求（F-2 回归）。
3. 正常完成语义不变（无断连时 TestClient 200）。
4. `engine.heartbeat_grace_s` 不新增 SCHEMA 键（保持 84 键口径）：由
   `heartbeat_timeout_s` 派生 + `RKLLM_HEARTBEAT_GRACE_S` 运行时覆盖，
   `_build_supervisor` 用真实配置栈把 `--heartbeat-grace-s` 传给 worker。
5. 非流式计时口径：`run_blocking` 的 TTFT/duration 从拿到执行槽起算，排队等待
   只由 queue_wait_* 计数。
"""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import json
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from realism_helpers import (  # noqa: E402
    capture_supervisor,
    load_real_config,
    make_app,
    worker_arg,
)

from rkllm.cli_engine import (  # noqa: E402
    HEARTBEAT_GRACE_ENV,
    _heartbeat_grace_s,
)
from rkllm.config import SCHEMA, load_config  # noqa: E402
from rkllm.server.app import create_app  # noqa: E402
from rkllm.server.pipeline_runner import run_blocking  # noqa: E402
from rkllm.server.queue import InferenceQueue  # noqa: E402
from rkllm.server.supervisor import RequestAborted  # noqa: E402


# ---------------------------------------------------------------------------
# 外部边界替身：只替换 NPU worker；uvicorn / ASGI / 队列 / Pipeline 全真实
# ---------------------------------------------------------------------------


class _BlockingSupervisor:
    """第一次请求卡在 prefill 直到 abort；之后请求正常返回（验证恢复可服务）。"""

    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self):
        self.started = threading.Event()
        self.abort_evt = threading.Event()
        self.aborts = 0
        self.calls = 0

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            if not self.abort_evt.wait(30):
                raise AssertionError("非流式断连未传导到 supervisor.abort()")
            raise RequestAborted("inference aborted before completion")
        yield "ok"

    def abort(self):
        self.aborts += 1
        self.abort_evt.set()


class _SlowSupervisor:
    """慢 token 流（占住执行槽）；用于“排队请求断连不得误伤在飞请求”。"""

    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self, tokens, delay=0.02):
        self.tokens = list(tokens)
        self.delay = delay
        self.aborts = 0
        self.started = threading.Event()
        self.finished = threading.Event()

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        self.started.set()
        for token in self.tokens:
            time.sleep(self.delay)
            yield token
        self.finished.set()

    def abort(self):
        self.aborts += 1


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


@contextlib.contextmanager
def _serve(app):
    """真实 uvicorn 监听 127.0.0.1（进程内线程），yield 端口。"""
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error", access_log=False))
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()),
                              daemon=True, name="p2-runtime2-http")
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn 未能启动"
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(5)


def _send(port: int, path: str, body: dict) -> http.client.HTTPConnection:
    """发一个非流式 POST，返回连接（尚未读响应）。"""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    payload = json.dumps(body).encode()
    conn.request("POST", path, body=payload,
                 headers={"Content-Type": "application/json"})
    return conn


def _abort_socket(conn: http.client.HTTPConnection) -> None:
    """模拟客户端在响应前断开：直接关闭 TCP 连接（发 FIN）。"""
    sock = getattr(conn, "sock", None)
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            sock.close()
    with contextlib.suppress(Exception):
        conn.close()


def _wait_until(predicate, timeout=15.0, interval=0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# 三个非流式端点的请求体（chat / legacy completions / responses 翻译层）
_NONSTREAM_CASES = [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]}),
    ("/v1/completions", {"prompt": "x"}),
    ("/v1/responses", {"input": "x"}),
]


# ---------------------------------------------------------------------------
# 1) 非流式断连：真实 uvicorn + 原始 socket
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,body", _NONSTREAM_CASES)
def test_nonstream_disconnect_cancels_request_and_aborts_worker(path, body):
    """客户端在 200 前断开：立即置位取消 + abort worker，而不是算到 900s。"""
    supervisor = _BlockingSupervisor()
    app = create_app(load_config(env={}), supervisor=supervisor)
    with _serve(app) as port:
        conn = _send(port, path, body)
        try:
            assert supervisor.started.wait(10), "请求未到达推理层"
            started = time.monotonic()
            _abort_socket(conn)                       # 200 之前断开
            assert _wait_until(lambda: supervisor.aborts >= 1, timeout=10), (
                "客户端断连未传导到 supervisor.abort()（请求会算到 900s）")
            assert time.monotonic() - started < 10
            # 取消后执行槽应归还：随后请求可正常服务（不会排队等旧请求）
            if supervisor.calls == 1:
                assert _wait_until(
                    lambda: app.state.svc.queue.queue_depth == 0, timeout=10)
            ok = _send(port, path, body)
            resp = ok.getresponse()
            raw = resp.read()
            ok.close()
            assert resp.status == 200, raw
            assert b"ok" in raw
        finally:
            with contextlib.suppress(Exception):
                conn.close()


@pytest.mark.parametrize("path,body", _NONSTREAM_CASES)
def test_nonstream_normal_completion_semantics_unchanged(path, body):
    """无断连时非流式语义不变（TestClient；异步路由改造的回归）。"""
    app, _sup = make_app(load_config(env={}), tokens=["ok"])
    with TestClient(app) as client:
        r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    assert "ok" in r.text


def test_queued_nonstream_disconnect_does_not_abort_inflight():
    """排队中的非流式请求断连：不得打断无关的在飞请求（F-2 回归）。"""
    tokens = ["t%d " % i for i in range(40)]          # 40×0.02s ≈ 0.8s
    supervisor = _SlowSupervisor(tokens, delay=0.02)
    app = create_app(load_config(env={}), supervisor=supervisor)
    with _serve(app) as port:
        flight = _send(port, "/v1/chat/completions",
                       {"messages": [{"role": "user", "content": "A"}]})
        queued = None
        try:
            assert supervisor.started.wait(10)
            queued = _send(port, "/v1/chat/completions",
                           {"messages": [{"role": "user", "content": "B"}]})
            assert _wait_until(
                lambda: app.state.svc.queue.queue_depth >= 1, timeout=10), (
                "第二个请求未进入排队，用例前提不成立")
            _abort_socket(queued)                     # 排队期断连
            queued = None
            time.sleep(0.4)                           # 旧实现此刻会 abort 在飞请求
            resp = flight.getresponse()
            payload = json.loads(resp.read())
            content = payload["choices"][0]["message"]["content"]
            assert content == "".join(tokens), "在飞请求被排队断连截断"
            assert supervisor.aborts == 0, (
                "排队请求断连 aborts 了无关的在飞请求（F-2 回归）")
        finally:
            for conn in (flight, queued):
                if conn is not None:
                    with contextlib.suppress(Exception):
                        conn.close()


# ---------------------------------------------------------------------------
# 1b) 断连监视单元：真断开返回哨兵；receive 异常不得误判为断开
# ---------------------------------------------------------------------------


class _FakeRequest:
    """可控的 ASGI receive：前 n 次返回请求体，之后返回 disconnect。"""

    def __init__(self, *, disconnect_after: int = 0):
        self.calls = 0
        self.disconnect_after = disconnect_after

    async def receive(self):
        self.calls += 1
        if self.calls > self.disconnect_after:
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": b"", "more_body": False}


class _RaisingRequest:
    """receive 直接抛错的 ASGI 适配层（不能当成客户端断开）。"""

    async def receive(self):
        raise RuntimeError("receive channel unavailable")


def test_run_blocking_disconnect_returns_sentinel_on_real_disconnect():
    from rkllm.server.pipeline_runner import (
        DISCONNECTED,
        run_blocking_disconnect,
    )

    aborts: list[int] = []

    class _Supervisor:
        def abort(self):
            aborts.append(1)

    def work(cancelled, on_acquire, on_release):
        on_acquire()
        deadline = time.monotonic() + 5
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(0.01)
        on_release()
        raise RequestAborted("inference aborted before completion")

    result = asyncio.run(run_blocking_disconnect(
        _FakeRequest(disconnect_after=0), work, supervisor=_Supervisor()))
    assert result is DISCONNECTED
    assert aborts, "真断开且持槽时必须 abort worker"


def test_run_blocking_disconnect_ignores_receive_failure():
    """receive 不可用 → 退回旧行为（等阻塞执行），不能误取消正常请求。"""
    from rkllm.server.pipeline_runner import run_blocking_disconnect

    def work(cancelled, on_acquire, on_release):
        on_acquire()
        on_release()
        return {"text": "ok"}

    result = asyncio.run(run_blocking_disconnect(
        _RaisingRequest(), work, supervisor=None))
    assert result == {"text": "ok"}


def test_asgi_receive_without_disconnect_is_not_treated_as_disconnect():
    """直接驱动 ASGI（receive 永远返回 http.request）时请求必须正常 200。

    回归：REQ-1 期间曾把「receive 提前结束/异常」误判成客户端断开，导致
    chunked body 等真实路径被 499/取消。
    """
    app, _sup = make_app(load_config(env={}), tokens=["ok"])
    payload = json.dumps({"messages": [{"role": "user", "content": "x"}]}
                         ).encode()
    pending = [payload]
    messages: list[dict] = []

    async def receive():
        if pending:
            return {"type": "http.request", "body": pending.pop(0),
                    "more_body": bool(pending)}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions", "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode())],
        "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8080),
        "state": {},
    }
    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages
                    if m["type"] == "http.response.body")
    assert start["status"] == 200, body
    assert json.loads(body)["choices"][0]["message"]["content"] == "ok"


# ---------------------------------------------------------------------------
# 2) heartbeat_grace_s：不新增 SCHEMA 键，派生 + 运行时 env 覆盖
# ---------------------------------------------------------------------------


def _cfg_with_timeout(timeout_s: int):
    return SimpleNamespace(
        engine=SimpleNamespace(heartbeat_timeout_s=timeout_s))


def test_heartbeat_grace_derived_from_heartbeat_timeout(monkeypatch):
    monkeypatch.delenv(HEARTBEAT_GRACE_ENV, raising=False)
    assert _heartbeat_grace_s(_cfg_with_timeout(20)) == 120.0   # 出厂默认
    assert _heartbeat_grace_s(_cfg_with_timeout(60)) == 120.0   # 下限 120
    assert _heartbeat_grace_s(_cfg_with_timeout(120)) == 240.0  # 2× 跟随


def test_heartbeat_grace_runtime_env_override(monkeypatch):
    monkeypatch.setenv(HEARTBEAT_GRACE_ENV, "45.5")
    assert _heartbeat_grace_s(_cfg_with_timeout(20)) == 45.5
    monkeypatch.setenv(HEARTBEAT_GRACE_ENV, "abc")              # 非法值回退
    assert _heartbeat_grace_s(_cfg_with_timeout(20)) == 120.0
    monkeypatch.setenv(HEARTBEAT_GRACE_ENV, "-5")               # 负值回退
    assert _heartbeat_grace_s(_cfg_with_timeout(20)) == 120.0


def test_heartbeat_grace_adds_no_schema_key():
    """保持 84/85 键口径：不新增 engine.heartbeat_grace_s 配置键。"""
    keys = {key.full for key in SCHEMA}
    assert "engine.heartbeat_grace_s" not in keys
    assert "engine.heartbeat_timeout_s" in keys


def _model_file(tmp_path, name="model.rkllm") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\0" * 4096)
    return path


def _build_supervisor_args(tmp_path, monkeypatch):
    model = _model_file(tmp_path)
    cfg = load_real_config(
        monkeypatch, tmp_path,
        toml_sections={"model": {"path": str(model)}},
        require_model_path=True)
    from rkllm.cli_engine import _build_supervisor
    captured = capture_supervisor(monkeypatch)
    _build_supervisor(cfg, use_fake=False)
    return cfg, captured.instances[-1].worker_args


def test_build_supervisor_passes_heartbeat_grace(tmp_path, monkeypatch):
    """真实配置栈：`--heartbeat-grace-s` 必须显式下发（默认 120s）。"""
    monkeypatch.delenv(HEARTBEAT_GRACE_ENV, raising=False)
    cfg, args = _build_supervisor_args(tmp_path, monkeypatch)
    assert float(worker_arg(args, "--heartbeat-grace-s")) == 120.0
    assert worker_arg(args, "--heartbeat-interval") == \
        str(cfg.engine.heartbeat_interval_s)


def test_build_supervisor_passes_env_override_grace(tmp_path, monkeypatch):
    monkeypatch.setenv(HEARTBEAT_GRACE_ENV, "45")
    _cfg, args = _build_supervisor_args(tmp_path, monkeypatch)
    assert float(worker_arg(args, "--heartbeat-grace-s")) == 45.0


# ---------------------------------------------------------------------------
# 3) 非流式计时口径：TTFT/duration 从拿到执行槽起算，排队等待单独计
# ---------------------------------------------------------------------------


class _ImmediatePipe:
    endpoint = "chat.completions"
    prompt_bytes = 0

    def iter_tokens(self):
        yield "t0"
        yield "t1"


def test_run_blocking_timing_starts_at_slot_acquire():
    """排队 0.25s 期间不算进 TTFT/duration；queue_wait_* 单独记录。"""
    queue = InferenceQueue(max_wait=5, wait_timeout_s=5)
    held = threading.Event()

    def _holder_tokens():
        held.set()
        time.sleep(0.25)          # 占住唯一执行槽，让后续请求排队约 0.25s
        yield "x"

    holder = threading.Thread(target=lambda: list(queue.run(_holder_tokens)),
                              daemon=True)
    holder.start()
    try:
        assert held.wait(5)
        svc = SimpleNamespace(queue=queue)
        result = run_blocking(svc, _ImmediatePipe())
        assert result["text"] == "t0t1"
        wait_s = queue.metrics["queue_wait_seconds_sum"]
        duration_s = queue.metrics["duration_sum"]
        ttft_s = queue.metrics["last_ttft_s"]
        assert wait_s >= 0.2, "排队等待未被单独计入 queue_wait_seconds_sum"
        assert duration_s < wait_s - 0.1, (
            "duration 把排队等待算进了推理耗时（口径未分离）："
            "duration=%.3f wait=%.3f" % (duration_s, wait_s))
        assert ttft_s < 0.2, "TTFT 把排队等待算进去了：%.3f" % ttft_s
    finally:
        holder.join(5)
