"""P0 流式 tools + 取消/背压语义回归（2026-10-27 审核修复）。

覆盖审核报告里的四条 P1/P0：
1. **P0**：流式 chat + tools 收尾帧读嵌套 `function` 结构（此前
   `pipeline.trailing_frames` 读 `c["name"]` 直接 KeyError，客户端只收到首帧，
   既无 `finish_reason` 也无 `[DONE]`，请求被静默截断）。
2. **P1-1**：排队期断连不得 abort 无关的在飞请求——`CancelToken` 只有在本
   请求真的持有执行槽时才允许 `supervisor.abort()`；同时 `OP_END.aborted`
   必须透传（不能再伪装 200 + `finish_reason="stop"`）。
3. **P1-3**：背压截断必须 abort worker 归还 NPU（只释放队列槽位会让旧请求
   继续算，下一请求首 token 被拖到旧生成结束）。
4. **F-3**：abort 帧携带最近一次 run id（worker 侧匹配属另一人改造范围，
   本侧先把字段带上并回归帧形状）。

测试全部用 fake supervisor / socketpair / 真实 `InferenceQueue`，不依赖 NPU；
唯一的 HTTP 端到端用例在临时端口起真实 uvicorn（模拟"排队客户端断开"）。
"""

from __future__ import annotations

import asyncio
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

from rkllm.agent.pipeline import Pipeline
from rkllm.config import load_config
from rkllm.server import streaming as streaming_mod
from rkllm.server.app import create_app
from rkllm.server.queue import InferenceQueue, QueueBusy, QueueTimeout
from rkllm.server.sse import SseSession
from rkllm.server.streaming import (STREAM_DONE, TRUNCATED, CancelToken,
                                    TokenStream)
from rkllm.server.supervisor import RequestAborted, WorkerClient
from rkllm.worker.frames import OP_END, read_frame, write_frame


# ---------------------------------------------------------------------------
# 公共替身
# ---------------------------------------------------------------------------

TOOL_TEXT = ('<tool_call>{"name": "get_weather", '
             '"arguments": {"city": "Paris"}}</tool_call>')
TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}}}}}]


class _SlowSupervisor:
    """确定 token 流，可被 abort() 打断并记录 abort 次数。"""

    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self, tokens=None, delay: float = 0.0):
        self.tokens = list(tokens) if tokens is not None else ["t0 ", "t1 ", "t2 "]
        self.delay = delay
        self.aborts = 0
        self._aborted = threading.Event()
        self.model_name = "fake"

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        for token in self.tokens:
            if self._aborted.is_set():
                break
            yield token
            if self.delay:
                time.sleep(self.delay)

    def abort(self):
        self.aborts += 1
        self._aborted.set()

    def shutdown(self):
        pass


class _AbortedSupervisor:
    """生成到一半就抛 RequestAborted（模拟 worker 回 aborted end 帧）。"""

    state = "ready"
    metrics = {"worker_restarts_total": 0}
    model_name = "fake"

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        yield "半"
        raise RequestAborted("inference aborted before completion "
                             "(cancelled or reset)")

    def abort(self):
        pass

    def shutdown(self):
        pass


def _http_client(supervisor):
    cfg = load_config(env={})
    app = create_app(cfg, supervisor=supervisor)
    return TestClient(app, raise_server_exceptions=False)


def _pipeline_svc():
    """满足 InferenceGateway 协议的最小 svc（只测纯函数收尾帧）。"""
    return SimpleNamespace(
        model_name="fake", tool_parser="auto", max_tool_calls=4,
        tools_mode="prompt", system_prompt="", context_len=0,
        default_max_tokens=0, max_tokens_limit=0, json_hint="",
        supervisor=None, queue=None,
        ttft_log=lambda seconds: None, default_sampling=None)


def _wait_pump_stops(stream: TokenStream, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    thread = stream._thread
    while thread is not None and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert thread is not None and not thread.is_alive(), "泵线程未退出"


def _drain(dqueue: asyncio.Queue) -> list:
    items = []
    while not dqueue.empty():
        items.append(dqueue.get_nowait())
    return items


# ---------------------------------------------------------------------------
# 1) P0：流式 chat + tools
# ---------------------------------------------------------------------------

def test_streaming_chat_tools_nested_function_and_done():
    """此前 trailing_frames 读扁平 c["name"] → KeyError：无 finish_reason/无 [DONE]。"""
    sup = _SlowSupervisor(tokens=list(TOOL_TEXT))
    with _http_client(sup) as client:
        with client.stream(
                "POST", "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "巴黎天气"}],
                      "stream": True, "tools": TOOLS}) as resp:
            assert resp.status_code == 200
            lines = [line for line in resp.iter_lines()
                     if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]", "流式 tools 必须正常收尾"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    calls = [call
             for chunk in chunks
             for call in (chunk["choices"][0]["delta"].get("tool_calls") or [])]
    assert calls, "缺少 tool_calls 增量帧"
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    assert "Paris" in calls[0]["function"]["arguments"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_trailing_frames_reads_nested_and_flat_compat():
    pipe = Pipeline(_pipeline_svc(),
                    {"messages": [{"role": "user", "content": "x"}]}, chat=True)
    session = SseSession(model="m")

    # OpenAI 标准嵌套结构（parser 的真实产出）
    pipe._tool_calls = [{"index": 0, "id": "call_1", "type": "function",
                         "function": {"name": "f",
                                      "arguments": '{"a": 1}'}}]
    frames = pipe.trailing_frames(session)
    delta = json.loads(frames[0][6:])["choices"][0]["delta"]["tool_calls"][0]
    assert delta["function"] == {"name": "f", "arguments": '{"a": 1}'}
    assert delta["id"] == "call_1" and delta["index"] == 0

    # 迁移期防御：早期扁平结构也不能再 500
    pipe._tool_calls = [{"index": 1, "id": "call_2", "type": "function",
                         "name": "g", "arguments": "{}"}]
    delta = json.loads(
        pipe.trailing_frames(session)[0][6:])["choices"][0]["delta"]["tool_calls"][0]
    assert delta["function"] == {"name": "g", "arguments": "{}"}


# ---------------------------------------------------------------------------
# 2) CancelToken：只有持槽才允许 abort
# ---------------------------------------------------------------------------

def test_cancel_token_aborts_only_while_slot_held():
    sup = _SlowSupervisor()
    token = CancelToken(sup)
    token.cancel()
    assert token.abort_once() is False
    assert sup.aborts == 0, "排队期（未持槽）不得 abort 在飞请求"

    token.hold_slot()
    assert token.slot_held is True
    assert token.abort_once() is True
    assert sup.aborts == 1
    assert token.abort_once() is False, "abort 必须幂等"
    assert sup.aborts == 1

    token.release_slot()
    assert token.slot_held is False
    assert token.abort_once() is False
    assert sup.aborts == 1, "槽归还后不得再 abort（否则会误伤下一个请求）"


def test_queued_cancel_does_not_abort_inflight():
    """排队期断连：只退出排队、不调用 supervisor.abort()。"""
    sup = _SlowSupervisor()
    queue = InferenceQueue(max_wait=8, wait_timeout_s=5)
    svc = SimpleNamespace(queue=queue, supervisor=sup)
    pipe = SimpleNamespace(iter_tokens=lambda: iter(["a", "b"]))
    queue._slot.acquire()          # 模拟另一个请求正持有 NPU 槽

    try:
        async def main():
            loop = asyncio.get_running_loop()
            dqueue: asyncio.Queue = asyncio.Queue()
            stream = TokenStream(svc, pipe)
            stream.start(loop, dqueue)
            for _ in range(300):
                if queue.queue_depth >= 1:
                    break
                await asyncio.sleep(0.01)
            assert queue.queue_depth == 1, "请求应处于排队状态"

            stream.cancel()        # 客户端在排队期断开
            for _ in range(600):
                if not stream._thread.is_alive():
                    break
                await asyncio.sleep(0.01)
            assert not stream._thread.is_alive()
            assert sup.aborts == 0, "排队期断连 abort 了无关的在飞请求"
            assert STREAM_DONE in _drain(dqueue)

        asyncio.run(main())
    finally:
        queue._slot.release()


def test_inflight_cancel_aborts_worker_once():
    """持槽期间取消：必须 abort（且只 abort 一次），泵随之退出。"""
    sup = _SlowSupervisor(tokens=["t%d " % i for i in range(200)], delay=0.005)
    queue = InferenceQueue(max_wait=8, wait_timeout_s=5)
    svc = SimpleNamespace(queue=queue, supervisor=sup)
    pipe = SimpleNamespace(iter_tokens=lambda: sup.execute("p"))

    async def main():
        loop = asyncio.get_running_loop()
        dqueue: asyncio.Queue = asyncio.Queue()
        stream = TokenStream(svc, pipe)
        stream.start(loop, dqueue)
        first = await asyncio.wait_for(dqueue.get(), 3)
        assert first == "t0 "        # 已拿到槽并开始生成
        assert stream.token.slot_held is True, "生成期必须处于持槽状态"
        stream.cancel()
        for _ in range(400):
            if not stream._thread.is_alive():
                break
            await asyncio.sleep(0.01)
        assert not stream._thread.is_alive()
        assert sup.aborts == 1
        assert STREAM_DONE in _drain(dqueue)

    asyncio.run(main())


# ---------------------------------------------------------------------------
# 2b) 执行槽回调：on_acquire / on_release 严格成对，且只在真正开始推理时触发
# ---------------------------------------------------------------------------

def test_queue_slot_callbacks_paired_on_normal_run():
    queue = InferenceQueue(max_wait=4, wait_timeout_s=5)
    events: list[str] = []
    items = list(queue.run(
        lambda: iter(["x"]),
        on_acquire=lambda: events.append("acquire"),
        on_release=lambda: events.append("release")))
    assert items == ["x"]
    assert events == ["acquire", "release"], "持槽/放槽回调必须成对"


def test_queue_slot_callbacks_skipped_when_cancelled_before_start():
    """拿到槽后才发现已取消：不 submit()、不触发任何槽回调、槽不泄漏。"""
    queue = InferenceQueue(max_wait=4, wait_timeout_s=5)
    events: list[str] = []
    state = {"n": 0}

    def cancelled() -> bool:
        state["n"] += 1
        return state["n"] >= 2      # 排队循环里 False（拿槽），拿槽后再查为 True

    submitted: list[int] = []
    with pytest.raises(QueueTimeout):
        list(queue.run(lambda: submitted.append(1) or iter(["x"]),
                       cancelled=cancelled,
                       on_acquire=lambda: events.append("acquire"),
                       on_release=lambda: events.append("release")))
    assert submitted == [], "已取消的请求不得 submit（不写 run 帧、不启动 NPU）"
    assert events == [], "未进入推理时两个回调都不能触发（严格配对）"
    # 槽必须已归还：后续请求可以正常执行
    assert list(queue.run(lambda: iter(["y"]))) == ["y"]


def test_queue_slot_callbacks_skipped_on_busy():
    queue = InferenceQueue(max_wait=0)
    events: list[str] = []
    queue._slot.acquire()
    try:
        with pytest.raises(QueueBusy):
            list(queue.run(lambda: iter(["x"]),
                           on_acquire=lambda: events.append("acquire"),
                           on_release=lambda: events.append("release")))
    finally:
        queue._slot.release()
    assert events == []
    assert list(queue.run(lambda: iter(["y"]))) == ["y"]


# ---------------------------------------------------------------------------
# 3) 背压截断必须 abort（归还 NPU）
# ---------------------------------------------------------------------------

def test_backpressure_truncation_aborts_worker(monkeypatch):
    """消费停滞（_offer 满）→ TRUNCATED，且必须在持槽期间 abort worker。"""
    monkeypatch.setattr(streaming_mod, "ITEM_QUEUE_MAXSIZE", 0)
    sup = _SlowSupervisor(tokens=["t0 ", "t1 ", "t2 "])
    queue = InferenceQueue(max_wait=8, wait_timeout_s=5)
    svc = SimpleNamespace(queue=queue, supervisor=sup)
    pipe = SimpleNamespace(iter_tokens=lambda: sup.execute("p"))

    async def main():
        loop = asyncio.get_running_loop()
        dqueue: asyncio.Queue = asyncio.Queue()
        stream = TokenStream(svc, pipe)
        stream.start(loop, dqueue)
        for _ in range(400):
            if not stream._thread.is_alive():
                break
            await asyncio.sleep(0.01)
        assert not stream._thread.is_alive()
        items = _drain(dqueue)
        assert TRUNCATED in items
        assert STREAM_DONE in items
        assert sup.aborts == 1, "背压截断没有 abort worker（只释放了槽位）"

    asyncio.run(main())


# ---------------------------------------------------------------------------
# 4) OP_END.aborted 透传 + abort 帧带 run id
# ---------------------------------------------------------------------------

def _socketpair_client():
    left, right = socket.socketpair()
    client = WorkerClient(left)
    rfile = right.makefile("rb")
    wfile = right.makefile("wb")
    return client, right, rfile, wfile


def test_worker_client_end_aborted_raises_request_aborted():
    client, right, rfile, wfile = _socketpair_client()

    def worker():
        frame = read_frame(rfile)
        assert frame["op"] == "run"
        write_frame(wfile, {"op": "token", "id": frame["id"], "text": "hi"})
        write_frame(wfile, {"op": OP_END, "id": frame["id"], "aborted": True})

    threading.Thread(target=worker, daemon=True).start()
    got = []
    try:
        with pytest.raises(RequestAborted):
            for item in client.run("prompt", request_timeout_s=5):
                got.append(item)
        assert got == ["hi"], "aborted 前已产出的 token 仍应交付"
    finally:
        client.close()
        right.close()


def test_worker_client_normal_end_still_finishes():
    """aborted 透传不能误伤正常 end（含 perf 元数据）。"""
    client, right, rfile, wfile = _socketpair_client()

    def worker():
        frame = read_frame(rfile)
        write_frame(wfile, {"op": OP_END, "id": frame["id"],
                            "perf": {"prefill_tokens": 3, "generate_tokens": 2}})

    threading.Thread(target=worker, daemon=True).start()
    try:
        items = list(client.run("prompt", request_timeout_s=5))
        assert items == [{"perf": {"prefill_tokens": 3, "generate_tokens": 2}}]
    finally:
        client.close()
        right.close()


def test_abort_frame_carries_last_run_id():
    client, right, rfile, wfile = _socketpair_client()
    seen: dict = {}

    def worker():
        run_frame = read_frame(rfile)
        seen["run"] = run_frame
        write_frame(wfile, {"op": OP_END, "id": run_frame["id"]})
        seen["abort"] = read_frame(rfile)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        assert list(client.run("prompt", request_timeout_s=5)) == []
        client.abort()
        thread.join(5)
        assert not thread.is_alive()
        assert seen["abort"]["op"] == "abort"
        assert seen["abort"].get("id") == seen["run"]["id"], (
            "abort 帧必须带最近一次 run 的 id（worker 侧匹配的前提）")
    finally:
        client.close()
        right.close()


def test_request_aborted_does_not_restart_worker():
    """RequestAborted 是请求级失败：execute() 不得当连接死亡去重启 worker。"""
    from rkllm.server import supervisor as sup_mod

    sup = sup_mod.Supervisor(["fake"])
    sup.state = "ready"
    restarted: list[int] = []

    class _Client:
        def run(self, prompt, **kwargs):
            yield "半"
            raise RequestAborted("aborted")

    sup._client = _Client()
    sup.restart = lambda: restarted.append(1)
    got = []
    with pytest.raises(RequestAborted):
        for item in sup.execute("p"):
            got.append(item)
    assert got == ["半"]
    assert restarted == [], "请求被打断不应触发 worker 重启（会白烧 14s 加载）"


# ---------------------------------------------------------------------------
# 5) HTTP 层：aborted 不能伪装成 200 + finish_reason=stop
# ---------------------------------------------------------------------------

def test_http_stream_aborted_emits_error_frame_not_fake_stop():
    with _http_client(_AbortedSupervisor()) as client:
        with client.stream(
                "POST", "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "x"}],
                      "stream": True}) as resp:
            assert resp.status_code == 200
            lines = [line for line in resp.iter_lines()
                     if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert any("error" in chunk for chunk in chunks), (
        "被打断的流必须以错误帧收尾，不能伪装正常结束")
    stops = [chunk["choices"][0].get("finish_reason")
             for chunk in chunks if chunk.get("choices")]
    assert "stop" not in stops, "aborted 不能报 finish_reason=stop"


def test_http_nonstream_aborted_is_503_worker_down():
    with _http_client(_AbortedSupervisor()) as client:
        r = client.post("/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "worker_down"
    assert "aborted" in err["message"]


# ---------------------------------------------------------------------------
# 6) 端到端：排队客户端断开不得打断正在生成的那个请求
# ---------------------------------------------------------------------------

def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _post_stream(port: int, body: dict) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", port), timeout=30)
    payload = json.dumps(body).encode()
    sock.sendall(("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                  "Content-Type: application/json\r\n"
                  "Content-Length: %d\r\n\r\n" % len(payload)).encode() + payload)
    return sock


def _recv_until(sock: socket.socket, needle: bytes, timeout: float = 15.0) -> bytes:
    sock.settimeout(timeout)
    buf = b""
    while needle not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
    return buf


def test_queued_disconnect_does_not_abort_inflight_http():
    import uvicorn

    tokens = ["t%d " % i for i in range(80)]
    sup = _SlowSupervisor(tokens=tokens, delay=0.03)   # A 约 2.4 s，留足排队窗口
    app = create_app(load_config(env={}), supervisor=sup)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error", access_log=False))
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()),
                              daemon=True, name="cancel-fix-http")
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn 未能启动"

    a = None
    b = None
    try:
        a = _post_stream(port, {"messages": [{"role": "user", "content": "A"}],
                                "stream": True})
        head = _recv_until(a, b"data: ", timeout=5)
        assert b"data: " in head
        time.sleep(0.15)           # A 正在生成

        b = _post_stream(port, {"messages": [{"role": "user", "content": "B"}],
                                "stream": True})
        _recv_until(b, b"data: ", timeout=5)   # B 已拿到首帧（随后进队列）
        # 确定 B 真的处于"排队等待"状态，避免慢机器上 A 已结束使用例失义
        queue_depth = app.state.svc.queue.queue_depth
        deadline = time.monotonic() + 5
        while queue_depth < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
            queue_depth = app.state.svc.queue.queue_depth
        assert queue_depth >= 1, "B 未进入排队，用例前提不成立"
        b.close()                             # 排队期断连
        b = None
        time.sleep(0.3)                       # 断连传导窗口（旧代码此刻 abort 了 A）

        rest = _recv_until(a, b"data: [DONE]", timeout=20)
        body = head + rest
        assert sup.aborts == 0, "排队请求断连打断了无关的在飞请求（P1-1 回归）"
        assert body.count(b"data: [DONE]") == 1
        deltas = [json.loads(line[6:]) for line in body.split(b"\n")
                  if line.startswith(b"data: ") and b"[DONE]" not in line]
        text = "".join(d["choices"][0]["delta"].get("content", "")
                       for d in deltas)
        assert text == "".join(tokens), "在飞请求被截断"
        assert deltas[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        for sock in (a, b):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        server.should_exit = True
        thread.join(5)
