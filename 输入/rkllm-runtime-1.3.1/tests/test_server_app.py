"""S2 服务端单测：schemas 校验、SSE 帧一致性（D4）、队列背压、HTTP 端到端
（fake supervisor，不经 NPU）。"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm import caps
from rkllm.config import load_config
from rkllm.server.app import create_app
from rkllm.server.queue import InferenceQueue, QueueBusy
from rkllm.server.schemas import ChatCompletionRequest, CompletionRequest
from rkllm.server.sse import SseSession


# -- 参数策略（D2/D7）---------------------------------------------------------


def test_schema_rejects_bad_temperature():
    with pytest.raises(Exception):
        ChatCompletionRequest(messages=[{"role": "user", "content": "x"}],
                              temperature=3.5)


def test_schema_rejects_extra_fields():
    with pytest.raises(Exception):
        ChatCompletionRequest(messages=[{"role": "user", "content": "x"}],
                              modalities=["audio"])


def test_caps_reject_and_warn_paths():
    reject, warn = caps.check_request_params({"seed": 1}, policy="warn")
    assert warn == ["seed"] and not reject
    reject, _ = caps.check_request_params({"seed": 1}, policy="reject")
    assert reject == ["seed"]


# -- SSE 帧一致性（D4：id/created 全程一致）-------------------------------------


def test_sse_session_fixed_identity():
    s = SseSession(model="m")
    f1 = s.first_frame()
    f2 = s.delta_frame(content="hi")
    f3 = s.finish_frame("stop")
    ids = [json.loads(f[6:])["id"] for f in (f1, f2, f3)]
    created = [json.loads(f[6:])["created"] for f in (f1, f2, f3)]
    assert len(set(ids)) == 1 and len(set(created)) == 1
    assert json.loads(f3[6:])["choices"][0]["finish_reason"] == "stop"
    assert s.done() == "data: [DONE]\n\n"
    assert s.heartbeat() == ": ping\n\n"


# -- 队列 ---------------------------------------------------------------------


def test_queue_rejects_when_full():
    q = InferenceQueue(max_wait=1, wait_timeout_s=1, busy_retry_after_s=3)
    gate = threading.Event()
    results = []

    def hold_tokens():
        # 占住执行槽：submit() 在拿到槽后被调用，等 gate 放行
        def submit():
            gate.wait(5)
            return iter(["x"])
        list(q.run(submit))

    threads = [threading.Thread(target=hold_tokens, daemon=True) for _ in range(2)]
    threads[0].start()
    time.sleep(0.3)          # t0 拿到槽并阻塞在 gate
    threads[1].start()
    time.sleep(0.3)          # t1 占据唯一等待位
    try:
        list(q.run(lambda: iter(["y"])))
        raise AssertionError("expected QueueBusy")
    except QueueBusy as exc:
        assert exc.retry_after_s == 3
    gate.set()
    threads[0].join(6)
    threads[1].join(6)


# -- HTTP 端到端（fake supervisor）---------------------------------------------


class FakeSupervisor:
    """确定 token 流；不 spawn 子进程（帧协议已被 test_worker_frames 覆盖）。"""

    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self, tokens=None):
        self.tokens = tokens or ["你", "好"]
        self.model_name = "fake"

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        for t in self.tokens:
            yield t

    def abort(self):
        pass


@pytest.fixture()
def client():
    cfg = load_config(env={})
    app = create_app(cfg, supervisor=FakeSupervisor())
    with TestClient(app) as c:
        yield c


def test_health_ready_metrics(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["context_len"] == cfg_context()
    assert r.json()["max_queue"] == 32
    assert client.get("/ready").status_code == 200
    m = client.get("/metrics").text
    assert "rkllm_requests_total" in m and "rkllm_queue_depth" in m


def cfg_context():
    return 4096


def test_capabilities_endpoint(client):
    caps_json = client.get("/v1/capabilities").json()
    assert caps_json["endpoints"]["chat.completions"] == "native"
    assert "tools" in caps_json["params"]["emulated"]


def test_models_endpoints(client):
    data = client.get("/v1/models").json()["data"]
    assert len(data) == 1
    assert client.get("/v1/models/whatever").json()["id"] == "rkllm"


def test_chat_completions_non_stream(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "打个招呼"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "你好"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["estimated"] is True


def test_chat_completions_stream(client):
    with client.stream("POST", "/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "x"}],
                             "stream": True}) as resp:
        lines = [l for l in resp.iter_lines() if l.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1] if l[6:] != "[DONE]"]
    ids = {c["id"] for c in chunks if "id" in c}
    assert len(ids) == 1                          # D4：流内 id 唯一
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_rejects_unsupported_param(client):
    """默认 warn 策略：logprobs/seed 走响应头放行；json_schema 恒 reject。"""
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x"}], "logprobs": True})
    assert r.status_code == 200
    assert r.headers.get("x-rkllm-unsupported") == "logprobs"
    r2 = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x"}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "o", "schema": {}}}})
    assert r2.status_code == 400
    assert r2.json()["error"]["code"] == "unsupported_parameter"


def test_chat_warn_param_gets_header(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x"}], "seed": 42})
    assert r.status_code == 200
    assert r.headers.get("x-rkllm-unsupported") == "seed"


def test_completions_thin_wrapper(client):
    r = client.post("/v1/completions", json={"prompt": "hi"})
    assert r.status_code == 200
    assert r.json()["choices"][0]["text"] == "你好"


def test_param_validation_400(client):
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 422                   # pydantic 校验失败
