"""OpenAI 契约 P1 修复回归（task-5）。

覆盖：
1. 非流式 `reasoning_content` 输出（原实现只累积不输出）；
2. 数据面 503 的 `Retry-After` 必须是 HTTP 头（chat / responses 都验）；
3. 参数校验失败 → OpenAI 400 `{"error": {...}}`（不再是 FastAPI 422 detail）；
4. Responses `max_output_tokens` 截断 → `status=incomplete` +
   `incomplete_details.reason=max_output_tokens`（流式/非流式）；
5. Responses 输入校验（input/max_output_tokens/NaN-Inf temperature）不 500；
6. 低风险项：json_schema 去围栏、tool_choice 明确语义、根路径鉴权。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.config import load_config                    # noqa: E402
from rkllm.server.app import create_app                 # noqa: E402
from rkllm.server.supervisor import WorkerDown          # noqa: E402


class FakeSupervisor:
    """确定性 token 流；`perf` 模拟 worker end 帧（finish_reason/usage 依赖）。"""

    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self, tokens=None, perf=None, raise_after=None):
        self.tokens = tokens if tokens is not None else ["你", "好"]
        self.perf = perf
        self.raise_after = raise_after
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        for idx, token in enumerate(self.tokens, start=1):
            yield token
            if self.raise_after is not None and idx >= self.raise_after:
                raise WorkerDown("simulated worker down")
        if self.perf:
            yield {"perf": self.perf}

    def abort(self):
        pass


def _client(tokens=None, perf=None, raise_after=None, env=None):
    cfg = load_config(env=env or {})
    sup = FakeSupervisor(tokens, perf, raise_after)
    app = create_app(cfg, supervisor=sup)
    return TestClient(app), app, sup


def _busy(app):
    """把队列置为“等待位已满”，触发 QueueBusy（不真占 NPU 槽）。"""
    svc = app.state.svc
    svc.queue._waiting = svc.queue.max_wait


def _openai_error(resp, status: int, *, err_type: str = "invalid_request_error"):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert "detail" not in body, body          # 不再是 FastAPI 422 形状
    err = body["error"]
    assert err["type"] == err_type, err
    assert err["message"], err
    assert "param" in err and "code" in err, err
    return err


def _sse_events(resp):
    events, frames = [], []
    for line in resp.iter_lines():
        if line.startswith("event: "):
            events.append(line[7:].strip())
        elif line.startswith("data: "):
            data = line[6:]
            if data != "[DONE]":
                frames.append(json.loads(data))
    return events, frames


MSG = {"messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------------------
# 1. 非流式 reasoning_content
# ---------------------------------------------------------------------------
def test_nonstream_reasoning_content_exposed():
    c, _, sup = _client(tokens=["<think>推理</think>答案"])
    r = c.post("/v1/chat/completions",
               json={**MSG, "reasoning_effort": "high"})
    assert r.status_code == 200, r.text
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "答案"
    assert msg["reasoning_content"] == "推理"
    # reasoning_effort 仍真实传给 runtime（修复只影响输出侧）
    assert sup.kwargs[-1]["enable_thinking"] is True


def test_nonstream_reasoning_respects_configured_field():
    c, _, _ = _client(tokens=["<think>r</think>a"],
                      env={"RKLLM_OPENAI_REASONING_FIELD": "reasoning"})
    r = c.post("/v1/chat/completions",
               json={**MSG, "reasoning_effort": "low"})
    assert r.status_code == 200, r.text
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "a"
    assert msg["reasoning"] == "r"
    assert "reasoning_content" not in msg


def test_nonstream_reasoning_hidden_when_field_none():
    c, _, _ = _client(tokens=["<think>r</think>a"],
                      env={"RKLLM_OPENAI_REASONING_FIELD": "none"})
    r = c.post("/v1/chat/completions",
               json={**MSG, "reasoning_effort": "low"})
    assert r.status_code == 200, r.text
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "a"
    assert "reasoning_content" not in msg and "reasoning" not in msg


# ---------------------------------------------------------------------------
# 2. 503 的 Retry-After 必须是 HTTP 头
# ---------------------------------------------------------------------------
def test_chat_queue_full_retry_after_header():
    c, app, _ = _client(tokens=["a"])
    _busy(app)
    r = c.post("/v1/chat/completions", json=MSG)
    assert r.status_code == 503, r.text
    assert r.headers.get("retry-after") == "5"
    err = r.json()["error"]
    assert err["code"] == "queue_full"
    assert "Retry-After" not in err          # 头信息不再塞进 body


def test_responses_queue_full_retry_after_header():
    c, app, _ = _client(tokens=["a"])
    _busy(app)
    r = c.post("/v1/responses", json={"input": "x"})
    assert r.status_code == 503, r.text
    assert r.headers.get("retry-after") == "5"
    assert r.json()["error"]["code"] == "queue_full"


def test_chat_worker_down_retry_after_header():
    c, _, _ = _client(tokens=["a"], raise_after=1)
    r = c.post("/v1/chat/completions", json=MSG)
    assert r.status_code == 503, r.text
    assert r.headers.get("retry-after") == "5"
    assert r.json()["error"]["code"] == "worker_down"


# ---------------------------------------------------------------------------
# 3. 参数校验失败 → OpenAI 400
# ---------------------------------------------------------------------------
def test_validation_error_bad_temperature_is_openai_400():
    c, _, _ = _client(tokens=["a"])
    r = c.post("/v1/chat/completions", json={**MSG, "temperature": 3})
    err = _openai_error(r, 400)
    assert err["param"] == "temperature"


def test_validation_error_empty_messages_is_openai_400():
    c, _, _ = _client(tokens=["a"])
    r = c.post("/v1/chat/completions", json={"messages": []})
    err = _openai_error(r, 400)
    assert err["param"] == "messages"


def test_validation_error_unknown_field_is_openai_400():
    c, _, _ = _client(tokens=["a"])
    r = c.post("/v1/chat/completions", json={**MSG, "totally_unknown": 1})
    err = _openai_error(r, 400)
    assert err["param"] == "totally_unknown"


def test_validation_error_bad_json_is_openai_400():
    c, _, _ = _client(tokens=["a"])
    r = c.post("/v1/chat/completions", content=b"{bad json",
               headers={"Content-Type": "application/json"})
    _openai_error(r, 400)


def test_nan_inf_temperature_rejected_before_runtime():
    c, _, sup = _client(tokens=["a"])
    for literal in (b"NaN", b"Infinity", b"-Infinity"):
        raw = b'{"messages": [{"role": "user", "content": "hi"}], ' \
              b'"temperature": ' + literal + b"}"
        r = c.post("/v1/chat/completions", content=raw,
                   headers={"Content-Type": "application/json"})
        _openai_error(r, 400)
    assert not sup.prompts          # 未进入 Pipeline/runtime


# ---------------------------------------------------------------------------
# 4. Responses max_output_tokens → incomplete
# ---------------------------------------------------------------------------
TRUNC_PERF = {"prefill_tokens": 5, "generate_tokens": 3}


def test_responses_nonstream_max_tokens_incomplete():
    c, _, _ = _client(tokens=["a", "b", "c"], perf=TRUNC_PERF)
    r = c.post("/v1/responses",
               json={"input": "x", "max_output_tokens": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "incomplete", body
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}
    assert body["output"][0]["status"] == "incomplete"


def test_responses_stream_max_tokens_incomplete():
    c, _, _ = _client(tokens=["a", "b", "c"], perf=TRUNC_PERF)
    with c.stream("POST", "/v1/responses",
                  json={"input": "x", "stream": True,
                        "max_output_tokens": 3}) as r:
        assert r.status_code == 200, r.text
        events, frames = _sse_events(r)
    assert events[-1] == "response.incomplete", events
    final = frames[-1]["response"]
    assert final["status"] == "incomplete"
    assert final["incomplete_details"] == {"reason": "max_output_tokens"}


def test_responses_normal_completion_stays_completed():
    c, _, _ = _client(tokens=["你", "好"])
    r = c.post("/v1/responses", json={"input": "x"})
    body = r.json()
    assert body["status"] == "completed"
    assert body["incomplete_details"] is None


# ---------------------------------------------------------------------------
# 5. Responses 输入校验 / NaN-Inf 不进 runtime
# ---------------------------------------------------------------------------
def test_responses_input_type_validation():
    c, _, sup = _client(tokens=["a"])
    r = c.post("/v1/responses", json={"input": 123})
    err = _openai_error(r, 400)
    assert err["param"] == "input" and err["code"] == "invalid_input"

    r = c.post("/v1/responses", json={"input": [123]})
    err = _openai_error(r, 400)
    assert err["param"] == "input"

    assert not sup.prompts


def test_responses_max_output_tokens_validation():
    c, _, sup = _client(tokens=["a"])
    for bad in ("abc", -5, 0, True):
        r = c.post("/v1/responses",
                   json={"input": "x", "max_output_tokens": bad})
        err = _openai_error(r, 400)
        assert err["param"] == "max_output_tokens", (bad, err)
    assert not sup.prompts


def test_responses_nan_inf_temperature_rejected():
    c, _, sup = _client(tokens=["a"])
    for literal in (b"NaN", b"Infinity", b"-Infinity"):
        raw = b'{"input": "x", "temperature": ' + literal + b"}"
        r = c.post("/v1/responses", content=raw,
                   headers={"Content-Type": "application/json"})
        _openai_error(r, 400)
    assert not sup.prompts


def test_responses_tools_shape_validation():
    c, _, sup = _client(tokens=["a"])
    r = c.post("/v1/responses", json={"input": "x", "tools": {"a": 1}})
    err = _openai_error(r, 400)
    assert err["param"] == "tools"
    assert not sup.prompts


# ---------------------------------------------------------------------------
# 6a. json_schema 非流式：content/output_text 本身是合法 JSON
# ---------------------------------------------------------------------------
JSON_SCHEMA = {"type": "json_schema",
               "json_schema": {"name": "o",
                               "schema": {"type": "object",
                                          "required": ["a"]}}}


def test_chat_json_schema_returns_plain_json():
    c, _, _ = _client(tokens=['```json\n{"a": 1}\n```'])
    r = c.post("/v1/chat/completions",
               json={**MSG, "response_format": JSON_SCHEMA})
    assert r.status_code == 200, r.text
    content = r.json()["choices"][0]["message"]["content"]
    assert json.loads(content) == {"a": 1}


def test_responses_json_schema_output_is_json():
    c, _, _ = _client(tokens=['```json\n{"a": 1}\n```'])
    r = c.post("/v1/responses", json={
        "input": "x",
        "text": {"format": {"type": "json_schema", "name": "o",
                            "schema": {"type": "object",
                                       "required": ["a"]}}}})
    assert r.status_code == 200, r.text
    assert json.loads(r.json()["output_text"]) == {"a": 1}


# ---------------------------------------------------------------------------
# 6b. tool_choice 明确语义
# ---------------------------------------------------------------------------
TOOL = {"type": "function", "function": {"name": "f"}}


def test_tool_choice_required_is_explicit_400():
    c, _, _ = _client(tokens=["a"])
    r = c.post("/v1/chat/completions",
               json={**MSG, "tools": [TOOL], "tool_choice": "required"})
    err = _openai_error(r, 400)
    assert err["param"] == "tool_choice"
    assert err["code"] == "unsupported_parameter"

    r = c.post("/v1/responses", json={
        "input": "x", "tools": [{"type": "function", "name": "f"}],
        "tool_choice": "required"})
    err = _openai_error(r, 400)
    assert err["param"] == "tool_choice"
    assert err["code"] == "unsupported_parameter"


def test_tool_choice_none_suppresses_tool_calls():
    tokens = ["<tool_call>", '{"name": "f", "arguments": {}}',
              "</tool_call>"]
    c, _, sup = _client(tokens=tokens)
    r = c.post("/v1/chat/completions",
               json={**MSG, "tools": [TOOL], "tool_choice": "none"})
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert "tool_calls" not in choice["message"]
    assert choice["finish_reason"] != "tool_calls"
    assert "Available tools" not in sup.prompts[-1]   # 工具提示词未注入


# ---------------------------------------------------------------------------
# 6c. 根路径鉴权
# ---------------------------------------------------------------------------
def test_root_is_protected_when_api_key_set():
    c, _, _ = _client(env={"RKLLM_SERVER_API_KEY": "secret"})
    r = c.get("/")
    assert r.status_code == 401, r.text
    assert r.json()["error"]["type"] == "authentication_error"

    ok = c.get("/", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    assert ok.json()["service"] == "rkllm-server"
    assert c.get("/ready").status_code == 200       # 仍豁免


def test_root_public_when_auth_disabled():
    c, _, _ = _client()
    r = c.get("/")
    assert r.status_code == 200
    assert r.json()["endpoints"]["models"] == ["/v1/models", "/models"]
