"""P2-1 回归：OpenAI / 流式契约（真实路径，不 mock SSE 生成器）。

仿真口径（与 tests/test_p1_completions_stream.py 一致）：
- 真实 `load_config()`（默认配置 + 环境变量覆盖）→ 真实 `create_app()` →
  FastAPI `TestClient`（完整 ASGI：路由 → StreamingResponse → SSE 字节流）；
- 唯一替身是 supervisor 边界（板端 NPU 不可用），token 按 worker 帧协议形状
  产出（含 `{"perf": ...}` 元数据）；
- 客户端侧用 pydantic 严格复刻 openai-python 的响应类型做反序列化，形状不对
  直接 ValidationError；Responses 事件同样按 `{type, sequence_number}` 校验。

覆盖（对应 task-13）：
1. 405/401/404/500 等 HTTP 错误统一 OpenAI `{"error": {...}}` 形状；
2. chat 的 usage-only 帧 `choices: []`（legacy completions 已由 P1-O1 覆盖）；
3. reasoning 流式增量（chat `delta.reasoning_content` / Responses
   `response.reasoning_text.*` + reasoning item），不再终帧一次性回填；
4. json_object 非流式校验+重试；非法 json_schema / `text.format` 明确 400；
5. `parallel_tool_calls` 从 caps emulated 降级为 unsupported；
   `repeat_penalty < 1.0` 明确 400（不再静默改成 1.1）；
6. `/ready` 不泄露模型名；`/v1/models` 补 `created`。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm import caps                                        # noqa: E402
from rkllm.config import load_config                          # noqa: E402
from rkllm.server.app import create_app                       # noqa: E402

MSG = {"messages": [{"role": "user", "content": "hi"}]}
TOOL = {"type": "function", "function": {"name": "f",
                                         "parameters": {"type": "object"}}}


# ---------------------------------------------------------------------------
# 假的 supervisor 边界（真实帧协议形状；可脚本化多轮）
# ---------------------------------------------------------------------------


class FakeSupervisor:
    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self, tokens=None, perf=None, error=None):
        self.tokens = list(tokens if tokens is not None else ["你", "好"])
        self.perf = perf
        self.error = error
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def is_ready(self):
        return self.state == "ready"

    def execute(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        if self.error is not None:
            raise self.error
        for token in self.tokens:
            yield token
        if self.perf:
            yield {"perf": self.perf}

    def abort(self):
        pass

    def shutdown(self):
        pass


class ScriptedSupervisor(FakeSupervisor):
    """每次 execute 依次消费一份 token 脚本（模拟模型重试后改好输出）。"""

    def __init__(self, scripts, perf=None):
        super().__init__(tokens=[], perf=perf)
        self.scripts = [list(script) for script in scripts]
        self.calls = 0

    def execute(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        index = min(self.calls, len(self.scripts) - 1)
        self.calls += 1
        for token in self.scripts[index]:
            yield token


class NotReadySupervisor(FakeSupervisor):
    state = "loading"

    def is_ready(self):
        return False


def _client(supervisor=None, env=None, *, raise_server_exceptions=True):
    cfg = load_config(env=env or {})
    app = create_app(cfg, supervisor=supervisor or FakeSupervisor())
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def _sse(resp):
    """返回 (原始 data 字典列表, 是否恰好一个结尾 [DONE])。"""
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    lines = list(resp.iter_lines())
    data_lines = [line for line in lines if line.startswith("data:")]
    done = [line for line in data_lines if line.strip() == "data: [DONE]"]
    assert data_lines and data_lines[-1].strip() == "data: [DONE]", (
        "[DONE] 不是最后一个 data 行：%r" % (data_lines[-1:],))
    assert len(done) == 1, "data: [DONE] 必须恰好出现一次"
    frames = [json.loads(line[len("data:"):].strip())
              for line in data_lines if line.strip() != "data: [DONE]"]
    return frames


def _response_events(resp):
    """解析 Responses SSE：返回 [(event_name, payload)]。"""
    assert resp.status_code == 200, resp.text
    events: list[tuple[str, dict]] = []
    current = None
    for line in resp.iter_lines():
        if line.startswith("event: "):
            current = line[7:].strip()
        elif line.startswith("data: ") and current:
            data = line[6:].strip()
            if data != "[DONE]":
                events.append((current, json.loads(data)))
    return events


# ---------------------------------------------------------------------------
# 严格客户端模型（等价 openai-python 的必填字段）
# ---------------------------------------------------------------------------


class StrictUsage(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class StrictError(BaseModel):
    model_config = ConfigDict(extra="allow")
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class StrictErrorResponse(BaseModel):
    error: StrictError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    object: Literal["model"]
    created: int
    owned_by: str


class StrictModelList(BaseModel):
    model_config = ConfigDict(extra="allow")
    object: Literal["list"]
    data: list[StrictModel]


class StrictChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["assistant"]
    content: str | None = None


class StrictChatChoice(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int
    message: StrictChatMessage
    finish_reason: Literal["stop", "length", "content_filter",
                           "tool_calls"] | None = None


class StrictChatCompletion(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[StrictChatChoice]
    usage: StrictUsage | None = None


class StrictDelta(BaseModel):
    model_config = ConfigDict(extra="allow")


class StrictChunkChoice(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int
    delta: StrictDelta
    finish_reason: Literal["stop", "length", "content_filter",
                           "tool_calls"] | None = None


class StrictChatChunk(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[StrictChunkChoice]
    usage: StrictUsage | None = None


class StrictEvent(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    sequence_number: int


class StrictResponsesObject(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    object: Literal["response"]
    status: str
    output: list[dict]
    output_text: str


# ---------------------------------------------------------------------------
# 1) HTTP 错误统一 OpenAI 形状
# ---------------------------------------------------------------------------


def test_405_is_openai_error_shape():
    c = _client()
    r = c.post("/v1/models")
    assert r.status_code == 405
    err = StrictErrorResponse.model_validate(r.json()).error
    assert err.type == "invalid_request_error"
    assert err.code == "method_not_allowed"
    assert "GET" in (r.headers.get("allow") or "")
    # 根路径别名同样走统一处理器
    r2 = c.get("/v1/chat/completions")
    assert r2.status_code == 405
    assert StrictErrorResponse.model_validate(r2.json()).error.code \
        == "method_not_allowed"


def test_404_is_openai_error_shape():
    r = _client().get("/v1/definitely-not-an-endpoint")
    assert r.status_code == 404
    err = StrictErrorResponse.model_validate(r.json()).error
    assert err.type == "invalid_request_error"
    assert err.code == "unknown_endpoint"
    assert "detail" not in r.json()


def test_401_is_openai_error_shape():
    r = _client(env={"RKLLM_SERVER_API_KEY": "k"}).get("/health")
    assert r.status_code == 401
    err = StrictErrorResponse.model_validate(r.json()).error
    assert err.type == "authentication_error"
    assert err.message


def test_500_is_openai_error_shape():
    """路由未捕获异常 → 500 也是 OpenAI error 对象（不是 HTML/`detail`）。"""
    c = _client(FakeSupervisor(error=RuntimeError("boom")),
                raise_server_exceptions=False)
    r = c.post("/v1/chat/completions", json=MSG)
    assert r.status_code == 500
    err = StrictErrorResponse.model_validate(r.json()).error
    assert err.type == "server_error"
    assert err.code == "internal_error"


def test_413_is_openai_error_shape():
    c = _client(env={"RKLLM_SERVER_MAX_BODY_BYTES": "1024"})
    r = c.post("/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "x" * 4096}]})
    assert r.status_code == 413
    err = StrictErrorResponse.model_validate(r.json()).error
    assert err.code == "request_too_large"


# ---------------------------------------------------------------------------
# 2) usage-only 帧 choices=[]
# ---------------------------------------------------------------------------


def test_chat_usage_frame_has_empty_choices():
    sup = FakeSupervisor(perf={"prefill_tokens": 5, "generate_tokens": 2})
    c = _client(sup)
    with c.stream("POST", "/v1/chat/completions", json={
            **MSG, "stream": True,
            "stream_options": {"include_usage": True}}) as resp:
        raw = _sse(resp)
    chunks = [StrictChatChunk.model_validate(item) for item in raw]
    usage_frames = [chunk for chunk in chunks if chunk.usage is not None]
    assert len(usage_frames) == 1
    usage_chunk = usage_frames[0]
    # 严格模型能解析空 choices；原始 JSON 也必须是 choices: []
    assert usage_chunk.choices == []
    raw_usage = [item for item in raw if item.get("usage") is not None]
    assert len(raw_usage) == 1 and raw_usage[0]["choices"] == []
    assert usage_chunk.usage.completion_tokens == 2
    # finish 帧在 usage 帧之前
    finish_at = [i for i, chunk in enumerate(chunks)
                 if chunk.choices and chunk.choices[0].finish_reason][0]
    assert chunks.index(usage_chunk) > finish_at


def test_chat_stream_parses_with_strict_client():
    c = _client()
    with c.stream("POST", "/v1/chat/completions",
                  json={**MSG, "stream": True}) as resp:
        raw = _sse(resp)
    chunks = [StrictChatChunk.model_validate(item) for item in raw]
    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(getattr(chunk.choices[0].delta, "content", None) or ""
                   for chunk in chunks if chunk.choices)
    assert text == "你好"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_nonstream_chat_completion_strict_shape():
    c = _client()
    r = c.post("/v1/chat/completions", json=MSG)
    assert r.status_code == 200
    body = StrictChatCompletion.model_validate(r.json())
    assert body.id.startswith("chatcmpl-")
    assert body.choices[0].message.role == "assistant"
    assert body.choices[0].message.content == "你好"
    assert body.choices[0].finish_reason == "stop"
    assert body.usage.total_tokens == (body.usage.prompt_tokens
                                       + body.usage.completion_tokens)


# ---------------------------------------------------------------------------
# 3) reasoning 流式增量（chat delta / Responses item+delta）
# ---------------------------------------------------------------------------

THINK_TOKENS = ["<think>", "思", "考", "</think>", "答", "案"]


def test_chat_stream_reasoning_is_incremental():
    sup = FakeSupervisor(tokens=THINK_TOKENS)
    c = _client(sup)
    with c.stream("POST", "/v1/chat/completions", json={
            **MSG, "stream": True, "reasoning_effort": "high"}) as resp:
        raw = _sse(resp)
    chunks = [StrictChatChunk.model_validate(item) for item in raw]
    reasoning_deltas: list[str] = []
    content_deltas: list[str] = []
    last_reasoning = first_content = None
    for idx, chunk in enumerate(chunks):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        content = getattr(delta, "content", None)
        if reasoning:
            reasoning_deltas.append(reasoning)
            last_reasoning = idx
        if content:
            content_deltas.append(content)
            if first_content is None:
                first_content = idx
    # 增量而非终帧一次性回填：思考阶段至少两帧
    assert len(reasoning_deltas) >= 2, raw
    assert "".join(reasoning_deltas) == "思考"
    assert "".join(content_deltas) == "答案"
    # 顺序：思考增量全部在正文之前（内容顺序即生成顺序）
    assert last_reasoning is not None and first_content is not None
    assert last_reasoning < first_content
    assert chunks[-1].choices[0].finish_reason == "stop"
    # reasoning_effort 真实传给 runtime
    assert sup.kwargs[0]["enable_thinking"] is True


def test_chat_nonstream_reasoning_still_exposed():
    """回归：非流式 reasoning_content 必须保持（流式改造不能搬走它）。"""
    c = _client(FakeSupervisor(tokens=THINK_TOKENS))
    r = c.post("/v1/chat/completions",
               json={**MSG, "reasoning_effort": "high"})
    body = StrictChatCompletion.model_validate(r.json())
    message = body.choices[0].message
    assert message.content == "答案"
    assert getattr(message, "reasoning_content") == "思考"


def test_responses_stream_reasoning_incremental():
    c = _client(FakeSupervisor(tokens=THINK_TOKENS))
    with c.stream("POST", "/v1/responses", json={
            "input": "x", "stream": True,
            "reasoning": {"effort": "high"}}) as resp:
        events = _response_events(resp)
    for _, payload in events:
        StrictEvent.model_validate(payload)
    names = [name for name, _ in events]
    reasoning_deltas = [payload for name, payload in events
                        if name == "response.reasoning_text.delta"]
    assert len(reasoning_deltas) >= 2, names
    assert "".join(item["delta"] for item in reasoning_deltas) == "思考"
    # reasoning item 在 body 输出之前声明（output_index 0）
    added = [payload for name, payload in events
             if name == "response.output_item.added"]
    reasoning_item = [item for item in added
                      if item["item"].get("type") == "reasoning"]
    assert reasoning_item and reasoning_item[0]["output_index"] == 0
    # 正文 delta 在思考 delta 之后
    first_text = names.index("response.output_text.delta")
    last_reasoning = max(i for i, name in enumerate(names)
                         if name == "response.reasoning_text.delta")
    assert last_reasoning < first_text
    completed = [payload for name, payload in events
                 if name == "response.completed"]
    assert completed, names
    final = completed[0]["response"]
    assert final["status"] == "completed"
    assert final["output"][0]["type"] == "reasoning"
    assert final["output"][0]["content"][0]["text"] == "思考"
    assert final["output"][1]["type"] == "message"
    assert final["output_text"] == "答案"
    # 不伪造 summary 事件
    assert not any(name.startswith("response.reasoning_summary")
                   for name in names)


def test_responses_nonstream_includes_reasoning_item():
    c = _client(FakeSupervisor(tokens=THINK_TOKENS))
    r = c.post("/v1/responses", json={"input": "x",
                                      "reasoning": {"effort": "low"}})
    assert r.status_code == 200
    body = StrictResponsesObject.model_validate(r.json())
    assert body.output[0]["type"] == "reasoning"
    assert body.output[0]["content"][0]["text"] == "思考"
    assert body.output[1]["type"] == "message"
    assert body.output_text == "答案"


# ---------------------------------------------------------------------------
# 4) json_object 校验/重试；非法 response_format / text.format 400
# ---------------------------------------------------------------------------


def test_json_object_nonstream_retries_until_valid():
    sup = ScriptedSupervisor([["这不是 JSON"], ['{"a": 1}']])
    c = _client(sup)
    r = c.post("/v1/chat/completions",
               json={**MSG, "response_format": {"type": "json_object"}})
    assert r.status_code == 200, r.text
    body = StrictChatCompletion.model_validate(r.json())
    content = body.choices[0].message.content
    assert json.loads(content) == {"a": 1}
    assert sup.calls == 2, "json_object 非法输出必须触发一次重试"
    assert "JSON" in sup.prompts[1], "重试 prompt 必须回灌格式要求"


def test_json_object_nonstream_fails_after_retries():
    sup = ScriptedSupervisor([["仍然不是 JSON"], ["还是不是"]])
    c = _client(sup, env={"RKLLM_OPENAI_JSON_SCHEMA_RETRIES": "1"})
    r = c.post("/v1/chat/completions",
               json={**MSG, "response_format": {"type": "json_object"}})
    assert r.status_code == 422, r.text
    err = StrictErrorResponse.model_validate(r.json()).error
    assert err.code == "json_object_invalid"
    assert sup.calls == 2


def test_json_schema_nonstream_still_validates_and_retries():
    sup = ScriptedSupervisor([["bad"], ['{"a": 1}']])
    c = _client(sup)
    r = c.post("/v1/chat/completions", json={
        **MSG, "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "o", "schema": {
                "type": "object", "required": ["a"]}}}})
    assert r.status_code == 200, r.text
    content = r.json()["choices"][0]["message"]["content"]
    assert json.loads(content) == {"a": 1}


@pytest.mark.parametrize("response_format, param, code", [
    ({"type": "json_schema"}, "response_format.json_schema", "invalid_schema"),
    ({"type": "json_schema", "json_schema": {"schema": "nope"}},
     "response_format.json_schema.schema", "invalid_schema"),
    ({"type": "json_schema",
      "json_schema": {"name": "o", "schema": {"type": "nonsense"}}},
     "response_format.json_schema.schema", "invalid_schema"),
    ({"type": "xml"}, "response_format", "invalid_response_format"),
])
def test_invalid_json_schema_is_explicit_400(response_format, param, code):
    sup = FakeSupervisor(tokens=["should not run"])
    c = _client(sup)
    r = c.post("/v1/chat/completions",
               json={**MSG, "response_format": response_format})
    assert r.status_code == 400, r.text
    err = StrictErrorResponse.model_validate(r.json()).error
    assert err.param == param
    assert err.code == code
    assert not sup.prompts, "非法格式必须在进入推理前 400"


@pytest.mark.parametrize("text, param, code", [
    ({"format": {"type": "xml"}}, "text.format", "unsupported_parameter"),
    ({"format": {"type": "json_schema", "schema": []}},
     "text.format.schema", "invalid_schema"),
    ("bad", "text", "invalid_response_format"),
])
def test_invalid_text_format_is_explicit_400(text, param, code):
    sup = FakeSupervisor(tokens=["should not run"])
    c = _client(sup)
    r = c.post("/v1/responses", json={"input": "x", "text": text})
    assert r.status_code == 400, r.text
    err = StrictErrorResponse.model_validate(r.json()).error
    assert err.param == param
    assert err.code == code
    assert not sup.prompts


def test_responses_json_object_retries_until_valid():
    sup = ScriptedSupervisor([["不是 JSON"], ['{"ok": true}']])
    c = _client(sup)
    r = c.post("/v1/responses", json={
        "input": "x", "text": {"format": {"type": "json_object"}}})
    assert r.status_code == 200, r.text
    body = StrictResponsesObject.model_validate(r.json())
    assert json.loads(body.output_text) == {"ok": True}
    assert sup.calls == 2


# ---------------------------------------------------------------------------
# 5) caps：parallel_tool_calls 降级；repeat_penalty < 1.0 明确拒绝
# ---------------------------------------------------------------------------


def test_parallel_tool_calls_downgraded_from_emulated():
    assert caps.param_tier("parallel_tool_calls") == "unsupported"
    assert "parallel_tool_calls" not in caps.CAPS["params"]["emulated"]
    assert "parallel_tool_calls" in caps.CAPS["params"]["unsupported"]
    c = _client()
    body = c.get("/v1/capabilities").json()
    assert "parallel_tool_calls" not in body["params"]["emulated"]
    assert "parallel_tool_calls" in body["params"]["unsupported"]


def test_parallel_tool_calls_warn_policy_reports_header():
    c = _client()
    r = c.post("/v1/chat/completions", json={
        **MSG, "tools": [TOOL], "parallel_tool_calls": True})
    assert r.status_code == 200, r.text
    assert "parallel_tool_calls" in (r.headers.get("x-rkllm-unsupported") or "")


def test_parallel_tool_calls_reject_policy_returns_400():
    c = _client(env={"RKLLM_OPENAI_PARAM_POLICY": "reject"})
    r = c.post("/v1/chat/completions", json={
        **MSG, "tools": [TOOL], "parallel_tool_calls": True})
    assert r.status_code == 400, r.text
    err = StrictErrorResponse.model_validate(r.json()).error
    assert err.param == "parallel_tool_calls"
    assert err.code == "unsupported_parameter"


def test_repeat_penalty_below_one_is_rejected():
    sup = FakeSupervisor()
    c = _client(sup)
    r = c.post("/v1/chat/completions",
               json={**MSG, "repeat_penalty": 0.8})
    assert r.status_code == 400, r.text
    err = StrictErrorResponse.model_validate(r.json()).error
    assert err.param == "repeat_penalty"
    assert err.code == "unsupported_parameter"
    assert "1.0" in err.message
    assert not sup.prompts

    # Responses 路径同样生效
    r2 = c.post("/v1/responses", json={"input": "x", "repeat_penalty": 0.8})
    assert r2.status_code == 400
    assert StrictErrorResponse.model_validate(r2.json()).error.param \
        == "repeat_penalty"


def test_repeat_penalty_at_least_one_reaches_runtime():
    sup = FakeSupervisor()
    c = _client(sup)
    r = c.post("/v1/chat/completions",
               json={**MSG, "repeat_penalty": 1.2})
    assert r.status_code == 200, r.text
    assert sup.kwargs[0]["sampling"]["repeat_penalty"] == 1.2

    r2 = c.post("/v1/responses",
                json={"input": "x", "repeat_penalty": 1.3})
    assert r2.status_code == 200, r2.text
    assert sup.kwargs[1]["sampling"]["repeat_penalty"] == 1.3


# ---------------------------------------------------------------------------
# 6) /ready 不泄露模型名；/v1/models 补 created
# ---------------------------------------------------------------------------


def test_ready_does_not_leak_model_name():
    c = _client()
    r = c.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"ready": True}
    assert "model" not in r.json()


def test_ready_503_does_not_leak_model_name():
    c = _client(NotReadySupervisor())
    r = c.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body == {"ready": False, "state": "loading"}
    assert "model" not in body
    assert r.headers.get("retry-after")


def test_models_have_created_field():
    c = _client()
    listed = StrictModelList.model_validate(c.get("/v1/models").json())
    assert len(listed.data) == 1
    model = listed.data[0]
    assert isinstance(model.created, int) and model.created >= 0
    # 根路径别名与单模型端点的 created/id 必须一致（同一进程/同一模型）
    alias = StrictModelList.model_validate(c.get("/models").json())
    single = StrictModel.model_validate(c.get("/v1/models/whatever").json())
    assert alias.data[0] == model
    assert single == model
    assert model.object == "model" and model.owned_by == "rkllm-runtime"
