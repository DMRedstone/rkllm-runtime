"""Responses 翻译层（emulated）：转换正确性 + HTTP 形状 + 事件序列。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.agent import responses as tr          # noqa: E402
from rkllm.config import load_config              # noqa: E402
from rkllm.server.app import create_app           # noqa: E402
from rkllm.server.routes import responses as rr   # noqa: E402


class FakeSupervisor:
    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self, tokens=None):
        self.tokens = tokens or ["你", "好"]
        self.model_name = "fake"

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        for token in self.tokens:
            yield token

    def abort(self):
        pass

    def shutdown(self):
        pass


def _client(tokens=None):
    cfg = load_config(env={})
    app = create_app(cfg, supervisor=FakeSupervisor(tokens))
    return TestClient(app)


# -- 请求翻译 ---------------------------------------------------------------

def test_string_input_becomes_user_message():
    chat, ignored = tr.to_chat_request({"input": "总结这段话",
                                        "instructions": "你是公文助手"})
    assert chat["messages"] == [
        {"role": "system", "content": "你是公文助手"},
        {"role": "user", "content": "总结这段话"}]
    assert ignored == []


def test_item_array_and_tool_history():
    """function_call / function_call_output 必须回填成 assistant.tool_calls + tool。"""
    body = {"input": [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "巴黎天气"}]},
        {"type": "function_call", "call_id": "call_1",
         "name": "get_weather", "arguments": '{"city":"Paris"}'},
        {"type": "function_call_output", "call_id": "call_1",
         "output": '{"temp":18}'},
    ]}
    msgs = tr.to_chat_request(body)[0]["messages"]
    assert msgs[0] == {"role": "user", "content": "巴黎天气"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert msgs[2] == {"role": "tool", "tool_call_id": "call_1",
                       "content": '{"temp":18}'}


def test_tools_and_tool_choice_translation():
    body = {"input": "x", "tools": [{"type": "function", "name": "f",
                                     "description": "d",
                                     "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "f"}}
    chat = tr.to_chat_request(body)[0]
    assert chat["tools"][0]["function"]["name"] == "f"
    assert chat["tool_choice"] == {"type": "function", "function": {"name": "f"}}


def test_text_format_and_limits():
    body = {"input": "x", "max_output_tokens": 128, "temperature": 0.2,
            "text": {"format": {"type": "json_schema", "name": "o",
                                "schema": {"type": "object"}}}}
    chat = tr.to_chat_request(body)[0]
    assert chat["max_tokens"] == 128
    assert chat["temperature"] == 0.2
    assert chat["response_format"]["type"] == "json_schema"


def test_stateless_fields_are_declared_not_silently_dropped():
    body = {"input": "x", "store": True, "previous_response_id": "resp_1",
            "include": ["reasoning"]}
    _, ignored = tr.to_chat_request(body)
    assert set(ignored) >= {"store", "previous_response_id", "include"}


# -- 响应翻译 ---------------------------------------------------------------

def test_chat_to_output_text():
    payload = {"choices": [{"message": {"role": "assistant", "content": "答复"},
                            "finish_reason": "stop"}]}
    output, finish = tr.chat_to_output(payload)
    assert finish == "stop"
    assert output[0]["type"] == "message"
    assert output[0]["content"][0]["text"] == "答复"


def test_chat_to_output_tool_calls():
    payload = {"choices": [{"message": {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "call_9", "type": "function",
                        "function": {"name": "f", "arguments": "{}"}}]},
        "finish_reason": "tool_calls"}]}
    output, finish = tr.chat_to_output(payload)
    assert finish == "tool_calls"
    assert output[0]["type"] == "function_call"
    assert output[0]["call_id"] == "call_9"


# -- HTTP ------------------------------------------------------------------

def test_non_streaming_response_shape():
    r = _client(["公文", "要素"]).post("/v1/responses",
                                      json={"input": "说说公文要素"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output_text"] == "公文要素"
    assert body["output"][0]["type"] == "message"
    assert body["usage"]["input_tokens"] >= 1
    assert r.headers.get("x-rkllm-emulated") == "responses"


def test_empty_input_rejected():
    r = _client().post("/v1/responses", json={"input": ""})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_input"


def test_streaming_event_sequence():
    """Agents SDK 依赖的事件顺序：created → in_progress → item/content →
    delta → done → completed。"""
    events = []
    with _client(["你", "好"]).stream(
            "POST", "/v1/responses",
            json={"input": "x", "stream": True}) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("event: "):
                events.append(line[7:].strip())
    assert events[0] == "response.created"
    assert "response.output_text.delta" in events
    assert events.count("response.output_text.delta") == 2
    assert events[-1] == "response.completed"
    for required in ("response.in_progress", "response.output_item.added",
                     "response.content_part.added", "response.output_text.done",
                     "response.content_part.done", "response.output_item.done"):
        assert required in events


def test_streaming_truncation_marks_incomplete(monkeypatch):
    """泵因背压截断时，Responses 不能伪装成正常 completed。"""
    async def fake_iterate(request, stream, heartbeat_s):
        yield "item", "部分"
        yield "item", {"truncated": True}
        yield "done", None

    monkeypatch.setattr(rr, "iterate_stream", fake_iterate)
    events, payloads = [], {}
    current = None
    with _client(["ignored"]).stream(
            "POST", "/v1/responses",
            json={"input": "x", "stream": True}) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("event: "):
                current = line[7:].strip()
                events.append(current)
            elif (line.startswith("data: ") and current
                  and line[6:].strip() != "[DONE]"):
                payloads[current] = json.loads(line[6:])
    assert events[-1] == "response.incomplete"
    body = payloads["response.incomplete"]["response"]
    assert body["status"] == "incomplete"
    assert body["incomplete_details"]["reason"] == "max_output_tokens"
    assert body["output"][0]["status"] == "incomplete"
    assert body["output_text"] == "部分"

