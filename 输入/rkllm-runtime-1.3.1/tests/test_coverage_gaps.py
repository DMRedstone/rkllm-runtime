"""覆盖率补口：JSON Schema 校验器、reasoning 分流、pipeline 的工具/推理路径、
连接器 main() 启动分支、cli 的构造辅助。

这几个模块此前 38–70%，是纯逻辑（不需要 NPU / 子进程），补齐后整体口径才站得住。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.agent.json_schema import (KNOWN_KEYWORDS, unsupported_keywords,  # noqa: E402
                                     validate)
from rkllm.agent.reasoning import ReasoningSplitter  # noqa: E402


# ---------------------------------------------------------------------------
# JSON Schema 校验器
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value,schema,ok", [
    ({"a": 1}, {"type": "object"}, True),
    (["a"], {"type": "object"}, False),
    (1, {"type": "integer"}, True),
    (True, {"type": "integer"}, False),          # bool 不是 integer
    (1.5, {"type": "number"}, True),
    (True, {"type": "number"}, False),
    ("x", {"type": "string"}, True),
    (None, {"type": "null"}, True),
    (5, {"type": "unknown-type"}, True),         # 不认识的类型不拦
])
def test_type_checks(value, schema, ok):
    assert (validate(value, schema) == []) is ok


def test_required_and_properties():
    schema = {"type": "object", "properties": {"a": {"type": "string"}},
              "required": ["a"]}
    assert validate({"a": "x"}, schema) == []
    errors = validate({}, schema)
    assert errors and "缺少必填" in errors[0]
    assert validate({"a": 1}, schema)          # 类型不符


def test_array_items_and_bounds():
    schema = {"type": "array", "items": {"type": "string"},
              "minItems": 2, "maxItems": 3}
    assert validate(["a", "b"], schema) == []
    assert validate(["a"], schema)             # 太少
    assert validate(["a", "b", "c", "d"], schema)   # 太多
    assert validate(["a", 1], schema)          # 元素类型错


def test_enum_and_string_bounds():
    schema = {"type": "string", "enum": ["a", "b"], "minLength": 1,
              "maxLength": 2}
    assert validate("a", schema) == []
    assert validate("c", schema)               # enum 外
    assert validate("", schema)                # 太短
    assert validate("abc", schema)             # 太长


def test_number_bounds():
    schema = {"type": "number", "minimum": 0, "maximum": 1}
    assert validate(0.5, schema) == []
    assert validate(-1, schema)
    assert validate(2, schema)


def test_nested_object_paths():
    schema = {"type": "object", "properties": {
        "meta": {"type": "object", "properties": {"n": {"type": "integer"}},
                 "required": ["n"]}}}
    errors = validate({"meta": {}}, schema)
    assert errors and "$.meta" in errors[0]


def test_additional_properties_false():
    schema = {"type": "object", "properties": {"a": {"type": "string"}},
              "additionalProperties": False}
    assert validate({"a": "x"}, schema) == []
    assert validate({"a": "x", "b": 1}, schema)


def test_unsupported_keywords_lists_unknown():
    schema = {"type": "object", "properties": {"a": {"type": "string",
                                                     "pattern": "^x"}},
              "oneOf": []}
    found = " ".join(unsupported_keywords(schema))
    assert "pattern" in found and "oneOf" in found
    assert set(KNOWN_KEYWORDS) & {"type", "properties"}


# ---------------------------------------------------------------------------
# reasoning 分流
# ---------------------------------------------------------------------------
def test_reasoning_splitter_basic():
    sp = ReasoningSplitter(enabled=True)
    reasoning, content = sp.feed("<think>思考中</think>正文")
    assert "思考中" in (reasoning or "")
    assert "正文" in (content or "")
    # 跨 chunk：半个标签要缓冲，不吐半个
    sp2 = ReasoningSplitter(enabled=True)
    r1, c1 = sp2.feed("<thi")
    assert not c1                       # 不能把半个标签当正文
    r2, c2 = sp2.feed("nk>想</think>正文")
    assert "想" in (r1 or "") + (r2 or "")
    assert "正文" in (c2 or "")


def test_reasoning_splitter_disabled():
    sp = ReasoningSplitter(enabled=False)
    _, c = sp.feed("<think>不该分流</think>")
    assert "不该分流" in (c or "")


def test_reasoning_splitter_unclosed_block():
    """未闭合的 think 段：结束时不吞内容（归入 reasoning）。"""
    sp = ReasoningSplitter(enabled=True)
    reasoning, content = sp.feed("<think>只思考没闭合")
    assert "只思考" in (reasoning or "")      # 未闭合段归 reasoning
    assert content is None
    assert sp.finish() == (None, None)        # 没有残留需要 flush


# ---------------------------------------------------------------------------
# pipeline：工具前缀注入 + json 提示
# ---------------------------------------------------------------------------
class _Svc:
    model_name = "fake"
    tool_parser = "auto"
    max_tool_calls = 4
    system_prompt = ""
    tools_mode = "prompt"
    default_max_tokens = 64


def test_pipeline_injects_tools_prefix():
    from rkllm.agent.pipeline import Pipeline

    body = {"messages": [{"role": "user", "content": "查天气"}],
            "tools": [{"type": "function",
                       "function": {"name": "get_weather", "description": "d",
                                    "parameters": {"type": "object"}}}]}
    pipe = Pipeline(_Svc(), body, chat=True)
    assert "get_weather" in pipe._prompt          # 工具定义进了提示词


def test_pipeline_tools_mode_off_disables_injection():
    from rkllm.agent.pipeline import Pipeline

    class Svc(_Svc):
        tools_mode = "off"

    body = {"messages": [{"role": "user", "content": "查天气"}],
            "tools": [{"type": "function",
                       "function": {"name": "get_weather", "description": "d",
                                    "parameters": {"type": "object"}}}]}
    pipe = Pipeline(Svc(), body, chat=True)
    assert "get_weather" not in pipe._prompt


def test_pipeline_json_object_hint():
    from rkllm.agent.pipeline import Pipeline

    pipe = Pipeline(_Svc(), {"messages": [{"role": "user", "content": "x"}],
                             "response_format": {"type": "json_object"}},
                    chat=True)
    assert "JSON" in pipe._prompt


def test_pipeline_sampling_merge():
    from rkllm.agent.pipeline import Pipeline

    class Svc(_Svc):
        default_sampling = {"temperature": 0.8, "top_k": 1}

    pipe = Pipeline(Svc(), {"messages": [{"role": "user", "content": "x"}],
                            "temperature": 0.2}, chat=True)
    merged = pipe.sampling()
    assert merged["temperature"] == 0.2      # 请求优先
    assert merged["top_k"] == 1              # 进程默认保留


def test_pipeline_tool_block_not_in_text():
    """tool_call 块不计入文本产出——这正是「调用前的 content」的定义。"""
    import types

    from rkllm.agent.pipeline import Pipeline

    class Sup:
        def __init__(self, chunks):
            self.chunks = chunks

        def execute(self, prompt, **kw):
            return iter(self.chunks)

        def abort(self):
            pass

    class Svc(_Svc):
        supervisor = Sup(["好的，我查一下。",
                          '<tool_call>{"name": "get_weather", '
                          '"arguments": {"city": "北京"}}</tool_call>'])
        queue = types.SimpleNamespace(request_timeout_s=5)

        def ttft_log(self, seconds):
            pass

    body = {"messages": [{"role": "user", "content": "查天气"}],
            "tools": [{"type": "function",
                       "function": {"name": "get_weather", "description": "d",
                                    "parameters": {"type": "object"}}}]}
    pipe = Pipeline(Svc(), body, chat=True)
    text: list[str] = []
    meta: dict = {}
    for item in pipe.iter_tokens():
        if isinstance(item, dict):
            meta.update(item)
        else:
            text.append(item)
    assert "".join(text) == "好的，我查一下。"       # 不含 tool_call 块
    assert meta["tool_calls"][0]["function"]["name"] == "get_weather"


def test_completion_payload_keeps_pre_tool_text():
    """带 tool_calls 时 message.content 必须是调用前的文本，不能是空串。"""
    import types

    from rkllm.server.routes.openai import _completion_payload

    class Pipe:
        def usage(self, result=None):
            return {"prompt_tokens": 3, "completion_tokens": 4,
                    "total_tokens": 7}

        def finish_reason(self):
            return "tool_calls"

    svc = types.SimpleNamespace(reasoning_field="none", model_name="fake")
    result = {"text": "好的，我查一下。",
              "tool_calls": [{"id": "call_1", "type": "function",
                              "function": {"name": "get_weather",
                                           "arguments": "{}"}}]}
    payload = _completion_payload(svc, Pipe(), result, 64)
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "好的，我查一下。"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    # 无前置文本时保持空串（不改成 None，避免影响既有客户端断言）
    empty = _completion_payload(svc, Pipe(), {"text": "", "tool_calls": [
        {"id": "c2", "type": "function", "function": {"name": "f",
                                                      "arguments": "{}"}}]}, 8)
    assert empty["choices"][0]["message"]["content"] == ""


# ---------------------------------------------------------------------------
# 连接器 main()：启动分支（不真跑 serve_forever）
# ---------------------------------------------------------------------------
def test_mcp_main_starts_stdio(monkeypatch, capsys):
    from rkllm.mcp import __main__ as entry

    started = {}
    monkeypatch.setattr(entry.StdioServer, "serve_forever",
                        lambda self: started.setdefault("ok", True))
    monkeypatch.setattr(entry.StdioServer, "_refresh_model_class", lambda self: None)
    monkeypatch.setenv("RKLLM_MCP_CONFIG", "/nonexistent")
    monkeypatch.setenv("RKLLM_ENDPOINT", "http://127.0.0.1:1/v1")
    assert entry.main([]) == 0
    assert started.get("ok") is True


def test_mcp_main_refuses_http_transport(monkeypatch, tmp_path):
    from rkllm.mcp import __main__ as entry

    toml = tmp_path / "mcp.toml"
    toml.write_text('[mcp]\ntransport = "http"\n', encoding="utf-8")
    monkeypatch.setenv("RKLLM_MCP_CONFIG", str(toml))
    assert entry.main([]) == 2          # 显式拒绝，不是静默按 stdio 跑


def test_mcp_main_disabled_returns_zero(monkeypatch, tmp_path):
    from rkllm.mcp import __main__ as entry

    toml = tmp_path / "mcp.toml"
    toml.write_text("[mcp]\nenabled = false\n", encoding="utf-8")
    monkeypatch.setenv("RKLLM_MCP_CONFIG", str(toml))
    assert entry.main([]) == 0


# ---------------------------------------------------------------------------
# cli 辅助：档位与预算（不启动服务）
# ---------------------------------------------------------------------------
def test_cli_memory_budget_downgrades(monkeypatch, tmp_path):
    from rkllm import cli
    from rkllm.config import load_config

    model = tmp_path / "m-4b.rkllm"
    model.write_bytes(b"\0" * 4096)
    cfg = load_config(env={"RKLLM_MODEL_PATH": str(model),
                           "RKLLM_MEMORY_RESERVE_MB": "1024"})
    monkeypatch.setattr("rkllm.platform.available_mb", lambda: 3000)
    cli._apply_memory_budget(cfg)          # 返回 None：不足时降级并告警
    assert cfg.model.context <= 4096


def test_cli_resolve_model_path(tmp_path):
    from rkllm import cli

    model = tmp_path / "qwen3-4b.rkllm"
    model.write_text("x")
    assert cli._resolve_model_path("qwen3-4b.rkllm", [str(tmp_path)]) == str(model)
