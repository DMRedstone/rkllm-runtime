"""S3 agent 能力单测：conversation 回填（D6）、reasoning 分离、tools 解析（D1）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.agent.conversation import build_prompt
from rkllm.agent.reasoning import ReasoningSplitter
from rkllm.agent.tools import ToolCallParser, build_tools_prefix, detect_parser


# -- conversation（D6）--------------------------------------------------------


def test_basic_transcript():
    prompt = build_prompt([
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
    ])
    assert "User: 你好" in prompt
    assert "Assistant: 你好！" in prompt
    assert prompt.rstrip().endswith("Assistant:")


def test_tool_history_roundtrip():
    prompt = build_prompt([
        {"role": "user", "content": "巴黎天气"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather",
                          "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "get_weather",
         "content": '{"temp": 18}'},
    ])
    assert "<tool_call>" in prompt
    assert '"name": "get_weather"' in prompt
    assert "call_1" in prompt                      # tool_call_id 不丢
    assert '{"temp": 18}' in prompt


def test_multimodal_rejected_not_silently_dropped():
    with pytest.raises(ValueError):
        build_prompt([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x"}}]}])


def test_system_prefix():
    prompt = build_prompt([{"role": "user", "content": "hi"}],
                          system="be terse")
    assert prompt.startswith("System: be terse")


# -- reasoning ----------------------------------------------------------------


def test_reasoning_split_whole():
    sp = ReasoningSplitter()
    r1, c1 = sp.feed("<think>内部推理</think>答案")
    assert r1 == "内部推理"
    assert c1 == "答案"
    r2, c2 = sp.finish()
    assert r2 is None and c2 is None


def test_reasoning_split_across_chunks():
    sp = ReasoningSplitter()
    seen_r, seen_c = [], []
    for chunk in ["<thi", "nk>思考中", "</thi", "nk>正文"]:
        r, c = sp.feed(chunk)
        if r:
            seen_r.append(r)
        if c:
            seen_c.append(c)
    r, c = sp.finish()
    if r:
        seen_r.append(r)
    if c:
        seen_c.append(c)
    assert "".join(seen_r) == "思考中"
    assert "".join(seen_c) == "正文"


def test_reasoning_unclosed_think_goes_to_reasoning():
    sp = ReasoningSplitter()
    r1, _ = sp.feed("<think>没闭合")     # 增量语义：unclosed 段即时下发
    r2, c2 = sp.finish()
    assert (r1 or "") + (r2 or "") == "没闭合"
    assert c2 is None


def test_reasoning_disabled_passthrough():
    sp = ReasoningSplitter(enabled=False)
    r, c = sp.feed("<think>x</think>y")
    assert r is None and c == "<think>x</think>y"


# -- tools 解析（D1）-----------------------------------------------------------


def test_tool_parser_whole_block():
    p = ToolCallParser()
    events = list(p.feed('好的。<tool_call>{"name": "f", '
                         '"arguments": {"a": 1}}</tool_call>'))
    kinds = [k for k, _ in events]
    assert "content" in kinds and "tool_call" in kinds
    calls = p.tool_calls()
    assert calls[0]["name"] == "f"
    assert json.loads(calls[0]["arguments"]) == {"a": 1}


def test_tool_parser_across_chunks():
    p = ToolCallParser()
    chunks = ['<tool_c', 'all>{"na', 'me": "g", "argu', 'ments": [1, 2]}</tool', '_call>']
    tool_events = []
    for chunk in chunks:
        for kind, payload in p.feed(chunk):
            if kind == "tool_call":
                tool_events.append(payload)
    assert tool_events[0]["name"] == "g"
    assert json.loads(tool_events[0]["arguments"]) == [1, 2]


def test_tool_parser_broken_json_degrades_to_text():
    p = ToolCallParser()
    out = list(p.feed('<tool_call>{"name": ')) + list(p.finish())
    text = "".join(v for k, v in out if k == "content")
    assert '<tool_call>{"name": ' in text      # 不吞内容
    assert p.parse_failed
    assert p.tool_calls() == []


def test_tool_parser_max_calls():
    p = ToolCallParser(max_calls=1)
    list(p.feed('<tool_call>{"name":"a","arguments":{}}</tool_call>'
                '<tool_call>{"name":"b","arguments":{}}</tool_call>'))
    assert len(p.tool_calls()) == 1


def test_tools_prefix_contains_schema():
    prefix = build_tools_prefix([{"name": "t", "parameters": {}}])
    assert '"name": "t"' in prefix
    assert "<tool_call>" in prefix


def test_detect_parser():
    assert detect_parser("qwen3-4b") == "qwen"
    assert detect_parser("deepseek-r1") == "deepseek"
    assert detect_parser("unknown-model", "hermes") == "hermes"
    assert detect_parser("unknown-model") == "qwen"
