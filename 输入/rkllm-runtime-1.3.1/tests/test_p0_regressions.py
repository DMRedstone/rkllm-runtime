"""P0 回归用例（1.3.1-17 审核整改）。

覆盖：
- P0-1 首包为「合法 JSON 但非对象」（数组/null/数字/字符串/布尔）时，reader
  线程不得抛异常、不得误设协议版本，并回 -32600；
- P0-2 tools/call 的 params 为真值非 dict 时，_tool_route 不得抛 AttributeError。
"""

from __future__ import annotations

import json

import pytest

from rkllm.mcp.stdio import StdioServer


class _Backend:
    def chat(self, messages, **kwargs):
        return {"text": "", "usage": None, "finish_reason": "stop"}

    def status(self):
        return {"state": "ready"}


def _cfg(**over):
    base = {
        "tool_prefix": "local_llm_", "default_max_tokens": 64,
        "default_temperature": 0.3, "max_input_chars": 4000,
        "max_result_chars": 4000, "batch_max_segments": 2,
        "task_mode": "auto", "task_threshold_s": 45, "job_ttl_s": 60,
        "job_max": 4, "job_poll_hint_s": 1, "progress_interval_s": 1,
        "extract_retries": 0, "max_concurrent": 1,
        "protocol_versions": ["2026-07-28", "2025-11-25", "2025-06-18",
                              "2024-11-05"],
        "endpoint": "http://x/v1",
    }
    base.update(over)
    return base


@pytest.mark.parametrize("raw", ["[1,2,3]", "null", "42", '"x"', "true"])
def test_p0_1_non_object_first_packet_does_not_kill_reader(monkeypatch, raw):
    """P0-1：非对象首包不得抛异常、不得误设协议，且回 -32600。"""
    srv = StdioServer(_Backend(), _cfg())
    sent = []
    monkeypatch.setattr(srv, "_send", lambda obj: sent.append(obj))
    srv._read_line(raw)                     # 修复前：AttributeError 杀死 reader
    assert srv._protocol_version is None
    assert any((m.get("error") or {}).get("code") == -32600 for m in sent)


def test_p0_2_tool_route_non_dict_params_does_not_raise():
    """P0-2：params 为真值非 dict 时 _tool_route 不得抛 AttributeError。"""
    srv = StdioServer(_Backend(), _cfg(), protocol_version="2024-11-05")
    raw = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                      "params": [1, 2, 3]})
    route, rid = srv._tool_route(raw)       # 修复前：AttributeError
    assert rid == 7
    assert route == "work"                  # 空名 → 默认 work 池
