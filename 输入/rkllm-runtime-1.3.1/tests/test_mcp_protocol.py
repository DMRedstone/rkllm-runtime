"""MCP 一致性测试：协议分发、四版差异、错误码（S4/MCP 设计 §10.2）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.mcp.protocol import (Dispatcher, JsonRpcError, PROTOCOL_VERSIONS,
                                detect_protocol, result_payload)


def rpc(method, params=None, msg_id=1, **extra):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    msg.update(extra)
    return json.dumps(msg)


def make_dispatcher(version="2024-11-05"):
    disp = Dispatcher(version)
    disp.register("ping", lambda params: {})
    disp.register("tools/list", lambda params: {"tools": []})
    disp.register("boom", lambda params: (_ for _ in ()).throw(
        RuntimeError("exploded")))
    disp.register("bad_params", lambda params: 1 / 0 if params else (
        lambda: (_ for _ in ()).throw(JsonRpcError(-32602, "missing x")))())
    disp.register("server/discover", lambda params: {"ok": True})
    return disp


def test_parse_error_continues():
    disp = make_dispatcher()
    resp = disp.handle_message("{not json")
    assert resp["error"]["code"] == -32700
    assert disp.handle_message(rpc("ping"))["result"] == {}   # 仍可服务


def test_unknown_method_and_bad_params():
    disp = make_dispatcher()
    assert disp.handle_message(rpc("nope"))["error"]["code"] == -32601
    assert disp.handle_message(rpc("bad_params", {"x": 1}))["error"]["code"] == -32603


def test_jsonrpc_error_passthrough():
    disp = make_dispatcher()
    resp = disp.handle_message(rpc("bad_params", None, msg_id=7))
    assert resp["id"] == 7


def test_notification_no_response():
    disp = make_dispatcher()
    assert disp.handle_message(json.dumps(
        {"jsonrpc": "2.0", "method": "notifications/x"})) is None


def test_detect_protocol_paths():
    assert detect_protocol({"method": "initialize",
                            "params": {"protocolVersion": "2025-06-18"}}) == "2025-06-18"
    assert detect_protocol({"method": "tools/call",
                            "params": {"_meta": {"protocolVersion": "2026-07-28"}}}) == "2026-07-28"
    assert detect_protocol({"method": "tools/call", "params": {}}) == "2024-11-05"
    assert detect_protocol({"method": "initialize",
                            "params": {"protocolVersion": "1999-01-01"}}) == "2024-11-05"


def test_result_type_2026():
    assert result_payload("2026-07-28", {"content": []})["resultType"] == "text"
    assert "resultType" not in result_payload("2024-11-05", {"content": []})


def test_version_gated_method():
    disp = make_dispatcher("2024-11-05")
    assert disp.handle_message(rpc("server/discover"))["error"]["code"] == -32601
    disp26 = make_dispatcher("2026-07-28")
    assert disp26.handle_message(rpc("server/discover"))["result"]["ok"] is True


def test_all_four_versions_handshake_shape():
    """四版 initialize 报文形状一致（一致性脚本 §10.2 的基线）。"""
    for version in PROTOCOL_VERSIONS:
        disp = make_dispatcher(version)
        result = disp.handle_message(rpc("ping"))["result"]
        if version == "2026-07-28":
            assert result == {"resultType": "text"}   # 2026 强制 resultType
        else:
            assert result == {}


def test_invalid_params_object():
    disp = make_dispatcher()
    assert disp.handle_message(json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "ping",
         "params": [1, 2]}))["error"]["code"] == -32602
