"""A 方案验收审查（e87bc87）后续 bug 的回归测试（2026-11-19）。

覆盖 P1-A 快照周期刷新、P1-C 板端默认生成长度回写、P2-A 数值溢出防护、
P2-B busy 错误分类。这些是本机纯 Python 可验证项，不依赖 ARM/NPU。
"""
from __future__ import annotations

import http.client
import math
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.mcp import jobs as jobs_mod
from rkllm.mcp.backend import BackendError, HttpBackend
from rkllm.mcp.stdio import BOARD_REFRESH_INTERVAL_S, StdioServer
from rkllm.mcp.__main__ import default_cfg


class _ReadyBackend:
    """模拟已跑过推理的板端：/health 带实测速率与能力上限。"""

    def __init__(self, *, decode_tok_s=7.5, default_max_tokens=1024):
        self.decode_tok_s = decode_tok_s
        self.default_max_tokens = default_max_tokens
        self.status_calls = 0

    def status(self):
        self.status_calls += 1
        return {
            "state": "ready", "model_class": "4b", "context_len": 4096,
            "perf": {"decode_tok_s": self.decode_tok_s,
                     "source": "measured", "samples": 50},
            "limits": {"max_tokens_limit": 0,
                       "default_max_tokens": self.default_max_tokens,
                       "context_margin_tokens": 32, "request_timeout_s": 600},
        }

    def chat(self, messages, **kwargs):
        return {"text": "ok", "finish_reason": "stop",
                "usage": None, "reasoning_chars": 0}


def _wait_board(cfg, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cfg.get("_board", {}).get("decode_tok_s"):
            return True
        time.sleep(0.01)
    return False


def test_p1a_periodic_refresh_reaches_measured_rate():
    """P1-A：构造后台线程应刷到板端实测速率，且刷新周期是常量。"""
    assert BOARD_REFRESH_INTERVAL_S > 0
    cfg = default_cfg("http://x/v1", "k")
    srv = StdioServer(_ReadyBackend(), cfg)
    try:
        assert _wait_board(cfg), "构造后应异步刷到 /health.perf.decode_tok_s"
        assert cfg["_board"]["decode_tok_s"] == 7.5
    finally:
        srv._board_refresh_stop.set()


def test_p1a_refresh_loop_repeats():
    """P1-A：刷新是周期性的——手动调 loop 的单次刷新不应是唯一机会。"""
    backend = _ReadyBackend()
    cfg = default_cfg("http://x/v1", "k")
    srv = StdioServer(backend, cfg)
    try:
        assert _wait_board(cfg)
        seen = backend.status_calls
        # 直接再触发一次刷新（模拟下一个周期），调用数应继续增长
        srv._refresh_model_class()
        assert backend.status_calls > seen
        assert cfg["_board"]["decode_tok_s"] == 7.5
    finally:
        srv._board_refresh_stop.set()


def test_p1c_board_default_max_tokens_written_back():
    """P1-C：板端 default_max_tokens 回写 cfg，tools handler 正常路径可拿到。"""
    cfg = default_cfg("http://x/v1", "k")
    assert cfg["default_max_tokens"] == 512        # 冷启动默认
    srv = StdioServer(_ReadyBackend(default_max_tokens=1024), cfg)
    try:
        assert _wait_board(cfg)
        assert cfg["default_max_tokens"] == 1024   # 板端值已回写
    finally:
        srv._board_refresh_stop.set()


def test_p2a_nonfinite_and_huge_values_do_not_overflow():
    """P2-A：inf/NaN/超大整数不得抛 OverflowError。"""
    assert jobs_mod._decode_rate("4b", math.inf) == 5.0
    assert jobs_mod._decode_rate("4b", math.nan) == 5.0
    assert jobs_mod._decode_rate("4b", None) == 5.0
    assert jobs_mod.sync_token_cap("4b", 10, 45, decode_tok_s=math.inf) >= 8
    assert jobs_mod.sync_token_cap("4b", 10, math.inf, 5.0) == 8
    est = jobs_mod.estimate_seconds("4b", 10, 10 ** 1000, 5.0)
    assert math.isfinite(est) and est > 0


def test_p2b_busy_becomes_iserror_with_retry_after():
    """P2-B：503 busy 转成 isError 结果并带 retryAfter，不再是 -32603。"""
    class BusyBackend(_ReadyBackend):
        def chat(self, messages, **kwargs):
            raise BackendError("模型忙", kind="busy", retry_after_s=5)

    cfg = default_cfg("http://x/v1", "k")
    srv = StdioServer(BusyBackend(), cfg)
    srv._board_refresh_stop.set()
    reply = srv._tools_call({"name": "local_llm_ask",
                             "arguments": {"prompt": "hi"}})
    assert reply.get("isError") is True
    assert reply["_meta"]["busy"] is True
    assert reply["_meta"]["retryAfter"] == 5


def test_p2b_non_busy_backend_error_still_raises():
    """P2-B：非 busy 的后端错误仍原样抛出（由 dispatcher 报 -32603）。"""
    class BrokenBackend(_ReadyBackend):
        def chat(self, messages, **kwargs):
            raise BackendError("其他错误", kind="error")

    cfg = default_cfg("http://x/v1", "k")
    srv = StdioServer(BrokenBackend(), cfg)
    srv._board_refresh_stop.set()
    with pytest.raises(BackendError):
        srv._tools_call({"name": "local_llm_ask",
                         "arguments": {"prompt": "hi"}})


# -- P2-D：非幂等 POST 的重试边界 -------------------------------------------

class _SockStub:
    def settimeout(self, _t):
        pass


class _FakeConn:
    def __init__(self, *, fail_at, calls):
        self.fail_at = fail_at          # "request" 或 "getresponse"
        self.calls = calls              # 外部计数列表
        # 与真实 http.client 连接一致：_request 先显式 connect()（短超时），
        # 成功后再 settimeout(读超时) 并 request()
        self.sock = None

    def connect(self):
        self.sock = _SockStub()

    def request(self, *a, **kw):
        self.calls.append("request")
        if self.fail_at == "request":
            raise OSError("connection refused")

    def getresponse(self):
        self.calls.append("getresponse")
        if self.fail_at == "getresponse":
            raise http.client.HTTPException("broken pipe")
        return "RESP"

    def close(self):
        pass


def _backend_with_conns(conns, monkeypatch):
    be = HttpBackend("http://x/v1")
    it = iter(conns)

    def _fake_conn(_self, timeout_s=None):
        return next(it)

    monkeypatch.setattr(HttpBackend, "_connection", _fake_conn)
    monkeypatch.setattr(HttpBackend, "_drop_connection",
                        lambda self: None)
    return be


def test_p2d_retries_when_request_not_sent(monkeypatch):
    """P2-D：conn.request() 失败（请求未送达）允许重建重试一次。"""
    calls = []
    be = _backend_with_conns(
        [_FakeConn(fail_at="request", calls=calls),
         _FakeConn(fail_at=None, calls=calls)], monkeypatch)
    assert be._request("/chat/completions", b"{}") == "RESP"
    assert calls == ["request", "request", "getresponse"]


def test_p2d_no_retry_after_request_sent(monkeypatch):
    """P2-D：getresponse() 阶段失败（请求已送达）绝不重发 POST。

    2026-11-19 审计 P1 起，该失败归一为 BackendError（不再是裸 HTTPException），
    便于上层给出可诊断错误；不重发的语义不变。
    """
    calls = []
    be = _backend_with_conns(
        [_FakeConn(fail_at="getresponse", calls=calls),
         _FakeConn(fail_at=None, calls=calls)], monkeypatch)
    with pytest.raises(BackendError) as ei:
        be._request("/chat/completions", b"{}")
    assert "连接中断" in str(ei.value)
    # 只有一次 request —— 第二次连接（若被误用会再发一次）不应被取用
    assert calls == ["request", "getresponse"]
