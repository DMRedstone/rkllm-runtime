"""2026-11-19 全仓库审计发现的回归测试（连接器侧）。

覆盖：
- P0  `mcp.tools` 配置归一化（默认 list / TOML 数组 / env 逗号串）；
- P1  `getresponse()` 失败归一为 BackendError 并丢弃半开连接；
- P2  connect 超时与读超时分离；
- P2  job 路径 busy/loading 在 error 文本带 retryAfter 提示；
- P2  JSON Infinity（非标准常量）不再抛 OverflowError；
- P2  板端 model_class 非字符串时不崩快照轮；
- P2  `max_concurrent=0` 不再回落 4 线程；
- P2  mcp.toml 解析失败在 stderr 留痕。

全部为本机纯 Python 可验证项，不依赖 ARM/NPU。
"""
from __future__ import annotations

import http.client
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.mcp import __main__ as mcp_main
from rkllm.mcp import backend as backend_mod
from rkllm.mcp.backend import BackendError, HttpBackend
from rkllm.mcp.protocol import JsonRpcError
from rkllm.mcp.stdio import StdioServer
from rkllm.mcp.tools import build_handlers


class _StubBackend:
    """最小后端：status 立刻返回、chat 返回固定文本（不碰网络）。"""

    def status(self):
        return {"state": "ready", "model_class": "4b", "context_len": 4096}

    def chat(self, messages, **kwargs):
        return {"text": "ok", "finish_reason": "stop",
                "usage": None, "reasoning_chars": 0}


@pytest.fixture
def make_server():
    """构造 StdioServer 并确保退出时停掉 /health 刷新线程。"""
    servers: list[StdioServer] = []

    def _make(cfg, backend=None):
        srv = StdioServer(backend or _StubBackend(), cfg)
        servers.append(srv)
        return srv

    yield _make
    for srv in servers:
        srv._board_refresh_stop.set()


def _cfg_without_file(monkeypatch, tmp_path) -> dict:
    monkeypatch.setenv("RKLLM_MCP_CONFIG", str(tmp_path / "absent.toml"))
    monkeypatch.delenv("RKLLM_MCP_TOOLS", raising=False)
    return mcp_main.load_cfg()


# -- P0：mcp.tools 配置归一化 ------------------------------------------------

def test_p0_default_tools_is_list_and_all_enabled(monkeypatch, tmp_path,
                                                  make_server):
    """默认配置下 tools 必须是空 list，六个工具全部可用（P0 根因）。"""
    cfg = _cfg_without_file(monkeypatch, tmp_path)
    assert cfg["tools"] == []
    srv = make_server(cfg)
    for short in ("ask", "chat", "extract", "batch", "status", "job"):
        assert srv._tool_enabled(short), short


def test_p0_toml_tools_array_filters(monkeypatch, tmp_path, make_server):
    """TOML 数组不再被 str() 化：列表元素精确匹配。"""
    cfg_path = tmp_path / "mcp.toml"
    cfg_path.write_text('[mcp]\ntools = ["ask"]\n', encoding="utf-8")
    monkeypatch.setenv("RKLLM_MCP_CONFIG", str(cfg_path))
    monkeypatch.delenv("RKLLM_MCP_TOOLS", raising=False)
    cfg = mcp_main.load_cfg()
    assert cfg["tools"] == ["ask"]
    srv = make_server(cfg)
    assert srv._tool_enabled("ask")
    assert not srv._tool_enabled("batch")
    assert srv._tool_enabled("status") and srv._tool_enabled("job")


def test_p0_env_tools_csv_filters(monkeypatch, tmp_path, make_server):
    """env 逗号串按元素比较，不会被打散成单字符集合。"""
    cfg = _cfg_without_file(monkeypatch, tmp_path)
    monkeypatch.setenv("RKLLM_MCP_TOOLS", "ask,chat")
    cfg = mcp_main.load_cfg()
    srv = make_server(cfg)
    assert srv._tool_enabled("ask")
    assert not srv._tool_enabled("extract")


# -- P1：getresponse 失败归一 ------------------------------------------------

class _SockStub:
    def settimeout(self, _t):
        pass


class _BrokenConn:
    """request() 成功、getresponse() 抛 RemoteDisconnected 的半开连接。"""

    def __init__(self):
        self.sock = _SockStub()
        self.closed = False

    def request(self, *_a, **_k):
        pass

    def getresponse(self):
        raise http.client.RemoteDisconnected("remote closed connection")

    def close(self):
        self.closed = True


def test_p1_getresponse_failure_wrapped_and_dropped(monkeypatch):
    b = HttpBackend("http://127.0.0.1:1/v1", timeout_s=5)
    broken = _BrokenConn()

    def _fake_conn(_t):
        # 模拟真实 _connection：把连接登记到 thread-local，_drop_connection 才拿得到
        b._local.conn = broken
        b._local.conn_timeout = 5.0
        return broken

    monkeypatch.setattr(b, "_connection", _fake_conn)
    with pytest.raises(BackendError) as ei:
        b._request("/chat/completions", b"{}")
    assert "连接中断" in str(ei.value)
    # 半开连接必须被丢弃，不能留在 thread-local 里给下一次调用复用
    assert broken.closed is True
    assert getattr(b._local, "conn", None) is None


# -- P2：connect 超时与读超时分离 --------------------------------------------

def test_p2_connect_timeout_separate_from_read_timeout():
    b = HttpBackend("http://127.0.0.1:1/v1", timeout_s=5)
    conn = b._connection(4200.0)
    # 建连阶段用短超时，读阶段才放宽到调用方要求的值
    assert conn.timeout == min(4200.0, backend_mod.CONNECT_TIMEOUT_S)
    assert b._local.conn_timeout == 4200.0
    b._drop_connection()


# -- P2：job 路径 busy 带 retryAfter 提示 ------------------------------------

def _busy_handler(_arguments, should_cancel=None, max_tokens_cap=None,
                  timeout_s=None):
    raise BackendError("模型忙：队列已满", retry_after_s=7, kind="busy")


def _wait_job(server, job_id, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = server.jobs.get(job_id)
        if job is not None and job.state != "working":
            return job
        time.sleep(0.01)
    return None


def test_p2_job_busy_error_carries_retry_hint(monkeypatch, tmp_path,
                                              make_server):
    cfg = _cfg_without_file(monkeypatch, tmp_path)
    srv = make_server(cfg)
    reply = srv._run_as_job("ask", _busy_handler, {}, None)
    import json as _json
    job_id = _json.loads(reply["content"][0]["text"])["job_id"]
    job = _wait_job(srv, job_id)
    assert job is not None and job.state == "failed"
    assert "7 秒后重试" in (job.error or "")


# -- P2：Infinity 不再抛 OverflowError --------------------------------------

def test_p2_ask_infinity_max_tokens_is_invalid_params(monkeypatch, tmp_path):
    cfg = _cfg_without_file(monkeypatch, tmp_path)
    handlers = build_handlers(_StubBackend(), cfg, None)
    with pytest.raises(JsonRpcError):
        handlers["ask"]({"prompt": "hi", "max_tokens": float("inf")})


def test_p2_estimate_handles_infinity(monkeypatch, tmp_path, make_server):
    cfg = _cfg_without_file(monkeypatch, tmp_path)
    srv = make_server(cfg)
    tokens = srv._estimate_decode_tokens({"max_tokens": float("inf")}, "ask")
    assert isinstance(tokens, int) and tokens >= 1


# -- P2：model_class 非字符串不崩快照轮 --------------------------------------

class _WeirdClassBackend:
    def status(self):
        return {"model_class": 123,
                "perf": {"decode_tok_s": 6.0, "source": "measured"},
                "limits": {"default_max_tokens": 1024,
                           "request_timeout_s": 600}}

    def chat(self, messages, **kwargs):
        return {"text": "ok", "finish_reason": "stop"}


def test_p2_refresh_tolerates_non_str_model_class(monkeypatch, tmp_path,
                                                  make_server):
    cfg = _cfg_without_file(monkeypatch, tmp_path)
    srv = make_server(cfg, _WeirdClassBackend())
    srv._refresh_model_class()      # 不得抛；同轮 perf/limits 必须仍能落盘
    board = srv.cfg.get("_board") or {}
    assert board.get("decode_tok_s") == 6.0
    assert board.get("default_max_tokens") == 1024


# -- P2：max_concurrent=0 不再回落 4 ----------------------------------------

def test_p2_zero_max_concurrent_clamped_to_one(monkeypatch, tmp_path,
                                               make_server):
    cfg = _cfg_without_file(monkeypatch, tmp_path)
    cfg["max_concurrent"] = 0
    srv = make_server(cfg)
    assert srv._pool._max_workers == 1


# -- P2：TOML 解析失败留痕 --------------------------------------------------

def test_p2_bad_toml_warns_on_stderr(monkeypatch, tmp_path, capsys):
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text("[mcp\nmissing_bracket = 1\n", encoding="utf-8")
    monkeypatch.setenv("RKLLM_MCP_CONFIG", str(cfg_path))
    cfg = mcp_main.load_cfg()
    assert "读取配置失败" in capsys.readouterr().err
    assert cfg["client_timeout_s"] == 60      # 回退默认值
