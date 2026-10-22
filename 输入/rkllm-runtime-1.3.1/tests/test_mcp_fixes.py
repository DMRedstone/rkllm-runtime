"""连接器审计修复的回归测试（2026-10-22 第二批）。

覆盖：2026 resultType、server/discover 全版本、tools/list 缓存字段、
`rkllm-mcp ask` 入口、job 进度不阻塞启动、会话记忆、extract 估算、取消态。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from rkllm.mcp import __main__ as entry
from rkllm.mcp.backend import BackendError
from rkllm.mcp.jobs import JobTable
from rkllm.mcp.stdio import StdioServer
from rkllm.mcp.tools import build_handlers


class _Backend:
    """最小后端：记录 messages，按需取消。"""

    def __init__(self, text: str = "回答"):
        self.text = text
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        check = kwargs.get("cancel_check")
        if check and check():
            raise BackendError("cancelled", kind="cancelled")
        return {"text": self.text, "usage": None, "finish_reason": "stop"}

    def status(self):
        return {"state": "ready"}


def _cfg(**over) -> dict:
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


def _call(server: StdioServer, method: str, params: dict | None = None,
          *, msg_id: int = 1) -> dict:
    raw = json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": method,
                      "params": params or {}}, ensure_ascii=False)
    return server.dispatcher.handle_message(raw)


def test_2026_result_type_is_complete_and_cache_fields():
    server = StdioServer(_Backend(), _cfg(), protocol_version="2026-07-28")
    server._refresh_model_class = lambda: None
    try:
        listed = _call(server, "tools/list")["result"]
        assert listed["ttlMs"] > 0 and listed["cacheScope"] == "global"
        discover = _call(server, "server/discover")["result"]
        assert discover["supportedVersions"][0] == "2026-07-28"
        assert discover["resultType"] == "complete"
    finally:
        server._control_pool.shutdown(wait=False)
        server._pool.shutdown(wait=False)


def test_server_discover_available_for_old_versions():
    server = StdioServer(_Backend(), _cfg(), protocol_version="2024-11-05")
    server._refresh_model_class = lambda: None
    try:
        result = _call(server, "server/discover")["result"]
        assert result["serverInfo"]["name"] == "rkllm-mcp"
    finally:
        server._control_pool.shutdown(wait=False)
        server._pool.shutdown(wait=False)


def test_cli_ask_uses_built_server(monkeypatch, capsys):
    class _Server:
        _handlers = {"ask": lambda args: {
            "content": [{"type": "text", "text": "ok:" + args["prompt"]}]}}

    monkeypatch.setattr(entry, "build_server", lambda: _Server())
    assert entry.cli_ask("hi") == 0
    assert "ok:hi" in capsys.readouterr().out


def test_job_progress_does_not_delay_start(monkeypatch):
    server = StdioServer(_Backend(), _cfg(progress_interval_s=1),
                         protocol_version="2024-11-05")
    server._refresh_model_class = lambda: None
    notes: list[float] = []
    monkeypatch.setattr(server, "_notify_progress",
                        lambda token, progress, total=None:
                        notes.append(progress))
    started: list[float] = []

    def handler(arguments, should_cancel=None):
        started.append(time.monotonic())
        time.sleep(1.2)
        return {"content": []}

    t0 = time.monotonic()
    reply = server._run_as_job("ask", handler, {}, progress_token=7)
    job_id = json.loads(reply["content"][0]["text"])["job_id"]
    time.sleep(0.25)
    try:
        assert started and started[0] - t0 < 0.2, "handler 被进度循环阻塞"
        time.sleep(2.0)
        assert server.jobs.get(job_id).state == "completed"
        assert notes, "任务执行期间没有进度通知"
    finally:
        server._control_pool.shutdown(wait=False)
        server._pool.shutdown(wait=False)


def test_chat_session_keeps_history():
    backend = _Backend()
    handlers = build_handlers(backend, _cfg(), JobTable(),
                              sessions={},
                              sessions_lock=threading.Lock())
    handlers["chat"]({"messages": [{"role": "user", "content": "第一问"}],
                      "session": "s1"})
    handlers["chat"]({"messages": [{"role": "user", "content": "第二问"}],
                      "session": "s1"})
    contents = [m["content"] for m in backend.calls[-1]]
    assert "第一问" in contents and "回答" in contents and "第二问" in contents


def test_ask_json_object_adds_instruction():
    backend = _Backend()
    handlers = build_handlers(backend, _cfg(), JobTable())
    handlers["ask"]({"prompt": "给出字段", "format": "json_object"})
    assert "只输出一个 JSON 对象" in backend.calls[-1][-1]["content"]


def test_extract_estimate_accounts_for_double_tokens():
    server = StdioServer(_Backend(), _cfg(), protocol_version="2024-11-05")
    server._refresh_model_class = lambda: None
    try:
        ask = server._default_estimate({"text": "同一段"}, "ask")
        extract = server._default_estimate({"text": "同一段"}, "extract")
        assert extract > ask
    finally:
        server._control_pool.shutdown(wait=False)
        server._pool.shutdown(wait=False)


def test_job_cancelled_even_if_handler_ignores_flag():
    server = StdioServer(_Backend(), _cfg(), protocol_version="2024-11-05")
    server._refresh_model_class = lambda: None

    def handler(arguments, should_cancel=None):
        time.sleep(0.3)              # 故意不检查 cancel 标志
        return {"content": []}

    try:
        reply = server._run_as_job("batch", handler, {}, None)
        job_id = json.loads(reply["content"][0]["text"])["job_id"]
        job = server.jobs.get(job_id)
        job.cancel()
        time.sleep(0.6)
        assert job.state == "cancelled"
    finally:
        server._control_pool.shutdown(wait=False)
        server._pool.shutdown(wait=False)


def test_default_config_path_is_install_dir():
    """配置文件默认放连接器安装目录（与 mcp.toml.example 同级）。"""
    expected = Path(entry.__file__).resolve().parents[2] / "mcp.toml"
    assert Path(entry.default_config_path()) == expected
