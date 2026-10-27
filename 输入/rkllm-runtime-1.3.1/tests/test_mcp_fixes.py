"""连接器审计修复的回归测试。

覆盖：2026 resultType、server/discover 全版本、tools/list 缓存字段、
`rkllm-mcp ask` 入口、job 进度不阻塞启动、会话记忆、extract 估算、取消态。

task-8（MCP 网关 P1）新增：
- 同步取消的底层读异常必须映射成 cancelled（不是 -32603）；
- 默认参数不再恒转 job、显式大 max_tokens 仍转 job、job 超时覆盖慢模型估算；
- 输入按 context_len 预算裁剪、chat 会话累计预算；
- `_inflight` 成功路径清理（100 次 status 后为空）；
- reasoning 模型空正文/finish_reason/usage 透传；
- 非法参数 -32602、extract 失败路径 structuredContent。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from rkllm.mcp import __main__ as entry
from rkllm.mcp.backend import BackendError, HttpBackend
from rkllm.mcp.jobs import JobTable, estimate_seconds
from rkllm.mcp.stdio import StdioServer
from rkllm.mcp.tools import build_handlers


class _Backend:
    """最小后端：记录 messages / kwargs，按需取消，可模拟 reasoning 截断。"""

    def __init__(self, text: str = "回答", *, finish_reason: str = "stop",
                 usage: dict | None = None, reasoning_chars: int = 0):
        self.text = text
        self.finish_reason = finish_reason
        self.usage = usage
        self.reasoning_chars = reasoning_chars
        self.calls: list[list[dict]] = []
        self.kwargs: dict = {}

    def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        self.kwargs = kwargs
        check = kwargs.get("cancel_check")
        if check and check():
            raise BackendError("cancelled", kind="cancelled")
        return {"text": self.text, "usage": self.usage,
                "finish_reason": self.finish_reason,
                "reasoning_chars": self.reasoning_chars}

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
        # task-8：估算/预算/超时新键
        "estimate_tokens_cap": 128, "context_window": 4096,
        "context_margin_tokens": 32, "target_timeout_s": 120,
        "job_timeout_factor": 1.5, "job_timeout_max_s": 1800,
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


def _new_server(backend=None, **cfg_over) -> StdioServer:
    server = StdioServer(backend or _Backend(), _cfg(**cfg_over),
                         protocol_version="2024-11-05")
    server._refresh_model_class = lambda: None
    return server


def _shutdown(server: StdioServer) -> None:
    for job in server.jobs.all():
        job.cancel()
    server._control_pool.shutdown(wait=False)
    server._pool.shutdown(wait=False)


def _tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 3)


# ---------------------------------------------------------------------------
# 协议/入口（原有）
# ---------------------------------------------------------------------------

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
        _shutdown(server)


def test_server_discover_available_for_old_versions():
    server = _new_server()
    try:
        result = _call(server, "server/discover")["result"]
        assert result["serverInfo"]["name"] == "rkllm-mcp"
    finally:
        _shutdown(server)


def test_cli_ask_uses_built_server(monkeypatch, capsys):
    class _Server:
        _handlers = {"ask": lambda args: {
            "content": [{"type": "text", "text": "ok:" + args["prompt"]}]}}

    monkeypatch.setattr(entry, "build_server", lambda: _Server())
    assert entry.cli_ask("hi") == 0
    assert "ok:hi" in capsys.readouterr().out


def test_default_config_path_is_install_dir():
    """配置文件默认放连接器安装目录（与 mcp.toml.example 同级）。"""
    expected = Path(entry.__file__).resolve().parents[2] / "mcp.toml"
    assert Path(entry.default_config_path()) == expected


# ---------------------------------------------------------------------------
# job / 进度 / 取消（原有 + task-8 调整）
# ---------------------------------------------------------------------------

def test_job_progress_does_not_delay_start(monkeypatch):
    server = _new_server(progress_interval_s=1)
    notes: list[float] = []
    monkeypatch.setattr(server, "_notify_progress",
                        lambda token, progress, total=None:
                        notes.append(progress))
    started: list[float] = []

    def handler(arguments, should_cancel=None, **kwargs):
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
        _shutdown(server)


def test_extract_job_start_has_structured_content():
    """outputSchema 工具转 job 的首次响应也必须带 structuredContent。"""
    server = _new_server(task_threshold_s=0)
    try:
        reply = server._run_as_job(
            "extract",
            lambda arguments, should_cancel=None, **kwargs: {
                "content": [], "structuredContent": {}},
            {}, None)
        assert reply["structuredContent"]["job_id"]
        assert reply["structuredContent"]["status"] == "working"
    finally:
        _shutdown(server)


def test_job_get_preserves_structured_content_and_errors():
    """声明了 output_schema 的工具经 job 路径也必须带回 structuredContent。"""
    server = _new_server()
    try:
        job = server.jobs.create()
        job.complete({"content": [{"type": "text", "text": "raw"}],
                      "structuredContent": {"ok": 1}})
        reply = server._job_tool({"job_id": job.id, "action": "get"})
        assert reply["structuredContent"] == {"ok": 1}
        assert reply["content"][0]["text"] == "raw"

        failed = server.jobs.create()
        failed.fail("boom")
        reply = server._job_tool({"job_id": failed.id, "action": "get"})
        assert reply["isError"] is True
        assert "boom" in reply["content"][0]["text"]
    finally:
        _shutdown(server)


def test_extract_job_failed_poll_has_structured_content():
    """outputSchema 工具 job 失败终帧也要带宽松 structuredContent（避免 -32600）。"""
    server = _new_server()
    try:
        job = server.jobs.create(short="extract")
        job.fail("boom")
        reply = server._job_tool({"job_id": job.id, "action": "get"})
        assert reply["isError"] is True
        assert reply["structuredContent"]["error"] == "boom"
    finally:
        _shutdown(server)


def test_job_cancelled_even_if_handler_ignores_flag():
    server = _new_server()

    def handler(arguments, should_cancel=None, **kwargs):
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
        _shutdown(server)


class _RaceResponse:
    """模拟 watchdog 打断后 http.client 抛 AttributError 的响应对象。"""

    status = 200

    def __init__(self, cancel_event: threading.Event):
        self._cancel = cancel_event

    def __iter__(self):
        yield b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n'
        for _ in range(200):
            if self._cancel.is_set():
                break
            time.sleep(0.01)
        raise AttributeError("'NoneType' object has no attribute 'peek'")

    def close(self):                     # pragma: no cover（防御路径）
        pass

    def getheader(self, name):
        return None


def test_sync_cancel_maps_low_level_exception_to_cancelled(monkeypatch):
    """task-8 P1-1：取消时 close 与读循环竞态抛的 AttributeError 不能变 -32603。"""
    cancel = threading.Event()
    backend = HttpBackend("http://127.0.0.1:1/v1", timeout_s=30)
    resp = _RaceResponse(cancel)
    monkeypatch.setattr(backend, "_request",
                        lambda path, body, timeout_s=None: resp)
    monkeypatch.setattr(backend, "_abort_remote", lambda r: None)

    def _fire():
        time.sleep(0.15)
        cancel.set()

    threading.Thread(target=_fire, daemon=True).start()
    with pytest.raises(BackendError) as info:
        backend.chat([{"role": "user", "content": "hi"}], max_tokens=32,
                     cancel_check=cancel.is_set)
    assert info.value.kind == "cancelled"


# ---------------------------------------------------------------------------
# task-8 P1-2：默认估算 / job 超时
# ---------------------------------------------------------------------------

def test_default_ask_estimate_is_sync_but_explicit_long_is_job():
    server = _new_server(default_max_tokens=512)
    try:
        default_estimate = server._default_estimate({}, "ask")
        explicit_estimate = server._default_estimate(
            {"max_tokens": 2000}, "ask")
        assert default_estimate <= 45, "默认 ask 不应恒转 job"
        assert explicit_estimate > 45, "显式大 max_tokens 仍应转 job"
    finally:
        _shutdown(server)


def test_extract_estimate_accounts_for_double_tokens_and_retries():
    server = _new_server()
    try:
        ask = server._default_estimate({"text": "同一段"}, "ask")
        extract = server._default_estimate({"text": "同一段"}, "extract")
        assert extract > ask
    finally:
        _shutdown(server)


def test_batch_estimate_multiplies_segments():
    server = _new_server()
    try:
        one = server._default_estimate({"segments": ["a"]}, "batch")
        three = server._default_estimate(
            {"segments": ["a", "b", "c"]}, "batch")
        assert three > one
    finally:
        _shutdown(server)


def test_job_timeout_covers_slow_model_estimate():
    """task-8：27B/large 512 token 估算 341 s，job 超时必须放宽（旧 120 s 必失败）。"""
    server = _new_server(default_max_tokens=512)
    server.model_class = "large"
    try:
        explicit = server._job_worst_estimate({"max_tokens": 512}, "ask")
        assert explicit > 120
        timeout = server._job_timeout(explicit)
        assert timeout > explicit, "job 读超时必须覆盖估算"
        assert timeout >= explicit * 1.5
        assert timeout <= 1800
        # 默认 ask 判 job（large 档预期 85 s）后实际仍可能生成 512 token：
        # 读超时必须按最坏值算，不能只看判定用的预期值。
        worst_default = server._job_worst_estimate({}, "ask")
        assert worst_default > 300
        assert server._job_timeout(worst_default) > worst_default
    finally:
        _shutdown(server)


def test_sync_token_cap_keeps_worst_case_within_threshold():
    server = _new_server(default_max_tokens=512)
    try:
        cap = server._sync_token_cap({}, "ask")
        assert 8 <= cap < 512
        worst = estimate_seconds(server.model_class, 1, cap)
        assert worst <= server.cfg["task_threshold_s"] + 1
    finally:
        _shutdown(server)


def test_job_runner_receives_relaxed_timeout():
    server = _new_server()
    captured: dict = {}

    def handler(arguments, should_cancel=None, **kwargs):
        captured.update(kwargs)
        return {"content": [{"type": "text", "text": "ok"}]}

    try:
        reply = server._run_as_job(
            "ask", handler, {}, None, estimate=341.0,
            timeout_s=server._job_timeout(341.0))
        job_id = json.loads(reply["content"][0]["text"])["job_id"]
        deadline = time.monotonic() + 3
        while server.jobs.get(job_id).state == "working" \
                and time.monotonic() < deadline:
            time.sleep(0.02)
        assert captured.get("max_tokens_cap") is None
        assert captured.get("timeout_s", 0) > 120
    finally:
        _shutdown(server)


# ---------------------------------------------------------------------------
# task-8 P1-3：输入预算（context_len / chat 累计）
# ---------------------------------------------------------------------------

def test_input_budget_trims_to_context_len():
    backend = _Backend()
    cfg = _cfg(context_len=4096, max_input_chars=12000,
               default_max_tokens=512)
    handlers = build_handlers(backend, cfg, JobTable())
    handlers["ask"]({"prompt": "字" * 8000})
    sent = backend.calls[-1][-1]["content"]
    # context 4096 - margin 32 - 生成预算 512 = 3552 token（中文 1 字≈1 token）
    assert _tokens(sent) <= 3552 + 64
    assert "截断" in sent


def test_chat_session_accumulates_context_budget():
    backend = _Backend()
    cfg = _cfg(context_len=4096, max_input_chars=12000,
               default_max_tokens=512)
    handlers = build_handlers(backend, cfg, JobTable(), sessions={},
                              sessions_lock=threading.Lock())
    last = None
    for _ in range(3):
        last = handlers["chat"]({
            "messages": [{"role": "user", "content": "字" * 2000}],
            "session": "budget"})
    request = backend.calls[-1]
    total = sum(_tokens(m["content"]) for m in request)
    assert total <= 3552 + 64, "chat 会话累计预算必须收窄到 context 内"
    assert last["_meta"].get("input_truncated") is True


def test_chat_history_and_session_shares_single_budget():
    backend = _Backend()
    cfg = _cfg(context_len=4096, max_input_chars=12000,
               default_max_tokens=512)
    handlers = build_handlers(backend, cfg, JobTable(), sessions={},
                              sessions_lock=threading.Lock())
    handlers["chat"]({"messages": [{"role": "user", "content": "问" * 1500}],
                      "session": "s2"})
    handlers["chat"]({"messages": [
        {"role": "user", "content": "问" * 1500},
        {"role": "assistant", "content": "答" * 300},
        {"role": "user", "content": "问" * 1500}], "session": "s2"})
    total = sum(_tokens(m["content"]) for m in backend.calls[-1])
    assert total <= 3552 + 64


# ---------------------------------------------------------------------------
# task-8 P1-4：_inflight 成功路径清理
# ---------------------------------------------------------------------------

def test_inflight_cleared_after_100_status_calls():
    server = _new_server()
    sent: list[dict] = []
    server._send = lambda obj: sent.append(obj)
    try:
        for i in range(100):
            line = json.dumps({
                "jsonrpc": "2.0", "id": 1000 + i, "method": "tools/call",
                "params": {"name": "local_llm_status", "arguments": {}}})
            server._read_line(line)
        deadline = time.monotonic() + 5
        while len(sent) < 100 and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.1)
        assert len(sent) == 100
        assert server._inflight == {}, \
            "status 成功响应后 _inflight 仍残留：%s" % list(server._inflight)
    finally:
        _shutdown(server)


def test_inflight_cleared_after_job_polls():
    server = _new_server()
    sent: list[dict] = []
    server._send = lambda obj: sent.append(obj)
    try:
        job = server.jobs.create()
        job.complete({"content": [{"type": "text", "text": "ok"}]})
        for i in range(20):
            line = json.dumps({
                "jsonrpc": "2.0", "id": 2000 + i, "method": "tools/call",
                "params": {"name": "local_llm_job",
                           "arguments": {"job_id": job.id}}})
            server._read_line(line)
        deadline = time.monotonic() + 5
        while len(sent) < 20 and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.1)
        assert len(sent) == 20
        assert server._inflight == {}
    finally:
        _shutdown(server)


# ---------------------------------------------------------------------------
# task-8 P1-5：reasoning 空正文 / finish_reason / usage
# ---------------------------------------------------------------------------

def test_reasoning_empty_with_length_is_error():
    backend = _Backend(text="", finish_reason="length",
                       reasoning_chars=200,
                       usage={"completion_tokens": 64})
    handlers = build_handlers(backend, _cfg(), JobTable())
    result = handlers["ask"]({"prompt": "用一句话说明"})
    assert result["isError"] is True
    assert "reasoning" in result["content"][0]["text"]
    assert result["_meta"]["finish_reason"] == "length"
    assert result["_meta"]["reasoning_chars"] == 200
    assert result["_meta"]["usage"]["completion_tokens"] == 64


def test_finish_reason_length_marks_truncation_and_passes_usage():
    backend = _Backend(text="半截回答", finish_reason="length",
                       usage={"completion_tokens": 64})
    handlers = build_handlers(backend, _cfg(), JobTable())
    result = handlers["ask"]({"prompt": "写长一点"})
    assert result.get("isError") is not True
    assert "截断" in result["content"][0]["text"]
    assert result["_meta"]["finish_reason"] == "length"
    assert result["_meta"]["usage"]["completion_tokens"] == 64


def test_empty_answer_without_length_is_error():
    backend = _Backend(text="", finish_reason="stop")
    handlers = build_handlers(backend, _cfg(), JobTable())
    result = handlers["ask"]({"prompt": "x"})
    assert result["isError"] is True
    assert "空响应" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# task-8 P1-6：非法参数 -32602 / extract 失败 structuredContent
# ---------------------------------------------------------------------------

def test_invalid_arguments_are_32602():
    server = _new_server()
    try:
        bad_arguments = _call(server, "tools/call", {
            "name": "local_llm_ask", "arguments": [1, 2]})
        assert bad_arguments["error"]["code"] == -32602
        empty_prompt = _call(server, "tools/call", {
            "name": "local_llm_ask", "arguments": {}})
        assert empty_prompt["error"]["code"] == -32602
        bad_mode = _call(server, "tools/call", {
            "name": "local_llm_ask",
            "arguments": {"prompt": "x", "mode": "nope"}})
        assert bad_mode["error"]["code"] == -32602
    finally:
        _shutdown(server)


def test_extract_failure_has_structured_content():
    backend = _Backend(text="这不是 JSON")
    handlers = build_handlers(backend, _cfg(), JobTable(),
                              extract_retries=0)
    result = handlers["extract"]({"text": "张三 35 岁",
                                  "schema": {"type": "object"}})
    assert result["isError"] is True
    structured = result.get("structuredContent")
    assert isinstance(structured, dict)
    assert structured.get("error")
    assert "raw" in structured


# ---------------------------------------------------------------------------
# task-8 P1-2 补充：默认同步 / task_mode=always
# ---------------------------------------------------------------------------

def test_default_ask_is_sync_not_job():
    server = _new_server(_Backend(text="短答"), default_max_tokens=512)
    try:
        result = server._tools_call({
            "name": "local_llm_ask", "arguments": {"prompt": "x"}})
        text = result["content"][0]["text"]
        assert "job_id" not in text, "默认短任务不应恒转 job"
        assert result["_meta"]["max_tokens"] < 512
    finally:
        _shutdown(server)


def test_task_mode_always_forces_job():
    server = _new_server(task_mode="always")
    try:
        result = server._tools_call({
            "name": "local_llm_ask", "arguments": {"prompt": "x"}})
        payload = json.loads(result["content"][0]["text"])
        assert payload["job_id"]
        assert payload["status"] == "working"
    finally:
        _shutdown(server)


# ---------------------------------------------------------------------------
# 工具层原有行为
# ---------------------------------------------------------------------------

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


def test_chat_session_expires_after_ttl():
    """P2：空闲超过 session_ttl_s 的会话必须被剔除，不再带入上下文。"""
    backend = _Backend()
    handlers = build_handlers(backend, _cfg(session_ttl_s=0.05), JobTable(),
                              sessions={}, sessions_lock=threading.Lock())
    handlers["chat"]({"messages": [{"role": "user", "content": "第一问"}],
                      "session": "s1"})
    assert "第一问" in [m["content"] for m in backend.calls[-1]]
    time.sleep(0.12)                     # 超过 TTL
    handlers["chat"]({"messages": [{"role": "user", "content": "第二问"}],
                      "session": "s1"})
    last = [m["content"] for m in backend.calls[-1]]
    assert "第一问" not in last, "过期会话仍被带入上下文"
    assert "第二问" in last


def test_batch_truncates_segment_count_as_valid_json():
    """P2：段数超限不再 raise，改为合法 JSON 截断（truncated/total/processed）。"""
    backend = _Backend()
    handlers = build_handlers(backend, _cfg(batch_max_segments=2), JobTable())
    result = handlers["batch"]({"segments": ["a", "b", "c", "d"]})
    payload = json.loads(result["content"][0]["text"])   # 必须是合法 JSON
    assert payload["truncated"] is True
    assert payload["total"] == 4
    assert payload["processed"] == 2
    assert payload["results"] == ["回答", "回答"]
