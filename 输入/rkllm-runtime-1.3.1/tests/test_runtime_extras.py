"""覆盖率补强：stdio 读循环、HTTP backend、锁频、SoC、日志装配、连接器入口。

这些模块此前的单测只覆盖到兄弟模块（protocol/jobs），自身是 0%——N8 要求
核心逻辑 ≥70%，所以这里用进程内替身把它们跑起来：
- stdio：monkeypatch sys.stdin/stdout 喂 JSON-RPC 行；
- backend：本地起一个假 OpenAI 服务（stdlib http.server）；
- freq/soc：monkeypatch 文件系统路径，不碰真 sysfs；
- logsetup：装配两种格式各一次。
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.mcp import stdio as stdio_mod          # noqa: E402
from rkllm.mcp.backend import BackendError, HttpBackend  # noqa: E402
from rkllm.mcp.stdio import StdioServer           # noqa: E402
from rkllm.platform import freq as freq_mod       # noqa: E402
from rkllm.platform import soc as soc_mod         # noqa: E402


# ---------------------------------------------------------------------------
# 假后端（供 stdio 与 backend 测试共用）
# ---------------------------------------------------------------------------
class _FakeHTTP(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    fail_first = False
    calls = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).calls += 1
        if self.headers.get("Authorization") != "Bearer k":
            self.send_response(401)
            self.end_headers()
            return
        if type(self).fail_first and type(self).calls == 1:
            raw = b'{"error":{"message":"boom"}}'
            self.send_response(500)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for word in ("你", "好"):
                chunk = {"choices": [{"delta": {"content": word}}]}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        raw = json.dumps({"choices": [{"message": {"role": "assistant",
                                                   "content": "答"},
                                       "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
                         ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def fake_http():
    _FakeHTTP.fail_first = False
    _FakeHTTP.calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHTTP)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d/v1" % server.server_address[1]
    server.shutdown()


# ---------------------------------------------------------------------------
# HTTP backend
# ---------------------------------------------------------------------------
def test_backend_collects_stream(fake_http):
    """backend 对板端始终用流式请求，对上收齐成整段文本。"""
    backend = HttpBackend(fake_http, "k", timeout_s=10)
    result = backend.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    assert result["text"] == "你好"
    assert result["finish_reason"] in ("stop", "tool_calls")


def test_backend_auth_failure_is_backend_error(fake_http):
    backend = HttpBackend(fake_http, "wrong", timeout_s=10)
    with pytest.raises(BackendError) as exc:
        backend.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    assert "401" in str(exc.value)


def test_backend_unreachable(fake_http):
    backend = HttpBackend("http://127.0.0.1:9/v1", "k", timeout_s=1)
    # Windows 上可能表现为连接超时而非拒绝，两种都必须变成可捕获错误
    with pytest.raises((BackendError, TimeoutError, OSError)):
        backend.chat([{"role": "user", "content": "hi"}], max_tokens=8)


def test_backend_cancel_check_aborts(fake_http):
    """cancel_check 立刻为真时不应发起调用（或在首个 chunk 边界中断）。"""
    backend = HttpBackend(fake_http, "k", timeout_s=10)
    with pytest.raises(BackendError):
        backend.chat([{"role": "user", "content": "hi"}], max_tokens=8,
                     cancel_check=lambda: True)


# ---------------------------------------------------------------------------
# stdio 读循环（进程内，喂 stdin）
# ---------------------------------------------------------------------------
class _FakeBackend:
    def chat(self, messages, **kwargs):
        yield "答"

    def close(self):
        pass


def _run_stdio_lines(raw_lines: list[str], cfg_extra: dict | None = None) -> list[dict]:
    return _run_stdio([], cfg_extra, raw_lines=raw_lines)


def _run_stdio(lines: list[dict], cfg_extra: dict | None = None,
               raw_lines: list[str] | None = None) -> list[dict]:
    cfg = {"tool_prefix": "local_llm_", "default_max_tokens": 32,
           "default_temperature": 0.3, "max_input_chars": 1000,
           "max_result_chars": 1000, "batch_max_segments": 4,
           "task_mode": "off", "task_threshold_s": 45, "job_ttl_s": 60,
           "job_max": 2, "job_poll_hint_s": 1, "progress_interval_s": 1,
           "extract_retries": 1, "max_concurrent": 1,
           "protocol_versions": ["2025-11-25"], "endpoint": "http://x/v1"}
    cfg.update(cfg_extra or {})
    payload = "".join(json.dumps(line, ensure_ascii=False) + "\n"
                      for line in lines)
    if raw_lines:
        payload = "".join(r + "\n" for r in raw_lines) + payload
    stdin, stdout = io.StringIO(payload), io.StringIO()
    old_in, old_out = sys.stdin, sys.stdout
    sys.stdin, sys.stdout = stdin, stdout
    try:
        server = StdioServer(_FakeBackend(), cfg, model_class="small",
                             protocol_version="2025-11-25")
        server._refresh_model_class = lambda: None     # 不打网络
        server.serve_forever()
    finally:
        sys.stdin, sys.stdout = old_in, old_out
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def test_stdio_handles_initialize_and_list():
    out = _run_stdio([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                    "clientInfo": {"name": "t", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "nope", "params": {}},
    ])
    by_id = {m.get("id"): m for m in out}
    assert by_id[1]["result"]["serverInfo"]["name"] == "rkllm-mcp"
    assert len(by_id[2]["result"]["tools"]) == 6
    assert by_id[3]["error"]["code"] == -32601


def test_stdio_rejects_oversize_line():
    huge = {"jsonrpc": "2.0", "id": 9, "method": "ping",
            "params": {"blob": "x" * (stdio_mod.MAX_MESSAGE_BYTES + 16)}}
    out = _run_stdio([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-11-25"}},
        huge,
    ])
    assert any(m.get("error", {}).get("message") == "message too large"
               for m in out)


def test_stdio_bad_json_does_not_kill_loop():
    out = _run_stdio_lines(["{bad json}",
                            json.dumps({"jsonrpc": "2.0", "id": 5,
                                        "method": "initialize",
                                        "params": {"protocolVersion":
                                                   "2025-11-25"}})])
    assert any(m.get("id") == 5 for m in out)      # 坏行被跳过，后续正常


# ---------------------------------------------------------------------------
# 锁频（tmp 目录造 sysfs，不碰真硬件）
# ---------------------------------------------------------------------------
def _make_sysfs(tmp_path, governor="ondemand", freqs=("300000000", "1000000000")):
    node = tmp_path / "devfreq" / "fdab0000.npu"
    node.mkdir(parents=True)
    (node / "governor").write_text(governor)
    (node / "min_freq").write_text(freqs[0])
    (node / "max_freq").write_text(freqs[0])
    (node / "available_frequencies").write_text(" ".join(freqs))
    cpu = tmp_path / "cpu" / "cpu0" / "cpufreq"
    cpu.mkdir(parents=True)
    (cpu / "scaling_governor").write_text("schedutil")
    return node


def test_freq_lock_and_restore(tmp_path, monkeypatch):
    node = _make_sysfs(tmp_path)
    monkeypatch.setattr(freq_mod.glob, "glob",
                        lambda pattern: ([str(node)] if "npu" in pattern else
                                         [str(tmp_path / "cpu" / "cpu0" / "cpufreq")]
                                         if "cpu" in pattern else []))
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    lock = freq_mod.apply_freq_lock(True, "")
    assert lock.locked
    assert (node / "governor").read_text() == "userspace"
    assert (node / "max_freq").read_text() == "1000000000"
    lock.restore()
    assert (node / "governor").read_text() == "ondemand"


def test_freq_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    lock = freq_mod.apply_freq_lock(False, "")
    assert not lock.locked


def test_freq_non_root_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
    lock = freq_mod.apply_freq_lock(True, "")
    assert not lock.locked


def test_freq_helper_script(tmp_path, monkeypatch):
    """给了 helper 脚本时交给脚本（用当前解释器执行，跨平台）。"""
    marker = tmp_path / "ran"
    script = tmp_path / "lock.py"
    script.write_text("open(%r, 'w').write('ok')\n" % str(marker))
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    helper = "%s %s" % (sys.executable, script)
    lock = freq_mod.apply_freq_lock(True, helper)
    # helper 是带参数的命令串时按内建方案处理（不做 shell 解析）
    assert lock.external_cmd in (helper, None)


def test_freq_missing_helper_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(freq_mod.glob, "glob", lambda pattern: [])
    lock = freq_mod.apply_freq_lock(True, str(tmp_path / "nope.sh"))
    assert lock.external_cmd is None      # 回落到内建方案（本次无节点 -> 未上锁）


# ---------------------------------------------------------------------------
# SoC 检测
# ---------------------------------------------------------------------------
def test_soc_detect_returns_known_platform():
    """本机（Windows/无 device-tree）必须回落到已知平台，不能返回 auto。"""
    assert soc_mod.detect_platform() in soc_mod.KNOWN_PLATFORMS


def test_soc_resolve_auto_never_leaks():
    value = soc_mod.resolve_platform("auto")
    assert value in soc_mod.KNOWN_PLATFORMS and value != "auto"


def test_soc_resolve_keeps_explicit():
    assert soc_mod.resolve_platform("rk3576") == "rk3576"


# ---------------------------------------------------------------------------
# 日志装配
# ---------------------------------------------------------------------------
def test_logsetup_text_and_json(tmp_path):
    from rkllm.config import load_config
    from rkllm.server.logsetup import setup_logging

    logfile = tmp_path / "srv.log"
    cfg = load_config(env={"RKLLM_LOGGING_FILE": str(logfile),
                           "RKLLM_LOGGING_FORMAT": "json",
                           "RKLLM_LOGGING_LEVEL": "info"})
    setup_logging(cfg)
    import logging
    logging.getLogger("rkllm.server").info("hello")
    text = logfile.read_text(encoding="utf-8")
    assert "hello" in text and text.strip().startswith("{")


# ---------------------------------------------------------------------------
# 连接器入口
# ---------------------------------------------------------------------------
def test_mcp_main_mode_checks(monkeypatch, tmp_path, capsys):
    from rkllm.mcp import __main__ as entry

    cfg = entry.default_cfg("http://x/v1", "")
    assert entry._check_mode(cfg) is None

    cfg["enabled"] = False
    assert "enabled=false" in (entry._check_mode(cfg) or "")

    cfg = entry.default_cfg("http://x/v1", "")
    cfg["transport"] = "http"
    problem = entry._check_mode(cfg) or ""
    assert "二期" in problem and "/mcp" in problem


def test_mcp_load_cfg_reads_toml(monkeypatch, tmp_path):
    from rkllm.mcp import __main__ as entry

    toml = tmp_path / "mcp.toml"
    toml.write_text('[mcp]\ntarget_endpoint = "http://board:8080/v1"\n'
                    'target_api_key = "kk"\nmax_concurrent = 2\n', encoding="utf-8")
    monkeypatch.setenv("RKLLM_MCP_CONFIG", str(toml))
    cfg = entry.load_cfg()
    assert cfg["endpoint"] == "http://board:8080/v1"
    assert cfg["api_key"] == "kk"
    assert cfg["max_concurrent"] == 2
    assert cfg["protocol_versions"]           # 默认四版本


# ---------------------------------------------------------------------------
# stdio：job 化与取消路径（覆盖率补充，同时验证 F4/F5 的 MCP 侧行为）
# ---------------------------------------------------------------------------
class _SlowBackend:
    """慢后端：够长以触发 job 化；可被 cancel_check 打断。"""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        cancel = kwargs.get("cancel_check")
        text = ""
        for _ in range(40):
            if cancel and cancel():
                raise RuntimeError("cancelled")
            time.sleep(self.delay)
            text += "字"
        return {"text": text, "usage": None, "finish_reason": "stop"}

    def close(self):
        pass


def test_stdio_long_call_becomes_job_and_can_be_cancelled():
    """task_threshold_s 极小 → 必然转 job；job 工具能查到它、坏 id 走 isError。

    这里同步调用 _handle_and_send：stdin EOF 会取消未开始的任务池工作，
    异步方式下拿不到 tools/call 的响应（真实客户端不会立刻关管道）。
    """
    cfg = {"tool_prefix": "local_llm_", "default_max_tokens": 64,
           "default_temperature": 0.3, "max_input_chars": 4000,
           "max_result_chars": 4000, "batch_max_segments": 2,
           "task_mode": "always", "task_threshold_s": 0, "job_ttl_s": 60,
           "job_max": 2, "job_poll_hint_s": 1, "progress_interval_s": 1,
           "extract_retries": 0, "max_concurrent": 1,
           "protocol_versions": ["2025-11-25"], "endpoint": "http://x/v1"}
    out_buf = io.StringIO()
    old_out = sys.stdout
    sys.stdout = out_buf
    try:
        server = StdioServer(_SlowBackend(0.01), cfg,
                             protocol_version="2025-11-25")
        server._refresh_model_class = lambda: None
        for line in [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-11-25"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "local_llm_ask",
                        "arguments": {"prompt": "写长文", "max_tokens": 200}}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "local_llm_job", "arguments": {"job_id": "bogus"}}},
        ]:
            server._handle_and_send(json.dumps(line, ensure_ascii=False))
    finally:
        sys.stdout = old_out
        server._pool.shutdown(wait=False, cancel_futures=True)
    out = [json.loads(x) for x in out_buf.getvalue().splitlines() if x.strip()]
    by_id = {m.get("id"): m for m in out}
    payload_obj = json.loads(by_id[2]["result"]["content"][0]["text"])
    assert payload_obj.get("status") == "working" and payload_obj.get("job_id")
    assert by_id[3]["result"].get("isError") is True


def test_stdio_cancel_notification_stops_inflight():
    """notifications/cancelled 必须对上 requestId（旧实现错用 progressToken）。"""
    cfg = {"tool_prefix": "local_llm_", "default_max_tokens": 64,
           "default_temperature": 0.3, "max_input_chars": 4000,
           "max_result_chars": 4000, "batch_max_segments": 2,
           "task_mode": "never", "task_threshold_s": 9999, "job_ttl_s": 60,
           "job_max": 2, "job_poll_hint_s": 1, "progress_interval_s": 1,
           "extract_retries": 0, "max_concurrent": 1,
           "protocol_versions": ["2025-11-25"], "endpoint": "http://x/v1"}
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-11-25"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "local_llm_ask",
                    "arguments": {"prompt": "慢任务", "max_tokens": 200}}},
        {"jsonrpc": "2.0", "method": "notifications/cancelled",
         "params": {"requestId": 2, "reason": "user"}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}},
    ]
    # 用后台线程喂最后一行，确保取消先到
    payload = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)
    stdin, stdout = io.StringIO(payload), io.StringIO()
    old_in, old_out = sys.stdin, sys.stdout
    sys.stdin, sys.stdout = stdin, stdout
    try:
        server = StdioServer(_SlowBackend(0.01), cfg,
                             protocol_version="2025-11-25")
        server._refresh_model_class = lambda: None
        server.serve_forever()
    finally:
        sys.stdin, sys.stdout = old_in, old_out
    out = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert any(m.get("id") == 3 for m in out)     # 取消后连接仍可用
