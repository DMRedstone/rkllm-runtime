"""P2.5 MCP 连接器回归（真实 stdio / JSON-RPC / 临时 TOML / 假板端 HTTP）。

覆盖待办 §2.5 的 9 项：
1) TOML 合法 `inf`/`1e400` 不崩、配置范围/枚举校验；
2) 板端 400/401/流中错误归一为有语义 MCP 结果（不是统一 -32603），保留 retryAfter；
3) 按需 `stream_options.include_usage`，usage 进 `_meta`；
4) `protocol_versions` 收缩时 fallback 一致 + 2026 无状态/旧版 initialize 边界；
5) `mcp_conformance.py`：recv 超时、--verbose、取消断言（由真实脚本执行验证）；
6) 无 id 的 tools/call 不阻塞 reader，明确 -32600；
7) prefill 用板端 ttft_s、token 预算用 bytes_per_token；
8) 会话 LRU 淘汰；
9) 畸形 JSON Schema -32602；8 MiB 限制在整行读入前生效。

仿真方式：连"假板端"用 stdlib HTTP server 真实走 HTTP/SSE；跨进程用例用真实
`python -m rkllm.mcp` stdio 子进程 + 真实 mcp.toml；只有 NPU/真实板端不可用。
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
CONNECTOR = REPO / "本地模型连接器"
CONFORMANCE = REPO / "工具" / "脚本" / "mcp_conformance.py"

sys.path.insert(0, str(PKG / "src"))

from rkllm.mcp import jobs as jobs_mod  # noqa: E402
from rkllm.mcp import protocol as protocol_mod  # noqa: E402
from rkllm.mcp.__main__ import default_cfg, load_cfg  # noqa: E402
from rkllm.mcp.backend import BackendError, HttpBackend  # noqa: E402
from rkllm.mcp.jobs import JobTable  # noqa: E402
from rkllm.mcp.stdio import MAX_MESSAGE_BYTES, StdioServer  # noqa: E402
from rkllm.mcp.tools import build_handlers  # noqa: E402


# ===========================================================================
# 假板端（真实 HTTP/SSE）
# ===========================================================================


class _BoardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    mode = "ok"                 # ok / 400 / 401 / 403 / 429 / 503 / stream_error
    delay = 0.0
    tokens = ("你", "好")
    health = {"state": "ready", "model_class": "4b", "context_len": 4096,
              "perf": {"decode_tok_s": 5.0, "ttft_s": 1.2, "source": "measured",
                       "samples": 10},
              "limits": {"max_tokens_limit": 2048, "default_max_tokens": 512,
                         "context_margin_tokens": 32, "request_timeout_s": 600},
              "tokens": {"backend": "estimate", "bytes_per_token": 3.0}}
    chat_calls = 0
    reset_calls = 0
    last_body: dict = {}

    def log_message(self, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, OSError):
            self.close_connection = True

    def _json(self, status: int, payload: dict, headers: dict | None = None):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._json(200, dict(_BoardHandler.health))

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        if self.path.rstrip("/").endswith("/reset"):
            _BoardHandler.reset_calls += 1
            self._json(200, {})
            return
        if _BoardHandler.mode == "400":
            self._json(400, {"error": {"type": "invalid_request_error",
                                       "message": "context exceeded"}})
            return
        if _BoardHandler.mode == "401":
            self._json(401, {"error": {"message": "invalid api key"}})
            return
        if _BoardHandler.mode == "403":
            self._json(403, {"error": {"message": "forbidden"}})
            return
        if _BoardHandler.mode == "429":
            self._json(429, {"error": {"message": "slow down"}},
                       headers={"Retry-After": "7"})
            return
        if _BoardHandler.mode == "503":
            self._json(503, {"error": {"message": "loading"}},
                       headers={"Retry-After": "3"})
            return
        _BoardHandler.chat_calls += 1
        if isinstance(body, dict):
            _BoardHandler.last_body = body
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if _BoardHandler.mode == "stream_error":
            err = {"error": {"type": "server_error", "code": "runtime_error",
                             "message": "NPU 推理中断"}}
            self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        for token in _BoardHandler.tokens:
            if _BoardHandler.delay:
                time.sleep(_BoardHandler.delay)
            chunk = {"choices": [{"delta": {"content": token}}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        finish = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(b"data: " + json.dumps(finish).encode() + b"\n\n")
        stream_options = body.get("stream_options") if isinstance(body, dict) else None
        if isinstance(stream_options, dict) and stream_options.get("include_usage"):
            usage = {"id": "chatcmpl-x", "object": "chat.completion.chunk",
                     "created": 0, "model": "rkllm", "choices": [],
                     "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                               "total_tokens": 7}}
            self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@contextmanager
def board(mode: str = "ok", delay: float = 0.0, tokens=None, health=None):
    _BoardHandler.mode = mode
    _BoardHandler.delay = delay
    _BoardHandler.tokens = tokens or ("你", "好")
    if health is not None:
        _BoardHandler.health = health
    _BoardHandler.chat_calls = 0
    _BoardHandler.reset_calls = 0
    _BoardHandler.last_body = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BoardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d/v1" % server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


# ===========================================================================
# 临时 TOML / stdio 子进程 / 内嵌 StdioServer
# ===========================================================================


def write_toml(tmp_path: Path, endpoint: str, extra: dict | None = None) -> Path:
    """写真实 mcp.toml（连接器默认读取的安装目录配置形态）。"""
    lines = ["[mcp]", 'target_endpoint = "%s"' % endpoint,
             'target_api_key = "test-key"']
    for key, value in (extra or {}).items():
        if isinstance(value, str):
            lines.append('%s = "%s"' % (key, value))
        elif isinstance(value, bool):
            lines.append("%s = %s" % (key, "true" if value else "false"))
        elif isinstance(value, (int, float)):
            lines.append("%s = %r" % (key, value))
        else:
            lines.append("%s = %s" % (key, json.dumps(value)))
    path = tmp_path / "mcp.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items()
            if not k.upper().startswith("RKLLM_")}


class Connector:
    """真实 `python -m rkllm.mcp` stdio 子进程客户端（后台读线程 + 队列超时）。"""

    def __init__(self, toml_path: Path):
        env = _clean_env()
        env.update({"PYTHONPATH": str(CONNECTOR),
                    "PYTHONIOENCODING": "utf-8",
                    "RKLLM_MCP_CONFIG": str(toml_path)})
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "rkllm.mcp"], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self._lines: queue.Queue = queue.Queue()
        self._pump = threading.Thread(target=self._read_stdout, daemon=True)
        self._pump.start()
        self._id = 0

    def _read_stdout(self) -> None:
        try:
            for line in self.proc.stdout:
                self._lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(None)

    def send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def send_raw(self, text: str) -> None:
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def next_id(self) -> int:
        self._id += 1
        return self._id

    def recv(self, want_id=None, timeout: float = 10.0) -> dict | None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                return None
            if line is None:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if want_id is None or msg.get("id") == want_id:
                return msg

    def call(self, method: str, params: dict | None = None,
             timeout: float = 10.0) -> dict | None:
        rid = self.next_id()
        self.send({"jsonrpc": "2.0", "id": rid, "method": method,
                   "params": params or {}})
        return self.recv(rid, timeout)

    def initialize(self, version: str = "2025-11-25") -> dict | None:
        reply = self.call("initialize", {"protocolVersion": version,
                                         "capabilities": {},
                                         "clientInfo": {"name": "pytest",
                                                        "version": "1"}})
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return reply

    def close(self) -> str:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        try:
            return self.proc.stderr.read() or ""
        except Exception:
            return ""


class _StubBackend:
    """进程内后端替身：可控 status/chat，记录调用。"""

    def __init__(self, text: str = "回答", *, status: dict | None = None,
                 error: BackendError | None = None):
        self.text = text
        self._status = status or {"state": "ready"}
        self.error = error
        self.calls: list[tuple[list[dict], dict]] = []
        self.status_calls = 0

    def status(self):
        self.status_calls += 1
        return dict(self._status)

    def chat(self, messages, **kwargs):
        self.calls.append((list(messages), dict(kwargs)))
        if self.error is not None:
            raise self.error
        return {"text": self.text, "usage": kwargs.get("usage"),
                "finish_reason": "stop", "reasoning_chars": 0}


def _server(backend=None, *, protocol_version="2024-11-05", **cfg_over):
    cfg = default_cfg("http://x/v1", "k")
    cfg.update(cfg_over)
    srv = StdioServer(backend or _StubBackend(), cfg,
                      protocol_version=protocol_version)
    srv._refresh_model_class = lambda: None
    return srv


def _shutdown(srv: StdioServer) -> None:
    srv._board_refresh_stop.set()
    for job in srv.jobs.all():
        job.cancel()
    srv._control_pool.shutdown(wait=False, cancel_futures=True)
    srv._pool.shutdown(wait=False, cancel_futures=True)


def _drive(srv: StdioServer, messages: list[dict]) -> list[dict]:
    """把消息喂给真实 `_read_line` 并等响应（tools/call 走线程池，是异步的）。"""
    out: list[dict] = []
    srv._send = lambda obj: out.append(obj)
    expected = sum(1 for m in messages
                   if not str(m.get("method", "")).startswith("notifications/"))
    for message in messages:
        srv._read_line(json.dumps(message, ensure_ascii=False))
    deadline = time.monotonic() + 10
    while len(out) < expected and time.monotonic() < deadline:
        time.sleep(0.01)
    return out


def _rpc(method: str, params: dict | None = None, msg_id: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "method": method,
            "params": params or {}}


def _tool_call(name: str, arguments: dict, msg_id: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
            "params": {"name": "local_llm_" + name, "arguments": arguments}}


# ===========================================================================
# 1) 配置：inf/1e400 不崩 + 范围/枚举校验（真实临时 TOML）
# ===========================================================================


def test_toml_inf_and_1e400_do_not_crash(monkeypatch, tmp_path, capsys):
    toml = tmp_path / "mcp.toml"
    toml.write_text(
        "[mcp]\n"
        "target_timeout_s = inf\n"
        "client_timeout_s = 1e400\n"
        "default_max_tokens = inf\n"
        "default_temperature = inf\n"
        "batch_max_segments = 1e400\n"
        "job_max = inf\n", encoding="utf-8")
    monkeypatch.setenv("RKLLM_MCP_CONFIG", str(toml))
    cfg = load_cfg()                       # 旧实现这里 OverflowError → 启动崩溃
    assert cfg["target_timeout_s"] == 120
    assert cfg["client_timeout_s"] == 60
    assert cfg["default_max_tokens"] == 512
    assert cfg["default_temperature"] == pytest.approx(0.3)
    assert cfg["batch_max_segments"] == 20
    assert cfg["job_max"] == 8
    err = capsys.readouterr().err
    assert "配置项不可用，已回退默认" in err


def test_config_range_enum_endpoint_and_versions(monkeypatch, tmp_path, capsys):
    toml = tmp_path / "mcp.toml"
    toml.write_text(
        "[mcp]\n"
        'target_endpoint = "not-a-url"\n'
        "batch_max_segments = 10000\n"
        "max_concurrent = 0\n"
        "session_ttl_s = -5\n"
        'task_mode = "bogus"\n'
        'backend = "nope"\n'
        'tools = ["ask", "not_a_tool"]\n'
        'protocol_versions = ["2024-11-05", "1999-01-01"]\n',
        encoding="utf-8")
    monkeypatch.setenv("RKLLM_MCP_CONFIG", str(toml))
    cfg = load_cfg()
    assert cfg["endpoint"] == "http://127.0.0.1:8080/v1"
    assert cfg["batch_max_segments"] == 20
    assert cfg["max_concurrent"] == 1
    assert cfg["session_ttl_s"] == 1800          # 负值非法 → 回退默认 TTL
    assert cfg["task_mode"] == "auto"
    assert cfg["backend"] == "auto"
    assert cfg["tools"] == ["ask"]
    assert cfg["protocol_versions"] == ["2024-11-05"]
    err = capsys.readouterr().err
    assert err.count("配置项不可用，已回退默认") >= 5


def test_config_keeps_legal_custom_values(monkeypatch, tmp_path, capsys):
    toml = tmp_path / "mcp.toml"
    toml.write_text(
        "[mcp]\n"
        "target_endpoint = \"http://192.168.1.9:8080/v1\"\n"
        "batch_max_segments = 7\n"
        "client_timeout_s = 90\n"
        "default_temperature = 0.7\n"
        "task_mode = \"never\"\n"
        "protocol_versions = [\"2025-11-25\", \"2024-11-05\"]\n",
        encoding="utf-8")
    monkeypatch.setenv("RKLLM_MCP_CONFIG", str(toml))
    cfg = load_cfg()
    assert cfg["endpoint"] == "http://192.168.1.9:8080/v1"
    assert cfg["batch_max_segments"] == 7
    assert cfg["client_timeout_s"] == 90
    assert cfg["default_temperature"] == pytest.approx(0.7)
    assert cfg["task_mode"] == "never"
    assert cfg["protocol_versions"] == ["2025-11-25", "2024-11-05"]
    assert "配置项不可用" not in capsys.readouterr().err


def test_env_inf_override_does_not_crash(monkeypatch, tmp_path, capsys):
    toml = tmp_path / "mcp.toml"
    toml.write_text("[mcp]\n", encoding="utf-8")
    monkeypatch.setenv("RKLLM_MCP_CONFIG", str(toml))
    monkeypatch.setenv("RKLLM_MCP_BATCH_MAX_SEGMENTS", "inf")
    monkeypatch.setenv("RKLLM_MCP_DEFAULT_MAX_TOKENS", "1e400")
    cfg = load_cfg()
    assert cfg["batch_max_segments"] == 20
    assert cfg["default_max_tokens"] == 512
    assert "配置项不可用" in capsys.readouterr().err


# ===========================================================================
# 2) 板端 400/401/流错误 → 有语义 MCP 错误（保留 retryAfter）
# ===========================================================================


@pytest.mark.parametrize("mode,kind,status", [
    ("400", "bad_request", 400),
    ("401", "auth", 401),
    ("403", "auth", 403),
    ("stream_error", "stream", None),
])
def test_http_and_stream_errors_are_classified(mode, kind, status):
    with board(mode) as endpoint:
        backend = HttpBackend(endpoint, "test-key")
        with pytest.raises(BackendError) as info:
            backend.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    assert info.value.kind == kind
    assert info.value.http_status == status
    assert info.value.retry_after_s is None


@pytest.mark.parametrize("mode,kind,retry", [
    ("429", "rate_limited", 7),
    ("503", "busy", 3),
])
def test_http_retry_after_preserved(mode, kind, retry):
    with board(mode) as endpoint:
        backend = HttpBackend(endpoint, "test-key")
        with pytest.raises(BackendError) as info:
            backend.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    assert info.value.kind == kind
    assert info.value.retry_after_s == retry
    assert info.value.meta()["retryAfter"] == retry


@pytest.mark.parametrize("kind,status", [
    ("bad_request", 400), ("auth", 401), ("stream", None),
    ("rate_limited", 429), ("busy", 503),
])
def test_tools_call_returns_semantic_iserror(kind, status):
    error = BackendError("板端错误", kind=kind, http_status=status,
                         retry_after_s=5 if kind in ("rate_limited", "busy")
                         else None)
    srv = _server(_StubBackend(error=error))
    try:
        # 同步路径（默认 ask 估算 < 45 s，不会转 job）
        reply = srv._tools_call({"name": "local_llm_ask",
                                 "arguments": {"prompt": "x"}})
    finally:
        _shutdown(srv)
    assert reply["isError"] is True
    meta = reply["_meta"]
    assert meta["kind"] == kind
    if status is not None:
        assert meta["httpStatus"] == status
    if kind in ("busy", "rate_limited"):
        assert meta["retryAfter"] == 5 and meta["busy"] is True
    assert "板端" in reply["content"][0]["text"]


def test_generic_backend_error_still_raises():
    """非语义通用错误保持旧语义（由 dispatcher 兜成 -32603，兼容既有回归）。"""
    error = BackendError("内部连接失败", kind="error")
    srv = _server(_StubBackend(error=error))
    try:
        with pytest.raises(BackendError):
            srv._tools_call({"name": "local_llm_ask",
                             "arguments": {"prompt": "x"}})
    finally:
        _shutdown(srv)


def test_job_failure_carries_semantic_meta():
    """job 路径的 auth 失败也要带 kind/httpStatus（客户端可据 meta 改配置）。"""
    error = BackendError("板端鉴权失败", kind="auth", http_status=401)
    srv = _server(_StubBackend(error=error), task_mode="always")
    try:
        reply = srv._tools_call({"name": "local_llm_ask",
                                 "arguments": {"prompt": "x"}})
        payload = json.loads(reply["content"][0]["text"])
        job = srv.jobs.get(payload["job_id"])
        deadline = time.monotonic() + 3
        while job.state == "working" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert job.state == "failed"
        assert job.error_meta.get("kind") == "auth"
        assert job.error_meta.get("httpStatus") == 401
        polled = srv._job_tool({"job_id": job.id, "action": "get"})
        assert polled["isError"] is True
        assert polled["_meta"]["kind"] == "auth"
    finally:
        _shutdown(srv)


# ===========================================================================
# 3) include_usage：请求按需发送 + usage 进 _meta
# ===========================================================================


def test_backend_requests_include_usage_and_parses_usage():
    with board() as endpoint:
        backend = HttpBackend(endpoint, "test-key")
        result = backend.chat([{"role": "user", "content": "hi"}], max_tokens=8)
        body = _BoardHandler.last_body
    assert body["stream_options"] == {"include_usage": True}
    assert result["usage"]["completion_tokens"] == 2
    with board() as endpoint:
        backend = HttpBackend(endpoint, "test-key")
        backend.chat([{"role": "user", "content": "hi"}], max_tokens=8,
                     include_usage=False)
        body = _BoardHandler.last_body
    assert "stream_options" not in body


def test_tool_meta_usage_from_board(tmp_path):
    with board() as endpoint:
        cfg = default_cfg(endpoint, "test-key")
        cfg["target_timeout_s"] = 30
        srv = StdioServer(HttpBackend(endpoint, "test-key"), cfg)
        try:
            reply = srv._tools_call({"name": "local_llm_ask",
                                     "arguments": {"prompt": "hi"}})
        finally:
            _shutdown(srv)
    assert reply.get("isError") is not True
    usage = reply["_meta"]["usage"]
    assert usage["prompt_tokens"] == 5 and usage["completion_tokens"] == 2
    assert reply["_meta"]["finish_reason"] == "stop"


# ===========================================================================
# 4) protocol_versions fallback / 2026 无状态 / 旧版 initialize 边界
# ===========================================================================


def test_detect_protocol_fallback_inside_supported():
    assert protocol_mod.detect_protocol(
        {"method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        supported=["2025-11-25"]) == "2025-11-25"
    assert protocol_mod.detect_protocol(
        {"method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        supported=["2026-07-28"]) == "2026-07-28"
    assert protocol_mod.detect_protocol(
        {"method": "tools/call", "params": {"_meta": {
            "protocolVersion": "2026-07-28"}}},
        supported=["2024-11-05"]) == "2024-11-05"
    # 默认配置保持历史行为
    assert protocol_mod.detect_protocol(
        {"method": "initialize", "params": {"protocolVersion": "1999-01-01"}}
    ) == "2024-11-05"


def test_shrunk_supported_initialize_returns_supported_version():
    srv = _server(protocol_versions=["2025-11-25"], protocol_version=None)
    try:
        frames = _drive(srv, [_rpc("initialize", {
            "protocolVersion": "2024-11-05"})])
    finally:
        _shutdown(srv)
    result = frames[0]["result"]
    assert result["protocolVersion"] == "2025-11-25"
    assert "2024-11-05" not in srv.dispatcher.supported


def test_2026_only_config_rejects_initialize_with_supported_list():
    srv = _server(protocol_versions=["2026-07-28"], protocol_version=None)
    try:
        frames = _drive(srv, [_rpc("initialize", {
            "protocolVersion": "2024-11-05"})])
    finally:
        _shutdown(srv)
    error = frames[0]["error"]
    assert error["code"] == -32601
    assert error["data"]["supportedVersions"] == ["2026-07-28"]


def test_2026_stateless_first_frame_without_initialize():
    srv = _server(protocol_versions=["2026-07-28"], protocol_version=None)
    try:
        frames = _drive(srv, [{
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "local_llm_status", "arguments": {},
                       "_meta": {"protocolVersion": "2026-07-28"}}}])
    finally:
        _shutdown(srv)
    assert frames[0]["result"]["resultType"] == "complete"
    assert frames[0]["result"]["content"][0]["text"].startswith("模型状态")


def test_old_initialize_then_initialize_again_is_ok():
    """旧版 initialize 后重复握手不应破坏服务（客户端重连常见）。"""
    srv = _server(protocol_version=None)
    try:
        frames = _drive(srv, [
            _rpc("initialize", {"protocolVersion": "2025-11-25"}, 1),
            _rpc("initialize", {"protocolVersion": "2025-11-25"}, 2),
            _rpc("tools/list", {}, 3),
        ])
    finally:
        _shutdown(srv)
    assert frames[0]["result"]["protocolVersion"] == "2025-11-25"
    assert len(frames[2]["result"]["tools"]) == 6


# ===========================================================================
# 5) mcp_conformance：脚本真实执行 + recv 超时真正生效
# ===========================================================================


def _load_conformance():
    spec = importlib.util.spec_from_file_location("mcp_conformance_under_test",
                                                  str(CONFORMANCE))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_conformance_script_45_of_45_verbose():
    result = subprocess.run(
        [sys.executable, str(CONFORMANCE), "--verbose"],
        cwd=str(REPO), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "45/45" in result.stdout
    assert result.stdout.count("PASS") >= 45          # --verbose 真的逐条打印


def test_conformance_recv_timeout_effective():
    module = _load_conformance()
    with board() as endpoint:
        port = int(endpoint.split(":")[2].split("/")[0])
        conn = module.Connector(port)
        try:
            t0 = time.monotonic()
            missing = conn.recv(want_id=99999, timeout=0.3)
            elapsed = time.monotonic() - t0
            assert missing is None
            assert elapsed < 2.0, "recv 超时未生效，阻塞了 %.1fs" % elapsed
        finally:
            conn.close()


# ===========================================================================
# 6) 无 id 的 tools/call：拒绝且不阻塞 reader
# ===========================================================================


def test_no_id_tools_call_rejected_locally():
    srv = _server()
    backend = srv.backend
    try:
        frames = _drive(srv, [{"jsonrpc": "2.0", "method": "tools/call",
                               "params": {"name": "local_llm_ask",
                                          "arguments": {"prompt": "x"}}}])
    finally:
        _shutdown(srv)
    assert frames and frames[0]["id"] is None
    assert frames[0]["error"]["code"] == -32600
    assert backend.calls == [], "无 id 的 tools/call 不应真的执行"


def test_no_id_request_rejected_by_dispatcher():
    from rkllm.mcp.protocol import Dispatcher
    disp = Dispatcher("2024-11-05")
    disp.register("ping", lambda params: {})
    resp = disp.handle_message(json.dumps(
        {"jsonrpc": "2.0", "method": "ping", "params": {}}))
    assert resp["error"]["code"] == -32600
    assert resp["id"] is None
    # 通知仍不回包、不报错
    assert disp.handle_message(json.dumps(
        {"jsonrpc": "2.0", "method": "notifications/initialized"})) is None


def test_real_stdio_no_id_tools_call_does_not_block(tmp_path):
    with board(delay=2.0) as endpoint:          # 若被执行会阻塞 ~4 s
        toml = write_toml(tmp_path, endpoint)
        conn = Connector(toml)
        try:
            assert conn.initialize()
            # 无 id 的 tools/call：必须立刻被拒且不执行
            conn.send({"jsonrpc": "2.0", "method": "tools/call",
                       "params": {"name": "local_llm_ask",
                                  "arguments": {"prompt": "长任务"}}})
            base = time.monotonic()
            nid = conn.next_id()
            conn.send({"jsonrpc": "2.0", "id": nid, "method": "tools/list",
                       "params": {}})
            listed = conn.recv(nid, timeout=5)
            elapsed = time.monotonic() - base
        finally:
            conn.close()
    assert listed and listed.get("result"), listed
    assert elapsed < 1.5, "reader 被无 id 的 tools/call 阻塞 %.1fs" % elapsed
    assert _BoardHandler.chat_calls == 0


# ===========================================================================
# 7) prefill 用板端 ttft_s；token 预算用板端 bytes_per_token
# ===========================================================================


def _wait_board(cfg, key, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (cfg.get("_board") or {}).get(key):
            return True
        time.sleep(0.02)
    return False


def test_board_refresh_reads_ttft_and_bytes_per_token():
    backend = _StubBackend(status={
        "state": "ready", "context_len": 4096,
        "perf": {"decode_tok_s": 8.0, "ttft_s": 2.5, "source": "measured"},
        "limits": {"default_max_tokens": 512, "context_margin_tokens": 32},
        "tokens": {"backend": "estimate", "bytes_per_token": 2.0}})
    cfg = default_cfg("http://x/v1", "k")
    srv = StdioServer(backend, cfg)
    try:
        assert _wait_board(cfg, "ttft_s") and _wait_board(cfg, "bytes_per_token")
        assert cfg["_board"]["ttft_s"] == pytest.approx(2.5)
        assert cfg["_board"]["bytes_per_token"] == pytest.approx(2.0)
    finally:
        _shutdown(srv)


def test_prefill_estimate_uses_board_ttft_not_constant():
    cold = _server()
    try:
        cold_est = cold._default_estimate({}, "ask")
    finally:
        _shutdown(cold)
    backend = _StubBackend(status={
        "state": "ready", "perf": {"decode_tok_s": 5.0, "ttft_s": 50.0,
                                   "source": "measured"},
        "limits": {"default_max_tokens": 512}})
    cfg = default_cfg("http://x/v1", "k")
    srv = StdioServer(backend, cfg)
    try:
        assert _wait_board(cfg, "ttft_s")
        warm_est = srv._default_estimate({}, "ask")
        assert warm_est > cold_est
        assert warm_est > 45, "ttft_s=50 时应超过转 job 阈值"
        assert srv._sync_token_cap({}, "ask") == 8
    finally:
        _shutdown(srv)


def test_jobs_cold_estimate_stays_compatible_and_ttft_used():
    """冷启动公式不变（既有测试基线），有 ttft 时用它。"""
    assert jobs_mod.estimate_seconds("4b", 0, 512) == pytest.approx(102.4, abs=0.1)
    assert jobs_mod.estimate_seconds("4b", 44, 0) == pytest.approx(1.0)
    assert jobs_mod.estimate_seconds("4b", 10, 0, ttft_s=10.0) == pytest.approx(10.0)
    # 长 prompt 不被短 prompt 的 ttft 低估（保守）
    assert jobs_mod.estimate_seconds("4b", 4400, 0, ttft_s=1.0) == \
        pytest.approx(100.0)


def test_token_budget_uses_board_bytes_per_token():
    class _Backend(_StubBackend):
        def chat(self, messages, **kwargs):
            self.calls.append((list(messages), dict(kwargs)))
            return {"text": "ok", "usage": None, "finish_reason": "stop",
                    "reasoning_chars": 0}

    backend = _Backend()
    cfg = default_cfg("http://x/v1", "k")
    cfg["_board"] = {"context_len": 4096, "context_margin_tokens": 32,
                     "bytes_per_token": 6.0}
    handlers = build_handlers(backend, cfg, JobTable(),
                              extract_retries=0)
    handlers["ask"]({"prompt": "字" * 12000})       # 36000 字节 > 预算字节
    sent = backend.calls[-1][0][-1]["content"]
    budget_tokens = 4096 - 32 - 512
    max_bytes = budget_tokens * 6 + 200             # +截断提示
    assert len(sent.encode("utf-8")) <= max_bytes
    assert "截断" in sent


# ===========================================================================
# 8) 会话淘汰：真 LRU（不是 FIFO）
# ===========================================================================


def test_session_eviction_is_lru():
    store: dict[str, list[dict]] = {}
    lock = threading.Lock()
    backend = _StubBackend()
    cfg = default_cfg("http://x/v1", "k")
    cfg.update({"session_max_count": 2, "session_max_messages": 0,
                "context_len": 8192, "default_max_tokens": 64})
    handlers = build_handlers(backend, cfg, JobTable(),
                              sessions=store, sessions_lock=lock)
    handlers["chat"]({"messages": [{"role": "user", "content": "一"}],
                      "session": "s1"})
    time.sleep(0.02)
    handlers["chat"]({"messages": [{"role": "user", "content": "二"}],
                      "session": "s2"})
    time.sleep(0.02)
    # 再次访问 s1 → s1 变成最近使用；s2 才是 LRU
    handlers["chat"]({"messages": [{"role": "user", "content": "一续"}],
                      "session": "s1"})
    time.sleep(0.02)
    handlers["chat"]({"messages": [{"role": "user", "content": "三"}],
                      "session": "s3"})
    assert set(store) == {"s1", "s3"}, "淘汰的不是最久未用的会话：%s" % list(store)
    # s2 的历史确实丢了：重建会话只剩本条
    handlers["chat"]({"messages": [{"role": "user", "content": "二新"}],
                      "session": "s2"})
    sent = [m["content"] for m in backend.calls[-1][0]]
    assert "二" not in sent and "二新" in sent


# ===========================================================================
# 9) 畸形 JSON Schema → -32602；8 MiB 限制在整行读入前生效
# ===========================================================================


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": []},
    {"type": "object", "required": "name"},
    {"type": "object", "items": 123},
    {"type": "not_a_type"},
    {"type": "object", "properties": {"a": {"type": 123}}},
    {"type": "object", "properties": {"a": {"minimum": "x"}}},
])
def test_malformed_schema_is_invalid_params(schema):
    srv = _server()
    try:
        frames = _drive(srv, [_tool_call("extract", {
            "text": "张三 35 岁", "schema": schema})])
    finally:
        _shutdown(srv)
    assert frames[0]["error"]["code"] == -32602, frames[0]


def test_deeply_nested_schema_is_invalid_params():
    schema: dict = {"type": "string"}
    for _ in range(40):
        schema = {"type": "object", "properties": {"x": schema}}
    srv = _server()
    try:
        frames = _drive(srv, [_tool_call("extract", {
            "text": "x", "schema": schema})])
    finally:
        _shutdown(srv)
    assert frames[0]["error"]["code"] == -32602


def test_valid_schema_still_works():
    """合法 schema 不误伤：extract 转 job 后轮询拿结构化结果（真实路径）。"""
    backend = _StubBackend(text='{"name": "张三"}')
    srv = _server(backend)
    try:
        reply = srv._tools_call({"name": "local_llm_extract",
                                 "arguments": {
                                     "text": "张三",
                                     "schema": {"type": "object",
                                                "required": ["name"],
                                                "properties": {
                                                    "name": {"type": "string"}}}}})
        payload = json.loads(reply["content"][0]["text"])
        job = srv.jobs.get(payload["job_id"])
        deadline = time.monotonic() + 5
        while job.state == "working" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert job.state == "completed", job.error
        polled = srv._job_tool({"job_id": job.id, "action": "get"})
    finally:
        _shutdown(srv)
    assert polled["structuredContent"] == {"name": "张三"}


class _TrackingStream(io.StringIO):
    """记录 readline(size) 的入参：证明 8 MiB 超长行没有整行进内存。"""

    def __init__(self, value: str = ""):
        super().__init__(value)
        self.sizes: list[int] = []

    def readline(self, size: int = -1) -> str:
        self.sizes.append(size)
        return super().readline(size)


def test_oversize_line_is_bounded_before_full_read(monkeypatch):
    huge_line = "x" * (MAX_MESSAGE_BYTES + 1024)
    stream = _TrackingStream(huge_line + "\n"
                             + json.dumps(_rpc("ping")) + "\n")
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stream)
    monkeypatch.setattr(sys, "stdout", out)
    srv = _server()
    try:
        srv.serve_forever()
    finally:
        _shutdown(srv)
    frames = [json.loads(line) for line in out.getvalue().splitlines() if line]
    assert any(f.get("error", {}).get("message") == "message too large"
               for f in frames)
    # 后续正常帧仍然被处理
    assert any(f.get("result") == {} for f in frames)
    # 关键：每次 readline 都带有限 size（没有 size=-1 的整行读取）
    assert stream.sizes, "没有走有界 readline"
    assert all(0 < size <= MAX_MESSAGE_BYTES + 2 for size in stream.sizes), \
        "存在无界整行读取：%r" % [s for s in stream.sizes if s <= 0][:5]


def test_real_stdio_oversize_then_alive(tmp_path):
    with board() as endpoint:
        toml = write_toml(tmp_path, endpoint)
        conn = Connector(toml)
        try:
            assert conn.initialize()
            conn.send_raw(json.dumps({
                "jsonrpc": "2.0", "id": 99, "method": "ping",
                "params": {"blob": "x" * (MAX_MESSAGE_BYTES + 1024)}}))
            reply = conn.recv(None, timeout=30)
            assert (reply or {}).get("error", {}).get("code") == -32700
            alive = conn.call("tools/list", {}, timeout=10)
        finally:
            conn.close()
    assert alive and alive.get("result")


# ===========================================================================
# 真实 stdio 端到端：板端错误语义 / usage / 畸形 schema
# ===========================================================================


def test_real_stdio_board_400_is_semantic_result(tmp_path):
    with board("400") as endpoint:
        toml = write_toml(tmp_path, endpoint)
        conn = Connector(toml)
        try:
            assert conn.initialize()
            reply = conn.call("tools/call", {
                "name": "local_llm_ask",
                "arguments": {"prompt": "hi", "max_tokens": 8}}, timeout=15)
        finally:
            conn.close()
    assert "result" in reply, reply          # 不是 JSON-RPC -32603
    result = reply["result"]
    assert result["isError"] is True
    assert result["_meta"]["kind"] == "bad_request"
    assert result["_meta"]["httpStatus"] == 400


def test_real_stdio_retry_after_semantic(tmp_path):
    with board("503") as endpoint:
        toml = write_toml(tmp_path, endpoint)
        conn = Connector(toml)
        try:
            assert conn.initialize()
            reply = conn.call("tools/call", {
                "name": "local_llm_ask",
                "arguments": {"prompt": "hi", "max_tokens": 8}}, timeout=15)
        finally:
            conn.close()
    result = reply["result"]
    assert result["isError"] is True
    assert result["_meta"]["kind"] == "busy"
    assert result["_meta"]["retryAfter"] == 3


def test_real_stdio_usage_roundtrip(tmp_path):
    with board() as endpoint:
        toml = write_toml(tmp_path, endpoint)
        conn = Connector(toml)
        try:
            assert conn.initialize()
            reply = conn.call("tools/call", {
                "name": "local_llm_ask",
                "arguments": {"prompt": "hi", "max_tokens": 8}}, timeout=15)
        finally:
            conn.close()
    result = reply["result"]
    assert result.get("isError") is not True
    assert result["_meta"]["usage"]["completion_tokens"] == 2
    assert _BoardHandler.last_body["stream_options"]["include_usage"] is True


def test_real_stdio_malformed_schema_invalid_params(tmp_path):
    with board() as endpoint:
        toml = write_toml(tmp_path, endpoint)
        conn = Connector(toml)
        try:
            assert conn.initialize()
            reply = conn.call("tools/call", {
                "name": "local_llm_extract",
                "arguments": {"text": "x",
                              "schema": {"type": "object",
                                         "properties": []}}}, timeout=15)
        finally:
            conn.close()
    assert reply["error"]["code"] == -32602
    assert not _BoardHandler.chat_calls
