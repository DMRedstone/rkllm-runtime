"""board_api 传输层回归（httpx 版）。

用一个本地 stdlib HTTP 服务当靶子，覆盖真实调用路径：/health、非流式 chat、
SSE 流式、4xx 语义、连接失败。不依赖板端、不依赖 pytest 之外的包
（httpx 是工具链自身的依赖）。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import board_api


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):        # 静默
        pass

    def _json(self, code: int, payload: dict, extra_headers=None):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                     # noqa: N802
        if self.path.startswith("/health"):
            if self.headers.get("Authorization") == "Bearer bad":
                self._json(401, {"error": {"message": "unauthorized"}})
                return
            self._json(200, {"state": "ready", "model_class": "4b"})
            return
        if self.path.startswith("/empty-health"):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):                    # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw or b"{}")
        except Exception:                 # noqa: BLE001
            payload = {}
        if self.path.startswith("/v1/chat/completions"):
            if payload.get("stream"):
                self._sse()
            else:
                self._json(200, {
                    "choices": [{"message": {"role": "assistant",
                                             "content": "答：%s"
                                                        % payload.get("messages", [{}])[0].get("content", "")}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2,
                              "total_tokens": 5}})
            return
        if self.path.startswith("/v1/reset"):
            self._json(200, {"ok": True})
            return
        if self.path.startswith("/v1/toomuch"):
            self._json(400, {"error": {"message": "max_tokens too large"}})
            return
        self._json(404, {"error": {"message": "not found"}})

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        frames = [
            {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"delta": {"content": "你"}}]},
            {"choices": [{"delta": {"content": "好"}}]},
        ]
        for frame in frames:
            self.wfile.write(("data: %s\n\n" % json.dumps(frame)).encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def setup_module(module):                # noqa: D103
    module.server, module.thread = _serve()
    module.host = "127.0.0.1:%d" % module.server.server_address[1]


def teardown_module(module):             # noqa: D103
    module.server.shutdown()
    module.server.server_close()


def test_health_ready():
    board = board_api.Board(host, key="k")
    info = board.health()
    assert info and info["state"] == "ready" and info["model_class"] == "4b"
    assert board.last_health_problem is None


def test_health_401_sets_problem():
    board = board_api.Board(host, key="bad")
    assert board.health() is None
    assert board.last_health_problem == "HTTP 401"


def test_ask_returns_text_and_usage():
    board = board_api.Board(host, key="k")
    status, text, elapsed, usage = board.ask("hi", max_tokens=8)
    assert status == 200 and text == "答：hi"
    assert usage["total_tokens"] == 5 and elapsed >= 0


def test_stream_collects_deltas_and_ttft():
    board = board_api.Board(host, key="k")
    status, text, _elapsed, ttft = board.stream(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    assert status == 200 and text == "你好"
    assert ttft is not None


def test_4xx_returns_status_not_exception():
    board = board_api.Board(host, key="k")
    status, body, _elapsed, usage = board.call("/v1/toomuch", {"max_tokens": 3000})
    assert status == 400 and "too large" in body and usage is None


def test_reset_accepts_200():
    board = board_api.Board(host, key="k")
    assert board.reset() is True


def test_module_request_without_key_sends_nothing():
    status, body, _elapsed, err = board_api.request(
        "http://" + host, "k", "/v1/toomuch", {}, timeout=5)
    assert status == 400 and err is None and body


def test_connection_error_maps_to_exc(monkeypatch):
    """传输层异常统一映射为 "EXC"，不把异常抛给各测试脚本。"""
    board = board_api.Board(host, key="k", timeout=1)

    def boom(*_args, **_kwargs):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(board._client, "request", boom)
    status, body, _elapsed, _usage = board.call("/v1/chat/completions", {})
    assert status == "EXC" and "connection reset" in body
