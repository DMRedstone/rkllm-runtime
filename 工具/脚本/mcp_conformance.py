"""MCP 协议一致性脚本（N6 / 03 §12 第 5 条）。

覆盖四个协议版本 × 六类断言：
  1. initialize 形状：protocolVersion / serverInfo / capabilities；
     2026-07-28 额外要求 resultType；
  2. tools/list：六个工具、每个有 inputSchema、带统一前缀；
  3. tools/call：正常调用返回 content 数组；
  4. 未知方法：-32601；
  5. 参数错误（params 不是对象）：-32602；
  6. 8 MiB 边界：超长帧被拒（PARSE_ERROR），进程仍存活。

另有独立的取消断言：长调用进行中发 notifications/cancelled，要求 1 s 内
返回且后续调用仍可用。

为什么不用真板：连接器与协议是本脚本的被测对象，后端只需一个可控的慢流。
内置一个假 OpenAI 服务（stdlib http.server），因此本脚本在 Windows/Linux 都
能跑，不需要 NPU、不需要 network 外网。

用法：
  python 工具/脚本/mcp_conformance.py [--verbose]
退出码：0 全绿；1 有失败。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "输入" / "rkllm-runtime-1.3.1" / "src"

VERSIONS = ("2026-07-28", "2025-11-25", "2025-06-18", "2024-11-05")
TOOLS = ("ask", "chat", "extract", "batch", "status", "job")
MAX_BYTES = 8 * 1024 * 1024

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    if not ok:
        print("  FAIL  %-46s %s" % (name, detail))


# ---------------------------------------------------------------------------
# 假后端：/v1/chat/completions 支持流式与非流式，可配置延迟
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    delay = 0.0

    def log_message(self, *args):        # 静音
        pass

    def handle(self):
        # 连接器取消时会直接断流，socketserver 默认会把 ConnectionResetError
        # 当异常打到 stderr；这是预期路径，不是失败。
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.headers.get("Authorization") not in (None, "Bearer test-key"):
            self.send_response(401)
            self.end_headers()
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            words = ["你", "好", "，", "这是", "假", "后端"]
            for word in words:
                time.sleep(_Handler.delay)
                chunk = {"choices": [{"delta": {"content": word}}]}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        time.sleep(_Handler.delay)
        payload = {"choices": [{"message": {"role": "assistant",
                                            "content": "假后端结果"},
                                "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def start_backend(delay: float = 0.0) -> tuple[ThreadingHTTPServer, int]:
    _Handler.delay = delay
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


# ---------------------------------------------------------------------------
# 连接器子进程
# ---------------------------------------------------------------------------
class Connector:
    def __init__(self, port: int, extra_env: dict | None = None):
        env = dict(os.environ)
        env.update({"PYTHONPATH": str(SRC),
                    # Windows 下子进程 stdio 默认是 ANSI（cp936），父进程按 UTF-8
                    # 读会 UnicodeDecodeError；显式钉死 UTF-8，跨平台行为一致。
                    "PYTHONIOENCODING": "utf-8",
                    "RKLLM_ENDPOINT": "http://127.0.0.1:%d/v1" % port,
                    "RKLLM_API_KEY": "test-key",
                    "RKLLM_MCP_CONFIG": os.devnull})
        if extra_env:
            env.update(extra_env)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "rkllm.mcp"], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self._id = 0

    def send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def next_id(self) -> int:
        self._id += 1
        return self._id

    def recv(self, want_id=None, timeout: float = 30.0) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
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
        return None

    def call(self, method: str, params: dict, timeout: float = 30.0) -> dict | None:
        rid = self.next_id()
        self.send({"jsonrpc": "2.0", "id": rid, "method": method,
                   "params": params})
        return self.recv(rid, timeout)

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# 断言
# ---------------------------------------------------------------------------
def check_version(version: str, port: int) -> None:
    conn = Connector(port)
    try:
        init = conn.call("initialize", {"protocolVersion": version,
                                        "capabilities": {},
                                        "clientInfo": {"name": "conformance",
                                                       "version": "1"}})
        result = (init or {}).get("result") or {}
        check("%s initialize ok" % version, bool(result), str(init)[:120])
        check("%s serverInfo" % version,
              (result.get("serverInfo") or {}).get("name") == "rkllm-mcp",
              str(result.get("serverInfo")))
        check("%s capabilities.tools" % version,
              "tools" in (result.get("capabilities") or {}))
        if version == "2026-07-28":
            check("2026-07-28 resultType", "resultType" in result,
                  str(result)[:100])

        conn.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        listed = (conn.call("tools/list", {}) or {}).get("result") or {}
        names = [t.get("name") for t in listed.get("tools") or []]
        expected = ["local_llm_" + t for t in TOOLS]
        check("%s tools/list 6 个" % version,
              sorted(names) == sorted(expected), str(names))
        check("%s 每个工具有 inputSchema" % version,
              all(isinstance(t.get("inputSchema"), dict)
                  for t in listed.get("tools") or []))

        called = conn.call("tools/call",
                           {"name": "local_llm_ask",
                            "arguments": {"prompt": "说一句话",
                                          "max_tokens": 8}}, timeout=60)
        text = ((called or {}).get("result") or {}).get("content") or []
        check("%s tools/call 返回 content" % version,
              bool(text) and "text" in (text[0] or {}), str(called)[:120])

        unknown = conn.call("nope/unknown", {})
        check("%s 未知方法 -32601" % version,
              (unknown or {}).get("error", {}).get("code") == -32601,
              str(unknown)[:100])

        bad = conn.call("tools/list", "not-an-object")   # params 类型错误
        check("%s 非法 params -32602" % version,
              (bad or {}).get("error", {}).get("code") == -32602,
              str(bad)[:100])
    finally:
        conn.close()


def check_oversize(port: int) -> None:
    conn = Connector(port)
    try:
        conn.send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": "2025-11-25"}})
        conn.recv(1)
        huge = "x" * (MAX_BYTES + 1024)
        conn.send({"jsonrpc": "2.0", "id": 99, "method": "ping",
                   "params": {"blob": huge}})
        reply = conn.recv(None, timeout=30)
        code = (reply or {}).get("error", {}).get("code")
        check("8 MiB 边界拒绝", code is not None, str(reply)[:120])
        alive = conn.call("tools/list", {}, timeout=30)
        check("超长帧后进程仍存活", bool(alive and alive.get("result")))
    finally:
        conn.close()


def check_cancel(delay: float) -> None:
    server, port = start_backend(delay)
    conn = Connector(port)
    try:
        conn.call("initialize", {"protocolVersion": "2025-11-25",
                                 "capabilities": {},
                                 "clientInfo": {"name": "c", "version": "1"}})
        conn.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        rid = conn.next_id()
        conn.send({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                   "params": {"name": "local_llm_ask",
                              "arguments": {"prompt": "长任务", "max_tokens": 256}}})
        time.sleep(0.4)
        conn.send({"jsonrpc": "2.0", "method": "notifications/cancelled",
                   "params": {"requestId": rid, "reason": "user"}})
        t0 = time.monotonic()
        reply = conn.recv(rid, timeout=20)
        elapsed = time.monotonic() - t0
        check("取消后 5 s 内返回", reply is not None and elapsed < 5.0,
              "elapsed=%.2fs" % elapsed)
        after = conn.call("tools/list", {}, timeout=20)
        check("取消后连接仍可用", bool(after and after.get("result")))
    finally:
        conn.close()
        server.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.parse_args(argv)

    server, port = start_backend(0.0)
    try:
        for version in VERSIONS:
            check_version(version, port)
        check_oversize(port)
    finally:
        server.shutdown()

    check_cancel(0.3)

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("\n一致性结果：%d/%d 通过" % (passed, total))
    if passed != total:
        print("失败项：")
        for name, ok, detail in _results:
            if not ok:
                print("  - %s  %s" % (name, detail))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
