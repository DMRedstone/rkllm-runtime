"""MCP 协议一致性脚本（N6 / 03 §12 第 5 条）。

覆盖四个协议版本 × 六类断言：
  1. initialize 形状：protocolVersion / serverInfo / capabilities；
     2026-07-28 额外要求 resultType；
  2. tools/list：六个工具、每个有 inputSchema、带统一前缀；
  3. tools/call：正常调用返回 content 数组；
  4. 未知方法：-32601；
  5. 参数错误（params 不是对象）：-32602；
  6. 8 MiB 边界：超长帧被拒（PARSE_ERROR），进程仍存活。

另有独立的取消断言：同步调用进行中发 notifications/cancelled，要求 3 s 内
返回 **isError(cancelled)**，且板端真的收到 /reset（abort 收尾）——旧版只断言
"5 s 内返回"，延时配得太短，取消不生效也会超时前自然完成，等于空转。
并校验 `server/discover` 全版本可用、2026 `resultType=complete`、
`tools/list` 的 `ttlMs`/`cacheScope`。

P2.5 修复（2026-11-23）：
- `Connector.recv` 改为后台读线程 + 队列超时——旧实现直接阻塞在
  `readline()`，deadline 循环永不生效；
- `--verbose` 真正逐条打印 PASS/FAIL 与细节（默认只在失败时输出）；
- 取消断言按上面的语义/板端副作用双重校验，不再空转；
- 8 MiB 边界断言收紧为 PARSE_ERROR(-32700)。

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
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONNECTOR = ROOT / "本地模型连接器"        # 客户端独立源码包（非服务端 src/）

VERSIONS = ("2026-07-28", "2025-11-25", "2025-06-18", "2024-11-05")
TOOLS = ("ask", "chat", "extract", "batch", "status", "job")
MAX_BYTES = 8 * 1024 * 1024

_results: list[tuple[str, bool, str]] = []
_verbose = False


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    if _verbose:
        print("  %s  %-46s %s" % ("PASS" if ok else "FAIL", name, detail[:140]))
    elif not ok:
        print("  FAIL  %-46s %s" % (name, detail))


# ---------------------------------------------------------------------------
# 假后端：/v1/chat/completions 支持流式与非流式，可配置延迟
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    delay = 0.0
    # 取消断言需要的证据：/reset 次数（连接器取消时先断流、再 POST /reset）。
    reset_calls = 0
    chat_calls = 0
    last_chat_body: dict = {}

    def log_message(self, *args):        # 静音
        pass

    def handle(self):
        # 连接器取消时会直接断流，socketserver 默认会把 ConnectionResetError
        # 当异常打到 stderr；这是预期路径，不是失败。
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, OSError):
            self.close_connection = True

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.rstrip("/").endswith("/reset"):
            _Handler.reset_calls += 1
            raw = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.headers.get("Authorization") not in (None, "Bearer test-key"):
            self.send_response(401)
            self.end_headers()
            return
        _Handler.chat_calls += 1
        if isinstance(body, dict):
            _Handler.last_chat_body = body
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
            # 板端只在请求带 stream_options.include_usage 时发 usage-only 帧
            if isinstance(body.get("stream_options"), dict) and \
                    body["stream_options"].get("include_usage"):
                usage = {"id": "chatcmpl-test", "object": "chat.completion.chunk",
                         "created": 0, "model": "rkllm", "choices": [],
                         "usage": {"prompt_tokens": 5, "completion_tokens": 3,
                                   "total_tokens": 8}}
                self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
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
    _Handler.reset_calls = 0
    _Handler.chat_calls = 0
    _Handler.last_chat_body = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


# ---------------------------------------------------------------------------
# 连接器子进程
# ---------------------------------------------------------------------------
class Connector:
    def __init__(self, port: int, extra_env: dict | None = None):
        env = dict(os.environ)
        env.update({"PYTHONPATH": str(CONNECTOR),
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
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        self._id = 0
        # P2.5：readline() 在管道上会一直阻塞，旧实现的 deadline 循环根本
        # 不生效（recv 超时形同虚设）。改由后台线程持续读 stdout 进队列，
        # recv 对队列做带超时的 get，超时真正可返回。
        self._lines: queue.Queue = queue.Queue()

        def _pump() -> None:
            try:
                for line in self.proc.stdout:
                    self._lines.put(line)
            except (OSError, ValueError):
                pass
            finally:
                self._lines.put(None)
        self._reader = threading.Thread(target=_pump, daemon=True,
                                        name="conformance-stdout")
        self._reader.start()

    def send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def next_id(self) -> int:
        self._id += 1
        return self._id

    def recv(self, want_id=None, timeout: float = 30.0) -> dict | None:
        """读一条响应；超时真正生效（后台读线程 + 队列超时）。"""
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                return None
            if line is None:                 # 子进程退出（EOF）
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
        try:
            self.proc.stdout.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 断言
# ---------------------------------------------------------------------------
def check_version(version: str, port: int) -> None:
    conn = Connector(port)
    try:
        meta = {"protocolVersion": version}
        if version == "2026-07-28":
            # 2026 是无状态协议：没有 initialize；首帧用 _meta 声明版本。
            discover = (conn.call("server/discover", {"_meta": meta}) or {}).get("result") or {}
            check("%s server/discover" % version,
                  (discover.get("serverInfo") or {}).get("name") == "rkllm-mcp",
                  str(discover)[:120])
            check("%s discover.supportedVersions" % version,
                  "2026-07-28" in (discover.get("supportedVersions") or []),
                  str(discover.get("supportedVersions")))
            check("2026-07-28 resultType=complete",
                  discover.get("resultType") == "complete", str(discover)[:100])
            # initialize 必须被拒绝，显性化版本误用（A-7）。
            init = conn.call("initialize", {"protocolVersion": version})
            check("2026-07-28 initialize rejected",
                  (init or {}).get("error", {}).get("code") == -32601,
                  str(init)[:100])
            listed = (conn.call("tools/list", {"_meta": meta}) or {}).get("result") or {}
        else:
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
            discover = (conn.call("server/discover", {}) or {}).get("result") or {}
            check("%s server/discover" % version,
                  (discover.get("serverInfo") or {}).get("name") == "rkllm-mcp",
                  str(discover)[:120])
            check("%s discover.supportedVersions" % version,
                  "2026-07-28" in (discover.get("supportedVersions") or []),
                  str(discover.get("supportedVersions")))
            conn.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            listed = (conn.call("tools/list", {}) or {}).get("result") or {}

        names = [t.get("name") for t in listed.get("tools") or []]
        expected = ["local_llm_" + t for t in TOOLS]
        check("%s tools/list 6 个" % version,
              sorted(names) == sorted(expected), str(names))
        check("%s 每个工具有 inputSchema" % version,
              all(isinstance(t.get("inputSchema"), dict)
                  for t in listed.get("tools") or []))
        if version == "2026-07-28":
            check("2026-07-28 tools/list ttlMs",
                  isinstance(listed.get("ttlMs"), int) and listed["ttlMs"] > 0,
                  str(listed.get("ttlMs")))
            check("2026-07-28 tools/list cacheScope",
                  listed.get("cacheScope") in ("global", "session"),
                  str(listed.get("cacheScope")))

        called = conn.call("tools/call",
                           {"name": "local_llm_ask",
                            "arguments": {"prompt": "说一句话",
                                          "max_tokens": 8},
                            **({"_meta": meta} if version == "2026-07-28" else {})},
                           timeout=60)
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
        check("8 MiB 边界拒绝（PARSE_ERROR 且不阻塞）", code == -32700,
              str(reply)[:120])
        alive = conn.call("tools/list", {}, timeout=30)
        check("超长帧后进程仍存活", bool(alive and alive.get("result")))
    finally:
        conn.close()


def _is_cancelled_reply(reply: dict | None) -> bool:
    """取消的回复必须是明确 isError/cancelled，而不是"恰好提前结束的成功"。"""
    result = (reply or {}).get("result")
    if not isinstance(result, dict):
        return False
    if result.get("isError") and (result.get("_meta") or {}).get("cancelled"):
        return True
    return "取消" in json.dumps(result, ensure_ascii=False)


def check_cancel(delay: float) -> None:
    """取消断言（P2.5：旧实现是空转——延时太短，取消与否都会在 5 s 内返回）。

    现在：假后端每 token delay 秒（6 个 token），显式 max_tokens=8 强制同步
    路径；0.4 s 后发 notifications/cancelled。若取消未生效，请求要 ~6×delay 秒
    才有结果；断言"3 s 内返回且为 isError(cancelled)"，并核对板端真的收到
    /reset（连接器 cancel 收尾），两者缺一都判失败。
    """
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
                              "arguments": {"prompt": "长任务",
                                            "max_tokens": 8}}})
        time.sleep(0.4)
        resets_before = _Handler.reset_calls
        conn.send({"jsonrpc": "2.0", "method": "notifications/cancelled",
                   "params": {"requestId": rid, "reason": "user"}})
        t0 = time.monotonic()
        reply = conn.recv(rid, timeout=20)
        elapsed = time.monotonic() - t0
        check("取消后 3 s 内返回 isError(cancelled)",
              reply is not None and elapsed < 3.0
              and _is_cancelled_reply(reply),
              "elapsed=%.2fs reply=%s" % (elapsed, str(reply)[:140]))
        after = conn.call("tools/list", {}, timeout=20)
        check("取消后连接仍可用且板端已中止",
              bool(after and after.get("result"))
              and _Handler.reset_calls > resets_before,
              "reset_calls=%d" % _Handler.reset_calls)
    finally:
        conn.close()
        server.shutdown()


def main(argv: list[str] | None = None) -> int:
    global _verbose
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="逐条打印 PASS/FAIL 与细节（默认只在失败时输出）")
    args = parser.parse_args(argv)
    _verbose = bool(args.verbose)

    server, port = start_backend(0.0)
    try:
        for version in VERSIONS:
            if _verbose:
                print("[%s] 协议一致性" % version)
            check_version(version, port)
        if _verbose:
            print("[8 MiB 边界]")
        check_oversize(port)
    finally:
        server.shutdown()

    if _verbose:
        print("[取消/中止]")
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
