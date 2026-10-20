# -*- coding: utf-8 -*-
"""服务端源码级回归检查（不需要模型、不需要 NPU、不需要板子）。

针对 rkllm.py 里那些靠 selftest 覆盖不到、但都曾真实出过问题的路径：
并发锁的异常安全、客户端断开的看门狗、慢速 body、请求体排空、以及
stop 序列对 usage 的影响。

用法（在仓库根目录，或用 --source 指定 rkllm.py）：

    python 工具/大模型测试工具链/verify_server.py
    python 工具/大模型测试工具链/verify_server.py --source /path/to/rkllm.py

退出码：0 全部通过 / 1 有失败项。只读，不改被检查的文件。
"""
from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import os
import socket
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SOURCE = os.path.join(ROOT, "输入", "rkllm-runtime-1.3.1", "cli", "rkllm.py")

RESULTS = []


def check(name, ok, detail=""):
    ok = bool(ok)
    RESULTS.append(ok)
    sys.stdout.write("[%s] %-52s %s\n" % ("PASS" if ok else "FAIL", name, detail))
    sys.stdout.flush()


def load_module(path):
    spec = importlib.util.spec_from_file_location("rkmod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 1. 异常安全
class _BrokenStderr(object):
    """写就抛：systemd journal 关闭、终端断开时的真实行为。"""

    def write(self, _s):
        raise BrokenPipeError("stderr gone")

    def flush(self):
        raise BrokenPipeError("stderr gone")


class _StubThread(object):
    """start() 同步跑完 target，但对外永远自称 alive。"""

    def __init__(self, target=None, name=None, daemon=None, **kwargs):
        self._target = target
        self.daemon = bool(daemon)

    def start(self):
        self._target()

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return True


def test_guard_release(m):
    """推理线程卡死 + stderr 不可写时，_stream_guard 必须仍然释放。"""

    class _FailLib(object):
        def rkllm_run(self, *args):
            return 3                                   # 非 0 -> _Failure

    engine = m.RealEngine.__new__(m.RealEngine)
    engine._destroyed = False
    engine._stream_guard = threading.Lock()
    engine._sink_lock = threading.Lock()
    engine._sink = None
    engine._run_keep = []
    engine._infer = m._InferParam()
    engine._handle = None
    engine.lib = _FailLib()
    engine.system_prompt = ""
    engine._last_perf = None

    saved_thread = m.threading.Thread
    saved_stderr = sys.stderr
    saved_join = m.STREAM_JOIN_TIMEOUT_S
    raised = None
    try:
        m.threading.Thread = _StubThread
        m.STREAM_JOIN_TIMEOUT_S = 0.2
        sys.stderr = _BrokenStderr()
        try:
            list(engine.stream("hi"))
        except Exception as exc:                       # noqa: BLE001
            raised = exc
    finally:
        m.threading.Thread = saved_thread
        sys.stderr = saved_stderr
        m.STREAM_JOIN_TIMEOUT_S = saved_join

    check("stderr 写失败时 stream() 仍报出原始失败",
          isinstance(raised, Exception) and "returned 3" in str(raised),
          repr(raised)[:60])
    if engine._stream_guard.acquire(blocking=False):
        engine._stream_guard.release()
        check("stderr 写失败后 _stream_guard 已释放", True, "可再次 acquire")
    else:
        check("stderr 写失败后 _stream_guard 已释放", False,
              "锁泄漏，引擎将永久拒绝请求")
    check("推理线程未退出时标记引擎不可用", engine._destroyed is True)


# ---------------------------------------------------------------- HTTP 部分
class _StopEngine(object):
    """固定输出 helloSTOPworld，用来检查 stop 裁剪与 usage。"""

    def __init__(self):
        self.model_name = "stop-test"
        self.context_len = 4096
        self.keep_history = 1
        self.system_prompt = ""
        self.answer = "helloSTOPworld"

    def stream(self, prompt, role="user", enable_thinking=False,
               sampling=None, max_new_tokens=None):
        for ch in self.answer:
            yield ch

    def last_perf(self):
        return None

    def clear_history(self):
        return 0

    def is_running(self):
        return False

    def abort(self):
        return 0

    def destroy(self):
        pass


def start_server(m, engine):
    options = argparse.Namespace(max_queue=4, queue_timeout=5.0,
                                 api_key=None, quiet=True)
    server = m.Server(("127.0.0.1", 0), m.Handler, engine, options)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def post(port, path, payload, conn=None, timeout=30):
    own = conn is None
    conn = conn or http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", path, body=json.dumps(payload),
                 headers={"Content-Type": "application/json"})
    reply = conn.getresponse()
    body = reply.read()
    if own:
        conn.close()
    return reply.status, body


def test_stop_usage(m, port):
    status, body = post(port, "/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "hi"}],
                         "stop": ["STOP"]})
    payload = json.loads(body) if status == 200 else {}
    answer = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = payload.get("usage", {})
    expect = m._estimate_tokens("hello")
    check("非流式 stop 裁剪正文", status == 200 and answer == "hello", "%r" % answer)
    check("completion_tokens 按裁剪后文本计数",
          usage.get("completion_tokens") == expect,
          "got %s, expect %s (raw would be %s)"
          % (usage.get("completion_tokens"), expect,
             m._estimate_tokens("helloSTOPworld")))

    status, body = post(port, "/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "hi"}],
                         "stop": ["STOP"], "stream": True,
                         "stream_options": {"include_usage": True}})
    text = body.decode("utf-8", "replace")
    usage_line = None
    for line in text.splitlines():
        if line.startswith("data: ") and '"usage"' in line:
            usage_line = json.loads(line[6:])
    check("流式 usage 按裁剪后文本计数",
          usage_line is not None
          and usage_line["usage"]["completion_tokens"] == expect,
          "%s" % (usage_line or {}).get("usage"))


def test_reset_body(m, port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    status, _ = post(port, "/v1/reset", {"why": "x" * 500}, conn=conn)
    check("POST /v1/reset 带 body 正常", status == 200, "status %s" % status)
    # 同一 keep-alive 连接上再发一个请求：body 没读完的话，它的字节会被当成
    # 下一个请求的请求行。
    conn.request("GET", "/health")
    reply = conn.getresponse()
    reply.read()
    check("reset 后同一连接可复用（body 已排空）", reply.status == 200,
          "status %s" % reply.status)
    conn.close()


def test_reset_slow_body(m, port):
    """声明 100 字节只发 1 字节：必须按 READ_TIMEOUT_S 超时，而不是挂住线程。"""
    saved = m.READ_TIMEOUT_S
    m.READ_TIMEOUT_S = 1
    data = b""
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        sock.sendall(b"POST /v1/reset HTTP/1.1\r\nHost: x\r\n"
                     b"Content-Length: 100\r\n\r\n{")
        sock.settimeout(10)
        data = sock.recv(4096)
        sock.close()
    finally:
        m.READ_TIMEOUT_S = saved
    text = data.decode("latin-1", "replace") if data else ""
    status_line = text.split("\r\n")[0] if text else "无响应"
    check("reset 慢速 body 被超时拒绝", " 408 " in status_line, status_line)


def test_oversize_closes(m, port):
    """413 之后必须关闭连接。

    body 还留在 socket 里，而 HTTP/1.1 默认 keep-alive：不关的话，下一个请求会
    从这些未消费的字节开始解析。这个分支曾经只在 /v1/reset 里关了连接，
    /v1/chat/completions 漏了。
    """
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                 b"Content-Type: application/json\r\n"
                 b"Content-Length: 10485760\r\n\r\n")
    sock.settimeout(10)
    head = sock.recv(4096).decode("latin-1", "replace")
    status_line = head.split("\r\n")[0] if head else "无响应"
    closed = False
    try:
        sock.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
        closed = (sock.recv(256) == b"")
    except OSError:
        closed = True
    sock.close()
    check("413 后关闭连接（不污染 keep-alive）",
          " 413 " in status_line and closed, "%s | closed=%s" % (status_line, closed))


def test_reset_chunked(m, port):
    """reset 也必须拒绝 chunked，否则 chunk 帧留在 socket 里当成下一个请求。"""
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.sendall(b"POST /v1/reset HTTP/1.1\r\nHost: x\r\n"
                 b"Transfer-Encoding: chunked\r\n\r\n"
                 b"2\r\n{}\r\n0\r\n\r\n")
    sock.settimeout(10)
    head = sock.recv(512).decode("latin-1", "replace")
    sock.close()
    status_line = head.split("\r\n")[0] if head else "无响应"
    check("reset 拒绝 chunked body（411）", " 411 " in status_line, status_line)


def test_watcher_join(m, port):
    """第一次请求结束后，看门狗不得干扰同一连接上的第二次请求。"""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    bodies = []
    for _ in range(2):
        status, body = post(conn, "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "q"}]},
                            conn=conn)
        bodies.append((status, json.loads(body).get("choices", [{}])[0]
                       .get("message", {}).get("content", "")))
    conn.close()
    check("keep-alive 上连续两次非流式请求都完整",
          all(s == 200 and b.startswith("answer") for s, b in bodies),
          "%s" % [(s, b[:18]) for s, b in bodies])


def test_health(m, port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/health")
    reply = conn.getresponse()
    payload = json.loads(reply.read())
    conn.close()
    check("/health 暴露 startup_system_prompt",
          "startup_system_prompt" in payload,
          "%s" % payload.get("startup_system_prompt"))


def test_json_hint(m):
    check("JSON_HINT 默认是中文提示", "只输出一个 JSON 对象" in m.JSON_HINT,
          repr(m.JSON_HINT[:24]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source",
                        default=os.environ.get("RKLLM_SOURCE") or DEFAULT_SOURCE,
                        help="rkllm.py 路径（默认取仓库内的源码模板）")
    args = parser.parse_args()
    if not os.path.isfile(args.source):
        sys.stderr.write("找不到 rkllm.py: %s\n" % args.source)
        return 2

    m = load_module(args.source)

    test_json_hint(m)
    test_guard_release(m)

    engine = m.FakeEngine(model_name="regress", delay=0.05, context_len=4096)
    server, port = start_server(m, engine)
    try:
        test_health(m, port)
        test_watcher_join(m, port)
        test_reset_body(m, port)
        test_reset_slow_body(m, port)
        test_reset_chunked(m, port)
        test_oversize_closes(m, port)
    finally:
        server.shutdown()
        server.server_close()

    engine2 = _StopEngine()
    server2, port2 = start_server(m, engine2)
    try:
        test_stop_usage(m, port2)
    finally:
        server2.shutdown()
        server2.server_close()

    passed = sum(1 for r in RESULTS if r)
    sys.stdout.write("\nverify_server: %d passed, %d failed\n"
                     % (passed, len(RESULTS) - passed))
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
