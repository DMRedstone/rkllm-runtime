"""Backend：MCP 工具到板端模型的两条通路（MCP 设计 §3.1）。

- HttpBackend：客户端侧（唯一必做），urllib 调板端 /v1/chat/completions；
- LocalBackend：板端侧（二期），进程内直接调 worker（本文件只留协议）。

工具层只依赖 Backend 协议，不碰 HTTP 或 ctypes → 可单测。
"""

from __future__ import annotations

import http.client
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

__all__ = ["HttpBackend", "BackendError"]

# 取消轮询粒度：长 prefill 期间板端不产出 chunk，主循环的检查点进不来，
# 由看门狗线程按这个间隔检查（P2：取消原先只在收到 chunk 后生效）。
CANCEL_POLL_S = 0.2


class BackendError(Exception):
    def __init__(self, message: str, *, retry_after_s: int | None = None,
                 kind: str = "backend"):
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.kind = kind        # busy / loading / error


class HttpBackend:
    """调板端 OpenAI 子集。流式收齐为整段文本（MCP 工具不流式回传）。"""

    def __init__(self, endpoint: str, api_key: str = "",
                 timeout_s: int = 120):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        # keep-alive（审查项：原实现每次 urlopen 都新建 TCP，内网穿透下还要
        # 重做 TLS 握手，一轮对话要付多次握手成本）。按线程复用连接——
        # http.client 的连接不可并发共享，而调用方是多线程的工具池/job 线程。
        parts = urllib.parse.urlsplit(self.endpoint)
        self._scheme = parts.scheme or "http"
        self._host = parts.hostname or "127.0.0.1"
        self._port = parts.port
        self._prefix = parts.path.rstrip("/")
        self._local = threading.local()

    # -- 连接管理 -----------------------------------------------------------
    def _connection(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            cls = (http.client.HTTPSConnection if self._scheme == "https"
                   else http.client.HTTPConnection)
            conn = cls(self._host, self._port, timeout=self.timeout_s)
            self._local.conn = conn
        return conn

    def _drop_connection(self) -> None:
        """丢弃本线程的连接（复用连接可能已被对端关闭，或响应未读完）。"""
        conn = getattr(self._local, "conn", None)
        self._local.conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _request(self, path: str, body: bytes):
        """POST 到板端并返回 `HTTPResponse`（复用 keep-alive 连接）。

        首次失败可能是复用的空闲连接已被对端关闭（keep-alive idle 超时），
        重建连接重试一次；仍失败才报不可达。
        """
        headers = {"Content-Type": "application/json",
                   "Accept": "text/event-stream",
                   **({"Authorization": "Bearer " + self.api_key}
                      if self.api_key else {})}
        last_exc: Exception | None = None
        for attempt in (0, 1):
            conn = self._connection()
            try:
                conn.request("POST", self._prefix + path, body=body,
                             headers=headers)
                return conn.getresponse()
            except (http.client.HTTPException, OSError) as exc:
                last_exc = exc
                self._drop_connection()
                if attempt == 0:
                    continue
        raise BackendError("无法连接板端：%s" % last_exc)

    def chat(self, messages: list[dict], *, max_tokens: int = 512,
             temperature: float = 0.3,
             cancel_check: Callable[[], bool] | None = None) -> dict:
        """返回 {"text": str, "usage": dict|None, "finish_reason": str}。

        cancel_check 由调用方周期性置位（job 取消 / 客户端断开）；HTTP 层
        无法中断已在途的请求，靠读循环在 chunk 边界检查。
        """
        body = json.dumps({
            "model": "rkllm",
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        resp = self._request("/chat/completions", body)
        if resp.status >= 400:
            detail = resp.read(2000).decode("utf-8", "replace")
            self._drop_connection()   # 错误响应后连接状态不确定，不复用
            if resp.status == 503:
                retry = resp.getheader("Retry-After")
                raise BackendError("模型忙：" + detail[:200],
                                   retry_after_s=int(retry) if retry else None,
                                   kind="busy")
            raise BackendError("板端 HTTP %d：%s"
                               % (resp.status, detail[:200]))

        text_parts: list[str] = []
        usage = None
        finish_reason = "stop"
        stop_watch = threading.Event()
        cancelled = threading.Event()

        def watchdog() -> None:
            """无输出期的取消检查（P2）。

            原实现只在收到 chunk 后检查 `cancel_check()`——长 prefill 期间
            板端一个 chunk 都没有，取消形同虚设，任务会一直算完。这里在
            后台线程定期检查，命中即断流+`/reset`，主循环随后退出。
            """
            while not stop_watch.wait(CANCEL_POLL_S):
                if cancel_check is not None and cancel_check():
                    cancelled.set()
                    self._abort_remote(resp)
                    return

        watcher = None
        if cancel_check is not None:
            watcher = threading.Thread(target=watchdog, daemon=True,
                                       name="backend-cancel")
            watcher.start()
        finished = False
        try:
            import time as _time
            _start = _time.monotonic()
            for raw in resp:
                # P2：urlopen(timeout=) 只是单次 socket 超时；慢速滴流可无限挂住
                # ——这里加总 deadline 兜底（finally 负责 close）。
                if _time.monotonic() - _start > self.timeout_s:
                    raise BackendError("板端响应超过总超时 %ss" % self.timeout_s)
                if cancelled.is_set():
                    raise BackendError("任务已取消", kind="cancelled")
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if cancel_check and cancel_check():
                    cancelled.set()
                    self._abort_remote(resp)
                    raise BackendError("任务已取消", kind="cancelled")
                err = chunk.get("error")
                if err:
                    # P2：板端中途错误的帧没有 choices，旧逻辑会静默吞成假成功
                    msg = err.get("message") if isinstance(err, dict) else err
                    raise BackendError("板端流中错误：%s" % str(msg)[:200])
                choices = chunk.get("choices") or [{}]
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text_parts.append(delta["content"])
                if chunk.get("usage"):
                    usage = chunk["usage"]
            finished = True
        except (OSError, ValueError) as exc:
            # 看门狗 close() 会让阻塞中的读抛异常——这是取消而不是板端故障
            if cancelled.is_set():
                raise BackendError("任务已取消", kind="cancelled") from exc
            raise BackendError("读板端流失败：%s" % exc) from exc
        finally:
            stop_watch.set()
            if finished:
                # 响应已读到 EOF：保留连接供下次复用（keep-alive）
                pass
            else:
                self._drop_connection()   # 取消/异常：连接不可复用
        return {"text": "".join(text_parts), "usage": usage,
                "finish_reason": finish_reason}

    def _abort_remote(self, resp) -> None:
        """取消收尾：先断流、再 `/reset` 兜底（顺序反了会把请求留在服务端算完）。"""
        try:
            resp.close()
        except OSError:
            pass
        req_abort = urllib.request.Request(
            self.endpoint + "/reset", data=b"{}", method="POST",
            headers={"Content-Type": "application/json",
                     **({"Authorization": "Bearer " + self.api_key}
                        if self.api_key else {})})
        try:
            urllib.request.urlopen(req_abort, timeout=5)
        except OSError:
            pass

    def status(self) -> dict:
        """板端健康摘要（local_llm_status 用）。

        鉴权：/health 与其他端点同受 Bearer 保护（实测 401 被旧代码的
        except 吞成 unreachable——wheel 冒烟抓出的缺陷）。
        """
        headers = {"Authorization": "Bearer " + self.api_key} \
            if self.api_key else {}
        # urljoin 正确处理 endpoint 带/不带 /v1、带/不带末尾斜杠的边界
        url = urllib.parse.urljoin(self.endpoint + "/", "../health")
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers),
                    timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (OSError, ValueError):
            return {"state": "unreachable"}
