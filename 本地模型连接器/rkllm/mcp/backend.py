"""Backend：MCP 工具到板端模型的两条通路（MCP 设计 §3.1）。

- HttpBackend：客户端侧（唯一必做），urllib 调板端 /v1/chat/completions；
- LocalBackend：板端侧（二期），进程内直接调 worker（本文件只留协议）。

工具层只依赖 Backend 协议，不碰 HTTP 或 ctypes → 可单测。
"""

from __future__ import annotations

import http.client
import json
import socket
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
    def _connection(self, timeout_s: float | None = None):
        want = float(timeout_s or self.timeout_s)
        conn = getattr(self._local, "conn", None)
        if conn is not None and want > float(
                getattr(self._local, "conn_timeout", 0) or 0):
            # 长 job 的读超时比同步调用大：换一条 socket timeout 更大的连接，
            # 否则长 prefill 会在旧连接的 socket 超时上先被掐断。
            self._drop_connection()
            conn = None
        if conn is None:
            cls = (http.client.HTTPSConnection if self._scheme == "https"
                   else http.client.HTTPConnection)
            conn = cls(self._host, self._port, timeout=want)
            self._local.conn = conn
            self._local.conn_timeout = want
        return conn

    def _drop_connection(self) -> None:
        """丢弃本线程的连接（复用连接可能已被对端关闭，或响应未读完）。"""
        conn = getattr(self._local, "conn", None)
        self._local.conn = None
        self._local.conn_timeout = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _request(self, path: str, body: bytes,
                 timeout_s: float | None = None):
        """POST 到板端并返回 `HTTPResponse`（复用 keep-alive 连接）。

        首次失败可能是复用的空闲连接已被对端关闭（keep-alive idle 超时），
        重建连接重试一次；仍失败才报不可达。
        """
        headers = {"Content-Type": "application/json",
                   "Accept": "text/event-stream",
                   **({"Authorization": "Bearer " + self.api_key}
                      if self.api_key else {})}
        t = float(timeout_s or self.timeout_s)
        last_exc: Exception | None = None
        for attempt in (0, 1):
            conn = self._connection(t)
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
             cancel_check: Callable[[], bool] | None = None,
             timeout_s: float | None = None) -> dict:
        """返回 {"text", "usage", "finish_reason", "reasoning_chars"}。

        - cancel_check：job 取消 / 客户端断开时返回 True；命中后必须抛
          `BackendError(kind="cancelled")`——绝不能把底层读循环异常（跨线程
          关闭 response 时的 AttributeError 等）裸露成 -32603；
        - timeout_s：本次调用的**总读超时**（job 按估算放宽，见 stdio）。
        """
        deadline_s = float(timeout_s or self.timeout_s)
        body = json.dumps({
            "model": "rkllm",
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        resp = self._request("/chat/completions", body, timeout_s=deadline_s)
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
        reasoning_chars = 0
        usage = None
        finish_reason = "stop"
        stop_watch = threading.Event()
        cancelled = threading.Event()

        def watchdog() -> None:
            """无输出期的取消检查（P2）。

            板端长 prefill 期间一个 chunk 都没有，主循环检查点进不来；由看门狗
            定期检查。打断方式优先 socket shutdown（不跨线程改 response 状态），
            再退化为静默 close，避免与主读循环的 `for raw in resp` 相互践踏。
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
                if _time.monotonic() - _start > deadline_s:
                    raise BackendError("板端响应超过总超时 %ss" % deadline_s)
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
                # reasoning 模型（Qwen3 / LM Studio 等）把思考放在
                # reasoning_content；只统计长度用于"空正文"判定，不回传给上下文。
                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(rc, str) and rc:
                    reasoning_chars += len(rc)
                if delta.get("content"):
                    text_parts.append(delta["content"])
                if chunk.get("usage"):
                    usage = chunk["usage"]
            finished = True
        except BackendError:
            raise
        except Exception as exc:
            # P1：取消时 watchdog 打断读循环会抛 AttributeError/ValueError
            # （"'NoneType' object has no attribute 'peek'"）——一律映射成 cancelled，
            # 不再变成 -32603 裸内部错误。
            if cancelled.is_set() or (cancel_check is not None and cancel_check()):
                cancelled.set()
                raise BackendError("任务已取消", kind="cancelled") from exc
            if isinstance(exc, (OSError, ValueError)):
                raise BackendError("读板端流失败：%s" % exc) from exc
            raise BackendError("读板端流失败：%s" % (exc or type(exc).__name__)
                               ) from exc
        finally:
            stop_watch.set()
            if finished:
                # 响应已读到 EOF：保留连接供下次复用（keep-alive）
                pass
            else:
                self._drop_connection()   # 取消/异常：连接不可复用
        return {"text": "".join(text_parts), "usage": usage,
                "finish_reason": finish_reason,
                "reasoning_chars": reasoning_chars}

    @staticmethod
    def _close_quietly(obj) -> None:
        try:
            obj.close()
        except Exception:
            pass

    def _interrupt_response(self, resp) -> None:
        """打断阻塞中的 response 读取（取消用）。

        优先 `socket.shutdown(SHUT_RDWR)`：只影响内核 socket，不改 http.client
        的 fp 状态，主读循环会拿到 OSError 而不是 AttributeError；拿不到 socket
        时才退化为 close（此后由 chat 的兜底 except 转成 cancelled）。
        """
        sock = getattr(getattr(getattr(resp, "fp", None), "raw", None),
                       "_sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
                return
            except OSError:
                pass
        self._close_quietly(resp)

    def _abort_remote(self, resp) -> None:
        """取消收尾：先断流、再 `/reset` 兜底（顺序反了会把请求留在服务端算完）。"""
        self._interrupt_response(resp)
        req_abort = urllib.request.Request(
            self.endpoint + "/reset", data=b"{}", method="POST",
            headers={"Content-Type": "application/json",
                     **({"Authorization": "Bearer " + self.api_key}
                        if self.api_key else {})})
        try:
            urllib.request.urlopen(req_abort, timeout=5)
        except (OSError, ValueError):
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
