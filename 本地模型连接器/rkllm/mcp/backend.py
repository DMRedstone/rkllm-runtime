"""Backend：MCP 工具到板端模型的两条通路（MCP 设计 §3.1）。

- HttpBackend：客户端侧（唯一必做），urllib 调板端 /v1/chat/completions；
- LocalBackend：板端侧（二期），进程内直接调 worker（本文件只留协议）。

工具层只依赖 Backend 协议，不碰 HTTP 或 ctypes → 可单测。
"""

from __future__ import annotations

import http.client
import json
import logging
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

__all__ = ["HttpBackend", "BackendError"]

_log = logging.getLogger("rkllm.mcp.backend")

# 取消轮询粒度：长 prefill 期间板端不产出 chunk，主循环的检查点进不来，
# 由看门狗线程按这个间隔检查（P2：取消原先只在收到 chunk 后生效）。
CANCEL_POLL_S = 0.2
# 连接建立阶段的独立超时：读超时（job 可达 4200 s）不能同时当成 TCP 连接超时——
# 板端不可达/隧道黑洞时，同步调用会在建连阶段挂到读超时（2026-11-19 审计 P2）。
CONNECT_TIMEOUT_S = 10.0


class BackendError(Exception):
    """板端调用错误；kind 决定上层如何归一为有语义的 MCP 错误（P2.5）。

    kind 取值（新增的是板端 HTTP/流语义，不再一律 -32603）：
    - busy / loading / rate_limited：可重试（带 retry_after_s）；
    - bad_request：板端 400/422，参数或上下文预算问题，不可重试；
    - auth：板端 401/403，key 错误；
    - not_found：板端 404；
    - stream：SSE 流中的错误帧；
    - upstream：板端 5xx / 其他 HTTP；
    - cancelled：客户端取消；
    - backend / error / config：连接失败或配置错误（沿用旧语义）。
    """

    def __init__(self, message: str, *, retry_after_s: int | None = None,
                 kind: str = "backend", http_status: int | None = None,
                 code: str | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.kind = kind
        self.http_status = http_status
        self.code = code

    def meta(self) -> dict:
        """转成 MCP tool result 的 `_meta`（不含 key、不含 URL 查询串）。"""
        meta: dict = {"kind": self.kind}
        if self.http_status is not None:
            meta["httpStatus"] = int(self.http_status)
        if self.code:
            meta["code"] = self.code
        if isinstance(self.retry_after_s, (int, float)):
            meta["retryAfter"] = self.retry_after_s
        return meta


# HTTP 状态码 → 语义 kind（P2.5：400/401/流错误不再统一 -32603）
_HTTP_KIND = {
    400: "bad_request", 422: "bad_request",
    401: "auth", 403: "auth",
    404: "not_found",
    429: "rate_limited",
    503: "busy",
}
# 这些 kind 是可重试的临时状态：上层回 isError + retryAfter，而不是内部错误
RETRYABLE_KINDS = frozenset({"busy", "loading", "rate_limited"})
# 这些 kind 是"有语义"的板端失败：上层回结构化 isError（不是 -32603）
SEMANTIC_KINDS = frozenset({"bad_request", "auth", "not_found", "stream",
                            "upstream", "http"}) | RETRYABLE_KINDS


def _parse_retry_after(resp) -> int | None:
    """Retry-After 只采信整数秒；HTTP-date/非法值返回 None（不猜）。"""
    try:
        raw = resp.getheader("Retry-After")
    except (AttributeError, TypeError):
        return None
    if not raw:
        return None
    try:
        seconds = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _http_error(status: int, detail: str, resp) -> BackendError:
    """把板端 HTTP 错误归一成带语义的 BackendError（P2.5）。"""
    kind = _HTTP_KIND.get(status, "upstream" if status >= 500 else "http")
    retry = _parse_retry_after(resp) if status in (429, 503) else None
    label = {
        "bad_request": "板端拒绝请求（参数/上下文预算）",
        "auth": "板端鉴权失败（API key 无效或缺失）",
        "not_found": "板端端点不存在",
        "rate_limited": "板端限流",
        "busy": "模型忙",
        "upstream": "板端服务错误",
        "http": "板端 HTTP 错误",
    }.get(kind, "板端 HTTP 错误")
    # 板端 OpenAI 错误帧通常是 {"error":{"message":...}}；取出 message 便于排障
    text = detail.strip()
    try:
        parsed = json.loads(text)
        err = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(err, dict) and err.get("message"):
            text = str(err["message"])
        elif isinstance(err, str):
            text = err
    except (json.JSONDecodeError, ValueError):
        pass
    return BackendError("%s（HTTP %d）：%s" % (label, status, text[:200]),
                        retry_after_s=retry, kind=kind, http_status=status)


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
            # 建连阶段用短超时（CONNECT_TIMEOUT_S），避免读超时（job 可达
            # 4200 s）被当成 TCP 连接超时；建连成功后由 _request 切回 want。
            conn = cls(self._host, self._port,
                       timeout=min(want, CONNECT_TIMEOUT_S))
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
            # P2-D：POST 非幂等——重试只允许发生在"请求尚未送达"阶段。
            # conn.request() 失败 = 连接没建立，可安全重建重试；一旦 request()
            # 成功，getresponse() 阶段再失败也绝不重发，否则同一推理执行两次。
            try:
                if conn.sock is None:
                    conn.connect()          # 建连阶段：短超时（见 _connection）
                conn.sock.settimeout(t)     # 读阶段：本次调用的读超时
                conn.request("POST", self._prefix + path, body=body,
                             headers=headers)
            except (http.client.HTTPException, OSError) as exc:
                last_exc = exc
                self._drop_connection()
                if attempt == 0:
                    continue
                raise BackendError("无法连接板端：%s" % last_exc) from exc
            # 请求已送达：此后的异常不重发（非幂等），但必须归一为 BackendError
            # 并丢弃半开连接，否则会以裸 OSError 逃成 -32603、且坏连接残留
            # 线程本地（2026-11-19 审计 P1）。
            try:
                return conn.getresponse()
            except (http.client.HTTPException, OSError) as exc:
                self._drop_connection()
                raise BackendError("板端连接中断（响应读取失败）：%s"
                                   % exc) from exc
        raise BackendError("无法连接板端：%s" % last_exc)  # pragma: no cover

    def chat(self, messages: list[dict], *, max_tokens: int = 512,
             temperature: float = 0.3,
             cancel_check: Callable[[], bool] | None = None,
             timeout_s: float | None = None,
             include_usage: bool = True) -> dict:
        """返回 {"text", "usage", "finish_reason", "reasoning_chars"}。

        - cancel_check：job 取消 / 客户端断开时返回 True；命中后必须抛
          `BackendError(kind="cancelled")`——绝不能把底层读循环异常（跨线程
          关闭 response 时的 AttributeError 等）裸露成 -32603；
        - timeout_s：本次调用的**总读超时**（job 按估算放宽，见 stdio）；
        - include_usage：P2.5 起按需发送 `stream_options.include_usage`。板端
          只在该开关下于 [DONE] 前发 usage-only 帧；工具层要 `_meta.usage`
          就必须带上它（默认 True；批处理等确实不需要时可关）。
        """
        deadline_s = float(timeout_s or self.timeout_s)
        payload: dict = {
            "model": "rkllm",
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if include_usage:
            payload["stream_options"] = {"include_usage": True}
        body = json.dumps(payload).encode("utf-8")
        resp = self._request("/chat/completions", body, timeout_s=deadline_s)
        if resp.status >= 400:
            detail = resp.read(2000).decode("utf-8", "replace")
            self._drop_connection()   # 错误响应后连接状态不确定，不复用
            raise _http_error(resp.status, detail, resp)

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
                    # P2：板端中途错误的帧没有 choices，旧逻辑会静默吞成假成功。
                    # P2.5：归一为 kind="stream" 的语义错误（带板端 code/type），
                    # 上层回 isError + _meta，而不是 -32603。
                    if isinstance(err, dict):
                        msg = err.get("message") or err.get("type") or str(err)
                        code = err.get("code") or err.get("type")
                    else:
                        msg, code = err, None
                    raise BackendError("板端流中错误：%s" % str(msg)[:200],
                                       kind="stream",
                                       code=str(code) if code else None)
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
        except (OSError, ValueError) as exc:
            # C-H1：abort 兜底的 /reset 失败不得静默——否则取消后板端继续空跑
            # NPU 而无任何迹证。保留原语义（不重试、不向主流程抛出），但留痕。
            _log.warning("abort /reset failed (endpoint=%s, path=/reset): %r",
                         self.endpoint, exc)

    def status(self) -> dict:
        """板端健康摘要（local_llm_status 用）。

        鉴权：/health 与其他端点同受 Bearer 保护（实测 401 被旧代码的
        except 吞成 unreachable——冒烟测试抓出的缺陷）。
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
        except urllib.error.HTTPError as exc:
            # M-1：区分 HTTP 层失败，别把 401/404/503 折叠成 unreachable
            return {"state": "unreachable", "error_kind": "http",
                    "http_status": exc.code, "detail": str(exc.reason)[:200]}
        except urllib.error.URLError as exc:
            return {"state": "unreachable", "error_kind": "connect",
                    "detail": str(exc.reason)[:200]}
        except ValueError as exc:          # 非 JSON / 解析失败
            return {"state": "unreachable", "error_kind": "bad_json",
                    "detail": str(exc)[:200]}
        except OSError as exc:
            return {"state": "unreachable", "error_kind": "os",
                    "detail": str(exc)[:200]}


class LocalBackend:
    """板端进程内后端（backend=local）。

    连接器以客户端源码交付，没有板端 worker 可进程内直调；本类仅用于
    「按配置选择后端」——选 local 时给出一致、可诊断的错误，而不是静默
    退回 http，也不假装成功。
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def _unavailable(self, *args, **kwargs):
        raise BackendError(
            "backend=local 仅板端进程可用；客户端连接器请用 backend=http/auto",
            kind="config")

    chat = _unavailable
    status = _unavailable
