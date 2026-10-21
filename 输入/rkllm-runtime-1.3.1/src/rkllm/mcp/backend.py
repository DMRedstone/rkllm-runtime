"""Backend：MCP 工具到板端模型的两条通路（MCP 设计 §3.1）。

- HttpBackend：客户端侧（唯一必做），urllib 调板端 /v1/chat/completions；
- LocalBackend：板端侧（二期），进程内直接调 worker（本文件只留协议）。

工具层只依赖 Backend 协议，不碰 HTTP 或 ctypes → 可单测。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator

__all__ = ["HttpBackend", "BackendError"]


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
        req = urllib.request.Request(
            self.endpoint + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     **({"Authorization": "Bearer " + self.api_key}
                        if self.api_key else {})})
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", "replace")
            if exc.code == 503:
                retry = exc.headers.get("Retry-After")
                raise BackendError("模型忙：" + detail[:200],
                                   retry_after_s=int(retry) if retry else None,
                                   kind="busy")
            raise BackendError("板端 HTTP %d：%s" % (exc.code, detail[:200]))
        except urllib.error.URLError as exc:
            raise BackendError("无法连接板端：%s" % exc.reason)

        text_parts: list[str] = []
        usage = None
        finish_reason = "stop"
        try:
            for raw in resp:
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
                    req_abort = urllib.request.Request(
                        self.endpoint + "/reset", data=b"{}", method="POST",
                        headers={"Content-Type": "application/json",
                                 **({"Authorization": "Bearer " + self.api_key}
                                    if self.api_key else {})})
                    try:
                        urllib.request.urlopen(req_abort, timeout=5)
                    except OSError:
                        pass
                    raise BackendError("任务已取消", kind="cancelled")
                choices = chunk.get("choices") or [{}]
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text_parts.append(delta["content"])
                if chunk.get("usage"):
                    usage = chunk["usage"]
        finally:
            resp.close()
        return {"text": "".join(text_parts), "usage": usage,
                "finish_reason": finish_reason}

    def status(self) -> dict:
        """板端健康摘要（local_llm_status 用）。"""
        try:
            with urllib.request.urlopen(self.endpoint.rsplit("/", 1)[0]
                                        + "/health", timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (OSError, ValueError):
            return {"state": "unreachable"}
