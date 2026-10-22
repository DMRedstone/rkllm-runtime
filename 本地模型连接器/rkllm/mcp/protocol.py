"""JSON-RPC 分发与 MCP 协议版本矩阵（MCP 设计 §1.2 / §8）。

四版协议的差异收敛在 PROTOCOL_MATRIX 一张表里，业务代码不做 if：
- 2026-07-28：无状态；请求 _meta 带 protocolVersion；实现 server/discover；
  结果带 resultType；
- 2025-11-25 / 2025-06-18 / 2024-11-05：initialize 握手 + 会话语义；
- 首包判定：带 _meta.protocolVersion → 2026 无状态；是 initialize → 旧版握手；
  其余 → 默认回退 2024-11-05 并告警（M 风险处置）。

WorkBuddy 是唯一适配承诺的客户端；协议按通用 MCP 实现。
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable

__all__ = ["JsonRpcError", "PROTOCOL_VERSIONS", "DEFAULT_PROTOCOL_VERSION",
           "Dispatcher", "result_payload"]

# 协议常量
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

PROTOCOL_VERSIONS = ("2026-07-28", "2025-11-25", "2025-06-18", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = "2024-11-05"   # 未知首包回退值（M1 处置）

SERVER_INFO = {"name": "rkllm-mcp", "version": "1.0.0"}

# 方法 → 支持的协议版本（未列出 = 全版本支持）；server/discover 全版本实现，
# 作为新客户端的探针与旧客户端的兼容探测（设计 §1.2）。
PROTOCOL_MATRIX: dict[str, tuple[str, ...]] = {
    # tasks 扩展不做（WorkBuddy 不声明）；tools/call 通用
}


class JsonRpcError(Exception):
    """携带 code 的 JSON-RPC 错误；data 透传给客户端。"""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def as_object(self) -> dict:
        err = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


def result_payload(protocol_version: str, result: dict) -> dict:
    """按协议版本补齐 result 包裹。

    2026-07-28 的 `resultType` 取值为 `complete` / `input_required`
    （tasks 另有 `task`）。本连接器没有需要人工补输入的流程，统一 `complete`；
    之前写成 `text`/`structured` 是错的（设计 §1.1）。
    """
    if protocol_version == "2026-07-28" and "resultType" not in result:
        merged = dict(result)
        merged["resultType"] = "complete"
        return merged
    return result


def detect_protocol(first_frame: dict,
                    supported: tuple[str, ...] | list[str] | None = None) -> str:
    """按首包判定协议版本（MCP 设计 §8：首包 _meta 或 initialize）。

    `supported` 来自 mcp.protocol_versions —— 收缩支持范围时，未列出的版本
    一律回落到默认版本（客户端会按 initialize 返回的版本继续）。
    """
    versions = tuple(supported) if supported else PROTOCOL_VERSIONS
    params = first_frame.get("params") or {}
    meta = params.get("_meta") or {}
    version = meta.get("protocolVersion")
    if version:
        return version if version in versions else DEFAULT_PROTOCOL_VERSION
    if first_frame.get("method") == "initialize":
        requested = params.get("protocolVersion")
        if requested in versions:
            return requested
        return DEFAULT_PROTOCOL_VERSION
    return DEFAULT_PROTOCOL_VERSION


class Dispatcher:
    """方法注册表 + JSON-RPC 信封处理（解析/校验/错误码集中在此）。"""

    def __init__(self, protocol_version: str,
                 supported: tuple[str, ...] | list[str] | None = None):
        self.protocol_version = protocol_version
        # mcp.protocol_versions：本连接器声明支持的版本（默认全部四版）
        self.supported = tuple(supported) if supported else PROTOCOL_VERSIONS
        self._handlers: dict[str, Callable[[dict], dict]] = {}
        # 当前正在处理的请求 id（线程本地）：stdio 用它把
        # notifications/cancelled 的 requestId 对上号
        self._local = threading.local()

    @property
    def current_request_id(self):
        return getattr(self._local, "request_id", None)

    def register(self, method: str, handler: Callable[[dict], dict]) -> None:
        self._handlers[method] = handler

    def supported_methods(self) -> list[str]:
        return sorted(self._handlers)

    def handle_message(self, raw: str) -> dict | None:
        """处理一行 JSON-RPC。返回响应 dict；通知/请求返回 None（无 id）。

        非法 JSON 回 -32700 后继续服务（M9：不退出）；
        未知方法回 -32601；方法与本协议版本不匹配也回 -32601。
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return {"jsonrpc": "2.0", "id": None,
                    "error": {"code": PARSE_ERROR,
                              "message": "invalid JSON"}}
        if not isinstance(msg, dict):
            return self._error(None, INVALID_REQUEST, "request must be an object")
        method = msg.get("method")
        msg_id = msg.get("id")
        if not method:
            return None                        # 对我们响应的回复：忽略
        if method.startswith("notifications/"):
            handler = self._handlers.get(method)
            if handler:
                try:
                    handler(msg.get("params") or {})
                except Exception:
                    pass                       # 通知失败不影响服务
            return None                        # 通知永不回包
        handler = self._handlers.get(method)
        if handler is None:
            return self._error(msg_id, METHOD_NOT_FOUND, f"unknown method: {method}")
        if method in PROTOCOL_MATRIX and \
                self.protocol_version not in PROTOCOL_MATRIX[method]:
            return self._error(msg_id, METHOD_NOT_FOUND,
                               f"{method} requires {'/'.join(PROTOCOL_MATRIX[method])}")
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            return self._error(msg_id, INVALID_PARAMS, "params must be an object")
        self._local.request_id = msg_id
        try:
            result = handler(params)
        except JsonRpcError as exc:
            return self._error(msg_id, exc.code, exc.message, exc.data)
        except Exception as exc:               # 业务异常统一 -32603，不带堆栈
            return self._error(msg_id, INTERNAL_ERROR, str(exc)[:500])
        finally:
            self._local.request_id = None
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": result_payload(self.protocol_version, result)}

    @staticmethod
    def _error(msg_id, code: int, message: str, data: Any = None) -> dict:
        err: dict = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": msg_id, "error": err}
