"""OpenAI 契约层错误出口（P1 修复）。

为什么单独成模块：
- `pipeline_runner.http_error()` 只能写 body；数据面 503 需要 HTTP
  `Retry-After` 响应头（网关/客户端按头退避），不能把语义塞进 JSON；
- FastAPI 默认把参数校验失败渲染成 422 `{"detail": ...}`，OpenAI 客户端读的是
  `error.message` / `error.param` / `error.code`，读不到就等于没有错误信息。

本模块是路由层唯一的错误出口，错误体始终是 OpenAI 形状：
`{"error": {"message", "type", "param", "code"}}`。
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

__all__ = ["http_error", "invalid_request", "validation_error_response",
           "install_handlers"]


def http_error(status: int, err_type: str, message: str, *,
               headers: dict[str, str] | None = None,
               **extra: Any) -> JSONResponse:
    """OpenAI 形状错误响应；`headers` 承载 Retry-After 等传输层语义。"""
    body: dict[str, Any] = {"error": {"type": err_type, "message": message}}
    body["error"].update(extra)
    return JSONResponse(body, status_code=status, headers=headers)


def invalid_request(message: str, *, param: str | None = None,
                    code: str = "invalid_request_error",
                    status: int = 400,
                    headers: dict[str, str] | None = None) -> JSONResponse:
    """400 invalid_request_error：参数/请求体不合法（OpenAI 语义）。"""
    return http_error(status, "invalid_request_error", message, headers=headers,
                      param=param, code=code)


def _loc_param(loc) -> str | None:
    """pydantic 的 loc=("body","temperature") → param="temperature"。

    列表下标（bad JSON 的 ("body", 0)）不当作参数名，否则 param 会变成 "0"。
    """
    parts = [str(p) for p in (loc or ())
             if p not in ("body", "query", "path") and not isinstance(p, int)]
    return ".".join(parts) or None


def validation_error_response(exc: RequestValidationError) -> JSONResponse:
    """把 pydantic/FastAPI 校验错误转成 OpenAI 400（含 param/code）。"""
    errors = exc.errors() or []
    first = errors[0] if errors else {}
    param = _loc_param(first.get("loc"))
    detail = str(first.get("msg") or "invalid request")
    message = ("%s: %s" % (param, detail)) if param else detail
    return invalid_request(message, param=param, code="invalid_request_error")


def install_handlers(app: FastAPI) -> None:
    """注册参数校验处理器：FastAPI 422/`detail` → OpenAI 400/`error`。"""

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request,
                                 exc: RequestValidationError) -> JSONResponse:
        return validation_error_response(exc)
