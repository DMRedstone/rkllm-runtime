"""OpenAI 契约层错误出口（P1 修复 / P2-1 扩面）。

为什么单独成模块：
- `pipeline_runner.http_error()` 只能写 body；数据面 503 需要 HTTP
  `Retry-After` 响应头（网关/客户端按头退避），不能把语义塞进 JSON；
- FastAPI 默认把参数校验失败渲染成 422 `{"detail": ...}`，OpenAI 客户端读的是
  `error.message` / `error.param` / `error.code`，读不到就等于没有错误信息。

本模块是路由层唯一的错误出口，错误体始终是 OpenAI 形状：
`{"error": {"message", "type", "param", "code"}}`。

P2-1：`response_format` / Responses `text.format` 的合法性校验也收在这里——
非法格式必须显式 400，不能静默退化成"没有 response_format"（agent 层转换
函数对不认识的字段是静默忽略的）。
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

__all__ = ["http_error", "invalid_request", "validation_error_response",
           "install_handlers", "JSON_OBJECT_SCHEMA", "validate_response_format",
           "validate_text_format"]

# json_object 的校验口径：OpenAI 语义是"一个 JSON 对象"，不是任意 JSON 值。
JSON_OBJECT_SCHEMA: dict[str, Any] = {"type": "object"}

_FORMAT_TYPES = ("text", "json_object", "json_schema")


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


# ---------------------------------------------------------------------------
# response_format / text.format 合法性（P2-1）
# ---------------------------------------------------------------------------


def _check_json_schema(schema: Any, param: str) -> JSONResponse | None:
    """schema 必须是 JSON 对象且能通过 Draft 2020-12 元校验。"""
    if not isinstance(schema, dict):
        return invalid_request(
            "`%s` 必须是 JSON 对象（JSON Schema）" % param,
            param=param, code="invalid_schema")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        return invalid_request(
            "`%s` 不是合法 JSON Schema：%s" % (param, exc.message),
            param=param, code="invalid_schema")
    return None


def validate_response_format(rf: Any
                             ) -> tuple[str | None, dict | None,
                                        JSONResponse | None]:
    """校验 chat 的 `response_format`。

    返回 `(mode, schema, error)`：mode ∈ {None, "json_object", "json_schema"}
    （`text`/缺省 = None）；error 非 None 时路由直接返回它（OpenAI 400）。
    此前 json_schema 缺少 schema、类型拼错都会静默当"无格式"处理——模型
    输出不受约束、客户端也读不到原因。
    """
    if rf is None:
        return None, None, None
    if not isinstance(rf, dict):
        return None, None, invalid_request(
            "`response_format` must be a JSON object", param="response_format",
            code="invalid_response_format")
    kind = rf.get("type", "text")
    if kind not in _FORMAT_TYPES:
        return None, None, invalid_request(
            "unsupported response_format.type=%r（支持 text/json_object/"
            "json_schema）" % (kind,), param="response_format",
            code="invalid_response_format")
    if kind == "json_object":
        return "json_object", None, None
    if kind != "json_schema":
        return None, None, None
    spec = rf.get("json_schema")
    if not isinstance(spec, dict):
        return None, None, invalid_request(
            "`response_format.json_schema` must be an object with a `schema`",
            param="response_format.json_schema", code="invalid_schema")
    schema = spec.get("schema")
    error = _check_json_schema(schema, "response_format.json_schema.schema")
    if error is not None:
        return None, None, error
    name = spec.get("name")
    if name is not None and not isinstance(name, str):
        return None, None, invalid_request(
            "`response_format.json_schema.name` must be a string",
            param="response_format.json_schema.name", code="invalid_schema")
    return "json_schema", schema, None


def validate_text_format(text: Any
                         ) -> tuple[str | None, dict | None,
                                    JSONResponse | None]:
    """校验 Responses 的 `text` / `text.format`（同 response_format 语义）。"""
    if text is None:
        return None, None, None
    if not isinstance(text, dict):
        return None, None, invalid_request(
            "`text` must be a JSON object", param="text",
            code="invalid_response_format")
    fmt = text.get("format")
    if fmt is None:
        return None, None, None
    if not isinstance(fmt, dict):
        return None, None, invalid_request(
            "`text.format` must be a JSON object", param="text.format",
            code="invalid_response_format")
    kind = fmt.get("type", "text")
    if kind not in _FORMAT_TYPES:
        return None, None, invalid_request(
            "unsupported text.format.type=%r（支持 text/json_object/"
            "json_schema）" % (kind,), param="text.format",
            code="unsupported_parameter")
    if kind == "json_object":
        return "json_object", None, None
    if kind != "json_schema":
        return None, None, None
    schema = fmt.get("schema")
    error = _check_json_schema(schema, "text.format.schema")
    if error is not None:
        return None, None, error
    name = fmt.get("name")
    if name is not None and not isinstance(name, str):
        return None, None, invalid_request(
            "`text.format.name` must be a string", param="text.format.name",
            code="invalid_schema")
    return "json_schema", schema, None


def install_handlers(app: FastAPI) -> None:
    """注册参数校验处理器：FastAPI 422/`detail` → OpenAI 400/`error`。"""

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request,
                                 exc: RequestValidationError) -> JSONResponse:
        return validation_error_response(exc)
