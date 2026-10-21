"""未实现端点的显式拒绝（G5「兼容性诚实」）。

设计取舍：不实现就不承诺，但**必须回话**。此前这些路径落到框架默认 404
（`{"detail":"Not Found"}`），客户端只能看到"地址错了"，无法区分
"服务没有这个能力"和"我把 URL 写错了"——这正是审计里"静默失败"的一类。

约定：
- HTTP 501 Not Implemented（语义是"服务不认识这个方法/能力"，不是 400 参数错）；
- 错误体沿用 OpenAI 形状，code=`unsupported_endpoint`，message 里给出替代路径，
  客户端可以据此自动降级；
- 响应头 `X-RKLLM-Endpoint: unsupported`，与参数级的 `X-RKLLM-Unsupported` 同族。

能力表 caps.ENDPOINTS 是唯一事实源：本文件的路由集合必须与
`tier == "unsupported"` 的端点保持一致（tests/test_unsupported.py 守住）。
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ... import caps

router = APIRouter()

# 路径 -> (能力表键, 建议替代)
_UNSUPPORTED: dict[str, tuple[str, str]] = {
    "/v1/embeddings": ("embeddings", "/v1/chat/completions（本服务不提供向量化）"),
    "/v1/messages": ("messages(anthropic)", "/v1/chat/completions（OpenAI 形状）"),
}


def _reject(path: str) -> JSONResponse:
    cap_key, alternative = _UNSUPPORTED[path]
    # 能力表若改了档位（例如将来真做了 responses），这里必须跟着改；
    # 万一漏改，仍然以能力表为准并打印告警，避免"表说支持、路由说没有"。
    tier = caps.ENDPOINTS.get(cap_key, "unsupported")
    body = {
        "error": {
            "message": (
                "endpoint %s is not implemented by this service (capability=%s). "
                "Use %s instead. See GET /v1/capabilities for the full matrix."
                % (path, tier, alternative)
            ),
            "type": "invalid_request_error",
            "param": None,
            "code": "unsupported_endpoint",
        }
    }
    return JSONResponse(status_code=501, content=body,
                        headers={"X-RKLLM-Endpoint": "unsupported"})


@router.post("/v1/embeddings", include_in_schema=False)
async def embeddings_not_implemented(request: Request) -> JSONResponse:
    return _reject("/v1/embeddings")


@router.post("/v1/messages", include_in_schema=False)
async def anthropic_messages_not_implemented(request: Request) -> JSONResponse:
    return _reject("/v1/messages")


__all__ = ["router", "_UNSUPPORTED"]
