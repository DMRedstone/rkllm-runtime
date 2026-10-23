"""FastAPI 组装 + 优雅退出（改造设计 §6.2 / D11）。

中间件实际顺序（**由外到内**）：CORS → 根路径改写 → 请求体上限 → 鉴权。

注意方向：`app.middleware("http")` 内部是 `add_middleware`，而它是
`insert(0)`——**后注册的在更外层**，所以执行顺序与本文件里注册的顺序相反。

1. CORS（最后注册，最外层）：OPTIONS 预检必须在鉴权之前放行——浏览器的
   预检请求不带 Authorization，若被鉴权拦下，跨域客户端整体不可用；
2. 根路径改写 root_alias：把 `/chat/completions` 等精确路径改写到 `/v1`，
   后续鉴权与路由都看改写后的路径；
3. 请求体上限（server.max_body_bytes）：含 chunked 的流式计数；
4. 鉴权（N4，最内层）：默认拒绝，仅 `/ready` 豁免；api_key 为空时不启用。

取舍：体限在鉴权外层，未授权的超大 body 会先拿到 413 而不是 401。这是既有
顺序的代价（把体限挪到鉴权内层需连同 root_alias 一起重排，而 CORS 必须
留在最外层不能被挤动）。

运维端点 /metrics 的路径与开关来自 metrics.enabled / metrics.path。
"""
from __future__ import annotations

import logging
import sys
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .auth import make_auth_middleware
from .logsetup import request_id_var
from .state import ServiceState

log = logging.getLogger("rkllm.server")


def create_app(cfg=None, *, supervisor=None) -> FastAPI:
    from .routes import ops, openai, responses, unsupported

    app = FastAPI(title="rkllm-server", version="1.3.1-16",
                  docs_url=None, redoc_url=None, openapi_url="/openapi.json")
    svc = ServiceState(cfg, supervisor=supervisor) if cfg else None
    app.state.svc = svc

    if cfg is not None and svc is not None:
        # -- 鉴权（最外层）--------------------------------------------------
        auth = make_auth_middleware(cfg)
        if auth is not None:
            app.middleware("http")(auth)

        # -- 请求体上限 ------------------------------------------------------
        max_body = int(cfg.server.max_body_bytes)

        async def body_limit(request: Request, call_next):
            declared = request.headers.get("content-length")
            if declared:
                try:
                    if int(declared) > max_body:
                        return JSONResponse(
                            {"error": {"type": "invalid_request_error",
                                       "message": "request body exceeds %d bytes"
                                                  % max_body}},
                            status_code=413)
                except ValueError:
                    return JSONResponse(
                        {"error": {"type": "invalid_request_error",
                                   "message": "invalid Content-Length"}},
                        status_code=400)
            elif request.method in ("POST", "PUT", "PATCH"):
                # P2：chunked / 无 Content-Length 时旧逻辑直接放行，可绕过上限。
                # 这里流式读入并计数字节，超限即 413；并把已读 body 缓存下来，
                # 供下游 FastAPI 解析（设置 _body 后 body() 直接返回缓存）。
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > max_body:
                        return JSONResponse(
                            {"error": {"type": "invalid_request_error",
                                       "message": "request body exceeds %d bytes"
                                                  % max_body}},
                            status_code=413)
                request._body = bytes(body)
            return await call_next(request)

        app.middleware("http")(body_limit)

        # -- 根路径兼容：OpenAI 端点在 /v1 与根下等价 -----------------------
        # 背景：不少客户端把 base_url 配成 http://host:port（不带 /v1），
        # 于是 /models、/chat/completions 全部落框架 404。这里把这类**精确**
        # 路径改写到 /v1 前缀；/v1/* 与运维端点原样透传。
        # 不用 301：部分客户端不跟随重定向，且 POST 会被改成 GET。
        # 不复制 router：route.path 已含 /v1，去前缀需重建 endpoint 元数据，
        # 与后续路由易漂移。
        _ROOT_ALIASES = frozenset({
            "/models", "/chat/completions", "/completions",
            "/responses", "/embeddings", "/moderations",
        })

        async def root_alias(request: Request, call_next):
            path = request.url.path
            if request.method == "OPTIONS":
                return await call_next(request)      # 预检交给 CORS 处理
            if path == "/":
                state = app.state.svc
                if state is None:
                    return JSONResponse({"service": "rkllm-server",
                                         "version": "1.3.1-16"})
                return JSONResponse({
                    "service": "rkllm-server",
                    "version": state.version,
                    "model": state.model_name,
                    "state": (state.supervisor.state
                              if state.supervisor else "unconfigured"),
                    "endpoints": {
                        "health": "/health",
                        "ready": "/ready",
                        "models": ["/v1/models", "/models"],
                        "chat_completions": ["/v1/chat/completions",
                                             "/chat/completions"],
                        "completions": ["/v1/completions", "/completions"],
                        "responses": ["/v1/responses", "/responses"],
                        "capabilities": "/v1/capabilities",
                        "reset": "/v1/reset",
                    },
                })
            if path in _ROOT_ALIASES:
                request.scope["path"] = "/v1" + path
            return await call_next(request)

        app.middleware("http")(root_alias)

        # -- CORS（空列表 = 不发 CORS 头）------------------------------------
        origins = list(cfg.server.cors_origins or ())
        if origins:
            from fastapi.middleware.cors import CORSMiddleware
            app.add_middleware(CORSMiddleware, allow_origins=origins,
                               allow_credentials=False,
                               allow_methods=["GET", "POST", "OPTIONS"],
                               allow_headers=["*"])

        # -- request_id（A-6：最后注册 = 最外层，预检与被拒请求也有 id）-------
        async def request_id_mw(request: Request, call_next):
            rid = (request.headers.get("x-request-id") or "").strip()[:64] \
                or uuid.uuid4().hex[:12]
            token = request_id_var.set(rid)
            try:
                response = await call_next(request)
            finally:
                request_id_var.reset(token)
            response.headers["X-Request-Id"] = rid
            return response

        app.middleware("http")(request_id_mw)

    app.include_router(ops.router)
    app.include_router(openai.router)
    app.include_router(responses.router)   # Responses 翻译层（emulated）
    # 未实现的 OpenAI/Anthropic 端点：显式 501，不落框架 404（G5）
    app.include_router(unsupported.router)

    # -- /metrics 的开关与路径来自配置（N5；键不再空转）----------------------
    if cfg is not None:
        if cfg.metrics.enabled:
            from fastapi.responses import PlainTextResponse
            metrics_path = cfg.metrics.path or "/metrics"

            @app.get(metrics_path)
            def _metrics(request: Request):
                return PlainTextResponse(_render_metrics(request),
                                         media_type="text/plain; version=0.0.4")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        # 不回显 traceback；异常详情交给日志（不含请求体，避免带出密钥）
        import logging
        logging.getLogger("rkllm.server").exception("unhandled: %s", exc)
        return JSONResponse({"error": {"type": "server_error",
                                       "message": str(exc)[:300]}}, status_code=500)

    return app


def _render_metrics(request: Request) -> str:
    """Prometheus 文本（设计 §7.4 指标集，含 prompt 缓存命中与占用）。"""
    svc = request.app.state.svc
    from .cacheview import cache_stats as _cache_stats   # 与 /health 共用同一份统计
    m = svc.queue.metrics
    sup = svc.supervisor.metrics if svc.supervisor else {}
    cache = _cache_stats(svc.cfg)          # 只读一次，避免 /metrics 双次统计
    lines = [
        "# TYPE rkllm_requests_total counter",
        "rkllm_requests_total %d" % m["requests_total"],
        "# TYPE rkllm_rejected_total counter",
        "rkllm_rejected_total %d" % m["rejected_total"],
        "# TYPE rkllm_queue_depth gauge",
        "rkllm_queue_depth %d" % svc.queue.queue_depth,
        "# TYPE rkllm_queue_wait_seconds_sum counter",
        "rkllm_queue_wait_seconds_sum %f" % m["queue_wait_seconds_sum"],
        "# TYPE rkllm_tokens_total counter",
        "rkllm_tokens_total %d" % m["tokens_total"],
        "# TYPE rkllm_worker_restarts_total counter",
        "rkllm_worker_restarts_total %d" % sup.get("worker_restarts_total", 0),
        "# TYPE rkllm_cache_hits_total counter",
        "rkllm_cache_hits_total %d" % cache["hits"],
        "# TYPE rkllm_cache_bytes gauge",
        "rkllm_cache_bytes %d" % cache["bytes"],
        "# TYPE rkllm_worker_state gauge",
        'rkllm_worker_state{state="%s"} 1' % (
            svc.supervisor.state if svc.supervisor else "none"),
        "# TYPE rkllm_ttft_seconds gauge",
        "rkllm_ttft_seconds %f" % m.get("last_ttft_s", 0.0),
        "# TYPE rkllm_request_duration_seconds_sum counter",
        "rkllm_request_duration_seconds_sum %f" % m.get("duration_sum", 0.0),
    ]
    return "\n".join(lines) + "\n"
