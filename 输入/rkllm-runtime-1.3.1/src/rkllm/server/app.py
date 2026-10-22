"""FastAPI 组装 + 优雅退出（改造设计 §6.2 / D11）。

装配顺序（中间件是洋葱模型，先加的在最外层）：
1. 鉴权（N4）：默认拒绝，仅 `/ready` 豁免；api_key 为空时不启用；
2. 请求体上限（server.max_body_bytes）：按 Content-Length 前置拒绝；
3. CORS（server.cors_origins 非空时）。
运维端点 /metrics 的路径与开关来自 metrics.enabled / metrics.path。
"""
from __future__ import annotations

import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..platform import derive_class, max_tokens_for
from .auth import make_auth_middleware


class ServiceState:
    """进程级服务状态：supervisor / 队列 / 配置的装配点（app.state.svc）。"""

    def __init__(self, cfg, *, supervisor=None):
        from .queue import InferenceQueue
        self.cfg = cfg
        self.queue = InferenceQueue(
            max_wait=cfg.queue.max_wait,
            wait_timeout_s=cfg.queue.wait_timeout_s,
            request_timeout_s=cfg.queue.request_timeout_s,
            busy_retry_after_s=cfg.queue.busy_retry_after_s,
            max_concurrent=cfg.queue.max_concurrent)
        self.supervisor = supervisor      # None = 未配置推理（仅运维端点）
        self.model_name = (cfg.model.path.rsplit("/", 1)[-1]
                           .rsplit("\\", 1)[-1].removesuffix(".rkllm")
                           or "rkllm")
        self.model_path = cfg.model.path
        self.context_len = cfg.model.context
        # 档位（F12）：显式配置优先，否则按文件名/体积推导。
        # 键名是 model.class（Python 关键字），只能用 getattr 取。
        self.model_class = derive_class(cfg.model.path,
                                       getattr(cfg.model, "class", "auto"))
        self.default_max_tokens = (cfg.model.max_tokens
                                   or max_tokens_for(self.model_class))
        self.param_policy = cfg.openai.param_policy
        self.tool_parser = cfg.openai.tool_parser
        self.max_tool_calls = cfg.openai.max_tool_calls
        # openai.tools_mode：native 需 rkllm_set_function_tools，ADR-06 明确不用
        # （handle 级永久生效会污染后续请求），配置成 native 时降级为 prompt 并告警
        self.tools_mode = cfg.openai.tools_mode
        if self.tools_mode == "native":
            sys.stderr.write(
                "[rkllm] openai.tools_mode=native 不受支持（ADR-06），已按 prompt 处理\n")
        self.reasoning_field = cfg.openai.reasoning_field
        self.json_schema_retries = cfg.openai.json_schema_retries
        self.stream_heartbeat_s = cfg.openai.stream_heartbeat_s
        self.include_usage_default = cfg.openai.include_usage_default
        # 进程级采样默认值（--temperature 等）：请求未指定时使用，
        # 由 cli 在启动时填入；空 dict 表示全用 runtime 默认。
        self.default_sampling: dict = {}
        self.system_prompt = ""
        self.version = "1.3.1-16"
        self.lib_fingerprint: dict | None = None

    def on_reset(self) -> None:
        if self.supervisor:
            # 走 worker 的 abort（帧协议控制帧）；KV 由下一次请求全量重放
            self.supervisor.abort()

    def ttft_log(self, seconds: float) -> None:
        """首 token 耗时打点（N3 性能定位；默认 info 级，便于板端直接看）。"""
        import logging
        logging.getLogger("rkllm.server.ttft").info(
            "supervisor 首 token %.2fs", seconds)


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
    from .routes.ops import _cache_stats      # 与 /health 共用同一份统计
    m = svc.queue.metrics
    sup = svc.supervisor.metrics if svc.supervisor else {}
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
        "rkllm_cache_hits_total %d" % _cache_stats(svc.cfg)["hits"],
        "# TYPE rkllm_cache_bytes gauge",
        "rkllm_cache_bytes %d" % _cache_stats(svc.cfg)["bytes"],
        "# TYPE rkllm_worker_state gauge",
        'rkllm_worker_state{state="%s"} 1' % (
            svc.supervisor.state if svc.supervisor else "none"),
        "# TYPE rkllm_ttft_seconds gauge",
        "rkllm_ttft_seconds %f" % m.get("last_ttft_s", 0.0),
        "# TYPE rkllm_request_duration_seconds_sum counter",
        "rkllm_request_duration_seconds_sum %f" % m.get("duration_sum", 0.0),
    ]
    return "\n".join(lines) + "\n"
