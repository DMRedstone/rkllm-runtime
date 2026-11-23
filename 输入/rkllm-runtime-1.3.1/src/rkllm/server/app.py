"""FastAPI 组装 + 优雅退出（改造设计 §6.2 / D11）。

中间件实际顺序（**由外到内**）：
request_id → CORS → 请求体上限 (+ 鉴权前置) → 鉴权。

注意方向：`app.middleware("http")` 内部是 `add_middleware`，而它是
`insert(0)`——**后注册的在更外层**，所以执行顺序与本文件里注册的顺序相反
（本文件注册顺序：鉴权 → 体限 → CORS → request_id）。

1. request_id（最后注册，最外层）：生成/透传请求 id，预检与被拒请求也有 id；
2. CORS：OPTIONS 预检必须在鉴权之前放行——浏览器的预检请求不带
   Authorization，若被鉴权拦下，跨域客户端整体不可用；
3. 请求体上限（server.max_body_bytes）：含 chunked 的流式计数；
   **P2-4 鉴权前置**：在读取 body 之前先按同样的 Bearer 规则校验一次，
   未授权直接 401、不读 body（chunked 请求此前会先被整段缓冲，未授权的
   慢体/大 body 可无凭据占用内存与连接）；`/ready` 豁免、未启用鉴权时不变；
4. 鉴权（N4，最内层）：默认拒绝，仅 `/ready` 豁免；api_key 为空时不启用。

运维端点 /metrics 的路径与开关来自 metrics.enabled / metrics.path。
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import VERSION
from .auth import (EXEMPT_PATHS, auth_status, make_auth_middleware,
                   resolve_api_key)
from .errors import install_handlers as install_error_handlers
from .logsetup import request_id_var
from .state import ServiceState
from .sysmetrics import ResourceSampler
from .watchdog import ready as watchdog_ready
from .watchdog import start_loop_watchdog, stopping as watchdog_stopping

log = logging.getLogger("rkllm.server")


def create_app(cfg=None, *, supervisor=None) -> FastAPI:
    from .routes import ops, openai, responses, unsupported

    @asynccontextmanager
    async def lifespan(_app):
        # systemd 看门狗：报到前 systemd 认为服务未就绪，报到中断则重启。
        # 不在 systemd 下（NOTIFY_SOCKET 未设置）时两处都是 no-op。
        watchdog_ready()
        start_loop_watchdog(asyncio.get_running_loop())
        # 资源时序：NPU/DDR 频率、内存、进程 RSS（Prometheus 每次抓取取最新值）
        sampler = ResourceSampler()
        sampler.start()
        _app.state.res_sampler = sampler
        try:
            yield
        finally:
            sampler.stop()
            watchdog_stopping()

    app = FastAPI(title="rkllm-server", version=VERSION,
                  docs_url=None, redoc_url=None, openapi_url="/openapi.json",
                  lifespan=lifespan)
    svc = ServiceState(cfg, supervisor=supervisor) if cfg else None
    app.state.svc = svc
    # 解析后的鉴权状态（P1-S1）：启动横幅、响应头、/health 共用同一份事实，
    # 不再用"配置了 key 文件"冒充"已鉴权"。
    app.state.auth = None

    if cfg is not None and svc is not None:
        # -- 鉴权（最内层）--------------------------------------------------
        # 即使未启用鉴权也挂中间件：它负责在响应头标注 enabled/disabled；
        # 构造时 strict 解析 key 文件，文件在 load_config 后消失也会拒绝启动。
        expected_key = resolve_api_key(cfg, strict=True)
        auth = make_auth_middleware(cfg)
        app.state.auth = getattr(auth, "auth_state", None) or auth_status(cfg)
        if auth is not None:
            app.middleware("http")(auth)

        # -- 请求体上限（含 P2-4 鉴权前置）------------------------------------
        max_body = int(cfg.server.max_body_bytes)
        _bearer_prefix = "bearer "

        def _authorized(request: Request) -> bool:
            """与 auth 中间件同口径的 Bearer 校验（未启用鉴权恒 True）。"""
            if not expected_key:
                return True
            header = request.headers.get("authorization", "")
            if header[:len(_bearer_prefix)].lower() != _bearer_prefix:
                return False
            token = header[len(_bearer_prefix):].strip()
            return bool(token) and hmac.compare_digest(
                token.encode("utf-8"), expected_key.encode("utf-8"))

        def _early_unauthorized() -> JSONResponse:
            state = app.state.auth or {}
            return JSONResponse(
                {"error": {"type": "authentication_error",
                           "message": "missing or invalid API key",
                           "param": None, "code": "invalid_api_key"}},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer",
                         "X-RKLLM-Auth": "enabled",
                         "X-RKLLM-Auth-Source": state.get("source", "none")})

        async def body_limit(request: Request, call_next):
            method = request.method
            # 先鉴权再读体：chunked 请求没有 Content-Length，旧实现会先
            # `request.stream()` 整段缓冲，未授权的慢体/大 body 可以无凭据
            # 占用内存与连接。这里在任何 body 读取之前挡下（/ready 豁免）。
            if method in ("POST", "PUT", "PATCH") \
                    and request.url.path not in EXEMPT_PATHS \
                    and not _authorized(request):
                return _early_unauthorized()
            declared = request.headers.get("content-length")
            if declared:
                try:
                    if int(declared) > max_body:
                        return JSONResponse(
                            {"error": {
                                "type": "invalid_request_error",
                                "message": "request body exceeds %d bytes"
                                           % max_body,
                                "param": None,
                                "code": "request_too_large"}},
                            status_code=413)
                except ValueError:
                    return JSONResponse(
                        {"error": {
                            "type": "invalid_request_error",
                            "message": "invalid Content-Length",
                            "param": None,
                            "code": "invalid_content_length"}},
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
                            {"error": {
                                "type": "invalid_request_error",
                                "message": "request body exceeds %d bytes"
                                           % max_body,
                                "param": None,
                                "code": "request_too_large"}},
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
        # 根路径别名不再靠中间件改写：数据面路由统一 mount 两次（见下方
        # include_router），新增端点不会再因为忘记加进别名表而漏挂（C-H1）。

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

    # -- 路由挂载：每个数据面 router 挂两次 ---------------------------------
    # 带 `/v1` 前缀的那次进 OpenAPI；根路径别名那次 `include_in_schema=False`，
    # 只服务"base_url 没写 /v1"的客户端。别名由挂载派生，不再有手工别名表。
    app.include_router(ops.router)         # /health /ready /admin/models（仅根路径）
    for prefix, in_schema in (("/v1", True), ("", False)):
        app.include_router(ops.v1_router, prefix=prefix,
                           include_in_schema=in_schema)      # capabilities / reset
        app.include_router(openai.router, prefix=prefix,
                           include_in_schema=in_schema)
        app.include_router(responses.router, prefix=prefix,
                           include_in_schema=in_schema)      # Responses 翻译层
        # 未实现的 OpenAI/Anthropic 端点：显式 501，不落框架 404（G5）
        app.include_router(unsupported.router, prefix=prefix,
                           include_in_schema=in_schema)

    # -- 根路径信息（与数据面同级鉴权）--------------------------------------
    # 路由注册在鉴权中间件之后，因此启用 api_key 时 `GET /` 也返回 401；
    # 只有 /ready 在 auth.EXEMPT_PATHS 里豁免。此前在别名中间件里直接
    # return 会绕过鉴权，泄露模型名/状态/版本。
    @app.get("/", include_in_schema=False)
    def root_info(request: Request):
        state = request.app.state.svc
        if state is None:
            return {"service": "rkllm-server", "version": VERSION}
        return {
            "service": "rkllm-server",
            "version": state.version,
            "model": state.model_name,
            "state": (state.supervisor.state if state.supervisor
                      else "unconfigured"),
            "endpoints": {
                "health": "/health",
                "ready": "/ready",
                "models": ["/v1/models", "/models"],
                "chat_completions": ["/v1/chat/completions",
                                     "/chat/completions"],
                "completions": ["/v1/completions", "/completions"],
                "responses": ["/v1/responses", "/responses"],
                "capabilities": ["/v1/capabilities", "/capabilities"],
                "reset": ["/v1/reset", "/reset"],
            },
        }

    # -- /metrics 的开关与路径来自配置（N5；键不再空转）----------------------
    if cfg is not None:
        if cfg.metrics.enabled:
            from fastapi.responses import PlainTextResponse
            metrics_path = cfg.metrics.path or "/metrics"

            @app.get(metrics_path)
            def _metrics(request: Request):
                return PlainTextResponse(_render_metrics(request),
                                         media_type="text/plain; version=0.0.4")

    from starlette.exceptions import HTTPException as _StarletteHTTPException

    @app.exception_handler(_StarletteHTTPException)
    async def http_error(request: Request, exc: _StarletteHTTPException):
        # P2-1：框架层 HTTP 错误（404/405/其它）一律 OpenAI 形状。
        # 此前只有 404 转换，405 仍是 Starlette `{"detail": "Method Not Allowed"}`
        # ——OpenAI 客户端读不到 error.type/message，重试/降级逻辑无从判断。
        status = int(getattr(exc, "status_code", 500) or 500)
        detail = getattr(exc, "detail", None)
        message = detail if isinstance(detail, str) and detail else ""
        if status == 404:
            err_type, code = "invalid_request_error", "unknown_endpoint"
            message = "未知端点：%s" % request.url.path
        elif status == 405:
            err_type, code = "invalid_request_error", "method_not_allowed"
            message = "method not allowed: %s %s" % (request.method,
                                                     request.url.path)
        elif status == 401:
            err_type, code = "authentication_error", "invalid_api_key"
            message = message or "missing or invalid API key"
        elif 400 <= status < 500:
            err_type, code = "invalid_request_error", None
            message = message or "invalid request"
        else:
            err_type, code = "server_error", "http_error"
            message = message or "internal server error"
        return JSONResponse(
            {"error": {"type": err_type, "message": message,
                       "param": None, "code": code}},
            status_code=status, headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        # 不回显 traceback；异常详情交给日志（不含请求体，避免带出密钥）
        import logging
        logging.getLogger("rkllm.server").exception("unhandled: %s", exc)
        return JSONResponse({"error": {"type": "server_error",
                                       "message": str(exc)[:300],
                                       "param": None,
                                       "code": "internal_error"}}, status_code=500)

    # 参数校验失败：FastAPI 默认 422 {"detail": ...}；改注册为 OpenAI 400 形状。
    install_error_handlers(app)

    return app


def _render_metrics(request: Request) -> str:
    """Prometheus 文本（设计 §7.4 指标集，含 prompt 缓存命中与占用）。

    直方图由 `prometheus_client` 对象持有（`svc.queue.metrics_registry`）；
    标量每渲染一次灌进临时 registry，避免全局可变注册状态——多实例与测试
    互不干扰，输出仍是标准 exposition 格式。
    """
    svc = request.app.state.svc
    from .cacheview import cache_stats as _cache_stats   # 与 /health 共用同一份统计
    from .metricsutil import render_text, scalar_registry
    m = svc.queue.metrics
    sup = svc.supervisor.metrics if svc.supervisor else {}
    cache = _cache_stats(svc.cfg)          # 只读一次，避免 /metrics 双次统计
    from ..agent.tokens import stats as _token_stats
    tok = _token_stats()                   # token 比例校准状态（counter + gauge 都用）
    counters = {
        "rkllm_requests_total": ("受理的请求数", m["requests_total"]),
        "rkllm_rejected_total": ("被拒绝/超时的请求数", m["rejected_total"]),
        "rkllm_queue_wait_seconds_total": ("排队等待累计秒数",
                                         m["queue_wait_seconds_sum"]),
        "rkllm_tokens_total": ("生成 token 总数", m["tokens_total"]),
        "rkllm_worker_restarts_total": ("worker 重启次数",
                                        sup.get("worker_restarts_total", 0)),
        "rkllm_worker_heartbeat_timeouts_total": (
            "worker 心跳超时次数",
            sup.get("worker_heartbeat_timeouts_total", 0)),
        "rkllm_cache_hits_total": ("prompt 缓存命中次数", cache["hits"]),
        "rkllm_request_duration_seconds_total": ("请求累计耗时秒数",
                                               m.get("duration_sum", 0.0)),
        # 计数器一律以 _total 结尾（Prometheus 约定）：prometheus_client 会给
        # Counter 追加 _total 与 _created，旧名字里的 _sum 后缀会被改写，
        # 与其被库改名，不如先按约定命名。
        "rkllm_token_calibrations_total": ("token 比例校准样本数",
                                           tok["samples"]),
        "rkllm_max_tokens_clamped_total": (
            "max_tokens 被 context 预算裁剪的次数",
            m.get("max_tokens_clamped_total", 0)),
    }
    sampler = getattr(request.app.state, "res_sampler", None)
    snap = sampler.snapshot() if sampler is not None else {}
    gauges = {
        "rkllm_token_bytes_per_token": ("token 估算的 bytes/token（自校准）",
                                        tok["bytes_per_token"]),
        "rkllm_npu_freq_mhz": ("NPU 当前频率（MHz）", snap.get("npu_freq_mhz") or 0.0),
        "rkllm_ddr_freq_mhz": ("DDR 当前频率（MHz）", snap.get("ddr_freq_mhz") or 0.0),
        "rkllm_mem_available_mb": ("系统可用内存（MB）",
                                   snap.get("mem_available_mb") or 0.0),
        "rkllm_mem_total_mb": ("系统总内存（MB）", snap.get("mem_total_mb") or 0.0),
        "rkllm_process_rss_mb": ("服务进程常驻内存（MB）",
                                 snap.get("proc_rss_mb") or 0.0),
        "rkllm_queue_depth": ("当前排队请求数", svc.queue.queue_depth),
        "rkllm_cache_bytes": ("prompt 缓存占用字节", cache["bytes"]),
        "rkllm_ttft_seconds": ("最近一次首 token 延迟（秒）",
                               m.get("last_ttft_s", 0.0)),
    }
    labelled = [("rkllm_worker_state", "worker 状态（1=当前状态）",
                 {"state": svc.supervisor.state if svc.supervisor else "none"}, 1.0)]
    scalars = scalar_registry(counters, gauges, labelled)
    return render_text(scalars, svc.queue.metrics_registry)
