"""FastAPI 组装 + 优雅退出（改造设计 §6.2 / D11）。"""

from __future__ import annotations

import contextlib
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ServiceState:
    """进程级服务状态：supervisor / 队列 / 配置的装配点（app.state.svc）。"""

    def __init__(self, cfg, *, supervisor=None):
        from .queue import InferenceQueue
        self.cfg = cfg
        self.queue = InferenceQueue(
            max_wait=cfg.queue.max_wait,
            wait_timeout_s=cfg.queue.wait_timeout_s,
            request_timeout_s=cfg.queue.request_timeout_s,
            busy_retry_after_s=cfg.queue.busy_retry_after_s)
        self.supervisor = supervisor      # None = 未配置推理（仅运维端点）
        self.model_name = (cfg.model.path.rsplit("/", 1)[-1]
                           .rsplit("\\", 1)[-1].removesuffix(".rkllm")
                           or "rkllm")
        self.model_path = cfg.model.path
        self.context_len = cfg.model.context
        self.default_max_tokens = cfg.model.max_tokens or 2048
        self.param_policy = cfg.openai.param_policy
        self.tool_parser = cfg.openai.tool_parser
        self.max_tool_calls = cfg.openai.max_tool_calls
        self.system_prompt = ""
        self.version = "1.4.0"
        self.lib_fingerprint: dict | None = None

    def on_reset(self) -> None:
        if self.supervisor:
            # 走 worker 的 clear_history（帧协议扩展：abort 通道复用控制帧）
            self.supervisor.abort()


def create_app(cfg=None, *, supervisor=None) -> FastAPI:
    from .routes import ops, openai

    app = FastAPI(title="rkllm-server", version="1.4.0",
                  docs_url=None, redoc_url=None, openapi_url="/openapi.json")
    svc = ServiceState(cfg, supervisor=supervisor) if cfg else None
    app.state.svc = svc
    app.include_router(ops.router)
    app.include_router(openai.router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        return JSONResponse({"error": {"type": "server_error",
                                       "message": str(exc)[:300]}}, status_code=500)

    return app
