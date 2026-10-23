"""日志装配（logging.* 五键：level / format / file / max_bytes / backup_count）。

N5 要求「JSON 日志、可追踪」。这里只做应用侧装配；worker 的 C 库日志由
supervisor 的排水线程处理（不入本配置）。

A-6 两处修正：
1. **uvicorn 系 logger 归并**：`uvicorn.access` 默认 `propagate=False` 且自带
   handler，配了 `logging.file` 时访问日志仍只进 stderr（双轨，运维在文件里
   看不到请求）。这里把它们挂到 root 的 handler 上、关闭各自 handler；
2. **request_id**：JSON 里加 `request_id`，长流式请求（一次请求多条日志）
   才能按请求关联。id 由 `app.py` 的中间件写入 contextvar，并回写
   `X-Request-Id` 响应头。

JSON 格式只输出稳定字段，避免把请求体（可能含密钥）带进日志：
  ts / level / logger / message / request_id
"""
from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import sys
import time

__all__ = ["setup_logging", "JsonFormatter", "request_id_var",
           "current_request_id"]

# 请求级上下文：middleware 写入，JsonFormatter 读取（跨线程随 context 传递）
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "rkllm_request_id", default="-")

# uvicorn 自己的 logger：不归并就会与 root 双轨（A-6）
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def current_request_id() -> str:
    return request_id_var.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S",
                                time.localtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            # A-6：长流式请求可能产生多条日志，靠 request_id 关联
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(cfg) -> None:
    level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    json_mode = cfg.logging.format == "json"
    fmt = JsonFormatter() if json_mode \
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    path = (cfg.logging.file or "").strip()
    if path:
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=int(cfg.logging.max_bytes),
            backupCount=int(cfg.logging.backup_count), encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    root.addHandler(handler)

    # A-6：uvicorn 的 logger 改走同一套 handler（否则配了 logging.file 时
    # 访问日志仍只写 stderr，且格式与 JSON 不一致）
    for name in _UVICORN_LOGGERS:
        logger = logging.getLogger(name)
        for existing in list(logger.handlers):
            logger.removeHandler(existing)
        logger.propagate = True
        logger.setLevel(level)
