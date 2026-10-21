"""日志装配（logging.* 五键：level / format / file / max_bytes / backup_count）。

N5 要求「JSON 日志、可追踪」。这里只做应用侧装配：uvicorn 自带日志仍在
level 上对齐；worker 的 C 库日志由 supervisor 的排水线程处理（不入本配置）。

JSON 格式只输出稳定字段，避免把请求体（可能含密钥）带进日志：
  ts / level / logger / message
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time

__all__ = ["setup_logging", "JsonFormatter"]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S",
                                time.localtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(cfg) -> None:
    level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    fmt = JsonFormatter() if cfg.logging.format == "json" \
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
