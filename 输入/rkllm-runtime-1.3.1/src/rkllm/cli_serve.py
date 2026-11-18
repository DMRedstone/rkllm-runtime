"""`rkllm serve`（A-4：从 cli.py 拆出）。

流程：旧参数 → 配置覆盖层 → 日志 → 模型路径/内存预算 → supervisor →
uvicorn。运维视角与 1.3.1-15 保持一致（命令名/单元名/conffile 不变）。
"""

from __future__ import annotations

import os
import sys

from . import VERSION
from .cli_engine import (_apply_memory_budget, _build_supervisor,
                         _resolve_model_path)
from .cli_env import _legacy_cli_overrides, _sampling_from_args

__all__ = ["_cmd_serve"]


def _cmd_serve(args) -> int:
    import uvicorn

    from .config import ConfigError, load_config
    from .server.app import create_app
    from .server.logsetup import setup_logging

    try:
        cfg = load_config(file=os.environ.get("RKLLM_TOML") or None,
                          cli=_legacy_cli_overrides(args),
                          require_model_path=True)
    except ConfigError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2

    setup_logging(cfg)
    from .platform import (apply_freq_lock, cache_max_mb_for, derive_class,
                       max_tokens_for, max_tokens_limit_for)
    cfg.values["model.path"] = _resolve_model_path(cfg.model.path,
                                                  cfg.model.search_dirs)
    # token 计数：有 tokenizer.json 就用精确实现，否则靠 prefill_tokens 自校准
    from .agent.tokens import configure as _configure_tokens, stats as _token_stats
    _tok = _configure_tokens(cfg.model.path)
    if _tok["backend"] == "tokenizers":
        sys.stderr.write("[rkllm] token 计数：精确 tokenizer（%s）\n" % _tok["source"])
    # 档位默认只覆盖 4b/7b；仅当用户未显式配置（source=default）时套用，
    # 显式配置永远优先。更大档位（14b/27b/未知）不内置，打印告警要求手动配置。
    model_class = derive_class(cfg.model.path,
                               getattr(cfg.model, "class", "auto"))
    if cfg.sources.get("model.max_tokens") == "default":
        cfg.values["model.max_tokens"] = max_tokens_for(model_class)
    if cfg.sources.get("model.max_tokens_limit") == "default":
        cfg.values["model.max_tokens_limit"] = max_tokens_limit_for(model_class)
    if cfg.sources.get("cache.max_mb") == "default":
        cfg.values["cache.max_mb"] = cache_max_mb_for(model_class)
    if model_class not in ("4b", "7b"):
        sys.stderr.write(
            "[rkllm] 档位 %s 无内置默认（内置仅 4B/7B）；请手动配置 "
            "model.max_tokens / max_tokens_limit / cache.max_mb\n" % model_class)

    use_fake = bool(getattr(args, "fake_engine", False))
    if not use_fake and not os.path.isfile(cfg.model.path):
        sys.stderr.write("[rkllm] 模型文件不存在：%s\n" % cfg.model.path)
        return 2
    _apply_memory_budget(cfg)

    supervisor = _build_supervisor(cfg, use_fake=use_fake)
    # 锁频：默认开（板端供电场景），非 root 或 sysfs 不支持时只告警
    freq_lock = apply_freq_lock(bool(cfg.model.freq_lock), cfg.model.freq_helper)

    app = create_app(cfg, supervisor=supervisor)
    if getattr(args, "system", None):
        app.state.svc.system_prompt = args.system
    app.state.svc.default_sampling = _sampling_from_args(args)

    sys.stderr.write(
        "rkllm %s 服务模型 '%s'，监听 http://%s:%d\n"
        "  endpoints : GET /health /ready /v1/capabilities /metrics, "
        "GET /v1/models, POST /v1/chat/completions /v1/completions /v1/reset\n"
        "  history   : %s   api key : %s   推理进程 : %s\n"
        "  freq lock : %s   档位 : %s   context : %d\n"
        % (VERSION, app.state.svc.model_name, cfg.server.bind_host,
           cfg.server.port,
           "on" if cfg.model.keep_history else "off",
           "set" if (cfg.server.api_key or cfg.server.api_key_file) else "none",
           "in_process" if cfg.engine.mode == "in_process" else "subprocess",
           "on" if cfg.model.freq_lock else "off",
           app.state.svc.model_class, cfg.model.context))
    if not (cfg.server.api_key or cfg.server.api_key_file) and \
            cfg.server.bind_host not in ("127.0.0.1", "::1", "localhost"):
        sys.stderr.write(
            "  WARNING   : 未设 API key 而监听 %s —— 能访问该端口的任何主机都能"
            "使用模型。请在 /etc/default/rkllm-server 里设 RKLLM_API_KEY 并 "
            "chmod 600。\n" % cfg.server.bind_host)

    tls = bool(cfg.server.tls_cert and cfg.server.tls_key)
    try:
        uvicorn.run(
            app, host=cfg.server.bind_host, port=cfg.server.port,
            log_level=cfg.logging.level,
            timeout_graceful_shutdown=cfg.server.shutdown_grace_s,
            access_log=cfg.server.access_log,
            # 以下四项此前是死键，现按 85 键表接线
            limit_concurrency=cfg.server.max_connections,
            timeout_keep_alive=cfg.server.read_timeout_s,
            backlog=cfg.server.request_queue_size,
            ssl_certfile=cfg.server.tls_cert if tls else None,
            ssl_keyfile=cfg.server.tls_key if tls else None,
        )
    finally:
        freq_lock.restore()          # 退出恢复原 governor 与频率
        supervisor.shutdown()
    return 0
