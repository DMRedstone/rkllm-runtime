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

__all__ = ["_cmd_serve", "_startup_banner", "_auth_banner"]


def _auth_banner(cfg) -> str:
    """启动横幅里的鉴权状态：必须显示**解析后**的真实状态（P1-S1）。

    旧实现按 `api_key or api_key_file` 判断"是否配置"，key 文件缺失时会打印
    "set" 而实际鉴权未启用。现在用 auth.auth_status()（与响应头/`/health` 同源）。
    """
    from .server.auth import auth_status
    state = auth_status(cfg)
    if not state["enabled"]:
        return "disabled（未鉴权，仅允许回环监听）"
    return "enabled（来源 %s）" % state["source"]


def _startup_banner(cfg, model_name: str, model_class: str) -> str:
    """构造启动横幅（抽成函数便于用真实 config 断言鉴权/绑定状态）。"""
    return (
        "rkllm %s 服务模型 '%s'，监听 http://%s:%d\n"
        "  endpoints : GET /health /ready /v1/capabilities /metrics, "
        "GET /v1/models, POST /v1/chat/completions /v1/completions /v1/reset\n"
        "  history   : %s   api key : %s   推理进程 : %s\n"
        "  freq lock : %s   档位 : %s   context : %d\n"
        % (VERSION, model_name, cfg.server.bind_host, cfg.server.port,
           "on" if cfg.model.keep_history else "off",
           _auth_banner(cfg),
           "in_process" if cfg.engine.mode == "in_process" else "subprocess",
           "on" if cfg.model.freq_lock else "off",
           model_class, cfg.model.context))


def _cmd_serve(args) -> int:
    import uvicorn

    from .config import ConfigError, load_config
    from .server.app import create_app
    from .server.logsetup import setup_logging

    try:
        cfg = load_config(file=os.environ.get("RKLLM_TOML") or None,
                          cli=_legacy_cli_overrides(args),
                          require_model_path=True,
                          # `.d` 是 TOML 之上的部署层：即使 RKLLM_TOML 显式换了
                          # TOML 文件也照读（与文档里的优先级链一致）。
                          use_dropins=True)
    except ConfigError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2

    setup_logging(cfg)
    from .platform import (apply_freq_lock, cache_max_mb_for, derive_class,
                       max_tokens_limit_for)
    from .platform.modelclass import effective_max_tokens_for
    cfg.values["model.path"] = _resolve_model_path(cfg.model.path,
                                                  cfg.model.search_dirs)
    # token 计数：有 tokenizer.json 就用精确实现，否则靠 prefill_tokens 自校准
    from .agent.tokens import configure as _configure_tokens
    _tok = _configure_tokens(cfg.model.path)
    if _tok["backend"] == "tokenizers":
        sys.stderr.write("[rkllm] token 计数：精确 tokenizer（%s）\n" % _tok["source"])
    # 档位默认只覆盖 4b/7b；仅当用户未显式配置（source=default）时套用，
    # 显式配置永远优先。P1-C2：更大档位（8b/14b/27b/未知）没有内置值，
    # 用 effective_max_tokens_for() 的保守正数回退——绝不能把 0 传给 worker
    # （argparse 会 exit 2，表现为"服务起不来"）。
    model_class = derive_class(cfg.model.path,
                               getattr(cfg.model, "class", "auto"))
    # 0（未配置/显式写 0=按档位）都要换成真实正数：worker 的 argparse 不接受 0。
    if cfg.sources.get("model.max_tokens") == "default" or not cfg.model.max_tokens:
        cfg.values["model.max_tokens"] = effective_max_tokens_for(model_class)
    if cfg.sources.get("model.max_tokens_limit") == "default":
        cfg.values["model.max_tokens_limit"] = max_tokens_limit_for(model_class)
    if cfg.sources.get("cache.max_mb") == "default":
        cfg.values["cache.max_mb"] = cache_max_mb_for(model_class)
    if int(cfg.model.max_tokens or 0) <= 0:
        sys.stderr.write(
            "[rkllm] 配置错误：model.max_tokens 必须为正整数（当前 %r）；"
            "请在 /etc/rkllm/rkllm.toml 或 conffile 中显式配置\n"
            % cfg.model.max_tokens)
        return 2
    if model_class not in ("4b", "7b"):
        src = cfg.sources.get("model.max_tokens", "default")
        sys.stderr.write(
            "[rkllm] 档位 %s 无内置默认（内置仅 4B/7B）：max_tokens 生效值 %d"
            "（来源 %s）；更大档位请按显存/上下文显式配置 model.max_tokens / "
            "max_tokens_limit\n" % (model_class, cfg.model.max_tokens, src))

    use_fake = bool(getattr(args, "fake_engine", False))
    if not use_fake and not os.path.isfile(cfg.model.path):
        sys.stderr.write("[rkllm] 模型文件不存在：%s\n" % cfg.model.path)
        return 2
    _apply_memory_budget(cfg)

    # 启动链从 spawn worker / 锁频开始就持有系统资源；之后的任何一步（create_app
    # 构造、横幅、uvicorn 起来前的配置错误、按键）失败都必须收尾。旧实现把
    # try/finally 只包在 uvicorn.run 外层，create_app/apply_freq_lock 抛错时
    # worker 会变成孤儿并抱着 NPU，锁频也不会恢复（P2.3）。
    supervisor = None
    freq_lock = None
    try:
        supervisor = _build_supervisor(cfg, use_fake=use_fake)
        # 锁频：默认开（板端供电场景），非 root 或 sysfs 不支持时只告警
        freq_lock = apply_freq_lock(bool(cfg.model.freq_lock),
                                    cfg.model.freq_helper)

        app = create_app(cfg, supervisor=supervisor)
        if getattr(args, "system", None):
            app.state.svc.system_prompt = args.system
        app.state.svc.default_sampling = _sampling_from_args(args)

        # 横幅里的 api key 状态来自 auth.auth_status()（解析后的真实状态）；
        # 非回环 + 无 key 的部署已在 load_config 的 fail-closed 校验里拒绝启动，
        # 不再"warning 后照常监听 0.0.0.0"（P1-S1）。
        sys.stderr.write(_startup_banner(cfg, app.state.svc.model_name,
                                         app.state.svc.model_class))

        tls = bool(cfg.server.tls_cert and cfg.server.tls_key)
        uvicorn.run(
            app, host=cfg.server.bind_host, port=cfg.server.port,
            log_level=cfg.logging.level,
            timeout_graceful_shutdown=cfg.server.shutdown_grace_s,
            access_log=cfg.server.access_log,
            # 以下四项此前是死键，现按键表接线
            limit_concurrency=cfg.server.max_connections,
            timeout_keep_alive=cfg.server.read_timeout_s,
            backlog=cfg.server.request_queue_size,
            ssl_certfile=cfg.server.tls_cert if tls else None,
            ssl_keyfile=cfg.server.tls_key if tls else None,
        )
    finally:
        if freq_lock is not None:
            freq_lock.restore()          # 退出恢复原 governor 与频率
        if supervisor is not None:
            supervisor.shutdown()        # 杀掉 worker，绝不留孤儿持 NPU
    return 0
