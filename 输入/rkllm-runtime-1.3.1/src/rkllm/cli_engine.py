"""引擎装配（A-4：从 cli.py 拆出）。

负责「模型路径解析 → 内存预算 → 构造 Supervisor/Engine」这条链路，
serve 与 run/chat 共用。
"""

from __future__ import annotations

import os
import sys

__all__ = ["_resolve_model_path", "_apply_memory_budget", "_build_supervisor",
           "_make_engine", "_search_dirs"]


def _resolve_model_path(path: str, search_dirs) -> str:
    """裸模型名按 model.search_dirs 查找（含自动补 .rkllm 后缀）。"""
    if not path:
        return path
    if os.path.sep in path or (os.path.altsep and os.path.altsep in path):
        return path
    names = [path] if path.endswith(".rkllm") else [path + ".rkllm", path]
    for directory in search_dirs or ():
        for name in names:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
    return path
def _apply_memory_budget(cfg) -> None:
    """启动前内存预算：不足则逐级降 context，仍不足拒绝启动（N10/F12）。"""
    from .platform import MemoryBudgetError, check_budget, derive_class
    if not cfg.engine.memory_check:
        return
    model_class = derive_class(cfg.model.path, getattr(cfg.model, "class", "auto"))
    try:
        plan = check_budget(model_class, cfg.model.context,
                            model_path=cfg.model.path,
                            reserve_mb=cfg.engine.memory_reserve_mb,
                            min_context=cfg.model.min_context)
    except MemoryBudgetError as exc:
        sys.stderr.write("[rkllm] %s\n" % exc)
        raise SystemExit(3)
    sys.stderr.write("[rkllm] %s（档位 %s）\n" % (plan.describe(), model_class))
    if plan.downgraded:
        sys.stderr.write("[rkllm] 警告：内存不足，context 已从 %d 降到 %d\n"
                         % (cfg.model.context, plan.context))
        cfg.values["model.context"] = plan.context
def _build_supervisor(cfg, *, use_fake: bool):
    from .platform import derive_class, max_tokens_for, resolve_platform
    model_class = derive_class(cfg.model.path, getattr(cfg.model, "class", "auto"))
    max_tokens = cfg.model.max_tokens or max_tokens_for(model_class)
    # model.platform=auto 必须解析成实际 SoC：worker 用平台名选 CPU 大小核
    # （engine.py 的 BIG_CORE_PLATFORMS 判断），"auto" 会落到小核、慢 3-4 倍。
    platform = resolve_platform(cfg.model.platform)
    if cfg.model.platform == "auto":
        sys.stderr.write("[rkllm] 平台检测：auto -> %s\n" % platform)

    if cfg.engine.mode == "in_process":
        from .server.supervisor import LocalSupervisor
        if use_fake:
            from .worker.fake import FakeEngine
            engine = FakeEngine(model_name=cfg.model.path or "fake",
                                context_len=cfg.model.context)
        else:
            from .worker.engine import RealEngine
            from .worker.promptcache import PromptCache
            cache = None
            if cfg.cache.enabled:
                cache = PromptCache(cfg.cache.dir, max_mb=cfg.cache.max_mb,
                                    ttl_s=cfg.cache.ttl_s)
            engine = RealEngine(
                model_path=cfg.model.path, platform=platform,
                context_len=cfg.model.context, max_new_tokens=max_tokens,
                keep_history=cfg.model.keep_history, threads=cfg.model.threads,
                prompt_cache=cache)
        supervisor = LocalSupervisor(engine,
                                     load_timeout_s=cfg.engine.load_timeout_s)
        supervisor.start()
        return supervisor

    from .server.supervisor import Supervisor
    worker_args = ["fake"] if use_fake else ["real"]
    if use_fake:
        worker_args += ["--model", cfg.model.path or "fake",
                        "--context", str(cfg.model.context)]
    else:
        worker_args += ["--model", cfg.model.path,
                        "--platform", platform,
                        "--context", str(cfg.model.context),
                        "--max-tokens", str(max_tokens),
                        "--threads", str(cfg.model.threads)]
        if not cfg.model.keep_history:
            worker_args.append("--no-history")
        # 前缀缓存：目录/容量/TTL 传到 worker（关闭时不传，等价 cache.enabled=false）
        if cfg.cache.enabled:
            worker_args += ["--cache-dir", cfg.cache.dir,
                            "--cache-max-mb", str(cfg.cache.max_mb),
                            "--cache-ttl-s", str(cfg.cache.ttl_s)]
    supervisor = Supervisor(
        worker_args,
        load_timeout_s=cfg.engine.load_timeout_s,
        restart_backoff_s=cfg.engine.restart_backoff_s,
        restart_backoff_max_s=cfg.engine.restart_backoff_max_s,
        restart_max_per_hour=cfg.engine.restart_max_per_hour,
        health_interval_s=cfg.engine.health_interval_s)
    supervisor.start()
    return supervisor
def _make_engine(args):
    from .platform import resolve_platform
    from .worker.engine import RealEngine
    return RealEngine(
        model_path=_resolve_model_path(args.model, args.model_dir),
        platform=resolve_platform(getattr(args, "platform", None) or "auto"),
        context_len=args.context,
        max_new_tokens=getattr(args, "max_new_tokens", None) or 2048,
        keep_history=not args.no_history,
        threads=args.threads,
        system_prompt=getattr(args, "system", None),
        lib_path=getattr(args, "lib", None))
def _search_dirs(extra) -> list[str]:
    dirs: list[str] = []
    for item in list(extra or []) + (os.environ.get("RKLLM_MODEL_DIR") or "").split(":"):
        if item and item not in dirs:
            dirs.append(item)
    for item in ("/var/lib/rkllm/models", os.path.expanduser("~/models")):
        if os.path.isdir(item) and item not in dirs:
            dirs.append(item)
    return dirs
