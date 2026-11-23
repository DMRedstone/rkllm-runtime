"""引擎装配（A-4：从 cli.py 拆出）。

负责「模型路径解析 → 内存预算 → 构造 Supervisor/Engine」这条链路，
serve 与 run/chat 共用。
"""

from __future__ import annotations

import os
import sys

__all__ = ["_resolve_model_path", "_apply_memory_budget", "_build_supervisor",
           "_make_engine", "_search_dirs", "_heartbeat_grace_s"]

# P2.2：prefill 心跳宽限期不新增 SCHEMA 键（保持 84/85 口径），来源：
# 运行时环境变量 > 由现有 heartbeat_timeout_s 派生（默认 20s → 120s）。
HEARTBEAT_GRACE_ENV = "RKLLM_HEARTBEAT_GRACE_S"
_HEARTBEAT_GRACE_FACTOR = 2.0
_HEARTBEAT_GRACE_MIN_S = 120.0


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
def _heartbeat_grace_s(cfg) -> float:
    """worker prefill 心跳宽限期（秒）。

    P2.2：此前 supervisor 从不传 `--heartbeat-grace-s`，worker 用硬编码 120s
    ——27B/长上下文 prefill 超过 120s 就会被心跳看门狗按"卡死"误杀。为保持
    84/85 键口径**不新增 SCHEMA 键**，使用两层来源：

    1. 运行时环境变量 `RKLLM_HEARTBEAT_GRACE_S`（显式覆盖，改环境即可，无需
       改配置键/文档键表）；
    2. 否则由现有 `engine.heartbeat_timeout_s` 派生：`max(120, 2×timeout)`。
       默认 timeout=20s → 120s（出厂行为不变）；调大 timeout 的运维通常也在
       容忍更长的无声 prefill，宽限期按 2 倍跟随。
    """
    raw = os.environ.get(HEARTBEAT_GRACE_ENV)
    if raw is not None and raw.strip():
        try:
            value = float(raw)
        except ValueError:
            sys.stderr.write("[rkllm] 忽略非法 %s=%r（需秒数，回退派生值）\n"
                             % (HEARTBEAT_GRACE_ENV, raw))
        else:
            if value >= 0:
                return value
            sys.stderr.write("[rkllm] 忽略非法 %s=%r（不能为负，回退派生值）\n"
                             % (HEARTBEAT_GRACE_ENV, raw))
    return max(_HEARTBEAT_GRACE_MIN_S,
               _HEARTBEAT_GRACE_FACTOR * float(cfg.engine.heartbeat_timeout_s))


def _build_supervisor(cfg, *, use_fake: bool):
    from .platform import derive_class, resolve_platform
    # P1-C2：无内置档位（8b/14b/27b/未知）不能把 0 传给 worker 的 argparse，
    # 否则 worker exit 2，表现为"服务起不来"。effective_max_tokens_for 恒 > 0，
    # 显式配置的 model.max_tokens 仍然优先（0 视为"未配置"）。
    from .platform.modelclass import effective_max_tokens_for
    model_class = derive_class(cfg.model.path, getattr(cfg.model, "class", "auto"))
    max_tokens = int(cfg.model.max_tokens or effective_max_tokens_for(model_class))
    if max_tokens <= 0:                     # 防御：任何路径都不许把 0 下发给 worker
        raise SystemExit("[rkllm] model.max_tokens 必须为正整数（当前 %r）"
                         % max_tokens)
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
                                    ttl_s=cfg.cache.ttl_s,
                                    similarity=cfg.cache.similarity)
            engine = RealEngine(
                model_path=cfg.model.path, platform=platform,
                context_len=cfg.model.context, max_new_tokens=max_tokens,
                keep_history=cfg.model.keep_history, threads=cfg.model.threads,
                prompt_cache=cache,
                lora_paths=list(getattr(cfg.model, "lora_paths", ()) or ()))
        supervisor = LocalSupervisor(engine,
                                     load_timeout_s=cfg.engine.load_timeout_s)
        _start_supervisor(supervisor)
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
                            "--cache-ttl-s", str(cfg.cache.ttl_s),
                            "--cache-similarity", str(cfg.cache.similarity)]
        # LoRA：adapter 在 worker 里 init 之后加载（handle 级）
        for lora in getattr(cfg.model, "lora_paths", ()) or ():
            worker_args += ["--lora-path", lora]
    worker_args += ["--heartbeat-interval",
                    str(cfg.engine.heartbeat_interval_s),
                    # P2.2：prefill 宽限期由 server 显式下发（env 覆盖或按
                    # heartbeat_timeout_s 派生），worker 不再依赖 argparse
                    # 里的硬编码默认值。
                    "--heartbeat-grace-s",
                    str(_heartbeat_grace_s(cfg))]
    supervisor = Supervisor(
        worker_args,
        load_timeout_s=cfg.engine.load_timeout_s,
        restart_backoff_s=cfg.engine.restart_backoff_s,
        restart_backoff_max_s=cfg.engine.restart_backoff_max_s,
        restart_max_per_hour=cfg.engine.restart_max_per_hour,
        health_interval_s=cfg.engine.health_interval_s,
        heartbeat_timeout_s=cfg.engine.heartbeat_timeout_s)
    _start_supervisor(supervisor)
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


def _start_supervisor(supervisor) -> None:
    """启动 supervisor；失败时先 shutdown 再抛出（P2.3 启动失败清理）。

    `start()` 可能已经 spawn 了 worker 子进程，然后在"等待就绪"阶段超时/报错：
    直接向上抛会把子进程留给 init 收养，它仍抱着 NPU 句柄与显存，下一次启动
    直接失败（且没有任何地方能看到这个孤儿）。shutdown() 是 supervisor 的既有
    清理路径，这里只保证异常路径也走到它。
    """
    try:
        supervisor.start()
    except BaseException:
        supervisor.shutdown()
        raise
