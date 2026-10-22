"""`rkllm` 命令入口 —— 1.3.1-16 起由新栈承载，接口与 1.3.1-15 保持一致。

设计约束（用户 2026-10-21 明确）：
- 命令名仍是 `/usr/bin/rkllm`，systemd 单元仍是 `rkllm-server.service`，
  conffile 仍是 `/etc/default/rkllm-server`——**运维视角零变化**；
- 实现是新的：FastAPI + 推理子进程 + UDS 帧协议 + 80 键配置；
- 旧 CLI 的每个子命令与每个 serve 参数都保留（包括从 `RKLLM_*` 环境变量取
  默认值的语义，这是 conffile 驱动服务的方式）。

子命令：
  rkllm serve <model> [--host H] [--port P] ...   启动 OpenAI 兼容服务
  rkllm run <model> <text...>                     一次性推理
  rkllm chat <model>                              交互式对话
  rkllm list                                      列出可发现的模型
  rkllm info                                      环境与驱动报告
  rkllm selftest                                  安装自检
  rkllm config show|validate|docs|init            配置表
"""
from __future__ import annotations

import argparse
import os
import sys

PROGRAM = "rkllm"


# ---------------------------------------------------------------------------
# 环境变量默认值（旧 conffile 的驱动方式，逐字保留）
# ---------------------------------------------------------------------------


def _env_default(name: str, kind, fallback):
    """`--xxx` 的默认值：先看 `$RKLLM_<NAME>`，再看内置默认。

    这是 `/etc/default/rkllm-server` 能驱动服务的根本机制（systemd 用
    EnvironmentFile= 注入）。值不合法时报错并回退——配置文件里的笔误不该让
    服务以 argparse 的 status=2 静默失败。
    """
    raw = os.environ.get("RKLLM_" + name)
    if raw is None or not raw.strip():
        return fallback
    try:
        return kind(raw)
    except (TypeError, ValueError):
        sys.stderr.write("[rkllm] 忽略 RKLLM_%s=%r（不是合法 %s），用 %r\n"
                         % (name, raw, kind.__name__, fallback))
        return fallback


def _env_flag(name: str, fallback: bool = False) -> bool:
    raw = os.environ.get("RKLLM_" + name)
    if raw is None or not raw.strip():
        return fallback
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# serve：旧参数 → 80 键配置
# ---------------------------------------------------------------------------


def _legacy_cli_overrides(args) -> dict:
    """把旧 serve 命令行映射为配置键（命令行最高优先级）。"""
    overrides: dict = {
        "model.path": args.model,
        "server.bind_host": args.host,
        "server.port": args.port,
        "model.context": args.context,
        "model.threads": args.threads,
        "queue.max_wait": args.max_queue,
        "queue.wait_timeout_s": int(args.queue_timeout),
        "model.keep_history": not args.no_history,
        "model.freq_lock": not args.no_freq_lock,
        "logging.level": "warning" if args.quiet else "info",
    }
    if getattr(args, "max_new_tokens", None):
        overrides["model.max_tokens"] = args.max_new_tokens
    if getattr(args, "api_key", None):
        overrides["server.api_key"] = args.api_key
    if getattr(args, "platform", None):
        overrides["model.platform"] = args.platform
    if getattr(args, "model_dir", None):
        overrides["model.search_dirs"] = list(args.model_dir)
    return overrides


SAMPLING_KEYS = ("top_k", "top_p", "temperature", "repeat_penalty",
                 "presence_penalty", "frequency_penalty")


def _sampling_from_args(args) -> dict:
    """把 --top-k/--top-p/--temperature/--repeat-penalty 变成进程级采样默认值。

    与 1.3.1-15 的语义一致：命令行（或 RKLLM_* 环境变量）给出的是**默认值**，
    请求体里的同名字段仍然优先（OpenAI 客户端习惯）。
    """
    return {key: getattr(args, key) for key in SAMPLING_KEYS
            if getattr(args, key, None) is not None}


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
    from .platform import apply_freq_lock, derive_class, max_tokens_for
    cfg.values["model.path"] = _resolve_model_path(cfg.model.path,
                                                  cfg.model.search_dirs)
    if not cfg.model.max_tokens:
        cfg.values["model.max_tokens"] = max_tokens_for(
            derive_class(cfg.model.path, getattr(cfg.model, "class", "auto")))

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
        "rkllm 1.3.1-16 服务模型 '%s'，监听 http://%s:%d\n"
        "  endpoints : GET /health /ready /v1/capabilities /metrics, "
        "GET /v1/models, POST /v1/chat/completions /v1/completions /v1/reset\n"
        "  history   : %s   api key : %s   推理进程 : %s\n"
        "  freq lock : %s   档位 : %s   context : %d\n"
        % (app.state.svc.model_name, cfg.server.bind_host, cfg.server.port,
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
            # 以下四项此前是死键，现按 80 键表接线
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


# ---------------------------------------------------------------------------
# run / chat：单次与交互推理（直接走引擎，不经 HTTP）
# ---------------------------------------------------------------------------


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


def _cmd_run(args) -> int:
    """一次性推理：纯文本到 stdout（与 1.3.1-15 输出一致）。"""
    engine = _make_engine(args)
    sampling = _sampling_from_args(args) or None
    try:
        for token in engine.stream(" ".join(args.text), sampling=sampling):
            sys.stdout.write(token)
            sys.stdout.flush()
        sys.stdout.write("\n")
    finally:
        engine.destroy()
    return 0


def _cmd_chat(args) -> int:
    """交互式对话（Ctrl-D 退出）。"""
    engine = _make_engine(args)
    sampling = _sampling_from_args(args) or None
    sys.stderr.write("rkllm chat: 输入问题，Ctrl-D 退出\n")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            for token in engine.stream(line, sampling=sampling):
                sys.stdout.write(token)
                sys.stdout.flush()
            sys.stdout.write("\n")
    except KeyboardInterrupt:
        pass
    finally:
        engine.destroy()
    return 0


# ---------------------------------------------------------------------------
# list / info / selftest
# ---------------------------------------------------------------------------


def _search_dirs(extra) -> list[str]:
    dirs: list[str] = []
    for item in list(extra or []) + (os.environ.get("RKLLM_MODEL_DIR") or "").split(":"):
        if item and item not in dirs:
            dirs.append(item)
    for item in ("/var/lib/rkllm/models", os.path.expanduser("~/models")):
        if os.path.isdir(item) and item not in dirs:
            dirs.append(item)
    return dirs


def _cmd_list(args) -> int:
    import json
    found = []
    for directory in _search_dirs(getattr(args, "model_dir", None)):
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for name in names:
            if not name.endswith(".rkllm"):
                continue
            path = os.path.join(directory, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            found.append({"name": name, "path": path,
                          "size_mb": round(size / 1048576, 1)})
    if getattr(args, "json", False):
        print(json.dumps(found, ensure_ascii=False, indent=2))
        return 0
    if not found:
        sys.stderr.write("未找到 .rkllm 模型；用 --model-dir 指定目录\n")
        return 1
    for item in found:
        print("%-60s %8.1f MB  %s" % (item["name"], item["size_mb"], item["path"]))
    return 0


def _cmd_info(args) -> int:
    import json
    import platform as py_platform
    from .platform import available_mb, detect_platform, npu_status
    npu_available, npu_detail = npu_status()
    info = {
        "rkllm": "1.3.1-16",
        "python": sys.version.split()[0],
        "platform_detected": detect_platform(),
        "machine": py_platform.machine(),
        "mem_available_mb": (round(available_mb(), 0)
                             if available_mb() is not None else None),
        "npu_available": npu_available,
        "npu_detail": npu_detail,
    }
    try:
        from .abi import load_library
        _lib, lib_path, fingerprint = load_library(getattr(args, "lib", None))
        info.update({"library": lib_path,
                     "library_sha256": fingerprint.get("sha256"),
                     "library_size": fingerprint.get("size")})
    except Exception as exc:                      # noqa: BLE001
        info["library_error"] = str(exc)[:200]
    info["model_dirs"] = _search_dirs(getattr(args, "model_dir", None))
    if getattr(args, "json", False):
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    for key, value in info.items():
        print("%-18s %s" % (key, value))
    return 0


def _cmd_selftest(args) -> int:
    """安装自检：解释器 / 依赖 / 库 / 模型 / 配置表五步。"""
    checks: list[tuple[str, bool, str]] = [
        ("python>=3.13", sys.version_info[:2] >= (3, 13),
         sys.version.split()[0])]
    try:
        import fastapi
        import pydantic
        import uvicorn
        checks.append(("fastapi/uvicorn/pydantic", True,
                       "%s / %s / %s" % (fastapi.__version__,
                                         uvicorn.__version__,
                                         pydantic.VERSION)))
    except ImportError as exc:
        checks.append(("fastapi/uvicorn/pydantic", False, str(exc)))
    try:
        from .abi import load_library
        _lib, lib_path, fingerprint = load_library(getattr(args, "lib", None))
        checks.append(("librkllmrt", True,
                       "%s sha256=%s" % (lib_path,
                                         (fingerprint.get("sha256") or "?")[:12])))
    except Exception as exc:                      # noqa: BLE001
        checks.append(("librkllmrt", False, str(exc)[:120]))
    models = []
    for directory in _search_dirs(getattr(args, "model_dir", None)):
        try:
            models += [n for n in os.listdir(directory) if n.endswith(".rkllm")]
        except OSError:
            pass
    checks.append(("model files", bool(models), "%d 个" % len(models)))
    from .config import SCHEMA
    checks.append(("config schema", len(SCHEMA) == 80, "%d 键" % len(SCHEMA)))

    failed = 0
    for name, ok, detail in checks:
        print("%s  %-26s %s" % ("PASS" if ok else "FAIL", name, detail))
        failed += 0 if ok else 1
    print("\n%d/%d 通过" % (len(checks) - failed, len(checks)))
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def _cmd_config(args) -> int:
    from .config import load_config, config_docs, config_init, config_show
    if args.action == "docs":
        print(config_docs())
        return 0
    if args.action == "init":
        target = "client" if args.client else "board"
        path = args.path or ("mcp.toml" if args.client
                             else "/etc/rkllm/rkllm.toml")
        path = os.path.expanduser(path)
        if os.path.exists(path) and not args.force:
            sys.stderr.write("[rkllm] 已存在，未覆盖：%s（用 --force 覆盖）\n" % path)
            return 1
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(config_init(target))
        print("已写入 %s（%s 模板）" % (path, target))
        return 0
    cfg = load_config(file=args.config, require_model_path=False)
    if args.action == "show":
        print(config_show(cfg, as_json=args.json))
        return 0
    return 0     # validate：load_config 抛错即非零退出


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _add_engine_args(parser, env: bool = False) -> None:
    """chat/run/serve 共享的推理参数（与 1.3.1-15 同名同义）。"""
    def default(name, kind, fallback):
        return _env_default(name, kind, fallback) if env else fallback

    parser.add_argument("--context", type=int, default=default("CONTEXT", int, 4096))
    parser.add_argument("--max-new-tokens", type=int,
                        default=default("MAX_NEW_TOKENS", int, 2048))
    parser.add_argument("--threads", type=int, default=default("THREADS", int, 4))
    parser.add_argument("--platform", default=default("PLATFORM", str, "auto"),
                        choices=("auto", "rk3588", "rk3576", "rk3562", "rv1126b"))
    parser.add_argument("--model-dir", action="append",
                        default=(default("MODEL_DIR", str, "") or "").split(":")
                        if env and default("MODEL_DIR", str, "") else [])
    parser.add_argument("--lib", default=None)
    parser.add_argument("--no-freq-lock", action="store_true",
                        default=_env_flag("NO_FREQ_LOCK") if env else False)
    parser.add_argument("--quiet", "-q", action="store_true",
                        default=_env_flag("QUIET") if env else False)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fake-engine", action="store_true")


def _add_sampling_args(parser, env: bool = False) -> None:
    """采样参数（与 1.3.1-15 同名同默认值：top_k=1 / top_p=0.9 / temp=0.8 / rp=1.1）。"""
    def default(name, kind, fallback):
        return _env_default(name, kind, fallback) if env else fallback

    parser.add_argument("--top-k", dest="top_k", type=int,
                        default=default("TOP_K", int, None))
    parser.add_argument("--top-p", dest="top_p", type=float,
                        default=default("TOP_P", float, None))
    parser.add_argument("--temperature", type=float,
                        default=default("TEMPERATURE", float, None))
    parser.add_argument("--repeat-penalty", dest="repeat_penalty", type=float,
                        default=default("REPEAT_PENALTY", float, None))
    parser.add_argument("--presence-penalty", dest="presence_penalty",
                        type=float, default=default("PRESENCE_PENALTY", float, None))
    parser.add_argument("--frequency-penalty", dest="frequency_penalty",
                        type=float, default=default("FREQUENCY_PENALTY", float, None))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM,
                                     description="RKLLM runtime + OpenAI 兼容服务")
    parser.add_argument("--version", action="version", version="rkllm 1.3.1-16")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="启动 OpenAI 兼容 HTTP 服务")
    serve.add_argument("model", help="模型名或 .rkllm 路径")
    serve.add_argument("--host", default=_env_default("HOST", str, "127.0.0.1"))
    serve.add_argument("--port", type=int,
                       default=_env_default("PORT", int, 8080))
    serve.add_argument("--api-key", dest="api_key",
                       default=_env_default("API_KEY", str, "") or None)
    serve.add_argument("--max-queue", dest="max_queue", type=int,
                       default=_env_default("MAX_QUEUE", int, 32))
    serve.add_argument("--queue-timeout", dest="queue_timeout", type=float,
                       default=_env_default("QUEUE_TIMEOUT", float, 600.0))
    serve.add_argument("--system", default=_env_default("SYSTEM", str, "") or None)
    serve.add_argument("--no-history", action="store_true",
                       default=_env_flag("NO_HISTORY"))
    _add_engine_args(serve, env=True)
    _add_sampling_args(serve, env=True)
    serve.set_defaults(func=_cmd_serve)

    run = sub.add_parser("run", help="一次性推理")
    run.add_argument("model")
    run.add_argument("text", nargs="+")
    run.add_argument("--system", default=None)
    run.add_argument("--no-history", action="store_true")
    _add_engine_args(run)
    _add_sampling_args(run)
    run.set_defaults(func=_cmd_run)

    chat = sub.add_parser("chat", help="交互式对话")
    chat.add_argument("model")
    chat.add_argument("--system", default=None)
    chat.add_argument("--no-history", action="store_true")
    _add_engine_args(chat)
    _add_sampling_args(chat)
    chat.set_defaults(func=_cmd_chat)

    listing = sub.add_parser("list", help="列出可发现的模型")
    listing.add_argument("--model-dir", action="append", default=[])
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=_cmd_list)

    info = sub.add_parser("info", help="环境与驱动报告")
    info.add_argument("--model-dir", action="append", default=[])
    info.add_argument("--lib", default=None)
    info.add_argument("--json", action="store_true")
    info.set_defaults(func=_cmd_info)

    selftest = sub.add_parser("selftest", help="安装自检")
    selftest.add_argument("--model-dir", action="append", default=[])
    selftest.add_argument("--lib", default=None)
    selftest.set_defaults(func=_cmd_selftest)

    conf = sub.add_parser("config", help="配置表命令")
    conf.add_argument("action", choices=("show", "validate", "docs", "init"))
    conf.add_argument("--config")
    conf.add_argument("--json", action="store_true")
    conf.add_argument("--board", action="store_true")
    conf.add_argument("--client", action="store_true")
    conf.add_argument("--path")
    conf.add_argument("--force", action="store_true")
    conf.set_defaults(func=_cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # systemd 会把未设置的变量展开成空参数；argparse 会当成非法位置参数。
    payload = [item for item in argv if item != ""]
    parser = build_parser()
    args = parser.parse_args(payload)
    if not getattr(args, "command", None):
        parser.print_help(sys.stderr)
        sys.stderr.write("\n%s: 缺少子命令；见 --help\n" % PROGRAM)
        return 2
    try:
        return args.func(args)
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130
    except SystemExit:
        raise
    except Exception as exc:                      # noqa: BLE001
        # 运维视角：一行可读错误即可；堆栈默认不打（避免把内部路径/参数刷屏），
        # 需要排障时设 RKLLM_DEBUG=1。
        sys.stderr.write("%s: 失败：%s\n" % (PROGRAM, exc))
        if os.environ.get("RKLLM_DEBUG"):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
