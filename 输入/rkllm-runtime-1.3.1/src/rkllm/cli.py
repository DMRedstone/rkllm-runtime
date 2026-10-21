"""新 CLI 入口：rkllm serve（装配 80 键配置 → FastAPI → uvicorn）。

与旧 cli/rkllm.py 并存；S6 发布时由打包链切换入口。
用法：
  python -m rkllm.cli serve --model /path/to.rkllm [--engine fake] [--port 8080]
  python -m rkllm.cli config show [--json]
  python -m rkllm.cli config validate
  python -m rkllm.cli config docs
"""

from __future__ import annotations

import argparse
import sys


def _load_cfg(args):
    from .config import load_config, ConfigError
    cli = {}
    if getattr(args, "model", None):
        cli["model.path"] = args.model
    if getattr(args, "port", None):
        cli["server.port"] = args.port
    if getattr(args, "bind", None):
        cli["server.bind_host"] = args.bind
    try:
        return load_config(file=getattr(args, "config", None), cli=cli,
                           require_model_path=True,
                           env={"RKLLM_ENGINE_MODE": "subprocess"}
                           if getattr(args, "engine", None) else None)
    except ConfigError as exc:
        sys.stderr.write(str(exc) + "\n")
        raise SystemExit(2)


def _cmd_serve(args) -> int:
    import uvicorn
    from .server.app import create_app

    cfg = _load_cfg(args)
    use_fake = getattr(args, "engine", None) == "fake"
    supervisor = None
    if not getattr(args, "no_worker", False):
        from .server.supervisor import Supervisor
        worker_args = ["fake"] if use_fake else ["real"]
        if use_fake:
            worker_args += ["--model", cfg.model.path or "fake",
                            "--context", str(cfg.model.context)]
        else:
            worker_args += ["--model", cfg.model.path,
                            "--platform", cfg.model.platform,
                            "--context", str(cfg.model.context),
                            "--max-tokens", str(cfg.model.max_tokens or 2048),
                            "--threads", str(cfg.model.threads)]
            if not cfg.model.keep_history:
                worker_args.append("--no-history")
        supervisor = Supervisor(
            worker_args,
            load_timeout_s=cfg.engine.load_timeout_s,
            restart_backoff_s=cfg.engine.restart_backoff_s,
            restart_backoff_max_s=cfg.engine.restart_backoff_max_s,
            restart_max_per_hour=cfg.engine.restart_max_per_hour)
        supervisor.start()

    app = create_app(cfg, supervisor=supervisor)
    try:
        uvicorn.run(app, host=cfg.server.bind_host, port=cfg.server.port,
                    log_level=cfg.logging.level,
                    timeout_graceful_shutdown=cfg.server.shutdown_grace_s,
                    access_log=cfg.server.access_log)
    finally:
        if supervisor:
            supervisor.shutdown()
    return 0


def _cmd_config(args) -> int:
    from .config import load_config, config_docs, config_show, ConfigError
    if args.action == "docs":
        print(config_docs())
        return 0
    cfg = load_config(file=args.config, require_model_path=False)
    if args.action == "show":
        print(config_show(cfg, as_json=args.json))
        return 0
    return 0     # validate：load_config 抛错即非零退出


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rkllm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="启动 OpenAI 兼容服务")
    serve.add_argument("--model", help=".rkllm 模型路径")
    serve.add_argument("--config", help="TOML 配置文件")
    serve.add_argument("--engine", choices=("real", "fake"), default="real")
    serve.add_argument("--port", type=int)
    serve.add_argument("--bind")
    serve.add_argument("--no-worker", action="store_true",
                       help="只起运维端点（调试用）")
    serve.set_defaults(func=_cmd_serve)

    conf = sub.add_parser("config", help="配置表命令")
    conf.add_argument("action", choices=("show", "validate", "docs"))
    conf.add_argument("--config")
    conf.add_argument("--json", action="store_true")
    conf.set_defaults(func=_cmd_config)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
