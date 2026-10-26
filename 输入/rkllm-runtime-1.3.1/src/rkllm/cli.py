"""`rkllm` 命令入口 —— 1.3.1-16 起由新栈承载，接口与 1.3.1-15 保持一致。

设计约束（用户 2026-10-21 明确）：
- 命令名仍是 `/usr/bin/rkllm`，systemd 单元仍是 `rkllm-server.service`，
  conffile 仍是 `/etc/default/rkllm-server`——**运维视角零变化**；
- 实现是新的：FastAPI + 推理子进程 + UDS 帧协议 + 83 键配置；
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

# 兼容导出面：A-4 拆分后，这些名字仍按旧路径 `rkllm.cli._xxx` 暴露给测试与
# 外部脚本。写进 __all__ 既说明意图，也让静态检查知道它们不是"未使用导入"。
__all__ = ["main", "build_parser", "PROGRAM",
           "_apply_memory_budget", "_build_supervisor", "_search_dirs",
           "_resolve_model_path", "_make_engine",
           "_env_default", "_env_flag", "_legacy_cli_overrides",
           "_sampling_from_args",
           "_cmd_serve", "_cmd_list", "_cmd_info", "_cmd_selftest",
           "_cmd_config", "_cmd_run", "_cmd_chat"]


# A-4：实现按职责拆到 cli_env / cli_engine / cli_info / cli_serve。
# 下面 re-export 旧名字是**有意为之**（pyflakes 会报 F401）：`rkllm.cli._xxx`
# 是测试与外部脚本的既有访问路径，拆分不应让它们 AttributeError。
from . import VERSION
from .cli_engine import (_apply_memory_budget, _build_supervisor,  # noqa: F401
                         _make_engine, _resolve_model_path, _search_dirs)
from .cli_env import (_env_default, _env_flag, _legacy_cli_overrides,  # noqa: F401
                      _sampling_from_args)
from .cli_info import _cmd_config, _cmd_info, _cmd_list, _cmd_selftest
from .cli_serve import _cmd_serve

def _cmd_run(args) -> int:
    """一次性推理：纯文本到 stdout（与 1.3.1-15 输出一致）。"""
    engine = _make_engine(args)
    sampling = _sampling_from_args(args) or None
    try:
        for token in engine.stream(" ".join(args.text), sampling=sampling,
                                   max_new_tokens=getattr(args, "max_new_tokens", None)):
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
            for token in engine.stream(line, sampling=sampling,
                                       max_new_tokens=getattr(args, "max_new_tokens", None)):
                sys.stdout.write(token)
                sys.stdout.flush()
            sys.stdout.write("\n")
    except KeyboardInterrupt:
        pass
    finally:
        engine.destroy()
    return 0
def _add_engine_args(parser, env: bool = False, defer: bool = False) -> None:
    """chat/run/serve 共享的推理参数（与 1.3.1-15 同名同义）。

    defer=True（serve）：argv 默认置 None，交由配置层（文件/env）决定；只有
    命令行显式给出才进入 cli 覆盖层，避免默认值击穿 file/env（P1-1）。
    """
    def default(name, kind, fallback):
        if defer:
            return None
        return _env_default(name, kind, fallback) if env else fallback

    parser.add_argument("--context", type=int, default=default("CONTEXT", int, 4096))
    parser.add_argument("--max-new-tokens", type=int,
                        default=default("MAX_NEW_TOKENS", int, 2048))
    parser.add_argument("--threads", type=int, default=default("THREADS", int, 4))
    parser.add_argument("--platform", default=default("PLATFORM", str, "auto"),
                        choices=("auto", "rk3588", "rk3576", "rk3562", "rv1126b"))
    parser.add_argument("--model-dir", action="append",
                        default=(default("MODEL_DIR", str, "") or "").split(":")
                        if (env and not defer and default("MODEL_DIR", str, "")) else [])
    parser.add_argument("--lib", default=None)
    parser.add_argument("--no-freq-lock", action="store_true",
                        default=None if defer else
                        (_env_flag("NO_FREQ_LOCK") if env else False))
    parser.add_argument("--quiet", "-q", action="store_true",
                        default=None if defer else
                        (_env_flag("QUIET") if env else False))
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
    parser.add_argument("--version", action="version", version="rkllm " + VERSION)
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="启动 OpenAI 兼容 HTTP 服务")
    serve.add_argument("model", help="模型名或 .rkllm 路径")
    # P1-1：serve 的 argv 默认全部置 None，交由配置层（文件/env）决定；仅显式
    # 给出才进入 cli 覆盖层，避免 argparse 默认值击穿 file/env 两级。
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--api-key", dest="api_key",
                       default=_env_default("API_KEY", str, "") or None)
    serve.add_argument("--max-queue", dest="max_queue", type=int, default=None)
    serve.add_argument("--queue-timeout", dest="queue_timeout", type=float,
                       default=None)
    serve.add_argument("--system", default=_env_default("SYSTEM", str, "") or None)
    serve.add_argument("--no-history", action="store_true", default=None)
    _add_engine_args(serve, defer=True)
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
