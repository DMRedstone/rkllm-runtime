"""CLI 的环境变量与命令行覆盖层（A-4：从 cli.py 拆出）。

`/etc/default/rkllm-server` 经 systemd `EnvironmentFile=` 注入 `RKLLM_*`，
这里是那套约定的唯一实现：`--xxx` 的默认值先看环境变量、再看内置值；
serve 的旧参数只在**显式给出**时才进入配置覆盖层（P1-1）。
"""

from __future__ import annotations

import os
import sys

__all__ = ["SAMPLING_KEYS", "_env_default", "_env_flag",
           "_legacy_cli_overrides", "_sampling_from_args"]


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
def _legacy_cli_overrides(args) -> dict:
    """把旧 serve 命令行映射为配置键（仅显式给出的参数才覆盖，P1-1）。

    serve 的 argv 默认值全部为 None：未出现即不进入覆盖层，让配置文件与
    RKLLM_* 环境变量（由 config.LEGACY_ENV_MAP 读取）真正生效。
    """
    overrides: dict = {"model.path": args.model}

    def put(key: str, val) -> None:
        if val is not None:
            overrides[key] = val

    put("server.bind_host", getattr(args, "host", None))
    put("server.port", getattr(args, "port", None))
    put("model.context", getattr(args, "context", None))
    put("model.threads", getattr(args, "threads", None))
    put("queue.max_wait", getattr(args, "max_queue", None))
    qt = getattr(args, "queue_timeout", None)
    if qt is not None:
        overrides["queue.wait_timeout_s"] = int(qt)
    nh = getattr(args, "no_history", None)
    if nh is not None:
        overrides["model.keep_history"] = not nh
    nf = getattr(args, "no_freq_lock", None)
    if nf is not None:
        overrides["model.freq_lock"] = not nf
    q = getattr(args, "quiet", None)
    if q is not None:
        overrides["logging.level"] = "warning" if q else "info"
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
