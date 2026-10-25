"""82 键 TOML 配置表（MCP 接入设计 §7 / 总体设计 §7.1，需求 N11/F13）。

设计要点：
- 所有影响行为的参数必须进表；代码中不得出现魔法数（协议常量与安全下限除外）。
- 优先级：内置默认 < 配置文件 < 环境变量（RKLLM_*）< 命令行。
- 启动校验：类型、范围、互斥项；错误一次性列出并拒绝启动（不再静默回退）。
- 纯 stdlib（tomllib 为 3.13 内置）：客户端 wheel 零第三方依赖的前提。
- 键计数由 tests/test_config.py 守住：server 13 + model 11 + engine 8 + queue 5
  + openai 8 + cache 4 + mcp 24 + logging 5 + metrics 2 = 80。

环境变量映射规则：RKLLM_<SECTION>_<NAME>（大写，如 RKLLM_SERVER_PORT）。
旧 conffile（/etc/default/rkllm-server）经 LEGACY_ENV_MAP 迁移。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Key", "SCHEMA", "Config", "ConfigError", "load_config",
           "config_docs", "config_init", "config_show"]

# ---------------------------------------------------------------------------
# schema 定义
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Key:
    """单个配置键的契约：类型、默认值、约束、说明。"""

    section: str
    name: str
    type: type                  # str / int / float / bool / list
    default: Any
    min_v: Any = None           # 数值下限（含）；list 时为长度下限
    max_v: Any = None           # 数值上限（含）；list 时为长度上限
    choices: tuple = ()         # 枚举取值
    desc: str = ""
    implemented: bool = True    # False = schema 存在但运行时尚未接线（G5 诚实）

    @property
    def full(self) -> str:
        return f"{self.section}.{self.name}"

    @property
    def env(self) -> str:
        return f"RKLLM_{self.section}_{self.name}".upper()


def _k(section: str, name: str, type_: type, default: Any, *,
       min_v: Any = None, max_v: Any = None, choices: tuple = (),
       desc: str = "", implemented: bool = True) -> Key:
    return Key(section, name, type_, default,
               min_v=min_v, max_v=max_v, choices=choices, desc=desc,
               implemented=implemented)


# 82 键全量 schema。分组顺序与《MCP 接入与配置设计》§7.2 一致。
SCHEMA: tuple[Key, ...] = (
    # -- server（13）--------------------------------------------------------
    _k("server", "bind_host", str, "127.0.0.1", desc="监听地址；默认只本机，对外需显式改"),
    _k("server", "port", int, 8080, min_v=1, max_v=65535, desc="监听端口"),
    _k("server", "api_key", str, "", desc="Bearer 鉴权；空=不鉴权"),
    _k("server", "api_key_file", str, "", desc="key 文件（0600），优先于 api_key"),
    _k("server", "cors_origins", list, (), desc="CORS 白名单；空=不发 CORS 头"),
    _k("server", "max_body_bytes", int, 8388608, min_v=1024, max_v=67108864,
       desc="请求体上限（默认 8 MiB）"),
    _k("server", "read_timeout_s", int, 30, min_v=1, max_v=600, desc="socket 读超时"),
    _k("server", "request_queue_size", int, 128, min_v=1, max_v=1024,
       desc="内核 accept 队列"),
    _k("server", "max_connections", int, 256, min_v=1, max_v=4096, desc="并发连接上限"),
    _k("server", "shutdown_grace_s", int, 30, min_v=0, max_v=300,
       desc="SIGTERM 后在飞请求排空时间"),
    _k("server", "tls_cert", str, "", desc="TLS 证书；与 tls_key 成对给出才启用"),
    _k("server", "tls_key", str, "", desc="TLS 私钥"),
    _k("server", "access_log", bool, True, desc="访问日志开关"),
    # -- model（11）---------------------------------------------------------
    _k("model", "path", str, "", desc=".rkllm 模型路径；serve 必填"),
    _k("model", "search_dirs", list, (), desc="裸模型名的搜索目录"),
    _k("model", "freq_helper", str, "auto",
       desc="锁频脚本路径；auto=用内建 sysfs 方案，给了路径则交给脚本"),
    _k("model", "class", str, "auto", choices=("auto", "small", "medium", "large"),
       desc="模型档位；auto 按文件名与大小推导"),
    _k("model", "context", int, 4096, min_v=512, max_v=32768,
       desc="上下文长度；与内存预算联动校验"),
    _k("model", "min_context", int, 1024, min_v=256, max_v=8192,
       desc="内存不足时最低降到此值，再低则拒绝启动"),
    _k("model", "max_tokens", int, 0, min_v=0, max_v=32768,
       desc="0=按档位（small 2048 / medium 1024 / large 512）"),
    _k("model", "threads", int, 4, min_v=1, max_v=8, desc="CPU 线程数"),
    _k("model", "platform", str, "auto",
       choices=("auto", "rk3588", "rk3576", "rk3562", "rv1126b"), desc="SoC 平台"),
    _k("model", "freq_lock", bool, True,
       desc="NPU/CPU 锁频（需 root；非 root 或 sysfs 不支持时告警跳过）"),
    _k("model", "keep_history", bool, False, desc="服务端保历史；默认无状态"),
    # -- engine（8）---------------------------------------------------------
    _k("engine", "mode", str, "subprocess", choices=("subprocess", "in_process"),
       desc="推理进程隔离；subprocess 崩溃不拖垮 HTTP"),
    _k("engine", "load_timeout_s", int, 300, min_v=30, max_v=900, desc="模型冷加载超时"),
    _k("engine", "restart_backoff_s", int, 5, min_v=1, max_v=120,
       desc="worker 重启退避起点（指数）"),
    _k("engine", "restart_backoff_max_s", int, 120, min_v=5, max_v=600,
       desc="重启退避上限"),
    _k("engine", "restart_max_per_hour", int, 3, min_v=0, max_v=60,
       desc="每小时自愈次数上限，超过停止自愈并告警"),
    _k("engine", "memory_check", bool, True, desc="启动前内存预算校验"),
    _k("engine", "memory_reserve_mb", int, 1024, min_v=256, max_v=8192,
       desc="给系统与 NPU 驱动的内存预留"),
    _k("engine", "health_interval_s", int, 10, min_v=1, max_v=60,
       desc="supervisor 巡检间隔"),
    # -- queue（5）----------------------------------------------------------
    _k("queue", "max_wait", int, 32, min_v=0, max_v=256, desc="允许等待的请求数；0=不排队"),
    _k("queue", "wait_timeout_s", int, 600, min_v=1, max_v=3600, desc="排队超时"),
    _k("queue", "request_timeout_s", int, 900, min_v=1, max_v=7200,
       desc="单请求总时长上限"),
    _k("queue", "busy_retry_after_s", int, 5, min_v=0, max_v=120,
       desc="忙时 Retry-After 秒数"),
    # implemented=False：键保留（旧的排队配置在这里有落点），但运行时恒夹紧
    # 为 1——单 NPU 单引擎串行，>1 只会有害无益。值与行为无关，故诚实标注
    # 「未接线」，config show/docs 会带 [未接线] 标记。
    _k("queue", "max_concurrent", int, 1, min_v=1, max_v=1,
       desc="并发上限；单 NPU 恒为 1（配置值不生效）", implemented=False),
    # -- openai（9）---------------------------------------------------------
    _k("openai", "param_policy", str, "warn", choices=("warn", "reject"),
       desc="不支持参数的处理策略（语义关键项恒为 reject）"),
    _k("openai", "tools_mode", str, "prompt", choices=("prompt", "native"),
       desc="prompt=提示词注入+解析；native=rkllm_set_function_tools（污染句柄，不用）"),
    _k("openai", "tool_parser", str, "auto",
       choices=("auto", "qwen", "llama3", "deepseek", "hermes"), desc="工具调用解析模板"),
    _k("openai", "json_schema_retries", int, 2, min_v=0, max_v=5,
       desc="json_schema 校验失败重试次数"),
    _k("openai", "reasoning_field", str, "reasoning_content",
       choices=("reasoning_content", "reasoning", "none"), desc="思考内容字段名"),
    _k("openai", "stream_heartbeat_s", int, 15, min_v=5, max_v=60,
       desc="SSE 心跳间隔"),
    _k("openai", "max_tool_calls", int, 8, min_v=1, max_v=32,
       desc="单条消息允许的最大 tool_call 数"),
    _k("openai", "include_usage_default", bool, False, desc="默认是否在流末尾发 usage 帧"),
    # 1.3.1-15 的 conffile 变量 RKLLM_JSON_HINT：json_object 追加的提示语。
    # 此前 pipeline 直读 os.environ，绕过本表（违反「影响行为的参数必须进表」），
    # 现已进表；旧变量名经 LEGACY_ENV_MAP 继续生效。
    _k("openai", "json_hint", str, "",
       desc="response_format=json_object 时追加的提示语；空=内置默认，"
            "填 none=不追加"),
    # -- cache（5）----------------------------------------------------------
    # 单槽前缀缓存（ADR-07）：上一轮 prompt 是新 prompt 的前缀时复用其 KV，
    # 避免多轮对话反复全量 prefill。worker/promptcache.py 实现，加载失败自动
    # 清槽重试一次不带缓存的推理（R5 的兜底）。
    _k("cache", "enabled", bool, False,
       desc="prompt 前缀缓存开关；默认关闭——当前 runtime 版本加载缓存文件"
            "返回 rkllm_run -1（内部码 10000），待上游修复后再开"),
    _k("cache", "dir", str, "/dev/shm/rkllm/prefix",
       desc="缓存目录；默认 tmpfs（内存），不可写时自动回退 /tmp/rkllm/prefix"),
    _k("cache", "max_mb", int, 512, min_v=0, max_v=8192,
       desc="缓存目录容量上限；超出按最旧淘汰；0=不限"),
    _k("cache", "ttl_s", int, 604800, min_v=0, max_v=2592000,
       desc="缓存文件 TTL；0=不过期"),
    # 是否接管单槽的判据：低于阈值视为「另一个会话」，不加载也不覆盖缓存
    # （worker/promptcache.py）。此前是模块级常量，不可配。
    _k("cache", "similarity", float, 0.5, min_v=0.0, max_v=1.0,
       desc="前缀相似度阈值；低于该值判定为无关会话（llama.cpp 默认 0.5）"),
    # -- mcp（24）-----------------------------------------------------------
    _k("mcp", "enabled", bool, True, desc="MCP 连接器总开关"),
    _k("mcp", "transport", str, "stdio", choices=("stdio", "http"), desc="传输方式"),
    _k("mcp", "http_path", str, "/mcp", desc="HTTP 模式端点路径"),
    _k("mcp", "backend", str, "auto", choices=("auto", "local", "http"),
       desc="板端=local（进程内），客户端=http"),
    _k("mcp", "target_endpoint", str, "http://127.0.0.1:8080/v1", desc="http 后端地址"),
    _k("mcp", "target_api_key", str, "", desc="板端 key；客户端配置文件 0600"),
    _k("mcp", "target_timeout_s", int, 120, min_v=5, max_v=3600, desc="单次 HTTP 读超时"),
    _k("mcp", "protocol_versions", list,
       ("2026-07-28", "2025-11-25", "2025-06-18", "2024-11-05"),
       desc="协议兼容矩阵，首个为主版本"),
    _k("mcp", "tools", list, (), desc="启用的工具列表；空=全部 6 个"),
    _k("mcp", "tool_prefix", str, "local_llm_", desc="工具名前缀"),
    _k("mcp", "task_mode", str, "auto", choices=("auto", "always", "never"),
       desc="长任务策略；第一期 job 工具恒可用"),
    _k("mcp", "task_threshold_s", int, 45, min_v=5, max_v=55,
       desc="超过此秒数转 job；必须 < 60 避开客户端硬超时"),
    _k("mcp", "job_ttl_s", int, 600, min_v=60, max_v=3600, desc="job 结果保留时长"),
    _k("mcp", "job_max", int, 8, min_v=1, max_v=64, desc="并存 job 上限"),
    _k("mcp", "job_poll_hint_s", int, 2, min_v=1, max_v=30, desc="建议客户端轮询间隔"),
    _k("mcp", "progress_interval_s", int, 5, min_v=0, max_v=30,
       desc="notifications/progress 间隔；0=不发"),
    _k("mcp", "max_input_chars", int, 12000, min_v=100, max_v=100000,
       desc="工具输入字符上限（约 4k token 中文）"),
    _k("mcp", "max_result_chars", int, 20000, min_v=1000, max_v=200000,
       desc="工具结果字符上限；超出截断并标注"),
    _k("mcp", "default_max_tokens", int, 512, min_v=1, max_v=8192,
       desc="工具未指定时的生成上限"),
    _k("mcp", "default_temperature", float, 0.3, min_v=0.0, max_v=2.0,
       desc="辅助任务偏确定"),
    _k("mcp", "batch_max_segments", int, 20, min_v=1, max_v=100,
       desc="batch 工具的分段上限"),
    _k("mcp", "max_concurrent", int, 1, min_v=1, max_v=1,
       desc="单 NPU 固定 1（与 queue.max_concurrent 同源）"),
    _k("mcp", "extract_retries", int, 2, min_v=0, max_v=5,
       desc="结构化抽取校验失败重试次数"),
    _k("mcp", "allow_remote_origin", list, (), desc="HTTP 模式 Origin 白名单"),
    # -- logging（5）--------------------------------------------------------
    _k("logging", "level", str, "info",
       choices=("debug", "info", "warning", "error"), desc="日志级别"),
    _k("logging", "format", str, "text", choices=("text", "json"),
       desc="json 便于采集"),
    _k("logging", "file", str, "", desc="日志文件；空=只 stderr/journal"),
    _k("logging", "max_bytes", int, 10485760, min_v=1048576, max_v=1073741824,
       desc="文件轮转阈值"),
    _k("logging", "backup_count", int, 5, min_v=0, max_v=100, desc="保留旧日志数"),
    # -- metrics（2）--------------------------------------------------------
    _k("metrics", "enabled", bool, True, desc="/metrics 开关"),
    _k("metrics", "path", str, "/metrics", desc="metrics 路径"),
)

assert len(SCHEMA) == len(set(k.full for k in SCHEMA)), "配置键名重复（N11）"

# 板端默认 TOML 落点。此前 82 键表只有 RKLLM_TOML 环境变量一个入口，deb 也不
# 提供模板，运维根本看不见这张表；现在 load_config 会自动读它（不存在则静默
# 跳过，默认行为零变化），模板由 `rkllm config init --board` 生成。
DEFAULT_TOML_PATH = "/etc/rkllm/rkllm.toml"

_BY_SECTION: dict[str, list[Key]] = {}
for _key in SCHEMA:
    _BY_SECTION.setdefault(_key.section, []).append(_key)


# ---------------------------------------------------------------------------
# 旧 conffile 迁移（MCP 设计 §7.4）
# ---------------------------------------------------------------------------

LEGACY_ENV_MAP: dict[str, str] = {
    "RKLLM_MODEL": "model.path",
    "RKLLM_MODEL_DIR": "model.search_dirs",
    "RKLLM_BIND": "server.bind_host",
    "RKLLM_PORT": "server.port",
    "RKLLM_API_KEY": "server.api_key",
    "RKLLM_MAX_QUEUE": "queue.max_wait",
    "RKLLM_QUEUE_TIMEOUT": "queue.wait_timeout_s",
    "RKLLM_READ_TIMEOUT": "server.read_timeout_s",
    "RKLLM_MAX_BODY": "server.max_body_bytes",
    "RKLLM_THREADS": "model.threads",
    "RKLLM_PLATFORM": "model.platform",
    "RKLLM_NO_HISTORY": "model.keep_history",
    "RKLLM_FREQ_HELPER": "model.freq_helper",
    "RKLLM_NO_FREQ_LOCK": "model.freq_lock",
    "RKLLM_UNSUPPORTED_POLICY": "openai.param_policy",
    "RKLLM_LOG_LEVEL": "logging.level",
    # 1.3.1-15 的 conffile 用的是单数 RKLLM_CORS_ORIGIN：新栈的对应键是
    # server.cors_origins（列表），加载时按逗号切分，旧设置继续生效。
    "RKLLM_CORS_ORIGIN": "server.cors_origins",
    # 1.3.1-15 的 json_object 提示语变量：新键 openai.json_hint（进表后仍认旧名）
    "RKLLM_JSON_HINT": "openai.json_hint",
}


# 反义开关：旧变量为真表示"关闭"，映射到新键时必须取反，否则语义完全相反。
_NEGATED_LEGACY: frozenset[str] = frozenset({
    "RKLLM_NO_HISTORY", "RKLLM_NO_FREQ_LOCK",
})

_TRUE_WORDS = ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# 加载与校验
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """启动校验失败；errors 为全部问题的可读列表（一次性列出）。"""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("配置校验失败：\n  - " + "\n  - ".join(errors))


@dataclass
class Config:
    """按 section 命名空间的生效配置（config.server.port 风格访问）。"""

    values: dict[str, Any] = field(default_factory=dict)   # "server.port" -> value
    sources: dict[str, str] = field(default_factory=dict)  # "server.port" -> 来源

    def __getattr__(self, section: str) -> Any:
        if section.startswith("_") or section not in _BY_SECTION:
            raise AttributeError(section)
        ns = _SectionView(self, section)
        object.__setattr__(self, section, ns)   # 缓存，避免重复构造
        return ns

    def source_of(self, full_key: str) -> str:
        return self.sources.get(full_key, "default")


class _SectionView:
    """section 视图：前缀与 values 引用预先算好，热路径只做一次 dict 查找。

    审查项（效率）：原实现每次属性访问都做 f-string 拼接 + 两次
    `object.__getattribute__` + `in` 判断 + 再取值（两次 dict 查找）。
    `Config.values` 只会被原地修改（`cli.py` 的 `cfg.values[...] = ...`），
    dict 对象本身不替换，因此缓存引用是安全的。
    """

    __slots__ = ("_cfg", "_section", "_prefix", "_values")

    def __init__(self, cfg: Config, section: str):
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_section", section)
        object.__setattr__(self, "_prefix", section + ".")
        object.__setattr__(self, "_values", cfg.values)

    def __getattr__(self, name: str) -> Any:
        key = self._prefix + name
        try:
            return self._values[key]
        except KeyError:
            raise AttributeError(key) from None


def _coerce(key: Key, raw: Any, origin: str, errors: list[str]) -> Any:
    """把原始值收敛为 schema 类型；问题记入 errors 而不中断（一次性列出）。"""
    try:
        if key.type is bool:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str):
                low = raw.strip().lower()
                if low in ("1", "true", "yes", "on"):
                    return True
                if low in ("0", "false", "no", "off"):
                    return False
            raise ValueError("需为布尔（true/false）")
        if key.type is int:
            if isinstance(raw, bool):
                raise ValueError("需为整数")
            val = int(raw)
            if key.min_v is not None and val < key.min_v:
                raise ValueError(f"需 ≥ {key.min_v}")
            if key.max_v is not None and val > key.max_v:
                raise ValueError(f"需 ≤ {key.max_v}")
            return val
        if key.type is float:
            if isinstance(raw, bool):
                raise ValueError("需为数值")
            val = float(raw)
            # P1-5：NaN/Inf 会绕过 min/max（nan<x 恒为 False），显式拒绝
            if val != val or val in (float("inf"), float("-inf")):
                raise ValueError("需为有限数值（不接受 NaN/Inf）")
            if key.min_v is not None and val < key.min_v:
                raise ValueError(f"需 ≥ {key.min_v}")
            if key.max_v is not None and val > key.max_v:
                raise ValueError(f"需 ≤ {key.max_v}")
            return val
        if key.type is list:
            if isinstance(raw, str):
                items = tuple(s.strip() for s in raw.split(",") if s.strip())
            elif isinstance(raw, (list, tuple)):
                items = tuple(raw)
            else:
                raise ValueError("需为列表")
            if key.min_v is not None and len(items) < key.min_v:
                raise ValueError(f"长度需 ≥ {key.min_v}")
            if key.max_v is not None and len(items) > key.max_v:
                raise ValueError(f"长度需 ≤ {key.max_v}")
            return items
        # str
        if not isinstance(raw, str):
            val = str(raw)
        else:
            val = raw
        if key.choices and val not in key.choices:
            raise ValueError(f"取值需为 {'/'.join(key.choices)}")
        return val
    except (ValueError, TypeError) as exc:
        errors.append(f"{key.full}（{origin}）：{exc}")
        return None


def load_config(*, file: str | None = None, cli: dict[str, Any] | None = None,
                env: dict[str, str] | None = None,
                require_model_path: bool = False) -> Config:
    """按 优先级 default < file < env < cli 装配配置并整体校验。

    require_model_path：serve 子命令置 True；config 命令与客户端 wheel 置 False
    （model.path 仅板端推理需要）。
    """
    env = dict(os.environ if env is None else env)
    errors: list[str] = []
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}

    # 旧 conffile 的 RKLLM_* 名先映射到新键（新键名优先，迁移期为旧名兜底）
    for legacy, target in LEGACY_ENV_MAP.items():
        section, name = target.split(".", 1)
        new_env = f"RKLLM_{section}_{name}".upper()
        if legacy in env and new_env not in env and env[legacy].strip():
            raw = env[legacy]
            if legacy in _NEGATED_LEGACY:
                # NO_HISTORY=1 → keep_history=false；NO_FREQ_LOCK=1 → freq_lock=false
                env[new_env] = ("false" if raw.strip().lower() in _TRUE_WORDS
                                else "true")
            else:
                env[new_env] = raw

    file_data: dict[str, Any] = {}
    # 未显式指定时，板端的默认 TOML 存在就用它（82 键的落点；此前只有
    # RKLLM_TOML 环境变量一条路，deb 也没打模板，等于运维看不见这张表）。
    # 文件不存在是正常状态，不算错误。
    path = file or DEFAULT_TOML_PATH
    if path and (file or os.path.isfile(path)):
        try:
            with open(path, "rb") as fh:
                file_data = tomllib.load(fh)
        except FileNotFoundError:
            if file:
                errors.append(f"配置文件不存在：{file}")
        except tomllib.TOMLDecodeError as exc:
            errors.append(f"配置文件 TOML 语法错误：{exc}")

    for key in SCHEMA:
        value, source = key.default, "default"
        raw = file_data.get(key.section, {})
        if key.name in raw:
            value, source = _coerce(key, raw[key.name], "file", errors), "file"
        env_val = env.get(key.env)
        if env_val is not None and env_val != "":
            value, source = _coerce(key, env_val, "env", errors), "env"
        if cli and key.full in cli and cli[key.full] is not None:
            value, source = _coerce(key, cli[key.full], "cli", errors), "cli"
        values[key.full] = value
        sources[key.full] = source

    _cross_validate(values, errors, require_model_path)
    if errors:
        raise ConfigError(errors)
    return Config(values=values, sources=sources)


def _cross_validate(values: dict[str, Any], errors: list[str],
                    require_model_path: bool) -> None:
    """跨键校验与安全下限（设计 §7.1 / MCP 设计 §2.1）。

    单键校验失败的键值为 None（错误已入列表），此处跳过含 None 的组合判断，
    避免二次异常掩盖原始错误。
    """
    if require_model_path and not values.get("model.path"):
        errors.append("model.path：serve 需要模型路径（必填）")
    min_ctx, ctx = values.get("model.min_context"), values.get("model.context")
    if min_ctx is not None and ctx is not None and min_ctx > ctx:
        errors.append("model.min_context 不得大于 model.context")
    backoff, backoff_max = (values.get("engine.restart_backoff_s"),
                            values.get("engine.restart_backoff_max_s"))
    if backoff is not None and backoff_max is not None and backoff_max < backoff:
        errors.append("engine.restart_backoff_max_s 不得小于 engine.restart_backoff_s")
    cert, key_pem = values.get("server.tls_cert"), values.get("server.tls_key")
    if cert is not None and key_pem is not None and bool(cert) != bool(key_pem):
        errors.append("server.tls_cert 与 server.tls_key 必须成对给出")
    threshold = values.get("mcp.task_threshold_s")
    if threshold is not None and threshold >= 60:
        errors.append("mcp.task_threshold_s 必须 < 60（客户端 60 s 硬超时）")
    if values.get("mcp.transport") == "http" and not values.get("mcp.allow_remote_origin"):
        errors.append("mcp.transport=http 时必须配置 mcp.allow_remote_origin（安全边界）")


# ---------------------------------------------------------------------------
# 命令入口（rkllm config show / validate / docs / init）
# ---------------------------------------------------------------------------


def config_show(cfg: Config, as_json: bool = False) -> str:
    """打印生效值与来源（default/file/env/cli）。未接线的键如实标注。"""
    import json
    if as_json:
        return json.dumps({"values": cfg.values,
                           "sources": cfg.sources,
                           "not_implemented": [k.full for k in SCHEMA
                                               if not k.implemented]},
                          ensure_ascii=False, indent=2)
    lines = []
    for section, keys in _BY_SECTION.items():
        lines.append(f"[{section}]")
        for key in keys:
            mark = "" if key.implemented else "  [未接线]"
            lines.append(f"  {key.name} = {cfg.values[key.full]!r}"
                         f"    # {cfg.sources[key.full]}  {key.desc}{mark}")
    return "\n".join(lines)


def config_docs() -> str:
    """由 schema 生成 Markdown 键表（文档同源，杜绝漂移）。"""
    lines = ["| 键 | 默认 | 取值/范围 | 说明 |", "|---|---|---|---|"]
    for key in SCHEMA:
        rng = "/".join(key.choices) if key.choices else ""
        if key.min_v is not None or key.max_v is not None:
            rng = (rng + " " if rng else "") + f"[{key.min_v}, {key.max_v}]"
        mark = "" if key.implemented else "（未接线）"
        lines.append(f"| `{key.full}` | `{key.default!r}` | {rng} "
                     f"| {key.desc}{mark} |")
    return "\n".join(lines)


def config_init(target: str = "board") -> str:
    """生成 TOML 模板（设计 §7.2 `rkllm config init --board|--client`）。

    只输出注释 + 少量必填项，保持与 `config docs` 的 82 键同源但不重复抄写
    全部默认值——默认值本来就不需要写进文件（优先级链已保证）。
    """
    if target == "client":
        return """# <连接器安装目录>/mcp.toml - rkllm-mcp 客户端配置
# 优先级：内置默认 < 本文件 < RKLLM_* 环境变量
# 完整键表：见板端 `rkllm-serve config docs` 的 [mcp] 组

[mcp]
# 板端 OpenAI 兼容端点（也可用 RKLLM_ENDPOINT 环境变量）
target_endpoint = "http://127.0.0.1:8080/v1"
# 板端 server.api_key（留空表示板端未启用鉴权）
target_api_key = ""
# 单次 HTTP 读超时（秒）
target_timeout_s = 120
# 超过此秒数转 job；必须 < 60（客户端工具硬超时）
task_threshold_s = 45
# 工具结果与输入上限（字符）
max_result_chars = 20000
max_input_chars = 12000
"""
    return """# /etc/rkllm/rkllm.toml - 板端 FastAPI 栈配置（`rkllm-serve serve`）
# 优先级：内置默认 < 本文件 < RKLLM_* 环境变量 < 命令行
# 全部 82 键与默认值：  rkllm-serve config docs
# 生效值与来源：        rkllm-serve config show
# 只填需要改的项即可。

[server]
# 默认只监听本机；对局域网提供服务时显式改这里，并同时配置 api_key。
#bind_host = "127.0.0.1"
#port = 8080
#api_key = ""              # 空 = 不鉴权（与 1.3.1-15 行为一致）
#api_key_file = ""         # 优先于 api_key；建议 0600

[model]
# serve 必须提供；也可用裸文件名并在 search_dirs 中查找
path = "/var/lib/rkllm/models/your-model.rkllm"
#search_dirs = ["/var/lib/rkllm/models"]
#class = "auto"            # auto 按文件名与体积推导
#context = 4096            # 内存不足时按 2048 / 1024 逐级降，不低于 min_context
#min_context = 1024
#max_tokens = 0            # 0 = 按档位（small 2048 / medium 1024 / large 512）

[engine]
#mode = "subprocess"        # in_process 为排障/回滚开关（崩溃不隔离）
#memory_check = true        # 启动前内存预算校验
#memory_reserve_mb = 1024

[queue]
#max_wait = 32
#request_timeout_s = 900

[openai]
#param_policy = "warn"      # 语义关键项（n>1 / tool_choice / 流式 json_schema）恒 400
#tool_parser = "auto"
#stream_heartbeat_s = 15

[logging]
#level = "info"
#format = "text"            # json 便于采集
#file = ""                  # 空 = 只 stderr/journal
"""
