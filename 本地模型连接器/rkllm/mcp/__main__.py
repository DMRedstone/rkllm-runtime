"""rkllm-mcp 命令入口：纯环境变量可启动（WorkBuddy mcp.json 的 env 方式）。

用法：
  rkllm-mcp                       # stdio，默认读 <安装目录>/mcp.toml
  rkllm-mcp --transport stdio     # 显式
  rkllm-mcp ask "把这段改成公文体"  # 排障 CLI，输出纯文本

配置优先级：内置默认 < `<连接器安装目录>/mcp.toml` < RKLLM_* 环境变量
（可用 RKLLM_MCP_CONFIG 显式覆盖配置文件路径）。配置文件与 `mcp.toml.example`
同级，跟着连接器安装目录走，不放到注册文件或共享目录里。

其中 `enabled` / `transport` / `http_path` / `allow_remote_origin` 属于"模式选择"，
在 main() 里判定——不实现的模式显式拒绝并说明，而不是静默按 stdio 跑。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import urllib.parse
from pathlib import Path

from .backend import BackendError, HttpBackend, LocalBackend
from .protocol import PROTOCOL_VERSIONS, normalize_supported_versions
from .stdio import StdioServer

DEFAULT_PROTOCOL_VERSIONS = PROTOCOL_VERSIONS
DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1"

# 值域校验表（P2.5）：TOML/env 里数值越界或非有限值统一回退默认并告警，
# 不再让 `inf`/`1e400`/0 之类的值在运行期制造 OverflowError/死循环。
_CFG_RANGES: dict[str, tuple[float, float]] = {
    "client_timeout_s": (5, 3600),          # 转 job 阈值的客户端约束
    "target_timeout_s": (5, 4200),
    "default_max_tokens": (1, 8192),
    "max_result_chars": (1000, 200000),
    "batch_max_segments": (1, 100),
    "extract_retries": (0, 5),
    "job_ttl_s": (60, 3600),
    "job_max": (1, 64),
    "job_poll_hint_s": (1, 30),
    "progress_interval_s": (0, 30),
    "session_max_messages": (0, 1000),
    "session_max_count": (1, 4096),
    "session_ttl_s": (0, 86400),
    "max_concurrent": (1, 32),
    "default_temperature": (0.0, 2.0),
}
# 枚举校验表（与板端 config.py 的 choices 对齐）
_CFG_ENUMS: dict[str, tuple[str, ...]] = {
    "task_mode": ("auto", "always", "never"),
    "backend": ("auto", "http", "local"),
    "transport": ("stdio", "http"),
}
# 连接器实际实现的 6 个工具短名（mcp.tools 只认这些）
_TOOL_SHORT = ("ask", "chat", "extract", "batch", "status", "job")


def default_config_path() -> str:
    """连接器安装目录下的 mcp.toml（与 mcp.toml.example 同级）。

    `rkllm/mcp/__main__.py` 的上两级就是安装根目录；repo 内则是
    `本地模型连接器/mcp.toml`。配置跟着安装走，key 不进注册文件。
    """
    return str(Path(__file__).resolve().parents[2] / "mcp.toml")


def default_cfg(endpoint: str, api_key: str) -> dict:
    """配置缺省（板端键表 mcp 组的客户端子集 + 少量客户端专有键；TOML 加载后合并）。"""
    return {
        # -- 模式选择 ------------------------------------------------------
        "enabled": True,                 # mcp.enabled：连接器总开关
        "transport": "stdio",            # mcp.transport：一期只支持 stdio
        "http_path": "/mcp",             # mcp.http_path：二期
        "backend": "auto",               # auto/http=HTTP 后端；local=板端进程内
        # 启用的工具（短名）；空=全部。必须是 list 而非 tuple：_coerce_cfg_types
        # 按本表类型模板归一化，tuple 不命中 list 分支会被 str() 掉，
        # 变成真值字符串 "()" 导致全部推理工具被判禁用（2026-11-19 审计 P0）。
        "tools": [],
        "allow_remote_origin": [],       # mcp.allow_remote_origin：二期
        "protocol_versions": list(DEFAULT_PROTOCOL_VERSIONS),
        "max_concurrent": 1,             # mcp.max_concurrent：在飞工具调用数
        # -- 工具行为 ------------------------------------------------------
        "tool_prefix": "local_llm_",
        # 生成上限 / 同步读超时的冷启动兜底：板端可达时分别由
        # /health.limits.default_max_tokens 与 request_timeout_s 覆盖。
        "default_max_tokens": 512,
        "default_temperature": 0.3,
        "max_result_chars": 20000,
        "batch_max_segments": 20,
        "task_mode": "auto",
        # 转 job 的耗时阈值 = client_timeout_s − 15 s（客户端硬超时余量）。
        # 这是 A 方案唯一保留的"判断参数"：它是客户端约束而非模型能力，
        # 板端无从得知调用方会等多久。
        "client_timeout_s": 60,
        "job_ttl_s": 600,
        "job_max": 8,
        "job_poll_hint_s": 2,
        "progress_interval_s": 5,
        "extract_retries": 2,
        "session_max_messages": 0,       # 0=不限制；>0 时裁剪会话历史
        "session_max_count": 256,        # 保留的会话数上限
        "session_ttl_s": 1800,           # 会话空闲过期（秒）；0=不过期
        # -- 目标 ----------------------------------------------------------
        # 同步读超时的冷启动兜底；板端 /health.limits.request_timeout_s 优先。
        # job 读超时 = max(板端请求超时, 估算×1.5+30)，上限 4200 s，
        # factor/上限均为传输层常量（见 stdio.py），不再作为配置项。
        "target_timeout_s": 120,
        "target_endpoint": endpoint,
        "target_api_key": api_key,
        # 内部字段（与 target_* 同值；键表用 target_* 命名）
        "endpoint": endpoint,
        "api_key": api_key,
    }


def load_cfg() -> dict:
    """安装目录 mcp.toml + RKLLM_* 环境变量覆盖（env 优先，键名与板端 mcp 组一致）。"""
    endpoint = os.environ.get("RKLLM_ENDPOINT", "http://127.0.0.1:8080/v1")
    api_key = os.environ.get("RKLLM_API_KEY", "")
    cfg = default_cfg(endpoint, api_key)
    toml_path = os.environ.get("RKLLM_MCP_CONFIG") or default_config_path()
    if os.path.isfile(toml_path):
        try:
            import tomllib
            with open(toml_path, "rb") as fh:
                data = tomllib.load(fh)
            section = data.get("mcp") or {}
            for key, value in section.items():
                if key in cfg:
                    cfg[key] = value
            # 键表用 target_* 命名：端点与 key 合并到内部字段
            if section.get("target_endpoint"):
                cfg["endpoint"] = str(section["target_endpoint"])
            if section.get("target_api_key") is not None:
                cfg["api_key"] = str(section["target_api_key"])
        except (OSError, ValueError) as exc:
            # 配置存在但读不出来必须留痕（审计 P2）：静默回退会让用户以为
            # 配置已生效，实际跑的是默认值。stdio 模式下 stderr 不进协议线。
            sys.stderr.write("[rkllm-mcp] 读取配置失败，已回退默认值：%s: %s\n"
                             % (toml_path, exc))
    # 环境变量最终覆盖 TOML（RKLLM_MCP_* 或兼容名 RKLLM_ENDPOINT / RKLLM_API_KEY）
    for key in list(cfg.keys()):
        env_key = "RKLLM_MCP_" + key.upper()
        if env_key in os.environ:
            val = os.environ[env_key]
            cfg[key] = val
            # P1-3：target_* 与内部键任一被 env 覆盖都双写，避免 build_server 读旧值
            if key in ("endpoint", "target_endpoint"):
                cfg["endpoint"] = cfg["target_endpoint"] = val
            elif key in ("api_key", "target_api_key"):
                cfg["api_key"] = cfg["target_api_key"] = val
    if "RKLLM_ENDPOINT" in os.environ:
        cfg["endpoint"] = cfg["target_endpoint"] = os.environ["RKLLM_ENDPOINT"]
    if "RKLLM_API_KEY" in os.environ:
        cfg["api_key"] = cfg["target_api_key"] = os.environ["RKLLM_API_KEY"]
    # P1-4：TOML/env 值都是字符串，按 schema 类型归一化，避免 "45" 直达比较逻辑
    # P2.5：再补值域/枚举/端点校验；所有问题一次性写 stderr，不崩启动。
    problems = _coerce_cfg_types(cfg)
    problems.extend(_validate_cfg(cfg))
    for problem in problems:
        sys.stderr.write("[rkllm-mcp] 配置项不可用，已回退默认：%s\n" % problem)
    return cfg


def _coerce_cfg_types(cfg: dict) -> list[str]:
    """按 default_cfg 的类型模板归一化；不合法值回退默认（绝不带坏类型）。

    P2.5：`int(inf)` 抛的是 OverflowError（不是 ValueError），TOML 合法字面量
    `inf`/`1e400` 会让旧实现启动直接 traceback。这里统一捕获并记录问题。
    """
    problems: list[str] = []
    for key, default in default_cfg("", "").items():
        if key not in cfg:
            continue
        val = cfg[key]
        try:
            if isinstance(default, bool):
                if not isinstance(val, bool):
                    cfg[key] = str(val).strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                if isinstance(val, float) and not math.isfinite(val):
                    raise OverflowError("非有限数值")
                cfg[key] = int(val)
            elif isinstance(default, float):
                cfg[key] = float(val)
                if not math.isfinite(cfg[key]):
                    raise ValueError("非有限数值")
            elif isinstance(default, list):
                if isinstance(val, str):
                    cfg[key] = [s.strip() for s in val.split(",") if s.strip()]
                elif not isinstance(val, list):
                    raise ValueError("需为数组")
            else:
                cfg[key] = str(val)
        except (TypeError, ValueError, OverflowError) as exc:
            problems.append("%s=%r 类型非法（%s），回退 %r"
                            % (key, val, exc, default))
            cfg[key] = default
    return problems


def _valid_endpoint(value: str) -> bool:
    try:
        parts = urllib.parse.urlsplit(value or "")
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def _validate_cfg(cfg: dict) -> list[str]:
    """值域/枚举/端点/版本列表校验；返回问题列表（调用方统一写 stderr）。

    只改"不可用"的值：合法配置原样保留（用户自定值不被悄悄改写）。
    """
    problems: list[str] = []
    defaults = default_cfg(DEFAULT_ENDPOINT, "")
    for key, (lo, hi) in _CFG_RANGES.items():
        value = cfg.get(key, defaults[key])
        valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(float(value))
                 and lo <= float(value) <= hi)
        if not valid:
            problems.append("%s=%r 超出允许范围 [%s, %s]，回退 %r"
                            % (key, value, lo, hi, defaults[key]))
            cfg[key] = defaults[key]
        elif isinstance(defaults[key], int) and not isinstance(value, int):
            # TOML `45.0` 是合法浮点；整型配置项按整数存，避免下游类型漂移
            cfg[key] = int(value)
    for key, choices in _CFG_ENUMS.items():
        value = cfg.get(key)
        if value not in choices:
            problems.append("%s=%r 不是合法枚举（%s），回退 %r"
                            % (key, value, "/".join(choices), defaults[key]))
            cfg[key] = defaults[key]
    endpoint = str(cfg.get("endpoint") or "")
    if not _valid_endpoint(endpoint):
        problems.append("target_endpoint=%r 不是 http(s) URL，回退 %r"
                        % (endpoint, DEFAULT_ENDPOINT))
        endpoint = DEFAULT_ENDPOINT
    cfg["endpoint"] = cfg["target_endpoint"] = endpoint
    cfg["api_key"] = str(cfg.get("api_key") or "")
    cfg["target_api_key"] = cfg["api_key"]

    # protocol_versions：过滤未知版本、去重、保持新→旧顺序；空/非法回默认四版
    raw_versions = cfg.get("protocol_versions")
    if isinstance(raw_versions, str):
        raw_versions = [s.strip() for s in raw_versions.split(",") if s.strip()]
    if isinstance(raw_versions, (list, tuple)):
        raw_list = [str(v).strip() for v in raw_versions]
        normalized = normalize_supported_versions(raw_list)
        if list(normalized) != raw_list:
            problems.append("protocol_versions=%r 含未知/重复版本，收敛为 %r"
                            % (raw_list, list(normalized)))
        cfg["protocol_versions"] = list(normalized)
    else:
        problems.append("protocol_versions 需为数组，回退默认四版")
        cfg["protocol_versions"] = list(DEFAULT_PROTOCOL_VERSIONS)

    # tools：短名列表；未知工具名告警并丢弃（mcp.tools 只实现了 6 个）
    tools = cfg.get("tools")
    if isinstance(tools, str):
        tools = [s.strip() for s in tools.split(",") if s.strip()]
    if isinstance(tools, (list, tuple)):
        cleaned = [str(t).strip() for t in tools if str(t).strip()]
        unknown = [t for t in cleaned if t not in _TOOL_SHORT]
        if unknown:
            problems.append("tools 含未实现的工具 %r，已忽略" % unknown)
        cfg["tools"] = [t for t in cleaned if t in _TOOL_SHORT]
    else:
        problems.append("tools 需为数组，回退空=全部启用")
        cfg["tools"] = []

    prefix = str(cfg.get("tool_prefix") or "")
    if not prefix.strip():
        problems.append("tool_prefix 不能为空，回退 %r" % defaults["tool_prefix"])
        cfg["tool_prefix"] = defaults["tool_prefix"]
    else:
        cfg["tool_prefix"] = prefix
    return problems


def build_server(cfg: dict | None = None) -> StdioServer:
    cfg = cfg or load_cfg()
    kind = str(cfg.get("backend") or "auto").lower()
    if kind == "local":
        # 连接器作为客户端交付时没有板端 worker，local 后端无法直调；
        # 按配置选择后端（而不是静默忽略），local 会给出可诊断错误。
        backend = LocalBackend()
    else:
        backend = HttpBackend(cfg["endpoint"], cfg.get("api_key", ""),
                              timeout_s=int(cfg.get("target_timeout_s", 120)))
    return StdioServer(backend, cfg,
                       model_class=os.environ.get("RKLLM_MODEL_CLASS", "4b"))


def cli_ask(prompt: str) -> int:
    """排障 CLI：单轮调用，输出纯文本（不是任何客户端的兜底）。"""
    server = build_server()
    try:
        result = server._handlers["ask"]({"prompt": prompt})
        sys.stdout.write(result["content"][0]["text"] + "\n")
        return 0
    except BackendError as exc:
        sys.stderr.write("error: %s\n" % exc)
        return 1


def _check_mode(cfg: dict) -> str | None:
    """返回错误说明；None 表示可以按 stdio 启动。"""
    if not cfg.get("enabled", True):
        return "mcp.enabled=false：连接器不启动（配置要求，不是错误）"
    transport = str(cfg.get("transport", "stdio")).lower()
    if transport != "stdio":
        # 二期能力：把"为什么没起来"说清楚，而不是悄悄按 stdio 跑
        return ("mcp.transport=%r 未实现：Streamable HTTP 属二期（端点 %s，"
                "远端来源白名单 %s）。一期只支持 stdio；需要远程调用请走 SSH "
                "隧道，或把 mcp.transport 改回 stdio。"
                % (transport, cfg.get("http_path"),
                   cfg.get("allow_remote_origin")))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rkllm-mcp")
    parser.add_argument("--transport", choices=("stdio",), default="stdio")
    sub = parser.add_subparsers(dest="sub")
    ask_p = sub.add_parser("ask", help="单轮排障调用")
    ask_p.add_argument("prompt")
    args = parser.parse_args(argv)
    if args.sub == "ask":
        return cli_ask(args.prompt)

    cfg = load_cfg()
    problem = _check_mode(cfg)
    if problem:
        disabled = not cfg.get("enabled", True)
        sys.stderr.write("[rkllm-mcp] %s\n" % problem)
        return 0 if disabled else 2

    # stdio：日志一律 stderr，stdout 只放协议
    server = build_server(cfg)
    sys.stderr.write("[rkllm-mcp] stdio up; endpoint=%s; versions=%s; "
                     "max_concurrent=%s\n"
                     % (server.cfg["endpoint"],
                        ",".join(server.cfg.get("protocol_versions") or []),
                        server.cfg.get("max_concurrent")))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
