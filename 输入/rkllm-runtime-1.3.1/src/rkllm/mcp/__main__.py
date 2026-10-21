"""rkllm-mcp 命令入口：纯环境变量可启动（WorkBuddy mcp.json 的 env 方式）。

用法：
  rkllm-mcp                       # stdio，读 RKLLM_ENDPOINT / RKLLM_API_KEY
  rkllm-mcp --transport stdio     # 显式
  rkllm-mcp ask "把这段改成公文体"  # 排障 CLI，输出纯文本

配置（mcp.toml / 环境变量）覆盖 80 键表的 [mcp] 组。与本进程相关的键都在
default_cfg 里；其中 `enabled` / `transport` / `http_path` /
`allow_remote_origin` 属于"模式选择"，在 main() 里判定——不实现的模式显式
拒绝并说明，而不是静默按 stdio 跑。
"""

from __future__ import annotations

import argparse
import os
import sys

from .backend import BackendError, HttpBackend
from .stdio import StdioServer

DEFAULT_PROTOCOL_VERSIONS = ("2026-07-28", "2025-11-25", "2025-06-18",
                             "2024-11-05")


def default_cfg(endpoint: str, api_key: str) -> dict:
    """配置缺省（80 键表的 mcp 组在客户端侧的子集；TOML 加载后合并）。"""
    return {
        # -- 模式选择 ------------------------------------------------------
        "enabled": True,                 # mcp.enabled：连接器总开关
        "transport": "stdio",            # mcp.transport：一期只支持 stdio
        "http_path": "/mcp",             # mcp.http_path：二期
        "allow_remote_origin": [],       # mcp.allow_remote_origin：二期
        "protocol_versions": list(DEFAULT_PROTOCOL_VERSIONS),
        "max_concurrent": 1,             # mcp.max_concurrent：在飞工具调用数
        # -- 工具行为 ------------------------------------------------------
        "tool_prefix": "local_llm_",
        "default_max_tokens": 512,
        "default_temperature": 0.3,
        "max_input_chars": 12000,
        "max_result_chars": 20000,
        "batch_max_segments": 20,
        "task_mode": "auto",
        "task_threshold_s": 45,
        "job_ttl_s": 600,
        "job_max": 8,
        "job_poll_hint_s": 2,
        "progress_interval_s": 5,
        "extract_retries": 2,
        # -- 目标 ----------------------------------------------------------
        "target_timeout_s": 120,
        "target_endpoint": endpoint,
        "target_api_key": api_key,
        # 内部字段（与 target_* 同值；键表用 target_* 命名）
        "endpoint": endpoint,
        "api_key": api_key,
    }


def load_cfg() -> dict:
    """环境变量 + 可选 TOML 覆盖。TOML 只覆盖已知键（未知键忽略）。"""
    endpoint = os.environ.get("RKLLM_ENDPOINT", "http://127.0.0.1:8080/v1")
    api_key = os.environ.get("RKLLM_API_KEY", "")
    cfg = default_cfg(endpoint, api_key)
    toml_path = os.environ.get("RKLLM_MCP_CONFIG",
                               os.path.expanduser("~/.config/rkllm/mcp.toml"))
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
        except (OSError, ValueError):
            pass
    return cfg


def build_server(cfg: dict | None = None) -> StdioServer:
    cfg = cfg or load_cfg()
    backend = HttpBackend(cfg["endpoint"], cfg.get("api_key", ""),
                          timeout_s=int(cfg.get("target_timeout_s", 120)))
    return StdioServer(backend, cfg,
                       model_class=os.environ.get("RKLLM_MODEL_CLASS", "small"))


def cli_ask(prompt: str) -> int:
    """排障 CLI：单轮调用，输出纯文本（不是任何客户端的兜底）。"""
    server = build_server()
    try:
        result = server._handler_for("ask")({"prompt": prompt})
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
