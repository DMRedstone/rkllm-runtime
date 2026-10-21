"""rkllm-mcp 命令入口：纯环境变量可启动（WorkBuddy mcp.json 的 env 方式）。

用法：
  rkllm-mcp                       # stdio，读 RKLLM_ENDPOINT / RKLLM_API_KEY
  rkllm-mcp --transport stdio     # 显式
  rkllm-mcp ask "把这段改成公文体"  # 排障 CLI，输出纯文本
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .backend import HttpBackend, BackendError
from .stdio import StdioServer


def default_cfg(endpoint: str, api_key: str) -> dict:
    """配置缺省（80 键表的 mcp 组在客户端侧的子集；TOML 加载后合并）。"""
    return {
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
        "endpoint": endpoint,
        "api_key": api_key,
    }


def build_server() -> StdioServer:
    endpoint = os.environ.get("RKLLM_ENDPOINT", "http://127.0.0.1:8080/v1")
    api_key = os.environ.get("RKLLM_API_KEY", "")
    # 可选 TOML 覆盖（客户端路径 ~/.config/rkllm/mcp.toml，缺省纯环境变量）
    cfg = default_cfg(endpoint, api_key)
    toml_path = os.environ.get("RKLLM_MCP_CONFIG",
                               os.path.expanduser("~/.config/rkllm/mcp.toml"))
    if os.path.isfile(toml_path):
        try:
            import tomllib
            with open(toml_path, "rb") as fh:
                data = tomllib.load(fh)
            for key, value in (data.get("mcp") or {}).items():
                if key in cfg:
                    cfg[key] = value
        except (OSError, ValueError):
            pass
    backend = HttpBackend(endpoint, api_key,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rkllm-mcp")
    parser.add_argument("--transport", choices=("stdio",), default="stdio")
    sub = parser.add_subparsers(dest="sub")
    ask_p = sub.add_parser("ask", help="单轮排障调用")
    ask_p.add_argument("prompt")
    args = parser.parse_args(argv)
    if args.sub == "ask":
        return cli_ask(args.prompt)
    # stdio：日志一律 stderr，stdout 只放协议
    server = build_server()
    sys.stderr.write("[rkllm-mcp] stdio up; endpoint=%s\n" % server.cfg["endpoint"])
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
