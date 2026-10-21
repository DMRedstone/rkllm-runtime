#!/usr/bin/env python3
"""构建客户端 MCP wheel：rkllm_mcp-<ver>-py3-none-any.whl（改造设计 §3.4）。

裁剪口径（单一源码树，避免双份实现漂移）：
  只打包 rkllm/{__init__,caps,config}.py + rkllm/mcp/*——纯 stdlib，
  Windows/Linux 通用；server/worker/agent/abi/cli 是板端部分，不进 wheel。

用法：
  python 工具/编译工具链/make_mcp_wheel.py [--version 1.0.0] [--out dist]
产物：dist/rkllm_mcp-<ver>-py3-none-any.whl
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "输入" / "rkllm-runtime-1.3.1" / "src"

# 客户端子集白名单（相对 rkllm/ 的路径）
CLIENT_FILES = ["__init__.py", "caps.py", "config.py"]
CLIENT_DIRS = ["mcp"]

STAGING_PYPROJECT = '''\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "rkllm-mcp"
version = "{version}"
description = "WorkBuddy MCP connector for the rkllm board-side model service"
requires-python = ">=3.11"
license = {{ text = "MIT" }}

[project.scripts]
rkllm-mcp = "rkllm.mcp:main"

[tool.setuptools.packages.find]
where = ["src"]
'''


def build(version: str, out_dir: Path) -> Path:
    if not SRC.is_dir():
        sys.exit(f"source tree not found: {SRC}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rkllm-mcp-wheel-") as tmp:
        staging = Path(tmp) / "src"
        pkg = staging / "rkllm"
        pkg.mkdir(parents=True)
        for name in CLIENT_FILES:
            shutil.copy2(SRC / "rkllm" / name, pkg / name)
        for dirname in CLIENT_DIRS:
            shutil.copytree(SRC / "rkllm" / dirname, pkg / dirname,
                            ignore=shutil.ignore_patterns("__pycache__"))
        (staging.parent / "pyproject.toml").write_text(
            STAGING_PYPROJECT.format(version=version), encoding="utf-8")
        # 排除板端残留引用：client 子集不得 import server/worker/agent/abi
        _check_no_board_imports(pkg)
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", str(staging.parent),
             "--no-deps", "-w", str(out_dir), "--no-cache-dir"],
            check=True)
    produced = sorted(out_dir.glob(f"rkllm_mcp-{version}-*.whl"))
    if not produced:
        sys.exit("wheel not produced")
    return produced[-1]


def _check_no_board_imports(pkg: Path) -> None:
    """裁剪完整性自检：客户端子集里禁止出现板端模块的真实 import 语句。

    按 import 语句匹配而非子串——docstring 里「不碰 ctypes」这类说明文本
    不是依赖（首轮误报即此原因）。
    """
    import re
    pattern = re.compile(
        r"^\s*(?:from|import)\s+(rkllm\.(?:server|worker|agent|abi|cli)"
        r"|fastapi|uvicorn|pydantic|ctypes)\b", re.M)
    offenders = []
    for py in pkg.rglob("*.py"):
        for match in pattern.finditer(py.read_text(encoding="utf-8")):
            offenders.append(f"{py.name}: {match.group(1)}")
    if offenders:
        sys.exit("client subset imports board-side modules:\n  "
                 + "\n  ".join(offenders))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--out", default=str(REPO / "dist"))
    args = parser.parse_args()
    wheel = build(args.version, Path(args.out))
    print(wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
