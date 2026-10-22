"""pytest 全局路径：服务端源码 + 根目录独立 MCP 连接器。

连接器已从服务端 `src/rkllm/mcp/` 移出，放到根目录 `本地模型连接器/rkllm/`
（客户端独立源码交付）。这里把该目录挂进 `rkllm` 包的搜索路径，使 MCP 用例
仍以 `rkllm.mcp.*` 导入；两个包在客户端/板端分别部署，不会同时存在。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

import rkllm  # noqa: E402  (先由 src 提供 rkllm 服务端包)

CONNECTOR = HERE.parent.parent.parent / "本地模型连接器" / "rkllm"
if CONNECTOR.is_dir() and str(CONNECTOR) not in rkllm.__path__:
    rkllm.__path__.append(str(CONNECTOR))
