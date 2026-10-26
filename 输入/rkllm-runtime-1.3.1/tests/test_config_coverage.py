"""N11 配置完备性守卫：每个键都必须有真实的读取点。

审计教训：`implemented=True` 是人工标记，不可信——上一轮按标记报「80/80 全
生效」，实际扫描发现 9 个键标了已接线却没有任何读取点。这里用机械口径：

- 属性式读取：`cfg.<section>.<name>`（含 `getattr(cfg.<section>, "<name>")`）；
- 字面量读取：`"<section>.<name>"`；
- 扁平等价：连接器的配置是扁平 dict，键名本身出现即算（`cfg["enabled"]`、
  `cfg.get("tool_prefix")`）——只在该键所属模块目录内判定，避免"别的模块
  恰好有个同名字符串"造成假阳性。

另外守一条 N11 的硬要求：代码里不得出现未登记的魔法数（见
`tests/test_magic_numbers.py`）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.config import SCHEMA  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src" / "rkllm"
# MCP 连接器已移到根目录独立源码包（客户端交付），配置读取点在那边
CONNECTOR_MCP = (Path(__file__).resolve().parents[3]
                 / "本地模型连接器" / "rkllm" / "mcp")


def _corpus(*dirs: str) -> str:
    files = []
    for d in dirs:
        base = SRC if d == "." else SRC / d
        files.extend(p for p in base.rglob("*.py") if p.name != "config.py")
    return "\n".join(p.read_text(encoding="utf-8") for p in files)


def _connector_corpus() -> str:
    if not CONNECTOR_MCP.is_dir():
        return ""
    return "\n".join(p.read_text(encoding="utf-8")
                     for p in CONNECTOR_MCP.rglob("*.py"))


def _has_reader(key, whole: str, scoped: str) -> bool:
    attr = re.escape(key.section) + r"\.\s*" + re.escape(key.name) + r"\b"
    if re.search(attr, whole):
        return True
    if re.search(r"[\"']" + re.escape(key.full) + r"[\"']", whole):
        return True
    getattr_pat = (r"getattr\([^)]*" + re.escape(key.section)
                   + r"[^)]*[\"']" + re.escape(key.name) + r"[\"']")
    if re.search(getattr_pat, whole):
        return True
    # 扁平读取（连接器）：只在所属目录范围内算
    if key.section == "mcp":
        return bool(re.search(r"[\"']" + re.escape(key.name) + r"[\"']", scoped))
    return False


def test_every_key_has_a_reader():
    whole = _corpus(".")
    mcp_scope = _connector_corpus()
    missing = [k.full for k in SCHEMA if not _has_reader(k, whole, mcp_scope)]
    assert missing == [], "配置键没有读取点（键表承诺了却不生效）：%s" % missing


def test_no_key_is_marked_implemented_while_unwired():
    """`implemented=False` 只允许出现在"确实不实现"的键上，且必须是显式声明。

    queue.max_concurrent 是唯一例外：单 NPU 恒夹紧为 1，配置值不生效，
    故诚实标注未接线（10-25 审计启用该机制）。新增其他未接线键必须改这里。
    """
    unwired = [k.full for k in SCHEMA if not k.implemented]
    assert unwired == ["queue.max_concurrent"]


def test_key_count_is_frozen():
    """83 键是对外契约（文档、conffile 注释、README 都引用了这个数字）。"""
    assert len(SCHEMA) == 83
