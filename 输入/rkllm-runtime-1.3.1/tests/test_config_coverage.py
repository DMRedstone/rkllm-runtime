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


def _corpus(*dirs: str) -> str:
    files = []
    for d in dirs:
        base = SRC if d == "." else SRC / d
        files.extend(p for p in base.rglob("*.py") if p.name != "config.py")
    return "\n".join(p.read_text(encoding="utf-8") for p in files)


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
    mcp_scope = _corpus("mcp")
    missing = [k.full for k in SCHEMA if not _has_reader(k, whole, mcp_scope)]
    assert missing == [], "配置键没有读取点（键表承诺了却不生效）：%s" % missing


def test_no_key_is_marked_implemented_while_unwired():
    """`implemented=False` 只允许出现在"确实不实现"的键上，且必须是显式声明。"""
    unwired = [k.full for k in SCHEMA if not k.implemented]
    # 当前 80 键全部接线；将来若要加回"未接线"键，必须同时改这条断言并在
    # config show/docs 里标出来（不静默）。
    assert unwired == []


def test_key_count_is_frozen():
    """80 键是对外契约（文档、conffile 注释、README 都引用了这个数字）。"""
    assert len(SCHEMA) == 80
