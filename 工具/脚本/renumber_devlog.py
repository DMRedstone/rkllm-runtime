#!/usr/bin/env python3
"""统一《开发记录》章编号为「卷.章」形式，消除缺号与重号。

在 restructure_devlog.py 输出的基础上执行：
- 39 个 `##` 章 + 6 个原附录内 `###` 小节（已随卷归位）统一改为 `## 卷.章 标题`；
- 章节内部既有的 `### 11.1` 式小节编号保持不变（与 git 提交历史对应）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path("文档/开发记录.md")

# 行号(1-based, 闭区间单行) -> (新编号, 是否由 ### 提升为 ##)
RENUMBER = {
    36:  ("1.1",  False),
    69:  ("1.2",  False),
    93:  ("1.3",  False),
    265: ("1.4",  False),
    326: ("1.5",  False),
    374: ("1.6",  False),
    423: ("1.7",  False),
    443: ("1.8",  False),
    491: ("2.1",  False),
    500: ("2.2",  False),
    543: ("2.3",  False),
    550: ("2.4",  False),
    592: ("2.5",  False),
    692: ("2.6",  False),
    742: ("2.7",  False),
    897: ("2.8",  False),
    949: ("2.9",  False),
    985: ("2.10", True),
    1005: ("2.11", True),
    1026: ("2.12", True),
    1048: ("2.13", True),
    1091: ("3.1", False),
    1284: ("4.1", False),
    1358: ("5.1", False),
    1411: ("5.2", False),
    1457: ("5.3", False),
    1499: ("5.4", False),
    1560: ("5.5", False),
    1628: ("5.6", False),
    1656: ("6.1", False),
    1696: ("6.2", False),
    1722: ("6.3", False),
    1766: ("7.1", False),
    1797: ("7.2", False),
    1890: ("7.3", False),
    1952: ("7.4", False),
    2004: ("7.5", False),
    2079: ("7.6", False),
    2131: ("7.7", False),
    2218: ("7.8", False),
    2285: ("7.9", False),
    1066: ("7.10", True),
    2377: ("7.11", True),
    2446: ("7.12", False),
    2487: ("附录 A", False),
}

# 标题里需要剥掉的旧前缀（按出现顺序尝试）
OLD_PREFIX = re.compile(
    r"^(?:#{2,3}\s*)?"
    r"(?:[一二三四五六七八九十]+、"
    r"|二十[一二三四五六]?、|十[一二三四五六七八九]?、|"
    r"\d{4}-\d{2}-\d{2}[^：]*：|"
    r"附录：)")


def strip_old_prefix(title: str, was_h3: bool) -> str:
    body = title.lstrip("#").strip()
    m = OLD_PREFIX.match(body)
    if m:
        body = body[m.end():]
    return body.lstrip("：:").strip()


def main() -> int:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    changed = 0
    for lineno, (new_id, was_h3) in sorted(RENUMBER.items()):
        idx = lineno - 1
        if idx >= len(lines):
            print(f"[失败] 行 {lineno} 超出文件范围")
            return 1
        old = lines[idx]
        if not old.startswith("## ") and not old.startswith("### "):
            print(f"[失败] 行 {lineno} 不是章节标题：{old[:60]!r}")
            return 1
        if was_h3 and not old.startswith("### "):
            print(f"[失败] 行 {lineno} 预期为 ### ，实际 {old[:40]!r}")
            return 1
        body = strip_old_prefix(old, was_h3)
        new = f"## {new_id} {body}" if new_id != "附录 A" \
            else f"## 附录 A {body}"
        lines[idx] = new
        changed += 1
        print(f"  {lineno:>5}: {old[:52]:<54} -> {new[:52]}")

    SRC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n已重编号 {changed} 个章节")

    # 校验：不得再有中文数字章号或裸日期章号
    leftover = []
    for i, line in enumerate(lines, 1):
        if line.startswith("## ") and not re.match(
                r"^## (\d+\.\d+ |附录 [A-Z] )", line):
            leftover.append((i, line[:60]))
    if leftover:
        print("[失败] 以下章节未取得合法编号：")
        for i, t in leftover:
            print(f"   {i}: {t}")
        return 1
    print("[通过] 所有章节均为「卷.章」或「附录 X」形式")
    return 0


if __name__ == "__main__":
    sys.exit(main())
