#!/usr/bin/env python3
"""重组《开发记录》：按时间顺序 + 开发版本分卷分章，取消混乱分章。

输入：文档/开发记录.md（3041 行，编号章节与日期章节混排、缺「十二」、
      「十五」编号冲突、附录内混入本应属主线的实施记录）。
输出：同路径新文件；原文件已备份 /tmp/开发记录.md.bak。

策略：按行号切片 → 建立「旧块 → 新卷/章」映射 → 重新拼接。
不改动块内任何文字，只新增卷/章标题行，确保内容零丢失。
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path("文档/开发记录.md")

# (起, 止) 行号为闭区间，1-based
BLOCKS = {
    "header":        (1, 15),
    # 第一卷：打包工具链（1.3.1-1 ~ 1.3.1-12）
    "s1":            (16, 465),
    # 第二卷：本地模型连接器（独立支线）
    "conn_head":     (466, 514),      # 原九、十
    "conn_dates":    (1198, 1636),    # 原二十四说明 + 各日期章节
    "conn_appendix": (2839, 2939),    # 附录 A 内 §九~§十二
    # 第三卷：1.3.1-13
    "s3":            (515, 702),
    # 第四卷：1.3.1-14
    "s4":            (703, 771),
    # 第五卷：1.3.1-15 六轮修复
    "s5":            (772, 1064),
    # 第六卷：1.3.1-15 上板与测试
    "s6":            (1065, 1169),
    # 第七卷：板端服务改造 1.3.1-16
    "s7_start":      (1170, 1197),    # 原二十三 工程启动
    "s7_main":       (1637, 2213),    # 各日期续篇
    "s7_appendix":   (2940, 3005),    # 附录 A 内 §十三~§十四
    "s7_root":       (3006, 3041),    # 根路径兼容
    # 附录 A：改造计划（§〇~§八）
    "appendixA":     (2214, 2838),
}


def slice_lines(lines: list[str], start: int, end: int) -> list[str]:
    """1-based 闭区间切片；越界返回空而不是报错。"""
    if start < 1 or end > len(lines) or start > end:
        return []
    return lines[start - 1:end]


def main() -> int:
    raw = SRC.read_text(encoding="utf-8")
    lines = raw.splitlines()
    total = len(lines)
    print(f"源文件 {total} 行")

    out: list[str] = []

    def emit(text: str = "") -> None:
        out.append(text)

    def emit_block(key: str) -> None:
        start, end = BLOCKS[key]
        chunk = slice_lines(lines, start, end)
        if not chunk:
            print(f"  [警告] 块 {key} 区间 {start}-{end} 为空")
            return
        out.extend(chunk)
        print(f"  块 {key:<14} {start:>5}-{end:<5} -> {len(chunk):>4} 行")

    # -- 卷首 ---------------------------------------------------------------
    emit_block("header")
    emit()
    emit("---")
    emit()
    emit("> **编排说明（2026-10-22 重组）**：本记录原为「编号章节」与「日期章节」"
         "混排，缺「十二」、「十五」编号冲突，附录内又混入本应属主线的实施记录。"
         "现按**时间顺序 + 开发版本**重编为七卷：")
    emit("> ")
    emit("> - **第一卷** 打包工具链与工程基础（1.3.1-1 ~ 1.3.1-12）")
    emit("> - **第二卷** 本地模型连接器（2026-10-15 ~ 10-22，独立支线）")
    emit("> - **第三卷** 审查驱动的健壮性与安全加固（1.3.1-13）")
    emit("> - **第四卷** 服务端 agent 客户端兼容改造（1.3.1-14）")
    emit("> - **第五卷** 第三方工程审查修复（1.3.1-15 六轮）")
    emit("> - **第六卷** 1.3.1-15 上板部署与全量测试")
    emit("> - **第七卷** 板端服务改造（1.3.1-16）")
    emit("> - **附录** 改造计划、审计报告结论存档、待办核对")
    emit("> ")
    emit("> 各章内的小节编号为原有编号，未随卷重排，以保留与 git 提交历史的对应。")
    emit()
    emit("---")
    emit()

    # -- 第一卷 -------------------------------------------------------------
    emit("# 第一卷 工程基础与打包工具链（1.3.1-1 ~ 1.3.1-12）")
    emit()
    emit_block("s1")

    # -- 第二卷：连接器 -----------------------------------------------------
    emit()
    emit("---")
    emit()
    emit("# 第二卷 本地模型连接器（2026-10-15 ~ 10-22，独立支线）")
    emit()
    emit_block("conn_head")
    emit()
    emit("---")
    emit()
    emit_block("conn_dates")
    emit()
    emit("---")
    emit()
    emit_block("conn_appendix")

    # -- 第三卷 -------------------------------------------------------------
    emit()
    emit("---")
    emit()
    emit("# 第三卷 审查驱动的健壮性与安全加固（1.3.1-13，2026-10-19）")
    emit()
    emit_block("s3")

    # -- 第四卷 -------------------------------------------------------------
    emit()
    emit("---")
    emit()
    emit("# 第四卷 服务端 agent 客户端兼容改造（1.3.1-14，2026-10-20）")
    emit()
    emit_block("s4")

    # -- 第五卷 -------------------------------------------------------------
    emit()
    emit("---")
    emit()
    emit("# 第五卷 第三方工程审查修复（1.3.1-15，2026-10-20 晚 ~ 深夜）")
    emit()
    emit_block("s5")

    # -- 第六卷 -------------------------------------------------------------
    emit()
    emit("---")
    emit()
    emit("# 第六卷 1.3.1-15 上板部署与全量测试（2026-10-20 深夜 ~ 10-21 凌晨）")
    emit()
    emit_block("s6")

    # -- 第七卷 -------------------------------------------------------------
    emit()
    emit("---")
    emit()
    emit("# 第七卷 板端服务改造（1.3.1-16，2026-10-21 ~ 10-22）")
    emit()
    emit_block("s7_start")
    emit()
    emit("---")
    emit()
    emit_block("s7_main")
    emit()
    emit("---")
    emit()
    emit_block("s7_appendix")
    emit()
    emit("---")
    emit()
    emit_block("s7_root")

    # -- 附录 ---------------------------------------------------------------
    emit()
    emit("---")
    emit()
    emit("# 附录")
    emit()
    emit_block("appendixA")

    # -- 落盘 ---------------------------------------------------------------
    new_text = "\n".join(out)
    if raw.endswith("\n"):
        new_text += "\n"
    SRC.write_text(new_text, encoding="utf-8")

    new_lines = new_text.splitlines()
    print(f"\n输出 {len(new_lines)} 行（源 {total} 行，"
          f"新增 {len(new_lines) - total} 行标题/分隔）")

    # 覆盖校验：源每一行都必须出现在输出中
    src_set = [l for l in lines if l.strip()]
    new_set = [l for l in new_lines if l.strip()]
    missing = []
    new_iter = iter(new_set)
    new_pool = list(new_set)
    for line in src_set:
        if line in new_pool:
            new_pool.remove(line)
        else:
            missing.append(line)
    if missing:
        print(f"[失败] 有 {len(missing)} 行未保留，示例：")
        for m in missing[:10]:
            print("   ", m[:100])
        return 1
    print(f"[通过] 源文件全部 {len(src_set)} 个非空行均已保留")
    return 0


if __name__ == "__main__":
    sys.exit(main())
