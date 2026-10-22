# -*- coding: utf-8 -*-
"""render_caps_doc.py —— 由 caps.py（单一事实源）渲染 OpenAI 支持矩阵 Markdown。

对话接口用法.md 的 §8 支持矩阵由本脚本生成；caps.py 变更后重跑一次即可，
**不要手改那个表格**（会与 caps.py 漂移，正是 caps 单一事实源要消灭的东西）。

用法（仓库根目录）：

    PYTHONPATH=输入/rkllm-runtime-1.3.1/src python 工具/编译工具链/render_caps_doc.py \
        --check 文档/对话接口用法.md     # 只校验文档里的矩阵是否与 caps.py 一致
    PYTHONPATH=... python 工具/编译工具链/render_caps_doc.py --append 文档/对话接口用法.md
                                     # 把矩阵追加到文档末尾
"""
from __future__ import annotations

import argparse
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "输入", "rkllm-runtime-1.3.1", "src"))
sys.path.insert(0, SRC)

from rkllm.caps import CAPS, NOTES, LIMITS   # noqa: E402

HEADER = "## 10. OpenAI 参数支持矩阵（新栈）"


def render() -> str:
    L = []
    L.append(HEADER)
    L.append("")
    L.append("> 本节由单一事实源 `src/rkllm/caps.py` 渲染生成（能力三档：native / emulated /")
    L.append("> ignored / unsupported）。caps.py 变更后重新生成即可，不要手改表格。")
    L.append("> 重新生成：`PYTHONPATH=输入/rkllm-runtime-1.3.1/src python 工具/编译工具链/render_caps_doc.py`")
    L.append("")
    L.append("### 端点")
    L.append("")
    L.append("| 端点 | 档位 |")
    L.append("|---|---|")
    for k, v in CAPS["endpoints"].items():
        L.append("| `%s` | %s |" % (k, v))
    L.append("")
    L.append("### 请求参数")
    L.append("")
    L.append("| 参数 | 档位 | 说明 |")
    L.append("|---|---|---|")
    for tier in ("native", "emulated", "ignored", "unsupported"):
        for p in CAPS["params"][tier]:
            note = NOTES.get(p.split(".")[-1] if p.startswith("response_format.") else p, "")
            L.append("| `%s` | %s | %s |" % (p, tier, note))
    L.append("")
    L.append("### 运行上限（schema 兜底值，可被 config 覆盖）")
    L.append("")
    L.append("| 项 | 默认 |")
    L.append("|---|---|")
    for k, v in LIMITS.items():
        L.append("| %s | %s |" % (k, v))
    L.append("")
    L.append("> unsupported 项的默认策略见 `openai.param_policy`：reject 全部 400；warn 时无害项")
    L.append("> 放行并标 `X-RKLLM-Unsupported` 头，但语义关键项（`response_format.json_schema`、")
    L.append("> `tool_choice`、`n>1`）即便 warn 也拒绝。")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", metavar="DOC", help="校验文档已含与 caps.py 一致的矩阵")
    ap.add_argument("--append", metavar="DOC", help="把矩阵追加到文档末尾")
    args = ap.parse_args()
    if not args.check and not args.append:
        sys.stdout.write(render())
        return 0

    doc = io.open(args.check or args.append, encoding="utf-8").read()
    block = render().strip()
    if args.check:
        if HEADER not in doc:
            print("FAIL: 文档缺少支持矩阵章节")
            return 1
        if block not in doc:
            print("FAIL: 文档中的支持矩阵与 caps.py 不一致（caps.py 变了，请重新渲染）")
            return 1
        print("OK: 支持矩阵与 caps.py 一致（%d 参数 / %d 端点）"
              % (sum(len(v) for v in CAPS["params"].values()), len(CAPS["endpoints"])))
        return 0

    if HEADER in doc:
        print("SKIP: 文档已有该章节（如需更新请先手工移除旧章节，或用 --check 校验）")
        return 1
    with io.open(args.append, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n" + block)
    print("已追加支持矩阵（%d 参数 / %d 端点）到 %s"
          % (sum(len(v) for v in CAPS["params"].values()), len(CAPS["endpoints"]), args.append))
    return 0


if __name__ == "__main__":
    sys.exit(main())
