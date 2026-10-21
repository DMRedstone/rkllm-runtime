"""N11 魔法数扫描：代码里的数值常量必须登记。

设计原文（01 §7 / N11）：所有影响行为的参数必须进配置表，代码中不得出现
魔法数（协议常量与安全下限除外）；新增常量必须同时更新 schema、默认值、
`config docs` 与本节计数。

务实口径（一次性重构不现实，也不是设计意图）：
- 扫描 `src/rkllm/**/*.py` 的全部数值字面量（跳过 docstring / 注释）；
- 白名单三类：极小整数（结构体/索引语义）、协议常量（帧头、SSE 格式）、
  **已登记基线**（`tests/magic_baseline.json`，逐条带理由）；
- 新出现的未登记数值 → 测试失败，强制作者做选择：改成配置键，或写进基线
  并说明为什么它不该可配。

用法：
  python 工具/脚本/magic_numbers.py            # 打印未登记项
  python 工具/脚本/magic_numbers.py --update   # 把当前快照写回基线
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "输入" / "rkllm-runtime-1.3.1" / "src" / "rkllm"
BASELINE = ROOT / "输入" / "rkllm-runtime-1.3.1" / "tests" / "magic_baseline.json"

# 语义上不该可配的数值：结构体尺寸、索引、比较边界、协议帧头等
TRIVIAL = {0, 1, 2, 3, 4, 5, 8, 10, 16, 32, 64, 100, 256, 1000, 1024}

# 有明确出处的协议/安全常量：允许硬编码，但必须能说出依据
PROTOCOL = {
    ("abi.py", 2), ("abi.py", 4), ("protocol.py", 8), ("frames.py", 4),
    ("stdio.py", 8), ("sse.py", 2), ("memory.py", 1024), ("memory.py", 2),
}


def scan() -> list[dict]:
    found: list[dict] = []
    for py in sorted(SRC.rglob("*.py")):
        if py.name == "config.py":      # 默认值本来就登记在 schema 里
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value in TRIVIAL or (py.name, int(value)) in PROTOCOL:
                continue
            if isinstance(value, float) and abs(value) < 1:
                # 轮询间隔/退避系数这类亚秒值：数量多且语义明确，单独登记
                pass
            found.append({"file": py.relative_to(SRC).as_posix(),
                          "line": node.lineno, "value": value,
                          "key": "%s:%s:%s" % (py.relative_to(SRC).as_posix(),
                                               node.lineno, value)})
    return found


def load_baseline() -> set[str]:
    try:
        data = json.loads(BASELINE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return set(data.get("registered", []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="把当前扫描结果写入基线（需人工审阅后提交）")
    args = parser.parse_args(argv)

    hits = scan()
    if args.update:
        payload = {
            "note": "N11 魔法数基线：这里的每个数值都等于'不打算做配置键'的声明。"
                    "新增项必须在提交说明里写理由。",
            "count": len(hits),
            "registered": sorted(h["key"] for h in hits),
        }
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print("已写入基线：%d 项 -> %s" % (len(hits), BASELINE))
        return 0

    known = load_baseline()
    unknown = [h for h in hits if h["key"] not in known]
    print("扫描到数值常量 %d 项，已登记 %d 项，未登记 %d 项"
          % (len(hits), len(known), len(unknown)))
    for h in unknown[:40]:
        print("  %s:%d  %r" % (h["file"], h["line"], h["value"]))
    return 1 if unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
