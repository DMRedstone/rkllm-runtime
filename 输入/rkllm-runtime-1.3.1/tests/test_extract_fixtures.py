"""F6：20 条结构化抽取样本，schema 校验通过率 100%（fixture 级）。

设计验收原文（03 §12 第 6 条）：「`local_llm_extract` 对 20 条样本的 schema
校验通过率 100%（fixture 级）」——即模型输出可能带代码块、前后缀、多余字段，
抽取链路必须全部吃下；而真正不合规的输出必须被拒并触发重试。

每条样本只声明「应通过 / 应拒绝」，不追求某个模型的真实输出；模型侧的质量
由 F3 的场景验收负责。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.agent.json_schema import validate   # noqa: E402
from rkllm.mcp.tools import _extract_json       # noqa: E402

# 通用 schema：公文要素抽取
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "doc_number": {"type": "string"},
        "date": {"type": "string"},
        "units": {"type": "array", "items": {"type": "string"}},
        "urgent": {"type": "boolean"},
    },
    "required": ["title", "date"],
}

GOOD = {
    "title": "关于加强建筑垃圾管理的通知",
    "doc_number": "柞政发〔2026〕19号",
    "date": "2026-10-21",
    "units": ["县住建局", "县生态环境局"],
    "urgent": False,
}

# (名称, 模型原始输出, 是否应通过)
SAMPLES: list[tuple[str, str, bool]] = [
    ("纯 JSON 对象", json.dumps(GOOD, ensure_ascii=False), True),
    ("```json 代码块", "```json\n%s\n```" % json.dumps(GOOD, ensure_ascii=False), True),
    ("无语言标记代码块", "```\n%s\n```" % json.dumps(GOOD, ensure_ascii=False), True),
    ("前置说明文字", "好的，抽取结果如下：\n%s" % json.dumps(GOOD, ensure_ascii=False), True),
    ("后置说明文字", "%s\n以上为抽取结果。" % json.dumps(GOOD, ensure_ascii=False), True),
    ("前后都有说明", "按 schema 抽取：\n%s\n如需调整请告知。" % json.dumps(GOOD, ensure_ascii=False), True),
    ("前后空白与空行", "\n\n  %s  \n\n" % json.dumps(GOOD, ensure_ascii=False), True),
    ("多余字段", json.dumps({**GOOD, "extra": "ignored"}, ensure_ascii=False), True),
    ("嵌套对象字段", json.dumps({"title": "通知", "date": "2026-10-21",
                                 "meta": {"level": 2, "tags": ["a", "b"]}},
                                ensure_ascii=False), True),
    ("数组字段多元素", json.dumps({"title": "通知", "date": "2026-10-21",
                                   "units": ["A", "B", "C"]}, ensure_ascii=False), True),
    ("空数组字段", json.dumps({"title": "通知", "date": "2026-10-21",
                               "units": []}, ensure_ascii=False), True),
    ("布尔与中文", json.dumps({"title": "紧急通知", "date": "2026年10月21日",
                               "urgent": True}, ensure_ascii=False), True),
    ("字符串内转义引号", json.dumps({"title": '关于"建筑垃圾"的通知',
                                     "date": "2026-10-21"},
                                    ensure_ascii=False), True),
    ("字符串内换行", json.dumps({"title": "第一行\n第二行",
                                 "date": "2026-10-21"}, ensure_ascii=False), True),
    ("代码块+前后说明", "结果：\n```json\n%s\n```\n完成" % json.dumps(
        GOOD, ensure_ascii=False), True),
    # -- 以下应被拒（校验失败 → 触发重试）------------------------------
    ("缺必填字段 date", json.dumps({"title": "通知"}, ensure_ascii=False), False),
    ("类型错误", json.dumps({"title": 123, "date": "2026-10-21"},
                            ensure_ascii=False), False),
    ("数组元素类型错误", json.dumps({"title": "通知", "date": "2026-10-21",
                                     "units": [1, 2]}, ensure_ascii=False), False),
    ("布尔字段给了字符串", json.dumps({"title": "通知", "date": "2026-10-21",
                                        "urgent": "yes"}, ensure_ascii=False), False),
    ("根本不是 JSON", "抱歉，我无法完成该抽取。", False),
]


def test_sample_count():
    assert len(SAMPLES) == 20, "F6 要求 20 条样本，当前 %d" % len(SAMPLES)


@pytest.mark.parametrize("name,raw,should_pass", SAMPLES,
                         ids=[s[0] for s in SAMPLES])
def test_extract_sample(name, raw, should_pass):
    try:
        parsed = _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        assert not should_pass, "%s：应当能提取出 JSON，实际失败" % name
        return
    errors = validate(parsed, SCHEMA)
    if should_pass:
        assert errors == [], "%s：应通过校验，实际 %s" % (name, errors)
    else:
        assert errors, "%s：应被拒绝，实际通过了校验" % name


def test_pass_rate_is_100_percent_for_compliant_samples():
    compliant = [s for s in SAMPLES if s[2]]
    ok = 0
    for name, raw, _ in compliant:
        try:
            parsed = _extract_json(raw)
            if not validate(parsed, SCHEMA):
                ok += 1
        except (json.JSONDecodeError, ValueError):
            pass
    rate = ok / len(compliant)
    assert rate == 1.0, "合规样本通过率 %.0f%%（%d/%d）" % (rate * 100, ok,
                                                            len(compliant))
