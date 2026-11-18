"""JSON Schema 校验（设计 §4.3 / F6「JSON 提取 → schema 校验 → 失败重试」）。

板端改用 `jsonschema`（Debian `python3-jsonschema`，随 deb Depends 安装）：
自制子集校验在实现 15 个关键字后仍不断遇到边界（`pattern`、`oneOf`、
`dependentRequired`、数组 tuple 形式 items……），而这些正是模型最容易踩的地方，
自写等价于持续补漏。这里把"校验语义"整体交给成熟实现，本模块只保留两件事：

1. **错误信息整形**：jsonschema 报英文 + 绝对路径，这里转成
   `$.a[0].b: <message>` 的单行列表，重试提示词直接可用。
2. **诚实降级**：schema 本身不合法（`SchemaError`）时不拦数据，返回空错误
   列表——不能因为调用方给了坏 schema 就把正常回答判死。

注意：这是**校验**，不是受约束解码（rkllm 不暴露 logits 钩子，做不到 GBNF）。
与 `/v1/capabilities` 的声明一致。

客户端连接器（`本地模型连接器/rkllm/agent/json_schema.py`）是**另一份零依赖副本**：
连接器要求纯 stdlib 源码交付，不能共用本模块。
"""
from __future__ import annotations

import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

__all__ = ["validate", "unsupported_keywords", "KNOWN_KEYWORDS"]

# 校验关键字 + 纯注解关键字（后者不参与校验，但属于合法 schema，不该报"不认识"）
KNOWN_KEYWORDS = frozenset(Draft202012Validator.VALIDATORS) | frozenset({
    "$schema", "$id", "$anchor", "$defs", "definitions", "$ref", "$comment",
    "title", "description", "default", "examples", "deprecated",
    "readOnly", "writeOnly", "contentMediaType", "contentEncoding",
    "format",                       # 未装 format 检查器时不强制，但合法
    "Discriminator", "discriminator",
})

_ANNOTATION_ONLY = frozenset({"title", "description", "default", "examples",
                              "deprecated", "readOnly", "writeOnly", "$comment"})


def unsupported_keywords(schema: Any, path: str = "$") -> list[str]:
    """列出 schema 中 Draft 2020-12 不认识的关键字（含嵌套）。

    纯注解关键字不算"不支持"——它们不影响校验结果。
    """
    found: list[str] = []
    if not isinstance(schema, dict):
        return found
    for key, value in schema.items():
        if key not in KNOWN_KEYWORDS and key not in _ANNOTATION_ONLY:
            found.append("%s.%s" % (path, key))
        if key == "properties" and isinstance(value, dict):
            for name, sub in value.items():
                found.extend(unsupported_keywords(sub, "%s.%s" % (path, name)))
        elif key in ("$defs", "definitions") and isinstance(value, dict):
            for name, sub in value.items():
                found.extend(unsupported_keywords(sub, "%s.%s" % (path, name)))
        elif key == "items":
            if isinstance(value, dict):
                found.extend(unsupported_keywords(value, path + "[]"))
            elif isinstance(value, list):
                for idx, sub in enumerate(value):
                    found.extend(unsupported_keywords(
                        sub, "%s[%d]" % (path, idx)))
        elif key in ("allOf", "anyOf", "oneOf") and isinstance(value, list):
            for idx, sub in enumerate(value):
                found.extend(unsupported_keywords(
                    sub, "%s.%s[%d]" % (path, key, idx)))
    return found


def _location(error) -> str:
    """把 jsonschema 的绝对路径渲染成 `$.a[0].b` 形式。"""
    out = "$"
    for part in error.absolute_path:
        out += "[%d]" % part if isinstance(part, int) else ".%s" % part
    return out


def _message(error) -> str:
    """把常用校验器的英文消息转成中文；未覆盖的原样返回。

    重试提示词会把这些消息回灌给模型，中文与"人可读"的要求一致；未覆盖的
    少数校验器保留 jsonschema 原文，避免臆造失真。
    """
    validator = error.validator
    instance = error.instance
    limit = error.validator_value
    if validator == "required":
        missing = re.findall(r"'([^']+)'", error.message)
        return "缺少必填字段 %s" % (missing[0] if missing else error.message)
    if validator == "type":
        return "类型应为 %s，实际 %s" % (limit, type(instance).__name__)
    if validator == "enum":
        return "取值应为 %s 之一，实际 %r" % (limit, instance)
    if validator == "const":
        return "取值应为 %r，实际 %r" % (limit, instance)
    if validator == "additionalProperties":
        return "不允许的额外字段（%s）" % error.message
    if validator == "minimum":
        return "需 ≥ %s" % limit
    if validator == "maximum":
        return "需 ≤ %s" % limit
    if validator == "minLength":
        return "长度需 ≥ %s" % limit
    if validator == "maxLength":
        return "长度需 ≤ %s" % limit
    if validator == "minItems":
        return "元素数需 ≥ %s" % limit
    if validator == "maxItems":
        return "元素数需 ≤ %s" % limit
    if validator == "pattern":
        return "不匹配模式 %s" % limit
    return error.message


def validate(value: Any, schema: Any, path: str = "$") -> list[str]:
    """返回错误列表（空 = 通过）。错误信息面向模型与人都可读。

    `path` 仅为兼容旧签名保留；路径一律从 `$` 起算（调用方拿到的就是完整位置）。
    """
    if not isinstance(schema, dict) or not schema:
        return []
    try:
        validator = Draft202012Validator(schema)
    except SchemaError:
        return []                    # schema 不合法 → 不拦数据（诚实降级）
    try:
        errors = sorted(validator.iter_errors(value),
                        key=lambda e: (len(e.absolute_path), _location(e)))
    except Exception:                # 递归 schema 深度异常等，同样降级
        return []
    return ["%s: %s" % (_location(e), _message(e)) for e in errors]
