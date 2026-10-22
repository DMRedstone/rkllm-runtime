"""JSON Schema 子集校验（设计 §4.3 / F6「JSON 提取 → schema 校验 → 失败重试」）。

零第三方依赖：客户端 wheel 要求纯 stdlib，板端 PEP 668 禁 pip，所以不引
`jsonschema`。支持的子集覆盖辅助任务的实际用法：

    type / properties / required / additionalProperties(bool) / items /
    enum / minimum / maximum / minLength / maxLength / minItems / maxItems

不支持的关键字（$ref、allOf、anyOf、oneOf、patternProperties、format…）会被
忽略但记录在 `unsupported` 清单里，调用方可据此判断"校验强度不足"，而不是
误以为校验完整。

注意：这是**校验**，不是受约束解码。与 `/v1/capabilities` 的声明一致。
"""
from __future__ import annotations

from typing import Any

__all__ = ["validate", "unsupported_keywords", "KNOWN_KEYWORDS"]

KNOWN_KEYWORDS = frozenset({
    "type", "properties", "required", "additionalProperties", "items",
    "enum", "minimum", "maximum", "minLength", "maxLength",
    "minItems", "maxItems", "title", "description", "default",
})

_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    py = _TYPE_MAP.get(expected)
    if py is None:
        return True                 # 未知类型名不拦（诚实：不认识就不假装校验）
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return isinstance(value, py)


def unsupported_keywords(schema: Any, path: str = "$") -> list[str]:
    """列出 schema 中本校验器不认识的关键字（含嵌套）。"""
    found: list[str] = []
    if not isinstance(schema, dict):
        return found
    for key, value in schema.items():
        if key not in KNOWN_KEYWORDS:
            found.append("%s.%s" % (path, key))
        if key == "properties" and isinstance(value, dict):
            for name, sub in value.items():
                found.extend(unsupported_keywords(sub, "%s.%s" % (path, name)))
        elif key == "items":
            if isinstance(value, dict):
                found.extend(unsupported_keywords(value, path + "[]"))
            elif isinstance(value, list):
                for idx, sub in enumerate(value):
                    found.extend(unsupported_keywords(
                        sub, "%s[%d]" % (path, idx)))
    return found


def validate(value: Any, schema: Any, path: str = "$") -> list[str]:
    """返回错误列表（空 = 通过）。错误信息面向模型与人都可读。"""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors

    expected = schema.get("type")
    if isinstance(expected, list):
        if not any(_type_ok(value, t) for t in expected if isinstance(t, str)):
            errors.append("%s: 类型应为 %s，实际 %s"
                          % (path, "/".join(map(str, expected)),
                             type(value).__name__))
            return errors
    elif isinstance(expected, str) and not _type_ok(value, expected):
        errors.append("%s: 类型应为 %s，实际 %s"
                      % (path, expected, type(value).__name__))
        return errors

    if "enum" in schema and isinstance(schema["enum"], list):
        if value not in schema["enum"]:
            errors.append("%s: 取值应为 %s 之一，实际 %r"
                          % (path, schema["enum"], value))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append("%s: 需 ≥ %s" % (path, schema["minimum"]))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append("%s: 需 ≤ %s" % (path, schema["maximum"]))

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append("%s: 长度需 ≥ %s" % (path, schema["minLength"]))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append("%s: 长度需 ≤ %s" % (path, schema["maxLength"]))

    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                errors.append("%s: 缺少必填字段 %r" % (path, name))
        for name, sub in props.items():
            if name in value:
                errors.extend(validate(value[name], sub,
                                       "%s.%s" % (path, name)))
        if schema.get("additionalProperties") is False:
            extra = [k for k in value if k not in props]
            if extra:
                errors.append("%s: 不允许的额外字段 %s" % (path, extra))

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append("%s: 元素数需 ≥ %s" % (path, schema["minItems"]))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append("%s: 元素数需 ≤ %s" % (path, schema["maxItems"]))
        items = schema.get("items")
        if isinstance(items, dict):
            for idx, item in enumerate(value):
                errors.extend(validate(item, items, "%s[%d]" % (path, idx)))
        elif isinstance(items, list):
            for idx, sub in enumerate(items[:len(value)]):
                errors.extend(validate(value[idx], sub, "%s[%d]" % (path, idx)))

    return errors
