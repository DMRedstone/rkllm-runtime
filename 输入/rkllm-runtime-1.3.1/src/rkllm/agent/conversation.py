"""messages → prompt（修 D6：tool 历史与 assistant.tool_calls 不再整条丢弃）。

无 tokenizer 时退化为结构化文本转录（设计 §6.4(2) 允许的兜底），
但 tool 轮必须保留 tool_call_id 与工具名——父级 agent 每轮都带回 tool 结果，
丢掉等于每轮失忆。
"""

from __future__ import annotations

import json

__all__ = ["build_prompt"]


def _render_tool_call(call: dict) -> str:
    name = call.get("function", {}).get("name") or call.get("name") or ""
    args = call.get("function", {}).get("arguments")
    if not isinstance(args, str):
        args = json.dumps(args or {}, ensure_ascii=False)
    return '<tool_call>{"name": "%s", "arguments": %s}</tool_call>' % (name, args)


def build_prompt(messages: list[dict], *, system: str = "",
                 tools_json: str | None = None) -> str:
    """把 OpenAI messages 转录为单条 user prompt。

    - system：请求级 system 覆盖/合并引擎级（旧版双拼语义收敛为：请求级优先）；
    - assistant.tool_calls：按模型模板形状回填；
    - role=tool：带 tool_call_id 与内容回填；
    - content 数组：只取 text 项；图片 → ValueError（D2：不再静默丢弃）。
    """
    lines: list[str] = []
    if system:
        lines.append("System: " + system)
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") != "text":
                    raise ValueError(
                        "multimodal content (%s) is not supported; use text only"
                        % part.get("type"))
            content = "\n".join(t for t in texts if t)
        if role == "user":
            if content:
                lines.append("User: " + content)
        elif role == "assistant":
            calls = msg.get("tool_calls") or []
            if calls:
                rendered = "\n".join(_render_tool_call(c) for c in calls)
                lines.append("Assistant:\n" + rendered)
            elif content:
                lines.append("Assistant: " + content)
        elif role == "tool":
            call_id = msg.get("tool_call_id", "")
            name = msg.get("name", "")
            body = content if isinstance(content, str) else json.dumps(
                content, ensure_ascii=False)
            lines.append('Tool result (%s) for %s:\n%s'
                         % (call_id, name or "tool", body))
        elif role == "system":
            lines.append("System: " + (content or ""))
    lines.append("Assistant:")
    if tools_json:
        lines.insert(0, tools_json)
    return "\n\n".join(lines)
