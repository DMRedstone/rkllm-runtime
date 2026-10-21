"""Responses API ↔ Chat Completions 翻译层（emulated，02 §4）。

为什么要做：OpenAI Agents SDK 默认打 `POST /v1/responses`。板端只有 chat 语义，
所以这里把 Responses 请求翻译成 chat 请求，再把结果翻译回 Responses 结构。

能力边界（写进 caps，客户端必须能看到）：
- 支持：input（字符串或 item 数组）、instructions、max_output_tokens、
  temperature / top_p、tools / tool_choice、text.format（json_schema/json_object）、
  reasoning.effort、stream；
- 不支持并声明：`store`、`previous_response_id`、`include`、`truncation`、
  `background`、`conversation` —— 无状态服务没有服务端会话存储，收到即忽略并
  在响应头 `X-RKLLM-Unsupported` 里列出；
- 语义非等价：`reasoning` 只映射 effort，不产生 summary 事件。

纯函数在这里，路由在 server/routes/responses.py —— 转换逻辑可单测，不碰 HTTP。
"""

from __future__ import annotations

import time
import uuid

__all__ = ["to_chat_request", "response_id", "message_item", "function_call_item",
           "response_object", "IGNORED_REQUEST_FIELDS"]

# 无状态服务无法实现、但客户端可能默认带的字段：忽略并声明，不报错。
IGNORED_REQUEST_FIELDS = ("store", "previous_response_id", "include",
                          "truncation", "background", "conversation",
                          "prompt_cache_key", "safety_identifier")


def response_id() -> str:
    return "resp_" + uuid.uuid4().hex[:24]


def _item_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:24]


# ---------------------------------------------------------------------------
# Responses -> Chat
# ---------------------------------------------------------------------------

def _content_text(content) -> str:
    """Responses 的 content 可能是字符串或 part 数组。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # input_text / output_text / text 都有 text 字段
                if "text" in part:
                    parts.append(str(part["text"]))
                elif part.get("type") == "input_image":
                    parts.append("[image omitted]")
        return "".join(parts)
    return ""


def _input_to_messages(items) -> list[dict]:
    """把 Responses 的 input 转成 chat messages（含 tool 历史回填）。"""
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    messages: list[dict] = []
    pending_calls: list[dict] = []          # 尚未挂到 assistant 消息上的 function_call

    def flush_calls() -> None:
        if pending_calls:
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": list(pending_calls)})
            pending_calls.clear()

    for item in items or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type") or ("message" if "role" in item else "")
        if kind == "function_call":
            # 上一轮的助手工具调用：累积成一条 assistant.tool_calls
            pending_calls.append({
                "id": item.get("call_id") or item.get("id") or _item_id("call_"),
                "type": "function",
                "function": {"name": item.get("name", ""),
                             "arguments": item.get("arguments") or "{}"},
            })
            continue
        if kind == "function_call_output":
            flush_calls()
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("id") or "",
                "content": _content_text(item.get("output")),
            })
            continue
        if kind in ("message", "") and item.get("role"):
            flush_calls()
            messages.append({"role": item["role"],
                             "content": _content_text(item.get("content"))})
            continue
        # reasoning / item_reference 等：无状态服务无处安放，忽略
    flush_calls()
    return messages


def _tools_to_chat(tools) -> list[dict]:
    """Responses 的工具是扁平的；chat 的嵌在 function 里。"""
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") in (None, "function") and "name" in tool:
            out.append({"type": "function", "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters") or {"type": "object",
                                                         "properties": {}},
            }})
        # web_search / file_search / mcp 等内置工具：本服务不提供，跳过
    return out


def _tool_choice_to_chat(choice):
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = choice.get("name") or (choice.get("function") or {}).get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return choice


def _text_format_to_chat(text_cfg):
    """`text.format` → chat 的 response_format。"""
    if not isinstance(text_cfg, dict):
        return None
    fmt = text_cfg.get("format")
    if not isinstance(fmt, dict):
        return None
    kind = fmt.get("type")
    if kind == "json_schema":
        schema = fmt.get("schema")
        if isinstance(schema, dict):
            return {"type": "json_schema",
                    "json_schema": {"name": fmt.get("name") or "output",
                                    "schema": schema}}
    if kind == "json_object":
        return {"type": "json_object"}
    return None


def to_chat_request(body: dict) -> tuple[dict, list[str]]:
    """返回 (chat 请求体, 被忽略的字段名列表)。"""
    messages = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": str(instructions)})
    messages.extend(_input_to_messages(body.get("input")))

    chat: dict = {"model": body.get("model", "rkllm"), "messages": messages}
    if body.get("max_output_tokens"):
        chat["max_tokens"] = int(body["max_output_tokens"])
    for key in ("temperature", "top_p", "stream"):
        if body.get(key) is not None:
            chat[key] = body[key]
    tools = _tools_to_chat(body.get("tools"))
    if tools:
        chat["tools"] = tools
        if body.get("tool_choice") is not None:
            chat["tool_choice"] = _tool_choice_to_chat(body["tool_choice"])
        if body.get("parallel_tool_calls") is not None:
            chat["parallel_tool_calls"] = bool(body["parallel_tool_calls"])
    rf = _text_format_to_chat(body.get("text"))
    if rf:
        chat["response_format"] = rf
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        chat["reasoning_effort"] = reasoning["effort"]

    ignored = [k for k in IGNORED_REQUEST_FIELDS
               if body.get(k) not in (None, [], {}, False)]
    return chat, ignored


# ---------------------------------------------------------------------------
# Chat -> Responses
# ---------------------------------------------------------------------------

def message_item(text: str, *, item_id: str | None = None,
                 status: str = "completed") -> dict:
    return {
        "id": item_id or _item_id("msg_"),
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": ([{"type": "output_text", "text": text, "annotations": []}]
                    if status == "completed" else []),
    }


def function_call_item(call: dict, *, index: int = 0) -> dict:
    fn = call.get("function") or {}
    return {
        "id": call.get("id") or _item_id("fc_"),
        "type": "function_call",
        "status": "completed",
        "call_id": call.get("id") or _item_id("call_"),
        "name": fn.get("name", ""),
        "arguments": fn.get("arguments") or "{}",
    }


def _usage(chat_usage: dict | None) -> dict:
    chat_usage = chat_usage or {}
    prompt = int(chat_usage.get("prompt_tokens") or 0)
    completion = int(chat_usage.get("completion_tokens") or 0)
    out = {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if chat_usage.get("estimated"):
        out["estimated"] = True          # 无 tokenizer 时的估算标注（G5）
    return out


def response_object(*, resp_id: str, model: str, created_at: int,
                    status: str, output: list[dict],
                    usage: dict | None = None,
                    request_body: dict | None = None) -> dict:
    text = "".join(
        part.get("text", "")
        for item in output if item.get("type") == "message"
        for part in item.get("content") or [] if part.get("type") == "output_text")
    body = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
        "parallel_tool_calls": bool((request_body or {}).get("parallel_tool_calls",
                                                             True)),
        "tool_choice": (request_body or {}).get("tool_choice", "auto"),
        "tools": (request_body or {}).get("tools") or [],
        "error": None,
        "incomplete_details": None,
        "instructions": (request_body or {}).get("instructions"),
        "max_output_tokens": (request_body or {}).get("max_output_tokens"),
        "metadata": (request_body or {}).get("metadata") or {},
        "temperature": (request_body or {}).get("temperature"),
        "top_p": (request_body or {}).get("top_p"),
    }
    if usage is not None:
        body["usage"] = usage
    # SDK 的便捷字段：多数客户端直接读 output_text，不自己拼 content
    body["output_text"] = text
    return body


def chat_to_output(chat_payload: dict) -> tuple[list[dict], str]:
    """从 chat 完成体提取 (output items, finish_reason)。"""
    choice = (chat_payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict] = []
    calls = message.get("tool_calls") or []
    for call in calls:
        output.append(function_call_item(call))
    content = message.get("content") or ""
    if content or not calls:
        output.append(message_item(content))
    return output, choice.get("finish_reason") or "stop"


def now() -> int:
    return int(time.time())
