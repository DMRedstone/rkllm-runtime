"""工具调用：提示词注入 + 多模板流式解析（修 D1，不用 rkllm_set_function_tools）。

理由：该 API 在 handle 上永久生效且无关闭接口（D1）——一次带 tools 的请求
污染之后所有对话。按模板逐请求注入、流式解析 <tool_call> 块：
- 开始前的内容作为普通 content 下发；
- 块内只发 tool_calls 增量（首帧 name/arguments 首片，后续 arguments 片段）；
- arguments 一律 JSON 字符串（客户端自己 parse）；
- 解析失败降级为文本（R6），不吞内容。
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Iterator

__all__ = ["ToolCallParser", "PARSERS", "build_tools_prefix", "detect_parser"]

_CALL_RE = {
    "qwen": re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S),
    "hermes": re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S),
    "deepseek": re.compile(r"<\|tool▁calls▁begin\|>.*?<\|tool▁call▁begin\|>.*?function<\|tool▁sep\|>(\{.*?\})<\|tool▁call▁end\|>", re.S),
}

# 提示词前缀（设计 §6.4(1)；按 parser 家族微调闭合标记）
_TOOLS_PREFIX = (
    "You are a function-calling assistant.\n"
    "Available tools (JSON Schema):\n%s\n"
    "When you need a tool, reply ONLY with:\n"
    '<tool_call>{"name": "...", "arguments": {...}}</tool_call>\n'
    "Otherwise reply in plain text.\n\n"
)


def build_tools_prefix(tools: list[dict]) -> str:
    return _TOOLS_PREFIX % json.dumps(tools, ensure_ascii=False)


def detect_parser(model_name: str, configured: str = "auto") -> str:
    if configured != "auto":
        return configured
    low = model_name.lower()
    if "qwen" in low:
        return "qwen"
    if "deepseek" in low or "r1" in low:
        return "deepseek"
    if "llama" in low:
        return "llama3"
    return "qwen"          # hermes 兼容 <tool_call>，qwen 解析器可覆盖


class ToolCallParser:
    """流式解析：feed(content_delta) 产出事件；finish() 收尾。

    事件：("content", text) | ("tool_call", {"index","id","type","name",
    "arguments"} 片段 dict)。
    """

    def __init__(self, parser_name: str = "qwen", max_calls: int = 8):
        self.name = parser_name
        self.max_calls = max_calls
        self._buf = ""
        self._in_call = False
        self._calls: list[dict] = []
        self._parse_failed = False
        # P1-16：按 parser 家族选用真实起止标记，而非一律 <tool_call>
        _tags = {"qwen": ("<tool_call>", "</tool_call>"),
                 "hermes": ("<tool_call>", "</tool_call>"),
                 "deepseek": ("<|tool▁call▁begin|>", "<|tool▁call▁end|>"),
                 "llama3": ("<|python_tag|>", "<|eom_id|>")}
        self._open_tag, self._close_tag = _tags.get(
            parser_name, ("<tool_call>", "</tool_call>"))

    # -- 事件 ---

    def feed(self, delta: str) -> Iterator[tuple[str, Any]]:
        self._buf += delta
        while True:
            if not self._in_call:
                start = self._buf.find(self._open_tag)
                if start < 0:
                    # 保留半个 open_tag 前缀
                    keep = self._hold(self._buf, self._open_tag)
                    emit = self._buf[:len(self._buf) - keep]
                    self._buf = self._buf[len(self._buf) - keep:]
                    if emit:
                        yield "content", emit
                    return
                if start > 0:
                    yield "content", self._buf[:start]
                self._buf = self._buf[start + len(self._open_tag):]
                self._in_call = True
                # OpenAI 规范：name/arguments 嵌在 function 里。
                # 早期实现是扁平的（{"name":...}），LangGraph / Agents SDK 这类
                # 标准客户端读 tool_call["function"]["name"] 会拿到空值，
                # 报 "Tool  not found in agent"——F10 实机才暴露。
                self._calls.append({"index": len(self._calls),
                                    "id": "call_" + uuid.uuid4().hex[:8],
                                    "type": "function",
                                    "function": {"name": None,
                                                 "arguments": ""}})
                if len(self._calls) > self.max_calls:
                    self._parse_failed = True
                    yield "content", self._open_tag
                    self._in_call = False
                    return
            else:
                end = self._buf.find(self._close_tag)
                if end < 0:
                    return          # 等更多数据
                raw = self._buf[:end]
                self._buf = self._buf[end + len(self._close_tag):]
                self._in_call = False
                call = self._calls[-1]
                extracted = self._extract_call(raw)
                if extracted is None:
                    self._parse_failed = True
                    yield "content", self._open_tag + raw + self._close_tag
                    self._calls.pop()
                    continue
                name, args_str = extracted
                call["function"]["name"] = name
                call["function"]["arguments"] = args_str
                yield "tool_call", call

    def finish(self) -> Iterator[tuple[str, Any]]:
        """流结束：未闭合块尝试解析（llama3 无闭合标记），否则降级为文本。"""
        if self._in_call and self._buf:
            extracted = self._extract_call(self._buf)
            if extracted is not None:
                call = self._calls[-1]
                name, args_str = extracted
                call["function"]["name"] = name
                call["function"]["arguments"] = args_str
                self._buf = ""
                self._in_call = False
                yield "tool_call", call
                return
            self._calls.pop()
            yield "content", self._open_tag + self._buf
            self._buf = ""
            self._parse_failed = True
        elif self._buf:
            yield "content", self._buf
            self._buf = ""

    @property
    def parse_failed(self) -> bool:
        return self._parse_failed

    def tool_calls(self) -> list[dict]:
        """只返回解析出工具名的调用，且是 OpenAI 标准嵌套结构。"""
        return [c for c in self._calls
                if (c.get("function") or {}).get("name")]

    @staticmethod
    def _extract_call(raw: str):
        """从标签内文本提取 (name, arguments_json_str)；失败返回 None。

        兼容：qwen/hermes 纯 JSON；deepseek 在 <|tool▁sep|> 后跟 JSON；
        llama3 用 parameters 键而非 arguments。
        """
        text = raw.strip()
        if "<|tool▁sep|>" in text:
            text = text.rsplit("<|tool▁sep|>", 1)[-1].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None
        name = parsed.get("name")
        if name is None and isinstance(parsed.get("function"), dict):
            name = parsed["function"].get("name")
        if name is None:
            return None
        args = parsed.get("arguments")
        if args is None:
            args = parsed.get("parameters")
        if args is None:
            args = {}
        return str(name), (args if isinstance(args, str)
                           else json.dumps(args, ensure_ascii=False))

    @staticmethod
    def _hold(buf: str, tag: str) -> int:
        for keep in range(min(len(tag) - 1, len(buf)), 0, -1):
            if buf.endswith(tag[:keep]):
                return keep
        return 0


PARSERS = ("qwen", "hermes", "deepseek", "llama3")
