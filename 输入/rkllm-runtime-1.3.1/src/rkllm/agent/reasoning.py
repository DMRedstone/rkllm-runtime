"""思考段分离（改造设计 §6.4(5)）：think 标签 → reasoning_content。"""

from __future__ import annotations

__all__ = ["ReasoningSplitter"]

_TAGS = ("<think>", "</think>")


class ReasoningSplitter:
    """流式把 <think>…</think> 路由到 reasoning 通道，其余到 content。

    半个标签跨 chunk 时缓冲待定文本；结束时若仍在 think 段，
    剩余内容视为 reasoning 下发（不吞内容）。
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._in_think = False
        self._pending = ""

    def feed(self, text: str) -> tuple[str | None, str | None]:
        """返回 (reasoning_delta | None, content_delta | None)。"""
        if not self.enabled:
            return None, text
        self._pending += text
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        while self._pending:
            if self._in_think:
                end = self._pending.find(_TAGS[1])
                if end < 0:
                    # 留住可能的半个 </think> 前缀
                    keep = self._safe_hold(self._pending, _TAGS[1])
                    emit = self._pending[:len(self._pending) - keep]
                    if emit:
                        reasoning_parts.append(emit)
                    self._pending = self._pending[len(self._pending) - keep:]
                    break
                reasoning_parts.append(self._pending[:end])
                self._pending = self._pending[end + len(_TAGS[1]):]
                self._in_think = False
            else:
                start = self._pending.find(_TAGS[0])
                if start < 0:
                    keep = self._safe_hold(self._pending, _TAGS[0])
                    emit = self._pending[:len(self._pending) - keep]
                    if emit:
                        content_parts.append(emit)
                    self._pending = self._pending[len(self._pending) - keep:]
                    break
                if start > 0:
                    content_parts.append(self._pending[:start])
                self._pending = self._pending[start + len(_TAGS[0]):]
                self._in_think = True
        return ("".join(reasoning_parts) or None,
                "".join(content_parts) or None)

    def finish(self) -> tuple[str | None, str | None]:
        if not self._pending:
            return None, None
        text, self._pending = self._pending, ""
        if self._in_think:
            return text, None
        return None, text

    @staticmethod
    def _safe_hold(pending: str, tag: str) -> int:
        """pending 尾部与 tag 前缀可能重叠的最大长度（留住等待跨 chunk 命中）。"""
        for keep in range(min(len(tag) - 1, len(pending)), 0, -1):
            if pending.endswith(tag[:keep]):
                return keep
        return 0
