"""6 个 MCP 工具（MCP 设计 §4.1–§4.4）。

描述文案是给调用方模型看的提示词：能做什么、不能做什么、耗时、输出上限
四件事必须写清（沿用旧连接器 ABILITY_NOTE 的做法）。

工具层只依赖 Backend 协议与配置值，不碰 socket/ctypes。
长任务判定的编排在 stdio.py。

本批（task-8）整改：
- 同步调用的生成上限按 `max_tokens_cap` 夹紧（stdio 按阈值算出），默认参数
  不再恒转 job；被夹断时给 `finish_reason=length` 提示；
- 输入按板端上下文预算（`context_len`/`context_window`）裁剪，chat 会话累计
  预算（含完整历史场景），避免 4k+ 汉字稳定 400；
- 结果透传 `finish_reason` / `usage` / `reasoning_chars`（放在 `_meta`）；
  reasoning 模型预算耗尽导致空正文时返回 isError，不再“空成功”；
- 非法参数抛 `JsonRpcError(-32602)`（MCP InvalidParams），不再混成 -32603；
- `extract` 失败路径也返回宽松 outputSchema 兼容的 `structuredContent`。
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

from ..agent.json_schema import validate as validate_schema
from .protocol import INVALID_PARAMS, JsonRpcError

__all__ = ["build_handlers", "tool_definitions", "Cancelled"]

# 模式预设：辅助任务的提示词骨架（中文语境，保留英文关键词）
_MODE_PROMPTS = {
    "summarize": "对下面的文本做摘要，输出不超过 300 字，只保留关键信息：\n\n",
    "rewrite": "改写下面的文本，保持原意，使表达更清晰规范：\n\n",
    "extract": "从下面的文本中抽取要求的信息，直接给出条目：\n\n",
    "classify": "对下面的文本分类，先给类别名，再用一句话说明理由：\n\n",
    "translate": "翻译下面的文本，保持格式，不要添加解释：\n\n",
    "qa": "根据下面的文本回答问题；文本中没有依据就明确说不知道：\n\n",
}

_TOOL_DOC_COMMON = (
    "耗时通常 3–45 秒；估算超过阈值会返回 job_id，用 %sjob 轮询或取消。"
    "同步调用的生成上限会被夹到安全值（结果里 finish_reason=length 表示截断）。"
)


class Cancelled(Exception):
    """工具内部信号：任务被取消（由 stdio 转成 isError 文本）。"""


def _tool(name: str, desc: str, schema: dict, *,
          readonly: bool = False, output_schema: dict | None = None) -> dict:
    out = {"name": name, "description": desc, "inputSchema": schema}
    if output_schema is not None:
        out["outputSchema"] = output_schema
    if readonly:
        out["annotations"] = {"readOnlyHint": True}
    return out


def tool_definitions(prefix: str = "local_llm_") -> list[dict]:
    """6 个工具的定义（tools/list 返回值）。"""
    p = prefix
    job_hint = _TOOL_DOC_COMMON % p
    return [
        _tool(p + "ask",
              "单轮辅助任务：摘要、改写、分类、抽取、翻译、短问答。"
              "适合 < 2000 字的输入。\n"
              "不适合：长文创作、代码生成、多步推理、需要联网核实的事实问题。\n"
              + job_hint,
              {"type": "object", "properties": {
                  "prompt": {"type": "string", "description": "任务输入文本"},
                  "system": {"type": "string", "description": "可选系统提示词"},
                  "mode": {"type": "string",
                           "enum": sorted(_MODE_PROMPTS),
                           "description": "任务预设；不填则原样执行"},
                  "max_tokens": {"type": "integer",
                                 "description": "生成上限，默认 512；显式传大值会转 job"},
                  "temperature": {"type": "number"},
                  "format": {"type": "string", "enum": ["text", "json_object"],
                             "description": "json_object 仅追加 JSON 提示（非受约束解码）"}},
               "required": ["prompt"]}),
        _tool(p + "chat",
              "多轮对话：需要上下文跟进时用；messages 为完整历史。\n"
              "可选 session 由连接器记住最近几轮（MCP 无状态，会话显式传参）；"
              "带 session 时只需传新增消息。连接器会按板端上下文预算自动裁剪。\n"
              "不适合长文生成与多步推理。" + job_hint,
              {"type": "object", "properties": {
                  "messages": {"type": "array", "items": {
                      "type": "object", "properties": {
                          "role": {"type": "string"},
                          "content": {"type": "string"}},
                      "required": ["role", "content"]}},
                  "system": {"type": "string"},
                  "session": {"type": "string",
                              "description": "可选会话 id；同一 id 连续调用保留最近上下文"}},
               "required": ["messages"]}),
        _tool(p + "extract",
              "结构化抽取：按 JSON Schema 从文本抽取，返回 structuredContent。"
              "输出会按 schema 校验，不符会自动重试；仍失败时返回 isError + "
              "包含错误与原始输出的 structuredContent。\n"
              "这是提示+校验，不是受约束解码。" + job_hint,
              {"type": "object", "properties": {
                  "text": {"type": "string"},
                  "schema": {"type": "object", "description": "JSON Schema"},
                  "instructions": {"type": "string"}},
               "required": ["text", "schema"]},
              output_schema={"type": "object", "additionalProperties": True}),
        _tool(p + "batch",
              "分段批量：对 segments 逐段执行同一任务，串行执行，"
              '返回与输入等长的结果数组（失败段为 {"error": ...}，不影响其他段）。'
              "单次最多 20 段；整体结果超长时返回 {truncated,...} 信封。" + job_hint,
              {"type": "object", "properties": {
                  "segments": {"type": "array", "items": {"type": "string"}},
                  "system": {"type": "string"},
                  "mode": {"type": "string", "enum": sorted(_MODE_PROMPTS)}},
               "required": ["segments"]}),
        _tool(p + "status",
              "查看板端模型状态与能力：模型名、档位、上下文、队列、负载。"
              "只读，秒回。",
              {"type": "object", "properties": {}}, readonly=True),
        _tool(p + "job",
              "长任务轮询与取消：action=get 取结果，action=cancel 取消任务"
              "（取消会立即停止推理并释放 NPU）。",
              {"type": "object", "properties": {
                  "job_id": {"type": "string"},
                  "action": {"type": "string", "enum": ["get", "cancel"]}},
               "required": ["job_id"]}),
    ]


def _extract_json(text: str) -> Any:
    """宽容提取 JSON：容忍代码块、前后缀说明文字。

    工程取舍：不做"逐个 { 尝试解析"的回溯（成本高、边界多），而是取第一个
    `{` 到最后一个 `}`。对"多个 JSON 片段"的输出会取错，但那种输出本身
    就不符合"只输出一个 JSON 对象"的提示约定，重试会重新要求它。
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:]
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start:end + 1]
    return json.loads(candidate)


def build_handlers(backend, cfg: dict, jobs, *,
                   extract_retries: int = 2,
                   sessions: dict[str, list[dict]] | None = None,
                   sessions_lock: threading.Lock | None = None) -> dict[str, Callable]:
    """返回 {工具短名: handler(params, should_cancel=None, max_tokens_cap=None,
    timeout_s=None)->result dict}。

    cfg 需要：default_max_tokens / default_temperature / max_input_chars /
    max_result_chars / batch_max_segments；上下文预算读 `context_len`
    （stdio 从 /health 刷新）或 `context_window`（缺省 4096）。

    `max_tokens_cap`：同步路径由 stdio 传入的安全生成上限；job 路径为 None
    （按调用方请求/默认值生成）。`timeout_s`：本次调用的 HTTP 总读超时。
    """
    from .backend import BackendError

    def _invalid(message: str) -> None:
        """参数非法 → JSON-RPC -32602（MCP InvalidParams），不再是 -32603。"""
        raise JsonRpcError(INVALID_PARAMS, message)

    def _as_check(value):
        if value is None or callable(value):
            return value
        snapshot = bool(value)
        return lambda: snapshot

    input_limit = int(cfg.get("max_input_chars", 12000) or 12000)
    result_limit = int(cfg.get("max_result_chars", 20000) or 20000)
    session_store = sessions if sessions is not None else {}
    session_lock = sessions_lock or threading.Lock()
    max_session_messages = int(cfg.get("session_max_messages", 20))
    session_locks: dict[str, threading.Lock] = {}
    max_sessions = int(cfg.get("session_max_count", 256) or 256)
    # P2：会话空闲 TTL（秒），0=不过期。陈旧会话不回收会持续占用内存/上下文。
    session_ttl_s = float(cfg.get("session_ttl_s", 0) or 0)
    session_seen: dict[str, float] = {}
    default_context = int(cfg.get("context_window", 4096) or 4096)
    context_margin = int(cfg.get("context_margin_tokens", 32) or 32)

    def _tokens(text: str) -> int:
        """与板端同口径的粗估：UTF-8 字节数 / 3（agent/tokens.py）。"""
        if not text:
            return 0
        return max(1, len(text.encode("utf-8")) // 3)

    def _ctx_tokens() -> int:
        raw = cfg.get("context_len") or default_context
        try:
            ctx = int(raw)
        except (TypeError, ValueError):
            ctx = default_context
        return max(256, ctx - context_margin)

    def _out_tokens(requested: Any, cap: Any = None) -> int:
        try:
            out = int(requested)
        except (TypeError, ValueError):
            out = int(cfg.get("default_max_tokens", 512) or 512)
        out = max(1, out)
        if cap is not None:
            try:
                out = min(out, max(8, int(cap)))
            except (TypeError, ValueError):
                pass
        return out

    def _fit_text(text: str, budget_tokens: int, note: str = "输入") -> str:
        budget = max(16, int(budget_tokens))
        limit_bytes = budget * 3
        raw = text.encode("utf-8")
        if len(raw) <= limit_bytes:
            return text
        cut = raw[:limit_bytes].decode("utf-8", "ignore")
        return cut + "\n…[%s超出板端上下文预算，已截断（约 %d token）]" % (
            note, budget)

    def _limit_input(text: str, budget_tokens: int | None = None,
                     note: str = "输入") -> str:
        if len(text) > input_limit:
            text = text[:input_limit] + \
                "\n…[输入超限已截断，上限 %d 字符]" % input_limit
        if budget_tokens is not None:
            text = _fit_text(text, budget_tokens, note)
        return text

    def _clip_result(text: str) -> str:
        if len(text) > result_limit:
            return text[:result_limit] + \
                "\n…[结果超限已截断，上限 %d 字符]" % result_limit
        return text

    def _purge_expired_locked() -> None:
        """剔除空闲超过 TTL 的会话（调用方须已持有 session_lock）。"""
        if session_ttl_s <= 0:
            return
        now = time.monotonic()
        for name, seen in list(session_seen.items()):
            if now - seen > session_ttl_s:
                session_store.pop(name, None)
                session_seen.pop(name, None)
        # 外部注入但无时间戳的会话按当前时刻补记，避免永不清理
        for name in session_store:
            session_seen.setdefault(name, now)

    def _session_lock(name: str) -> "threading.Lock":
        """同一 session 的读→推理→写回必须串行，否则并发丢上下文（P2）。"""
        with session_lock:
            lk = session_locks.get(name)
            if lk is None:
                lk = threading.Lock()
                session_locks[name] = lk
            return lk

    def _chat_once(system: str, user: str, max_tokens: int,
                   temperature: float, should_cancel=None,
                   timeout_s: float | None = None) -> dict:
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        try:
            out = backend.chat(messages, max_tokens=max_tokens,
                               temperature=temperature,
                               cancel_check=_as_check(should_cancel),
                               timeout_s=timeout_s)
        except BackendError as exc:
            if getattr(exc, "kind", "") == "cancelled":
                raise Cancelled() from exc
            raise
        if not isinstance(out, dict):
            return {"text": str(out or ""), "usage": None,
                    "finish_reason": "stop", "reasoning_chars": 0}
        return out

    def _meta_of(out: dict, max_tokens_used: Any = None,
                 extra: dict | None = None) -> dict:
        meta: dict = {}
        finish = out.get("finish_reason")
        if finish:
            meta["finish_reason"] = finish
        usage = out.get("usage")
        if usage is not None:
            meta["usage"] = usage
        reasoning_chars = int(out.get("reasoning_chars") or 0)
        if reasoning_chars:
            meta["reasoning_chars"] = reasoning_chars
        if max_tokens_used is not None:
            meta["max_tokens"] = int(max_tokens_used)
        if extra:
            meta.update(extra)
        return meta

    def _answer_result(out: dict, used: int,
                       extra_meta: dict | None = None) -> dict:
        """把后端结果转成 CallToolResult；空正文/截断给明确信号。"""
        text = str(out.get("text") or "")
        finish = str(out.get("finish_reason") or "stop")
        meta = _meta_of(out, used, extra_meta)
        if not text.strip():
            if finish == "length":
                message = (
                    "本地模型未输出正文：生成预算 max_tokens=%d 被耗尽"
                    "（finish_reason=length，reasoning_chars=%d）。"
                    "请增大 max_tokens；需要长输出时直接传较大 max_tokens 让其转 job。"
                    % (int(used), int(out.get("reasoning_chars") or 0)))
            else:
                message = ("本地模型返回空响应（finish_reason=%s）；"
                           "请重试或调整提示词。" % finish)
            return {"content": [{"type": "text", "text": message}],
                    "isError": True, "_meta": meta}
        text = _clip_result(text)
        if finish == "length":
            text += "\n…[输出达到 max_tokens 截断（finish_reason=length）]"
        return {"content": [{"type": "text", "text": text}], "_meta": meta}

    def _fit_messages(messages: list[dict],
                      budget_tokens: int) -> tuple[list[dict], bool]:
        """按 token 预算保留最新的消息后缀；最后一条超预算时截断后保留。

        返回 (messages, truncated)。chat 的输入预算包含 system + 全部历史，
        否则“messages 为完整历史 + session”很容易把板端上下文挤爆（400）。
        """
        try:
            budget = max(16, int(budget_tokens))
        except (TypeError, ValueError):
            budget = 16
        total = sum(_tokens(m["content"]) for m in messages)
        if total <= budget:
            return list(messages), False
        kept: list[dict] = []
        remaining = budget
        truncated = False
        for message in reversed(messages):
            cost = _tokens(message["content"])
            if cost <= remaining:
                kept.insert(0, dict(message))
                remaining -= cost
            elif not kept:
                content = _fit_text(message["content"], remaining, "消息")
                kept.insert(0, {"role": message["role"], "content": content})
                truncated = True
            else:
                truncated = True
                break
        return kept, truncated

    def _temperature(params: dict) -> float:
        value = params.get("temperature")
        try:
            return float(value) if value is not None \
                else float(cfg.get("default_temperature", 0.3))
        except (TypeError, ValueError):
            _invalid("temperature 必须是数字")
            return 0.0                       # pragma: no cover（_invalid 必抛）

    def ask(params: dict, should_cancel=None, max_tokens_cap=None,
            timeout_s=None) -> dict:
        if not isinstance(params, dict):
            _invalid("arguments 必须是对象")
        prompt = str(params.get("prompt") or "")
        if not prompt.strip():
            _invalid("prompt 不能为空")
        mode = params.get("mode")
        if mode is not None and mode not in _MODE_PROMPTS:
            _invalid("mode 仅支持：" + ", ".join(sorted(_MODE_PROMPTS)))
        fmt = str(params.get("format") or "text")
        if fmt not in ("text", "json_object"):
            _invalid("format 仅支持 text / json_object")
        try:
            requested = int(params.get("max_tokens")
                            or cfg["default_max_tokens"])
        except (TypeError, ValueError):
            _invalid("max_tokens 必须是整数")
            requested = int(cfg["default_max_tokens"])
        used = _out_tokens(requested, max_tokens_cap)
        system = str(params.get("system") or "")
        prefix = _MODE_PROMPTS.get(mode, "") if mode else ""
        budget = _ctx_tokens() - used - _tokens(system) - _tokens(prefix)
        prompt = _limit_input(prefix + prompt, max(64, budget))
        if fmt == "json_object":
            # 提示约束（不是受约束解码）：与 extract 的 JSON 提示同一口径
            prompt = prompt + "\n只输出一个 JSON 对象，不要输出其他内容。"
        out = _chat_once(system, prompt, used, _temperature(params),
                         should_cancel, timeout_s=timeout_s)
        return _answer_result(out, used)

    def chat(params: dict, should_cancel=None, max_tokens_cap=None,
             timeout_s=None) -> dict:
        if not isinstance(params, dict):
            _invalid("arguments 必须是对象")
        messages = params.get("messages") or []
        if not isinstance(messages, list) or not messages:
            _invalid("messages 不能为空")
        clean: list[dict] = []
        for message in messages:
            if not isinstance(message, dict):
                _invalid("messages 元素必须是 {role, content} 对象")
            role, content = message.get("role"), message.get("content")
            if role not in ("user", "assistant", "system"):
                _invalid("messages 元素 role 仅支持 user/assistant/system")
            if not isinstance(content, str):
                _invalid("messages 元素 content 必须是字符串")
            clean.append({"role": role, "content": content})
        system = str(params.get("system") or "")
        session = str(params.get("session") or "").strip()
        if session and len(session) > 128:
            _invalid("session 过长（最多 128 字符）")
        used = _out_tokens(cfg["default_max_tokens"], max_tokens_cap)
        budget = _ctx_tokens() - used - _tokens(system)
        temperature = _temperature(params)

        def _ask(base: list[dict]) -> dict:
            request = ([{"role": "system", "content": system}]
                       if system else []) + base
            try:
                return backend.chat(
                    request, max_tokens=used, temperature=temperature,
                    cancel_check=_as_check(should_cancel), timeout_s=timeout_s)
            except BackendError as exc:
                if getattr(exc, "kind", "") == "cancelled":
                    raise Cancelled() from exc
                raise

        if not session:
            base, truncated = _fit_messages(clean, max(64, budget))
            out = _ask(base)
            extra = {"input_truncated": True} if truncated else None
            return _answer_result(out, used, extra)

        # P2：同一 session 的「读历史 → 推理 → 写回」必须串行，否则并发请求
        # 各自读到同一历史、互相覆盖，丢一轮上下文。不同 session 互不阻塞。
        with _session_lock(session):
            with session_lock:
                _purge_expired_locked()
                history = list(session_store.get(session, []))
            base, truncated = _fit_messages(history + clean, max(64, budget))
            out = _ask(base)
            with session_lock:
                # 会话数上限：超限时逐出最早插入的（dict 保序），防无界增长
                if (session not in session_store
                        and len(session_store) >= max_sessions):
                    try:
                        evicted = next(iter(session_store))
                        session_store.pop(evicted)
                        session_seen.pop(evicted, None)
                    except (StopIteration, KeyError):
                        pass
                stored = base + [{"role": "assistant",
                                  "content": str(out.get("text") or "")}]
                # 会话累计预算：存入前再按当前预算裁一次，避免下次请求才丢上下文
                stored, _ = _fit_messages(stored, max(64, budget))
                if max_session_messages > 0:
                    stored = stored[-max_session_messages:]
                session_store[session] = stored
                session_seen[session] = time.monotonic()
        extra = {"input_truncated": True} if truncated else None
        return _answer_result(out, used, extra)

    _try_json = _extract_json        # 保持内层调用点不变

    def extract(params: dict, should_cancel=None, max_tokens_cap=None,
                timeout_s=None) -> dict:
        if not isinstance(params, dict):
            _invalid("arguments 必须是对象")
        text = str(params.get("text") or "")
        schema = params.get("schema")
        if not isinstance(schema, dict):
            _invalid("schema 必须是 JSON Schema 对象")
        try:
            requested = int(cfg["default_max_tokens"]) * 2
        except (TypeError, ValueError):
            requested = int(cfg.get("default_max_tokens", 512)) * 2
        used = _out_tokens(requested, max_tokens_cap)
        schema_json = json.dumps(schema, ensure_ascii=False)
        instructions = str(params.get("instructions") or "")
        head = (instructions +
                "\n从下面的文本中按 JSON Schema 抽取，"
                "只输出一个 JSON 对象，不要输出其他内容。\n"
                "Schema:\n" + schema_json + "\n\n文本：\n")
        budget = _ctx_tokens() - used - _tokens(head)
        text = _limit_input(text, max(64, budget), "抽取文本")
        instruction = head + text
        last_raw, last_error = "", ""
        last_meta: dict = {}
        for _attempt in range(max(0, extract_retries) + 1):
            out = _chat_once("", instruction, used,
                             float(cfg.get("default_temperature", 0.3)),
                             should_cancel, timeout_s=timeout_s)
            raw = str(out.get("text") or "")
            last_raw = raw
            last_meta = _meta_of(out, used)
            try:
                parsed = _try_json(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = "输出不是合法 JSON：%s" % exc
            else:
                errors = validate_schema(parsed, schema)
                if not errors:
                    return {"content": [{"type": "text",
                                         "text": _clip_result(raw)}],
                            "structuredContent": parsed,
                            "_meta": _meta_of(out, used)}
                last_error = "不符合 schema：" + "；".join(errors[:5])
            instruction = (head + text +
                           "\n\n上次输出不可用（%s），请重新只输出符合 schema 的 JSON。"
                           % last_error[:200])
        # 失败兜底也必须带 structuredContent：extract 声明了 outputSchema，
        # MCP 客户端会对每一次 tools/call 校验（job 路径此前已踩过 -32600）。
        failure = {"error": last_error,
                   "raw": _clip_result(last_raw)[:2000]}
        return {"content": [{"type": "text",
                             "text": "结构化抽取失败（校验重试已用尽）。\n"
                                     "校验错误：" + last_error +
                                     "\n模型原始输出：\n" + _clip_result(last_raw)}],
                "structuredContent": failure,
                "isError": True,
                "_meta": last_meta}

    def batch(params: dict, should_cancel=None, max_tokens_cap=None,
              timeout_s=None) -> dict:
        if not isinstance(params, dict):
            _invalid("arguments 必须是对象")
        segments = params.get("segments") or []
        if not isinstance(segments, list) or not segments:
            _invalid("segments 不能为空")
        # P2：段数超限不再 raise（上游未必捕获，可能破坏 JSON-RPC 帧），
        # 改为只处理前 batch_max_segments 段，并按合法 JSON 标记 truncated。
        max_segments = int(cfg["batch_max_segments"])
        total_segments = len(segments)
        truncated = total_segments > max_segments
        work = segments[:max_segments] if truncated else segments
        mode = params.get("mode")
        if mode is not None and mode not in _MODE_PROMPTS:
            _invalid("mode 仅支持：" + ", ".join(sorted(_MODE_PROMPTS)))
        system = str(params.get("system") or "")
        try:
            requested = int(cfg["default_max_tokens"])
        except (TypeError, ValueError):
            requested = 512
        used = _out_tokens(requested, max_tokens_cap)
        prefix = _MODE_PROMPTS.get(mode, "") if mode else ""
        budget = _ctx_tokens() - used - _tokens(system) - _tokens(prefix)
        results: list[Any] = []
        finishes: list[str] = []
        check = _as_check(should_cancel)
        for seg in work:
            if check is not None and check():
                results.append({"error": "cancelled"})
                finishes.append("cancelled")
                continue
            user = _limit_input(prefix + str(seg), max(64, budget), "分段")
            try:
                out = _chat_once(system, user, used,
                                 float(cfg.get("default_temperature", 0.3)),
                                 should_cancel, timeout_s=timeout_s)
            except Cancelled:
                # 取消要向上传播（runner 才会把 job 记成 cancelled）；
                # 单段普通失败仍按 F8 隔离成等长数组里的 error 项。
                if check is not None and check():
                    raise
                results.append({"error": "cancelled"})
                finishes.append("cancelled")
                continue
            except Exception as exc:      # F8：单段失败不中断整批
                results.append({"error": str(exc)[:200]})
                finishes.append("error")
                continue
            text = str(out.get("text") or "")
            finish = str(out.get("finish_reason") or "stop")
            finishes.append(finish)
            if not text.strip():
                results.append({"error": "空响应（finish_reason=%s）" % finish})
                continue
            text = _clip_result(text)
            if finish == "length":
                text += "\n…[输出达到 max_tokens 截断（finish_reason=length）]"
            results.append(text)
        # P2：对整体 json.dumps 做字符串裁剪会得到非法 JSON；超限时返回带
        # truncated 标记的合法 JSON，并尽量保留能放下的完整结果项。
        payload = json.dumps(results, ensure_ascii=False)
        limit = int(cfg.get("max_result_chars", 20000))
        kept = results
        if len(payload) > limit:
            truncated = True
            kept = []
            for item in results:
                if len(json.dumps(kept + [item], ensure_ascii=False)) > limit:
                    break
                kept.append(item)
        if truncated:
            payload = json.dumps({"truncated": True,
                                  "total": total_segments,
                                  "processed": len(results),
                                  "results": kept}, ensure_ascii=False)
        return {"content": [{"type": "text", "text": payload}],
                "_meta": {"finish_reasons": finishes, "max_tokens": used}}

    def status(params: dict, should_cancel=None, max_tokens_cap=None,
               timeout_s=None) -> dict:
        info = backend.status()
        if isinstance(info, dict):
            context_len = info.get("context_len")
            if isinstance(context_len, int) and context_len > 0:
                # /health 是板端上下文的事实源；工具层预算随后按它收窄
                cfg["context_len"] = context_len
            model_class = info.get("model_class")
            if model_class in ("small", "medium", "large"):
                cfg["model_class"] = model_class
        lines = ["模型状态：" + json.dumps(info, ensure_ascii=False)]
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    def job_tool(params: dict, should_cancel=None, max_tokens_cap=None,
                 timeout_s=None) -> dict:
        # 真正的 job 表操作由 stdio.py 注册（需要进度回调）；此处占位防缺注册
        raise JsonRpcError(-32603, "job 工具未注册（内部错误）")

    return {"ask": ask, "chat": chat, "extract": extract,
            "batch": batch, "status": status, "job": job_tool}
