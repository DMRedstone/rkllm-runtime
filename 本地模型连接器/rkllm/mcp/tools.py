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
import math
import threading
import time
from typing import Any, Callable

from ..agent.json_schema import validate as validate_schema
from .jobs import DEFAULT_BYTES_PER_TOKEN, estimate_tokens
from .protocol import INVALID_PARAMS, JsonRpcError

__all__ = ["build_handlers", "tool_definitions", "Cancelled"]

# JSON Schema 定义的合法 type 取值（畸形 schema 必须在进校验器前拒绝，
# 否则会以 AttributeError/TypeError 逃成 -32603；P2.5）。
_SCHEMA_TYPES = frozenset({"object", "array", "string", "number", "integer",
                           "boolean", "null"})
# schema 嵌套深度上限：防止客户端用深层嵌套拖垮递归校验（DoS 边界）。
_SCHEMA_MAX_DEPTH = 32
# 数值型 schema 关键字（出现时必须是非 bool 的有限数值）。
_SCHEMA_NUMERIC = ("minimum", "maximum", "minLength", "maxLength",
                   "minItems", "maxItems")


def _safe_int(value: Any, default: int, *, lo: int | None = None,
              hi: int | None = None) -> int:
    """数值配置安全转换：inf/NaN/超大整数/非法字符串一律回落（P2.5）。"""
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        result = int(default)
    if lo is not None:
        result = max(lo, result)
    if hi is not None:
        result = min(hi, result)
    return result


def validate_schema_definition(schema: Any, path: str = "$",
                               depth: int = 0) -> None:
    """畸形 JSON Schema 显式拒绝（-32602，不是 -32603）。

    `json_schema.validate` 假定 schema 结构合法（`properties.items()`、
    `required` 可迭代…）；客户端传 `{"properties": []}` / `{"required": "x"}`
    会让它抛 AttributeError/TypeError，被 dispatcher 兜成 -32603。这里做
    结构/类型/深度校验，把"参数错误"和"服务端内部错误"分开（P2.5）。

    放在模块级是为了让 stdio 在"转 job"之前就能拒绝畸形参数（否则 extract
    默认估算会先转 job，参数错误被延迟到 job 终帧，客户端拿不到 -32602）。
    """
    if depth > _SCHEMA_MAX_DEPTH:
        raise JsonRpcError(INVALID_PARAMS,
                           "%s：schema 嵌套过深（>%d）"
                           % (path, _SCHEMA_MAX_DEPTH))
    if not isinstance(schema, dict):
        raise JsonRpcError(INVALID_PARAMS, "%s：必须是 JSON Schema 对象" % path)
    stype = schema.get("type")
    if isinstance(stype, str):
        if stype not in _SCHEMA_TYPES:
            raise JsonRpcError(INVALID_PARAMS,
                               "%s.type 非法：%r（允许 %s）"
                               % (path, stype, "/".join(sorted(_SCHEMA_TYPES))))
    elif isinstance(stype, list):
        if not stype or any(not isinstance(t, str) or t not in _SCHEMA_TYPES
                            for t in stype):
            raise JsonRpcError(INVALID_PARAMS,
                               "%s.type 必须是合法类型名或其非空数组" % path)
    elif stype is not None:
        raise JsonRpcError(INVALID_PARAMS,
                           "%s.type 必须是字符串或字符串数组" % path)

    if "enum" in schema and not isinstance(schema["enum"], list):
        raise JsonRpcError(INVALID_PARAMS, "%s.enum 必须是数组" % path)
    for keyword in _SCHEMA_NUMERIC:
        if keyword in schema:
            value = schema[keyword]
            if isinstance(value, bool) \
                    or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)):
                raise JsonRpcError(INVALID_PARAMS,
                                   "%s.%s 必须是有限数值" % (path, keyword))
    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list) \
                or any(not isinstance(name, str) for name in required):
            raise JsonRpcError(INVALID_PARAMS,
                               "%s.required 必须是字符串数组" % path)
    props = schema.get("properties")
    if props is not None:
        if not isinstance(props, dict):
            raise JsonRpcError(INVALID_PARAMS,
                               "%s.properties 必须是对象" % path)
        for name, sub in props.items():
            if not isinstance(name, str):
                raise JsonRpcError(INVALID_PARAMS,
                                   "%s.properties 的键必须是字符串" % path)
            validate_schema_definition(sub, "%s.%s" % (path, name), depth + 1)
    items = schema.get("items")
    if items is not None:
        if isinstance(items, dict):
            validate_schema_definition(items, path + "[]", depth + 1)
        elif isinstance(items, list):
            for index, sub in enumerate(items):
                validate_schema_definition(sub, "%s[%d]" % (path, index),
                                           depth + 1)
        else:
            raise JsonRpcError(INVALID_PARAMS,
                               "%s.items 必须是对象或对象数组" % path)
    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool):
        raise JsonRpcError(INVALID_PARAMS,
                           "%s.additionalProperties 必须是布尔值" % path)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None

# 冷启动兜底常量：仅当板端 `/health` 不可达（cfg["_board"] 为空）时使用。
# 正常运行时 stdio 会把板端 context_len / limits 写进快照，_bval 优先读快照。
# 与旧 cfg 键（context_window / context_margin_tokens / default_max_tokens）解耦：
# 那些键 A 方案后不再是判断依据，兜底值集中在这里，避免两处漂移。
DEFAULT_CONTEXT = 4096
DEFAULT_CONTEXT_MARGIN = 32
DEFAULT_MAX_TOKENS = 512

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

    cfg 需要：default_temperature / max_result_chars / batch_max_segments。
    上下文预算与生成上限优先读板端能力快照 `cfg["_board"]`（stdio 从
    `/health` 刷新：context_len / limits.*）；板端不可达时回落到本模块
    冷启动常量。旧的 `max_input_chars`/`context_window` 等键不再参与判断。

    `max_tokens_cap`：同步路径由 stdio 传入的安全生成上限；job 路径为 None
    （按调用方请求/板端默认值生成）。`timeout_s`：本次调用的 HTTP 总读超时。
    """
    from .backend import BackendError

    def _invalid(message: str) -> None:
        """参数非法 → JSON-RPC -32602（MCP InvalidParams），不再是 -32603。"""
        raise JsonRpcError(INVALID_PARAMS, message)

    def _check_schema_definition(schema: Any) -> None:
        """模块级校验的薄封装（保留内层调用点；逻辑只有一份）。"""
        validate_schema_definition(schema)

    def _as_check(value):
        if value is None or callable(value):
            return value
        snapshot = bool(value)
        return lambda: snapshot

    result_limit = _safe_int(cfg.get("max_result_chars", 20000), 20000,
                             lo=1000, hi=200000)
    session_store = sessions if sessions is not None else {}
    session_lock = sessions_lock or threading.Lock()
    max_session_messages = _safe_int(cfg.get("session_max_messages", 0), 0,
                                     lo=0, hi=1000)
    session_locks: dict[str, threading.Lock] = {}
    max_sessions = _safe_int(cfg.get("session_max_count", 256), 256,
                             lo=1, hi=4096)
    # P2：会话空闲 TTL（秒），0=不过期。陈旧会话不回收会持续占用内存/上下文。
    session_ttl_s = _finite_float(cfg.get("session_ttl_s", 0)) or 0.0
    session_seen: dict[str, float] = {}

    def _bval(key: str, default):
        """读板端能力快照。每次现读 `cfg["_board"]`（stdio 会整体替换它）。"""
        board = cfg.get("_board") or {}
        value = board.get(key, default)
        return default if value is None else value

    def _bytes_per_token() -> float:
        """板端 tokenizer 口径（`/health.tokens.bytes_per_token`），取不到用 3。

        P2.5：输入裁剪、上下文预算、prefill 估算必须与板端 tokenizer 同口径，
        不再用写死的"UTF-8 字节/3"（4B/27B 的 bytes_per_token 实测并不相同）。
        """
        raw = _bval("bytes_per_token", DEFAULT_BYTES_PER_TOKEN)
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return DEFAULT_BYTES_PER_TOKEN
        if 1.0 <= value <= 16.0 and math.isfinite(value):
            return value
        return DEFAULT_BYTES_PER_TOKEN

    def _tokens(text: str) -> int:
        """与板端同口径的 token 估算（UTF-8 字节 / bytes_per_token）。"""
        return estimate_tokens(text, _bytes_per_token())

    def _ctx_tokens() -> int:
        """可用输入预算 = 板端上下文 − 板端保留余量。

        两者都来自 `/health`（`context_len` 与 `limits.context_margin_tokens`）；
        板端不可达时才用冷启动常量，保证工具仍能构造请求让板端自己拒绝。
        """
        raw = cfg.get("context_len") or _bval("context_len", 0) or DEFAULT_CONTEXT
        try:
            ctx = int(raw)
        except (TypeError, ValueError, OverflowError):
            ctx = DEFAULT_CONTEXT
        try:
            margin = int(_bval("context_margin_tokens", DEFAULT_CONTEXT_MARGIN))
        except (TypeError, ValueError, OverflowError):
            margin = DEFAULT_CONTEXT_MARGIN
        return max(256, ctx - margin)

    def _out_tokens(requested: Any, cap: Any = None) -> int:
        try:
            out = int(requested)
        except (TypeError, ValueError, OverflowError):
            # OverflowError：JSON 的 Infinity（json.loads 默认接受）转 int 会抛
            # 且非 ValueError，漏捕会逃成 -32603（2026-11-19 审计 P2）。
            # 默认生成长度也由板端给（limits.default_max_tokens）
            out = int(_bval("default_max_tokens", DEFAULT_MAX_TOKENS))
        out = max(1, out)
        # 不得越过板端声明的一次输出上限，否则板端会直接 400
        try:
            limit = int(_bval("max_tokens_limit", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            limit = 0
        if limit > 0:
            out = min(out, limit)
        if cap is not None:
            try:
                out = min(out, max(8, int(cap)))
            except (TypeError, ValueError, OverflowError):
                pass
        return out

    def _fit_text(text: str, budget_tokens: int, note: str = "输入") -> str:
        budget = max(16, int(budget_tokens))
        # 板端 tokenizer 口径：budget token ≈ budget × bytes_per_token 字节
        limit_bytes = max(16, int(budget * _bytes_per_token()))
        raw = text.encode("utf-8")
        if len(raw) <= limit_bytes:
            return text
        cut = raw[:limit_bytes].decode("utf-8", "ignore")
        return cut + "\n…[%s超出板端上下文预算，已截断（约 %d token）]" % (
            note, budget)

    def _limit_input(text: str, budget_tokens: int | None = None,
                     note: str = "输入") -> str:
        # 输入裁剪只按板端上下文预算（token，UTF-8 字节安全切分）。
        # 旧的 max_input_chars 字符硬截断已移除：它与 token 预算是两套口径，
        # 会互相打架；而且三处调用都传了 budget，不存在"完全不裁"的路径。
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
        # P2（2026-11-19 审计）：per-session 锁表只增不减——会话被 TTL/上限
        # 逐出后锁对象仍残留。只清"既不在 store 也不在 seen、且当前未持有"的
        # 锁；与 _session_lock() 的取用都在 session_lock 下串行，窗口极小，
        # 且仅影响"首次会话在首个请求途中被清理"，该场景 store 本无历史。
        # 放在 TTL 判断之外：TTL=0（不过期）时 max_sessions 淘汰同样要清锁。
        for name in list(session_locks):
            if name in session_store or name in session_seen:
                continue
            lk = session_locks.get(name)
            if lk is not None and not lk.locked():
                session_locks.pop(name, None)
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

    def _evict_one_locked(current: str | None = None) -> bool:
        """按 LRU 逐出最久未使用的会话（调用方须已持有 session_lock）。

        P2.5：旧实现用 `next(iter(session_store))`（插入序 = FIFO），与
        mcp.toml.example 里写的"LRU 淘汰"不符——常用会话可能比冷会话先被
        淘汰。现在以 `session_seen` 的最近访问时间为准，并排除即将写入的
        当前会话。
        """
        candidates = [name for name in session_store if name != current]
        if not candidates:
            return False
        victim = min(candidates, key=lambda name: session_seen.get(name, 0.0))
        session_store.pop(victim, None)
        session_seen.pop(victim, None)
        return True

    def _chat_once(system: str, user: str, max_tokens: int,
                   temperature: float, should_cancel=None,
                   timeout_s: float | None = None) -> dict:
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        try:
            # include_usage 由 HttpBackend 默认 True（P2.5），这里不显式传：
            # 保持对旧式 fake backend（测试替身）的调用兼容。
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
            raw = float(value) if value is not None \
                else float(cfg.get("default_temperature", 0.3))
        except (TypeError, ValueError, OverflowError):
            _invalid("temperature 必须是数字")
            return 0.0                       # pragma: no cover（_invalid 必抛）
        # P2.5：NaN/Inf（TOML `inf`/JSON `Infinity` 都能进来）不得进 runtime 请求；
        # 0~2 与板端 schema 的范围一致。
        if not math.isfinite(raw) or not 0.0 <= raw <= 2.0:
            _invalid("temperature 必须是 0~2 的有限数值")
        return raw

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
                            or cfg.get("default_max_tokens", DEFAULT_MAX_TOKENS))
        except (TypeError, ValueError, OverflowError):
            # OverflowError：客户端传 JSON Infinity 时 int() 会抛，按非法参数报
            _invalid("max_tokens 必须是整数")
            requested = int(DEFAULT_MAX_TOKENS)   # pragma: no cover（_invalid 必抛）
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
        used = _out_tokens(cfg.get("default_max_tokens", DEFAULT_MAX_TOKENS),
                           max_tokens_cap)
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
                # 访问即刷新 recency：LRU 淘汰依赖最近使用时间（P2.5）
                session_seen[session] = time.monotonic()
                history = list(session_store.get(session, []))
            base, truncated = _fit_messages(history + clean, max(64, budget))
            out = _ask(base)
            with session_lock:
                # 会话数上限：按 LRU 逐出最久未用的会话，防无界增长
                if (session not in session_store
                        and len(session_store) >= max_sessions):
                    _evict_one_locked(session)
                stored = base + [{"role": "assistant",
                                  "content": str(out.get("text") or "")}]
                # 会话累计预算：存入前再按当前预算裁一次，避免下次请求才丢上下文
                stored, _ = _fit_messages(stored, max(64, budget))
                if max_session_messages > 0:
                    stored = stored[-max_session_messages:]
                # dict 重新插入 = 移到末尾，保持 LRU 顺序可观测
                session_store.pop(session, None)
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
        if not text.strip():
            _invalid("text 不能为空")
        schema = params.get("schema")
        if not isinstance(schema, dict):
            _invalid("schema 必须是 JSON Schema 对象")
        # P2.5：畸形 schema → -32602（而非校验器内部异常逃成 -32603）
        _check_schema_definition(schema)
        requested = _safe_int(cfg.get("default_max_tokens", DEFAULT_MAX_TOKENS),
                              DEFAULT_MAX_TOKENS, lo=1, hi=8192) * 2
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
        max_segments = _safe_int(cfg.get("batch_max_segments", 20), 20,
                                 lo=1, hi=100)
        total_segments = len(segments)
        truncated = total_segments > max_segments
        work = segments[:max_segments] if truncated else segments
        mode = params.get("mode")
        if mode is not None and mode not in _MODE_PROMPTS:
            _invalid("mode 仅支持：" + ", ".join(sorted(_MODE_PROMPTS)))
        system = str(params.get("system") or "")
        try:
            requested = int(cfg["default_max_tokens"])
        except (TypeError, ValueError, OverflowError):
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
            # 非字符串真值调 .endswith 会抛 AttributeError（2026-11-19 审计 P2）
            if isinstance(model_class, str) and model_class.endswith("b"):
                cfg["model_class"] = model_class
        lines = ["模型状态：" + json.dumps(info, ensure_ascii=False)]
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    def job_tool(params: dict, should_cancel=None, max_tokens_cap=None,
                 timeout_s=None) -> dict:
        # 真正的 job 表操作由 stdio.py 注册（需要进度回调）；此处占位防缺注册
        raise JsonRpcError(-32603, "job 工具未注册（内部错误）")

    return {"ask": ask, "chat": chat, "extract": extract,
            "batch": batch, "status": status, "job": job_tool}
