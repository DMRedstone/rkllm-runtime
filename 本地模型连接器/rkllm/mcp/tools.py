"""6 个 MCP 工具（MCP 设计 §4.1–§4.4）。

描述文案是给调用方模型看的提示词：能做什么、不能做什么、耗时、输出上限
四件事必须写清（沿用旧连接器 ABILITY_NOTE 的做法）。

工具层只依赖 Backend 协议与配置值，不碰 socket/ctypes。
长任务判定的编排在 stdio.py。

本批新增（审计整改）：
- 所有 handler 接受 `should_cancel`（由 stdio 传入 job/取消标志），贯通到
  `Backend.chat(cancel_check=...)`——此前形参存在但无人传，取消永不生效；
- `batch` 单段异常隔离（F8：单段失败不中断整批，返回等长数组）；
- `extract` 增加 JSON Schema 校验（F6：提示 + 校验 + 重试，非受约束解码）。
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable

from ..agent.json_schema import validate as validate_schema

__all__ = ["TOOL_DEFS", "build_handlers", "tool_definitions", "Cancelled"]

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
    "耗时 3–60 秒；超过 45 秒会返回 job_id，需调用 %slocal_llm_job 轮询。"
    "输出默认 ≤ 512 token。"
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
                                 "description": "生成上限，默认 512"},
                  "temperature": {"type": "number"},
                  "format": {"type": "string", "enum": ["text", "json_object"],
                             "description": "json_object 仅追加 JSON 提示（非受约束解码）"}},
               "required": ["prompt"]}),
        _tool(p + "chat",
              "多轮对话：需要上下文跟进时用；messages 为完整历史。\n"
              "可选 session 由连接器记住最近几轮（MCP 无状态，会话显式传参）。\n"
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
              "输出会按 schema 校验，不符会自动重试；仍失败时返回原始文本与错误。\n"
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
              "单次最多 20 段。" + job_hint,
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
                   progress_cb: Callable[[str], None] | None = None,
                   extract_retries: int = 2,
                   sessions: dict[str, list[dict]] | None = None,
                   sessions_lock: threading.Lock | None = None) -> dict[str, Callable]:
    """返回 {工具短名: handler(params, should_cancel=None)->result dict}。

    cfg 需要：tool_prefix / default_max_tokens / default_temperature /
    max_input_chars / max_result_chars / batch_max_segments。

    sessions/sessions_lock 给 `local_llm_chat` 的可选会话记忆（MCP 2026 无状态，
    会话只能作为工具参数显式传）；不传则每次只处理本次 messages。

    should_cancel 是 stdio 传入的 `() -> bool`：job 取消或客户端断开时应返回
    True。工具层把它一路传给 Backend，使 HTTP 读循环在 chunk 边界能中断。
    传入非可调用值（bool）时自动包装——历史上曾经把 property 的值直接传进来，
    取消检查因此恒为 False（真机踩过）。
    """
    from .backend import BackendError

    def _as_check(value):
        if value is None or callable(value):
            return value
        snapshot = bool(value)
        return lambda: snapshot

    prefix = cfg["tool_prefix"]
    input_limit = cfg["max_input_chars"]
    result_limit = cfg["max_result_chars"]
    session_store = sessions if sessions is not None else {}
    session_lock = sessions_lock or threading.Lock()
    max_session_messages = int(cfg.get("session_max_messages", 20))
    session_locks: dict[str, threading.Lock] = {}
    max_sessions = int(cfg.get("session_max_count", 256) or 256)

    def _session_lock(name: str) -> "threading.Lock":
        """同一 session 的读→推理→写回必须串行，否则并发丢上下文（P2）。"""
        with session_lock:
            lk = session_locks.get(name)
            if lk is None:
                lk = threading.Lock()
                session_locks[name] = lk
            return lk

    def _validate_input(text: str) -> str:
        if len(text) > input_limit:
            return text[:input_limit] + "\n…[输入超限已截断，上限 %d 字符]" % input_limit
        return text

    def _clip_result(text: str) -> str:
        if len(text) > result_limit:
            return text[:result_limit] + "\n…[结果超限已截断，上限 %d 字符]" % result_limit
        return text

    def _chat_once(system: str, user: str, max_tokens: int,
                   temperature: float, should_cancel=None) -> str:
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        try:
            out = backend.chat(messages, max_tokens=max_tokens,
                               temperature=temperature,
                               cancel_check=_as_check(should_cancel))
        except BackendError as exc:
            if getattr(exc, "kind", "") == "cancelled":
                raise Cancelled() from exc
            raise
        return out["text"]

    def ask(params: dict, should_cancel=None) -> dict:
        prompt = _validate_input(str(params.get("prompt", "")))
        if not prompt.strip():
            raise ValueError("prompt 不能为空")
        mode = params.get("mode")
        if mode and mode in _MODE_PROMPTS:
            prompt = _MODE_PROMPTS[mode] + prompt
        fmt = str(params.get("format") or "text")
        if fmt not in ("text", "json_object"):
            raise ValueError("format 仅支持 text / json_object")
        if fmt == "json_object":
            # 提示约束（不是受约束解码）：与 extract 的 JSON 提示同一口径
            prompt = prompt + "\n只输出一个 JSON 对象，不要输出其他内容。"
        text = _chat_once(str(params.get("system", "")), prompt,
                          int(params.get("max_tokens") or cfg["default_max_tokens"]),
                          float(params.get("temperature")
                                if params.get("temperature") is not None
                                else cfg["default_temperature"]),
                          should_cancel)
        return {"content": [{"type": "text", "text": _clip_result(text)}]}

    def chat(params: dict, should_cancel=None) -> dict:
        messages = params.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages 不能为空")
        clean = []
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if role not in ("user", "assistant", "system") or not isinstance(content, str):
                raise ValueError("messages 元素需含合法 role 与 content")
            clean.append({"role": role, "content": _validate_input(content)})
        system = str(params.get("system", ""))
        session = str(params.get("session") or "").strip()
        if session and len(session) > 128:
            raise ValueError("session 过长（最多 128 字符）")

        def _ask(base: list[dict]) -> dict:
            request = ([{"role": "system", "content": system}]
                       if system else []) + base
            try:
                return backend.chat(request,
                                    max_tokens=cfg["default_max_tokens"],
                                    temperature=cfg["default_temperature"],
                                    cancel_check=_as_check(should_cancel))
            except BackendError as exc:
                if getattr(exc, "kind", "") == "cancelled":
                    raise Cancelled() from exc
                raise

        if not session:
            out = _ask(clean)
            return {"content": [{"type": "text",
                                 "text": _clip_result(out["text"])}]}

        # P2：同一 session 的「读历史 → 推理 → 写回」必须串行，否则并发请求
        # 各自读到同一历史、互相覆盖，丢一轮上下文。不同 session 互不阻塞。
        with _session_lock(session):
            with session_lock:
                history = list(session_store.get(session, []))
            base = history + clean
            out = _ask(base)
            with session_lock:
                # 会话数上限：超限时逐出最早插入的（dict 保序），防无界增长
                if (session not in session_store
                        and len(session_store) >= max_sessions):
                    try:
                        session_store.pop(next(iter(session_store)))
                    except (StopIteration, KeyError):
                        pass
                session_store[session] = (
                    base + [{"role": "assistant", "content": out["text"]}]
                )[-max_session_messages:]
        return {"content": [{"type": "text", "text": _clip_result(out["text"])}]}

    _try_json = _extract_json        # 保持内层调用点不变

    def extract(params: dict, should_cancel=None) -> dict:
        text = _validate_input(str(params.get("text", "")))
        schema = params.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("schema 必须是 JSON Schema 对象")
        schema_json = json.dumps(schema, ensure_ascii=False)
        instruction = (str(params.get("instructions", "")) +
                       "\n从下面的文本中按 JSON Schema 抽取，"
                       "只输出一个 JSON 对象，不要输出其他内容。\n"
                       "Schema:\n" + schema_json + "\n\n文本：\n" + text)
        last_raw, last_error = "", ""
        for attempt in range(max(0, extract_retries) + 1):
            raw = _chat_once("", instruction, cfg["default_max_tokens"] * 2,
                             cfg["default_temperature"], should_cancel)
            last_raw = raw
            try:
                parsed = _try_json(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = "输出不是合法 JSON：%s" % exc
            else:
                errors = validate_schema(parsed, schema)
                if not errors:
                    return {"content": [{"type": "text", "text": _clip_result(raw)}],
                            "structuredContent": parsed}
                last_error = "不符合 schema：" + "；".join(errors[:5])
            instruction = (instruction +
                           "\n\n上次输出不可用（%s），请重新只输出符合 schema 的 JSON。"
                           % last_error[:200])
        return {"content": [{"type": "text",
                             "text": "结构化抽取失败（校验重试已用尽）。\n"
                                     "校验错误：" + last_error +
                                     "\n模型原始输出：\n" + _clip_result(last_raw)}],
                "isError": True}

    def batch(params: dict, should_cancel=None) -> dict:
        segments = params.get("segments") or []
        if not isinstance(segments, list) or not segments:
            raise ValueError("segments 不能为空")
        if len(segments) > cfg["batch_max_segments"]:
            raise ValueError("segments 超过上限 %d" % cfg["batch_max_segments"])
        mode = params.get("mode")
        system = str(params.get("system", ""))
        results: list[Any] = []
        check = _as_check(should_cancel)
        for seg in segments:
            if check is not None and check():
                results.append({"error": "cancelled"})
                continue
            user = _MODE_PROMPTS.get(mode, "") + _validate_input(str(seg))
            try:
                results.append(_chat_once(system, user,
                                          cfg["default_max_tokens"],
                                          cfg["default_temperature"],
                                          should_cancel))
            except Cancelled:
                # 取消要向上传播（runner 才会把 job 记成 cancelled）；
                # 单段普通失败仍按 F8 隔离成等长数组里的 error 项。
                if check is not None and check():
                    raise
                results.append({"error": "cancelled"})
            except Exception as exc:      # F8：单段失败不中断整批
                results.append({"error": str(exc)[:200]})
        # P2：对整体 json.dumps 做字符串裁剪会得到非法 JSON；超限时返回带
        # truncated 标记的合法 JSON，并尽量保留能放下的完整结果项。
        payload = json.dumps(results, ensure_ascii=False)
        limit = int(cfg.get("max_result_chars", 20000))
        if len(payload) > limit:
            kept: list[Any] = []
            for item in results:
                if len(json.dumps(kept + [item], ensure_ascii=False)) > limit:
                    break
                kept.append(item)
            payload = json.dumps({"truncated": True, "total": len(results),
                                  "results": kept}, ensure_ascii=False)
        return {"content": [{"type": "text", "text": payload}]}

    def status(params: dict, should_cancel=None) -> dict:
        info = backend.status()
        lines = ["模型状态：" + json.dumps(info, ensure_ascii=False)]
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    def job_tool(params: dict, should_cancel=None) -> dict:
        # 真正的 job 表操作由 stdio.py 注册（需要进度回调）；此处占位防缺注册
        raise ValueError("job 工具未注册（内部错误）")

    return {"ask": ask, "chat": chat, "extract": extract,
            "batch": batch, "status": status, "job": job_tool}
