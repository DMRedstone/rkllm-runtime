"""6 个 MCP 工具（MCP 设计 §4.1–§4.4）。

描述文案是给调用方模型看的提示词：能做什么、不能做什么、耗时、输出上限
四件事必须写清（沿用旧连接器 ABILITY_NOTE 的做法）。

工具层只依赖 Backend 协议与配置值，不碰 socket/ctypes。
长任务判定在 ToolRunner.should_job 化；job 的执行与轮询在 stdio.py 编排。
"""

from __future__ import annotations

import json
from typing import Any, Callable

__all__ = ["TOOL_DEFS", "build_handlers", "max_tool_prefix"]

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


def max_tool_prefix(prefix: str) -> str:
    return prefix


def _tool(name: str, desc: str, schema: dict, *,
          readonly: bool = False) -> dict:
    out = {"name": name, "description": desc, "inputSchema": schema}
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
                  "temperature": {"type": "number"}},
               "required": ["prompt"]}),
        _tool(p + "chat",
              "多轮对话：需要上下文跟进时用；messages 为完整历史。\n"
              "不适合长文生成与多步推理。" + job_hint,
              {"type": "object", "properties": {
                  "messages": {"type": "array", "items": {
                      "type": "object", "properties": {
                          "role": {"type": "string"},
                          "content": {"type": "string"}},
                      "required": ["role", "content"]}},
                  "system": {"type": "string"}},
               "required": ["messages"]}),
        _tool(p + "extract",
              "结构化抽取：按 JSON Schema 从文本抽取，返回 structuredContent。"
              "校验失败会自动重试 " + str(2) + " 次；仍失败时返回原始文本与错误。\n"
              "这是提示+校验，不是受约束解码。" + job_hint,
              {"type": "object", "properties": {
                  "text": {"type": "string"},
                  "schema": {"type": "object", "description": "JSON Schema"},
                  "instructions": {"type": "string"}},
               "required": ["text", "schema"]}),
        _tool(p + "batch",
              "分段批量：对 segments 逐段执行同一任务，串行执行，"
              "返回与输入等长的结果数组。单次最多 " + str(20) + " 段。" + job_hint,
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
              "长任务轮询与取消：action=get 取结果，action=cancel 取消任务。",
              {"type": "object", "properties": {
                  "job_id": {"type": "string"},
                  "action": {"type": "string", "enum": ["get", "cancel"]}},
               "required": ["job_id"]}),
    ]


def build_handlers(backend, cfg: dict, jobs, *,
                   progress_cb: Callable[[str], None] | None = None,
                   extract_retries: int = 2) -> dict[str, Callable]:
    """返回 {工具短名: handler(params)->result dict}。

    cfg 需要：tool_prefix / default_max_tokens / default_temperature /
    max_input_chars / max_result_chars / batch_max_segments。
    """
    prefix = cfg["tool_prefix"]
    input_limit = cfg["max_input_chars"]
    result_limit = cfg["max_result_chars"]

    def _validate_input(text: str) -> str:
        if len(text) > input_limit:
            return text[:input_limit] + "\n…[输入超限已截断，上限 %d 字符]" % input_limit
        return text

    def _clip_result(text: str) -> str:
        if len(text) > result_limit:
            return text[:result_limit] + "\n…[结果超限已截断，上限 %d 字符]" % result_limit
        return text

    def _chat_once(system: str, user: str, max_tokens: int,
                   temperature: float, cancel_check=None) -> str:
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        out = backend.chat(messages, max_tokens=max_tokens,
                           temperature=temperature, cancel_check=cancel_check)
        return out["text"]

    def ask(params: dict) -> dict:
        prompt = _validate_input(str(params.get("prompt", "")))
        if not prompt.strip():
            raise ValueError("prompt 不能为空")
        mode = params.get("mode")
        if mode and mode in _MODE_PROMPTS:
            prompt = _MODE_PROMPTS[mode] + prompt
        text = _chat_once(str(params.get("system", "")), prompt,
                          int(params.get("max_tokens") or cfg["default_max_tokens"]),
                          float(params.get("temperature")
                                if params.get("temperature") is not None
                                else cfg["default_temperature"]))
        return {"content": [{"type": "text", "text": _clip_result(text)}]}

    def chat(params: dict) -> dict:
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
        if system:
            clean.insert(0, {"role": "system", "content": system})
        out = backend.chat(clean, max_tokens=cfg["default_max_tokens"],
                           temperature=cfg["default_temperature"])
        return {"content": [{"type": "text", "text": _clip_result(out["text"])}]}

    def _try_json(text: str) -> Any:
        """宽容提取 JSON：容忍代码块与前后缀。"""
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

    def extract(params: dict) -> dict:
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
                             cfg["default_temperature"])
            last_raw = raw
            try:
                parsed = _try_json(raw)
                return {"content": [{"type": "text", "text": _clip_result(raw)}],
                        "structuredContent": parsed}
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
                instruction = (instruction +
                               "\n\n上次输出不是合法 JSON（%s），请重新只输出 JSON。"
                               % last_error[:120])
        return {"content": [{"type": "text",
                             "text": "结构化抽取失败（校验重试已用尽）。\n"
                                     "校验错误：" + last_error +
                                     "\n模型原始输出：\n" + _clip_result(last_raw)}],
                "isError": True}

    def batch(params: dict) -> dict:
        segments = params.get("segments") or []
        if not isinstance(segments, list) or not segments:
            raise ValueError("segments 不能为空")
        if len(segments) > cfg["batch_max_segments"]:
            raise ValueError("segments 超过上限 %d" % cfg["batch_max_segments"])
        mode = params.get("mode")
        system = str(params.get("system", ""))
        results = []
        for seg in segments:
            user = _MODE_PROMPTS.get(mode, "") + _validate_input(str(seg))
            results.append(_chat_once(system, user, cfg["default_max_tokens"],
                                      cfg["default_temperature"]))
        return {"content": [{"type": "text",
                             "text": _clip_result(json.dumps(results,
                                                             ensure_ascii=False))}]}

    def status(params: dict) -> dict:
        info = backend.status()
        lines = ["模型状态：" + json.dumps(info, ensure_ascii=False)]
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    def job_tool(params: dict) -> dict:
        # 真正的 job 表操作由 stdio.py 注册（需要进度回调）；此处占位防缺注册
        raise ValueError("job 工具未注册（内部错误）")

    return {"ask": ask, "chat": chat, "extract": extract,
            "batch": batch, "status": status, "job": job_tool}
