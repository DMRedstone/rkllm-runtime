"""推理管线：conversation 回填 → worker token 流 → reasoning/tools 解析。

iter_tokens() 产出 str（内容 token）或 dict（一次性元数据：perf/reasoning/
tool_calls/parse_error）。finish_reason() 与 usage() 在流结束后可用。
json_object/json_schema 仿真由 apply_response_format 在 prompt 侧注入，
校验重试在非流式路径由 complete_with_retry 处理（设计 §6.4(3)）。
"""

from __future__ import annotations

import json
from typing import Iterator

from .conversation import build_prompt
from .gateway import InferenceGateway
from .reasoning import ReasoningSplitter
from .tools import ToolCallParser, build_tools_prefix, detect_parser


_SAMPLING_KEYS = ("top_k", "top_p", "temperature", "repeat_penalty",
                  "presence_penalty", "frequency_penalty")


class Pipeline:
    def __init__(self, svc: InferenceGateway, body: dict, *, chat: bool):
        # A-5：svc 只需满足 InferenceGateway 契约（模型名/tool 配置/system/
        # supervisor/queue/ttft_log），不再要求完整 ServiceState。
        self.svc = svc
        self.body = body
        self.chat = chat
        self.max_tokens = int(body.get("max_tokens") or 0) or None
        # P1-14：reasoning_effort 真正接线——非 none 即开启思考（runtime 只有
        # enable_thinking 布尔开关）；显式 enable_thinking 存在时优先。
        _re = body.get("reasoning_effort")
        enable_thinking = bool(body.get("enable_thinking", False))
        if _re is not None:
            enable_thinking = (_re != "none")
        self._enable_thinking = enable_thinking
        self._splitter = ReasoningSplitter(enabled=enable_thinking)
        # P1-14：stop 序列真正接线（流式截断 + 命中 abort，见 _emit_text）
        self._stop = self._norm_stop(body.get("stop"))
        self._stop_hit = False
        self._stopbuf = ""
        tools = body.get("tools") or None
        # tools_mode 只影响"是否把工具注入提示词"；native 已在 app 层降级为 prompt
        if tools and getattr(svc, "tools_mode", "prompt") == "off":
            tools = None
        self._parser = ToolCallParser(
            detect_parser(svc.model_name, svc.tool_parser),
            max_calls=svc.max_tool_calls) if (chat and tools) else None
        tools_prefix = build_tools_prefix(tools) if (chat and tools) else None
        if chat:
            prompt = build_prompt(
                body.get("messages") or [],
                system=body.get("_system_override") or svc.system_prompt or "",
                tools_json=tools_prefix)
        else:
            prompt = str(body.get("prompt", ""))
        rf = body.get("response_format") or {}
        if isinstance(rf, dict) and rf.get("type") == "json_object":
            # 与 1.3.1-15 一致：提示语可由旧 conffile 的 RKLLM_JSON_HINT 覆盖，
            # 空=用内置默认，填 "none"=不追加任何提示。
            import os
            hint = (os.environ.get("RKLLM_JSON_HINT") or "").strip()
            if hint.lower() == "none":
                hint = ""
            elif not hint:
                hint = ("Respond with a single valid JSON object and "
                        "nothing else.")
            if hint:
                prompt += "\n\n" + hint
        elif isinstance(rf, dict) and rf.get("type") == "json_schema":
            # emulated：schema 提示注入；校验与重试在非流式路径（openai.py）
            spec = rf.get("json_schema") or {}
            schema = spec.get("schema") if isinstance(spec, dict) else None
            prompt += ("\n\nRespond with a single JSON value that conforms to "
                       "this JSON Schema and nothing else:\n"
                       + json.dumps(schema or {}, ensure_ascii=False))
        self._prompt = prompt
        self._reasoning: list[str] = []
        self._content_before: list[str] = []
        self._tool_calls: list[dict] = []
        self._perf: dict | None = None

    # -- token 流 -----------------------------------------------------------

    def iter_tokens(self) -> Iterator[str | dict]:
        """经队列调用：从 supervisor 拉 token，流经 reasoning/tools 解析，
        流结束后自动 drain（tool_calls 汇总、未闭合块降级、reasoning 收尾）。"""
        import time as _time
        _t0 = _time.monotonic()
        stream = self.svc.supervisor.execute(
            self._prompt, enable_thinking=self._enable_thinking,
            max_new_tokens=self.max_tokens,
            sampling=self.sampling(),
            # 多轮对话才值得写前缀缓存（单轮存了不会命中）
            cache_hint=len(self.body.get("messages") or []) >= 2,
            request_timeout_s=self.svc.queue.request_timeout_s)
        _first = True
        for token in stream:
            if _first:
                _first = False
                # TTFT 打点：supervisor 层返回首个 token 的耗时（N3 定位用）
                self.svc.ttft_log(_time.monotonic() - _t0)
            yield from self._consume(token)
            if self._stop_hit:
                # stop 命中即终止生成并释放 NPU（不是丢弃后续 token 继续转）
                try:
                    self.svc.supervisor.abort()
                except Exception:
                    pass
                break
        yield from self.drain()

    def sampling(self) -> dict | None:
        """请求级采样参数 → runtime 的 _SamplingParam。

        顺序：进程默认（serve 的 --temperature 等 / default_sampling）
              < 请求体（OpenAI 的 temperature / top_p / top_k / *_penalty）。
        此前这些参数只进了 usage 估算、从未传给 runtime —— 能力表却标成
        native（审计发现的"静默不生效"），这里补上真正的传递。
        """
        merged: dict = dict(getattr(self.svc, "default_sampling", None) or {})
        for key in _SAMPLING_KEYS:
            value = self.body.get(key)
            if value is not None:
                merged[key] = value
        return merged or None

    @staticmethod
    def _norm_stop(stop) -> list[str]:
        if stop is None:
            return []
        if isinstance(stop, str):
            return [stop] if stop else []
        return [s for s in stop if isinstance(s, str) and s]

    def _emit_text(self, text: str, flush: bool = False) -> Iterator[str]:
        """P1-14：真实 stop 序列处理。命中则截断并置 _stop_hit；未命中保留可能
        构成 stop 前缀的尾巴，以捕获跨 chunk 的序列；flush=True 输出全部残留。"""
        if self._stop_hit:
            return
        if not self._stop:
            if text:
                yield text
            return
        self._stopbuf += text
        hit = None
        for s in self._stop:
            idx = self._stopbuf.find(s)
            if idx >= 0 and (hit is None or idx < hit[0]):
                hit = (idx, len(s))
        if hit is not None:
            emit = self._stopbuf[:hit[0]]
            self._stopbuf = ""
            self._stop_hit = True
            if emit:
                yield emit
            return
        if flush:
            emit = self._stopbuf
            self._stopbuf = ""
            if emit:
                yield emit
            return
        keep = 0
        for s in self._stop:
            for k in range(min(len(s) - 1, len(self._stopbuf)), 0, -1):
                if self._stopbuf.endswith(s[:k]):
                    keep = max(keep, k)
                    break
        if keep:
            emit, self._stopbuf = self._stopbuf[:-keep], self._stopbuf[-keep:]
        else:
            emit, self._stopbuf = self._stopbuf, ""
        if emit:
            yield emit

    def _consume(self, token: str) -> Iterator[str | dict]:
        reasoning, content = self._splitter.feed(token)
        if reasoning:
            self._reasoning.append(reasoning)
        if content:
            yield from self._route_content(content)

    def _route_content(self, text: str) -> Iterator[str | dict]:
        if self._parser is None:
            self._content_before.append(text)
            yield from self._emit_text(text)
            return
        for kind, payload in self._parser.feed(text):
            if kind == "content":
                self._content_before.append(payload)
                yield from self._emit_text(payload)
            else:
                yield {"tool_call": payload}

    def drain(self) -> Iterator[str | dict]:
        """流收尾：finish 段、tool_calls 汇总、perf。"""
        reasoning, content = self._splitter.finish()
        if reasoning:
            self._reasoning.append(reasoning)
        if content:
            yield from self._route_content(content)
        if self._parser is not None:
            for kind, payload in self._parser.finish():
                if kind == "content":
                    self._content_before.append(payload)
                    yield from self._emit_text(payload)
            calls = self._parser.tool_calls()
            if calls:
                self._tool_calls = calls
                yield {"tool_calls": calls}
            if self._parser.parse_failed:
                yield {"parse_error": True}
        # 输出可能被 stop 前缀判断扣住的尾巴
        yield from self._emit_text("", flush=True)

    # -- 收尾视图 ------------------------------------------------------------

    def finish_reason(self) -> str:
        if self._tool_calls:
            return "tool_calls"
        if self._parser is not None and self._parser.parse_failed:
            return "stop"
        return "stop"

    def usage(self, result: dict | None = None) -> dict:
        text = "".join(self._content_before)
        reasoning_chars = len("".join(self._reasoning))
        usage = {"prompt_tokens": max(1, len(self._prompt) // 2),
                 "completion_tokens": max(1, (len(text) + reasoning_chars) // 2),
                 "total_tokens": 0}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        usage["estimated"] = True      # 无 tokenizer：估算并标注（设计 §4.3）
        perf = result.get("perf") if result else None
        if perf:
            usage["prompt_tokens"] = perf.get("prefill_tokens",
                                              usage["prompt_tokens"])
            usage["completion_tokens"] = perf.get("generate_tokens",
                                                  usage["completion_tokens"])
            usage["total_tokens"] = (usage["prompt_tokens"]
                                     + usage["completion_tokens"])
            usage["estimated"] = False
        return usage

    @property
    def reasoning_text(self) -> str:
        return "".join(self._reasoning)

    @property
    def content_before_tools(self) -> str:
        return "".join(self._content_before)

    def trailing_frames(self, session) -> list[str]:
        """流式收尾附加帧：tool_calls 增量帧与 reasoning 帧。"""
        frames = []
        if self._tool_calls:
            frames.append(session.delta_frame(tool_calls=[
                {"index": c["index"], "id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in self._tool_calls]))
        if self._reasoning:
            frames.append(session.delta_frame(reasoning=self.reasoning_text))
        return frames
