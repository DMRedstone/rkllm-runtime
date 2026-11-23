"""P1-O1 回归：`/v1/completions` 流式必须是 OpenAI **legacy** completions 契约。

旧实现把 chat 的 `delta` 信封套在 `object=text_completion` 上，并沿用
`chatcmpl-` id：标准客户端（openai-python 的 `CompletionChoice.text` 是必填
字段）解析流式帧直接失败。

本文件刻意提高仿真度，**不 mock SSE 生成器**：

- 真实 `load_config()`（默认配置）+ 真实 `create_app()` + FastAPI `TestClient`
  （完整 ASGI 请求 → 路由 → StreamingResponse → SSE 字节流）；
- 唯一替身是 supervisor 边界（板端 NPU 不可用），token 由假的 worker 客户端
  按真实帧协议的形状产出（含 `{"perf": ...}` 元数据）；
- 客户端侧用 pydantic 严格复刻 openai-python 的 `Completion` /
  `CompletionChoice` / `CompletionUsage` 必填字段做反序列化：形状不对（例如
  `delta` 取代 `text`）会直接 ValidationError。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.config import load_config  # noqa: E402
from rkllm.server.app import create_app  # noqa: E402

STREAM_USAGE = {"stream_options": {"include_usage": True}}


# -- 严格客户端模型（等价 openai-python 的 legacy completions 类型） -------------


class StrictUsage(BaseModel):
    """openai.types.CompletionUsage（多出的 estimated 字段按 SDK 行为放行）。"""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class StrictCompletionChoice(BaseModel):
    """openai.types.CompletionChoice：`text`/`index`/`finish_reason` 必填。"""

    model_config = ConfigDict(extra="allow")

    text: str
    index: int
    logprobs: dict | None = None
    finish_reason: Literal["stop", "length", "content_filter",
                           "tool_calls"] | None = None


class StrictCompletion(BaseModel):
    """openai.types.Completion（流式 chunk 与整包用同一类型解析）。"""

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[StrictCompletionChoice]
    usage: StrictUsage | None = None


# -- 假的 worker 边界（真实帧协议形状；可参数化 token / perf） ------------------


class FakeSupervisor:
    """确定性 token 流；`perf` 在末尾产出 worker end 帧同款元数据。

    真实路径由 `WorkerClient.run` 在 OP_END 时产出 `{"perf": ...}`，
    `Pipeline.iter_tokens` 捕获后供 finish_reason=length 与精确 usage 使用。
    """

    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self, tokens=None, perf=None):
        self.tokens = tokens or ["你", "好"]
        self.perf = perf
        self.model_name = "fake"

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        for token in self.tokens:
            yield token
        if self.perf:
            yield {"perf": self.perf}

    def abort(self):
        pass


def _make_client(env=None, supervisor=None):
    """真实 load_config + 真实 create_app；返回 TestClient 上下文管理器。"""
    cfg = load_config(env=env or {})
    app = create_app(cfg, supervisor=supervisor or FakeSupervisor())
    return cfg, TestClient(app)


@pytest.fixture()
def client():
    _, c = _make_client(supervisor=FakeSupervisor(
        perf={"prefill_tokens": 5, "generate_tokens": 7}))
    with c as value:
        yield value


# -- SSE 解析（只做传输层拆行，不做任何形状清洗） -------------------------------


class Stream:
    def __init__(self, lines: list[str]):
        self.lines = lines
        self.raw: list[dict] = []
        self.done_count = 0
        self.data_after_done = False
        seen_done = False
        for line in lines:
            if not line.startswith("data:"):
                continue                      # 注释帧（心跳）不算数据
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                self.done_count += 1
                seen_done = True
                continue
            if seen_done:
                self.data_after_done = True
            self.raw.append(json.loads(payload))

    def parse(self) -> list[StrictCompletion]:
        """严格反序列化每个 chunk：形状不符即 ValidationError。"""
        return [StrictCompletion.model_validate(item) for item in self.raw]

    @property
    def text(self) -> str:
        return "".join(chunk.choices[0].text
                       for chunk in self.parse() if chunk.choices)

    def content_chunks(self) -> list[StrictCompletion]:
        return [c for c in self.parse() if c.choices]


def _stream_completions(client, payload: dict) -> Stream:
    with client.stream("POST", "/v1/completions", json=payload) as resp:
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = list(resp.iter_lines())
    data_lines = [line for line in lines if line.startswith("data:")]
    # [DONE] 必须恰好出现一次、且是最后一个 data 行
    done_lines = [line for line in data_lines
                  if line.strip() == "data: [DONE]"]
    assert done_lines, "SSE 缺少 data: [DONE]"
    assert data_lines[-1].strip() == "data: [DONE]", (
        "[DONE] 不是最后一帧：%r" % (data_lines[-1],))
    return Stream(data_lines)


# -- 契约：标准客户端可解析的 legacy 流式形状 -----------------------------------


def test_stream_parses_with_strict_legacy_client(client):
    stream = _stream_completions(client, {"prompt": "你好", "stream": True})
    chunks = stream.parse()

    assert chunks, "SSE 无数据帧"
    assert all(c.object == "text_completion" for c in chunks)
    assert all(c.id.startswith("cmpl-") for c in chunks)
    assert all(not c.id.startswith("chatcmpl-") for c in chunks)
    # D4：全流 id/created 唯一且一致（客户端按首帧 id 归并）
    assert len({c.id for c in chunks}) == 1
    assert len({c.created for c in chunks}) == 1
    assert len({c.model for c in chunks}) == 1

    # 首帧：空 text、无 role、finish_reason null（legacy 没有 role 首帧）
    first = chunks[0].choices[0]
    assert first.text == "" and first.finish_reason is None
    assert "role" not in stream.raw[0]["choices"][0]

    # 中间帧只有 text 增量；拼起来是完整答案
    assert [c.choices[0].text for c in chunks if c.choices
            and c.choices[0].text] == ["你", "好"]
    assert stream.text == "你好"

    # 末帧：finish_reason=stop、空 text；[DONE] 干净收尾
    last = chunks[-1].choices[0]
    assert last.finish_reason == "stop" and last.text == ""
    assert stream.done_count == 1 and not stream.data_after_done


def test_stream_never_uses_chat_delta_envelope(client):
    """P1-O1 的核心：choice 里不得再出现 chat 的 delta/message 字段。"""
    stream = _stream_completions(client, {"prompt": "你好", "stream": True})
    for raw in stream.raw:
        for choice in raw["choices"]:
            assert "delta" not in choice
            assert "message" not in choice
            # legacy 契约每个 choice 都有 text（即使为空串）
            assert isinstance(choice["text"], str)
            assert "logprobs" in choice       # OpenAI 每个 choice 都带（可为 null）


def test_stream_id_matches_nonstream_prefix(client):
    """与非流式 `/v1/completions` 一致：两者都是 cmpl- 前缀。"""
    stream = _stream_completions(client, {"prompt": "你好", "stream": True})
    stream_id = stream.parse()[0].id
    non_stream = client.post("/v1/completions", json={"prompt": "你好"}).json()
    assert non_stream["object"] == "text_completion"
    assert non_stream["id"].startswith("cmpl-")
    assert stream_id.startswith("cmpl-")


# -- stream_options.include_usage ----------------------------------------------


def test_include_usage_emits_usage_only_chunk_before_done(client):
    stream = _stream_completions(
        client, {"prompt": "你好", "stream": True, **STREAM_USAGE})
    chunks = stream.parse()

    usage_chunks = [c for c in chunks if c.usage is not None]
    assert len(usage_chunks) == 1
    usage_chunk = usage_chunks[0]
    # OpenAI 文档：额外 usage 帧的 choices 恒为空数组（客户端据此区分 token 帧）
    assert usage_chunk.choices == []
    assert raw_choices_empty(stream.raw, usage_chunk)
    usage = usage_chunk.usage
    assert usage.completion_tokens == 7            # perf exact counts 未变
    assert usage.prompt_tokens == 5
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    # finish 帧之后、[DONE] 之前，且全流只有这一个 usage 帧
    finish_at = next(i for i, c in enumerate(chunks)
                     if c.choices and c.choices[0].finish_reason is not None)
    assert chunks.index(usage_chunk) == finish_at + 1
    assert chunks[-1] is usage_chunk
    # token 帧不带 usage 字段（Chat 的 "all other chunks usage=null" 口径不在
    # legacy 契约要求内，这里只保证不会伪造 usage）
    assert all(c.usage is None for c in chunks if c is not usage_chunk)
    assert stream.done_count == 1 and not stream.data_after_done


def raw_choices_empty(raw: list[dict], usage_chunk: StrictCompletion) -> bool:
    """usage-only 帧在原始 JSON 里也是 choices: []（不是空文本 choice）。"""
    for item in raw:
        if item.get("usage") is not None:
            return item.get("choices") == []
    return False


def test_no_usage_chunk_by_default(client):
    """默认 include_usage_default=false：流里没有 usage 字段。"""
    stream = _stream_completions(client, {"prompt": "你好", "stream": True})
    assert all(item.get("usage") is None for item in stream.raw)
    assert stream.text == "你好"


def test_include_usage_from_config_default():
    """openai.include_usage_default=true 时，不带 stream_options 也发 usage 帧。"""
    _, c = _make_client(env={"RKLLM_OPENAI_INCLUDE_USAGE_DEFAULT": "true"})
    with c as value:
        stream = _stream_completions(value, {"prompt": "你好", "stream": True})
    assert [item for item in stream.raw if item.get("usage") is not None]


# -- finish_reason / usage 行为保持（只改帧形状，不改语义） ----------------------


def test_length_finish_reason_and_usage_unchanged():
    """perf 触顶（generate_tokens=7, max_tokens=8）→ length + exact usage。"""
    _, c = _make_client(supervisor=FakeSupervisor(
        perf={"prefill_tokens": 5, "generate_tokens": 7}))
    with c as value:
        stream = _stream_completions(value, {"prompt": "数到十", "stream": True,
                                             "max_tokens": 8, **STREAM_USAGE})
    chunks = stream.parse()
    finish = [c for c in chunks if c.choices
              and c.choices[0].finish_reason is not None]
    assert [c.choices[0].finish_reason for c in finish] == ["length"]
    usage = [c for c in chunks if c.usage is not None][0].usage
    assert usage.completion_tokens == 7


def test_root_alias_stream_uses_legacy_contract(client):
    """base_url 不带 /v1 的客户端（/completions）也拿到同一 legacy 契约。"""
    with client.stream("POST", "/completions",
                       json={"prompt": "你好", "stream": True}) as resp:
        assert resp.status_code == 200
        lines = list(resp.iter_lines())
    stream = Stream(lines)
    chunks = stream.parse()
    assert chunks[0].id.startswith("cmpl-")
    assert stream.text == "你好"
    assert stream.done_count == 1
