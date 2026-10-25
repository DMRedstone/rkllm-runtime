"""P0 回归（2026-10-25 审计整改）：finish_reason=length + completions 流式。

- length 判定口径复刻 1.3.1-15（git 950005f 的 _finish_reason）：runtime 的
  generate_tokens 触顶时比 max_new_tokens 少 1（真机实测 8→7、40→39、6→5）；
- perf 帧此前在 WorkerClient.run 的 OP_END 分支被丢弃，usage 的 exact counts
  从未生效——现在 end 帧 perf 经 {"perf": ...} 元数据透传到 run_blocking；
- /v1/completions 流式此前忽略 stream 参数，现与 chat 同机制（F-1 取消/心跳）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.agent.pipeline import Pipeline
from rkllm.config import load_config
from rkllm.server.app import create_app


# -- Pipeline 层：finish_reason 的 length 判定 ---------------------------------


def _pipe(max_tokens=None):
    svc = type("S", (), {"ttft_log": staticmethod(lambda _s: None),
                         "supervisor": None,
                         "system_prompt": "",
                         "tools_mode": "prompt",
                         "tool_parser": "auto",
                         "max_tool_calls": 4,
                         "default_sampling": None,
                         "model_name": "fake"})()
    return Pipeline(svc, {"max_tokens": max_tokens} if max_tokens else {},
                    chat=True)


def test_finish_reason_length_when_generate_tops_out():
    """generate_tokens >= max_tokens-1 即 length（15 版真机口径）。"""
    pipe = _pipe(max_tokens=8)
    assert pipe.finish_reason({"perf": {"generate_tokens": 7}}) == "length"


def test_finish_reason_stop_when_well_below_limit():
    pipe = _pipe(max_tokens=40)
    assert pipe.finish_reason({"perf": {"generate_tokens": 9}}) == "stop"


def test_finish_reason_stop_without_max_tokens():
    """未给 max_tokens 不判 length（引擎默认上限对上层不可见，15 版同口径）。"""
    pipe = _pipe()
    assert pipe.finish_reason({"perf": {"generate_tokens": 2048}}) == "stop"


def test_finish_reason_prefers_result_over_cached_perf():
    pipe = _pipe(max_tokens=8)
    pipe._perf = {"generate_tokens": 7}
    assert pipe.finish_reason() == "length"          # 流式路径：无参回落 _perf
    assert pipe.finish_reason({"perf": {"generate_tokens": 3}}) == "stop"


def test_usage_takes_exact_counts_from_perf_frame():
    """end 帧 perf 透传后 usage 用 exact counts（此前恒为估算值）。"""
    pipe = _pipe(max_tokens=8)
    pipe._content_before.append("你好")
    usage = pipe.usage({"text": "你好",
                        "perf": {"prefill_tokens": 13, "generate_tokens": 7}})
    assert usage["prompt_tokens"] == 13
    assert usage["completion_tokens"] == 7


# -- HTTP 层：completions 流式 + length 端到端 ---------------------------------


@pytest.fixture()
def client():
    cfg = load_config(env={})
    app = create_app(cfg, supervisor=_Fake(perf={"prefill_tokens": 5,
                                                 "generate_tokens": 7}))
    with TestClient(app) as c:
        yield c


class _Fake:
    """带 perf 元数据的确定 token 流（真路径由 WorkerClient.run 产出）。"""

    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def __init__(self, tokens=None, perf=None):
        self.tokens = tokens or ["你", "好"]
        self.perf = perf
        self.model_name = "fake"

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        for t in self.tokens:
            yield t
        if self.perf:
            yield {"perf": self.perf}

    def abort(self):
        pass


def test_completions_nonstream_reports_length(client):
    r = client.post("/v1/completions",
                    json={"prompt": "数到十", "max_tokens": 8})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert r.json()["usage"]["completion_tokens"] == 7   # exact counts 生效


def test_completions_stream_uses_text_completion_envelope(client):
    r = client.post("/v1/completions",
                    json={"prompt": "你好", "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = [json.loads(line[6:]) for line in r.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert frames, "SSE 无数据帧"
    assert all(f["object"] == "text_completion" for f in frames)
    assert frames[0]["choices"][0]["delta"] == {"text": ""}   # 无 role 首帧
    text = "".join(f["choices"][0]["delta"].get("text", "")
                   for f in frames)
    assert text == "你好"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert r.text.rstrip().endswith("data: [DONE]")


def test_completions_stream_include_usage(client):
    r = client.post("/v1/completions",
                    json={"prompt": "你好", "stream": True,
                          "stream_options": {"include_usage": True}})
    frames = [json.loads(line[6:]) for line in r.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    usage_frames = [f for f in frames if f.get("usage")]
    assert usage_frames and usage_frames[-1]["usage"]["completion_tokens"] == 7
