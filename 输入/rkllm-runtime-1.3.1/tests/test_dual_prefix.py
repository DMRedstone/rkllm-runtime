"""双前缀挂载回归：数据面端点必须 `/v1/x` 与 `/x` 行为一致。

替代原来的 root_alias 中间件（手工别名表，C-H1 就是漏加 `/reset` `/capabilities`）。
这里按端点逐个比对两次挂载的状态码，新增端点如果只挂了一侧，这些断言会直接暴露。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.config import load_config
from rkllm.server.app import create_app


class _FakeSupervisor:
    state = "ready"
    metrics = {"worker_restarts_total": 0}

    def is_ready(self):
        return True

    def execute(self, prompt, **kwargs):
        yield "ok"

    def abort(self):
        pass


@pytest.fixture()
def client():
    cfg = load_config(env={"RKLLM_MODEL": "/tmp/x.rkllm"})
    with TestClient(create_app(cfg, supervisor=_FakeSupervisor())) as c:
        yield c


# (方法, 路径（不带 /v1）, 请求体)
CASES = [
    ("GET", "/models", None),
    ("GET", "/models/fake", None),
    ("POST", "/chat/completions", {"messages": [{"role": "user",
                                                 "content": "hi"}]}),
    ("POST", "/completions", {"prompt": "hi"}),
    ("POST", "/responses", {"input": "hi"}),
    ("GET", "/capabilities", None),
    ("POST", "/reset", {}),
    # 未实现端点：两侧都应是显式 501，而不是一侧 404
    ("POST", "/embeddings", {"input": "x"}),
    ("POST", "/moderations", {"input": "x"}),
]


@pytest.mark.parametrize("method,path,payload", CASES)
def test_v1_and_root_alias_agree(client, method, path, payload):
    def call(target):
        return client.request(method, target, json=payload)

    prefixed = call("/v1" + path)
    alias = call(path)
    assert prefixed.status_code == alias.status_code, (
        "%s %s: /v1 版 %d，根路径版 %d（某一侧没挂载）"
        % (method, path, prefixed.status_code, alias.status_code))
    assert prefixed.status_code != 404, "%s %s 根本没挂载" % (method, path)


def test_unknown_path_still_404(client):
    r = client.get("/v1/nope")
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_no_manual_alias_table_left():
    """别名必须由挂载派生：代码里不应再有手工别名集合。"""
    src = (Path(__file__).resolve().parents[1]
           / "src" / "rkllm" / "server" / "app.py").read_text(encoding="utf-8")
    assert "_ROOT_ALIASES" not in src
