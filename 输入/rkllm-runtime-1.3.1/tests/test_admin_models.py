"""只读模型清单 `GET /admin/models` 回归。"""

from __future__ import annotations

import sys
from pathlib import Path

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


def _client(**env):
    cfg = load_config(env={"RKLLM_MODEL": "/tmp/x.rkllm", **env})
    return TestClient(create_app(cfg, supervisor=_FakeSupervisor()))


def test_admin_models_requires_auth():
    client = _client(RKLLM_API_KEY="secret")
    assert client.get("/admin/models").status_code == 401
    ok = client.get("/admin/models", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_admin_models_lists_files_in_search_dirs(tmp_path):
    (tmp_path / "qwen3.5-4b.rkllm").write_bytes(b"x" * 1024)
    (tmp_path / "other.rkllm").write_bytes(b"y" * 2048)
    client = _client(RKLLM_MODEL=str(tmp_path / "qwen3.5-4b.rkllm"),
                     RKLLM_MODEL_SEARCH_DIRS=str(tmp_path))
    payload = client.get("/admin/models").json()
    ids = [m["id"] for m in payload["models"]]
    assert ids == ["other", "qwen3.5-4b"]
    assert payload["current"]["path"].endswith(".rkllm")
    assert payload["switchable"] is False        # 热切换未实现，如实告知
    assert any(m["current"] for m in payload["models"])


def test_admin_models_declared_in_caps():
    client = _client()
    caps = client.get("/v1/capabilities").json()
    assert caps["endpoints"]["admin.models"] == "native"
    # 对话页已明确不做，不留在能力表里
    assert "ui" not in caps["endpoints"]
