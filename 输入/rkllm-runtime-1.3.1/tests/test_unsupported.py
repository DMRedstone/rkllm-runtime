"""未实现端点必须"回话"，不能落框架 404（G5 兼容性诚实）。"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm import caps                                    # noqa: E402
from rkllm.config import load_config                      # noqa: E402
from rkllm.server.app import create_app                   # noqa: E402
from rkllm.server.routes.unsupported import _UNSUPPORTED  # noqa: E402


def _client():
    # 不挂 supervisor：这些路由在任何 supervisor 状态下都应直接拒绝
    return TestClient(create_app(load_config(env={}), supervisor=None))


def test_unimplemented_endpoints_answer_501():
    client = _client()
    for path in ("/v1/embeddings", "/v1/messages"):
        r = client.post(path, json={"model": "rkllm", "input": "x"})
        assert r.status_code == 501, f"{path} -> {r.status_code}"
        err = r.json()["error"]
        assert err["code"] == "unsupported_endpoint"
        assert path in err["message"]
        assert "/v1/chat/completions" in err["message"]   # 给出替代路径
        assert r.headers.get("x-rkllm-endpoint") == "unsupported"


def test_rejected_endpoints_match_capability_table():
    """能力表标 unsupported 的端点必须有拒绝路由，且不能多。"""
    table = {name for name, tier in caps.ENDPOINTS.items() if tier == "unsupported"}
    routed = {cap_key for cap_key, _ in _UNSUPPORTED.values()}
    assert routed == table, f"能力表 {table} vs 路由 {routed}"


def test_unknown_path_still_404():
    """只有声明过的端点才回 501；拼错的路径仍是 404。"""
    r = _client().post("/v1/chat/completionsX", json={})
    assert r.status_code == 404


def test_capabilities_endpoint_lists_tiers():
    """/v1/capabilities 必须把三档都渲染出来（客户端据此决定降级）。"""
    caps_body = _client().get("/v1/capabilities").json()
    endpoints = caps_body.get("endpoints") or {}
    # 键是能力名而不是路径（能力表就是这么定义的）
    assert endpoints.get("responses") == "emulated"     # 翻译层已实现
    assert endpoints.get("embeddings") == "unsupported"
    assert endpoints.get("chat.completions") == "native"
