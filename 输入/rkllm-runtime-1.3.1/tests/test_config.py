"""config.py 单测：80 键契约、优先级、校验、迁移（N11/F13 验收）。"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm.config import SCHEMA, ConfigError, config_docs, load_config

# 83 键分组计数（与《MCP 接入与配置设计》§7.2 一致）
GROUP_COUNTS = {"server": 13, "model": 12, "engine": 8, "queue": 5,
                "openai": 9, "cache": 5, "mcp": 24, "logging": 5, "metrics": 2}


def test_schema_is_exactly_83_keys():
    assert len(SCHEMA) == 83
    counts: dict[str, int] = {}
    for key in SCHEMA:
        counts[key.section] = counts.get(key.section, 0) + 1
    assert counts == GROUP_COUNTS


def test_schema_keys_unique():
    fulls = [k.full for k in SCHEMA]
    assert len(fulls) == len(set(fulls)), "存在重复键"


def test_defaults_load():
    cfg = load_config(env={})
    assert cfg.server.bind_host == "127.0.0.1"
    assert cfg.server.port == 8080
    assert cfg.mcp.task_threshold_s == 45
    assert cfg.mcp.default_max_tokens == 512
    assert cfg.model.keep_history is False
    assert cfg.queue.max_concurrent == 1
    assert cfg.source_of("server.port") == "default"


def test_priority_file_env_cli(tmp_path):
    f = tmp_path / "rkllm.toml"
    f.write_text(textwrap.dedent("""
        [server]
        port = 9000
        [mcp]
        task_threshold_s = 30
    """), encoding="utf-8")
    env = {"RKLLM_SERVER_PORT": "9100"}
    cfg = load_config(file=str(f), env=env, cli={"server.port": 9200})
    assert cfg.server.port == 9200
    assert cfg.source_of("server.port") == "cli"

    cfg = load_config(file=str(f), env=env)
    assert cfg.server.port == 9100
    assert cfg.source_of("server.port") == "env"

    cfg = load_config(file=str(f), env={})
    assert cfg.server.port == 9000
    assert cfg.source_of("server.port") == "file"
    assert cfg.mcp.task_threshold_s == 30


def test_env_bool_and_list_coercion():
    cfg = load_config(env={"RKLLM_MODEL_KEEP_HISTORY": "1",
                           "RKLLM_MODEL_SEARCH_DIRS": "/a,/b"})
    assert cfg.model.keep_history is True
    assert cfg.model.search_dirs == ("/a", "/b")


def test_validation_collects_all_errors():
    with pytest.raises(ConfigError) as excinfo:
        load_config(env={"RKLLM_SERVER_PORT": "70000",
                         "RKLLM_MCP_TASK_THRESHOLD_S": "90",
                         "RKLLM_MODEL_CONTEXT": "100"})
    errs = "\n".join(excinfo.value.errors)
    assert "server.port" in errs
    assert "mcp.task_threshold_s" in errs   # >= 60 被跨键校验拦截
    assert "model.context" in errs
    assert len(excinfo.value.errors) >= 3   # 一次性列出，不是首个即停


def test_cross_validation():
    with pytest.raises(ConfigError) as e1:
        load_config(env={"RKLLM_SERVER_TLS_CERT": "/a.pem"})   # tls 不成对
    assert "成对" in "\n".join(e1.value.errors)
    with pytest.raises(ConfigError) as e2:
        load_config(env={"RKLLM_MODEL_MIN_CONTEXT": "8192"})   # > context
    assert "min_context" in "\n".join(e2.value.errors)


def test_http_transport_requires_origin():
    with pytest.raises(ConfigError) as excinfo:
        load_config(env={"RKLLM_MCP_TRANSPORT": "http"})
    assert "allow_remote_origin" in "\n".join(excinfo.value.errors)


def test_legacy_env_migration():
    # NO_HISTORY / NO_FREQ_LOCK 是**反义**开关：0 表示"没有禁用"→ 保留。
    # 旧实现把它的值直接搬过去，语义完全颠倒（2026-10-21 修正）。
    cfg = load_config(env={"RKLLM_MODEL": "/m/qwen3-4b.rkllm",
                           "RKLLM_NO_HISTORY": "0"})
    assert cfg.model.path == "/m/qwen3-4b.rkllm"
    assert cfg.model.keep_history is True
    assert load_config(env={"RKLLM_NO_HISTORY": "1"}
                       ).model.keep_history is False
    assert load_config(env={"RKLLM_NO_FREQ_LOCK": "1"}
                       ).model.freq_lock is False
    # 空值不参与映射（conffile 里留下 KEY= 不应炸启动）
    assert load_config(env={"RKLLM_NO_HISTORY": ""}
                       ).model.keep_history is False   # 内置默认
    # 旧 conffile 的单数 CORS 变量仍然生效
    assert list(load_config(env={"RKLLM_CORS_ORIGIN": "http://a,http://b"}
                            ).server.cors_origins) == ["http://a", "http://b"]
    # 新键名优先于旧名
    cfg2 = load_config(env={"RKLLM_MODEL": "/old.rkllm",
                            "RKLLM_MODEL_PATH": "/new.rkllm"})
    assert cfg2.model.path == "/new.rkllm"


def test_require_model_path():
    with pytest.raises(ConfigError) as excinfo:
        load_config(env={}, require_model_path=True)
    assert "model.path" in "\n".join(excinfo.value.errors)


def test_config_docs_mentions_all_keys():
    doc = config_docs()
    for key in SCHEMA:
        assert f"`{key.full}`" in doc
