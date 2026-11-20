"""2026-11-20 全仓库审计（板端补审）回归测试。

覆盖 config 的 section 非表 / 文件读取异常 / inf 与小数 / 旧 MODEL_DIR 分隔符 /
heartbeat 组合，caps 的 seed=0 与 reasoning_effort 档位 / n 标注，
abi 的可选符号归位，以及 completions 端点的 native 参数。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm import abi, caps
from rkllm.config import ConfigError, load_config
from rkllm.server.schemas import CompletionRequest


def _load(tmp_path, toml_text: str, *, env=None):
    path = tmp_path / "rkllm.toml"
    path.write_text(toml_text, encoding="utf-8")
    return load_config(file=str(path), env=env or {})


# -- config：section 写成标量/数组 --------------------------------------------

def test_config_section_scalar_is_config_error(tmp_path):
    """`server = 1` 是合法 TOML；旧实现抛 TypeError（traceback），应转 ConfigError。"""
    with pytest.raises(ConfigError) as ei:
        _load(tmp_path, "server = 1\n")
    assert any("段必须是表" in e for e in ei.value.errors)


def test_config_section_array_is_config_error(tmp_path):
    with pytest.raises(ConfigError) as ei:
        _load(tmp_path, 'server = ["port"]\n')
    assert any("段必须是表" in e for e in ei.value.errors)


def test_config_section_error_reported_once(tmp_path):
    """同段多个键不应重复同一条段错误。"""
    with pytest.raises(ConfigError) as ei:
        _load(tmp_path, "server = 1\nmodel = 2\n")
    msgs = [e for e in ei.value.errors if "段必须是表" in e]
    assert len(msgs) == 2      # server、model 各一条，不按键重复


# -- config：文件读取异常 -----------------------------------------------------

def test_config_unreadable_path_is_config_error(tmp_path):
    """file= 指向目录时 IsADirectoryError（OSError）应转 ConfigError。"""
    with pytest.raises(ConfigError) as ei:
        load_config(file=str(tmp_path), env={})
    assert any("读取失败" in e for e in ei.value.errors)


def test_config_undecodable_toml_is_config_error(tmp_path):
    """非 UTF-8 文件：UnicodeDecodeError 应转 ConfigError。"""
    path = tmp_path / "bad.toml"
    path.write_bytes(b"\xff\xfe[server]\n")
    with pytest.raises(ConfigError) as ei:
        load_config(file=str(path), env={})
    assert any("编码错误" in e or "语法错误" in e for e in ei.value.errors)


# -- config：数值转换 ---------------------------------------------------------

def test_config_int_key_rejects_inf(tmp_path):
    """TOML `inf` 合法；int(inf) 抛 OverflowError，须转 ConfigError。"""
    with pytest.raises(ConfigError) as ei:
        _load(tmp_path, "[server]\nport = inf\n")
    assert any("server.port" in e for e in ei.value.errors)


def test_config_int_key_rejects_fraction(tmp_path):
    """`context = 4096.9` 不再静默截断。"""
    with pytest.raises(ConfigError) as ei:
        _load(tmp_path, "[model]\ncontext = 4096.9\n")
    assert any("不接受小数" in e for e in ei.value.errors)


# -- config：旧 MODEL_DIR 的冒号分隔 -----------------------------------------

def test_config_legacy_model_dir_colon_split():
    cfg = load_config(env={"RKLLM_MODEL_DIR": "/var/a:/var/b"})
    assert cfg.model.search_dirs == ("/var/a", "/var/b")


# -- config：heartbeat 组合校验 ----------------------------------------------

def test_config_heartbeat_timeout_must_be_twice_interval(tmp_path):
    with pytest.raises(ConfigError) as ei:
        _load(tmp_path, "[engine]\nheartbeat_interval_s = 10\n"
                        "heartbeat_timeout_s = 5\n")
    assert any("heartbeat_timeout_s" in e for e in ei.value.errors)


def test_config_default_heartbeat_pair_is_valid():
    cfg = load_config(env={})           # 默认 5 / 20
    assert cfg.engine.heartbeat_timeout_s >= 2 * cfg.engine.heartbeat_interval_s


# -- caps：seed=0 / 空容器 / n 标注 ------------------------------------------

def test_caps_seed_zero_is_rejected():
    reject, warn = caps.check_request_params({"seed": 0}, policy="reject")
    assert "seed" in reject


def test_caps_empty_containers_are_not_reported():
    reject, warn = caps.check_request_params(
        {"logit_bias": {}, "modalities": []}, policy="reject")
    assert reject == [] and warn == []


def test_caps_n_zero_labelled_as_lt_one():
    reject, _ = caps.check_request_params({"n": 0}, policy="reject")
    assert "n<1" in reject


def test_caps_reasoning_effort_is_emulated():
    assert caps.param_tier("reasoning_effort") == "emulated"


# -- abi：可选符号 -----------------------------------------------------------

def test_abi_unused_symbols_are_optional():
    assert "rkllm_set_function_tools" in abi._OPTIONAL_SYMBOLS
    assert "rkllm_set_chat_template" in abi._OPTIONAL_SYMBOLS
    assert "rkllm_set_function_tools" not in abi._REQUIRED_SYMBOLS
    assert "rkllm_init" in abi._REQUIRED_SYMBOLS


# -- schemas：completions 端点接受 native 采样参数 ---------------------------

def test_completions_accepts_native_sampling_params():
    body = CompletionRequest(prompt="hi", top_k=5, repeat_penalty=1.2,
                             presence_penalty=0.1, frequency_penalty=0.1,
                             reasoning_effort="low", enable_thinking=True,
                             max_completion_tokens=64)
    assert body.top_k == 5 and body.enable_thinking is True
