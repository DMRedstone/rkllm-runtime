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


# -- 2026-11-20 观察项修复回归 ------------------------------------------------

def test_queue_bump_is_atomic_under_concurrency():
    """metrics 的 += 在多线程下会丢更新；bump 必须在锁内累加。"""
    import threading
    from rkllm.server.queue import InferenceQueue

    q = InferenceQueue()
    threads = [threading.Thread(
        target=lambda: [q.bump("requests_total") for _ in range(500)])
        for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert q.metrics["requests_total"] == 4000


def test_hist_stats_reads_samples_without_private_attrs():
    """ops._hist_stats 用公开 collect() 解析均值，不读 _sum/_count。"""
    from rkllm.server.metricsutil import (TTFT_BUCKETS, new_labelled_histogram,
                                          new_registry)
    from rkllm.server.routes.ops import _hist_stats

    hist = new_labelled_histogram(new_registry(), "t_hist_seconds", "d",
                                  TTFT_BUCKETS)
    for value in (1.0, 3.0):
        hist.labels(endpoint="chat.completions").observe(value)
    mean, samples = _hist_stats(hist, "chat.completions")
    assert samples == 2 and abs(mean - 2.0) < 1e-9
    assert _hist_stats(hist, "completions") == (None, 0)


def test_stream_timer_handles_pipeline_without_perf():
    """StreamTimer 只认 dict；perf 为 None 时不得崩（tuple 分支已删）。"""
    import time

    class _Pipe:
        endpoint = "chat.completions"

        def last_perf(self):
            return None

    class _Hist:
        def labels(self, **_kw):
            return self

        def observe(self, _v):
            pass

    class _Q:
        metrics = {"duration_sum": 0.0}
        ttft_hist = _Hist()
        decode_hist = _Hist()
        tok_s_hist = _Hist()
        bump = staticmethod(lambda k, d=1: None)
        gauge = staticmethod(lambda k, v: None)

    class _Svc:
        queue = _Q()

    from rkllm.server.pipeline_runner import StreamTimer
    timer = StreamTimer()
    timer.first_token()
    time.sleep(0.001)
    timer.finish(_Svc(), _Pipe())        # 不抛即通过


def test_config_empty_env_clears_toml_string(tmp_path, monkeypatch):
    """空串对 str 键是显式清空上游层（如关掉 TOML 里的 api_key）。"""
    path = tmp_path / "rkllm.toml"
    path.write_text('[server]\napi_key = "secret"\n', encoding="utf-8")
    cfg = load_config(file=str(path), env={"RKLLM_SERVER_API_KEY": ""})
    assert cfg.server.api_key == ""


def test_config_unknown_toml_key_warns(tmp_path, caplog):
    """拼错的键不再静默丢弃：TOML 未知键打 warning，启动照常。"""
    path = tmp_path / "rkllm.toml"
    path.write_text("[server]\nprot = 9090\n", encoding="utf-8")
    with caplog.at_level("WARNING", logger="rkllm.config"):
        cfg = load_config(file=str(path), env={})
    assert cfg.server.port == 8080          # 默认值，未知键未生效
    assert any("prot" in r.getMessage() for r in caplog.records)


def test_config_explicit_file_skips_dropin_dir(tmp_path, monkeypatch):
    """显式 file= 时不吃系统 .d（隔离配置）。"""
    from rkllm import config as cfgmod
    dropin = tmp_path / "d"
    dropin.mkdir()
    (dropin / "99-override.conf").write_text("RKLLM_SERVER_PORT=9999",
                                             encoding="utf-8")
    monkeypatch.setattr(cfgmod, "DEFAULT_DROPIN_DIR", str(dropin))
    path = tmp_path / "rkllm.toml"
    path.write_text("[server]\nport = 8080\n", encoding="utf-8")
    assert load_config(file=str(path), env={}).server.port == 8080
    # 不传 file 时 .d 仍生效
    assert load_config(env={}).server.port == 9999


def test_config_list_items_are_stringified(tmp_path):
    path = tmp_path / "rkllm.toml"
    path.write_text('[server]\ncors_origins = [1, "http://a"]\n',
                    encoding="utf-8")
    cfg = load_config(file=str(path), env={})
    assert cfg.server.cors_origins == ("1", "http://a")


def test_abi_describe_rc_and_struct_sizes():
    assert abi.describe_rc(0) == "code 0（成功）"
    assert "prompt cache" in abi.describe_rc(10000)
    assert abi.describe_rc(12345) == "code 12345"      # 未登记码不猜原因
    sizes = abi.struct_sizes()
    assert sizes["_Param"] > 0 and "_PromptCacheParam" in sizes


def test_config_read_timeout_desc_mentions_keepalive():
    from rkllm.config import SCHEMA
    key = next(k for k in SCHEMA if k.full == "server.read_timeout_s")
    assert "keep-alive" in key.desc
