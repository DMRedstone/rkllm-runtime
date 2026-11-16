"""配置 drop-in（`/etc/default/rkllm-server.d/*.conf`）优先级测试。"""
from __future__ import annotations

import pathlib

from rkllm import config


def _write(d: pathlib.Path, name: str, text: str) -> None:
    (d / name).write_text(text, encoding="utf-8")


def test_dropin_overrides_default_and_later_file_wins(tmp_path, monkeypatch):
    _write(tmp_path, "10-model.conf", "RKLLM_PORT=9000\nRKLLM_MODEL=/m/a.rkllm\n")
    _write(tmp_path, "20-later.conf", "RKLLM_PORT=9001\n")
    monkeypatch.setattr(config, "DEFAULT_DROPIN_DIR", str(tmp_path))
    cfg = config.load_config(env={})
    assert cfg.values["server.port"] == 9001        # 后文件覆盖先文件
    assert cfg.values["model.path"] == "/m/a.rkllm"  # 旧变量名在 .d 里也生效
    assert cfg.sources["server.port"] == ".d"


def test_env_overrides_dropin(tmp_path, monkeypatch):
    _write(tmp_path, "10-model.conf", "RKLLM_PORT=9000\n")
    monkeypatch.setattr(config, "DEFAULT_DROPIN_DIR", str(tmp_path))
    cfg = config.load_config(env={"RKLLM_SERVER_PORT": "9002"})
    assert cfg.values["server.port"] == 9002
    assert cfg.sources["server.port"] == "env"


def test_missing_dropin_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_DROPIN_DIR", str(tmp_path / "nope"))
    cfg = config.load_config(env={})
    assert cfg.values["server.port"] == 8080
    assert cfg.sources["server.port"] == "default"
