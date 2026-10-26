"""CLI 面：命令集合与 config 子命令（不需要模型、不碰 NPU）。

cli.py 是运维唯一入口，此前 0% 覆盖。这里覆盖：解析器骨架、config
show/docs/validate/init 四条路径、以及"非法配置必须非零退出"。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rkllm import cli  # noqa: E402


def _run(argv, capsys):
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_parser_keeps_all_subcommands():
    parser = cli.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    for name in ("chat", "run", "serve", "list", "info", "selftest", "config"):
        assert name in choices, "缺少子命令 %s" % name


def test_serve_keeps_legacy_arguments():
    """旧 unit 的 ExecStart 依赖这些参数名，升级后必须还在。"""
    parser = cli.build_parser()
    serve = parser._subparsers._group_actions[0].choices["serve"]
    options = {a.dest for a in serve._actions}
    for legacy in ("host", "port", "api_key", "model", "max_queue",
                   "queue_timeout", "system", "no_history", "context",
                   "threads", "platform", "model_dir", "top_k", "top_p",
                   "temperature", "repeat_penalty"):
        assert legacy in options, "serve 丢了旧参数 --%s" % legacy


def test_config_docs_lists_all_keys(capsys):
    code, out, _ = _run(["config", "docs"], capsys)
    assert code == 0
    assert out.count("| `") >= 85


def test_config_show_json(capsys):
    code, out, _ = _run(["config", "show", "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert "values" in payload and "sources" in payload
    assert len(payload["values"]) == 85


def test_config_show_text(capsys):
    code, out, _ = _run(["config", "show"], capsys)
    assert code == 0
    assert "[server]" in out and "bind_host" in out


def test_config_validate_rejects_bad_value(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("[server]\nport = 999999\n", encoding="utf-8")
    code, _, err = _run(["config", "validate", "--config", str(bad)], capsys)
    assert code != 0
    assert "port" in err


def test_config_init_writes_template(tmp_path, capsys):
    target = tmp_path / "rkllm.toml"
    code, out, _ = _run(["config", "init", "--path", str(target)], capsys)
    assert code == 0 and target.is_file()
    text = target.read_text(encoding="utf-8")
    assert "[server]" in text and "[model]" in text


def test_config_init_refuses_overwrite(tmp_path, capsys):
    target = tmp_path / "rkllm.toml"
    target.write_text("existing", encoding="utf-8")
    code, _, err = _run(["config", "init", "--path", str(target)], capsys)
    assert code == 1 and "已存在" in err


def test_config_init_client_template(tmp_path, capsys):
    target = tmp_path / "mcp.toml"
    code, _, _ = _run(["config", "init", "--client", "--path", str(target)],
                      capsys)
    assert code == 0
    assert "target_endpoint" in target.read_text(encoding="utf-8")


def test_parse_returns_error_for_bad_cli_value(capsys):
    """argparse 层面的非法值应走 SystemExit（由 main 之外的入口处理）。"""
    with pytest.raises(SystemExit):
        cli.main(["serve", "--port", "not-a-number"])


# ---------------------------------------------------------------------------
# list / info / selftest / run（不需要真实模型）
# ---------------------------------------------------------------------------
def test_list_finds_models(tmp_path, capsys):
    (tmp_path / "qwen3-4b.rkllm").write_bytes(b"\0" * 2048)
    (tmp_path / "notes.txt").write_text("x")
    code, out, _ = _run(["list", "--model-dir", str(tmp_path), "--json"], capsys)
    assert code == 0
    items = json.loads(out)
    assert items and items[0]["name"] == "qwen3-4b.rkllm"
    assert items[0]["size_mb"] >= 0


def test_list_reports_empty(tmp_path, capsys):
    code, _, err = _run(["list", "--model-dir", str(tmp_path)], capsys)
    assert code == 1 and "未找到" in err


def test_info_json(capsys):
    code, out, _ = _run(["info", "--json"], capsys)
    assert code == 0
    info = json.loads(out)
    assert info["rkllm"] and info["python"]
    assert "platform_detected" in info and "model_dirs" in info


def test_selftest_runs_and_reports(capsys):
    code, out, err = _run(["selftest"], capsys)
    assert code in (0, 1)
    assert out or err                     # 有结论输出即可


def test_run_without_library_fails_cleanly(capsys):
    """没有 librkllmrt.so 时必须给出可读错误，而不是 traceback。"""
    code = cli.main(["run", "missing.rkllm", "hello"])
    out = capsys.readouterr()
    assert code != 0
    assert "Traceback" not in (out.err + out.out)


def test_search_dirs_dedupes_and_keeps_order(tmp_path):
    extra = [str(tmp_path), str(tmp_path)]
    dirs = cli._search_dirs(extra)
    assert dirs.count(str(tmp_path)) == 1
    assert dirs[0] == str(tmp_path)


def test_resolve_model_path_passthrough(tmp_path):
    explicit = tmp_path / "m.rkllm"
    explicit.write_text("x")
    assert cli._resolve_model_path(str(explicit), []) == str(explicit)


def test_serve_startup_path(monkeypatch, tmp_path):
    """serve 主流程（不真起 uvicorn、不 spawn worker）：预算→装配→run→收尾。"""
    model = tmp_path / "qwen3-4b.rkllm"
    model.write_bytes(b"\0" * 4096)

    class _Sup:
        state = "ready"
        metrics = {"worker_restarts_total": 0}

        def __init__(self):
            self.shut = False

        def is_ready(self):
            return True

        def shutdown(self):
            self.shut = True

    sup = _Sup()
    ran = {}
    # A-4：serve 的实现已拆到 cli_serve 模块，patch 目标随之调整（断言不变）
    monkeypatch.setattr("rkllm.cli_serve._build_supervisor",
                        lambda cfg, use_fake: sup)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: ran.update(kwargs=k))
    code = cli.main(["serve", str(model), "--port", "18099",
                     "--host", "127.0.0.1", "--api-key", "k"])
    assert code == 0
    assert ran["kwargs"].get("port") == 18099
    assert sup.shut is True                  # 退出时释放 supervisor


def test_serve_rejects_missing_model(capsys):
    code, _, err = _run(["serve", "/nonexistent/x.rkllm"], capsys)
    assert code == 2 and "模型文件不存在" in err
