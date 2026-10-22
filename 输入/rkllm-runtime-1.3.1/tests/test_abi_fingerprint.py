"""库指纹路径解析的单测（D10）。

ctypes 的 _name 是候选名不是真实路径，直接 stat 会失败；这里覆盖
「候选就是文件」与「候选不存在」两条路径，不依赖板端 /proc。
"""
import os

from rkllm import abi


def test_resolve_existing_file(tmp_path, monkeypatch):
    lib = tmp_path / "librkllmrt.so"
    lib.write_bytes(b"\x7fELF")
    assert abi._resolve_library_path(str(lib)) == os.path.realpath(str(lib))


def test_resolve_missing_file_returns_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(abi.os.path, "isfile", lambda path: False)
    assert abi._resolve_library_path("librkllmrt.so") == "librkllmrt.so"
