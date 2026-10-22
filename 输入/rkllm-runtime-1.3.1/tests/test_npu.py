"""NPU 探测的单元测试（monkeypatch 路径常量，不依赖真实板端）。

覆盖五种探测信号与“全都没有”的拒绝路径；DRM render node 用假目录树 +
monkeypatch realpath 模拟，避免 Windows 上建符号链接需要特权。
"""
import os

from rkllm.platform import npu
from rkllm.platform.npu import npu_status


def _layout(tmp_path, monkeypatch, name):
    root = tmp_path / name
    root.mkdir()
    monkeypatch.setattr(npu, "NPU_DEBUGFS", str(root / "debugfs"))
    monkeypatch.setattr(npu, "LEGACY_DEVICE", str(root / "dev-rknpu"))
    monkeypatch.setattr(npu, "DRM_CLASS", str(root / "drm"))
    monkeypatch.setattr(npu, "MODULE_PATH", str(root / "module"))
    return root


def test_version_file_wins(tmp_path, monkeypatch):
    root = _layout(tmp_path, monkeypatch, "version")
    debugfs = root / "debugfs"
    debugfs.mkdir()
    (debugfs / "version").write_text("RKNPU driver: v0.9.8\n", encoding="utf-8")
    ok, detail = npu_status()
    assert ok is True
    assert "v0.9.8" in detail


def test_debugfs_dir_without_version(tmp_path, monkeypatch):
    root = _layout(tmp_path, monkeypatch, "dir")
    (root / "debugfs").mkdir()
    ok, detail = npu_status()
    assert ok is True
    assert "version not readable" in detail


def test_drm_render_node(tmp_path, monkeypatch):
    root = _layout(tmp_path, monkeypatch, "drm")
    node = root / "drm" / "renderD128"
    node.mkdir(parents=True)
    real = os.path.realpath
    monkeypatch.setattr(
        npu.os.path, "realpath",
        lambda path: "/sys/bus/platform/drivers/rknpu" if "renderD128" in path
        else real(path))
    ok, detail = npu_status()
    assert ok is True
    assert "renderD128" in detail


def test_legacy_device_node(tmp_path, monkeypatch):
    root = _layout(tmp_path, monkeypatch, "legacy")
    (root / "dev-rknpu").write_text("", encoding="utf-8")
    ok, detail = npu_status()
    assert ok is True
    assert "legacy" in detail


def test_module_without_node(tmp_path, monkeypatch):
    root = _layout(tmp_path, monkeypatch, "module")
    (root / "module").mkdir()
    ok, detail = npu_status()
    assert ok is True
    assert "module loaded" in detail


def test_nothing_found(tmp_path, monkeypatch):
    _layout(tmp_path, monkeypatch, "none")
    ok, detail = npu_status()
    assert ok is False
    assert "no NPU driver found" in detail
