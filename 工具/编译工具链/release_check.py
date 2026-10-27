"""发布前一致性检查：版本、changelog、配置键、caps 文档、deb 元数据。

用法：
    python 工具/编译工具链/release_check.py
    python 工具/编译工具链/release_check.py --deb 工具/dist/rkllm-runtime_1.3.1-17_arm64.deb
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PACKAGING = ROOT / "输入" / "rkllm-runtime-1.3.1"
SRC = PACKAGING / "src"
EXPECTED_KEYS = 85


def _read_version() -> str:
    text = (SRC / "rkllm" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^DEFAULT_VERSION\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        raise SystemExit("DEFAULT_VERSION not found")
    return match.group(1)


def _check_config() -> list[str]:
    sys.path.insert(0, str(SRC))
    from rkllm.config import SCHEMA
    return [] if len(SCHEMA) == EXPECTED_KEYS else [
        "config keys %d != %d" % (len(SCHEMA), EXPECTED_KEYS)]


def _check_changelog(version: str) -> list[str]:
    text = (PACKAGING / "changelog.Debian.in").read_text(encoding="utf-8")
    first = text.splitlines()[0]
    if "(@VERSION@)" not in first:
        return ["changelog first entry is not @VERSION@: %r" % first]
    return []


def _check_pyproject() -> list[str]:
    text = (PACKAGING / "pyproject.toml").read_text(encoding="utf-8")
    if 'dynamic = ["version"]' not in text:
        return ["pyproject.toml must declare dynamic version"]
    return []


def _check_caps_doc() -> list[str]:
    doc = ROOT / "文档" / "对话接口用法.md"
    result = subprocess.run(
        [sys.executable, str(HERE / "render_caps_doc.py"),
         "--check", str(doc)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return [] if result.returncode == 0 else [
        "caps doc check failed: %s" % result.stdout.strip()[:200]]


def _check_deb(path: Path, version: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, str(HERE / "verify_deb.py"), str(path)],
        cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return ["verify_deb failed: %s" % result.stdout.strip()[-300:]]
    if version not in result.stdout:
        return ["verify_deb did not report version %s" % version]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deb", default="")
    args = ap.parse_args()
    version = _read_version()
    problems: list[str] = []
    problems += _check_pyproject()
    problems += _check_changelog(version)
    problems += _check_config()
    problems += _check_caps_doc()
    if args.deb:
        deb = Path(args.deb)
        if not deb.exists():
            problems.append("deb not found: %s" % deb)
        else:
            problems += _check_deb(deb, version)
    if problems:
        for problem in problems:
            print("[FAIL] %s" % problem)
        return 1
    print("[PASS] release check: version=%s, keys=%d%s"
          % (version, EXPECTED_KEYS,
             ", deb verified" if args.deb else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
