"""发布前一致性检查：版本、changelog、配置键、caps 文档、deb 元数据、发布 tag。

用法：
    python 工具/编译工具链/release_check.py
    python 工具/编译工具链/release_check.py --deb <deb 路径>

changelog 检查覆盖**全部**条目，而不只是第一行：模板必须恰好有一个
`@VERSION@` 占位（构建时替换成 DEFAULT_VERSION），渲染后的版本序列必须严格
递减、无重复，每个条目都必须以 ` -- Maintainer <email>  <date>` 结尾——否则
两个 `@VERSION@` 会构建出两条同名发布（1.3.1-21 的历史问题），dpkg 照装不误，
只有读包的人才发现。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# make_deb.py / verify_deb.py 和本文件同目录；目录被脚本或测试直接 import 时，
# sys.path[0] 不一定已在，这里显式补上。
sys.path.insert(0, str(HERE))

PACKAGING = ROOT / "输入" / "rkllm-runtime-1.3.1"
SRC = PACKAGING / "src"
CHANGELOG = PACKAGING / "changelog.Debian.in"
# 2026-11-18：新增 model.lora_paths → 86 键；2026-11-19：A 方案移除
# mcp.task_threshold_s / mcp.max_input_chars 两个孤儿键 → 84 键（.d 45 键 + 启动最小集）
EXPECTED_KEYS = 84
EXPECTED_PACKAGE = "rkllm-runtime"
# 每一个模板条目都以这行结尾；构建时 @MAINTAINER@/@DATE@ 由 make_deb 替换。
PLACEHOLDER_TRAILER = " -- @MAINTAINER@  @DATE@"
TAG_VERSION_RE = re.compile(r"^v?(?P<version>\d+\.\d+\.\d+-\d+)$")


def _read_version() -> str:
    text = (SRC / "rkllm" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^DEFAULT_VERSION\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        raise SystemExit("DEFAULT_VERSION not found")
    return match.group(1)


def _render_changelog(template: str, version: str) -> str:
    """用构建器同一张替换表渲染模板（epoch 固定，检查结果与运行时刻无关）。"""
    import make_deb

    upstream, _, revision = version.partition("-")
    ctx = make_deb.BuildContext(
        arch="arm64", package=make_deb.DEFAULT_PACKAGE,
        upstream_version=upstream, revision=revision,
        maintainer=make_deb.DEFAULT_MAINTAINER, profile="full", epoch=0)
    ctx.finalize()
    return ctx.apply(template)


def _check_changelog(version: str, changelog: Path | None = None) -> list[str]:
    """检查模板的全部条目，以及渲染后（即打进 deb 的那份）的完整结构。"""
    import verify_deb

    path = changelog or CHANGELOG
    if not path.is_file():
        return ["changelog template not found: %s" % path]
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []

    # 1) 模板层：最新条目只能是一个 @VERSION@ 占位，否则一次构建会产出
    #    多个同名版本头（历史 bug：两条 1.3.1-21）。
    placeholders = text.count("@VERSION@")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if placeholders != 1:
        problems.append(
            "changelog has %d @VERSION@ placeholders; exactly one (the newest "
            "entry) is allowed, otherwise the build emits duplicate versions"
            % placeholders)
    if "(@VERSION@)" not in first_line:
        problems.append(
            "changelog first entry is not the @VERSION@ placeholder: %r"
            % first_line[:90])

    # 2) 模板层：每个条目都必须以占位形式的 maintainer/date 结尾。
    entries, parse_problems = verify_deb.parse_changelog(text)
    problems += parse_problems
    if not entries:
        problems.append("changelog template has no entries")
    for entry in entries:
        if entry.trailer != PLACEHOLDER_TRAILER:
            problems.append(
                "changelog entry %s (line %d) trailer is %r, expected %r"
                % (entry.version, entry.line_no, entry.trailer,
                   PLACEHOLDER_TRAILER))

    # 3) 渲染层：与 make_deb 实际打包内容一致，执行完整 Debian 结构校验。
    if "-" not in version or not version.rsplit("-", 1)[1].isdigit():
        problems.append(
            "DEFAULT_VERSION %r is not <upstream>-<revision>" % version)
        return problems
    rendered = _render_changelog(text, version)
    rendered_entries, rendered_problems = verify_deb.parse_changelog(rendered)
    problems += ["rendered: " + problem for problem in rendered_problems]
    problems += ["rendered: " + problem for problem in verify_deb.validate_changelog(
        rendered_entries, expected_package=EXPECTED_PACKAGE,
        expected_version=version)]
    return problems


def _check_pyproject() -> list[str]:
    text = (PACKAGING / "pyproject.toml").read_text(encoding="utf-8")
    if 'dynamic = ["version"]' not in text:
        return ["pyproject.toml must declare dynamic version"]
    return []


def _check_config() -> list[str]:
    sys.path.insert(0, str(SRC))
    from rkllm.config import SCHEMA
    return [] if len(SCHEMA) == EXPECTED_KEYS else [
        "config keys %d != %d" % (len(SCHEMA), EXPECTED_KEYS)]


def _child_env() -> dict[str, str]:
    """子进程统一用 UTF-8 写管道。

    Windows 下 Python 写管道默认用本地代码页，而这里按 UTF-8 解码；中文路径
    （工具/dist/…）会被解成 U+FFFD，控制台再打印时直接 UnicodeEncodeError。
    """
    return dict(os.environ, PYTHONIOENCODING="utf-8")


def _check_caps_doc() -> list[str]:
    doc = ROOT / "文档" / "对话接口用法.md"
    result = subprocess.run(
        [sys.executable, str(HERE / "render_caps_doc.py"),
         "--check", str(doc)],
        cwd=str(ROOT), env=_child_env(), capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    return [] if result.returncode == 0 else [
        "caps doc check failed: %s" % result.stdout.strip()[:200]]


def _check_deb(path: Path, version: str) -> list[str]:
    # 子进程 cwd 是工具目录；调用者给的可能是相对仓库根的路径，先变成绝对路径，
    # 否则 verify_deb 会在自己目录下找不到 deb（旧实现的真实缺陷）。
    resolved = path.resolve()
    result = subprocess.run(
        [sys.executable, str(HERE / "verify_deb.py"), str(resolved)],
        cwd=str(HERE), env=_child_env(), capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return ["verify_deb failed: %s" % result.stdout.strip()[-300:]]
    if version not in result.stdout:
        return ["verify_deb did not report version %s" % version]
    return []


# ---------------------------------------------------------------- release tag
def _git(arguments: list[str], root: Path = ROOT) -> str | None:
    """本机有 git 时跑一条只读命令；否则返回 None（不访问网络、不推送）。"""
    executable = shutil.which("git")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "-C", str(root)] + arguments,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def tag_version_problem(tag: str, tagged_version: str) -> str | None:
    """发布 tag 名里的版本与它指向的树内 DEFAULT_VERSION 不一致时返回问题。

    纯函数，便于不依赖 git 的单测。`tag` 是本地 tag 名（可带 v 前缀），
    `tagged_version` 是该 tag 提交里读到的 DEFAULT_VERSION。例如 1.3.1-18
    指向的树实际是 1.3.1-17 —— 该 tag 无效，发布前必须重打或声明；本检查
    只读，不创建/移动/推送任何 tag。
    """
    match = TAG_VERSION_RE.match(tag)
    if match is None:
        return None
    expected = match.group("version")
    if tagged_version == expected:
        return None
    return ("tag %s names version %s but points at a tree whose "
            "DEFAULT_VERSION is %s" % (tag, expected, tagged_version))


def _check_tags(version: str, root: Path = ROOT) -> list[str]:
    """对比本地 tag 名与 tag 指向的 DEFAULT_VERSION（当前版本不一致才算失败）。"""
    tags = _git(["tag", "--list"], root)
    if tags is None:
        print("[SKIP] tag/version check: git or a local tag list is unavailable")
        return []
    init_rel = "输入/rkllm-runtime-1.3.1/src/rkllm/__init__.py"
    problems: list[str] = []
    current_tag_present = False
    for tag in (line.strip() for line in tags.splitlines() if line.strip()):
        match = TAG_VERSION_RE.match(tag)
        if match is None:
            continue
        if match.group("version") == version:
            current_tag_present = True
        body = _git(["show", "%s:%s" % (tag, init_rel)], root)
        if body is None:
            continue
        found = re.search(r'^DEFAULT_VERSION\s*=\s*"([^"]+)"', body, re.M)
        if found is None:
            continue
        problem = tag_version_problem(tag, found.group(1))
        if problem is None:
            continue
        if match.group("version") == version:
            problems.append(problem + " (the release tag is invalid)")
        else:
            # 历史 tag 不阻断构建，但必须作为发布决策被记录/处理。
            print("[WARN] %s (historical tag; decide before publishing)" % problem)
    if not current_tag_present:
        print("[INFO] no local tag encodes the current version %s" % version)
    return problems


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
    problems += _check_tags(version)
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
