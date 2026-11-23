"""P1-R1 / P1-R2 回归：Debian changelog 结构与发布检查。

P1-R1（1.3.1-17..21 拆条）历史问题：`changelog.Debian.in` 只有两个
`rkllm-runtime (@VERSION@)` 头，构建时两个占位都替换成 DEFAULT_VERSION，于是
包里出现两条 1.3.1-21；1.3.1-16 及更新的条目又缺 Debian 必需的
` -- Maintainer <email>  <date>` 结尾行。dpkg 只比较第一条，所以坏 changelog
能正常安装，只有读包的人才会发现。

这些用例不检查字符串常量，而是：
  * 把**真实的** `changelog.Debian.in` 复制到临时目录，交给 release_check 的
    真实检查函数（与构建使用同一张 @TOKEN@ 替换表）渲染、解析、校验；
  * 对真实模板做定点破坏（重复占位 / 缺 trailer / 版本重复 / 乱序 / 坏日期 /
    缺空行），确认检查会拒绝，并确认两个 @VERSION@ 确实渲染成两条同版本；
  * 用 make_deb 的自写 ar+tar 打包器构建一次最小 deb，按 ar/tar 解包取出
    `changelog.Debian.gz`，验证版本序列单调、无重复、每条都有 maintainer/date，
    再走 verify_deb 的打包检查路径（不依赖 dpkg-deb）。

P1-R2：1.3.1-18 tag 实际指向 DEFAULT_VERSION=1.3.1-17 的树。本次不改 tag、
不推送，只锁定 release_check 的 tag/version 一致性检查逻辑（读本地 tag，
纯只读；历史坏 tag 记 WARN 不阻断，当前版本的 tag 不一致才 FAIL）。
"""

from __future__ import annotations

import gzip
import io
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent        # 输入/rkllm-runtime-1.3.1/tests
PKG = HERE.parent                             # 输入/rkllm-runtime-1.3.1
REPO = PKG.parents[1]                         # 仓库根 rkllm-runtime/
TOOLS = REPO / "工具" / "编译工具链"
CHANGELOG_IN = PKG / "changelog.Debian.in"

sys.path.insert(0, str(TOOLS))

import make_deb        # noqa: E402
import release_check   # noqa: E402
import verify_deb      # noqa: E402

VERSION = release_check._read_version()
PACKAGE = "rkllm-runtime"

# 真实模板里每一条都必须有的占位 trailer（构建时替换成 Maintainer/日期）。
TRAILER = " -- @MAINTAINER@  @DATE@"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _template() -> str:
    """真实的 changelog 模板（不是测试里手写的近似字符串）。"""
    return CHANGELOG_IN.read_text(encoding="utf-8")


def _write_template(directory: Path, text: str) -> Path:
    """写到临时的 changelog.Debian.in，模拟发布目录里的那一份。"""
    path = directory / "changelog.Debian.in"
    path.write_text(text, encoding="utf-8")
    return path


def _render(text: str) -> str:
    """用 release_check 的真实渲染（make_deb 的替换表）得到打包内容。"""
    return release_check._render_changelog(text, VERSION)


def _problems(directory: Path, text: str) -> list[str]:
    return release_check._check_changelog(
        VERSION, _write_template(directory, text))


def _joined(problems: list[str]) -> str:
    return "\n".join(problems)


def _minimal_deb(directory: Path, changelog_text: str) -> Path:
    """用 make_deb 的自写 ar+tar 打包器构建最小 deb（无 dpkg-deb 也可用）。

    只带 changelog + 两个 conffile，够验证「构建 -> ar/tar 解包 -> 校验」
    这条链路；完整 payload 的 deb 由发布流程单独构建。
    """
    epoch = 0
    ctx = make_deb.BuildContext(
        arch="arm64", package=PACKAGE,
        upstream_version=make_deb.UPSTREAM_VERSION,
        revision=VERSION.rsplit("-", 1)[1],
        maintainer=make_deb.DEFAULT_MAINTAINER,
        profile="full", epoch=epoch)
    ctx.finalize()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9,
                       mtime=epoch) as gz:
        gz.write(changelog_text.encode("utf-8"))
    items = [
        make_deb.Shipped(dest=make_deb.SERVER_DEFAULTS,
                         data=b"RKLLM_MODEL=\n", mode=0o644),
        make_deb.Shipped(dest=make_deb.SERVER_DROPIN,
                         data=b"# drop-in\n", mode=0o644),
        make_deb.Shipped(dest=ctx.docdir + "/changelog.Debian.gz",
                         data=buffer.getvalue(), mode=0o644),
    ]
    return make_deb.build_deb(items, ctx, directory / "dist", directory / "build")


def _unpack_changelog(deb: Path) -> str:
    """按 ar + tar 解包（不调用 dpkg-deb / dpkg），返回 changelog 文本。"""
    members = verify_deb.parse_ar(deb.read_bytes())
    assert "data.tar.xz" in members, members.keys()
    target = "./usr/share/doc/%s/changelog.Debian.gz" % PACKAGE
    with tarfile.open(fileobj=io.BytesIO(members["data.tar.xz"]),
                      mode="r:xz") as tf:
        member = tf.getmember(target)
        payload = tf.extractfile(member).read()
    return gzip.decompress(payload).decode("utf-8")


# ---------------------------------------------------------------------------
# P1-R1：模板结构（真实 changelog.Debian.in）
# ---------------------------------------------------------------------------


def test_template_has_exactly_one_version_placeholder():
    """两个 @VERSION@ 会构建出两条同名发布；模板只允许一个（最新条目）。"""
    text = _template()
    assert text.count("@VERSION@") == 1, "模板又出现多个 @VERSION@ 占位"
    first = text.splitlines()[0]
    assert "(@VERSION@)" in first, first


def test_template_every_entry_has_maintainer_date_placeholder():
    """1.3.1-16 及以后（以及更早）每条都必须有 maintainer/date 结尾行。"""
    entries, problems = verify_deb.parse_changelog(_template())
    assert problems == [], problems
    assert len(entries) >= 19, "模板条目数异常: %d" % len(entries)
    for entry in entries:
        assert entry.trailer == TRAILER, (
            "条目 %s 缺 maintainer/date 占位行: %r"
            % (entry.version, entry.trailer))
        assert entry.blank_after_header, (
            "条目 %s 版本头后缺空行" % entry.version)
    versions = {entry.version for entry in entries}
    assert entries[0].version == "@VERSION@"
    assert entries[-1].version == "1.3.1-1"
    # 本次修复的核心：1.3.1-17..21 必须是各自独立的条目（21 是 @VERSION@ 占位，
    # 渲染后等于当前版本，见下面的渲染用例）。
    for revision in range(17, 21):
        assert "1.3.1-%d" % revision in versions, "缺 1.3.1-%d 条目" % revision


def test_real_template_in_temp_dir_passes_release_check(tmp_path):
    """把真实的 changelog.Debian.in 放到临时目录，真实检查函数必须通过。"""
    path = _write_template(tmp_path, _template())
    assert release_check._check_changelog(VERSION, path) == []


def test_real_template_renders_to_valid_debian_changelog(tmp_path):
    """渲染结果就是 make_deb 会打进 deb 的内容，必须是合法 Debian changelog。"""
    rendered = _render(_template())
    assert "@VERSION@" not in rendered
    assert "@MAINTAINER@" not in rendered
    assert "@DATE@" not in rendered
    assert not [token for token in ("@UPSTREAM_VERSION@", "@ARCH@")
                if token in rendered]
    entries, problems = verify_deb.parse_changelog(rendered)
    assert problems == [], problems
    assert verify_deb.validate_changelog(
        entries, expected_package=PACKAGE, expected_version=VERSION) == []
    versions = [entry.version for entry in entries]
    assert versions[0] == VERSION
    assert versions[-1] == "1.3.1-1"
    # 单调递减且无重复
    assert len(versions) == len(set(versions)), "版本序列有重复: %s" % versions
    keys = [verify_deb._changelog_version_key(v) for v in versions]
    assert all(a > b for a, b in zip(keys, keys[1:])), versions


# ---------------------------------------------------------------------------
# P1-R1：定点破坏真实模板 -> 检查必须拒绝
# ---------------------------------------------------------------------------


def test_duplicate_version_placeholder_is_rejected_and_reproduces_bug(tmp_path):
    """历史 bug 的精确复现：两个 @VERSION@ 渲染成两条 1.3.1-21。"""
    broken = _template().replace(
        "rkllm-runtime (1.3.1-20) unstable", "rkllm-runtime (@VERSION@) unstable", 1)
    problems = _problems(tmp_path, broken)
    joined = _joined(problems)
    assert "@VERSION@" in joined
    assert "duplicate" in joined, joined
    # 真实渲染：两个占位确实变成两条 1.3.1-21（不是字符串断言而是构建结果）。
    entries, _ = verify_deb.parse_changelog(_render(broken))
    rendered_versions = [entry.version for entry in entries]
    assert rendered_versions.count(VERSION) == 2, rendered_versions


def test_missing_maintainer_date_trailer_is_rejected(tmp_path):
    broken = _template().replace(
        "  * 未知路径返回 OpenAI 形状 404；extract 空 text 校验；空 meta 不再计缓存命中。"
        "\n\n -- @MAINTAINER@  @DATE@\n\nrkllm-runtime (1.3.1-18)",
        "  * 未知路径返回 OpenAI 形状 404；extract 空 text 校验；空 meta 不再计缓存命中。"
        "\n\nrkllm-runtime (1.3.1-18)", 1)
    assert broken != _template(), "定点破坏没生效，模板内容变了？"
    joined = _joined(_problems(tmp_path, broken))
    assert "1.3.1-19" in joined
    assert "trailer" in joined or "maintainer" in joined.lower(), joined


def test_duplicate_version_header_is_rejected(tmp_path):
    broken = _template().replace(
        "rkllm-runtime (1.3.1-19) unstable", "rkllm-runtime (1.3.1-20) unstable", 1)
    joined = _joined(_problems(tmp_path, broken))
    assert "duplicate" in joined and "1.3.1-20" in joined, joined


def test_non_monotonic_version_sequence_is_rejected(tmp_path):
    """把 -20 与 -19 对调：版本序列不再严格递减。"""
    text = _template()
    text = text.replace("rkllm-runtime (1.3.1-20) unstable",
                        "rkllm-runtime (SWAPPED-HEADER) unstable", 1)
    text = text.replace("rkllm-runtime (1.3.1-19) unstable",
                        "rkllm-runtime (1.3.1-20) unstable", 1)
    text = text.replace("rkllm-runtime (SWAPPED-HEADER) unstable",
                        "rkllm-runtime (1.3.1-19) unstable", 1)
    joined = _joined(_problems(tmp_path, text))
    assert "not strictly decreasing" in joined, joined


def test_unparsable_trailer_date_is_rejected(tmp_path):
    broken = _template().replace(
        " -- @MAINTAINER@  @DATE@\n\nrkllm-runtime (1.3.1-19)",
        " -- @MAINTAINER@  whenever the wind blows\n"
        "\n\nrkllm-runtime (1.3.1-19)", 1)
    assert broken != _template()
    joined = _joined(_problems(tmp_path, broken))
    assert "date" in joined.lower(), joined


def test_missing_blank_line_after_version_header_is_rejected(tmp_path):
    broken = _template().replace(
        "rkllm-runtime (1.3.1-19) unstable; urgency=medium\n\n  [ 1.3.1-19",
        "rkllm-runtime (1.3.1-19) unstable; urgency=medium\n  [ 1.3.1-19", 1)
    assert broken != _template()
    joined = _joined(_problems(tmp_path, broken))
    assert "blank line" in joined, joined


# ---------------------------------------------------------------------------
# P1-R1：构建一次 deb，按 ar/tar 解包验证 changelog
# ---------------------------------------------------------------------------


def test_built_deb_changelog_sequence_is_monotonic_and_unique(tmp_path):
    """make_deb（自写 ar+tar，无 dpkg-deb）构建 -> 解包 changelog -> 全量校验。"""
    deb = _minimal_deb(tmp_path, _render(_template()))
    assert deb.is_file() and deb.stat().st_size > 0
    packaged = _unpack_changelog(deb)
    entries, problems = verify_deb.parse_changelog(packaged)
    assert problems == [], problems
    assert verify_deb.validate_changelog(
        entries, expected_package=PACKAGE, expected_version=VERSION) == []
    versions = [entry.version for entry in entries]
    assert versions[0] == VERSION, versions
    assert len(versions) == len(set(versions)), "解包后版本序列有重复: %s" % versions
    keys = [verify_deb._changelog_version_key(v) for v in versions]
    assert all(a > b for a, b in zip(keys, keys[1:])), versions
    assert all(entry.trailer for entry in entries)


def test_verify_deb_check_data_passes_on_built_deb(tmp_path):
    """verify_deb 的打包检查路径（check_container/check_control/check_data）同步。"""
    deb = _minimal_deb(tmp_path, _render(_template()))
    verify_deb.results.clear()
    members = verify_deb.check_container(deb)
    control_info = verify_deb.check_control(members)
    verify_deb.check_data(members, control_info)
    changelog_failures = [
        row for row in verify_deb.results
        if row[0] == "FAIL" and row[1].startswith("changelog")]
    assert changelog_failures == [], changelog_failures
    passed = [row for row in verify_deb.results
              if row[0] == "PASS" and row[1] == "changelog structure"]
    assert passed, "verify_deb 没有做 changelog 结构检查"
    assert "strictly decreasing" in passed[0][2]


def test_verify_deb_check_data_catches_broken_packaged_changelog(tmp_path):
    """把坏 changelog 打进 deb，verify_deb 必须报 FAIL（不是只看首行版本）。"""
    broken = _render(_template()).replace(
        "rkllm-runtime (1.3.1-19) unstable", "rkllm-runtime (1.3.1-20) unstable", 1)
    deb = _minimal_deb(tmp_path, broken)
    verify_deb.results.clear()
    members = verify_deb.check_container(deb)
    control_info = verify_deb.check_control(members)
    verify_deb.check_data(members, control_info)
    failures = [
        row for row in verify_deb.results
        if row[0] == "FAIL" and row[1] == "changelog structure"]
    assert failures, "坏 changelog 没有被 verify_deb 拦下"
    assert any("duplicate" in message for _, _, message in failures), failures


# ---------------------------------------------------------------------------
# P1-R2：tag / version 一致性（只读本地 tag；本次不改 tag、不推送）
# ---------------------------------------------------------------------------


def test_tag_version_problem_detects_the_invalid_1_3_1_18_tag():
    """已知发布决策：1.3.1-18 tag 指向 86b6061，而该树 DEFAULT_VERSION=1.3.1-17。"""
    problem = release_check.tag_version_problem("1.3.1-18", "1.3.1-17")
    assert problem is not None
    assert "1.3.1-18" in problem and "1.3.1-17" in problem
    assert release_check.tag_version_problem("v1.3.1-18", "1.3.1-18") is None
    assert release_check.tag_version_problem("1.3.1-21", "1.3.1-21") is None
    assert release_check.tag_version_problem("nightly", "1.3.1-17") is None


def test_check_tags_flags_mismatched_current_release_tag(monkeypatch):
    def fake_git(arguments, root=None):
        if arguments == ["tag", "--list"]:
            return "1.3.1-21\n"
        return 'DEFAULT_VERSION = "1.3.1-20"\n'

    monkeypatch.setattr(release_check, "_git", fake_git)
    problems = release_check._check_tags(VERSION, Path("."))
    assert problems, "当前版本的 tag 指向别的版本却未报错"
    assert "1.3.1-21" in problems[0]


def test_check_tags_reports_historical_invalid_tag_as_warning(monkeypatch, capsys):
    """历史坏 tag（1.3.1-18）只 WARN，不阻断本次发布检查。"""
    def fake_git(arguments, root=None):
        if arguments == ["tag", "--list"]:
            return "1.3.1-21\n1.3.1-18\n"
        if arguments[1].startswith("1.3.1-18:"):
            return 'DEFAULT_VERSION = "1.3.1-17"\n'
        return 'DEFAULT_VERSION = "1.3.1-21"\n'

    monkeypatch.setattr(release_check, "_git", fake_git)
    problems = release_check._check_tags(VERSION, Path("."))
    captured = capsys.readouterr().out
    assert problems == [], problems
    assert "1.3.1-18" in captured and "WARN" in captured


def test_check_tags_skips_cleanly_without_git(monkeypatch, capsys):
    """本机无 git 时 tag 检查必须 SKIP，不能把发布检查变成失败。"""
    monkeypatch.setattr(release_check.shutil, "which", lambda name: None)
    assert release_check._check_tags(VERSION, Path(".")) == []
    assert "SKIP" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 发布检查自身：deb 模式必须把相对路径解析成绝对路径
# ---------------------------------------------------------------------------


def test_check_deb_hands_verify_deb_an_absolute_path(monkeypatch, tmp_path):
    """旧缺陷：调用者在仓库根给 `工具/dist/x.deb`，verify_deb 子进程 cwd 是工具目录，
    相对路径解析不到 deb。检查必须先 resolve，并用 UTF-8 子进程管道。"""
    captured: dict[str, object] = {}

    class FakeResult:
        returncode = 0
        stdout = "control Version %s\n" % VERSION

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env") or {}
        return FakeResult()

    monkeypatch.setattr(release_check.subprocess, "run", fake_run)
    relative = (Path("工具") / "dist"
                / ("rkllm-runtime_%s_arm64.deb" % VERSION))
    problems = release_check._check_deb(relative, VERSION)
    assert problems == []
    passed_path = Path(captured["command"][-1])
    assert passed_path.is_absolute(), captured["command"]
    assert captured["env"].get("PYTHONIOENCODING") == "utf-8"

