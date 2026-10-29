#!/usr/bin/env python3
"""Build a Debian binary package (".deb") from the Rockchip RKLLM release tarball.

The package is assembled without dpkg-deb so that it can be produced on hosts
that have none (e.g. Windows). The same staged tree can be repacked with
dpkg-deb later for cross-checking, see --emit-tree.

Layout follows Debian Policy / FHS:
    /usr/lib/<multiarch-triplet>/librkllmrt.so     runtime shared library
    /usr/include/rkllm/rkllm.h                     public API header
    /usr/lib/<triplet>/{pkgconfig,cmake/rkllm}     build integration
    /usr/lib/<package>/librknnrt.so                private RKNN runtime copy
    /etc/ld.so.conf.d/<package>.conf               private lib dir for ldconfig
    /usr/bin/rkllm-*                               board tuning scripts
    /usr/share/doc/<package>/                      docs, license, changelog
    /usr/share/<package>/examples/                 upstream example sources

Clean-removal contract:
    * every shipped file is listed in /var/lib/dpkg/info/<package>.list
    * nothing is written outside that list at install time (no cache models,
      no /usr/local writes, no users, no systemd units, no DKMS module)
    * the one file dpkg would normally keep behind is declared in
      DEBIAN/conffiles - /etc/default/rkllm-server, so that local edits survive
      upgrades - and postrm unlinks it on `remove` as well as `purge`
    * postinst/postrm only run ldconfig, tell systemd to re-read units, and
      remove that one config file
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import posixpath
import re
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGING = Path(os.environ.get("RKLLM_PACKAGING_DIR") or (ROOT.parent / "输入" / "rkllm-runtime-1.3.1"))

# 上游 release tarball 位置（软编码）：优先环境变量 RKLLM_RELEASE_TARBALL，
# 未设置时使用仓库内随附的精简上游库。换机器 / 换版本只需设该环境变量，不必改脚本。
SOURCE_VERSION_FILE = PACKAGING / "src" / "rkllm" / "__init__.py"


def _source_version() -> tuple[str, str]:
    """版本唯一来源：src/rkllm/__init__.py 的 DEFAULT_VERSION。

    返回 (upstream_version, revision)。构建器不再维护第二份版本号；显式
    --revision 与源码不一致时直接报错，避免 dpkg 版本与运行时自报版本双源。
    """
    try:
        text = SOURCE_VERSION_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read {SOURCE_VERSION_FILE}: {exc}")
    match = re.search(r'^DEFAULT_VERSION\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        raise SystemExit(f"DEFAULT_VERSION not found in {SOURCE_VERSION_FILE}")
    version = match.group(1)
    if "-" not in version:
        raise SystemExit(
            f"DEFAULT_VERSION={version!r} must look like 1.3.1-17")
    upstream, revision = version.split("-", 1)
    return upstream, revision


UPSTREAM_VERSION, DEFAULT_REVISION = _source_version()
SOURCE_TARBALL = Path(
    os.environ.get("RKLLM_RELEASE_TARBALL")
    or ROOT.parent / "输入" / ("rknn-llm-release-v%s-minimal.tar.gz"
                               % UPSTREAM_VERSION))
TARBALL_PREFIX = "rknn-llm-release-v%s" % UPSTREAM_VERSION
DEFAULT_PACKAGE = "rkllm-runtime"
DEFAULT_MAINTAINER = "RedStone <DMRedStone@users.noreply.github.com>"

# multiarch triplet -> upstream directory naming inside the release tarball
ARCHITECTURES = {
    "arm64": {"triplet": "aarch64-linux-gnu", "upstream": "aarch64", "opencv": "opencv-linux-aarch64"},
}

# Dependencies derived from DT_NEEDED of librkllmrt.so:
#   libgomp.so.1, libpthread.so.0, libstdc++.so.6, libdl.so.2, libm.so.6,
#   libgcc_s.so.1, libc.so.6 with GLIBC_2.29 symbol versions.
# python3 (>= 3.13): the new stack behind `rkllm serve` uses tomllib and
#   3.13-only stdlib; trixie ships 3.13 as the default python3.
# python3-fastapi / python3-uvicorn / python3-pydantic: the new FastAPI stack
#   (apt route, decided 2026-10-21 - no vendored wheels). trixie versions meet
#   the floors (pydantic >= 2.9 supports 3.13); verified on board before
#   release, see the appendix in 文档/开发记录.md.
DEPEND_TEMPLATE = ("libc6 (>= 2.29), libgomp1, libstdc++6, "
                   "python3:any (>= 3.13), python3-fastapi, python3-uvicorn, "
                   "python3-pydantic")

# The API server unit keeps a fixed name independent of --package so that
# operators always find it under the same name.
SERVER_UNIT = "rkllm-server.service"
# New-stack python sources (server only). The MCP connector is client-side
# source kept in the repo root 本地模型连接器/ and is NOT packaged into the deb.
SRC_TREE = ROOT.parent / "输入" / "rkllm-runtime-1.3.1" / "src" / "rkllm"
# The service configuration is a dpkg conffile: dpkg keeps local edits across an
# upgrade instead of overwriting them with the shipped template. Its text is
# deliberately free of @VERSION@ so that a revision bump does not change it -
# dpkg only rewrites a conffile whose packaged content actually changed, and a
# rewritten file would also lose a `chmod 600` an operator applied for the API
# key. Removal is handled by postrm, see packaging/postrm.
SERVER_DEFAULTS = "/etc/default/rkllm-server"

FILE_MODE = 0o644
EXEC_MODE = 0o755
DIR_MODE = 0o755

ROOT_UID, ROOT_GID = 0, 0
ROOT_USER, ROOT_GROUP = "root", "root"


@dataclass
class Shipped:
    """One actionable payload entry."""

    dest: str                      # absolute POSIX path inside the package
    data: bytes | None = None      # regular file content
    mode: int = FILE_MODE
    link: str | None = None        # symlink target, when data is None
    origin: str = ""               # provenance for the audit manifest
    note: str = ""


@dataclass
class BuildContext:
    arch: str
    package: str
    upstream_version: str
    revision: str
    maintainer: str
    profile: str
    epoch: int
    # derived
    triplet: str = ""
    upstream_arch: str = ""
    libdir: str = ""               # usr/lib/<triplet>
    privlibdir: str = ""           # /usr/lib/<package>
    examples_dir: str = ""         # /usr/share/<package>/examples
    docdir: str = ""
    version: str = ""
    subst: dict = field(default_factory=dict)

    def finalize(self):
        meta = ARCHITECTURES[self.arch]
        self.triplet = meta["triplet"]
        self.upstream_arch = meta["upstream"]
        self.version = f"{self.upstream_version}-{self.revision}"
        self.libdir = f"usr/lib/{self.triplet}"
        self.privlibdir = f"/usr/lib/{self.package}"
        self.examples_dir = f"/usr/share/{self.package}/examples"
        self.docdir = f"/usr/share/doc/{self.package}"
        self.subst = {
            "@PACKAGE@": self.package,
            "@VERSION@": self.version,
            "@UPSTREAM_VERSION@": self.upstream_version,
            "@UPSTREAM_ARCH@": self.upstream_arch,
            "@ARCH@": self.arch,
            "@TRIPLET@": self.triplet,
            "@MAINTAINER@": self.maintainer,
            "@DEPENDS@": DEPEND_TEMPLATE,
            "@PKGINCLUDE_SUBDIR@": "rkllm",
            "@INCLUDEDIR_ABS@": "/usr/include",
            "@PKGINCLUDE_DIR_ABS@": "/usr/include/rkllm",
            "@LIBDIR_ABS@": f"/{self.libdir}",
            "@LIBDIR@": self.libdir,
            "@PRIVLIBDIR_ABS@": self.privlibdir,
            "@PKGEXAMPLESDIR@": self.examples_dir,
            "@DOCDIR_ABS@": self.docdir,
            "@SERVER_UNIT@": SERVER_UNIT,
            "@SERVER_DEFAULTS@": SERVER_DEFAULTS,
            "@PREFIX@": "/usr",
            "@DATE@": time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(self.epoch)),
            "@DATE_ISO@": time.strftime("%Y-%m-%d", time.gmtime(self.epoch)),
        }

    def apply(self, text: str) -> str:
        for token, value in self.subst.items():
            text = text.replace(token, value)
        return text


# --------------------------------------------------------------------------
# classification: upstream tarball path -> installed path, or None to drop
# --------------------------------------------------------------------------

class Classifier:
    def __init__(self, ctx: BuildContext):
        self.ctx = ctx
        self.dropped: dict[str, int] = {}      # reason -> count
        self.dropped_bytes: dict[str, int] = {}

    def drop(self, rel: str, size: int, reason: str):
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        self.dropped_bytes[reason] = self.dropped_bytes.get(reason, 0) + size
        return None

    def map(self, rel: str, size: int) -> tuple[str, int] | None:
        """Return (destination path, mode) or None when the file is excluded."""
        ctx = self.ctx
        low = rel.lower()
        # --- foreign architectures / hosts -----------------------------------
        if rel.startswith("rkllm-runtime/Android") or "/OpenCV-android-sdk" in rel or "/Android/" in rel:
            return self.drop(rel, size, "Android 目标")
        if "uclibc" in low:
            return self.drop(rel, size, "uclibc 目标")
        if rel.startswith("rkllm-toolkit/"):
            return self.drop(rel, size, "PC 端转换工具(x86_64 wheel)")
        if rel.startswith("rknpu-driver/"):
            return self.drop(rel, size, "NPU 内核驱动(已安装,不属于本包)")
        other_arch = (["armhf", "armeabi", "arm64-v8a"] if ctx.arch == "arm64"
                      else ["aarch64", "arm64-v8a"])
        if _mentions_arch(low, other_arch):
            return self.drop(rel, size, "非目标架构")
        if ctx.profile == "slim" and ("/opencv-linux-" in low or "/haarcascades" in low):
            return self.drop(rel, size, "slim: 剔除自带 OpenCV(依赖系统 libopencv-dev)")
        if low.endswith(("/.gitkeep", "/.gitignore")):
            return self.drop(rel, size, "空占位文件")
        if rel == "LICENSE":
            return self.drop(rel, size, "内容并入 Debian copyright")

        name = posixpath.basename(rel)

        # --- runtime library -------------------------------------------------
        rt = f"rkllm-runtime/Linux/librkllm_api/{ctx.upstream_arch}/librkllmrt.so"
        if rel == rt:
            return f"/{ctx.libdir}/librkllmrt.so", FILE_MODE
        if rel == "rkllm-runtime/Linux/librkllm_api/include/rkllm.h":
            return "/usr/include/rkllm/rkllm.h", FILE_MODE

        # --- board tuning / measurement scripts ------------------------------
        if rel.startswith("scripts/") and rel.endswith(".sh"):
            stem = name[:-3].replace("_", "-")
            return f"/usr/bin/rkllm-{stem}", EXEC_MODE

        # --- documentation ---------------------------------------------------
        if rel in ("README.md", "benchmark.md", "CHANGELOG.md"):
            return f"{ctx.docdir}/{name}", FILE_MODE
        if rel.startswith("doc/") and rel.endswith(".pdf"):
            return f"{ctx.docdir}/{name}", FILE_MODE
        if rel == "res/framework.jpg":
            return f"{ctx.docdir}/framework.jpg", FILE_MODE

        # --- examples --------------------------------------------------------
        if rel.startswith("examples/"):
            parts = rel.split("/")
            demo = parts[1]
            if demo == "multimodal_model_demo":
                mm = self._map_multimodal(rel, size)
                if mm is None:
                    return None
                return mm
            if name == "build-android.sh":
                return self.drop(rel, size, "Android 构建脚本")
            rest = "/".join(parts[2:])
            return f"{ctx.examples_dir}/{demo}/{rest}", EXEC_MODE if name.endswith(".sh") else FILE_MODE

        return self.drop(rel, size, "未归类内容")

    def _map_multimodal(self, rel: str, size: int) -> tuple[str, int] | None:
        ctx = self.ctx
        base = "examples/multimodal_model_demo/"
        tail = rel[len(base):]
        low = tail.lower()

        if low.startswith(("deploy/3rdparty/librknnrt/android", "deploy/3rdparty/librknnrt/linux/rknn_server")):
            return self.drop(rel, size, "多模态示例: RKNN proxy server(不属于 RKLLM)")
        if tail.startswith("deploy/3rdparty/librknnrt/Linux/librknn_api/include/"):
            return f"/usr/include/rknn/{posixpath.basename(tail)}", FILE_MODE
        if tail == f"deploy/3rdparty/librknnrt/Linux/librknn_api/{ctx.upstream_arch}/librknnrt.so":
            return f"{ctx.privlibdir}/librknnrt.so", FILE_MODE
        if "opencv-linux-" in low:
            opencv_dir = ARCHITECTURES[ctx.arch]["opencv"]
            if opencv_dir not in tail:
                return self.drop(rel, size, "非目标架构的 OpenCV 包")
            return f"{ctx.examples_dir}/multimodal_model_demo/{tail}", FILE_MODE
        mode = EXEC_MODE if tail.endswith(".sh") else FILE_MODE
        return f"{ctx.examples_dir}/multimodal_model_demo/{tail}", mode


# --------------------------------------------------------------------------
# content transformations
# --------------------------------------------------------------------------

def fix_shell_script(name: str, data: bytes) -> bytes:
    """Boards ship /bin/sh, several upstream scripts carry an Android shebang."""
    text = data.decode("utf-8", "replace")
    lines = text.split("\n")
    if lines and lines[0].startswith("#!"):
        lines[0] = "#!/bin/sh" if "/system/bin/sh" in lines[0] else lines[0]
    else:
        lines.insert(0, "#!/bin/sh")
    return "\n".join(lines).encode("utf-8")


def load_templates() -> tuple[dict[str, str], list[dict]]:
    overrides_dir = PACKAGING / "overrides"
    overrides: dict[str, str] = {}
    for path in overrides_dir.rglob("*"):
        if path.is_file() and "/examples/" in str(path).replace("\\", "/"):
            rel_posix = str(path.relative_to(overrides_dir)).replace("\\", "/")
            if rel_posix.startswith("examples/"):
                overrides[rel_posix[len("examples/"):]] = path.read_text(encoding="utf-8")
    patches = json.loads((overrides_dir / "text-patches.json").read_text(encoding="utf-8"))["patches"]
    return overrides, patches


def read_packaging(name: str) -> str:
    return (PACKAGING / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# staging
# --------------------------------------------------------------------------

def stage(ctx: BuildContext, source_tarball: Path) -> tuple[list[Shipped], dict]:
    overrides, patches = load_templates()
    patch_index = {p["file"]: p for p in patches}
    classifier = Classifier(ctx)

    files: dict[str, Shipped] = {}

    def add(dest: str, data: bytes, mode: int, origin: str, note: str = ""):
        if dest in files:
            raise SystemExit(f"duplicate destination: {dest}")
        files[dest] = Shipped(dest=dest, data=data, mode=mode, origin=origin, note=note)

    with tarfile.open(source_tarball, "r:gz") as src:
        for member in src:
            rel = member.name
            if rel.startswith(TARBALL_PREFIX + "/"):
                rel = rel[len(TARBALL_PREFIX) + 1:]
            if not rel or rel == TARBALL_PREFIX:
                continue
            if member.isdir():
                continue
            if not member.isfile():
                continue

            handle = src.extractfile(member)
            blob = handle.read()

            # keep upstream license text aside for the Debian copyright file
            if rel == "LICENSE":
                files["__license__"] = Shipped(dest="__license__", data=blob, mode=FILE_MODE,
                                               origin=rel, note="embedded into Debian copyright")
                classifier.map(rel, member.size)
                continue

            mapped = classifier.map(rel, member.size)
            if mapped is None:
                continue
            dest, mode = mapped

            # in-place textual patches
            patch = patch_index.get(rel)
            if patch:
                text = blob.decode("utf-8", "replace")
                applied = []
                for rep in patch["replacements"]:
                    find, new = rep["find"], ctx.apply(rep["replace"])
                    if find == new:
                        continue
                    if find not in text:
                        raise SystemExit(f"patch target not found in {rel}:\n{find[:120]}")
                    text = text.replace(find, new)
                    applied.append(rep["find"].split("\n")[0][:60])
                blob = text.encode("utf-8")
                note = "text patch: " + "; ".join(applied)
            else:
                note = ""

            if dest.startswith("/usr/bin/rkllm-") or rel.startswith("scripts/"):
                blob = fix_shell_script(posixpath.basename(dest), blob)
                note = (note + "; " if note else "") + "shebang 归一化为 #!/bin/sh"

            add(dest, blob, mode, rel, note)

    # --- replace upstream example build files with the adapted ones ----------
    for upstream_rel, content in overrides.items():
        full = "examples/" + upstream_rel
        mapped_classifier = classifier
        # destination follows the same mapping rules as the file it replaces
        parts = full.split("/")
        demo = parts[1]
        rest = "/".join(parts[2:])
        dest = f"{ctx.examples_dir}/{demo}/{rest}" if demo != "multimodal_model_demo" \
            else f"{ctx.examples_dir}/multimodal_model_demo/{rest}"
        mode = EXEC_MODE if upstream_rel.endswith(".sh") else FILE_MODE
        text = ctx.apply(content)
        files.pop(dest, None)
        add(dest, text.encode("utf-8"), mode, f"packaging/overrides/examples/{upstream_rel}",
            "改写: 路径从 SDK 相对路径改为系统安装路径")

    # --- generated metadata --------------------------------------------------
    license_text = files.pop("__license__").data.decode("utf-8", "replace")

    copyright_body = ctx.apply(read_packaging("copyright.head"))
    for line in license_text.strip().splitlines():
        copyright_body += (" " + line if line.strip() else " .") + "\n"
    add(f"{ctx.docdir}/copyright", copyright_body.encode("utf-8"), FILE_MODE,
        "packaging/copyright.head + upstream LICENSE", "Debian copyright(含完整许可证正文)")

    changelog = ctx.apply(read_packaging("changelog.Debian.in"))
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=ctx.epoch) as gz:
        gz.write(changelog.encode("utf-8"))
    add(f"{ctx.docdir}/changelog.Debian.gz", buf.getvalue(), FILE_MODE,
        "packaging/changelog.Debian.in", "gzip -9, mtime 归零(可复现构建)")

    add(f"{ctx.docdir}/README.Debian", ctx.apply(read_packaging("README.Debian.in")).encode("utf-8"),
        FILE_MODE, "packaging/README.Debian.in", "安装使用说明")

    pc = ctx.apply(read_packaging("rkllm.pc.in"))
    add(f"/{ctx.libdir}/pkgconfig/rkllm.pc", pc.encode("utf-8"), FILE_MODE,
        "packaging/rkllm.pc.in", "pkg-config 元数据")

    cmake_dir = f"/{ctx.libdir}/cmake/rkllm"
    add(f"{cmake_dir}/rkllm-config.cmake", ctx.apply(read_packaging("rkllm-config.cmake.in")).encode("utf-8"),
        FILE_MODE, "packaging/rkllm-config.cmake.in", "CMake 包配置(导入目标 rkllm::rkllmrt)")
    add(f"{cmake_dir}/rkllm-config-version.cmake",
        ctx.apply(read_packaging("rkllm-config-version.cmake.in")).encode("utf-8"), FILE_MODE,
        "packaging/rkllm-config-version.cmake", "CMake 版本文件")

    add(f"/etc/ld.so.conf.d/{ctx.package}.conf", (ctx.privlibdir + "\n").encode("utf-8"), FILE_MODE,
        "generated", "ldconfig 私有库路径;随包删除,由 postinst/postrm 刷新缓存")

    # --- rkllm: local chat + OpenAI-compatible API ---------------------------
    # /usr/bin/rkllm：唯一命令入口。1.3.1-16 起它把 serve/run/chat/list/info/
    # selftest/config 全部委派给新栈（/usr/lib/rkllm/current），而 systemd 单元
    # 与 conffile 保持 1.3.1-15 的名字与内容不变。
    add("/usr/bin/rkllm",
        ctx.apply(read_packaging("launcher/rkllm")).encode("utf-8"), EXEC_MODE,
        "packaging/launcher/rkllm",
        "统一命令入口:校验 python3>=3.13 → sys.path 挂 /usr/lib/rkllm/current → rkllm.cli")

    # /usr/lib/systemd/system is the location dh_installsystemd uses on merged-usr
    # systems; /lib is a symlink into /usr on every supported release.
    add(f"/usr/lib/systemd/system/{SERVER_UNIT}",
        ctx.apply(read_packaging("systemd/rkllm-server.service")).encode("utf-8"), FILE_MODE,
        "packaging/systemd/rkllm-server.service",
        "可选 API 服务单元;不 enable 不 start,由使用者决定")

    add(SERVER_DEFAULTS, ctx.apply(read_packaging("default/rkllm-server")).encode("utf-8"),
        FILE_MODE, "packaging/default/rkllm-server",
        "systemd 单元的环境配置(conffile:升级保留本地修改,postrm 在 remove/purge 时删除)")

    # --- manual pages --------------------------------------------------------
    # Policy requires man pages to be compressed; the mtime is pinned so the
    # archive stays reproducible. The translated copy goes under zh_CN, which is
    # how man-db picks a localised page for the user's locale.
    for source, dest, note in (
            ("man/rkllm.1", "/usr/share/man/man1/rkllm.1.gz", "英文 man 手册"),
            ("man/zh_CN/rkllm.1", "/usr/share/man/zh_CN/man1/rkllm.1.gz", "简体中文 man 手册")):
        text = ctx.apply(read_packaging(source))
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=ctx.epoch) as gz:
            gz.write(text.encode("utf-8"))
        add(dest, buffer.getvalue(), FILE_MODE, f"packaging/{source}",
            note + "(gzip -9, mtime 归零)")

    files[f"/usr/include/rkllm.h"] = Shipped(dest="/usr/include/rkllm.h", data=None, mode=FILE_MODE,
                                            link="rkllm/rkllm.h", origin="generated",
                                            note="兼容按上游习惯 #include \"rkllm.h\" 的工程")

    # --- service stack (2026-10-21, apt route) ------------------------------
    # Sources ship as plain .py under /usr/lib/rkllm/current: small tree, the
    # board operator can read the installed file, and dpkg upgrade/rollback
    # semantics stay untouched. The MCP connector is client-side source
    # (repo root 本地模型连接器/) and is deliberately not part of this deb.
    #
    # 注意：只有一套入口。/usr/bin/rkllm 是 launcher（把本目录挂进 sys.path
    # 后调 rkllm.cli），"serve" 子命令在本目录内实现；systemd 单元仍是
    # rkllm-server.service，conffile 仍是 /etc/default/rkllm-server
    # —— 运维视角零变化。
    new_pkg = "/usr/lib/rkllm/current/rkllm"
    for path in sorted(SRC_TREE.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(SRC_TREE).as_posix()
        add(f"{new_pkg}/{rel}", path.read_bytes(), FILE_MODE,
            f"src/rkllm/{rel}", "新栈模块树(FastAPI 服务端/worker/agent)")

    items = sorted(files.values(), key=lambda f: f.dest)
    return items, {"dropped": classifier.dropped, "dropped_bytes": classifier.dropped_bytes}


# --------------------------------------------------------------------------
# tar / ar writers
# --------------------------------------------------------------------------

def _base_info(name: str, ctx: BuildContext) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.uid = ROOT_UID
    ti.gid = ROOT_GID
    ti.uname = ROOT_USER
    ti.gname = ROOT_GROUP
    ti.mtime = ctx.epoch
    return ti


def write_tar_xz(target: Path, entries, ctx: BuildContext, *, is_control: bool) -> None:
    """Write a tar.xz stream. `entries` is a list of (name, mode, data|None, link|None)."""
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as raw, tarfile.open(fileobj=raw, mode="w:xz",
                                                format=tarfile.GNU_FORMAT) as tf:
        for name, mode, data, link in entries:
            ti = _base_info(name, ctx)
            if data is None and link is None:
                ti.type = tarfile.DIRTYPE
                ti.mode = DIR_MODE
                tf.addfile(ti)
                continue
            if data is None and link is not None:
                ti.type = tarfile.SYMTYPE
                ti.mode = EXEC_MODE
                ti.linkname = link
                tf.addfile(ti)
                continue
            ti.type = tarfile.REGTYPE
            ti.mode = mode
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def collect_dirs(paths) -> list[str]:
    dirs = set()
    for p in paths:
        parts = p.strip("/").split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    return sorted(dirs)


def _mentions_arch(rel_low: str, markers) -> bool:
    """路径里是否出现了某个架构标记，按"非字母数字边界"匹配。

    原来的 `marker in low` 是纯子串：`libmyarmhflib.so`、`armhfnotes.txt`
    这类不含架构语义的文件名会被当成"非目标架构"丢掉，而丢弃只进统计、
    构建照常成功——查都查不出来。

    边界用 `[^a-z0-9]` 而不是 `[-_.]` 分词：`arm64-v8a` 自带 `-`，分词会把
    它拆成 `arm64` + `v8a` 反而匹配不上。
    """
    for marker in markers:
        pattern = r"(?:^|[^a-z0-9])" + re.escape(marker) + r"(?:$|[^a-z0-9])"
        if re.search(pattern, rel_low):
            return True
    return False


def ar_member(name: str, payload: bytes) -> bytes:
    header = f"{name:<16}{0:<12}{ROOT_UID:<6}{ROOT_GID:<6}{0o100644:<8o}{len(payload):<10}`\n"
    # ar headers are 60 bytes; values are space padded, not zero padded.
    # assert 在 `python -O` 下会被整条剔除，坏头就会静默产出损坏的 deb。
    if len(header) != 60:
        raise SystemExit("ar header for %r is %d bytes, expected 60: %r"
                         % (name, len(header), header))
    blob = header.encode("ascii") + payload
    if len(blob) % 2:
        blob += b"\n"
    return blob


def build_deb(items, ctx: BuildContext, out_dir: Path, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_paths = [f.dest for f in items]
    # A conffile that is not part of the payload is a packaging error dpkg only
    # reports at install time, so refuse to build one.
    conffile_paths = [SERVER_DEFAULTS]
    for cf in conffile_paths:
        if cf not in file_paths:
            raise SystemExit(f"conffile {cf} is not in the payload")
    entries = [("./" + d, DIR_MODE, None, None) for d in collect_dirs(file_paths)]
    md5_lines = []
    for item in items:
        name = "./" + item.dest.lstrip("/")
        md5 = hashlib.md5(item.data).hexdigest() if item.data is not None else None
        entries.append((name, item.mode, item.data, item.link))
        if md5 is not None:
            md5_lines.append(f"{md5}  {item.dest.lstrip('/')}")

    data_tar = write_tar_xz(tmp_dir / "data.tar.xz", entries, ctx, is_control=False)

    scripts = {
        "postinst": (read_packaging("postinst"), EXEC_MODE),
        "prerm": (read_packaging("prerm"), EXEC_MODE),
        "postrm": (read_packaging("postrm"), EXEC_MODE),
    }
    control_entries = [("./", DIR_MODE, None, None)]
    # Paths are absolute, one per line, sorted - dpkg reads the list verbatim
    # and only treats a file as a conffile when this member names it.
    control_entries.append(("./conffiles", FILE_MODE,
                            ("\n".join(conffile_paths) + "\n").encode("utf-8"), None))
    control_text = ctx.apply(read_packaging("control.in"))
    control_entries.append(("./control", FILE_MODE, control_text.encode("utf-8"), None))
    control_entries.append(("./md5sums", FILE_MODE, ("\n".join(md5_lines) + "\n").encode("utf-8"), None))
    for script, (body, mode) in scripts.items():
        control_entries.append((f"./{script}", mode, ctx.apply(body).encode("utf-8"), None))
    write_tar_xz(tmp_dir / "control.tar.xz", control_entries, ctx, is_control=True)

    deb_name = f"{ctx.package}_{ctx.version}_{ctx.arch}.deb"
    deb_path = out_dir / deb_name
    with deb_path.open("wb") as fh:
        fh.write(b"!<arch>\n")
        fh.write(ar_member("debian-binary", b"2.0\n"))
        fh.write(ar_member("control.tar.xz", (tmp_dir / "control.tar.xz").read_bytes()))
        fh.write(ar_member("data.tar.xz", (tmp_dir / "data.tar.xz").read_bytes()))
    return deb_path


def emit_tree(items, ctx: BuildContext, tree_dir: Path) -> None:
    """Materialise the payload as real files, so dpkg-deb can be used later.

    Deleting an existing tree is deliberately left to the caller: the host's
    safe-delete policy intercepts unlink() calls made from a Python process, and
    a build that silently cannot refresh its staging tree is worse than one that
    stops and asks. Remove the directory (or pass --tree-dir) and re-run.
    """
    if tree_dir.exists():
        raise SystemExit(
            "staging tree already exists: %s\n"
            "Remove it first (rm -rf %s) or pass --tree-dir <path>."
            % (tree_dir, tree_dir))
    debian_dir = tree_dir / "DEBIAN"
    debian_dir.mkdir(parents=True)
    for item in items:
        target = tree_dir / item.dest.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.data is None:
            try:
                target.symlink_to(item.link)
            except OSError as exc:
                raise SystemExit(
                    "cannot create symlink %s -> %s: %s\n"
                    "The staged tree is only useful if it matches the real package; "
                    "writing a regular file in its place would make dpkg-deb's view "
                    "differ from what we build. Fix permissions/filesystem, or drop "
                    "--emit-tree." % (item.dest, item.link, exc)) from exc
            # Windows 上无权限时 symlink_to 可能不抛异常，却留下一个解析不了的
            # 条目——只捕获 OSError 挡不住这种情况，必须回读确认。
            if not target.is_symlink():
                raise SystemExit(
                    "symlink %s -> %s was not created as a symlink (platform or "
                    "permission). On Windows this needs Developer Mode or an elevated "
                    "shell; alternatively build in WSL." % (item.dest, item.link))
        else:
            target.write_bytes(item.data)
            os.chmod(target, item.mode)
    for name in ("conffiles", "control", "postinst", "prerm", "postrm"):
        if name == "conffiles":
            body = SERVER_DEFAULTS + "\n"
        else:
            body = ctx.apply(read_packaging(name if name != "control" else "control.in"))
        target = debian_dir / name
        target.write_text(body, encoding="utf-8")
        target.chmod(EXEC_MODE if name in ("postinst", "prerm", "postrm") else FILE_MODE)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Build the RKLLM Debian package.")
    ap.add_argument("--arch", choices=sorted(ARCHITECTURES), default="arm64")
    ap.add_argument("--profile", choices=("full", "slim"), default="full",
                    help="full: 自带 OpenCV,示例开箱可编译; slim: 依赖系统 libopencv-dev")
    ap.add_argument("--package", default=DEFAULT_PACKAGE)
    ap.add_argument("--revision", default=DEFAULT_REVISION,
                    help="默认取 src/rkllm/__init__.py DEFAULT_VERSION 的修订号")
    ap.add_argument("--maintainer", default=DEFAULT_MAINTAINER)
    ap.add_argument("--source", default=str(SOURCE_TARBALL))
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--emit-tree", action="store_true",
                    help="同时输出 staging 目录,便于 dpkg-deb --build 复核;"
                         "目录已存在时会报错而不是自行删除")
    ap.add_argument("--tree-dir", default=None,
                    help="staging 目录位置(默认 build/tree-<arch>)")
    args = ap.parse_args()

    if args.revision != DEFAULT_REVISION:
        raise SystemExit(
            "--revision %s != src/rkllm/__init__.py DEFAULT_VERSION revision %s; "
            "bump DEFAULT_VERSION instead of overriding the revision"
            % (args.revision, DEFAULT_REVISION))

    source_path = Path(args.source)
    if not source_path.exists():
        raise SystemExit(f"source tarball not found: {source_path}")

    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    ctx = BuildContext(arch=args.arch, package=args.package, upstream_version=UPSTREAM_VERSION,
                       revision=args.revision, maintainer=args.maintainer,
                       profile=args.profile, epoch=epoch)
    ctx.finalize()

    items, stats = stage(ctx, source_path)

    build_dir = ROOT / "build"
    deb = build_deb(items, ctx, Path(args.out), build_dir / "tmp")

    if args.emit_tree:
        tree_dir = Path(args.tree_dir) if args.tree_dir else build_dir / f"tree-{ctx.arch}"
        emit_tree(items, ctx, tree_dir)

    manifest = build_dir / f"{ctx.package}_{ctx.version}_{ctx.arch}.manifest.json"
    manifest.write_text(json.dumps([{k: v for k, v in vars(i).items() if k != "data"} for i in items],
                                   indent=2, ensure_ascii=False), encoding="utf-8")

    size = sum(len(i.data or b"") for i in items)
    print(f"已构建: {deb}")
    print(f"  架构={ctx.arch}  三元组={ctx.triplet}  profile={ctx.profile}")
    print(f"  文件数={len(items)}  解包体积={size/1048576:.1f} MB  deb 体积={deb.stat().st_size/1048576:.2f} MB")
    print(f"  清单: {manifest}")
    if stats["dropped"]:
        print("  裁剪统计:")
        for reason, count in sorted(stats["dropped"].items(), key=lambda kv: -kv[1]):
            print(f"    {count:>5d} 文件  {stats['dropped_bytes'][reason]/1048576:>8.1f} MB  {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
