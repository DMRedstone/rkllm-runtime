#!/usr/bin/env python3
"""End-to-end content check: compare what ended up inside the .deb against the
upstream release tarball (hashes) and print the installed metadata so the mapping
can be reviewed without opening the package on a board."""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 2026-10-27 audit P2-6：默认用法（不带参数）会走 `ROOT / "dist"` 自动发现
# deb，而 ROOT 从未定义 → NameError。ROOT 与 make_deb.py 同口径：工具目录
# （dist/、build/ 都在它下面）。
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from verify_deb import parse_ar, read_tar_xz  # noqa: E402
# 上游 release tarball 位置（软编码）：优先环境变量 RKLLM_RELEASE_TARBALL，未设置时使用仓库内随附的精简上游库。
SOURCE = Path(os.environ.get("RKLLM_RELEASE_TARBALL")
              or HERE.parent.parent / "输入" / "rknn-llm-release-v1.3.1-minimal.tar.gz")
PREFIX = "rknn-llm-release-v1.3.1"

# upstream path -> expected path inside the package; byte identical unless noted
IDENTICAL = [
    ("rkllm-runtime/Linux/librkllm_api/@ARCHDIR@/librkllmrt.so", "/usr/lib/@TRIPLET@/librkllmrt.so"),
    ("rkllm-runtime/Linux/librkllm_api/include/rkllm.h", "/usr/include/rkllm/rkllm.h"),
    ("examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/@ARCHDIR@/librknnrt.so",
     "/usr/lib/rkllm-runtime/librknnrt.so"),
    ("examples/multimodal_model_demo/deploy/src/main.cpp",
     "/usr/share/rkllm-runtime/examples/multimodal_model_demo/deploy/src/main.cpp"),
    ("examples/rkllm_api_demo/deploy/src/llm_demo.cpp",
     "/usr/share/rkllm-runtime/examples/rkllm_api_demo/deploy/src/llm_demo.cpp"),
    ("README.md", "/usr/share/doc/rkllm-runtime/README.md"),
    ("doc/Rockchip_RKLLM_SDK_CN_1.3.1.pdf", "/usr/share/doc/rkllm-runtime/Rockchip_RKLLM_SDK_CN_1.3.1.pdf"),
]

ARCHINFO = {
    "arm64": {"triplet": "aarch64-linux-gnu", "archdir": "aarch64"},
    "armhf": {"triplet": "arm-linux-gnueabihf", "archdir": "armhf"},
}


def source_hashes() -> dict[str, str]:
    digests: dict[str, str] = {}
    with tarfile.open(SOURCE, "r:gz") as tf:
        for member in tf:                      # stream instead of random access
            if not member.isfile():
                continue
            rel = member.name
            if rel.startswith(PREFIX + "/"):
                rel = rel[len(PREFIX) + 1:]
            f = tf.extractfile(member)
            if f is None:
                continue
            digests[rel] = hashlib.md5(f.read()).hexdigest()
    return digests


def deb_payload(deb: Path):
    import verify_deb as vd  # reuse the ar reader
    return vd.parse_ar(deb.read_bytes())


def main() -> int:
    if len(sys.argv) > 1:
        debs = [Path(p) for p in sys.argv[1:]]
    else:
        debs = sorted((ROOT / "dist").glob("*.deb"))
    if not debs:
        print("no .deb found under dist/")
        return 2

    upstream = source_hashes()
    rc = 0
    for deb in debs:
        arch = "arm64" if "arm64" in deb.name else "armhf"
        info = ARCHINFO[arch]
        print(f"\n===== {deb.name} =====")
        members = deb_payload(deb)
        with read_tar_xz(members["data.tar.xz"], "data") as tf:
            payload = {}
            for m in tf.getmembers():
                key = m.name[2:] if m.name.startswith("./") else m.name
                payload["/" + key] = m
            for up_tpl, pkg_tpl in IDENTICAL:
                up = up_tpl.replace("@ARCHDIR@", info["archdir"])
                pkg = pkg_tpl.replace("@TRIPLET@", info["triplet"])
                member = payload.get(pkg)
                if member is None:
                    print(f"  MISSING  {pkg}")
                    rc = 1
                    continue
                got = hashlib.md5(tf.extractfile(member).read()).hexdigest()
                want = upstream.get(up)
                status = "OK  " if got == want else "DIFF"
                if got != want:
                    rc = 1
                print(f"  {status}   {up}  ->  {pkg}")

            interesting = [
                f"/usr/lib/{info['triplet']}/librkllmrt.so",
                "/usr/lib/rkllm-runtime/librknnrt.so",
                "/usr/include/rkllm/rkllm.h",
                "/usr/include/rkllm.h",
                "/etc/ld.so.conf.d/rkllm-runtime.conf",
                f"/usr/lib/{info['triplet']}/pkgconfig/rkllm.pc",
                "/usr/bin/rkllm-fix-freq-rk3588",
                "/usr/share/doc/rkllm-runtime/copyright",
            ]
            print("  -- 权限与类型 --")
            for path in interesting:
                m = payload.get(path)
                if m is None:
                    print(f"     {path}: MISSING")
                    continue
                kind = {tarfile.REGTYPE: "file", tarfile.SYMTYPE: "symlink",
                        tarfile.DIRTYPE: "dir"}.get(m.type, "?")
                extra = f" -> {m.linkname}" if m.type == tarfile.SYMTYPE else f" {m.size:,} B"
                print(f"     {oct(m.mode & 0o777)} {kind:8s} {path}{extra}")

            pc = payload.get(f"/usr/lib/{info['triplet']}/pkgconfig/rkllm.pc")
            if pc is not None:
                print("  -- rkllm.pc --")
                text = tf.extractfile(pc).read().decode()
                for line in text.splitlines():
                    if line.startswith(("libdir", "includedir", "Libs", "Cflags")):
                        print("     " + line)
    return rc


if __name__ == "__main__":
    sys.exit(main())
