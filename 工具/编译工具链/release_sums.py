#!/usr/bin/env python3
"""发布产物哈希（不签名）：生成/校验 `SHA256SUMS`。

为什么只做哈希：板端是 `dpkg -i` 手动安装、也没有 apt 仓库，签名要引入 OpenPGP
密钥管理（`gpg`、`reprepro`、`signed-by`），维护成本高于收益；而 `.deb` 的完整性
用哈希即可核对——前提是构建可复现，本项目 `dpkg-deb` 后端两次独立构建 sha256 一致，
所以"别人拿源码自建、对哈希"这条路是通的。

用法：
    python3 release_sums.py                        # 扫 dist/ 下 *.deb，写 SHA256SUMS
    python3 release_sums.py --dir /path/to/out     # 指定目录
    python3 release_sums.py --verify               # 校验（板端/下载后自查）
    python3 release_sums.py --verify SHA256SUMS    # 校验指定清单

输出格式与 `sha256sum` 一致（`<hex>  <filename>`），因此也能直接：
    sha256sum -c SHA256SUMS
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "工具" / "dist"
SUMS_NAME = "SHA256SUMS"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def collect(directory: Path) -> list[Path]:
    """目录下的发布产物（按文件名排序，保证清单稳定可复现）。"""
    if not directory.is_dir():
        raise SystemExit(f"目录不存在：{directory}")
    files = sorted(p for p in directory.glob("*.deb") if p.is_file())
    if not files:
        raise SystemExit(f"{directory} 下没有 .deb；先跑 make_deb.py")
    return files


def write_sums(directory: Path, out: Path | None = None) -> Path:
    files = collect(directory)
    lines = ["%s  %s" % (sha256_of(path), path.name) for path in files]
    target = out or directory / SUMS_NAME
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    print(f"\n已写入 {target}（{len(lines)} 个文件）")
    return target


def verify(sums: Path) -> int:
    if not sums.is_file():
        print(f"清单不存在：{sums}", file=sys.stderr)
        return 2
    base = sums.parent
    bad = 0
    total = 0
    for raw in sums.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            expected, name = line.split("  ", 1)
        except ValueError:
            print(f"[BAD] 行格式不对：{line}", file=sys.stderr)
            bad += 1
            continue
        total += 1
        path = base / name
        if not path.is_file():
            print(f"[MISS] {name}")
            bad += 1
            continue
        actual = sha256_of(path)
        if actual != expected:
            print(f"[FAIL] {name}\n       期望 {expected}\n       实际 {actual}")
            bad += 1
        else:
            print(f"[ OK ] {name}")
    print(f"\n{total} 个文件，失败 {bad}")
    return 0 if bad == 0 and total else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="生成或校验发布产物 SHA256SUMS（不做签名）")
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help="产物目录（默认 工具/dist）")
    ap.add_argument("--out", default=None, help="清单输出路径（默认 <dir>/SHA256SUMS）")
    ap.add_argument("--verify", nargs="?", const="", default=None,
                    metavar="SUMS",
                    help="校验模式：不带参数则校验 <dir>/SHA256SUMS")
    args = ap.parse_args(argv)

    if args.verify is not None:
        sums = Path(args.verify) if args.verify else Path(args.dir) / SUMS_NAME
        return verify(sums)

    write_sums(Path(args.dir), Path(args.out) if args.out else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
