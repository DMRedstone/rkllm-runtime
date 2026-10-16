"""Render a packaging template with the Debian substitution table.

Used while developing the CLI: it applies the same @TOKEN@ replacement that
make_deb.py performs, so the file can be executed locally.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_deb as m  # noqa: E402

ctx = m.BuildContext(arch="arm64", package=m.DEFAULT_PACKAGE,
                     upstream_version=m.UPSTREAM_VERSION, revision="1",
                     maintainer=m.DEFAULT_MAINTAINER, profile="full", epoch=0)
ctx.finalize()
source = Path(sys.argv[1])
target = Path(sys.argv[2])
target.write_text(ctx.apply(source.read_text(encoding="utf-8")),
                  encoding="utf-8", newline="\n")
print("rendered %s -> %s (%d bytes)" % (source, target, target.stat().st_size))
