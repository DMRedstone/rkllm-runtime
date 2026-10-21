#!/usr/bin/env python3
"""Structural validator for the generated .deb (a lintian stand-in for hosts
without dpkg tooling). Re-reads the ar container and both tar members and
reports everything a reviewer would look at before installing on a board."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import lzma
import posixpath
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VALID_SECTIONS = {"libs", "devel", "utils", "misc", "doc", "python", "science"}
REQUIRED_FIELDS = ["Package", "Version", "Architecture", "Maintainer",
                   "Homepage", "Description"]
# 本包由本仓库维护，Homepage 必须指向它而非上游 airockchip/rknn-llm——
# 1.3.1-16 之前漏改，`apt show` 直接暴露给了运维。
EXPECTED_HOMEPAGE = "https://github.com/DMRedstone/rkllm-runtime"
KNOWN_MANDATORY_TREE = ["usr", "usr/share", "usr/share/doc", "usr/bin", "etc"]

# Declared in DEBIAN/conffiles: dpkg keeps local edits of this file across an
# upgrade, and only deletes it on purge - which is why postrm has to unlink it
# explicitly for `remove` as well.
SERVER_DEFAULTS = "/etc/default/rkllm-server"

results: list[tuple[str, str, str]] = []   # (level, check, message)


def record(level: str, check: str, message: str = "") -> None:
    results.append((level, check, message))


def fail(check: str, message: str) -> None:
    record("FAIL", check, message)


def ok(check: str, message: str = "") -> None:
    record("PASS", check, message)


def warn(check: str, message: str) -> None:
    record("WARN", check, message)


# ---------------------------------------------------------------- ar reading
def parse_ar(blob: bytes) -> dict[str, bytes]:
    if not blob.startswith(b"!<arch>\n"):
        fail("ar magic", "missing !<arch> header")
        return {}
    members: dict[str, bytes] = {}
    pos = 8
    while pos + 60 <= len(blob):
        header = blob[pos:pos + 60]
        if header[58:60] != b"`\n":
            fail("ar header", f"bad magic at offset {pos}")
            break
        name = header[0:16].decode("ascii", "replace").strip()
        try:
            size = int(header[48:58].decode("ascii").strip())
        except ValueError:
            fail("ar header", f"unparsable size for {name!r}")
            break
        payload = blob[pos + 60:pos + 60 + size]
        if len(payload) != size:
            fail("ar payload", f"{name}: truncated member")
        name = name.rstrip("/")
        members[name] = payload
        pos += 60 + size + (size % 2)
    trailing = blob[pos:]
    if trailing.strip(b"\n"):
        warn("ar trailing", f"{len(trailing)} bytes of unexpected trailing data")
    else:
        ok("ar trailing", "clean")
    return members


def read_tar_xz(payload: bytes, label: str):
    return tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz")


# ---------------------------------------------------------------- checks
def check_container(deb: Path) -> None:
    blob = deb.read_bytes()
    members = parse_ar(blob)
    for required, note in (("debian-binary", ""), ("control.tar.xz", ""), ("data.tar.xz", "")):
        if required in members:
            ok(f"ar member {required}", f"{len(members[required]):,} bytes")
        else:
            fail(f"ar member {required}", "absent")
    order = [n for n in members]
    if order[:3] != ["debian-binary", "control.tar.xz", "data.tar.xz"]:
        warn("ar member order", f"got {order}, canonical order is debian-binary/control/data")
    else:
        ok("ar member order", "debian-binary -> control -> data")
    binary = members.get("debian-binary")
    if binary == b"2.0\n":
        ok("debian-binary", "2.0")
    elif binary is not None:
        fail("debian-binary", f"expected b'2.0\\n', got {binary!r}")
    return members


def check_control(members: dict[str, bytes]) -> dict:
    if "control.tar.xz" not in members:
        fail("control.tar.xz", "missing")
        return {}
    with read_tar_xz(members["control.tar.xz"], "control") as tf:
        names = [m.name for m in tf.getmembers()]
        control_files = {}
        for m in tf.getmembers():
            control_files[m.name] = m
            if m.uid != 0 or m.gid != 0 or not (m.uname, m.gname) in (("root", "root"), ("", "")):
                fail("control ownership", f"{m.name}: {m.uid}:{m.gid} {m.uname}/{m.gname}")
        for expected in ("./conffiles", "./control", "./md5sums", "./postinst",
                         "./prerm", "./postrm"):
            if expected in names:
                ok(f"control member {expected}", "")
            else:
                fail(f"control member {expected}", "missing")
        for script in ("./postinst", "./prerm", "./postrm"):
            member = control_files.get(script)
            if member is None:
                continue
            if member.mode & 0o777 != 0o755:
                fail(f"{script} mode", f"{oct(member.mode & 0o777)}, must be 0755")
            else:
                ok(f"{script} mode", "0755")
            body = tf.extractfile(member).read().decode("utf-8", "replace")
            if not body.startswith("#!/"):
                fail(f"{script} shebang", "missing shebang")
            elif body.splitlines()[0] not in ("#!/bin/sh", "#!/bin/bash"):
                fail(f"{script} shebang", body.splitlines()[0])
            else:
                ok(f"{script} shebang", body.splitlines()[0])
            if "set -e" not in body:
                warn(f"{script} strictness", "no `set -e`")
            else:
                ok(f"{script} strictness", "set -e present")
            # scan executable statements only - comments may legitimately mention these
            code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
            lowered = code.lower()
            # `rm -f <the declared conffile>` is the one deliberate exception and
            # is checked separately below ("postrm deletes the conffile"); a
            # recursive rm or a write into /etc is never acceptable.
            for risky in ("rm -rf", "rm -fr", "mkdir -p /usr", "> /etc/", "useradd",
                          "apt-get", "pip install", "curl ", "wget "):
                if risky in lowered:
                    fail(f"{script} residue risk", f"contains {risky!r}")
            # systemctl is acceptable only for refreshing caches and for bouncing
            # a service the operator already started. Enabling or starting a unit
            # would create state dpkg cannot remove later, so those stay banned.
            allowed_verbs = ("daemon-reload", "daemon-reexec", "is-active", "restart")
            seen_verbs = []
            for match in re.finditer(
                    r"systemctl\s+(?:--no-reload\s+|--quiet\s+|--no-pager\s+)?([a-z-]+)", code):
                verb = match.group(1)
                seen_verbs.append(verb)
                if verb not in allowed_verbs:
                    fail(f"{script} residue risk", f"systemctl {verb} is not allowed")
            if "restart" in seen_verbs and "is-active" not in seen_verbs:
                fail(f"{script} residue risk",
                     "systemctl restart without an is-active guard could start a service "
                     "the operator never started")
            # deb-systemd-helper is the standard way to *undo* an operator's
            # `systemctl enable`, which is exactly what we want on removal.
            ok(f"{script} residue risk", "no filesystem mutation outside dpkg")
            if script == "./postrm":
                # dpkg keeps conffiles when a package is merely removed and only
                # deletes them on purge. This package promises that removing it
                # leaves nothing behind, so the file has to go on `remove` too.
                if re.search(r"^\s*rm\s+-f\s+" + re.escape(SERVER_DEFAULTS) + r"\s*$",
                             code, re.M):
                    ok("postrm deletes the conffile",
                       f"{SERVER_DEFAULTS} is gone after remove, not only after purge")
                else:
                    fail("postrm deletes the conffile",
                         f"dpkg keeps conffiles on a plain remove and nothing deletes "
                         f"{SERVER_DEFAULTS}")

        control_member = control_files.get("./control")
        fields: dict[str, str] = {}
        if control_member is None:
            return {}
        raw = tf.extractfile(control_member).read()
        if b"\r\n" in raw:
            fail("control line endings", "CRLF detected")
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            fail("control termination", "must end with exactly one trailing newline")
        else:
            ok("control termination", "single trailing newline")
        text = raw.decode("utf-8", "replace")
        current = None
        for line in text.splitlines():
            if line.startswith((" ", "\t")):
                if current is None:
                    fail("control syntax", "continuation line before any field")
                fields[current] += "\n" + line
                continue
            if not line.strip():
                fail("control syntax", "blank line inside control file")
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            if key in fields:
                fail("control duplicate field", key)
            fields[key] = value.strip()
            current = key
        for field_name in REQUIRED_FIELDS:
            if field_name in fields and fields[field_name]:
                ok(f"control field {field_name}", fields[field_name].splitlines()[0][:70])
            else:
                fail(f"control field {field_name}", "missing or empty")
        if fields.get("Section") not in VALID_SECTIONS:
            warn("control field Section", f"{fields.get('Section')!r} is not a common section")
        if fields.get("Priority") not in {"required", "important", "standard", "optional", "extra"}:
            warn("control field Priority", f"{fields.get('Priority')!r}")
        version = fields.get("Version", "")
        if not re.fullmatch(r"\d[\w.+:~-]*", version):
            fail("control Version format", version)
        elif "-" not in version:
            warn("control Version", "no Debian revision suffix")
        else:
            ok("control Version", version)
        if fields.get("Architecture") not in {"all", "any", "amd64", "arm64", "armhf", "i386"}:
            fail("control Architecture", fields.get("Architecture", ""))
        else:
            ok("control Architecture", fields["Architecture"])
        description = fields.get("Description", "")
        body_lines = description.splitlines()
        if not body_lines or not body_lines[0].strip():
            fail("control Description", "empty synopsis")
        else:
            ok("control Description synopsis", body_lines[0])
        for line in body_lines[1:]:
            if line == " .":
                continue
            if not line.startswith(" "):
                fail("control Description", f"unindented continuation line {line[:40]!r}")
        depends = fields.get("Depends", "")
        for dep in [d.strip() for d in depends.split(",") if d.strip()]:
            # package name, optional :arch/:any qualifier, optional version clause
            if not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*(:[a-z0-9]+)?( \(.*\))?", dep):
                fail("control Depends", f"malformed dependency {dep!r}")
        ok("control Depends", depends)
        homepage = fields.get("Homepage", "").strip()
        if homepage != EXPECTED_HOMEPAGE:
            fail("control Homepage",
                 f"{homepage!r} != {EXPECTED_HOMEPAGE!r} (must point at this repo)")
        else:
            ok("control Homepage", homepage)
        suggests = fields.get("Suggests", "")
        stale = [s.strip() for s in suggests.split(",")
                 if s.strip().startswith("python3-flask")]
        if stale:
            fail("control Suggests",
                 "python3-flask is a legacy-stack leftover; the service uses FastAPI")
        else:
            ok("control Suggests", suggests or "(none)")
        md5_member = control_files.get("./md5sums")
        md5_map = {}
        if md5_member is not None:
            for line in tf.extractfile(md5_member).read().decode().splitlines():
                if not line.strip():
                    continue
                digest, _, path = line.partition("  ")
                md5_map[path] = digest
            ok("md5sums entries", f"{len(md5_map)} files")

        # ---- conffiles: the file dpkg must not overwrite on upgrade ---------
        conffiles: list[str] = []
        conffiles_member = control_files.get("./conffiles")
        if conffiles_member is None:
            fail("conffiles declared",
                 "no DEBIAN/conffiles: dpkg would replace every /etc file on upgrade")
        else:
            if conffiles_member.mode & 0o777 not in (0o644, 0o755):
                fail("conffiles mode", oct(conffiles_member.mode & 0o777))
            raw_conf = tf.extractfile(conffiles_member).read()
            if b"\r\n" in raw_conf:
                fail("conffiles line endings", "CRLF detected")
            raw_text = raw_conf.decode("utf-8", "replace")
            if not raw_text.endswith("\n") or raw_text.endswith("\n\n"):
                fail("conffiles termination", "must end with exactly one trailing newline")
            conffiles = [line.strip() for line in raw_text.splitlines() if line.strip()]
            bad = [p for p in conffiles if not p.startswith("/") or ".." in p.split("/")]
            if bad:
                fail("conffiles entry", f"not an absolute in-package path: {bad[:3]}")
            if len(conffiles) != len(set(conffiles)):
                fail("conffiles duplicate", str(conffiles))
            elif conffiles != sorted(conffiles):
                warn("conffiles order", "entries are not sorted")
            (fail if SERVER_DEFAULTS not in conffiles else ok)(
                "conffiles declare the service config",
                f"{SERVER_DEFAULTS} is a conffile" if SERVER_DEFAULTS in conffiles
                else f"{SERVER_DEFAULTS} missing from {conffiles}")
        return {"fields": fields, "md5sums": md5_map, "names": names,
                "conffiles": conffiles}


def check_data(members: dict[str, bytes], control_info: dict) -> None:
    if "data.tar.xz" not in members:
        fail("data.tar.xz", "missing")
        return
    md5_map = control_info.get("md5sums", {})
    with read_tar_xz(members["data.tar.xz"], "data") as tf:
        regulars = {}
        dirs = set()
        symlinks = {}
        modes = set()
        problems = []
        for m in tf.getmembers():
            name = m.name
            if not name.startswith("./"):
                problems.append((name, "path does not start with ./"))
                continue
            rel = name[2:]
            if name.startswith("./.") or ".." in rel.split("/"):
                problems.append((name, "unexpected dot segment"))
            if m.uid != 0 or m.gid != 0:
                problems.append((name, f"not root-owned ({m.uid}:{m.gid})"))
            if m.type == tarfile.DIRTYPE:
                dirs.add(rel)
            elif m.type == tarfile.SYMTYPE:
                symlinks[rel] = m.linkname
                if m.linkname.startswith("/") or ".." in m.linkname.split("/"):
                    problems.append((name, f"symlink target escapes the package: {m.linkname}"))
            elif m.type == tarfile.REGTYPE:
                regulars[rel] = m
                modes.add(m.mode & 0o777)
                if m.mode & 0o7000:
                    problems.append((name, f"setuid/setgid/sticky bit set: {oct(m.mode)}"))
                if m.mode & 0o022:
                    problems.append((name, f"group/other writable: {oct(m.mode)}"))
                if not (m.mode & 0o444) == 0o444:
                    problems.append((name, f"not world readable: {oct(m.mode)}"))
                top = rel.split("/")[0]
                if top not in {"usr", "etc"}:
                    problems.append((name, f"file outside /usr and /etc: /{top}"))
                computed = hashlib.md5(tf.extractfile(m).read()).hexdigest()
                expected = md5_map.get(rel)
                if expected is None:
                    problems.append((name, "not listed in md5sums"))
                elif expected != computed:
                    problems.append((name, "md5 mismatch"))
            else:
                problems.append((name, f"unexpected tar member type {m.type!r}"))

        for name, issue in problems:
            fail("data payload", f"{name}: {issue}")
        if not problems:
            ok("data payload integrity", f"{len(regulars)} files, all owned by root, listed and md5-verified")

        missing_dirs = []
        for rel in list(regulars) + list(symlinks):
            parts = rel.split("/")
            for i in range(1, len(parts)):
                parent = "/".join(parts[:i])
                if parent not in dirs:
                    missing_dirs.append(parent)
        missing_dirs = sorted(set(missing_dirs))
        if missing_dirs:
            fail("data directories", f"{len(missing_dirs)} parent directories absent from the tar")
        else:
            ok("data directories", f"{len(dirs)} directory entries cover every file")

        extra = set(md5_map) - set(regulars)
        if extra:
            fail("md5sums", f"{len(extra)} entries without a matching file: {sorted(extra)[:3]}")
        else:
            ok("md5sums", "matches the file set exactly")

        by_top: dict[str, int] = {}
        for rel, m in regulars.items():
            key = "/".join(rel.split("/")[:3]) if rel.startswith("usr/share") else "/".join(rel.split("/")[:2])
            by_top[key] = by_top.get(key, 0) + m.size
        record("INFO", "installed footprint", "")
        for key, size in sorted(by_top.items(), key=lambda kv: -kv[1])[:12]:
            record("INFO", f"  /{key}", f"{size/1048576:8.2f} MB")

        # ---- package specific expectations ---------------------------------
        expected = [
            f"usr/lib/{ARCH_TRIPLET['arm64'] if 'arm64' in control_info['fields'].get('Architecture','') else ''}/librkllmrt.so",
        ]
        arch = control_info["fields"].get("Architecture", "")
        triplet = "aarch64-linux-gnu" if arch == "arm64" else "arm-linux-gnueabihf"
        for path, why in (
            (f"usr/lib/{triplet}/librkllmrt.so", "runtime library in the multiarch path"),
            ("usr/include/rkllm/rkllm.h", "public header"),
            (f"usr/lib/{triplet}/pkgconfig/rkllm.pc", "pkg-config metadata"),
            (f"usr/lib/{triplet}/cmake/rkllm/rkllm-config.cmake", "CMake package"),
            ("usr/share/doc/" + control_info["fields"].get("Package", "") + "/copyright", "Debian copyright"),
            ("usr/share/doc/" + control_info["fields"].get("Package", "") + "/changelog.Debian.gz", "Debian changelog"),
        ):
            if path in regulars:
                ok("expected file", path)
            else:
                fail("expected file", f"{path} missing ({why})")

        # --- /usr/bin/rkllm: the user facing front end -----------------------
        cli_rel = "usr/bin/rkllm"
        if cli_rel not in regulars:
            fail("cli installed", "/usr/bin/rkllm missing")
        else:
            member = regulars[cli_rel]
            body = tf.extractfile(member).read().decode("utf-8", "replace")
            if member.mode & 0o777 != 0o755:
                fail("cli mode", f"{oct(member.mode & 0o777)} != 0755")
            else:
                ok("cli mode", "0755")
            first = body.splitlines()[0] if body.splitlines() else ""
            if first != "#!/usr/bin/python3":
                fail("cli shebang", repr(first) or "missing")
            else:
                ok("cli shebang", first)
            # Unsubstituted template tokens would reach the board verbatim.
            leftovers = sorted(set(re.findall(r"@[A-Z][A-Z0-9_]*@", body)))
            if leftovers:
                fail("cli rendered", f"unsubstituted placeholders: {leftovers[:5]}")
            else:
                ok("cli rendered", "no @TOKEN@ left in the script")
            try:
                ast.parse(body)
                ok("cli syntax", "parses as Python 3")
            except SyntaxError as exc:
                fail("cli syntax", f"line {exc.lineno}: {exc.msg}")
            # standard library only: the package must not need pip on the board
            third_party = [mod for mod in ("flask", "gradio", "requests", "numpy", "torch")
                           if re.search(r"^\s*(?:import|from)\s+%s\b" % mod, body, re.M)]
            if third_party:
                fail("cli dependencies", f"imports third-party modules: {third_party}")
            else:
                ok("cli dependencies", "standard library only, nothing to pip install")
            # declared dependency for the interpreter that will run it
            if "python3" not in control_info["fields"].get("Depends", ""):
                fail("cli interpreter dependency", "Depends lacks python3")
            else:
                ok("cli interpreter dependency", "python3 declared in Depends")
            # helpers it calls by absolute path must really be in the package
            called = sorted(set(re.findall(r"/usr/bin/rkllm-[a-z0-9-]+", body)))
            missing = [p for p in called if p.lstrip("/") not in regulars]
            if missing:
                fail("cli helper paths", f"references helpers that are not shipped: {missing}")
            else:
                ok("cli helper paths", f"{len(called)} helper path(s) all shipped")

        # --- systemd unit ----------------------------------------------------
        unit_rel = "usr/lib/systemd/system/rkllm-server.service"
        if unit_rel not in regulars:
            fail("systemd unit", "rkllm-server.service missing")
        else:
            unit = tf.extractfile(regulars[unit_rel]).read().decode("utf-8", "replace")
            ok("systemd unit", "shipped under /usr/lib/systemd/system")
            for label, condition in (
                    ("ExecStart points at rkllm", "/usr/bin/rkllm serve" in unit),
                    ("EnvironmentFile", SERVER_DEFAULTS in unit),
                    ("[Install] section", "[Install]" in unit),
                    ("restart is rate limited", "StartLimitBurst=" in unit),
                    # Guarding on /dev/rknpu looks sensible but is wrong: 0.9.x
                    # drivers register the NPU as a DRM render node and never
                    # create that path, so the unit gets skipped on working
                    # boards. Whether the NPU works is decided by rkllm_init().
                    ("no bogus NPU path guard",
                     "ConditionPathExists=/dev/rknpu" not in unit)):
                (ok if condition else fail)(f"unit {label}", "present" if condition else "missing")
            # the file the unit reads must be the file declared in conffiles,
            # otherwise upgrades silently replace the operator's settings
            env_file = re.search(r"^EnvironmentFile=-?(\S+)", unit, re.M)
            declared_conffiles = control_info.get("conffiles", [])
            if env_file is None:
                fail("unit EnvironmentFile target", "the unit reads no configuration file")
            elif env_file.group(1) != SERVER_DEFAULTS:
                fail("unit EnvironmentFile target",
                     f"reads {env_file.group(1)}, but the conffile is {SERVER_DEFAULTS}")
            elif SERVER_DEFAULTS not in declared_conffiles:
                fail("unit EnvironmentFile target",
                     f"{SERVER_DEFAULTS} is not declared in DEBIAN/conffiles")
            else:
                ok("unit EnvironmentFile target", f"{SERVER_DEFAULTS}, declared as a conffile")
            # Direct ${VAR} expansion in ExecStart is a trap: systemd turns an
            # unset variable into an EMPTY ARGUMENT, which argparse rejects as an
            # unrecognised positional. The unit then dies with exit code 2 and
            # `systemctl status` shows nothing but "status=2".
            exec_lines = [line for line in unit.splitlines() if line.startswith("ExecStart=")]
            wrapped = bool(exec_lines) and "sh -c" in exec_lines[0]
            (ok if wrapped else fail)(
                "unit ExecStart survives unset vars",
                "wrapped in /bin/sh, where an empty variable disappears"
                if wrapped else
                "bare ${VAR} expansion would pass empty arguments to rkllm (exit code 2)")
            # A literal $$ is required: systemd expands $VAR itself, so a single
            # $ would leave the shell with an empty argument list.
            if "$$RKLLM_MODEL" not in unit:
                fail("unit escaping", "ExecStartPre must escape $$ for the shell")
            else:
                ok("unit escaping", "$$ stops systemd from pre-expanding the variables")
            # shipped but never enabled: enabling is the operator's decision
            if "enable" in unit.lower():
                fail("unit not self-enabling", "unit text mentions enable")
            else:
                ok("unit not self-enabling", "installed but not enabled or started")

        # --- service configuration: a conffile, documented in both languages --
        defaults_rel = SERVER_DEFAULTS.lstrip("/")
        if defaults_rel not in regulars:
            fail("conffile shipped", f"{SERVER_DEFAULTS} missing from the payload")
        else:
            body = tf.extractfile(regulars[defaults_rel]).read().decode("utf-8", "replace")
            issues = []
            if "RKLLM_MODEL=" not in body:
                issues.append("RKLLM_MODEL is not settable")
            # A conffile whose text depends on the package version gets rewritten
            # by every upgrade, which discards the operator's edits and any
            # `chmod 600` applied because an API key lives in it.
            version = control_info["fields"].get("Version", "")
            if version and version in body:
                issues.append(f"embeds the package version {version!r}: every revision "
                              "bump would rewrite the file")
            leftovers = sorted(set(re.findall(r"@[A-Z][A-Z0-9_]*@", body)))
            if leftovers:
                issues.append(f"unsubstituted placeholders: {leftovers[:3]}")
            # systemd reads this file (EnvironmentFile=), not a shell
            for line in body.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", stripped):
                    issues.append(f"line systemd cannot parse: {stripped[:40]!r}")
                elif "$(" in stripped or "`" in stripped:
                    issues.append(f"shell syntax in a value: {stripped[:40]!r}")
            # every parameter the service accepts has to be documented here, with
            # a commented example - that is the whole point of the file
            unit_text = tf.extractfile(regulars[unit_rel]).read().decode("utf-8", "replace") \
                if unit_rel in regulars else ""
            cli_text = tf.extractfile(regulars[cli_rel]).read().decode("utf-8", "replace") \
                if cli_rel in regulars else ""
            wanted = set(re.findall(r"\$\$?(RKLLM_[A-Z0-9_]+)", unit_text))
            wanted |= set(re.findall(r'os\.environ\.get\("(RKLLM_[A-Z0-9_]+)"', cli_text))
            wanted |= {"RKLLM_" + name for name in
                       re.findall(r'option_(?:default|flag)\("([A-Z0-9_]+)"', cli_text)}
            wanted.add("RKLLM_EXTRA_ARGS")
            missing = sorted(v for v in wanted if v not in body)
            if missing:
                issues.append(f"undocumented parameter(s): {missing}")
            uncovered = sorted(v for v in wanted if f"#{v}=" not in body)
            if uncovered:
                issues.append(f"no commented example (optional form) for: {uncovered}")
            cjk = [line for line in body.splitlines() if re.search(r"[\u4e00-\u9fff]", line)]
            if len(cjk) < 10 or not re.search(r"English|\bEN\b", body):
                issues.append(f"not bilingual: {len(cjk)} Chinese line(s)")
            if issues:
                fail("conffile content", "; ".join(issues[:4]))
            else:
                ok("conffile content", "%d lines, bilingual, version-independent, "
                                       "documents all %d parameters"
                   % (len(body.splitlines()), len(wanted)))

        # --- manual pages ----------------------------------------------------
        for rel, label, need_cjk in (
                ("usr/share/man/man1/rkllm.1.gz", "man page: English", False),
                ("usr/share/man/zh_CN/man1/rkllm.1.gz", "man page: zh_CN", True)):
            if rel not in regulars:
                fail(label, "missing")
                continue
            try:
                page = gzip.decompress(tf.extractfile(regulars[rel]).read()).decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                fail(label, f"not a readable gzip/UTF-8 man page: {exc}")
                continue
            issues = []
            if not re.search(r"^\.TH\s+RKLLM\s+1\b", page, re.M):
                issues.append("missing `.TH RKLLM 1` header")
            leftovers = sorted(set(re.findall(r"@[A-Z][A-Z0-9_]*@", page)))
            if leftovers:
                issues.append(f"unsubstituted placeholders: {leftovers[:3]}")
            cjk = re.search(r"[\u4e00-\u9fff]", page)
            if need_cjk and not cjk:
                issues.append("no Chinese text - is this a copy of the English page?")
            if not need_cjk and cjk:
                issues.append("English page contains CJK text")
            # a man page must not carry raw source markup that man would print
            if "<html" in page.lower() or "&nbsp;" in page:
                issues.append("contains markup that would be shown verbatim")
            if issues:
                fail(label, "; ".join(issues))
            else:
                ok(label, "%d bytes, %d sections, %d options documented"
                   % (len(page), page.count("\n.SH "), page.count("\n.BI ")))

        # examples must not reference the upstream SDK relative layout
        offenders = []
        for rel in regulars:
            if "/share/" in rel and rel.endswith((".txt", ".sh", ".py", ".cmake")):
                body = tf.extractfile(regulars[rel]).read().decode("utf-8", "replace")
                # ignore comments: they legitimately describe the upstream layout
                code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
                if "../../../rkllm-runtime" in code or "~/opts/gcc-arm" in code:
                    offenders.append(rel)
        if offenders:
            fail("examples relocated", f"{len(offenders)} example files still point at the SDK tree: {offenders[:3]}")
        else:
            ok("examples relocated", "no upstream relative path left in build scripts")

        kept_proxies = [r for r in regulars if "/rknn_server/" in r]
        if kept_proxies:
            fail("examples relocated", f"RKNN proxy server binaries still shipped: {kept_proxies[:2]}")
        else:
            ok("examples relocated", "RKNN proxy server (not an RKLLM component) excluded")

        # install() rules must stay inside the build tree, otherwise `make install`
        # writes files dpkg does not own and cannot remove -> violates the
        # "no residue after removal" requirement.
        escaped = []
        for rel in regulars:
            if rel.endswith("CMakeLists.txt"):
                body = tf.extractfile(regulars[rel]).read().decode("utf-8", "replace")
                code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
                for m in re.finditer(r"install\s*\(([^)]*)\)", code, re.S):
                    dest = re.search(r"DESTINATION\s+(\S+)", m.group(1))
                    if dest and dest.group(1).startswith(("/", "${CMAKE_INSTALL_PREFIX}", "$ENV{")):
                        escaped.append(f"{rel}: DESTINATION {dest.group(1)}")
        if escaped:
            fail("install() sandboxed", f"{len(escaped)} absolute DESTINATION(s) could escape dpkg: {escaped[:3]}")
        else:
            ok("install() sandboxed", "every install(DESTINATION) is relative to the build-tree prefix")

        # no compiled junk / VCS data
        junk = [r for r in regulars if re.search(r"(\.o|\.a|\.gitignore)$", r) and "/share/" not in r and "opencv" not in r]
        if junk:
            warn("data payload content", f"{len(junk)} suspicious objects outside examples: {junk[:3]}")
        else:
            ok("data payload content", "no stray build artefacts outside examples")

        mtimes = {m.mtime for m in tf.getmembers()}
        if len(mtimes) == 1:
            ok("reproducible timestamps", f"all members share mtime {mtimes.pop()}")
        else:
            warn("reproducible timestamps", f"{len(mtimes)} distinct mtimes")


def simulate_removal(members: dict[str, bytes]) -> None:
    """Everything dpkg deletes must be either a file or a directory this package created."""
    with read_tar_xz(members["data.tar.xz"], "data") as tf:
        owned_dirs = {m.name[2:].rstrip("/") for m in tf.getmembers() if m.type == tarfile.DIRTYPE}
        owned_files = {m.name[2:] for m in tf.getmembers() if m.type != tarfile.DIRTYPE}
    system_dirs = {"usr", "usr/bin", "usr/share", "usr/share/doc", "usr/include", "usr/lib",
                   "etc", "etc/default", "usr/lib/" + ("aarch64-linux-gnu"),
                   "usr/share/man", "etc/ld.so.conf.d"}
    leftover = sorted(d for d in owned_dirs if d in system_dirs)
    record("INFO", "removal: shared system dirs touched", ", ".join("/" + d for d in leftover) or "none")
    private = sorted("/" + d for d in owned_dirs if d not in system_dirs)
    if private:
        record("INFO", "removal: package-private dirs removed with the package",
               ", ".join(private))
    record("INFO", "removal: files tracked by dpkg", f"{len(owned_files)} entries, " +
           "nothing is ever written outside this list (postinst/postrm only run ldconfig "
           "and unlink " + SERVER_DEFAULTS + ", the one conffile, on remove as well as purge)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("deb", nargs="?", default=None,
                    help="package to check; defaults to the single .deb in dist/")
    args = ap.parse_args()
    if args.deb:
        deb = Path(args.deb)
    else:
        # Discover it rather than hardcoding a version, otherwise every revision
        # bump silently breaks the default invocation.
        candidates = sorted((ROOT / "dist").glob("*.deb"))
        if not candidates:
            print("no .deb in dist/ - build one first")
            return 2
        if len(candidates) > 1:
            print("several packages in dist/, pick one explicitly:")
            for candidate in candidates:
                print("  %s" % candidate)
            return 2
        deb = candidates[0]
    if not deb.exists():
        print(f"deb not found: {deb}")
        return 2
    members = check_container(deb)
    control_info = check_control(members)
    if control_info:
        check_data(members, control_info)
        simulate_removal(members)

    width = max(len(c) for _, c, _ in results)
    failures = warnings = 0
    for level, check, message in results:
        if level == "FAIL":
            failures += 1
        if level == "WARN":
            warnings += 1
        line = f"[{level:4s}] {check:<{width}}  {message}" if message else f"[{level:4s}] {check}"
        print(line)
    print("-" * 72)
    print(f"deb      : {deb}  ({deb.stat().st_size/1048576:.2f} MB)")
    print(f"checks   : {len(results)}   FAIL={failures}   WARN={warnings}")
    return 1 if failures else 0


ARCH_TRIPLET = {"arm64": "aarch64-linux-gnu", "armhf": "arm-linux-gnueabihf"}

if __name__ == "__main__":
    sys.exit(main())
