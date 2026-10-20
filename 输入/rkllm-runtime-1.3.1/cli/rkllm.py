#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
rkllm - command line front end for the RKLLM runtime shipped by the
        rkllm-runtime Debian package.

    rkllm chat <model>            interactive local chat (REPL)
    rkllm run  <model> "prompt"   one-shot inference, prints the answer
    rkllm serve <model>           OpenAI-compatible HTTP API server
    rkllm list                    list discoverable .rkllm models
    rkllm info                    runtime / NPU driver / environment report
    rkllm selftest                end-to-end self test, needs no model

<model> may be a bare name ("qwen2.5-1.5b", resolved against the model search
directories) or any path to a .rkllm file.

Only the Python 3 standard library is used, so nothing has to be pip-installed
on the board.  All user-facing output is plain ASCII/English on purpose: board
images frequently run with the C locale.
"""

from __future__ import annotations

import argparse
import ctypes
import hmac
import json
import os
import queue
import select
import socket
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

PROGRAM = "rkllm"
VERSION = "@VERSION@"
UPSTREAM_VERSION = "@UPSTREAM_VERSION@"
PACKAGE = "@PACKAGE@"
ARCH = "@ARCH@"
TRIPLET = "@TRIPLET@"
LIBDIR = "@LIBDIR_ABS@"

MODEL_SUFFIX = ".rkllm"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_CONTEXT = 4096
DEFAULT_MAX_NEW_TOKENS = 2048

# A client that stops sending mid-body would otherwise pin a worker thread for
# as long as it likes; request_queue_size lets 256 of them queue up.
READ_TIMEOUT_S = 30
# Wildcard by default (LAN tool), but a deployment that serves a browser UI from
# one known origin should pin it: "*" lets any page send an Authorization header
# cross-origin.
ALLOW_ORIGIN = os.environ.get("RKLLM_CORS_ORIGIN", "").strip() or "*"
# How long to wait for the inference thread after abort() before deciding the
# runtime is wedged. Blocking forever takes a request thread with it.
STREAM_JOIN_TIMEOUT_S = 60
DEFAULT_MAX_QUEUE = 32
DEFAULT_QUEUE_TIMEOUT = 600.0
MAX_BODY_BYTES = 8 * 1024 * 1024

# Frequency-lock helpers installed by this package, per detected SoC.
FREQ_HELPERS = {
    "rk3588": "/usr/bin/rkllm-fix-freq-rk3588",
    "rk3576": "/usr/bin/rkllm-fix-freq-rk3576",
    "rk3562": "/usr/bin/rkllm-fix-freq-rk3562",
    "rv1126b": "/usr/bin/rkllm-fix-freq-rv1126b",
}
PLATFORMS = ("rk3588", "rk3576", "rk3562", "rv1126b")

# Boards with four big cores in the 4..7 mask (the upstream demos use the same
# split); everything else falls back to cores 0..3.
BIG_CORE_PLATFORMS = ("rk3588", "rk3576")

NPU_DEBUGFS = "/sys/kernel/debug/rknpu"
LEGACY_NPU_DEVICE = "/dev/rknpu"
DRM_CLASS = "/sys/class/drm"

# sysfs knobs the board tuning helpers write. These are candidates: a path that
# does not exist on this SoC or kernel is skipped, which is what lets one table
# serve RK3588/RK3576/RK3562/RV1126B.
FREQ_GOVERNOR_PATHS = (
    "/sys/class/devfreq/fdab0000.npu/governor",
    "/sys/class/devfreq/fb000000.gpu/governor",
    "/sys/class/devfreq/dmc/governor",
    "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor",
    "/sys/devices/system/cpu/cpufreq/policy4/scaling_governor",
    "/sys/devices/system/cpu/cpufreq/policy6/scaling_governor",
)
FREQ_CPUIDLE_PATHS = tuple(
    "/sys/devices/system/cpu/cpu%d/cpuidle/state1/disable" % cpu for cpu in range(8))

# Snapshot taken right before a helper runs, replayed by restore_frequencies().
_FREQ_SNAPSHOT = []


def npu_status():
    """Report whether the RKNPU driver is reachable, and how.

    Returns (available, description). It deliberately checks several signals:
    drivers from 0.9.x onwards register the NPU as a DRM render node and do NOT
    create /dev/rknpu at all, so looking for that path reports "no NPU" on a
    perfectly working board.
    """
    try:
        with open(os.path.join(NPU_DEBUGFS, "version")) as handle:
            text = handle.read().strip()
        if text:
            return True, text.replace("\n", " ")
    except OSError:
        pass

    if os.path.isdir(NPU_DEBUGFS):
        return True, "present at %s (version not readable)" % NPU_DEBUGFS

    try:
        entries = sorted(os.listdir(DRM_CLASS))
    except OSError:
        entries = []
    for entry in entries:
        if not entry.startswith("renderD"):
            continue
        target = os.path.realpath(os.path.join(DRM_CLASS, entry, "device", "driver"))
        if "rknpu" in target.lower():
            return True, "bound to /dev/dri/%s" % entry

    if os.path.exists(LEGACY_NPU_DEVICE):
        return True, "present at %s (legacy device node)" % LEGACY_NPU_DEVICE
    if os.path.isdir("/sys/module/rknpu"):
        return True, "kernel module loaded, no device node yet"

    return False, ("no NPU driver found: neither %s/version, nor a DRM render node "
                   "bound to rknpu, nor %s" % (NPU_DEBUGFS, LEGACY_NPU_DEVICE))
DEVICE_TREE_COMPATIBLE = "/proc/device-tree/compatible"


class RKLLMError(Exception):
    """Anything the user is supposed to read instead of a traceback."""


# ---------------------------------------------------------------------------
# environment probing
# ---------------------------------------------------------------------------

def detect_platform():
    """Best-effort SoC detection from the device tree. Returns None if unknown."""
    blob = ""
    try:
        with open(DEVICE_TREE_COMPATIBLE, "rb") as handle:
            blob = handle.read().decode("latin-1").lower()
    except OSError:
        blob = ""
    if not blob:
        try:
            with open("/proc/cpuinfo", "r") as handle:
                blob = handle.read().lower()
        except OSError:
            blob = ""
    for platform in PLATFORMS:
        if platform in blob:
            return platform
    return None


def geteuid():
    getter = getattr(os, "geteuid", None)
    return getter() if getter is not None else 0


def model_search_dirs(extra_dirs=()):
    """Directories scanned when the user passes a bare model name."""
    dirs = []
    env = os.environ.get("RKLLM_MODEL_DIR")
    if env:
        dirs.extend([d for d in env.split(os.pathsep) if d])
    for path in extra_dirs or ():
        if path:
            dirs.append(path)
    dirs.extend([
        os.path.join("/usr/share", PACKAGE, "models"),
        "/var/lib/rkllm/models",
        os.path.join(os.path.expanduser("~"), ".local", "share", "rkllm", "models"),
    ])
    dirs.append(os.getcwd())
    seen, unique = set(), []
    for d in dirs:
        d = os.path.abspath(os.path.expanduser(d))
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def resolve_model(name, extra_dirs=()):
    """Turn a bare model name or a path into an existing .rkllm file."""
    if os.path.isabs(name) or os.path.dirname(name):
        candidate = os.path.abspath(os.path.expanduser(name))
        if os.path.isfile(candidate):
            return candidate
        raise RKLLMError("no such model file: %s" % candidate)
    names = [name] if name.endswith(MODEL_SUFFIX) else [name + MODEL_SUFFIX, name]
    dirs = model_search_dirs(extra_dirs)
    for directory in dirs:
        for candidate in names:
            path = os.path.join(directory, candidate)
            if os.path.isfile(path):
                return os.path.abspath(path)
    raise RKLLMError(
        "model %r not found. Searched for %s in:\n  %s\n"
        "Pass an absolute path, place the file in one of those directories, or set RKLLM_MODEL_DIR."
        % (name, ", ".join(names), "\n  ".join(dirs)))


def discover_models(extra_dirs=()):
    """Return [(name, path, size)] for every .rkllm file found in the search path."""
    found, seen = [], set()
    for directory in model_search_dirs(extra_dirs):
        try:
            entries = sorted(os.listdir(directory))
        except OSError:
            continue
        for entry in entries:
            if not entry.endswith(MODEL_SUFFIX):
                continue
            path = os.path.join(directory, entry)
            if not os.path.isfile(path):
                continue
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            found.append((entry[:-len(MODEL_SUFFIX)], path, size))
    return found


def _run_quiet(cmd, input_text=None):
    try:
        done = subprocess.run(cmd,
                              input=input_text.encode("utf-8") if input_text is not None else None,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return done.returncode, done.stdout.decode("utf-8", "replace")
    except (OSError, ValueError) as exc:
        return 127, str(exc)


def _snapshot_freq_state():
    """Read the knobs a helper is about to change; unreadable ones are skipped."""
    state = []
    for path in FREQ_GOVERNOR_PATHS + FREQ_CPUIDLE_PATHS:
        try:
            with open(path) as handle:
                state.append((path, handle.read().strip()))
        except OSError:
            continue
    return state


def _read_sysfs(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


def _write_sysfs(path, value):
    """Write one sysfs knob; False when this process may not (or cannot)."""
    if geteuid() == 0:
        try:
            with open(path, "w") as handle:
                handle.write(value + "\n")
            return True
        except OSError:
            return False
    sudo = shutil.which("sudo")
    if not sudo:
        return False
    # tee takes the value on stdin, so no shell ever parses it - shlex.quote()
    # already kept the old form safe, this removes the shell entirely.
    code, _output = _run_quiet([sudo, "-n", "tee", path], input_text=value + "\n")
    return code == 0


def restore_frequencies(quiet=False):
    """Put back every knob the tuning helper touched.

    The upstream helpers only ever lock the clocks: cpuidle is disabled, the
    governors are pinned to "userspace" and the frequencies to their maximum.
    Nothing in them - and nothing in prerm/postrm - undoes that, so a one-shot
    `rkllm run` used to leave the board at full clocks with deep idle disabled
    until the next reboot.

    Governors are restored before cpuidle on purpose: once the governor is back
    to schedutil/ondemand the kernel picks frequencies itself, which is the
    state cpuidle was switched off on top of.
    """
    global _FREQ_SNAPSHOT
    state, _FREQ_SNAPSHOT = _FREQ_SNAPSHOT, []
    if not state:
        return "nothing to restore"
    failed = [path for path, value in state if not _write_sysfs(path, value)]
    if failed and not quiet:
        sys.stderr.write("[rkllm] could not restore %d sysfs knob(s): %s\n"
                         % (len(failed), ", ".join(failed[:3])))
    return "ok" if not failed else "partial (%d failed)" % len(failed)


def lock_frequencies(platform, enabled=True, quiet=False):
    """Raise CPU/NPU/DDR frequencies using the helper shipped with this package.

    Upstream's Python servers shell out to a *relative* `fix_freq_<soc>.sh`,
    which silently does nothing unless the current directory happens to contain
    it.  We call the installed helper by absolute path instead.
    """
    if not enabled:
        return "disabled"
    if platform is None:
        return "skipped (unknown SoC, pass --platform)"
    helper = FREQ_HELPERS.get(platform)
    if not helper or not os.path.exists(helper):
        return "skipped (helper %s not installed)" % (helper or platform)
    global _FREQ_SNAPSHOT
    _FREQ_SNAPSHOT = _snapshot_freq_state()      # taken before anything changes
    cmd = [helper]
    if geteuid() != 0:
        sudo = shutil.which("sudo")
        if not sudo:
            return "skipped (not root and sudo unavailable)"
        cmd = [sudo, "-n"] + cmd
    code, output = _run_quiet(cmd)
    if not quiet and code != 0:
        sys.stderr.write("[rkllm] frequency lock helper exited %s: %s\n"
                         % (code, output.strip()[:200]))
    # The helper does not check what it writes: when the requested frequency is
    # not in this board's available_frequencies table, devfreq rejects it and the
    # script still exits 0 - with the governor already pinned to "userspace",
    # which can leave the board slower than it started. At least say so.
    if code == 0 and not any(_read_sysfs(path) == "userspace"
                             for path in FREQ_GOVERNOR_PATHS):
        if not quiet:
            sys.stderr.write("[rkllm] frequency lock reported success but no governor "
                             "switched to userspace - the board's frequency table may "
                             "not contain the hardcoded value\n")
        return "suspect (no governor changed)"
    return "ok" if code == 0 else "failed (exit %s)" % code


def load_library(explicit=None):
    """Load librkllmrt.so, preferring the copy installed by this package."""
    candidates = []
    env = os.environ.get("RKLLM_LIB")
    if env:
        candidates.append(env)
    if explicit:
        candidates.append(explicit)
    candidates.extend([
        os.path.join(LIBDIR, "librkllmrt.so"),
        "/usr/lib/%s/librkllmrt.so" % TRIPLET,
        "/usr/lib/librkllmrt.so",
        "librkllmrt.so",
    ])
    errors = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError as exc:
            errors.append("  %s: %s" % (candidate, exc))
    raise RKLLMError("cannot load librkllmrt.so. Tried:\n%s\n"
                     "Install rkllm-runtime, or point RKLLM_LIB at the library."
                     % "\n".join(errors))


# ---------------------------------------------------------------------------
# ctypes bindings - mirror of /usr/include/rkllm/rkllm.h (upstream 1.3.1)
# ---------------------------------------------------------------------------

RKLLM_RUN_NORMAL = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH = 2
RKLLM_RUN_ERROR = 3

RKLLM_INPUT_PROMPT = 0
RKLLM_INFER_GENERATE = 0

CPU_MASK_LOW = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
CPU_MASK_HIGH = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)


class _ExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class _Param(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("ignore_eos_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("extend_param", _ExtendParam),
    ]


class _LoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float),
    ]


class _LoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class _PromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class _SamplingParam(ctypes.Structure):
    _fields_ = [
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
    ]


class _InferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(_LoraParam)),
        ("prompt_cache_params", ctypes.POINTER(_PromptCacheParam)),
        ("sampling_params", ctypes.POINTER(_SamplingParam)),
        ("keep_history", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int32),
    ]


class _EmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class _TokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class _ImageInput(ctypes.Structure):
    _fields_ = [
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_start", ctypes.c_char_p),
        ("image_end", ctypes.c_char_p),
        ("image_content", ctypes.c_char_p),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class _VideoInput(ctypes.Structure):
    _fields_ = [
        ("video_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_frame_tokens", ctypes.c_size_t),
        ("n_frame_per_video", ctypes.c_size_t),
        ("n_video", ctypes.c_size_t),
        ("video_start", ctypes.c_char_p),
        ("video_end", ctypes.c_char_p),
        ("video_content", ctypes.c_char_p),
        ("frame_width", ctypes.c_size_t),
        ("frame_height", ctypes.c_size_t),
    ]


class _AudioInput(ctypes.Structure):
    _fields_ = [
        ("audio_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_audio_tokens", ctypes.c_size_t),
        ("n_audio", ctypes.c_size_t),
        ("audio_start", ctypes.c_char_p),
        ("audio_end", ctypes.c_char_p),
        ("audio_content", ctypes.c_char_p),
    ]


class _MultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image", _ImageInput),
        ("video", _VideoInput),
        ("audio", _AudioInput),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", _EmbedInput),
        ("token_input", _TokenInput),
        ("multimodal_input", _MultiModalInput),
    ]


class _Input(ctypes.Structure):
    _anonymous_ = ("input_data",)
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input_data", _InputUnion),
    ]


class _HiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class _Logits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class _PerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class _Result(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32),
        ("last_hidden_layer", _HiddenLayer),
        ("logits", _Logits),
        ("perf", _PerfStat),
    ]


_ResultCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_Result), ctypes.c_void_p, ctypes.c_int)
_TokenizerCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p,
                                      ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32)
_EmbedCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                  ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)


class _Callback(ctypes.Structure):
    _fields_ = [
        ("result_callback", _ResultCallback),
        ("result_userdata", ctypes.c_void_p),
        ("tokenizer_callback", _TokenizerCallback),
        ("tokenizer_userdata", ctypes.c_void_p),
        ("embed_callback", _EmbedCallback),
        ("embed_userdata", ctypes.c_void_p),
    ]


class _Aborted(Exception):
    pass


class _Failure(Exception):
    pass


_END = object()


def _bind_library(lib):
    """Declare prototypes; fail loudly when a symbol is missing."""
    signatures = {
        "rkllm_init": ([ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_Param),
                        ctypes.POINTER(_Callback)], ctypes.c_int),
        "rkllm_run": ([ctypes.c_void_p, ctypes.POINTER(_Input),
                       ctypes.POINTER(_InferParam), ctypes.c_void_p], ctypes.c_int),
        "rkllm_abort": ([ctypes.c_void_p], ctypes.c_int),
        "rkllm_is_running": ([ctypes.c_void_p], ctypes.c_int),
        "rkllm_destroy": ([ctypes.c_void_p], ctypes.c_int),
        "rkllm_clear_kv_cache": ([ctypes.c_void_p, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
        "rkllm_set_chat_template": ([ctypes.c_void_p, ctypes.c_char_p,
                                     ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int),
        "rkllm_set_function_tools": ([ctypes.c_void_p, ctypes.c_char_p,
                                      ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        try:
            func = getattr(lib, name)
        except AttributeError:
            raise RKLLMError("librkllmrt.so is missing the symbol %s "
                             "(is this really the RKLLM runtime?)" % name)
        func.argtypes = argtypes
        func.restype = restype
    return lib


# ---------------------------------------------------------------------------
# inference engine
# ---------------------------------------------------------------------------

class EngineBase(object):
    """Common surface used by the CLI and the HTTP server."""

    model_name = "unknown"
    context_len = DEFAULT_CONTEXT

    def stream(self, prompt, role="user", enable_thinking=False,
               sampling=None, max_new_tokens=None):
        raise NotImplementedError

    def clear_history(self):
        return 0

    def abort(self):
        return None

    def is_running(self):
        return False

    def destroy(self):
        return None

    def diagnostics(self):
        return {}


class RealEngine(EngineBase):
    """ctypes wrapper around a live RKLLM model."""

    # Class-level defaults, not just __init__ assignments: selftest builds an
    # instance with object.__new__ to record the C calls without loading a model,
    # and set_function_tools() now reads these on its first line.
    _function_tools_enabled = False
    _function_tools_signature = None

    def __init__(self, model_path, platform, context_len=DEFAULT_CONTEXT,
                 max_new_tokens=DEFAULT_MAX_NEW_TOKENS, keep_history=True,
                 threads=None, system_prompt=None, lib_path=None, verbose=False):
        """Load librkllmrt.so, build the runtime parameters and init the handle.

        The handle owns the KV cache for the whole process, so its lifetime is the
        service lifetime; release() destroys it. ctypes does not keep the buffers
        behind c_char_p alive on its own, hence the explicit _keep lists.
        """
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        if self.model_name.endswith(MODEL_SUFFIX):
            self.model_name = self.model_name[:-len(MODEL_SUFFIX)]
        self.context_len = context_len
        self.max_new_tokens = max_new_tokens
        self.keep_history = 1 if keep_history else 0
        self.platform = platform
        self.lib, self.lib_path = load_library(lib_path)
        _bind_library(self.lib)

        # Everything handed to C is kept alive for the lifetime of the engine:
        # ctypes does not copy the buffers behind c_char_p on its own.
        self._keep = []          # long-lived buffers (model path & friends)
        self._run_keep = []      # per-inference buffers, dropped after each run
        self._sink = None
        self._sink_lock = threading.Lock()
        # The docstring says "serialised by the caller"; this turns the implicit
        # contract into an explicit failure. _infer is shared state, so two
        # concurrent runs corrupt each other's sampling parameters.
        self._stream_guard = threading.Lock()
        # Real counts from the runtime's own perf stats (RKLLM_RUN_FINISH).
        # A tuple (prefill_tokens, generate_tokens), or None before the first
        # completed run. Preferred over the byte estimator for usage.
        self._last_perf = None
        self._handle = ctypes.c_void_p()
        self._destroyed = False
        self._function_tools_signature = None
        # Set once rkllm_set_function_tools has been called: the runtime offers
        # no documented way back, so /health reports it and the caller is warned.
        self._function_tools_enabled = False

        param = _Param()
        param.model_path = model_path.encode("utf-8")
        self._keep.append(param.model_path)
        param.max_context_len = int(context_len)
        param.max_new_tokens = int(max_new_tokens)
        param.n_keep = -1
        param.top_k = 1
        param.top_p = 0.9
        param.temperature = 0.8
        param.repeat_penalty = 1.1
        param.frequency_penalty = 0.0
        param.presence_penalty = 0.0
        param.mirostat = 0
        param.mirostat_tau = 5.0
        param.mirostat_eta = 0.1
        param.skip_special_token = True
        param.ignore_eos_token = False
        param.is_async = False
        param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1
        param.extend_param.n_batch = 1
        param.extend_param.use_cross_attn = 0
        cpu_count = int(threads) if threads else 4
        # num and mask have to agree: a four-bit mask with num=2 leaves the
        # runtime to pick which two of the four cores to use, which is not a
        # decision anyone can reason about afterwards.
        base_mask = CPU_MASK_HIGH if platform in BIG_CORE_PLATFORMS else CPU_MASK_LOW
        candidates = [i for i in range(32) if base_mask & (1 << i)]
        chosen = candidates[:max(1, min(cpu_count, len(candidates)))]
        param.extend_param.enabled_cpus_num = len(chosen)
        param.extend_param.enabled_cpus_mask = sum(1 << i for i in chosen)

        self._callback_type = _ResultCallback(self._on_result)
        callback = _Callback()
        callback.result_callback = self._callback_type
        callback.result_userdata = None
        callback.tokenizer_callback = _TokenizerCallback()
        callback.tokenizer_userdata = None
        callback.embed_callback = _EmbedCallback()
        callback.embed_userdata = None
        self._callback = callback
        self._param = param

        if verbose:
            sys.stderr.write("[rkllm] loading %s with %s\n" % (model_path, self.lib_path))
        code = self.lib.rkllm_init(ctypes.byref(self._handle), ctypes.byref(param),
                                   ctypes.byref(callback))
        if code != 0:
            available, detail = npu_status()
            hint = ("the NPU driver is present (%s)" % detail if available
                    else "and no NPU driver could be found")
            raise RKLLMError("rkllm_init failed with code %d. Common causes: the model "
                             "file is not a valid .rkllm for this runtime version, %s, "
                             "or NPU memory is exhausted." % (code, hint))

        self._infer = _InferParam()
        ctypes.memset(ctypes.byref(self._infer), 0, ctypes.sizeof(_InferParam))
        self._infer.mode = RKLLM_INFER_GENERATE
        self._infer.lora_params = None
        self._infer.prompt_cache_params = None
        self._infer.sampling_params = None
        self._infer.keep_history = self.keep_history
        self._infer.max_new_tokens = 0

        # Kept as text and prepended in stream(). Pushing it through
        # rkllm_set_chat_template would disable the model's built-in template.
        self.system_prompt = (system_prompt or "").strip()

    # -- callbacks ---------------------------------------------------------

    def _on_result(self, result, userdata, state):
        sink = self._sink
        try:
            if state == RKLLM_RUN_NORMAL:
                chunk = result.contents.text
                if chunk and sink is not None:
                    sink.put(chunk.decode("utf-8", "replace"))
            elif state == RKLLM_RUN_FINISH:
                # The final callback carries the runtime's own token counters -
                # the only exact numbers available without a tokeniser.
                try:
                    perf = result.contents.perf
                    self._last_perf = (int(perf.prefill_tokens), int(perf.generate_tokens))
                except Exception:                            # never unwind into C
                    pass
            elif state == RKLLM_RUN_ERROR:
                if sink is not None:
                    sink.put(_Failure("the runtime reported RKLLM_RUN_ERROR"))
        except Exception:                                    # never unwind into C
            if sink is not None:
                try:
                    sink.put(_Failure("callback failure"))
                except Exception:
                    pass
        return 0

    # -- api ---------------------------------------------------------------

    def set_system_prompt(self, system_prompt):
        """Install a custom chat template. USE WITH CARE.

        This maps to rkllm_set_chat_template, which the runtime documents as
        *disabling the built-in automatic chat template parsing* (including
        enable_thinking). After calling it the model receives bare text with none
        of its special tokens, so it rambles or never emits EOS - and because the
        setting sticks for the life of the handle, it poisons every later
        request too. That is why nothing in this script calls it: a system prompt
        is prepended to the prompt text instead.

        Two further traps, both observed on real hardware:
        - prompt_prefix / prompt_postfix are dereferenced unconditionally, so
          passing NULL there segfaults the process. Always pass a real string.
        - An empty system_prompt still resets the template, it does not restore
          the built-in one.
        """
        return self.lib.rkllm_set_chat_template(
            self._handle, system_prompt.encode("utf-8"), b"", b"")

    def set_function_tools(self, system_prompt, tools_json, tool_response_str="tool_response"):
        """Enable function calling on this handle. USE WITH CARE.

        rkllm_set_function_tools has the same permanence trap as
        rkllm_set_chat_template: the setting applies to every later rkllm_run
        call on this handle and the runtime documents no way to switch it off,
        so one request that carries "tools" changes the behaviour of all the
        requests that follow until the process restarts. That is why nothing in
        this script calls it unless the client asked for tools.

        An empty system_prompt is dangerous for the same reason it is in
        set_system_prompt - the runtime treats an empty value as "reset", not
        "keep" - so a caller without a system prompt gets a neutral placeholder
        rather than an empty string.

        ctypes does not keep the buffers behind c_char_p alive, and the runtime
        may retain these pointers for later runs; they are parked in _keep for
        the lifetime of the engine.
        """
        # The runtime has no documented way to switch function calling off, so
        # a second call would only append another set of buffers to _keep - and
        # _keep lives as long as the engine. An agent client that sends "tools"
        # with every request would leak its whole tool schema (several KB) per
        # request, which on an 8 GB board is a real leak rather than a rounding
        # error. Set it once and report the no-op on later calls.
        signature = hash((system_prompt or "", tools_json, tool_response_str))
        if self._function_tools_enabled:
            if signature != self._function_tools_signature and not getattr(self, "_warned_tools_change", False):
                self._warned_tools_change = True
                sys.stderr.write(
                    "[rkllm] tools changed after function calling was enabled; the "
                    "runtime keeps the first definition until restart\n")
                sys.stderr.flush()
            return 0
        if not (system_prompt or "").strip():
            system_prompt = "You are a helpful assistant."
        buffers = [system_prompt.encode("utf-8"), tools_json.encode("utf-8"),
                   tool_response_str.encode("utf-8")]
        self._keep.extend(buffers)
        self._function_tools_enabled = True
        self._function_tools_signature = signature
        return self.lib.rkllm_set_function_tools(self._handle, *buffers)

    def stream(self, prompt, role="user", enable_thinking=False,
               sampling=None, max_new_tokens=None):
        """Yield generated text fragments. Serialised by the caller.

        Concurrency is enforced rather than assumed: _infer is engine-wide, so a
        second concurrent run would read sampling parameters another run is
        mid-way through tearing down.
        """
        if self._destroyed:
            raise RKLLMError("engine already destroyed")
        if not self._stream_guard.acquire(blocking=False):
            raise RKLLMError("concurrent stream() calls on one engine are not supported")

        self._last_perf = None
        sink = queue.Queue()
        with self._sink_lock:
            self._sink = sink

        if self.system_prompt:
            prompt = "%s\n\n%s" % (self.system_prompt, prompt)

        rkllm_input = _Input()
        role_bytes = role.encode("utf-8")
        rkllm_input.role = role_bytes
        rkllm_input.enable_thinking = bool(enable_thinking)
        rkllm_input.input_type = RKLLM_INPUT_PROMPT
        encoded = prompt.encode("utf-8")
        rkllm_input.prompt_input = encoded
        # The prompt buffer has to stay alive until rkllm_run returns; hold it in
        # a per-run list that is dropped in the finally block below, so a server
        # that answers thousands of requests does not accumulate memory.
        self._run_keep = [role_bytes, encoded]

        self._infer.sampling_params = ctypes.pointer(sampling) if sampling is not None else None
        self._infer.max_new_tokens = int(max_new_tokens) if max_new_tokens else 0

        def runner():
            try:
                code = self.lib.rkllm_run(self._handle, ctypes.byref(rkllm_input),
                                          ctypes.byref(self._infer), None)
                if code != 0:
                    sink.put(_Failure("rkllm_run returned %d" % code))
            except Exception as exc:                          # pragma: no cover
                sink.put(_Failure("inference crashed: %s" % exc))
            finally:
                sink.put(_END)

        thread = threading.Thread(target=runner, name="rkllm-infer")
        thread.daemon = True
        thread.start()
        try:
            while True:
                item = sink.get()
                if item is _END:
                    break
                if isinstance(item, _Failure):
                    raise item
                yield item
        finally:
            thread.join(timeout=STREAM_JOIN_TIMEOUT_S)
            if thread.is_alive():
                # abort() was not enough; the handle is in an unknown state, so
                # refuse further work instead of corrupting the next request.
                sys.stderr.write("[rkllm] inference thread did not exit within %ds of "
                                 "abort; marking the engine unusable\n" % STREAM_JOIN_TIMEOUT_S)
                sys.stderr.flush()
                self._destroyed = True
            with self._sink_lock:
                self._sink = None
            self._run_keep = []
            self._infer.sampling_params = None
            self._infer.max_new_tokens = 0
            self._stream_guard.release()

    def last_perf(self):
        """(prefill_tokens, generate_tokens) from the last completed run, or None."""
        return self._last_perf

    def clear_history(self):
        return self.lib.rkllm_clear_kv_cache(self._handle, 1, None, None)

    def abort(self):
        try:
            return self.lib.rkllm_abort(self._handle)
        except Exception:
            return None

    def is_running(self):
        try:
            return bool(self.lib.rkllm_is_running(self._handle))
        except Exception:
            return False

    def destroy(self):
        if self._destroyed:
            return
        self._destroyed = True
        try:
            self.lib.rkllm_destroy(self._handle)
        except Exception:
            pass

    def diagnostics(self):
        return {
            "library": self.lib_path,
            "model_path": self.model_path,
            "keep_history": self.keep_history,
        }


class FakeEngine(EngineBase):
    """Deterministic stand-in used by `rkllm selftest` - no model, no NPU."""

    def __init__(self, model_name="selftest", context_len=DEFAULT_CONTEXT,
                 keep_history=True, delay=0.0, system_prompt=None, **kwargs):
        self.model_name = model_name
        self.model_path = "<fake>"
        self.context_len = context_len
        self.keep_history = 1 if keep_history else 0
        self.delay = delay
        self.system_prompt = (system_prompt or "").strip()
        self._turns = 0
        self._destroyed = False

    def stream(self, prompt, role="user", enable_thinking=False,
               sampling=None, max_new_tokens=None):
        self._turns += 1
        if self.system_prompt:
            prompt = "%s\n\n%s" % (self.system_prompt, prompt)
        answer = "answer %d to: %s" % (self._turns, prompt)
        for char in answer:
            if self.delay:
                time.sleep(self.delay)
            yield char

    def last_perf(self):
        return None

    def clear_history(self):
        self._turns = 0
        return 0

    def destroy(self):
        self._destroyed = True

    def diagnostics(self):
        return {"library": "<fake>", "model_path": "<fake>", "keep_history": self.keep_history}


class _GateEngine(FakeEngine):
    """Fake engine that parks inside the first inference until released.

    Used by `rkllm selftest` to make the queue behaviour deterministic: while
    one request is parked in the "NPU", a second one may take the single queue
    slot and anything beyond that must be refused.
    """

    def __init__(self):
        FakeEngine.__init__(self, model_name="gate")
        self.started = threading.Event()
        self.release = threading.Event()

    def stream(self, prompt, role="user", enable_thinking=False,
               sampling=None, max_new_tokens=None):
        self.started.set()
        self.release.wait(15)
        for fragment in FakeEngine.stream(self, prompt):
            yield fragment


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def build_sampling(args):
    """RKLLMSamplingParam from CLI/API values, or None when nothing was set."""
    sampler = _SamplingParam()
    sampler.top_k = int(args.get("top_k", 1))
    sampler.top_p = float(args.get("top_p", 0.9))
    sampler.temperature = float(args.get("temperature", 0.8))
    sampler.repeat_penalty = float(args.get("repeat_penalty", 1.1))
    sampler.frequency_penalty = float(args.get("frequency_penalty", 0.0))
    sampler.presence_penalty = float(args.get("presence_penalty", 0.0))
    sampler.mirostat = 0
    sampler.mirostat_tau = 5.0
    sampler.mirostat_eta = 0.1
    return sampler


def make_engine(args, model_path):
    """Instantiate the real engine unless --fake-engine was given."""
    if getattr(args, "fake_engine", False):
        name = os.path.basename(model_path)
        if name.endswith(MODEL_SUFFIX):
            name = name[:-len(MODEL_SUFFIX)]
        return FakeEngine(model_name=name, context_len=args.context,
                          system_prompt=getattr(args, "system", None))
    return RealEngine(
        model_path,
        platform=args.platform,
        context_len=args.context,
        max_new_tokens=args.max_new_tokens,
        keep_history=not args.no_history,
        threads=args.threads,
        system_prompt=args.system,
        lib_path=args.lib,
        verbose=args.verbose,
    )


def engine_options(args):
    return {
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repeat_penalty": args.repeat_penalty,
    }


def human_size(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return "%.1f %s" % (value, unit)
        value /= 1024.0
    return "%.1f GiB" % value


# ---------------------------------------------------------------------------
# chat / run
# ---------------------------------------------------------------------------

BANNER = """\
%s %s - RKLLM runtime front end
model      : %s
file       : %s
context    : %d tokens (max new tokens %s)
history    : %s
commands   : /help /clear /set <name> <value> /exit
"""

HELP_TEXT = """\
/help                     show this help
/clear                    drop the KV cache (forget the conversation)
/set <name> <value>       change top_k, top_p, temperature or repeat_penalty
/exit, /quit              leave
"""


def _apply_set_command(line, args):
    parts = line.split()
    allowed = ("top_k", "top_p", "temperature", "repeat_penalty")
    if len(parts) != 3 or parts[1] not in allowed:
        return "usage: /set <%s> <value>" % "|".join(allowed)
    try:
        value = float(parts[2])
    except ValueError:
        return "value must be numeric"
    if parts[1] == "top_k":
        args.top_k = int(value)
    else:
        setattr(args, parts[1], value)
    return "%s = %s" % (parts[1], parts[2])


def cmd_chat(args):
    """Interactive chat on this machine (REPL)."""
    model_path = resolve_model(args.model, args.model_dir)
    engine = make_engine(args, model_path)
    state = lock_frequencies(args.platform, enabled=not args.no_freq_lock, quiet=args.quiet)
    if not args.quiet:
        sys.stderr.write(BANNER % (PROGRAM, VERSION, engine.model_name,
                                   getattr(engine, "model_path", "?"), args.context,
                                   args.max_new_tokens,
                                   "on" if not args.no_history else "off"))
        sys.stderr.write("frequency lock: %s\n" % state)

    interactive = sys.stdin.isatty()
    exit_code = 0
    try:
        while True:
            if interactive:
                sys.stdout.write("\nuser> ")
                sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                break
            text = line.strip()
            if not text:
                continue
            if text in ("/exit", "/quit", "exit", "quit"):
                break
            if text == "/help":
                sys.stdout.write(HELP_TEXT)
                continue
            if text in ("/clear", "clear"):
                code = engine.clear_history()
                sys.stdout.write("kv cache cleared (rc=%s)\n" % code)
                continue
            if text.startswith("/set "):
                sys.stdout.write(_apply_set_command(text, args) + "\n")
                continue

            sys.stdout.write("bot> " if interactive else "")
            sys.stdout.flush()
            try:
                for fragment in engine.stream(text, sampling=build_sampling(engine_options(args)),
                                              max_new_tokens=args.max_new_tokens):
                    sys.stdout.write(fragment)
                    sys.stdout.flush()
            except _Failure as exc:
                sys.stdout.write("\n[error] %s\n" % exc)
                exit_code = 1
            except KeyboardInterrupt:
                engine.abort()
                sys.stdout.write("\n[interrupted]\n")
            sys.stdout.write("\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
    finally:
        engine.destroy()
        restore_frequencies(quiet=args.quiet)
    return exit_code


def cmd_run(args):
    """Answer a single prompt, print it, and exit."""
    prompt = " ".join(args.text).strip()
    if not prompt:
        raise RKLLMError("no prompt given")
    model_path = resolve_model(args.model, args.model_dir)
    engine = make_engine(args, model_path)
    lock_frequencies(args.platform, enabled=not args.no_freq_lock, quiet=args.quiet)
    try:
        for fragment in engine.stream(prompt, sampling=build_sampling(engine_options(args)),
                                      max_new_tokens=args.max_new_tokens):
            sys.stdout.write(fragment)
            sys.stdout.flush()
        sys.stdout.write("\n")
    finally:
        engine.destroy()
        restore_frequencies(quiet=args.quiet)
    return 0


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP server
# ---------------------------------------------------------------------------

def _estimate_tokens(text):
    """Approximate a token count for the usage report.

    There is no tokeniser on the device, so this is a byte-based approximation
    and it is deliberately the same one the size guard uses. It replaced
    len(text.split()), which treated any space-free string as a single token and
    therefore under-reported every Chinese request by roughly an order of
    magnitude: a 15-token Chinese question was reported as "1".

    Accuracy is language dependent - about three bytes per token fits Chinese
    (roughly one token per character, three bytes each), while Latin text runs
    nearer four bytes per token, so English counts come out somewhat high. High
    is the safe direction for a budget check; the numbers are an estimate, not a
    measurement.
    """
    if not text:
        return 0
    return max(1, len(text.encode("utf-8")) // 3)


def _openai_response(model, content, finish_reason="stop", prompt_tokens=0, completion_tokens=0):
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _openai_usage_chunk(model, prompt_tokens, completion_tokens):
    """Final SSE frame for stream_options.include_usage (OpenAI shape).

    choices is empty on purpose: the usage frame carries no text, and clients
    that ask for it know to look for an empty choices array.
    """
    payload = {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return "data: %s\n\n" % json.dumps(payload, ensure_ascii=False)


def _openai_chunk(model, delta, finish_reason=None, role=None):
    body = {}
    if role:
        body["role"] = role
    if delta is not None:
        body["content"] = delta
    payload = {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": body,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
    }
    return "data: %s\n\n" % json.dumps(payload, ensure_ascii=False)


def content_text(content):
    """Flatten a message's content field to plain text.

    Handles the OpenAI multimodal shape ([{"type": "text", "text": ...}, ...]),
    plain strings, and anything else, which yields an empty string. Callers that
    need the text of a system prompt must go through this too: a client is free
    to send the system prompt in the multimodal shape, and treating that as a
    string used to raise AttributeError further down.
    """
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") == "text")
    if isinstance(content, str):
        return content
    return ""


def build_conversation(messages, budget_tokens=None, include_history=True):
    """Flatten an OpenAI messages array into (role, prompt, system_prompt).

    Stateless contract: an agent client resends the whole conversation on every
    request, so the earlier turns have to travel inside the prompt - the service
    keeps no history of its own (serve answers every request independently by
    default). Only the last user/tool turn is handed to the runtime as the
    prompt proper; the turns before it are rendered as a readable transcript so
    the model can follow the thread.

    budget_tokens trims the transcript from the oldest turn instead of rejecting
    the request: a long conversation against a 4096-token window must still get
    an answer, just with its earliest turns dropped.

    include_history=False is for the legacy mode where the server itself keeps
    the KV history (RKLLM_NO_HISTORY=0): replaying the transcript there would
    double every earlier turn in the model's context.
    """
    system = ""
    turns = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        text = content_text(message.get("content"))
        if role == "system":
            if text.strip():
                system = text              # the last system message wins
            continue
        if role in ("user", "assistant", "tool") and text.strip():
            turns.append((role, text))

    if not include_history:
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") in ("user", "tool"):
                text = content_text(message.get("content"))
                if text.strip():
                    return "user", text, system
        turns = turns[-1:]               # fall back to the last usable turn

    if not turns:
        # Clients probe a freshly configured model with a system-only or empty
        # request; answering beats a 400 that reads as "the model is broken".
        for message in reversed(messages):
            if isinstance(message, dict):
                text = content_text(message.get("content"))
                if text.strip():
                    if message.get("role") == "system":
                        system = ""
                    return "user", text, system
        return "user", "", system

    def render(items):
        lines = []
        for role, text in items:
            # English labels on purpose: this server also backs non-Chinese
            # models, and "用户/助手" is noise for them. Qwen parses either.
            label = {"user": "User", "assistant": "Assistant", "tool": "Tool"}.get(role, role)
            lines.append("%s: %s" % (label, text))
        return "\n".join(lines)

    dropped = 0
    if budget_tokens:
        while len(turns) > 1:
            if (_estimate_tokens(render(turns)) + _estimate_tokens(system)) <= budget_tokens:
                break
            turns.pop(0)
            dropped += 1

    if len(turns) == 1:
        role, current = turns[0]
        return ("user" if role == "tool" else role), current, system

    prefix = "Previous conversation:\n"
    if dropped:
        prefix = ("(the earliest %d turn(s) were dropped to fit the context window)\n"
                  % dropped) + prefix
    role, current = turns[-1]
    prompt = prefix + render(turns[:-1]) + "\n\n" + current
    return ("user" if role == "tool" else role), prompt, system


def _normalise_stop(value):
    """OpenAI 'stop' -> list of non-empty strings (at most 4, like OpenAI)."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
        if len(out) >= 4:
            break
    return out


def _stop_index(text, stops):
    """Offset of the earliest stop sequence in text, or None."""
    best = None
    for seq in stops:
        found = text.find(seq)
        if found >= 0 and (best is None or found < best):
            best = found
    return best


def _apply_stop(text, stops):
    """(visible_text, did_stop) - cut at the first stop sequence."""
    if not stops:
        return text, False
    cut = _stop_index(text, stops)
    if cut is None:
        return text, False
    return text[:cut], True


def _finish_reason(stopped_at_stop, completion_tokens, max_tokens):
    """'length' when generation ran into max_tokens, 'stop' otherwise.

    Reporting 'stop' for a truncated answer is worse than useless: an agent
    treats the text as complete and concatenates or parses it as if the model
    had finished.

    The runtime's generate_tokens counter tops out one short of max_new_tokens
    when it cuts generation off (measured on 1.3.1-14: max_tokens 8 -> 7,
    40 -> 39, 6 -> 5), while a natural end reports well below it, so the
    comparison allows that single-token slack.
    """
    if stopped_at_stop:
        return "stop"
    if max_tokens and completion_tokens and completion_tokens >= max_tokens - 1:
        return "length"
    return "stop"


def _tokens_for(engine, prompt, answer):
    """Exact counts from the runtime when available, byte estimate otherwise."""
    perf = None
    getter = getattr(engine, "last_perf", None)
    if callable(getter):
        try:
            perf = getter()
        except Exception:                                    # noqa: BLE001
            perf = None
    if perf:
        prefill, generated = perf
        if prefill > 0 or generated > 0:
            return max(int(prefill), 1), max(int(generated), 0)
    return _estimate_tokens(prompt), _estimate_tokens(answer)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # socketserver defaults this to 5, which makes the kernel drop connections
    # under a burst instead of letting them wait: a stress run of 60 concurrent
    # requests produced 15 client-side connection errors and no 503 at all,
    # because the requests never reached our queue logic. Sized so the queue
    # logic (and its 503 with Retry-After) is what sheds load.
    request_queue_size = 256

    def __init__(self, address, handler, engine, options):
        ThreadingHTTPServer.__init__(self, address, handler)
        self.engine = engine
        self.options = options
        self.infer_lock = threading.Lock()
        # max_queue counts the requests that may *wait*. The one that is
        # actually running on the NPU needs a slot of its own, hence the +1.
        self.queue_slots = threading.BoundedSemaphore(options.max_queue + 1)
        self.waiting = 0                      # slots held but not yet inferring
        self._waiting_lock = threading.Lock()
        self.stats = {"requests": 0, "rejected": 0, "errors": 0,
                      "prompt_tokens": 0, "completion_tokens": 0, "started": time.time()}

    def adjust_waiting(self, delta):
        with self._waiting_lock:
            self.waiting += delta
            return self.waiting

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        if not self.options.quiet:
            sys.stderr.write("[rkllm] request from %s failed: %s\n" % (client_address, exc))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = PROGRAM + "/" + VERSION
    sys_version = ""

    # -- plumbing ---------------------------------------------------------

    def log_message(self, fmt, *args):
        if self.server.options.quiet:
            return
        sys.stderr.write("[rkllm] %s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, payload, status=200, headers=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", ALLOW_ORIGIN)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message, status=400, error_type="invalid_request_error",
                         headers=None):
        self._send_json({"error": {"message": message, "type": error_type,
                                   "param": None, "code": None}}, status, headers)

    def _reject(self, message, status=400, error_type="invalid_request_error"):
        """Count the refusal, say why in the journal, then answer.

        The journal line is the only place an operator can see what a client
        actually sent: a UI reports our 400 as "the model is unavailable", which
        hides the real cause entirely.
        """
        self.server.stats["rejected"] += 1
        if not self.server.options.quiet:
            sys.stderr.write("[rkllm] rejected (%d): %s\n" % (status, message))
            sys.stderr.flush()
        self._send_error_json(message, status, error_type)

    def _authorised(self):
        key = self.server.options.api_key
        if not key:
            return True
        header = self.headers.get("Authorization", "")
        # constant-time compare: a plain == would leak the key length by timing
        return hmac.compare_digest(header.encode("utf-8"), ("Bearer " + key).encode("utf-8"))

    def _read_body(self):
        # A client can announce a large Content-Length and then trickle bytes;
        # without a socket timeout the worker thread stays busy indefinitely.
        try:
            self.connection.settimeout(READ_TIMEOUT_S)
        except OSError:
            pass
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None, "invalid Content-Length"
        if length <= 0:
            return None, "empty request body"
        if length > MAX_BODY_BYTES:
            return None, "request body larger than %d bytes" % MAX_BODY_BYTES
        try:
            raw = self.rfile.read(length)
        except (socket.timeout, TimeoutError):
            return None, ("timed out after %ds reading a %d-byte body"
                          % (READ_TIMEOUT_S, length))
        try:
            return json.loads(raw.decode("utf-8")), None
        except (UnicodeDecodeError, ValueError) as exc:
            return None, "invalid JSON body: %s" % exc

    # -- routes -----------------------------------------------------------

    def do_GET(self):
        """Handle GET /health, /v1/models, / and /v1."""
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if not self._authorised():
            self._send_error_json("missing or wrong API key", 401, "authentication_error")
            return
        if path == "/health":
            engine = self.server.engine
            self._send_json({
                "status": "ok",
                "model": engine.model_name,
                "busy": engine.is_running(),
                "function_tools": bool(getattr(engine, "_function_tools_enabled", False)),
                "queue_length": self.server.waiting,
                "uptime_s": round(time.time() - self.server.stats["started"], 1),
                "stats": dict(self.server.stats),
            })
            return
        # Both /v1/... and the bare path are accepted: clients differ in whether
        # the "/v1" belongs in the base URL they were configured with, and a 404
        # caused by that is hard for an operator to diagnose.
        if path in ("/v1/models", "/models"):
            self._send_json({
                "object": "list",
                "data": [{"id": self.server.engine.model_name, "object": "model",
                          "created": int(time.time()), "owned_by": "rkllm"}],
            })
            return
        if path in ("/", "/v1"):
            self._send_json({"service": PROGRAM, "version": VERSION,
                             "model": self.server.engine.model_name,
                             "endpoints": ["/health", "/v1/models", "/v1/chat/completions",
                                           "/v1/reset"],
                             "note": "base_url for clients is http://<host>:<port>/v1"})
            return
        self._send_error_json("unknown endpoint %s; use /v1/models or set the client "
                              "base_url to http://<host>:<port>/v1" % path, 404, "not_found")

    def do_POST(self):
        """Handle POST /v1/chat/completions and /v1/reset."""
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if not self._authorised():
            self._send_error_json("missing or wrong API key", 401, "authentication_error")
            return
        if path in ("/v1/reset", "/reset"):
            # Drain any body before answering. protocol_version is HTTP/1.1, so
            # an unread Content-Length body stays in the socket and the next
            # pipelined request parses those bytes as its request line - most
            # HTTP clients send "{}" here, which makes this a certainty rather
            # than a corner case.
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length > 0:
                if length > MAX_BODY_BYTES:
                    self.close_connection = True
                    self._reject("request body larger than %d bytes" % MAX_BODY_BYTES, 413)
                    return
                self.rfile.read(length)
            self.server.engine.clear_history()
            self._send_json({"object": "reset", "status": "ok"})
            return
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_error_json(
                "unknown endpoint %s; the completion endpoint is /v1/chat/completions and the "
                "base_url for clients is http://<host>:<port>/v1" % path, 404, "not_found")
            return

        # A chunked body carries no Content-Length and BaseHTTPRequestHandler
        # does not decode it: reading zero bytes would leave the chunk frames in
        # the socket, where the next request on this keep-alive connection would
        # parse them as part of its own body. Refuse the framing instead of
        # smuggling it through.
        if (self.headers.get("Transfer-Encoding") or "").strip().lower() not in ("", "identity"):
            self.close_connection = True
            self._reject("chunked request bodies are not supported; send the body with "
                         "Content-Length and without Transfer-Encoding", 411)
            return

        body, problem = self._read_body()
        if problem:
            self._reject(problem)
            return
        if not isinstance(body, dict) or "messages" not in body:
            self._reject("missing 'messages' in request body")
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._reject("'messages' must be a non-empty array")
            return

        engine = self.server.engine
        model_name = body.get("model") or engine.model_name
        stream = bool(body.get("stream", False))

        try:
            n = int(body.get("n", 1) or 1)
        except (TypeError, ValueError):
            self._reject("'n' must be an integer")
            return
        if n > 1:
            # One NPU generates one candidate; silently returning a single
            # choice for n=2 makes a caller believe it got two.
            self._reject("'n' > 1 is not supported: this server runs a single NPU and "
                         "always returns one choice", 400, "invalid_request_error")
            return

        stream_options = body.get("stream_options")
        include_usage = bool(stream_options.get("include_usage", False)) \
            if isinstance(stream_options, dict) else False
        stop_seqs = _normalise_stop(body.get("stop"))

        # Stateless contract: the client resends the whole conversation, so the
        # transcript is rebuilt here. Leave a quarter of the window for the
        # answer and trim the oldest turns rather than refusing a long chat.
        context_hint = max(int(getattr(engine, "context_len", DEFAULT_CONTEXT) or DEFAULT_CONTEXT), 1)
        carries = bool(getattr(engine, "keep_history", 0))
        role, prompt, system_prompt = build_conversation(
            messages, budget_tokens=max(context_hint - context_hint // 4, 128),
            include_history=not carries)
        response_format = body.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            # Kept in Chinese: the deployment serves Chinese documents, where a
            # Chinese instruction lands better than an English one. Move it into
            # a config knob if this ever serves another locale.
            prompt = prompt + "\n\n只输出一个 JSON 对象，不要任何解释、前后缀或代码块标记。"
        if not prompt.strip():
            # Say which shape arrived: a UI reports this 400 as "the model is
            # unavailable", so the journal has to carry the real cause.
            shapes = []
            for message in messages[:8]:
                if isinstance(message, dict):
                    content = message.get("content")
                    shapes.append("%s/%s/%dchars" % (
                        message.get("role"), type(content).__name__,
                        len(content_text(content))))
                else:
                    shapes.append(type(message).__name__)
            self._reject("no message with usable text: every message has empty "
                         "or null content (got [%s]). Send a non-empty 'content' "
                         "string." % ", ".join(shapes))
            return

        try:
            max_tokens = int(body.get("max_tokens", DEFAULT_MAX_NEW_TOKENS))
        except (TypeError, ValueError):
            self._reject("'max_tokens' must be an integer")
            return
        max_tokens = max(1, min(max_tokens, engine.context_len))

        try:
            sampling = build_sampling(body)
        except (TypeError, ValueError):
            self._reject("sampling parameters must be numeric")
            return
        thinking = bool(body.get("enable_thinking", False))
        tools = body.get("tools")

        if not self.server.queue_slots.acquire(blocking=False):
            self.server.stats["rejected"] += 1
            self._send_error_json("server queue is full (%d already waiting), retry later"
                                  % self.server.options.max_queue, 503, "server_error",
                                  {"Retry-After": "5"})
            return
        self.server.adjust_waiting(1)
        try:
            acquired = self.server.infer_lock.acquire(timeout=self.server.options.queue_timeout)
            self.server.adjust_waiting(-1)
            if not acquired:
                self.server.stats["rejected"] += 1
                # A single explicit response: _send_json would emit a second status line.
                body = json.dumps({"error": {"message": "timed out waiting for the NPU",
                                             "type": "server_error"}}).encode("utf-8")
                self.send_response(503)
                self.send_header("Retry-After", "5")
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            try:
                self._run_inference(engine, model_name, prompt, role, system_prompt,
                                    stream=stream, sampling=sampling,
                                    max_tokens=max_tokens, thinking=thinking, tools=tools,
                                    stop=stop_seqs, include_usage=include_usage)
            finally:
                self.server.infer_lock.release()
        finally:
            self.server.queue_slots.release()

    # -- inference --------------------------------------------------------

    def _client_gone(self):
        """客户端是否已断开（非流式推理期间没有写操作，靠 BrokenPipe 发现不了）。

        可读 + MSG_PEEK 读到 EOF 才算断开；可读但仍有数据是 pipelined 的下一
        请求，不能当断开。MSG_PEEK 不消费数据，后续解析不受影响。
        """
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True                      # 套接字已不可用，按断开处理

    def do_OPTIONS(self):
        """CORS preflight: browser-based clients cannot POST cross-origin otherwise."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", ALLOW_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _begin_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", ALLOW_ORIGIN)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _run_inference(self, engine, model_name, prompt, role, system_prompt,
                       stream=False, sampling=None, max_tokens=None,
                       thinking=False, tools=None, stop=None, include_usage=False):
        """Serve one request: apply the size guards, run the engine, emit the reply.

        Guards matter here. A prompt that does not fit the model's window makes the
        native runtime overrun its KV cache and die, so the size is checked before the
        call, and a system prompt that cannot fit is dropped rather than failing the
        whole request.
        """
        # A request-level system message is prepended to the prompt rather than
        # pushed through rkllm_set_chat_template. That call disables the model's
        # built-in template parsing (the runtime warns about it), after which the
        # model receives bare text without any of its special tokens - it then
        # rambles or never emits EOS, and because the setting persists for the
        # whole handle it poisons every later request as well.
        user_prompt = prompt
        system_bytes = 0
        if system_prompt:
            system_bytes = len(system_prompt.encode("utf-8"))
            prompt = "%s\n\n%s" % (system_prompt.strip(), prompt)

        # How many tokens a byte produces depends on the language: English runs
        # about four bytes per token, Chinese about three bytes per character at
        # roughly one token per character. An earlier version of this guard
        # allowed eight bytes per token, which let a 32 KB English prompt (about
        # 8000 tokens) past a 4096-token window - the runtime then overran its
        # KV cache and segfaulted, killing the service. Three bytes per token is
        # the safe side for both languages.
        context_len = max(engine.context_len, 1)
        # server-side --system / RKLLM_SYSTEM is prepended inside engine.stream(),
        # after this guard runs. Leaving it out of the estimate let a long startup
        # preamble sail past a 4096-token window - the exact KV-overrun path that
        # segfaults the service.
        engine_system = (getattr(engine, "system_prompt", "") or "").strip()
        engine_system_bytes = len(engine_system.encode("utf-8"))
        user_tokens = (len(user_prompt.encode("utf-8")) + engine_system_bytes) // 3
        estimated_tokens = (len(user_prompt.encode("utf-8")) + system_bytes
                            + engine_system_bytes + 2) // 3

        if estimated_tokens >= context_len:
            # A chat client's own system prompt can be enormous: an agent UI sent
            # 57 KB of preamble alongside a two-character "hello", which no 4096
            # token window can hold, so the user could not even say hello. Drop
            # the preamble and serve the request - the model loses its
            # instructions but still answers, and that is far more useful than a
            # 400 the user cannot act on. Only do so when the user's own text
            # fits, and always say so in the log.
            if system_prompt and user_tokens < context_len:
                if not self.server.options.quiet:
                    sys.stderr.write("[rkllm] system prompt dropped: %d bytes does not "
                                     "fit a %d-token window (user text is about %d "
                                     "tokens)\n" % (system_bytes, context_len, user_tokens))
                    sys.stderr.flush()
                prompt = user_prompt
                system_prompt = ""
                system_bytes = 0
                estimated_tokens = user_tokens
            else:
                detail = ""
                if system_prompt:
                    detail = (" Dropping the system prompt would not help either: the "
                              "user text alone is about %d tokens." % user_tokens)
                self._reject(
                    "prompt too long: about %d tokens for a %d-token context window "
                    "(%d bytes).%s Shorten the prompt, raise --context, or use a model "
                    "with a larger context." % (estimated_tokens, context_len,
                                                len(prompt.encode("utf-8")), detail))
                return

        # Generation shares the context window with the prompt. Trim the request
        # rather than letting the runtime overrun its KV cache: a caller asking
        # for 2048 new tokens on top of a 3000-token prompt gets a shorter answer
        # instead of a crash.
        if max_tokens and estimated_tokens + max_tokens > context_len:
            trimmed = max(1, context_len - estimated_tokens)
            if not self.server.options.quiet:
                sys.stderr.write("[rkllm] max_tokens trimmed %s -> %d "
                                 "(prompt is about %d of %d tokens)\n"
                                 % (max_tokens, trimmed, estimated_tokens, context_len))
            max_tokens = trimmed

        if not self.server.options.quiet:
            # Printed before inference starts, so if the runtime crashes the last
            # line in the log still says how big the request that killed it was.
            sys.stderr.write("[rkllm] inference: prompt ~%d tokens (%d bytes: request-system "
                             "%d, server-system %d, user %d), max_tokens %s, stream=%s, "
                             "history=%s\n"
                             % (estimated_tokens,
                                len(prompt.encode("utf-8")) + engine_system_bytes,
                                system_bytes, engine_system_bytes,
                                len(user_prompt.encode("utf-8")), max_tokens, stream,
                                engine.keep_history))
            sys.stderr.flush()

        # Counted only now: every guard above may still reject the request, and
        # counting a rejected request as served made /health read as "9 requests,
        # 4 of them rejected" for the same call.
        self.server.stats["requests"] += 1

        if tools:
            try:
                engine.set_function_tools(system_prompt, json.dumps(tools))
                if not getattr(self.server, "_warned_function_tools", False):
                    self.server._warned_function_tools = True
                    if not self.server.options.quiet:
                        sys.stderr.write(
                            "[rkllm] function calling enabled: this setting persists for "
                            "every later request on this handle - the runtime offers no "
                            "documented way to switch it off. Restart the service to "
                            "return to plain chat.\n")
                        sys.stderr.flush()
            except Exception as exc:
                self.server.stats["errors"] += 1
                self._send_error_json("failed to enable function calling: %s" % exc, 500,
                                      "server_error")
                return

        stop_seqs = stop or []
        # Kept for stream mode too: the stop-sequence scan and the token totals
        # both need the whole text, and max_out caps it at a few thousand tokens.
        pieces = []
        if stream:
            self._begin_stream()
            try:
                self.wfile.write(_openai_chunk(model_name, None, role="assistant").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                engine.abort()
                return
            sent_len = 0
            stopped_at_stop = False
            try:
                for fragment in engine.stream(prompt, role=role, enable_thinking=thinking,
                                              sampling=sampling, max_new_tokens=max_tokens):
                    pieces.append(fragment)
                    if stop_seqs:
                        # A stop sequence can straddle fragments, so look at the
                        # whole text generated so far. On a hit, flush only the
                        # part before the sequence and stop the NPU early.
                        joined = "".join(pieces)
                        cut = _stop_index(joined, stop_seqs)
                        if cut is not None:
                            if cut > sent_len:
                                self.wfile.write(_openai_chunk(
                                    model_name, joined[sent_len:cut]).encode("utf-8"))
                                self.wfile.flush()
                            stopped_at_stop = True
                            engine.abort()
                            break
                    self.wfile.write(_openai_chunk(model_name, fragment).encode("utf-8"))
                    self.wfile.flush()
                    sent_len += len(fragment)
            except (BrokenPipeError, ConnectionResetError):
                engine.abort()
                return
            except _Failure as exc:
                self.server.stats["errors"] += 1
                try:
                    self.wfile.write(("data: %s\n\n" % json.dumps(
                        {"error": {"message": str(exc), "type": "server_error"}})).encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            prompt_tokens, completion_tokens = _tokens_for(engine, prompt, "".join(pieces))
            finish = _finish_reason(stopped_at_stop, completion_tokens, max_tokens)
            try:
                self.wfile.write(_openai_chunk(model_name, "", finish_reason=finish).encode("utf-8"))
                if include_usage:
                    self.wfile.write(_openai_usage_chunk(
                        model_name, prompt_tokens, completion_tokens).encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            # 非流式请求在客户端断开后仍在跑满整段推理，白占 NPU 与 infer_lock
            # （下一次请求要等它跑完）。每 0.5 秒探一次，断开即 abort。
            # Prefill yields no fragments, so probing inside the loop misses the
            # whole prefill window - seconds of NPU time on a long prompt after
            # the client is already gone. A watchdog covers both phases.
            watch_stop = threading.Event()
            client_gone = threading.Event()

            def watch_client():
                while not watch_stop.wait(0.5):
                    if self._client_gone():
                        client_gone.set()
                        engine.abort()
                        return

            watcher = threading.Thread(target=watch_client, name="client-watch", daemon=True)
            watcher.start()
            try:
                for fragment in engine.stream(prompt, role=role, enable_thinking=thinking,
                                              sampling=sampling, max_new_tokens=max_tokens):
                    if client_gone.is_set():
                        self.close_connection = True
                        if not self.server.options.quiet:
                            sys.stderr.write(
                                "[rkllm] client disconnected during a non-streaming "
                                "request; inference aborted\n")
                            sys.stderr.flush()
                        return
                    pieces.append(fragment)
            except _Failure as exc:
                self.server.stats["errors"] += 1
                self._send_error_json(str(exc), 500, "server_error")
                return
            except RKLLMError as exc:
                self.server.stats["errors"] += 1
                self._send_error_json(str(exc), 500, "server_error")
                return
            finally:
                watch_stop.set()
            raw_answer = "".join(pieces)
            answer, stopped_at_stop = _apply_stop(raw_answer, stop_seqs)
            prompt_tokens, completion_tokens = _tokens_for(engine, prompt, raw_answer)
            self._send_json(_openai_response(
                model_name, answer,
                finish_reason=_finish_reason(stopped_at_stop, completion_tokens, max_tokens),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens))

        prompt_tokens, completion_tokens = _tokens_for(engine, prompt, "".join(pieces))
        self.server.stats["prompt_tokens"] += prompt_tokens
        self.server.stats["completion_tokens"] += completion_tokens


def cmd_serve(args):
    """Serve the OpenAI-compatible HTTP API in the foreground."""
    model_path = resolve_model(args.model, args.model_dir)
    engine = make_engine(args, model_path)
    state = lock_frequencies(args.platform, enabled=not args.no_freq_lock, quiet=args.quiet)

    server = Server((args.host, args.port), Handler, engine, args)
    server.timeout = 1.0

    stop = threading.Event()

    def shutdown(signum, frame):
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, shutdown)
        except (ValueError, OSError):
            pass

    probe_host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    sys.stderr.write("%s %s serving model '%s' on http://%s:%d\n"
                     % (PROGRAM, VERSION, engine.model_name, args.host, args.port))
    sys.stderr.write("  endpoints : GET /health, GET /v1/models, "
                     "POST /v1/chat/completions, POST /v1/reset\n")
    sys.stderr.write("  history   : %s   api key : %s\n"
                     % ("on" if not args.no_history else "off",
                        "set" if args.api_key else "none"))
    if not args.api_key and args.host not in ("127.0.0.1", "::1", "localhost"):
        # Open by default is a deliberate choice for a LAN tool, but an operator
        # who never set a key should not have to read the manual to find out that
        # every host on the network can now reset the conversation and burn NPU
        # time. This is a warning rather than a refusal: existing deployments
        # start this way and must keep working.
        sys.stderr.write("  WARNING   : no API key while listening on %s - any host that "
                         "can reach this port can use the model and POST /v1/reset.\n"
                         "              Set RKLLM_API_KEY in /etc/default/rkllm-server "
                         "(and chmod 600 that file) before exposing the board.\n"
                         % args.host)
    sys.stderr.write("  freq lock : %s\n" % state)
    sys.stderr.write("  status    : model loaded, listening. This is a foreground "
                     "service, so a quiet\n"
                     "              terminal from here on is expected.\n")
    sys.stderr.write("  try       : curl http://%s:%d/health        (Ctrl-C stops it)\n"
                     % (probe_host, args.port))
    sys.stderr.flush()

    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2})
    thread.daemon = True
    thread.start()
    try:
        while not stop.is_set():
            stop.wait(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stderr.write("\n[rkllm] shutting down\n")
        server.shutdown()
        server.server_close()
        engine.destroy()
        restore_frequencies(quiet=args.quiet)
    return 0


# ---------------------------------------------------------------------------
# list / info / selftest
# ---------------------------------------------------------------------------

def cmd_list(args):
    """List the .rkllm models found on this machine."""
    models = discover_models(args.model_dir)
    if args.json:
        sys.stdout.write(json.dumps(
            [{"name": n, "path": p, "size": s} for n, p, s in models], indent=2) + "\n")
        return 0
    dirs = model_search_dirs(args.model_dir)
    if not models:
        sys.stdout.write("no .rkllm model found. Directories searched:\n")
        for directory in dirs:
            sys.stdout.write("  %s%s\n" % (directory, "" if os.path.isdir(directory) else "  (missing)"))
        sys.stdout.write("\nCopy a converted model there, or pass an absolute path.\n")
        return 0
    width = max(len(n) for n, _, _ in models)
    for name, path, size in models:
        sys.stdout.write("%-*s  %10s  %s\n" % (width, name, human_size(size), path))
    return 0


def cmd_info(args):
    """Report environment, library and model information."""
    npu_available, npu_detail = npu_status()
    report = {"program": PROGRAM, "package_version": VERSION,
              "upstream_version": UPSTREAM_VERSION, "arch": ARCH,
              "triplet": TRIPLET, "python": sys.version.split()[0],
              "platform_detected": None, "platform_used": None,
              "library_path": None, "library_loaded": False, "library_error": None,
              "npu_available": npu_available, "npu_detail": npu_detail,
              "running_as_root": geteuid() == 0,
              "model_dirs": model_search_dirs(args.model_dir),
              "models_found": []}

    detected = detect_platform()
    report["platform_detected"] = detected
    report["platform_used"] = args.platform

    try:
        lib, path = load_library(args.lib)
        report["library_path"] = path
        report["library_loaded"] = True
    except RKLLMError as exc:
        report["library_error"] = str(exc)

    for name, path, size in discover_models(args.model_dir):
        report["models_found"].append({"name": name, "path": path, "size": size})

    if args.json:
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
        return 0

    def line(label, value):
        sys.stdout.write("%-20s: %s\n" % (label, value))

    line("package", "%s %s (upstream rknn-llm %s)" % (PACKAGE, VERSION, UPSTREAM_VERSION))
    line("arch", "%s (%s)" % (ARCH, TRIPLET))
    line("python", report["python"])
    line("SoC detected", detected or "unknown")
    line("frequency helper", FREQ_HELPERS.get(detected or args.platform, "n/a"))
    line("NPU driver", npu_detail if npu_available else "NOT FOUND - " + npu_detail)
    line("runtime library", report["library_path"] or "NOT LOADABLE")
    if report["library_error"]:
        line("  error", report["library_error"].replace("\n", " "))
    line("running as root", "yes" if report["running_as_root"] else "no")
    line("models found", str(len(report["models_found"])))
    for entry in report["models_found"]:
        sys.stdout.write("  %-24s %10s  %s\n" % (entry["name"], human_size(entry["size"]),
                                                 entry["path"]))
    sys.stdout.write("model search path  :\n")
    for directory in report["model_dirs"]:
        sys.stdout.write("  %s%s\n" % (directory, "" if os.path.isdir(directory) else "  (missing)"))
    if not npu_available:
        sys.stdout.write(
            "\nnote: no RKNPU driver was found, so no model can run on this machine.\n"
            "      The driver comes from your board image; this package deliberately\n"
            "      does not ship or build it. Note that drivers from 0.9.x onwards do\n"
            "      NOT create %s - they register the NPU as a DRM render node\n"
            "      (/dev/dri/renderD*), so test with `ls -l /sys/class/drm/renderD*/device/driver`\n"
            "      or `sudo cat %s/version` instead of looking for a device node.\n"
            % (LEGACY_NPU_DEVICE, NPU_DEBUGFS))
    return 0


def _selftest_result(name, status, detail=""):
    """status: True = pass, False = fail, None = skipped (needs real hardware)."""
    label = "SKIP" if status is None else ("PASS" if status else "FAIL")
    sys.stdout.write("[%s] %-42s %s\n" % (label, name, detail))
    return status


def cmd_selftest(args):
    """Exercise everything that does not need a model, a driver or an NPU."""
    results = []

    # 1. the script itself is syntactically loaded and knows its own version.
    #    An unrendered template still carries the literal placeholder, which
    #    always starts with "@" - that is what we detect here.
    results.append(_selftest_result("cli loaded",
                                    bool(VERSION) and not VERSION.startswith("@"),
                                    "version %s" % VERSION))

    # 2. library discovery. Missing hardware makes this a skip, but a library
    #    the user pointed at explicitly and that still fails is a hard error.
    try:
        lib, path = load_library(args.lib)
        results.append(_selftest_result("librkllmrt.so loadable", True, path))
    except RKLLMError as exc:
        forced = bool(args.lib or os.environ.get("RKLLM_LIB"))
        results.append(_selftest_result("librkllmrt.so loadable", False if forced else None,
                                        str(exc).splitlines()[0]))

    # 3. the RKNPU driver, via whichever signal the kernel exposes. Not a device
    #    node check: 0.9.x drivers register the NPU as a DRM render node and
    #    never create /dev/rknpu.
    npu_available, npu_detail = npu_status()
    results.append(_selftest_result(
        "npu driver reachable", True if npu_available else None,
        npu_detail if npu_available
        else "not found here; expected off-board, required on the target"))

    # 3. model name resolution refuses to invent a path
    try:
        resolve_model("definitely-not-a-model", [os.getcwd()])
        results.append(_selftest_result("model resolution", False, "accepted a bogus name"))
    except RKLLMError:
        results.append(_selftest_result("model resolution", True, "bogus names rejected"))

    # 4. engine + HTTP surface against the fake engine
    engine = FakeEngine(model_name="selftest")
    options = argparse.Namespace(max_queue=4, queue_timeout=5.0, api_key=None, quiet=True)
    server = Server(("127.0.0.1", 0), Handler, engine, options)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1})
    thread.daemon = True
    thread.start()

    import http.client as http_client
    import urllib.error
    import urllib.request

    def http(method, path, payload=None, timeout=30, auth=None):
        url = "http://127.0.0.1:%d%s" % (port, path)
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if auth is not None:
            request.add_header("Authorization", auth)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # 4xx/5xx are part of what we assert on, not an error here
            return exc.code, exc.read().decode("utf-8", "replace")

    try:
        status, body = http("GET", "/v1/models")
        payload = json.loads(body)
        results.append(_selftest_result("GET /v1/models", status == 200
                                        and payload["data"][0]["id"] == "selftest",
                                        "status %s" % status))

        # clients disagree about whether "/v1" belongs in their base_url
        status_alias, body_alias = http("GET", "/models")
        status_alias_post, _ = http("POST", "/chat/completions",
                                    {"messages": [{"role": "user", "content": "x"}]})
        results.append(_selftest_result("works with and without /v1",
                                        status_alias == 200 and status_alias_post == 200,
                                        "GET /models %s, POST /chat/completions %s"
                                        % (status_alias, status_alias_post)))

        status, body = http("GET", "/health")
        results.append(_selftest_result("GET /health", status == 200
                                        and json.loads(body)["status"] == "ok",
                                        "status %s" % status))

        status, body = http("POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "ping"}]})
        payload = json.loads(body)
        text = payload["choices"][0]["message"]["content"]
        results.append(_selftest_result("POST /v1/chat/completions",
                                        status == 200 and "ping" in text,
                                        "%r" % text[:40]))

        # A response whose usage block is all zeros makes chat UIs display
        # "0 tokens"; the counts must at least be internally consistent.
        usage = payload.get("usage") or {}
        counted = usage.get("completion_tokens", 0)
        results.append(_selftest_result(
            "usage reported, not zeros",
            counted > 0 and usage.get("total_tokens") ==
            usage.get("prompt_tokens", 0) + counted,
            "prompt=%s completion=%s total=%s"
            % (usage.get("prompt_tokens"), counted, usage.get("total_tokens"))))

        # Chinese has no spaces, so a word-based count reports the entire
        # question as a single token. This case exists because the old
        # len(text.split()) estimator passed every other usage check while being
        # wrong by an order of magnitude on Chinese input.
        chinese = "请介绍一下杭州这座城市的历史与风景" * 3
        status, body = http("POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": chinese}],
                             "max_tokens": 8})
        reported = json.loads(body)["usage"]["prompt_tokens"]
        expected = len(chinese.encode("utf-8")) // 3
        results.append(_selftest_result(
            "token estimate handles space-free text",
            reported >= expected * 0.8,
            "Chinese prompt (%d bytes, expect ~%d) -> %s"
            % (len(chinese.encode("utf-8")), expected, reported)))

        # A request-level system message has to reach the model as prompt text,
        # never through rkllm_set_chat_template (which disables the model's
        # built-in template and makes it ramble). The fake engine echoes its
        # prompt, so the marker proves where the text ended up.
        status, body = http("POST", "/v1/chat/completions",
                            {"messages": [{"role": "system", "content": "SYS-MARKER"},
                                          {"role": "user", "content": "ping"}]})
        echo = json.loads(body)["choices"][0]["message"]["content"]
        results.append(_selftest_result("system message reaches the model",
                                        "SYS-MARKER" in echo,
                                        "%r" % echo[:46]))

        # Clients probe a freshly configured model with a system-only request;
        # that has to be answered rather than refused with a 400 that reads as
        # "the model is broken".
        status, body = http("POST", "/v1/chat/completions",
                            {"messages": [{"role": "system", "content": "SYS-ONLY-MARKER"}],
                             "max_tokens": 16})
        echo = (json.loads(body)["choices"][0]["message"]["content"]
                if status == 200 else body[:70])
        results.append(_selftest_result(
            "system-only request answered",
            status == 200 and "SYS-ONLY-MARKER" in echo
            and echo.count("SYS-ONLY-MARKER") == 1,
            "%s  %r" % (status, echo[:46])))

        status, body = http("POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "stream"}], "stream": True})
        chunks = [line[len("data: "):] for line in body.splitlines()
                  if line.startswith("data: ")]
        done = chunks and chunks[-1] == "[DONE]"
        assembled = "".join(json.loads(chunk)["choices"][0]["delta"].get("content", "")
                            for chunk in chunks[:-1] if chunk != "[DONE]")
        results.append(_selftest_result("streaming SSE", status == 200 and done
                                        and "stream" in assembled,
                                        "%d chunks" % len(chunks)))

        status, body = http("POST", "/v1/chat/completions", {"messages": []})
        results.append(_selftest_result("invalid request rejected", status == 400,
                                        "status %s" % status))

        # A chunked body carries no Content-Length. It must be refused, not read
        # as an empty body: undecoded chunk frames would stay on the socket and
        # be parsed as the next request on this keep-alive connection.
        try:
            conn = http_client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.putrequest("POST", "/v1/chat/completions")
            conn.putheader("Transfer-Encoding", "chunked")
            conn.putheader("Content-Type", "application/json")
            conn.endheaders()
            raw = b'{"messages":[{"role":"user","content":"x"}]}'
            conn.send(b"%x\r\n%s\r\n0\r\n\r\n" % (len(raw), raw))
            response = conn.getresponse()
            chunked_status = response.status
            response.read()
            conn.close()
        except Exception as exc:                              # pragma: no cover
            chunked_status = repr(exc)
        results.append(_selftest_result("chunked body refused", chunked_status == 411,
                                        "status %s" % chunked_status))

        # The handle-level function-tools flag is permanent, so /health has to
        # expose it: otherwise its effect on later requests is invisible.
        status, body = http("GET", "/health")
        results.append(_selftest_result(
            "health reports function_tools", status == 200
            and "function_tools" in json.loads(body),
            "status %s" % status))

        # Two back-pressure exits (queue full, NPU wait timeout) must speak the
        # same dialect: a bare 503 tells a client "retry later" without saying
        # when. Emptying the semaphore makes the queue-full branch deterministic
        # instead of racing real concurrency.
        saved_slots = server.queue_slots
        server.queue_slots = threading.BoundedSemaphore(0)
        try:
            probe = http_client.HTTPConnection("127.0.0.1", port, timeout=10)
            probe.request("POST", "/v1/chat/completions",
                          body=json.dumps({"messages": [{"role": "user", "content": "x"}]}),
                          headers={"Content-Type": "application/json"})
            reply = probe.getresponse()
            queue_status, retry_after = reply.status, reply.getheader("Retry-After")
            reply.read()
            probe.close()
        finally:
            server.queue_slots = saved_slots
        results.append(_selftest_result(
            "queue-full 503 carries Retry-After",
            queue_status == 503 and retry_after == "5",
            "status %s, Retry-After %r" % (queue_status, retry_after)))

        # A non-streaming request has no writes, so a client that walks away is
        # invisible until the whole generation finishes - it keeps the NPU and
        # infer_lock busy for nothing. The probe must tell three states apart:
        # idle, data waiting (a pipelined request - not a disconnect), and EOF.
        left, right = socket.socketpair()
        try:
            probe_handler = Handler.__new__(Handler)      # skip __init__: it makes a connection
            probe_handler.connection = left
            idle = probe_handler._client_gone()
            right.sendall(b"GET / HTTP/1.1\r\n")          # bytes still queued
            with_data = probe_handler._client_gone()
        finally:
            right.close()
            left.close()

        # EOF needs a connection with nothing left to read: MSG_PEEK can only see
        # the queued bytes, never the FIN behind them.
        left2, right2 = socket.socketpair()
        try:
            probe2 = Handler.__new__(Handler)
            probe2.connection = left2
            right2.close()                                # peer gone -> FIN
            deadline = time.monotonic() + 1.0             # FIN 到达是异步的
            eof = False
            while time.monotonic() < deadline:
                eof = probe2._client_gone()
                if eof:
                    break
                time.sleep(0.02)
        finally:
            left2.close()
        results.append(_selftest_result(
            "non-streaming disconnect probe",
            idle is False and with_data is False and eof is True,
            "idle=%s with_data=%s eof=%s" % (idle, with_data, eof)))

        # An oversized prompt must be refused here, not handed to the native
        # runtime: a 32 KB English prompt is about 8000 tokens against a 4096
        # token window, and the runtime segfaults when it overruns the KV cache.
        # The guard allows 3 bytes per token, so this is just past the line.
        oversized = "x" * (engine.context_len * 3 + 1)
        status, body = http("POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": oversized}]})
        results.append(_selftest_result("oversized prompt refused", status == 400,
                                        "status %s (%d bytes)"
                                        % (status, len(oversized))))

        # Prompt and generation share the window, so an over-large max_tokens has
        # to be trimmed rather than overrun it. This prompt takes roughly two
        # thirds of the window, leaving room for about a third of the request.
        status, body = http("POST", "/v1/chat/completions",
                            {"messages": [{"role": "user",
                                           "content": "y" * (engine.context_len * 2)}],
                             "max_tokens": engine.context_len})
        results.append(_selftest_result(
            "oversized max_tokens trimmed", status == 200,
            "status %s for a ~%d-token prompt asking for %d new tokens"
            % (status, engine.context_len * 2 // 3, engine.context_len)))

        # A chat client's own system prompt can be far larger than the window -
        # an agent UI sent 57 KB of preamble with a two-character greeting. The
        # request must still be served, by dropping the preamble, rather than
        # failing with a 400 the user cannot act on.
        status, body = http("POST", "/v1/chat/completions",
                            {"messages": [
                                {"role": "system", "content": "S" * (engine.context_len * 6)},
                                {"role": "user", "content": "hello"}],
                             "max_tokens": 16})
        echo = (json.loads(body)["choices"][0]["message"]["content"]
                if status == 200 else body[:70])
        results.append(_selftest_result(
            "oversized system prompt dropped", status == 200 and "hello" in echo,
            "%s  %r  (preamble was %d bytes, window %d tokens)"
            % (status, echo[:40], engine.context_len * 6, engine.context_len)))

        status, body = http("POST", "/v1/reset")
        results.append(_selftest_result("POST /v1/reset", status == 200, "status %s" % status))

        status, body = http("GET", "/health")
        stats = json.loads(body)["stats"]
        results.append(_selftest_result("request accounting", stats["requests"] >= 2,
                                        "%d served, %d rejected" % (stats["requests"],
                                                                    stats["rejected"])))

        # 5. queue behaviour: one request runs, one may wait, the rest get 503
        gate = _GateEngine()
        server.engine = gate
        server.options.max_queue = 1
        server.queue_slots = threading.BoundedSemaphore(2)   # 1 running + 1 waiting
        statuses = {}

        def call(tag):
            try:
                status, _ = http("POST", "/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "q"}]}, timeout=30)
                statuses[tag] = status
            except Exception as exc:                              # pragma: no cover
                statuses[tag] = repr(exc)

        running = threading.Thread(target=call, args=("running",))
        running.start()
        gate.started.wait(10)               # the first request is inside the NPU call
        waiting = threading.Thread(target=call, args=("waiting",))
        waiting.start()
        time.sleep(0.5)                     # let it take the single queue slot
        overflow = threading.Thread(target=call, args=("overflow",))
        overflow.start()
        overflow.join(20)
        gate.release.set()                  # let the parked inference finish
        running.join(20)
        waiting.join(20)
        results.append(_selftest_result(
            "queue: 1 runs, 1 waits, extra 503",
            statuses.get("running") == 200 and statuses.get("waiting") == 200
            and statuses.get("overflow") == 503, "%s" % statuses))

        # 7. API key authentication
        server.options.api_key = "sekret"
        no_key, _ = http("GET", "/health")
        good_key, _ = http("GET", "/health", auth="Bearer sekret")
        bad_key, _ = http("GET", "/health", auth="Bearer wrong")
        server.options.api_key = None
        results.append(_selftest_result("api key auth",
                                        no_key == 401 and good_key == 200 and bad_key == 401,
                                        "missing %s / correct %s / wrong %s"
                                        % (no_key, good_key, bad_key)))

        # 8. The C API must never receive NULL where it dereferences a pointer.
        #    Passing NULL for the chat template prefix/postfix segfaults the
        #    runtime, which is exactly what any client sending a system message
        #    used to trigger. Verified against a recording stub, so it needs no
        #    model and no library.
        class _RecordingLib(object):
            def __init__(self):
                self.calls = []

            def rkllm_set_chat_template(self, handle, system_prompt, prefix, postfix):
                self.calls.append((system_prompt, prefix, postfix))
                return 0

            def rkllm_set_function_tools(self, handle, system_prompt, tools, tool_response):
                self.calls.append((system_prompt, tools, tool_response))
                return 0

        probe = object.__new__(RealEngine)        # bypass __init__: no model needed
        probe._handle = ctypes.c_void_p(1)
        probe.lib = _RecordingLib()
        probe._keep = []                          # __init__ normally creates this
        probe.set_system_prompt("be nice")
        probe.set_function_tools("be nice", "[]")
        nulls = [call for call in probe.lib.calls if any(arg is None for arg in call)]
        results.append(_selftest_result(
            "no NULL passed to the C API", not nulls,
            "%d call(s) checked, %d with NULL" % (len(probe.lib.calls), len(nulls))))

        # 9. REPL driving, same code path as `rkllm chat`
        import io
        saved_stdin, saved_stdout, saved_stderr = sys.stdin, sys.stdout, sys.stderr
        sys.stdin = io.StringIO("hello repl\n/clear\n/set temperature 0.5\n/exit\n")
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        chat_args = argparse.Namespace(
            model="no-such-model", model_dir=[os.getcwd()], platform=None, no_freq_lock=True,
            quiet=True, context=512, max_new_tokens=32, no_history=True, threads=None,
            system=None, lib=None, verbose=False, fake_engine=True, top_k=1, top_p=0.9,
            temperature=0.8, repeat_penalty=1.1)
        try:
            # bypass model resolution: the fake engine ignores the path
            original_resolve = globals()["resolve_model"]
            globals()["resolve_model"] = lambda name, extra=(): "<fake>"
            try:
                cmd_chat(chat_args)
            finally:
                globals()["resolve_model"] = original_resolve
            repl_output = sys.stdout.getvalue()
        finally:
            sys.stdin, sys.stdout, sys.stderr = saved_stdin, saved_stdout, saved_stderr
        results.append(_selftest_result("REPL chat path", "hello repl" in repl_output
                                        and "temperature = 0.5" in repl_output,
                                        "%d bytes of transcript" % len(repl_output)))

        # 10. An unset systemd variable arrives as an empty argument. That must
        #     not be treated as an unrecognised positional: the unit then fails
        #     with exit code 2 and `systemctl status` shows nothing but
        #     "status=2", which is very hard to diagnose.
        buffer = io.StringIO()
        saved_out, saved_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = buffer
        try:
            # only empty arguments: after filtering, this is "no command". Without
            # the filter argparse sees two empty positionals and calls sys.exit().
            try:
                argv_result = main(["", ""])
            except SystemExit as exc:
                argv_result = "SystemExit(%s)" % exc.code
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err
        noise = buffer.getvalue()
        complained = "unrecognized arguments" in noise
        results.append(_selftest_result(
            "empty arguments tolerated", argv_result == 2 and not complained,
            "exit %s%s" % (argv_result,
                           " - rejected them as unrecognised" if complained else "")))

        # 11. Every serve option can be preset through the environment, which is
        #     how /etc/default/rkllm-server drives the service. A value that does
        #     not parse must be reported and ignored rather than aborting the
        #     parser, which would leave the journal with nothing but "status=2".
        env_names = ("RKLLM_TEMPERATURE", "RKLLM_MAX_QUEUE", "RKLLM_QUEUE_TIMEOUT",
                     "RKLLM_NO_HISTORY")
        saved_env = {key: os.environ.get(key) for key in env_names}
        buffer = io.StringIO()
        saved_err = sys.stderr
        sys.stderr = buffer
        try:
            os.environ["RKLLM_TEMPERATURE"] = "0.25"
            os.environ["RKLLM_MAX_QUEUE"] = "7"
            os.environ["RKLLM_QUEUE_TIMEOUT"] = "not-a-number"
            os.environ["RKLLM_NO_HISTORY"] = "1"
            preset = build_parser().parse_args(["serve", "selftest-model"])
        finally:
            sys.stderr = saved_err
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        env_ok = (preset.temperature == 0.25
                  and preset.max_queue == 7
                  and preset.queue_timeout == DEFAULT_QUEUE_TIMEOUT
                  and preset.no_history is True
                  and "ignoring RKLLM_QUEUE_TIMEOUT" in buffer.getvalue())
        results.append(_selftest_result(
            "serve options accept environment defaults", env_ok,
            "temperature=%s max_queue=%s queue_timeout=%s no_history=%s%s"
            % (preset.temperature, preset.max_queue, preset.queue_timeout,
               preset.no_history,
               "" if env_ok else " (bad value not reported: %r)" % buffer.getvalue()[:80])))
    except SystemExit as exc:
        # A check that triggers sys.exit() would otherwise abort the whole run
        # without printing the summary, hiding every later result.
        sys.stdout.write("\n[FAIL] a check called sys.exit(%s); selftest aborted\n" % exc.code)
        results.append(False)
    except Exception as exc:                                      # noqa: BLE001
        sys.stdout.write("\n[FAIL] selftest aborted: %r\n" % exc)
        results.append(False)
    finally:
        server.shutdown()
        server.server_close()
        engine.destroy()
        thread.join()

    passed = sum(1 for entry in results if entry is True)
    failed = sum(1 for entry in results if entry is False)
    skipped = sum(1 for entry in results if entry is None)
    sys.stdout.write("\nselftest: %d passed, %d failed, %d skipped\n"
                     % (passed, failed, skipped))
    if skipped:
        sys.stdout.write("Skipped checks need the real librkllmrt.so and the rknpu driver, "
                         "so they only pass on the board itself.\n")
    if failed:
        sys.stdout.write("The failures above mean this installation is broken on this "
                         "machine.\n")
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def option_default(name, kind, fallback, enabled=True):
    """Default for a serve option: $RKLLM_<NAME>, then `fallback`.

    The systemd unit loads /etc/default/rkllm-server through EnvironmentFile=,
    so this is what makes every serve parameter configurable in that file
    without repeating it on the command line. A flag given on the command line
    still wins, because argparse only consults the default when the option is
    absent.

    A value that does not parse is reported and ignored: a typo in the config
    file must not make the service die with an argparse error, which would show
    up in the journal as nothing but "status=2".
    """
    raw = os.environ.get("RKLLM_" + name) if enabled else None
    if raw is None or not raw.strip():
        return fallback
    try:
        return kind(raw)
    except (TypeError, ValueError):
        sys.stderr.write("[rkllm] ignoring RKLLM_%s=%r: not a valid %s, using %r\n"
                         % (name, raw, kind.__name__, fallback))
        return fallback


def option_flag(name, enabled=True, fallback=False):
    """Boolean form of option_default() for the --no-* style switches."""
    raw = os.environ.get("RKLLM_" + name) if enabled else None
    if raw is None or not raw.strip():
        return fallback
    return raw.strip().lower() in ("1", "true", "yes", "on")


def add_common_arguments(parser, env=False):
    """Options shared by chat/run/serve.

    With `env` set they also accept $RKLLM_<NAME>, which is how the service
    unit's config file drives them. Only `serve` opts in: an interactive shell
    that happens to export RKLLM_TEMPERATURE should not silently change chat.
    """
    parser.add_argument("--context", type=int,
                        default=option_default("CONTEXT", int, DEFAULT_CONTEXT, env),
                        help="max context length in tokens (default %d)" % DEFAULT_CONTEXT)
    parser.add_argument("--max-new-tokens", type=int,
                        default=option_default("MAX_NEW_TOKENS", int,
                                               DEFAULT_MAX_NEW_TOKENS, env),
                        help="max tokens to generate per answer (default %d)"
                             % DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--threads", type=int,
                        default=option_default("THREADS", int, None, env),
                        help="CPU cores used for inference (default 4)")
    parser.add_argument("--platform", default=None, choices=list(PLATFORMS),
                        help="override SoC detection (rk3588/rk3576/rk3562/rv1126b)")
    parser.add_argument("--model-dir", action="append", default=[],
                        help="extra directory to search for models (repeatable; "
                             "$RKLLM_MODEL_DIR is a pathsep-separated list too)")
    parser.add_argument("--lib", default=None,
                        help="path to librkllmrt.so ($RKLLM_LIB works too)")
    parser.add_argument("--no-freq-lock", action="store_true",
                        default=option_flag("NO_FREQ_LOCK", env),
                        help="do not raise CPU/NPU/DDR clocks before running")
    parser.add_argument("--quiet", "-q", action="store_true",
                        default=option_flag("QUIET", env),
                        help="less chatter on stderr")
    parser.add_argument("--verbose", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fake-engine", action="store_true", help=argparse.SUPPRESS)


def add_sampling_arguments(parser, env=False):
    """Sampling options, shared by chat/run/serve (`env` as in add_common_arguments)."""
    parser.add_argument("--top-k", dest="top_k", type=int,
                        default=option_default("TOP_K", int, 1, env),
                        help="sample from the k most likely tokens (1 = greedy)")
    parser.add_argument("--top-p", dest="top_p", type=float,
                        default=option_default("TOP_P", float, 0.9, env),
                        help="nucleus sampling: shortest token list whose probabilities "
                             "sum to this value")
    parser.add_argument("--temperature", type=float,
                        default=option_default("TEMPERATURE", float, 0.8, env),
                        help="sampling temperature; lower is more focused, 0 is greedy")
    parser.add_argument("--repeat-penalty", dest="repeat_penalty", type=float,
                        default=option_default("REPEAT_PENALTY", float, 1.1, env),
                        help="penalty for repeating tokens (1.0 disables it)")


def build_parser():
    """Build the argument parser: shared options plus subcommands."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Run .rkllm models locally through the RKLLM runtime "
                    "(RK3588 / RK3576 / RK3562 / RV1126B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  rkllm chat qwen2.5-1.5b                 interactive local chat
  rkllm chat /data/models/qwen.rkllm      same, by explicit path
  rkllm run  qwen2.5-1.5b "hello"         one-shot answer
  rkllm serve qwen2.5-1.5b --port 8080    OpenAI-compatible API
  rkllm list                              which models can be found
  rkllm selftest                          verify the installation
""")
    parser.add_argument("--version", action="version",
                        version="%s %s (upstream rknn-llm %s, %s)"
                                % (PROGRAM, VERSION, UPSTREAM_VERSION, ARCH))
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser("chat", help="interactive local chat")
    chat.add_argument("model", help="model name or path to a .rkllm file")
    chat.add_argument("--system", default=None, help="system prompt")
    chat.add_argument("--no-history", action="store_true",
                      help="do not keep the conversation context")
    add_common_arguments(chat)
    add_sampling_arguments(chat)
    chat.set_defaults(func=cmd_chat)

    run = subparsers.add_parser("run", help="one-shot inference")
    run.add_argument("model", help="model name or path to a .rkllm file")
    run.add_argument("text", nargs="+", help="prompt text")
    run.add_argument("--system", default=None, help="system prompt")
    run.add_argument("--no-history", action="store_true", help=argparse.SUPPRESS)
    add_common_arguments(run)
    add_sampling_arguments(run)
    run.set_defaults(func=cmd_run)

    serve = subparsers.add_parser("serve", help="OpenAI-compatible HTTP API")
    serve.add_argument("model", help="model name or path to a .rkllm file")
    serve.add_argument("--host", default=DEFAULT_HOST, help="bind address (default %s)" % DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT,
                       help="TCP port (default %d)" % DEFAULT_PORT)
    serve.add_argument("--api-key", dest="api_key",
                       default=os.environ.get("RKLLM_API_KEY") or None,
                       help="require 'Authorization: Bearer <key>'; falls back to "
                            "$RKLLM_API_KEY. Prefer the environment variable: a value "
                            "given on the command line is visible in `ps`.")
    # serve is the one subcommand a service runs, so every option below can be
    # preset from the environment (see option_default): /etc/default/rkllm-server
    # reaches the process through EnvironmentFile= and needs no command line.
    serve.add_argument("--max-queue", dest="max_queue", type=int,
                       default=option_default("MAX_QUEUE", int, DEFAULT_MAX_QUEUE, True),
                       help="requests allowed to wait (default %d)" % DEFAULT_MAX_QUEUE)
    serve.add_argument("--queue-timeout", dest="queue_timeout", type=float,
                       default=option_default("QUEUE_TIMEOUT", float,
                                              DEFAULT_QUEUE_TIMEOUT, True),
                       help="seconds a request may wait for the NPU (default %.0f)"
                            % DEFAULT_QUEUE_TIMEOUT)
    serve.add_argument("--system", default=os.environ.get("RKLLM_SYSTEM") or None,
                       help="system prompt (also $RKLLM_SYSTEM)")
    serve.add_argument("--no-history", action="store_true",
                       default=option_flag("NO_HISTORY", True, fallback=True),
                       help="answer every request independently (default: the client "
                            "resends the conversation). Set RKLLM_NO_HISTORY=0 to let "
                            "the server carry history between requests instead")
    add_common_arguments(serve, env=True)
    add_sampling_arguments(serve, env=True)
    serve.set_defaults(func=cmd_serve)

    listing = subparsers.add_parser("list", help="list discoverable models")
    listing.add_argument("--model-dir", action="append", default=[],
                        help="extra directory to search for models (repeatable)")
    listing.add_argument("--json", action="store_true",
                        help="print machine-readable JSON instead of text")
    listing.set_defaults(func=cmd_list)

    info = subparsers.add_parser("info", help="environment and driver report")
    info.add_argument("--model-dir", action="append", default=[],
                        help="extra directory to search for models (repeatable)")
    info.add_argument("--lib", default=None, help="path to librkllmrt.so")
    info.add_argument("--json", action="store_true",
                        help="print machine-readable JSON instead of text")
    info.set_defaults(func=cmd_info)

    selftest = subparsers.add_parser("selftest", help="verify the installation")
    selftest.add_argument("--model-dir", action="append", default=[],
                        help="extra directory to search for models (repeatable)")
    selftest.add_argument("--lib", default=None, help="path to librkllmrt.so")
    selftest.set_defaults(func=cmd_selftest)

    return parser


def main(argv=None):
    """Entry point: parse arguments, dispatch, return an exit code."""
    if argv is None:
        argv = sys.argv[1:]
    # systemd expands every unset variable into an EMPTY ARGUMENT, and argparse
    # would reject those as unrecognised positionals - the unit then failed with
    # exit code 2 and `systemctl status` showed only "status=2". Empty arguments
    # carry no meaning for this tool, so drop them.
    argv = [item for item in argv if item != ""]

    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # to stderr, so the reason ends up in the journal rather than nowhere
        parser.print_help(sys.stderr)
        sys.stderr.write("\n%s: no command given; see --help\n" % PROGRAM)
        return 2

    # SoC detection is shared by every subcommand that can touch hardware.
    if hasattr(args, "platform") and not args.platform:
        args.platform = detect_platform()
    for attribute, default in (("quiet", False), ("verbose", False), ("json", False),
                               ("fake_engine", False), ("no_freq_lock", False),
                               ("model_dir", []), ("lib", None), ("system", None),
                               ("platform", None)):
        if not hasattr(args, attribute):
            setattr(args, attribute, default)

    try:
        return args.func(args)
    except RKLLMError as exc:
        sys.stderr.write("%s: error: %s\n" % (PROGRAM, exc))
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130
    except Exception:
        sys.stderr.write("%s: unexpected failure\n" % PROGRAM)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
