"""ctypes ABI 镜像 + 上游库加载（迁移自旧 cli/rkllm.py，修 D10）。

改造点（对照旧实现）：
- 库加载时记录 sha256 与文件大小，供 /health 与启动日志暴露（D10：旧版只校验
  符号不校验版本，结构体布局漂移会变成静默内存破坏）；
- 保留 ctypes 保活纪律：_keep / _run_keep 两级缓冲列表；
- 签名缺失时显式失败（沿用旧版 fail-loudly 语义）。

仅限板端使用；客户端连接器（源码交付）不导入本模块。
"""

from __future__ import annotations

import ctypes
import hashlib
import os

__all__ = ["RKLLMError", "load_library", "bind_library",
           "describe_rc", "struct_sizes", "RETURN_CODES",
           "MODEL_SUFFIX", "DEFAULT_CONTEXT", "DEFAULT_MAX_NEW_TOKENS",
           "CPU_MASK_LOW", "CPU_MASK_HIGH", "BIG_CORE_PLATFORMS",
           "RKLLM_RUN_NORMAL", "RKLLM_RUN_WAITING", "RKLLM_RUN_FINISH",
           "RKLLM_RUN_ERROR", "RKLLM_INPUT_PROMPT", "RKLLM_INFER_GENERATE",
           "_Param", "_InferParam", "_Input", "_Callback", "_Result",
           "_ResultCallback", "_PerfStat", "_SamplingParam",
           "_PromptCacheParam", "_ExtendParam", "_LoraParam"]

MODEL_SUFFIX = ".rkllm"
DEFAULT_CONTEXT = 4096
DEFAULT_MAX_NEW_TOKENS = 2048

RKLLM_RUN_NORMAL = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH = 2
RKLLM_RUN_ERROR = 3
RKLLM_INPUT_PROMPT = 0
RKLLM_INFER_GENERATE = 0

CPU_MASK_LOW = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
CPU_MASK_HIGH = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
# 大核位于第 4-7 号 CPU 的 SoC。RK3576 是 4×A72(4-7)+4×A53(0-3)，官方 demo
# （examples/rkllm_server_demo/rkllm_server/flask_server.py）对 rk3576/rk3588
# 都用高掩码；此前只列 rk3588 会让 RK3576 落到 A53 小核，prefill/解码慢 3-4 倍
# （2026-10-27 审核 P1-2）。rk3562 是 4×A53、rv1126b 是 4×A7，保持低掩码。
BIG_CORE_PLATFORMS = frozenset({"rk3588", "rk3576"})

# rkllm 返回码 → 人话（2026-11-20 审计：此前只把数字拼进错误文案，负值与内部码
# 只能靠注释推测）。**只登记有实测/代码依据的码**；未登记的码由 describe_rc
# 原样回显数字，绝不猜原因。
RETURN_CODES: dict[int, str] = {
    0: "成功",
    -1: "runtime 内部失败（已实测：prompt+max_new_tokens 超出上下文；"
        "cache 文件加载失败同为此码，见 engine 的熔断重试）",
    10000: "prompt cache 加载失败（内部码，2026-10-22 实测；promptcache.disable 熔断）",
}


def describe_rc(code: int) -> str:
    """返回码的可读解释；未知码回显数字，绝不猜原因。"""
    reason = RETURN_CODES.get(code)
    return "code %d（%s）" % (code, reason) if reason else "code %d" % code


class RKLLMError(Exception):
    pass


# ---------------------------------------------------------------------------
# 结构体镜像（/usr/include/rkllm/rkllm.h，上游 1.3.1）
# ---------------------------------------------------------------------------


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
    """`rkllm_load_lora` 的入参：adapter 文件路径 + 名字 + 缩放系数。

    与 `_LoraParam`（推理时按名字选中已加载的 adapter）是两件事：先用本结构
    把 adapter 挂到 handle 上，再在 infer_params 里指定用哪一个。
    """
    _fields_ = [("lora_adapter_path", ctypes.c_char_p),
                ("lora_adapter_name", ctypes.c_char_p),
                ("scale", ctypes.c_float)]


class _LoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class _PromptCacheParam(ctypes.Structure):
    _fields_ = [("save_prompt_cache", ctypes.c_int),
                ("prompt_cache_path", ctypes.c_char_p)]


class _SamplingParam(ctypes.Structure):
    _fields_ = [("top_k", ctypes.c_int32), ("top_p", ctypes.c_float),
                ("temperature", ctypes.c_float),
                ("repeat_penalty", ctypes.c_float),
                ("frequency_penalty", ctypes.c_float),
                ("presence_penalty", ctypes.c_float),
                ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float),
                ("mirostat_eta", ctypes.c_float)]


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
    _fields_ = [("embed", ctypes.POINTER(ctypes.c_float)),
                ("n_tokens", ctypes.c_size_t)]


class _TokenInput(ctypes.Structure):
    _fields_ = [("input_ids", ctypes.POINTER(ctypes.c_int32)),
                ("n_tokens", ctypes.c_size_t)]


class _ImageInput(ctypes.Structure):
    _fields_ = [("image_embed", ctypes.POINTER(ctypes.c_float)),
                ("n_image_tokens", ctypes.c_size_t), ("n_image", ctypes.c_size_t),
                ("image_start", ctypes.c_char_p), ("image_end", ctypes.c_char_p),
                ("image_content", ctypes.c_char_p),
                ("image_width", ctypes.c_size_t), ("image_height", ctypes.c_size_t)]


class _VideoInput(ctypes.Structure):
    _fields_ = [("video_embed", ctypes.POINTER(ctypes.c_float)),
                ("n_frame_tokens", ctypes.c_size_t),
                ("n_frame_per_video", ctypes.c_size_t), ("n_video", ctypes.c_size_t),
                ("video_start", ctypes.c_char_p), ("video_end", ctypes.c_char_p),
                ("video_content", ctypes.c_char_p),
                ("frame_width", ctypes.c_size_t), ("frame_height", ctypes.c_size_t)]


class _AudioInput(ctypes.Structure):
    _fields_ = [("audio_embed", ctypes.POINTER(ctypes.c_float)),
                ("n_audio_tokens", ctypes.c_size_t), ("n_audio", ctypes.c_size_t),
                ("audio_start", ctypes.c_char_p), ("audio_end", ctypes.c_char_p),
                ("audio_content", ctypes.c_char_p)]


class _MultiModalInput(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p), ("image", _ImageInput),
                ("video", _VideoInput), ("audio", _AudioInput)]


class _InputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p), ("embed_input", _EmbedInput),
                ("token_input", _TokenInput), ("multimodal_input", _MultiModalInput)]


class _Input(ctypes.Structure):
    _anonymous_ = ("input_data",)
    _fields_ = [("role", ctypes.c_char_p), ("enable_thinking", ctypes.c_bool),
                ("input_type", ctypes.c_int), ("input_data", _InputUnion)]


class _HiddenLayer(ctypes.Structure):
    _fields_ = [("hidden_states", ctypes.POINTER(ctypes.c_float)),
                ("embd_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]


class _Logits(ctypes.Structure):
    _fields_ = [("logits", ctypes.POINTER(ctypes.c_float)),
                ("vocab_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]


class _PerfStat(ctypes.Structure):
    _fields_ = [("prefill_time_ms", ctypes.c_float),
                ("prefill_tokens", ctypes.c_int),
                ("generate_time_ms", ctypes.c_float),
                ("generate_tokens", ctypes.c_int),
                ("memory_usage_mb", ctypes.c_float)]


class _Result(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p), ("token_id", ctypes.c_int32),
                ("last_hidden_layer", _HiddenLayer), ("logits", _Logits),
                ("perf", _PerfStat)]


_ResultCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_Result),
                                   ctypes.c_void_p, ctypes.c_int)
_TokenizerCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int32)
_EmbedCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
    ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)


class _Callback(ctypes.Structure):
    _fields_ = [("result_callback", _ResultCallback),
                ("result_userdata", ctypes.c_void_p),
                ("tokenizer_callback", _TokenizerCallback),
                ("tokenizer_userdata", ctypes.c_void_p),
                ("embed_callback", _EmbedCallback),
                ("embed_userdata", ctypes.c_void_p)]


_REQUIRED_SYMBOLS = (
    "rkllm_init", "rkllm_run", "rkllm_abort", "rkllm_is_running",
    "rkllm_destroy", "rkllm_clear_kv_cache",
)

# 已绑定但**不被调用**的符号：rkllm_set_function_tools 是 ADR-06 明确不走的
# 路径（handle 级污染），rkllm_set_chat_template 由本项目自拼 prompt 取代
# （agent/conversation.py）。旧 runtime 缺这两个导出属正常情况，列为必需会把
# 未使用的能力当硬依赖、无谓收紧兼容性（2026-11-20 审计）。
_OPTIONAL_SYMBOLS = ("rkllm_load_lora", "rkllm_set_chat_template",
                     "rkllm_set_function_tools")

_SIGNATURES = {
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
    # LoRA：handle 级加载（可选符号——旧 runtime 没有该导出，缺失时按不支持处理）
    "rkllm_load_lora": ([ctypes.c_void_p, ctypes.POINTER(_LoraAdapter)],
                        ctypes.c_int),
}


def _resolve_library_path(candidate: str) -> str:
    """把 ctypes 的加载名解析成可读的真实路径（D10 指纹用）。

    `ctypes.CDLL("librkllmrt.so")` 的 `_name` 只是候选名，不是链接器实际
    加载的绝对路径；直接对它 os.stat 会失败，指纹就成了 None。顺序：
    候选本身是文件 → /proc/self/maps 里 basename 匹配的映射 → 原候选。
    """
    try:
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
    except OSError:
        pass
    base = os.path.basename(candidate)
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                mapped = line.rsplit(None, 1)[-1]
                # 精确比较 basename：子串匹配会命中 librkllmrt.so.1.2.3、
                # librkllmrt.so.bak 等不同文件，指纹记到别的库上（2026-11-20 审计）
                if os.path.basename(mapped) != base:
                    continue
                if os.path.isfile(mapped):
                    return mapped
    except OSError:
        pass
    return candidate


def load_library(lib_path: str | None = None) -> tuple[ctypes.CDLL, str, dict]:
    """加载 librkllmrt.so 并记录指纹（D10）。

    返回 (lib, 实际路径, 指纹 dict)。指纹含 sha256 / size / mtime，
    由调用方写入 /health 与启动日志：换库但符号仍存在时至少能立刻看出。
    """
    candidates = []
    if lib_path:
        candidates.append(lib_path)
    env = os.environ.get("RKLLM_LIB")
    if env:
        candidates.append(env)
    # 新布局（deb 打进 multiarch 目录 /usr/lib/aarch64-linux-gnu/）在
    # 默认搜索路径里就能命中；旧布局 /usr/lib/rkllm-runtime/ 已不再安装，
    # 留着只会让"找不到库"的报错多一条误导性尝试（10-25 审计 P2）。
    candidates += ["librkllmrt.so"]
    errors = []
    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append("  %s: %s" % (candidate, exc))
            continue
        # 2026-11-20 审计：仅 dlopen 成功就返回会让"名字对但内容不是 rkllm"的库
        # （错架构副本、别的同名 .so）提前终止候选链，随后 bind_library 报错却
        # 不提其他候选。这里先预检必需符号，缺则继续下一个候选。
        missing = [n for n in _REQUIRED_SYMBOLS if not hasattr(lib, n)]
        if missing:
            errors.append("  %s: 缺少必需符号 %s（不是 RKLLM runtime）"
                          % (candidate, ", ".join(missing)))
            continue
        # CDLL 只记住搜索名；映射到真实路径后再算指纹
        fp_path = _resolve_library_path(getattr(lib, "_name", candidate))
        fingerprint = {"path": fp_path, "struct_sizes": struct_sizes()}
        try:
            st = os.stat(fp_path)
            with open(fp_path, "rb") as fh:
                fingerprint["sha256"] = hashlib.sha256(fh.read()).hexdigest()[:16]
            fingerprint["size"] = st.st_size
        except OSError:
            fingerprint["sha256"] = None
        return lib, fp_path, fingerprint
    raise RKLLMError("cannot load librkllmrt.so. Tried:\n%s\n"
                     "Install rkllm-runtime, or point RKLLM_LIB at the library."
                     % "\n".join(errors))


def struct_sizes() -> dict:
    """各结构体的运行时 sizeof（字节）。

    2026-11-20 审计：模块没有尺寸校验，换 runtime 导致布局漂移仍是静默内存破坏。
    但"硬断言"需要头文件里的真实常量——我们的 `_fields_` 就是照头文件写的，
    拿自己的 sizeof 去断言等于自证。因此改为**暴露尺寸供比对**：写入
    `/health.lib_fingerprint.struct_sizes` 与启动日志，换库后可直接对照。
    """
    return {
        "_Param": ctypes.sizeof(_Param),
        "_ExtendParam": ctypes.sizeof(_ExtendParam),
        "_SamplingParam": ctypes.sizeof(_SamplingParam),
        "_InferParam": ctypes.sizeof(_InferParam),
        "_Input": ctypes.sizeof(_Input),
        "_Result": ctypes.sizeof(_Result),
        "_PromptCacheParam": ctypes.sizeof(_PromptCacheParam),
    }


def bind_library(lib: ctypes.CDLL) -> ctypes.CDLL:
    """声明函数原型；**必需**符号缺失时显式失败（不静默）。

    可选符号（`_OPTIONAL_SYMBOLS`，目前只有 LoRA）缺失时只记录在
    `lib.rkllm_optional_missing` 里：旧版 runtime 没有该导出属于正常情况，
    由调用方决定降级（拒绝带 adapter 的请求）还是忽略。
    """
    for name in _REQUIRED_SYMBOLS:
        try:
            func = getattr(lib, name)
        except AttributeError:
            raise RKLLMError("librkllmrt.so is missing the symbol %s "
                             "(is this really the RKLLM runtime?)" % name)
        argtypes, restype = _SIGNATURES[name]
        func.argtypes = argtypes
        func.restype = restype
    missing = []
    for name in _OPTIONAL_SYMBOLS:
        try:
            func = getattr(lib, name)
        except AttributeError:
            missing.append(name)
            continue
        argtypes, restype = _SIGNATURES[name]
        func.argtypes = argtypes
        func.restype = restype
    lib.rkllm_optional_missing = tuple(missing)
    return lib
