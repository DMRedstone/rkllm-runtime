"""RealEngine：ctypes 封装的板端引擎（迁移自旧 cli/rkllm.py，S1）。

迁移要点：
- 保活纪律不变：_keep（引擎寿命缓冲）/ _run_keep（单次推理缓冲）；
- _stream_guard 语义不变：同一引擎并发 stream 显式失败（D15 之外的既有防线）；
- 回调异常永不 unwind 进 C（沿用旧版 _on_result 的防御）；
- D15：增量 UTF-8 解码器兜底多字节汉字被回调切成两块的场景。
"""

from __future__ import annotations

import codecs
import ctypes
import queue
import sys
import threading

from ..abi import (RKLLMError, RKLLM_INFER_GENERATE, RKLLM_INPUT_PROMPT,
                   RKLLM_RUN_ERROR, RKLLM_RUN_FINISH, RKLLM_RUN_NORMAL,
                   BIG_CORE_PLATFORMS, CPU_MASK_HIGH, CPU_MASK_LOW,
                   MODEL_SUFFIX, _Callback, _InferParam, _Input, _Param,
                   _PromptCacheParam, _ResultCallback, _SamplingParam,
                   _TokenizerCallback,
                   _EmbedCallback, bind_library, load_library)

STREAM_JOIN_TIMEOUT_S = 60

# 请求级采样参数的基线（与 1.3.1-15 及 _Param 的加载默认值一致）。
#
# 为什么必须有这份基线：rkllm 的 sampling_params 是指向结构体的指针，
# 结构体里**每个字段都生效**。旧实现按请求里出现的键新建结构体，没给的
# 字段就是 ctypes 的 0——top_k=0 / top_p=0.0 / repeat_penalty=0.0 全部非法，
# 采样直接崩坏（真机表现为词语重复、`</think>` 标签残留）。
# 正确语义是"请求字段覆盖基线"，不是"重建结构体"。
DEFAULT_SAMPLING = {
    "top_k": 1, "top_p": 0.9, "temperature": 0.8, "repeat_penalty": 1.1,
    "frequency_penalty": 0.0, "presence_penalty": 0.0,
    "mirostat": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1,
}


def _sanitize_sampling(param: dict) -> None:
    """夹紧非法取值（原地修改）。

    这些不是"用户设了个怪值"，而是结构体默认 0 或客户端传了越界值时必须
    兜住的底线——采样器不接受 top_k<=0 / top_p<=0 / repeat_penalty<1。
    """
    if param["top_k"] is None or param["top_k"] < 1:
        param["top_k"] = DEFAULT_SAMPLING["top_k"]
    if param["top_p"] is None or not (0.0 < param["top_p"] <= 1.0):
        param["top_p"] = DEFAULT_SAMPLING["top_p"]
    if param["temperature"] is None or param["temperature"] < 0:
        param["temperature"] = DEFAULT_SAMPLING["temperature"]
    if param["repeat_penalty"] is None or param["repeat_penalty"] < 1.0:
        param["repeat_penalty"] = DEFAULT_SAMPLING["repeat_penalty"]
    for key in ("frequency_penalty", "presence_penalty"):
        if param[key] is None:
            param[key] = 0.0
    if param["mirostat"] is None:
        param["mirostat"] = 0


def build_sampling(sampling: dict | None) -> dict:
    """请求字段覆盖基线，得到完整的 _SamplingParam 字段集（纯函数，可单测）。"""
    merged = dict(DEFAULT_SAMPLING)
    if sampling:
        merged.update({k: v for k, v in sampling.items()
                       if k in DEFAULT_SAMPLING and v is not None})
    _sanitize_sampling(merged)
    return merged


class _Failure(Exception):
    pass


class _Aborted(Exception):
    pass


_END = object()


class RealEngine:
    """ctypes wrapper around a live RKLLM model."""

    def __init__(self, model_path, platform, context_len=4096,
                 max_new_tokens=2048, keep_history=False,
                 threads=4, system_prompt=None, lib_path=None, verbose=False,
                 prompt_cache=None):
        self.model_path = model_path
        self.model_name = model_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if self.model_name.endswith(MODEL_SUFFIX):
            self.model_name = self.model_name[:-len(MODEL_SUFFIX)]
        self.context_len = context_len
        self.max_new_tokens = max_new_tokens
        self.keep_history = 1 if keep_history else 0
        # platform=auto 必须解析成实际 SoC：CPU mask 的选择依赖它，
        # 未解析时 "auto" 会落到小核（A55），prefill/解码慢 3-4 倍（真机实测）。
        from ..platform.soc import resolve_platform
        self.platform = resolve_platform(platform)
        self.lib, self.lib_path, self.lib_fingerprint = load_library(lib_path)
        bind_library(self.lib)

        self._keep = []            # 引擎寿命缓冲（model path 等）
        self._run_keep = []        # 单次推理缓冲，run 结束即清
        self._sink = None
        self._sink_lock = threading.Lock()
        self._stream_guard = threading.Lock()
        self._last_perf = None
        self._handle = ctypes.c_void_p()
        self._destroyed = False
        # D15：增量解码器——回调可能把多字节字符切在两个 chunk
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        # 前缀缓存（cache.*）：None = 关闭；由 cli 按配置构造后注入
        self.prompt_cache = prompt_cache

        param = _Param()
        param.model_path = model_path.encode("utf-8")
        self._keep.append(param.model_path)
        param.max_context_len = int(context_len)
        param.max_new_tokens = int(max_new_tokens)
        param.n_keep = -1
        # 加载期默认值与请求期基线同源（DEFAULT_SAMPLING），避免两处漂移
        param.top_k = DEFAULT_SAMPLING["top_k"]
        param.top_p = DEFAULT_SAMPLING["top_p"]
        param.temperature = DEFAULT_SAMPLING["temperature"]
        param.repeat_penalty = DEFAULT_SAMPLING["repeat_penalty"]
        param.skip_special_token = True
        param.ignore_eos_token = False
        param.is_async = False
        param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1
        param.extend_param.n_batch = 1
        cpu_count = int(threads) if threads else 4
        base_mask = (CPU_MASK_HIGH if self.platform in BIG_CORE_PLATFORMS
                     else CPU_MASK_LOW)
        candidates = [i for i in range(32) if base_mask & (1 << i)]
        chosen = candidates[:max(1, min(cpu_count, len(candidates)))]
        param.extend_param.enabled_cpus_num = len(chosen)
        param.extend_param.enabled_cpus_mask = sum(1 << i for i in chosen)

        self._callback_type = _ResultCallback(self._on_result)
        callback = _Callback()
        callback.result_callback = self._callback_type
        callback.tokenizer_callback = _TokenizerCallback()
        callback.embed_callback = _EmbedCallback()
        self._callback = callback
        self._param = param

        if verbose:
            sys.stderr.write("[rkllm] loading %s with %s\n" % (model_path, self.lib_path))
        code = self.lib.rkllm_init(ctypes.byref(self._handle),
                                   ctypes.byref(param), ctypes.byref(callback))
        if code != 0:
            raise RKLLMError("rkllm_init failed with code %d. Common causes: the "
                             "model file is not a valid .rkllm for this runtime "
                             "version, or NPU memory is exhausted." % code)

        self._infer = _InferParam()
        ctypes.memset(ctypes.byref(self._infer), 0, ctypes.sizeof(_InferParam))
        self._infer.mode = RKLLM_INFER_GENERATE
        self._infer.keep_history = self.keep_history
        self._infer.max_new_tokens = 0
        self.system_prompt = (system_prompt or "").strip()

    # -- 回调（绝不 unwind 进 C）--------------------------------------------

    def _on_result(self, result, userdata, state):
        sink = self._sink
        try:
            if state == RKLLM_RUN_NORMAL:
                chunk = result.contents.text
                if chunk and sink is not None:
                    text = self._decoder.decode(chunk)
                    if text:
                        sink.put(text)
            elif state == RKLLM_RUN_FINISH:
                try:
                    perf = result.contents.perf
                    self._last_perf = (int(perf.prefill_tokens),
                                       int(perf.generate_tokens))
                except Exception:
                    pass
            elif state == RKLLM_RUN_ERROR:
                if sink is not None:
                    sink.put(_Failure("the runtime reported RKLLM_RUN_ERROR"))
        except Exception:
            if sink is not None:
                try:
                    sink.put(_Failure("callback failure"))
                except Exception:
                    pass
        return 0

    # -- 推理 ----------------------------------------------------------------

    def stream(self, prompt, role="user", enable_thinking=False,
               sampling=None, max_new_tokens=None, cache_hint=True):
        """产出文本片段。同一引擎并发调用显式失败（_stream_guard）。

        前缀缓存（cache.*）在这里决策：命中则让 runtime 加载缓存、只算新增
        部分；未命中则本次推理后保存。若加载失败（缓存文件损坏或语义不兼容）
        且尚未吐出任何 token，则清槽重试一次不带缓存的推理——缓存只能是加速
        器，不能成为可用性的单点。
        """
        if self._destroyed:
            raise RKLLMError("engine already destroyed")
        if not self._stream_guard.acquire(blocking=False):
            raise RKLLMError("concurrent stream() calls on one engine are not supported")
        try:
            if self.system_prompt:
                prompt = "%s\n\n%s" % (self.system_prompt, prompt)
            cache_path, save = (None, 0)
            # cache_hint：只有"后续还会以这段 prompt 为前缀"的请求才值得存缓存。
            # 单轮请求（messages 只有一条）存了也不会命中——实测 35 次单轮请求
            # 每次写 185 MB 缓存文件，纯属 I/O 浪费与存储磨损。
            if self.prompt_cache is not None and cache_hint:
                cache_path, save = self.prompt_cache.prepare(prompt)
            recovered = False
            while True:
                produced = False
                try:
                    for item in self._stream_once(
                            prompt, role, enable_thinking, sampling,
                            max_new_tokens, cache_path, save):
                        produced = True
                        yield item
                    break
                except RKLLMError:
                    # 缓存加载失败（rkllm_run -1 / 内部码 10000）：实测换 save=1
                    # 重跑仍失败——失败后运行时状态已脏，唯一可靠出路是
                    # **完全不带缓存**干净重跑，并熔断本进程的缓存功能。
                    # 若重跑也失败（状态确实坏了），向上抛由 supervisor 重启。
                    if (not produced and cache_path is not None
                            and not recovered and self.prompt_cache is not None):
                        recovered = True
                        self.prompt_cache.invalidate()
                        self.prompt_cache.disable()
                        cache_path, save = None, 0
                        continue
                    raise
            if self.prompt_cache is not None and cache_path:
                self.prompt_cache.commit(prompt)
        finally:
            self._stream_guard.release()

    def _stream_once(self, prompt, role, enable_thinking, sampling,
                     max_new_tokens, cache_path=None, save=0):
        """一次完整推理尝试；并发锁与缓存命中判断由 stream() 负责。"""
        self._last_perf = None
        self._decoder.reset()
        sink = queue.Queue()
        with self._sink_lock:
            self._sink = sink

        rkllm_input = _Input()
        role_bytes = role.encode("utf-8")
        rkllm_input.role = role_bytes
        rkllm_input.enable_thinking = bool(enable_thinking)
        rkllm_input.input_type = RKLLM_INPUT_PROMPT
        encoded = prompt.encode("utf-8")
        rkllm_input.prompt_input = encoded
        self._run_keep = [role_bytes, encoded]   # 保活到 run 返回

        sampling_ptr = None
        run_sampling = None
        if sampling is not None:
            run_sampling = _SamplingParam(**build_sampling(sampling))
            sampling_ptr = ctypes.pointer(run_sampling)
        self._infer.sampling_params = sampling_ptr
        self._infer.max_new_tokens = int(max_new_tokens) if max_new_tokens else 0

        # 前缀缓存指针：结构体与路径 bytes 都必须活到 rkllm_run 返回
        cache_param = None
        if cache_path:
            cache_bytes = cache_path.encode("utf-8")
            cache_param = _PromptCacheParam(int(save), cache_bytes)
            self._run_keep.append(cache_bytes)
            self._infer.prompt_cache_params = ctypes.pointer(cache_param)
        else:
            self._infer.prompt_cache_params = None

        def runner():
            try:
                code = self.lib.rkllm_run(self._handle, ctypes.byref(rkllm_input),
                                          ctypes.byref(self._infer), None)
                if code != 0:
                    sink.put(_Failure("rkllm_run returned %d" % code))
            except Exception as exc:
                sink.put(_Failure("inference crashed: %s" % exc))
            finally:
                sink.put(_END)

        thread = threading.Thread(target=runner, name="rkllm-infer", daemon=True)
        thread.start()
        try:
            while True:
                item = sink.get()
                if item is _END:
                    break
                if isinstance(item, _Failure):
                    raise RKLLMError(str(item))
                yield item
        finally:
            try:
                try:
                    thread.join(timeout=STREAM_JOIN_TIMEOUT_S)
                    if thread.is_alive():
                        self._destroyed = True
                        try:
                            sys.stderr.write(
                                "[rkllm] inference thread did not exit within %ds; "
                                "marking the engine unusable\n"
                                % STREAM_JOIN_TIMEOUT_S)
                            sys.stderr.flush()
                        except (OSError, ValueError):
                            pass
                except Exception:
                    self._destroyed = True
            finally:
                with self._sink_lock:
                    self._sink = None
                self._run_keep = []
                self._infer.sampling_params = None
                self._infer.prompt_cache_params = None
                self._infer.max_new_tokens = 0

    def last_perf(self):
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
        return {"library": self.lib_path, "model_path": self.model_path,
                "keep_history": self.keep_history,
                "lib_sha256": self.lib_fingerprint.get("sha256"),
                "lib_size": self.lib_fingerprint.get("size")}
