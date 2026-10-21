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
                   _ResultCallback, _SamplingParam, _TokenizerCallback,
                   _EmbedCallback, bind_library, load_library)

STREAM_JOIN_TIMEOUT_S = 60


class _Failure(Exception):
    pass


class _Aborted(Exception):
    pass


_END = object()


class RealEngine:
    """ctypes wrapper around a live RKLLM model."""

    def __init__(self, model_path, platform, context_len=4096,
                 max_new_tokens=2048, keep_history=False,
                 threads=4, system_prompt=None, lib_path=None, verbose=False):
        self.model_path = model_path
        self.model_name = model_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if self.model_name.endswith(MODEL_SUFFIX):
            self.model_name = self.model_name[:-len(MODEL_SUFFIX)]
        self.context_len = context_len
        self.max_new_tokens = max_new_tokens
        self.keep_history = 1 if keep_history else 0
        self.platform = platform
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
        param.skip_special_token = True
        param.ignore_eos_token = False
        param.is_async = False
        param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1
        param.extend_param.n_batch = 1
        cpu_count = int(threads) if threads else 4
        base_mask = CPU_MASK_HIGH if platform in BIG_CORE_PLATFORMS else CPU_MASK_LOW
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
               sampling=None, max_new_tokens=None):
        """产出文本片段。同一引擎并发调用显式失败（_stream_guard）。"""
        if self._destroyed:
            raise RKLLMError("engine already destroyed")
        if not self._stream_guard.acquire(blocking=False):
            raise RKLLMError("concurrent stream() calls on one engine are not supported")
        self._last_perf = None
        self._decoder.reset()
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
        self._run_keep = [role_bytes, encoded]   # 保活到 run 返回

        sampling_ptr = None
        run_sampling = None
        if sampling is not None:
            run_sampling = _SamplingParam(**{
                k: v for k, v in sampling.items()
                if k in {f[0] for f in _SamplingParam._fields_}})
            sampling_ptr = ctypes.pointer(run_sampling)
        self._infer.sampling_params = sampling_ptr
        self._infer.max_new_tokens = int(max_new_tokens) if max_new_tokens else 0

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
                    raise item
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
                try:
                    with self._sink_lock:
                        self._sink = None
                    self._run_keep = []
                    self._infer.sampling_params = None
                    self._infer.max_new_tokens = 0
                finally:
                    self._stream_guard.release()

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
