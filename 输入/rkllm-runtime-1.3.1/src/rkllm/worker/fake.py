"""FakeEngine：确定性替身，无模型无 NPU（selftest / 本机测试 / fake-worker）。

与旧 cli/rkllm.py 的差异：keep_history 默认 False（修 D5——旧版默认 True 导致
selftest 全部跑在 stateful 分支，生产默认的 stateless 路径无回归）。
"""

from __future__ import annotations

import threading
import time

__all__ = ["FakeEngine", "GateEngine", "DEFAULT_CONTEXT"]

DEFAULT_CONTEXT = 4096


class FakeEngine:
    """确定性 stand-in：逐字符回显 "answer N to: <prompt>"。"""

    def __init__(self, model_name="selftest", context_len=DEFAULT_CONTEXT,
                 keep_history=False, delay=0.0, system_prompt=None, **kwargs):
        self.model_name = model_name
        self.model_path = "<fake>"
        self.context_len = context_len
        self.keep_history = 1 if keep_history else 0
        self.delay = delay
        self.system_prompt = (system_prompt or "").strip()
        self._turns = 0
        self._last_perf = (0, 0)
        self._destroyed = False

    def stream(self, prompt, role="user", enable_thinking=False,
               sampling=None, max_new_tokens=None, cache_hint=True):
        self._turns += 1
        if self.system_prompt:
            prompt = "%s\n\n%s" % (self.system_prompt, prompt)
        answer = "answer %d to: %s" % (self._turns, prompt)
        # 与真 runtime 同口径给出非固定 perf：prompt 按 UTF-8 字节/3 保守折算，
        # 输出按 4 字符/token 粗估。目的不是精确，而是让 golden 基线能反映
        # prompt/输出长度变化，而不是永远锁死 (3, 9)。
        prefill = max(1, len(prompt.encode("utf-8")) // 3)
        generate = max(1, len(answer) // 4)
        self._last_perf = (prefill, generate)
        for char in answer:
            if self.delay:
                time.sleep(self.delay)
            yield char

    def last_perf(self):
        return self._last_perf

    def abort(self):
        return None

    def destroy(self):
        self._destroyed = True


class GateEngine(FakeEngine):
    """停在第一次推理里直到放行：队列行为测试用（确定性阻塞）。"""

    def __init__(self):
        FakeEngine.__init__(self, model_name="gate")
        self.started = threading.Event()
        self.release = threading.Event()

    def stream(self, prompt, role="user", enable_thinking=False,
               sampling=None, max_new_tokens=None, cache_hint=True):
        self.started.set()
        self.release.wait(15)
        return FakeEngine.stream(self, prompt, role=role, cache_hint=cache_hint,
                                 enable_thinking=enable_thinking,
                                 sampling=sampling,
                                 max_new_tokens=max_new_tokens)
