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
        return (3, len(str(self._turns)) + 9)   # 确定的假 perf，供 end 帧测试

    def clear_history(self):
        self._turns = 0
        return 0

    def abort(self):
        return None

    def is_running(self):
        return False

    def destroy(self):
        self._destroyed = True

    def diagnostics(self):
        return {"library": "<fake>", "model_path": "<fake>",
                "keep_history": self.keep_history}


class GateEngine(FakeEngine):
    """停在第一次推理里直到放行：队列行为测试用（确定性阻塞）。"""

    def __init__(self):
        FakeEngine.__init__(self, model_name="gate")
        self.started = threading.Event()
        self.release = threading.Event()

    def stream(self, prompt, role="user", enable_thinking=False,
               sampling=None, max_new_tokens=None):
        self.started.set()
        self.release.wait(15)
        return FakeEngine.stream(self, prompt, role=role,
                                 enable_thinking=enable_thinking,
                                 sampling=sampling,
                                 max_new_tokens=max_new_tokens)
