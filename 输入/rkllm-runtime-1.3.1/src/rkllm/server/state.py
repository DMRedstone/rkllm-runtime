"""进程级服务状态（A-1：拆掉上帝对象，保留对外 facade）。

改造前 `ServiceState` 四位一体——配置装配（InferenceQueue）、元数据派生
（模型名 / 档位 / 上下文）、12 个策略默认值、运行期可变状态。任何一处调整
都要动这个大对象，agent 层也无法脱离 server 单测。

拆分后：

- `PolicyView`：构造期从 cfg 派生的**只读**策略与元数据。请求处理全程只读，
  可以在测试里单独构造/断言。
- `ServiceState`：facade，组合 `policy` 与运行期状态（`supervisor` / `queue` /
  `default_sampling` / `version` / `lib_fingerprint`）。policy 字段通过
  `__getattr__` 转发，`svc.model_name` 这类既有访问方式完全不变；新代码可以
  用 `svc.policy.xxx` 明确表达「这是只读策略」。
"""

from __future__ import annotations

import logging

from .. import VERSION
from ..platform import derive_class, max_tokens_for

__all__ = ["PolicyView", "ServiceState"]

log = logging.getLogger("rkllm.server")


class PolicyView:
    """从配置派生的只读策略与模型元数据（构造后不再变化）。"""

    def __init__(self, cfg):
        self.model_name = (cfg.model.path.rsplit("/", 1)[-1]
                           .rsplit("\\", 1)[-1].removesuffix(".rkllm")
                           or "rkllm")
        self.model_path = cfg.model.path
        self.context_len = cfg.model.context
        # 档位（F12）：显式配置优先，否则按文件名/体积推导。
        # 键名是 model.class（Python 关键字），只能用 getattr 取。
        self.model_class = derive_class(cfg.model.path,
                                       getattr(cfg.model, "class", "auto"))
        self.default_max_tokens = (cfg.model.max_tokens
                                   or max_tokens_for(self.model_class))
        # 请求 max_tokens 的部署级上限；0=不额外限制，由 context 预算裁剪
        self.max_tokens_limit = cfg.model.max_tokens_limit
        self.param_policy = cfg.openai.param_policy
        self.tool_parser = cfg.openai.tool_parser
        self.max_tool_calls = cfg.openai.max_tool_calls
        # openai.tools_mode：native 需 rkllm_set_function_tools，ADR-06 明确不用
        # （handle 级永久生效会污染后续请求），配置成 native 时降级为 prompt 并告警
        self.tools_mode = cfg.openai.tools_mode
        if self.tools_mode == "native":
            # A-3：告警走日志系统（原先 sys.stderr.write，配 logging.file 时
            # 运维在日志文件里完全看不到）
            log.warning("openai.tools_mode=native 不受支持（ADR-06），已按 prompt 处理")
        self.reasoning_field = cfg.openai.reasoning_field
        self.json_schema_retries = cfg.openai.json_schema_retries
        self.json_hint = cfg.openai.json_hint
        self.stream_heartbeat_s = cfg.openai.stream_heartbeat_s
        self.include_usage_default = cfg.openai.include_usage_default


class ServiceState:
    """进程级服务状态 facade（`app.state.svc`）。

    只读策略走 `self.policy`（并转发属性）；运行期可变状态直接挂在实例上。
    """

    def __init__(self, cfg, *, supervisor=None):
        from .queue import InferenceQueue

        self.cfg = cfg
        self.policy = PolicyView(cfg)
        self.queue = InferenceQueue(
            max_wait=cfg.queue.max_wait,
            wait_timeout_s=cfg.queue.wait_timeout_s,
            request_timeout_s=cfg.queue.request_timeout_s,
            busy_retry_after_s=cfg.queue.busy_retry_after_s,
            max_concurrent=cfg.queue.max_concurrent)
        self.supervisor = supervisor      # None = 未配置推理（仅运维端点）
        # 进程级采样默认值（--temperature 等）：请求未指定时使用，
        # 由 cli 在启动时填入；空 dict 表示全用 runtime 默认。
        self.default_sampling: dict = {}
        self.system_prompt = ""
        self.version = VERSION
        self.lib_fingerprint: dict | None = None

    def __getattr__(self, name: str):
        """转发只读 policy 字段（`svc.model_name` 等既有访问方式不变）。"""
        policy = self.__dict__.get("policy")
        if policy is None:
            raise AttributeError(name)
        try:
            return getattr(policy, name)
        except AttributeError:
            raise AttributeError(name) from None

    def on_reset(self) -> None:
        if self.supervisor:
            # 走 worker 的 abort（帧协议控制帧）；KV 由下一次请求全量重放
            self.supervisor.abort()

    def ttft_log(self, seconds: float) -> None:
        """首 token 耗时打点（N3 性能定位；默认 info 级，便于板端直接看）。"""
        logging.getLogger("rkllm.server.ttft").info(
            "supervisor 首 token %.2fs", seconds)
