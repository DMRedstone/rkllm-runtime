"""推理管线的依赖契约（A-5）。

Pipeline 原先直接依赖 `server.app.ServiceState`：`iter_tokens` 与构造期共 8 处
属性访问，agent 层想单测就得伪造整个 ServiceState（队列、supervisor、配置
视图一应俱全）。这里把「管线真正需要什么」写成 Protocol：

    model_name / tool_parser / max_tool_calls / tools_mode / system_prompt
    supervisor / queue / ttft_log()

`ServiceState` 结构上即满足（policy 字段经属性转发），不需要继承；测试可以
只提供满足协议的最小对象。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["InferenceGateway"]


@runtime_checkable
class InferenceGateway(Protocol):
    """推理管线的最小依赖面（server 的 ServiceState 天然满足）。"""

    model_name: str
    tool_parser: str
    max_tool_calls: int
    tools_mode: str
    system_prompt: str
    context_len: int
    default_max_tokens: int
    max_tokens_limit: int
    json_hint: str
    supervisor: Any        # 提供 execute()/abort()；None = 未配置推理
    queue: Any             # 提供 run(cancelled=) 与 request_timeout_s

    def ttft_log(self, seconds: float) -> None:
        ...
