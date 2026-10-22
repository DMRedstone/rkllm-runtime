"""rkllm 板端模型服务 —— 模块化实现（改造设计 v2 / MCP 接入设计 v3）。

模块划分（改造设计 §4.4 / §8）：
- config      80 键 TOML 配置表：schema、加载、优先级、校验、config 命令
- caps        OpenAI 能力三档表（单一事实源）
- abi         ctypes 结构体镜像 + 上游库校验（S1）
- worker/     推理子进程：UDS 帧协议、引擎、prefix cache、FakeEngine（S1）
- server/     FastAPI：队列、supervisor、路由、SSE（S2）
- agent/      tools 解析、json_schema、reasoning、conversation（S3）
- platform/   SoC 探测与锁频（迁移自旧实现）

MCP 连接器是**客户端独立源码包**，在仓库根 `本地模型连接器/`，不随板端 deb
分发；板端 deb 只含服务端模块。本包不再包含 `mcp/`。
"""

__version__ = "1.3.1.16"
