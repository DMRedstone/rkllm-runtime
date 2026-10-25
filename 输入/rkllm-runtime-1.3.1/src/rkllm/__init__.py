"""rkllm 板端模型服务 —— 模块化实现（改造设计 v2 / MCP 接入设计 v3）。

模块划分（改造设计 §4.4 / §8）：
- config      82 键 TOML 配置表：schema、加载、优先级、校验、config 命令
- caps        OpenAI 能力三档表（单一事实源）
- abi         ctypes 结构体镜像 + 上游库校验（S1）
- worker/     推理子进程：UDS 帧协议、引擎、prefix cache、FakeEngine（S1）
- server/     FastAPI：队列、supervisor、路由、SSE（S2）
- agent/      tools 解析、json_schema、reasoning、conversation（S3）
- platform/   SoC 探测与锁频（迁移自旧实现）

MCP 连接器是**客户端独立源码包**，在仓库根 `本地模型连接器/`，不随板端 deb
分发；板端 deb 只含服务端模块。本包不再包含 `mcp/`。
"""

# 版本：单点定义，其余（--version / 启动横幅 / app.title / /v1 根 / state）
# 一律引用，避免发布漏改后各处自报的版本互不一致。
#   __version__  PEP 440 口径（pyproject / importlib.metadata）
#   VERSION      deb 修订号口径（对外展示：1.3.1-16 这种连字符形式）
__version__ = "1.3.1.16"
VERSION = "1.3.1-16"
