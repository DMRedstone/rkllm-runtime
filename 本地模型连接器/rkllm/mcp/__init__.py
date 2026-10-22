"""自写 MCP 连接器（S4，MCP 接入设计 §1.4/§8）。

- protocol.py  JSON-RPC 分发 + 四版协议差异表 + 错误码
- stdio.py     读循环（线程+队列）、写锁、取消与进度
- tools.py     6 个工具（只依赖 Backend，可单测）
- jobs.py      长任务状态机、TTL、取消
- backend.py   HttpBackend（urllib）/ LocalBackend 协议

零第三方依赖（纯 stdlib），目标客户端 WorkBuddy（stdio）。
"""
