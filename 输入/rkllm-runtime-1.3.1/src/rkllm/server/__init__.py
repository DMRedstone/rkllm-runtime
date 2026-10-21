"""FastAPI 服务端层（S2，改造设计 §6.1/§6.2/§6.6）。

- queue.py       单槽 FIFO + 背压（503 + Retry-After）
- supervisor.py  worker 子进程生命周期 + 重启退避 + 帧客户端
- schemas.py     pydantic 请求模型 + caps 参数策略
- sse.py         SSE 帧工具（固定 id、心跳、[DONE]）
- routes/ops.py      health / ready / metrics / capabilities / reset
- routes/openai.py   chat/completions / completions / models
- app.py         FastAPI 组装 + 优雅退出
"""
