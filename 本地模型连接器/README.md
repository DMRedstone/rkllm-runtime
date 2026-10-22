# rkllm MCP 连接器（客户端独立源码）

把板端小模型接成 WorkBuddy 可调用的 MCP 工具。**纯标准库、零第三方依赖**，
以源码目录交付（无 wheel/deb），装在客户端主机上用 stdio 拉起；板端只跑
FastAPI 服务，两者互不混放。

## 目录

```
本地模型连接器/
├── rkllm/                 # 拷到客户端 PYTHONPATH 的包
│   ├── __init__.py
│   ├── agent/json_schema.py
│   └── mcp/               # JSON-RPC、stdio、6 工具、job、backend
├── mcp.toml.example       # 客户端配置样例
└── README.md
```

## 安装（客户端主机）

1. 把本目录的 `rkllm/` 整体拷到：

   ```
   ~/.workbuddy/mcp-servers/rkllm-mcp/rkllm/
   ```

2. 复制 `mcp.toml.example` 到 `~/.config/rkllm/mcp.toml`，填 endpoint 与
   api_key，权限 `chmod 600`；密钥不要写进 mcp.json。

3. 在 `~/.workbuddy/mcp.json` 注册（`PYTHONPATH` 指 `rkllm` 的父目录）：

   ```json
   {
     "mcpServers": {
       "local-rkllm": {
         "command": "python",
         "args": ["-m", "rkllm.mcp"],
         "env": {
           "PYTHONPATH": "C:/Users/<you>/.workbuddy/mcp-servers/rkllm-mcp",
           "RKLLM_MCP_CONFIG": "C:/Users/<you>/.config/rkllm/mcp.toml"
         }
       }
     }
   }
   ```

4. 完全退出并重启 WorkBuddy → 「自定义连接器」里信任 `local-rkllm`。

5. 验证：

   ```
   python -m rkllm.mcp ask "用一句话说明你在运行"
   ```

## 工具

`local_llm_ask` / `chat` / `extract` / `batch` / `status` / `job`；估算超过
`task_threshold_s`（默认 45 s，必须 < 60）自动转 job，用 `local_llm_job`
轮询或取消。协议支持四版（2026-07-28 无状态 + 三个 initialize 旧版）。

2026-10-22 修复后：2026 `resultType=complete`、`server/discover` 全版本可用、
`tools/list` 带 `ttlMs`/`cacheScope`；`ask` 支持 `format=json_object`（提示约束），
`chat` 支持可选 `session` 记忆，`extract` 声明 `outputSchema`；job 进度不再阻塞
启动，`status`/`job` 走独立控制池，长 ask 期间仍可轮询。

## 测试

服务端 pytest 通过 `tests/conftest.py` 把本目录挂进 `rkllm.__path__` 覆盖
MCP 用例；协议一致性脚本：

```bash
python 工具/脚本/mcp_conformance.py
```

## 相关文档

- 设计与决策：`rkllm-设计方案/rkllm-mcp-接入与配置设计.md`
- 落地安装与踩坑：`文档/开发记录.md`「MCP 连接器落地安装」
