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
   <WorkBuddy 配置目录>/mcp-servers/rkllm-mcp/rkllm/
   ```

2. 复制 `mcp.toml.example` 为本目录（安装目录）下的 `mcp.toml`，填 endpoint 与
   api_key，权限 `chmod 600`；密钥只留在这个文件里，不要写进 mcp.json。
   连接器默认就读 `<安装目录>/mcp.toml`，无需配置路径。

3. 在 `<WorkBuddy 配置目录>/mcp.json` 注册（`PYTHONPATH` 指 `rkllm` 的父目录，
   `RKLLM_MCP_CONFIG` 可省略——默认就是安装目录下的 `mcp.toml`）：

   ```json
   {
     "mcpServers": {
       "local-rkllm": {
         "command": "python",
         "args": ["-m", "rkllm.mcp"],
         "env": {
           "PYTHONPATH": "<WorkBuddy 配置目录>/mcp-servers/rkllm-mcp"
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

## 定位（2026-10-27）

本连接器是**保留 6 个工具的模型调用网关**，不是“完整子智能体”：不承诺任务分解、
多步规划、自主工具循环或单实例多模型路由。单进程只绑定一个 endpoint；需要同时
接 4B/27B 时，注册两个 connector 实例/两个 MCP server，用不同 `tool_prefix` 区分。

2026-10-27 稳定性修复：同步取消不再返回 `-32603`；默认估算按 `estimate_tokens_cap`
避免所有请求恒转 job；显式大 `max_tokens` 仍转 job，job 超时按最坏生成放宽；
输入按 `context_window - context_margin_tokens - 生成预算` 裁剪；`_inflight` 成功
路径也清理；reasoning 模型空正文且 `finish_reason=length` 返回 `isError`，并在
`_meta` 透传 `finish_reason/usage/reasoning_chars`；invalid args 返回 `-32602`。
新键见 `mcp.toml.example`。

## 测试

服务端 pytest 通过 `tests/conftest.py` 把本目录挂进 `rkllm.__path__` 覆盖
MCP 用例；协议一致性脚本：

```bash
python 工具/脚本/mcp_conformance.py
```

## 相关文档

- 设计与决策：`rkllm-设计方案/rkllm-mcp-接入与配置设计.md`
- 落地安装与踩坑：`文档/开发记录.md`「MCP 连接器落地安装」
