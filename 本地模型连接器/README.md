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
转 job 阈值（= `client_timeout_s` − 15 s，默认 60−15=45 s）自动转 job，用
`local_llm_job` 轮询或取消。模型能力从板端 `/health` 读取，不在连接器配置：

- `perf.decode_tok_s`：解码速率（job 判定/超时/同步生成上限）；
- `perf.ttft_s`：首 token 延迟，作为 prefill 估算的实测值（冷启动才回落
  连接器内置的 44 tok/s 兜底）；
- `limits.*`：上下文、输出上限、请求超时；
- `tokens.bytes_per_token`：tokenizer 口径，连接器的输入裁剪/预算按它折算。

协议支持四版（2026-07-28 无状态 + 三个 initialize 旧版）；`protocol_versions`
收缩时，回退版本一定落在该列表内，2026 收到 `initialize` 返回 -32601（显性化
版本误用），旧版握手则返回实际支持的版本。

## 错误语义

工具执行失败一律是 MCP `CallToolResult`（`isError=true` + 结构化 `_meta`），
不是统一 `-32603`：

| 场景 | `_meta.kind` | 额外字段 |
|---|---|---|
| 板端 503 / 429 | `busy` / `rate_limited` | `retryAfter`（秒） |
| 板端 400/422（参数/上下文预算） | `bad_request` | `httpStatus` |
| 板端 401/403（key 错） | `auth` | `httpStatus` |
| 板端 404 | `not_found` | `httpStatus` |
| SSE 流中错误帧 | `stream` | `code`（板端错误码） |
| 板端 5xx/其他 HTTP | `upstream`/`http` | `httpStatus` |

job 路径把同一份 `kind/httpStatus/retryAfter` 带到失败终帧的 `_meta`，客户端
可据此退避或修配置，而不用解析文案。客户端参数错误仍是 JSON-RPC `-32602`
（含畸形 JSON Schema）；无 id 的非通知请求（典型：`tools/call` 漏 id）直接
`-32600`，不会阻塞 reader 或回 `id=null`。

## 定位（2026-10-27）

本连接器是**保留 6 个工具的模型调用网关**，不是“完整子智能体”：不承诺任务分解、
多步规划、自主工具循环或单实例多模型路由。单进程只绑定一个 endpoint；需要同时
接 4B/27B 时，注册两个 connector 实例/两个 MCP server，用不同 `tool_prefix` 区分。

## 2026-11-23 P2.5 修复

- 配置 `inf`/`1e400`/越界/未知枚举不再让启动崩溃：统一回退默认并在 stderr 告警；
  endpoint 必须是 http(s) URL，`tools` 只认 6 个实现名。
- 板端 400/401/流中错误归一为有语义的 `_meta`（见上表），保留 `retryAfter`。
- 请求按需带 `stream_options.include_usage`，工具 `_meta.usage` 可用（精确
  `prompt_tokens/completion_tokens` 来自板端 usage 帧）。
- 无 id 的 `tools/call` 拒绝（-32600）；8 MiB 上限在整行读入前生效（分块读取）。
- prefill 用板端 `ttft_s`；token 预算用板端 `tokens.bytes_per_token`；会话淘汰是
  真 LRU；畸形 JSON Schema 返回 -32602。
- `mcp_conformance.py` 修好 `recv` 超时、`--verbose` 与取消断言（现在同时校验
  MCP 取消结果语义与板端 `/reset` 证据），保持 **45/45**。

## 测试

服务端 pytest 通过 `tests/conftest.py` 把本目录挂进 `rkllm.__path__` 覆盖
MCP 用例（`tests/test_mcp_*.py`，P2.5 起含真实 stdio/JSON-RPC/临时 TOML 集成
测试 `tests/test_p2_mcp.py`）；协议一致性脚本：

```bash
python 工具/脚本/mcp_conformance.py           # 45/45
python 工具/脚本/mcp_conformance.py --verbose # 逐条 PASS/FAIL + 细节
```

## 相关文档

- 落地安装与踩坑：`文档/开发记录.md` §5.2（MCP 连接器段）
- A 方案能力下沉：`文档/开发记录.md` §16、§17（审查后续修复）
- 设计取舍与决策：`文档/开发记录.md` §15.9
