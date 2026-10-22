# mcp_conformance 说明

## 定位

N6 协议一致性脚本：对 MCP 连接器跑四个协议版本 × 六类断言，替代设计原文中的
「自写一致性脚本」（已在设计中登记为 pytest + 本脚本的组合方案）。

## 覆盖范围

四个协议版本（`2026-07-28` / `2025-11-25` / `2025-06-18` / `2024-11-05`）× 六类断言：

1. `initialize` 形状：`protocolVersion` / `serverInfo` / `capabilities`
   （`2026-07-28` 额外要求 `resultType`）；
2. `tools/list`：六个工具、每个有 `inputSchema`、带统一前缀；
3. `tools/call`：正常调用返回 `content` 数组；
4. 未知方法 → `-32601`；
5. 参数错误（`params` 不是对象）→ `-32602`；
6. 8 MiB 边界：超长帧被拒（`PARSE_ERROR`），进程仍存活。

另有独立的取消断言：长调用进行中发 `notifications/cancelled`，要求 1 s 内生效。

## 用法

```bash
python mcp_conformance.py              # 需连接器已在 stdio 模式启动
python mcp_conformance.py --verbose    # 打印每版本每断言明细
```

## 注意

脚本通过 stdio 与连接器通信，需先在 `~/.workbuddy/mcp.json` 配好
`local-rkllm` 并重启客户端；板端服务需就绪（`/ready` 返回 200）。
