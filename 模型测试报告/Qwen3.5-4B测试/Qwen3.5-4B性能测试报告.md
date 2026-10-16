# Qwen3.5-4B 板端推理服务 · 性能与稳定性测试报告

- 目标：`http://192.0.2.10:8080`

- 模型：`qwen3.5-4b_w8a8_rk3588`

- 开发板：OrangePi 5 Plus 8G（RK3588 / 8GB）

- 生成时间：2026-10-15 18:51:24

## 执行摘要

服务单流推理稳定可靠，边界处理健壮；不具备并发吞吐能力（NPU 单请求串行），无背压限流；存在两处行为偏差：max_tokens 不被严格遵守、以及一次低概率并发崩溃（可被守护进程自动拉起）。

- 单流生成速率约 8–9 tok/s，短问答端到端 1–3s

- 并发不带来吞吐增益：QPS 恒定 0.86–0.94，仅增加排队延迟（16 并发 p99 = 18.2s）

- 无背压：96 个瞬时并发全部被接受，全部排队（无 503/429）

- 稳定性良好：压测 265 次请求 errors=0，40 次连续请求延迟无漂移

- 边界健壮：超长/畸形请求均被 400 干净拒绝

## 测试环境与资源占用

| 进程 | VIRT | RES | CPU% | MEM% | 累计CPU时间 | load average |
| --- | --- | --- | --- | --- | --- | --- |
| rkllm | 5.0g | 4.4g | 186.4 | 66.1 | 14:17.00 | 3.28, 3.08, 2.21 |

rkllm 常驻内存 4.4G/8G（66%），内存为最紧资源；CPU 约 1.86 核，8 核平台未饱和。

> 

## 压力测试

| 段 | 指标 | 结果 |
| --- | --- | --- |
| A | short prompt, 32 tok | 200  2.32s  in=2 out=15  6.5 tok/s |
| A | medium prompt, 32 tok | 200  3.28s  in=200 out=10  3.0 tok/s |
| A | long prompt, 32 tok | 200  11.98s  in=1200 out=10  0.8 tok/s |
| A | generate 128 tok | 200  22.72s  in=17 out=200  8.8 tok/s |
| A | generate 256 tok | 200  45.68s  in=17 out=370  8.1 tok/s |
| C | 40 sequential requests | 40 ok  0 failed  median 1.22s  agg 0.8 tok/s |
| C | latency drift | first quarter 1.21s -> last quarter 1.22s  (delta +0.01s) |
| C | server counters | requests=169 errors=0 |
| D | burst 1: 32 simultaneous | wall 49.0s  statuses {200: 32} |
| D | burst 2: 32 simultaneous | wall 49.4s  statuses {200: 32} |
| D | burst 3: 32 simultaneous | wall 49.5s  statuses {200: 32} |
| D | totals | statuses {200: 96} |
| D | no explicit shed observed | 96 requests all queued/accepted |
| D | server counters | requests=265 rejected=0 errors=0 |
| E | advertised guard | context 4096 tokens, 32768 byte budget |
| E | max_tokens = 0 | 200  0.98s  {"id": "chatcmpl-2f1c8744bbc24968be3735ec", "object": "chat.completion |
| E | max_tokens = -1 | 200  0.88s  {"id": "chatcmpl-ed887f9145104c8ab7b25861", "object": "chat.completion |
| E | max_tokens huge | 200  1.64s  {"id": "chatcmpl-362b20391fc44349bce25475", "object": "chat.completion |
| E | temperature = -1 | 200  1.73s  {"id": "chatcmpl-51ccfddee7da499d8b668187", "object": "chat.completion |
| E | top_p = 2.0 | 200  1.58s  {"id": "chatcmpl-47be84c77dbc4183b8db7ce7", "object": "chat.completion |
| E | 2000 messages | 200  1.65s  {"id": "chatcmpl-24cae213d010413e8ddb60bf", "object": "chat.completion |
| E | unknown role | 200  1.65s  {"id": "chatcmpl-d96f9932f6574a578baf98f3", "object": "chat.completion |
| E | control chars | 200  1.55s  {"id": "chatcmpl-b718f52f4ae14aedbdb0dd28", "object": "chat.completion |
| E | emoji + CJK + rtl | 200  1.72s  {"id": "chatcmpl-19fd647c592a4c7b969f429a", "object": "chat.completion |
| E | prompt just under budget | 400  0.06s  {"error": {"message": "prompt too long: about 10890 tokens for a 4096- |
| E | prompt just over budget | 400  0.06s  {"error": {"message": "prompt too long: about 10956 tokens for a 4096- |
| E | CJK prompt near budget | 400  0.06s  {"error": {"message": "prompt too long: about 10922 tokens for a 4096- |
| E2 | not JSON | 400  0.07s  {"error": {"message": "invalid JSON body: Expecting value: l |
| E2 | JSON array | 400  0.06s  {"error": {"message": "missing 'messages' in request body", |
| E2 | JSON string | 400  0.06s  {"error": {"message": "missing 'messages' in request body", |
| E2 | truncated JSON | 400  0.06s  {"error": {"message": "invalid JSON body: Expecting property |
| E2 | empty body | 400  0.06s  {"error": {"message": "empty request body", "type": "invalid |
| E2 | deeply nested | 400  0.06s  {"error": {"message": "no message with usable text: every me |
| E2 | 8MB body | 400  0.02s  {"error": {"message": "request body larger than 8388608 byte |

### 并发 → 吞吐

| 并发 | 请求 | 成功 | 失败 | 耗时(s) | QPS | p50(s) | p90(s) | p99(s) | 聚合tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 24 | 24 | 0 | 25.74 | 0.93 | 1.07 | 1.09 | 1.09 | 0.9 |
| 2 | 24 | 24 | 0 | 25.52 | 0.94 | 1.60 | 2.14 | 2.16 | 0.9 |
| 4 | 24 | 24 | 0 | 25.90 | 0.93 | 2.71 | 4.31 | 4.37 | 0.9 |
| 8 | 24 | 24 | 0 | 26.52 | 0.91 | 4.97 | 8.43 | 8.98 | 0.9 |
| 16 | 24 | 24 | 0 | 27.81 | 0.86 | 7.43 | 15.75 | 18.19 | 0.9 |

## 关键缺陷：max_tokens 参数不被遵守

| 请求 max_tokens | 实际输出 tokens | 倍数 |
| --- | --- | --- |
| 8 | 10 | 1.25x |
| 16 | 20 | 1.25x |
| 64 | 97 | 1.52x |
| 128 | 200 | 1.56x |
| 256 | 370 | 1.45x |

finish_reason 均为 stop（自然结束）。客户端无法用 max_tokens 可靠约束输出长度，估算超时需留 1.5–1.6 倍余量。

> 

## 仿真测试

| 场景 | 请求数 | 成功 |
| --- | --- | --- |
| handshake | 6 | 6 |
| conversation | 8 | 8 |
| paste | 8 | 6 |
| interruption | 2 | 2 |
| session | 3 | 3 |
| background | 3 | 3 |
| mixed | 12 | 12 |

- 服务最终状态：存活

- 长文档场景：800/3000 字符成功，8000/16000 字符被 HTTP 400 拒绝（超上下文）

## 稳定性事件：一次并发下的进程崩溃

- 触发时机：首轮仿真 第3步『后台标题』——标题请求与带历史的对话请求并发

- 现象：两个请求均返回 RemoteDisconnected，随后 /health 连接被拒绝（进程死亡）

- 恢复：守护进程自动重启，\<30s 恢复服务

- 是否复现：未复现（15 次并发组合实验）

- 备注：首次崩溃时服务已运行约 1130s、处理约 280 次请求；后续 5 类共 15 次并发组合实验均未复现，判定为低概率偶发

## 发现与建议

| 级别 | 发现 | 影响 | 建议 |
| --- | --- | --- | --- |
| 高 | 无背压限流，并发请求全部排队 | 高峰时客户端挂起直至超时 | 网关加并发闸门与队列上限，超限快速失败 |
| 高 | max_tokens 不生效（超出 25%–56%） | 输出长度/延迟不可控 | 服务端修复；客户端按 1.6 倍估算超时 |
| 中 | 低概率并发崩溃（1 次，未复现） | 极端负载下服务中断 | 保留守护自动重启；加存活告警与重试 |
| 中 | 长文档粘贴被拒（8000+ 字符 → 400） | 真实客户端粘贴长文报错 | 限制粘贴长度或客户端分块摘要 |
| 中 | 上下文窗口仅 4096 tokens | 多轮长对话很快触顶 | 调整量化档位或做历史压缩 |
| 中 | 参数不校验（0/-1/2.0 均 200） | 非法参数被静默接受 | 服务端加参数范围校验 |
| 低 | 内存占用 4.4G/8G（66%） | 内存余量有限 | 监控内存，避免同机部署其他大内存服务 |
| 低 | 并发无吞吐增益 | 扩容需加设备 | 按 QPS≈0.9 规划容量，多设备横向扩展 |
