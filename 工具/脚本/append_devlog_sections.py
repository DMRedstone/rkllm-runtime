#!/usr/bin/env python3
"""向重组后的《开发记录》追加三块尚未入档的内容。

1. §7.13 真机安装与缓存链路闭环（2026-10-22 21:00 ~ 22:15）；
2. 附录 B 《十五版本以来改造审计报告》结论存档（原报告 §一~§七 浓缩）；
3. 附录 C 待办事项核对清单（原 11 项 + 当日新增 4 项，逐项附代码证据）。
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path("文档/开发记录.md")

SECTION_713 = """
---

## 7.13 真机安装与缓存链路闭环（2026-10-22 21:00 ~ 22:15）

### 一、打包与上板

`make_deb.py --arch arm64 --revision 16` 产出
`rkllm-runtime_1.3.1-16_arm64.deb`（13.98 MB，441 文件），
`verify_deb.py` **86 项校验 FAIL=0**，MD5 与板上一致。

**踩坑记录（重要）**：首次 `dpkg -i` 被中断后进入半装状态（`iU`）。
此时若只执行 `dpkg --configure -a`，**它只配置已解包的内容，不会重新解包**
——板上于是停留旧 deb（表现为 `Maintainer` 还是旧值）。换 deb 后必须重跑：

```
sudo dpkg -i --force-confdef --force-confold <deb> </dev/null
```

`--force-confold` 保留本地 conffile（`/etc/default/rkllm-server` 被本地改过，
含实际生效的 `RKLLM_MODEL` / `RKLLM_API_KEY`，包方版本这两项是空值）；
`</dev/null` 防止 dpkg 进 conffile 交互提示后挂死。

### 二、根路径兼容修复的由来与实施

用户实测反馈：`http://10.0.0.14:8080`（不带 `/v1`）不能正常获取模型。
根因是 OpenAI 端点全部挂在 `/v1` 前缀下，而不少客户端（OpenWebUI /
LobeChat / 各类 SDK）把 base_url 配成 `http://host:port`，请求
`/models`、`/chat/completions` 全部落框架 404。

实施见 §7.12。真机复核：`/`、`/models`、`/chat/completions` 均 200，
`/v1/*` 不回归，`/nope` 仍 404，`/ready` 鉴权豁免不被别名破坏。

### 三、缓存链路实测结论

配置写入 `/etc/default/rkllm-server`（追加两键，其余不动）：

```
RKLLM_CACHE_ENABLED=true
RKLLM_CACHE_DIR=/dev/shm/rkllm/prefix
```

| 观测项 | 结果 |
|---|---|
| worker 命令行 | `--cache-dir /dev/shm/rkllm/prefix --cache-max-mb 512 --cache-ttl-s 604800` |
| 缓存文件 | `prefix.cache` 80,017,336 B，mtime 21:48:37 后**从未变化** |
| 前缀元数据 | `prefix.meta` 每轮更新（22:11:14） |
| hits / misses | **8 / 1**（89%；miss 是建立缓存的第一轮） |
| eMMC 写入 | **零** —— `/var/cache/rkllm/prefix` mtime 仍为 19:17:57 |
| 服务稳定性 | `NRestarts=0`，无 `rkllm_run -1/10000`，无熔断 |

**缓存只在多轮对话（`messages` 含 ≥2 条）时才启用**：`agent/pipeline.py`
的 `cache_hint = len(messages) >= 2`。单轮请求不建目录、不写缓存，这是设计
而非故障——单轮没有可复用前缀，存了也永不命中。

**为什么能命中**：真实客户端（deepseek-harness）每轮带完整历史（user/
assistant 成对）、稳定的 tools 列表（渲染为 prompt 头部 `tools_json`）、
稳定的 runtime context（仅 cwd 等固定信息），prompt 严格前缀递增，
`prepare()` 的 `prompt.startswith(meta)` 判定成立。本地实测确认
`p2.startswith(p1) == True`。

### 四、两处被实测推翻的错误结论（如实记录）

1. **「hits 仍为 0」错误**：该数据引用自 21:40 重启前的旧 stats。worker
   重启后 `PromptCache` 计数器归零并重写 `stats.json`，旧值已失效；把
   "21:48–21:50 期间 stats 停更"误判成熔断，实际那是单轮请求
   （`cache_hint=False`）不写 stats。
2. **「转录式全量重渲染导致前缀不成立」错误**：`conversation.build_prompt`
   末尾固定的 `"Assistant:"` 在第 N+1 轮恰好构成第 N 轮 prompt 的前缀，
   结构上天然支持复用。真正破坏前缀的只有「多条独立 user」与
   「system 每轮变化」两种客户端行为，本例均不属于。

由此，原拟把 `cache_hint` 改为「要求存在 assistant 历史」的修订**撤销**：
客户端本来就带完整历史，`misses: 1` 是必要开销，改了没有收益。

### 五、eMMC 残留清理

按用户指示删除 `/var/cache/rkllm/prefix/` 下故障遗留的 3 个文件
（`prefix.cache` 55.7 MiB + `prefix.meta` 194 B + `stats.json` 43 B），
释放 56 MiB，eMMC 根分区使用率降至 16%。该路径不受 dpkg 跟踪（运行时产物），
父目录保留。删除后服务仍 active、`NRestarts=0`、tmpfs 缓存完好。

> 创建日期：2026-10-22

"""

APPENDIX_B = """
---

## 附录 B 《十五版本以来改造审计报告》结论存档

> 原报告 `文档/十五版本以来改造审计报告.md`（2026-10-21）的结论部分。
> 其 §八「整改执行记录」已按时间线并入 §7.2 起各章，此处存档 §一~§七 的
> 分析结论，便于不回原文件即可掌握审计判断。

### B.1 结论摘要

**代码实现度约七成半，工程质量高，但不具备发布条件。**

1. 主体架构已按设计落地且质量良好：worker 子进程隔离、UDS 帧协议、单槽
   背压队列、supervisor 自愈、四版 MCP 协议矩阵、6 工具、job 状态机、
   80 键配置表、能力单一事实源 `caps.py`。
2. 三个 P0 级目标未实现，且都不是「未开工」，而是链路断在半途：服务端
   鉴权整体缺失（N4 回归）、取消传导断裂（F5 不可用）、内存预算与档位
   （N10/F12）完全未实现。
3. 「73 项 pytest 全绿」在最新提交上已不成立：ready 帧从 stdout 改到连接后，
   `test_worker_frames.py` 仍从 stdout 读，导致 worker 侧 9 项测试不可运行。

### B.2 需求达成矩阵（审计时点判定）

| 维度 | 判定分布 |
|---|---|
| 设计目标 G1–G7 | 基本达成 2、部分 4、**未达成 1**（G3 档位） |
| 功能需求 F1–F14 | ✅ 5、部分 6、**未达成 3**（F5 取消、F8 批量隔离、F12 档位/预算） |
| 非功能 N1–N11 | ✅ 1、部分 7、**未达成 2**（N4 安全、N10 内存预算）、**未测 1**（N3 性能基线） |
| 阶段 DoD S0–S6 | 基本达成 2、部分 3、**未开始 1**（S0） |

### B.3 发布门槛判定（设计 §9.2，8 条）

整改前：明确通过 1 条，部分满足 1 条，其余 6 条未通过或未验证。
2026-10-21 夜真机后：**通过 5 条**（工具列表/短长任务/取消、TTFT 0.80 s、
kill -9 自愈 21.1 s、config show 80 键、日志无 key 与 401），**部分 3 条**
（三档位仅 4B 实测、一致性以握手测试替代、干净系统装通未做）。

### B.4 缺陷清单与最终状态

| ID | 缺陷 | 状态 |
|---|---|---|
| P0-1 | 服务端鉴权整体缺失 | **关闭**（§7.2） |
| P0-2 | 取消传导断裂（三处断点） | **关闭**；真机 5.0 s 内 NPU 停（§7.3） |
| P0-3 | 内存预算与模型档位未实现 | **关闭**（§7.2） |
| P1-4 | 约 32 个配置键无生效路径 | **部分关闭**——接线 26 键；`cache.*` 已于 §7.7 补接线并迁 tmpfs |
| P1-5 | 测试套件在最新提交上挂起 | **关闭**（§7.2） |
| P1-6 | 批量任务单段失败中断整批 | **关闭**（§7.2） |
| P2-7 | 结构化抽取缺 schema 校验 | **关闭**（§7.2，新增零依赖校验器） |
| P2-8 | 能力表与实际行为不一致 | **关闭**（§7.2） |
| P2-9 | SSE 心跳未接线 | **关闭**（§7.2） |
| P2-10 | 进度通知语义弱 | 保留（设计定位为辅助手段） |
| P2-11 | 同步调用产生永不终结的伪 job | **关闭**（§7.2 改 CancelFlag） |
| P2-12 | 运维命令与回滚开关缺失 | **关闭**（§7.2） |
| P3-13 | 文档重复段落 | **关闭**（本次重组进一步消除章节混乱） |
| P3-14 | 死代码 | 未处理（不影响行为） |
| P3-15 | 客户端 `target_timeout_s` 不生效 | **关闭**（§2.13） |

### B.5 与设计基线的偏离（需回写设计或修实现）

| 项 | 设计 | 实现 | 处置 |
|---|---|---|---|
| 依赖路线 | vendor wheel | apt Depends | 已拍板走 apt，设计 §5.2/§10 R7 未回写 |
| `json_schema` | emulated | 恒 reject → 非流式 emulated（§7.7） | 已收敛为实现侧 |
| `reserves` 端点 | 未列 | 曾声明 emulated 无路由 | 改 `unsupported`（§7.2） |
| 一致性脚本 | 自写 | `test_mcp_protocol.py` + `mcp_conformance.py`（§7.7） | 可接受，需登记替代 |
| S5 配置完备 | 键覆盖率 100% | 计数 100%，生效约 60% → 大幅改善 | P1-4 |

### B.6 建议行动顺序（审计提出，现已全部完成或明确后置）

- **发布前必修**：补鉴权、接通取消、实现内存预算与档位 —— 均已完成。
- **发布前应修**：修 `test_worker_frames.py`、batch 隔离、配置键处置 —— 已完成。
- **发布前需补验证**：S0 spike、golden 回放、N3 性能基线、干净系统装通、
  WorkBuddy 取消验收 —— 已完成或在 §7.8/§7.9 落实。
- **可后置**：extract schema 校验（已做）、SSE 心跳（已做）、真实进度、
  伪 job 清理（已做）、`config init`（已做）、文档去重（本次重组）、
  死代码清理（仍未做）。

### B.7 对「实现思路」的评价（非缺陷，工程约定）

1. worker 单读线程分发模型正确，且注释写清了「为什么这样」。
2. ready 帧改走 UDS 是必要的真机修复（C 库日志污染 stdout）。
3. `caps.py` / `config.py` 单表驱动让「文档漂移」在结构上难以发生。
4. 能力收紧（json_schema/tool_choice）判断本身合理——4B 无受约束解码时
   假装支持比明确拒绝更难排查。
5. 45 s 阈值 + 真实公文负载校准（54/78/39 s）是本批最有说服力的验证。

**教训（已写入工程约定）**：本地全绿只能证明逻辑自洽，不能证明协议/驱动/
平台行为；取消类功能必须以「NPU 是否停」为准；任何 `auto` 型配置项必须有
显式解析与日志回显。

> 创建日期：2026-10-22（原报告创建于 2026-10-21）

"""

APPENDIX_C = """
---

## 附录 C 待办事项核对清单

> 原 `文档/待办事项.md` 的 11 项 + 当日新增 4 项，逐项附代码证据。
> 原文件已按 2026-10-21 约定保留为占位。

### C.1 原待办 11 项（2026-10-22 审计修复项）

| # | 事项 | 状态 | 证据 |
|---|---|---|---|
| 1 | prompt 前缀缓存默认落 tmpfs | ✅ | `config.py` 默认 `/dev/shm/rkllm/prefix`；板上 `prefix.cache` 80 MB 落内存 |
| 2 | 请求级错误走 `RequestFailed`，不重启 worker | ✅ | `supervisor.py` 5 处 `RequestFailed`；真机 `NRestarts=0` |
| 3 | 连接器 `load_cfg()` env 覆盖 TOML | ✅ | `__main__.py` TOML 先加载、env 后覆盖 |
| 4 | 补 `session_max_messages` 默认键 | ✅ | `default_cfg()` 含该键 |
| 5 | `tools/call` 入池前预注册 `CancelFlag` | ✅ | `stdio.py` reader 入池前写 `_inflight[key] = flag` |
| 6 | `/v1/completions` 接入 caps 检查 | ✅ | `openai.py` 2 处 `check_request_params` |
| 7 | Responses 非流式复用 json_schema 重试 | ✅ | `responses.py` 2 处 `_run_blocking_checked` |
| 8 | Responses 流式接入断连 watcher | ✅ | `responses.py` 4 处 `receive()` / watcher |
| 9 | `HttpBackend.status()` 改用 urljoin | ✅ | `backend.py` 2 处 `urljoin` |
| 10 | 版本号统一 `1.3.1-16` | ✅ | `pyproject.toml` / `__init__.py` / `app.py` / `cli.py` |
| 11 | N11 魔法数基线更新 | ✅ | `magic_baseline.json` 随行号更新，全量测试通过 |

### C.2 当日新增 4 项

| # | 事项 | 状态 | 证据 |
|---|---|---|---|
| 12 | 根路径兼容（base_url 不带 `/v1`） | ✅ | `app.py` `root_alias` 中间件；真机 `/`、`/models`、`/chat/completions` 均 200（§7.12） |
| 13 | deb 打包上板实际安装 | ✅ | `verify_deb.py` 86 项 FAIL=0；板上 `dpkg -s` 显示 `install ok installed`（§7.13） |
| 14 | 缓存链路闭环验证 | ✅ | hits 8 / misses 1；eMMC 零写入；无 `-1/10000`（§7.13） |
| 15 | eMMC 残留缓存清理 | ✅ | 删除 3 个文件释放 56 MiB；服务仍 active（§7.13） |

### C.3 测试状态

- 全量 `pytest 247 passed`（含根路径兼容新增 7 例）。
- 本地 240 → 247，唯一 warning 为测试故意构造的 `QueueBusy` 线程异常。
- 板端 8 个 chat 请求全部 200，`/v1/*` 与根路径双通道均可用。

> 创建日期：2026-10-22

"""


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines()

    # §7.13 插在 "# 附录" 分隔标题之前
    marker = "# 附录"
    idx = next((i for i, l in enumerate(lines) if l == marker), None)
    if idx is None:
        print("[失败] 未找到附录分隔标题")
        return 1
    insert_at = idx
    # 回退到该标题前的 "---" 分隔线，保持格式整齐
    while insert_at > 0 and lines[insert_at - 1].strip() in ("", "---"):
        insert_at -= 1
    new_lines = (lines[:insert_at]
                 + SECTION_713.strip("\n").splitlines()
                 + ["", "---", ""]
                 + lines[insert_at:])
    print(f"§7.13 已插入于第 {insert_at + 1} 行前")

    SRC.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # 附录 B / C 追加到文末
    with SRC.open("a", encoding="utf-8") as fh:
        fh.write(APPENDIX_B)
        fh.write(APPENDIX_C)
    print("附录 B、C 已追加到文末")

    total = len(SRC.read_text(encoding="utf-8").splitlines())
    print(f"文档现共 {total} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
