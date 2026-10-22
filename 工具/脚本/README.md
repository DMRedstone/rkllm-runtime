# 脚本目录说明

本目录收录 rsync 硬链接快照备份脚本（`rsync-backup.sh`）及其全部测试基建，以及资料库检查工具（`space_sync_check.py` / `space_name_check.py`）。本 README 只做总览：每个脚本的功能见下表，详细用法、配置项、注意事项与测试覆盖见文末「脚本说明书」索引对应文档。

## 脚本一览

| 文件 | 功能 |

| --- | --- |

| `README.md` | 本说明：目录总览与说明书索引 |

| `rsync-backup.sh` | rsync 硬链接快照备份主脚本（本目录核心，其余备份相关脚本均为其测试基建） |

| `harness.sh` | rsync-backup 函数级隔离测试（T1–T11，53 项断言） |

| `e2e.sh` | rsync-backup 端到端场景测试（E1–E11，101 项断言） |

| `mock-rsync.sh` | 测试用 rsync 模拟器（覆盖被测脚本全部实际参数，语义对齐真实 rsync） |

| `mock-flock.sh` | 测试用 flock 桩（MOCK_FLOCK_BUSY=1 模拟锁占用） |

| `mock-mountpoint.sh` | 测试用 mountpoint 桩（恒返回 0） |

| `space_sync_check.py` | 资料库空间 ↔ 本地目录双向比对（报缺、多、内容不同步） |

| `space_name_check.py` | 资料库节点 ↔ 本地目录按名称核对与数量统计 |

| `space_rename_strip.py` | 批量剥离去序号前缀后残留的旧名引用 |

| `magic_numbers.py` | 魔法数守卫：扫描源码中的裸数字，与 `tests/magic_baseline.json` 比对 |

| `coverage_report.py` | 按分组口径统计 pytest 覆盖率，产出可读报告 |

| `mcp_conformance.py` | MCP 四版本协议一致性脚本（N6 验收） |

| `f3_scenarios.py` / `f10_frameworks.py` | F3 七类场景与 F10 tools 闭环的真机验收脚本 |

| `restructure_devlog.py` | 开发记录切片重组（按行号映射重排章节，带零丢失校验） |

| `renumber_devlog.py` | 开发记录章编号统一为「卷.章」形式 |

| `append_devlog_sections.py` | 向开发记录追加新章节/附录 |

## 脚本说明书

每个脚本的详细说明（用法、配置、注意事项、测试覆盖）见下列文档；本地副本与
云端链接一一对应，任选其一：

| 脚本 | 本地文档 | 云端 |
| --- | --- | --- |
| `rsync-backup.sh` | `rsync-backup 说明.md` | [20261016 rsync-backup 说明](https://www.workbuddy.cn/space/d/C11o85Qmmr3iBf55tGaW9K) |
| `harness.sh` | `harness 说明.md` | [20261016 harness 说明](https://www.workbuddy.cn/space/d/rW01idj78d4XfDWt1jOtPk) |
| `e2e.sh` | `e2e 说明.md` | [20261016 e2e 说明](https://www.workbuddy.cn/space/d/orxiHZawJs4zHeZUpMDB0Q) |
| `mock-rsync.sh` | `mock-rsync 说明.md` | [20261016 mock-rsync 说明](https://www.workbuddy.cn/space/d/ovkWohcRpZrYwRTfoUBeXE) |
| `mock-flock.sh` | `mock-flock 说明.md` | [20261016 mock-flock 说明](https://www.workbuddy.cn/space/d/DU8t2hqiObQD9OZA88piz6) |
| `mock-mountpoint.sh` | `mock-mountpoint 说明.md` | [20261016 mock-mountpoint 说明](https://www.workbuddy.cn/space/d/SVXUaC0pw5E1iDF6TPdh5E) |
| `space_sync_check.py` | `space_sync_check 说明.md` | [20261016 space_sync_check 说明](https://www.workbuddy.cn/space/d/hNQJTOyBmUqrrNwQ8svtTc) |
| `space_name_check.py` | `space_name_check 说明.md` | [20261016 space_name_check 说明](https://www.workbuddy.cn/space/d/6JXulw9ttROS0amUr1TRIE) |
| `space_rename_strip.py` | `space_rename_strip 说明.md` | — |
| `coverage_report.py` | `coverage_report 说明.md` | — |
| `magic_numbers.py` | `magic_numbers 说明.md` | — |
| `mcp_conformance.py` | `mcp_conformance 说明.md` | — |
| `f3_scenarios.py` | `f3_scenarios 说明.md` | — |
| `f10_frameworks.py` | `f10_frameworks 说明.md` | — |
| `restructure_devlog.py` | `restructure_devlog 说明.md` | — |
| `renumber_devlog.py` | `renumber_devlog 说明.md` | — |
| `append_devlog_sections.py` | `append_devlog_sections 说明.md` | — |

2026-10-16 说明：本文档原为 rsync-backup 测试基建说明，当日重定位为目录 README；原有使用方式、覆盖场景、环境注意事项与测试结论等内容已迁入 rsync-backup 说明书附录。
