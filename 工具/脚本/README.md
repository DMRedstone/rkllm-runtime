# 脚本目录说明

本目录收录 rsync 硬链接快照备份脚本（`rsync-backup.sh`）及其测试基建，以及运行时相关的辅助脚本。本 README 只做总览：每个脚本的功能见下表，详细用法见同目录的说明书。

## 脚本一览

| 文件 | 功能 |
| --- | --- |
| `README.md` | 本说明：目录总览与说明书索引 |
| `rsync-backup.sh` | rsync 硬链接快照备份主脚本（其余备份相关脚本均为其测试基建） |
| `harness.sh` | rsync-backup 函数级隔离测试（T1–T11，53 项断言） |
| `e2e.sh` | rsync-backup 端到端场景测试（E1–E11，101 项断言） |
| `mock-rsync.sh` | 测试用 rsync 模拟器（覆盖被测脚本全部实际参数，语义对齐真实 rsync） |
| `mock-flock.sh` | 测试用 flock 桩（MOCK_FLOCK_BUSY=1 模拟锁占用） |
| `mock-mountpoint.sh` | 测试用 mountpoint 桩（恒返回 0） |
| `magic_numbers.py` | 魔法数守卫：扫描源码中的裸数字，与 `tests/magic_baseline.json` 比对 |
| `coverage_report.py` | 按分组口径统计 pytest 覆盖率，产出可读报告 |
| `mcp_conformance.py` | MCP 四版本协议一致性脚本（N6 验收） |
| `f3_scenarios.py` / `f10_frameworks.py` | F3 七类场景与 F10 tools 闭环的真机验收脚本 |

## 脚本说明书

每个脚本的详细说明（用法、配置、注意事项、测试覆盖）见同目录下的 `* 说明.md`：

| 脚本 | 本地文档 |
| --- | --- |
| `rsync-backup.sh` | `rsync-backup 说明.md` |
| `harness.sh` | `harness 说明.md` |
| `e2e.sh` | `e2e 说明.md` |
| `mock-rsync.sh` | `mock-rsync 说明.md` |
| `mock-flock.sh` | `mock-flock 说明.md` |
| `mock-mountpoint.sh` | `mock-mountpoint 说明.md` |
| `coverage_report.py` | `coverage_report.py 说明.md` |
| `magic_numbers.py` | `magic_numbers.py 说明.md` |
| `mcp_conformance.py` | `mcp_conformance.py 说明.md` |
| `f3_scenarios.py` | `f3_scenarios.py 说明.md` |
| `f10_frameworks.py` | `f10_frameworks.py 说明.md` |

> 本目录的脚本与说明均只引用仓库内文件；开发记录维护脚本已随记录重写而退役。
