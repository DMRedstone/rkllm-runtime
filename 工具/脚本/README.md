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

## 脚本说明书

每个脚本的详细说明（用法、配置、注意事项、测试覆盖）见下列文档：

- [20261016 rsync-backup 说明（2026.10.16）](https://www.workbuddy.cn/space/d/C11o85Qmmr3iBf55tGaW9K)：主脚本功能、全部可配置项、防护机制、崩溃恢复语义，附测试基建使用说明（原 README 内容迁移于此）

- [20261016 harness 说明（2026.10.16）](https://www.workbuddy.cn/space/d/rW01idj78d4XfDWt1jOtPk)：函数级隔离测试的方法、自证机制与 T1–T11 覆盖明细

- [20261016 e2e 说明（2026.10.16）](https://www.workbuddy.cn/space/d/orxiHZawJs4zHeZUpMDB0Q)：端到端测试的副本生成、mock 注入与 E1–E11 场景明细

- [20261016 mock-rsync 说明（2026.10.16）](https://www.workbuddy.cn/space/d/ovkWohcRpZrYwRTfoUBeXE)：rsync 模拟器的语义对齐点与 4 个环境变量

- [20261016 mock-flock 说明（2026.10.16）](https://www.workbuddy.cn/space/d/DU8t2hqiObQD9OZA88piz6)：flock 桩行为与并发场景用法

- [20261016 mock-mountpoint 说明（2026.10.16）](https://www.workbuddy.cn/space/d/SVXUaC0pw5E1iDF6TPdh5E)：mountpoint 桩的定位与测试盲区

- [20261016 space_sync_check 说明（2026.10.16）](https://www.workbuddy.cn/space/d/hNQJTOyBmUqrrNwQ8svtTc)：资料库 ↔ 本地目录双向比对

- [20261016 space_name_check 说明（2026.10.16）](https://www.workbuddy.cn/space/d/6JXulw9ttROS0amUr1TRIE)：按名称核对与数量统计

2026-10-16 说明：本文档原为 rsync-backup 测试基建说明，当日重定位为目录 README；原有使用方式、覆盖场景、环境注意事项与测试结论等内容已迁入 rsync-backup 说明书附录。
