# rsync-backup.sh 说明书

## 一、定位

`rsync-backup.sh` 是 rsync 硬链接快照备份脚本：对若干源目录做定时快照备份。每个源在 `BACKUP_BASE/<名称>/` 下按时间戳保留一份完整快照；未改动的文件通过 `--link-dest` 硬链接到上一份成功快照，不占双倍空间。本目录的 harness / e2e / 三个 mock 均为它的测试基建（见文末附录）。

## 二、运行方式

典型 cron（每天 03:00）：

```
0 3 * * * /usr/local/sbin/rsync-backup.sh
```

- 必须以 root 运行（`REQUIRE_ROOT=1`）：`-aHAX --numeric-ids` 在非 root 下会静默丢 owner/ACL/xattr。

- 退出码：`0` 全部成功；`1` 存在失败或警告（用于 cron 报警）。单个源失败不耽误后续源。

## 三、依赖

bash 4.4+（更低版本 `set -u` 遇空数组直接退出）、GNU find（`-printf`）、GNU df（`--output`）、GNU stat、rsync、flock（util-linux）、mountpoint、nice；有 `ionice` 时自动加用（仅 CFQ/BFQ 有效）。脚本启动时对以上全部做存在性自检，缺任一命令直接退出。

## 四、可配置项（只改脚本头部配置区）

核心路径与保留策略

- `BACKUP_BASE`：快照根目录，必须挂在独立磁盘上（配合 `MOUNT_CHECKS` 防止写到系统盘）。

- `KEEP=8`：每源保留的「成功」快照数，超出从最旧删除。硬链接占 inode，机械盘上可能先耗尽 inode 而非空间。

- `FAILED_KEEP_DAYS=14` / `FAILED_KEEP_COUNT=2`：失败目录（`*.failed.*`）按天与按数量双淘汰。

rsync 行为

- `BW_LIMIT=""`：带宽上限（SMR/归档盘建议 `"30M"`）。

- `RSYNC_TIMEOUT=0`：I/O 超时；本地机械盘保持 0（SMR 卡顿会被误杀），NFS 可设 300。

- `ONE_FILE_SYSTEM=1`：对应 `rsync -x`，不跟随源上的其他挂载点。

- `EXCLUDES`：rsync 排除规则数组，元素必须是完整参数（不要漏 `--exclude=`）。

安全防护开关（值非法会拒绝启动，绝不静默关闭）

- `REQUIRE_ROOT=1`：非 root 退出。

- `ALLOW_ROOT_SRC=0`：禁止把 `/` 当源。

- `REJECT_SRC_ON_ROOTFS=1`：源与 `/` 同设备号则拒绝（拦「未挂载的空挂载点 + --delete 清空历史」）。btrfs 子卷与父卷同 st_dev，此场景置 0 改由 `MOUNT_CHECKS` 把关。

- `CHECK_SRC_EMPTY=1`：源一层目录为空（忽略 lost+found）则拒绝。

- `SHRINK_RATIO=10`：源文件数少于上次快照的 1/N 则拒绝（防挂错盘后做出空快照、旧快照被淘汰）。

- `MIN_FREE_PCT=5` / `MIN_FREE_INODE_PCT=5`：备份盘空间/inode 低于阈值不开跑；逐源检查时空间不足跳过当前源、后续源继续。

- `CHMOD_SNAPSHOT_RO=1`：成功快照顶层去写权限（只 chmod 顶层不递归；防不住 root 自身，仅防其他身份）。

rsync 退出码分类

- `OK_RSYNC_CODES=(0 24)`：0 完全成功；24 传输中源文件消失（活跃系统常见，可接受）→ 快照照常转正。

- `WARN_RSYNC_CODES=(23)`：部分文件失败 → 快照保留但整体退出码 1，便于 cron 报警。

- 其余退出码 → 快照改名 `.failed.<pid>`，不占用正式快照名额。

源与检查配置

- `SOURCES`：每项 `源目录|备份子目录名`。NAME 只允许 `[A-Za-z0-9._-]`、禁止点开头（避免与 `.inprogress` 混淆）、禁止重复（重复会互相淘汰快照，启动即拦）。

- `MOUNT_CHECKS`：必须已挂载的路径列表（填 findmnt 能看到的真实挂载点，不填其子目录）。

- `LOG` / `LOCK` / `LOG_MAX_SIZE` / `LOG_KEEP_DAYS`：主日志路径、按大小轮转与按天清理。

## 五、快照目录布局

```
BACKUP_BASE/<NAME>/
2026-10-16_030000/        正式快照（含 .complete 标记，顶层 a-w）
2026-10-16_030000.failed.1234/   失败/多余残留，按天+数量清理
.2026-10-16_030000.inprogress.1234/   正在写入的工作目录
（快照内 .rsync-partial/ 为 rsync 半截文件目录，转正前删除）
```

## 六、崩溃恢复语义（重点）

- 上次崩溃留下的 `.*.inprogress.*`：已含 `.complete` → 提升为正式快照（目标已存在则改判 failed）；未完成 → 按文件数挑最厚的一份续传（不是按时间戳——最新的一份可能才跑 5%，更早的可能已完成 90%），其余改判 failed。

- EXIT trap 保证中途被杀时在写目录被改名 `.failed`，退出码保留原值（INT 130 / TERM 143）。

---

## 附录：测试基建使用说明

本脚本由两层测试覆盖：函数级隔离测试（`harness.sh`，不跑主流程，直接抽取脚本函数逐场景验证）与端到端场景测试（`e2e.sh`，生成「仅配置不同」的测试副本跑完整主流程）。各测试脚本的详细说明见其各自说明书；此处保留原测试 README 的使用要点与结论。

### 使用方式

1. 把三个 mock 放入 PATH 首位（在没有真实 `rsync`/`flock`/`mountpoint` 的环境中使用；有真实命令的环境可不用 mock，把 `bin/` 从 PATH 去掉即可）。

1. 修改 `e2e.sh` 顶部的 `SB`（沙箱根目录）与 `SRC`（被测脚本路径）两个变量。

1. `bash harness.sh` 跑函数级测试；`bash e2e.sh` 跑端到端全量（小数据量约十几分钟）。

1. e2e 会自动生成「仅配置不同」的测试副本，并以「行数一致 + 函数区逐字节一致」双重自证逻辑零改动，差异行只涉及配置区。

### 覆盖场景

函数级（53 项断言）：rsync 退出码分类（0/24→ok、23→warn、其他→fail）、文件计数（含 `-xdev`）、完整快照列举（只认 `.complete`、新的在前、排除 failed/inprogress）、残留目录处理（按完成度选最厚一份续传、单份免统计、已 complete 提升、目标已存在改判 failed）、快照淘汰（KEEP）、失败目录清理（数量封顶 + 超龄）、日志轮转（大小 + 超龄）、空间检查（软失败 / 解析失败留痕 / inode 不误报）、失败标记、EXIT trap 清理（保留原退出码）、backup_one 输入校验（坏 NAME、SRC=/、源不存在、空源、三段配置）。

端到端（101 项断言）：首次完整备份、二次运行硬链接去重（inode 相同）、增删改后快照不可变性、KEEP 淘汰、SHRINK_RATIO 拒绝（且不影响后续源）、掉电窗口（已 complete 未改名 → 提升）、崩溃续传（按完成度挑选）、rsync 退出码 23（warn，快照保留）与 11（fail，标记 .failed）、启动自检与输入校验 12 种、并发互斥（第二个实例退出）、日志轮转与超龄清理、限速/超时参数变体。

### 环境注意事项

- 被测脚本的目标运行环境是 Linux（bash 4.4+、GNU find/df、rsync、flock、mountpoint、以 root 运行）。

  - MSYS bash 的 `command -v` 不解析 Windows 风格（`C:/...`）的 PATH 项，沙箱路径需用 MSYS 风格（如 `/tmp/...`），否则启动自检误报「缺少命令」；

  - MSYS 的 `stat -c %h` 在 link+unlink 循环后会虚高，硬链接数以 `find -inum` 实测为准；

  - `chmod a-w` 在 MSYS 下不生效，快照只读化（CHMOD_SNAPSHOT_RO）的效果需在 Linux 目标机验证。

- 测试基建在 MSYS2 下开发验证。以下为 MSYS 平台特性，不是被测脚本的问题，在 Linux 上不存在：

- 使用 mock rsync 时，真实 rsync 的传输语义（权限/ACL/xattr 保留、`-aHAX --numeric-ids` 的实际行为）需在 Linux 目标机另行验证；mock 只保证脚本状态机与参数构造正确。

### 测试结论与版本记录

- 测试已完成收尾：154 项断言（函数级 53 + 端到端 101）全部通过，未发现脚本缺陷。静态走查另发现 1 项 P2 遗漏：启动自检的命令清单缺少 date、sleep、wc、tail、dirname、basename 六个命令，其中 tail 缺失会令空间检查被静默跳过；建议补丁与两项 P3 加固建议见测试报告第四节。

- e2e.sh 已于 2026-10-15 更新：newlog 改为 inode 感知（修复日志轮转后本轮日志断言失效）、E6.6 改为前后差分选取快照（修复硬编码日期问题）。harness.sh、三个 mock 主体内容不变。
