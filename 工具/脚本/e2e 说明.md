# e2e.sh 说明书

## 一、定位

`e2e.sh` 是 `rsync-backup.sh` 的端到端场景测试（E1–E12，脚本内 1–13 节，101 项断言）：生成「仅配置不同」的被测副本，跑完整主流程，验证从启动自检到快照落盘的整条状态机。

## 二、运行方式

```
bash e2e.sh [FROM] [TO]    # 场景编号 1..13，默认 1..99 全量
```

- 分段运行（TO\<13）仅用于排查；结论以 FROM=1 的完整运行为准（前段场景为后段提供状态）。

- 全量耗时取决于文件操作速度，小数据量约十几分钟。FAIL 非 0 时退出码非 0。

## 三、测试机制

- 副本生成：`gen_copy` 用 awk 对被测脚本做 1:1 行替换，只改配置区（`BACKUP_BASE`/`LOG`/`LOCK`/`SOURCES`/`REQUIRE_ROOT` 等），行数不变；附加 `sed` 覆盖指定键值。双重自证：副本与原件行数一致 + 函数区（按 `# 基础工具函数` 标记动态定位，非写死行号）逐字节一致——逻辑零改动，差异行只涉及配置区。

- mock 注入：把 mock（mock-rsync、mock-flock、mock-mountpoint）复制到沙箱 `bin/` 并置 PATH 首位，注入后自证三者可执行。本机 MSYS2 无真实 rsync/flock/mountpoint，正好覆盖脚本自身的回退分支（无 ionice 时退化为仅 nice）。
- 2026-10-22 前此处用 `ln -sf` 建软链，但 Git Bash 的 `ln -s` 会静默失败（exit 0 却不产生文件），导致 PATH 首位为空、被测副本报「缺少命令: rsync」而全量失败；已改为复制。

- 沙箱：固定在 `/tmp/rsync-test-sb`（back/log/lock/src/bin）。注意用 MSYS 风格路径——MSYS bash 的 `command -v` 不解析 Windows 风格 PATH 项，否则启动自检误报「缺少命令」。

- 日志断言：`newlog` 感知日志轮转的 inode 变化——轮转后整读新文件，未轮转则从上轮字节偏移续读。

## 四、覆盖场景

- E1 首次完整备份：退出码 0、每源 1 份快照、argv 含 `-aHAX`/`--delete`/排除规则、首跑无 `--link-dest`、`.complete` 标记与文件完整性；同节内以 `E12.x` 编号验证排除规则落地（`.cache/`、`lost+found/`、`*.tmp` 均不落盘）。

- E2 二次运行硬链接去重：未改动文件 nlink=2、两代同 inode、argv 出现 `--link-dest` 且指向上一代。

- E3 源增删改与快照不可变性：新快照反映变更、旧快照不被污染、改动文件换 inode。

- E4 KEEP 淘汰：五轮后仅剩 3 份、最旧淘汰、日志有删除记录。

- E5 缩小保护：源骤减时拒绝备份、不生成快照、失败源不影响后续源。

- E5b 掉电窗口：已 complete 未改名的残留被提升为正式快照。

- E6 崩溃续传：按完成度（非时间戳）选最厚残留续传，薄残留改判 failed，续传目录陈旧文件被刷新。

- E7 退出码分类：23 → warn（快照保留、整体 rc=1）；11 → fail（标记 .failed、无 inprogress 残留）。

- E8 启动自检与输入校验 12 种：非整数、KEEP=0、布尔开关填 yes、SHRINK 非法、NAME 重复、缺分隔符、三段配置、SOURCES 为空、路径穿越、SRC=/、源不存在、空源。

- E9 并发互斥：mock flock 模拟占用，第二实例退出 1。

- E10 日志轮转与超龄清理（真实触发轮转路径）。

- E11 限速/超时变体：argv 含 `--bwlimit=1M`、`--timeout=300`、`--one-file-system`。

## 五、产物与清理

沙箱内：`e2e.out`（被测脚本 stderr 捕获）、`argv.log`（rsync 命令行记录）、`log/rsync-backup.log` 及轮转文件、测试副本 `rsync-test.sh`/`rt-*.sh`。场景 1 每轮自动重置沙箱。

## 六、环境注意

- 在 MSYS2（bash 5.x）下开发验证；Linux 上可去掉 mock 直接用真实命令（把 `bin/` 从 PATH 移除）。

- MSYS 的 `stat -c %h` 在 link+unlink 后会虚高，硬链接数以 `find -inum` 实测为准。

- mock rsync 只保证状态机与参数构造；真实传输语义（权限/ACL/xattr）需在 Linux 目标机验证。

- 函数区起点按 `# 基础工具函数` 标记动态定位，被测脚本增删函数不影响自证；但若该标记被删除或改名，脚本会 FATAL 退出，需同步恢复标记。
