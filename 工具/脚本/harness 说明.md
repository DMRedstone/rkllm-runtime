# harness.sh 说明书

## 一、定位

`harness.sh` 是 `rsync-backup.sh` 的函数级隔离测试（T1–T11，54 项断言）。不跑备份主流程：从被测脚本动态定位并抽取全部函数，在桩全局配置下逐场景直接调用，只验证函数行为。

## 二、运行方式

```
bash harness.sh
```

要求与 `rsync-backup.sh` 位于同一目录（脚本内 `SRC="$SB/rsync-backup.sh"`）。结尾输出 `PASS=N FAIL=M`，FAIL 非 0 时退出码非 0。

## 三、测试方法与自证

- 动态定位函数区写入 `funcs.sh`：起点用 `grep` 找 `^log() {$`，终点用 `awk` 找 `backup_one()` 之后第一个行首 `}`；随后仍以两处断言自证（起点必须是 `log() {`、终点必须是 `}`），定位失败即 FATAL 退出。
- 2026-10-22 前此处硬编码 `sed -n '144,781p'`，被测脚本函数区后移到 157–890 后测试直接 FATAL——已改为动态定位，不再随行号漂移失效。

- 桩全局配置与被测脚本可配置段同构（`KEEP=3`、`MIN_FREE_PCT=1`、`REQUIRE_ROOT=0`、`SHRINK_RATIO=0` 等），`source funcs.sh` 后直接调用函数。

- 幂等：每轮先清空全部场景目录（t2…t11、h-back），避免读到上轮产物。

## 四、覆盖场景

- T1 `in_array` / `rsync_class`：退出码 0/24→ok、23→warn、其余→fail。

- T2 `count_files`：计数正确性、`-xdev` 开关。

- T3 `list_complete_snapshots`：只认 `.complete`、新的在前、排除 failed/inprogress。

- T4 `handle_leftovers`：4 个场景（T4a–T4d）——多残留按完成度选最厚续传（非时间戳）、仅一份时免全树统计、已 complete 的残留提升为正式快照、目标已存在改判 failed。

- T5 `prune_snapshots`：KEEP 淘汰、对只读顶层也能删（先 chmod u+w）。

- T6 `prune_failed`：数量封顶 + 超龄清理。

- T7 `check_free_space`：软失败返回 1、阈值通过、df 解析失败跳过且留日志、inode 不误报。

- T8 `rotate_logs`：按大小轮转、按天清理旧轮转文件。

- T9 `mark_failed`：改名 `.failed.<pid>`、内容保留。

- T10 `cleanup`（EXIT trap）：保留原退出码、残留目录改名 failed。

- T11 `backup_one` 输入校验：三段配置、路径穿越 NAME、SRC=/、源不存在、空源；校验失败不创建任何目录。

## 五、产物与清理

运行目录内生成：`funcs.sh`、`harness.log`、场景目录 `t2/`–`t11/`、`h-back/`。全部在每轮开头自动重建/清空，可随时整目录删除。

## 六、依赖与环境

bash、GNU find/stat；桩中 `REQUIRE_ROOT=0`，无需 root；无需 rsync/flock/mountpoint（T11 不触达 rsync）。被测脚本改动函数区后需同步更新抽取行号。
