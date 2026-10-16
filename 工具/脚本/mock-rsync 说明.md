# mock-rsync.sh 说明书

## 一、定位

`mock-rsync.sh` 是 `rsync-backup.sh` 测试用 rsync 模拟器：覆盖被测脚本实际使用的全部参数，在本机没有真实 rsync（或不想动真实数据）时，让端到端测试完整跑通备份状态机。

## 二、覆盖的参数

`-aHAX --numeric-ids --delete --partial-dir= --stats --human-readable --one-file-system --bwlimit= --timeout= --exclude= --link-dest=`，以及任意未知 `-` 开头参数（忽略）。最后两个非参数位置分别取为 src 与 dest。

## 三、模拟语义（对齐真实 rsync 的关键行为）

1. `--link-dest` 硬链接入场：参考快照存在时，`cp -al` 把参考快照整体硬链接进 dest——未改动文件不占双倍空间。

1. 增量拷贝：逐个遍历源文件；与参考副本内容一致 → 保留硬链接不动；不一致（内容变化或不存在）→ 先删后拷，换 inode，不污染旧快照。

1. `--delete`：清掉 dest 中源已不存在的条目（`.rsync-partial` 除外）。

1. exclude 匹配：按 rsync 语义——无斜杠模式匹配任意层级的 basename；含斜杠模式按路径匹配（精确段/前缀/后缀/包含）。

## 四、环境变量

- `MOCK_RSYNC_RC`：强制退出码（默认 0）。E7 场景用它模拟 23（warn）/11（fail）。

- `MOCK_RSYNC_SLEEP`：开跑前睡眠秒数（并发/时序测试用）。

- `MOCK_RSYNC_ARGV_LOG`：把命令行原样追加到该文件（供断言 argv 组装正确性）。

- `MOCK_RSYNC_DEBUG`：把动作明细（入场/保留/先删后拷/exclude/delete/DONE）追加到该文件。

## 五、用法与位置

放入 PATH 首位（e2e 的 `bin/` 目录），被测副本即自动调到它而非真实 rsync：

```
export PATH="$SB/bin:$PATH"
```

## 六、边界与限制

- 只保证状态机与参数构造正确：权限/ACL/xattr 保留、`-aHAX --numeric-ids` 的真实传输语义不模拟，需在 Linux 目标机用真 rsync 验证。

- `--stats` 输出仅为占位摘要（被测脚本只记录不解析）。

- 参数解析失败（缺 src/dest）退出 2。
