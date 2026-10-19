# mock-flock.sh 说明书

## 一、定位

`mock-flock.sh` 是测试用 flock 桩：本机（MSYS2）没有真实 flock 时，替身 `rsync-backup.sh` 的防重叠锁调用，让启动流程走通、并发场景可模拟。

## 二、行为

- `flock -n <FD>`：默认获取成功，退出 0。

- `MOCK_FLOCK_BUSY=1` 时：打印「mock-flock: 锁被占用」到 stderr 并退出 1——模拟「已有实例在运行」。

- 其他参数：忽略，恒退出 0。

## 三、用法与位置

与 mock-rsync 一起放入 PATH 首位即可，被测副本无需任何改动：

```
export PATH="$SB/bin:$PATH"
```

被测脚本以 `flock -n 9` 抢占 `$LOCK` 文件描述符；桩收到 `-n` 即按上述规则返回。

## 四、对应的测试场景

- 正常路径：E1–E11 各场景均经此桩，锁永远可得。

- 并发互斥（E9）：`MOCK_FLOCK_BUSY=1` 断言第二个实例退出码 1、日志记录「已有备份在运行」。

## 五、限制

不实现真实锁定语义（无文件锁、无 FD 绑定、无等待模式）；只覆盖被测脚本实际使用的 `flock -n` 单一形态。真实互斥行为需在 Linux 目标机验证。
