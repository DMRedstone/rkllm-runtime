# mock-mountpoint.sh 说明书

## 一、定位

测试用 mountpoint 桩：本机没有真实 mountpoint 命令时的占位替身，使 `rsync-backup.sh` 启动自检的命令存在性检查通过、挂载检查流程可执行。

## 二、行为

恒 `exit 0`（视为「路径是挂载点」）。

## 三、何时被调用

被测脚本对 `MOUNT_CHECKS` 列表逐项执行 `mountpoint -q "$m"`。e2e 的测试副本把 `MOUNT_CHECKS` 配置为空，因此实际不会被调用——它的存在意义是让脚本启动自检（`for cmd in … mountpoint …`）不因缺命令而退出，从而覆盖其余主流程。

## 四、用法与位置

与 mock-rsync、mock-flock 一起放入 PATH 首位：

```
export PATH="$SB/bin:$PATH"
```

## 五、限制

恒成功意味着挂载检查在测试中形同虚设；「挂载点未挂载则退出」的真实行为（`mountpoint -q` 返回非 0 → 脚本记日志退出 1）未被任何场景覆盖，属已知测试盲区，需在 Linux 目标机验证。
