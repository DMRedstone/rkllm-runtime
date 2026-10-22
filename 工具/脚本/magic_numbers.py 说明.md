# magic_numbers 说明

## 定位

N11 魔法数守卫：代码里的数值常量必须登记，新增未登记常量会让测试失败。

## 口径

- 扫描 `输入/rkllm-runtime-1.3.1/src/rkllm/**/*.py` 的全部数值字面量
  （跳过 docstring 与注释）；
- 白名单三类：极小整数（结构体/索引语义）、协议常量（帧头、SSE 格式）、
  **已登记基线**（`输入/rkllm-runtime-1.3.1/tests/magic_baseline.json`，
  逐条带理由）；
- 新出现的未登记数值 → 失败，强制作者二选一：改成配置键，或写进基线并说明
  为什么它不该可配。

## 用法

```bash
python magic_numbers.py            # 校验，发现未登记常量即非零退出
python magic_numbers.py --update   # 把当前快照写入基线（代码改动后行号漂移时用）
```

## 注意

`--update` 会覆盖基线，执行前先 `git diff` 确认变化只是行号迁移、没有夹带
新增未登记常量。该脚本同时被 `tests/test_magic_numbers.py` 调用。
