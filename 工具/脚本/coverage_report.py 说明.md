# coverage_report 说明

## 定位

N8（可维护）覆盖率判定：区分「纯逻辑模块」与「平台 I/O 模块」后统计，输出可读报告。

## 为什么分组

`engine.py`（ctypes 绑 `librkllmrt.so`）、`supervisor.py`（spawn 子进程）、
`worker/main.py`（子进程入口）、`platform/freq.py`（写 sysfs，需 root）在
Windows 开发机上**结构上不可覆盖**。把它们算进分母只会让数字失真——这些模块的
验收靠板端实测（kill -9 自愈、锁频、真推理），不是单测。

## 判定口径

- `CORE_AVG ≥ 70%` → N8 通过；
- `PLATFORM` 组只报告，不作为门槛，逐项在报告里列出验证方式。

## 用法

```bash
python coverage_report.py            # 跑 pytest 并生成分组报告
python coverage_report.py --json     # 额外输出 JSON
python coverage_report.py --no-run   # 不跑测试，只汇总已有 .coverage 数据
```

## 注意

需先安装 `pytest-cov`；报告同时给出每个模块的行数、覆盖率与未覆盖行号范围，
便于定位补测点。
