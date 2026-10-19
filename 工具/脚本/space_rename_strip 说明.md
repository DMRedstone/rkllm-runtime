## 用途

批量剥离云端 doc 标题中的时间戳（前缀建档日期 `YYYYMMDD` + 后缀最后修改 `（YYYY.M.D）`）。

背景（2026-10-16）：项目内容文档迁入 git 仓库后，版本由提交历史表达，标题不再带时间戳（否则每次修改变成「删除+新增」，历史断裂）；行为类文档（「智能体行为记录」夹）仍保留时间戳，用范围排除即可。

## 用法

```bash
python space_rename_strip.py --space-root <节点id> [--space-id <空间id>] 
[--skip 名称 ...] [--apply]
```

- 默认 dry-run：只打印「旧标题 -> 新标题」映射；`--apply` 才真正调用 `rename-node`。

- `--space-root` 递归其子树；只处理 `doc`（folder/drive 不动）。

- 匹配模式：`^8位日期 空格 名称（YYYY.M.D）$`。

## 设计要点

- 纯标准库；dry-run 只读；自动定位 `space_api.py`（`--space-api` → `$SPACE_API` → 插件目录）；带重试。

- 与 `space_sync_check.py`（路径比对）、`space_name_check.py`（名称核对）同属资料库维护工具族。

## 备注

2026-10-16 部署机会话已完成全局去戳，本工具留作后续批量改名/防呆复用。

---

创建日期：2026-10-16

> 
