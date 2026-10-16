# space_name_check 说明

## 用途

云端资料库节点 ↔ 本地目录，按「名称」核对（忽略层级差异）+ 数量统计。

## 与 space_sync_check 的分工

| 脚本 | 比对口径 | 擅长发现 |

| --- | --- | --- |

| `space_sync_check.py` | 逐路径（相对目录 + 名字） | 结构差异（层级、多余/缺失目录） |

| `space_name_check.py` | 只比名字（扩展名容错） | 「同一批文件两边是否都在」+ 数量 |

两个配合：结构差异看前者，内容齐不齐看后者。

## 用法

```bash
python space_name_check.py --cloud-root <节点id> --local-dir <本地目录> 
[--space-id <空间id>] [--skip-cloud 目录名 ...] [--local-only 相对路径 ...] 
[--space-api /path/to/space_api.py]
```

## 输出

- `[仅云端有]` / `[仅本地有]` 两份清单 + 「云端叶子文件数 / 本地文件数」

- 退出码：`0` = 一致，`1` = 有差异，`2` = 环境错误

## 设计要点

- 纯标准库、只读、不写死路径

- 自动定位 `space_api.py`：`--space-api` → `$SPACE_API` → 常见插件目录

- 名称比对做扩展名容错（`.md` `.docx` `.html` `.json` `.py`… 的后缀都能去）

- 云端 folder 与本地目录名同样参与比对（只比名字，不比位置）

- 默认忽略 `__pycache__` / `.git`；`list-nodes` 自带重试

- 递归云端子树；`--skip-cloud` 跳过根层指定节点（如本地不承载的 `文档`/`交付`）

---

创建日期：2026-10-16

> 
