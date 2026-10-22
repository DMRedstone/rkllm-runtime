# space_sync_check.py —— 资料库空间 ↔ 本地目录双向比对

一句话：把「云端资料库结构」和「本地工作目录」对一遍，报出缺谁、多谁、谁的内容不同步，用来保证"换台电脑也能从云端接着干"。

## 为什么需要它

资料库（网盘/知识库）和本地工程是两份副本，时间一长必然漂移：本地改了没传、云端传了本地没拉、同名文件内容却是两版。肉眼比对不现实，`library_search` 又只返回已绑定资源（会误判"不存在"）。这个脚本直接用节点接口遍历云端节点树，与本地目录做双向比对。

## 两个口径

| 口径 | 开关 | 判据 | 用途 |

| --- | --- | --- | --- |

| 结构级 | 默认 | 云端节点标题 ↔ 本地同名文件/目录（扩展名不限，`.csv` 对 `database` 节点也算命中） | 查漏、查多余 |

| 内容级 | `--content` | 下载云端内容，与本地文件比归一化指纹 | 名字对得上但内容是两版 |

| 内容级·文档 | `--content-doc` | doc 节点读正文，与本地 `.md` 比文本骨架长度 | 文档是否同步 |

为什么归一化：CRLF/LF、行尾空格、BOM 造成的哈希差异是噪声，不是内容差异。默认按归一化文本比；要严格字节比对加 `--raw`。

为什么 doc 用宽松口径：doc 是 `.md` 转的，格式必然不同（标题符号、表格标记），严格指纹只会全报差异。改用"去空白 + 去 Markdown 标记后的字符数"，相对差 ≤3% 视为同步，能抓住实质性内容差异，不误报格式差异。

## 用法

```bash
# 结构级：整个空间对某个本地目录
python space_sync_check.py --space-id <SPACE_ID> --local-dir ./myproject
# 只比某个节点子树（空间 ID 从节点信息自动取）
python space_sync_check.py --cloud-root <NODE_ID> --local-dir ./myproject
# 排除项（可重复）：云端忽略的标题 / 本地专有项
python space_sync_check.py --space-id <S> --local-dir ./p 
--skip-cloud 文档 --skip-cloud 交付 --local-only 本地过程存档
# 结构 + 内容级，并连带核对 doc
python space_sync_check.py --cloud-root <NODE_ID> --local-dir ./p 
--content --content-doc
```

常用参数：

- `--content-ext py,sh,json,md,...`：参与内容比对的扩展名（默认覆盖常见文本格式）

- `--max-size`：云端单文件下载上限，超出跳过（默认 4 MiB）

- `--max-depth`：云端递归深度上限（默认 12）

- `--space-api` / `--python`：接口脚本路径与解释器；默认自动查找

退出码：`0` 无差异 / `1` 有差异 / `2` 运行错误 —— 可直接接进 CI 或批处理。

## 设计要点（写通用脚本的几条经验）

1. 自动定位接口脚本：不写死安装路径，按 `--space-api` → 环境变量 `SPACE_API` → `~/.workbuddy/plugins/cache/**` 逐级查找，换机器/换版本不用改代码。

1. 接口调用带重试：本地 IPC 端点偶发被抢占（`E_ENDPOINT_UNTRUSTED`），统一重试 4 次 + 间隔 1.2s，避免把偶发失败当成真实差异。

1. 默认忽略 `__pycache__`：Python 字节码缓存不是交付物，任何目录比对都应默认排除。

1. 命名匹配双轨：先按完整文件名比，再按去扩展名的基名比 —— 云端节点常常不带扩展名（`压测并发曲线` vs `压测并发曲线.csv`）。

1. 不碰用户数据：只读云端、只读本地，下载内容落在 `tempfile` 临时目录；不上传、不删除、不改名。

1. 纯标准库：只用到 `argparse/subprocess/urllib/tempfile` 等，任何装了 Python 3 的机器能直接跑。
