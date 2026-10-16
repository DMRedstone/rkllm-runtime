# rkllm-runtime

把 Rockchip RKLLM SDK 裁打成板端「装完即用」的 Debian 安装包，并对它提供的推理服务做性能与真实场景测试。

## 入口文档

| 文档 | 说明 |
| --- | --- |
| **[项目总览](项目总览.md)** | **先看这份** —— 两条工作线 + 附属设施的全貌、产物、结论与命名规范 |
| [冷启动说明](冷启动说明.md) | 换机器后只凭本仓库把项目重新跑起来 |
| [本地目录说明](本地目录说明.md) | 本地目录与各文档、脚本的对应关系 |

## 目录结构

```
.
├── 输入/                   构建输入：打包脚手架 + 上游 SDK 源料
├── 工具/
│   ├── 编译工具链/         make_deb · verify_deb · crosscheck_deb · render_template
│   ├── 大模型测试工具链/   00_run_all … 09_make_rw_report
│   └── 脚本/               运维脚本：备份、空间同步校验、测试 harness
├── 文档/                   部署说明 · 文件清单 · 对话接口用法 · 开发记录
├── 交付/                   构建产物不入库，发在 GitHub Release
├── 测试数据/               结果 JSON/CSV、10 份脱敏真实公文、输出样本
├── 模型测试报告/           性能测试报告 · 真实公文场景测试报告
├── 本地模型连接器/         把板端小模型接成 MCP 工具（可独立部署）
├── 项目总览.md
├── 冷启动说明.md
└── 本地目录说明.md
```

## 三条常用命令

```bash
# 构建 deb（需先指定上游 SDK 与打包脚手架的位置）
export RKLLM_RELEASE_TARBALL=/path/to/rknn-llm-release-v1.3.1.tar.gz
export RKLLM_PACKAGING_DIR=$PWD/输入/rkllm-runtime-1.3.1
python 工具/编译工具链/make_deb.py --arch arm64 --emit-tree

# 静态自检（83 项）
python 工具/编译工具链/verify_deb.py 交付/rkllm-runtime_1.3.1-12_arm64.deb

# 板端测试全流程
cd 工具/大模型测试工具链 && python 00_run_all.py --host <板端地址> --key <令牌>
```

## 关于上游源料

构建器 `make_deb.py` 只读取 **`.tar.gz`** 形态的上游 SDK（`tarfile.open(..., "r:gz")`），不读解包目录，且要求包内顶层目录名为 `rknn-llm-release-v1.3.1/`。两个路径都通过环境变量覆盖，换机器、换版本不必改脚本。

上游 SDK 属第三方源料，与本项目分离存放；本仓库内的 `输入/rknn-llm-release-v1.3.1.tar.gz` 是精简归档（27.1 MiB，只做删除，不改内容与结构），原始发行包 160 MiB。

## 目标平台

OrangePi 5 Plus 8G（RK3588，aarch64），只构建 arm64。板端需自带 NPU 驱动，模型 `.rkllm` 需另行在 x86 主机上用 `rkllm-toolkit` 转换，不含在本仓库内。
