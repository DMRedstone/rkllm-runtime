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
│   ├── 大模型测试工具链/   10 个脚本 + board_api.py + quality.py
│   └── 脚本/               运维脚本：备份、空间同步校验、测试 harness
├── 文档/                   部署说明 · 对话接口用法 · 开发记录
├── 测试数据/Qwen3.5-4B测试/    结果 JSON/CSV、10 份脱敏真实公文、输出样本
├── 模型测试报告/Qwen3.5-4B测试/ 性能测试报告 · 真实公文场景测试报告
├── 本地模型连接器/         客户端 MCP 连接器（独立源码交付，不随 deb）
├── 项目总览.md
├── 冷启动说明.md
└── 本地目录说明.md
```

> MCP 连接器是**客户端独立源码包**：`本地模型连接器/rkllm/`，用
> `python -m rkllm.mcp` 启动；板端 deb 只含服务端，不含连接器。

## 常用命令

```bash
# 构建 deb（上游 tarball 不入库，按实际路径指定；打包脚手架可省，省了按仓库布局查找）
export RKLLM_RELEASE_TARBALL=/path/to/rknn-llm-release-v1.3.1.tar.gz
export RKLLM_PACKAGING_DIR=$PWD/输入/rkllm-runtime-1.3.1
python 工具/编译工具链/make_deb.py --arch arm64        # 产物落 工具/dist/

# 静态自检（88 项）
python 工具/编译工具链/verify_deb.py 工具/dist/rkllm-runtime_1.3.1-17_arm64.deb

# 板端测试全流程
cd 工具/大模型测试工具链 && python run_all.py --host <板端地址> --key <令牌>

# 本地对话
rkllm chat qwen2.5-1.5b

# OpenAI 兼容 API
rkllm serve qwen2.5-1.5b
# 或 systemctl enable --now rkllm-server（配置在 /etc/default/rkllm-server）
```

## 关于上游源料

构建器 `make_deb.py` 只读取 **`.tar.gz`** 形态的上游 SDK（`tarfile.open(..., "r:gz")`），不读解包目录，且要求包内顶层目录名为 `rknn-llm-release-v1.3.1/`。两个路径都通过环境变量覆盖，换机器、换版本不必改脚本。

上游 SDK 属第三方源料，**不入库**——原始发行包 160 MiB，此前随附的精简归档（27.1 MiB）已移除，避免仓库体积膨胀。构建前自行下载并用 `RKLLM_RELEASE_TARBALL` 指定路径；需要瘦身可参照 `输入/构建资料说明.md` 里的剔除清单自行裁剪（只删不加，不改目录结构）。

## 目标平台

OrangePi 5 Plus 8G（RK3588，aarch64），只构建 arm64。板端需自带 NPU 驱动，模型 `.rkllm` 需另行在 x86 主机上用 `rkllm-toolkit` 转换，不含在本仓库内。
