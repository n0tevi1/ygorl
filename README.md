# ygorl

游戏王（Yu-Gi-Oh!）强化学习引擎：遵循当前环境规则（卡池 + 禁限表）进行对局、探索组牌策略、并推断对手隐藏信息。

## 项目状态

设计已评审通过，进入实施（里程碑 M0 骨架）。技术栈与方向见下。

## 简介

三个目标：

1. **对局**：给定一套牌组，与其他玩家 / agent 决斗。
2. **组牌**：给定卡池与当前环境信息，探索组牌策略，并能发现并实验验证非版本主流的强势构筑。
3. **对手预测**：对局中估计对手的手牌、剩余卡组、盖牌与牌组类型。

核心结构是「牌组无关的通用对局策略（内层）」+「以该策略为评估器的组牌搜索（外层）」，对手预测作为内层网络的子模块。

技术栈：edo9300/ygopro-core（EDOPro 核心）+ Project Ignis 卡片脚本与数据库，C++17 + pybind11 向量化环境，Python 3.11 + PyTorch 训练，pyribs 质量-多样性搜索。优先 Master Duel 格式，架构适用于 OCG / TCG。

## 文档

- [设计文档](docs/design/README.md)：目标、引擎裁决、RL 挑战、对局策略、对手预测、组牌与 off-meta 发现、架构、风险。
- [工程计划](docs/eng-plan.md)：里程碑 M0–M6、任务清单、依赖、验收标准、推进顺序。GitHub issues 与任务一一对应。

## 快速开始

依赖：Linux、[uv](https://docs.astral.sh/uv/)（≥ 0.8）、CMake（≥ 3.20）、支持 C++17 的编译器（GCC ≥ 9 / Clang ≥ 10）。
Python 3.11 由 uv 自动选择或下载；pybind11 与 scikit-build-core 作为构建依赖由 uv 自动安装。

```bash
git clone https://github.com/n0tevi1/ygorl && cd ygorl
uv sync                                   # 创建 .venv，编译并安装 C++ 扩展 ygorl._core
uv run python -c "import ygorl._core"     # 冒烟测试
uv run pytest                             # 跑单测
```

修改 `csrc/`、`CMakeLists.txt` 或 `pyproject.toml` 后，`uv sync` / `uv run` 会自动重新编译扩展；
需要强制重编时用 `uv sync --reinstall-package ygorl`。

## 目录结构

```
.
├── README.md
├── CLAUDE.md                # 给 AI 协作工具的项目约定
├── pyproject.toml           # uv 项目 + scikit-build-core 构建配置
├── uv.lock
├── CMakeLists.txt           # 构建 C++ 扩展 ygorl._core
├── csrc/                    # C++ 源码（pybind11 绑定）
├── src/ygorl/               # Python 包
├── tests/                   # pytest 单测
├── docs/
│   ├── design/              # 设计文档（按主题拆分）
│   └── eng-plan.md          # 工程计划
├── .editorconfig
└── .gitignore
```

## 参考资料

- 游戏王 OCG 官方站：https://www.yugioh-card.com/japan/
- 游戏王 TCG 官方站：https://www.yugioh-card.com/en/
- 卡片数据 API（YGOPRODeck）：https://ygoprodeck.com/api-guide/
- 规则核心（EDOPro）：https://github.com/edo9300/ygopro-core
- 卡片脚本 / 数据库 / 禁限表（Project Ignis）：https://github.com/ProjectIgnis

## 许可证

待定（注意：edo9300 核心与 Project Ignis 脚本为 AGPL-3.0）。
