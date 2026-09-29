# CLAUDE.md

本文件为 AI 协作工具（如 Claude Code）提供本仓库的约定与背景。

## 项目背景

`ygorl` 是游戏王（Yu-Gi-Oh!）强化学习引擎：对局、组牌（含 off-meta 发现）、对手预测。
设计文档在 `docs/design/`，工程计划与任务清单在 `docs/eng-plan.md`；设计文档是唯一权威，设计变更先改文档再改代码。

技术栈（已定）：edo9300/ygopro-core + Project Ignis 脚本/数据库（git submodule），C++17 + pybind11
（CMake + scikit-build-core），Python 3.11 + uv，PyTorch，pyribs，pytest。不要引入其它运行时或框架。

「环境」（格式 + 卡池 + 禁限表 + 规则 flag + meta 卡表 + 版本号）是一等公民配置，位于
`environments/<version>/`；所有训练、评估、组牌产物必须绑定环境版本。

## 游戏王术语约定

代码与文档中涉及游戏王概念时，统一使用以下命名（英文标识符 / 中文含义）：

| 标识符          | 含义                       |
| --------------- | -------------------------- |
| `monster`       | 怪兽卡                     |
| `spell`         | 魔法卡                     |
| `trap`          | 陷阱卡                     |
| `attack` / `defense` | 攻击力 / 守备力       |
| `level` / `rank` / `link` | 等级 / 阶级 / 连接值 |
| `attribute`     | 属性（光、暗、地、水、炎、风、神） |
| `race`          | 种族                       |
| `main_deck` / `extra_deck` / `side_deck` | 主卡组 / 额外卡组 / 副卡组 |
| `banlist`       | 禁限卡表                   |
| `lp`            | 生命值（Life Points）      |

卡片以官方卡片密码（8 位数字 `password`）作为唯一标识，不要用卡名做主键。

上表是核心子集；区域、对局流程、引擎与训练、组牌等完整术语见 [docs/glossary.md](docs/glossary.md)。新增概念先补术语表再使用。

## 工作约定

- 提交信息使用简洁的祈使句，说明做了什么以及为什么。
- 提交前跑 `tools/presubmit.sh`（ruff 格式化 + lint，与 CI 的 `--check` 一致）。
- 合并 PR 前在本机跑 `tools/presubmit.sh --test`（全部单测）；GitHub CI 只在 PR 与 main 的推送上跑，文档改动不跑。
- 新增依赖前先在 README 的「快速开始」里写明安装方式。
- 目录结构变化时同步更新 README 的「目录结构」一节。
