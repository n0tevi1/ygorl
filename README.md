# ygorl

游戏王（Yu-Gi-Oh!）强化学习引擎：遵循当前环境规则（卡池 + 禁限表）进行对局、探索组牌策略、并推断对手隐藏信息。

## 项目状态

设计已评审通过。M0（骨架）完成；M1（引擎绑定）实现中：核心绑定、卡片数据、合法性、消息解码、单局 API、确定性重放、回放格式已落地。M3（基线与评估）：GreedyAgent、配对种子 Arena、对局矩阵与 Nash / alpha-rank、命令行 `ygorl duel / arena / matrix / replay` 已落地。技术栈与方向见下。

## 简介

三个目标：

1. **对局**：给定一套牌组，与其他玩家 / agent 决斗。
2. **组牌**：给定卡池与当前环境信息，探索组牌策略，并能发现并实验验证非版本主流的强势构筑。
3. **对手预测**：对局中估计对手的手牌、剩余卡组、盖牌与牌组类型。

核心结构是「牌组无关的通用对局策略（内层）」+「以该策略为评估器的组牌搜索（外层）」，对手预测作为内层网络的子模块。

技术栈：edo9300/ygopro-core（EDOPro 核心）+ Project Ignis 卡片脚本与数据库，C++17 + pybind11 向量化环境，Python 3.11 + PyTorch 训练，pyribs 质量-多样性搜索。优先 Master Duel 格式，架构适用于 OCG / TCG。

## 文档

- [设计文档](docs/design/README.md)：目标、引擎裁决、RL 挑战、对局策略、对手预测、组牌与 off-meta 发现、架构、风险。
- [引擎层](docs/engine.md)：核心绑定、消息解码、动作模型、单局 API、确定性补丁。
- [回放](docs/replays.md)：回放文件格式、环境绑定、`.yrpX` 导出。
- [基线与评估](docs/evaluation.md)：Agent 协议、Random / Greedy / PolicyAgent、配对种子 Arena、对局矩阵与 Nash / alpha-rank。
- [命令行](docs/cli.md)：`ygorl duel`、`ygorl replay`、`ygorl branch`、`ygorl arena`、`ygorl matrix` 的参数、输出与退出码。
- [分支探索](docs/branching.md)：`fork(replay, t)` 从任意决策点分叉、候选 rollout 比较、`ygorl branch` 命令行、限制。
- [课程与开局配平](docs/curriculum.md)：单人展开 / 仅手坑 / 完整三种课程模式、先后攻配平、增广开局标志位。
- [信念校准评估](docs/belief-eval.md)：信念头的 ECE / AUC / top-k 等指标定义、掩码约定、随机与先验预测器基线数字。
- [基准结果](docs/benchmarks.md)：Greedy vs Random 2,000 局等实测数字。
- [观测编码](docs/encoding.md)：卡片表、全局向量、候选动作表的每一列。
- [环境规范](docs/environments.md)：`environments/<version>/` 的文件格式、来源与版本约定。
- [语义协同图](docs/synergy.md)：CardScripts 脚本挖掘、边语义、解析覆盖率、代理召回检验、引擎包枚举。
- [工程计划](docs/eng-plan.md)：里程碑 M0–M6、任务清单、依赖、验收标准、推进顺序。GitHub issues 与任务一一对应。

## 快速开始

依赖：Linux、[uv](https://docs.astral.sh/uv/)（≥ 0.8）、CMake（≥ 3.20）、支持 C++17 的编译器（GCC ≥ 9 / Clang ≥ 10）。
Python 3.11 由 uv 自动选择或下载；pybind11 与 scikit-build-core 作为构建依赖由 uv 自动安装。
Python 运行时依赖写在 `pyproject.toml`、锁定在 `uv.lock`，`uv sync` 一并安装：numpy（观测编码、评估指标）；
[nashpy](https://github.com/drvinceknight/Nashpy)（连带 scipy、networkx 等）用于对局矩阵的 Nash 均衡（`ygorl.eval.matchup`）。
新增依赖用 `uv add <包名>`。

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/n0tevi1/ygorl && cd ygorl
# 已克隆但没带子模块时：git submodule update --init --recursive
uv sync                                   # 创建 .venv，编译并安装 C++ 扩展 ygorl._core
uv run python -c "import ygorl._core"     # 冒烟测试
uv run pytest                             # 跑单测
```

命令行 `ygorl`：单局、回放、分叉探索、Arena、对局矩阵（全部参数与输出说明见 [docs/cli.md](docs/cli.md)，`uv run ygorl <命令> --help` 查看帮助）。
在仓库根目录依次运行，产物写到 git 已忽略的 `out/`；`tests/test_readme.py` 会逐行执行下面的示例（局数调小）：

```bash
# 单局：snake_eye（greedy）对 kashtira（random），打印胜者、终局原因、回合、LP、决策数；存回放并导出 EDOPro .yrpX
uv run ygorl duel tests/decks/snake_eye.ydk tests/decks/kashtira.ydk --seed 1 --agent-a greedy --save-replay out/game.json.gz --yrpx out/game.yrpX
# 回放：显示元数据，按应答日志重新模拟并核对是否到达录制的终局，再导出一份 .yrpX
uv run ygorl replay out/game.json.gz --verify --export-yrpx out/again.yrpX
# 分叉：从第 12 个 agent 步逐个尝试全部合法动作并用 greedy 下完；第二行每个候选 20 次随机 rollout，输出胜/平/负与均值
uv run ygorl branch out/game.json.gz --at 12 --try all --policy greedy --seed 1
uv run ygorl branch out/game.json.gz --at 12 --try 0,1,2 --rollouts 20
# Arena：greedy 驾驶 snake_eye 对 random 驾驶 kashtira，20 局配对种子对局，输出胜率与 95% Wilson 区间
uv run ygorl arena tests/decks/snake_eye.ydk --vs tests/decks/kashtira.ydk --games 20 --workers 2
# 对局矩阵：3 套牌两两各 10 局（greedy 驾驶双方），输出胜率矩阵、Nash 混合与 alpha-rank
uv run ygorl matrix tests/decks/snake_eye.ydk tests/decks/kashtira.ydk tests/decks/yubel.ydk --games 10 --workers 2 --out out/matrix.json
```

加 `--env <版本或目录>` 即按该环境的规则对局，产物绑定环境版本；`ygorl matrix --env <版本>` 不给牌组时用环境的 meta 卡组，
结果写到 `environments/<版本>/artifacts/matrix/`（见 [docs/cli.md](docs/cli.md)）。Python API 见 [docs/engine.md](docs/engine.md)、
[docs/replays.md](docs/replays.md)、[docs/evaluation.md](docs/evaluation.md)。

修改 `csrc/`、`patches/`、`CMakeLists.txt` 或 `pyproject.toml` 后，`uv sync` / `uv run` 会自动重新编译扩展；
需要强制重编时用 `uv sync --reinstall-package ygorl`。

### Claude Code 云端会话

`.claude/hooks/session-start.sh` 是 SessionStart hook（在 `.claude/settings.json` 注册），只在 Claude Code on the web
（`CLAUDE_CODE_REMOTE=true`）中运行：补装缺失的工具（uv / cmake / ninja / g++ / ccache）、拉取子模块、`uv sync` 编译扩展，
会话开始时即可直接 `uv run pytest`。脚本幂等，也可以粘贴进云环境的 setup script，或手动执行：

```bash
CLAUDE_CODE_REMOTE=true .claude/hooks/session-start.sh
```

### 第三方子模块

规则核心与卡片数据以 git submodule 形式放在 `third_party/`，每个都固定到明确的 commit（`git submodule status` 查看）：

| 路径 | 上游 | 用途 |
|------|------|------|
| `third_party/ygopro-core` | [edo9300/ygopro-core](https://github.com/edo9300/ygopro-core)（含 Lua 5.4 嵌套子模块） | 规则核心，CMake 编成静态库链接进 `ygorl._core` |
| `third_party/CardScripts` | [ProjectIgnis/CardScripts](https://github.com/ProjectIgnis/CardScripts) | 卡片效果 Lua 脚本 |
| `third_party/BabelCDB` | [ProjectIgnis/BabelCDB](https://github.com/ProjectIgnis/BabelCDB) | 卡片数据库 `cards.cdb` |
| `third_party/LFLists` | [ProjectIgnis/LFLists](https://github.com/ProjectIgnis/LFLists) | 禁限表 `.lflist.conf` |

脚本与数据库子模块标记为 shallow，只拉取固定的那个 commit。更新某个子模块到上游最新：

```bash
git submodule update --remote third_party/CardScripts   # 换成要更新的路径
git add third_party/CardScripts && git commit -m "Bump CardScripts to <short-sha>"
uv sync --reinstall-package ygorl                        # 更新 ygopro-core 后需要重编
```

更新核心或脚本会改变对局结果，提交前跑一遍 `uv run pytest`，并在提交信息里写明新旧 commit。

## 目录结构

```
.
├── README.md
├── CLAUDE.md                # 给 AI 协作工具的项目约定
├── .claude/                 # Claude Code 配置：settings.json + hooks/session-start.sh（云端会话初始化）
├── .github/workflows/       # CI：构建扩展 + pytest
├── pyproject.toml           # uv 项目 + scikit-build-core 构建配置
├── uv.lock
├── CMakeLists.txt           # 构建 C++ 扩展 ygorl._core
├── cmake/                   # CMake 片段（ocgcore.cmake：复制核心、打补丁、编成静态库）
├── patches/ygopro-core/     # 对规则核心的补丁（确定性遍历顺序），构建时应用
├── third_party/             # git submodule：ygopro-core、CardScripts、BabelCDB、LFLists
├── csrc/                    # C++：core_backend（OCG_* 封装）、duel_pool（线程池）、host / obs_encoder（C++ 主机层与观测编码）、host_pool + worker_pool（C++ 步进环境）、binding（pybind11）
├── src/ygorl/               # Python 包
│   ├── cli.py               # 命令行入口 `ygorl`（argparse 子命令）
│   ├── commands/            # 各子命令一个模块：duel / replay / branch / arena / matrix；__init__.py 放共用选项（牌组、环境、agent）
│   ├── agents/              # Agent 协议、RandomAgent、GreedyAgent、PolicyAgent；registry.py（按规格构造 agent 与可 pickle 的 factory，供 CLI）
│   ├── build/               # 组牌：Lua 脚本读取器、过滤条件 IR、脚本挖掘协同图（synergy_graph）、引擎包枚举（packages）
│   ├── cards/               # cards.cdb、禁限表（.lflist.conf）、牌组（.ydk）、合法性校验
│   ├── data/                # Environment 加载与校验
│   ├── engine/              # 消息解码、动作模型、单局 Duel、回放、分支探索（branch.py）、课程模式（curriculum.py）；constants.py 为生成文件
│   ├── env/                 # 向量化环境：VecDuelEnv（C++ 线程池）、DuelEnv、run_games、paired_specs；encoding.py 参考编码器；encoded.py 为 C++ 步进的 EncodedVecEnv
│   └── eval/                # 评估：配对种子 Arena、对局矩阵与 Nash / alpha-rank、信念头校准指标与基线
├── tools/                   # 开发脚本：常量生成、测试牌组 / 代理引擎包生成、协同图构建、引擎包列表、压力测试、确定性扫描、YGOPRODECK 核对、arena 基准（ygorl arena 的包装）、信念基线表、吞吐基准、课程模式检查
├── tests/                   # pytest 单测（test_readme.py 执行 README 的命令行示例）；decks/ 放 10 套测试牌组，data/ 放测试数据（含代理引擎包）
├── docs/
│   ├── design/              # 设计文档（按主题拆分）
│   ├── engine.md            # 引擎层：CoreBackend、消息、动作模型、确定性
│   ├── encoding.md          # 观测编码规范（卡片表、全局向量、候选动作表）
│   ├── evaluation.md        # 基线 agent 与评估
│   ├── belief-eval.md       # 信念校准评估：指标定义与基线数字
│   ├── benchmarks.md        # 基准结果（实测数字、commit、日期）
│   ├── environments.md      # environments/<version>/ 目录规范
│   ├── replays.md           # 回放格式与 .yrpX 导出
│   ├── cli.md               # 命令行 ygorl 各子命令
│   ├── branching.md         # 分支探索：fork(replay, t)、ygorl branch、限制
│   ├── curriculum.md        # 课程模式、先后攻配平、增广开局标志位
│   ├── synergy.md           # 脚本挖掘协同图与引擎包
│   ├── spikes/              # 技术调研结论（combo-solver.md）
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
