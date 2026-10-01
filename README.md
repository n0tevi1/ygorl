# ygorl

游戏王（Yu-Gi-Oh!）强化学习引擎：遵循当前环境规则（卡池 + 禁限表）进行对局、探索组牌策略、并推断对手隐藏信息。

## 项目状态

设计已评审通过。逐任务的完成情况见 [工程计划](docs/eng-plan.md) 与 GitHub issues。

- **M0 骨架**：完成。
- **M1 引擎绑定**：完成；待办是两项人工核对（T1.2 卡片字段与效果串的人工抽检、T1.8 `.yrpX` 在 EDOPro 客户端中回看）。
- **M2 向量化环境**：完成（C++ 线程池、C++ 步进与观测编码、多选可行集核对、事件 token 流、训练态真值、课程模式、arena 快照、分支探索）；待办是 16 核吞吐数字，以及 C++ 步进路径上的课程模式。
- **M3 基线与评估**：完成（Greedy、配对种子 Arena、对局矩阵与 Nash / alpha-rank、信念校准指标、命令行 `ygorl duel / replay / branch / arena / matrix / env`）。
- **M4 策略训练**：进行中（PyTorch 作为可选依赖组 `train` 已接入）；策略网络、特权 critic 与 PPO 自博弈训练循环（T4b.1–T4b.4）已落地。T4c.1 信念头已落地：五个头以 HDT 后验为残差初始化，随机自博弈 1500 局上各头均优于 HDT 过滤基线（牌组类型 top-1 0.547 → 0.863、ECE 0.31 → 0.01），见 [docs/belief-heads.md](docs/belief-heads.md)。T4a.2 BC 预热：求解器第 1 回合示范 + Greedy 战斗阶段示范（b2）对 Random 0.795（0.734–0.845）、对 Greedy 0.415，未见起手目标场面 41/100（求解器 62）；训练起手线复现 86.7%（bc60）。只用求解器示范时对 Random 0.305（示范只覆盖先攻第 1 回合，不进战斗阶段）；设计 I4 已相应修改，见 [docs/bc.md](docs/bc.md)「补救实验」。
- **M5 数据与组牌**：数据抓取与环境快照（T5.1，`ygorl env build`，快照 `environments/md-2026-09`，禁限表已与 YGOPRODECK、Yugipedia 交叉核对）、协同图（T5.3，真实 meta 召回 0.935）、引擎包枚举（T5.4）、基因型与算子（T5.5）、漏斗第一层求解器起手分析（T5.6：120 手配对检验与真实首回合无显著差异，p = 0.63；每套牌平均 71 求解器进程秒；预算研究（`tools/funnel_budget.py`）：120 秒/套合理，但每手 5 秒对长 combo 牌组有系统性假卡手（+17–19 pp），默认改为每手 10 秒（均值 101 秒/套））、代理模型（T5.7）已落地；T5.2 文本嵌入尚未开始。
- **M6**：未开始。

技术栈与方向见下。

## 简介

三个目标：

1. **对局**：给定一套牌组，与其他玩家 / agent 决斗。
2. **组牌**：给定卡池与当前环境信息，探索组牌策略，并能发现并实验验证非版本主流的强势构筑。
3. **对手预测**：对局中估计对手的手牌、剩余卡组、盖牌与牌组类型。

核心结构是「牌组无关的通用对局策略（内层）」+「以该策略为评估器的组牌搜索（外层）」，对手预测作为内层网络的子模块。

技术栈：edo9300/ygopro-core（EDOPro 核心）+ Project Ignis 卡片脚本与数据库，C++17 + pybind11 向量化环境，Python 3.11 + PyTorch 训练，pyribs 质量-多样性搜索。优先 Master Duel 格式，架构适用于 OCG / TCG。

## 文档

- [设计文档](docs/design/README.md)：目标、引擎裁决、RL 挑战、对局策略、对手预测、组牌与 off-meta 发现、架构、风险。
- [引擎层](docs/engine.md)：核心绑定、消息解码、动作模型、单局 API、确定性补丁、快照（snapshot / restore）。
- [回放](docs/replays.md)：回放文件格式、环境绑定、`.yrpX` 导出与 `.yrp` / `.yrpX` 读取。
- [调研：ygo-combo-solver](docs/spikes/combo-solver.md)：与本仓库核心的兼容性、封装方案、arena 快照移植评估（T1.7）。
- [基线与评估](docs/evaluation.md)：Agent 协议、Random / Greedy / PolicyAgent、配对种子 Arena、对局矩阵与 Nash / alpha-rank。
- [命令行](docs/cli.md)：`ygorl duel`、`ygorl replay`、`ygorl branch`、`ygorl arena`、`ygorl matrix`、`ygorl strength`、`ygorl env` 的参数、输出与退出码。
- [分支探索](docs/branching.md)：`fork(replay, t)` 从任意决策点分叉、候选 rollout 比较、`ygorl branch` 命令行、限制。
- [课程与开局配平](docs/curriculum.md)：单人展开 / 仅手坑 / 完整三种课程模式、先后攻配平、增广开局标志位。
- [信念校准评估](docs/belief-eval.md)：信念头的 ECE / AUC / top-k 等指标定义、掩码约定、随机与先验预测器基线数字。
- [信念头](docs/belief-heads.md)：五个头（牌组类型 / 剩余份数 / 手牌 + 角色位 / 盖卡 / 被响应）、损失掩码、meta 先验初始化与 HDT 式过滤特征、随机自博弈上的实验数字。
- [基准结果](docs/benchmarks.md)：Greedy vs Random 2,000 局、吞吐、PPO 自博弈 1 小时等实测数字。
- [策略训练](docs/training.md)：rollout 数据布局与两人零和的符号约定、截断对局的 critic 自举、GAE(λ) 对照、Expected-SARSA(λ) 回报与 Q-boosted 优势（VRPO）、候选动作 Q 头 + V 头；PPO 自博弈训练循环（可插拔策略目标、熵、KL 到慢速参考 / BC 先验、快照池 + keep-best、牌组池、checkpoint 与日志）。
- [观测编码](docs/encoding.md)：卡片表、全局向量、候选动作表的每一列；事件 token 流与响应窗口 / 放弃 token。
- [策略网络](docs/nets.md)：卡片 / 效果编码器、局面 Transformer、事件历史模块（GTrXL / LSTM）、动作打分头、冻结文本向量接口、给 critic / 信念头的接口。
- [环境规范](docs/environments.md)：`environments/<version>/` 的文件格式、来源与版本约定。
- [数据抓取与环境快照](docs/data.md)：YGOPRODECK / masterduelmeta / Yugipedia 抓取器与解析器、卡片对应、MD 禁限表生成与人工校对流程、meta 份额与代表卡表、`ygorl env build / check`、md-2026-09 快照统计。
- [语义协同图](docs/synergy.md)：CardScripts 脚本挖掘、边语义、解析覆盖率、真实 meta 引擎包召回检验（md-2026-09）与代理召回（历史）、可选 Yugipedia 关系边、引擎包枚举。
- [求解器示范集](docs/solver.md)：封装 ygo-combo-solver 求解起手展开线（含 `--fire` 手坑变体），在我们的核心里新鲜重放验证并转成动作下标，示范集 JSONL 格式、批量驱动与成本。
- [行为克隆预热](docs/bc.md)：求解器示范 → 策略网络的 BC（T4a.2）：样本构造、检查点格式与 PolicyAgent 加载（policy:PATH）、线复现率 / 未见起手场面质量 / 对 Random 的实测；对 Random 失败的根因与补救实验（Greedy 战斗示范、PPO 热启动、只在第 1 回合的 BC 先验）。
- [组牌基因型](docs/genotype.md)：引擎包份数 + 泛用槽 + 额外卡组的表示、禁限 / 同名 3 张 / 40–60 / ≤ 15 硬约束与修复、变异 / 交叉算子、计数向量编码、10k 合法性验收。
- [代理模型](docs/surrogate.md)：牌组特征（计数向量、引擎包、卡片结构、可插拔的卡文本嵌入均值）、自助 ridge 集成与不确定性、DSA-ME 在线更新与采集规则、真实对局标签缓存、留出集误差实测。
- [漏斗第一层：求解器起手分析](docs/funnel.md)：候选牌组固定种子抽起手，用求解器判定最佳线存在率、卡手率、抗手坑率（`--fire`）与 combo 长度，作为廉价过滤与 QD 描述符；与真实首回合的配对检验（McNemar）和每套牌耗时 vs 预算（T5.6）。
- [调卡组](docs/tuning.md)：给定一套牌的单卡替换局部搜索：候选卡（同类型卡表 + meta 常见卡）、公共随机数的配对比较、逐轮淘汰，对手为环境 meta（T6.2）。
- [术语表](docs/glossary.md)：代码标识符与中文术语对照（卡片、区域、对局流程、引擎、组牌）。
- [训练规模与训练信号](docs/scaling.md)：为什么从零训练进展小、先前项目（ygo-agent、DouZero、Suphx 等）的做法、扩规模与训练信号的选项及其与设计的冲突点。
- [工程计划](docs/eng-plan.md)：里程碑 M0–M6、任务清单、依赖、验收标准、推进顺序。GitHub issues 与任务一一对应。

## 快速开始

依赖：Linux、[uv](https://docs.astral.sh/uv/)（≥ 0.8）、CMake（≥ 3.20）、支持 C++17 的编译器（GCC ≥ 9 / Clang ≥ 10）。
Python 3.11 由 uv 自动选择或下载；pybind11 与 scikit-build-core 作为构建依赖由 uv 自动安装。
Python 运行时依赖写在 `pyproject.toml`、锁定在 `uv.lock`，`uv sync` 一并安装：numpy（观测编码、评估指标）；
[nashpy](https://github.com/drvinceknight/Nashpy)（连带 scipy、networkx 等）用于对局矩阵的 Nash 均衡（`ygorl.eval.matchup`）。
[pyribs](https://pyribs.org)（PyPI 包名 `ribs`，连带 pandas、scikit-learn、numba 等）用于组牌进化的 MAP-Elites 档案（`ygorl.build.archive`）。
策略训练（M4，`ygorl.nets` / `ygorl.train`）另需 PyTorch，放在可选依赖组 `train` 里：`uv sync --extra train`
（从 PyPI 安装 Linux 版 torch，自带 CUDA 运行库，安装后约 5 GB；只跑引擎、评估与组牌不需要它，相关测试在未安装时自动跳过）。
AMD GPU（ROCm，例如 Ryzen AI Max+ 395 的 Radeon 8060S 核显）：`uv sync --extra train` 之后换成同版本的 ROCm 轮子
`uv pip install "torch==<uv.lock 中的版本>" --index-url https://download.pytorch.org/whl/rocm<系统 ROCm 版本>`（例如 `rocm7.2`），
之后用 `uv run --no-sync` 运行，否则 `uv run` 会按锁文件换回 PyPI 版；训练加 `--device cuda`（ROCm 也叫 `cuda`，见 [docs/training.md](docs/training.md)）。
卡文本 / 效果文本嵌入的离线生成（`tools/build_text_embeddings.py`，T5.2）另需 sentence-transformers，在可选依赖组 `text` 里：`uv sync --extra train --extra text`
（ROCm 机器上先按上面装好 ROCm 版 torch，再 `uv pip install "sentence-transformers>=3"`，以免换回 PyPI 的 torch）；首次运行从 Hugging Face 下载所选模型。训练与推理只读生成好的 `.npy`，不需要它。
开发工具在 `dev` 依赖组（`uv sync` 默认安装）：pytest（单测）、ruff（格式化与 lint）。
提交前跑 `tools/presubmit.sh`：原地格式化（`ruff format`）并 lint（`ruff check`），加 `--test` 再跑单测。CI 跑的是只检查、不改动的 `tools/presubmit.sh --check`。
可以装成 git pre-commit 钩子（只检查，不改动暂存的内容，装法见脚本开头注释）。
个别工具另有系统依赖：`tools/tsan/check.sh` 需要 ninja、GCC 的 libtsan 与 `setarch`（util-linux）；`tools/check_ygoprodeck.py` 需要能访问
YGOPRODECK API 的网络；`tools/crosscheck_banlist.py` 需要能访问 YGOPRODECK 与 Yugipedia 的网络（`--from` 离线重跑）。CI 与云端会话 hook 另装 ccache 以加速重编。
新增依赖用 `uv add <包名>`（可选组用 `uv add --optional <组> <包名>`）。

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
# 策略对局矩阵：random 与 greedy 在 3 套牌的 4 个牌组配对上对局（每个配对 4 局），输出强度排名、胜率矩阵、Nash 与 alpha-rank
uv run ygorl strength random greedy --decks tests/decks/snake_eye.ydk tests/decks/kashtira.ydk tests/decks/yubel.ydk --pairings 4 --workers 2 --out out/strength.json
# 环境：校验仓库里的 Master Duel 快照（卡池、禁限表、meta 卡组合法性），打印卡池大小、禁限张数与 meta 份额
uv run ygorl env check md-2026-09
```

加 `--env <版本或目录>` 即按该环境的规则对局，产物绑定环境版本；`ygorl matrix --env <版本>` 不给牌组时用环境的 meta 卡组，
结果写到 `environments/<版本>/artifacts/matrix/`（见 [docs/cli.md](docs/cli.md)）。Python API 见 [docs/engine.md](docs/engine.md)、
[docs/replays.md](docs/replays.md)、[docs/evaluation.md](docs/evaluation.md)。

重新抓取并生成 MD 环境：`uv run ygorl env build md-YYYY-MM`（需能访问 YGOPRODECK、masterduelmeta、Yugipedia；`--offline` 只用已下载的原始文件），流程与禁限表人工校对见 [docs/data.md](docs/data.md)。

修改 `csrc/`、`patches/`、`CMakeLists.txt` 或 `pyproject.toml` 后，`uv sync` / `uv run` 会自动重新编译扩展；
需要强制重编时用 `uv sync --reinstall-package ygorl`。

### 可选：PPO 自博弈训练（T4b.4）

需要 `uv sync --extra train`。单牌组对训练，产物写到 `out/train/<名字>/`（配置、词表、`metrics.jsonl`、`eval.jsonl`、
`checkpoints/`、`best.pt`；给 `--env` 时写到该环境的 `artifacts/train/`）。参数与产物说明见 [docs/training.md](docs/training.md) §8。

```bash
uv run python tools/train_ppo.py tests/decks/snake_eye.ydk tests/decks/kashtira.ydk --minutes 60 --out out/train/snake-kashtira
uv run python tools/train_ppo.py --resume out/train/snake-kashtira/checkpoints/latest.pt --minutes 30   # 续训
uv run python tools/train_ppo.py --summary out/train/snake-kashtira/metrics.jsonl                      # 首末指标
```

训练出的 checkpoint 可以作为 agent 用在任何命令里：`--agent-a policy:out/train/snake-kashtira/best.pt`（按策略采样；
`policy-greedy:` 取 argmax），例如 `ygorl arena tests/decks/snake_eye.ydk --vs tests/decks/kashtira.ydk --agent-a policy:... --agent-b greedy`。

### 可选：combo 求解器（示范集，T4a.1）

[ygo-combo-solver](https://github.com/96jonesa/ygo-combo-solver)（AGPL-3.0）不随仓库分发，由脚本在 `build/combo-solver/` 里按固定提交拉取源码、
用本仓库打过补丁的核心编译。另需 sqlite3 开发包（Debian/Ubuntu：`sudo apt install libsqlite3-dev`）、git、patch；
只有生成示范集时需要，默认测试在没有二进制时自动跳过相关用例。用法与示范集格式见 [docs/solver.md](docs/solver.md)。

```bash
tools/build_combo_solver.sh                                   # 首次约 40 秒；输出二进制路径
uv run python tools/solve_openings.py tests/decks/labrynth.ydk --hands 2 --solve-ms 10000 --workers 2   # 写 out/demos/solver.jsonl
uv run python tools/solve_blocking.py tests/decks/snake_eye.ydk --hands 2 --solve-ms 8000 --workers 2   # 自动阻断目标，写 out/demos/blocking.jsonl
```

### Claude Code 云端会话

两个脚本分工（参见 [Cloud environments 文档](https://code.claude.com/docs/en/cloud-environments)）：

- `.claude/cloud-setup.sh`：云环境的 **setup script**，以 root 在 Claude Code 启动前运行，结果被快照、约 7 天内的新会话直接复用。
  装 ccache 与 libsqlite3-dev（combo 求解器用），并执行一次 `uv sync --extra train` 预热 uv 缓存（PyTorch 约 3 GB 下载）与 ccache；
  若运行时仓库尚未克隆，则只把 Python 3.11 和固定版本的 torch 放进 uv 缓存（版本写在脚本里，升级 torch 时同步改）。冷启动实测约 1 分钟，
  始终以 0 退出。
- `.claude/hooks/session-start.sh`：SessionStart hook（在 `.claude/settings.json` 注册），只在 Claude Code on the web
  （`CLAUDE_CODE_REMOTE=true`）中运行，每个会话都跑且不进快照：补装缺失的工具（uv / cmake / ninja / g++ / ccache）、拉取子模块、
  `uv sync --extra train` 编译扩展并安装 PyTorch，会话开始时即可直接 `uv run pytest`。缓存已预热时约 15 秒（ccache 命中率 > 95%），
  没配 setup script 时约 1 分钟。脚本幂等，也可手动执行：`CLAUDE_CODE_REMOTE=true .claude/hooks/session-start.sh`。

在 [claude.ai/code](https://claude.ai/code) 的环境选择器里编辑（或新建）云环境：

1. **Setup script**：粘贴 `.claude/cloud-setup.sh` 的全部内容。
2. **Network access**：选 **Custom**，勾选 *Also include default list of common package managers*，Allowed domains 填下面的列表。
   默认的 Trusted 级别只放行包管理源与 GitHub，T5.1 数据抓取（YGOPRODECK、masterduelmeta、Yugipedia）与 T5.2 文本嵌入（HuggingFace 模型下载）
   需要的站点都被拒；`ppa.launchpadcontent.net` 是镜像自带的 deadsnakes PPA，缺了它 `apt-get update` 会报 403。

   ```text
   db.ygoprodeck.com
   ygoprodeck.com
   masterduelmeta.com
   www.masterduelmeta.com
   yugipedia.com
   huggingface.co
   *.huggingface.co
   *.hf.co
   ppa.launchpadcontent.net
   ```

3. **Environment variables**：不需要。GitHub 由会话的 GitHub 代理认证；若将来要用需授权的 HuggingFace 模型，Pro / Max 计划用环境的
   API credentials（host 填 `huggingface.co`），不要把 token 写进环境变量（用该环境的人都能读到）。

云端 VM 约 4 vCPU / 16 GB 内存 / 30 GB 磁盘、无 GPU：适合开发、单测、数据抓取与离线嵌入生成；T2.7 的 16 核吞吐数字与 M4 正式训练
需要在自有机器上跑（[self-hosted environment](https://code.claude.com/docs/en/self-hosted-environments) 或 Remote Control）。

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
├── .claude/                 # Claude Code 配置：settings.json + hooks/session-start.sh（每个云端会话的初始化）+ cloud-setup.sh（云环境 setup script）
├── .github/workflows/       # CI：ruff 格式与 lint（tools/presubmit.sh --check）；构建扩展 + pytest（另一 job 装 `train` 可选依赖跑依赖 torch 的测试）
├── pyproject.toml           # uv 项目 + scikit-build-core 构建配置
├── uv.lock
├── .python-version         # uv 使用的 Python 版本（3.11）
├── .gitmodules              # 子模块定义（third_party/）
├── CMakeLists.txt           # 构建 C++ 扩展 ygorl._core
├── cmake/                   # CMake 片段（ocgcore.cmake：复制核心、打补丁、编成静态库）
├── patches/ygopro-core/     # 对规则核心的补丁（确定性遍历顺序、Lua 字符串哈希种子、Lua 分配器钩子），构建时应用
├── third_party/             # git submodule：ygopro-core、CardScripts、BabelCDB、LFLists
├── environments/            # 环境版本目录（规范见 docs/environments.md）：md-2026-09 快照由 ygorl env build 生成；*/raw/ 原始抓取文件不进 git；<版本>/artifacts/meta_packages.json 为协同图召回检验用的 meta 引擎包；artifacts/decks/ + deck_corpus.json 为牌组语料（tools/build_deck_corpus.py）
├── csrc/                    # C++：core_backend（OCG_* 封装）、duel_pool（线程池）、host / obs_encoder（C++ 主机层与观测编码）、privileged（训练态对手真值）、event_encoder + event_binding（事件 token 流）、host_pool + worker_pool（C++ 步进环境）、arena（每局内存 arena 与快照）、binding（pybind11）；exports.map 为链接导出表
├── src/ygorl/               # Python 包
│   ├── cli.py               # 命令行入口 `ygorl`（argparse 子命令）
│   ├── paths.py             # 子模块数据路径（cards.cdb、脚本目录、禁限表）与环境根目录
│   ├── commands/            # 各子命令一个模块：duel / replay / branch / arena / matrix / env；__init__.py 放共用选项（牌组、环境、agent）
│   ├── agents/              # Agent 协议、RandomAgent、GreedyAgent、PolicyAgent；checkpoint.py（policy:<checkpoint> agent，锁步 C++ 主机）；registry.py（按规格构造 agent 与可 pickle 的 factory，供 CLI）
│   ├── build/               # 组牌：Lua 脚本读取器、过滤条件 IR、脚本挖掘协同图（synergy_graph）、Yugipedia 关系边（relations）、引擎包枚举（packages）、基因型与算子（genotype）、代理模型（surrogate）与真实对局标签（labels）、漏斗第一层起手分析（funnel）与真实首回合验证（first_turn）、调卡组（tuner）、进化的逐卡信号（diagnose、signals）、评估预算（selection：前二 Thompson 采样与序贯复核）与 critic 控制变量（control）、一组改动的部分析因评估（factorial）、进化步骤（evolve：一轮进化、谱系与续跑）、档案精英交叉子代（crossover）、语料关联规则候选（rules）、引擎成员与加入池（deck_engine）、掩码卡组模型与学习的子代（deck_model、learned）、改动价值模型与它的标签（value_model、edit_labels）、与 RL 的共同进化循环（coevolve）、信号库热启动（warmstart）与 MAP-Elites 档案（archive）
│   ├── cards/               # cards.cdb、禁限表（.lflist.conf）、牌组（.ydk）、合法性校验
│   ├── data/                # Environment 加载与校验；数据抓取与环境构建（fetch、cardmap、ygoprodeck、masterduelmeta、yugipedia、build）
│   ├── engine/              # 消息解码、动作模型、单局 Duel、卡片查询解析（query.py）、回放（含 .yrp / .yrpX 读取）、分支探索（branch.py）、课程模式（curriculum.py）、残局构造（puzzle.py）、逐步推进与快照（duel.py 的 DuelSession）、主机追踪器（tracker.py 的 DuelTracker）；constants.py 为生成文件
│   ├── env/                 # 向量化环境：VecDuelEnv（C++ 线程池）、DuelEnv、run_games、paired_specs；driver.py 为槽位上整局对弈的对局驱动（规格队列、开局失败、槽位复用、决策批、放弃对局）；encoding.py 参考编码器；privileged.py 训练态对手真值与信念头目标；belief_prior.py 公开证据、meta 卡表与 HDT 式过滤（信念头的先验、输入特征与基线）；events.py 事件 token 流参考实现；encoded.py 为 C++ 步进的 EncodedVecEnv；observer.py 为 DecisionPoint 的观测（参考编码器 + 事件流，与 EncodedVecEnv 一致）
│   ├── nets/                # 策略网络（PyTorch，train 可选依赖）：config、text（冻结文本表）、batch（观测拼批）、encoders、history（GTrXL / LSTM）、heads、policy（PolicyNet）、actor_critic（PolicyNet + 特权 Q / V critic）、belief（信念头、掩码损失、BeliefPolicy）、agent（检查点读写、PolicyAgent 用的 NetPolicy）
│   ├── eval/                # 评估：配对种子 Arena、对局矩阵与 Nash / alpha-rank、信念头校准指标与基线
│   ├── solver/              # combo 求解器封装（combo_solver.py）、目标场面（targets.py）、线的重放验证与示范集格式（demo.py）、起手批量求解（batch.py）
│   └── train/               # 策略训练（需 train 可选依赖）：advantages.py（GAE / Expected-SARSA(λ) / VRPO 优势）、critic.py（特权 Q 头 + V 头与损失）、rollout.py（EncodedVecEnv 上的 rollout 收集）、ppo.py（PPO 更新与可插拔策略目标）、selfplay.py（快照池 + keep-best、牌组池、配对发局）、trainer.py（训练循环、评估、续训、日志）、cli.py（训练配置的命令行：每个字段一个参数）、checkpoint.py、toy.py（玩具博弈 Nim）、bc.py（求解器示范的行为克隆预热与评估）、heuristic_demos.py（启发式 agent 第 2 回合起的决策 → BC 样本）
├── tools/                   # 开发脚本：提交前检查（presubmit.sh：ruff 格式化 + lint）、PPO 自博弈训练（train_ppo.py）、combo 求解器构建（build_combo_solver.sh）、起手批量求解（solve_openings.py）、任意牌组的阻断场面示范（solve_blocking.py）与示范集复验（verify_demos.py）、常量生成、测试牌组 / 代理引擎包生成、meta 引擎包推导（make_meta_packages.py）、协同图构建、引擎包列表、基因型采样与合法性检查、代理模型实验（surrogate_experiment：标注 + 留出集误差）、漏斗第一层评估与验收实验（funnel_eval.py、validate_funnel.py）、预算研究（funnel_budget.py）、压力测试、确定性扫描、YGOPRODECK 核对、MD 禁限表交叉核对（crosscheck_banlist.py）、arena 基准（ygorl arena 的包装）、信念基线表、信念头实验（train_beliefs.py）、行为克隆训练与评估（train_bc.py）、Greedy 示范录制（greedy_demos.py）、BC 对 Random 失败的根因诊断（diagnose_bc.py）、吞吐基准、课程模式检查、快照检查、线程池与逐局比对（check_pool.py）、C++ 编码 / 事件流交叉校验、组牌进化的先行测量与评估器对照（deckevo_*.py：M1–M5、每接受一次改动的对局数与被接受改动的重验、析因评估与逐个筛的对照）、组牌进化的一轮（evolve_decks.py）、牌组数据集构建（build_deck_dataset.py：masterduelmeta 全部历史卡表 → out/deck_dataset/<版本>/）、掩码卡组模型的训练与评估（train_deck_model.py → out/deckmodel/）、改动价值模型的拟合与留一牌组交叉验证（fit_value_model.py → out/valuemodel/）、共同进化循环（coevolve.py）；tsan/ 为 ThreadSanitizer 检查
├── tests/                   # pytest 单测（test_readme.py 执行 README 的命令行示例）；decks/ 放 10 套测试牌组及其求解目标（solver_targets.json），data/ 放测试数据（含代理引擎包、泛用卡池）
├── docs/
│   ├── design/              # 设计文档（按主题拆分）
│   ├── glossary.md          # 术语表（标识符与中文术语对照）
│   ├── engine.md            # 引擎层：CoreBackend、消息、动作模型、确定性
│   ├── encoding.md          # 观测编码规范（卡片表、全局向量、候选动作表、训练态真值、事件 token 流）
│   ├── nets.md              # 策略网络：编码器、局面 Transformer、历史模块（GTrXL / LSTM）、动作打分头、文本向量接口
│   ├── evaluation.md        # 基线 agent 与评估
│   ├── belief-eval.md       # 信念校准评估：指标定义与基线数字
│   ├── belief-heads.md      # 信念头：五个头、损失掩码、meta 先验与实验数字
│   ├── training.md          # 策略训练：优势估计与特权 critic 的公式、符号约定、rollout 数据布局；PPO 自博弈训练循环
│   ├── benchmarks.md        # 基准结果（实测数字、commit、日期）
│   ├── scaling.md           # 训练规模与训练信号：现状、先前项目、选项
│   ├── environments.md      # environments/<version>/ 目录规范
│   ├── data.md              # 数据抓取、禁限表校对流程、环境快照统计
│   ├── replays.md           # 回放格式、.yrpX 导出与 .yrp / .yrpX 读取
│   ├── solver.md            # combo 求解器构建、线的验证与转换、示范集格式、批量驱动
│   ├── bc.md                # 行为克隆预热：样本、训练、检查点、评估与实测
│   ├── cli.md               # 命令行 ygorl 各子命令
│   ├── branching.md         # 分支探索：fork(replay, t)、ygorl branch、限制
│   ├── curriculum.md        # 课程模式、先后攻配平、增广开局标志位
│   ├── synergy.md           # 脚本挖掘协同图与引擎包
│   ├── genotype.md          # 组牌基因型、硬约束与变异 / 交叉算子
│   ├── surrogate.md         # 代理模型：特征、ridge 集成、DSA-ME 在线更新、标签与留出集误差
│   ├── funnel.md            # 漏斗第一层：求解器起手分析、过滤与描述符、配对检验验收、预算研究
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
