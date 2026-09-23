# 漏斗第一层：求解器起手分析（`ygorl.build.funnel`，T5.6）

> 对应设计文档 [05-deck-building.md](design/05-deck-building.md)「ygo-combo-solver 作漏斗第一层：最佳线存在性 + 抗手坑率（`--fire`）替代单纯的起手概率」
> 与 [06-architecture.md](design/06-architecture.md)「漏斗 (1) 求解器起手分析（最佳线存在性、卡手率、抗手坑率）」「描述符 = {…, combo 长度, 卡手率}」，
> 工程计划 T5.6（[#43](https://github.com/n0tevi1/ygorl/issues/43)）。代码在 `src/ygorl/build/funnel.py` 与 `src/ygorl/build/first_turn.py`，
> 求解器本身见 [solver.md](solver.md)（T4a.1），基因型见 [genotype.md](genotype.md)（T5.5）。

候选牌组（或基因型解码出的牌组）抽固定种子的一批起手，每手交给 T4a.1 求解器找到目标场面的展开线；找到的线再在 `--fire` 手坑下重测。
汇总成：最佳线存在率、卡手率（含 Wilson 区间）、抗手坑率、combo 长度与评估耗时；再给出一个廉价的通过 / 淘汰判定和供 QD 档案（T5.8）使用的描述符。

```
ygorl.build.funnel       FunnelConfig / evaluate_deck / evaluate_genotype → FunnelResult；FunnelFilter；paired_table / mcnemar_exact
ygorl.build.first_turn   「真实首回合」：标准对局里重放求解器的线、Greedy、随机探索 rollout（验证用）
tools/funnel_eval.py     批量评估 .ydk，每套牌一行 JSON（可续跑）
tools/validate_funnel.py 验收实验：漏斗卡手 vs 真实首回合 / 大预算求解器的配对检验与耗时
```

## 用法

```python
from ygorl.build.funnel import FunnelConfig, FunnelFilter, evaluate_deck, evaluate_genotype
from ygorl.cards.ydk import load_ydk

cfg = FunnelConfig(hands=12, solve_ms=5000, fire=(14558127,), fire_ms=3000, workers=2)
res = evaluate_deck(load_ydk("tests/decks/labrynth.ydk"), ["1225009", "5380979@szone:fd"], cfg, filter=FunnelFilter(max_brick_rate=0.5))
res.passed, res.reasons            # 过滤结论（淘汰时给出原因）
res.brick_rate, res.brick_interval(), res.hand_trap_survival, res.combo_length()
res.descriptors()                  # {"brick_rate", "combo_length", "hand_trap_survival"}，供 T5.8
res.solver_s, res.within_budget    # 求解器进程秒与预算
res.to_json()                      # 可存档（format "ygorl-funnel"），FunnelResult.from_json 读回

res = evaluate_genotype(space, g, targets, cfg, name="elite-17")   # space.decode(g) 后同上；结果带 genotype_to_json(g)
```

```bash
export YGORL_COMBO_SOLVER=build/combo-solver/bin/combosolver       # 或先 tools/build_combo_solver.sh
uv run python tools/funnel_eval.py tests/decks --hands 12 --workers 2                       # 默认带过滤与早停
uv run python tools/funnel_eval.py tests/decks --no-filter --keep-demos --out out/funnel/validate.jsonl
uv run python tools/validate_funnel.py out/funnel/validate.jsonl --rollouts 1000 --reference-ms 30000 --workers 2
```

`targets` 是一个目标场面（卡片密码列表，语法同 [solver.md](solver.md#目标场面)），或若干个备选场面（列表的列表）：
一手按顺序尝试备选，到达任一即算有线（有备选时卡手的代价是 备选数 × `solve_ms`）。

## 定义

| 量 | 定义 |
|----|------|
| 起手 | 第 `i` 手 = 主机洗牌 `hand_seed(seed, "funnel", i)` 的顶上 5 张（`sample_hand`，与对局、求解器相同）。种子**不含牌组名**：同样张数的牌组用同一组洗牌置换（共同随机数），候选之间的比较是配对的；`opening_hands(deck, cfg)` 列出这些起手 |
| 有线 / 卡手（brick） | 求解器在 `solve_ms` 内找到**且在我们核心里重放验证通过**的线（`solve_hand` 的 `solved`）→ 有线；预算内没找到（`unsolved`）→ 卡手。`error` / `unverified` 是工具链故障，不计入分母，单独计数 |
| 最佳线存在率 / 卡手率 | 有线手数 / 非错误手数；卡手率 = 1 − 存在率，附 95% Wilson 区间 |
| 手坑存活 | 对有线的手，按 `--fire` 手坑逐张跑 `solve_fire`：`no_window`（这条线上手坑无可发动时点）算存活；`solved` 且求解器**从每个**可发动时点都恢复了终场（`converted == windows`，对手自选时点）算存活；其余不存活 |
| 抗手坑率 `hand_trap_survival` | 「起手有线**且**对每张 `--fire` 手坑都存活」的手数 / 非错误手数（无条件）；`survival(fire, given_line=True)` 是以有线为条件的比例；`window_recovery(fire)` 是全部可发动时点中被恢复的比例（较宽松的补充指标） |
| combo 长度 | 有线手的最佳线（求解器评分最优）：`combo_actions` = 召唤与发动次数（求解器评分 `actions`），`combo_decisions` = 来自求解器的动作步数；报告均值、中位数、最小、最大 |
| 耗时 | `solver_s` = 各手求解器墙钟之和（plain + fire），即**单线程求解器进程秒**，与并行进程数无关；`wall_s` 是整套评估的墙钟 |

`FunnelResult.descriptors()`：`brick_rate`、`combo_length`（`combo_actions` 均值）、`hand_trap_survival`。
没有一手有线时 `combo_length` 为 NaN，不带 `--fire` 时 `hand_trap_survival` 为 NaN；这类牌组先被过滤器淘汰，不进档案。
设计文档描述符里的「手坑数」直接取基因型的 `role_counts(g)["hand_trap"]`（T5.5），不在这里算。

## 过滤器与早停（`FunnelFilter`）

- `max_brick_rate`（默认 0.5）：卡手率上限；`min_hand_trap_survival`（默认 0，即不设门槛）：抗手坑率下限；全部手都是 `error` 的牌组淘汰。
- **早停**：已出现的卡手数超过 `max_brick_rate × 计划手数` 时，这套牌已不可能通过，剩余的手不再求解（`stopped_early`，结论为淘汰）。
  卡手每手要花满 `solve_ms`，而随机基因型大多是高卡手率，早停把它们的成本压到约 `max_brick_rate × hands × solve_ms`。
- 过滤器只看漏斗第一层自己的量；是否把 `brick_rate` 的 Wilson 上界、抗手坑率等作为 fitness 的一部分由 T5.8 决定。

## 预算

设计文档与工程计划没有给出第一层的数字预算，这里按用途定为 **每套牌 120 求解器进程秒**（`DEFAULT_BUDGET_S`）：
第一层要在代理模型（T5.7）和真实对局（第三层：每候选微调 B 局 + 数百局评估，按进程小时计）之前把大部分候选筛掉，
目标是 16 核机器一天能筛约 1 万个候选（16 × 86,400 / 10,000 ≈ 138 秒 / 套）。默认参数（12 手、`solve_ms` 5 秒、Ash Blossom 一张、`fire_ms` 3 秒）
的最坏情况是 12 × 5 秒卡手 = 60 秒 plain，再加有线手的 `--fire`。**这个数字是工程取值，需要设计方确认**（见文末「待定」）。

## 验收实验：卡手率与真实首回合一致（配对检验）

### 设计

同一批起手上比较两种判定（配对）：

- **漏斗**：上面的默认配置，每手 `solved` / `unsolved`。
- **真实首回合**（`ygorl.build.first_turn`）：用我们的主机开一局**标准对局**——同样的卡组顺序（起手在顶上）、同样的核心种子字、同样的白板对手，
  但**不带 `DUEL_PSEUDO_SHUFFLE`**（效果里的「洗切卡组」真的洗）；先攻玩家打完第 1 回合，到第 2 回合第一个决策时检查目标卡是否在场。三个玩家：
  1. **求解器的线**（有线的手）：在伪洗牌对局里逐步重放示范线，每一步在真实对局的候选里按「做什么」（动作种类、卡片密码、区域；卡组 / 手牌内的序号不比较）匹配；
     匹配不到即线在真实对局里断了，之后被动结束回合再检查场面；
  2. **Greedy**（`GreedyAgent`，4 个种子）；
  3. **随机探索**（`ExplorerAgent`：空闲阶段以 0.9 概率做一个非阶段切换的动作，其余决策均匀随机），从开局快照出发 1,000 次 rollout。

  任一玩家到达目标即该手「真实有线」。
- **参照求解器**：漏斗判为卡手的手用 6 倍预算（30 秒、另一个求解器种子）再解一次。

比较四组配对：漏斗 vs (a) 真实首回合（三个玩家合并），(b) 只用 Greedy + 随机探索（与求解器独立的玩家），(c) 参照求解器，(d) 已知最佳（a 或 c）。
检验用精确 McNemar（不一致对的二项检验，`mcnemar_exact`），H0：两种判定的卡手率相同。

### 数据与环境

| 项 | 值 |
|----|----|
| 命令 | `tools/funnel_eval.py tests/decks --hands 12 --no-filter --keep-demos --workers 2 --out out/funnel/validate.jsonl`；`tools/validate_funnel.py out/funnel/validate.jsonl --rollouts 1000 --greedy-seeds 4 --reference-ms 30000 --workers 2` |
| 代码 | 本提交（基于 main `7547920`），求解器 `e0c7221` |
| 日期 | 2026-09-23 |
| 环境 | 无（MR5、8000 LP、起手 5 张）；`tests/decks/` 10 套测试牌组 × 12 手 = 120 手，目标见 `tests/decks/solver_targets.json`；`--fire` 为 Ash Blossom & Joyous Spring（14558127） |
| 机器 | 4 核云容器，与其他任务共用（负载 4–6）；求解器 2 进程 × 1 线程、固定求解器种子 1。求解按墙钟计预算，负载会让同一预算下解出的手变少 |

### 结果：配对检验

| 漏斗卡手 vs … | 两者都卡手 | 只有漏斗卡手 | 只有对方卡手 | 两者都有线 | 漏斗卡手率 | 对方卡手率 | 一致率 | McNemar p |
|------|------|------|------|------|------|------|------|------|
| (a) 真实首回合（线 + Greedy + 探索） | 48 | 3 | 1 | 68 | 0.425 | 0.408 | 96.7% | **0.625** |
| (b) 仅 Greedy + 探索 | 48 | 3 | 11 | 58 | 0.425 | 0.492 | 88.3% | 0.057 |
| (c) 参照求解器（30 秒） | 47 | 4 | 0 | 69 | 0.425 | 0.392 | 96.7% | 0.125 |
| (d) 已知最佳（a 或 c） | 46 | 5 | 0 | 69 | 0.425 | 0.383 | 95.8% | 0.063 |

- **(a) 验收项**：漏斗卡手率 42.5% 与真实首回合 40.8% 在 120 对上无显著差异（p = 0.63），一致率 96.7%。
  - 「只有对方卡手」1 手（purrely 第 4 手）：求解器的线在真实对局第 22 步断开——Purrely 的效果先洗切卡组再看卡组顶，伪洗牌下与真实洗牌下顶上的卡不同（[solver.md 的限制](solver.md#限制)）。
    69 条线中 68 条在真实对局里原样走通并到达目标。
  - 「只有漏斗卡手」3 手：snake_eye 第 1、7 手与 purrely 第 10 手，随机探索在 1,000 次 rollout 中各有 1 次到达目标；其中两手参照求解器 30 秒也解出（snake_eye 第 1 手 30 秒仍未解出）。
- **(b)**：去掉求解器的线后，Greedy 与随机探索在漏斗有线的 69 手里只到达 58 手（Greedy 20 手；480 局 Greedy 首回合只有 46 局到达目标），
  独立玩家的卡手率 49.2% 高于漏斗——弱玩家的「卡手」混着「不会打」，不能单独作为真值；这一行 p = 0.057，接近显著。
- **(c)**：6 倍预算只多解出 4 / 51 个卡手（purrely 2、snake_eye 1、tenpai 1，用时 5.5–17.1 秒），方向一致（漏斗只会高估卡手）。
- **(d) 偏差**：相对已知最佳，漏斗把 5 / 120 = 4.2% 的手误判为卡手（95% Clopper-Pearson 区间 1.4%–9.5%），没有反向误判；
  p = 0.063 未过 0.05，但全部不一致都在同一方向，检验的功效有限（5 个不一致对时 p 的最小可能值就是 0.0625）。
  所以准确的表述是：**在 120 手的样本上未检出差异；漏斗卡手率存在约 4 个百分点的系统性高估（预算不足导致的假卡手），样本增大后很可能显著**。
  对过滤器而言这是保守方向（误淘汰而非误放行），对描述符而言是近似同幅度的平移。

### 结果：每套牌

| 牌组 | 有线 / 12 | 卡手率 [95% Wilson] | 抗手坑率 | 有线时存活 | 时点恢复率 | combo 长度（动作） | 线的步数 | 求解器进程秒 | 墙钟（2 进程） | 真实首回合卡手 | 已知最佳卡手 |
|------|------|------|------|------|------|------|------|------|------|------|------|
| branded_despia | 7 | 0.42 [0.19, 0.68] | 0.00 | 0.00 | 0.16 | 5.0 | 25.0 | 85.5 | 46.4 | 5 | 5 |
| fiendsmith_ryzeal | 8 | 0.33 [0.14, 0.61] | 0.17 | 0.25 | 0.14 | 4.3 | 28.1 | 55.7 | 29.5 | 4 | 4 |
| kashtira | 8 | 0.33 [0.14, 0.61] | 0.42 | 0.62 | 0.56 | 6.1 | 37.3 | 60.3 | 31.4 | 4 | 4 |
| labrynth | 5 | 0.58 [0.32, 0.81] | 0.33 | 0.80 | 0.00 | 1.8 | 10.4 | 47.8 | 24.7 | 7 | 7 |
| purrely | 4 | 0.67 [0.39, 0.86] | 0.17 | 0.50 | 0.53 | 8.5 | 40.3 | 89.1 | 45.0 | 8 | 6 |
| snake_eye | 8 | 0.33 [0.14, 0.61] | 0.00 | 0.00 | 0.08 | 9.0 | 43.5 | 115.9 | 58.4 | 2 | 2 |
| tearlaments | 4 | 0.67 [0.39, 0.86] | 0.17 | 0.50 | 0.33 | 4.3 | 17.3 | 54.8 | 28.4 | 8 | 8 |
| tenpai | 9 | 0.25 [0.09, 0.53] | 0.17 | 0.22 | 0.13 | 7.2 | 35.7 | 85.9 | 44.7 | 3 | 2 |
| voiceless_voice | 7 | 0.42 [0.19, 0.68] | 0.17 | 0.29 | 0.00 | 4.9 | 29.9 | 67.5 | 36.2 | 5 | 5 |
| yubel | 9 | 0.25 [0.09, 0.53] | 0.50 | 0.67 | 0.20 | 2.1 | 12.7 | 45.6 | 23.8 | 3 | 3 |
| **合计** | **69 / 120** | **0.425** | | 25 / 69 | | 5.3（1–15） | | 均值 70.8 | | 49 | 46 |

`--fire` 69 次：19 次 `no_window`、17 次 `solved`（其中只有部分时点恢复的不算存活）、33 次 `unsolved`；有线的手里 25 / 69 存活。
测试牌组每套带约 20 张手坑与泛用卡，卡手多为「只有手坑」的起手；目标场面是 T4a.1 挑的代表性终点，不是最优终场，所以 combo 长度只在同一目标定义下可比。

### 结果：耗时与预算

| 项 | 值 |
|----|----|
| 每套牌求解器进程秒 | 均值 **70.8**，最小 45.6（yubel），最大 **115.9**（snake_eye）；**10 / 10 在 120 秒预算内** |
| 每套牌墙钟（2 进程） | 23.8–58.4 秒，10 套共 369 秒 |
| 单手 | 有线 plain 平均 2.0 秒（69 手），卡手 5.3 秒（51 手，即预算 + 启动），`--fire` 每次 4.4 秒 |
| 早停 | 本实验关闭（验证需要每手结果）；开启时卡手率 > 0.5 的牌组在第 7 个卡手处停止，成本上限约 7 × 5.3 ≈ 37 秒 + 此前有线手 |

snake_eye 接近预算上限：它有线的手多、线长（平均 9 个动作、43.5 步），`--fire` 在 12 个时点上逐个恢复。预算紧张时先降 `fire_ms` 或只对前若干有线手跑 `--fire`。

验证本身的成本不计入漏斗：120 手的真实首回合玩家中位 3.5 秒 / 手，但有 8 手超过 100 秒——随机探索碰到「宣言卡名」决策时，
动作模型要为整个卡池逐张判断可否宣言（`engine/actions.py` 的 `is_declarable`，约 1.5 万张，每次约 2 秒）。参照求解器 51 手 × 30 秒。

## 局限

- **目标场面要人给。** 求解器只判定「到达给定目标」。测试牌组的目标来自 `tests/decks/solver_targets.json`；基因型解码出的牌组还没有自动目标——
  需要从引擎包推出（例如包内额外卡组怪兽作为备选场面），这属于 T5.8 接入时的决定。`evaluate_genotype` 目前要求调用方传目标。
- **卡手 = 预算内没找到。** 求解按墙钟计预算，受机器负载影响，结果不可逐位复现（`--threads 1` + 固定求解器种子只固定 rollout 顺序）；
  实测相对 6 倍预算有约 4 个百分点的假卡手（上节）。需要可复现时可改用求解器的 `--max-rollouts`（尚未接入 `HandJob`）。
- **伪洗牌。** 求解器的线在 `DUEL_PSEUDO_SHUFFLE` 下成立；依赖「洗牌后看卡组顶」的牌组（本实验中的 purrely）在真实对局里可能断线，漏斗会高估这类牌组的有线率（本实验 1 / 69）。
- **对手是白板、手坑逐张单次。** 抗手坑率只测 `--fire` 列出的手坑、最佳一条线、每个时点发动一次；多张手坑、对手的其它互动不在范围内。
  「存活」要求从每个时点都恢复，比「对手随机时点」严格。
- **验证的「真实首回合」是下界。** Greedy 与随机探索都弱；因此验收检验以「真实首回合玩家 ∪ 求解器的线」为观测，另报了独立玩家与大预算求解器两组对照。
- **样本小。** 每套 12 手时单套卡手率的 Wilson 区间宽约 ±0.25；QD 描述符的分辨率受此限制，T5.8 可按档案格宽调整 `hands`。

## 结果格式（`format = "ygorl-funnel"`，`format_version = 1`）

| 键 | 含义 |
|----|------|
| `deck` | `{"name", "main", "extra"}` |
| `targets` | 备选目标场面列表，`password@zone[:fd]` |
| `config` | `FunnelConfig` 全部字段（手数、种子、`solve_ms`、`fire`、`fire_ms`、进程数、`budget_s`…） |
| `environment` | `{"version", "fingerprint"}`，无环境时为 `null` |
| `genotype` | `genotype_to_json(g)`（只在 `evaluate_genotype` 时） |
| `hands[]` | 每手：`index`、`hand_seed`、`hand`、`status`（solved / brick / error）、`target`（到达的备选序号）、`combo_actions`、`combo_decisions`、`burned`、`fire`（按手坑密码：`status`、`windows`、`converted`、`survives`、`wall_s`）、`solver_s`、`error`、`demos`（`keep_demos` 时为求解器的 `Demonstration` 记录） |
| `planned_hands`, `stopped_early`, `wall_s`, `passed`, `reasons` | 计划手数、是否早停、墙钟、过滤结论与原因 |
| `summary` | `FunnelResult.summary()`：各项比率、区间、combo 长度统计、`solver_s`、`within_budget` |

## 待定

- 第一层预算的正式数字（本文按 120 求解器进程秒 / 套取值）与默认过滤门槛（卡手率 ≤ 0.5、不设抗手坑门槛）需要设计方确认。
- 基因型的目标场面如何自动生成（见「局限」），决定后 T5.8 才能把本层接到 MAP-Elites 上。
