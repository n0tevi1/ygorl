# 基线 agent 与评估（`ygorl.agents`、`ygorl.eval`）

本文说明 M3 的对手与裁判：Agent 协议与基线 agent（T3.1）、配对种子 Arena（T3.2）、对局矩阵与 Nash / alpha-rank（T3.3）。对应 [工程计划](eng-plan.md) M3。

## Agent 协议（`ygorl/agents/base.py`）

```python
class Agent(Protocol):
    def act(self, point: DecisionPoint) -> int: ...   # 返回 point.actions 的下标

AgentFactory = Callable[[int], Agent]                 # 种子 -> 新 agent
```

约定：

- 返回 `range(len(point.actions))` 内的整数。列出的动作全部合法，越界时 `Duel` 抛 `ValueError`。
- 一次调用 = 一步：多选按张拆步，同一个 `point.decision` 可能连续问多次（见 [engine.md](engine.md) 动作模型）。
- `point.player` 是**引擎玩家**（0 先攻），不是牌组侧 a/b。
- agent 有状态（RNG、回合内记忆），每局每个座位一个实例；由 `AgentFactory(seed)` 构造，**同种子同对局必须做出同样的选择**。这是回放、配对种子和「结果与进程数无关」的前提。
- 可选属性 `last_probs`：上一次调用时对各动作给出的概率，`Duel(record_steps=True)` 会记进回放。
- 并行 `Arena` 需要把 factory 发到子进程，所以 factory 必须可 pickle：类本身（`GreedyAgent`）、模块级函数或 `functools.partial`。
- `agent_name(x)` 给类、实例、函数或 partial 取可读名字（优先类属性 `name`）。

## 基线 agent

| agent | 行为 |
|-------|------|
| `RandomAgent(seed)` | 均匀随机合法动作 |
| `GreedyAgent(seed, cards=None)` | 确定性启发式，见下 |
| `PolicyAgent(policy, seed=None, greedy=False, temperature=1.0)` | `policy(point)` 对每个动作打分（logit），按 `softmax(score / T)` 采样或取 argmax；暴露 `last_probs`。M4 的网络策略从这里接入 |

### GreedyAgent 规则

只看当前决策与少量回合内记忆，**不追踪场面**；攻击力 / 守备力取卡片数据库的原始值（`CardDB`，默认 `default_cards()`），不含场上增减。

| 决策 | 选择（按优先级） |
|------|------------------|
| 主要阶段 IDLECMD | 发动 > 特殊召唤 > 通常召唤（攻击力最高）> 盖放怪兽 > 盖放魔陷 > 进入战斗阶段 > 结束阶段 |
| 战斗阶段 BATTLECMD | 直接攻击 > 攻击力最高的怪兽攻击 > 发动 > 主要阶段 2 > 结束阶段 |
| 攻击目标（`MSG_HINT` 549 之后的 SELECT_CARD） | 攻击力严格高于其攻击力（攻击表示）/ 守备力（守备表示）的目标中最强的一个；打不过任何目标时取消攻击，本回合不再用这只怪兽攻击；不能取消时选最弱的目标 |
| CHAIN | 能连锁就连锁 |
| EFFECTYN / YESNO | 是 |
| POSITION | 攻击力 ≥ 守备力 → 表侧攻击，否则表侧守备 |
| SELECT_CARD / UNSELECT_CARD / SUM | 能选就继续选（种子随机），没得选才 finish；从不 unselect |
| TRIBUTE | 攻击力最低的先解放 |
| 其他（OPTION、PLACE、ANNOUNCE、SORT 等） | 种子随机 |

防卡死：同一发动（卡、位置、效果）每回合最多 `max_repeats = 3` 次，每回合主动的主要阶段动作最多 `max_turn_actions = 100` 个，之后只进战斗 / 结束阶段。

「有攻击者才进战斗阶段」的近似：agent 不追踪场面，主要阶段 1 能进战斗阶段就进，到了 BATTLECMD 没有可攻击的怪兽再去主要阶段 2 / 结束阶段。

验收（T3.1）：`tests/test_agents.py` 让 Greedy 在 10 套测试牌组上各打一局（对 Random），零 retry、零未知消息，且都以胜负或回合上限结束；开发时的 200 局抽样全部以 `MSG_WIN` 结束，Greedy 胜率约 90%。

`AGENTS = {"random": RandomAgent, "greedy": GreedyAgent}` 按名字登记基线 agent，供工具脚本与 CLI 使用。

## Arena（`ygorl/eval/arena.py`，T3.2）

```python
from ygorl.agents import GreedyAgent, RandomAgent
from ygorl.eval import Arena

arena = Arena(GreedyAgent, RandomAgent, env=None, workers=2)   # 参数是 factory：seed -> agent
report = arena.run(deck_a, deck_b, pairs=100, seed=0)          # 200 局
print(report.summary())
reports = arena.run_many([(a1, b1), (a2, b2, 7)], pairs=10)     # 多个对阵共用一个进程池；第三项可指定种子
```

**配对种子**：第 `p` 对用同一个对局种子 `s_p = derive_seed(seed, p)` 打两局，牌组相同、先后攻互换（`first=0` / `first=1`）。
Arena 自己用 `shuffle_deck(main, s_p, 0/1)` 洗好双方主卡组，再以 `DuelConfig(shuffle_decks=False)` 按原顺序加载，
所以同一对的两局起手与抽卡顺序完全相同；两局的 agent 种子也相同。这样能抵消大部分抽牌运气和先攻优势。
`DuelConfig.shuffle_decks=False` 传给 Arena 时不洗牌（按牌组文件顺序）。

**确定性**：每局由 `GameSpec`（种子、先攻、洗好的牌组、agent 种子、环境、配置）完全决定；
`workers > 1` 时用 `multiprocessing` 进程池 `map`，结果按局序排列，与进程数无关（`test_results_do_not_depend_on_worker_count`）。
factory 必须可 pickle。某局抛异常时记为 `reason="exception"`（胜者为空）并计入 `errors`，不中断整个评估。

**报告** `ArenaReport`（均为 agent a 视角）：

| 字段 | 含义 |
|------|------|
| `games`、`wins` / `losses` / `draws` | 局数与胜负平 |
| `win_rate` | `(wins + draws / 2) / games`，平局算半胜 |
| `ci`、`confidence` | `win_rate` 的 Wilson 区间（默认 95%）；`significant()`：区间不含 0.5 |
| `as_first` / `as_second` | a 先攻 / 后攻时的 `SideStats`（局数、胜负平、胜率） |
| `first_player_win_rate` | 先攻方胜率（衡量先攻优势） |
| `reasons`、`retries`、`unknown_messages`、`errors`、`mean_turns` | 终局原因计数与健康指标 |
| `environment` | 有环境时为 `Environment.stamp()` |
| `records` | 每局的 `GameRecord`（对号、种子、先攻、胜者、原因、回合、决策数、LP 等） |

`to_dict()` 可直接写 JSON；`merge(reports)` 把同一对 agent 的多份报告合并成一份。

统计说明：平局按半胜计时，胜负平得分的方差不超过 `p(1 - p)`，所以 Wilson 区间偏保守；
但区间假设各局独立，而配对的两局相关（同一起手），严格的配对检验留待需要时再加。

**基准**：`uv run python tools/arena.py --games 2000 --workers 2` 让两个 agent 在 10 套测试牌组的全部有序组合
（含镜像，100 组）上各打 `2 × pairs` 局并汇总，结果记在 [benchmarks.md](benchmarks.md)。

## 对局矩阵、Nash 与 alpha-rank（`ygorl/eval/matchup.py`，T3.3）

```python
from ygorl.agents import GreedyAgent
from ygorl.eval.matchup import analyze, build_matrix, MetaGame

matrix = build_matrix(decks, GreedyAgent, pairs=50, seed=0, env=env, workers=2)  # decks: [Deck] 或 {名字: Deck}
meta = analyze(matrix, alpha=10.0, population_size=50)
meta.nash_by_deck(), meta.alpha_rank_by_deck()
meta.save(env=env, name="greedy-2026-10")        # -> environments/<v>/artifacts/matrix/greedy-2026-10.json
meta.save("out/matrix.json")                     # 无环境时必须给路径
MetaGame.load(path, env=env)                     # 环境不符时抛 EnvironmentConfigError
```

`ygorl.eval.matchup` 依赖 nashpy（及 scipy），没有从 `ygorl.eval` 顶层导出，只用 Arena 时不必加载它们。

**矩阵**：同一个 agent factory 驾驶双方，对每一对不同牌组用 Arena 打 `2 × pairs` 局配对对局。
`win_rate[i][j]` 是牌组 i 对牌组 j 的胜率（平局算半胜），`win_rate[j][i] = 1 - win_rate[i][j]`，
对角线（镜像）不打、记 0.5、局数 0；另有每格的局数与 Wilson 区间（`ci_low` / `ci_high`）。
每格按牌组名字排序后以 `derive_seed(seed, hash(名字1), hash(名字2))` 为种子，
所以**矩阵与牌组列表顺序无关，增删牌组不影响其他格子**；牌组名字必须唯一。固定种子下矩阵与 Nash 混合逐位可复现，且与进程数无关
（`tests/test_matchup.py` 用 3 套牌、每格 2 局验证）。

**元游戏**：对称零和博弈，收益矩阵 `win_rate - 0.5`。

- `nash_mixture(M)`：nashpy 的线性规划解（`Game(A).linear_program()`），得到一个没有任何单一牌组能平均胜过的牌组混合。
  均衡不唯一时返回线性规划给出的那个（确定性）。
- `alpha_rank(M, alpha, population_size)`：单种群 alpha-rank（Omidshafiei 等，2019）。种群有 `m = population_size` 个玩家，
  都用牌组 s；一个变异者改用 r。有 k 个变异者时
  `f_r(k) = ((k-1) M[r,r] + (m-k) M[r,s]) / (m-1)`，`f_s(k) = (k M[s,r] + (m-k-1) M[s,s]) / (m-1)`，
  Fermi 过程的固定概率 `ρ(r,s) = 1 / (1 + Σ_{l=1}^{m-1} exp(-α Σ_{k=1}^{l} (f_r(k) - f_s(k))))`（对数域计算，大 α 不溢出）。
  单态之间的马尔可夫链 `s → r` 的转移概率为 `ρ(r,s) / (n-1)`，其平稳分布即各牌组的 alpha-rank 质量。
  `alpha` 越大选择越强；默认 `alpha = 10`、`population_size = 50`（胜率在 0–1 之间，差值量级 0.1 时已是强选择）。
- 单测：石头剪刀布（Nash 与 alpha-rank 都均匀）、加权石头剪刀布（Nash = 1/4, 1/2, 1/4）、占优策略（Nash 纯策略、alpha-rank 质量 > 0.99）、
  固定概率的中性与常数适应度闭式解、随机矩阵上 Nash 不可被剥削。

**产物格式**（JSON，`format = "ygorl-matchup"`，`format_version = 1`）：
`environment`（`Environment.stamp()` 或 `null`）、`decks`、`deck_hashes`（主/额外/副卡组的 sha256 前 16 位）、
`agent`、`seed`、`pairs`、`confidence`、`errors`、`win_rate`、`games`、`ci_low`、`ci_high`、`nash`、`alpha_rank`、`alpha`、`population_size`。
传入 `env` 保存时，矩阵必须是在同一环境（版本与指纹）下构建的，否则抛 `EnvironmentConfigError`。
