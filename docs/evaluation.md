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
- 可选方法 `observe(point, core)`：`Duel.run` 在对局的**每个**决策点（双方的都算）按顺序、在行动方 `act` 之前调用，`core` 是活的核心句柄（只做查询）。编码观测的 agent 需要它：事件流跨越对手的决策，卡片表要查询核心（`NetPolicy`，[bc.md](bc.md)）。同一实例坐两个座位时每个点只调用一次。
- 并行 `Arena` 需要把 factory 发到子进程，所以 factory 必须可 pickle：类本身（`GreedyAgent`）、模块级函数或 `functools.partial`。
- `agent_name(x)` 给类、实例、函数或 partial 取可读名字（优先类属性 `name`）。
- 可选钩子（T4b.4）：`Duel.run` 开局时对每个 agent 调一次 `on_duel_start(duel)`（拿到种子、规则、装载顺序的牌组），
  每个决策答完后对**两个座位**的 agent 都调 `on_decision(point, index)`（同一实例坐两个座位时只调一次）。
  需要跟踪整局的 agent（如下面的策略检查点 agent）用它们；只走 `Duel.run` 路径，`VecDuelEnv` / 回放 / 分叉不调用。

## 基线 agent

| agent | 行为 |
|-------|------|
| `RandomAgent(seed)` | 均匀随机合法动作 |
| `GreedyAgent(seed, cards=None)` | 确定性启发式，见下 |
| `PolicyAgent(policy, seed=None, greedy=False, temperature=1.0)` | `policy(point)` 对每个动作打分（logit），按 `softmax(score / T)` 采样或取 argmax；暴露 `last_probs`。M4 的网络策略从这里接入：`PolicyAgent(NetPolicy.from_checkpoint(path))`，或规格 `policy:PATH`（[bc.md](bc.md)） |
| `CheckpointAgent(path, seed, greedy=False, temperature=1.0)` | 训练出的 PPO checkpoint（T4b.4），见下「策略检查点 agent」；登记名 `policy:PATH[@greedy][@t=T]`（与 BC 检查点共用一个名字，按文件的 `format` 字段选加载方式）；`policy-greedy:PATH` 是 `@greedy` 的简写 |
| `LethalAgent(agent, seed, rollouts=12, ...)` | 任意 agent 外加本回合的致死搜索（T4e.1 原型，[#62](https://github.com/n0tevi1/ygorl/issues/62)）：跟随整局的影子对局（带引擎快照），在本方主要 / 战斗阶段做若干次本回合推演（己方 Greedy / 随机、对手只放弃），找到赢下来的线就照走，真实对局一偏离就交回原 agent；严格模式不用己方抽卡、投币 / 骰子、对手连锁的线（不利用隐藏信息）。登记名 `lethal:<agent 规格>`；`YGORL_LETHAL_LOG` 记录每局的搜索统计 |

### 策略检查点 agent（`ygorl/agents/checkpoint.py`）

策略网络吃的是 C++ 编码器的观测（[encoding.md](encoding.md)：卡片表、全局向量、候选表、事件 token 流），要引擎状态和整局的事件流；
Agent 协议只给一个 `DecisionPoint`。适配方式是**锁步 C++ 主机**：

1. `on_duel_start(duel)`：用这局的核心种子（`duel.core_seed`）、规则、起始 LP / 手牌与装载顺序的牌组（`duel.loaded_decks()`）
   启动一个 `_core.HostDuel`（事件窗口长度取 checkpoint 的配置、词表取 checkpoint 里保存的 `CardVocab`）——同一局棋。
2. `on_decision(point, index)`：两个座位的每个决策都在自己的主机里重放一遍，主机的事件历史与对局一致。
3. `act(point)`：先核对主机问的座位与候选数和 Python 追踪器一致（不一致即抛错，说明两套主机分叉），`host.observe()` 就是
   `EncodedVecEnv` 在同一决策上给出的观测；网络采样（或 argmax）一个候选行号。`last_probs` 为各动作概率（第 128 个之后为 0）。

于是 `ygorl duel` / `ygorl arena` / `ygorl matrix` 直接可用：`--agent-a policy:out/train/run1/best.pt`。单测
`tests/test_train_loop.py::test_policy_agent_plays_legal_moves_in_lockstep` 把一局按同样的动作序列在 `EncodedVecEnv` 里重放，
逐个决策比对观测完全相同。限制：只能选前 128 个候选（编码截断）；不支持课程模式与增广开局（C++ 步进路径尚无课程过滤）；
checkpoint 每个进程按路径只读一次；并行 Arena 的子进程里 PyTorch 限 1 线程。需要 `train` 可选依赖（未安装时构造报错）。

另一条路是批量评估器：直接在 `EncodedVecEnv` 上让检查点对另一个检查点 / 随机对局（训练里的快照对手就是这样打的）。
Greedy 依赖 `DecisionPoint`，不能走那条路，所以对基线的评估统一用上面的 Arena 路径。

### GreedyAgent 规则

只看当前决策与少量回合内记忆，**不追踪场面**；攻击力 / 守备力取卡片数据库的原始值（`CardDB`，默认 `default_cards()`），不含场上增减。

| 决策 | 选择（按优先级） |
|------|------------------|
| 主要阶段 IDLECMD | 发动 > 特殊召唤 > 通常召唤（攻击力最高）> 盖放怪兽 > 盖放魔陷 > 进入战斗阶段 > 结束阶段 |
| 战斗阶段 BATTLECMD | 直接攻击 > 攻击力最高的怪兽攻击 > 发动 > 主要阶段 2 > 结束阶段 |
| 攻击目标（`MSG_HINT` 549 之后的 SELECT_CARD） | 攻击力严格高于其攻击力（攻击表示）/ 守备力（守备表示）的目标中最强的一个；打不过任何目标时取消攻击，本回合不再用这只怪兽攻击；不能取消时选最弱的目标。里侧目标的身份对决策方隐藏（卡密为 0），按 `HIDDEN_STAT = 1500` 估计，不偷看真实数值 |
| CHAIN | 能连锁就连锁 |
| EFFECTYN / YESNO | 是 |
| POSITION | 攻击力 ≥ 守备力 → 表侧攻击，否则表侧守备 |
| SELECT_CARD / UNSELECT_CARD / SUM | 能选就继续选（种子随机），没得选才 finish；从不 unselect |
| TRIBUTE | 攻击力最低的先解放 |
| 其他（OPTION、PLACE、ANNOUNCE、SORT 等） | 种子随机 |

防卡死：同一发动（卡、位置、效果）每回合最多 `max_repeats = 3` 次，每回合主动的主要阶段动作最多 `max_turn_actions = 100` 个，之后只进战斗 / 结束阶段。

「有攻击者才进战斗阶段」的近似：agent 不追踪场面，主要阶段 1 能进战斗阶段就进，到了 BATTLECMD 没有可攻击的怪兽再去主要阶段 2 / 结束阶段。

验收（T3.1）：`tests/test_agents.py` 让 Greedy 在 10 套测试牌组上各打一局（对 Random），零 retry、零未知消息，且都以胜负或回合上限结束；开发时的 200 局抽样全部以 `MSG_WIN` 结束，Greedy 胜率约 90%。

`AGENTS = {"random": RandomAgent, "greedy": GreedyAgent}` 按名字列出基线 agent。命令行用 `ygorl.agents.registry` 的规格 `name[:arg]`：`make_agent(spec, seed)` 构造 agent，`agent_factory(spec)` 返回可 pickle、以规格为名字的 factory（`AgentSpec`），可直接传给 `Arena` / `build_matrix`。

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
进程启动方式默认随平台（Linux 为 fork）；父进程已加载 PyTorch 时（例如校验过 `policy:` 规格）改用 `spawn`——fork 一个初始化过
PyTorch 线程池的进程会让子进程死锁；`Arena(mp_context=...)` 可显式指定。
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

命令行：`ygorl arena`（见 [cli.md](cli.md)）。

**基准**：`uv run python tools/arena.py --games 2000 --workers 2`（`ygorl arena tests/decks` 的包装）让两个 agent 在 10 套测试牌组的全部有序组合
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

命令行：`ygorl matrix`（见 [cli.md](cli.md)）。

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

## 策略对局矩阵：谁打得更好（`ygorl/eval/agent_matrix.py`，T4b.8，#85）

牌组对局矩阵问「哪套牌强」（同一 agent 驾驶所有牌组）。策略对局矩阵问「哪个 agent 强」：agent 对 agent，在同一批牌组配对上比。
它是对局强度的尺子（[eng-plan.md](eng-plan.md) T4b.8）：Greedy 已接近被打饱和，所以之后的强度比较都在这张矩阵里做。
按设计 C7，排名用 Nash / alpha-rank，而不是单一 Elo，因为克制关系不一定可传递。

```python
from ygorl.agents import AgentSpec
from ygorl.eval.agent_matrix import AgentMatrix, build_agent_matrix

m = build_agent_matrix({"greedy": AgentSpec("greedy"), "best": AgentSpec("policy:out/run/best.pt")}, decks,
                       pairings=50, seed=0, env=env, workers=8)
m.ranking(), m.against("greedy"), m.nash, m.alpha_rank
m.save(env=env, name="baseline")      # -> environments/<v>/artifacts/agent-matrix/baseline.json
AgentMatrix.load(path, env=env)
```

命令行：`ygorl strength`（见 [cli.md](cli.md)）。

**对局**：
- 按 `seed` 从牌组池抽 `pairings` 个有序配对 `(d1, d2)`。
- 每对 agent `(x, y)`（按名字排序，x 坐 a 位）在每个配对上打两种牌组分配：x 用 d1 对 y 用 d2，再 x 用 d2 对 y 用 d1。
- 每种分配先后攻各一局，共 4 局。所以配对内的牌组强弱和先后攻优势互相抵消。

**公共随机数**：
- 每个配对的两套牌是两个「槽位」。洗牌、以及拿着某个槽位的 agent 的随机种子，都按槽位派生（`derive_seed(seed, k)` 与槽位号），与座位无关。
  座位（a / b）只按名字顺序决定，不影响局面。
- 所以同一个 agent 在每一格里拿同一个槽位时，看到的起手完全相同。
- 结果：矩阵与 agent 的顺序、名字、进程数都无关；加一个 agent 不改变已有的格子（小矩阵是大矩阵的子矩阵）；不同行在同一批局面上比较。
  测试检查：两个名字排在对手前后的 Greedy 副本，对同一个对手的结果逐局相同。
- 所有格子的对局交给同一个进程池。

**统计口径**：
- `win_rate[i][j]` 是 agent i 对 agent j 的胜率，平局算半胜；`win_rate[j][i] = 1 - win_rate[i][j]`，对角线 0.5。
- 抛异常（`exception`）或主机以错误结束（`error`：引擎步数上限、脚本预算）的局不算胜负平：计入 `errors`，不进 `games` 与胜率。
  这与 `Arena` 的报告不同，后者把它们记作平局。
- 每格给 Wilson 区间（`ci_low` / `ci_high`）。

**排名**：
- `ranking()` 按 alpha-rank 质量、Nash 权重、平均胜率排序；完全打平时次序无意义。
- Nash 混合与 alpha-rank 用 `ygorl.eval.matchup` 的同一套求解器（收益矩阵 `win_rate - 0.5`）。

**产物格式**（JSON，`format = "ygorl-agent-matrix"`，`format_version = 1`）：
- `agents`、`specs`（每个 agent 的构造方式，即 agent 规格）、`fingerprints`（检查点内容哈希）；
- `win_rate`、`games`、`errors`、`ci_low`、`ci_high`；
- `decks`、`deck_hashes`、`pairings`（牌组下标对）、`seed`、`max_turns`、`max_decisions`；
- `nash`、`alpha_rank`、`alpha`、`population_size`、`confidence`、`environment`。
- 带 `env` 保存时必须是同一环境下构建的。

**增量扩展**（#86）：`extend_agent_matrix(matrix, 新 agent, decks, env=..., config=...)`。
- 只打含新 agent 的格子，用矩阵自己的配对、种子和规则；旧格子原样复制。结果与一次建出全部 agent 的矩阵逐格相同（测试检查，并数了实际开的局数）。
- 旧 agent 按记录的规格用注册表重建；重建不了（例如当初传的是自定义函数）就报错，请把它和新 agent 一起传入。
- `fingerprints` 记录每个策略 agent 的检查点内容哈希（`sha256:` 前 16 位，穿过 `lethal:` 包装）；规则 agent 为空串。
- 以下情况报错，并给出原因：
  - 同名 agent 的规格或检查点内容不同（同名同内容则跳过）；
  - 旧 agent 的检查点文件已被改写；
  - 牌组池的名字、顺序或内容不同；
  - `max_turns` / `max_decisions` 不同；
  - 环境不同。
- 不含 `fingerprints` 的旧文件照常加载（视为全空）。

**尚未做**：
- 策略对策略的格子走 GPU 批量路径（#87）。
- 目前全部走 `Arena`，策略 agent 在 CPU 工作进程里推理，所有格子共用一个进程池。

## 批量评估（策略对策略，`ygorl.eval.batched`）

`Arena` 每局是一个 Python `Duel`，策略 agent 在 CPU 上一次只答一个决策。调卡组要在很多牌组上打很多局网络策略之间的对局，
所以 `play_policies(env, specs, policy_a, policy_b)` 把对局放在 C++ 步进路径（`EncodedVecEnv`，`skip_forced=True`）上跑，
每轮把同一策略所有就绪的决策合成一次前向（可在 GPU 上）。

- **配对**同 Arena：`paired_specs(deck_a, deck_b, pairs, seed, config)` 给每个种子先后攻各一局、两局同一牌序；记录是 deck a 一侧的
  `GameRecord`，`summarize` 与配对比较照用。
- **采样**：每个决策用由（对局种子、先攻方、座位、该座位第几个决策）导出的均匀数从掩码后的 softmax 里抽，结果与批怎么拼、线程快慢无关
  （测试：1 个环境与 4 个环境逐局相同）。与 Arena 同分布，但不是同一串随机数。
- **失败**：开不了局（`reset` 抛异常）记为 `exception` 并接着打下一局；引擎在局中出错（`reason="error"`）也记为 `exception`、不算平局，
  都进报告的 `errors`。
- 只能打网络策略（greedy / random 在 Python 里，仍用 Arena）。

命令行：`tools/eval_batched.py CKPT --decks DIR [--opponents DIR] [--opponent-checkpoint CKPT2] [--pairings 200] [--pairs 1]
[--device cuda] [--envs 256] [--out report.json]`：CKPT 驾驶 `--decks` 里的牌，对手策略驾驶 `--opponents` 里的牌，按 `--seed`
抽一次对阵；打印总胜率（Wilson 区间）与先后攻分项，`--out` 另写按驾驶牌组拆开的胜率与耗时统计。吞吐见 [benchmarks.md](benchmarks.md)。
