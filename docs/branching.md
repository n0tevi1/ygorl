# 分支探索（`ygorl.engine.branch`，T2.9）

从一局录好的对局的任意决策点 `t` 分叉：在 `t` 处尝试多个候选动作，每个候选用给定策略 rollout 到终局，比较结果。用途：反事实分析与调试（「这里换一手会怎样」）、中局开局的数据来源（T4d.1）、后期 PIMC 搜索的基础。设计依据见 [06-architecture.md](design/06-architecture.md)「回放与分支探索」。

## 用法

```python
from ygorl.agents import RandomAgent
from ygorl.engine.branch import RecordedAgent, fork, recorded_actions
from ygorl.engine.replay import Replay

rep = Replay.load("game.json.gz")
branch = fork(rep, t=30, env=env)          # 有环境的回放必须传入同一环境（同 Replay.play）
branch.point                               # t 处的 DecisionPoint：point.actions 即候选动作
branch.side                                # t 处行动方：0 = 卡组 a，1 = 卡组 b
branch.recorded_action                     # 原局在 t 处的选择（回放没记录到时为 None）

res = branch.rollout(3, RandomAgent(1), RandomAgent(2))   # t 处走动作 3，之后由策略 (a, b) 下完
outcomes = branch.try_all(RandomAgent, rollouts=10, seed=0)  # 每个合法动作 10 次 rollout
for o in outcomes:
    print(o.index, o.action.kind, o.wins, o.draws, o.losses)   # 胜负按 t 处行动方统计

acts = recorded_actions(rep, env)          # 原局全部动作下标
same = branch.rollout(acts[30], RecordedAgent(acts), RecordedAgent(acts))  # 按原动作继续 → 与原局终局一致
```

- `rollout` 返回整局的 `DuelResult`（`responses` / `actions` 含重放的前缀），策略只在 `t` 之后被询问。
- `try_all(policy_factory, candidates=None, rollouts=1, seed=0)`：`policy_factory(seed)` 为每一方、每次 rollout 新建策略。第 `r` 次 rollout 对**所有候选**都用种子 `seed + 2r`（a）与 `seed + 2r + 1`（b），即公共随机数，候选之间的差异不被策略随机性淹没。
- `RecordedAgent(actions)` 按 `point.index` 取原局动作，两方共用。

## 命令行

```bash
uv run ygorl branch game.json.gz --at 30 --try all --policy random --seed 1
uv run ygorl branch game.json.gz --at 30 --try 0,5,10 --rollouts 20
```

| 选项 | 含义 |
|------|------|
| `--at T` | 分叉的决策点（见下节定义） |
| `--try all\|I,J,...` | 候选动作下标，默认全部合法动作 |
| `--policy AGENT` | 双方的 rollout 策略，按名字从 agent 注册表构造（现有 `random`） |
| `--seed S` | 第一次 rollout 的策略种子 |
| `--rollouts N` | 每个候选 rollout 次数；N > 1 时输出胜/平/负、胜率与均值 |
| `--env PATH\|VERSION` | 回放的环境；省略时按回放记录的版本在环境根目录（`$YGORL_ENVIRONMENTS` 或 `./environments`）下查找 |

输出示例（`*` 标出原局的选择）：

```
replay    game.json.gz: seed 1, a moves first, environment none
fork      t=30: turn 3, side a to act (engine player 0), MSG_SELECT_IDLECMD, 13 legal actions
recorded: action 6; the game ended winner=draw reason=turn_limit turns=7 lp=8000/8000
rollouts  policy random, seed 1, 10 rollouts per candidate; * = recorded action
          wins/draws/losses for side a; turns and lp are means

  cand  action                                        wins  draws  losses   win%   turns    lp_a    lp_b
     0  summon 45663742 Snake-Eye Oak @hand              1      8       1   10.0     7.0    7900    7970
     1  spsummon 41999284 Linkuriboh @extra              0     10       0    0.0     7.0    8000    8000
   ...
*    6  sset 89023486 Original Sinful Spoils - Sn...     0     10       0    0.0     7.0    8000    8000
```

`N = 1` 时每行给出 `winner reason turns lp_a lp_b`。卡片以 `password` 标识，卡名只用于显示。错误（`t` 越界、未知策略、候选越界、环境不符、文件不存在）打印为 `ygorl branch: error: ...`，退出码 2。

新增策略（Greedy、策略 checkpoint）：`ygorl.agents.register_agent(name, factory, description)`，`factory(arg, seed)` 返回 agent；`--policy name:arg` 把 `arg`（如 checkpoint 路径）传给工厂。新增子命令：在 `ygorl.commands` 下加一个模块（提供 `add_parser(subparsers)`），列入 `ygorl.cli.COMMANDS`。

## 决策点 `t` 的定义

`t` 是 **agent 步**的下标，从 0 开始，与 `DecisionPoint.index`、`DuelResult.actions` 的下标一致，**不是**引擎应答的下标。多选（SELECT_CARD / TRIBUTE / SUM / SORT / PLACE / COUNTER / ANNOUNCE_RACE|ATTRIB）按 [engine.md](engine.md) 的动作模型拆成逐张决策，一次 `set_response` 对应多步，`t` 可以落在多选的中间（例如已选一张、正在选第二张），此时候选只含剩余可选的卡与 finish。

合法范围：`0 ≤ t < N`，`N` 为回放能确定的动作步数。回放在一个无人应答的决策处截断（`log_exhausted`，或决策数上限卡在多选中间）时，那个决策点本身也可以分叉（`recorded_action` 为 `None`）；更后面的点不可达，报错信息给出可达范围 `0..last`。

## 实现：重放到 t

回放文件只存引擎应答字节，不存动作下标。`fork` 按回放重建对局（`Replay.duel`），用一个两方共用的 agent 逐步作答：每个决策开始时，把下一条记录的应答**反解**为动作下标序列（`actions_for_response`：在决策状态的副本上深度优先搜索，按应答里的选卡顺序、区域、计数等剪枝），然后逐步喂给引擎，到第 `t` 步停下并交出该决策点。回放带有逐步记录（`record_steps`）时直接用其中的 `chosen`，并核对它能生成记录的应答，不一致即报错；逐步记录还能补上被截断在多选中间的最后几步。反解与原局一致由测试保证：`recorded_actions(replay) == result.actions`。

`rollout` 再从头重放：`t` 之前按前缀动作、`t` 处走候选、之后问策略；并核对 `t` 处的决策与分叉时相同，否则报「replay diverged」。

测试（`tests/test_branch.py`）：从一局的每个 `t` 分叉并按原动作继续，终局（胜者、原因、回合、LP、全部应答）与原局一致；多选中间分叉；每种多选的反解都能还原任意路径的应答；`try_all` 每个合法动作一个结果且可复现；越界报错。`tests/test_cli.py` 覆盖命令行输出表。

## 限制

- **上帝视角分支**：各分支保留原局的全部隐藏状态（双方手牌、卡组顺序、核心随机数状态都来自同一种子），分叉只改变 `t` 处的选择。对手未知信息的重采样（确定化，determinization）需要按信念重建局面，留作后续；在此之前，用 rollout 结果评估一手棋会高估「知道对手手牌」带来的收益，不能直接当作 PIMC 的估值。
- **重放成本**：每次 `fork` 和每次 `rollout` 都从开局重放到 `t`，代价与 `t` 成正比。实测（本机，约 100 步的小局）：`fork` 约 20–30 ms，`rollout` 约 30 ms，其中建局与加载基础脚本占大头；`try_all` 的代价为 候选数 × rollout 次数 × 整局。T2.8 的 arena 快照落地后，`fork` 保存快照、`rollout` 从快照恢复，接口不变。
- rollout 用回放记录的上限（`max_turns` / `max_decisions`），与原局相同。
- 策略从 `t + 1` 才被询问，看不到 `t` 之前的事件（`DecisionPoint.events` 只含上一个决策点以来的消息）；依赖完整历史的策略（循环网络、信念头）需要以后加一个「旁观前缀」的钩子。
- 反解搜索有节点上限（`MAX_SEARCH_NODES`）；对 agent 产生的应答不会触及。无法反解的应答（非本动作模型产生、或曾被核心拒绝的应答）报 `BranchError`。
