# 基线 agent 与评估（`ygorl.agents`、`ygorl.eval`）

本文说明 M3 的对手与裁判：Agent 协议与基线 agent（T3.1）。对应 [工程计划](eng-plan.md) M3。

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
