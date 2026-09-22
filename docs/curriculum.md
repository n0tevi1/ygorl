# 课程模式与开局配平（T2.6）

对应设计文档 [02-challenges.md](design/02-challenges.md) C3(c)「单人展开 → 对手只有手坑 → 完整对局，外加中局起始状态」与 [03-play-policy.md](design/03-play-policy.md) I7「增广开局标志位、先后攻配平」。本票只提供开关与载体；三阶段调度与中局开局的数据来源属于 T4d.1（以及 T2.8 / T2.9）。

## 配置

全部是 `DuelConfig` 的字段（`GameSpec.config` 同一个对象），默认值与改动前完全一致：

| 字段 | 默认 | 含义 |
|------|------|------|
| `curriculum` | `"full"` | `"solo"`（单人展开）/ `"handtrap"`（仅手坑）/ `"full"`（完整对局） |
| `learner` | `0` | 学习方：0 = `deck_a`，1 = `deck_b`。按**牌组**而不是按座位指定，先后攻配平时学习方轮流先攻、后攻 |
| `augmented_start` | `False` | 增广开局标志，见下文 |

非法取值在构造 `DuelConfig` 时抛 `ValueError`。三个字段都写入回放（[replays.md](replays.md)）。

```python
from ygorl.engine.duel import DuelConfig
from ygorl.env import GameSpec, paired_specs, run_games

cfg = DuelConfig(curriculum="handtrap", learner=0)
specs = paired_specs(GameSpec(seed=s, deck_a=mine, deck_b=theirs, config=cfg) for s in range(50))  # 100 局
results = run_games(specs, agent_factory)
```

## 限制的范围

课程模式只限制**对手**（非学习方），并且只在**学习方是回合玩家**时生效。对手自己的回合永远完整进行，由对手的 agent 正常决策；学习方在任何回合都不受限制。

理由：C3 要解决的是「先攻一回合几十个决策、一步选错整条线崩掉」的探索问题，需要的是学习方展开时不被打断；展开的价值最终要由对手回合的突破来检验，所以对手回合保持完整。两种模式因此是同一条规则的两档：`solo` 允许的对手发动集合为空，`handtrap` 只允许从手牌发动。

需要「只看展开、不打对手回合」的纯单人局时，用 `solo` + 学习方先攻 + `max_turns=1`：进入第 2 回合的第一个决策时以 `turn_limit` 结束（LP 相同即平局），终局评价由训练侧给出（如价值头或求解器对终场的打分），引擎本身不做 shaping。

## 规则

「放弃类」动作：`pass`（连锁不响应）、`no`（EFFECTYN / YESNO 选否）、`cancel`（可取消的选卡）。对手的一个决策（学习方回合内）按下表处理，纯函数在 `ygorl.engine.curriculum`（`allowed_actions` / `auto_action`），由 `DuelTracker` 应用：

| 决策 | `solo` | `handtrap` | `full` |
|------|--------|------------|--------|
| `SELECT_CHAIN`（非强制） | 主机代答 `pass` | 只保留从手牌发动的选项 + `pass`；没有手牌选项时主机代答 `pass` | 不限制 |
| `SELECT_EFFECTYN` | 主机代答 `no` | 卡片在手牌：交给 agent；否则主机代答 `no` | 不限制 |
| `SELECT_YESNO`、可取消的 `SELECT_CARD` / `SELECT_UNSELECT_CARD` | 主机代答 `no` / `cancel` | 不限制（属于正在处理的效果，如手坑自身的结算） | 不限制 |
| 没有放弃选项的决策：强制连锁（必发诱发）、必须选的卡、区域、表示形式、宣言等 | 不限制 | 不限制 | 不限制 |

- 过滤后只剩唯一的放弃动作时由主机代答，agent 不会被问到；否则 agent 看到的 `DecisionPoint.actions` 就是过滤后的列表，返回的下标也指这个列表（`DuelResult.actions` 与 `record_steps` 记录的都是过滤后的下标和候选，所以同一配置下 `ScriptedAgent` 可以重演）。
- 强制连锁从不过滤：核心要求必须选一个，对手的必发诱发在 `solo` 下也照常发动。
- 只有单个必发诱发时核心不询问而直接发动（`processor.cpp` 强制诱发分支），这不经过主机，也不受限制。可选诱发只有一个且连锁为空时核心用 `SELECT_EFFECTYN` 询问，多个时用非强制的 `SELECT_CHAIN`，两者都按上表处理。

### 手坑的判定

判据是**发动位置**：`SELECT_CHAIN` 选项 / `SELECT_EFFECTYN` 卡片的位置为 `LOCATION_HAND`。灰流丽、增殖的 G、效果遮蒙者、原始生命态 尼比鲁、幽鬼兔、屋敷童等都在其中；从手牌发动的陷阱（无限泡影）选项位置同样是手牌——核心在发动时才把它移到魔陷区，所以 `MSG_CHAINING` 里的位置是魔陷区，而选项里是手牌。盖放的陷阱（神之宣告等）、场上怪兽的诱发即时效果、墓地效果都被过滤。

不用卡表的原因：与环境版本无关、不需要维护名单，而且与规则一致——只要能从手牌发动，就是对手在学习方回合里能用的「手坑」。代价是少数从手牌发动但不算传统手坑的效果（如战斗中从手牌丢弃的效果）也被放行，这对「对手只有手牌互动」的课程目的没有影响。需要按名单细分时，可以在 `allowed_actions` 里加一层按 `password` 的过滤，并把名单放进环境的 meta 卡表。

## 记录与重放

- 主机代答同样经过 `set_response`，应答字节追加到 `DuelResult.responses`；应答日志仍是唯一真相，`Duel.replay` / `Replay.play` 不需要知道课程模式就能逐字节重现（`test_replay_roundtrip_keeps_the_mode` 比对消息流）。
- 代答不算 agent 决策：不进 `DuelResult.actions`、`steps`，不计入 `decisions` 与 `max_decisions`，单独计数在 `DuelResult.auto_decisions`。
- 代答期间产生的消息留在事件缓冲里，随下一个 `DecisionPoint.events` 交给下一个被询问的 agent，不丢失。
- `VecDuelEnv` / `DuelEnv` / `run_games` 与 `Duel.run` 共用 `DuelTracker.auto_response()`，结果逐局一致（`test_pool_matches_sequential_duels` 对三种模式各测一次）。
- 回放文件新增 `curriculum`、`learner`、`augmented_start` 三个键；旧文件缺这些键时按 `"full"`、`0`、`false` 加载，`format_version` 不变。

## 先后攻配平

`paired_specs(specs)` 把每个 `GameSpec` 展开成两局：`first=0`（`deck_a` 先攻）和 `first=1`（`deck_b` 先攻），其余字段（种子、牌组、配置）不变。学习方按牌组指定，因此同一种子下学习方先攻、后攻各一局。两局的核心随机数种子相同；主机洗牌按引擎座位播种（`shuffle_deck(main, seed, player)`，为保持已有结果不变而没有改），所以交换先后攻后各方的牌序不同，配平消除的是先后攻偏差而不是起手偏差。

## 增广开局标志（I7）

`DuelConfig.augmented_start` → `DecisionPoint.augmented_start` → 观测全局向量第 21 列 `augmented_start`（[encoding.md](encoding.md)），参考编码器 `ObservationEncoder` 写入 0/1。用途是让策略知道当前局面来自增广分布（从求解器线或对局记录采样的中局状态），避免训练分布与真实对局分布混在一起导致均衡偏移（DAGS 2026）。

本票只提供标志位的载体：中局状态本身如何构造（T2.8 快照、T2.9 分支 API，或从种子重放到 t）由 T4d.1 负责，届时生成中局开局的代码设置 `augmented_start=True`。

## 验收

`uv run python tools/check_curriculum.py --games 100 --threads 2 --envs 8 --compare`：每种模式 100 局随机对随机（50 个种子 × 先后攻配平，10 套测试牌组轮换，学习方为 `deck_a`，`max_turns=200`），在 DuelPool 上运行；`--compare` 再用 `Duel.run` 逐局重跑、用 `Duel.replay` 从应答日志重放，比对结果。

2026-09-22 的一次运行（4 核共享机器，2 线程 × 8 槽位；三种模式连同 `--compare` 共 6 分 21 秒）：

| 指标 | `full` | `solo` | `handtrap` |
|------|--------|--------|------------|
| 对局数 | 100 | 100 | 100 |
| 错误 / retry / 未知或无法解码的消息 / 有 Lua 报错的对局 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| 结束原因 | 100 局 `win` | 100 局 `win` | 100 局 `win` |
| 平均回合数 | 51.9 | 52.1 | 53.5 |
| agent 决策数 | 120,221 | 99,739 | 99,513 |
| 主机代答数（`auto_decisions`） | 0 | 19,967 | 20,356 |
| 学习方回合内交给对手 agent 的决策 | 22,238 | 7 | 478 |
| 其中对手连锁：手牌 / 其他（非强制）/ 强制 | 170 / 564 / 3 | 0 / 0 / 3 | 192 / 0 / 1 |
| 对手非手牌的 EFFECTYN 选「是」 | 89 | 0 | 0 |
| 学习方胜率：先攻 / 后攻 | 0.54 / 0.52 | 0.50 / 0.48 | 0.52 / 0.52 |
| 违反限制 | 0 | 0 | 0 |
| 与 `Duel.run` 逐局比对、应答日志重放不一致 | 0 | 0 | 0 |
| 池化耗时（秒） | 54.4 | 54.5 | 56.2 |

随机对随机时学习方回合里对手 agent 被问到的决策几乎全是空的连锁窗口（只有 `pass`），`solo` / `handtrap` 下这些都由主机代答，因此 agent 决策数少了约 17%。胜率只是随机策略下的数字，用来确认各模式都能打完整局，不代表课程效果。

测试：`tests/test_curriculum.py`——手工构造决策上的过滤单测（三种模式 × 连锁 / EFFECTYN / YESNO / 可取消选卡 / 强制连锁 / 无放弃选项的决策）、配置校验、`paired_specs`；小规模真实对局验证 `solo` 下对手在学习方回合从不自愿连锁、`handtrap` 下对手的每次连锁都从手牌发动（且确实发生过）、`full` 与改动前逐字节相同、池化与逐局一致、回放往返、旧回放按 `full` 加载、标志位出现在观测第 21 列。
