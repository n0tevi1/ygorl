# 回放（`ygorl.engine.replay`，T1.8）

回放是一等公民：任意一局都能记录、存盘、重新加载并逐字节重放到同一终局，也能导出 EDOPro 可打开的 `.yrpX`。

## 录制与重放

```python
from ygorl.engine.duel import Duel
from ygorl.engine.replay import Replay

duel = Duel(seed, env, deck_a, deck_b, record_steps=True)   # record_steps 可选
result = duel.run(agent_a, agent_b)
rep = Replay.from_duel(duel, result)
rep.save("game.json.gz")                                   # .json 或 .json.gz

rep = Replay.load("game.json.gz")
again = rep.play(env=env)                                   # 按应答日志重放
rep.to_yrpx("game.yrpX", names=("A", "B"), env=env)         # 导出 EDOPro 回放
```

## 文件格式（`format_version = 1`）

JSON 对象（`.json.gz` 为 gzip 压缩的同一内容）：

| 键 | 含义 |
|----|------|
| `format`, `format_version` | 固定 `"ygorl-replay"`、`1`；版本不符时加载报错 |
| `environment` | `{"version", "fingerprint"}`，无环境对局为 `null` |
| `engine` | `{"ocgcore": [major, minor]}` |
| `seed`, `first` | 对局种子；先攻方（0 = a，1 = b） |
| `rule_flags`, `player`, `shuffle_decks`, `max_turns`, `max_decisions` | 规则 flag、LP/起手/抽卡数、是否主机洗牌、上限 |
| `curriculum`, `learner`, `augmented_start` | 课程模式、学习方（0 = a，1 = b）、增广开局标志（见 [curriculum.md](curriculum.md)）；旧文件缺省为 `"full"`、`0`、`false` |
| `decks` | `{"a": {name, main, extra, side}, "b": {...}}`，洗牌前的原始卡组 |
| `responses` | 每次 `set_response` 的负载（十六进制字符串），按时间顺序 |
| `result` | 录制时的胜者、原因、回合、LP、决策数（参考信息） |
| `steps` | 可选：每个动作步的候选动作列表、所选下标、玩家、决策类型，以及 agent 暴露的 `last_probs`（策略概率） |

应答日志是回放的唯一真相：核心种子由 `seed` 经 `expand_seed` 得到，洗牌由 `shuffle_deck` 从同一 `seed` 得到（见 [engine.md](engine.md)）。

## 环境绑定

录制时环境的版本号与指纹写进回放。`play()` / `to_yrpx()`：

- 回放有环境而未传 `env` → `ReplayEnvironmentMismatch: replay requires environment <v>`；
- 传入的环境版本不同 → `replay was recorded in environment <v>, not <w>`；
- 版本相同但文件被改过（指纹不同）→ `environment <v> has changed since the replay was recorded`。

## `.yrpX` 导出

布局抄自 edo9300/edopro 的 `gframe/replay.cpp` 与 `gframe/generic_duel.cpp`：

- 扩展头（72 字节）：`id='yrpX'`、客户端版本（EDOPro 41.0 + 核心 11.0）、flag = `LUA64 | NEWREPLAY | 64BIT_DUELFLAG | EXTENDED_HEADER`（不压缩）、`datasize`、4 个 u64 核心种子。
- 正文：双方名字（按座位：先攻方在前）、u64 规则 flag、数据包流 `[u8 消息][u32 长度][负载]`：`MSG_START` + 核心产生的全部消息（与 EDOPro 主机一致，去掉决策消息、`MSG_RETRY` 和仅发给单方的提示）。
- 最后一个包（消息号 231，`OLD_REPLAY_MODE`）内嵌完整的 `yrp1`：同一种子、LP/起手/抽卡、规则 flag、按加载顺序（洗牌后）的双方卡组、应答日志。EDOPro 勾选「yrp 模式」时用自己的核心按它重新模拟。

测试：`test_yrpx_export_structure` 用与 EDOPro 相同的解析逻辑读回文件并核对各字段；`test_embedded_yrp1_resimulates_like_edopro_old_mode` 只凭文件内容按 EDOPro 旧模式的步骤重跑核心，得到与原局逐字节相同的消息流。

**尚待人工确认**：验收要求在 EDOPro 客户端里实际打开一次导出的 `.yrpX`。需要 EDOPro 使用与本仓库相同版本的核心与脚本，否则 yrp 模式的重新模拟会分叉；流式模式不依赖脚本版本，但缺少主机发送的 `MSG_UPDATE_DATA` 刷新包，卡组/额外卡组的卡面信息可能显示不全。
