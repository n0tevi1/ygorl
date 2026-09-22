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

命令行：`ygorl duel ... --save-replay game.json.gz --yrpx game.yrpX` 录制并导出，`ygorl replay game.json.gz --verify --export-yrpx out.yrpX` 显示元数据、重新模拟核对终局并导出（见 [cli.md](cli.md)）。

## 文件格式（`format_version = 1`）

JSON 对象（`.json.gz` 为 gzip 压缩的同一内容）：

| 键 | 含义 |
|----|------|
| `format`, `format_version` | 固定 `"ygorl-replay"`、`1`；版本不符时加载报错 |
| `environment` | `{"version", "fingerprint"}`，无环境对局为 `null` |
| `engine` | `{"ocgcore": [major, minor]}` |
| `seed`, `first` | 对局种子；先攻方（0 = a，1 = b） |
| `seed_words` | 可选：4 个核心种子字，只在来自其他主机的回放（`Replay.from_yrp`，见文末）里出现；缺省时由 `seed` 经 `expand_seed` 得到 |
| `rule_flags`, `player`, `shuffle_decks`, `max_turns`, `max_decisions` | 规则 flag、LP/起手/抽卡数、是否主机洗牌、上限 |
| `curriculum`, `learner`, `augmented_start` | 课程模式、学习方（0 = a，1 = b）、增广开局标志（见 [curriculum.md](curriculum.md)）；旧文件缺省为 `"full"`、`0`、`false` |
| `decks` | `{"a": {name, main, extra, side}, "b": {...}}`，洗牌前的原始卡组 |
| `responses` | 每次 `set_response` 的负载（十六进制字符串），按时间顺序 |
| `result` | 录制时的胜者、原因、回合、LP、决策数（参考信息） |
| `steps` | 可选：每个动作步的候选动作列表、所选下标、玩家、决策类型，以及 agent 暴露的 `last_probs`（策略概率） |

应答日志是回放的唯一真相：核心种子由 `seed` 经 `expand_seed` 得到，洗牌由 `shuffle_deck` 从同一 `seed` 得到（见 [engine.md](engine.md)）。从回放的任意决策点分叉、尝试其它动作见 [branching.md](branching.md)。

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

## 读取 `.yrp` / `.yrpX`（T4a.1）

```python
from ygorl.engine.replay import Replay, load_yrp

yrp = load_yrp("solution_00_b2_a5.yrp")   # YrpFile：id、flag、seed（4 个核心种子字）、names、player、decks、responses、packets
inner = yrp.replayable()                   # yrp1 本身，或 yrpX 内嵌的 yrp1
rep = Replay.from_yrp(yrp)                 # 可重放的 ygorl Replay
result = rep.play()                        # 按文件里的种子、规则、加载顺序的卡组与应答重新模拟
```

- 解析照 `gframe/replay.cpp`：普通头 32 字节，带 `EXTENDED_HEADER` 时 72 字节（含 4 个 u64 种子）；`COMPRESSED` 标志时正文是 LZMA（头里存 5 字节属性、`datasize` 为解压后长度），
  用标准库 `lzma` 以 `.lzma` 格式重组后解压；`yrpX` 的数据包流中 `OLD_REPLAY_MODE` 包即内嵌的 `yrp1`；应答以长度 0 结尾或读到数据末尾为止（求解器写结尾 0，我们的导出不写）。
- 截断、标识不对、解压失败、声明长度超过实际数据都抛 `YrpError`（`ValueError` 子类），不会返回半截结果。
- `Replay.from_yrp` 只接受 1 对 1、带扩展头的回放：`seed_words` 取文件的种子字（不再由 `seed` 经 `expand_seed` 得到），`shuffle_decks=False`、卡组按文件的加载顺序，
  `first=0`（文件的第一套卡组是引擎玩家 0，记为 a）；规则 flag 与 LP / 起手 / 抽卡数取自文件，不带环境。
- 相应地，`Duel(..., core_seed=[w0, w1, w2, w3])` 可以直接指定核心种子字；`Replay` 的 `seed_words` 只在不等于 `expand_seed(seed)` 时写进 JSON，普通对局的回放文件与之前逐字节相同。
- `Replay.to_yrpx(path, names, env, cards=..., scripts=...)` 的额外参数传给 `Duel`。
