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
| `recorded_at` | 可选：录制时刻（unix 秒），`from_duel` 写入，作为导出 `.yrpX` 的日期；旧文件没有此键，照常加载 |
| `steps` | 可选：每个动作步的候选动作列表、所选下标、玩家、决策类型，以及 agent 暴露的 `last_probs`（策略概率） |

应答日志是回放的唯一真相：核心种子由 `seed` 经 `expand_seed` 得到，洗牌由 `shuffle_deck` 从同一 `seed` 得到（见 [engine.md](engine.md)）。从回放的任意决策点分叉、尝试其它动作见 [branching.md](branching.md)。

## 环境绑定

录制时环境的版本号与指纹写进回放。`play()` / `to_yrpx()`：

- 回放有环境而未传 `env` → `ReplayEnvironmentMismatch: replay requires environment <v>`；
- 传入的环境版本不同 → `replay was recorded in environment <v>, not <w>`；
- 版本相同但文件被改过（指纹不同）→ `environment <v> has changed since the replay was recorded`。

## `.yrpX` 导出

布局抄自 edo9300/edopro 的 `gframe/replay.cpp`、`gframe/generic_duel.cpp` 与 `gframe/core_utils.cpp`：

- 扩展头（72 字节）：`id='yrpX'`、客户端版本（EDOPro 41.0 + 核心 11.0）、flag = `LUA64 | NEWREPLAY | 64BIT_DUELFLAG | EXTENDED_HEADER`（不压缩）、
  时间戳 = `recorded_at`（没有时写 0，EDOPro 列表里显示为 1970/01/01；不用当前时间，保证同一回放文件每次导出逐字节相同）、`datasize`、4 个 u64 核心种子。
  内嵌 `yrp1` 的头用同一时间戳。
- 正文：双方名字（按座位：先攻方在前）、u64 规则 flag、数据包流 `[u8 消息][u32 长度][负载]`，内容与 EDOPro 主机（`GenericDuel`）录进回放的逐字节相同：
  - `MSG_START` 之后：双方卡组（`PseudoRefreshDeck`，flag `0x1181fff`，核心查询原样拷贝）、双方额外卡组（`RefreshExtra`）的 `MSG_UPDATE_DATA`；
  - 每条核心消息照 `GenericDuel::Analyze`：去掉决策消息和仅发给单方的提示（1/2/3/5）；`BeforeParsing` 在待机/战斗指令前刷新双方怪兽区、魔陷区、手牌，
    在连锁选择与 `MSG_NEW_TURN` 前刷新怪兽区、魔陷区，`MSG_FLIPSUMMONING` 前刷新该卡，这几条消息本身排在刷新包之后（`record_last`）；
    `AfterParsing` 在抽卡/洗手牌、移动（换区或换控制者）、翻开、交换、召唤成功、阶段、连锁、伤害步骤等之后追加刷新（flag 与 `generic_duel.h` 的默认参数一致）；
  - 刷新包的负载照 `CoreUtils::QueryStream/Query::GenerateBuffer(check_hidden=false)` 重新编码：按 flag 从小到大写出，去掉没有位置的 reason/equip card；
  - 在 `MSG_WIN` 处停止；读到决策消息即不再处理同一缓冲里其后的消息（与主机相同）。与主机唯一的不同：`MSG_RETRY` 不写入（主机会就此结束对局）。
  - 导出时按应答日志重跑一局，经 `Duel.replay(..., observer=...)` 在每次 `process()` 之后、下一次应答之前查询核心（`query_location` / `query`），与主机查询的时机一致。
- 回放结果记为 `turn_limit` / `decision_limit`（核心没有发 `MSG_WIN`）时，末尾追加一个主机式的 `MSG_WIN` 包（负载 `[玩家, 原因]`，同主机 `replay_stream.emplace_back(wbuf, 2)`）：
  胜者取回放记录的 `result.winner`（换算成座位，平局 = 2），原因 `0x3`，EDOPro 显示为「[败者] Time limit up」，平局显示 Draw Game。没有记录终局的回放（如 `from_yrp` 读入的）不追加。
- 最后一个包（消息号 231，`OLD_REPLAY_MODE`）内嵌完整的 `yrp1`：同一种子、LP/起手/抽卡、规则 flag、按加载顺序（洗牌后）的双方卡组、应答日志。EDOPro 勾选「yrp 模式」时用自己的核心按它重新模拟。

刷新包让文件变大（每次刷新卡组都带 40 张卡的查询）：64 回合的一局约 3.3 MB，11 回合约 1 MB（EDOPro 自己的回放用 LZMA 压缩，我们不压缩）。

测试：`test_yrpx_export_structure` 用与 EDOPro 相同的解析逻辑读回文件并核对各字段；`test_embedded_yrp1_resimulates_like_edopro_old_mode` 只凭文件内容按 EDOPro 旧模式的步骤重跑核心，
得到与原局逐字节相同的消息流；`test_stream_carries_the_host_refresh_packets` 核对刷新包的位置与形状（开局卡组/额外卡组与直接查询核心一致、新回合前的四个场地刷新、抽卡后的手牌刷新、
移动后的单卡刷新、决策前的六个刷新、负载 flag 升序）；`test_recorded_at_is_the_header_date_of_both_replays`、`test_limit_games_end_with_a_host_msg_win` 覆盖日期与限时终局。

### 与 EDOPro 的对照（人工验证，2026-09）

- 客户端：从源码构建 EDOPro 41.0.2（edo9300/edopro，premake5 gmake2，无音频），静态链接本仓库打补丁后的核心（0001、0002），脚本与 `cards.cdb` 用本仓库 `third_party/`，
  在 Xvfb + llvmpipe 下打开导出的回放：流式模式与「Old Replay Mode」都能播放到底，胜者、LP、回合与 ygorl 一致，没有 replay error；
  流式模式的场上卡片现在显示等级与攻守（与 yrp 模式同一时刻的画面一致），列表显示录制日期，限回合对局显示「Time limit up」/ Draw Game。
- 逐字节对照：用 EDOPro 自己的代码（`replay.cpp` 解析、`core_utils.cpp`、`generic_duel.cpp` 的 `BeforeParsing` / `Sending` / `AfterParsing` / `Refresh*` 原文编译，网络部分打桩）
  从内嵌 `yrp1` 的种子、卡组、应答生成主机会录下的 `replay_stream`，与导出文件的数据包流逐字节比较：

| 对局 | 说明 | 数据包 | 结果 |
|------|------|-------:|------|
| snake_eye vs kashtira，seed 1 | a 先攻，11 回合 LP 胜 | 3453 | 逐字节相同 |
| labrynth vs tenpai，seed 7，`--first b` | b 先攻，21 回合 | 5797 | 逐字节相同 |
| purrely vs yubel，seed 3 | 64 回合抽干胜 | 9357 | 逐字节相同 |
| kashtira vs tenpai，seed 1 | 超量召唤（10 次叠放移动） | 4434 | 逐字节相同 |
| tearlaments vs branded_despia，seed 11，`--max-turns 4` | 限回合平局 | 886 + 1 | 相同，另加 `MSG_WIN [2, 3]` |
| purrely vs fiendsmith_ryzeal，seed 5，`--first b --max-turns 6` | 限回合按 LP 判胜 | 1509 + 1 | 相同，另加 `MSG_WIN [1, 3]` |

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
  `first=0`（文件的第一套卡组是引擎玩家 0，记为 a）；规则 flag 与 LP / 起手 / 抽卡数取自文件，不带环境；头里的时间戳作为 `recorded_at`（为 0 时记 `None`），`seed` 记 0
  （种子字已显式给出，时间戳不再当作种子），因此导出再读回、再导出，日期与种子不变。
- 相应地，`Duel(..., core_seed=[w0, w1, w2, w3])` 可以直接指定核心种子字；`Replay` 的 `seed_words` 只在不等于 `expand_seed(seed)` 时写进 JSON，普通对局的回放文件与之前逐字节相同。
- `Replay.to_yrpx(path, names, env, cards=..., scripts=...)` 的额外参数传给 `Duel`。
