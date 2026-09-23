# 观测编码规范（T2.2）

每个决策点对**决策方**（viewer）编码成定长 int32 数组。Python 参考实现是 `ygorl.env.encoding.ObservationEncoder`；C++ 实现须与之逐元素一致（交叉校验测试）。方案参考 ygo-agent 的 `docs/`（卡片表 + 全局向量 + 候选动作表），扩展了效果 id 与可见性规则。

| 数组 | 形状 | 说明 |
|------|------|------|
| `cards` | `[N_CARDS=160, F=23]` | 卡片 token 表，按下文顺序排列，未用行全 0 |
| `globals` | `[G=22]` | 全局向量 |
| `actions` | `[MAX_OPTIONS=128, A=10]` | 当前 step 的合法动作（多选已拆步，见 [engine.md](engine.md)） |
| `action_mask` | `[MAX_OPTIONS]` | 1 = 该行是合法动作，且是其等价类的第一行（见下「等价动作去重」） |

卡片身份用 `CardVocab` 下标（`0` 填充，`1` 未知/背面，真实卡从 `2` 起；见 `ygorl.cards.cdb.CardVocab`）。`CardVocab.from_db(db)` 按卡密排序新建，卡库增卡后下标会整体移动；训练产物必须与所用词表一起保存（`CardVocab.save`），卡库更新时用 `CardVocab.from_db(db, base=旧词表)` 只追加新卡，旧下标不变（T6.3 热启动依赖这一点）。

## 可见性

卡片身份对 viewer 可见，当且仅当：

1. 核心标记为公开（`QUERY_IS_PUBLIC`：表侧、正在连锁中、被公开效果影响的手牌/场上卡）；或
2. 是 viewer 自己控制的卡，位于手牌、怪兽区、魔陷区、墓地、除外区或额外卡组；或
3. 是超量素材；或
4. 是 viewer 自己的卡组——但只作为**构成**：卡组行按卡片下标排序、`sequence` 置 0，不泄露顺序。

不可见的行只保留位置信息（区域、序号、控制者、表示形式、叠放序号），身份与数值列全部为 0，`card_index = 1`。对手的卡组不入表（张数在全局向量里）；对手额外卡组只收录公开的卡（如表侧灵摆）。

## 卡片表行序

先 viewer 后对手；每方依次：怪兽区（按序号）、魔陷区（按序号）、手牌（按序号）、墓地、除外、额外卡组、超量素材（按所属怪兽序号、素材序号），最后是 viewer 的卡组构成。超过 160 行时从末尾截断（先截卡组构成）。

## 卡片表列（`F = 23`）

| 列 | 名称 | 取值 |
|----|------|------|
| 0 | `card_index` | 词表下标；0 填充，1 未知 |
| 1 | `location` | 0 无，1 卡组，2 手牌，3 怪兽区，4 魔陷区，5 墓地，6 除外，7 额外卡组，8 超量素材 |
| 2 | `sequence` | 区域内序号（超量素材为所属怪兽的序号；卡组行为 0） |
| 3 | `overlay_index` | 超量素材序号 + 1，否则 0 |
| 4 | `controller` | 0 viewer，1 对手 |
| 5 | `owner` | 0 viewer，1 对手 |
| 6 | `position` | 0 无，1 表侧攻击，2 里侧攻击，3 表侧守备，4 里侧守备 |
| 7 | `visible` | 身份是否可见（0/1） |
| 8 | `public` | 核心的公开标志（0/1） |
| 9 | `type` | `TYPE_*` 位掩码（不可见为 0） |
| 10 | `attribute` | 属性位序号 + 1（地 1 … 神 7），0 未知 |
| 11 | `race` | 种族位序号 + 1（1–63），0 未知 |
| 12 | `level` | 等级 |
| 13 | `rank` | 阶级 |
| 14 | `link` | 连接值 |
| 15 | `lscale` | 左灵摆刻度 |
| 16 | `rscale` | 右灵摆刻度 |
| 17 | `attack` | 当前攻击力，截断到 [0, 65535] |
| 18 | `defense` | 当前守备力，截断到 [0, 65535] |
| 19 | `link_marker` | 连接标记位掩码 |
| 20 | `counters` | 所有指示物总数 |
| 21 | `materials` | 超量素材数 |
| 22 | `disabled` | 效果被无效（`STATUS_DISABLED`） |

数值列取核心查询的**当前值**（受效果修改后）；超量素材行没有查询数据，数值列取卡片数据库的原始值。

## 全局向量（`G = 22`）

| 列 | 名称 | 取值 |
|----|------|------|
| 0 | `viewer` | 决策方的引擎玩家号（0 先攻方） |
| 1 | `is_first` | viewer 是否先攻 |
| 2 | `is_my_turn` | 当前回合玩家是否为 viewer |
| 3 | `turn` | 回合数，截断到 999 |
| 4 | `phase` | 阶段位序号 + 1（抽卡 1 … 结束 10），0 未开始 |
| 5 | `my_lp` | viewer 的 LP，截断到 [0, 65535] |
| 6 | `op_lp` | 对手 LP，同上 |
| 7–11 | `my_deck, my_hand, my_grave, my_removed, my_extra` | viewer 各区张数 |
| 12–16 | `op_deck, op_hand, op_grave, op_removed, op_extra` | 对手各区张数 |
| 17 | `chain` | 当前连锁长度 |
| 18 | `decision` | 决策消息号（`MSG_SELECT_*` 等） |
| 19 | `substep` | 多选已选数量（单步决策为 0） |
| 20 | `num_actions` | 合法动作总数（可能大于 128，见下） |
| 21 | `augmented_start` | 增广开局标志：`DuelConfig.augmented_start`（经 `DecisionPoint.augmented_start`，见 [curriculum.md](curriculum.md)） |

## 候选动作表（`A = 10`）

| 列 | 名称 | 取值 |
|----|------|------|
| 0 | `kind` | 动作类型编号 + 1（`ygorl.env.encoding.ACTION_KINDS` 的顺序），0 填充 |
| 1 | `card_row` | 所指卡在卡片表中的行号 + 1，0 无；卡组中的卡按卡片下标匹配构成行 |
| 2 | `card_index` | 所指卡（宣言动作为被宣言的卡）的词表下标；该卡对决策方隐藏时为 0（见下「决策中的隐藏信息」），此时第 1 列仍指向卡片表里那张（未知的）卡 |
| 3 | `effect_card` | 效果描述串所属卡的词表下标（`description >> 20` 是已知卡时），否则 0 |
| 4 | `effect_index` | 该卡的第几个效果串 + 1（`(description & 0xfffff) + 1`，即 `str1..16`），否则 0 |
| 5 | `system_string` | 描述不是卡片效果串时的系统串编号（截断到 65535），否则 0 |
| 6 | `position` | 表示形式动作的目标形式（同卡片表第 6 列编号），否则 0 |
| 7 | `zone` | 选区域动作：`控制者(0 viewer/1 对手) × 16 + (怪兽区 0 / 魔陷区 8) + 序号 + 1`，否则 0 |
| 8 | `value` | 与类型相关的小整数：宣言数字的数值、种族/属性的位序号 + 1、猜拳 1–3、指示物剩余数、祭品计数、求和数值（低 16 位），否则 0 |
| 9 | `index` | 动作在其所属列表中的下标 + 1（截断到 255），无则 0 |

EDOPro 脚本里 `aux.Stringid(code, n) = code << 20 | n`（`utility.lua`），所以描述的高位是卡片密码、低 20 位是串序号；比 `1 << 20` 小的描述是系统串。`effect_card` + `effect_index` 就是效果级文本嵌入（设计 I5）的查表键。

**决策中的隐藏信息**：核心在 SELECT_CARD / SELECT_TRIBUTE / SELECT_UNSELECT_CARD 里写的是真实卡密，包括对手里侧、手牌、卡组中的卡（例如选择破坏对手盖放的卡）；EDOPro 服务端把这三种消息里对手卡的卡密清零后才发给玩家（`generic_duel.cpp` `Sending`）。主机在解码决策时做同样的事（`messages.hide_private`，C++ `host::hide_private`）：对手的卡若不在墓地 / 超量素材中，且在卡组 / 手牌中或里侧表示（SELECT_TRIBUTE 不带表示形式，未知一律按隐藏处理），卡密置 0。表侧的对手卡保留卡密（它在场上本就可见）。因此 `DecisionPoint.actions`、动作表第 2 列、以及所有 agent（含 GreedyAgent）都看不到这些卡的身份；原始消息日志（回放、`.yrpX` 导出）不受影响。这是审计发现的泄露（修复前动作表第 2 列会带出对手里侧卡的身份），`tests/test_visibility.py` 覆盖 Python 与 C++ 两条路径。

**等价动作去重**：同一张卡的多张副本常各占一行（手里 3 张灰流丽 → 3 行「连锁灰流丽」），它们对决策方没有区别，却让均匀初始策略偏向发动（3/4 而不是 1/2），熵奖励也被摊到等价行上。因此 `action_mask` 只保留每个等价类的**第一行**，其余行照常编码、mask 置 0。第 i、j 行（i < j，都在前 128 行内）等价，当且仅当：

1. 两行都指向卡片表中的一行（第 1 列 > 0），且卡的身份对决策方可见（第 2 列 > 0）；
2. 动作行除 `card_row`（1）与 `index`（9）外各列相同；SELECT_UNSELECT_CARD（`globals[18]`）中 `value`（8）是列表下标，也不比较；
3. 两张卡在卡片表中的行除 `overlay_index`（3）外各列相同；卡组、手牌、墓地、除外、额外卡组里的卡另外不比较 `sequence`（2）。怪兽区、魔陷区的序号决定纵列与连接方向，不同序号不等价；超量素材行的 `sequence` 是所属怪兽的序号，也必须相同。

于是公开标志不同（一张已经给对手看过）、表示形式不同（表侧 / 里侧除外）、数值被效果改过、效果串不同的都不算等价；身份对决策方隐藏的卡（对手手牌、里侧卡）也不合并。多选拆步不受影响：选了代表行之后，下一步列出剩余的副本，任何多重集仍然可达。主机的 `skip_forced`（[training.md](training.md)）按 mask 判定：去重后只剩一行的决策也视为强制。`globals[20]` 仍是去重前的合法动作总数。`ygorl.env.encoding.canonical_action(obs, i)` 把任意一行映射到它的代表行（行为克隆、回放标签用）。10 套测试牌组 40 局随机对局里，7.4% 的非强制决策有被合并的行（SELECT_CARD 36%，多为从卡组检索多张同名卡之一；SELECT_IDLECMD 10%；SELECT_CHAIN 3%），其中 75 个因此变成强制决策。

已知简化：手牌序号在对手看来可能带信息（例如对手见过被检索公开的那张卡加到了手牌末位），合并后总是用序号最小的副本；需要隐藏时仍可选洗切手牌（`shuffle`）。

**截断（未解决）**：合法动作超过 128 个时（主要是 ANNOUNCE_CARD 宣言卡名，可达上千个）只编码前 128 个，`globals[20]` 记录真实数量，`action_mask` 只标前 128 行，**第 128 行之后的动作 agent 选不到**。T2.3 只处理了多选拆步，没有处理这一点；可能的方案是把宣言拆成「先选类别 / 种族 / 属性再选卡」的多步，或按信念头排序只列前 128 个。待网络（T4b.1）落地后决定，见 [eng-plan.md](eng-plan.md) T2.3 备注。

## 实现与交叉校验

- Python 参考：`ygorl.env.encoding.ObservationEncoder`（`tests/test_encoding.py`）。
- C++：`csrc/host.{h,cpp}`（决策解码、动作状态机、主机侧 tracker）与 `csrc/obs_encoder.cpp`（编码），通过 `ygorl._core.HostDuel` / `ygorl._core.DecisionState` 暴露。
- 校验：`tests/test_cpp_host.py` 对 19 种决策消息做随机报文 + 随机选择路径的差分测试（动作列表与应答字节必须与 `ygorl.engine.actions` 一致），并在 5 局真实对局中逐步比对动作、四个观测数组与终局；`uv run python tools/check_cpp_encoder.py --points 10000` 做验收。2026-09-22 的一次运行：8 局、10,530 个决策点、0 处不一致，覆盖 16 种决策类型（其余类型由差分测试覆盖）。

## 训练态真值（privileged，T2.5）

训练时环境另输出一组**真值张量**：对手（相对 viewer，即 `1 - viewer`）的隐藏信息，供特权 critic（设计 I2/I9）与信念头的损失
（[设计 04](design/04-opponent-model.md)）使用。推理态不计算、不输出。

**隔离约定（actor 拿不到）**：

- 真值从不进入上文四个 actor 数组，也不进入 actor 的 obs dict。它放在单独的结构里：`EncodedEvent.privileged`
  （`EncodedVecEnv`）、`ObservationEncoder.encode_privileged()`（Python 参考）、`HostDuel.observe_privileged()`（C++ 调试句柄）。
  actor 只接 `event.obs`，critic / 信念头的损失另接 `event.privileged`。
- 模式是构造期开关：`EncodedVecEnv(..., privileged=False)`、`_core.HostPool(..., privileged=False)`、
  `ObservationEncoder(..., privileged=False)`，默认关闭（推理态）。推理态下 C++ 工作线程不做任何真值查询，
  `event.privileged is None`；`ObservationEncoder.encode_privileged()` 直接抛 `RuntimeError`。评估 / 对战（arena、agent）一律用推理态。
- 两种模式下 actor 观测逐元素相同（真值开关不改变 actor 输入）；actor 编码本身从不查询对手卡组（单测用查询探针检查）。

### 张量

全部是 `int32`，每一行是 `(card_index, public, sequence)`：`card_index` 为 `CardVocab` 下标（真实身份，不做可见性遮蔽），
`public` 为核心的 `QUERY_IS_PUBLIC`（1 = 该张对 viewer 已公开，actor 表里本就可见），`sequence` 为区域内序号。
未用行全 0（`card_index = 0` 即填充）。

| 键 | 形状 | 内容 | 行序 |
|----|------|------|------|
| `op_hand` | `[P_HAND=32, 3]` | 对手全部手牌（含已公开的） | 手牌序号（与 actor 表中对手手牌行按 `sequence` 对齐） |
| `op_deck` | `[P_DECK=64, 3]` | 对手主卡组剩余构成 | 按 `(card_index, public)` 排序，`sequence` 置 0（只给构成，不泄露顺序） |
| `op_extra` | `[P_EXTRA=32, 3]` | 对手额外卡组全部卡（里侧 + 表侧灵摆，`public` 区分） | 同 `op_deck` |
| `op_set` | `[P_SET=15, 3]` | 对手场上**里侧**的卡：第 0–6 行 = 怪兽区序号 0–6，第 7–14 行 = 魔陷区序号 0–7 | 按区域固定；空区域或表侧卡为全 0 行 |
| `op_removed` | `[P_REMOVED=64, 3]` | 对手里侧除外的卡 | 除外区序号（与 actor 表中除外行对齐） |
| `counts` | `[5]` | 截断前的真实张数：手牌、主卡组、额外卡组、里侧场上、里侧除外 | — |

各列表超出宽度时截断（保留排序后的前若干行），`counts` 记录真实张数，据此可发现截断；宽度按「主卡组 ≤ 60、额外 ≤ 15」留了余量：
2026-09-22 用 10 套测试牌组随机对局 200 局（240,587 个决策点）统计的最大张数为手牌 8、主卡组 37、额外 15、里侧场上 8、
里侧除外 21，没有截断。训练态每个决策点多 6 次位置查询，同一批对局的 C++ 步进吞吐约低 5%（5,478 → 5,208 决策/秒，8 个 env、4 线程）。超量素材总是公开的，不在真值里；viewer 自己的卡组顺序也不输出（真值只描述对手）。

### 与信念头目标的对应

`ygorl.env.privileged.belief_targets(priv, candidates)` 把真值换算成 [belief-eval.md](belief-eval.md) 的
`Head(targets, mask)` 形状（可带前导批维）。`candidates` 是候选卡集合（「meta 并集 + 泛用卡」，按 8 位 `password` 给出，
经 `CandidateCards(vocab, passwords)` 映射成 C 列；不在集合里的卡不计数）：

| 头 | targets | mask（True = 参与损失 / 评估） |
|----|---------|------|
| `hand` | `[C]` 0/1：该卡 ≥ 1 张在对手手牌 | 手牌里没有该卡的已公开张（已公开 → 按构造置 1，掩掉） |
| `remaining_copies` | `[C]` ∈ 0..3：主卡组 + 额外卡组中**未公开**的份数，截断到 3 | 全 True（已现份数不计入，即「已现份数从剩余中扣除」） |
| `set_cards` | `[15]`：里侧卡的候选列号，空区域 / 非候选卡为 −1 | 区域有里侧卡、未公开且是候选卡 |

`copy_counts(rows, candidates)` 给出任意列表在候选集上的原始份数 `[C]`（不截断），例如 `copy_counts(priv["op_deck"], cands)`
即主卡组剩余构成；传入 `CardVocab` 本身时按整个词表计数（`[len(vocab)]`，填充 / 未知不计）。这两个函数是纯 numpy，C++ / Python 两条路径共用。

### 实现与校验

- Python 参考：`ygorl.env.privileged.encode_privileged(core, viewer, vocab)`；C++：`csrc/privileged.{h,cpp}`（只查
  `QUERY_CODE | QUERY_POSITION`，`QUERY_IS_PUBLIC` 总会返回）。
- `tests/test_privileged.py`：推理态 `privileged is None` 且 Python 入口抛错；训练态与直接的引擎查询（逐张 `core.query` +
  `query_count`，不经位置查询解析）一致；C++ 与 Python 在 5 局随机对局的全部决策点上逐元素一致；`EncodedVecEnv` 两种模式的
  actor 观测逐元素相同、训练态真值与单局 `HostDuel` 一致；`belief_targets` 的形状能直接构造 `BeliefBatch`。
- `tools/check_cpp_encoder.py` 同时比对真值：2026-09-22 的一次运行 8 局、10,530 个决策点、0 处不一致，其中 8,025 个点
  `op_set` 非空、921 个点 `op_removed` 非空。

## 事件 token 流（T2.4）

设计 I6（[03-play-policy.md](design/03-play-policy.md)）与「不响应」证据（[04-opponent-model.md](design/04-opponent-model.md)）的落地：把核心发出的 `MSG_*` 事件按 **viewer 可见性**过滤后编码成定长整数行，另插入「响应窗口 + 放弃」token。Python 参考实现是 `ygorl.env.events.EventHistory`，C++ 实现是 `csrc/event_encoder.cpp`（`EventHistory`），两者逐元素一致。

| 数组 | 形状 | 说明 |
|------|------|------|
| `events` | `[L, E=20]` | viewer 视角最近 `L` 个事件 token，按时间先后排在第 0…n−1 行（最新在第 n−1 行），其余行全 0 |
| `event_mask` | `[L]` | 1 = 该行是事件 |

`L` 可配（`EventHistory(length=L)`、`EncodedVecEnv(event_length=L)`、`_core.HostDuel(..., event_length=L)`；默认 `DEFAULT_EVENT_LENGTH = 128`，0 = 不输出）。历史从对局开始累积（每个 viewer 各一份），只保留最近 `L` 个。

**状态与输入**：历史吃的是**全部**核心消息（上帝视角，按产生顺序；Python 侧即依次喂入每个 `DecisionPoint.events`，C++ 侧即 `HostDuel` 收到的每个消息缓冲），同时为两个 viewer 生成 token；每个 token 在生成时就按该 viewer 的可见性过滤，之后不再改动。决策点的观测取决策方（`DecisionPoint.player`）那一份。

### 列（`E = 20`）

| 列 | 名称 | 取值 |
|----|------|------|
| 0 | `type` | 事件类型编号（下表），0 填充 |
| 1 | `player` | 事件主体玩家：0 无，1 viewer，2 对手（含义见下表） |
| 2 | `card` | 主卡的词表下标：0 无 / 未追踪，1 对 viewer 隐藏，≥ 2 卡片 |
| 3 | `card2` | 第二张卡（攻击目标、装备对象、交换的另一张；`CHAINING` 为效果串所属卡），取值同上 |
| 4–7 | `from_controller, from_location, from_sequence, from_position` | 起点位置 |
| 8–11 | `to_controller, to_location, to_sequence, to_position` | 终点位置 |
| 12–14 | `value1, value2, value3` | 与类型相关的整数（下表） |
| 15 | `turn` | 事件后的回合数，截断到 999 |
| 16 | `phase` | 事件后的阶段位序号 + 1（同全局向量第 4 列） |
| 17 | `my_turn` | 事件后的回合玩家是否为 viewer |
| 18 | `my_lp` | 事件后 viewer 的 LP，截断到 [0, 65535] |
| 19 | `op_lp` | 事件后对手的 LP，同上 |

位置四列的编码：`controller` 0 无、1 viewer、2 对手；`location` 同卡片表第 1 列（`LOCATION_OVERLAY` 位优先，记 8），0 无；`sequence` 为区域内序号（超量素材为所属怪兽序号），**卡组内一律记 0**（不泄露卡组顺序，同卡片表规则 4）；`position` 1–4 同卡片表第 6 列，其余表侧形式（魔陷、除外等）记 5、其余里侧形式记 6，超量素材与无位置信息记 0。没有该位置的事件四列全 0。

### 事件类型

| 编号 | 名称 | 来源 | `player` | `card` / `card2` | 位置 | `value1` / `value2` / `value3` |
|----|------|------|------|------|------|------|
| 1 | `draw` | `MSG_DRAW`，每张卡一行 | 抽卡者 | 抽到的卡 | to = 抽卡者手牌 | 本条消息抽卡张数 |
| 2 | `move` | `MSG_MOVE` | 终点控制者（终点为空时取起点） | 移动的卡 | from / to | `reason` 位掩码（`& 0x7fffffff`） |
| 3 | `pos_change` | `MSG_POS_CHANGE` | 控制者 | 该卡 | from = 原表示形式，to = 新表示形式 | |
| 4 | `set` | `MSG_SET` | 控制者 | 该卡 | to | |
| 5 | `swap` | `MSG_SWAP` | — | 卡 1 / 卡 2 | from = 卡 1 原位置，to = 卡 2 原位置 | |
| 6–8 | `summoning` / `spsummoning` / `flipsummoning` | `MSG_SUMMONING` 等 | 控制者 | 被召唤的卡 | to | |
| 9 | `chaining` | `MSG_CHAINING` | 发动者 | 发动的卡 / 效果串所属卡 | from = 该卡位置 | 连锁序号 / 效果串序号 + 1 / 系统串编号（同候选动作表第 3–5 列规则） |
| 10–12 | `chain_solving` / `chain_negated` / `chain_disabled` | 同名消息 | 该连锁的发动者 | 该连锁发动的卡 | | 连锁序号 |
| 13 | `chain_end` | `MSG_CHAIN_END` | | | | |
| 14 | `new_turn` | `MSG_NEW_TURN` | 新回合玩家 | | | |
| 15 | `new_phase` | `MSG_NEW_PHASE` | | | | 阶段位序号 + 1 |
| 16–18 | `damage` / `recover` / `pay_lpcost` | 同名消息 | 受影响玩家 | | | 数值（截断到 65535） |
| 19 | `lp_update` | `MSG_LPUPDATE` | 受影响玩家 | | | 新 LP（截断） |
| 20 | `attack` | `MSG_ATTACK` | 攻击怪兽控制者 | 攻击怪兽 / 攻击对象 | from = 攻击怪兽，to = 攻击对象（直接攻击为空） | 直接攻击 1，否则 0 |
| 21 | `battle` | `MSG_BATTLE` | 攻击怪兽控制者 | 攻击怪兽 / 攻击对象 | from / to | 攻击怪兽攻击力 / 对象攻击力 / 对象守备力（截断到 [0, 65535]） |
| 22 | `attack_disabled` | `MSG_ATTACK_DISABLED` | | | | |
| 23 | `equip` | `MSG_EQUIP` | 装备卡控制者 | 装备卡 / 装备对象 | from / to | |
| 24 | `unequip` | `MSG_UNEQUIP` | 控制者 | 装备卡 | from | |
| 25–26 | `card_target` / `cancel_target` | 同名消息 | 控制者 | 卡 / 对象 | from / to | |
| 27–29 | `become_target` / `card_selected` / `random_selected` | 同名消息，每个位置一行 | 该位置控制者（`random_selected` 为选择者） | 该位置的卡 | from | |
| 30–31 | `add_counter` / `remove_counter` | 同名消息 | 控制者 | 该位置的卡 | from | 指示物类型 / 数量 |
| 32–34 | `confirm_cards` / `confirm_decktop` / `confirm_extratop` | 同名消息，每张卡一行 | 被出示者（`confirm_cards`）/ 卡组持有者 | 该卡 | from | |
| 35 | `deck_top` | `MSG_DECK_TOP` | 卡组持有者 | 卡组顶的卡 | | |
| 36–38 | `shuffle_deck` / `shuffle_hand` / `shuffle_extra` | 同名消息 | 该玩家 | | | 洗切张数（卡组为 0） |
| 39 | `shuffle_set_card` | `MSG_SHUFFLE_SET_CARD` | | | from_location = 区域 | 张数（洗切后这些区域仍视为有卡，但各区域的卡身份在场地表中变为未知，之后从其中移出的卡会清掉该区域） |
| 40 | `swap_grave_deck` | `MSG_SWAP_GRAVE_DECK` | 该玩家 | | | |
| 41 | `reverse_deck` | `MSG_REVERSE_DECK` | | | | |
| 42 | `field_disabled` | `MSG_FIELD_DISABLED` | | | | viewer 的区域位掩码（16 位）/ 对手的 |
| 43–44 | `toss_coin` / `toss_dice` | 同名消息 | 投掷者 | | | 次数 / 结果（硬币：第 i 次为正面置位 i；骰子：第 i 个点数放在第 3i 位起的 3 位，至多 10 个） |
| 45 | `hand_res` | `MSG_HAND_RES` | | | | viewer 出的拳 / 对手出的拳（1–3） |
| 46 | `abstain` | 合成（见下） | 放弃响应者 | 触发卡 | from = 触发卡位置 | 触发类型位掩码 / 放弃者场上卡数 / 放弃者手牌数 |

「该位置的卡」来自历史自己维护的场上图（见下）；不在怪兽区 / 魔陷区的位置（墓地、除外、手牌等）不追踪，`card` 记 0。

**不编码的消息**：全部决策消息（`MSG_SELECT_*`、`MSG_SORT_*`、`MSG_ANNOUNCE_*`、`MSG_ROCK_PAPER_SCISSORS`）——它们只发给决策方，内容（有哪些可选项）正是隐藏信息；`MSG_HINT` / `MSG_CARD_HINT` / `MSG_PLAYER_HINT` / `MSG_SHOW_HINT`（界面提示，部分只发给一方）；`MSG_MISSED_EFFECT`（只发给该卡控制者）；`MSG_RETRY`、`MSG_WAITING`、`MSG_WIN`、`MSG_RELOAD_FIELD`、`MSG_TAG_SWAP`、`MSG_REMOVE_CARDS`、`MSG_AI_NAME`、`MSG_MATCH_KILL`（非对局事件或对局已结束）；与相邻 token 冗余的 `MSG_SUMMONED` / `MSG_SPSUMMONED` / `MSG_FLIPSUMMONED`（紧跟对应的 `*SUMMONING`）、`MSG_CHAINED`（紧跟 `CHAINING`）、`MSG_CHAIN_SOLVED`（下一个 `chain_solving` 或 `chain_end` 隐含）、`MSG_DAMAGE_STEP_START` / `MSG_DAMAGE_STEP_END`（由 `battle` 隐含）；以及核心从不发出的消息。决策消息虽不编码，仍参与下文响应窗口的关闭判定（只用其类型，类型在对局规则下是公开的）。

### 卡片身份的可见性

核心消息里的卡片密码是上帝视角的；token 的 `card` / `card2` 只在 viewer 本就能知道时填卡片下标，否则记 1。与 EDOPro 服务器转发消息时的过滤一致，并补上「来源公开」的情形。先定义位置 `loc` **公开**：超量素材或墓地 → 公开；卡组、手牌 → 不公开；怪兽区、魔陷区、除外、额外卡组 → 表侧表示时公开；其他 → 不公开。

| 事件 | 卡片可见，当且仅当 |
|------|------|
| `draw` | viewer 是抽卡者，或该卡以表侧抽出（卡组翻转时） |
| `move` | viewer 是终点控制者（终点为空时取起点控制者），或终点公开，或起点公开 |
| `pos_change` | viewer 是控制者，或原 / 新表示形式之一为表侧 |
| `set` | viewer 是控制者 |
| `swap` | 每张卡分别：viewer 是其新控制者，或其原位置公开 |
| `*summoning` | viewer 是控制者，或召唤位置公开（里侧特殊召唤对对手隐藏） |
| `chaining`、`chain_*`、`abstain` | 总是可见（发动即公开） |
| `confirm_cards` | viewer 是被出示者，或该卡不在卡组（手牌等的出示双方都看到，EDOPro 同此） |
| `confirm_decktop` / `confirm_extratop` | 总是可见 |
| `deck_top` | 该卡为表侧 |
| 场上图查得的卡（`attack`、`battle`、`equip`、`*target`、`*selected`、指示物） | viewer 是该卡控制者，或该卡当前为表侧 |

对手抽到的卡、对手盖放的卡、加入对手手牌 / 卡组的卡、对手的里侧除外，因此都记 1；对手从卡组检索后按效果出示时，出示本身产生可见的 `confirm_cards` token。位置、张数、LP、连锁结构等**公开信息**不受影响。

**场上图**：历史按 `move` / `pos_change` / `swap` / `set` / `*summoning` 维护两名玩家怪兽区（序号 0–6）与魔陷区（序号 0–7）上每个格子的卡与表示形式（`move` 只在起点格子仍是同一密码时才清空起点，以正确处理两条 `MSG_MOVE` 组成的交换），同时按 `draw` / `move` 维护双方手牌张数。场上卡数与手牌张数都是公开信息；测试逐决策点核对它们与核心查询一致。

### 响应窗口与放弃 token

「对手本可以响应却没有响应」是信念头的重要负证据（04 §4.5），但**对手是否真有合法响应是隐藏信息**：核心只在有可发动效果时才对某些窗口发 `MSG_SELECT_CHAIN`，决策消息的内容也只发给决策方。因此放弃 token 只由**公开**事件按规则判定——「在触发 X 之后按规则开了一个响应窗口，而某玩家在窗口内没有发动任何效果」——与该玩家手里有没有能发动的卡、核心有没有问他都无关。对 viewer 而言，「对手有灰流丽但放弃」与「对手根本没有灰流丽」产生**完全相同**的 token 流（测试 `test_abstain_stream_is_public` 逐行核对）。

触发类型位掩码（`value1`）：

| 位 | 名称 | 含义 |
|----|------|------|
| 1 | `search` | 该连锁处理时有卡从卡组加入手牌（`MSG_MOVE` 卡组 → 手牌） |
| 2 | `spsummon_deck` | 该连锁处理时有卡从卡组特殊召唤（`MSG_MOVE` 卡组 → 怪兽区） |
| 4 | `send_deck_grave` | 该连锁处理时有卡从卡组送去墓地（堆墓） |
| 8 | `fifth_summon` | 回合玩家本回合第 5 次召唤 / 特殊召唤（尼比鲁的条件） |
| 16 | `attack` | 攻击宣言 |
| 32 | `other` | 该连锁处理时以上三种都没有发生（包括被无效） |

两类窗口：

1. **发动窗口**（灰流丽、效果遮蒙者、无限泡影类）：连锁第 `n` 环由玩家 A 发动。它处理完毕（`MSG_CHAIN_SOLVED n`）时，若对手 B 在第 `n` 环之后没有加入任何一环（B 在第 n 环之上没有连锁），就生成一个 `abstain` token：`player` = B，`card` / from = 第 `n` 环发动的卡，触发类型取第 `n` 环处理期间（`MSG_CHAIN_SOLVING n` 到 `MSG_CHAIN_SOLVED n`）观察到的卡组移动（`search` / `spsummon_deck` / `send_deck_grave` 的并集，都没有则 `other`）。处理期间才归类，是因为「这个效果含检索」只在处理后才由公开事件确认；被 B 连锁（包括被灰流丽无效）的环不生成 token——连锁本身已经是可见的 `chaining` token。双方对称：B 发动、A 不连锁同样生成 `player` = A 的 token。
2. **事件窗口**（尼比鲁、攻击反应类）：`MSG_ATTACK`（放弃者 = 攻击怪兽控制者的对手）与回合玩家本回合第 5 次 `summoning` / `spsummoning`（按召唤位置的控制者计，`MSG_NEW_TURN` 清零；放弃者 = 非回合玩家）各打开一个窗口；放弃者在窗口内发动任何效果（`MSG_CHAINING` 的发动者是他）即视为已响应；窗口在下一次回到开放局面时关闭——即下一个 `MSG_SELECT_IDLECMD` / `MSG_SELECT_BATTLECMD`（任何一方）、`MSG_NEW_PHASE` 或 `MSG_NEW_TURN`，此时对每个未响应的窗口生成 `abstain` token（先于该消息自身的 token）。`card` 为攻击怪兽 / 第 5 次召唤的怪兽（按 `*summoning` 的可见性），from 为其位置。

`abstain` token 的 `value2` / `value3` 是放弃者当时的场上卡数（怪兽区 + 魔陷区）与手牌张数，`my_lp` / `op_lp` / `turn` 列给出 LP 与回合——即设计中的「对手场面、LP、回合」。

**已知局限**：从卡组以外（额外卡组、墓地）的检索与特召、「宣言卡名」等只在提示消息里出现的公开信息不单独编码；场上图之外的位置（墓地、除外）被取对象时 `card` 记 0；手牌里被公开（`QUERY_IS_PUBLIC`）的卡在之后的移动中仍按上表规则判定；增殖的 G 类「对方特殊召唤时」、屋敷童 / 墓穴类「墓地效果发动时」没有专门的触发位（落在 `other` 或不产生窗口）。

### 实现与交叉校验

- Python 参考：`ygorl.env.events.EventHistory`；C++：`csrc/event_encoder.{h,cpp}`，经 `_core.HostDuel(..., event_length=L)`、`_core.HostPool(..., event_length=L)`（`EncodedVecEnv(event_length=L)`）输出 `events` / `event_mask`，另以 `_core.EventHistory` 单独暴露供差分测试。
- `tests/test_events.py`：手工构造的灰流丽场景（A 发动增援检索，B 手里 5 张灰流丽、被问到连锁且选择不连锁 → 双方流里出现 `player` = B、触发 `search`、B 场上 0 张 / 手牌 5 张的 `abstain` token；B 连锁灰流丽时该环不产生 token）、信息集检验（B 手里是灰流丽还是无法响应的通常怪兽，A 在每个决策点的 token 流逐行相同）、对手抽卡 / 盖放 / 检索的隐藏、卡组序号置 0、手牌与场上张数逐决策点与核心查询一致、`L` 可配（0 / 1 / 8 / 300 与完整历史的末尾一致）、合成的第 5 次召唤与攻击宣言窗口、全部 46 种事件类型的随机报文（含截断与尾随字节）C++ 与 Python 逐元素一致、3 局真实对局 `HostDuel` 逐决策点一致、`EncodedVecEnv` 的形状与开关。
- 验收：`uv run python tools/check_cpp_events.py --points 100000 --length 512`。2026-09-22 的一次运行：81 局、100,842 个决策点、0 处不一致（viewer 0 共 85,057 个 token；`abstain` 触发位：other 4,308、attack 476、search 443、spsummon_deck 213、send_deck_grave 61、fifth_summon 42）；`--length 8` 另跑 16 局 20,268 个决策点，0 处不一致。随机对局平均每回合约 20 个 token（每决策点 0.8 个）；真实 combo 回合会多得多，训练时按需调大 `L`。
