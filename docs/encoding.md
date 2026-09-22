# 观测编码规范（T2.2）

每个决策点对**决策方**（viewer）编码成定长 int32 数组。Python 参考实现是 `ygorl.env.encoding.ObservationEncoder`；C++ 实现须与之逐元素一致（交叉校验测试）。方案参考 ygo-agent 的 `docs/`（卡片表 + 全局向量 + 候选动作表），扩展了效果 id 与可见性规则。

| 数组 | 形状 | 说明 |
|------|------|------|
| `cards` | `[N_CARDS=160, F=23]` | 卡片 token 表，按下文顺序排列，未用行全 0 |
| `globals` | `[G=22]` | 全局向量 |
| `actions` | `[MAX_OPTIONS=128, A=10]` | 当前 step 的合法动作（多选已拆步，见 [engine.md](engine.md)） |
| `action_mask` | `[MAX_OPTIONS]` | 1 = 该行是合法动作 |

卡片身份用 `CardVocab` 下标（`0` 填充，`1` 未知/背面，真实卡从 `2` 起；见 `ygorl.cards.cdb.CardVocab`）。

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
| 2 | `card_index` | 所指卡（宣言动作为被宣言的卡）的词表下标 |
| 3 | `effect_card` | 效果描述串所属卡的词表下标（`description >> 20` 是已知卡时），否则 0 |
| 4 | `effect_index` | 该卡的第几个效果串 + 1（`(description & 0xfffff) + 1`，即 `str1..16`），否则 0 |
| 5 | `system_string` | 描述不是卡片效果串时的系统串编号（截断到 65535），否则 0 |
| 6 | `position` | 表示形式动作的目标形式（同卡片表第 6 列编号），否则 0 |
| 7 | `zone` | 选区域动作：`控制者(0 viewer/1 对手) × 16 + (怪兽区 0 / 魔陷区 8) + 序号 + 1`，否则 0 |
| 8 | `value` | 与类型相关的小整数：宣言数字的数值、种族/属性的位序号 + 1、猜拳 1–3、指示物剩余数、祭品计数、求和数值（低 16 位），否则 0 |
| 9 | `index` | 动作在其所属列表中的下标 + 1（截断到 255），无则 0 |

EDOPro 脚本里 `aux.Stringid(code, n) = code << 20 | n`（`utility.lua`），所以描述的高位是卡片密码、低 20 位是串序号；比 `1 << 20` 小的描述是系统串。`effect_card` + `effect_index` 就是效果级文本嵌入（设计 I5）的查表键。

**截断**：合法动作超过 128 个时（主要是宣言卡名）只编码前 128 个，`globals[20]` 记录真实数量，`action_mask` 只标前 128 行；这类决策的处理在 T2.3 细化。

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
