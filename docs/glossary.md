# 术语表

代码标识符（英文）与中文含义的对照。新代码、新文档按本表命名；新增概念先补进本表。
CLAUDE.md 的「游戏王术语约定」是本表的核心子集。

## 卡片与卡组

| 标识符 | 中文 | 说明 |
|--------|------|------|
| `password` | 卡片密码 | 官方 8 位数字卡密，卡片的唯一主键；不要用卡名做键（卡名只用于显示） |
| `alias` | 同名卡 / 异画 | cdb 的 `alias` 字段；异画卡按原卡计张数（`cards.legality.canonical_password`） |
| `monster` / `spell` / `trap` | 怪兽卡 / 魔法卡 / 陷阱卡 | 卡片大类（cdb `type` 位） |
| `token` | 衍生物 | 不能放进卡组 |
| `attack` / `defense` | 攻击力 / 守备力 | 不写 `atk` / `def`（仅在逐字对应 cdb 列名或核心消息字段的局部变量里例外） |
| `level` / `rank` / `link` | 等级 / 阶级 / 连接值 | |
| `lscale` / `rscale` | 左 / 右灵摆刻度 | |
| `link_marker` | 连接标记 | 位掩码 |
| `attribute` | 属性 | 光、暗、地、水、炎、风、神 |
| `race` | 种族 | |
| `setcode` / `archetype` | 字段 / 系列 | `setcode` 是 cdb 里的系列编码；讨论组牌时称 archetype |
| `main_deck` / `extra_deck` / `side_deck` | 主卡组 / 额外卡组 / 副卡组 | 独立变量与公开 API 用全称；`Deck` 对象与回放 JSON 内部的字段简写为 `main` / `extra` / `side`（作用域已限定在卡组里） |
| `banlist` | 禁限卡表 | `.lflist.conf` |
| `pool` | 卡池 | 环境允许使用的卡 |
| `hand_trap` | 手坑 | 从手牌发动、干扰对手展开的卡 |

## 区域（location）

区域名与核心的 `LOCATION_*` 常量一一对应，代码里一律用这些名字。

| 标识符 | 中文 | 核心常量 |
|--------|------|----------|
| `deck` | 卡组 | `LOCATION_DECK` |
| `hand` | 手牌 | `LOCATION_HAND` |
| `mzone` | 怪兽区（含额外怪兽区，序号 0–6） | `LOCATION_MZONE` |
| `szone` | 魔法陷阱区（序号 0–4；场地 5；灵摆 6–7） | `LOCATION_SZONE` |
| `grave` | 墓地 | `LOCATION_GRAVE`；不用 `gy` / `graveyard`（如 `send_to_grave`） |
| `removed` | 除外区 | `LOCATION_REMOVED`；动作动词用 `banish`（除外） |
| `extra` | 额外卡组所在区域 | `LOCATION_EXTRA`（区域名；卡组本身见 `extra_deck`） |
| `overlay` | 超量素材 | `LOCATION_OVERLAY`；`materials` 指素材数量或素材卡 |
| `field` / `onfield` | 场上 | 怪兽区 + 魔法陷阱区 |
| `sequence` | 区域内序号 | |
| `position` | 表示形式 | `faceup` / `facedown`（表侧 / 里侧），`attack` / `defense` 表示 |
| `controller` / `owner` | 控制者 / 持有者 | |

## 对局流程

| 标识符 | 中文 | 说明 |
|--------|------|------|
| `lp` | 生命值 | 核心中是有符号 int32，致命伤害后可为负 |
| `turn` / `turn_player` | 回合 / 回合玩家 | |
| `first` | 先攻方 | `Duel(first=1)` 让 b 先攻 |
| `phase` | 阶段 | draw、standby、main1、battle、main2、end |
| `summon` / `spsummon` / `flip_summon` | 通常召唤 / 特殊召唤 / 反转召唤 | |
| `set` / `mset` / `sset` | 盖放 / 盖放怪兽 / 盖放魔陷 | |
| `activate` | 发动 | |
| `chain` / `chain link` | 连锁 / 连锁环 | |
| `tribute` | 祭品（解放） | |
| `material` | 素材 | 融合 / 同调 / 超量 / 连接素材 |
| `search` | 检索 | 从卡组加入手牌 |
| `win_reason` | 胜利原因 | 1 = LP，2 = 卡组耗尽，0x10 以上为卡片特殊胜利 |
| `turn_limit` / `decision_limit` | 回合上限 / 决策数上限 | 主机侧的保险；训练时视为截断而非真实终局 |

## 引擎与主机

| 标识符 | 中文 | 说明 |
|--------|------|------|
| engine player / seat | 引擎玩家 / 座位 | 0 先攻；座位 p 持卡组 `(first + p) % 2`（0 = a），只由 `deck_of_seat` / `seat_of_deck`（`GameSpec` 同名方法、`Duel.deck_of`）回答 |
| side a / b | a 方 / b 方 | 按卡组区分，结果按 (a, b) 顺序报告 |
| `decision` | 决策 | 需要玩家应答的 `MSG_SELECT_*` 等消息 |
| `response` | 应答 | `set_response` 的字节 |
| `action` / step | 动作 / agent 步 | 多选拆成多步，一次应答可对应多个动作 |
| `viewer` | 视角玩家 | 观测按其可见性编码 |
| public / hidden | 公开 / 隐藏 | `messages.is_hidden_from` 定义对决策方隐藏的卡 |
| equivalent action / `canonical_action` | 等价动作 / 代表行 | 同一张卡的多张副本在同一决策里各占一行时只保留第一行（`action_mask`），见 encoding.md「等价动作去重」 |
| no-op undo / `DecisionPoint.undo` | 撤销类空操作 | 退出刚开始的命令、立即反悔的选 / 取消选；局面回到完全相同的决策，`action_mask` 遮住，见 encoding.md |
| `abstain` | 放弃响应 token | 有响应窗口而未连锁（事件流） |
| `snapshot` / `restore` | 快照 / 恢复 | 每局 arena 的内存拷贝（T2.8） |
| `fork` / `branch` / `rollout` | 分叉 / 分支 / 推演 | 从某个决策点尝试候选并下完（T2.9） |
| `curriculum` / `learner` | 课程模式 / 学习方 | full / solo / handtrap（T2.6） |
| `privileged` | 训练态真值 | 只给 critic 与信念头，绝不喂给 actor |

## 训练与搜索

| 标识符 | 中文 | 说明 |
|--------|------|------|
| BC / warm start | 行为克隆 / 热启动 | 用求解器与 Greedy 示范预训练，再接 PPO（bc.md） |
| `target_kl` | 按 KL 提前停 | 一次更新内 minibatch 的 `approx_kl` 超过 1.5 × 目标即停（training.md） |
| pinned opponent | 固定对手 | `--pin` 钉进快照池的固定策略，不被逐出、不入 `state_dict` |
| lethal search / `lethal:<agent>` | 致死搜索 | 影子对局上推演本回合，找到赢下来的线就照走（T4e.1，evaluation.md） |
| PIMC | 完美信息蒙特卡洛 | 按信念采样对手隐藏信息后在每个样本里搜索（T4e.1 阶段 C） |
| GRPO | 组相对策略优化 | 同一起点采多条样本，以组内相对奖励作优势、无 critic；本仓库只考虑第 1 回合展开（scaling.md T11） |
| DPO | 直接偏好优化 | 用同一输入的优 / 劣样本对直接微调策略；本仓库只考虑求解器线对未达成线（scaling.md T12） |
| `interruption` | 阻断点 / 妨害 | 能在对手回合打断对手的一张卡：场上表侧的诱发即时效果、盖放的陷阱或速攻魔法、手里的手坑、墓地的诱发即时效果，效果须无效或能处理对手的卡（`ygorl.solver.blocking`，solver.md「阻断场面」） |
| blocking board | 阻断场面 | 先攻第 1 回合结束时阻断点多的场面；求解器示范的自动目标（solver.md） |
| piece | 阻断件 | 卡组里能作为求解器目标的场上阻断怪兽（`blocking.Piece`） |

## 组牌与评估

| 标识符 | 中文 | 说明 |
|--------|------|------|
| `environment` | 环境 | 格式 + 卡池 + 禁限表 + 规则 + meta + 版本 |
| `meta` / meta pool | 环境主流卡组 / meta 池 | |
| `deck corpus` | 牌组语料 | 环境下合法的历史卡表（每个类型至多 3 份，含娱乐 / 活动卡组），`artifacts/deck_corpus.json`（`ygorl.data.corpus`） |
| `deck dataset` | 牌组数据集 | masterduelmeta 历史里卡片全部对应的每一份卡表（按计数去重，含不合法的，带合法性标记），学习型组牌模型的训练数据，`out/deck_dataset/<版本>/`（`ygorl.data.deck_dataset`） |
| `game log` | 对局日志 | 训练时每局结束一行的记录（双方牌组、先攻、胜者、回合、更新号、环境戳），`--log-games` → 运行目录 `games.jsonl.gz` |
| `deck tuning` / `tuner` | 调卡组 | 给定一套牌，找单卡替换后对环境 meta 更强的版本（`ygorl.build.tuner`，[tuning.md](tuning.md)） |
| `package` | 引擎包 | 协同图上的连通卡组组件（T5.4） |
| `genotype` | 基因型 | 引擎包份数 + 泛用槽 + 额外卡组（T5.5） |
| generic slot | 泛用槽 | 手坑、解场等不属于引擎包的卡 |
| `surrogate` | 代理模型 | 由牌组特征预测胜率与描述符（T5.7） |
| `arena` | 竞技场评估 | 配对种子对局（与内存 arena 同名，按上下文区分） |
| win rate / Wilson CI | 胜率 / Wilson 区间 | 平局算半胜 |
| matchup matrix | 牌组对局矩阵 | 同一 agent 驾驶双方，牌组对牌组的胜率矩阵，配 Nash 混合与 alpha-rank（evaluation.md，T3.3） |
| agent matchup matrix | 策略对局矩阵 | agent 对 agent、在一批固定牌组配对上（双方轮流先攻）的胜率矩阵；衡量对局强度的尺子，`ygorl strength`，`ygorl.eval.agent_matrix`（evaluation.md，#83 / #85） |
| evolved deck | 进化卡组 | 进化进程（#105）产生、训练中加入牌组池的卡组；状态 `probation`（试用）/ `active`（正式）/ `history`（历史，只作对手卡组） |
| deck-pool manifest | 牌组池清单 | 列出进化卡组的 `ygorl-deck-pool` JSON，训练器每隔几次更新重读（`ygorl.train.selfplay.EvolvedDecks`，training.md §8.3） |
| opening-hand effect | 起手效应 | 同一牌组内，某卡在起手与不在起手的对局胜率差（`ygorl.build.diagnose`，spikes/deck-evolution.md §3） |
| leave-one-out value | 留一值 | 把一份某卡换成空白通常怪兽后，亲本与子代配对胜率差；逐卡边际价值的真值（M2） |
| common random numbers / paired difference | 公共随机数 / 配对差 | 亲本与子代在同一种子、同一对手、先后攻各一局上对局，逐对比较子代 − 亲本（`ygorl.build.tuner.PairedEvaluator`，[tuning.md](tuning.md)） |
| top-two Thompson sampling | 前二 Thompson 采样 | 按后验把对局批次分给「可能最好」与「可能次好」的子代（`ygorl.build.selection.top_two_thompson`，tuning.md「进化评估器」） |
| control variate / opening luck | 控制变量 / 起手手气 | 每局得分减去 critic 读出的起手手气：实际起手的 V 减同一卡组其它洗牌的平均 V；期望不变（`ygorl.build.control`） |
| sequential validation | 序贯复核 | 新种子上分批看、按 α 支出界判定的复核（O'Brien–Fleming），大效应早停（`ygorl.build.selection.sequential_validate`） |
| evolution step / round | 进化步骤 / 一轮进化 | 选亲本 → 诊断 → 子代 → L0 筛 → 前二 Thompson 采样 → 序贯复核 → 通过者以试用状态入池；独立进程，状态可续跑（`ygorl.build.evolve`，`tools/evolve_decks.py`，tuning.md「进化步骤」） |
| deck lineage | 卡组谱系 | 每个被评估子代一条记录：亲本、改动、预测增益、实测配对差与区间、对局数、检查点、环境、是否通过（`lineage.jsonl`） |
| association rule / support / confidence / lift | 关联规则 / 支持度 / 置信度 / 提升度 | 「跑 A（和 B）的卡表也跑 C」：同时含前件与 C 的卡表占比 / 含前件的卡表里含 C 的比例 / 置信度 ÷ P(C)；牌组语料上挖出，只作进化的候选来源（`ygorl.build.rules`，tuning.md「关联规则候选」） |
| child generator | 子代生成器 | 提出子代的来源：`mutation`（`informed_children`：有依据 / 探索）、`crossover`（交叉）、`rules`（关联规则）；记在谱系的 `generator`，报告按它统计通过率 |
| package completion | 引擎包补全 | 给定部分引擎包，返回语料里通常与之同在的缺失成员（`DeckRules.complete`，#113） |
| L0 screen | L0 筛 | 零对局成本的预筛：critic 在公共随机数起手上比较子代与亲本的开局值；M4 通过前默认关闭（`shadow` 只计数） |
| MAP-Elites archive / admission | MAP-Elites 档案 / 入选 | 按五个描述符分格、每格留胜率下界最高的卡组（pyribs）；填入空格、严格胜过格内精英，或取代对手分布已变的过期精英即入选（`ygorl.build.archive`） |
| crossover child / mate | 交叉子代 / 配偶 | 本轮亲本与一个档案精英（配偶，按格子距离偏向远处格子抽取）按卡表交叉、换入配偶不超过一半差异的子代；以亲本为配对基准，谱系 `generator = "crossover"`（`ygorl.build.crossover`，tuning.md「交叉子代」） |
| cell distance | 格子距离 | 两组描述符在 MAP-Elites 网格上各维格序号差之和（只算双方都有值的描述符） |
| descriptor proxy | 描述符代理 | 真值要求解器的描述符的廉价近似：combo 长度 ≈ 检索链深度，卡手率 ≈ 起手没有启动卡的概率（tuning.md「进化步骤」） |
| engine member | 引擎成员 | 卡组里与本卡组另一张卡有协同图边（任一方向、卡组内扇出 ≤ 500）的非泛用卡，检索者、启动卡与被检索者都算；模型没有证据前进化不换下它（`ygorl.build.deck_engine.deck_engine`，tuning.md「引擎感知的候选」，#145） |
| addition pool | 加入池 | 进化子代可换入的卡：泛用卡，或在协同图上与亲本的非泛用卡相连（扇出 ≤ 30）的卡（`deck_engine.addition_pool`） |
| cold-start breadth / screen | 冷启动广度 / 初筛 | 模型对亲本类型证据不足时，一轮生成约 40 个单卡替换、各打一批（25 对），只让第一批最好的几个进前二 Thompson 采样（`ygorl.build.selection.screen`） |
| signal-library warm start | 信号库热启动 | 把已有的非自适应配对数据（M1、M2 留一、首轮谱系的 `learned`、调卡组对比的复核）作为观测先写进卡片价值模型（`ygorl.build.warmstart`）；与策略的 BC 热启动无关 |
| multiplicity-aware stop | 多重性校正的停止 | 前二 Thompson 采样判「有把握」要求领先子代至少 100 对，且 P(Δ > 0) > 1 − 0.05 / 候选数（Bonferroni） |
| winner's curse / re-validation | 胜者诅咒 / 重验 | 只接受复核里显著的改动，复核的差因此平均偏高；在从未用过的新种子上重打被接受的改动来量它（`tools/deckevo_eval_compare.py --revalidate`） |
| fractional factorial design / variant | 部分析因设计 / 变体 | 把 k 处相容改动按 2^(k−p) 个两水平组合（变体）同时评估，所有变体打同样的公共随机数对局，一次回归估出各改动的效应（`ygorl.build.factorial`，tuning.md「析因评估」，#152） |
| main effect / interaction | 主效应 / 交互效应 | 主效应：有这处改动的平均得分 − 没有的（对其它改动取值平均）；交互 AB：B 在与不在时 A 的效应之差的一半 |
| alias / resolution | 别名 / 分辨度 | 部分析因里列完全相同（带符号）、分不开的效应互为别名（如 AB = CD）；分辨度 = 定义关系里最短词的长度，IV 表示主效应不与两两交互混杂 |
| legality repair (factorial) | 合法性修复（析因） | 单独合法、合起来不合法的变体留下最少的改动使其合法，估计用实际打的水平；被去掉的记在 `repairs` |
