# Spike：终局胜负之外的训练信号（奖励塑形、方差、信用分配）

日期：2026-09-26。输入：[scaling.md](../scaling.md) §3 的先前项目表与 §4.2 的 T6 / T7（这里不重复），设计 C3（[02-challenges.md](../design/02-challenges.md)）、I1 / I2 / I7（[03-play-policy.md](../design/03-play-policy.md)），以及当天的实测：

- **平台期是随机游走**：热启动后再跑 2,400+ 次更新（每次 2,048 行，约 12 局完整对局），策略对局矩阵上强度不涨；检查点之间首选动作一致率只有 59–72%，策略一直在动。
- **梯度信噪比低**：平台期检查点上，两份独立 2,048 行批的策略梯度余弦均值 +0.10，4 批合并对 4 批合并 +0.29；单批噪声的平方范数约为真梯度的 9 倍。早期检查点是 +0.16 / +0.52。
- **大批有效，大网络无效**：8 倍批（16k 行，约 124 局 / 次）、总数据相同，一个种子明显超过平台（平均胜率 0.596 对 0.53），另一个略好；宽 128 × 2 层在同数据量下没用。
- **牌组对局方差大**：每局从 837 套语料牌组里随机抽两套。
- **对局长**：对 Greedy 的中位数 9–10 回合（Greedy 对 Greedy 同样长，是资源消耗战）；采样时有 13% 的机会跳过能赢的战斗阶段；γ = 1 不奖励早赢。
- 宽 128 的一次运行，新 critic 学得慢，策略严重漂移（熵上升、27 回合的对局）。

## 结论

瓶颈是**每次更新的独立胜负样本太少、优势估计噪声太大**，不是奖励「太稀疏以致学不到」。所以先做降方差、不改目标的方法；塑形只在势函数形式下使用，而这种形式在我们的 actor-critic 里等价于 critic 的参数化。排序：

| # | 建议 | 对准的瓶颈 | 预期 | 风险 | 设计 |
|---|------|-----------|------|------|------|
| R1 | **每次更新 ≥ 100 局 + 按牌组配对分层**：16k 行 / 次（1–2 轮）设为默认；训练端也用「四重配对」（同一对牌组，双方交换卡组 × 先后攻各一局，洗牌按槽位同种子），让每批的牌组对局均衡，并得到同一发牌下的组内基线 | 梯度噪声、牌组方差 | 已有单种子 +0.066；分层后再降方差 | 每秒更新次数下降；学习率 / KL 阈值要重调 | I1 改批大小；C3 不变 |
| R2 | **更低方差的优势**：扫 λ ∈ {0.95, 0.8, 0.5}，并试 `vrpo_mode="critic"` 与「全动作」策略梯度 Σ_a π(a\|s)·(Q(s,a) − V̄(s))·∇log π（Q 头已对全部候选打分） | 梯度噪声 | 去掉动作采样与后续转移的噪声；用现有梯度余弦工具先测信噪比 | critic 偏差直接进策略；critic 差时（如新宽度）会更糟 | I2 范围内；在 training.md 记录 |
| R3 | **分支推演给 critic 做低方差目标**（TRPO「vine」+ 公共随机数 / 专家迭代的简化版）：在少数关键决策（进入战斗阶段、进入结束阶段、致死搜索报「有线」）用快照分叉，每个候选用当前策略、同一组随机数推演到回合结束 + critic 自举，得到 Q̂ 作为 Q 头的额外目标 | 长时程信用分配、13% 跳过战斗 | 在关键点上的 Q 目标几乎无噪声 | 算力；推演用的是真实隐藏信息，只能进特权 critic，不能直接蒸馏进 actor（策略融合），actor 目标要等 PIMC | I2 扩展、#62 阶段 B；C3 不变 |
| R4 | **势函数只作 critic 的残差参数化 + 辅助预测头**：Q(s,a) = Φ(s) + q_θ(s,a)，Φ 取冻结的旧 critic 或手写的 LP / 资源差；信念头与「本回合末 LP 变化」「下回合是否被斩杀」等辅助损失接到共享主干 | 新 critic 学得慢导致的漂移 | 换宽度 / 卡片视图 / 新牌组时 critic 起步快；平台期本身收益小 | 手写 Φ 若在终局不为 0 或非反对称会改变目标 | **C3 措辞要改**，见 §1 |
| R5 | **中局开局（I7 / T9）+ 小幅每回合折扣作诊断** | 长时程、样本效率 | 更多样本落在后期决策；短局提高每批局数 | 状态分布偏移（DAGS 的标志位可缓解）；γ_turn < 1 改变目标 | 中局开局属 I7；T6 仍与 C3 冲突，不建议改默认 |

不建议：非势函数的密集奖励（cjiang 式 Δ卡数、「妨碍对方展开」奖励）、RUDDER 类回报重分配（我们的 Q critic 已在做同一件事）、UPGO（T8，另议）。

设计文档要改的只有两处：C3 的「Gin Rummy：shaping 无益」改为更准确的表述（见 §1），I1 的批大小。

## 1. 势函数塑形（T7）

**机制与条件**。Ng、Harada、Russell（1999）：F(s, s′) = γΦ(s′) − Φ(s) 是保持最优策略不变的充要形式。有限局、γ = 1 时，整局累加得 Φ(s_T) − Φ(s_0)，只有**终局状态的 Φ 为 0**（或所有终局取同一常数）才不改变胜负的比较（Grześ 2017 专门指出了这一点）。Devlin & Kudenko（2011）证明多智能体随机博弈里势函数塑形不改变 Nash 均衡；2012 年的动态势函数 F = γΦ(s′, t′) − Φ(s, t) 在势函数随训练变化时仍保持这一保证。对我们：

- 我们的递推已用 c_t = γ(1 − done_t)σ_t，写成 r′_t = r_t + c_t·Φ(s_{t+1}) − Φ(s_t) 时，终局行 c_t = 0，等于自动取 Φ(终局) = 0；**截断行**要保留 Φ（截断靠 critic 自举，不是终局）。
- 零和：Φ 必须对座位反对称（Φ_p = −Φ_{1−p}，如 LP 差、卡差），否则博弈不再零和。

**与 critic 的关系**。Wiewiora（2003）：势函数塑形与「用 Φ 初始化 Q 值」逐步等价。在带自举 critic 的 PPO / VRPO 里也一样：塑形后 critic 学的是 V − Φ，δ 不变；λ < 1 时只改变偏差—方差的分配。所以它**不会给平台期新增信息**，作用是让 critic 起步更好。R4 直接把它写成残差 critic，比改奖励更清楚，也不碰 C3 的「主奖励只用胜负」。

**别人的结果**：

- **Gin Rummy 2026**（Kelidari、Haghi、Salmani，arXiv 2607.06854）：摘要原文是「Short-term and longer-term reward shaping … were each unhelpful」。失败的是**按专家后 L 步一致度打分的密集奖励**，不是势函数形式：「a horizon of two showed no real learning and a horizon of five stopped improving」，智能体变得短视、刷即时分。而他们最终的「well-aimed reward」恰恰带一个**小的 deadwood 减少奖励**（系数与是否势函数形式，论文未给）。所以 C3 应改为：「非势函数的密集奖励有害（Gin Rummy 2026）；势函数形式等价于 critic 初始化，只作 critic 参数化使用」。
- **PerfectDou**（NeurIPS 2022）：节点奖励 r_t ∝ Adv_t − Adv_{t−1}，Adv 是双方「最少几手出完」之差，即势函数形式；去掉它（RewardlessDou）对 DouZero 的胜率 WP 相同（0.738 对 0.732），主要损失的是分差 ADP（0.490 对 1.270）。附录说它「extremely useful for accelerating convergence」，最后阶段去掉它再微调。
- **VRPO**（Fan & Farina 2026，arXiv 2605.19235）：只用终局奖励，约 40% 的步数超过 PerfectDou。这是我们 I2 的依据，也说明塑形不是必需品。
- **IRumAI**（arXiv 2606.21975）：Φ = −deadwood/130 的势函数塑形，对 4 个对手 +1.7 至 +4.9 个百分点，**单种子**。
- Suphx 的「全局奖励预测」（GRU 从局级信息预测整场终局奖励，逐局奖励 = Φ(x^k) − Φ(x^{k−1})）也是学出来的势函数。

候选 Φ：LP 差（最便宜，但 LP 在游戏王里和胜率关系弱，资源消耗战里常常误导）、资源差（手牌 + 场上 + 可用额外卡组）、冻结的旧 critic（最准）、信念特征（对手手坑概率等，只能做 actor 可见的部分）。建议只用冻结 critic 与 LP / 资源差两种做 R4 的对照。

## 2. 快赢压力（T6）

- **ygo-agent `greedy_reward`**（`ygoenv/ygoenv/ygopro/ygopro.h` 约 2247 行，@ `dbf5142`）：座位 0 获胜时幅度按回合 16 / 8 / 4 / 2（≤ 1 / 3 / 5 / 7 回合），之后 0.5 + 1/(t − 7)；座位 1 获胜的档位错一档（8 / 4 / 2，之后 0.5 + 1/(t − 5)）；败方取相反数，零和。`scripts/cleanba.py` 训练默认 `greedy_reward=False`，**评估时强制为 True**（`args.greedy_reward if not eval else True`）；γ = 1、λ = 0.95、UPGO 开。即他们主要拿它当评估口径，不是训练奖励。
- **OpenAI Five**（Berner et al. 2019，arXiv 1912.06680）：视界 H = T/(1 − γ)，训练中从 180 s（γ = 0.9993）逐步加到 360 s；图 6：已训练好的智能体改用更长视界继续训练，胜率一直在提高，直到试过的最长 6–12 分钟。塑形奖励（表 6）「loosely after potential-based shaping … though the guarantees therein do not apply」，减去对方平均奖励使其零和，并乘 0.6^(T/10 分钟) 按游戏时间衰减。
- **Metamon**（`metamon/interface.py` 的 `DefaultShapedReward`，@ `0a00a75`）：±100 胜负 + 伤害 / 回血 / 状态 / 击倒各 0.5–1 的逐步项（差分形式，但不是严格势函数）；另有 `AggressiveShapedReward`（胜 +200 / 负 0），注释说是为了去掉「clinging to lost positions」。

对我们：折扣只能**按回合**计（按决策计会惩罚 30–80 步的展开）。γ_turn = 0.99 意味着愿意用约 1% 的胜率换早一回合，并且让败方倾向拖延；13% 跳过战斗的本质是「跳过的代价在噪声里看不出」，γ = 1 本来就会惩罚它（对手多一回合找解），所以 R2 / R3 是更直接的解法。T6 只建议作诊断臂（γ_turn = 0.995），不改默认。实现在 c_t 里乘 γ^{Δ回合}，需要 rollout 带回合号。

## 3. 「打断对方展开」类中间奖励

可验证的定义只有两种：(a) 对方回合结束时的资源差 / 场面评分的变化——这就是 §1 的势函数，只要写成 Φ 的差分就不会被钻空子；(b) 事件计数（成功无效化、手坑命中）——非势函数，会诱导「为发动而发动」、在无关时机打出手坑。结论：不单独做，归入 R4 的 Φ 候选；「对方第 1 回合的展开达成度」可借用 `opening_report` 的场面评分当 Φ 的一项。

## 4. 辅助任务

UNREAL（Jaderberg et al. 2016）：奖励预测、价值回放与像素控制等辅助任务显著提高 Labyrinth 上的数据效率（具体倍数未复核）。OpenAI Five 的胜率等辅助头以很小的权重回传到主干，标签用段末预测自举。对我们：信念头已训练好（[belief-heads.md](../belief-heads.md)），把它们的损失接到共享主干、再加「本回合末 LP 变化」「本回合是否存在致死线」（由致死搜索标注）两个便宜标签。它们不降低策略梯度噪声，但能让 critic 与主干更快可用——正对「新 critic 学得慢 → 漂移」。Hearthstone 的 SkyNet / BEPAL：**未找到一手来源，不引用**。

## 5. 信用分配与回报重分配

- RUDDER（Arjona-Medina et al. 2019）用 LSTM 预测终局回报，把预测值的差分重分配成逐步奖励——本质是学出来的势函数；我们的 Q critic + Expected-SARSA(λ) 已经在做同一件事，不另做。
- Hindsight Credit Assignment（Harutyunyan et al. 2019）：需要额外的后见分布模型，大规模对局中未见成功案例，不做。
- **λ 与 critic 的作用**（R2）：VRPO 的 `return` 模式 A_t = [Q − V̄] + Σ(λc)^k δ^Q，λ = 0.95 时有效窗口约 20 行，仍含抽卡 / 对方隐藏信息带来的转移噪声；λ 越小、越依赖 critic。`critic` 模式与全动作梯度（Mean Actor-Critic，Allen et al. 2017）把动作采样噪声也去掉。代价是 critic 偏差，所以 R2 必须与 critic 质量诊断（Q 头的解释方差）一起看。

## 6. 胜负噪声的方差缩减

- **重复发牌（duplicate）**：桥牌 / 扑克评估的做法，同一发牌双方换位各打一次。我们的 arena 已按槽位派生种子做到这一点（[evaluation.md](../evaluation.md)），训练端还没有。R1 把它搬到收集：同一配对四局共享洗牌种子，局级基线取组内均值（与 GRPO 的组基线同理），并保证每批牌组配对均衡。
- **AIVAT**（Burch et al. 2018）：对机会节点与已知策略的动作都加控制变量，无偏；人机扑克赛标准差降 85%、所需局数少 44 倍。它是**评估**方法，要求双方策略已知——自博弈矩阵正好满足，可用于缩短 `ygorl strength` 的局数（后续工作，不在前五）。
- **公共随机数**：TRPO 的 vine 流程在同一状态对不同动作用相同随机数推演，以降低 Q 值差的方差；`branch.try_all` 已实现同样的约定（[branching.md](../branching.md)），R3 直接复用。
- **按对局条件化 critic**：优势 = G − V̄，牌组方差只通过 critic 的误差进入。先诊断：第一个决策点的 V̄ 与牌组矩阵里该配对的经验胜率的相关性；若低，把双方牌组的语料 ID / 构成摘要作为特权输入给 critic（只进 critic，不违反 I9）。
- 优势归一化：现在对整个 rollout 做 `standard`；R1 加大批后这一项自动更稳。

## 7. 搜索当老师

AlphaZero / ExIt（Anthony et al. 2017）把搜索结果作为策略目标、把搜索值作为价值目标，比只用终局胜负的 REINFORCE 密集得多。完美信息推演直接蒸馏给 actor 会有策略融合问题（Long et al. 2010 分析过 PIMC 为何在某些游戏中仍有效）；ReBeL（Brown et al. 2020）与 Student of Games（Schmid et al. 2023）在公共信念状态上搜索来解决它，但工程量远超当前阶段。可行的路线：R3 先把分支推演的 Q̂ 只给特权 critic（它本来就看隐藏信息）；之后用信念头采样对手手牌 / 盖卡做 PIMC，才给 actor 做策略目标（#62 阶段 C）。致死搜索对强度贡献小（scaling.md §4.3），但作为「关键决策点检测器」对 R3 很有用。

## 8. 课程与中局开局

DAGS（Lanier、Monette、Baldi、Fox 2026，arXiv 2605.14379）：从离线数据的中间状态开局可加速自博弈探索，但会使均衡有偏，用多任务观测标志位缓解——即设计 I7 的「增广开局标志位」。Go-Explore（Ecoffet et al. 2021）的「先回到状态再探索」需要可恢复的模拟器，我们有快照。R5 的简化版：从最近几次更新自博弈对局的快照中，按「距终局 ≤ 3 回合」抽开局点，占收集的 20–30%，并加 `augmented_start`。第 1 回合 GRPO / DPO（T11 / T12）已在 scaling.md，属 I4，与本文不冲突。

## 9. 其它卡牌 / 不完全信息游戏的做法

| 项目 | 奖励 | 与本文相关的做法 |
|------|------|------------------|
| DouZero（Zha et al. 2021） | 只用终局 WP / ADP，蒙特卡洛回报 | 靠约 10¹¹ 帧的规模 |
| PerfectDou | 终局 ADP + 势函数形式的节点奖励，最后阶段去掉 | 特权 critic；节点奖励主要提高 ADP |
| Suphx | 学出来的全局奖励预测（势函数差分） | oracle guiding：训练初期给完美信息特征，逐步 dropout 到 0 |
| OpenAI Five | 零和、按时间衰减的塑形 + 长视界 | 胜率辅助头 |
| Metamon | ±100 胜负 + 轻塑形 | 离线 RL + 自博弈 |
| ygo-agent | 终局 ±1（快赢奖励主要用于评估） | GAE + UPGO、约 32k 步 / 次 |
| VRPO | 只用终局奖励 | Q-boosting，就是我们的 I2 |

共同点：成功的项目要么只用终局奖励 + 大规模 / 低方差估计，要么用**势函数形式**的塑形并在后期去掉；非势函数的密集奖励都伴随「短视」或「刷分」的报告（Gin Rummy、Metamon 的注释）。

## 验证实验

统一口径：从同一个热启动检查点出发；总数据量与 16k 实验相同（约 490 万行）；每臂 3 个种子；强度用策略对局矩阵（`ygorl strength`，语料池 100 个牌组配对 × 4 局 = 每格 400 局，约 ±0.05），各臂最终检查点与基线检查点同矩阵、报 Nash / alpha-rank 与平均胜率；同时记录梯度余弦（2 × 2,048 行与 4 × 4 合并两档）、首选动作一致率、对局回合数、跳过可赢战斗阶段的比例。

| 建议 | 对照 | 通过标准 |
|------|------|----------|
| R1 | 2,048 行 × 4 轮（现默认） / 16k 行 × 2 轮随机抽牌组 / 16k 行 × 2 轮 + 四重配对分层 | 分层臂 3 个种子平均胜率都 ≥ 随机抽牌组臂，且梯度余弦更高 |
| R2 | λ = 0.95 `return`（默认）/ λ = 0.8 / λ = 0.5 / `critic` 模式 / 全动作梯度；先在固定检查点上只测梯度余弦（便宜，1 小时内），信噪比最高的两臂再训练 | 余弦提升且训练臂强度不低于默认；Q 头解释方差不下降 |
| R3 | R1 最佳配置 ± vine Q 目标（关键点占决策的 ≤ 5%，每候选 4 次公共随机数推演到回合结束） | 强度提升、跳过战斗比例下降；记录额外 CPU 开销 |
| R4 | 宽 128 新 critic（已知会漂移）：原样 / 残差 critic（Φ = 冻结旧 critic）/ 残差 + 辅助头 | 不再出现熵上升与 27 回合对局；强度不低于宽 64 基线 |
| R5 | R1 最佳配置 ± 中局开局 20% ± γ_turn = 0.995 | 强度（从空场开局的矩阵）提升；γ_turn 臂只看回合数与强度的取舍，不作默认 |

## 来源

- Ng, Harada, Russell. *Policy invariance under reward transformations: Theory and application to reward shaping*. ICML 1999. https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Wiewiora. *Potential-Based Shaping and Q-Value Initialization are Equivalent*. JAIR 19, 2003. https://arxiv.org/abs/1106.5267
- Grześ. *Reward Shaping in Episodic Reinforcement Learning*. AAMAS 2017. https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf
- Devlin, Kudenko. *Theoretical considerations of potential-based reward shaping for multi-agent systems*. AAMAS 2011；*Dynamic potential-based reward shaping*. AAMAS 2012. https://dl.acm.org/doi/10.5555/2030470.2030503 ，https://dl.acm.org/doi/10.5555/2343576.2343638
- Kelidari, Haghi, Salmani. *A Gold-Standard Study of What Makes a Lightweight Game-Playing Agent Strong*（Gin Rummy）. 2026. https://arxiv.org/abs/2607.06854
- Yang et al. *PerfectDou: Dominating DouDizhu with Perfect Information Distillation*. NeurIPS 2022. https://arxiv.org/abs/2203.16406 （式 4–5、表 4、附录 D.1）
- Fan, Farina. *GAE Falls Short in Imperfect-Information Self-Play Reinforcement Learning*（VRPO）. 2026. https://arxiv.org/abs/2605.19235
- *IRumAI: Reinforcement Learning for Indian Rummy*. 2026. https://arxiv.org/abs/2606.21975
- Li et al. *Suphx: Mastering Mahjong with Deep Reinforcement Learning*. 2020. https://arxiv.org/abs/2003.13590
- Berner et al. *Dota 2 with Large Scale Deep Reinforcement Learning*（OpenAI Five）. 2019. https://arxiv.org/abs/1912.06680 （式 3、图 6、附录 G、表 6）
- Grigsby et al. *Human-Level Competitive Pokémon via Scalable Offline RL with Transformers*（Metamon）. RLC 2025. https://arxiv.org/abs/2504.04395 ；代码 github.com/UT-Austin-RPL/metamon `metamon/interface.py`（`DefaultShapedReward` / `AggressiveShapedReward`，@ `0a00a75`）
- sbl1996/ygo-agent：`ygoenv/ygoenv/ygopro/ygopro.h`（`greedy_reward_`）、`scripts/cleanba.py`（@ `dbf5142`）. https://github.com/sbl1996/ygo-agent
- Schulman et al. *Trust Region Policy Optimization*. ICML 2015（vine 流程与公共随机数，§5）. https://arxiv.org/abs/1502.05477
- Burch, Schmid, Moravčík, Morrill, Bowling. *AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games*. AAAI 2018. https://arxiv.org/abs/1612.06915
- Lanier, Monette, Baldi, Fox. *Data-Augmented Game Starts for Accelerating Self-Play Exploration in Imperfect Information Games*（DAGS）. 2026. https://arxiv.org/abs/2605.14379
- 以下只按摘要 / 已知内容引用，本 spike 未逐条复核细节：Jaderberg et al. *Reinforcement Learning with Unsupervised Auxiliary Tasks*（UNREAL）2016, https://arxiv.org/abs/1611.05397 ；Arjona-Medina et al. *RUDDER* NeurIPS 2019, https://arxiv.org/abs/1806.07857 ；Harutyunyan et al. *Hindsight Credit Assignment* NeurIPS 2019, https://arxiv.org/abs/1912.02503 ；Allen et al. *Mean Actor Critic* 2017, https://arxiv.org/abs/1709.00503 ；Anthony, Tian, Barber. *Thinking Fast and Slow with Deep Learning and Tree Search*（ExIt）NeurIPS 2017, https://arxiv.org/abs/1705.08439 ；Silver et al. AlphaZero, Science 2018；Brown et al. *ReBeL* NeurIPS 2020, https://arxiv.org/abs/2007.13544 ；Schmid et al. *Student of Games* Science Advances 2023, https://arxiv.org/abs/2112.03178 ；Long et al. *Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search* AAAI 2010；Ecoffet et al. *First return, then explore*（Go-Explore）Nature 2021, https://arxiv.org/abs/2004.12919 ；Zha et al. *DouZero* ICML 2021, https://arxiv.org/abs/2106.06135 ；Shao et al. GRPO（DeepSeekMath）2024, https://arxiv.org/abs/2402.03300 。
