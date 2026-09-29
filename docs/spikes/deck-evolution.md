# Spike：牌组池与策略共同进化——有依据、省对局的组牌迭代（#105）

日期：2026-09-27。输入：[#105](https://github.com/n0tevi1/ygorl/issues/105) 的 spec；设计 [05-deck-building.md](../design/05-deck-building.md)；
[genotype.md](../genotype.md)（引擎包 + 泛用槽 + 额外卡组，算子 `mutate` / `crossover` / `repair`）、[surrogate.md](../surrogate.md)（ridge 集成，留出 MAE 7.25 pp，约 100 个标签进入 10 pp）、
[funnel.md](../funnel.md)（每套 120 求解器进程秒）、[tuning.md](../tuning.md)（调卡组 MVP）、[evaluation.md](../evaluation.md)（批量路径每格 200 局约 8.6 局/秒）、
[data.md](../data.md)（md-2026-09 牌组语料 1,021 套 / 397 个类型，训练划分 837 套）、[branching.md](../branching.md)、[encoding.md](../encoding.md)。

## 结论

**效率的瓶颈不在「生成候选」，而在「确认一个小改动」。** 一张卡的改动通常只值 1–3 pp（tuning.md）。配对差的每对标准差近似为 σ_d ≈ 0.5·√(1 − ρ)，ρ 是同一种子下亲本局与子代局胜负的相关系数（p ≈ 0.5，一对 = 先后攻各一局；M1 实测 ρ = 0.84、σ_d = 0.212，见文末「测量结果」）。要在 95% 双侧、80% 功效下确认 Δ，需要 n ≈ (2.8·σ_d / Δ)² 对：

| ρ | σ_d | Δ = 2 pp 所需对数 | Δ = 5 pp 所需对数 | 调卡组 MVP 的 200 对复核在 Δ = 2 pp 时的功效 |
|---|-----|------|------|------|
| 0.5 | 0.354 | 2,450 | 392 | 0.12 |
| 0.8 | 0.224 | 980 | 157 | 0.24 |
| 0.9 | 0.158 | 490 | 78 | 0.43 |

所需对局随 1/Δ² 与 (1 − ρ) 缩放。所以高效的做法只有三条：**(a) 让每个被评估的子代的真实 Δ 更大**：用数据选改动，并把几处同方向的改动打包进同一个子代；**(b) 提高 ρ、降低方差**：公共随机数加上用 critic 做控制变量；**(c) 预算只花在后验上可能最好的子代上**。盲目的随机单卡替换三条都做不到。

**推荐的循环**（进化步骤是独立进程，按 #105 只读写牌组池清单与档案）：

1. **选亲本**：按 #105 的规则（适应度、档案空格、距上次变异的时间）。
2. **诊断**（每个亲本约 200 对 = 400 局，同时充当公共随机数下的亲本基线，不另打）：当前策略驾驶亲本，对手按「Nash 权重 + 环境 meta 份额」从牌组池中抽取。逐卡记录以下统计：
   - **起手效应**：同一牌组内，某卡在起手与不在起手的胜率差。起手由洗牌随机决定，所以这是一个自然实验，见 §3。
   - 抽到 / 使用 / **死卡率**：进过手牌，但到离开手牌或终局为止从未出现在合法动作里。
   - **critic 反事实值**：在第一个决策点把手牌里的这张卡换成卡组里的另一张卡，比较 V̄。
   
   训练回放里全池累积的同类统计作为先验，经层级收缩（卡片主效应 + 卡片 × 类型）合并进来。
3. **生成子代（有依据）**：
   - **移出分** = 低边际价值（起手效应与 critic 反事实）+ 高死卡率。
   - **加入分** = 这张卡在同类型或相近牌组里的收缩后起手效应 + 协同图边 + 代理模型的预测增益。
   - 取 (移出, 加入) 组合的前若干名，组成 **6 个子代，每个 1–3 处同向改动**；再加 **2 个探索子代**：现有算子的随机变异，或经漏斗预筛的换包 / 加包。
   - 所有子代先过合法性检查与 **L0 筛**：critic 起手值，每个子代 256 个公共随机数起手，比较第一个决策点的平均 V̄。只算前向，不打对局。
4. **分配对局**：8 个子代各打 25 对起步，之后用 **前二 Thompson 采样**（top-two Thompson，TTTS）按 25 对一批分配预算。后验以第 3 步的预测 Δ 为先验均值，结果用 critic 控制变量降方差（§4）。某个子代满足 P(Δ > 0) > 0.95，或 600 对用完，就停止。
5. **复核**：在新种子上做**序贯**检验，每 100 对看一次（α 支出界），上限 1,000 对。区间下限 > 0 就以「试用」状态入池；明显为负时提前停。
6. **档案与多样性**：pyribs MAP-Elites，用设计 05 的 5 个描述符。格内胜出或填入空格即进档案（#105 用户故事 16）；进化卡组在训练配对中占 30%；同格内过于相似的卡组被淘汰。
7. **试用期**：训练 N 次更新后，用第 2 步同样的诊断协议复评。复评同时刷新逐卡信号，所以不是额外开销。不及格就淘汰。
8. **回流**：每个真实配对差都写进代理模型，也写进一张「信号 → 实测 Δ」校准表，在线调整第 3 步各信号的权重。哪个信号不能预测实测 Δ，就停用哪个。

**每接受一次改动的对局数（估计，前提是 §3 的信号有效，需由文末的测量证实）：**

| | 盲目（调卡组 MVP 默认参数） | 推荐循环 |
|---|---|---|
| 候选 | 64 个随机单卡替换 | 6 个有依据的子代（1–3 处改动）+ 2 个探索子代 |
| 候选生成 | 0 局 | 诊断 400 局（兼作亲本基线）+ 前向计算 |
| 搜索 | 逐轮淘汰 25 → 800 对，按 `successive_halving` 的代码算：候选 6,000 对 + 亲本 800 对 = **13,600 局** | 约 400–600 对 = **800–1,200 局** |
| 复核 | 4 套 × 200 对 = **1,600 局** | 序贯检验，平均约 600 对 = **约 1,200 局**（上限 2,000 局） |
| 每轮合计 | **约 15,200 局**（批量路径约 30 分钟） | **约 2,400–3,200 局** |
| 每轮通过的概率 | 即使 64 个候选里真有一个 +2 pp，ρ = 0.8 时复核功效也只有 0.24 | 打包后的改动若有 +4–5 pp，600 对时功效 > 0.9 |
| **每接受一次** | **≥ 6 万局** | **约 3–6 千局**（按 50–80% 的轮次里「最好的子代真的变强」计算） |

即便信号完全无效，推荐循环也只是退化成「8 个随机子代 + TTTS」，每轮仍比盲目基线便宜约 5 倍。只是那时它能接受的只有大改动。

## 1. 进化与质量多样性（QD）组牌

- **García-Sánchez 等 2016（CIG）**：用 µGP 遗传算法组 Hearthstone 卡组（MetaStone 模拟器）。种群 10、50 代，每个 meta 对手卡组打 16 局，适应度按字典序分三级。这是最早的「模拟器打分 + 进化」基线，评估全靠盲目变异。
- **Bhatt 等 2018（FDG）**：用进化策略（ES）组卡。进化出的卡组能打败内置卡组，对未见过的卡组也有一定泛化能力。把第一轮的产物当对手再进化一次，新卡组又能打败它们（论文称「some degree of transitivity」）。这是两阶段对手池的前身。
- **Fontaine 等 2019（GECCO，MESB）**：
  - 变异：随机换 k 张卡，P(k) 按几何分布递减。
  - 适应度：200 局的英雄生命差。
  - 描述符：平均法力值与法力值方差（组卡前即可算出）；滑动边界每 100 个体按分位数重划一次。
  - 实验分三组：对入门卡组；以第一组的精英为对手（elite adversaries）；平衡性分析。
  - 成本：**每组实验 10,000 次评估 × 200 局 = 200 万局**，在 500 个 CPU 节点上跑（代码 `tehqin/EvoSabber`）。
- **Zhang、Fontaine、Hoover、Nikolaidis 2022（GECCO，DSA-ME）**：
  - 结构：外循环打真实对局；内循环在 MLP 代理模型 [128, 32, 16] 上跑一整轮 MAP-Elites（10⁶ 次迭代）。代理的输入是份数向量（bag-of-cards），代理档案里的精英拿去真实评估，结果回流训练代理。
  - 设置与成本：Rogue 职业、180 张候选卡；预算 N = 10,000 次评估 × 200 局；200 个 Xeon CPU 上每次试验 10–12 小时。
  - 结果：QD-score 338.5，无代理的 MAP-Elites 为 136.9；最高胜率 98.4%，对 90.6%。
  - 另两点：离线训练的代理（10,000 个随机卡组）明显差于在线训练；加入辅助预测目标反而略降 QD-score。
  - 代码 `icaros-usc/EvoStone2`。
  
  这条管线我们已有：surrogate.md 的 `select` 就是它的采集步。它提高的是**每次评估的命中率**，每次评估仍要 200 局。
- **CMA-ME / CMA-MAE（Fontaine & Nikolaidis，GECCO 2023）**：解决 CMA-ME 过早放弃目标、在平坦目标上探索不动的问题。它在连续空间上工作，要经 `from_vector` 修复才能用于我们的离散基因型，适合做包份数这类连续比例的辅助 emitter，不做主力。
- **Dominated Novelty Search（Bahlous-Boldi 等，GECCO 2025）**：竞争适应度 = 到 k 个**更优**个体的平均描述符距离；没有更优个体的记为 +∞，必然保留。它不需要网格边界，可作为档案之外的局部竞争方式。#105 把它放在范围外，这里同意。

**对我们的启示**：这些工作的变异都是盲目的，靠几百万局的规模换覆盖面；我们的预算小 3 个数量级，QD 只该承担**保留多样性**（档案），改进的来源要换成 §3 的有依据变异。它们的评估也都用固定的启发式 AI 驾驶，没有「评估器还没学会这套牌」的问题。我们有这个问题，所以需要试用期（§5）。

## 2. 推荐与学习式组牌

- **Q-DeckRec（Chen 等，CIG 2018）**：
  - 建模：把「对给定对手组卡」建模为 MDP。状态 = 当前卡组 + 对手卡组；动作 = 换一张卡或不改；每回合 D = 15 步；奖励取胜率的指数放大形式；Q 函数用 MLP 近似（1,000 个 ReLU）。
  - 评估：每次在 MetaStone 上打 300 局，约 5 秒，胜率标准差约 5%。
  - 成本：训练 3 天、调用 6.2 万次评估（约 4 千个 episode）后，达到 GA 25 分钟搜索的水平（胜率 0.93 对 0.94，GA 为 315 次评估、15.9 CPU 小时）；此后单个实例只需 9.6 CPU 秒。
  
  启示：**把对局成本挪到训练期，摊到多个实例上**。我们有 837 套语料，是天然的多实例，但 6.2 万次评估 × 数百局对我们太贵。更划算的是用已有的 critic 与训练回放直接当「学好的评估器」（§3）。
- **草稿（draft）智能体**：
  - Vieira、Tavares、Chaimowicz 2020（SBGames，LOCM）：用 RL 学选卡，卡片用特征表示，因此能处理没见过的卡。
  - Kowalski & Miernik 2020（CEC）：用「活跃基因」的进化算法做 arena 组卡。
  - Ward 等 2020（Draftsim）：约 10 万份人类草稿，神经网络与人类的选卡一致率 48.7%。
  
  这些都是逐张选卡，与我们的「构筑 + 调卡」不同，可借用的是特征化的卡片表示。
- **卡片嵌入**（card2vec、decksage）与 DraftFM（2026）：设计 05 已列，本 spike 未复核。共现嵌入只衡量「像不像 meta」，不衡量「对这套牌有没有用」，只作加入分的弱先验。
- **游戏王**：未找到基于对局结果的游戏王组牌论文（检索到的多是卡图识别与嵌入项目）。

## 3. 用对局数据指导变异（重点）

**17lands 的指标**：GP WR 是卡组里有这张卡时的胜率；OH WR 是这张卡在起手时的胜率；GD WR 是之后才抽到时的胜率；GIH WR 是两者合计，即「这张卡在任一时刻进过手」的胜率（Sierkovitz，17lands blog 2021）。IWD 通常说明为 GIH WR 减去「没见到该卡的对局」的胜率。17lands 的定义页由 JS 渲染，未能直接读取，**IWD 的逐字定义未核对**。ALSA 只与轮抽有关，对我们无用。

已知偏差：
- **牌组强度混杂**：17lands blog 明说，强卡组里的平庸卡与弱卡组里的强卡，GP WR 会很接近。
- **对局长度混杂**：局越长见到的卡越多，GIH WR 偏向控制型卡组。此条来自二手说明，未在一手来源核对。游戏王里还有**检索**：被检索的卡「进过手」与展开成功是同一件事，因果方向是反的。
- **修正方法**：mtgds（2022）用 logistic 回归，把玩家胜率、先后攻、双方调度（mulligan）作为控制变量，加上逐卡强度项与成对协同项。预测与结果的相关从 0.15 提到 0.24。

**我们能做得更干净，原因有三：**

1. **组内比较消掉牌组强度混杂**：只在同一套牌、同一对手分布、同一策略下，比较「有 / 无」某卡的对局。
2. **起手是随机分配的**：主机洗牌决定哪 5 张进起手，所以组内「在起手」与「不在起手」的胜率差是**因果效应**：这张卡在起手，与换成这套牌里任意一张别的卡在起手相比。它不受对局长度与检索的影响，比 GIH WR 干净。代价是只有起手的 5 张参与，每张卡约 1/8 的局在处理组。200 对诊断局下，单卡效应的标准误约 ±8 pp。所以单局统计只用来排序，靠的是跨牌组的层级收缩：同一张泛用卡（手坑、解场卡）在上百套牌里都出现。
3. **critic 反事实值不花对局**：在第一个决策点构造「起手换一张」的局面，用特权 critic 比较 V̄。开局用 funnel.md 的「起手在顶上」设定。encoding.md 规定 actor 表含己方卡组构成，所以 critic 能看到卡组构成，也能直接给**改后卡组**打分：对同一组公共随机数起手取 V̄ 的均值，就是第 1 步的 L0 筛。风险有两条：一是 critic 对从没进过训练的卡外推失真；二是 V̄ 只反映开局，看不到后期抽卡的价值。

补充的逐卡信号：
- **死卡率**：进手后始终不在合法动作里。它区分「卡弱」与「策略不会用」：合法但从不选，是策略问题；从不合法，是卡组结构问题。
- **使用后 Q − V̄**：该卡相关动作的平均优势。这是训练里已有的量，但混着策略偏差。

分支推演（branching.md）只能改选择，不能中途换卡的身份，所以不用于逐卡估值。**Shapley / 留一法**：对 40 张卡做真实留一，每张要上千局，太贵；我们只把它当作少量卡的**真值标定**（文末 M2），不当生产信号。没找到把 Shapley 用于卡组逐卡归因的一手文献。

## 4. 评估预算分配

- **逐轮淘汰 / Hyperband**（Karnin 等 ICML 2013；Li 等 JMLR 2018）：已在 tuner 里。它不利用先验，淘汰也只看当前均值。
- **F-race**（Birattari 等 GECCO 2002）：用 Friedman 检验逐批淘汰，天然是配对的，与公共随机数一致。
- **前二 Thompson 采样**（Russo，COLT 2016）：面向「找最好的臂」，按后验在最优与次优之间随机分配样本，有最优性保证。它能**直接吃先验**（§3 的预测 Δ），支持批量与序贯停止，所以推荐它替代固定的逐轮淘汰。
- **组合贝叶斯优化**：BOCS（Baptista & Poloczek，ICML 2018：稀疏二阶模型 + Thompson + SDP 采集）；COMBO（Oh 等，NeurIPS 2019：图笛卡尔积 + 扩散核）。它们适合几十维、评估极贵的问题。我们的代理模型（ridge 一阶 + 包聚合）就是 BOCS 的一阶版；二阶协同项等真实标签到数千个再考虑。
- **多保真**：L0 critic 起手值（毫秒级）→ L1 求解器漏斗（只用于换包、加包，每套 120 秒）→ L2 配对短评估（25 对）→ L3 复核。
- **AIVAT**（Burch 等，AAAI 2018）：在机会节点与已知策略的动作上加控制变量，保持无偏，人机扑克赛中标准差降低 85%。我们的简化版：每局得分减去 [V̄(实际起手) − 在同一卡组的 K 个替代起手上的平均 V̄]。起手分布已知（均匀洗牌），所以只要 V̄ 是固定函数就无偏。这相当于把「手气」从配对差里扣掉，提高有效 ρ。由于亲本与子代的卡组不同，控制变量要各自按自己的卡组算。
- **序贯复核**：固定 200 对在小 Δ 下功效太低（见结论的表）。改用 α 支出的分组序贯检验，大效应早停，小效应要么补够样本，要么放弃。

## 5. 牌组与策略的共同进化

- **POET**（Wang、Lehman、Clune、Stanley 2019，arXiv 1901.01753）：环境与智能体成对共同进化。新环境要满足「最低标准」（对现有智能体既不太难也不太易），并定期尝试把智能体迁移到别的环境。对应到我们：进化卡组入池前，要求当前策略驾驶它的胜率在合理区间；太弱的卡组对训练没有信号。
- **PSRO / 双重预言机（double oracle）**（McMahan 等 ICML 2003；Lanctot 等 NeurIPS 2017）：每轮对「元博弈」的 Nash 混合求最佳应对，加进种群。对应到我们：适应度对手按**牌组对局矩阵的 Nash 权重**与 meta 份额混合抽取，防止「打赢上一代」式的循环。种群（档案）只增不删精英，被取代的卡组留作历史对手。
- **AlphaStar 联赛**（Vinyals 等，Nature 2019）：主智能体、主利用者、联赛利用者，配合按胜率加权的优先虚拟自博弈（PFSP）。对应到训练配对：进化卡组的采样权重按 f(当前策略用它时的胜率) 调，而不是均匀。
- **Fontaine 2019 与 Bhatt 2018 的两阶段对手池**：见 §1。

**风险与对策：**
- 评估器偏向它已经会打的卡组（tuning.md 已观察到「偏向不改」）→ 试用期后复评；设计 05 的「难开但强」要求记录学习曲线。
- §3 的信号来自策略自己的对局，会低估「策略还不会用」的卡 → 死卡率拆开「合法未选」与「不合法」；探索子代保留随机改动；对新加入的卡给出不确定性加成。
- 非传递 / 循环 → 用 Nash 加权的对手，并在档案里保留历史精英。
- 牌组池漂向少数强卡组 → 30% 占比上限，外加档案。

## 6. 游戏王相关

- **ygo-agent**（`sbl1996/ygo-agent` @ `dbf5142`）：固定卡组目录 `assets/deck`（31 个 `.ydk`）；`deck="random"` 时每局均匀抽一套（`ygoenv/ygoenv/ygopro/ygopro.h` 约 2969 行）。它没有组牌或调卡。
- **求解器**（`96jonesa/ygo-combo-solver`）：已作为漏斗第一层（funnel.md），给出卡手率、combo 长度、抗手坑率。
- **外部数据**：MD 只有单卡使用率与胜率差，TCG / OCG 站点只有卡表与 top-cut 占比（设计 05）。它们只能做先验，没有可用的对局矩阵。
- *Optimal play in Yu-Gi-Oh! TCG is hard*（arXiv 2603.02863）：证明判定一个策略是否必胜不可判定。与组牌无直接关系。

## 先做的测量（按依赖顺序）

| # | 测量 | 做法 | 通过标准 / 用途 |
|---|------|------|------|
| M1 | 公共随机数相关 ρ | 5 套语料牌组 × 10 个随机单卡替换，每个 500 对；亲本与子代逐局胜负的相关系数 | 决定全部预算（结论的表）；ρ < 0.6 时先做 M5 |
| M2 | 留一真值 | 5 套牌组 × 10 张卡，把该卡换成一张固定的「空白」通常怪兽，每个 1,000 对（共约 10 万局，批量路径约 3.2 小时） | 得到 50 个逐卡真实边际值，作为下面几项的标定 |
| M3 | 信号能否预测真值 | 与 M2 比较：组内起手效应、GIH 式效应、死卡率、使用后 Q − V̄、critic 起手反事实 | 至少一个信号 Spearman ≥ 0.5；用于设定第 3 步各信号的权重 |
| M4 | critic 起手值当代理 | 100 个改动（单卡、三卡、换包）：256 个起手的 V̄ 差与 M1 式实测配对差比较 | 相关 ≥ 0.5 才启用 L0 筛；另报对从未训练过的卡的误差 |
| M5 | 控制变量降方差 | 在 M1 的数据上，比较扣除「起手手气」前后配对差的方差 | 方差降 ≥ 30% 才进评估器 |
| M6 | 加入分 | 对 30 个「按加入分挑的卡」与 30 个「随机加入的卡」测实测 Δ | 前者均值显著更高 |
| M7 | 试用期长度 | 3 套进化卡组入池后，每 N 次更新复评一次，看 Δ 随训练的变化 | 定下试用期的 N，量化「难开但强」的低估幅度 |

## 测量结果（#107，2026-09-27）

数字见 [benchmarks.md](../benchmarks.md)「牌组迭代的先行测量 M1–M3」。

- **M1 通过**：ρ = 0.84（逐局 0.80），σ_d = 0.212，与上表 ρ = 0.8 一行一致。确认 Δ = 2 pp 要约 885 对，5 pp 约 142 对。M5 不再是前置条件。
  50 个随机单卡替换的实测差在 −4.3 到 +2.2 pp，没有一个显著变好，印证「随机替换不值得评估」。
- **M2**：多数单卡（一份）的留一值在 0 到 +5 pp；胜利条件卡（Exodia 三件）+27 pp。单卡改动的真实 Δ 普遍小于 3 pp，所以**微调必须打包**（§3 第 3 步），单卡子代基本确认不了。
- **M3 未通过**：起手效应合计 Spearman 0.28（对半独立局 0.19，去掉 Exodia 0.34），critic 起手 V̄ 0.14；而真值自身的噪声允许的上限约 0.88（去掉 Exodia 0.86）。
  起手效应衡量「早拿到」的价值，对胜利条件卡、被检索的卡符号相反，对展开起点则高估。

**因此改动：**
1. 起手效应与 critic 起手 V̄ **只作为候选排序的弱先验**，不再用于淘汰候选（不做门槛），在 Thompson 采样的先验里给小权重（先验方差按 M3 的相关取大）。
2. 移出分以「卡的角色」修正：胜利条件、被检索目标（卡组内有能检索它的卡）不按起手效应扣分；这类信息由卡片文本 / 脚本的检索关系给出（#109）。
3. 真正的逐卡边际值来自**配对评估本身**：打包子代的实测差按包内各卡摊回（ridge 一阶，§4 的代理），逐代累积，取代单局统计作为主信号。
4. 死卡率与「使用后 Q − V̄」尚未测（要逐动作记录），挪到 #109 与 M6 一起测；L0 critic 筛（M4）在 critic 解释方差提高前暂缓。

**评估器的实现（#110）**：前二 Thompson 采样、critic 控制变量与序贯复核见 [tuning.md](../tuning.md)「进化评估器」。与上文不同或补充之处：
- 「P(Δ > 0) > 0.95 即停」在第一轮（每个子代 25 对）就可能被中性子代触发（8 个子代时约 10%），误停交给序贯复核拦下；
- 600 对的上限只算子代的对局，亲本对局另计（可复用诊断的亲本对局）；
- 复核的无望界不用 α 支出界，而用固定 z = 1.645、最小效应 2 pp（非约束，不影响 α），否则无效应时几乎总要打满 1,000 对；
- 控制变量的 β 默认 0.5，可改为在不相交的局上拟合的值（仍无偏）；M5 以留出牌组上拟合 β 的降幅为准。critic 噪声大时控制变量会增大方差，M5 未达 30% 前默认关闭。M5 与「每接受一次改动的对局数」待 GPU 实测。

**进化步骤的实现（#111）**：一轮、谱系、本轮报告与 MAP-Elites 档案见 [tuning.md](../tuning.md)「进化步骤」。与上文不同或补充之处：
- 选亲本只按「作为亲本的次数」轮转（同次数取档案胜率高者），档案空格尚不参与；
- 诊断默认关闭（起手效应的校准权重为 0，只剩亲本基线的作用）；L0 筛默认关闭，`shadow` 模式只统计会筛掉的子代；
- 每个亲本只复核前二 Thompson 采样选中的一个子代，所以每轮每个亲本至多接受一个；
- 回流只用非自适应的对局（每个子代的第一批、被复核子代的复核对局），搜索中后续的对局有选择偏差，不进信号库；
- 档案的 combo 长度与卡手率用卡表上的代理（检索链深度、起手无启动卡的概率），不跑求解器；
- 档案的目标值是胜率的单侧下界；对手分布或检查点变了，旧精英过期、按格重置；
- 入池前的最低标准、试用期复评与淘汰留给 #113。
- 交叉子代（#141，默认关闭）：亲本 × 远处格子的档案精英，按卡表交叉，换入不超过一半差异，走同一评估路径，见 [tuning.md](../tuning.md)「交叉子代」。

## 来源

- García-Sánchez, Tonda, Squillero, Mora, Merelo. *Evolutionary Deckbuilding in HearthStone*. IEEE CIG 2016. https://hal.science/hal-01494733 （表 I：µ = 10、50 代、每个对手卡组 16 局）
- Bhatt, Lee, de Mesentier Silva, Watson, Togelius, Hoover. *Exploring the Hearthstone Deck Space*. FDG 2018. https://dl.acm.org/doi/10.1145/3235765.3235791 （只读了摘要）
- Fontaine, Lee, Soros, de Mesentier Silva, Togelius, Hoover. *Mapping Hearthstone Deck Spaces through MAP-Elites with Sliding Boundaries*. GECCO 2019. https://arxiv.org/abs/1904.10656 ；代码 https://github.com/tehqin/EvoSabber （§3.1、§4）
- Zhang, Fontaine, Hoover, Nikolaidis. *Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding*. GECCO 2022. https://arxiv.org/abs/2112.03534 ；代码 https://github.com/icaros-usc/EvoStone2 （表 1、附录 A.1）
- Fontaine, Nikolaidis. *Covariance Matrix Adaptation MAP-Annealing*. GECCO 2023. https://arxiv.org/abs/2205.10752 （只读了摘要）
- Bahlous-Boldi, Faldor, Grillotti, Janmohamed, Coiffard, Spector, Cully. *Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity*. GECCO 2025. https://arxiv.org/abs/2502.00593
- Chen, Amato, Nguyen, Cooper, Sun, Seif El-Nasr. *Q-DeckRec: A Fast Deck Recommendation System for Collectible Card Games*. IEEE CIG 2018. https://arxiv.org/abs/1806.09771 （表 I、II）
- Vieira, Tavares, Chaimowicz. *Drafting in Collectible Card Games via Reinforcement Learning*. SBGames 2020. https://www.sbgames.org/proceedings2020/ComputacaoFull/209690.pdf （只读了摘要）
- Kowalski, Miernik. *Evolutionary Approach to Collectible Card Game Arena Deckbuilding using Active Genes*. IEEE CEC 2020. https://arxiv.org/abs/2001.01326 （只读了摘要）
- Ward, Brooks, Troha, Mills, Khakhalin. *AI solutions for drafting in Magic: the Gathering*（Draftsim）. 2020. https://arxiv.org/abs/2009.00655 （只读了摘要）
- Sierkovitz. *Using Win Rate Data*. 17lands blog, 2021. https://blog.17lands.com/posts/using-win-rate-data/ ；17lands 指标定义页 https://www.17lands.com/metrics_definitions （JS 渲染，未能读取；IWD 的定义未逐字核对）
- mtgds. *Knowledge and Power: Estimating Adjusted Win Rate in Magic: the Gathering Limited*. 2022. https://mtgds.wordpress.com/2022/02/28/knowledge-and-power-estimating-adjusted-win-rate-in-magic-the-gathering-limited/ （博客，非同行评审）
- Russo. *Simple Bayesian Algorithms for Best Arm Identification*. COLT 2016. https://arxiv.org/abs/1602.08448
- Baptista, Poloczek. *Bayesian Optimization of Combinatorial Structures*（BOCS）. ICML 2018. https://arxiv.org/abs/1806.08838
- Oh, Tomczak, Gavves, Welling. *Combinatorial Bayesian Optimization using the Graph Cartesian Product*（COMBO）. NeurIPS 2019. https://arxiv.org/abs/1902.00448
- Burch, Schmid, Moravčík, Morrill, Bowling. *AIVAT*. AAAI 2018. https://arxiv.org/abs/1612.06915 （数字引自 [reward-signal.md](reward-signal.md)）
- sbl1996/ygo-agent @ `dbf5142`：`ygoenv/ygoenv/ygopro/ygopro.h`、`scripts/cleanba.py`、`assets/deck/`. https://github.com/sbl1996/ygo-agent
- 以下按已知内容引用，本 spike 未复核细节：
  - Karnin, Koren, Somekh. *Almost Optimal Exploration in Multi-Armed Bandits*. ICML 2013.
  - Li et al. *Hyperband*. JMLR 2018. https://arxiv.org/abs/1603.06560
  - Birattari, Stützle, Paquete, Varrentrapp. *A Racing Algorithm for Configuring Metaheuristics*（F-race）. GECCO 2002.
  - Wang, Lehman, Clune, Stanley. *POET*. 2019. https://arxiv.org/abs/1901.01753
  - McMahan, Gordon, Blum. *Planning in the Presence of Cost Functions Controlled by an Adversary*（double oracle）. ICML 2003.
  - Lanctot et al. *A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning*（PSRO）. NeurIPS 2017. https://arxiv.org/abs/1711.00832
  - Vinyals et al. *Grandmaster level in StarCraft II using multi-agent reinforcement learning*（AlphaStar）. Nature 2019.
  - *Optimal play in Yu-Gi-Oh! TCG is hard*. 2026. https://arxiv.org/abs/2603.02863
  - card2vec / decksage、DraftFM（arXiv 2608.19568）：见设计 05。
