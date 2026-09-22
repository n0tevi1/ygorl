# 代理模型（T5.7）

> 对应设计文档 [05-deck-building.md](design/05-deck-building.md)（DSA-ME：在线代理模型 + MAP-Elites；描述符
> {先攻胜率, 后攻胜率, 手坑数, combo 长度, 卡手率}）与 [06-architecture.md](design/06-architecture.md)
> 「漏斗 (2) 代理模型（计数向量 + 卡文本嵌入均值 → 胜率 + 描述符，在线更新）」，风险表
> [08-risks.md](design/08-risks.md)「代理模型对新系列泛化差 → 加卡 / 效果文本嵌入均值；在线更新」；
> 工程计划 T5.7（[#44](https://github.com/n0tevi1/ygorl/issues/44)），验收「留出集误差 < 10 个百分点」。
> 代码在 `src/ygorl/build/surrogate.py`（特征、模型、在线更新、采集）与 `src/ygorl/build/labels.py`（真实对局标签与缓存），
> 实验脚本 `tools/surrogate_experiment.py`。

代理模型是组牌评估漏斗的第二层：求解器起手分析（T5.6）之后、真实对局之前，用一个廉价的回归模型预测候选牌组
**对 meta 池的胜率**（总体 / 先攻 / 后攻）与 QD 描述符，并给出**不确定性**，决定哪些候选值得花真实对局去评估；
真实对局的结果再回流训练代理（在线更新）。全部是 numpy，不依赖 PyTorch。

```python
import numpy as np
from ygorl.build.labels import LabelCache, label_decks
from ygorl.build.surrogate import FeatureMap, Surrogate

fm = FeatureMap(space, db)                         # space: GenotypeSpace，db: CardDB（结构特征用）
sur = Surrogate(fm)                                # 目标默认 win_rate / win_rate_first / win_rate_second

# 冷启动：随机 / 种子基因型打真实对局，写入缓存
seeds = space.sample_many(100, np.random.default_rng(0))
recs = label_decks([space.decode(g) for g in seeds], meta_pool, pairs=2, workers=4,
                   cache=LabelCache("out/surrogate/labels.jsonl"))
sur.update(seeds, [r["labels"] for r in recs], weights=[r["labels"]["games"] for r in recs])

pred = sur.predict(candidates)                     # pred.mean["win_rate"], pred.std["win_rate"]
desc = sur.descriptors(candidates)                 # {"win_rate_first", "win_rate_second", "hand_traps"}
pick = sur.select(candidates, 16, beta=1.0, cells=cell_ids)   # 送去真实对局的候选下标
```

## 特征 `FeatureMap`

输入是基因型、`Deck` 或计数向量（同一 `GenotypeSpace` 的索引；`Deck` 按卡密计数，异画折叠为原卡，空间外的卡忽略，不做修复），
输出按组拼接的实数向量。`fm.groups` 给出每组的列区间，`fm.names` 给出每列的名字。所有特征只由牌组本身决定：
解码成同一套牌的两个基因型特征相同。

| 组 | 列 | 含义 |
|----|----|------|
| `counts` | `len(space)` | 基因型的计数向量（`space.vector(g)`），即设计文档里的「计数向量」 |
| `packages` | 引擎包数 | 每个引擎包成员的总份数（包级聚合：ridge 能在同包成员之间共享信息） |
| `structure` | 约 140 | 卡片结构统计：主卡组中怪兽 / 魔法 / 陷阱、调整、灵摆、等级 1–4 / 5–6 / 7+、各 `attribute` 与 `race`、魔法 / 陷阱子类型、平均 `attack` / `defense`、cdb 检索分类位的**按份数加权比例**；额外卡组中融合 / 同调 / 超量 / 连接、阶级与连接值、属性种族等的比例；主卡组张数、额外卡组张数、主卡组不同卡数 |
| `roles` | 角色数 | 每种泛用角色（`hand_trap`、`board_breaker`、`staple`、`extra`）的份数 |
| `text` | `D` | 可选：卡文本嵌入按份数加权的均值（见下） |

默认启用前四组；`FeatureMap(space, groups=("counts",))` 只用计数向量；`structure` 需要卡片数据库（`cards=`）。
每张卡的结构特征由 `card_feature_matrix(passwords, cards)` 给出，可单独复用（例如给学习型描述符或新系列先验）。

### 卡文本嵌入：可插拔，当前关闭

T5.2（sentence-transformers 离线生成 `.npy`）**尚未完成**：本环境访问不了 HuggingFace 等模型托管站，无法下载编码模型。
因此文本特征做成可插拔：

```python
from ygorl.build.surrogate import TextEmbeddings
emb = TextEmbeddings.load("environments/<v>/artifacts/card_text.npy", "environments/<v>/artifacts/card_vocab.json",
                          missing=cards_without_text)
fm = FeatureMap(space, db, text=emb)            # 自动追加 text 组
emb.coverage(space.passwords)                   # 空间内有嵌入的卡的比例
```

- 约定 T5.2 的产物是 `[len(vocab), D]` 的 `.npy` 表，行号即 `CardVocab.index(password)`（第 0、1 行是填充 / 未知占位），
  配套 `CardVocab.save` 写出的 JSON 词表——与 T4b.1 的卡片编码器共用同一词表对齐方式。
- 牌组的文本特征 = 有嵌入的卡按份数加权的均值；`missing` 中的卡（无文本、占位行）和词表外的卡不参与平均。
- 不给 `text` 时该组不存在，模型只用计数向量与结构特征。**本文档的全部实测数字都是在没有文本嵌入的情况下得到的**；
  文本嵌入的价值主要在跨系列泛化（新卡、新包没有计数历史），这正是本实验的留出集不考察的场景，需要 T5.2 落地后另行验证。

## 模型 `RidgeEnsemble`

- 多目标 ridge 回归，特征先标准化（常数列自动忽略）；每个目标的惩罚系数 `alpha` 用**广义交叉验证（GCV）**在
  `alphas × 行数` 网格上自动选（SVD 一次算完整条路径），`residual_std_` 是对应的残差标准差估计（标签噪声 + 拟合误差）。
- **自助集成**：`n_members`（默认 16）个成员各在一次有放回重采样上拟合，预测取均值，成员间标准差即**认知不确定性**：
  数据越多越小（`test_ensemble_uncertainty_shrinks_with_data`），离训练数据越远越大（`test_ensemble_is_less_certain_far_from_the_data`）。
- 缺失标签：某个目标为 NaN 的行只在该目标上不参与拟合（例如只有部分牌组有 combo 长度）；有效行少于 `min_rows`（默认 10）的目标
  不拟合、预测为 NaN。
- 行权重：可按对局数加权（`weights`），同一目标内归一化到均值 1。
- 为什么是 ridge：几百个带噪标签、几百维稀疏计数特征，线性模型 + 强正则是稳健的基线；GCV 省去手动调参；闭式解使在线重拟合
  几乎免费（400 行 × 642 维、16 个成员、3 个目标：单线程拟合约 0.45 s，预测 1 万个候选约 0.5 s）。
  `Surrogate(model=...)` 可以换成任何有 `fit(x, y, weights)` / `predict(x) -> (mean, std)` 的回归器（例如 PyTorch MLP 集成），接口不变。
- 注意：numpy 的 OpenBLAS 在负载高的机器上用多线程解这类小矩阵会慢上百倍（实测 100 × 157 的 SVD 从 2 ms 变成 0.5–2 s），
  `tests/conftest.py` 与实验脚本都在导入 numpy 前把 BLAS 线程数设为 1；在 QD 主循环里用代理时也应如此。

## 目标与描述符

- **目标**（被回归的标签名）默认 `("win_rate", "win_rate_first", "win_rate_second")`，可配置：T5.6 落地后加入
  `combo_length`、`brick_rate`（卡手率）即可，标签里缺某个目标的行自动跳过。以 `win_rate` 开头的目标预测值裁剪到 [0, 1]，
  其它目标可用 `bounds` 指定范围。
- **描述符**（`descriptors=`，供 T5.8 的 pyribs 档案）默认 `("win_rate_first", "win_rate_second", "hand_traps")`：
  目标里有的取代理预测值，`hand_traps` 等**精确描述符**直接由牌组算出（`EXACT_DESCRIPTORS`；手坑数 = 角色 `hand_trap` 的份数，
  与 `space.role_counts(g)["hand_trap"]` 一致），不需要模型。设计文档五个描述符中的 combo 长度与卡手率待 T5.6 提供标签后作为目标加入。
- 未知的描述符名（既不是目标也不是精确描述符）在构造时报错。

## 在线更新与采集（DSA-ME）

`Surrogate` 保存全部已标注的牌组，**以牌组（计数向量）为键**：同一套牌再次评估时按权重（对局数）合并标签而不是重复加行，
精英被反复评估时标签越来越准。`update(items, labels, weights)` = 加入 + 重拟合；`add` 只加入、`fit` 只重拟合。

DSA-ME（Zhang & Fontaine 2022）的外循环，在本项目中的对应：

1. **冷启动**：随机基因型与 meta 牌组的变异体各若干，真实对局标注（`label_decks`），`update`。
2. **内循环（代理上的 MAP-Elites）**：T5.8 的 pyribs 档案用 `sur.predict` 的 `win_rate` 作 fitness、`sur.descriptors` 作描述符，
   跑若干代，得到「代理档案」。
3. **采集**：从代理档案的精英中挑一批做真实评估：`sur.select(elites, k, beta, cells, explore)`，即 `acquire`：
   - 按上置信界 `mean + beta · std` 排序（`beta = 0` 纯利用，越大越偏向不确定的候选）；
   - 给出 `cells`（每个候选的档案格子）时，先取每个格子的最优者、再取任一格子的第二名，保证一批评估覆盖整个档案；
   - `explore` 比例的名额留给其余候选中 `std` 最大者（纯不确定性采样），用于降低代理在陌生区域的误差。
4. **回流**：真实结果写入标签缓存并 `update`；真实评估过的精英以真实标签进入真实档案。回到 2。

## 标签：真实对局

`label_decks(decks, pool, pairs, seed, agent, opponent, workers, cache)`（`src/ygorl/build/labels.py`）：

- 每个候选（`agent` 驾驶）对 meta 池中每套牌（`opponent` 驾驶）打 `2 × pairs` 局配对种子对局（`Arena`，互换先后攻、共享起手）；
- **公共随机数**：对第 `j` 个对手的对局种子是 `derive_seed(seed, j)`，**所有候选相同**，对手的起手与 agent 种子一致，候选之间的差异更少被运气掩盖；
- 标签 = 对各对手胜率按 `weights`（默认均匀；以后可用 meta 占比或 Nash 权重）加权平均：`win_rate`、`win_rate_first`、`win_rate_second`，
  另有 `games`、`draws`、`errors`、`se`（二项标准误）、`ci_half_width`（按有效样本量 `1 / Σ w_j² / n_j` 的 95% Wilson 半宽）、
  `mean_turns`、`per_opponent`；
- **缓存** `LabelCache`：JSON Lines，每行一条记录，键 = (牌组指纹, 标注配置指纹)。牌组指纹 = 排序后主卡组与额外卡组卡密的 SHA-256
  （与牌组名、顺序、副卡组无关）；配置指纹覆盖 meta 池（名字 + 牌组指纹）、权重、双方 agent、`pairs`、`seed`、回合上限、环境戳。
  已有记录直接返回不再对局；中断的运行可以续跑（被截断的最后一行会被忽略）；同一套牌在一次调用里只打一次。
- 绑定环境：传 `env=` 时按环境规则对局，环境戳进入配置指纹。

## 验收实验

```bash
uv run python tools/surrogate_experiment.py --n-train 400 --n-test 60 --test-pairs 8 --workers 4
uv run python tools/surrogate_experiment.py --n-train 400 --n-test 60 --test-pairs 8 --analyze-only   # 只用缓存重算分析
```

**格式**（不绑定环境，MR5 默认规则）：meta 池 = `tests/decks/` 的 10 套测试牌组（均匀权重）。基因型空间的引擎包 = 10 套牌各自的卡片集合
+ 协同图排名前 20 的引擎包，泛用卡 = `tests/data/generic_pool.json`，`max_packages = 3`：466 张卡（主卡组 368 / 额外卡组 98）、30 个包、35 张泛用卡，
特征 642 维（计数 466、包 30、结构 142、角色 4）。

**候选**：40% 随机基因型（`space.sample`），60% meta 牌组的变异体（`from_deck` 后随机施加 1–12 个变异算子），
所以标签从「接近 meta 的牌」到「随机拼凑的牌」都有。每个候选由自己的种子生成，扩大规模时复用已有缓存。

**标签**：候选由 greedy 驾驶，meta 牌组也由 greedy 驾驶，配对种子、公共随机数。

| 划分 | 数量 | 每个候选的对局 | 种子 | 用途 |
|------|------|----------------|------|------|
| train | 400 | 40（每个对手 2 对） | S | 训练代理 |
| test | 60（与 train 无重复牌组） | 160（每个对手 8 对） | S + 1 | **留出集参考标签**（近似真值） |
| retest | 同 test 的 60 个 | 40 | S + 2 | 「对同一候选再打一次廉价的真实评估」能达到的误差，即标签噪声底 |

共 28,000 局（test 9,600、retest 2,400、train 16,000），用时约 1.7 小时（4 个 worker，机器同时被其它任务占用，2.9–7.0 局/秒）。

### 结果（2026-09-22）

标签分布与噪声（`win_rate`；先攻 / 后攻类似）：

| 项 | 值 |
|----|----|
| train 标签均值 / 标准差 | 0.466 / 0.171（10% 分位 0.225，90% 分位 0.700）；随机基因型 0.476（168 个），meta 变异体 0.459（232 个） |
| 40 局标签的噪声 | 二项标准误平均 **7.4 个百分点**，95% Wilson 半宽平均 13.9 个百分点 |
| 方差分解（train） | 标签方差 0.0292 = 真实差异 0.0237（标准差 15.4 pp）+ 噪声 0.0055（7.4 pp） |
| 160 局参考标签的噪声 | 标准误 3.7 pp，Wilson 半宽 7.2 pp |

**留出集误差**（60 个候选，对 160 局参考标签；三个数字依次为 `win_rate` / `win_rate_first` / `win_rate_second`）：

| 预测方式 | MAE（百分点） | RMSE（百分点） | Spearman（`win_rate`） |
|----------|---------------|----------------|------------------------|
| 常数（train 均值） | 13.49 / 13.83 / 13.81 | 16.96 / 17.29 / 17.44 | — |
| 再打一次 40 局真实评估（retest） | 6.67 / 9.46 / 7.71 | 8.26 / 11.63 / 9.61 | 0.857 |
| 代理：仅计数向量 | 7.32 / 8.12 / 8.63 | 9.24 / 9.68 / 10.73 | 0.851 |
| 代理：计数 + 包 | 7.38 / 8.02 / 8.69 | 9.28 / 9.63 / 10.81 | 0.854 |
| 代理：计数 + 包 + 角色 | 7.30 / 7.94 / 8.60 | 9.15 / 9.53 / 10.65 | 0.859 |
| 代理：包 + 结构 + 角色（无计数） | 7.14 / 7.89 / 8.91 | 8.87 / 9.44 / 10.99 | 0.883 |
| **代理：默认特征（计数 + 包 + 结构 + 角色）** | **7.25 / 7.96 / 8.41** | 8.91 / 9.48 / 10.37 | 0.871 |

- **验收通过**：默认代理在留出集上的 MAE 为 7.25（总胜率）/ 7.96（先攻）/ 8.41（后攻）个百分点，均 < 10。
  按来源：随机基因型 7.34 pp（17 个）、meta 变异体 7.21 pp（43 个）。手坑数描述符是精确计算的，误差为 0。
- 参考标签自身有 3.7 pp 的噪声；从 RMSE 中按平方扣除后，代理对真实胜率的 RMSE 约 **8.1 pp**（常数基线约 16.6 pp，40 局真实评估约 7.4 pp）。
- **与标签噪声底比较**：代理不打任何对局，误差已接近「再打 40 局」（6.7 pp）；先攻胜率上代理反而更准（7.96 vs 9.46，
  因为单侧只有 20 局，噪声更大，而代理在相似牌组之间平均掉了噪声）。
- **特征消融**：各组合差距在 ±0.3 pp 内，小于 60 个留出样本本身的抽样误差（约 ±0.8 pp）。在这个格式里「牌组来自哪几个包、各多少份」
  已经决定了大部分可预测的差异；结构特征与计数向量信息高度重叠。
- **学习曲线**（默认特征，4 个成员，5 次随机子集平均）：25 → 12.35，50 → 10.15，100 → 8.80，200 → 7.97，300 → 7.65，400 → 7.72 pp
  （成员少，n = 400 时比上表 16 个成员的 7.25 略高）。
  约 100 个标注样本即进入 10 pp 以内；300 以后基本持平，剩余误差主要是线性模型的偏差（非线性协同、对手相关的克制关系）加参考标签噪声，
  继续加同分布的廉价标签收益很小。
- **非线性模型探索**（未合入）：在 192 个训练样本时，RBF 核岭回归比 ridge 低约 0.8 pp（7.6 vs 8.4），差距与抽样误差同量级；
  包占比平方项、与各 meta 牌组的重合度等手工非线性特征无改善。需要更大、分布更广的数据再比较（接口已支持替换 `model=`）。

**不确定性**：集成标准差平均 3.5 pp，GCV 残差标准差 8.8 pp；把两者（并换算到参考标签的噪声）合成的 95% 预测区间覆盖率 0.88（略窄）；
集成标准差与实际误差的 Spearman 相关 ≈ 0（−0.01）。在同分布的留出集上，误差主要来自模型偏差与标签噪声，自助集成的**认知不确定性**
不反映这部分误差——它衡量的是「这类牌组见得多不多」，适合用来决定探索方向，不适合当误差条。

**采集模拟**（DSA-ME 外循环的离线版本：以 train 的 400 个已标注候选为池，从 50 个随机样本起步，每轮按规则挑 25 个「揭晓」标签后重拟合；
各规则起点相同）：

| 已标注数 | 随机 | 纯不确定性（`explore=1`） | UCB（`beta=1`） |
|----------|------|---------------------------|-----------------|
| 100 | 9.23 pp（所选均值 0.469） | 9.57 pp（0.505） | 9.15 pp（**0.564**） |
| 200 | 8.60 pp（0.454） | 7.87 pp（0.471） | 7.69 pp（**0.554**） |
| 300 | 8.38 pp（0.454） | 7.04 pp（0.470） | 7.51 pp（**0.516**） |

表中「pp」是留出集 `win_rate` MAE，括号内是已选候选的平均标签（越高说明真实评估的预算越多花在强牌上）。
UCB 用同样的真实评估预算把代理误差降得比随机采样快，同时把评估集中到强牌上（前 100 个的平均胜率 0.56 vs 0.47），这正是 DSA-ME
需要的「边提高代理精度、边找精英」；纯不确定性采样最利于降低误差，但不偏向强牌。单次模拟、留出集只有 60 个，差距约 1 pp 量级，仅作定性参考。

### 局限与后续

- **格式小、分布窄**：留出集与训练集来自同一生成分布（同一空间、同样的 10 套 meta 牌组的变异体 + 随机基因型），
  没有检验对**新系列 / 新卡**的泛化——这正是卡文本嵌入要解决的问题，需等 T5.2 的嵌入落地后另做跨系列留出（按包留出）实验。
- **对手是 greedy**：标签衡量的是「greedy 对 greedy」的胜率，不是强策略下的真实强度；换成 M4 的策略后需要重新标注
  （配置指纹不同，缓存自动区分），代理结构不变。
- **误差预算**：要把 MAE 进一步压到约 5 pp，瓶颈不在标签数量（学习曲线已持平），而在模型偏差：需要非线性模型（核方法 / MLP 集成）
  与更大的数据；若只是想让参考标签更准，把 test 的每对手对局数翻倍（320 局）可把参考噪声从 3.7 降到 2.6 pp。
- combo 长度、卡手率两个描述符等待 T5.6 的求解器标签；加入后作为 `targets` 的新列，接口不变。
