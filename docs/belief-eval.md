# 信念校准评估（`ygorl.eval.calibration` / `ygorl.eval.beliefs`，T3.5）

对手预测（[设计文档 04](design/04-opponent-model.md)）的各个信念头既要「猜得准」也要「说得诚实」：
过度自信的信念喂给策略比不喂更糟，所以 M4c 的验收同时看准确率 / AUC 与 ECE。本页定义评估用到的指标、
掩码约定、报告格式，并给出随机 / 先验预测器在合成数据上的基线数字。纯 numpy 实现，不依赖 torch。

## 用法

```python
import numpy as np
from ygorl.eval.beliefs import BeliefBatch, Head, evaluate_beliefs

batch = BeliefBatch(
    deck_type=Head(deck_probs, deck_type_id),                  # [N, K], [N]
    remaining_copies=Head(copy_probs, copies, mask=~public),   # [N, C, 4], [N, C]
    hand=Head(hand_probs, in_hand, mask=~public),              # [N, C], [N, C]
    set_cards=Head(set_probs, set_class, mask=occupied),       # [N, S, C], [N, S]
    responded=Head(resp_probs, was_responded),                 # [N], [N]
)
report = evaluate_beliefs(batch, n_bins=15)   # {"deck_type/top1": 0.51, "hand/auc": 0.86, ...}
```

单个指标也可直接调用：`calibration.ece(p, y, n_bins=15, strategy="adaptive", mask=m)`、
`calibration.roc_auc(...)`、`calibration.top_k_accuracy(p, t, k=3)` 等。
基线表：`uv run python tools/belief_baselines.py [--n 20000] [--seed 0] [--bins 15]`。

## 头与形状

N = 样本（决策点），K = 牌组类型数（meta 类型 + other），C = 候选卡（meta 并集 + 泛用卡，按固定的
卡片 `password` 词表编号），S = 对手盖卡区域数，R = 手牌角色位数。每个头可选；缺席的头不出现在报告里。

| 头 | `probs` | `targets` | 类型 | 报告指标 |
|----|---------|-----------|------|----------|
| `deck_type` | `[N, K]` | `[N]` int | 多分类 | top1、top3、ece、nll、brier |
| `remaining_copies` | `[N, C, 4]` | `[N, C]` int ∈ 0..3 | 多分类（每卡一个 4 类头） | accuracy、ece、nll、brier |
| `hand` | `[N, C]` | `[N, C]` 0/1 | 二分类（≥ 1 张在手） | auc、auc_macro、ece、brier、log_loss、accuracy |
| `hand_roles` | `[N, R]` | `[N, R]` 0/1 | 二分类（手坑 / 灰流丽 / 增 G …） | 同 `hand` |
| `set_cards` | `[N, S, C]` | `[N, S]` int | 多分类（每个盖卡区域） | top1、top3、ece、nll、brier |
| `responded` | `[N]` | `[N]` 0/1 | 二分类（下一步检索 / 特召被响应） | auc、ece、brier、log_loss、accuracy |

每个头还报告 `<head>/n`：参与评估（未被掩码）的条目数。报告是扁平的 `{"<head>/<metric>": float}`，
方便直接写日志。`BeliefBatch` 构造时校验形状（维数、`remaining_copies` 末维为 4、targets / mask 形状、
各头 N 一致），`evaluate_beliefs` 校验取值（概率在 [0, 1] 且有限、多分类行和为 1（容差 1e-3）、
标签为 0/1、类别下标在范围内），错误信息带头名。

## 掩码

每个指标都接受可选的布尔 `mask`，形状同 `labels` / `targets`：**True = 参与评估**，False = 排除。
用途与设计中的损失掩码一致：

- 已公开信息（按构造置 1 的卡、已全部公开的份数）不计入评估，否则公开信息会虚高准确率；
- 已现份数从剩余份数真值中扣除后再评估；
- 空的盖卡区域掩掉，targets 可填 `-1` 之类的占位值。

被排除的条目不做取值校验（可以是 NaN 或 -1）。掩码后没有任何条目时指标为 NaN；只有一个类别时 AUC 为 NaN。
mask 必须是 bool 数组（0/1 整数数组会被拒绝，避免与下标数组混淆）。

## 指标定义

记参与评估的样本为 i = 1..n。二分类：pᵢ = P(正类)，yᵢ ∈ {0, 1}。多分类：pᵢ ∈ Δᴷ，目标类 tᵢ。

**分箱。** `strategy="uniform"`：B 个等宽箱 [b/B, (b+1)/B)，最后一箱在 1 处闭合；
`strategy="adaptive"`：把预测值排序后切成 B 个样本数近似相等的箱（等质量，适合预测集中在两端的头）。
默认 B = 15。

**可靠性图（reliability diagram）。** 每箱给出 `confidence`（箱内平均预测）、`accuracy`（箱内平均正确率）、
`count`，以及箱边界 `lower` / `upper`（adaptive 时为箱内最小 / 最大预测值）。空箱 count = 0、其余为 NaN。

- 二分类：置信度 = pᵢ，「正确」= yᵢ（即箱内正例比例）。
- 多分类（top-label）：置信度 = maxₖ pᵢₖ，「正确」= top-1 命中（并列按下文的分数计）。

**ECE / MCE。**

ECE = Σ_b (n_b / n) · |acc_b − conf_b|，MCE = max_b |acc_b − conf_b|（只看非空箱）。

完美校准的预测器 ECE 的期望随样本数趋于 0（有限样本下有 O(√(B/n)) 的正偏差，所以比较 ECE 时样本量和箱数要一致）。

**Brier。** 二分类 mean (pᵢ − yᵢ)²，∈ [0, 1]；多分类 mean Σₖ (pᵢₖ − 1[k = tᵢ])²，∈ [0, 2]。

**log-loss / NLL。** 二分类 −mean[yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ)]，多分类 −mean log pᵢ,tᵢ；
p 裁剪到 [ε, 1 − ε]，默认 ε = 1e-7（一次满自信的错误最多记 −log ε ≈ 16.1）。

**ROC-AUC。** 按 Mann–Whitney U 统计量用平均秩计算：

AUC = (Σ_{正例} rank − n₊(n₊ + 1)/2) / (n₊ n₋) = P(s₊ > s₋) + ½ P(s₊ = s₋)

即正负对中正例得分更高的比例，并列计 ½；完美排序 = 1，完全反序 = 0，常数预测 = 0.5。得分可以是任意实数。

- `auc`：把所有（样本, 卡）条目合在一起算（pooled）。它会因为「卡 A 比卡 B 更常在手」这类**卡间基础频率**而高于 0.5；
- `auc_macro`：每张卡（每个角色位）沿样本维单独算 AUC 再取平均（跳过只有一个类别的卡）。只知道每张卡基础频率的先验恰好得 0.5，
  衡量的是「根据对局证据区分这一局有没有这张卡」的能力。与 HDT 过滤基线比较时应看这一项。

**Top-k 准确率。** 目标类是否在概率最高的 k 类之中。并列按随机打破并列的**期望**计：若 g 个类严格高于目标、另有 e 个类与目标并列，
则记 clip((k − g)/(e + 1), 0, 1)。这样均匀预测器恰好得 k/K，而不是依赖下标顺序。`remaining_copies/accuracy` 即 0..3 份数上的 top-1。

**阈值准确率。** 预测 p ≥ τ 为正类（默认 τ = 0.5）的准确率。注意均匀预测器 p = 0.5 会被判为全正，因此它的
`accuracy` 等于正例比例。

## 基线预测器

| 预测器 | 定义 | 预期 |
|--------|------|------|
| `uniform` | 二分类恒为 0.5，多分类恒为 1/K | top-k = k/K，AUC = 0.5；多分类 top-label ECE = 0，二分类 ECE = \|0.5 − 正例率\| |
| `random` | 二分类 U(0, 1)，多分类 Dirichlet(1) | top-k ≈ k/K，AUC ≈ 0.5，但 ECE 大（不知道却很自信） |
| `prior` | 从（掩码后的）真值统计频率，忽略观测：牌组类型份额、每卡份数分布、每卡 / 每角色基础率、盖卡类别份额、被响应基础率；可 Laplace 平滑、可在另一批数据上拟合 | 样本内完全校准（ECE = 0）；pooled AUC > 0.5，`auc_macro` = 0.5 |
| `oracle` | 合成数据的真实生成后验 | 上界；ECE ≈ 0（只剩有限样本噪声） |

`prior` 对应设计中「meta 先验」一类的硬基线（牌组类型按 meta 份额、卡按出现频率）；T4c.1 的真实基线是按已现卡过滤的 HDT 式先验，
在有对局数据后同样用 `Head` 包装、用 `evaluate_beliefs` 评估。

### 合成数据

`synthetic_batch(n, seed=…)` 生成一个 `probs` 就是真实后验的批次（每个 target 都从自己的 probs 采样，因此按构造完美校准）：
牌组类型与盖卡类别份额为 Zipf 式（∝ 1/rank），每卡在手基础率 ~ Beta(1, 4)，角色位基础率 ~ Beta(2, 3)，每卡份数分布为随机
softmax；每个样本在这些先验的 logit 上叠加尺度为 `signal`（默认 1.5）的高斯「证据」。20% 的卡条目按「已公开」掩掉，
每个盖卡区域有 40% 概率为空（掩掉，target = −1）。默认 K = 8、C = 40、S = 5、R = 5。

### 基线数字

`uv run python tools/belief_baselines.py`（n = 20000，seed = 0，15 个等宽箱）：

| 指标 | uniform | random | prior | oracle |
|---|---:|---:|---:|---:|
| deck_type/top1 | 0.125 | 0.125 | 0.290 | 0.513 |
| deck_type/top3 | 0.375 | 0.373 | 0.600 | 0.833 |
| deck_type/ece | 0.000 | 0.214 | 0.000 | 0.008 |
| deck_type/nll | 2.079 | 2.594 | 1.934 | 1.343 |
| deck_type/brier | 0.875 | 0.972 | 0.833 | 0.627 |
| remaining_copies/accuracy | 0.250 | 0.250 | 0.425 | 0.661 |
| remaining_copies/ece | 0.000 | 0.271 | 0.000 | 0.001 |
| remaining_copies/nll | 1.386 | 1.834 | 1.242 | 0.828 |
| remaining_copies/brier | 0.750 | 0.900 | 0.678 | 0.456 |
| hand/auc | 0.500 | 0.500 | 0.726 | 0.864 |
| hand/auc_macro | 0.500 | 0.500 | 0.500 | 0.822 |
| hand/ece | 0.266 | 0.320 | 0.000 | 0.001 |
| hand/brier | 0.250 | 0.333 | 0.158 | 0.118 |
| hand/log_loss | 0.693 | 1.000 | 0.483 | 0.371 |
| hand/accuracy | 0.234 | 0.501 | 0.772 | 0.833 |
| hand_roles/auc | 0.500 | 0.497 | 0.684 | 0.847 |
| hand_roles/auc_macro | 0.500 | 0.498 | 0.500 | 0.817 |
| hand_roles/ece | 0.072 | 0.258 | 0.000 | 0.004 |
| set_cards/top1 | 0.025 | 0.026 | 0.174 | 0.337 |
| set_cards/top3 | 0.075 | 0.075 | 0.351 | 0.577 |
| set_cards/ece | 0.000 | 0.081 | 0.000 | 0.005 |
| set_cards/nll | 3.689 | 4.254 | 3.235 | 2.392 |
| responded/auc | 0.500 | 0.499 | 0.500 | 0.819 |
| responded/ece | 0.089 | 0.255 | 0.000 | 0.013 |
| responded/brier | 0.250 | 0.333 | 0.242 | 0.169 |
| responded/log_loss | 0.693 | 1.004 | 0.677 | 0.510 |

（完整表另含 `hand_roles` / `responded` 的 brier、log_loss、accuracy 以及各头 `n`，运行脚本即可得到。）

解读：

- 随机预测器给出了应有的「无信息」数字：牌组类型 top-1 ≈ 1/K = 0.125、top-3 ≈ 3/K = 0.375，盖卡 top-3 ≈ 3/C = 0.075，
  份数 accuracy ≈ 1/4，各二分类头 AUC ≈ 0.5；同时 ECE 在 0.08–0.32，Brier / NLL 都比均匀预测器差——排序能力相同，
  但过度自信被校准指标惩罚。
- 均匀预测器在二分类头上 ECE 并不为 0：0.5 偏离了真实正例率（手牌约 0.23），这是二分类 ECE 与 top-label ECE 的区别。
- 频率先验完全校准但没有逐局区分能力：`auc_macro` = 0.5，pooled `hand/auc` 却有 0.73，只来自卡间基础率差异。
  训练出的手牌头应以 `auc_macro` 与先验 / HDT 基线比较。
- oracle 的 ECE 不为 0 只是有限样本噪声（`responded` 每箱约千余样本，ECE ≈ 0.01），可作为同样样本量下「校准良好」的参考量级。
- 合成数据只用于验证评估管线与给出参照量级；真实的 M4c 数字来自自博弈数据上的信念头（T4c.1），并绑定环境版本记录。

## 真实数据上的数字（T4c.1）

合成数据之外，训练出的信念头与 HDT 式过滤基线在随机自博弈数据上的对比（同一套指标）见
[belief-heads.md](belief-heads.md)「实验」。那里的 `prior` 列就是本页的 `prior_predictor`（在训练局上拟合），
`hdt` 列是 `ygorl.env.belief_prior.hdt_prior`。
