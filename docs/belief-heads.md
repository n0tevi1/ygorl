# 信念头（`ygorl.nets.belief` / `ygorl.env.belief_prior`，T4c.1）

对手预测（[设计文档 04](design/04-opponent-model.md)）的五个信念头、它们的损失掩码、meta 先验初始化与 HDT 式过滤特征，
以及一次真实自博弈数据上的小实验。评估指标与掩码约定沿用 [belief-eval.md](belief-eval.md)（T3.5），真值来自
[encoding.md](encoding.md)「训练态真值」（T2.5）。

| 模块 | 内容 | 依赖 |
|------|------|------|
| `ygorl.env.belief_prior` | `MetaTable`（meta 卡表 + 候选卡 + 角色位 + 哈希桶）、`observe` / `EvidenceTracker`（公开证据）、`hdt_prior`（HDT 式过滤）、`role_targets`、`responded_labels` | numpy |
| `ygorl.nets.belief` | `BeliefConfig`、`BeliefHeads`、`BeliefOutput`、`belief_losses`、`loss_weights`、`evaluation_batch`、`BeliefPolicy`（接策略网络） | PyTorch（`train` 可选依赖） |
| `tools/train_beliefs.py` | 自博弈采数 → 短训练 → T3.5 评估的可复现实验 | 两者 |

```python
from ygorl.env.belief_prior import Evidence, EvidenceTracker, MetaTable, default_roles, hdt_prior
from ygorl.nets import NetConfig, PolicyNet, collate
from ygorl.nets.belief import BeliefConfig, BeliefHeads, BeliefPolicy, belief_losses, prior_tensors

meta = MetaTable(vocab, db, meta_decks, shares=shares, other_share=0.1, generic=generic, roles=default_roles(pool))
cfg = BeliefConfig.from_meta(meta, context_dim=128, action_dim=128)
model = BeliefPolicy(PolicyNet(NetConfig(vocab_size=len(vocab), belief_dim=cfg.policy_dim)), BeliefHeads(cfg))

tracker = EvidenceTracker(meta)                     # 每局、每个 viewer 一个；新局 reset()
evidence = tracker.update(ev.obs["cards"], ev.obs["globals"])   # 只读 actor 观测（公开信息）
policy_out, beliefs = model(collate([ev.obs]), prior_tensors(hdt_prior(Evidence.stack([evidence]), meta)))
losses = belief_losses(beliefs, targets, loss_weights(progress), prior=prior)   # targets 来自 ev.privileged
```

## 五个头

K = meta 牌组类型数，C = 候选卡数，R = 角色位数，S = 15 个盖卡区域（`op_set` 的行：怪兽区 0–6、魔陷区 7–14），
A = 候选动作行。形状与 `ygorl.eval.beliefs` 一致，可直接用 `evaluate_beliefs` 评估。

| 头 | 输出 | 真值（训练态） | 损失掩码（True = 参与） |
|----|------|------|------|
| `deck_type` | `[B, K+1]` softmax（最后一类 other） | 自博弈采样牌组时记下的类型（离群噪声牌组仍记原 meta 类型，rogue 牌组记 other） | 标签 ≥ 0 |
| `remaining_copies` | `[B, C, 4]`，每卡一个 0–3 份数的 4 类头 | `belief_targets`：主 + 额外卡组中**未公开**的份数（已现份数已扣除），截断到 3 | 全部；另丢弃 `> 3 − visible` 的目标（按构造不可能） |
| `hand` | `[B, C]` P(≥1 张在手) | `belief_targets` | 手牌里有该卡的已公开张 → 掩掉（按构造置 1） |
| `hand_roles` | `[B, R]` P(≥1 张该角色在手) | `role_targets`（`op_hand` × 角色表） | 已公开的手牌已有该角色 → 掩掉 |
| `set_cards` | `[B, S, C]` 每个盖卡区域的类别分布 | `belief_targets`：候选列号，空区域 / 非候选卡为 −1 | 有里侧候选卡的区域 |
| `responded` | `[B, A]`（动作级，读候选动作嵌入）或 `[B]` | `responded_labels`：我方下一次发动在其连锁环处理前是否被对手连锁 | 标签已知（选中动作是发动且流未截断），取选中行 |

**类别数受控**：输出只覆盖 C 张候选卡 = meta 并集 + 泛用卡（`MetaTable` 构造时给出，顺序固定、按 8 位 `password`）。
其余卡只出现在**输入侧**：对手已公开的非候选卡按 `crc32(password) mod H` 落进 H 个哈希桶（默认 64），作为「见过离表卡」的证据；
它们在盖卡头里是 −1（掩掉），不计入份数与手牌头。卡片身份嵌入由共享主干的 `CardIdentity` 负责（主干读卡片表与事件流）。

**角色位**：`default_roles(generic_pool, extenders)` 给出设计里的五位——`hand_trap`（泛用池中 role = hand_trap 的卡）、
`ash`（灰流丽 14558127）、`maxx_c`（增殖的 G 23434538）、`nibiru`（原始生命态 尼比鲁 27204311）、`extender`（延伸）。
前四位由卡密确定；「延伸」没有权威卡表，由环境提供（`MetaTable(roles=...)` 接受任意 `role -> passwords`）。
实验脚本用一个粗略规则：meta 牌组里非泛用的主卡组怪兽、脚本含 `CATEGORY_SPECIAL_SUMMON` 且有 `SetRange(LOCATION_HAND)`（手牌自特召类）。

## 公开证据与 HDT 式过滤（`hdt_prior`）

`observe(cards, globals, meta)` 只读 actor 观测（[encoding.md](encoding.md) 卡片表 / 全局向量），得到 `Evidence`：

| 字段 | 含义 |
|------|------|
| `visible` `[C]` | 当前可见的、对手**持有**（owner）的各候选卡张数（不含衍生物） |
| `seen` `[C]` / `seen_other` `[H]` | 本局至今「同时可见」张数的最大值（`EvidenceTracker` 逐决策点取 max），是对手卡表份数的下界；离表卡按哈希桶 |
| `public_hand` `[C]` | 对手手牌中已公开的张数 |
| `counts` `[5]` | 对手主卡组张数、隐藏手牌、里侧场上、里侧除外、隐藏额外卡组（由全局向量与卡片表推出） |
| `set_zones` `[15]` | 对手哪些区域有里侧卡 |

单测在真实对局的每个决策点核对：`counts` / `set_zones` 与特权真值逐项相等；未改动的 meta 牌组从不被过滤掉；
`3 − visible` 确实是剩余份数的上界；`public_hand` 与真值中的公开手牌一致。

过滤（Hearthstone Deck Tracker / Bursztein 2016 的做法）：

- **牌组类型**：meta 类型 k 一致 ⇔ 对所有候选卡 `seen ≤ list_k` 且没见过离表卡；后验 ∝ 份额 × 一致，other 保留自己的份额
  （没有一致类型时 other = 1）。
- **每个假设 h 的未现份数**：`u = clip(list_h − visible, 0, 3 − visible)`（other 假设用份额加权平均卡表取整）。
- **剩余份数**：主卡组卡的 u 张均匀分布在隐藏主卡组池（主卡组 + 隐藏手牌 + 里侧场上 + 里侧除外，共 n 张）里，
  留在卡组的张数 ~ 超几何(n, u, 主卡组张数)；额外卡组卡的 u 张都在额外卡组（不超过隐藏额外张数）。
- **手牌**：P(≥1 在手) = 1 − 超几何(n, u, 隐藏手牌数) 取 0 的概率；已公开在手 → 1。
- **角色位**：同上，u 换成该角色所有卡的未现份数之和。
- **盖卡**：区域 z 的分布 ∝ Σ_h 后验 × u × 该区域可放的卡（怪兽区 = 主卡组怪兽，魔陷区 = 魔法 / 陷阱，场地区 = 场地魔法）。
- 以上对假设按后验混合；多分类输出加 ε = 1e-4 平滑，避免 0 概率。

`BeliefPrior.features`（宽 `K+1 + 1 + 4C + R + H + 5`）：后验、一致类型数、手牌 / 期望剩余份数 / 角色的 HDT 概率、
`seen` / `visible`、哈希桶、各区张数——作为信念头的输入特征（设计：「同时作为硬基线与输入特征」）。

## 头结构：meta 先验初始化

```
h = MLP(LayerNorm(features ⊕ trunk context))                       trunk context = PolicyNet.features(obs).context
deck_type        = s₀ · log p_HDT + W₀ h
remaining_copies = s₁ · log p_HDT + W₁ h          （类 j > 3 − visible 置 −1e9）
hand / hand_roles = s · logit p_HDT + W h        （已公开在手置 +1e9）
set_cards[z, c]  = s₄ · log p_HDT + <q(h + zone_z), e_c> / √d + b_c
responded[a]     = MLP(relu(g(h) + W_a action_a))        （无动作嵌入时 = MLP(h)）
```

`W₀…W₃` 与 `q` 的末层初始化为 0、`s = 1`，所以**未训练的头恰好输出 HDT 后验**（单测）；训练只学残差。
结构约束（已公开置 1、份数上界）在前向里施加，训练后也永远成立（单测扰动参数后检查）。

## 损失

`belief_losses(out, targets, weights, prior=prior)`：逐头交叉熵（多分类 CE、二分类 BCE），**在各头未掩码条目上取平均**
（份数头有 C 个条目，也不会压过牌组类型头），`total = Σ w_h · loss_h`。掩码见上表；被掩码的条目取任何值都不改变损失和梯度（单测）；
某头全部被掩码时损失为 0 而不是 NaN。`loss_weights(progress, late=2.0)`：手牌 / 角色 / 盖卡头的权重随训练进度从 1 线性升到
`late`（设计：「后期加大手牌 / 盖卡头权重」）。

## 接入策略网络（三通道中的 (a)(b)）

`BeliefPolicy(net, heads, feed_policy=True)`：

- 信念头读 `PolicyNet.features` 的 `context`（和候选动作嵌入，给动作级 `responded`）；信念损失回传进共享主干（辅助损失通道 (b)）。
  `BeliefConfig(detach_context=True)` 切断这条梯度，留作 T4c.2 消融。
- `BeliefOutput.policy_features()` = `[牌组类型概率, 手牌概率, 期望剩余份数 / 3, 角色概率]`，宽 `BeliefConfig.policy_dim =
  K+1 + 2C + R`；以 `NetConfig(belief_dim=policy_dim)` 构造网络后由 `net.logits(f, belief)` 输入 actor（通道 (a)）。
  `ActionHead` 内部 detach：策略损失不会训练信念头（单测）。宽度不一致时构造即报错；`PolicyNet` 本身未改动，旧调用方不受影响。
- actor 侧只用公开信息：`hdt_prior` 的输入是 `EvidenceTracker`（读 actor 观测），真值只进损失。

特权 critic（通道 (c)）已有 `ygorl.train.critic`；三通道消融是 T4c.2。

## 实验（`tools/train_beliefs.py`）

```bash
uv run python tools/train_beliefs.py            # 默认：1500 局、每个模型训练 300 秒、2 线程；--out DIR 写 report.json / report.md
```

**设置**（2026-09-23，基于提交 `7547920`，环境：仓库自带卡库 + `tests/decks`，没有 `environments/<版本>/`——这是管线验证用的小实验，
不是某个环境版本的正式数字）：

- **meta 卡表**：`tests/decks` 按名字排序的前 8 套为 meta 类型（branded_despia … tenpai），份额按 1/名次（0.31 → 0.04），
  other 份额 0.15，由后 2 套（voiceless_voice、yubel）充当 rogue 牌组。30% 的 meta 牌组随机换 1–3 张主卡组
  （一半从全卡池、一半从泛用池抽），标签仍记原 meta 类型——设计中「off-meta 噪声以容忍离群」。
- **候选卡** C = 158（meta 并集 150 张 + 不在任何 meta 牌组里的泛用池卡 8 张；泛用池 35 张中 27 张已在 meta 并集里），角色位 R = 5（`extender` 按上文规则得 18 张），哈希桶 H = 64，特征宽 716。
- **数据**：`EncodedVecEnv(privileged=True)`，双方均匀随机合法动作，12 回合截断；1500 局、410,402 个决策点，保留 10% 的决策点 +
  全部「选中发动」的决策点（只有它们有 `responded` 标签），共 46,029 个样本。按局切分：每 5 局 1 局测试（9,414），
  局号 ≡ 1 (mod 10) 为验证（4,795，早停用），其余训练（31,820）。`responded` 标签：训练 3,926 / 测试 1,150，正例率 7.3%。
  采数按局设定随机种子并按（局，决策序号）重排，与线程调度无关；训练按墙钟预算，步数随机器负载变化。
- **模型**：(a) `heads`：只读 HDT 特征的信念头（606,528 参数，隐层 256，dropout 0.1）；(b) `trunk+heads`：同样的头接在小号
  `PolicyNet` 主干上（d = 64、局面 1 层、GTrXL 历史 1 层、最近 48 个事件 token、卡片 ID 嵌入；共 2,135,680 参数），
  读主干 context 与候选动作嵌入（动作级 `responded`）。只用信念损失，AdamW（lr 3e-4、wd 1e-4）、批 128、`loss_weights` 调度，
  每 50 步在验证集上算各头损失之和，保留最好的参数。4 核机器与其它任务共享、torch 2 线程：(a) 300 秒跑了 3,418 步（最佳第 1,000 步），
  (b) 300 秒只跑了 200 步（约 0.8 个 epoch，验证损失仍在下降）。
- **基线**：`uniform`；`prior` = `prior_predictor(fit=训练局)`（频率先验，Laplace 平滑 1）；`hdt` = `hdt_prior`（`responded`
  没有 HDT 式模型，取训练集正例率）。

测试局上的 `evaluate_beliefs`（15 个等宽箱；`n` = 参与评估的条目数）：

| 指标 | uniform | prior | hdt | heads | trunk+heads | n |
|---|---:|---:|---:|---:|---:|---:|
| deck_type/top1 | 0.111 | 0.310 | 0.547 | **0.863** | 0.837 | 9414 |
| deck_type/top3 | 0.333 | 0.625 | 0.914 | **0.938** | 0.924 | 9414 |
| deck_type/ece | 0.000 | 0.003 | 0.309 | 0.012 | 0.042 | 9414 |
| deck_type/nll | 2.197 | 1.965 | 1.322 | **0.415** | 0.493 | 9414 |
| deck_type/brier | 0.889 | 0.828 | 0.619 | **0.178** | 0.204 | 9414 |
| remaining_copies/accuracy | 0.250 | 0.857 | 0.895 | **0.953** | 0.944 | 1487412 |
| remaining_copies/ece | 0.000 | 0.002 | 0.037 | 0.004 | 0.004 | 1487412 |
| remaining_copies/nll | 1.386 | 0.381 | 0.321 | **0.136** | 0.166 | 1487412 |
| remaining_copies/brier | 0.750 | 0.217 | 0.147 | **0.074** | 0.088 | 1487412 |
| hand/auc | 0.500 | 0.905 | 0.948 | **0.967** | 0.965 | 1487397 |
| hand/auc_macro | 0.500 | 0.500 | 0.879 | **0.892** | 0.890 | 1487397 |
| hand/ece | 0.489 | 0.000 | 0.001 | 0.001 | 0.001 | 1487397 |
| hand/brier | 0.250 | 0.010 | 0.010 | 0.009 | 0.009 | 1487397 |
| hand/log_loss | 0.693 | 0.046 | 0.040 | **0.035** | **0.035** | 1487397 |
| hand_roles/auc | 0.500 | 0.805 | 0.844 | **0.895** | 0.889 | 47055 |
| hand_roles/auc_macro | 0.500 | 0.500 | 0.767 | **0.792** | 0.779 | 47055 |
| hand_roles/ece | 0.293 | 0.005 | 0.014 | 0.009 | 0.009 | 47055 |
| hand_roles/brier | 0.250 | 0.128 | 0.112 | **0.101** | 0.103 | 47055 |
| hand_roles/log_loss | 0.693 | 0.396 | 0.357 | **0.311** | 0.320 | 47055 |
| hand_roles/accuracy | 0.207 | 0.815 | 0.854 | **0.859** | 0.855 | 47055 |
| set_cards/top1 | 0.006 | 0.210 | 0.171 | **0.290** | 0.280 | 9601 |
| set_cards/top3 | 0.019 | 0.382 | 0.528 | **0.621** | 0.620 | 9601 |
| set_cards/ece | 0.000 | 0.020 | 0.035 | 0.034 | 0.031 | 9601 |
| set_cards/nll | 5.063 | 3.372 | 2.604 | **2.106** | 2.151 | 9601 |
| set_cards/brier | 0.994 | 0.927 | 0.876 | **0.813** | 0.818 | 9601 |
| responded/auc | 0.500 | 0.500 | 0.500 | **0.640** | 0.567 | 1150 |
| responded/ece | 0.449 | 0.022 | 0.022 | 0.019 | 0.019 | 1150 |
| responded/brier | 0.250 | 0.049 | 0.049 | 0.049 | 0.049 | 1150 |
| responded/log_loss | 0.693 | 0.206 | 0.206 | **0.201** | 0.204 | 1150 |

（`hand/accuracy` 与 `responded/accuracy` 在 0.5 阈值下除 uniform 外各列都等于多数类比例（0.989、0.949），从表中省略；完整表见 `report.md`。）

各角色位的 AUC（测试局；括号内为正例率）：

| 角色 | hdt | heads | trunk+heads |
|---|---:|---:|---:|
| hand_trap (0.557) | 0.820 | 0.822 | 0.821 |
| ash (0.101) | 0.720 | 0.720 | 0.723 |
| maxx_c (0.004) | 0.789 | 0.840 | 0.828 |
| nibiru (0.249) | 0.686 | 0.708 | 0.701 |
| extender (0.126) | 0.821 | 0.868 | 0.825 |

解读：

- **验收（各头高于 HDT 过滤基线）**：`heads` 在五个头（含角色位）的排序 / 准确率指标与 NLL / log-loss / Brier 上全部优于 `hdt`；
  持平的只有 `hand/ece`（都约 0.001）、`set_cards/ece`（0.034 vs 0.035）、`responded/brier`（0.049）和 ash 单项 AUC（0.720）。
  `trunk+heads` 同样在各头优于 `hdt`（`responded/brier` 持平），但只训练了 200 步，多数指标不如 `heads`（见最后一条）。
- **牌组类型**：HDT 精确过滤在离群牌组上会把真实类型排除掉（换进一张离表卡或份数超表即判为不一致），所以 top-1 只有 0.55，
  而且过度自信（ECE 0.31）；学到的头在同样的特征上把 top-1 提到 0.86、ECE 降到 0.01——正是设计要的「保留 other 类与离群逻辑」。
- **校准（ECE）**：学到的头各头 ECE ≤ 0.04，`deck_type` / `remaining_copies` 比 HDT 好得多，其余与 HDT 持平。
  频率先验按构造样本内校准（ECE ≈ 0），但没有逐局区分能力（`auc_macro` = 0.5）。
- **`hand` 的 pooled 指标**：148 万条目里大多数候选卡根本不在对手牌组里（正例率约 1%），`accuracy` / `ece` / `brier` 被这些平凡的 0 主导，
  应看 `auc_macro`（每张卡跨局的区分能力）：0.879 → 0.892。
- **`responded`**：随机对手被问到连锁时才可能响应，正例率只有 7%，标签也少（3,926）；学到的头有一定排序能力（AUC 0.64），
  但 log-loss 只比常数基率好一点（0.201 vs 0.206），Brier 持平。这个头在策略自博弈数据（对手会策略性地藏手坑）上才有意义，
  留给 T4c.2 的「放弃响应」场景评估。
- **主干没有带来额外收益**：在这个预算下 `trunk+heads` 只跑了 0.8 个 epoch，验证损失仍在下降；只读 HDT 特征的 `heads` 已经能拿到
  随机对局里大部分可用的信息。事件流里的放弃 token 等历史信息要在更长的训练与策略数据上才能体现，三通道消融归 T4c.2。
- 数据是**随机**自博弈：对手的出牌与盖卡不反映真实策略，数字只说明管线与损失正确、学到的头能超过 HDT 基线，不代表真实对局的信念质量。

## 限制与后续

- `responded` 标签是「我方下一次发动（任何发动）是否被对手连锁」：被响应的发动往往不会处理完（例如被灰流丽无效），
  所以无法按设计原文只统计「检索 / 特召」类发动；需要动作本身的效果类别（效果文本 / 脚本分类）才能细分。
- 盖卡头只预测候选卡类别，设计里的「角色」维度（盖的是手坑 / 解场 / 陷阱类别）没有单独的头；非候选卡的盖卡被掩掉而不是归入 other 类。
- `extender` 角色表目前是脚本启发式；应由环境的 meta 卡表（T5.1）提供。份数上界只用「每张最多 3 张」，未利用禁限表（限制卡 ≤ 1）。
- `EvidenceTracker.seen` 用「同时可见张数的最大值」做份数下界；曾公开后回到隐藏区的卡按单张计，下界可能偏松（不会错）。
- 训练按墙钟预算，CPU 共享时步数差别很大；正式实验应改为固定步数并在 `docs/experiments/` 记录环境版本。
