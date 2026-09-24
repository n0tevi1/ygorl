# 策略网络（T4b.1 / T4b.2）

`ygorl.nets` 是对局策略的 PyTorch 网络：卡片 / 效果编码器、局面 Transformer、事件历史模块（GTrXL / LSTM 可切换）、
动作打分头。输入就是 C++ 步进环境 `EncodedVecEnv` 的观测数组（[encoding.md](encoding.md)），归一化全部在网络里做。
设计依据：[03-play-policy.md](design/03-play-policy.md) I5 / I6 / I9、[06-architecture.md](design/06-architecture.md)「网络」、
[04-opponent-model.md](design/04-opponent-model.md)「三通道消费」。需要 `train` 可选依赖（`uv sync --extra train`）。

```python
from ygorl.nets import NetConfig, PolicyNet, TextFeatures, collate

text = TextFeatures.load("environments/<版本>/artifacts/text", vocab)  # 没有文件 → None（文本特征关闭）
net = PolicyNet(NetConfig(vocab_size=len(vocab)).with_text(text), text)
events = [ev for ev in env.recv(8) if ev.obs is not None]
actions, logp = net.act(collate([ev.obs for ev in events]))  # 窗口模式：每个观测自带最近 L 个事件
for ev, a in zip(events, actions.tolist()):
    env.step(ev.env_id, a)                                   # 动作 = 候选动作表的行号
print(net.parameter_report())
```

## 结构

```
cards [160,23] ─ CardEncoder ─┐
globals [22] ─ GlobalEncoder ─┼─ 局面 Transformer（pre-norm，填充行掩码）─┬─ global 输出 ─┐
history summary ─ Linear ─────┘                                           │               ├─ context MLP ─ context [d]
events [L,20] ─ EventEmbedding ─ 历史模块（因果）─ summary [d] ──────────────┼───────────────┤
                                                                          └─ 卡片均值池化 ─┘
actions [128,10] ─ ActionEncoder（+ 第 1 列所指卡片的上下文输出）─ action [128,d]
logits = <action_a, MLP(context ⊕ belief.detach())> / √d，非法行置 -1e9
```

- **卡片身份**（`CardIdentity`，卡片表、动作表、事件流三处共享）= 状态嵌入（无 / 背面未知 / 已知）⊕ ID 嵌入（`id_embedding`，可关）
  ⊕ 冻结卡文本向量投影 ⊕ 该卡全部效果串向量的均值投影。「⊕」实现为各部分投影后相加，等价于拼接后过一层线性层。
- **卡片表**：类别列（区域、序号、超量素材序号、控制者、持有者、表示形式、可见、公开、属性、种族、等级、阶级、连接值、
  灵摆刻度、指示物、素材数、无效）各自嵌入；`type` 与 `link_marker` 位掩码展开成位；`attack` / `defense` 取 `x/5000`
  与 `log1p(x)/log(65536)` 两种尺度。卡片表的行没有位置编码——区域 / 序号 / 控制者列本身就是位置。
- **全局向量**：viewer、先攻、回合方、阶段、连锁长度、决策消息号、多选已选数、增广开局标志嵌入；回合数、LP、各区张数、
  合法动作总数作数值。
- **动作表**：类型、效果串序号、系统串（按 2048 取模分桶）、表示形式、区域、数值、下标嵌入；第 1 列（卡片表行号 + 1）
  取局面 Transformer 对该卡的**上下文输出**；第 2 列加卡片身份（宣言的卡不一定在表里）；第 3 / 4 列查冻结效果文本（设计 I5）。
  动作之间不互相注意，填充行的内容不影响合法行的 logit（单测）。
- **事件 token**：类型、主体、起终点四列、三个数值（小值嵌入 + `log1p` + `value1` 的 32 位位掩码，覆盖 `move` 的 reason
  与 `abstain` 的触发类型）、回合、阶段、回合方、LP；`card` / `card2` 走共享卡片身份（`card2` 另有投影）；`chaining`
  事件额外查 `(card2, value2)` 的效果文本。
- **动作打分头**：点积 + 掩码 softmax（设计 06）。非法行用有限值 `MASKED_LOGIT = -1e9` 而不是 `-inf`，
  这样 softmax 后概率严格为 0，熵 / KL 不出 NaN。查询网络末层按 0.1 缩放初始化，初始策略接近均匀。

## 冻结文本向量（T5.2 的接口）

T5.2 离线生成、本仓库不下载模型。目录内（例如 `environments/<版本>/artifacts/text/`）两组 `.npy`，任一组可缺：

| 文件 | 形状 | 内容 |
|------|------|------|
| `card_text.npy` / `card_text_passwords.npy` | `[N, Dc]` / `[N]` | 每张卡的卡文本向量，按 8 位 `password` 对齐 |
| `effect_text.npy` / `effect_text_keys.npy` | `[M, De]` / `[M, 2]` | 每条效果串（cdb `str1..16`）的向量；键 = `(password, 串序号 0..15)` |

`TextFeatures.load(dir, vocab)` 按 `CardVocab` 重排成 `card [V, Dc]` 与 `effect [M+1, De]` + `effect_lookup [V, 17]`
（第 `s` 列 = 串序号 `s-1`，与动作表第 4 列、`chaining` 的 `value2` 同编号；0 = 无）。不在词表里的密码忽略，
没有向量的卡、填充与未知行为零向量。两组都缺时返回 `None`。

开关：`NetConfig.card_text` / `effect_text`（默认开）决定「有表时是否使用」；`cfg.with_text(text)` 把实际宽度写进
`card_text_dim` / `effect_text_dim`（0 = 关）。文本表是**非持久 buffer**，不进 `state_dict`（检查点不必带上百 MB 的冻结表）；
从检查点重建时用同一目录重新 `load` 并传给 `PolicyNet`，宽度或词表大小不符直接报错。`TextFeatures.random` 生成随机表供测试。

## 卡片事实（实验，默认关闭）

动机：只在 10 套牌上训练的策略，对没见过的卡几乎只剩结构化列可用（ID 嵌入未训练，范数约为初始化水平；trace 审查），
泛化差（对 Greedy 0.83 → 0.53）。卡片表里原本没有系列（setcode），而 T5.3 从 Lua 脚本挖出的效果语义也没进网络。

`tools/build_card_facts.py OUT_DIR` 写 `OUT_DIR/card_facts.npz`（按卡密；与文本表放同一目录，`TextFeatures.load` 一并读、按词表对齐）：

| 数组 | 形状 | 含义 |
|------|------|------|
| `setcodes` | `[N, 4]` | cdb 系列码 |
| `references` | `[N, 8]` | 脚本引用的系列（系列部分、去重）：检索 / 特召等过滤器里的 `IsSetCard`（否定条件里的不算）、`listed_series`、素材系列 |
| `categories` | `[N]` uint64 | cdb 效果类别位 |
| `queries` / `query_names` | `[N, Q]` / `[Q]` | 脚本「从哪里拿什么」的（动作, 区域）对（`ygorl.build.scripts`，至少 20 张卡用到的，Q = 46） |

系列码只取系列部分（低 12 位），在成员与引用的并集上编号 1..（0 = 无）。数据库 14,755 张卡：8,235 张有系列、5,740 张引用了系列、7,953 张有查询；490 个系列。

网络（`CardIdentity`，`NetConfig.card_facts` 开关，**默认关闭、显式开启**——特征目录里后来多出的 `card_facts.npz` 不会改变续训中的网络；维度由 `with_text` 从表里填）：卡片身份再加四项——
系列成员嵌入之和、引用系列嵌入之和经一层投影（**与成员共用一张系列表**，「检索 X 系列」与「是 X 系列」在同一空间）、
类别多热的线性投影、查询多热的线性投影。四项的权重都**从零开始**（加上这些视角的热启动网络输出与原来完全相同）。`id_dropout`（训练时按概率把每张卡的 ID 嵌入项置零，留下的按 1/(1−p) 放大，期望与评估模式一致）逼网络用可泛化的视角。
热启动（`init_from`）允许新配置只多出卡片视角（文本、事实、ID 丢弃）：旧权重照载，新模块从零开始（`CARD_VIEW_FIELDS`）。
检查点不带这些表，加载时按训练配置的 `text_dir` 重读。命令行：`tools/train_ppo.py --text-dir DIR [--card-facts] [--no-text] [--id-dropout P]`。

## 卡片表示探针

`tools/probe_card_embeddings.py TEXT_DIR...`：给每张卡（去掉异画）贴上系列、cdb 效果类别、脚本（动作, 区域）三组标签，
对每种表示（网络已有的结构化列、每个文本模型的「卡文本 ⊕ 效果串均值」、两者拼接）各训一个线性多标签探针，报宏平均 ROC-AUC；
另报同系列近邻比例（余弦 top-10）。`--split archetype`（默认）按系列整体划分训练 / 测试——新系列才是泛化要面对的（按卡划分时同系列兄弟卡会泄露）；
这时「系列」一项没有意义（测试集的系列在训练里没出现过），只看类别与查询。结果见 [benchmarks.md](benchmarks.md)「卡片表示」。

## 历史模块（T4b.2）

两种骨干实现同一接口 `HistoryEncoder`：

```python
summary, state, tokens = net.history(events, event_mask, state=None)
```

| 返回 | 形状 | 含义 |
|------|------|------|
| `summary` | `[B, d]` | 最后一个有效 token 的输出；本块无 token 时沿用 `state` 的；空历史为固定向量 |
| `state` | `TransformerState` / `LSTMState` | 跨块携带的状态（已 detach）；`state.index(rows)` 取子批 |
| `tokens` | `[B, T, d]` | 逐 token 因果输出（第 t 行只依赖 ≤ t 的行与 state），供 NTP / 辅助头 |

- `TransformerHistory`：GTrXL（Parisotto et al. 2020）。pre-norm 块，残差换成 GRU 式门控（`z` 的偏置 `gate_bias = 2`，
  初始接近恒等映射）；TrXL 式逐层记忆（缓存各层输入、stop-gradient，保留最近 `history_mem_len` 个**有效** token）；
  位置用 ALiBi 相对偏置（按 token 的绝对序号差），因此同一段 token 放在窗口哪里输出都一样，分块喂入也一样。
- `LSTMHistory`：消融基线（`history="lstm"`），多层 `nn.LSTM`，按有效长度 pack。要求每行有效 token 在前（事件数组本就如此）。
- `history="none"`：不用历史（summary 为零，局面 Transformer 不加历史 token）。

**两种用法（选择与理由）**：

1. **窗口模式（训练默认）**：环境已经在每个观测里给出最近 `L` 个 token（`EncodedVecEnv(event_length=L)`），
   每个观测单独以 `state=None` 过一遍。样本彼此独立，PPO 可以任意打乱 minibatch，不需要 BPTT、不需要在 rollout 里存隐状态，
   actor 与 learner 的输入完全相同。代价是每步重算 `L` 个 token（`L=128`、`d=128` 时仍很小）。
2. **流式模式**：只喂自上次调用以来的新 token，并传回 `state`（`net(obs, state=out.state)`）。输出与把整段 token
   一次性喂入完全一致（LSTM 精确；Transformer 在总长不超过 `history_mem_len` 时精确，超过后只看最近的记忆）。
   用于超出 `L` 的长历史或推理时省算力；调用方负责只取新 token（观测里的窗口是滑动的）。

单测覆盖：两者接口一致；改动 t 之后的事件不改变 ≤ t 的输出；前缀窗口的 summary 等于全序列第 t 行；
分块（含每行新 token 数不同的参差块）流式输出与全序列一致；`PolicyNet` 流式与窗口 logits 一致。

## 给后续任务的接口

- **T4b.3 特权 critic**：`f = net.features(obs)` 给出 `context [B,d]`、`actions [B,128,d]`、`action_mask`、`cards`、
  `history`、`history_tokens`。Q 头可对 `f.actions` 与 critic 自己的上下文（`f.context` ⊕ 特权真值编码，设计 I9：
  critic 看「历史 + 特权信息」）打分；V 头接同一上下文。特权张量（`EncodedEvent.privileged`）绝不进入 `features`。
- **T4b.4 PPO**：`out = net(batch)`，`out.log_probs()`、`out.entropy()`、`out.sample()`、`net.act(batch)`；
  批处理 `collate(list_of_obs)` / `to_tensors(batched_arrays)`（int32 → int64，掩码 → bool；没有 `events` 键时历史为空）。
  训练用的组合模型是 `ygorl.nets.actor_critic.ActorCritic`：actor = `PolicyNet`，critic = T4b.3 的 `Critic` 接
  `f.context` ⊕ `PrivilegedEncoder(对手真值)`、对 `f.actions` 打 Q 分；默认与 actor 共享主干（`shared_backbone=False` 时
  critic 另有一个 `PolicyNet` 主干）。特权真值只进 critic（[training.md](training.md) §8）。
- **T4c.1 信念头**：接在 `f` 上（辅助损失通道），输出 `[B, belief_dim]` 以 `net.logits(f, belief)` / `net(obs, belief=...)`
  传入；网络内部 `detach`（设计 04：防止策略把信念头当旁路），`NetConfig.belief_dim = 0` 时不接收。已实现为
  `ygorl.nets.belief`（`BeliefHeads` + `BeliefPolicy`，`belief_dim = BeliefConfig.policy_dim`），见 [belief-heads.md](belief-heads.md)。
- **NTP / 胜负辅助头（I6）**：接 `f.history_tokens`（因果，可直接做下一 token 预测）与 `f.context`。
- **重建**：`NetConfig.to_dict()` / `from_dict()`；参数量 `count_parameters(net)`、`net.parameter_report()`。

## 规模

默认配置（`d_model=128`、4 头、局面 2 层、历史 2 层、FF 256，词表 14,757）：

| 模块 | 参数量（transformer 历史） |
|------|------|
| 卡片身份（ID 嵌入 14,757 × 128 为主） | 1,889,280 |
| 局面编码 | 369,152 |
| 历史（GTrXL 2 层 + 事件嵌入） | 739,328 |
| 动作编码 | 397,312 |
| context + 打分头 | 164,736 |
| **合计** | **3,559,808** |

LSTM 历史合计 3,164,288；关 ID 嵌入且不用历史 915,072；加 384 维卡文本与效果文本（三个投影：卡文本、效果均值、效果串）+147,456。

## 实现上的性能处理（结果不变）

CPU 训练时的热点做了等价改写，都有与原写法逐项比对的单测（`tests/test_nets.py`、`tests/test_train_loop.py`）：

- **裁掉填充**：`PolicyNet.features` 与 `ActorCritic.forward` 先把一批观测裁到最长的有效前缀（`nets.policy.trim_padding`），
  输出再补回原宽度（logits 补 `MASKED_LOGIT`、Q 补 0）；critic 的 Q 头因此不再给 128 行全打分。
- **类别嵌入求和**（`CategoricalEmbedding`）：原写法 `table(idx).sum(-2)` 会生成 `[B, N, 列数, d]` 的中间张量，反向还要对
  上百万个下标排序。现在前向用 `F.embedding_bag(mode="sum")`；不超过 1,024 行的表（卡片 288、事件 453、全局 331 行）反向用
  多热计数矩阵乘梯度（`_SummedRows`，`gradcheck` 精确），更大的表（动作 2,454 行）用 `embedding_bag` 自带的反向。
- **局面自注意力**：`BoardEncoder` 逐层手写 pre-norm 层（打包投影 + `scaled_dot_product_attention`），参数与
  `nn.TransformerEncoderLayer` 完全相同（state dict 不变），省掉 `multi_head_attention_forward` 的投影拷贝；
  eval 模式也不再切到融合推理核，训练 / 评估两种模式输出一致。

合计每次 PPO 更新约快 1.6 倍，端到端约 1.46 倍（[benchmarks.md](benchmarks.md)「PPO 训练吞吐」）。

## 已知局限

- Python `DecisionPoint` 路径（`Duel.run` / Arena）上有两个适配器，登记名都是 `policy:PATH`，按检查点格式选择：
  PPO 训练的 checkpoint 走 `ygorl.agents.checkpoint`（T4b.4）：用锁步的 C++ `HostDuel` 生成与 `EncodedVecEnv` 相同的观测，
  见 [evaluation.md](evaluation.md)「策略检查点 agent」；策略检查点（BC 等，`ygorl.nets.agent`）走 `NetPolicy`（T4a.2，见 [bc.md](bc.md)）：
  Agent 协议加了可选的 `observe(point, core)`（`Duel.run` 在每个决策点、双方的点都调用），适配器用 `ygorl.env.observer.PointObserver`
  （参考编码器 + 事件流）编码，与 `EncodedVecEnv` 逐元素一致（`tests/test_bc.py`）。两者都每个决策点前向一次，只适合评估；
  训练直接在 `EncodedVecEnv` 上用网络。
- 合法动作超过 128 个（宣言卡名）时只能在前 128 行中选（同编码规范的截断）。
- 窗口模式只看最近 `L` 个 token；真实 combo 回合 token 多时按需调大 `L` 或用流式模式。
