# 策略训练：优势估计、特权 critic 与 PPO 自博弈（`ygorl.train`，T4b.3 / T4b.4）

对应设计 [03-play-policy.md](design/03-play-policy.md) I2「特权 Q-critic + Q-boosted 优势（VRPO）：以候选动作打分头再出一个
Q 头，用 Expected-SARSA(λ) 回溯替代 GAE」、I9「critic 看历史 + 特权信息，特权信息绝不喂 actor」，以及
[02-challenges.md](design/02-challenges.md) C3(b)。§1–§7 给出精确公式、符号约定、rollout 数据布局，以及
GAE / VRPO 对照开关（T4b.3，纯 PyTorch，与网络结构无关：critic 只接通用特征张量）；§8 是 PPO 自博弈训练循环（T4b.4，
设计 I1 / I2 / I8：裁剪目标、熵 0.05–0.2、KL 到慢速参考 / BC 先验、当前策略自博弈 + 小历史快照池 + keep-best、牌组池采样、
checkpoint / 日志）。需要 `uv sync --extra train`。

| 模块 | 内容 |
|------|------|
| `ygorl.train.advantages` | `gae`、`expected_values`、`q_advantages`、`expected_sarsa_returns`、`vrpo_advantages`、`normalize_advantages`、`terminal_rewards`、开关 `estimate` |
| `ygorl.train.critic` | `Critic`（历史 ⊕ 特权 → 上下文 → `CandidateQHead` + `ValueHead`）、损失 `q_loss` / `v_loss` |
| `ygorl.train.rollout` | `RolloutCollector`：在 `EncodedVecEnv` 上收集 §1 布局的 `Rollout`；`Assignment`（一局的规格与对手）、`FinishedGame` |
| `ygorl.train.ppo` | `PPOConfig`、`PPOLearner`（一次更新）、可插拔的策略目标 `PolicyObjective` / `register_objective`（`ppo_clip`、`pg`） |
| `ygorl.train.selfplay` | `SnapshotPool`（快照池 + keep-best）、`DeckPool`（牌组池与对阵）、`SelfPlaySchedule`（配对先后攻的发局） |
| `ygorl.train.trainer` | `TrainConfig`、`Trainer`（训练循环、评估、checkpoint、续训、日志）、`evaluate_checkpoint`、`summarize_metrics` |
| `ygorl.train.checkpoint` | checkpoint 读写、`load_policy`（只重建 actor，供对战 / 评估） |
| `ygorl.train.toy` | 玩具博弈 Nim（与 `EncodedVecEnv` 同接口的 `NimEnv` + `NimModel`），单测与调试用 |
| `ygorl.nets.actor_critic` | `ActorCritic`：`PolicyNet` actor + 特权 Q / V critic；`PrivilegedEncoder`（对手真值 → 特征） |

## 1. 数据布局（rollout 收集器的约定）

时间在前：`[T, B]`，逐候选的量为 `[T, B, A]`（A = 候选动作表宽度，编码里是 128）。列 `b` 是一个环境槽位，
行 `t` 是**一次 agent 决策**。

| 张量 | 形状 / 类型 | 含义 |
|------|-------------|------|
| `players` | `[T, B]` int | 这一行做决策的座位（0 / 1，引擎座位或牌组下标均可，只比较是否相等） |
| `actions` | `[T, B]` int | 选中的候选下标（候选动作表的行号） |
| `action_mask` | `[T, B, A]` bool | 合法候选（即观测里的 `action_mask`） |
| `probs` | `[T, B, A]` float | 行为策略 π_old 在这一行的候选概率（rollout 时记录；非法位的值被忽略，合法位按合法集合重新归一） |
| `rewards` | `[T, B]` float | 这一行转移得到的奖励，**从 `players[t]` 的视角**。只用胜负：终局行 +1 胜 / −1 负 / 0 平，其余行 0 |
| `dones` | `[T, B]` bool | 这一行之后该局结束；同列下一行（若有）是新一局的第一行 |
| `truncated` | `[T, B]` bool，可选 | `dones` 的子集：这一局被上限或引擎错误截断，不是胜负（见下「截断的对局」） |
| `valid` | `[T, B]` bool，可选 | 填充标记；填充只能出现在终局行之后，且之后不能再有真实行 |
| `q` | `[T, B, A]` float | critic 的 Q 头输出（VRPO 需要） |
| `values` | `[T, B]` float | critic 的 V 头输出（GAE 需要） |
| `bootstrap_*` | `[B]` / `[B, A]` | 段尾之后那个状态：座位 `bootstrap_player`，GAE 用 V 头 `bootstrap_value`，VRPO 用 `bootstrap_q` / `bootstrap_probs` / `bootstrap_mask` 求 V̄ |

两种排法都支持，也可混用：

- **定长段 + 自动重开**（PPO 常规）：每列 T 行，一局结束就在同列接着放下一局；最后一行若不是终局，就用段尾之后的状态自举。
- **每列一局 + 尾部填充**：`valid = False` 的行放在终局行之后。

游戏王的特殊情况：

- **两人交替**：当前策略自博弈时，同一局双方的行按时间顺序放在同一列，所有行都参与训练。行的视角随 `players` 切换，见 §2。
- **多步决策**：多选拆步（T2.3，逐张选 + Finish）就是同一座位的连续多行，中间行奖励为 0；不需要特殊标记。
  连锁响应、效果内选择同理：谁被问到谁出一行。
- **强制决策**（只有一个合法动作，约占全部决策的 60%，绝大多数是「对方发动时只能选择不连锁」；也包括合法动作都是同一张卡的等价副本、mask 只剩一行的决策，见 [encoding.md](encoding.md)「等价动作去重」）：训练默认 `TrainConfig.skip_forced = True`，由 C++ 步进环境直接走掉（`EncodedVecEnv(skip_forced=True)`），**不出行**、不做推理。对局轨迹与不跳过时完全相同（`tests/test_host_pool.py`）；这样的行本来就没有策略梯度（log 概率为 0），跳过后每局的行数减少约 60%，信用分配的步数也随之缩短。`tools/train_ppo.py --keep-forced` 恢复逐行记录。统计里的「决策数」只算出行和快照对手的决策，不含被跳过的强制决策。
- **裁掉填充**：观测的卡片（160）、动作（128）、事件（64）行都按上限补零，而实际平均只有约 66 张卡、2.8 个动作。`PolicyNet.features` 先把一批观测裁到这批里最长的有效前缀（`nets.policy.trim_padding`），算完再把输出补回原宽度；PPO 更新在整条 rollout 上裁一次，之后每个 minibatch 都从小张量里取。有效行的输出与不裁时一致（`tests/test_nets.py`、`tests/test_train_loop.py`）。加上跳过强制决策，每 2048 行一次更新从约 19 s 降到约 10 s。
- **主机代答**（课程模式 T2.6 的 `solo` / `handtrap` 下替对手自动放弃）：不是 agent 决策，**不出行**
  （与 `DuelResult.actions` / `record_steps` 一致）；它的后果体现在下一行的状态里。
- **终局奖励**：挂在这一局在本列里的最后一行上，从该行座位的视角给出（可用 `terminal_rewards(players, dones, winner)`，
  `winner` 为胜方座位或 −1 平局）。即使终局是对手的动作或主机代答触发的，也挂在最后一个 agent 行上。
- **截断的对局**（T4b.4 起）：回合上限（`turn_limit`）、决策数上限（`decision_limit`）与引擎错误（`error`，含 C++ 池里
  重开失败报回的错误事件）结束的局**不是胜负**，是截断：最后一个 agent 行 `done = True`、`truncated = True`、奖励 0。
  主机按 LP 判的「胜负」（[engine.md](engine.md)，上限的计分方式以后还可能改）一律不进训练目标。`estimate(..., truncated=...)`
  在截断行上用 critic 自举：Expected-SARSA 回报的奖励换成该行自己的 `Q(s_t, a_t)`、GAE 的换成 `V(s_t)`，于是截断行的
  TD 误差为 0，之前的行经 critic 回溯（等价于在该行把列切开、以同座位的 critic 估计自举；单测
  `test_truncated_rows_bootstrap_from_the_critic`）。截断行之后同列仍可接下一局。之所以不用「下一状态」自举：
  上限在引擎要下一个决策时触发，C++ 主机不再编码那个观测；`Q(s_t, a_t)` 本身就是 critic 对 `r + γ V̄(s_{t+1})` 的估计。
- **快照池对手**（非学习方，I1）：对手的行不训练。推荐只收学习方的行（按学习方座位抽出的子轨迹：所有行同一座位，σ 恒为 +1，
  终局奖励挂在学习方的最后一行）；此时对手的回合属于「环境」。也可以保留双方的行、只在策略损失里把对手行掩掉，
  这时回溯穿过对手行要用对手（快照）的 `probs`，critic 估计的是「学习方 + 该快照」联合策略的价值。
- **截断**：候选超过 128 个时只编码前 128 个（[encoding.md](encoding.md)），V̄ 只在编码出的候选上求期望。

## 2. 符号约定

所有价值、Q 值、奖励、回报都**从当前行的行动座位 `players[t]` 的视角**。两人零和：对手视角的价值是本方的相反数，
所以跨到另一座位的行要乘 −1。定义

```
σ_t = +1  若 players[t+1] == players[t]，否则 −1      （最后一行用 bootstrap_player）
c_t = γ · (1 − done_t) · σ_t                         （填充行 c_t = 0）
```

所有递推都用 `c_t` 代替通常的 `γ(1 − done_t)`。默认 γ = 1（有限局、只有终局奖励），λ = 0.95。
等价的直接写法（测试用它做独立参照）：把第 k 行的奖励换到第 t 行视角时乘 `+1 / −1`（`players[k]` 与 `players[t]` 相同 / 不同），
蒙特卡罗回报 `G_t = Σ_k γ^{k−t} · (±1) · r_k`。

## 3. 公式

### GAE(λ)（对照开关）

```
δ_t = r_t + c_t · V(s_{t+1}) − V(s_t)
A_t = δ_t + λ · c_t · A_{t+1}              （段尾 A_T = 0，V(s_T) = bootstrap_value）
V 目标 = A_t + V(s_t)                        （λ-回报）
```

λ = 1 即蒙特卡罗回报减 V，λ = 0 即 TD 误差。

### Expected-SARSA(λ) 回报（Q 头目标）

```
V̄(s) = Σ_{a 合法} π(a|s) · Q(s, a)                           （π 在合法集合上重新归一）
G_t   = r_t + c_t · [ V̄(s_{t+1}) + λ · (G_{t+1} − Q(s_{t+1}, a_{t+1})) ]
      = Q(s_t, a_t) + Σ_{k≥0} (λ c)^k · δ^Q_{t+k}                （(λc)^k 指 λ^k · c_t · … · c_{t+k−1}）
δ^Q_t = r_t + c_t · V̄(s_{t+1}) − Q(s_t, a_t)
```

这是 Sutton & Barto（2018）第 12 章「带控制变量的动作价值 λ-回报」在同策略（ρ = 1）下的形式，加上逐行的视角符号。
段尾：`V̄(s_T)` 由 `bootstrap_q / bootstrap_probs / bootstrap_mask` 求出，迹在段尾截断（`G_T − Q(s_T, a_T)` 取 0）。

- λ = 0：一步 Expected-SARSA 目标 `r_t + c_t · V̄(s_{t+1})`。
- λ = 1 且 Q ≡ 0：蒙特卡罗回报。
- Q 为真值 Q^π 时 `E[G_t | s_t, a_t] = Q^π(s_t, a_t)`（任意 λ）；若转移确定（随机性只来自策略），则**逐样本**
  `G_t = Q^π(s_t, a_t)`：后续采样动作的噪声被 V̄ 积分掉了。

选这一形式而不是「`(1 − λ) V̄ + λ G` 混合」：后者在 λ > 0 时仍含后续采样动作的方差，而设计 I2 的动机正是去掉它
（Fan & Farina 2026：自博弈里随机均衡策略把 GAE 的方差放大）。

### Q-boosted 优势（VRPO，默认）

设计文档只写了「Q 头 + Expected-SARSA(λ) + 合法动作期望」，没有给出优势的精确式子。本实现定义为

```
A_t = G_t − V̄(s_t)
    = [Q(s_t, a_t) − V̄(s_t)]  +  Σ_{k≥0} (λ c)^k · δ^Q_{t+k}          （mode="return"，默认）
A_t = Q(s_t, a_t) − V̄(s_t)                                          （mode="critic"）
```

即：以 critic 的 `Q − V̄` 为主项（「Q-boosted」），用 Expected-SARSA(λ) 的 TD 残差修正 critic 误差。理由：

- 期望正确：Q 为真值时两种模式的条件期望都是 `Q^π(s, a) − V^π(s)`；`return` 模式在 λ = 1 时等于「蒙特卡罗回报 − V̄」
  加零均值控制变量，critic 有偏时仍向真实回报靠拢。
- 方差：`δ^Q` 只含转移的随机性（抽牌、投币、对手的隐藏信息对特权 critic 而言部分已知），不含后续采样动作的随机性；
  GAE 的 `δ` 则含每一步的动作采样噪声，按 λ 累积。玩具 MDP 上（测试 `test_vrpo_removes_the_variance_of_policy_sampling`）：
  各 (s, a) 条件方差之和，确定转移的链上 VRPO 为 0、GAE（λ = 0.95、真值 V）为 1.14；带投币的两人博弈中 3.4 对 5.7。
- 基线 `V̄` 由 Q 与 π 算出，与 Q 头自洽；V 头不参与 VRPO 的优势。
- `mode="critic"` 是纯 critic 版本（零采样方差、偏差全来自 critic），留作消融。

逐候选形式 `q_advantages(q, probs, mask) = Q(s, ·) − V̄(s)`（非法位为 0，π 加权和为 0）供以后做 all-action 策略梯度
（`Σ_a π(a|s) A(s, a) ∇log π(a|s)`）使用；T4b.4 的 PPO 裁剪目标只用选中动作的 `A_t`。

### 归一化

`normalize_advantages(adv, valid, mode)`：`"standard"`（在 valid 上减均值、除标准差）、`"scale"`（只除标准差，保留符号；
VRPO 的 `Q − V̄` 本身在每个状态上 π 加权均值为 0，不必再中心化）、`"none"`。填充位置 0。

## 4. 开关

```python
from ygorl.train.advantages import estimate

est = estimate("vrpo", rewards=r, dones=d, players=p, actions=a, action_mask=m, probs=pi_old, q=q, values=v,
               valid=valid, gamma=1.0, lam=0.95, bootstrap_player=bp, bootstrap_value=v_T,
               bootstrap_q=q_T, bootstrap_probs=pi_T, bootstrap_mask=m_T, truncated=cut, normalize="standard")
est.advantages   # PPO 用的优势（已按 normalize 处理）
est.q_targets    # Q 头目标 G_t（Expected-SARSA(λ)）
est.v_targets    # V 头目标
```

| `estimator` | 优势 | Q 头目标 | V 头目标 |
|-------------|------|----------|----------|
| `"vrpo"`（默认路线） | `G_t − V̄(s_t)`（或 `vrpo_mode="critic"`：`Q − V̄`） | Expected-SARSA(λ) `G_t` | `G_t`（其关于 a_t 的期望是 V^π(s_t)） |
| `"gae"`（对照） | GAE(λ) | 给了 Q 时仍为 `G_t`（两种设置训练同一个 critic），否则 `None` | GAE λ-回报 |

`q` / `values` 由更新前的 critic 在 rollout 数据上算出，`probs` 是行为策略 π_old；目标在一次 rollout 内算一次，
PPO 的多个 epoch 共用（与 GAE 的常规做法相同，不做重要性修正）。所有估计函数在 `torch.no_grad()` 下运行。

## 5. 特权 critic

```
ctx = MLP(history ⊕ privileged)                                  [..., H]
Q(s, a) = <f(ctx), g(cand_a)> / √H + b(cand_a)，非法候选置 0       [..., A]
V(s)    = MLP(ctx)                                               [...]
```

- `history`：决策点的历史条件特征（事件流 Transformer / LSTM 的输出，actor 也能看到的那部分）。I9 要求 critic 必须看历史，
  只看特权状态有偏。
- `privileged`：对手真值（`EncodedEvent.privileged`，[encoding.md](encoding.md)「训练态真值」）编码成的特征向量；
  由网络侧（T4b.1 / T4c）负责把 int32 表编码成向量。**只进 critic 与信念损失，从不进 actor**；
  `Critic(privileged_dim=0)` 是非特权 critic，供 T4c.2 消融。构造时声明了特权维度却不给（或反之）直接报错。
- `candidates`：候选动作嵌入（与 actor 打分头用的同一组行），Q 头和 actor 一样是点积形状。
- `squash=True` 时 Q、V 过 tanh，落在终局奖励的 [−1, 1] 区间。

损失：`q_loss(q, actions, q_targets, mask) = mean_mask ½ (Q(s_t, a_t) − G_t)²`，只对选中的候选回传梯度；
`v_loss(v, v_targets, mask)` 同理。`mask` 可以是 bool（valid / 训练行）或浮点权重。

## 6. 测试

`tests/test_advantages.py`（`pytest.importorskip("torch")`，CPU，约 3 秒）。玩具博弈有限且无环：Q^π / V^π 由 Bellman 线性方程
`(I − γM)V = b` 精确求出，所有对局连同概率全部枚举，期望是精确加权和而不是采样估计。

- GAE：λ = 1 等于蒙特卡罗回报减 V、λ = 0 等于 TD 误差（对照朴素的逐座位参照实现，随机交替座位、段中终局、段尾自举）；
  真值 V 下 `E[A | s, a] = Q^π − V^π`。
- Expected-SARSA(λ)：真值 Q 下确定转移逐样本等于 Q^π、随机转移期望等于 Q^π；λ = 0 / λ = 1 两个极限；一般 λ 的递推恒等式。
- VRPO：两种模式都与解析的 Q^π − V^π 一致；逐候选形式；方差对比。
- 两人交替：期望最小最大递归与「贪心策略 + 线性方程」给出同一组价值；蒙特卡罗回报按座位取反；多步决策（同座位连续行）。
- 非法候选的 Q / π 取任意值（1e6、残余概率质量）结果不变；多局打包在同一列、段尾截断、尾部填充与逐局单独计算一致；布局错误报错。
- critic：形状、掩码、特权输入开关、损失只作用于选中动作与有效行；在随机转移链上用 Expected-SARSA(λ) 目标做拟合策略评估，
  Q / V 头收敛到 Q^π / V^π（误差 < 0.02）。

## 7. 限制与后续

- 列中间的截断（上限 / 错误）用 `truncated` 表达（§1），不需要拆列；它以 critic 在截断行的估计自举，而不是真正的下一状态。
- 不做离策略修正（重要性比 / Retrace 的 `c = λ min(1, ρ)`）：PPO 的数据接近同策略；快照池复用旧数据时再加，接口位置在 `λ · c_t`。
- 特权特征的编码器、critic 与 actor 是否共享主干及其梯度隔离，属于网络侧（T4b.1 / T4b.2）与 T4c.2 消融。

## 8. PPO 自博弈训练循环（T4b.4）

一次迭代 = 收集一段 rollout → 一次 PPO 更新 → 联赛簿记（快照、checkpoint、定期评估与 keep-best）。

```python
from ygorl.train.trainer import TrainConfig, Trainer

cfg = TrainConfig(decks=("tests/decks/snake_eye.ydk", "tests/decks/kashtira.ydk"), num_envs=32, steps=64)
trainer = Trainer(cfg, "out/train/run1")
trainer.train(max_minutes=60)                         # 或 max_updates=N
Trainer.resume("out/train/run1/checkpoints/latest.pt").train(max_minutes=60)   # 续训
```

命令行：`uv run python tools/train_ppo.py DECK... [--minutes 60] [--updates N] [--out DIR]`，`--resume CKPT` 续训，
`--summary RUN/metrics.jsonl` 打印首末 10 次更新的平均指标；参数见 `--help`。

### 8.1 收集（`RolloutCollector`）

- 环境是 `EncodedVecEnv(privileged=True)`（训练态：事件带对手真值，只进 critic）。每个环境槽位是一列，每次 `collect()`
  每列收 `T = steps` 行，得到 §1 布局的 `Rollout`（外加 `log_probs` 行为对数概率、`truncated`、结束的对局列表）。
- 每局开局由 `next_game() -> Assignment` 决定：`spec`（`GameSpec`）、`opponent`（None = 当前策略自博弈；否则快照 id）、
  学习方座位。**自博弈**局双方的每个决策都是一行；**快照对手**局只收学习方的行，对手的决策用快照的 `policy_logits` 采样、
  计入 `Rollout.opponent_decisions`，属于「环境」（§1 的推荐做法）。快照在开局时解析并随局保存，局中被逐出池也不影响。
- 就绪的决策按「谁来下」分组、每组一次前向：学习方组同时得到 logits、Q、V（行为策略与 critic 值在动作时记录，
  不再重算）；动作从 softmax 采样（可复现的 `torch.Generator`）。
- 终局：`reason ∈ {turn_limit, decision_limit, error}` 为截断（§1），其余按胜负给最后一行奖励；错误事件照常记账、槽位立刻开新局，
  不会中断训练。
- 段尾：一列满 `T` 行后，其环境在下一个**学习方**决策上暂停（对手的决策继续推进），这个待答决策就是该列的自举状态
  （`bootstrap_*` 由它前向得到），下一次 `collect()` 从它继续。所有列都暂停时这一段结束；先满的列等其余列，代价是少量空转。

### 8.2 更新（`PPOLearner`）

1. `estimate(cfg.estimator, ..., truncated=...)` 算一次目标（默认 VRPO，`estimator="gae"` 为对照），优势按
   `adv_norm`（默认 `standard`）在整段上归一化。
2. 慢速参考策略（学习方参数的 EMA，设计 I8 的 MMD / R-NaD 简化版）与可选的 BC 先验对全段各打一次分（不求导）。
3. `epochs` 轮、每轮打乱后按 `minibatch_size` 切批：

```
loss = L_policy                                    （可插拔，默认 ppo_clip：−mean min(ρA, clip(ρ, 1±ε)A)）
     − entropy_coef · H(π)                         （合法候选上的熵，设计 I1：0.05–0.2，默认 0.05）
     + kl_ref_coef  · KL(π ‖ π_ref)                （默认 0.05）
     + kl_prior_coef · KL(π ‖ π_BC)                （给了 --bc-prior 时，默认 0；kl_prior_turns > 0 时只在部分行上，见下）
     + q_coef · L_Q + v_coef · L_V                 （critic.q_loss / v_loss，默认各 0.5）
```

   梯度裁剪到 `max_grad_norm`（0.5），Adam（lr 1e-3；默认 4 轮 × 256 行的批，每次更新 32 步，见下「步长」）。KL 与熵只在合法候选上求和（掩码 logit 为 −1e9，概率严格为 0）。
   **热启动与先验**：`--init-from CKPT`（`TrainConfig.init_from`）把 actor 设成某个检查点的网络、critic 从头训；`--bc-prior CKPT`
   给 `π_BC`。两者都接受 PPO 训练的 checkpoint 或 BC 等导出的策略检查点（`ygorl.nets.agent`，[bc.md](bc.md)），按文件的 `format`
   字段区分（`train.checkpoint.load_actor`）。热启动时本次运行沿用检查点的卡片词表，网络配置（`d_model`、层数、历史模块等）必须与
   `--d-model` 等参数一致，否则报错；先验的词表必须与本次运行相同（同一下标要是同一张卡），网络大小可以不同。
   **只在第 1 回合用先验**：`--kl-prior-turns N`（`PPOConfig.kl_prior_turns`，默认 0 = 所有行）只对「回合玩家自己的决策、回合数 ≤ N」
   的行（观测 `globals` 的 `is_my_turn` 与 `turn` 两列）计先验 KL，其余行按 0 计入同一个均值（被选中的行权重与不限制时相同）；
   日志多一个 `kl_prior_rows`（本段被选中的行数）。`N = 1` 即求解器示范覆盖的先攻第 1 回合，理由与对比实验见 [bc.md](bc.md)「补救实验」。
   **按 KL 提前停**：`--target-kl X`（`PPOConfig.target_kl`，默认 0.01；`--target-kl 0` 关闭）——某个 minibatch 的 `approx_kl` 超过 1.5 X 时，本次更新余下的
   minibatch 都跳过（日志 `minibatches` / `early_stop`）。步长按「策略实际移动了多少」封顶，而不是按固定的轮数：同一组学习率与轮数，
   从零开始（熵约 1.3）每次约 0.006–0.009，从 BC 热启动（熵约 0.7）则到 0.023–0.029、裁剪比例约 0.2（[benchmarks.md](benchmarks.md)）。
4. `reference ← (1 − τ) reference + τ θ`，`τ = reference_ema`（默认 0.02 / 次更新）。

**策略目标可插拔**：`PolicyObjective` 有两个钩子——`prepare(rollout, estimate) -> [T, B]` 在整段上算每行权重
（默认就是估计器的优势），`loss(ObjectiveInputs) -> (loss, stats)` 在每个 minibatch 上算策略损失（输入：当前 log π、
动作、行为 log π、权重、Q 目标、掩码、行号）。`register_objective(name, factory)` 注册，`PPOConfig.objective` 选择；
内置 `ppo_clip`（默认）与 `pg`（无比值、无裁剪的同策略梯度，消融用）。MaxRL 式目标（[eng-plan.md](eng-plan.md)「M4 待议的消融」：
同一起点多次 rollout 估计成功概率的对数梯度）只需在 `prepare` 里按起点分组算权重、在 `loss` 里用它，收集器、联赛、critic 不变。

日志指标（每次更新）：`loss`、`policy_loss`、`entropy`、`kl_ref`、`kl_prior`、`q_loss`、`v_loss`、`approx_kl`、`clip_frac`、
`grad_norm`、`q_explained_var`（行为时 Q 对 Q 目标的解释方差）、`adv_mean` / `adv_std`、`minibatches` / `early_stop`；
诊断用的按动作类型拆开的优势：`adv/<kind>`（选中动作属于该类型的行的平均优势，类型见 `ACTION_KINDS`）与 `rows/<kind>`（行数）——
例如检验「搜索 / 抽卡类动作被系统性地判为负优势」（局面靠卡组耗尽决胜时的现象）。

### 8.3 联赛与牌组池（`selfplay`）

- **发局**（`SelfPlaySchedule`）：每次成对发局——一个种子、一组对阵、一个对手，先后攻各一局（先后攻配平）。
  以 `selfplay_fraction`（默认 0.75）的概率是当前策略自博弈，否则从快照池均匀抽一个对手、学习方随机坐一侧。
- **牌组池**（`DeckPool`）：牌组列表 + 对阵集合，`all`（全部有序对，含镜像）、`cross`（不同牌组，默认）、`mirror`，
  或显式的下标对；均匀抽（按 meta 份额加权留给 T4d.1）。单牌组对实验给两套牌、`cross` 即可（两个方向都打）。
- **快照池**（`SnapshotPool`）：每 `snapshot_every` 次更新冻结一份当前模型，超过 `pool_size` 逐出最旧的；
  **keep-best**：每 `eval_every` 次更新评估一次，对 `keep_best_by`（默认 greedy）的胜率创新高就把该 checkpoint 复制为
  `best.pt`，并把这份模型钉在池里（不被逐出，直到更好的替换它）。
- **PFSP 与入池门槛**（#61，默认关闭）：池记录学习方对每个快照的结果（只计非截断的池局，`learner_win_rate` 带一局 0.5 的先验）。
  `pool_sampling="pfsp"`（`--pool-sampling pfsp`）按 `(1 − p) ** pfsp_power`（默认 2，AlphaStar 的 hard 权重）抽快照，学习方打不过的对手更常出现；
  `snapshot_min_win_rate`（`--snapshot-min-win-rate 0.55`，ygo-agent 的 OSFP）让到期的快照只在「上次入池以来学习方在池局里的得分 > 门槛、且至少
  `snapshot_min_games` 局」时入池，否则跳过并计入 `snapshots_skipped`；池空时总是入池。入池以来的窗口（`league`）随 checkpoint 保存。
- **固定对手**（`TrainConfig.pin_opponents` / `--pin CKPT`，可重复）：把策略检查点（例如 BC 热启动用的那份）或 PPO checkpoint 整局钉在池里，
  不被逐出、不进 checkpoint（续训时按配置重新钉上，id 为 −1、−2……）；池局里固定对手合占 `pinned_share`（默认 0.5），其余均分给快照。
  词表与事件窗口长度必须与本次运行相同。用意：自博弈早期的历史快照都接近随机，固定一个会进战斗、会展开的对手让信号更有用
  （Gin Rummy 2026「更强对手的课程」、cjiang1209/yugioh-agent 默认对 greedy 训练）。

### 8.4 评估

`evaluate_checkpoint`：把 checkpoint 作为 `policy:<路径>` agent，在训练对阵上对每个基线（默认 greedy、random）各打
`eval_pairs` 对配对种子（每对先后攻各一局），走 `Arena`（与 `ygorl arena` 同一条路径，同一个评估种子，结果可跨 checkpoint 比较）。
策略 agent 如何在 Python `Duel` 上跑 C++ 编码的观测见 [evaluation.md](evaluation.md)「策略检查点 agent」。
评估期间训练暂停；PyTorch 已加载时 Arena 用 `spawn` 起子进程（fork 已初始化的 PyTorch 会让子进程死锁）。

### 8.5 checkpoint、续训与目录

一个 checkpoint 是一个自包含的 `.pt`（`torch.save` 普通容器，`weights_only=True` 可读）：`config`（`TrainConfig`，含
`PPOConfig`）、`net_config`、`vocab`（`CardVocab.save` 写出的 JSON 原文——词表下标是模型的一部分）、`environment`
（训练环境的 `stamp()`，没有环境时为 None）、`learner`（模型、EMA 参考、优化器、更新数）、`pool`（全部快照含 keep-best）、
`schedule`（发局计数与 RNG）、`counters`、`best`、`rng`。`load_policy(path)` 只重建 actor（`PolicyNet`）。
续训恢复以上全部状态；进行中的对局不保存，续训时各槽位开新局。

运行目录（`tools/train_ppo.py` 默认 `out/train/<时间戳>`；给 `--env` 时是 `environments/<版本>/artifacts/train/<名字>`，
产物绑定环境版本）：

| 文件 | 内容 |
|------|------|
| `config.json` / `vocab.json` | 训练配置；`CardVocab.save` 的词表 |
| `metrics.jsonl` | 每次更新一行：更新号、行数、决策数、收集 / 更新秒数、决策/秒、行/秒、结束局数、自博弈 / 快照局数、对快照胜率、自博弈先攻胜率、截断与错误局数、终局原因、平均局长，§8.2 的损失指标，累计计数 |
| `eval.jsonl` | 每次评估每个基线一行：局数、胜 / 负 / 平、胜率与 Wilson 区间、终局原因、耗时、是否新 best |
| `checkpoints/latest.pt`、`checkpoints/update_N.pt`、`best.pt` | 最新（每 `checkpoint_every` 次）、每次评估的、keep-best |

### 8.6 默认规模与吞吐

`TrainConfig` 默认是面向 4 核 CPU 的小网络：`d_model = 64`、4 头、局面 / 历史各 1 层、事件窗口 64、ID 嵌入开
（actor 136 万参数，其中卡片 ID 嵌入 94 万）、特权编码 64 维、critic 隐层 128，critic 与 actor 共享主干；32 个槽位 × 64 行 =
每次更新 2,048 行。4 核云容器上（与其他任务共享 CPU，PyTorch 2 线程）收集约 670 决策/秒（含推理），更新是瓶颈
（前向 + 反向约 3–4 ms/行，卡片 ID 嵌入的稠密梯度占相当一部分；每次更新约 21 秒、收集约 3.4 秒），整体约 84 行/秒。

**T4b.4 验收**（[benchmarks.md](benchmarks.md)「PPO 自博弈：单牌组对 1 小时」）：snake_eye 对 kashtira 训练 60 分钟、140 次更新、
286,720 行、256 局，**0 崩溃、0 引擎错误**，内存平稳；玩具环境（Nim）上 PPO 在数秒内收敛到最优策略（§8.7）。
1 小时的 CPU 训练对 random 0.537（0.429–0.643）、对 greedy 0.150（0.088–0.244），与未训练策略（0.475 / 0.113）的区间重叠——
还谈不上学会；原因与调参方向见下。

**GPU**（`TrainConfig.device` / `--device cuda`，ROCm 也叫 `cuda`）：学习器、行动网络、BC 先验与钉住的对手都放到该设备；
收集的整段观测在设备上拼好，优势估计仍在 CPU 上算（逐行数据小），checkpoint 一律按 CPU 加载。ROCm 上权重梯度走 `ygorl.nets.gemm` 的按 `K` 切分，
并自动打开融合注意力（原因与实测见 [benchmarks.md](benchmarks.md)「GPU 学习器」）。两个**实验开关**，默认关闭：

- `overlap_collect`（`--overlap`）：更新进行时在另一个线程（GPU 上另一个 stream）用「更新开始前的权重」的副本收集下一段；联赛记账
  （快照、评估、checkpoint）只在两步之间、没有收集在跑时进行。代价是每段数据落后一次更新，偏离设计的同步 PPO（[scaling.md](scaling.md) S3），采用前要先改设计。
- `bf16`（`--bf16`）：行动与更新都在 bf16 autocast 下跑；GPU 上更新约快 10–20%，CPU 上更慢。

`tools/bench_train.py` 测 `Trainer.step()` 的收集 / 更新耗时、行/秒与（AMD GPU 上的）忙碌率。

### 8.7 测试

- `tests/test_ppo.py`（玩具博弈，约 15 秒）：Nim 自博弈在 VRPO 与 GAE 下都收敛到最优策略（每个必胜局面取 `n mod 4`），
  对快照池训练同样收敛；收集器布局（交替座位、终局奖励、段间衔接自举状态）、快照局只含学习方的行、截断局标记且无奖励、
  引擎错误事件（开局失败 / 局中错误）记为截断并立即开新局；默认超参数符合设计（熵系数在 0.05–0.2）；EMA 参考；
  KL 为 0 与梯度方向；先验 KL 只在第 1 回合（`kl_prior_turns`）时的行选择、权重与更新结果；学习器状态往返；快照池逐出与 keep-best；策略目标可插拔（注册自定义目标，`prepare` / `loss` 被调用）。
- `tests/test_advantages.py::test_truncated_rows_bootstrap_from_the_critic`：截断行的目标等于 critic 自身估计、与「切列 + 同座位自举」一致、
  截断行的奖励被忽略。
- `tests/test_train_loop.py`（真实对局，约 15 秒）：`EncodedVecEnv` 上自博弈布局（双方交替、决策上限截断在第 30 行、特权真值只进
  critic 批）、快照局只留学习方的行；`Trainer` 写出指标 / 评估 / checkpoint / `best.pt`，词表随 checkpoint 保存；
  checkpoint 往返与续训（模型、参考、优化器、快照池、计数一致，续训后更新号接续）；`load_policy` 的 actor 与训练模型打分一致；
  `policy:` agent 在 `Duel.run` 上合法落子，且每个决策的观测与 `EncodedVecEnv` 重放同一局时逐元素相同；并行 Arena
  与单进程结果一致；`ygorl duel --agent-a policy:...`；`tools/train_ppo.py` 训练、续训、汇总。

### 8.8 限制与后续

- **步长**（T4b.5 起）：T4b.4 的 1 小时实验每次更新只有 4 次 Adam 步（2 轮 × 2 个 1,024 行的批），`approx_kl` 约 1e-5、
  裁剪比例约 0，策略几乎不动。Adam 每步把每个参数挪动约一个学习率，与梯度大小无关，所以步数与学习率决定每次更新走多远。
  扫描后默认改为 lr 1e-3、4 轮 × 256 行（每次更新 32 步）：`approx_kl` 约 5e-3–9e-3、裁剪比例约 0.08，落在 PPO 常见的
  1e-3–1e-2 区间（[benchmarks.md](benchmarks.md)「PPO 步长」）。同样 1 小时，更新次数翻倍（281 次），对 greedy 从 0.05 升到
  0.188（区间 0.117–0.287，与未训练不重叠），但几乎全部来自驾驶 kashtira；驾驶 snake_eye（长 combo）基本没学到。
- 从零训练在 CPU 预算内进展很小，原因、先前项目的做法与扩规模 / 训练信号的选项见 [scaling.md](scaling.md)（#60、#61、#62）。
- 更新是瓶颈：CPU 上 2,048 行 × 2 轮约 20–25 秒，收集只占约 10%。GPU 已接入（上文「GPU」，Radeon 8060S 上同算法 2.35 倍、
  叠加实验开关与更多并行局 1.7 倍，[benchmarks.md](benchmarks.md)「GPU 学习器」）；收集与更新默认仍串行（同步 PPO）。
  再往上是稀疏 / 行级的 ID 嵌入更新与减少 kernel 数（[scaling.md](scaling.md) S5）。
- 进行中的对局不进 checkpoint；续训时槽位重新开局（少量半局数据丢弃）。
- 评估走 Python `Duel` + 锁步 C++ 主机，单局推理批量为 1，比训练路径慢；每次评估的局数因此较少（区间宽），正式对比用
  `ygorl arena` 多打。
- 课程模式（T2.6）与中局开局在 C++ 步进路径上还不支持（`EncodedVecEnv` 报 `NotImplementedError`），T4d.1 前需要移植。
