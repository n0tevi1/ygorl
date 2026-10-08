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
| `ygorl.train.checkpoint` | checkpoint 读写；所有网络都在这里重建（见 §8.5「从检查点重建网络」）：`load_actor` / `load_policy`（只重建 actor，供对战 / 评估）、`load_actor_critic`（整个 actor-critic 连同 critic 选项）、`build_actor_critic`、`warm_start`，以及 `Signature`（词表 + 事件窗口长度）的兼容性检查 |
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
  重开失败报回的错误事件、引擎步数上限的「engine loop」、脚本指令预算的「script budget」，[engine.md](engine.md)）结束的局**不是胜负**，是截断：最后一个 agent 行 `done = True`、`truncated = True`、奖励 0。
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

**critic 目标的 λ**（`target_lam` / `PPOConfig.critic_lam` / `--critic-lam`，默认与 `lam` 相同）：只改 Q / V 头的目标，优势仍用 `lam`。
λ = 0.5 让策略梯度的噪声小（设计 I2），但同一个 λ 也让 critic 的目标大半是它自己下一步的估计：训练目标上的解释方差约 0.96，
对真实胜负却只有约 0.3（[benchmarks.md](benchmarks.md)「critic 的改进：预测与强度」）；几个回合后才起作用的信息（例如未来的抽卡）几乎传不到 critic。
λ = 1 仍是带控制变量的 Expected-SARSA 回报：递推中保留后续 `V̄ − Q_taken`，rollout 段末仍自举，**不是完整终局 Monte Carlo 目标**。
例如两步同座位轨迹奖励 `[0, 1]`，第二步两个动作的 Q 为 `[1, -1]`、等概率且实际选到 +1，λ = 1 的目标为 `[0, 1]`，纯 MC 则为 `[1, 1]`。
**该 λ = 1 设置实测更差**：400 次更新后对真实胜负的解释方差为 0.18–0.24，旧参考小组得分落后，因此默认保持与 `lam` 相同。
这不排除完整终局监督或动作价值排序校准的收益；不能把 TD 目标 EV 当作真实胜负预测准确率，也不能把预测动作差的方差占比叫作真实信号比例。

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
- **卡组顺序**（`TrainConfig.critic_deck_order` / `--critic-deck-order`，默认关）：特权编码器另读双方接下来的 10 张抽卡
  （`my_next` / `op_next`），每张的卡片嵌入加上「距卡组顶深度」嵌入，按抽卡顺序拼接。用意：未来抽卡是胜负方差里最大的一块，
  critic 知道它就能把这部分运气从优势里扣掉。风险：critic 依赖 actor 永远不可能知道的信息，状态 critic 在部分可观测下的偏差会变大，
  是否值得以策略对局矩阵的实测为准。
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

**配置即命令行**（`ygorl.train.cli`，#121）：`TrainConfig` / `PPOConfig` 的每个字段恰好对应一个参数，每个选项只声明一次——
类型与默认值取自配置 dataclass（或工具给的基础配置），`cli.FLAGS` 表只补命令行独有的信息：参数名、帮助、取反（`--no-privileged` →
`privileged_critic=False`、`--separate-critic` → `shared_backbone=False`、`--keep-forced` → `skip_forced=False`）、可选值、
「0 表示 None」（`--target-kl 0`、`--stall-timeout 0`）、逗号分隔（`--eval-opponents`）或可重复（`--pin`）。网络开关写进
`TrainConfig.net`（NetConfig 覆盖项，只写与默认不同的键；`--layers` 同时设 `board_layers` 与 `history_layers`，`--no-text` 同时关
`card_text` 与 `effect_text`）。接口是三个函数：`add_arguments(parser, base)` 建参数组、`from_args(args, decks, base)` 得到配置、
`to_argv(cfg, base)` 反过来给出重建该配置的参数（与 `base` 相同的字段省略；命令行表达不了的值报错）。牌组是工具的位置参数，不是 flag。
`tools/train_ppo.py` 与 `tools/bench_train.py` 都经这条路径建配置（后者的基础配置是 `--envs 32 --steps 64 --minibatch 256`、
评估 / 快照 / checkpoint 关闭）。参数名与默认值和以前一致，`config.json` / checkpoint 里的字段名不变；以前解析了却没传进配置的
`--overlap`、`--bf16` 现在生效，以前没有参数的字段（`--min-batch`、`--privileged-dim`、`--critic-hidden`、`--stall-timeout`、
`--eval-greedy-policy`、`--gamma`、`--clip`、`--q-coef`、`--v-coef`、`--adam-eps`、`--max-grad-norm`、`--adv-norm`）有了参数。
新增配置字段须在 `cli.FLAGS` 补一行，否则 `tests/test_train_cli.py` 失败。

### 8.1 收集（`RolloutCollector`）

- 环境是 `EncodedVecEnv(privileged=True)`（训练态：事件带对手真值，只进 critic）。每个环境槽位是一列，每次 `collect()`
  每列收 `T = steps` 行，得到 §1 布局的 `Rollout`（外加 `log_probs` 行为对数概率、`truncated`、结束的对局列表）。
- 每局开局由 `next_game() -> Assignment` 决定：`spec`（`GameSpec`）、`opponent`（None = 当前策略自博弈；否则快照 id）、
  学习方座位。**自博弈**局双方的每个决策都是一行；**快照对手**局只收学习方的行，对手的决策用快照的 `policy_logits` 采样、
  计入 `Rollout.opponent_decisions`，属于「环境」（§1 的推荐做法）。快照在开局时解析并随局保存，局中被逐出池也不影响。
- 就绪的决策按「谁来下」分组、每组一次前向：学习方组同时得到 logits、Q、V（行为策略与 critic 值在动作时记录，
  不再重算）；动作从 softmax 采样（可复现的 `torch.Generator`）。
- 收集器有自己的调度循环，不走整局对弈的对局驱动（`ygorl.env.driver`，见 [evaluation.md](evaluation.md)「批量评估」）：
  列满 `T` 行时要扣住待答决策、还要做卡死检测，这两件事是收集特有的。快照对手与 KL 参考的 logits 与批量评估共用
  `nets.batch.policy_logits`。
- 终局：`reason ∈ {turn_limit, decision_limit, error}` 为截断（§1），其余按胜负给最后一行奖励；错误事件照常记账、槽位立刻开新局，
  不会中断训练。
- 段尾：一列满 `T` 行后，其环境在下一个**学习方**决策上暂停（对手的决策继续推进），这个待答决策就是该列的自举状态
  （`bootstrap_*` 由它前向得到），下一次 `collect()` 从它继续。所有列都暂停时这一段结束；先满的列等其余列，代价是少量空转。

**卡死保护**：`RolloutCollector` 带超时等事件；连续 `TrainConfig.stall_timeout`（默认 900 秒）没有任何环境产生事件、而仍有环境欠着行时，
抛 `RuntimeError`，列出沉默最久的环境及其对局（种子、先攻方、牌组），而不是永远等下去（环境 i 固定在第 i % T 个线程上，一个卡死的对局会连带卡住同线程的环境）。
`Trainer.train` 先存 `checkpoints/latest.pt` 再抛出 `RolloutStalled`；卡住的工作线程让环境池无法正常析构（析构要等所有线程结束），所以 `tools/train_ppo.py` 捕获后用 `os._exit(3)` 结束进程（直接调用 `Trainer` 的代码也应如此），之后可 `--resume` 续训。引擎错误截断的每一局追加到运行目录的 `errors.jsonl`（种子、先攻方、牌组、错误、十六进制应答日志），可据此复现。

训练限时/决策上限另存`truncations.jsonl`，健康异常仍存`errors.jsonl`（即使最终reason为win）。
两者包含完整`spec`、环境stamp、`skip_forced`和跨更新累计的`action_indices`，以及原始response/健康/终局字段。
动作索引需经相同版本的EncodedVecEnv、相同skip_forced设置重放；只用seed与最近checkpoint不能重现局中变化的策略。
记录不改变采样、奖励或终局计数。正常终局不导出完整trace；独立EncodedVecEnv默认关闭记录，Trainer启用。
旧来源未收集动作时该字段为null，不伪装为可复现的空序列；运行中的旧实验不会被补写不存在的动作日志。
已知限制：若每一局都在第一个决策之前出错，收集器会不停开新局而凑不满一列（事件一直有，看门狗不触发）；只在引擎整体损坏时出现。

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
   `--d-model` 等参数一致，只允许多出卡片视角（`train.checkpoint.warm_start`，[nets.md](nets.md)「卡片事实」），否则报错；
   先验的词表必须与本次运行相同（同一下标要是同一张卡；`Signature.mismatches(..., event_length=False)`），网络大小可以不同。
   **只在第 1 回合用先验**：`--kl-prior-turns N`（`PPOConfig.kl_prior_turns`，默认 0 = 所有行）只对「回合玩家自己的决策、回合数 ≤ N」
   的行（观测 `globals` 的 `is_my_turn` 与 `turn` 两列）计先验 KL，其余行按 0 计入同一个均值（被选中的行权重与不限制时相同）；
   日志多一个 `kl_prior_rows`（本段被选中的行数）。`N = 1` 即求解器示范覆盖的先攻第 1 回合，理由与对比实验见 [bc.md](bc.md)「补救实验」。
   **按 KL 提前停**：`--target-kl X`（`PPOConfig.target_kl`，默认 0.01；`--target-kl 0` 关闭）——某个 minibatch 的更新前 `approx_kl` 超过 1.5 X 时，该批及本次更新余下的
   minibatch 都跳过（日志 `minibatches` / `early_stop`）。这是按已测策略变化提前停止的启发式，不保证KL硬上界；刚接受的一步仍可超限：同一组学习率与轮数，
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

**critic 预热**（`--critic-warmup N`，`--critic-warmup-ev EV`，默认关闭）：从 BC 或只含策略的检查点热启动时 critic 是新的，
优势会依赖尚未校准的critic估值，可能造成不利的策略漂移；这需要受控干预验证，不能从预测量的相关性直接断言是噪声。
2026-09-26 宽度128的历史运行中，种子1约第1,000次更新的Q解释方差达到0.66；此前熵涨到1.2、
平均对局27回合，2,000次时对Greedy为0.30，种子0更早达到类似自举目标拟合指标。
这里的Q解释方差是对当前λ自举目标的拟合，**不等于真实终局预测准确性或动作排序质量**，也没有证明预热能改善强度。
[首步诊断](spikes/ppo-first-step-2026-10-05.md)支持继续调查critic经advantage的间接路径，保留warmup为默认关闭的实验选项。
- 开启后，前面的更新只训 critic：损失只有 Q / V 两项，反向后清掉策略参数（`model.actor`，含 critic 共用的主干）的梯度再步进。
  Adam 跳过没有梯度的参数，所以策略逐位不变（测试）。
- 结束条件：Q 解释方差最近 5 次的平均达到 `EV`（默认 0.6），或满 `N` 次更新，二者先到为准。之后照常训练。
- 预热的更新也计入更新数与数据量。结束时的更新数记在 `counters["critic_warmup_done"]`，续训不会重新预热。
- 每次更新的统计里有 `critic_warmup`（1 = 这次只训了 critic）。

**按回合折扣**（`--turn-discount G`，默认 1.0 = 关闭；快赢压力 T6，与设计 C3 冲突，目前只作诊断臂，见 [spikes/reward-signal.md](spikes/reward-signal.md) R5）：
- 分出胜负的对局，终局奖励从 ±1 改为 ±G^回合数：早赢奖励大、晚赢奖励小；输的一方拖得越久，负奖励的绝对值越小。截断局仍为 0。
- 与「每过一回合折扣一次」只差按行所在回合 t 的系数 G^t。这个系数对同一局面的所有候选动作相同，不改变该局面上的梯度方向，只改变不同回合之间的权重。
- 按回合而不按决策计，所以不惩罚一个回合内几十步的展开。

### 8.3 联赛与牌组池（`selfplay`）

- **发局**（`SelfPlaySchedule`）：每次成对发局——一个种子、一组对阵、一个对手，先后攻各一局（先后攻配平）。
  以 `selfplay_fraction`（默认 0.75）的概率是当前策略自博弈，否则从快照池均匀抽一个对手、学习方随机坐一侧。
- **牌组池**（`DeckPool`）：牌组列表 + 对阵集合，`all`（全部有序对，含镜像）、`cross`（不同牌组，默认）、`mirror`，
  或显式的下标对；均匀抽（按 meta 份额加权留给 T4d.1）。单牌组对实验给两套牌、`cross` 即可（两个方向都打）。
- **快照池**（`SnapshotPool`）：每 `snapshot_every` 次更新冻结一份当前模型，超过 `pool_size` 逐出最旧的；
  **keep-best**：每 `eval_every` 次更新评估一次，对 `keep_best_by`（默认 greedy）的胜率创新高就把该 checkpoint 复制为
  `best.pt`，并把这份模型钉在池里（不被逐出，直到更好的替换它）。
- **PFSP 与入池门槛**（#61）：池记录学习方对每个快照的结果（只计非截断的池局，`learner_win_rate` 带一局 0.5 的先验）。
  **默认** `pool_sampling="pfsp"`（设计 I1；`--pool-sampling uniform` 为旧的均匀抽样）按 `(1 − p) ** pfsp_power`（默认 2，AlphaStar 的 hard 权重）抽快照，学习方打不过的对手更常出现；
  `snapshot_min_win_rate`（`--snapshot-min-win-rate 0.55`，ygo-agent 的 OSFP；默认关闭，消融里没有帮助）让到期的快照只在「上次入池以来学习方在池局里的得分 > 门槛、且至少
  `snapshot_min_games` 局」时入池，否则跳过并计入 `snapshots_skipped`；池空时总是入池。入池以来的窗口（`league`）随 checkpoint 保存。
- **进化牌组池**（`EvolvedDecks`，#108，spec #105）：`--deck-pool MANIFEST`（`TrainConfig.deck_pool`）指向进化进程维护的牌组池清单：

  ```json
  {"format": "ygorl-deck-pool", "version": 1,
   "decks": [{"id": "evo-0001", "file": "decks/evo-0001.ydk", "status": "probation", "weight": 1.0, "parent": "..."}]}
  ```

  `file` 相对清单所在目录，`id` 唯一、同时作为日志里的牌组名；`status` 为 `probation` / `active`（作为进化一方发出）或 `history`（被取代的精英，只作对手卡组）；
  `weight` ≥ 0，默认 1；其余字段（谱系、描述符，以及进化步骤写的顶层 `environment` 环境戳，见 [tuning.md](tuning.md)「进化步骤」）归进化进程所有，训练器不读。训练器每 `deck_pool_every`（默认 10）次更新在两次更新之间重读清单与它引用的 `.ydk`，
  内容变了才重建（清单相对路径在命令行里先解析成绝对路径）。清单由另一个进程在训练中写，所以**清单的问题不会中断训练**：读不了或写到一半的清单保留上一版牌组池；
  不合法（环境下逐套检查）、读不到文件、id 重复、状态或权重不对的条目留在外面；两者都记入日志。
  发局时以 `evolved_share`（默认 0.3）的概率抽一套 probation / active 卡组，权重 = `weight × (1 − p) ** evolved_power`（默认 1），
  `p` 是当前策略驾驶它的近期得分（每局折扣 0.99，约最近 100 局；带一局 0.5 的先验；自博弈局双方都算学习方，池局只计学习方那一侧），它与一套语料或历史卡组均匀配对、随机坐 a / b 一侧；
  否则照常从 `DeckPool` 抽。**没有 probation / active 卡组（或占比为 0）时不多抽一个随机数**，所以与固定牌组池逐位相同。
  每局 `info` 带 `evolved` 与 `evolved_seat`；`metrics.jsonl` 多一列 `evolved_games`。池（条目、各卡组的卡表、得分）随 checkpoint 保存（`evolved`），
  续训不需要清单与卡组文件还在，先恢复保存时的池，到下一个重读点再读当前文件。
- **固定对手**（`TrainConfig.pin_opponents` / `--pin CKPT`，可重复）：把策略检查点（例如 BC 热启动用的那份）或 PPO checkpoint 整局钉在池里，
  不被逐出、不进 checkpoint（续训时按配置重新钉上，id 为 −1、−2……）；池局里固定对手合占 `pinned_share`（默认 0.5），其余均分给快照。
  词表与事件窗口长度必须与本次运行相同（`Signature.mismatches`，报错写明哪里不同）。用意：自博弈早期的历史快照都接近随机，固定一个会进战斗、会展开的对手让信号更有用
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
`schedule`（发局计数与 RNG）、`counters`、`best`、`rng`、`evolved`（进化牌组池的条目、卡表与逐卡组得分，没有时为 None）。
续训恢复以上全部状态；进行中的对局不保存，续训时各槽位开新局。
`rng`包含CPU torch流和独立的CPU collector generator；GPU训练还保存learner设备的CUDA/ROCm全局流，
供PPO minibatch randperm等使用，在模型和快照池重建后恢复。CPU checkpoint不访问CUDA RNG。
旧GPU checkpoint没有该字段时仍可载入，但无法恢复缺失的GPU随机历史；跨设备迁移不保证相同随机序列。
对局重开和GPU数值波动仍可能改变轨迹，恢复RNG不代表整段续训逐位相同。

**从检查点重建网络**（#122）：训练器、对战 agent、评估、智能体矩阵与研究脚本都经 `ygorl.train.checkpoint` 重建网络，不各自拼装：

| 函数 | 作用 |
|------|------|
| `load_actor(path, text_dir=None)` | 任一检查点（PPO 或 BC 等策略检查点，按 `format` 区分）的 actor（`PolicyNet`，eval、无梯度、CPU）、词表、事件窗口长度；`load_policy` 同上但只收 PPO 检查点 |
| `load_actor_critic(path, text_dir=None)` | PPO 检查点的整个 `ActorCritic`，critic 选项（`privileged`、`privileged_dim`、`critic_hidden`、`shared_backbone`、`critic_deck_order`）取自该次运行的配置（`CriticConfig`）；策略检查点没有 critic，报错 |
| `build_actor_critic(net_config, text, critic)` | 新建 `ActorCritic` 的唯一入口（训练器、快照池） |
| `warm_start(actor, loaded)` | 热启动：把加载的 actor 拷进新网络，只允许新网络多出卡片视角（`CARD_VIEW_FIELDS`，新模块保持初始化），返回多出的配置项；不需要构造 `Trainer` 即可测 |
| `loaded.signature` / `Signature.mismatches(other)` | 网络读什么（词表的卡密顺序、事件窗口长度）；两份检查点 / 运行能否共用观测，不能时逐条给出可读的原因（空表 = 兼容）；可哈希，批量路径按它分组 |

冻结文本表与卡片事实不在检查点里：从 `text_dir` 读，PPO 检查点缺省用其运行配置的 `text_dir`；缺表时所有加载器走同一个检查
（`nets.text.require_tables`，报错列出缺的视角，如 `card_facts`）。

运行目录（`tools/train_ppo.py` 默认 `out/train/<时间戳>`；给 `--env` 时是 `environments/<版本>/artifacts/train/<名字>`，
产物绑定环境版本）：

| 文件 | 内容 |
|------|------|
| `config.json` / `vocab.json` | 训练配置；`CardVocab.save` 的词表 |
| `metrics.jsonl` | 每次更新一行：更新号、行数、决策数、收集 / 更新秒数、决策/秒、行/秒、结束局数、自博弈 / 快照局数、对快照胜率、自博弈先攻胜率、截断与错误局数、终局原因、平均局长，§8.2 的损失指标，累计计数 |
| `eval.jsonl` | 每次评估每个基线一行：局数、胜 / 负 / 平、胜率与 Wilson 区间、终局原因、耗时、是否新 best |
| `checkpoints/latest.pt`、`checkpoints/update_N.pt`、`best.pt` | 最新（每 `checkpoint_every` 次）、每次评估的、keep-best |
| `games.jsonl.gz` | 仅 `--log-games`（`log_games`，默认关）：每局结束一行，见下 |

**对局日志**（`--log-games`，#149）：给学习型组牌的 Δ 代理（[设计 05 §5.3](design/05-deck-building.md)）攒标签。每次更新把这批 rollout 里结束的每一局
追加一行到 `games.jsonl.gz`（每次更新一个 gzip 成员，`gzip.open` 可直接逐行读）：

```json
{"update": 12, "decks": ["branded", "evo-0003"], "evolved": 1, "first": 0, "winner": 1, "turns": 7, "reason": "win",
 "truncated": false, "opponent": null, "learner": null, "seed": 123, "environment": "md-2026-09", "fingerprint": "65ca…"}
```

`decks` 是牌组 a、b：语料牌组用 `.ydk` 文件名（路径在 `config.json` 的 `decks`），进化牌组用清单 id（`evolved` 标出发下去的那套是 a 还是 b，
清单路径是 `config.json` 的 `deck_pool`；作对手的 history 牌组也是清单 id）；`first`、`winner`、`learner` 都是牌组下标（0 = a、1 = b），
`winner` 为 null 表示无胜者；`truncated` 为真的局（上限截断、引擎错误）不是训练意义上的胜负；`opponent` 为 null 是自博弈，否则是快照 id，
`learner` 是学习者所用牌组。`update` 是该局结束所在 rollout 所喂的那次更新。
该 rollout 的学习者使用 `update - 1` 次更新后的策略（`--overlap` 时 `update - 2`）；
固定段收集的一局可能跨越多次更新，不能把这个编号解释成整局使用同一个 checkpoint。
完整游戏收集才保证一局的学习者策略在一次更新边界内固定。
开销：md-2026-09 的 6 套 meta 牌组、32 槽 × 64 步、`max_decisions=40`（为了多出局）跑 4 次更新共 407 局，写日志合计 6 ms（约 16 µs/局，
占训练时间 0.01%），压缩后约 17 字节/局。
读取：`ygorl.build.edit_labels.read_game_log`（解析牌组名、去掉截断局与续训重放的重复行），用作改动价值模型的辅助损失（[tuning.md](tuning.md)「改动价值模型」，#151）。

### 8.6 默认规模与吞吐

`TrainConfig` 默认是面向 4 核 CPU 的小网络：`d_model = 64`、4 头、局面 / 历史各 1 层、事件窗口 64、ID 嵌入开
（actor 136 万参数，其中卡片 ID 嵌入 94 万）、特权编码 64 维、critic 隐层 128，critic 与 actor 共享主干；**2026-09-27 起**
128 个槽位 × 128 行 = 每次更新 16,384 行、minibatch 2,048、2 个 epoch、λ = 0.5（设计 I1 / I2，依据见 [benchmarks.md](benchmarks.md)「训练平台期」「训练与评估的速度」；
此前是 32 × 64 = 2,048 行、minibatch 256、λ = 0.95，下文的吞吐数字按旧默认测）。4 核云容器上（与其他任务共享 CPU，PyTorch 2 线程）收集约 670 决策/秒（含推理），更新是瓶颈
（前向 + 反向约 3–4 ms/行，卡片 ID 嵌入的稠密梯度占相当一部分；每次更新约 21 秒、收集约 3.4 秒），整体约 84 行/秒。

**T4b.4 验收**（[benchmarks.md](benchmarks.md)「PPO 自博弈：单牌组对 1 小时」）：snake_eye 对 kashtira 训练 60 分钟、140 次更新、
286,720 行、256 局，**0 崩溃、0 引擎错误**，内存平稳；玩具环境（Nim）上 PPO 在数秒内收敛到最优策略（§8.7）。
1 小时的 CPU 训练对 random 0.537（0.429–0.643）、对 greedy 0.150（0.088–0.244），与未训练策略（0.475 / 0.113）的区间重叠——
还谈不上学会；原因与调参方向见下。

**GPU**（`TrainConfig.device` / `--device cuda`，ROCm 也叫 `cuda`）：学习器、行动网络、BC 先验与钉住的对手都放到该设备；
收集的整段观测在设备上拼好，优势估计仍在 CPU 上算（逐行数据小），checkpoint 一律按 CPU 加载。ROCm 上权重梯度走 `ygorl.nets.gemm` 的按 `K` 切分，
并自动打开融合注意力（原因与实测见 [benchmarks.md](benchmarks.md)「GPU 学习器」）。以下**实验选项**默认不启用：

- `overlap_collect`（`--overlap`）：更新进行时在另一个线程（GPU 上另一个 stream）用「更新开始前的权重」的副本收集下一段；联赛记账
  （快照、评估、checkpoint）只在两步之间、没有收集在跑时进行。代价是每段数据落后一次更新，偏离设计的同步 PPO（[scaling.md](scaling.md) S3），采用前要先改设计。
- `bf16`（`--bf16`）：行动与更新都在 bf16 autocast 下跑，包含critic-only warmup。
  2026-10-05修复warmup引入时的缩进回归：此前该选项只覆盖采样，update实际在autocast之外。
  更新作用域关闭参数cast缓存，保证每个Adam步后的forward读取新权重；采样时权重不变，保留缓存。
  早期特定配置曾测得GPU更新快10–20%，但不是当前硬件/配置保证；回归期间的CLI对照不能证明更新阶段BF16无收益。
  新配置需分别验证真实dtype、成本与强度，CPU上也不保证更快。
- `learner_precision`（`--learner-precision inherit|fp32|bf16`）：默认`inherit`沿用`--bf16`的更新精度，
  旧checkpoint也保持这个默认值。显式`fp32`或`bf16`只覆盖学习器整个update（含warmup、reference和BC prior），
  采样与bootstrap仍由`--bf16`控制；权重保持FP32，更新内继续禁用cast缓存。
  例如`--device cuda --learner-precision bf16`保留FP32采样，只对学习器启用BF16；
  `--device cuda --bf16 --learner-precision fp32`则反过来。配置和checkpoint保存该选择，benchmark也输出此字段。
  用相同训练预算独立验证成本、实际minibatch数与棋力后再决定是否采用，不能仅据微基准提速替换正式配方。

`tools/bench_train.py` 测 `Trainer.step()` 的收集 / 更新耗时、行/秒与（AMD GPU 上的）忙碌率。
另外输出一次更新内部的细分（#72）：
- 各段：`targets`（优势估计）、`setup`（数据搬运、裁剪填充、打乱）、`reference`（参考策略与 BC 先验的前向，设计 I8 的 KL 项）、
  `forward`（模型前向与损失）、`backward`、`optimizer`（清梯度、梯度裁剪、步进）、`bookkeeping`（逐 minibatch 取统计值）、`ema`（参考策略的 EMA 更新）；
  剩下的记为 `other`，各项之和等于 `total`。
- 输出位置：命令行的 `update breakdown:` 一行，以及 JSON 的 `update_breakdown`。JSON 还附带 `commit` 与 `rocm`（HIP 版本、`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`、`HSA_OVERRIDE_GFX_VERSION`）。
- 实现：`PPOLearner.timing = True` 时，每段结束前等设备同步再读时钟。同步只改变读表的时机，不改变计算：同种子下参数逐位相同（`tests/test_ppo.py`）。
  默认关闭，关闭时不做任何同步。

### 8.7 测试

- `tests/test_train_cli.py`（约 5 秒）：每个配置字段都有且只有一个参数（枚举 dataclass 字段）；每个字段取非默认值后经 `to_argv` → 解析 →
  `from_args` 往返不变，`to_dict` / `from_dict` 同样往返；不给参数得到 `TrainConfig()`；运行脚本用过的旧参数名含义不变；
  `tools/train_ppo.py` 的 `--overlap` / `--bf16` 进入配置。

- `tests/test_ppo.py`（玩具博弈，约 15 秒）：Nim 自博弈在 VRPO 与 GAE 下都收敛到最优策略（每个必胜局面取 `n mod 4`），
  对快照池训练同样收敛；收集器布局（交替座位、终局奖励、段间衔接自举状态）、快照局只含学习方的行、截断局标记且无奖励、
  引擎错误事件（开局失败 / 局中错误）记为截断并立即开新局；默认超参数符合设计（熵系数在 0.05–0.2）；EMA 参考；
  KL 为 0 与梯度方向；先验 KL 只在第 1 回合（`kl_prior_turns`）时的行选择、权重与更新结果；学习器状态往返；快照池逐出与 keep-best；策略目标可插拔（注册自定义目标，`prepare` / `loss` 被调用）。
- `tests/test_checkpoint.py`（手工写的小检查点，不构造 `Trainer`、不开局）：各种 critic 选项（含牌序 critic、非共享主干、非特权）的
  actor-critic 往返；策略检查点能当 actor 加载、不能当 actor-critic；所有加载器缺表时同一报错，PPO 检查点从运行配置找回表；
  `Signature` 的兼容性原因；从 PPO / BC 检查点热启动、热启动多出卡片视角、网络不同则报错。
- `tests/test_advantages.py::test_truncated_rows_bootstrap_from_the_critic`：截断行的目标等于 critic 自身估计、与「切列 + 同座位自举」一致、
  截断行的奖励被忽略。
- `tests/test_deck_pool.py`（约 10 秒）：没有 probation / active 卡组（清单缺失、只有 history 或占比为 0）时发局与固定牌组池逐位相同（经 `Trainer` 构建的发局也相同；整次训练不比较：哪个槽位打哪一局取决于引擎线程的时序，同一配置跑两次权重就不同）；进化占比、只发 probation / active、
  对手来自语料或历史、坐两侧；打得差的卡组更常出现、`weight` 生效；重读后只发当前卡组；原地改写的卡组文件会被重读；不合法 / 重复 / 读不到 / 状态或权重错的条目被跳过并记录，写到一半的清单保留上一版池；
  池状态往返（删掉清单与卡组文件后）发局相同；
  `Trainer` 记录进化卡组得分、续训后池与发局状态一致、训练中重读清单。
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

### 实验：完整游戏与纯终局 critic 标签（#61）

`--complete-games` 每个环境槽收集一整局后等待其它槽；`--steps` 在此模式下不参与预算。
批次按游戏拼为单列，长度可变，无 padding 或跨更新续局；终局处阻断回溯。
截断/错误游戏的所有训练行丢弃，`discarded_rows` 计数并计入 `decisions`，不自动补采。
全批没有有效学习行时失败；内存随完整游戏长度增加，应先用少量槽冒烟。
默认固定步数模式不变。

`--critic-target terminal --complete-games` 让 Q(s,a) 和 V(s) 使用真实终局 z，
不包含 critic/bootstrap/control-variate；要求 gamma=1、turn_discount=1、无 critic_lam 覆盖。
完整游戏模式暂不支持 overlap。策略优势仍使用配置的 VRPO/GAE，不是 MC policy gradient。
对照应同样启用 complete-games、只设 `--critic-target lambda`，避免将采样变化误作目标效果。
完整游戏更新的 `q_terminal_ev` / `v_terminal_ev`、对应 MSE 来自更新前的新游戏，
其终局标签在两个臂相同定义；`q_explained_var` 仍是各自训练目标的 EV，不能跨目标直接比较。
当前协议见 [终局 MC pilot](spikes/terminal-mc-2026-10-04.md)，实验选项不改变默认训练配方。
完整游戏与常规固定段的近似等行数对照见
[采样实验](spikes/collection-control-2026-10-04.md)；对照同时包含跨更新续局、事件顺序和截断行处理的差异，
不能将结果单独归因于段末自举。

### 独立矩阵消费者（#90）

先用 `ygorl strength` 创建固定基线矩阵，再训练时传
`--register-every 250 --register-matrix /absolute/path/matrix.json`。默认登记关闭；
需要完全异步评估时同时设 `--eval-every 0`。登记保存在 RUN/registrations，
checkpoint 永不覆盖；续训更新号延续，分叉历史更新应使用新 RUN。

另一个进程执行：

```bash
uv run --no-sync python tools/consume_registrations.py out/train/run --matrix /absolute/path/matrix.json --env md-2026-09 --workers 2
```

未传 `--decks` 时使用环境 meta 牌组（顺序/内容必须与矩阵一致）；可显式传评估牌组路径。
每次执行消费当前快照，重复执行安全；只写完整 checkpoint 的矩阵结果，异常中断可重跑。
输出默认在矩阵旁的 `<matrix-stem>.strength.json`。曲线列出累计活动 wall 秒数和训练秒数，
对初始固定 agent 的胜率及当前矩阵名次。消费前应冻结所有基线 checkpoint 的路径/内容；
原始基线清单写在 `<matrix>.registrations.json`。多个消费者通过锁互斥，勿同时用普通
`ygorl strength` 改写同一目标。错误矩阵不发布；已有矩阵计分仍须配合逐局健康审计。

矩阵计分会把带Lua错误、retry、未知/不可解码消息或error文本的局计为errors，
即使该局原reason为win；批量路径保留相同健康字段。消费者拒绝发布含errors的矩阵。
此前版本产出的历史矩阵需要另行审计/重建，不能通过这次修复推断它们没有污染。

同步evaluate的任何基线有健康错误或零有效局时，不更新best.pt或best池；
`eval.jsonl`保留attempted_games、全部健康计数与valid，完整无效报告另写`eval-errors.jsonl`。
checkpoint与诊断仍保存，训练可继续。这样不能靠剔除异常后剩余的小样本高分晋级。

2026-10-05修正KL检查顺序：检查发生在backward/optimizer之前。minibatches是实际Adam步数，
evaluated_minibatches是已检查批数，stop_approx_kl是触发值；loss/entropy/KL均值包括最后
拒绝的forward，grad_norm只统计真正更新。第一批即超限时minibatches=0，诊断仍完整有限。
critic-only warmup不受policy guard限制，策略参数仍被冻结。旧32-update诊断保持原实现作为对照，
不把检查顺序的修复自动宣称为实战提升；下一轮要独立检验。

### 慢reference与共享参数（2026-10-05）

每次更新后对每个唯一参数/持久buffer执行一次EMA，不能对state_dict里的共享别名重复更新。
[修复与复现](spikes/ema-shared-parameters-2026-10-05.md)：现有Transformer-history模型的CardIdentity
有4个名字，旧τ=.02实际成为.07763184；修后按配置执行。旧checkpoint保留已有reference，
修复不重建过去的历史。首步优化前reference仍等于learner，这个错误不是首步KL超调的原因。
