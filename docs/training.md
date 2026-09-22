# 优势估计与特权 critic（`ygorl.train`，T4b.3）

对应设计 [03-play-policy.md](design/03-play-policy.md) I2「特权 Q-critic + Q-boosted 优势（VRPO）：以候选动作打分头再出一个
Q 头，用 Expected-SARSA(λ) 回溯替代 GAE」、I9「critic 看历史 + 特权信息，特权信息绝不喂 actor」，以及
[02-challenges.md](design/02-challenges.md) C3(b)。本页给出精确公式、符号约定、rollout 收集器（T4b.4）要交出的数据布局，以及
GAE / VRPO 对照开关。纯 PyTorch 实现，与网络结构无关：critic 只接通用特征张量。需要 `uv sync --extra train`。

| 模块 | 内容 |
|------|------|
| `ygorl.train.advantages` | `gae`、`expected_values`、`q_advantages`、`expected_sarsa_returns`、`vrpo_advantages`、`normalize_advantages`、`terminal_rewards`、开关 `estimate` |
| `ygorl.train.critic` | `Critic`（历史 ⊕ 特权 → 上下文 → `CandidateQHead` + `ValueHead`）、损失 `q_loss` / `v_loss` |

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
- **主机代答**（课程模式 T2.6 的 `solo` / `handtrap` 下替对手自动放弃）：不是 agent 决策，**不出行**
  （与 `DuelResult.actions` / `record_steps` 一致）；它的后果体现在下一行的状态里。
- **终局奖励**：挂在这一局在本列里的最后一行上，从该行座位的视角给出（可用 `terminal_rewards(players, dones, winner)`，
  `winner` 为胜方座位或 −1 平局）。即使终局是对手的动作、主机代答或回合上限触发的，也挂在最后一个 agent 行上。
  回合上限（`turn_limit`）平局是真正的终局：`done = True`，奖励 0。
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
               bootstrap_q=q_T, bootstrap_probs=pi_T, bootstrap_mask=m_T, normalize="standard")
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

- 段只在列尾截断自举；收集器若在一列中间强行截断（非终局），需要把该段拆成单独的列。
- 不做离策略修正（重要性比 / Retrace 的 `c = λ min(1, ρ)`）：PPO 的数据接近同策略；快照池复用旧数据时再加，接口位置在 `λ · c_t`。
- 特权特征的编码器、critic 与 actor 是否共享主干及其梯度隔离，属于网络侧（T4b.1 / T4b.2）与 T4c.2 消融。
