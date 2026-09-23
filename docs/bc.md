# 行为克隆预热（`ygorl.train.bc`，T4a.2）

用 T4a.1 的求解器示范集（[solver.md](solver.md)）对策略网络（[nets.md](nets.md)）做行为克隆（BC），输出的检查点是 PPO 的初始化与
I8 的 KL 参考策略。设计依据：[03-play-policy.md](design/03-play-policy.md) I4「先攻展开求解器 → 示范：BC 预热 + KL 先验，再 RL」、
I8「对慢速参考策略加 KL 项」；[02-challenges.md](design/02-challenges.md) C3(a)。需要 `train` 可选依赖（`uv sync --extra train`）。

```
ygorl.env.observer   PointObserver：DecisionPoint → 与 EncodedVecEnv 逐元素一致的观测（参考编码器 + 事件流）
ygorl.train.bc       示范 → 样本（line_steps / build_dataset）、训练（train_bc）、评估（step_accuracy / play_opening / opening_report）
ygorl.nets.agent     检查点（save_checkpoint / load_checkpoint）、NetPolicy（PolicyAgent 的网络策略）、agent 规格 policy:PATH
tools/train_bc.py    训练 + 评估报告（report.json）
```

## 流程

1. **样本**：示范只存动作下标（[solver.md](solver.md)「示范集格式」）。`line_steps` 从记录的起始对局（`demo.replay`）新建 `DuelSession`，
   逐步喂示范动作；**每一步**（双方的）都把 `point.events` 喂给事件流，在研究对象（引擎玩家 0，先攻）的决策点用
   `PointObserver.encode(point, core)` 编码观测。样本 = (观测, 示范动作在候选动作表中的行号)。
   - 只有 1 个合法动作的步不入样本（损失恒为 0）；示范动作行号 ≥ 128（编码截断）的步不入样本（本数据集中 0 个）。
   - **选中 / 取消来回切换**不入样本（`undone_steps`）：求解器的线有时在 SELECT_UNSELECT_CARD 里把同一张卡选中又立刻取消（一条 tenpai 线在选同调素材时来回 320 次），
     每一对都回到同一决策状态，不示范任何东西。不去掉时这类步占训练样本的 13%、held-out 样本的 67%，并教会网络原地打转。去掉后的动作序列在同一起始对局里同样到达目标（17 条受影响的 plain 线全部复核）。
   - plain 与 `--fire` 记录都用；`--fire` 线里对手发动手坑的那一步是玩家 1 的，只进事件流不进样本，我方在被手坑后的「恢复」决策进样本。
2. **观测**：`PointObserver` = `ObservationEncoder`（卡片表、全局、候选动作）+ `EventHistory`（事件 token 窗口），与 C++ `EncodedVecEnv` 的观测
   **逐元素一致**（`tests/test_bc.py` 在真实对局上逐步比对 6 个数组），所以 BC 与 PPO 看到的是同一种输入，检查点可直接用于 `EncodedVecEnv`。
3. **损失**：`PolicyNet` 的 logits 在非法行上是 `MASKED_LOGIT`，对 logits 做交叉熵即「只在合法候选上的交叉熵」（`bc_loss`；可选标签平滑，平滑质量只分给合法候选）。
   AdamW（lr 3e-4、weight decay 1e-4）、梯度裁剪 1.0、100 步线性预热 + 余弦退火、batch 64。
4. **去填充**（`trim_padding`）：每个 batch 截掉所有样本都不用的尾部卡片行 / 动作行 / 事件 token。对 `PolicyNet` 精确（填充卡片行在注意力里被掩掉、动作之间不互相注意、历史因果且有效 token 在前；单测比对 logits），
   第 1 回合数据平均只用 60/160 张卡片行、4/128 个动作行、38/128 个事件 token，一步训练耗时约为不截断时的 40%。

## 检查点与 PolicyAgent

`save_checkpoint(path, net, vocab, event_length=..., environment=..., meta=...)` 写一个 `torch.save` 字典（只含普通数据，`torch.load(weights_only=True)` 可读）：

| 键 | 内容 |
|----|------|
| `format`, `format_version` | `"ygorl-policy"`, 1 |
| `config` | `NetConfig.to_dict()` |
| `state_dict` | 网络参数（冻结文本表是非持久 buffer，不在内；用过文本表的检查点加载时要传同一个 `TextFeatures`） |
| `vocab` | `CardVocab` 的密码表（按下标顺序），卡库更新后仍按训练时的下标解释 |
| `event_length` | 每个观测的事件 token 数 |
| `environment` | 训练数据绑定的环境 `{"version", "fingerprint"}`，无环境为 `null` |
| `meta` | `trainer`、数据规模、`BCConfig`、最后一个 epoch 的指标 |

加载与对局：

```python
from ygorl.agents import PolicyAgent, make_agent
from ygorl.nets.agent import NetPolicy, load_checkpoint

ckpt = load_checkpoint("out/bc12/policy.pt")        # .net（eval 模式）、.vocab、.event_length、.environment、.meta
agent = PolicyAgent(NetPolicy(ckpt.net, ckpt.vocab, event_length=ckpt.event_length), seed=0)
agent = make_agent("policy:out/bc12/policy.pt@greedy")   # 同上，按名字；@greedy 取 argmax，@t=0.5 改温度
```

```bash
uv run ygorl arena tests/decks --agent-a policy:out/bc12/policy.pt --agent-b random --games 200 --workers 2
```

`NetPolicy` 需要看到对局的**每个**决策点（事件流跨越对手的决策；卡片表要查询核心），为此 Agent 协议加了可选的 `observe(point, core)`：
`Duel.run` 在每个决策点（双方的）按顺序、在行动方 `act` 之前调用它（[evaluation.md](evaluation.md)「Agent 协议」），`PolicyAgent` 转发给策略。
每个决策点单独编码、前向一次，适合评估（200 局 arena 在 2 进程下约 13–17 分钟）；训练走 `EncodedVecEnv`。并行 arena 的子进程各自加载一次检查点（按路径与修改时间缓存）。
合法动作超过 128 个时，第 128 个之后的动作概率为 0（与编码截断一致）。

## 评估的定义

- **等价副本**：观测的 `action_mask` 只保留同一张卡等价副本中的第一行（[encoding.md](encoding.md)「等价动作去重」）。示范若选了另一张副本，
  训练标签换成它的代表行（`canonical_action`）；去重后只剩一行的决策按强制决策跳过（`skipped["forced"]`）。
- **步准确率**（teacher forcing，`step_accuracy`）：在示范线的每个样本上，网络 argmax 是否等于示范动作（代表行）；同时给出均匀随机猜中的期望（`uniform_accuracy`，按去重后的可选行数）作参照。
- **自由对局**（`play_opening`）：从记录的起始对局（与求解器完全相同的种子字、卡组顺序、`DUEL_PSEUDO_SHUFFLE` 与白板对手；求解器没解出的起手用 `start_replay` 按
  `hand_seed` 重建，单测核对重建结果与求解器记录的起始对局一致）让 agent 下第 1 回合，对手以被动选项应答（与示范补完回合相同），
  到第 2 回合第一个决策为止；agent 超过 300 步时由主机用被动选项收尾（`capped`）。终局场面按求解器判定线的标准评分：目标卡是否全部在场（`board_summary_missing`）。
  - **线复现率**：agent 的动作序列（玩家 0）与该起手某条示范线（去掉来回切换后）**逐步相同**的比例；另报「与示范线的最长公共前缀 / 线长」的平均。
    逐步比较用 `action_key`（决策类型、动作类型、卡、效果串、区域；场上的卡带序号），不用动作下标：选了另一张等价副本后手牌顺序不同，之后的下标会错开。
    下文 2026-09-23 的数字是等价动作去重之前测的（当时按下标比较）。
  - **场面质量**：目标场面达成率（`reach_rate`）、目标卡在场比例（`placed_fraction`）；同一批起手上求解器自己的解出率作上限参照（求解器在 20 秒预算内解出 = 存在一条到达目标的线）。
- 基线：`RandomAgent`、`GreedyAgent` 在同样的起手上自由对局。

## 数据与命令（2026-09-23）

示范集（`out/` 已被 git 忽略，按下面的命令重建；求解器二进制见 [solver.md](solver.md)「构建」）：

```bash
export YGORL_COMBO_SOLVER=build/combo-solver/bin/combosolver
# 训练集：10 套测试牌组 × 起手 0–29，plain + --fire（Ash Blossom），每手最多 2 条线（4 核共享机器上 2 进程，墙钟 48 分钟）
uv run python tools/solve_openings.py tests/decks --hands 30 --solve-ms 20000 --fire 14558127 --fire-ms 10000 \
    --workers 2 --threads 1 --lines 2 --out out/demos/bc_train.jsonl
# held-out：同 10 套牌 × 起手 1000–1009（与训练起手不重叠），只要 plain（墙钟 9 分钟）
uv run python tools/solve_openings.py tests/decks --first-hand 1000 --hands 10 --solve-ms 20000 --no-fire \
    --workers 2 --threads 1 --lines 2 --out out/demos/bc_heldout.jsonl
uv run python tools/verify_demos.py out/demos/bc_train.jsonl
uv run python tools/verify_demos.py out/demos/bc_heldout.jsonl
# 训练 + 评估（推荐检查点：12 个 epoch）；--baselines 另跑 Random / Greedy 的自由对局
uv run python tools/train_bc.py --train out/demos/bc_train.jsonl --heldout out/demos/bc_heldout.jsonl \
    --out out/bc12 --epochs 12 --threads 2 --baselines
OMP_NUM_THREADS=1 uv run ygorl arena tests/decks --agent-a policy:out/bc12/policy.pt --agent-b random \
    --games 200 --workers 2 --out out/bc12/arena_random.json
```

求解器的搜索按墙钟预算、未固定 `--solver-seed`，重跑得到的线可能不同（数量与下表同一量级）。

| 数据 | 训练集 | held-out |
|------|------|------|
| 起手（plain） | 300（解出 158、未解出 141、`unverified` 1） | 100（解出 62） |
| `--fire` 记录 | 300（解出 62、未解出 76、无时点 20、跳过 142） | — |
| 已验证的线 / 动作步 | 440（plain 316 + fire 124）/ 18,643 | 124 / 10,090 |
| 样本（玩家 0、≥ 2 个合法动作、去掉来回切换） | 8,701（跳过：强制步 3,970、来回切换 1,336、超过 128 行 0） | 2,528 |
| 与训练集相同的起手（同牌组、同 5 张） | — | 0 |

`tools/verify_demos.py` 独立复验：训练集 440 条、held-out 124 条线全部通过。`unverified` 的 1 手（labrynth 第 13 手）：求解器写出的 4 条线在我们的核心里重放、补完回合后终局都缺 Arianna（1225009），
原因未查（T4a.1 的问题，记录按设计被排除，不影响本任务）。

网络：默认 `NetConfig`（`d_model` 128、局面 2 层、GTrXL 历史 2 层、事件窗口 128、无文本表），3,559,808 个参数；`torch` 2 线程，一个 epoch 约 42 秒（4 核共享机器，其他任务同时在跑）。

### 训练曲线

held-out 步准确率在第 4–6 个 epoch 到顶（约 0.58–0.59），之后 held-out NLL 单调上升（第 6 个 epoch 0.96 → 第 60 个 3.7），训练准确率继续涨到 0.97：数据只有 316 个不同起手，网络很快开始背线。
所以报告两个检查点：**bc12**（12 个 epoch，推荐作 PPO 初始化与 KL 参考——没有过度自信）与 **bc60**（60 个 epoch，衡量「能否记住固定起手的线」）。

| 检查点 | 训练时间 | 训练集步准确率 / NLL | held-out 步准确率 / NLL | 均匀猜中期望 |
|------|------|------|------|------|
| bc12 | 615 s | 0.723 / 0.63 | **0.595** / 1.09 | 0.32 |
| bc60 | 2,857 s | 0.966 / 0.08 | 0.562 / 3.73 | 0.32 |

### 固定起手：线复现率（训练集的 299 个 plain 起手，其中求解器解出 158）

| agent | 逐步复现示范线 | 与示范线的平均公共前缀 | 目标场面达成（解出的 158 手） | 目标场面达成（全部 299 手） |
|------|------|------|------|------|
| bc60 | **137 / 158 = 86.7%** | 0.49 | **158 / 158 = 100%** | 159（含 1 手求解器 20 秒内没解出的） |
| bc12 | 5 / 158 = 3.2% | 0.10 | 129 / 158 = 81.6% | 134（含 5 手求解器没解出的） |
| Greedy | 0 | 0.04 | 28 / 158 = 17.7% | 29 |
| Random | 0 | 0.01 | 9 / 158 = 5.7% | 9 |

bc12 很少逐步复现（展开顺序有很多等价排列，它学到的是「到达目标」而不是某一条线的顺序），但 82% 的解出起手仍到达目标。

### 未见起手：场面质量（held-out 100 手，求解器在 20 秒内解出 62 手）

| agent | 目标场面达成 | 其中在求解器解出的手上 | 目标卡在场比例 |
|------|------|------|------|
| 求解器（上限参照） | 62 / 100 | — | — |
| **bc12** | **41 / 100** | **41 / 62 = 66.1%** | 0.44 |
| bc60 | 32 / 100 | 31 / 62 = 50.0% | 0.35 |
| Greedy | 13 / 100 | 13 / 62 = 21.0% | 0.15 |
| Random | 3 / 100 | 3 / 62 = 4.8% | 0.05 |

按牌组（bc12 达成 / 求解器解出，各 10 手）：branded_despia 3/5、fiendsmith_ryzeal 4/6、kashtira 3/5、labrynth 4/5、purrely 2/7、snake_eye 6/8、tearlaments 3/4、tenpai 8/9、voiceless_voice 4/7、yubel 4/6。
没有一手是 bc12 达成而求解器没解出的；在训练集上则有 5 手（求解器 20 秒预算内没找到、网络找到了）。

### Arena：对 Random（**未达到验收**）

`ygorl arena tests/decks --agent-b random --games 200`：10 套测试牌组的 100 个有序配对（含镜像）× 1 对配对种子，BC 策略按温度 1 采样，默认回合上限 200。

| agent a | 局数 | 胜 / 负 / 平 | 胜率（95% Wilson） | 先攻 / 后攻 | a 的胜局：卡组耗尽 / LP | a 的负局：卡组耗尽 / LP |
|------|------|------|------|------|------|------|
| bc12 | 200 | 57 / 143 / 0 | **0.285（0.227–0.351）** | 0.320 / 0.250 | 41 / 16 | 133 / 10 |
| bc60 | 200 | 56 / 143 / 1 | **0.282（0.225–0.349）** | 0.310 / 0.255 | 49 / 7 | 128 / 14 |
| 参照：Greedy | 2,000 | — | 0.921 | — | — | — |

两个检查点都**显著低于** 50%（区间上限 0.35），验收「在 Arena 里对 Random 明显 > 50%」**没有达到**。原因很清楚，在数据而不在训练：

- 示范只覆盖**先攻第 1 回合**的展开（第 1 回合不能战斗）加上「结束阶段」收尾，从来没有战斗阶段、攻击、后攻、对手回合的连锁响应。
  网络学到的是「每回合都尽量展开，然后进结束阶段」：bc12 在 6 局抽样里 506 次可以进战斗阶段只进了 27 次（5%），Random 是 64/362（18%）。
- 每回合都全力展开 = 大量检索、抽卡、送墓，卡组消耗远快于 Random；200 局里 133 局（bc12）是 **BC 方先卡组耗尽**输掉的，被 LP 打死的只有 10 局。
- bc60 背线更彻底，但对局表现没有差别（0.282 vs 0.285）：第 1 回合的质量不是瓶颈。

这正是设计 I4 的定位：BC 只是 RL 的起点与 KL 先验，打赢对局要靠 T4b 的 PPO 自博弈。要在 BC 阶段就过这条验收，需要**第 1 回合之外**的示范来源，这超出了「求解器示范集」的设计，
需要先改设计文档（见下「待决定」）。

## 局限与待决定

- **Arena 验收未达成**（上节）。可选方向，需要决定后先改 [03-play-policy.md](design/03-play-policy.md) I4 再实现：
  (a) 把这条验收移到 T4b（PPO 从 bc12 初始化后对 Random 的胜率），T4a.2 只验收第 1 回合的两项；
  (b) 为第 1 回合之后补一个示范来源（例如 Greedy 在第 2 回合起的决策做「启发式示范」，或求解器的「后攻 / 手坑」变体），与求解器示范混合做 BC；
  (c) 把 BC 先验只用在第 1 回合（KL 项只在示范覆盖的状态上加），对局其余部分交给 PPO。
- **过拟合**：只有 316 个不同的 plain 起手（+124 条 `--fire` 线），held-out NLL 从第 6 个 epoch 起上升。更多示范（T4a.1 的全量 10 × 1,000 起手）是最直接的改进；
  作 KL 参考时推荐 bc12 这样早停的检查点（bc60 在未见状态上过度自信，KL 项会过强）。
- **只学先攻**：`--fire` 线提供了「被手坑后恢复」的决策，但没有后攻、没有对手的真实互动（对手是白板）。
- **求解器目标是人工指定的**（[solver.md](solver.md)），「场面质量」只衡量是否到达这个目标，不是最优终场；没解出的起手里混有真卡手与预算不足。
- `NetPolicy` 走 Python 编码路径，每步单独前向，arena 评估较慢；PPO 训练用 `EncodedVecEnv`（C++ 编码，与之逐元素一致）。
