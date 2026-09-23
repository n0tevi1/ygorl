# 行为克隆预热（`ygorl.train.bc`，T4a.2）

用 T4a.1 的求解器示范集（[solver.md](solver.md)）对策略网络（[nets.md](nets.md)）做行为克隆（BC），输出的检查点是 PPO 的初始化与
I8 的 KL 参考策略。设计依据：[03-play-policy.md](design/03-play-policy.md) I4「先攻展开求解器 → 示范：BC 预热 + KL 先验，再 RL」、
I8「对慢速参考策略加 KL 项」；[02-challenges.md](design/02-challenges.md) C3(a)。需要 `train` 可选依赖（`uv sync --extra train`）。

```
ygorl.env.observer   PointObserver：DecisionPoint → 与 EncodedVecEnv 逐元素一致的观测（参考编码器 + 事件流）
ygorl.train.bc       示范 → 样本（line_steps / build_dataset）、训练（train_bc）、评估（step_accuracy / play_opening / opening_report）
ygorl.nets.agent     检查点（save_checkpoint / load_checkpoint）、NetPolicy（PolicyAgent 的网络策略）、agent 规格 policy:PATH
tools/train_bc.py    训练 + 评估报告（report.json）；--extra 混入启发式示范
ygorl.train.heuristic_demos  启发式 agent（GreedyAgent）第 2 回合起的决策 → 同格式样本（DemoRecorder / record_games / 子集 all、battle）
tools/greedy_demos.py        录制 Greedy 示范（.npz），见「补救实验」
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

两个检查点都**显著低于** 50%（区间上限 0.35），验收「在 Arena 里对 Random 明显 > 50%」**没有达到**。原因在数据覆盖而不在训练或管线，逐项证据见下节「对 Random 失败的根因分析」：

- 示范只覆盖**先攻第 1 回合**的展开（第 1 回合不能战斗）加上「结束阶段」收尾，候选动作里从来没有「进战斗阶段」「攻击」「主要阶段 2」「改变表示形式」。
  网络在第 2 回合起的主要阶段 1 收尾时 82% 选结束阶段：可以进战斗阶段的回合只进了 18%（Random 51%，Greedy 100%），场上有斩杀也 87% 不打。
- 于是对局拖到 50 回合以上、双方都很少掉 LP，胜负由「谁先抽干卡组」决定；BC 每回合比 Random 多消耗约 0.2 张卡组，133 / 143 个负局是 **BC 方先卡组耗尽**。
- bc60 背线更彻底，但对局表现没有差别（0.282 vs 0.285）：第 1 回合的质量不是瓶颈。

这正是设计 I4 的定位：BC 只是 RL 的起点与 KL 先验，打赢对局要靠 T4b 的 PPO 自博弈。要在 BC 阶段就过这条验收，需要**第 1 回合之外**的示范来源，这超出了「求解器示范集」的设计，
需要先改设计文档（见下「待决定」）。
**补救**（下文「补救实验」）：补上 Greedy 在第 2 回合起的战斗阶段决策做示范（b2）后对 Random 0.795（0.734–0.845），过了这条验收，第 1 回合的展开能力不变。

## 对 Random 失败的根因分析

`tools/diagnose_bc.py`（2026-09-23，bc12）。结论：**不是管线 bug，也不是先后攻；全部差距来自「第 2 回合起不进战斗阶段、不攻击」**，
卡组耗尽只是它的结果（没人造成伤害 → 对局拖长 → 谁先抽干谁输，而 BC 展开多、卡组消耗稍快）。第 1 回合的展开能力对「打赢 Random」既无帮助也无害处。

### 1. 管线：训练与对局看到的输入一致

| 检查 | 方法 | 结果 |
|------|------|------|
| 训练路径 vs `Duel.run` 路径 | `diagnose_bc.py consistency`：held-out 前 30 个解出记录的 60 条线，用脚本 agent 按示范动作走 `Duel.run`（双方同一个 agent，`observe` 看到每个决策点），在玩家 0 的每个非强制步取 `PolicyAgent(NetPolicy)` 的观测与 logits，与 `line_steps`/`build_dataset` 的观测逐数组比较，logits 与训练时的批量（`trim_padding`）前向及逐条前向比较 | 1,355 个决策：6 个观测数组 **全部逐元素相同**；logits 最大差 7.4e-6、概率最大差 1.7e-6，argmax **0** 处不同 |
| arena 复现 | `diagnose_bc.py play --agent bc` 用 `Arena.game_specs` 重建 `ygorl arena` 的 200 局（同配对、种子、卡组顺序、agent 种子），`fork` 两进程 | 与 `out/bc12/arena_random.json` **200 / 200 局**胜者、回合数、决策数、终局 LP 全同 |
| 并行 worker / 检查点缓存 | 同样 4 局用 1 个进程重跑 | 与 2 进程结果逐局相同 |
| 视角（后攻 = 引擎玩家 1） | 读 `ObservationEncoder`、`EventHistory`：卡片表按「我方 / 对方」排、事件 token 的玩家与控制者都相对观察者、LP 列是「我 / 对手」 | 观测相对决策者；只有全局向量的 `viewer`、`is_first` 两列按设计编码座位（训练只见过 `viewer = 0`、`is_first = 1`）。后攻的影响见下节，可以忽略 |
| 采样 vs argmax | `policy:PATH@greedy` | argmax 不更好：0.305（0.245–0.372），与采样的配对差 +0.020（−0.052, +0.092）；它**从不**进战斗阶段（6 / 4,365 个回合）、0 次攻击，且 24 / 200 局陷入确定性循环（「特殊召唤 → 选素材时取消」或同一张卡选中 / 取消来回），在 4,000 个决策处被截断（`--max-decisions 4000`；默认 20,000 的上限下这 200 局跑了 40 分钟还没完）。采样的温度 1 反而是它走出循环、偶尔进战斗阶段的原因 |

训练样本覆盖（`diagnose_bc.py coverage`，8,701 个样本）：全部是 `turn = 1`、`viewer = 0`、自己的回合、主要阶段 1 为主；决策类型里**没有** `SELECT_BATTLECMD`；
候选动作里 `battle_phase`、`attack`、`main2`、`reposition` 出现 **0 次**（这些动作种类的嵌入行从未得到梯度），`end_phase` 在 2,120 个 `SELECT_IDLECMD` 样本里被示范 389 次，而且每次都是一条线的收尾。
全局向量的回合（≥ 2）、`is_first = 0`、`is_my_turn = 0`、战斗阶段等类别嵌入同样从未训练过。这就是「分布偏移」的具体形态。

### 2. 分解实验：谁在哪些决策上替 BC 做决定（配对种子，对 Random，各 200 局）

同一批 200 局（与上节 arena 同配对、同种子），a 方是下表的组合 agent，b 方是 Random。「配对差」是逐局得分（胜 1、平 0.5、负 0）之差的均值与 95% 正态区间。

| a 方 | BC 决定 | 其余由 | 胜率（95% Wilson） | 先攻 / 后攻 | 负局：耗尽 / LP | 平均回合 | 与 bc 的配对差 | 与 Greedy 的配对差 |
|------|------|------|------|------|------|------|------|------|
| bc（= `policy:out/bc12/policy.pt`） | 全部 | — | **0.285**（0.227–0.351） | 0.320 / 0.250 | 133 / 10 | 51.3 | — | −0.625（−0.696, −0.554） |
| Random（对照） | — | Random | 0.495（0.426–0.564） | 0.500 / 0.490 | 67 / 34 | 53.0 | +0.210（+0.128, +0.292） | |
| Greedy（对照） | — | Greedy | **0.910**（0.862–0.942） | 0.910 / 0.910 | 11 / 7 | 22.2 | +0.625（+0.554, +0.696） | — |
| bc1_greedy | 先攻第 1 回合 | Greedy | 0.895（0.845–0.930） | 0.880 / 0.910 | 15 / 6 | 23.2 | +0.610 | −0.015（−0.058, +0.028） |
| bcown1_greedy | 自己的第一个回合（先攻 T1 / 后攻 T2） | Greedy | 0.910（0.862–0.942） | 0.880 / 0.940 | 10 / 8 | 22.9 | +0.625 | +0.000（−0.056, +0.056） |
| bc1_random | 先攻第 1 回合 | Random | 0.470（0.402–0.539） | 0.450 / 0.490 | 69 / 37 | 51.5 | +0.185 | 与 Random：−0.025（−0.080, +0.030） |
| greedy_bclate | 自己第一个回合以外的全部 | Greedy（第一个回合） | 0.355（0.292–0.423） | 0.360 / 0.350 | 111 / 18 | 49.5 | +0.070（−0.003, +0.143） | −0.555 |
| bc_bpd | 除战斗阶段内的决策外全部 | Greedy（战斗阶段内） | 0.405（0.339–0.474） | 0.440 / 0.370 | 109 / 10 | 47.4 | +0.120（+0.047, +0.193） | −0.505 |
| **bc_bp** | 同 bc_bpd，另外主要阶段 1 里 BC 选「结束阶段」而战斗阶段可进时改成「进战斗阶段」 | Greedy（战斗阶段内） | **0.910**（0.862–0.942） | 0.890 / 0.930 | 16 / 2 | 25.9 | **+0.625**（+0.552, +0.698） | **+0.000**（−0.056, +0.056） |

读法：

- **第 1 回合的 BC 对胜率无贡献、也无害**：把 BC 放进 Greedy 的第 1 回合（bc1_greedy）或自己的第一个回合（bcown1_greedy），与纯 Greedy 没有可分辨的差别；放进 Random 同样（bc1_random vs Random）。加到 1,000 局（同一 arena 的前 5 对配对种子）：Greedy 0.925（0.907–0.940），bc1_greedy 0.919，配对差 **−0.006（−0.022, +0.010）**；bcown1_greedy 0.927，配对差 **+0.002（−0.020, +0.024）**——第 1 回合的 BC 对打赢 Random 的影响在 ±2 个百分点以内。
- **损失全部来自第 1 回合之后**：只在后面的回合用 BC（greedy_bclate）就掉到 0.355，与全程 BC 的配对差只有 +0.07。
- **后面回合里，出问题的只有战斗**：BC 继续做所有展开、连锁、选卡、表示形式的决定，只把「进战斗阶段」和战斗阶段内的决策换掉（bc_bp），胜率与 Greedy **完全相同**（配对差 0.000 ± 0.056）。
  只换战斗阶段内的决策、不管进不进战斗阶段（bc_bpd）只追回 0.12：BC 很少走到战斗阶段。按 bc → bc_bpd → bc_bp 的顺序拆分 0.625 的差距：**+0.12 来自战斗阶段内的决策（进了也不攻击），+0.505 来自进入战斗阶段本身**（两者有交互：进了战斗阶段还得会攻击，反过来也一样，所以按另一种顺序拆分比例会不同，但「不进战斗阶段」是主因）。
- **先后攻不是原因**：bc12 先攻 0.320、后攻 0.250（区间重叠）；BC 在后攻的第一个回合（`viewer = 1`，训练没见过的座位）做决定（bcown1_greedy 后攻 0.940）不损失任何胜率。

### 3. 战斗：进不进、打不打

对「自己的回合、主要阶段 1、可以进战斗阶段」的决策点与回合统计（`diagnose_bc.py report`，200 局）：

| a 方 | 可进战斗阶段的回合里实际进入 | 主要阶段 1 收尾选结束阶段的点 / 选战斗阶段的点 | BC 在这些点上的平均概率：战斗阶段 / 结束阶段 | 对方场上无怪、我方有攻击表示怪时仍不进战斗 | 可一回合斩杀时仍不进战斗 | 攻击次数 / 局 | 造成伤害 / 局 |
|------|------|------|------|------|------|------|------|
| bc | **880 / 4,913 = 18%** | 4,077 / 890 | **0.045 / 0.205** | **715 / 852（84%）** | **115 / 132（87%）** | 1.2 | 2,337 |
| Random | 2,598 / 5,139 = 51% | 2,601 / 2,658 | — | 214 / 425（50%） | 31 / 64（48%） | 3.4 | 2,839 |
| Greedy | 2,146 / 2,150 = 100% | 0 / 2,171 | — | 0 / 568 | 0 / 149 | 6.9 | 6,178 |
| bc_bp | 2,510 / 2,511 = 100% | 0 / 2,545 | 0.054 / 0.236 | 0 / 485 | 0 / 121 | 7.6 | 6,264 |

「斩杀」按主要阶段 1 收尾时对方怪兽区为空、我方表侧攻击表示怪兽的当前攻击力之和 ≥ 对方 LP 估计（不考虑本回合不能攻击的怪兽，是上界估计）。
进了战斗阶段后，bc 在有攻击可选的 `SELECT_BATTLECMD` 上只有 240 / 806 次攻击（30%），其余是结束阶段 384、主要阶段 2 153。
第 2 回合起 BC 的 `SELECT_IDLECMD` 选择（bc_bpd，18,139 个）：发动 26%、结束阶段 22%、**改变表示形式 15%**（训练里从未出现的动作种类）、特殊召唤 12%；「进战斗阶段」只有约 4%。

### 4. 卡组耗尽是结果，不是原因

| a 方 | 平均回合 | ≥ 40 回合的局 | 卡组耗尽结束的局 / 其中 a 先耗尽 | 每个自己的回合离开卡组的卡（a / Random 方） | 其中检索 / 送墓 / 从卡组召唤 / 除外（a） | 额外抽卡 / 回合（a） |
|------|------|------|------|------|------|------|
| bc | 51.3 | 183 | 174 / **133** | **1.48 / 1.30** | 0.18 / 0.07 / 0.09 / 0.05 | 0.06 |
| Random | 53.0 | 176 | 135 / 67 | 1.27 / 1.27 | 0.11 / 0.03 / 0.04 / 0.04 | 0.02 |
| Greedy | 22.2 | 27 | 12 / 11 | 1.90 / 1.36 | 0.35 / 0.10 / 0.14 / 0.07 | 0.08 |
| bc_bp | 25.9 | 44 | 22 / 16 | 1.87 / 1.32 | 0.28 / 0.09 / 0.15 / 0.06 | 0.08 |

（「离开卡组」= 抽卡（不含起手 5 张）+ 从卡组移到其他位置的 `MSG_MOVE`；每局按自己的回合数平均。）

- 没有抽卡 / 送墓循环：BC 的额外抽卡每回合 0.06 张，检索 0.18 张；它的卡组消耗只比 Random 快约 0.2 张 / 回合，**比 Greedy（1.90）还慢**。
- 真正的区别是对局长度：BC 与 Random 对局平均 51 回合（Random 对 Random 53），双方都几乎不造成伤害（BC 每局造成 2,337、受到 1,557）；拖到卡组见底时，消耗稍快的一方先输。
  Random 对 Random 里卡组耗尽平分（67 / 68），bc 这边是 133 / 41。连 Greedy 在少数拖长的局里也是先耗尽的一方（11 / 12）。
- 一旦进战斗阶段（bc_bp），卡组消耗照旧（1.87），但对局在 26 回合内以 LP 结束，卡组耗尽的负局从 133 降到 16。

### 5. 未见过的决策上策略是什么样

bc12 在第 2 回合起的决策上**并不「迷茫」**（熵远低于均匀分布）：`SELECT_IDLECMD`（第 2 回合起，19,579 个）平均熵 1.02（均匀 1.85）、最大概率 0.62；
`SELECT_CHAIN`（对手回合，14,538 个）熵 0.13（均匀 0.79）、最大概率 0.96，几乎总是不连锁（约 92%）；从未训练过的 `SELECT_BATTLECMD`（1,121 个）熵 0.87（均匀 1.12）。
也就是说网络把第 1 回合学到的偏好（主要阶段 1 以「结束阶段」收尾）自信地外推到了所有回合，而「进战斗阶段 / 攻击」这类从未出现在候选里的动作只拿到随机初始化嵌入给出的低分。

### 6. 对三个方向的含义

| 方向 | 能解决 | 不能解决 / 风险 |
|------|------|------|
| (a) 把 Arena 验收移到 T4b（PPO 从 bc12 起步） | 验收放在对的地方：打赢 Random 只需要「进战斗阶段 + 攻击」（bc_bp = Greedy = 0.91），这是奖励直接可学的；bc12 采样时仍有 18% 的回合进战斗阶段，PPO 有探索信号 | bc12 本身不会变好；若 I8 的 KL 项对**所有**状态以 bc12 为参考，它会把策略拉回「结束阶段、不攻击」（bc12 在收尾点上只给战斗阶段 4.5%），与 RL 的改进相抵 |
| (b) 补第 1 回合之后的示范（Greedy 第 2 回合起的决策；后攻求解器线） | Greedy 示范直接覆盖缺失的动作种类（进战斗阶段、攻击、主要阶段 2、表示形式）：bc_bp 说明只要「进战斗阶段 + 战斗阶段内」像 Greedy，BC 就能到 0.91，其余决策保持 BC 也不吃亏；甚至只需补战斗相关的决策 | 上限约是 Greedy 的水平（示范是启发式）；后攻的**求解器**线不解决问题：座位不是原因，而求解器线以场面为目标、同样不含战斗；混入 Greedy 示范可能冲淡第 1 回合的展开，需要重测 held-out 场面质量 |
| (c) BC 先验 / KL 只用在第 1 回合状态 | 与证据一致：第 1 回合 BC 无害（bc1_greedy、bcown1_greedy ≈ Greedy），其外 BC 是未训练嵌入的外推，不该当参考；配合 (a) 时避免 KL 把 PPO 拉回不攻击 | 不改变 BC 检查点本身的 Arena 结果；第 2 回合起的初始化仍是「不攻击」，PPO 要先学会进战斗阶段（从 0.285 起步，而不是 Random 的 0.5） |

推荐：(a) + (c)；如果要在 BC 阶段就过验收，(b) 只需补「进战斗阶段 + 战斗阶段内」的 Greedy 示范。

复现（每条 200 局，两进程，与 `ygorl arena tests/decks --agent-b random --games 200` 同一批对局；BC 全程的配置约 11–13 分钟，其余 1–6 分钟）：

```bash
uv run --frozen python tools/diagnose_bc.py consistency --records 30          # 管线一致性
uv run --frozen python tools/diagnose_bc.py coverage                          # 训练样本覆盖
for a in bc random greedy bc1_greedy bc1_random bcown1_greedy greedy_bclate bc_bpd bc_bp; do
  OMP_NUM_THREADS=1 uv run --frozen python tools/diagnose_bc.py play --agent $a --games 200 --workers 2 --out out/diag/$a.json
done
uv run --frozen python tools/diagnose_bc.py report out/diag/*.json --baseline bc   # 以上各表
for a in greedy bc1_greedy bcown1_greedy; do   # 1,000 局的第 1 回合对照
  OMP_NUM_THREADS=1 uv run --frozen python tools/diagnose_bc.py play --agent $a --games 1000 --workers 2 --out out/diag/k1/$a.json
done
uv run --frozen python tools/diagnose_bc.py report out/diag/k1/*.json --baseline greedy
OMP_NUM_THREADS=1 uv run --frozen python tools/diagnose_bc.py play --agent bc_argmax --games 200 --max-decisions 4000 \
    --workers 2 --out out/diag/bc_argmax_d4000.json   # argmax（约 20 分钟）
```

## 补救实验：(a) (b) (c)

2026-09-23，接上节「对三个方向的含义」，issue #28。结论先行：

- **(b) 补第 1 回合之后的 Greedy 示范**是唯一在 BC 阶段就过验收的方向：对 Random **0.845（0.788–0.889）**（b1，全部 Greedy 决策）/ **0.795（0.734–0.845）**（b2，只补战斗相关决策），
  对 Greedy 0.405 / 0.415（基线 0.115）。b2 不损失第 1 回合展开能力（held-out 目标场面 41/100，与基线相同）；b1 少 6 手（35/100，差异在统计上不显著）。
- **(a) PPO 从 BC 起步**：在每个 arm 200 次更新（约 41 万行）的预算内，从 bc12 型 BC 起步与从零开始没有差别（对 Random 都是 0.515，第 1 回合展开几乎全丢）；
  从 (b) 的 BC 起步则到 **0.925**（= Greedy）、对 Greedy **0.635**（唯一显著超过 Greedy 的 arm），代价是第 1 回合展开从 37 降到 18。
- **(c) 先验 KL 只在第 1 回合**（新增 `--kl-prior-turns`）：不再像全状态先验那样把策略钉在「不攻击」上（全状态 0.305 vs 只在第 1 回合 0.525）；
  配合 (b) 与系数 2.0（P6）保住大部分第 1 回合展开（31 / 100），对 Random 0.855、对 Greedy 0.485。系数 0.2 太弱。
- **建议**：#28 用 b2 收尾（T4a.2 达标），T4b.5 从 b 型 BC 起步做 PPO，先验 KL 只加在第 1 回合；I4 需要改写（见文末「结论与建议」）。

### 0. 共同设置与重训的基线

等价动作去重（[encoding.md](encoding.md)「等价动作去重」）改变了标签与观测，旧 bc12 作废，先在当前 main 上用同一数据、同一配置重训（`out/bc12`）。
所有对局评估都是 `ygorl arena tests/decks` 的同一批 200 局（10 套测试牌组的 100 个有序配对 × 1 对配对种子，评估种子 0；对 Random 用
`tools/diagnose_bc.py play --agent bc`，与 `ygorl arena --agent-b random` 逐局相同，另带战斗统计），策略**按温度 1 采样**（argmax 会陷入选中 / 取消循环，见上节「采样 vs argmax」）；
区间是 95% Wilson，「配对差」是同一批对局上逐局得分差的均值与 95% 正态区间。第 1 回合场面质量是 held-out 100 手的目标场面达成率（`opening_report`，argmax，与上文相同的评分）。

| 检查点 | 训练集 / held-out 步准确率（NLL） | 对 Random | 对 Greedy | held-out 目标场面 |
|------|------|------|------|------|
| 旧 bc12（去重之前，上文） | 0.723 / 0.595（1.09） | 0.285（0.227–0.351） | — | 41 / 100 |
| **bc12（当前 main 重训）** | 0.721 / 0.585（1.02） | **0.305（0.245–0.372）** | **0.115（0.078–0.167）** | **41 / 100**（解出的 62 手中 40） |
| 参照：Greedy | — | 0.910（同一批 200 局，上节） | 0.500（0.431–0.569） | 13 / 100 |

重训后的 bc12 与旧的在误差内一致：样本 8,656（去重后多了 45 个强制步），可进战斗阶段的回合只进 29%（旧 18%），场上有斩杀仍不打 111 / 159，负局 116 / 139 是卡组耗尽。根因分析的结论不变。

### (b) 第 1 回合之后的示范：Greedy

**数据**（`tools/greedy_demos.py` → `ygorl.train.heuristic_demos`）：GreedyAgent 坐 a 方（先后攻各一局），对 Random 与对 Greedy 各打 100 个有序配对 × 1 对种子 = 400 局
（种子 7001，与评估种子 0 不重叠；Greedy 在其中对 Random 胜 0.925、对 Greedy 0.545），记录 a 方**第 2 回合起**所有非强制决策：观测用 `PointObserver`（与 `NetPolicy` / `EncodedVecEnv` 同一编码），
标签映射到等价副本的代表行。共 **49,262** 个样本（跳过：强制 78,441、第 1 回合 7,153），墓地与事件窗口比第 1 回合满，单个样本的训练开销约是求解器样本的 3 倍。
另用种子 9001 再录 400 局作 held-out（50,039 个样本）。决策分布（训练集）：选区域 18%、选卡 13%、选中 / 取消 9%、连锁 8%、进战斗阶段 7%、战斗中转主要阶段 2 7%、发动 7%、攻击 6%……

两个子集（`--extra-subset`）：

- **b1 = all**：全部 Greedy 决策（包括对手回合的连锁、选区域、表示形式等）；
- **b2 = battle**：诊断指出的最小集合——自己回合战斗阶段内的全部决策（`SELECT_BATTLECMD`、攻击对象、伤害步的连锁 / 效果）+ 主要阶段 1 里 Greedy 选「进战斗阶段」的那一步
  （12,541 个）。Greedy 在可进战斗阶段时从不选结束阶段，所以这就是「收尾时进战斗而不是结束」的示范；Greedy 在主要阶段 1 的其余选择（发动、召唤）不进 b2。

**配比**：两个子集都随机抽 **8,656** 个（= 求解器样本数），与求解器样本 1 : 1 混合（共 17,312），其余设置与 bc12 相同（12 个 epoch、同一种子）；因此 b1、b2 的样本数与梯度步数相同，
都是 bc12 的 2 倍。第 1 回合样本占一半，不会被第 2 回合起的大量决策淹没（全部 4.9 万个都用会让求解器样本只占 15%）。

| 检查点 | 求解器训练集 / held-out 步准确率 | Greedy held-out 步准确率（同子集，4,000 个；均匀猜中期望） | 对 Random | 对 Greedy | held-out 目标场面 argmax / 采样 |
|------|------|------|------|------|------|
| bc12（基线） | 0.721 / 0.585 | all 0.382（0.356）；battle **0.165**（0.423） | 0.305（0.245–0.372） | 0.115（0.078–0.167） | 41 / 28 |
| **b1**（+ 全部 Greedy 决策） | 0.719 / 0.592 | all **0.695** | **0.845（0.788–0.889）** | **0.405（0.339–0.474）** | 35 / 35 |
| **b2**（+ 战斗相关决策） | 0.741 / 0.588 | battle **0.857** | **0.795（0.734–0.845）** | **0.415（0.349–0.484）** | **41** / 37 |
| bc12s（bc12 的小网络版，PPO 用） | 0.680 / 0.594 | — | 0.270（0.213–0.335） | 0.120（0.082–0.172） | 36 / 23 |
| b2s（b2 的小网络版，PPO 用） | 0.699 / 0.597 | — | 0.775（0.712–0.827） | 0.380（0.316–0.449） | 37 / 25 |
| 参照：Greedy | — | — | 0.910 | 0.500 | 13 / — |

「采样」一列是按温度 1 采样下第 1 回合（每手一次，种子 = 手号；`train_bc.py --sample-openings`），单次采样噪声大（±9 手量级），只作 argmax 的补充：
BC 策略在 argmax 下更稳，PPO 之后的策略 argmax 会打转（下文「封顶」= 300 步上限内没结束回合）。

配对差（同一批 200 局）：对 Random，b1 − bc12 **+0.540（+0.460, +0.620）**、b2 − bc12 **+0.490（+0.410, +0.570）**、b1 − b2 +0.050（−0.023, +0.123）；
对 Greedy，b1 − b2 −0.010（−0.101, +0.081）、bc12 − b2 −0.300（−0.380, −0.220）。held-out 目标场面逐手比较：b1 相对 bc12 丢 9 手、多 3 手（符号检验 p ≈ 0.15）；b2 丢 6、多 6。

战斗统计（对 Random，`diagnose_bc.py report`）：

| 检查点 | 可进战斗阶段时进入 | 攻击 / 局 | 造成伤害 / 局 | 有斩杀仍不打 | 平均回合 | 负局：卡组耗尽 / LP |
|------|------|------|------|------|------|------|
| bc12 | 1,338 / 4,677 = 29% | 2.0 | 2,480 | 111 / 159 | 48.8 | 116 / 23 |
| b1 | 2,268 / 2,271 = 100% | 7.4 | 5,981 | 0 / 135 | 23.4 | 9 / 22 |
| b2 | 2,724 / 2,724 = 100% | 7.4 | 5,714 | 0 / 108 | 28.1 | 21 / 20 |

读法：

- 两种补法都把「进战斗阶段、攻击」学会了（100% 进入、0 次放过斩杀），对局从约 49 回合缩到 23–28 回合，卡组耗尽的负局从 116 降到 9–21：与根因分析的 bc_bp 一致。
- 对 Random 还差 Greedy 约 0.07–0.12：剩下的负局一半是被 Random 打穿 LP（b1 22、b2 20 局）。b1 略好于 b2（+0.05，区间含 0）。
- 对 Greedy 两者都约 0.41，远高于 bc12 的 0.115，但仍低于 0.5（Greedy 自我对局 0.500）。
- **第 1 回合的展开能力**：b2 完全保留（41 / 100，同 bc12，逐手 6 丢 6 得）；b1 降到 35（丢 9 得 3，不显著）。一个可能的原因：b1 含第 2 回合起的全部主要阶段决策（Greedy 的「能发动就发动、能召唤就召唤」），与求解器的第 1 回合线在同一类决策上给出不同的标签（未验证）。
  求解器 held-out 步准确率三者相同（0.585–0.592）。
- 代价：训练样本与时间翻倍（本机 4 核共享：bc12 519 s，b2 3,015 s，b1 3,896 s，后两者与其他任务同时跑）；示范的上限是 Greedy 的水平。

### (a) PPO 从 BC 起步，(c) 先验 KL 只在第 1 回合

**(c) 的实现**：`PPOConfig.kl_prior_turns`（`tools/train_ppo.py --kl-prior-turns N`，默认 0 = 原行为）只对「回合玩家自己的决策、回合数 ≤ N」的行（观测 `globals` 第 2 列
`is_my_turn`、第 3 列 `turn`）计先验 KL，其余行按 0 计入同一个均值，所以被选中的行与不限制时权重相同（[training.md](training.md) §8.2；`tests/test_ppo.py::test_prior_kl_can_be_restricted_to_first_turn_rows`）。
`N = 1` 即求解器示范覆盖的先攻第 1 回合。自博弈里这样的行只在每局开头出现，占训练行的 2.5%（P3）–5%（P5、P6：局短、开局多）。

**设置**（全部 arm 相同）：`tools/train_ppo.py` 的训练器（经一个每 50 次更新另存一份 checkpoint 的脚本调用；训练与 `tools/train_ppo.py tests/decks --updates 200 ...` 相同），
**10 套测试牌组、`--pairings all`**（100 个有序配对含镜像，与 arena 评估同分布；没有用 benchmarks.md 1 小时实验的单牌组对，因为 T4a.2 的验收是 10 套牌上的 arena），
默认 PPO 超参数（VRPO、`ppo_clip` 0.2、熵 0.05、KL 到 EMA 参考 0.05、lr 3e-4、32 局 × 64 步 = 2,048 行 / 次、2 轮 × 512 行、种子 0），**默认的小网络**
（`d_model 64`、局面 / 历史各 1 层、事件窗口 64）。bc12 的网络（`d_model 128`、2 层、窗口 128）试跑时一次更新约 5 分钟（与其他任务同时跑；同样条件下小网络 20–50 秒），
预算内只够几十次更新，所以用**同样的数据与训练配置**另训了小网络版 bc12s 与 b2s（`--d-model 64 --layers 1 --event-length 64`，b2s 用 64 个事件 token 重录的同一批 Greedy 对局）
作热启动与先验；两者的对局胜率与大网络版在误差内，held-out 第 1 回合场面略低（36、37 vs 41，见汇总表）。先验系数 `--kl-prior 0.2`（熵系数的 4 倍），P6 另试 2.0；没有扫更多取值。

**预算：每个 arm 200 次更新 = 409,600 个训练行**，即相同的数据量，而不是相同的墙钟：机器同时在跑评估，P0–P3 四个 arm 并行（各 1 个 PyTorch 线程），
会攻击的策略局短、观测小，同样 200 次更新快一倍。

| arm | 初始化 | BC 先验 KL | 墙钟（收集 + 更新） | 自博弈局数 | 熵：首次 → 末 10 次 |
|------|------|------|------|------|------|
| P0 | 随机 | — | 99 min | 887 | 1.32 → 1.28 |
| P1 = (a) | bc12s | — | 102 min | 862 | 0.78 → 1.22 |
| P2 | bc12s | bc12s，所有行，0.2 | 109 min | 781 | 0.78 → 0.87 |
| P3 = (a)+(c) | bc12s | bc12s，只在第 1 回合，0.2 | 108 min | 936 | 0.78 → 1.25 |
| P4 = (b)+(a) | b2s | — | 47 min | 1,777 | 0.66 → 0.65 |
| P5 = (b)+(a)+(c) | b2s | b2s，只在第 1 回合，0.2 | 50 min | 1,597 | 0.66 → **0.08**（第 180 次更新起崩塌，见下） |
| P6 = 同 P5，系数 2.0 | b2s | b2s，只在第 1 回合，2.0 | 47 min | 1,712 | 0.66 → 0.64 |

P5 在第 131 次更新的收集阶段崩溃（训练器的既有 bug，与本改动无关：`SelfPlaySchedule` 成对发局，第二局沿用第一局抽到的快照 id，两局之间该快照被逐出池时
`SnapshotPool.get` 抛 `KeyError`；会攻击的策略局短、发局频繁才碰上），从第 130 次更新的 `latest.pt` 续训到 200（进行中的对局按续训规则重开；续训时加的「id 已逐出就用最新快照」兜底一次也没触发）。

训练曲线（`metrics.jsonl`，每 50 次更新的均值：策略熵 / 自博弈局的平均回合 / 结束的局数）。训练中没有做对基线的评估（`--eval-every 0`：10 套牌的一次评估就要 400 局），
对 Random 的中间点只有 P5 第 150 次更新一个（汇总表）。

| arm | 更新 1–50 | 51–100 | 101–150 | 151–200 |
|------|------|------|------|------|
| P0 | 1.29 / 53 / 206 | 1.31 / 54 / 227 | 1.27 / 50 / 220 | 1.28 / 50 / 234 |
| P1 | 1.01 / 50 / 195 | 1.14 / 49 / 212 | 1.23 / 49 / 222 | 1.26 / 49 / 233 |
| P2 | 0.84 / 50 / 194 | 0.87 / 51 / 200 | 0.87 / 51 / 186 | 0.89 / 51 / 201 |
| P3 | 1.00 / 51 / 200 | 1.17 / 52 / 218 | 1.18 / 49 / 229 | 1.22 / 43 / 289 |
| P4 | 0.71 / 20 / 377 | 0.77 / 18 / 430 | 0.78 / 16 / 500 | 0.65 / 12 / 470 |
| P5 | 0.75 / 20 / 384 | 0.73 / 17 / 474 | 0.67 / 13 / 475 | 0.41 / 13 / 264 |
| P6 | 0.67 / 18 / 392 | 0.68 / 16 / 447 | 0.67 / 15 / 437 | 0.64 / 14 / 436 |

P1、P3 的熵单调上升（向均匀策略漂移），P2 被先验拉住；从 b2s 起步的 P4–P6 自博弈局从 20 回合缩到 12–14 回合（双方都更早打穿对方），P5 最后 50 次更新熵掉到 0.41、结束的局减半（崩塌）。

### 汇总：所有 arm（同一批 200 局；温度 1 采样）

| arm | 对 Random | 与起点的配对差 | 可进战斗阶段时进入 | 对 Greedy | 与起点的配对差 | held-out 目标场面 argmax / 采样（100 手） | 求解器步准确率 训练 / held-out |
|------|------|------|------|------|------|------|------|
| bc12（基线） | 0.305（0.245–0.372） |  | 29% | 0.115（0.078–0.167） |  | 41 / 28 | 0.721 / 0.585 |
| **b1** = (b) 全部 | 0.845（0.788–0.889） | +0.540（+0.460, +0.620） | 100% | 0.405（0.339–0.474） | +0.290（+0.216, +0.364） | 35 / 35 | 0.719 / 0.592 |
| **b2** = (b) 战斗 | 0.795（0.734–0.845） | +0.490（+0.410, +0.570） | 100% | 0.415（0.349–0.484） | +0.300（+0.220, +0.380） | 41 / 37 | 0.741 / 0.588 |
| bc12s（小网络） | 0.270（0.213–0.335） |  | 12% | 0.120（0.082–0.172） |  | 36 / 23 | 0.680 / 0.594 |
| b2s（小网络 b2） | 0.775（0.712–0.827） | +0.505（+0.420, +0.590） | 100% | 0.380（0.316–0.449） | +0.260（+0.185, +0.335） | 37 / 25 | 0.699 / 0.597 |
| P0 从零 | 0.515（0.446–0.583） | +0.245（+0.161, +0.329） | 52% | 0.095（0.062–0.144） | −0.025（−0.081, +0.031） | 0 / 3 | 0.302 / 0.288 |
| P1 = (a) | 0.515（0.446–0.583） | +0.245（+0.168, +0.322） | 51% | 0.095（0.062–0.144） | −0.025（−0.083, +0.033） | 6（封顶 35） / 1 | 0.393 / 0.414 |
| P2 全状态先验 | 0.305（0.245–0.372） | +0.035（−0.035, +0.105） | 15% | 0.100（0.066–0.149） | −0.020（−0.066, +0.026） | 26 / 15 | 0.636 / 0.572 |
| P3 = (a)+(c) | 0.525（0.456–0.593） | +0.255（+0.173, +0.337） | 57% | 0.105（0.070–0.155） | −0.015（−0.075, +0.045） | 21 / 9 | 0.591 / 0.548 |
| **P4** = (b)+(a) | 0.925（0.880–0.954） | +0.150（+0.088, +0.212） | 99% | 0.635（0.566–0.699） | +0.255（+0.167, +0.343） | 18（封顶 12） / 9 | 0.530 / 0.506 |
| P5 = (b)+(a)+(c) 0.2，第 200 次（已崩塌） | 0.757（0.694–0.812）* | −0.018（−0.088, +0.053） | 98% | 未测（见下） |  | 26（封顶 3） / 21 | 0.602 / 0.557 |
| P5 第 150 次更新（崩塌前） | 0.905（0.856–0.938） | +0.130（+0.067, +0.193） | 99% | 0.550（0.481–0.617） | +0.170（+0.088, +0.252） | 27（封顶 2） / 14 | 0.595 / 0.547 |
| **P6** = (b)+(a)+(c) 2.0 | 0.855（0.800–0.897） | +0.080（+0.010, +0.150） | 98% | 0.485（0.417–0.554） | +0.105（+0.018, +0.192） | 31（封顶 1） / 20 | 0.652 / 0.580 |

「起点」：b1、b2 对 bc12；P0–P3 对 bc12s；P4–P6 对 b2s。参照：Greedy 对 Random 0.910、对 Greedy 0.500，held-out 目标场面 13（argmax）；Random 对 Random 0.495。

读法：

- **(a) 单独不够**：P1（bc12s 热启动、无先验）200 次更新后对 Random 0.515、对 Greedy 0.095，与从零开始的 P0 **完全一样**（两者都约等于 Random）。
  熵从 0.78 涨到 1.22（P0 1.28）：熵奖励（0.05）加上稀疏的终局奖励，在这个预算里把 BC 的偏好稀释成接近均匀的策略——进战斗阶段的比例从 12% 回到 Random 的 51%，
  而第 1 回合的展开能力几乎全丢（held-out 目标场面 36 → 6 / 1）。「从 bc12 起步的 PPO 自己学会进战斗阶段」在 41 万行内没有发生。
- **所有状态上的 BC 先验把策略钉在 BC 上**（P2）：对 Random 0.305、进战斗阶段 15%，与 bc12s 没有差别（+0.035，区间含 0）；它保住了一部分第 1 回合能力（26 / 15），但也保住了「不进战斗阶段」——
  正是根因分析预言的副作用。
- **(c) 只在第 1 回合加先验**（P3）不再阻止攻击（0.525、进入 57%，同 P1），第 1 回合的步准确率比 P1 高（held-out 0.548 vs 0.414），但系数 0.2 保不住展开：目标场面 21 / 9。
  第 1 回合的行只占约 2.5%，而共享网络的其余部分在漂移。
- **(b) 之后再 PPO 最好**（P4）：从 b2s 起步，200 次更新后对 Random **0.925（0.880–0.954）**（= Greedy），对 Greedy **0.635（0.566–0.699）**——**唯一显著超过 Greedy 的 arm**；
  但第 1 回合的展开同样被侵蚀（37 → 18 / 25 → 9，argmax 下 12 手在 300 步内打转）。
- **(b) + (c)**：P6（系数 2.0）保住了大部分第 1 回合能力（目标场面 31 / 20，相对 b2s 逐手丢 11 得 5，不显著；求解器步准确率 0.652 / 0.580），同时对 Random +0.080（+0.010, +0.150）、
  对 Greedy +0.105（+0.018, +0.192），但提升小于 P4。P5（系数 0.2）在第 150 次更新时对 Random 0.905、对 Greedy 0.550，第 1 回合 27 / 14；之后在第 180 次更新附近**熵崩塌**
  （0.53 → 0.04，自博弈局陷入循环不再结束，第 195 次起每段结束的局为 0），第 200 次的策略对局时大量打转：对 Random 用 `--max-decisions 4000` 才跑完（107 / 200 局撞上 4,000 个决策的上限、按终局 LP 判或平局，胜 / 负 / 平 133 / 30 / 37，表中打 *），对 Greedy 的 arena（无决策上限）没有在合理时间内跑完，未测。

### 结论与建议（#28）

- **T4a.2 的「Arena 对 Random 明显 > 50%」只有 (b) 在 BC 阶段达到**：b2 0.795（0.734–0.845）、b1 0.845（0.788–0.889），区间下限都远高于 0.5；
  b2 的代价为零（held-out 第 1 回合目标场面 41 / 100 不变），b1 少 6 手（不显著）。只靠 PPO（(a)、(a)+(c)）在 200 次更新内都没有明显超过 0.5（P1 0.515、P3 0.525，区间含 0.5）。
- **建议 #28 采用 b2**：求解器第 1 回合示范 + Greedy 的战斗阶段示范（进战斗阶段、战斗阶段内的决策）1 : 1 混合做 BC，作为 T4a.2 的交付检查点，也作为 T4b.5 的 PPO 热启动。
  b1 对 Random 略好（+0.05，不显著）但更大、稀释第 1 回合，不推荐为默认。
- **PPO 阶段**：从 b 型 BC 起步（P4：对 Random 0.925 = Greedy，对 Greedy 0.635，唯一显著超过 Greedy 的 arm），**先验 KL 只加在第 1 回合**（(c)）。
  不加先验时 PPO 用第 1 回合的展开换胜率（P4 held-out 37 → 18，逐手丢 22 得 3）；系数 2.0 的第 1 回合先验（P6）保住了大部分（31，丢 11 得 5，不显著），
  代价是对局提升变小（P6 − P4：对 Random −0.070（−0.125, −0.015），对 Greedy −0.150（−0.228, −0.072））。**不要**在只含第 1 回合的 BC 上对所有状态加先验（P2：钉死「不进战斗阶段」）。
  系数 0.2（P3、P5）太弱，保不住展开；两者之间的取值、以及胜率与展开的取舍，留给 T4b.5 在更长的训练上决定。
- **没有结论的部分**（预算所限，如实）：每个 arm 只有一个种子、200 次更新（约 41 万行、800–1,800 局自博弈），PPO 的曲线还远没有收敛；先验系数只试了 0.2 与 2.0；
  P5 在第 180 次更新附近熵崩塌到 0.04（自博弈局陷入循环、不再结束），是否与 P5 的续训有关、是否会在其他 arm 的更长训练中出现，都没有查；
  (a)「BC 热启动对 PPO 有没有用」只能说在这个预算下 bc12 热启动（P1）与从零开始（P0）没有差别，b2 热启动（P4）则明显有用。
- **设计文档**：需要改 [03-play-policy.md](design/03-play-policy.md) I4（本文不改设计文档）。建议措辞：「**先攻展开求解器 → 示范**：用 ygo-combo-solver 离线求解各牌组起手的展开线作第 1 回合示范，
  第 2 回合起的战斗决策（进战斗阶段、战斗阶段内的选择）用启发式 agent 的示范补足，两者混合做 BC 预热；KL 先验只加在求解器示范覆盖的状态（先攻第 1 回合、回合玩家自己的决策），再 RL（ExIt 思路）」。
- **训练器的既有 bug**（P5 碰到）：`SelfPlaySchedule` 成对发局时第二局沿用第一局抽到的快照 id，两局之间快照被逐出就在 `SnapshotPool.get` 抛 `KeyError`，训练中断。
  修法之一：发局时就解析快照模型（或让池子保留被待发局引用的快照）。未在本分支修复（不属于本任务、文件由 T4b.4 维护）。

### 复现

`out/` 不入库；以下命令重建本节的全部产物（4 核共享机器上合计约 5 小时，大头是 PPO 与 arena 评估）。

```bash
D="--train out/demos/bc_train.jsonl --heldout out/demos/bc_heldout.jsonl"
# (b) Greedy 示范：400 局，2 进程约 2 分钟；小网络用 64 个事件 token 的同一批对局另录一份
uv run --frozen python tools/greedy_demos.py --out out/greedy_demos/train.npz --seed 7001
uv run --frozen python tools/greedy_demos.py --out out/greedy_demos/heldout.npz --seed 9001
uv run --frozen python tools/greedy_demos.py --out out/greedy_demos/train_e64.npz --seed 7001 --event-length 64
# 基线重训与 b1 / b2（bc12 的网络）
uv run --frozen python tools/train_bc.py $D --out out/bc12 --epochs 12 --threads 2 --baselines
uv run --frozen python tools/train_bc.py $D --out out/bc_b1 --epochs 12 --threads 2 --openings heldout \
    --extra out/greedy_demos/train.npz --extra-subset all --extra-max 8656 --extra-heldout out/greedy_demos/heldout.npz
uv run --frozen python tools/train_bc.py $D --out out/bc_b2 --epochs 12 --threads 2 --openings heldout \
    --extra out/greedy_demos/train.npz --extra-subset battle --extra-max 8656 --extra-heldout out/greedy_demos/heldout.npz
# PPO 用的小网络版
S="--d-model 64 --layers 1 --event-length 64 --epochs 12 --threads 2 --openings heldout"
uv run --frozen python tools/train_bc.py $D --out out/bc12s $S
uv run --frozen python tools/train_bc.py $D --out out/bc_b2s $S --extra out/greedy_demos/train_e64.npz --extra-subset battle --extra-max 8656
# PPO：每个 arm 200 次更新（P0–P3 当时并行跑）
P="tests/decks --pairings all --d-model 64 --layers 1 --event-length 64 --eval-every 0 --seed 0 --updates 200 \
   --torch-threads 1 --collect-threads 1 --env-threads 1"
uv run --frozen python tools/train_ppo.py $P --out out/ppo/P0
uv run --frozen python tools/train_ppo.py $P --out out/ppo/P1 --init-from out/bc12s/policy.pt
uv run --frozen python tools/train_ppo.py $P --out out/ppo/P2 --init-from out/bc12s/policy.pt --bc-prior out/bc12s/policy.pt --kl-prior 0.2
uv run --frozen python tools/train_ppo.py $P --out out/ppo/P3 --init-from out/bc12s/policy.pt --bc-prior out/bc12s/policy.pt --kl-prior 0.2 --kl-prior-turns 1
uv run --frozen python tools/train_ppo.py $P --out out/ppo/P4 --init-from out/bc_b2s/policy.pt
uv run --frozen python tools/train_ppo.py $P --out out/ppo/P5 --init-from out/bc_b2s/policy.pt --bc-prior out/bc_b2s/policy.pt --kl-prior 0.2 --kl-prior-turns 1
uv run --frozen python tools/train_ppo.py $P --out out/ppo/P6 --init-from out/bc_b2s/policy.pt --bc-prior out/bc_b2s/policy.pt --kl-prior 2.0 --kl-prior-turns 1
# 评估。CKPT = BC 的 policy.pt，或 PPO 的 actor 导出成的策略检查点（下一行），这样 PPO 与 BC 都走同一条 NetPolicy 路径
# （diagnose_bc.py 也直接接受 PPO 的 checkpoint，结果逐局相同；ygorl arena 的 policy:<PPO checkpoint> 走 C++ 锁步路径，同分布但采样序列不同）
uv run --frozen python -c "import sys; from ygorl.nets.agent import save_checkpoint; from ygorl.train.checkpoint import load_actor; \
    p = load_actor(sys.argv[1]); save_checkpoint(sys.argv[2], p.net, p.vocab, event_length=p.event_length)" out/ppo/P4/checkpoints/latest.pt out/ppo/P4/policy.pt
OMP_NUM_THREADS=1 uv run --frozen python tools/diagnose_bc.py play --agent bc --name NAME --checkpoint CKPT --games 200 --workers 2 \
    --out out/eval/NAME_random.json            # 对 Random（P5 另加 --max-decisions 4000，见上）
OMP_NUM_THREADS=1 uv run --frozen ygorl arena tests/decks --agent-a policy:CKPT --agent-b greedy --games 200 --workers 2 --out out/eval/NAME_greedy.json
uv run --frozen python tools/train_bc.py $D --out out/eval/NAME --no-train --checkpoint CKPT --openings heldout [--sample-openings]
uv run --frozen python tools/train_bc.py $D --out out/eval/bc12_battle --no-train --checkpoint out/bc12/policy.pt --openings none \
    --extra-heldout out/greedy_demos/heldout.npz --extra-subset battle      # 基线在 Greedy held-out 上的步准确率
uv run --frozen python tools/diagnose_bc.py report out/eval/*_random.json --baseline bc12
```

本节用的 PPO checkpoint 是训练脚本在第 200 次更新时另存的副本（`update_000200.pt`），与 `--updates 200` 结束时的 `latest.pt` 相同。

## 局限与待决定

- **Arena 验收**：只用求解器示范的 bc12 未达成（上节）；三个方向的实验见「补救实验」：(b) 的 b2 达成且不损失第 1 回合，推荐采用。
  采用前需要先改 [03-play-policy.md](design/03-play-policy.md) I4（示范来源加上启发式 agent 的战斗决策；KL 先验只加在示范覆盖的状态上）。
- **启发式示范的上限是 Greedy**：b1 / b2 对 Random 0.80–0.85、对 Greedy 约 0.41，超过 Greedy 要靠 PPO（P4 对 Greedy 0.635）。PPO 预算内的结论只有一个种子、200 次更新，见「补救实验」的「没有结论的部分」。
- **过拟合**：只有 316 个不同的 plain 起手（+124 条 `--fire` 线），held-out NLL 从第 6 个 epoch 起上升。更多示范（T4a.1 的全量 10 × 1,000 起手）是最直接的改进；
  作 KL 参考时推荐 bc12 这样早停的检查点（bc60 在未见状态上过度自信，KL 项会过强）。
- **只学先攻**：`--fire` 线提供了「被手坑后恢复」的决策，但没有后攻、没有对手的真实互动（对手是白板）。
- **求解器目标是人工指定的**（[solver.md](solver.md)），「场面质量」只衡量是否到达这个目标，不是最优终场；没解出的起手里混有真卡手与预算不足。
- `NetPolicy` 走 Python 编码路径，每步单独前向，arena 评估较慢；PPO 训练用 `EncodedVecEnv`（C++ 编码，与之逐元素一致）。
