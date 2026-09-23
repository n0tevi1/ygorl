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

- **步准确率**（teacher forcing，`step_accuracy`）：在示范线的每个样本上，网络 argmax 是否等于示范动作；同时给出均匀随机猜中的期望（`uniform_accuracy`）作参照。
- **自由对局**（`play_opening`）：从记录的起始对局（与求解器完全相同的种子字、卡组顺序、`DUEL_PSEUDO_SHUFFLE` 与白板对手；求解器没解出的起手用 `start_replay` 按
  `hand_seed` 重建，单测核对重建结果与求解器记录的起始对局一致）让 agent 下第 1 回合，对手以被动选项应答（与示范补完回合相同），
  到第 2 回合第一个决策为止；agent 超过 300 步时由主机用被动选项收尾（`capped`）。终局场面按求解器判定线的标准评分：目标卡是否全部在场（`board_summary_missing`）。
  - **线复现率**：agent 的动作序列（玩家 0）与该起手某条示范线（去掉来回切换后）**逐步相同**的比例；另报「与示范线的最长公共前缀 / 线长」的平均。
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
