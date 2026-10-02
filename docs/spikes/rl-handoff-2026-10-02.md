# RL 接手记录：测量修正与后续实验

日期：2026-10-02（America/Los_Angeles）。主线基点 `c530ff0`，工作分支 `codex/rl-handoff`。
跟踪：[强度 #83](https://github.com/n0tevi1/ygorl/issues/83)、[训练信号 #61](https://github.com/n0tevi1/ygorl/issues/61)、[搜索老师 #62](https://github.com/n0tevi1/ygorl/issues/62)。

**本轮最新决定**：固定局面 pilot 的公开重排收益未通过完整外部对局复核（−1.28pp，CI [−3.35,+0.79]pp）。
自对弈辅助测试也不确定；不扩训练。先对齐老师样本与部署窗口、加入发动倾向对照，并继续处理 #169 素材搜索复杂度。
后续实现位于 `codex/response-diagnostics`，详细结果见文末。

## 接手时的状态

Claude 的 strength-diag 会话最后一条实质更新在 UTC 2026-10-02 05:41（本地 10-01 22:41），当时两条 400-update 训练已结束、等待最终评估。
`out/why/seat_split_eval/panel.log` 后来写出 `EVAL2_DONE`；27-agent 矩阵完整，自对局报告也完成。没有需要盲目重启的丢失训练。
原 Claude worktrees 和检查点保留，本分支从 main 创建，未把多个未审 PR 整体合并。

| 工作 | 状态 | 目前能支持的结论 |
|---|---|---|
| survival teacher KL=2，400 updates | 训练与旧 panel 完成 | 未证明稳定改善；本次补充明确环境的新种子测试 |
| 先后手分网，400 updates | 训练与旧 panel 完成 | 当前设置未见优势，不排除共享主干条件化方案 |
| 回应效果 oracle 诊断 | 已有 113 个可比局面，200 个基础游戏 | 留出续局上有收益迹象，但用真实隐藏状态，不是合法老师已验收 |
| 共同进化 c151 | 3 轮完成，29,200 评估局，接受 2 个改动 | 卡组搜索的局部增益，不能当成通用策略强度提升 |
| 新增 GPU 训练 | 未启动 | 先测出可靠、可执行的老师，再扩大训练预算 |

## 修正的解释

1. `diagnose_losses` 旧 `1 - within_variance / total_variance` 在每组两局时过拟合。无牌组效应的独立硬币反例也给约 50%；去掉“解释比例”，只留描述性 spread。
   即使改为 ddof=1，牌组/起手/座位仍混杂。识别牌组效应需要同牌组对的多个独立洗牌种子和交叉先后攻。
2. 自对局先攻者得分要汇总双方。survival 旧输出 deck-a 先攻 0.427、后攻 0.549，合并先攻者得分是 0.439；控制组约 0.464。
   各自自对局不是面对同一个固定后手模型的策略干预。
3. 旧 panel 的 baseline 用 10 个对手、新候选用 11 个。统一 11 anchors、对角线约定 0.5 后，baseline 为 0.59682，不是 0.6065。
4. `critic_lam=1` 保留后续 `Vbar-Q_taken` 控制变量与段末 bootstrap，不是完整终局 MC。该臂失败不排除纯终局监督的价值。
5. `Var(Q_hat_taken-Vbar_hat)/Var(A_hat)` 只是当前模型预测量之比。恒为零的 critic 即使在确定性 ±1 bandit 上也会报告零；这不是实际动作信号比例。
6. 有限 top-k oracle 搜索的表现不是真实搜索上限；旧“PIMC 只能更差”推断撤回。弱利用者没找到漏洞同样不能证明不可利用。

默认训练配方保持不变；这些更正使解释与实现一致，并不证明尚未执行的替代算法会更好。

## 旧结果重新计分

来源：`out/why/seat_split_eval/panel.json` 与 `out/strength/panel_agents.txt`。
均为相同 11 anchors 的平均；baseline 对自己 0.5 是对称约定，不是新增实测。

| checkpoint | 11-anchor 得分 | 对 BBL05-u400 |
|---|---:|---:|
| BBL05-u400 | 0.59682 | 0.5000（约定） |
| TEACH_ctrl-u400 | 0.59227 | 0.5025 |
| SURV_kl2-u400 | 0.59636 | 0.5450 |
| SEAT-u400 | 0.59045 | 0.4750 |
| ENTFREE2-u400 | 0.61409 | 0.5700 |
| PLAT_big-u500 | 0.60977 | 0.5700 |
| PLAT_lowlr-u800 | 0.61386 | 0.4975 |

旧表每格 50 配对 × 4 局、环境 stamp 为 null。跨对手复用配对且缺逐局记录，不能从格子均值补出配对差区间。
它是探索性结果，不能与下面 MD meta 表直接相减。

## 新的固定预算评估：已完成

配置在看新结果前固定：环境 `md-2026-09` 的 20 套合法 meta 牌组，均匀抽不同牌组对；种子 `20261002`、128 个 pairing、四重配对、max_decisions=4000。
候选为 `teacher_ctrl/u400`、`survival_kl2/u400`、`bb_lam05/u400`；控制组 teacher_ctrl，外部对手为 `bb_lam05_s1/u400`、`league_s1/u400`、`bb_lam05/u1200`。
每候选 1,536 局，总计 4,608 局。该种子未用于旧 panel；这是新对局测试，不声称这些牌组从未进入训练数据。

主比较为 survival 减控制组，baseline 减控制组是辅助比较。继续扩大训练的门槛为点差至少 +2 pp 且区间下界大于零，然后补训练种子；
不逐次查看结果后追加样本求显著。128 个配对不保证检出 2 pp 效应；区间含零时保留“不确定”，不宣称等效。

| checkpoint | 固定外部 panel 得分 | 相对 teacher_ctrl | 95% 配对 cluster bootstrap 区间 |
|---|---:|---:|---:|
| teacher_ctrl | 0.48864 | 0 | — |
| survival_kl2 | 0.49380 | +0.52 pp | [−2.75, +3.75] pp |
| bb_lam05 起点 | 0.49897 | +1.03 pp | [−2.10, +4.20] pp |

**判断**：survival 未通过继续扩大训练的门槛，当前不增加其 KL 权重或续跑。结果不排除小收益，也不支持宣布 teacher 整体无效。
每格完整逐局结果已保存，相同命令再次运行复用了全部 9 格，没有新增对局，输出相同。

**异常及边界**：12/4,608 局触发 Lua script budget（ctrl 2、survival 5、base 5），涉及 7/128 个 pairing。
为保持配对与分母一致，三行共同剔除这些 pairing，使用 121 个配对、每候选 1,452 局；区间条件于这批无错误配对。
另有 31 局达到决策上限，按当前裁判规则计平局；没有把引擎错误计为平局。
将错误结果在 [0,1] 内作最坏/最好补全，全样本的 survival−ctrl 点差范围为 [+0.13,+0.59] pp（这只是样本缺失值范围，不是置信区间）。
因此这个敏感性检查也未显示达到 +2 pp 门槛的点差；仍需排查非随机错误，不能把剔除当作根治。

**运行与复现**：`out/research/rl-handoff-2026-10-02/run.sh`，产物在同目录的 `md-panel/{manifest.json,cell-*.json,summary.json}`。
脚本 `tools/compare_checkpoints.py`，基点 `c530ff0` 加本分支改动；manifest 记录实际导入源码和工具哈希、core/cdb 哈希、子模块版本、环境及 checkpoint 内容哈希。
使用原 ROCm 环境、单评估进程、CPU nice=19，未替换 torch。命令参数与记录格式见 [evaluation.md](../evaluation.md)。
区间来自整体配对重采样 10,000 次，仅覆盖固定检查点、固定对手下的对局采样；不包含训练种子方差，也不是多重比较校正后的结论。

## 下一步验收

| 优先级 / 跟踪 | 下一步 | 验收 |
|---|---|---|
| P0 / #83 | 统一外部 panel、逐局记录、配对区间、环境与代码指纹 | 本分支完成；保留旧实验口径，后续候选用新种子复核 |
| P0 / [#169](https://github.com/n0tevi1/ygorl/issues/169) | 12 次错误已复现；继续处理素材搜索复杂度 | 在固定回放上降低调用成本，并验证消息/合法动作/终局语义一致；不只提高预算 |
| P1 / #62 | 对齐老师训练与部署分布，加入简单发动倾向对照，再研究完整回应动作 | 固定候选/对手/首回应窗口收集；按配对分训练验证；新种子完整对局复核至少 +2pp 且 CI 下界 >0 |
| P1 / #61 | 用同局面的成对续局校准 critic 排序与优势符号 | 分回应/展开/解场报告排序准确率、Delta Q 区间；不用全局 EV 或预测方差比例代替 |
| P2 / #61 | 学生真实访问状态上的示范与蒸馏 | 报真实洗牌/干扰下整条线成功率、恢复能力、PPO 后保持率；老师通过门槛后补多训练种子 |
| P2 / #61 | 完整终局 MC 的 matched 对照 | 明确 z 标签、按游戏隔离验证、相同 actor 起点/预算；不能复用 λ=1 旧臂名称 |

survival teacher 原路线使用 `DUEL_PSEUDO_SHUFFLE`，真实执行约 35% 路线偏离是已知限制；提高旧路线 KL 不能修复此分布偏移。
搜索与网络交替改进可参考 [ExIt 原论文](https://arxiv.org/abs/1705.08439)，但其 Hex 结果不保证直接推广到这个不完全信息游戏。
报告不确定性的原则参考 [Agarwal 等，2021](https://arxiv.org/abs/2108.13264)；本工具只做固定检查点下的配对 bootstrap，不冒充多训练种子的算法评估。


## 接手后的回应诊断与错误复现（2026-10-02）

实施分支 `codex/response-diagnostics`；原始数据根目录 `out/research/response-diagnostics-2026-10-02/`。
本轮分为错误复现、固定局面诊断、冻结公开信息重排器的完整对局复核；不使用旧 panel 的点估计追跑训练。

### 脚本预算：定位完成，复杂度尚未修复（#169）

Lunalight Fusion 失败点在 2 倍预算后继续，Kewl Tune Mix 在 5 倍预算后继续，至少这两个是有限但昂贵的素材检查。
对原 12 个错误局做 5 倍预算隔离重试，11 局完成；base–league / pairing 73 / game 0 仍在 T10 / decision 333 超限，
栈指向 `proc_synchro.lua` 的手牌素材检查与 `CheckP42` 递归。全部重试前缀在原预算下复现相同失败回合、决策数和 LP。
[PR #171](https://github.com/n0tevi1/ygorl/pull/171) 保存两个无 GPU 的真实回放 fixture，并把本次核心调用的首条 traceback 带入批量错误结果。
预算默认值未变；下一步应在语义一致的条件下优化素材搜索，不能把提升预算当成已修复。

仅作敏感性分析，用成功重试替换缺失结果并共同排除剩余一组后，127 个配对上 survival−control 为 +0.46pp，
95% CI [−2.62,+3.58]pp；baseline−control 为 +0.82pp [−2.23,+3.90]pp。原始 panel 不改写，扩大 survival 训练的门槛仍未通过。

### 固定局面 pilot：公开信息重排有信号，critic 未测出收益（#61 / #62）

`tools/response_probe.py`，BBL05-u400，md-2026-09，seed=20261003，64 个四局配对，共 256 个回应局面 / 18,400 次续局。
所有候选共享续局种子，前 16 次用于选动作、后 16 次独立评分；偶数配对给公开重排器训练（151 个动作差标签），
奇数配对留出（128 个局面、32 个 cluster）。6 次续局错误按同局面全部候选共同剔除，没有剔除完整配对。

| 选动作方式 | 相对原策略的回应方终局得分差 | 95% cluster CI |
|---|---:|---:|
| 同一真实隐藏状态下的有限续局选动作（诊断 oracle） | +3.86pp | [+2.00,+5.85]pp |
| 公开 actor 表征 + 冻结 ridge 重排 | +2.59pp | [+0.51,+4.80]pp |
| 特权 critic Q 最大值 | −0.23pp | [−1.56,+1.21]pp |
| 原 actor argmax | +0.24pp | [−0.41,+0.89]pp |
| 总是 pass | +0.52pp | [−2.66,+3.41]pp |

这不是固定候选玩家的完整对局胜率：每个原局只取第一个合格回应方，再比较该方的反事实续局得分。
代价、对象和后续动作仍由原策略采样，不是完整回应线优化或已实现 PIMC，也不是理论搜索上限。
公开重排器推断只调用 actor 的合法观测表征，特权续局可用于训练标签但 Q/真实隐藏状态不进入推断。

覆盖有明显偏向：200/256 个窗口在 T1，46 个在 T2；121 个空连锁、134 个链长 1、1 个链长 2，主要是手坑。
留出集里公开重排在空连锁 56 个局面的条件均差 +3.21pp、非空连锁 72 个 +2.11pp；这两个探索性子组各自未取得排除零的贡献区间。
不能将结果直接推广到后期场面打断。独立评分里按名义 1.96×SE 筛出的 21 个动作差，critic 符号只对 7 个；
筛选存在多重比较与小样本噪声，只是诊断线索，不是严格置信的“正确动作”标签，更不能据此宣称 critic 实现有 bug。

复现产物：`pilot/{manifest.json,states.json,obs-*.npz,rollouts-*.json,public-reranker.npz,summary.json,strata.json}`。
manifest 固定 checkpoint/core/工具/统计模块哈希与环境；原始局面和逐次结果均保留，可重新评分。


### 冻结重排器的完整对局：未通过训练门槛

局面 pilot 通过后，按 #62 预登记的新 seed=20261004 运行 128 个四局配对、三个固定外部对手，合计 3,072 局。
候选方只在自己前 4 回合第一个合格回应窗口重排一次，其余步骤与原策略相同；环境 `privileged=False`。
权重保持冻结，不用新结果重拟合。1 个 control 局触发脚本预算，共同排除该 pairing，剩余 127 组 / 每候选 1,524 局。

| 策略 | 等权外部 panel 得分 | 相对原策略 | 95% cluster CI |
|---|---:|---:|---:|
| 原 BBL05-u400 | 0.50656 | — | — |
| 冻结公开重排器 | 0.49377 | −1.28pp | [−3.35,+0.79]pp |

对 seed1 / league / long 的点差分别为 +0.39 / −2.46 / −1.77pp。原策略 1,536 局里 1,382 局有合格窗口，861 局会改动作；
三组对手的干预前窗口、原采样动作、重排动作逐局一致，未改动作的全部 675 局终局也完全一致。
另有 48 局 control 冒烟回归与 #170 的独立评估器逐局匹配胜负、回合、决策数、LP、种子与先后攻。

**决定**：不扩大该 reranker，不启动其 PPO 蒸馏。固定局面上的 responder 得分提升不能代替完整对局的强度复核；
这个负结果同样不能证明所有回应老师无效。后续诊断保持权重与完整对局协议，换回 pilot 的自对弈对手，
用 seed=20261005、128 个四局配对检查更接近训练的对手分布；它仍不同于 pilot 的“每局第一个回应方”选样，不能唯一归因。

探索性复用 pilot 数据：仅用训练半份选出的 pass-logit bias=−4，在留出评分半份为 +1.66pp [+0.15,+3.30]pp，
总是发动（多动作时取原 actor 最高分的非 pass）为 +1.76pp [+0.25,+3.39]pp。没有新增独立测试，不能当作确认性发现；
它提示下轮要加入简单发动倾向的对照，不能把 ridge 的局面收益直接归因于更好的时机推理。

完整对局工具 `tools/evaluate_response_reranker.py`；产物 `deployed-panel/{manifest.json,control-*.json,reranked-*.json,summary.json,validation.json}`。
manifest 绑定公开重排权重、检查点、环境、实际 core/cdb/Python 源码与脚本版本；按完整格子恢复，不改写已完成实验。


### 自对弈辅助复核：仍不确定，不推翻主实验

相同冻结重排器与完整对局协议，对手换回 BBL05-u400，seed=20261005，128 个四局配对，共 1,024 局、零引擎错误。
原策略 0.50000，重排器 0.51758；差 +1.76pp，95% CI [−0.98,+4.49]pp。460/512 个候选局有窗口，250 局改变动作。
这既没有通过门槛，也不足以证明外部对手变化是唯一原因。样本规模、老师只有 151 个训练标签、回应方选样差异和有限续局噪声仍混杂。
结果在 `selfplay-diagnostic/summary.json`。本轮固定样本完成，不根据区间位置追加样本求显著。

**可执行的后续顺序**：先按真实部署方式（固定候选、固定对手、候选自己的首个回应窗口）重建训练/验证采样，
加入训练集选择的 pass-bias 与总是发动对照；再用未见种子完整对局验证。只有该门槛通过，才进入 matched 蒸馏/PPO 与多训练种子。
发动后的代价/对象当前仍由原策略处理；完整回应线搜索是另一个尚未完成的实验轴，不能把本轮负结果解释为其上限。

**验证**：新 native 编译通过；全套测试 1,336 passed / 9 skipped，15 个缓存路径 setup error 在设置 `YGORL_CACHE_DIR` 后全部通过，
合计 1,351 项通过。GPU 测试在沙箱内跳过；GPU 侧已实际跑完上述实验和 48 局 control 对照。presubmit lint/format 通过。
运行用共享 ROCm venv，显式 `PYTHONPATH`；未执行依赖同步、未修改原 Claude worktrees/检查点/默认训练配方。


复跑命令（在已配置好的 ROCm Python 环境与本分支中执行；原运行设 `PYTHONPATH=src`，未同步依赖）：

```sh
export PYTHONPATH=src
python tools/response_probe.py --checkpoint out/why/bb_lam05/checkpoints/update_000400.pt --pairings 64 --continuations 32 --seed 20261003 --out out/research/response-diagnostics-2026-10-02/pilot
python tools/evaluate_response_reranker.py --pilot out/research/response-diagnostics-2026-10-02/pilot --panel out/research/rl-handoff-2026-10-02/md-panel/manifest.json --pairings 128 --seed 20261004 --out out/research/response-diagnostics-2026-10-02/deployed-panel
python tools/evaluate_response_reranker.py --pilot out/research/response-diagnostics-2026-10-02/pilot --panel out/research/response-diagnostics-2026-10-02/selfplay-diagnostic-panel.json --pairings 128 --seed 20261005 --out out/research/response-diagnostics-2026-10-02/selfplay-diagnostic
```

第二条命令完成后再次运行已确认复用全部 6 个格子，无新增对局。改变代码/权重/环境时 manifest 拒绝原目录，需使用新的输出目录。
