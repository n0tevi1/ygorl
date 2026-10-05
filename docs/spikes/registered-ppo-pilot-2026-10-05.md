# 修正 BC 后的可观测 PPO pilot（2026-10-05，训练前协议）

目的：在较长预算前验证干净训练、梯度/critic 状态和独立强度曲线，继续调查 BC→RL 漂移。
这不是 #92 的20,000更新验收；单训练种子、32更新不能回答长期容量收益。
选择修正课程末尾64×1与128×2，不挑epoch。全部20合法md-2026-09牌组，
TrainConfig/PPOConfig当前默认算法（PPO/VRPO return、lambda .5、target_kl .01、PFSP、
128env×128steps、2epochs×2048minibatch、selfplay .75、pool8、snapshot10），
同seed2026100518、FP32/TF32 off、无AMP/overlap、无prior/pinned、critic_warmup=0。
初始critic按默认随机初始化，不把实验改成不同优化器配方。
运行差异仅为模型结构/BC权重；实际收集受异步引擎调度影响，不宣称逐轨迹配对训练。

每臂32更新（524,288行），顺序使用GPU，64后128；登记update0/16/32，关闭同步eval，
checkpoint_every16，log_games开启，CPU collect/torch线程2。
每个rollout在优化前检查全部终局的error/Lua/retry/unknown；异常立刻存诊断并停止该臂，
不晋级故障权重。非有限指标停止。累计≥100终局后若截断率>1%停止并调查，不悄悄补游戏。
其它早停只能明确记录原因，不按见到的胜率挑检查点。全部3个固定节点都进入矩阵。

新矩阵seed2026100519，10个预抽样有序异牌组配对，每cell40局（换驾驶牌组、双先后攻）。
固定基线Random、Greedy、原未修正256×2（旧四容量固定面板Greedy点估计最高32/80，
不是已证明最强历史策略）。初始两模型的update0也登记，便于与各自起点比较。
使用CPU独立消费者、2workers/每worker1 Torch线程、nice19，不占训练GPU。
矩阵牌组按deck.name排序，配置/种子/牌组与checkpoint均绑定身份；正式长跑另需更充足面板。

消费者保留每局GameSpec身份/agent seeds/结果全部健康字段，并在发布前检查。
任何异常或上限均保留并停止该批发布，先调查，不将LP上限胜利当正常win。
研究驱动对Arena.play作只读结果记录包装，登记/矩阵扩展使用生产实现。
报告所有检查点对基线胜率/排名及update和累计活动wall秒数，同时列训练终局数、截断、
实际minibatches/target-KL停步、entropy、critic EV、Q/V及loss；不把训练loss当强度。

继续条件：流水线健康且无大规模能力退化，再预注册更长双配置/多种子计划；若退化，
先依据数据区分critic初始化、KL/熵漂移、课程覆盖/决策缺口，不直接扩大GPU预算。

## 完成结果

producer保持冻结3bd1856（生产实现9bd7871、旧post-step KL guard）。两臂各完成32更新/524,288 rows；
64×1有2,144个终局，128×2有1,964个终局，全部健康、零截断。六个固定节点全部完成独立矩阵：
9 agents、36 cells、**1,440局均正常胜负**，没有Lua/retry/unknown/undecodable/error或上限。

每个表格单元40局；下列固定baseline胜率不是容量间逐轨迹配对训练。

|模型|更新|对Greedy|对旧256×2|对Random|
|---|---:|---:|---:|---:|
|64×1|0|37.5%|42.5%|72.5%|
|64×1|16|32.5%|57.5%|65.0%|
|64×1|32|47.5%|45.0%|77.5%|
|128×2|0|15.0%|47.5%|65.0%|
|128×2|16|47.5%|55.0%|70.0%|
|128×2|32|35.0%|55.0%|70.0%|

对各自PPO update0，按相同seed/decks/先后攻配对，10 seed clusters×4局，bootstrap20,000、seed2026100520。
update32 − update0的百分点差及95%区间：

|模型|Greedy|旧256×2|Random|
|---|---:|---:|---:|
|64×1|+10.0 [+2.5,+17.5]|+2.5 [−12.5,+15.0]|+5.0 [−15.0,+25.0]|
|128×2|+20.0 [+2.5,+35.0]|+7.5 [−7.5,+20.0]|+5.0 [−12.5,+22.5]|

这些是未作多重比较校正的探索区间、单训练seed；10个随机有序牌组对并不均衡覆盖全部20牌组。
不能据此宣布显著整体提升、128×2胜出或已经跨过长期验收阈值。128×2的Greedy曲线从16到32回落，
因此不筛掉最后一个节点只报最高点。矩阵完整排名保留在analysis.json，新增agent改变了排名参照集合，
强度曲线优先读固定baseline。

|模型|实际Adam步 / 最大512|触发KL早停的更新 /32|Q EV首→末|entropy首→末|活动wall秒|
|---|---:|---:|---:|---:|---:|
|64×1|380|18|−3.78208→.43605|.79942→.84034|779.210|
|128×2|225|32|−.28729→.17688|.71865→.81347|1121.359|

Q EV是各自rollout上的预测指标，不能跨状态分布当因果比较；不是critic问题全部解决的证据。
活动wall由registration读取（metrics.jsonl当时的wall计数落后一更新），机器同时有其它任务，不能视为隔离成本基准。
图表按update和wall两种横轴完整报告：`out/research/registered-ppo-pilot-2026-10-05/strength-curves.svg`（另有PNG）。
原始配置/实现身份/登记/1440局/统计/图表与失败日志封存于同目录validation.json。
画图初次因训练环境无matplotlib失败，数值分析已完成；之后用临时隔离matplotlib环境生成图，未改项目依赖。

另一个比较边界：BC PolicyAgent用Python random的inverse-CDF，而PPO CheckpointAgent用torch multinomial。
相同权重不保证相同seed动作；不能把前一BC面板与本次PPO update0差异解释为训练退化。
各自PPO 0/16/32使用同一adapter和固定面板。

**下一步根因**：[PR197](https://github.com/n0tevi1/ygorl/pull/197)已修复“已测出KL超限仍多执行Adam”。
固定真实rollout显示第一Adam步本身仍严重过冲，见[首步诊断](ppo-first-step-2026-10-05.md)。
先完成同数据loss分解/LR敏感性，再立下一轮训练协议；本pilot不是#92的20,000更新交付。
