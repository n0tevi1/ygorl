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
