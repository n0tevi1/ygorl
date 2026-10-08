# 终局 critic：128→512 更新续训（运行前协议）

128 更新研究已通过继续门槛。本阶段优先检验完整对局收益，不引入 masker、纠错复习、冻结 critic 或新的辅助损失。

## 固定训练

恢复 `terminal-critic-policy-long-2026-10-06` 的全部三个 seed × cold/warm 六个 update128 检查点。
恢复模型、EMA reference、Adam、对手池、schedule、CPU/collector/GPU RNG 和累计计数，不重新初始化优化器。
每臂追加384更新；累计128/256/512为固定评估节点，共追加2,304更新/37,748,736行。
源训练配置不变：128×128 rollout、LR1e-4、FP32、2 epochs、minibatch2048、selfplay .75、PFSP/pool8。
cold/warm表示历史是否经过终局监督预热，恢复时两者actor已经不同，不能声称新一轮起始actor相同。

检查点不含仍在进行的原生对局；六臂均从新对局恢复，保留schedule的已消耗序号。
这不是逐步等价于从未中断的训练。新记录单独保存，不回写旧研究；统计新增终局、错误、截断时扣除源计数。
中断的半成品不自动覆盖或追加。串行GPU训练，CPU四进程评估；整体48小时硬上限，预计24–36小时。

## 固定评估与判定

新评估seed **2026100804**，64个牌组组合/发牌cluster，各交换牌组与先后手4局。
原三对手 Greedy、旧BC256、初始BC128继续等权作为主面板。
第四个对手固定为上一研究 **seed0/warm/update128**，按索引指定，不按本轮成绩挑选；单独报告历史RL对手成绩。
初始BC公共节点以及六臂128/256/512节点，每个节点四对手×256局，共 **19,456局**。
全部候选共用新配对输入；128节点也重新评估，不与旧面板胜率直接相减。
原20套训练牌组不变：新发牌/对手不等于未见牌组泛化。本轮不声称跨牌组分布验收完成。

固定主终点为512更新：主面板 warm512−warm128（训练增长），以及 warm512−cold512（历史预热优势）。
同时报告cold增长、相对初始BC、逐对手成绩和历史RL对手的512−128增长。
20,000次训练seed×deal交叉bootstrap，seed **2026100805**，另报条件deal区间；不按中间成绩追加局数或选checkpoint。
继续扩展warm训练预算须满足：主面板warm增长95% CI下界>0、三个seed增长均>0，且历史RL对手增长95% CI下界>0。
是否保留预热优势另看warm512−cold512，不将cold追平自动解释为停止训练。区间为名义95%，非多重比较校正。

## 保留与健康检查

对六臂128/256/512节点，用已封存256局BC参考任务重算Q/V终局预测，比较512−128。
这仅检查旧任务保留，不作为当前策略校准或有害遗忘的单独证据，不触发自动冻结或调参。
本轮不把13个旧自灰窗口当作自然失误率；一般战术错误的频率需要另建覆盖充分的对局审计。

恢复预检逐项核对六个真实GPU状态与源checkpoint，使用历史非正式评估对局验证推理。
引擎错误、非有限值立即停止；每臂新增终局至少100后新增截断率>1%停止。
任何评估错误/上限停止并保留原始动作轨迹，不替换、不计平。
独立监控检查服务身份、输入SHA、训练/评估进展和心跳；完成后独立复核原始胜负与bootstrap。
issues #61/#83/#92同步研究状态。完成前不发布部分seed的整体棋力结论；本阶段仍不完成#92的两万更新验收。

正式重启工件目录：`out/research/terminal-critic-continuation-restart-2026-10-08/`。
执行脚本：`tools/critic_continuation/`；启动时复制封存到工件目录，服务执行封存副本。

## 预检与执行约束

六个真实ROCm检查点逐项恢复核验已通过，覆盖`state_dict`全部保存字段；四种对手的历史非正式面板对局均健康。
旧研究分数的临时schema fixture通过独立bootstrap复算；故意改动增长均值被拒绝，历史RL对手零增长会阻止扩预算。
该fixture不是512更新结果。64行旧BC任务预测与封存u128预测一致；监控新增计数扣除旧128更新、心跳/STOP/服务错误和身份篡改检查通过。
证据：`preflight.json`、`monitor-preflight.json`及对应脚本/日志。完整回归测试另行记录。

复现需上述历史工件（非git内数据）和同一已编译native。先复制七个脚本至新的研究工件目录；这些脚本按其所在目录寻找兄弟历史工件。
使用本worktree的`PYTHONPATH=src`、`YGORL_THIRD_PARTY=/home/ya0guang/Code/ygorl/third_party`，
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`，运行该目录`preflight.py`。
测试和正式服务均必须显式设置
`YGORL_COMBO_SOLVER=/home/ya0guang/Code/ygorl/out/research/declaration-list-2026-10-05/build/bin/combosolver`。
首次完整测试漏设该变量，落到旧的默认`build/combo-solver/bin/combosolver`，导致三项旧solver回归失败；
指定协议绑定二进制后这三项 **3 passed / 2.57秒**。两种二进制SHA及原失败日志均保留，未修改测试或跳过失败。
`registration.json`记录协议提交、协议SHA、完整测试日志SHA及监控预检SHA；再执行`run.py prepare`封存身份。
`launch.py all`负责训练/评估/保留诊断及最终统计审计；由systemd用户服务whole-cgroup管理，超时48小时、无自动重启。
独立`watch.py --publish`服务读取`monitor/config.json`中的正式服务InvocationID/身份SHA，每30秒检查，
每10分钟或终态更新三个issue。任何失败保留目录，不以同一身份覆盖重跑。


## 首次启动失败与修复

初次协议提交`ec1a775`，身份`8fd93d61ff65b0162dad05930a23354776f78db05bff6491ee339aba47703dca`。
2026-10-08 08:44:57 PDT启动后健康检查停止：公共initial误用BC格式checkpoint，`PolicyAgent`没有
诊断包装器所需的`host`属性；此前四个预检对局只把PPO历史模型放在candidate侧，漏掉这个接口差异。
完整测试已通过 **1548 passed/3 skipped，286.97秒**，但没有覆盖这一研究专用执行路径。
训练零更新、零新增行；仅两条正式评估记录（一胜负、一异常）写入，全部保留在原工件目录，不作为研究结果。
原producer/consumer/retention均已退出，原监控已发布停止状态。

修复恢复原128更新研究的**同actor权重、PPO格式update0**公共initial与初始BC128对手，
明确candidate须支持HostDuel；预检新增BC格式拒绝检查、权重逐张量一致性和实际失败对局完整回放。
重启使用新目录、新身份、eval seed2026100804、bootstrap2026100805（保留诊断2026100806），
原失败面板不再用于正式估计。预算、终点和门槛不变；重新预检和验证后启动，不修改旧STOP或覆盖旧产物。

重启预检通过：同一失败deal/先后手在修复后健康结束（9回合/622决策），实际出现3个素材取消窗口，
三个均按原策略mask屏蔽，0错误；这是一条旧失败用例回归，不计入新评估面板。六个GPU状态恢复、
四种对手历史对局、统计审计、保留预测与监控故障检查均再次通过。


## 正式重启与验证（2026-10-08）

修复预登记提交`3b83845`；完整本机ROCm测试 **1548 passed /3 skipped，288.77秒**，格式/lint通过。
三个跳过为snapshot和两项opt-in网络检查；GitHub当前未返回CI check，未将其称为远端CI通过。
新面板覆盖20套牌组，64个deal seed与旧研究/失败尝试均无交集；`panel-input-audit.json`仅核验输入，没有预看新结果。

正式训练服务`ygorl-terminal-continuation-restart-20261008.service`于 **2026-10-08 08:54:58 PDT**启动，
InvocationID `3b3c2d85654e4881a0de18b0a493f6c1`。独立监控08:55:01启动，InvocationID `bbeed33109de40618f9d0b5b7c9e0270`。
身份SHA `078a307af7b9cf3c80cca588288814728ae8e31e4aa2a456cf73a552a538bb4e`；两服务均为systemd用户服务，whole-cgroup管理，GPU研究48小时上限。

首个新增更新已实际完成：seed0/cold **128→129，16,384新行，63正常终局，0截断/错误**。
GPU使用率观测99%；CPU正式新面板评估与固定BC保留诊断同步运行，监控无报警。
`monitor/publication.json`确认#61/#83/#92已发布running状态，不把旧停止服务算作当前实验。
这些是启动/健康证据，不构成512更新棋力结论。预计24–36小时，实际以进展为准，48小时是硬上限。
