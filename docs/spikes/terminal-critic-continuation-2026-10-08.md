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

新评估seed **2026100801**，64个牌组组合/发牌cluster，各交换牌组与先后手4局。
原三对手 Greedy、旧BC256、初始BC128继续等权作为主面板。
第四个对手固定为上一研究 **seed0/warm/update128**，按索引指定，不按本轮成绩挑选；单独报告历史RL对手成绩。
初始BC公共节点以及六臂128/256/512节点，每个节点四对手×256局，共 **19,456局**。
全部候选共用新配对输入；128节点也重新评估，不与旧面板胜率直接相减。
原20套训练牌组不变：新发牌/对手不等于未见牌组泛化。本轮不声称跨牌组分布验收完成。

固定主终点为512更新：主面板 warm512−warm128（训练增长），以及 warm512−cold512（历史预热优势）。
同时报告cold增长、相对初始BC、逐对手成绩和历史RL对手的512−128增长。
20,000次训练seed×deal交叉bootstrap，seed **2026100802**，另报条件deal区间；不按中间成绩追加局数或选checkpoint。
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

工件目录：`out/research/terminal-critic-continuation-2026-10-08/`。
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
