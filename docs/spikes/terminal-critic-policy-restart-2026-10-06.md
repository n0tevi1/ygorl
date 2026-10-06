# 融合修复后的终局critic PPO对照重跑（2026-10-06，运行前协议）

旧研究在seed1/cold采样触发融合脚本预算，70/192更新后停止；原数据、STOP和异常证据保留。
[融合失败状态记忆修复](fusion-repeat-memo-2026-10-06.md)通过完整检查并合并后，另建
`out/research/terminal-critic-policy-restart-2026-10-06/`，从六个分支的起点全部重跑。
不继续旧optimizer、不删除STOP、不复用旧评估cell或拼接旧训练数据。

## 固定设计与预算

权重来源仍为`terminal-critic-fit-2026-10-06`三个critic初始化的epoch0/epoch1。
同对cold/warm actor相同，只有critic私有权重不同，fresh Trainer/Adam/pool/counters；
真实GPU预检再次验证三对actor/reference actor和起始RNG/schedule。预热模型选择不再改变。
三个新PPO seed为 **2026100551/52/53**；异步事件顺序不保证同RNG逐步同轨迹。

继续[原协议](terminal-critic-policy-2026-10-06.md)的六臂×32更新、共**192更新/3,145,728行**，
LR1e−4、lambda .5、128env×128steps、FP32/TF32 off、2epochs×2048 minibatch，
其余参数及epoch0/1私有权重转移不变。保存不可变0/16/32节点与各臂首个实际rollout。
warm额外使用共享512游戏/162,019行的一轮监督训练，80 Adam步/critic，不声称相同总计算/样本预算。
此前拟合、确认、旧中断PPO和本次修复验证的成本另行保留，不隐藏重试成本。

新eval seed **2026100554**：64个ordered deck-pair/deal cluster×交换牌组/先后手4局，
对手Greedy/原BC256/初始BC128等权。公共initial768局，六臂各16/32节点每节点768局，
共 **9,984新评估局**。新bootstrap seed **2026100555**，20,000次交叉重采样三个训练seed和64个deal cluster。

唯一主终点仍为update32：warm−cold及warm−initial的交叉95%区间下界均>0、三个seed的warm−cold差均>0，
才进入更长预算确认。update16仅描述，不能更换终点或在本面板追加样本求显著。
三训练seed、共享critic训练语料、固定三个对手都限制外推，不是#92两万更新或顶尖对局验收。

## 版本、健康与编排

原训练critic来自修复前脚本，但已验证修复对原失败的高预算成功参照及128新对局完整消息等价；
研究身份绑定实际Python/native/上游脚本/CDB/环境/牌组、四个driver、模型SHA和本次修复验证报告。
修复不改变游戏奖励、动作空间、卡片脚本条件或默认指令预算，未知脚本版本仍保持不变。

沿用严格健康规则：训练引擎错误/非有限值即停；至少100个训练终局后累计截断>1%停止。
任何评估错误/上限停止全研究并保留原局动作；不计平、替换或补局。
独立systemd用户服务、8小时上限、whole-cgroup管理、4-worker单线程CPU评估与单GPU训练并行。
原子阶段记录和SHA核验，可恢复完成阶段/分析收尾；不覆盖半份训练或评估。
正式数据只能在协议提交、完整测试通过、修复合并及GPU预检通过之后产生。

## 运行前核验

协议/修复提交`ceda223`早于新数据；完整ROCm检查1,511 passed/3 skipped。
三个真实GPU epoch0/1配对再次验证actor/reference actor与初始RNG/schedule相同、Adam/pool/计数为空；
历史非研究Arena游戏正常完成。4个研究driver lint/format/compile通过，预检SHA已保存。
截至本记录尚未生成新训练行，等待修复合并后冻结研究身份并启动独立服务。

## 正式启动

修复[PR #212](https://github.com/n0tevi1/ygorl/pull/212)已合并为`ff3d28d`。
正式研究身份SHA `67b4fc0aa9a32a2576f633c1dfa5671542c43b9e7cdb7708a8ea70bc2f0a750b`，
源码`ceda223`、native未改，四个driver/模型和修复验证报告已绑定，旧研究STOP也纳入来源记录。
独立服务`ygorl-terminal-policy-restart-20261006.service`于2026-10-06 **13:19:06 PDT**启动，
InvocationID `2365b9fa1c2e4ffe8580f0e081d0cfa3`，工作目录`codex-fusion-repeat-bound`。

已验证首个seed0/cold实际更新：16,384行、4个正常训练终局、0截断/错误，约621 rows/s（单次更新，非吞吐对照）。
训练producer与评估consumer同时运行，initial/Greedy原始评估记录已开始写入，无STOP。
尚无新三seed棋力结论；运行快照不等于后续持续健康，最新状态须检查service和`pipeline-status.json`。

## 独立持续监控（15:11 PDT启用）

监控服务`ygorl-terminal-policy-monitor-20261006.service`与训练服务分开，
InvocationID `a8eac145c75549948bd4c07f6a390abf`；30秒检查一次，8小时上限，独占写锁。
只读取已登记研究，记录独立目录`out/research/terminal-critic-policy-monitor-2026-10-06/`，
不改训练参数、driver、STOP或原始数据，也不自动替换失败游戏。

- 核对研究身份、四个driver哈希、实际systemd调用、训练累计量和新完成cell的SHA/逐局胜负及健康字段。
- 心跳超过180秒、训练/原始评估10分钟无进展、服务异常、STOP或健康门槛失败时标记需要处理。
  这些是监控警报，不能自行把停滞诊断成引擎死循环，也不会因此改写研究或杀进程。
- 每10分钟，以及完成/异常状态转换时，更新#61/#83/#92中由标记限定的监控段落，保留其余正文。
  GitHub写入失败单独落日志并重试；网络失败不改变实验状态。
- 完成后先核对39个评估cell/9,984局、配对输入、阶段与模型SHA及实际首批rollout，再用抽样次数权重
  独立重算20,000次交叉及条件bootstrap、三个seed差和固定update32门槛。通过后才同步完整结论。

首次真实快照：**173/192更新、11,035个已计数训练终局、0截断/错误，6,400/9,984完整评估局**；
25个cell原始文件核验通过，训练和评估服务正常，三个issues实际发布已确认。
这仍是阶段记录，最终强度结论须等待剩余数据与独立审计。

监控driver lint/format通过；故障检查在临时元数据副本上验证心跳过期、STOP、服务失败和身份篡改检测。
审计器用上一轮封存的9,984局与六个PPO阶段复核通过：仅在临时fixture补齐旧schema的npz哈希、
从阶段列表去掉三个pretrain记录，原始研究文件未改，适配来源和SHA单独记录。
监控身份SHA `57c07498692d9e0210cf8eb83ce3382cfaad8201def2ac160c6ef53e8c6c6c7b`，
`latest.json`为最近检查，`events.jsonl`保留进度/警报，`publication.json`记录实际issues更新时间；
完成须有`completion-audit.json`和`finished.json`，不能仅凭旧的running快照判断。
