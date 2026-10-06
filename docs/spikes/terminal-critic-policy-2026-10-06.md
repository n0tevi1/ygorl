# 早期终局critic权重的PPO强度对照（2026-10-06，运行前协议）

终局监督epoch32过拟合；仅由validation选择的公共epoch1已在256独立新游戏通过
Q终局MSE对零/冻结train常数的事前门槛。现检验该预测能力是否改善实际PPO策略，
不将校准改善或状态预测正确率直接解释为棋力。

## 权重干预与预算

使用`terminal-critic-fit-2026-10-06`三个critic初始化的epoch0（cold）与epoch1（warm），
actor均为修正BC128×2 epoch12，产品源码/native guard保持PR204后冻结版本。
每臂建立全新Trainer，严格检查词表、环境、net/critic配置和源SHA，仅加载model权重；
Adam为空、计数为0、reference actor相同、对手池为空。不得转移监督拟合的optimizer、RNG或任何游戏状态。
同对cold/warm使用相同新PPO seed **2026100546/47/48**；初始CPU/GPU/collector RNG和schedule核对一致，
但异步采样不保证逐步同轨迹，这一限制明确保留。保存各臂首个实际rollout。

六臂各固定**32 PPO更新/524,288行**，总**192更新/3,145,728行**；单GPU按seed逐对cold→warm。
128env×128steps、FP32/TF32 off、overlap off；VRPO lambda=.5，LR1e−4，2epochs×2048 minibatch，
KL guard .01、entropy/reference系数各.05、selfplay .75、PFSP/pool8、snapshot10，其余配置不变。
保存不可变0/16/32节点；32为唯一主终点，16仅描述，不挑seed/节点。

warm额外使用此前512完整训练游戏的162,019行、一轮监督拟合（80 Adam steps/critic）；
三个critic共享训练数据，只独立初始化/打乱。已发生的32epoch探索、验证与独立确认成本单独披露，
不宣称相同总计算/样本预算或更高样本效率。本研究不再训练或选择critic预热模型。

## 新评估与判定

新eval seed **2026100549**，64个ordered deck-pair/deal cluster，每cluster交换牌组/先后手4局，
三个固定对手Greedy/原BC256/初始128等权；每节点每对手256局。
initial actor逐位相同后共用768局，六臂×两个训练节点×768局，总**9,984新游戏**。
沿用上一健康对照的Arena、PPO候选/原BC基线采样接口和取消监控；绑定原始GameSpec、agent seeds、
模型SHA和健康结果，异常保留动作/response证据。候选取消计数是访问分布上的描述量。

主指标update32 warm−cold等权胜率；另测warm−initial，避免将较少退化称学习。
bootstrap seed **2026100550**、20,000次，训练seed（3）和deal-cluster（64）交叉重采样，
两臂/全部对手共享抽样，保留cluster四局；同时报告仅deal条件区间。
仅当两主比较的交叉区间下界均>0，且三个warm−cold seed差均>0，才值得更长预算确认。
不通过就保留结果，不在同面板追加预算求显著。3个seed、共享critic训练语料、有限固定对手仍限制外推；
不是#92两万更新或顶尖对局验收，也未直接验证未选动作的Q排序。

## 健康与编排

训练健康错误/非有限值即停；累计至少100终局后截断率>1%停止。
评估任意错误/上限立即全研究STOP，保留原局，不计平或替换。已完成节点/cell可SHA核验复用，
不自动覆盖半份训练/评估。先用真实GPU核对三个权重转移配对和空Adam/RNG合同，再冻结driver与身份。
独立systemd用户服务最多8小时，GPU producer和4-worker单线程CPU consumer并行；
whole-cgroup管理，阶段/heartbeat日志、可恢复分析收尾，completed只在原始证据核验与分析完成后写入。
研究根`out/research/terminal-critic-policy-2026-10-06/`。

## 运行前核验与启动

协议提交`3c094ab`早于数据；GPU预检核对全部3个实际epoch0/1配对：actor/reference actor一致，
CPU/GPU/collector RNG和schedule一致、Adam/pool为空，critic私有权重确实不同，零训练行。
同一Arena接口的历史非研究面板对局正常完成。首次研究入口把critic配置误当作config字典字段，
预检捕获后改用TrainConfig的派生配置解析并通过；原错误/旧driver留存，正式数据尚未产生时完成修正。

研究身份SHA `304292e15a85f9b71ffc201e9d26eca6eea903d69214fea9e79ea050f96dbcdc`，
四个driver及全部源模型已冻结。`ygorl-terminal-policy-20261006.service`托管，
InvocationID `53719e6ee8c74880a28d6533347855ea`，8小时上限；工作目录为`codex-lr-control-report`。
首个cold PPO更新正常完成（6个终局、0截断），CPU评估消费者同步运行。
当前尚无本轮胜率结论；`pipeline-status.json`和逐阶段日志记录进度。
