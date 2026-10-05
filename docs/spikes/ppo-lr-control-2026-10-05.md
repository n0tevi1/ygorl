# 修复guard后的学习率对照（2026-10-05，训练前协议）

[固定rollout诊断](ppo-first-step-2026-10-05.md)显示首步过冲不限于critic梯度：policy-only也过大；
两容量完整loss的lr1e−4单步KL为.0018/.0099，1e−3为.133/.669。尚不能证明小LR实战收益。

固定四臂：64×1/128×2 × lr={1e−3,1e−4}，全部使用PR197的guard，原修正BC各12epoch末尾权重。
训练seed2026100518、32更新、默认128env×128steps/2epochs×2048minibatch、FP32/TF32 off，
其它算法配置、20合法md-2026-09牌组、PFSP、熵.05、reference .05、warmup0均与上一pilot相同。
顺序64高LR、128高LR、64低LR、128低LR；每臂登记0/16/32，不按中途表现挑终点。
每段优化前检查健康、所有指标有限、累计100终局后截断率>1%则停并保留；不推广诊断权重。

每臂独立矩阵保留相同三个固定基线Random/Greedy/原BC256×2。seed2026100519、10 ordered pairs、
每cell40局，与前pilot相同协议用于诊断；它不是新保留测试集。每臂完成后有6 agents/15 cells/600局。
三个基线彼此的120局只生成一次，记录来源与SHA后复用到四矩阵；其余480局/臂均新评估，总新打2040局。
对0/16/32全部节点保留完整身份/健康/原始结果；无异常与上限才发布，任何故障先调查。
独立CPU消费者2workers×1Torch线程、nice19，不阻塞训练；各矩阵排名参照不同，不能直接跨臂比较名次。

主要诊断比较同容量、同训练seed、同evalpanel的update32低LR−高LR固定baseline胜率，另报16和各自起点。
按10个seed cluster×4游戏做20,000次配对bootstrap，seed2026100522，完整报告未校正探索区间。
不同在线策略的数据分布会分开，不宣称逐rollout配对训练。另报告实际Adam数、触发/拒绝KL、entropy、
Q EV、吞吐/活动wall（共享机器负载，非隔离benchmark）和健康。四臂全跑完再作下一阶段决定。
小样本单训练seed仅判断下一轮多seed长训练配置；#92的20,000更新目标仍未达到。

运行目录：`out/research/ppo-lr-control-2026-10-05/`。模型/source/driver/牌组/引擎/数据库/脚本hash全部绑定。
