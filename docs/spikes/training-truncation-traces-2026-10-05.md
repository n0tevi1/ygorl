# 训练上限诊断缺少跨更新动作（2026-10-05）

四臂LR控制的128×2高LR在第32次更新出现1个decision_limit：牌组elfnote-kewl-tune对dracotail，
seed2067908758125888073、turn9、snapshot opponent2，所有健康字段为零。
1/1,691低于事前1%停止阈值，因此保留全部数据继续既定四臂；不把它计正常胜负。
现有games.jsonl.gz只有seed/结果，errors.jsonl只保留engine error，未保存此截断的动作或responses。
学习方跨多次更新改变权重，checkpoint只有0/16/32，不能用最终权重精确重演原局。
该历史截断的具体循环原因仍未定位；此次补齐诊断能力不等于已修复这个未知循环。

设计4183287先于实现。EncodedVecEnv可选记录显式step索引，Trainer启用；每slot的序列跨rollout保留，
健康普通终局丢弃，错误/上限导出，不复制observations/模型，不改C++模块。
相同skip_forced模式会重放相同原生强制选择；显式host动作还能覆盖未提交engine response的中间选择。
Trainer新truncations.jsonl记录非error截断，原errors.jsonl补足相同spec/动作/健康字段；不健康win也保留诊断。
旧数据动作字段为null，区分“不存在记录”和真实空序列。

修前新契约测试 **3 failed，5.08s**：真实跨更新上限无诊断文件，env尚无可选记录接口。
修后训练/登记/记录定向 **44 passed，43.36s**（真实ROCm可用），包括：

- 使用新native env，按保存的完整spec与indices重放跨更新上限，终局/decision数/逐response完全一致。
- 同seed记录开关不改变正常局和上限局结果，正常局不导出trace；单slot复用与双slot并行不串局。
- 原生非法动作报error时保留尝试的索引；不健康win仍保留原reason及诊断。
- 原engine error及真实Lua错误回归保持通过，登记/checkpoint契约不变。

全量测试在整合当前main后执行；证据根目录`out/research/training-truncation-traces-2026-10-05/`。
当前四臂producer源码和native模块保持冻结，不在运行途中加日志或改变采样。未来长训练使用该诊断。
