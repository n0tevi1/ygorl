# BF16更新作用域回归（2026-10-05）

检查Trainer.step发现`with self._autocast()`只包住warm条件计算，`learner.update`在块外。
git历史定位4fa15b9752d907fcb0be1e80d98d185ff120292f（2026-09-26加入critic warmup）引入这次缩进变化。
因此BF16标志仍影响collector，但PPO更新静默回到FP32；不只影响warmup打开的运行。
此前测试只检查更新完成且loss有限，并不检查实际计算精度，故未捕获回归。

设计7933070先于修复。真实GPU回归分别检查FP32控制、BF16普通更新和BF16 warmup，
捕获训练forward中的autocast状态与实际logits dtype。修前 **2 failed /1 passed，4.30s**，
两个BF16案例均观察到`(False, torch.float32)`。一行缩进修复使update重新进入已有作用域。
初次训练循环与GPU/Split-K定向 **45 passed，32.49s**，包含现有overlap+BF16训练，随后发现下面的缓存问题，未合并该中间版本。

另外用此前冻结的真实128×128 rollout，在64×1和128×2上经Trainer.step验证大token路径，
记录实际Split-K调用数/行数、BF16 forward、有限参数与梯度。此处替换_collect复用固定数据，
收集耗时是原数据的历史值，不能拿这次日志当吞吐benchmark；不推进其更新权重。
初次探针产物保留于`out/research/ppo-autocast-2026-10-05/`；它只有dtype/有限性检查，不能视为正确更新的验证。

## 第二层缺陷：同作用域跨Adam使用旧权重副本

大批探针在仅修缩进的版本上，64×1/128×2分别执行16/6步，策略移动异常偏小，触发进一步检查。
新增真实GPU回归在每次Adam后，将当前作用域的forward与关闭缓存的fresh forward比较：
**1 failed，4.22s**，首步logits最大绝对差.0107422。autocast参数cast缓存在跨minibatch的with块内未失效，
更新后的FP32参数与后续forward读取的BF16副本不一致；这不是普通BF16舍入误差。

补充设计bdf3058先于修复：PPO更新的autocast关闭cast缓存，collector权重不变时仍保留缓存。
新增回归检查至少两次真实Adam后的当前权重forward逐位一致。最终定向 **46 passed，30.85s**。
重跑同一固定rollout，两容量都实际执行1步并在第二批拒绝；所有forward均BF16且梯度/参数有限。
64×1有46次大token Split-K forward、最大196,608行；128×2有72次、最大200,704行。
数据在`out/research/ppo-autocast-final-2026-10-05/`，原探针未覆盖。

发现缓存缺陷后主动停止中间版全量测试（exit143，日志保留），不把不完整运行称为green。
最终完整实际ROCm **1487 passed /3 skipped，488.38s**（1个snapshot选项、2个网络opt-in）；前后证据及产物hash统一归档于原目录validation.json。

对历史解释：09-29文档的bench_train调用Trainer.step，该回归期间CLI BF16对照不能验证更新阶段BF16性能。
不凭文档日期声称精确还原了某个没有源码指纹的原始运行；历史计时保留并明确此限制。
修复不证明BF16更快或强度相同，需要独立成本和训练验证。当前四臂LR实验明确FP32，源码保持冻结，完全不改配方。
