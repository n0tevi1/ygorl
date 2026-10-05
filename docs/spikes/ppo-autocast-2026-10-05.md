# BF16更新作用域回归（2026-10-05）

检查Trainer.step发现`with self._autocast()`只包住warm条件计算，`learner.update`在块外。
git历史定位4fa15b9752d907fcb0be1e80d98d185ff120292f（2026-09-26加入critic warmup）引入这次缩进变化。
因此BF16标志仍影响collector，但PPO更新静默回到FP32；不只影响warmup打开的运行。
此前测试只检查更新完成且loss有限，并不检查实际计算精度，故未捕获回归。

设计7933070先于修复。真实GPU回归分别检查FP32控制、BF16普通更新和BF16 warmup，
捕获训练forward中的autocast状态与实际logits dtype。修前 **2 failed /1 passed，4.30s**，
两个BF16案例均观察到`(False, torch.float32)`。一行缩进修复使update重新进入已有作用域。
训练循环与GPU/Split-K定向 **45 passed，32.49s**，包含现有overlap+BF16训练。

另外用此前冻结的真实128×128 rollout，在64×1和128×2上经Trainer.step验证大token路径，
记录实际Split-K调用数/行数、BF16 forward、有限参数与梯度。此处替换_collect复用固定数据，
收集耗时是原数据的历史值，不能拿这次日志当吞吐benchmark；不推进其更新权重。
完整实际ROCm测试通过后合并，结果及产物hash归档于`out/research/ppo-autocast-2026-10-05/validation.json`。

对历史解释：09-29文档的bench_train调用Trainer.step，该回归期间CLI BF16对照不能验证更新阶段BF16性能。
不凭文档日期声称精确还原了某个没有源码指纹的原始运行；历史计时保留并明确此限制。
修复不证明BF16更快或强度相同，需要独立成本和训练验证。当前四臂LR实验明确FP32，源码保持冻结，完全不改配方。
