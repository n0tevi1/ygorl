# PPO KL guard 的边界缺陷（2026-10-05）

32-update固定诊断中，128×2前两次更新的平均approx_kl为.10196/.17319，均已触发早停，
却各执行2次Adam。检查实现发现：kl_now来自optimizer之前的forward，条件却在optimizer.step后判断，
因此会对**已经测出超限**的当前batch再做一次更新。先前“提前停止、少于全epochs”的测试无法检查这个边界。
它会额外推动策略；不能只凭这个缺陷断言此前所有能力退化都由它造成。

独立Nim真实rollout复现：threshold=.00015，两次forward的KL为0/.364731，旧实现两次都执行Adam。
新增拦截optimizer.step的回归和首批即超限的回归，修前2 failed。设计a909058先于实现d13f2ec；
现在当前batch已知超限时，在backward/optimizer前拒绝它和后续batch。critic-only warmup仍可训练critic，
actor参数不动。关闭与不触发guard两路权重逐元素一致。

新日志区分minibatches（实际Adam步）与evaluated_minibatches（forward检查数），记录stop_approx_kl。
forward指标均值含拒绝batch，grad_norm只对实际更新平均；0步也输出完整有限诊断。
update仍计处理的rollout次数，EMA约定保留。**这不是KL硬上界**，第一步或最后一步本身仍可超限。
文档已纠正“策略位移封顶”的过强说法。

修后相同独立Nim复现只执行KL=0对应的一步，拒绝.364731对应的第二步。
PPO与真实对局训练定向 **54 passed，42.13s**，包含现有学习收敛和新增4项回归。
完整实际ROCm测试进行中，结果之后归档于 `out/research/ppo-kl-guard-2026-10-05/validation.json`。
旧两组32-update训练已按原协议跑完并保留为对照，运行中的矩阵消费者保持原冻结源码。

## 下一项受控诊断（运行前协议）

测试通过后，对64×1/128×2各从相同修正BC初始化，重新采一份默认128×128的真实自博弈rollout。
每份rollout固定后，旧/新PPOLearner使用同一初始模型、critic/reference/Adam状态以及相同CPU/CUDA RNG，
独立更新一次；不把两次异步收集叫配对数据。记录实际每次Adam前KL与每步完整权重hash，
验证被两路接受的共同步骤逐位一致，保留拒绝位置。分别在该固定rollout上测实际更新后双向策略KL。
seed2026100521；只比较guard顺序，不改LR/熵/critic_warmup，也不据结果筛牌组或样本。
保留CPU化的完整rollout与起始state，旧PPO源文件hash，实际源/引擎/数据库/脚本身份。
这项小实验检验“少走了哪一步、少走后策略变化多少”，不是对局强度结论；不推进这些权重。
