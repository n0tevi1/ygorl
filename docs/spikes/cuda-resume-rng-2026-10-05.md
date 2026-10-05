# GPU续训丢失minibatch RNG（2026-10-05）

审计Trainer.state_dict/_restore发现只保存CPU torch RNG和collector的独立CPU generator。
PPO使用`torch.randperm(n, device=learner_device)`打乱minibatch；CUDA/ROCm全局RNG未保存，
恢复时Trainer初始化的manual_seed还会把这个流重置。因此相同checkpoint不能恢复下一次GPU排列。
这与“进行中的对局不保存、续训重开”是不同的缺口，不能用后者解释前者。

真实GPU回归：小型YGO Trainer完成一步更新、加入快照池、保存checkpoint，然后记录下一次GPU randperm、
CPU随机数和collector抽样；扰乱种子后resume。修前在GPU排列比较处 **1 failed，5.04s**。
首次测试错误地假定collector generator也在GPU，得到设备类型错误；修正测试后才得到上述真正复现。
两个失败日志都保留，不把测试自身错误算作产品根因。

设计fcd2581先于实现。新checkpoint在`rng.cuda`记录learner所在设备的RNG（CPU ByteTensor），
在模型/优化器/reference/pool/schedule重建之后恢复。不读取所有GPU，不给CPU checkpoint新增CUDA RNG访问。
旧GPU checkpoint没有字段时仍正常加载，但无法恢复从未保存的历史GPU随机流；不承诺跨设备同序列。

新增三项回归覆盖真实GPU三个流的连续性、CPU不调用CUDA RNG API、旧GPU检查点继续训练。
训练循环/checkpoint/登记定向 **49 passed，30.09s**。全量实际ROCm **1483 passed /3 skipped，430.21s**；1个snapshot选项和2个网络opt-in跳过。
这恢复随机流，不保证整段训练逐位重现：原生异步调度、重开的半局和GPU数值波动仍存在。
运行中的四臂LR实验保留冻结源01d473e，全从BC新建，因此不改动这些producer来应用续训补丁。

验证产物：`out/research/cuda-resume-rng-2026-10-05/validation.json`。
