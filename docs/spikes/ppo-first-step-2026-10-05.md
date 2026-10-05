# PPO首步过冲与数值重复性诊断（2026-10-05）

前置：[KL guard修复](ppo-kl-guard-2026-10-05.md)，PR197已合并18f8bf4；真实ROCm全量1480 passed/3 skipped。
同rollout配对诊断首个64×1案例的共同Adam步权重hash不一致，断言失败报告保留于
`out/research/ppo-kl-guard-2026-10-05/paired-64x1/`。两路minibatch索引相同，初始KL均约0，
第二批KL均.122452；不能因此声称所有权重逐位一致，也不能先认定数值误差来自某个kernel。
旧实现最终KL(old policy‖new policy)=.055225，新实现只走一步为.132638：少走一步不保证该KL单调下降。
该pilot只定位更新行为，不据此推广策略强度。

## 补充实验协议（运行前）

1. 保留原paired.py、日志与失败产物，不重采或覆盖rollout。64×1/128×2分别用已冻结的完整rollout、
   初始model/reference/Adam、CPU/CUDA RNG，依次运行old-A、old-B、fixed-A、fixed-B四路。
   每路保留第一Adam步完整CPU权重；比较共同索引、逐位相等、最大绝对参数差和L2参数差。
   先测同实现重复的波动，再判断跨实现差异；不预设“足够接近”的通过阈值。
2. 加一条开启PyTorch确定性算法的单步路径；若不支持则保存完整异常，不能静默降为warning。
3. 之后在同一固定第一minibatch上分解policy surrogate、entropy、reference-KL、Q、V梯度，报告
   actor/trunk与critic-only的范数及对齐关系。用相同初始Adam对默认完整loss固定扫
   lr={1e-3,3e-4,1e-4}，另在lr=1e-3分别做policy-only、entropy-only、critic-only单步。
   每路完整rollout测双向KL、熵、probability drift；不选择样本、不推进这些诊断权重。
4. 所有比较都是固定数据上的单步敏感性，不能替代多seed训练/实战评估。只有诊断结束后才另立训练协议。

另一个待审计项：Trainer checkpoint目前记录CPU torch RNG与collector generator，尚未看到全局CUDA RNG。
进行中的对局本就不存盘，不能要求续训完整轨迹逐位一致；但GPU minibatch RNG是否应恢复需要单独复现。
