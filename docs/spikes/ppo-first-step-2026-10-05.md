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

## 完成结果

两容量原paired.py都在“共同Adam步骤逐位一致”处失败，原脚本/日志/产物未覆盖。
追加重复试验复用原始frozen.pt（完整rollout、初始learner与CPU/CUDA RNG），无重采样。
旧旧、新新及旧新比较的首步权重最大绝对差分别为：64×1全部2.98e−8，128×2为5.96e−8/5.96e−8/2.98e−8。
差异L2为0.9e−7至1.9e−7量级（state_dict包含共享权重别名，计数不是独立参数数）。
同实现重跑也有数值波动；因此不能要求默认GPU路径hash相等。尚未定位具体kernel，不把它归罪于某算子。
开启确定性算法的附加路径两模型均可运行，但只跑了一次，不宣称已验证其逐位重复性。
这些参数微差未解释0.1至1量级的KL；默认重复两次的最终KL差最多约3e−8。

每份固定rollout 16,384行均为多动作状态；全部已结束游戏健康。共同第一minibatch索引hash相同。
下表为完整rollout上的真实双向KL，区别于第二minibatch的采样approx_kl。

|模型|第一步后下一batch approx_kl|新guard步数|新guard KL(old‖new) / KL(new‖old)|旧guard步数|旧guard KL(old‖new) / KL(new‖old)|
|---|---:|---:|---:|---:|---:|
|64×1|.122452|1|.132638 / .138579|2|.055225 / .061206|
|128×2|.590497|1|.668931 / .936508|2|.471283 / .617251|

**少走一步不保证最终KL更小**：这里第二步反而拉回部分位移。修复保证不执行已知越界batch，
不是硬trust region，也不能据此承诺强度更高。旧guard保留下来的训练结果仍是有用的历史对照。

首个固定minibatch的梯度分解保留所有默认加权系数。actor参数上的policy/entropy/critic梯度范数：
64×1为.37084/.04342/.21354；128×2为.45452/.03994/.34871。
完整critic-only参数梯度范数.41739/.64217。reference初始与actor相同，loss为0，梯度只有约1.6e−8数值残差，
因此EMA reference惩罚无法在初始点预先约束第一步位移。不能把梯度范数直接当Adam后的影响权重。

|单步loss / LR|64×1 KL(old‖new)|128×2 KL(old‖new)|
|---|---:|---:|
|完整 / 1e−3|.132638|.668931|
|完整 / 3e−4|.015114|.080807|
|完整 / 1e−4|.001763|.009852|
|policy-only / 1e−3|.166128|.899741|
|entropy-only / 1e−3|.065702|.195234|
|critic-only / 1e−3|.030167|.057027|

这里critic-only允许Q/V更新共享actor trunk，**不是**冻结actor的policy=False warmup。
完整loss 1e−3复算与原第一步最大参数差2.98e−8/5.96e−8，KL相同；分解驱动在恢复Adam后显式设置LR，
避免load_state_dict把扫描学习率覆盖回1e−3。完整loss的mean total variation在1e−3为.13378/.24292，
1e−4为.01538/.02738；argmax改变比例21.38%/34.82%降至2.19%/3.68%。

判断：默认1e−3对这些BC权重的首步过大，KL早停只能在一步之后发现。policy-only仍更严重过冲，
排除了“Q/V loss直接更新共享trunk是唯一来源”，但它仍用原随机critic算出的VRPO advantage，
**不能排除critic初始化通过advantage间接造成过冲**。本实验既未实施也未排除warmup的收益。
entropy和共享critic也会改变策略，但这些独立Adam干预不构成可相加的因果贡献分解。
更小LR是值得进入真实训练的干预；单步KL好看仍不证明学习更快、胜率更高或长期不会失稳。
只在本次单seed/两份rollout上成立，不更改生产默认。

产物封存：`out/research/ppo-kl-guard-2026-10-05/diagnostic-validation.json`，包括失败记录、完整frozen.pt、
所有第一步权重、梯度、身份、脚本与报告。下一轮预先保留新guard下的1e−3对照与1e−4干预，各跑两容量。

## Adam首步机制审计

原始Adam state为空，默认无weight decay。首步bias correction后有
`delta = -lr * clipped_gradient / (abs(clipped_gradient) + eps)`；从已保存梯度及初始权重重算，
与真实GPU首步最大权重差两容量均5.96e−8。global norm clipping系数为.81817/.57578，
但当梯度大于eps时，Adam的归一化会抵消大部分统一缩放，所以max_grad_norm不是策略KL约束。
独立actor参数首步delta L2为.38954/.75812，最大单坐标约.001。
这是对保存产物的解析核验，不新增训练干预；`adam-audit.json`及单独manifest保留证据。

## advantage来源补充审计协议（运行前）

在原固定rollout上只作诊断，先保留默认VRPO λ=.5的未归一化advantage，然后把Q及bootstrap Q全部置零
（reward、done、seat和概率不变）重算。估计器对Q/reward线性，full−zeroQ可描述该份数据的critic项，
报告范数、方差、相关性、实际terminal reward非零行数、纯reward trace覆盖和可见终局的完整前缀比例。
不能把zeroQ当正确bootstrap，也不替换训练目标；它仅分开稀疏奖励项与当前估值项。
再在同一个首批样本比较两组各自按原标准化方式归一的policy梯度方向/范数，完整记录，
不据此宣称warmup已经有效。后续是否需要完整终局监督或warmup实验，应由该诊断和四臂训练一起判断。

### advantage来源审计结果

同一份固定rollout的分解，重构`full = reward_only + critic_only`最大误差1.19e−7。
两容量均只有10个非零终局奖励；能在本段看见后续终局的行数为1,007/1,095（共16,384行）。
λ=.5令reward trace的绝对值>1e−6仅200行、>1e−3仅100行；其余行仍从critic估值获得梯度权重。

|模型|full raw advantage std|reward-only std|critic-only std|full与critic项相关性|full与reward项相关性|policy梯度full vs reward-only cosine|
|---|---:|---:|---:|---:|---:|---:|
|64×1|.47942|.02852|.47832|.99823|.06841|.26624|
|128×2|.40473|.02851|.40451|.99752|.04302|.18700|

梯度使用相同首批样本与网络，但各组按原standard规则归一；完整梯度与原存档policy梯度cosine约1。
结果说明这两份fresh-critic首轮更新的数值方向主要依赖估值项，而非已观测终局奖励；
**不是99.8%的梯度是噪声，也不是实际动作信号比例或Q正确率**。zeroQ丢掉合法bootstrap，不能直接用来训练。
仍需真实分支续局/校准或受控warmup干预，才能判断估值方向是否错误、预热能否改善强度。

不能把它当成重新延长旧MC配方的理由：[先前完整终局监督对照](terminal-mc-2026-10-04.md)在两个种子上
比lambda低4.56pp，完整游戏采样的后续三seed对照也未通过门槛（[报告](collection-control-2026-10-04.md)）。
那些实验没有冻结actor预热、使用旧actor与guard，因而既不证明新初始化warmup有效，也不完全排除它。
本轮先完成已经启动的四臂LR对照，不改变其配置或混入预热。

补充数据/梯度/解析Adam审计与失败记录在同目录`advantage-validation.json`单独封存，原诊断manifest不覆盖。
