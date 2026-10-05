# 固定actor的critic预热校准（2026-10-05，事前协议）

问题：冷启动critic通过advantage影响首步策略方向，但预测量的高相关性不是实际动作信号比例，
自举目标Q EV也不能证明critic已学会真实终局。先固定actor，检验现有lambda-target预热
是否在新对局上改善真实终局预测，再决定是否值得做warmup的策略强度对照。
这不是复跑已经失败的terminal-MC joint-training：此次不改critic目标，actor始终冻结。

只用修正BC的128×2、12epoch末尾actor。选择它是基于已归档的首步机制诊断（KL .669）
和历史冷启动问题，不根据尚未齐全的四臂LR中途胜率选容量。四臂LR既定评估仍先完成；
本实验使用PR203之后的新源码，不能与旧EMA训练拼接成一条曲线。

固定训练seed **2026100524**，32次critic-only更新，每次128env×128steps=16384行，
FP32/TF32 off，默认VRPO lambda=.5、Q/V各.5、lr1e-3、2epochs×2048minibatch。
配置critic_warmup=32、critic_warmup_ev=2（禁用EV提前结束），actor和共享主干冻结；
每次更新逐位验证actor不变、实际16个Adam仅更新critic参数，保留0/8/32完整checkpoint。
20套合法MD-2026-09牌组；selfplay=1、snapshot_every=0，不引入变化的对手池。
这是固定BC行为下的critic干预，不等同于默认75%selfplay的PPO配方。
训练健康错误/非有限值即停；>=100终局后截断率>1%即停。保留异常完整轨迹，不换seed补跑。

训练完成后，用全新seed **2026100525**的固定BC自对弈采集**128个完整游戏**，8env×16批。
SelfPlaySchedule每个seed生成先后手两局，所以共有64个独立发牌seed cluster，不能把128局当独立样本。
按每slot恰好一局采集，不优先补短局。任意健康错误、上限、空游戏都保留数据并停止，
不把上限当平局或用替代游戏悄悄补齐。三个critic读取完全相同的实际访问状态、特权输入、
已采取动作和真实终局z；不再采样三份不同的游戏，也不使用自举目标作为测试标签。
保存原始rollout、所有游戏的完整spec/健康结果、逐行预测、模型/source/core/卡库/脚本指纹。
验证游戏与packed-row边界的对应关系，以及0/8/32的actor权重完全相同。

主要诊断：逐游戏等权的Q_taken终局MSE，固定比较update32减update0；
先对每个seed的两局取平均，再对64个seed cluster做20,000次配对bootstrap，seed **2026100526**，报95%区间。
同时完整报告0/8/32的逐行与逐游戏等权MSE、Q/V终局EV、预测范围/均值，以及零预测基线MSE=1；
所有节点均展示，不挑8或32中更好的作为终点。V与update8为描述性结果。
仅当32对0的MSE差上界<0且32对零预测的MSE差上界<0，才认为有证据值得继续验证该预热的终局预测收益。
未通过时不按同一面板追加训练求显著；这不是强度晋级门槛。

限制：单训练seed、固定BC自对弈分布、同一20牌组池；deal-seed-cluster区间不涵盖训练seed方差。
终局预测改善也不能证明同局面的动作排序、advantage符号或完整对局胜率改善。
即使通过，也要做冻结预算、多seed的warmup/无warmup策略对照及新的外部对手评估；
不会据此推广checkpoint或宣布#92两万更新目标完成。

证据根：`out/research/critic-warmup-calibration-2026-10-05/`。

运行前修订：四臂LR评估已因4个融合素材取消循环按协议停止；全量复现后先修进度mask。
本实验将在该修复的完整测试通过后开始，train/heldout使用同一新mask与PR203修复后EMA。
修复把无游戏进展选择链的material cancel预算设为32，不改变原始合法动作或终局规则。
固定actor指网络权重固定；这不是旧mask下行为分布的原样复用。训练与heldout预算、seed、
比较节点、主指标和停止阈值均不变；还未生成本实验训练/heldout数据。
