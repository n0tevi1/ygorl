# Reference EMA重复更新共享参数（2026-10-05）

初步真实ActorCritic 64×1、history=none检查：identity.status.weight与
identity.id_embedding.weight分别在actor.identity、board.cards.identity、
action_encoder.identity出现三个state_dict名字，但各自只有一个Parameter。
将learner每个唯一参数增加1，reference_ema=0.02时共享参数实际移动0.058808，
等于`1-(1-0.02)^3`，而不是约定的0.02。其他唯一参数应只移动0.02。
因此慢reference的部分权重追赶约3倍快，是实现错误；它削弱对应参数的平滑，
但尚未证明对策略KL、胜率或长期学习的净影响。也不能解释第一次更新前
reference与learner相等时的首步KL超调；该EMA发生在optimizer更新之后。

先提交设计，再添加真实共享网络与buffer回归，保留修前失败证据，再修复并全量验证。
证据根：`out/research/ema-shared-parameters-2026-10-05/`。
当前四臂LR实验仍以原有冻结EMA完成，结果不会与修复后实验混写。
