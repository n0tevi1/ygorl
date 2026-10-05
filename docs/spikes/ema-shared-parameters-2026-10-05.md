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

设计c5ce1c7先于实现37023b0；EMA循环从首次PPO实现4c691e7（2026-09-22）就按state_dict别名执行。
原Nim测试无共享参数，不能覆盖PolicyNet的实际结构。新增5例修前全部失败（3.08s）；
修后含原PPO测试共26 passed（11.92s），覆盖history none/transformer、共享/独立critic主干、
连续两次EMA、checkpoint往返、共享浮点/整数持久buffer及非持久buffer不更新。

结构审计audit.py读取当前四臂的实际net配置（仅把vocab_size缩到20）：
64×1与128×2都启用了Transformer history，identity的两个Parameter各有**4个别名**，
原有效τ为**1−.98⁴=.07763184**；history=none的3别名例为.058808。
修正后所有唯一参数与单次EMA解析公式最大误差2.384e−7（FP32舍入）。
该审计没有重新采样对局、训练策略或测胜率；不把公式修正解释成已验证的强度提升。

实现用state_dict(keep_vars=True)保留对象身份，再按id去重，维持持久state_dict契约；
不改变键名、别名或checkpoint载入。保留旧reference值，后续更新才按正确τ执行。
整合PR202后源码ef11ae7进行实际ROCm全量验证。

最终整合全量：**1497 passed，3 skipped，409.86s**（真实ROCm，源码ef11ae7）。
3个skip为snapshot选项与2个显式网络测试；validation.json封存修前失败、定向、结构审计与全量日志。
