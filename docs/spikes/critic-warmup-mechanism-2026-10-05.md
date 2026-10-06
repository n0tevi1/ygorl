# 预热critic如何改变固定数据的advantage（2026-10-05，事后机制诊断）

在三种子策略对照已经冻结并运行时，另作只读诊断：使用上一轮校准已封存的128完整BC自对弈游戏、
0/8/32 critic和同一actor概率，重新计算所有合法动作Q并分解VRPO lambda=.5的原始advantage。
线性拆分为真实reward trace（Q置零）与critic项（reward置零），核对两者之和等于完整advantage。
报告原始std/均值、full与两项的相关性、距终局learner-row距离分组，以及每种已选动作的样本数/均值。

目的：区分“critic有很强终局预测”与“减少初始化误差，改变奖励项相对幅度”这两个解释。
仅计算冻结数据，不更新actor/critic、不采新对局、不调整正在运行的策略实验、预算或胜率判定。
这是事后探索，不是新增保留集：完整游戏的终局密度与PPO固定长度rollout不同，不直接量化训练中的占比。
reward trace也不是完整的真实advantage，critic项也不能一概称噪声；零Q是代数诊断，非新的训练目标。
不把相关系数或方差占比当作动作正确率或棋力因果证据。距终局按learner rows计，非游戏回合数。

证据根：`out/research/critic-warmup-mechanism-2026-10-05/`。使用CPU单线程低优先级，不占用训练GPU。
