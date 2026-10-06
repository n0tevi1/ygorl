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

## 完成结果

同一39,613行、同一actor行为概率，奖励项std固定为 **.065531**。

| Critic更新 | full std | critic项std | corr(full,reward) | corr(full,critic) |
|---|---:|---:|---:|---:|
| 0 | .321149 | .311348 | .249306 | .979006 |
| 8 | .071829 | .041841 | .818250 | .435182 |
| 32 | .065347 | .032726 | .874951 | .244779 |

预热把critic项的原始std缩小约9.5倍，完整advantage与奖励项更一致。
这提供了一个比“已学会预测胜负”更窄的可能机制：消除冷启动估值的大幅度项，改变它与奖励项的相对幅度。
不能从.875相关性声称“87.5%真实信号”，也不能从std比值推导可加的贡献比例，两项有协方差。

**长程credit assignment仍未解决。** reward trace绝对值>1e−6仅2,560行（6.46%），>1e−3仅1,280行。
31,421行（79.32%）距终局>=64个learner rows，奖励项std约4e−21；这些行full与critic项在FP32下相同。
该组std虽从.305402降到.025420，但多数早期决策依然依靠critic项；全局相关性升高不代表它们被真实奖励纠正。
这里只指出该面板上lambda=.5的直接reward trace衰减，不据此认定某个新lambda/target必然提高棋力。
旧terminal-MC与collection-control负结果继续保留；当前三种子策略实验不变。

所有已选cancel合计只有14行，且不限于已知融合素材循环。其raw mean从.087266降到.004669，
样本太少，未测实际policy gradient，不能宣称已解释或解决“训练强化取消”的具体因果链。

验证：所有源raw/checkpoint SHA核对，三个actor逐位一致；CPU重算对原collector概率最大误差6.20e−6，
对保存的Q_taken最大误差1.30e−6；full−reward−critic最大误差1.19e−7。
保留全部合法动作Q、逐行分解/距离/动作类型/终局标签、报告与PNG/SVG图，`validation.json`封存。
