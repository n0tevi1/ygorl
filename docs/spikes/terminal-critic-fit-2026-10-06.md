# 固定actor的终局监督critic诊断（2026-10-06，运行前协议）

三seed lambda预热的策略主终点未通过加长预算条件。下一步单独检验固定BC行为分布下，
现有critic是否能拟合真实终局，以及这种拟合是否能泛化。**不训练actor，不更改奖励，不测棋力增益**。
这不同于此前失败的terminal-MC联合策略训练，也不同于继续做lambda自举预热。

## 数据与身份

沿用PR204之后冻结产品源码、material-cancel native guard、修正BC128×2 epoch12，
md-2026-09全部20合法牌组。全程FP32、TF32关闭、selfplay=1，完整游戏采集、skip_forced保持原配置。
新独立split：train **512局/seed2026100537**，validation **128局/2026100538**，
test **256局/2026100539**，共**896局**。每次8env，发牌/先后攻配对两局同属一个cluster，
整个cluster只属于一个split；检查实际游戏seed集合互斥。相同牌组池的新发牌，不声称新牌组泛化。
保留逐batch原始rollout、完整GameSpec与健康结果；任意错误/截断/无效标签立即停止并保留证据，
不换seed补齐。不因已有16,636训练局而重复利用旧策略训练数据充当固定actor数据。

以实际winner和每行acting player独立生成z∈{−1,+1}，逐行核对terminal_returns及游戏边界。
同局所有行拥有从各自行棋方看待的终局标签；这是价值监督，**不代表赢局每个动作都正确**。
actor权重始终逐位固定，缓存其context和实际所选action embedding；保存原始privileged输入，
privileged encoder仍可学习。现有Q头的每candidate计算独立，故只算taken candidate与完整头取出Q_taken
应等价：正式数据前用旧语料核对输出及全部critic私有参数梯度，原始数据仍保留可复核。

## 拟合与终点

三个新初始化seed **2026100540/41/42**，相同actor与缓存数据，独立critic私有参数和Adam。
仅训练privileged encoder与critic；冻结共享actor主干。
固定**32 epochs**，batch2048，Adam lr1e−3/eps1e−5，grad norm clip .5，
loss=.25×(Q_taken−z)²+.25×(V−z)²，使用全数据归一化的逆游戏长度权重，使每局等权。
每epoch固定seed shuffle，无重采样；不提前停止/选checkpoint/调参。
0/1/4/8/16/32保存诊断节点并计算train/validation；test仅在全部32epoch完成后分析0和32。
参考项：零预测，以及**仅在train拟合并冻结的逐局等权z均值常数**；不用test标签拟合常数。

主指标为test上逐局等权Q_taken终局MSE，报告三seed均值及每seed差、V MSE、EV、
训练/验证曲线、距终局>=64 learner rows的MSE与覆盖率。MSE的统计单位是游戏/配对发牌cluster，
不能把相邻行当独立样本。三模型共用数据，训练集抽样不确定性未被覆盖。
20,000次交叉bootstrap，seed2026100543，分别重采样3个critic seed和128个test clusters，
同一抽样用于全部比较。仅当Q32相对零及train常数的MSE差区间上界均<0，
且三个seed相对常数的均值差均<0，才值得下一步动作排序/策略验证；不是棋力验收。
主终点无明显优势时，如train下降但test不下降支持泛化不足；train也难下降则进一步查表示/优化，
不以一个结果证明唯一根因。Q_taken监督不能验证未选动作排序，需后续独立分支实验。

## 执行与恢复

证据根`out/research/terminal-critic-fit-2026-10-06/`，driver/源码/native/卡库/脚本/BC绑定SHA，
正式数据前提交本协议并完成缓存等价预检。
用独立systemd用户服务运行有界顺序任务，保留service状态和逐阶段日志；每阶段产物原子登记。
重复执行runner时先核验并跳过已完成阶段，可重做尚未执行的分析收尾；
不自动覆盖半份采样/训练，不承诺恢复未结束游戏。父服务失败则整组子进程受控退出，最长8小时。
分析输出成功后才标记complete；分析可独立幂等核验，避免仅凭陈旧heartbeat判断是否完成。

## 运行前核验与启动记录

协议提交`a252136`早于正式数据。实际ROCm GPU上，缓存与原网络Q/V输出误差均0，
所有critic私有参数梯度最大绝对误差**2.98e−8**，actor无梯度；缓存等价预检通过。
第一次预检发现研究入口的inference checkpoint loader将critic也冻结，已在研究driver显式解冻
privileged encoder/critic，actor保持冻结；原错误日志和旧driver保留，不涉及产品源码改动或研究数据替换。
成功预检绑定最终driver SHA，再初始化正式研究身份。

服务`ygorl-terminal-critic-20261006.service`已启动，工作目录为`codex-lr-control-report` worktree，
8小时上限、整组子进程管理、每阶段独立日志；`service-launch.txt`记录InvocationID/工作目录/进程身份。
首16局正常、4,507行，正式阶段仍在采样，尚无拟合或泛化结论。
`pipeline-status.json`表示当前阶段，采样进度在各split的`progress.json`；最终成功须同时有
`analysis.json`、`validation.json`及complete状态。测试集结果不用于中途调整配方。

## 完成结果

896局全部正常：train162,019行、validation40,486行、test82,407行；actor始终逐位固定。
三个seed各32epochs/2,560 Adam steps，总计15,553,824次训练行访问是重复使用162,019训练行，
**不是同等数量的新游戏数据**。独立用户服务自动完成全部阶段及分析封存，耗时约15.2分钟。

| seed | epoch32 train Q MSE | validation Q MSE | test Q MSE |
|---|---:|---:|---:|
| 0 | .153928 | 1.260535 | 1.183450 |
| 1 | .107079 | 1.183885 | 1.233460 |
| 2 | .096330 | 1.315898 | 1.265763 |
| 均值 | **.119112** | **1.253439** | **1.227558** |

test零预测MSE1；仅在train拟合的常数z=.132272，test MSE **.984798**。
主终点Q32−constant **+.242759，95% CI[+.116863,+.367775]**；
Q32−zero **+.227558，[+.100612,+.353683]**，三个seed均更差，继续条件为**false**。
V32 MSE1.231468，也更差。Q32−随机初始化仅−.032320，[−.152931,+.089291]。
test中80.12%行距终局>=64 learner rows；其逐游戏等权Q MSE三seed均值1.290314。

**本预算下能拟合，但明显过拟合且预测幅度过大。** 训练误差持续下降，验证先降后升；
三个seed在所保存节点中的最低验证Q MSE都出现在epoch1（.779501/.759845/.808964），
而固定主终点32较差。不能事后用epoch1替换本轮test主终点；旧test只分析0/32。
独立新游戏确认之前，不把早期验证优势称为已验证泛化或策略收益。

事后描述量显示Q32胜负符号正确率为65.24–67.03%（逐游戏等权的**状态预测**，不是对局胜率）；
E[Q²]=.854–.861，E[Qz]=.298–.336，满足MSE=1+E[Q²]−2E[Qz]。
仍含预测信息，但输出幅度与正确性不相称；不能将失败归为完全没有信号或模型根本拟合不了。
仅512个训练游戏反复32遍、共享固定表示/单一牌组池等因素仍需分开调查，未证明唯一根因。

核验：独立numpy重算所有保存节点train/validation/test逐游戏MSE；test的256局以原始winner/player
重新核对每行标签；用抽样次数权重独立复现全部交叉bootstrap区间。均通过，`audit.json`封存。
图`fit-generalization.png/.svg`及`plot-validation.json`保存完整曲线；不筛选seed或节点。
下一步仅用validation选择的公共epoch1，在事前冻结的新独立游戏面板确认，保留本轮32epoch失败结论。
