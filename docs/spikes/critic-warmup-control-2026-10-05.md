# Critic预热权重对策略强度的三种子控制（2026-10-05，运行前协议）

固定actor校准仅得到很弱的终局信号：Q32 MSE .9882、EV .0131。它通过预定继续条件，
但不足以证明advantage或棋力改善。本实验检验预热后的critic权重是否改善后续PPO，
避免同时带入warmup阶段的Adam状态、采样RNG、未结束游戏或对手池。

## 固定干预和预算

仅使用修正BC 128×2的12epoch末尾actor、md-2026-09全部20合法牌组，源码77f4f84之后保持冻结，
包括修正EMA和跨菜单取消guard。FP32，TF32/overlap关闭。全部使用128env×128steps、
VRPO lambda=.5、2epochs×2048minibatch、KL guard .01、entropy/reference系数各.05。

三个独立配对：warmup seed **2026100527/28/29**，对应策略训练seed **2026100530/31/32**。
每个warmup按已验证配方：actor及共享主干冻结32更新，selfplay=1、无snapshot、critic LR1e−3，
禁用EV提前结束；保留update0和32，逐更新核对actor完全不变。三个均使用新seed，不选择旧校准中表现好的critic。

随后每对各建两个全新Trainer，分别加载该warmup的update0（cold）或32（warm）的model权重。
actor、词表、reference的actor逐位相同；仅critic专用参数不同。两边Adam为空，计数为0，
相同的新策略seed、collector/schedule/RNG、空对手池；不恢复warmup checkpoint的optimizer/RNG。
比较的是**critic权重预热**，不是默认Trainer连续warmup所保留的整套状态。

六个策略run各固定 **32次PPO更新/524,288行**，保存不可变0/16/32完整checkpoint与SHA。
策略LR统一 **1e−4**，selfplay=.75、PFSP、pool8、snapshot10，其余默认不变。
LR来自此前首步KL控制的机制证据，不是从健康停止的四臂胜率挑赢家。
每个配对按warmup→cold→warm执行，配对按seed顺序；单GPU串行，独立CPU消费者读取已登记节点。
不在看到中途结果后停换配方或加长预算。

总预算：3×32 critic-only + 6×32 PPO = **288更新、4,718,592行**。
主比较匹配PPO更新数；warm臂额外预热524,288行/seed，**不是相同总算力或总样本预算**。
分别报告预热和策略成本，不由此宣称更高样本效率。保存每个策略run第一次实际rollout，供失败归因复查。

## 独立强度面板

新eval seed **2026100533**，64个有放回抽取的ordered deck pair/deal cluster，每个cluster交换牌组和
先后手4局；每个节点对三个固定对手各256局：Greedy、原BC256×2、修正BC128×2初始actor。
所有对手均冻结；候选与initial128使用PPO CheckpointAgent相同采样接口，原BC256基线保留其BC PolicyAgent接口。
共用发牌和slot agent seeds，不跨实现拼接旧面板。
固定对手等权。先验证所有初始actor逐位相同，再复用同一初始节点的768局，其余6×2节点各768局，
合计 **9,984新局**。报告全部0/16/32；16为描述性节点，32为唯一主终点。
用现有pairing_slots/cell_specs协议和独立逐节点对手比较，无需创建全候选两两矩阵或解读跨矩阵排名。
每局保留完整GameSpec/模型SHA/结果；异常局另外保留显式动作前缀，以及引擎已返回的responses和候选/概率trace。
若agent抛异常而无DuelResult，也保留已提交的动作前缀，不声称拥有未返回的响应记录。
统计候选策略material cancel的可用次数、mask次数、选择次数；mask包含既有各规则，非rule 6独有计数。
这些是策略访问分布上的描述量，不能当作同状态因果效应。

主比较：update32三个固定对手等权的warm−cold胜率。
报告每个训练seed差值、三个seed均值；bootstrap seed **2026100534**，20,000次对训练seed（3个）和
发牌cluster（64个）分别有放回重采样，同一次抽取共享到所有臂/对手，保留配对和每cluster四局结构。
只有3个训练seed，该区间仍不能充分覆盖训练方差；同时报只对deal cluster重采样的条件区间。
另比较warm32−初始actor，防止把“退化较少”叫成学习收益。原始预热/策略SHA、完整健康和所有节点必须齐全。
仅当两项主终点比较区间下界>0，且三个warm−cold seed差均>0，才认为值得进入更长的固定预算确认；
这仍不是顶尖对局或#92两万更新验收。其余结果完整保留，不在同一面板加样本求显著。

## 健康与停止

训练每轮先查引擎健康/非有限值；累计>=100终局后截断率>1%停止，保留部分checkpoint与原轨迹。
评估任意错误/上限立即保留并标记全研究停止，其他任务在下个安全检查点停止，不把上限计平或换seed补齐。
完成cell绑定输入SHA后可复用；部分cell不自动覆盖或补跑。任何源码/driver变更均须新的研究身份或明确事前修订。
若出现新失败，先调查修复，再决定独立复验；旧失败结果保留。

证据根：`out/research/critic-warmup-control-2026-10-05/`。

运行前集成检查：真实GPU上以不同critic权重验证actor、CPU/GPU/collector RNG与schedule一致，
Adam和pool为空；没有生成训练行。预检最初把BC格式checkpoint传给要求PPO候选的监控器，触发
PolicyAgent缺少host属性，已保留setup错误并修正预检使用真实PPO格式。历史非研究面板对局复检通过。
正式研究尚未初始化；这次入口修正不替换任何研究数据。driver增加agent异常时的已提交动作前缀记录。
