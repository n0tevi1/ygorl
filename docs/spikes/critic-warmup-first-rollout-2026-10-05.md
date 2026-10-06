# 实际PPO首批数据与异步采样审计（2026-10-05，事后诊断）

三种子权重预热对照运行期间，读取seed 0 cold/warm各自第一次真实PPO rollout，
不更新模型或修改冻结的研究driver、预算、健康规则和强度终点。
两边初始actor、CPU/GPU/collector RNG和schedule逐位相同，Adam和pool为空。
这不意味着异步采样得到同一份训练数据。

## 差异从第一步抽样开始

128个环境各自第一行的所有公开obs、privileged输入、player和行为概率完全相同，
但其中**48个环境第一步动作不同**。因此起始观测/概率差异不能解释这些动作差异。
后续完整rollout的观测、动作和奖励均不同。

源码机制：`csrc/worker_pool.h`按worker完成顺序发布事件；
`train/rollout.py`按收到的事件顺序，使用同一个collector generator执行批量multinomial。
同一随机数流会映射到不同环境/动作；完成游戏后，发牌schedule也由异步完成顺序调用。
初始随机数状态相同不保证逐步训练轨迹或后续每个环境的发牌相同。

独立Nim控制只倒转ready事件顺序，保持模型、seed、首行obs/概率一致，
**32行动作中26行改变，且两边最终generator状态依然相同**。
这复现了事件顺序敏感性。真实历史ready batch未记录，不能逐批重放历史线程调度；
也不把这一机制宣称为此前棋力退化的原因。正式三种子对照仍有效地比较随机训练过程，
但不能称为逐步相同数据的critic干预，不能用两份首批数据直接作同状态因果比较。

## 保留实际open-tail bootstrap的advantage分解

各臂16,384行；VRPO lambda=.5，原始未标准化advantage。
保留实际bootstrap player/probabilities/Q，将full拆成reward项（Q及bootstrap Q置零）
与critic项（reward置零）；线性重建最大误差<6e−8。

| seed 0实际首批 | cold | warm |
|---|---:|---:|
| 正常终局/非零reward行 | 4 | 5 |
| 所在segment中后续有终局的行 | 385 | 586 |
| reward项绝对值>1e−6的行 | 80 | 100 |
| reward项std | .018038 | .020167 |
| critic项std | .291152 | .026681 |
| full std | .292047 | .030567 |
| corr(full,reward) | .080443 | .510315 |
| corr(full,critic) | .998091 | .759924 |

这是不同采样数据上的描述量，与此前同一完整游戏数据的分解分开解释。
预热后critic项幅度更小，但实际训练首批的终局密度远低于完整游戏；
不能把完整游戏上的.875相关性直接移植到训练，也不能把相关性叫作真实信号比例。
尚无本研究的完整棋力结论；update32、全部三个seed和固定9,984局评估才用于判定。

## 动作容量排查的阴性结果

另对此前校准的39,613个learner rows核查原始合法动作数：最大**56**，超过128的行数为**0**。
这排除了该语料被128候选上限截断的解释；不解决一般ANNOUNCE_CARD的已知容量限制，
也不覆盖被skip的forced decisions或其他游戏。
本地Crossout Designator（65681983）脚本将可宣言集合限制为自己牌组中可除外的卡号，
不能据全卡库ANNOUNCE_CARD的一般风险，断言这张卡在本面板被截掉必要宣言。

证据：`out/research/critic-warmup-first-rollout-2026-10-05/`内
`seed-0.json/.pt`、`order-audit.json`及可复现driver，绑定研究身份、raw/checkpoint manifest和源码SHA。
容量证据：`out/research/observation-capacity-audit-2026-10-05/`内
`report.json`、`verify.py`、`validation.json`；独立复核全部16个raw batch的SHA。

## 三seed补全（2026-10-06）

| seed | cold/warm首批终局数 | cold/warm critic项std | cold/warm corr(full,reward) |
|---|---:|---:|---:|
| 0 | 4 / 5 | .29115 / .02668 | .0804 / .5103 |
| 1 | 9 / 7 | .70252 / .03124 | .0376 / .5348 |
| 2 | 3 / 3 | .30489 / .02623 | .0635 / .4099 |

幅度变化在三个seed中均出现，但[完整强度对照](critic-warmup-control-2026-10-05.md)的warm−cold
主终点仍未过预定门槛；机制变化不等于稳定胜率收益。首批终局数不能代表整段训练的稳态终局密度。
旁路watcher只自动完成seed 1，进程随后不存在；seed 2已用原run.py处理现存rollout补齐，
无新增对局/训练步。全部三份raw绑定、分解和driver已通过validation.json封存，并记录手动恢复。
