# 128×2低LR终点出现决策上限（2026-10-05）

四臂LR对照的全部2,040局新评估已执行。128×2低LR update32新增的200局有4局decision_limit，
各到6,000次决策，在turn9/7/4/6结束；没有Lua/retry/unknown/undecodable/error。
一局对原BC256×2，一局对同臂update0，两局对同臂update16。该节点按事前协议未写入矩阵，
完整四臂健康分析停止；原始200条结果保留。不能把这些上限悄悄记平、剔除后宣布小LR更强，
也不能因后续修复而改写原研究的失败状态。

立即调查全部4局：使用原牌组load order、config、game/agent seeds及冻结checkpoint重新运行，
增加只读逐动作候选/概率、response index、undo标记、事件与引擎消息记录。先核对原GameRecord
所有结果字段；原记录没有完整动作，故不声称逐动作核对了历史轨迹。新轨迹再做无推理冷回放，
逐动作/response/终局一致后才能作为具体循环的复现证据。默认6,000上限、200回合及脚本预算不变。
区分重复host选择、引擎内重复效果与真实长局；先定位触发条件再决定修复和受控复验。
不按有无改善挑seed，不启动更长PPO训练来掩盖当前异常。

此前128×2高LR训练中的1个decision_limit仍缺原动作，不能直接视为同一原因。
新批量评估中的4局是固定模型，可重新运行获取可检查轨迹。
[固定actor的critic校准](critic-warmup-calibration-2026-10-05.md)协议已提交但尚未启动，
先处理本轮评估新暴露的上限；EMA修复本身不改变现有checkpoint推理，不会消除已学到的异常行为。

证据根：`out/research/ppo-decision-limits-2026-10-05/`；原始研究根保持不改写。

## 四局全部复现：合法取消导致目标/素材往返

原GameRecord四局所有结果字段完全复现。新记录的四条完整动作轨迹又经冷回放，逐动作、
response、原始引擎消息与终局字段全部一致。每次取消都确实发送引擎响应，旧undo列表为空；
这不是此前教师数据里尚未提交的host命令，也不是单次引擎调用挂住。

循环在SELECT_CARD选择融合目标，再到SELECT_UNSELECT_CARD取消素材，回到目标菜单。
涉及Guardian Chimera、Lunalight Perfume Dancer/Sabre Dancer和Blue-Eyes Tyrant Dragon。
旧rule 5只限制unselect，且每次回到目标菜单都清零；ChainSolving已清除旧命令取消标志。
因此旧guard对这种跨菜单取消没有累计约束，四局约5,500–5,800步消耗在这里。

同一旧轨迹输入上离线比较128×2低LR的0/16/32三个actor，首次被新规则改变的状态如下：

| case | 轨迹索引（从0起） | update0取消概率 | update16 | update32 |
|---|---:|---:|---:|---:|
| 0 | 223 | .08983 | .66054 | .99880 |
| 1 | 365 | .03702 | .58303 | .99769 |
| 2 | 309 | .11777 | .71830 | .99951 |
| 3 | 390 | .02217 | .38717 | .99855 |

另保留每局5900附近的素材状态，趋势相同；update32概率逐项核对原trace相符。
这证明训练后的actor在这些相同输入上更偏向取消，不能只归因于BC起点本来如此。
但它们是事后选出的失败状态，不能由此推导总体错误率，也不能分离critic、actor loss、
reference更新或在线分布的因果贡献。为什么训练强化此行为仍需继续查。

## 修复和独立复验

设计先行加入policy progress rule 6：同玩家、无游戏事件、连续目标/素材选择链累计取消；
最初32次仍允许，从第33次起mask素材cancel。玩家变更、其他decision类型或游戏事件重置；
Hint不重置。保留原始合法actions/responses，唯一出口不mask，显式旧trace仍可回放。
Python/C++同步实现，snapshot保存计数。这是策略进度约束，不是把合法引擎动作改成非法。

真实Blue-Eyes前缀回归在修前因第33次cancel仍可选而失败；修后两host逐字段encoding一致，
snapshot与reset/唯一出口契约通过。定向39 passed（11.49s）。完整实际ROCm测试 **1499 passed、3 skipped（287.34s）**；测试源码ef24ffc，证据已封存。

所有四局使用相同模型、牌组、配置和随机种子，在新实现下独立复验：

| case | 原决策数 | 新决策数 | 新回合数 | 正常终局 | 首次变化前动作完全一致 |
|---|---:|---:|---:|---|---:|
| 0 | 6000 | 726 | 13 | 是 | 223 |
| 1 | 6000 | 664 | 12 | 是 | 365 |
| 2 | 6000 | 745 | 13 | 是 | 309 |
| 3 | 6000 | 506 | 6 | 是 | 390 |

首次变化均精确对应旧cancel被新guard屏蔽；没有Lua/retry/unknown/undecodable/error。
四条新trace再次冷回放，动作、responses、原始消息与终局完全一致。
这些是针对已知失败的修复验证，不能当新强度评测，更不能替换旧矩阵失败数据。
新证据单独在`out/research/material-cancel-guard-2026-10-05/`。
本地原始third_party未改写；新native构建与旧构建的104个core/Lua源文件逐字节相同。
