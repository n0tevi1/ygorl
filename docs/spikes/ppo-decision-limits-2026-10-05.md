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
