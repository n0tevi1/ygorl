# Lunalight 失败起手可达性诊断（2026-10-05）

## 冻结协议

本轮是失败归因，不是追加上一轮样本或修改已关闭的 gate。
输入固定为 `material-guidance-2026-10-05` 两臂共同未解出的 13 个普通洗牌起手：
indices 0,1,2,3,5,6,7,8,9,10,12,13,14（hand seed 来源 2026100506）。
直接复用记录的双方牌序、核心 seed、规则及目标 Liger Dancer 54701958。

每手一次新搜索，solver seed=2026100507，120,000ms，1 native thread，深度 692，
`--no-ref --start --novelty 0 --no-serial --reenter 0`；无 recipe/backward、参考引导、prefix、fire。
显式关闭 novelty pruning 以寻找被短预算/剪枝漏掉的路线；这同时改变预算、solver seed、novelty，
只能用来证明“存在合法成功线”，不能归因为某一个开关的因果收益。
4 CPU workers、nice19、max-written=4、保留 1 条完整验证线，timeout=150s。
所有失败保留；无解只能标为 unknown。成功线须核对精确起始身份、turn2、原目标并独立重放。

结果将决定后续工作：先核对新增可达证据，再针对仍未知起手检查发动/资源路径。
不同诊断阶段独立记录成本和条件，不能回填上一轮 30 秒 gate，也不能声称穷尽搜索。

## 结果

尚未运行。
