# 普通洗牌全牌组覆盖复验（2026-10-05）

## 冻结协议

失败起手诊断找回 6 条 Lunalight 完整线，证明旧开发面板至少 9/16 可达，但成本与参数已改变，
不能宣布通过旧 30 秒 gate。本轮用独立新开发种子验证一套明确教师配置，仍不是正式训练/留出数据。

环境 md-2026-09 的全部 20 套合法牌组，各 indices 0–15；hand/solver seed=2026100508，共 320 手。
目标固定为 `md-bc-targets-2026-10-05.json`，不降低任何牌组目标。
所有任务 `--no-ref --start` 普通洗牌、无参考/prefix/fire/recipes/backward；
Lunalight 120 秒、novelty=0，其它牌组 30 秒、novelty=12；共同 `--no-serial --reenter 0`。
显式深度 `12*(main+extra)+32`，1 native thread，max-written=4，保留一条完整成功线，
timeout=solve_ms/1000+30 秒；4 CPU workers、nice19，按起手 index 交错牌组安排。

覆盖门槛：每个牌组 ≥8/16，全部 0 error/unverified/拒收；不能用总体平均隐藏单牌组缺口。
所有输入先检查完整环境、合法性、目标在牌组内。每条成功线精确起始身份、turn2、最终目标、
合法动作和引擎错误独立回放；未解出保留精确 start。所有 native 文件、命令、日志和 manifest 保存。
通过后才冻结正式训练/留出；若失败，保留逐牌组缺口，继续定向诊断，不追加本面板后改写 gate。

## 结果

共 320 手：**265 solved / 54 unsolved / 1 unverified**。0 native error/timeout；Sky Striker 5 的四条候选在未结算连锁中途达标，结束回合后失去 Shizuku，均被 host 正确拒收。

| 牌组 | 解出 / 16 |
|---|---:|
| blue-eyes | 11 |
| branded | 14 |
| clown-crew | 15 |
| dracotail | 13 |
| elfnote | 16 |
| elfnote-kewl-tune | 10 |
| heros | 15 |
| kewl-tune | 15 |
| lunalight | 7 |
| magistus-fairy-tail | 14 |
| maliss | 15 |
| odion | 5 |
| orcust | 16 |
| radiant-typhoon-zoodiac | 11 |
| resonators | 14 |
| ryzeal-mitsurugi | 14 |
| sky-striker | 15 |
| tearlaments | 16 |
| vanquish-soul-k9 | 14 |
| yummy | 15 |

**Gate 失败**：Lunalight 7/16、Odion 5/16，且有上述候选拒收。不能用总体 82.8% 或接近 8/16 代替逐项通过。

全部输入的环境、合法卡组、普通洗牌规则、手牌/牌序/core seed、被动对手、目标及实际命令已核对。
265 条成功线 **18,042 个 host 动作**独立回放通过；原生枚举 **18,024/18,024 响应**覆盖及 snapshot 检查通过。
多选会拆成多个 host 动作，不能将两种步数混用。四条拒收错误独立复现。
BC 开发编码 **6,060 样本**，重复编码和带身份保存/读取逐元素一致，0 beyond_128。
原始 JSONL SHA256：`eaaaa3af29b41444483edd3e93865ad0a45d2060272d69106f83c9c8effdfd3b`。

已继续[未解出根因调查](solver-failure-causes-2026-10-05.md)：捕获声明卡名漏路、连锁中途目标误报，
对全部 55 个失败记录开展单独的修复后诊断；不追加原面板凑过线。正式数据冻结与容量对照仍未完成。

