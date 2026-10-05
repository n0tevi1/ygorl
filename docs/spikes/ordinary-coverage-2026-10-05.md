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

尚未运行。
