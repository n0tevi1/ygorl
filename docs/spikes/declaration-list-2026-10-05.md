# 有限声明列表漏路（2026-10-05）

## 捕获与实现协议

55 手原生修复复验中唯一仍出现默认回答缺失的 Ryzeal-Mitsurugi 1，捕获实际 prompt 142。
其过滤器为 23 个 `[password, OPCODE_ISCODE]` 通过 OR 左结合连接；这是有限明确的候选列表，
不能当作无答案或拿单个默认卡名应付。原生捕获前缀和完整 payload 保存于上一轮诊断工件。

修复只扩展已支持的单密码：识别完整的左结合 ISCODE/OR 过滤器，逐个做数据库、alias/token 检查，
去重后枚举所有合法密码，不使用 max-subsets 截断这种有限列表。复杂 RPN 仍不猜测答案。
先用 host 对真实前缀核对全部 23 个声明，然后逐一追加合法回答，检查原生枚举覆盖；
同一语义下的单密码与 unresolved-chain 回归必须保持通过。
固定 start/目标/30秒/seed2026100508 重测 Ryzeal 1，修复漏路不自动等于恢复完整解。

下一轮可达性诊断：对原 54 个 unsolved 中扣除 5 个有局限规则论证的起手，剩余 **49 手**使用修复后教师。
每牌组参考线固定为原 320 手中最小成功 index 的完整线；参考获取成本已包含原覆盖批次，不当免费教师。
保持原精确 start 和最终目标；solver seed=2026100509，普通洗牌；Lunalight 360秒，其它120秒，novelty=0，
深度 `12*(main+extra)+32`，1线程/max-written4/keep1，关闭serial/reenter，4workers/nice19。
完整参考引导、预算、seed 与部分剪枝联合改变，**只调查存在路径，不能分离各项因果收益**。
完整回放不通过的候选继续拒收；若仍未解出标 unknown，不回填原面板或据此宣布 gate 通过。

## 有限列表修复结果

实际 host 前缀 2 个响应后得到恰好 23 个合法密码，逐一追加它们的回答。
旧 PR #185 的单密码实现不覆盖此 prompt；修复版每个密码均 **3/3** native 覆盖及 snapshot 检查通过，
单密码和连锁稳定性回归保持通过，全部 solver tests **37 passed，10.95 s**。
相同 start/seed/30 秒重新搜索 Ryzeal 1 后，无默认回答缺失日志，但仍未找到完整线；不把漏路消失等同于恢复完整解。
49 手联合指导/预算诊断已按上述协议启动，执行端保持旧稳定 Python 实现，不受 BC 分支修改影响。

## 执行偏差与纠正

首次执行 `guided/run_guided.py` 漏设 API 的 `reference_guidance=True`，实际命令包含 `--no-plan`。
因此该批不是计划中的参考引导；保存原 driver/manifest/命令/日志，按实际的 **goal-only 较长预算诊断**解释。
此发现时已看到 Lunalight 2 和 ElfNote-Kewl Tune 4 恢复；这些仍是有效可达性证据，但不能归因于参考引导。
不修改正在运行的产物/身份，不把目录名当实际参数。

纠正后另开 `guided-enabled/`：同 49 手、同二进制、同 seed/预算/novelty/深度、相同参考文件内容，显式启用引导，
并在调用 solver 前断言 `--no-plan` 和 `--no-ref` 均不存在。原批继续作为实际配置已冻结的诊断对照。
比较发生在发现原批部分结果之后，只用于机制调查，不宣称预注册确认性比较或独立 gate 通过。
两批均4workers/nice19，总共8个单线程任务；并行时墙钟预算仍有调度波动，记录实际搜索时间和状态数。

## 完整验证

已测合并实现 **ae51a87**：全套 presubmit **1,431 passed / 3 skipped，328.43 s**，含实际 ROCm。
跳过 snapshot 编译选项与两个可选联网测试；无 hosted checks。该实现已包含 PR #186 的 BC 标签边界。
实验仍在运行的两批产物不封存为最终结果；稳定的捕获、回归、构建和完整测试证据另行封存。

## 49 手诊断完成（后续独立审计）

| 实际配置 | Solved | Unsolved | error / unverified / 拒收 | native 秒 |
|---|---:|---:|---:|---:|
| goal-only，目录 `guided/` | 6 | 43 | 0 / 0 / 0 | 6,112.58 |
| reference-guided，目录 `guided-enabled/` | 6 | 43 | 0 / 0 / 0 | 5,967.60 |

共同恢复 Dracotail 5/6、ElfNote-Kewl Tune 4、Lunalight 2/13；goal-only 独有 ElfNote-Kewl Tune 15，
guided 独有 Lunalight 8。因此 **7 手新增可达性证据，42 手仍 unknown**。
参考引导没有净覆盖收益；这不是稳定优于 goal-only 的证据，不继续把单参考引导当默认改进。
两批联合改变的预算/seed/novelty 与原面板不同；其并集花费双臂预算，只证明存在路径，不回填旧 gate。
这些教师仍只达成单卡目标，原生找到的长线可能消耗大量资源，并非实战最优展开。

独立检查全部98条的 start/手牌/环境/目标/参考hash/实际命令；配对命令除路径和 `--no-plan` 外一致。
两组恢复线分别484/498个 host 动作，482/494个 native 响应；**982步完整回放、976/976原生覆盖及 snapshot 通过**。
没有新的 `NO default answer` 日志；没有该警告不等于证明所有合法分支均被枚举。
两批零工作续跑均保持49条和原始hash。首批 manifest 内 exclusion 用了 Python tuple，回读JSON成为list，
需显式零工作 adapter 做JSON表示规范化；原 producer/manifest 内容与hash保持不变，不隐瞒此续跑偏差。

工件：`out/research/declaration-list-2026-10-05/comparison-audit.json`，两批原始命令/回放/日志和续跑记录。
下一步继续逐手规则/候选覆盖调查，同时按 [有限课程容量协议](bounded-capacity-2026-10-05.md)
生成全新训练/验证数据；这项推进显式保留旧覆盖 gate 失败，不再让单一50%阈值代替教师质量判断。
