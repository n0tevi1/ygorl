# 检查点登记与矩阵健康计分（#90，2026-10-05）

训练之前只有同步evaluate，无法让长RL稳定产出独立矩阵曲线。设计80cc108先于实现fda59d1：
`register_every=0`默认关闭，打开后先保存不可覆盖PPO checkpoint，再原子发布含run/update、
rows、累计训练秒数/活动wall秒数、环境/目标/权重hash的登记。消费者锁定矩阵，复用增量扩展，
逐checkpoint原子提交矩阵，重建固定基线胜率和当前矩阵排名。历史更新冲突拒绝覆盖，
矩阵已存但曲线未写的崩溃可修复；只保证完整checkpoint级恢复，未提交的当前评估可能重跑。
使用说明见 [训练文档](../training.md)；不同机器需要相同的共享绝对路径及冻结特征表。

定向94 passed；登记实现实际ROCm全套 **1,454 passed /3 skipped，509.13s**。
覆盖默认关闭无新目录/计数、真实训练按间隔登记、续训更新号/wall累计、不可覆盖、消费者重复运行
不重打任何cell、曲线中断后恢复、环境/牌组/权重变化防护、有错误矩阵不发布。

## 新发现的根因

现有 `agent_matrix.score` 仅按reason排除error/exception，即使GameRecord携带retry、Lua、
unknown/undecodable或error文本但最终reason=win，也会计入胜率。另一个问题是
`play_policies` 转换原生结果时丢弃健康计数：实际注入Ash Lua错误，结果正确为exception，
但script_errors错误显示0。前者可污染计分，后者掩盖诊断；不能由此反推历史污染比例。

新增5种正常win附异常字段测试，以及真实Lua的批量评估测试：修前 **6 failed**；
修复后矩阵、批量评估、登记和CLI定向 **125 passed，77.62s**。
现在所有这些异常整局计入errors并排除胜负分母，批量路径传递计数；独立消费者拒绝发布含errors矩阵，
保留rejected工件。健康干净的旧1,280局研究面板不受计分修复影响，逐局原始字段均已审计。
修复不重写历史矩阵，不将缺字段的旧记录推定健康，也不改变现有回合/决策上限的计分规则；
正式实验需要保留并单独审计上限局。

原生日志二进制仍为PR192的5f64e558；此次仅Python计分与传递修复。
最终完整测试及源身份随 `out/research/training-registration-2026-10-05/validation.json` 归档。
接续实验已经 [预注册短程PPO协议](registered-ppo-pilot-2026-10-05.md)，不冒充#92长跑验收。
