# Lunalight 融合材料引导（2026-10-05）

## 先验审计与问题

前轮单参考线 7/16 vs 5/16，离 ≥8/16 开发门槛只差一手，但仅四对起手结果不同（3 赢/1 输）。
门槛是预先指定的筛选条件，不是统计显著性或正式教师验收；不能通过追加当前种子起手凑过线。
本轮换机制、使用新开发种子；旧 16 手仅用于模型诊断。

以旧 12 条完整成功线做 native `--operators --target 54701958 --growth --growth-max 0` 审计：
780 个响应均可枚举，发动均有脚本声明匹配，已检查的区域/次数前置条件无违反；
材料 LP 沿全部成功线没有不可达判断，最终距离均为零。但每条线仅两次距离下降，
合计 24/780，说明该距离在已知长展开上仍然稀疏，不能把“解析通过”当作高质量引导。
共 87 次发动中有 7 次只能按卡片匹配、存在效果歧义；实际检查了 36 次区域条件、16 次资源条件。
该检查只覆盖声明提取和有限真实路径，不证明抽象模型对所有卡效完备/可靠。
LP 距离与本轮实际使用的 recipe 距离是不同指标；24/780 只是前置诊断，不能拿来直接解释新配方的失败原因。
本轮已关闭 LP serial，实际 recipe/backward 运行数据另见 `diagnostics.json`。

脚本与 native 源码均支持 Kaleido Chick 取得 Leo Dancer 名称。Liger 的声明是 Leo 名称材料
加三个 Lunalight，而不是必须先召唤 Leo。当前 `--op-recipes` 已有名称获取边，不预设该功能缺失。

## 冻结对照协议（求解前）

- seed / solver seed：2026100506；当前合法 MD Lunalight，起手 indices 0–15，共 32 个任务。
- 两臂普通洗牌 `--no-ref --start`，无参考动作、无 prefix、无 fire；原最终目标 54701958。
- 每臂 30,000 ms、固定深度 692、1 native thread、max-written=4、保留一条成功线、timeout=60s。
- 共同参数 `--no-serial --reenter 0`：关闭材料 LP 的自动分阶段/重新入场，避免混入第三种机制。
- material 臂额外 `--recipes 1 --no-seed-recipes --op-recipes --backward`。
  仅用 Lua 声明初始化材料图，不混入英文文本启发；保留 native 基于实际召唤更新图的行为。
  本对照评价“材料图距离+反向分解”的组合，不分别归因于每个开关。
- 必须先在一个已知起手确认 native 日志确实启用配方、名称获取边、backward，关闭 serial/reenter；
  该 smoke 仅检查参数接线，不计入新种子面板，不能据此调参。
- 4 CPU workers，nice19，按起手奇偶交替安排两臂；报告实际时间，保留所有失败和 native 输出。
- 推进门槛保持 material ≥8/16 且净增 ≥2 手，0 error/unverified/拒收；通过后仅允许独立确认，
  不直接宣称教师数据、容量训练或策略强度验收。不通过则停止扩展此固定配方。
- 所有记录（包括未解出）检查环境、seed、双方牌序及原目标；成功线独立 replay 至 turn2；
  检查 native 必要响应覆盖；manifest 绑定环境、deck/目标/预算、实际 solver/core/cards/scripts/Python/driver。

## 结果

生成提交 `a75652d`（运行代码与 main `c3e79b4` 相同），环境 `md-2026-09`，指纹
`65ca28f79233e73d540a23c2046718ef42749ae883baa920216b94365b3e42d9`；
Ryzen AI MAX+ 395，4 CPU workers × 1 native thread。

| 臂 | 成功 | 未解出 | error / unverified / 拒收 | native 累计秒 | 准备+求解+验证累计秒 |
|---|---:|---:|---:|---:|---:|
| control | 2/16 | 14 | 0 / 0 / 0 | 444.29 | 445.18 |
| material | 2/16 | 14 | 0 / 0 / 0 | 443.32 | 444.30 |

共同成功手 11；material 独有手 4；control 独有手 15；共同未解出 13 手。
**零净覆盖收益，且未过 ≥8/16 门槛**，不扩大这个固定的材料图+backward 组合。
两臂并集 3/16 用了双倍尝试预算，不能当单臂结果。

这不直接复测上一轮单参考线方案：除种子不同，上一轮两臂保留参考捕获，仅移除/保留 plan；
本轮两臂 `--no-ref`。源码 `MeasureWidth` 从参考线校准 novelty patience，上一轮为 19，
本轮全部 32 条为 12。两轮各自内部公平；**不能把 7/16 → 2/16 解释为材料引导的因果效果，
也不能把变化全部归因于种子**。本轮材料组合的因果对照是同面板的 2/16 vs 2/16。

## 机制与结果复核

- smoke 在已暴露起手上仅用 1ms 搜索预算检查接线（没有新种子数据），未用于选参数。
- 所有 material 任务日志都有 recipe weight=1、Lua operator 初始化、Chick→Leo 名称获取边、
  backward 开启；没有英文文本初始化或 serial/reenter 激活。两个名称获取边分别对应 Leo、Panther。
- 图确实参与了 rollout 估值和分解，不是空开关。某些失败任务的平均距离下降、backward 计数非零，
  仍未达最终目标；中间指标改善不等于覆盖改善。全部逐任务诊断在 `diagnostics.json`。
- 全部 32 条（包括未解出）的环境、hand seed、核心 seed、双方牌序、普通规则、原目标核对一致。
  16 对命令只相差预注册的 5 个 material 参数 token（忽略各自文件路径）。
- 成功线 control 170 步、material 126 步，共 **296 步**独立合法动作/响应/目标/turn2/引擎错误检查通过；
  native 枚举覆盖 **296/296**，snapshot 检查通过。
- 零工作续跑：pending=0，32 条全量汇总相同，原始 JSONL SHA256 不变：
  `2aeff4bd8000689e70049a1dd89d5d12bde545e4b48eb8719ec3cfc4cda33b55`。

## 关于“只差一手”的判断

上一轮 7/16 距离预设 8/16 的确只差一手（6.25pp），但这是开发筛选门槛。
把起手视为独立样本的粗略 Wilson 95% 区间约 23%–67%，不包含搜索运行随机性的全部不确定性；
不能把样本上的接近当作真实覆盖已稳定接近 50%。更不能把 50% 起手教师目标覆盖等同顶尖完整对局能力。
本轮不追加同面板起手或预算来“补过线”，也不宣布所有材料启发无效。

## 决策与下一步

这两轮都没有验证足够的教师覆盖。下一步从收益筛选转为失败归因：
冻结包括 novelty 在内的全部搜索参数，对本轮 13 个共同失败起手做独立的可达性诊断。
更长预算或其它教师找到完整线，可证明该手是搜索漏解；仍未找到只能记 unknown，不能记不可达。
用这个证据区分起手本身的限制与 30 秒搜索损失，再选择教师预算/路线；不把诊断追加结果混回本轮 gate。

不继续堆叠默认启发开关，不降低 Liger 最终目标，不因一轮 negative 宣称 RL 容量上限。
正式 20 牌组普通洗牌覆盖、训练/留出冻结、#88 大网络和 #92 长训练仍开放；当前无新策略 checkpoint。

## 工件与验证

`out/research/material-guidance-2026-10-05/` 保存：`audit_operators.py`、`audit_materials.py`、
旧线逐任务声明/材料审计、`model-audit-summary.json`，`run_pilot.py`、smoke、全部新求解输入/命令/日志/候选，
`paired.jsonl` 及 manifest、两臂 JSONL、`summary.json`、`audit_results.py`/`audit.json`、
`audit_native_coverage.py`/`native-coverage.json`、`diagnostics.json`、`resume.log` 和测试日志。
manifest 记录实际 driver/Python/native/solver/cards/scripts 哈希；求解器源为
`e0c7221802a23657d5a9a0b020025e0ceca04675`。

完整 `tools/presubmit.sh --test`：**1,425 passed / 3 skipped，292.02 秒**，包含实际 ROCm；
跳过项为 snapshot 编译选项和两个可选联网测试。已测提交 `a75652d`，本轮仅文档变动，
运行代码与已合并 main `c3e79b4` 相同；`validation.json` 封存 1,811 个工件文件内容哈希。
