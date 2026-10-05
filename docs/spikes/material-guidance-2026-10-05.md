# Lunalight 融合材料引导（2026-10-05）

## 先验审计与问题

前轮单参考线 7/16 vs 5/16，离 ≥8/16 开发门槛只差一手，但仅四对起手结果不同（3 赢/1 输）。
门槛是预先指定的筛选条件，不是统计显著性或正式教师验收；不能通过追加当前种子起手凑过线。
本轮换机制、使用新开发种子；旧 16 手仅用于模型诊断。

以旧 12 条完整成功线做 native `--operators --target 54701958 --growth --growth-max 0` 审计：
780 个响应均可枚举，发动均有脚本声明匹配，已检查的区域/次数前置条件无违反；
材料 LP 沿全部成功线没有不可达判断，最终距离均为零。但每条线仅两次距离下降，
合计 24/780，说明该距离在已知长展开上仍然稀疏，不能把“解析通过”当作高质量引导。
该检查只覆盖声明提取和有限真实路径，不证明抽象模型对所有卡效完备/可靠。

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

尚未运行。
