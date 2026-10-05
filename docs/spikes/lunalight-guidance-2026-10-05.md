# Lunalight 完整参考线引导（2026-10-05）

## 预注册协议

目的：检验保留 native semantic plan 是否能提高普通洗牌下完整 Liger Dancer（54701958）教师覆盖。
先前 prefix continuation 未增加覆盖；本轮不是追加相同配方预算。

参考固定为 `out/research/teacher-continuation-2026-10-05/control.jsonl` 中唯一成功的 Lunalight
（hand_index=5，hand_seed=9734033099336890053，55 host actions / 50 solver steps）。
它本身是普通洗牌成功线；参考获取成本独立报告，不计作新起手求解成功。
新 hand/solver seed 均为 2026100505，Lunalight indices 0–15，开发集，不是正式留出集。

两臂都使用同一参考、`--start` 指定的真实起手、30,000ms、1 native thread、
max-decisions=692（12×55+32）、max-written=4、保留一条成功线、60s timeout、默认 finisher 分配。
control 加 `--no-plan`；guided 不加；均不使用 `--no-ref`、prefix、fire 或中间目标。
按 hand 奇偶交替安排两臂，4 个 CPU worker，nice19；报告实际 native 时间和准备/验证总时间。
默认深度依赖参考动作数，故必须显式固定；本对照测 native 引导整体而非单一内部机制。

开发推进门槛：guided ≥8/16，且比 control 净多解出 ≥2 个起手，错误/未验证/拒收均为零。
未过则不扩展此配方；过关后独立新 seed 确认，再进行所有 20 卡组的普通洗牌覆盖门槛，
不直接进入容量训练。两组所有成功线独立重放，核对实际 start、turn2、目标和引擎错误。
保存全部失败、命令、日志、源与实现指纹；不按中间结果修改配方或追加样本。

## 结果

尚未运行。
