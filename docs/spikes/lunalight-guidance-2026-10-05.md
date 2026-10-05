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

生成实现：`b7a17ca`；环境 `md-2026-09`，完整指纹
`65ca28f79233e73d540a23c2046718ef42749ae883baa920216b94365b3e42d9`。
机器 AMD Ryzen AI MAX+ 395，本轮求解用 CPU；GPU 仅用于后续回归测试。

| 臂 | 解出 | 未解出 | error / unverified / 拒收 | native 累计秒 | 准备+求解+验证累计秒 |
|---|---:|---:|---:|---:|---:|
| control（清空 plan） | 5/16 | 11 | 0 / 0 / 0 | 398.16 | 400.90 |
| guided（保留 plan） | 7/16 | 9 | 0 / 0 / 0 | 366.86 | 370.38 |

共同成功 4 手，guided 独有 3 手（6、13、15），control 独有 1 手（8），共同未解出 8 手。
引导净增 2/16，但绝对覆盖 **7/16 < 8/16**，预设推进门槛失败，不扩大此单参考线配方。
这不是引导无效的证明，也不是模型容量/长期 RL 的上限结论；小开发面板不足以宣称稳定收益。
两臂并集 8/16 使用了每手两次预算，不能报告成一个 30 秒教师的覆盖。

参考不是免费获取的：原 control 的 8 手 Lunalight 搜索合计 218.49 native 秒，
被选中的唯一成功线为 6.97 秒；更早中间线生成成本另见前轮报告。新面板未重新计入这些成功。

## 独立核对

- 所有 32 条（含失败）环境、手牌、双方牌序、核心 seed、普通规则、目标一致。
- 16 对仅命令中的 `--no-plan` 不同（忽略各自文件路径）；固定深度均为 692，native 报告的默认深度为 114。
- 参考线解析出 41 个 semantic plan 步骤、0 个未识别步骤；control 明确 `SET ASIDE`，guided 保留。
- control 296 步、guided 484 步，共 **780 步**独立 host 回放，最终目标、turn 2、合法动作、引擎错误检查全过。
- 对所有 12 条成功线额外做 native 枚举覆盖和 snapshot 检查：**780/780** 必要响应可覆盖。
  使用 `--growth --growth-max 0` 启用检查但不展开搜索；不计作额外求解。
- 注意 upstream 的普通 replay 默认只检查 snapshot；`--verbose` 不输出 `selfChecks` JSON。
  此外 `selfChecks.pass` 不能代替 coverage 分子/分母检查；本审计显式要求非零且相等。
  初次诊断脚本对 verbose JSON 的错误假设已修正，原日志保留，未改变主实验。
- 同一 manifest 零工作续跑，32 条汇总一致，原始 JSONL SHA256 不变：
  `83b2249ebce60dd68e7e2f109f92b0f61ce766b73a86f99696520ee3dc8281b7`。

## 判断与下一步

单一完整参考线改善部分起手，也会损失其它路线；当前不够支撑正式教师数据冻结。
源码的 `max_subsets=24` 确实可能截断组合，但本轮 ladder 日志没有非零 subset 截断，
且已知成功路线全部可枚举，**没有证据把这轮缺口归因于材料枚举 bug**；对未知失败路线仍不能排除。

下一项优先检查最终目标的材料分解：Liger 脚本要求 Leo Dancer 名称材料加三个 Lunalight，
Kaleido Chick 的改名能力使“必须先召唤 Leo”不成立。先沿已知完整成功线核对 native
`--recipes / --op-recipes / --backward` 对这些条件的建模，排除错误约束，再冻结新 seed、
同总预算的目标分解 vs 普通搜索对照。保留原最终目标，不把中间材料到位当成功，
不在已暴露的本面板上选参数后宣称独立验证。

正式 20 牌组普通洗牌覆盖、训练/留出冻结、#88 容量对照和 #92 长训练仍未完成；
本轮没有新 policy checkpoint 或对局强度结论。

## 工件与复现

`out/research/lunalight-guidance-2026-10-05/` 保留 `run_pilot.py`、`paired.jsonl` 及 manifest、
两臂 JSONL、`summary.json`、全部逐任务命令/日志/参考/start/候选、`audit_outputs.py`/`audit.json`、
`audit_native_coverage.py`/`native-coverage.json`、零工作 `resume.log` 和完整测试日志。
运行前固定实现与输入；driver 通过输出锁和 manifest 拒绝混入不同实验。

参考记录 canonical JSON SHA256：
`414ec95fbb9565d8f973ba075293545d84f6a94c7f454aa53cac0b79e5c2f0e6`。
求解器源 `e0c7221802a23657d5a9a0b020025e0ceca04675`，实际 binary/core/cards/scripts/Python/driver 内容哈希在 manifest。

验证：solver 单测 33 passed；已测实现 `b7a17ca` 完整 `tools/presubmit.sh --test`
**1,425 passed / 3 skipped，294.34 秒**，包含实际 ROCm。三个跳过分别为 snapshot 编译选项和两个可选联网测试；
后续只有文档改动。`validation.json` 封存本轮工件哈希；复核零工作续跑保持原始 JSONL 哈希。


后续已完成：[材料图/backward 新种子对照](material-guidance-2026-10-05.md) 为 2/16 vs 2/16，
未过门槛；该轮不带参考捕获、novelty 默认为 12，与本轮 19 不同，不能当作本轮单参考线配方的直接复测。
当前下一步为失败起手可达性诊断，正式教师/容量状态见后续报告。
