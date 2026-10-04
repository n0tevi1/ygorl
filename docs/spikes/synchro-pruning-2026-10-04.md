# 同调素材搜索：保留回调语义的剪枝（2026-10-04，#169）

已知两个同调预算失败在默认预算下恢复推进。修复通过内容哈希限定的内存脚本覆盖默认启用，
不改 CardScripts 子模块，不提高 100,000 千条指令的默认预算。融合路径未修改，#169 保持开放。
设计依据见 [引擎设计 §2.4](../design/01-engine.md#24-版本限定的同调搜索剪枝2026-10-04169)。

## 为什么旧原型不能直接采用

此前 [隔离原型](aligned-response-2026-10-04.md) 的 75 局消息一致不足以证明回调安全。
本轮构造了真实 Lua/core 反例：当前调整/非调整素材等级为 7 + 8，目标 9；未来手牌素材的未知检查回调
将已选等级 8 改为 1，再加入自己的等级 1，形成合法的 7 + 1 + 1。
旧原型在 7 + 8 时提前拒绝；原脚本与本轮修复都接受。旧原型从未用于正式 RL 运行。
这也说明不能仅检查已选素材，必须检查所有剩余候选上的回调和 custom material 效果。

`Synchro.CheckP42` 中调整组固定，递归只增加非调整素材。只有当前和剩余素材的等级都固定、
为 1–65535、同调等级等于普通等级，而且没有能改变判断的自定义约束时，超目标的等级和才不可能恢复。
超过 65535 的数值可能被核心解释为两个 16-bit 可选等级，不能按普通正整数剪枝。

手牌素材检查仅放行本版本 `Synchro.CreateHandMaterialEffect` 创建的 `synchktg` 闭包：
该闭包只检查标签并筛选/返回排除组。私有弱键表保存函数身份；任意未知函数或创建后被替换的函数退回原搜索，
不能用效果标签冒充安全回调。等级修改、特殊/双值同调等级、custom material、素材限制、
`req2`、`reqm`、`Synchro.CheckAdditional` 等均退回原搜索。普通属性与效果查询沿用引擎的只读契约。
剪枝之后仍执行原有组恢复，不改变成功组合顺序、召唤目标或操作阶段。

## 实现与版本边界

- `ScriptDirectory` 支持不可变的文件名 → 内容覆盖，在 native 读路径优先使用；未覆盖文件仍按目录顺序加载。
  `find()` 仍返回磁盘路径，`read()` 返回有效内容；调用方修改原字典不会影响已创建对象。没有每次读脚本的 Python 回调。
- `default_scripts()` 只对已审查的 `proc_synchro.lua` SHA-256 启用覆盖；未知版本原样加载。
  直接构造不带覆盖的 `ScriptDirectory` 可复现原始错误，现有 traceback 回归已显式使用它。
- 基点 `8afe1f5827ac5d82d4da53aa860417780cfe01d5`；环境 `md-2026-09`，指纹
  `65ca28f79233e73d540a23c2046718ef42749ae883baa920216b94365b3e42d9`。
- CardScripts `1e28935380407e8f4e5a7edf50875b4b38fe56a6`；core `122e0d091a0f399221a4510cc98a406ae485905f`；
  BabelCDB `52d5221df32c943877c8a63639edcc4f3d918916`。
- 原脚本 SHA-256 `cacd92d496ab653e6e5f304cb2b38b831e4f6d41ab1e842ffab909a0efb1543d`；
  有效脚本 `edb6a0efa4df301aaff17a05ee97fccf7655d2bc466d1ff5cf6a60659eaba4f0`。

## 验证结果与局限

| 验证 | 结果 |
|---|---|
| 语义与真实失败回归 | 14 passed：10 种素材边界、版本回退、2 条真实失败前缀、1 条完整 golden 回放 |
| 完整本地 presubmit | **1,379 passed / 8 skipped**，294.92s；格式与 lint 通过 |
| 11 条旧失败高预算轨迹 | 完整消息、决策、应答、终局全部一致；两侧都用 5× 预算，融合未修故仍需要高预算 |
| 512 次新随机策略执行 | 510 次完整一致，2 次原脚本默认预算失败而修复后完成；无其他差异 |
| 原卡死于 decision 333 的策略对局 | 默认预算完成，436 decisions、12 turns、winner 1、LP −1100/7200；前 333 个动作与应答全部匹配旧记录 |

随机实验使用 seed=2026100404、128 个抽样牌组配对、四重角色安排、max_decisions=4000。
两边随机策略共用同一动作随机流，所以四重角色中有物理开局重复：**512 次执行是 256 个不同引擎开局各检查两次**，
不能称为 512 个独立新开局。两次恢复来自同一开局 `random-88-0-0` / `random-88-1-0`，
原脚本在 decision 158 超限，修复后完成；峰值从 100,038 降到 450 千条指令。
合计 523 次对照执行：521 次完整一致，2 次上述原脚本失败恢复。

已知完整 454-action 同调轨迹有原脚本 5× 预算的成功参照：920 个消息缓冲区及全部应答逐字节一致。
本轮原脚本 43.05s、峰值 296,589 千条指令；修复后 0.358s、峰值 202 千条。
512 次随机执行中共同完成的 510 次累计耗时 200.89s → 177.61s；机器另有训练负载，
这是诊断计时，不是严格吞吐基准，也不保证所有对局都更快。

decision 333 那局原脚本此前给 20× 预算仍失败。本轮用原 manifest 的 base–league / pairing 73 / game 0
冻结策略在 CPU 上续行，前缀和旧 GPU 记录一致；**失败点之后没有原脚本的完整成功参照**，不宣称该段已做完整等价验证。
14 项回归中的完整 golden 只包含有成功参照的 454-action 局；另一个 fixture 保存 333-action 失败前缀。

8 个跳过项为 snapshot 配置 1、显式启用网络测试 2、ROCm GPU 测试 5。
GPU 当时正被另一个 worktree 的训练使用，本轮 CPU 验证没有抢占它；不把旧分支 GPU 测试算作本轮通过。

## 复现与证据

随代码提交 `tests/test_synchro_pruning.py`、`tests/data/synchro-search-replays.json`，
无需模型检查点即可检查合法素材、原始失败前缀和完整消息/应答 digest。digest 对每段缓冲区先写 uint64 little-endian 长度再写内容。
安装本项目并构建当前 native 扩展后运行：

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_synchro_pruning.py
bash tools/presubmit.sh --test
```

本机完整产物在 `out/research/synchro-pruning-2026-10-04/`（不入 Git）：

- `manifest.json`：源码、实际 native binary、CDB、原始/有效脚本与环境指纹。
- `validate_messages.py`、`synchro-message-parity.json`、`parity.log`：523 次对照的执行脚本与逐局结果。
- `build_fixture.py`、`fixture.log`：原始 5× 完整参照及修复后独立验证。
- `old-prototype-counterexample.json`：旧原型误拒合法组合的实际 Lua 断言。
- `recovered-deep.json`、`recovered-deep-validation.json`：436 步完整策略回放及 333 步前缀核对。
- `guard-tests.log`、`presubmit.log`：测试证据；`native-build/`：独立构建，未替换其它 worktree 的扩展。

## 下一步

优先处理 #169 的融合组合搜索，先从既有失败回放定位可保持合法素材/消息顺序的优化。
上轮 6,144 局回应策略 panel 的 17 次错误经 traceback 复核全部为融合，**本次同调修复不会消除这些错误**。
保留历史 panel 原始统计及共同排除规则，不把预算恢复或搜索提速当作策略强度提升。
回应老师仍未通过扩训门槛，不启动其 PPO 蒸馏；新的训练容量/表示实验仍需独立完整对局验收。
