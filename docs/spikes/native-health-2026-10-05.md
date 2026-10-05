# 原生 RL 环境丢失脚本错误（2026-10-05）

## 已确认根因

`HostDuel::advance` 调用 `core_->pop_logs()` 后直接丢弃日志；HostPool 的结果也不携带脚本错误。
因此 Python 的健康回放检查不能证明实际 RL 环境健康。真实复现：保留 Ash 原有初始化，随后抛 Lua error，
旧 EncodedVecEnv 仍以正常 win 结束（turn48、972决策），与无错误对照的完整响应及结果完全一样。
同一脚本源的 Python 对局确实记录6条 Lua error。旧训练按 reason=win 发终局奖励，错误不可见。
这证明管道可漏报，**不证明此前每次训练已经受影响**；缺失的旧日志不能事后恢复。

## 修复契约（代码之前）

原生 HostDuel 消费日志时保留 OCG_LOG_TYPE_ERROR 的原文，立即结束为 reason=error、无 winner，
不继续暴露错误后的观测或胜负。普通脚本/debug日志不作为错误。
HostDuel 与 HostPool 的结果均包含 script_errors；HostPool 同时补齐已有 retries/unknown_messages 计数。
现有 rollout 把 reason=error 标为 truncated，不发终局胜负奖励；回归检查实际日志与该训练结果契约。
完整游戏日志记录这些健康字段，使后续实验可审计；历史不含字段的报告不可当成零错误证据。

另建原生二进制及 build 目录，不覆盖冻结实验使用的旧 binary。
验证真实初始化和运行中 Lua error、普通 Debug.Message，以及无错误对局与旧响应/编码的一致性。
容量 BC 参数不受此代码路径影响，旧自由展开重评仍用 Python健康检查。
128×16 PPO 代价测量推迟至修复后，四组统一使用新二进制并记录身份；既定数据、种子、更新预算不变。
这是成本诊断，不将6次更新作为强度证据。

## 已完成修复与验证

实现 `666fc1b`，与已合并PR #191对齐后测试提交 **685a7c6**。
新模块SHA256 `5f64e558cf55b40be95bfbdc1dc146d54d118328a424ee4f1d60f17f26dee277`，
位于 `out/research/native-health-2026-10-05/build/`，旧模块不覆盖。
真实初始化/运行中错误、普通debug日志、HostDuel/HostPool两条路径，以及实际rollout零奖励截断均通过。
定向 **51 passed，20.30s**；实际错误rollout/错误日志 **2 passed，2.52s**。
完整ROCm presubmit **1,443 passed /3 skipped，301.54s**。
三项跳过为snapshots=False分支和两个opt-in联网测试。初次全套只有worktree中误写的OCG fixture软链接失败；
修正并恢复全部9份上游禁限表后完整重跑，失败日志保留；不是改测试期望绕过实现问题。

原复现对照重跑：干净游戏完整响应/结果与旧模块相同；注入错误立即返回error、winner=None、
0决策，并保留6条Lua原文。审计脚本最初把Python的LP tuple与JSON list直接比较导致类型误报，
保留失败日志，统一JSON表示后对所有原字段逐项严格比较通过。
完整游戏与errors日志均保留脚本错误/重试/未知消息字段；未提供这些字段的旧记录用null，不能当零。
原生现有重试/未知消息计数现在可见，本修复未改变其处理规则；成本审计将显式要求计数为零。

证据与脚本封存在 `out/research/native-health-2026-10-05/validation.json`；build产物只封实际模块和构建日志，
不将临时中间对象冒充独立验证。四组既定成本测量在此源和新模块上执行；长期训练仍待教师数据问题修复。
