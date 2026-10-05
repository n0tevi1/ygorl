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
