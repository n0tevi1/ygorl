# ygorl 设计文档

状态：已评审通过（2026-09-22）。工程拆解、排期与验收见 [../eng-plan.md](../eng-plan.md)。

设计文档按主题拆成多个文件。文内的 `§N` 编号沿用整体设计的章节号（§1 目标 … §8 风险），跨文件引用时按下表查找。

| 文件 | 原章节 | 内容 |
|------|--------|------|
| [00-overview.md](00-overview.md) | §1 | 三个目标（对局、组牌 + off-meta、对手预测）、内外两层结构、已确认约束 |
| [01-engine.md](01-engine.md) | §2 | 引擎需求 R1–R8、两个核心对照、裁决（edo9300 + Project Ignis） |
| [02-challenges.md](02-challenges.md) | §3 | 游戏王对 RL 的 12 个独特挑战 C1–C12 与应对 |
| [03-play-policy.md](03-play-policy.md) | §4.2 | 对局策略：ygo-agent 起点与 2023–2026 改进 I1–I9 |
| [04-opponent-model.md](04-opponent-model.md) | §4.5 | 对手预测：预测目标、损失、「不响应」证据、meta 先验、三通道消费 |
| [05-deck-building.md](05-deck-building.md) | §4.3–4.4 | 组牌通用组件、off-meta 发现（脚本挖掘协同图、QD + DNS、三元组报告） |
| [06-architecture.md](06-architecture.md) | §5 | 架构图、仓库布局、关键设计决定 |
| [07-building-blocks.md](07-building-blocks.md) | §4.1 | 引擎、脚本、数据库、API、求解器等可复用资源清单 |
| [08-risks.md](08-risks.md) | §8 | 风险与备选 |
| [09-activation-health.md](09-activation-health.md) | 补充 | 发动与结算的合法性、脚本故障边界、监控接手 |

阅读顺序建议：00 → 01 → 02 → 06，再按兴趣读 03 / 04 / 05。
