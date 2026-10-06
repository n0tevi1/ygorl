# 规则引擎：需求与裁决

> 属于 [ygorl 设计文档](README.md)。状态：已评审通过（2026-09-22）。


## 2.1 我们需要引擎做到什么

| # | 需求 | 原因 |
|---|------|------|
| R1 | 完整、持续更新的卡片效果脚本（~13.5k 张） | 规则不是数据而是代码，必须复用社区脚本库 |
| R2 | MR5 规则 + 可切换的规则变体（TCG 专属裁定、Speed/Rush/GOAT） | MD 优先但要适用其它格式 |
| R3 | 同一进程内并行跑成百上千局，每局状态完全隔离 | 单机向量化环境是吞吐唯一来源 |
| R4 | 给定种子 + 应答序列，字节级可复现 | 回放、配对种子评估、回归测试、bug 复现 |
| R5 | 引擎枚举全部合法应答（`MSG_SELECT_*`），且能查询任意卡/区域信息 | 动作即输入 + 观测编码都依赖它 |
| R6 | 已有 RL 集成先例可参考 | 降低消息→动作映射的设计风险 |
| R7 | 卡片数据库与禁限表可机器读取 | 组牌约束、环境版本化 |
| R8 | 构建简单，Linux 下可作为静态库嵌入 pybind11 模块 | 单机、CI |

## 2.2 两个核心对照

| 需求 | Fluorohydride/ygopro-core（MyCard 系） | edo9300/ygopro-core（EDOPro / Project Ignis） |
|------|------|------|
| R1 脚本 | ygopro-scripts ~13.5k，9 月 1 日更新 | ProjectIgnis/CardScripts `official/` 13,461 + rush/speed/pre-errata，9 月 21 日更新 |
| R2 规则变体 | MR3/4/5 + 7 个 flag，无 TCG 专属 flag、无格式预设 | 37 个细粒度 flag，预设 `DUEL_MODE_MR1..MR5 / SPEED / RUSH / GOAT`，含 `TCG_SEGOC_NONPUBLIC`、`TCG_FAST_EFFECT_IGNITION` 等；MD = MR5 + MD 禁限表 |
| R3 并行隔离 | 每局有独立 `pduel` 句柄，但 card/script/message 回调是**全局**的（ygo-agent 在 envpool 线程池里跑通过，说明可行但靠约定） | 每局 `OCG_DuelOptions` 自带回调 + payload，**按设计无全局状态**，线程池天然安全 |
| R4 确定性 | mt19937 + seed_seq，`create_duel_v2(seed[8])`，确定 | Xoshiro256** 4×64bit 种子，ygo-harness 实测「同种子 → 字节一致消息流」 |
| R5 消息/查询 | `process/get_message/set_responsei/b`，`query_*`；消息集完整 | `OCG_DuelProcess/GetMessage/SetResponse`，`OCG_DuelQuery*`；消息集更新（如 `MSG_REMOVE_CARDS`），`OCG_VERSION` 显式版本化 |
| R6 RL 先例 | **ygo-agent**（最成熟：编码方案、1 亿局训练） | cjiang1209/yugioh-agent（ctypes + PPO，2026-09 活跃）、ygo-harness、YGO-Bench；ygo-agent 的 edopro 后端未维护 |
| R7 数据 | MyCard cards.cdb；禁限表需自取 | BabelCDB（含 rush/prerelease）+ LFLists（TCG/OCG/GOAT/Speed/Rush 的 `.lflist.conf`）；**无 MD 表** |
| R8 构建 | 源码 + Lua，简单 | 源码 + Lua 5.4（核心自带子模块），meson/premake，C++17，简单 |

两者 API 形状几乎同构（创建 → 加卡 → 循环 process/get_message/set_response），脚本互不兼容（edo9300 明确声明与非其派生的分支不兼容）。**MD 禁限表两边都没有**，都得自己维护（来源：masterduelmeta 非官方 JSON + 手工校对）。

## 2.3 裁决

**选 edo9300/ygopro-core + ProjectIgnis CardScripts / BabelCDB / LFLists。** 决定性因素是 R2 和 R3：多格式是明确需求，只有它有格式预设与 TCG 专属裁定；每局独立回调让 C++ 线程池不依赖全局约定。R1/R4/R5/R7 它同样不弱或更强。Fluorohydride 唯一的实质优势是 R6（ygo-agent 先例），但 ygo-agent 的价值在**编码方案与训练循环**，这部分与核心无关，可以移植。

缓解措施：`ygorl.engine` 定义一个极薄的 `CoreBackend` 接口（`create / process / get_message / set_response / query`），消息解码器按核心分文件。若将来因许可证或社区原因要换回 Fluorohydride，只需重写绑定与解码器（消息负载格式略有差异：edo9300 的位置信息是 `{u8,u8,u32,u32}`，Fluorohydride 是 4 个 u8），上层不动。

## 2.4 版本限定的同调搜索剪枝（2026-10-04，#169）

默认 100,000 千条脚本指令预算保持不变。对已审查的 CardScripts `proc_synchro.lua` 内容哈希提供内存覆盖，
不修改子模块或写入共享脚本目录；未识别的版本原样加载。`ScriptDirectory` 可接收只读的文件名→内容覆盖，Python 单局和 C++ 批量主机共用。
直接构造不带覆盖的 `ScriptDirectory` 保留原脚本，供语义对照与原始失败诊断。

剪枝仅作用于 `Synchro.CheckP42` 的非调整素材递归：调整组固定，每次递归只增加一个非调整素材。
当当前素材等级和已超过目标，且所有当前/剩余候选都是固定正等级（1–65535）、同调等级等于普通等级时，增加素材不可能恢复等式。
更大的数值会被核心求和接口解释为两个 16-bit 可选等级，即使没有特殊等级效果也必须退回原搜索。
在这一条件下返回 false，并仍执行原来的组恢复逻辑。不能调换成功组合的搜索顺序，也不改目标/操作阶段。

保护边界：存在特殊同调等级、custom material、`EFFECT_SYNCHRO_CHECK`、素材限制、等级修改效果，或 `req2` / `reqm` /
`Synchro.CheckAdditional` 时保留原搜索。对 `EFFECT_HAND_SYNCHRO + EFFECT_SYNCHRO_CHECK`，只允许本版本
`Synchro.CreateHandMaterialEffect` 自己创建的 `synchktg` 闭包；用 Lua 内私有弱键表记录函数身份，未知/替换的回调一律退回。
该闭包仅检查效果标签与筛选/返回排除组，不修改卡片属性或已选择组；其之后的 `CheckHand` 也仅检查标签。
因此这些回调不能使已超目标的固定正等级和降低。普通查询接口仍遵循引擎对只读属性/效果查询的约定。

验收包含：两个真实默认预算失败前缀、可完成高预算参照的完整消息/应答/终局一致性、新随机对局、
特殊/双值同调等级、custom material、未选择素材上的自定义回调/效果，以及组约束。未知回调改变等级的构造必须仍保留合法组合。
原始失败 traceback 测试显式使用未覆盖脚本。该同调修复本身不解决融合搜索（后续见 §2.5），也不重写历史 RL 结果。

## 2.5 融合完整素材组的约束检查顺序（2026-10-04，#169）

`CheckMixGoal` 先对完整素材组检查 `Fusion.CheckAdditional`，再枚举该组内的素材角色排列和检查可用区域。
月光融合中含两张牌组/额外牌组素材的完整组合因此无需进入昂贵的角色排列；搜索树与成功组合顺序不改，预算不变。
变换只应用于内容哈希匹配的 `proc_fusion.lua`。沿用融合条件函数的只读判定约定，不缓存结果、不修改素材组。

**不能对未完成素材组做通用剪枝。** 上游虽有单调性注释，可重复素材路径也据此提前检查，但实际卡片并非都遵守：
Fusion Destiny（52947044）的 `fcheck` 要求组中至少一个 Destiny HERO，在不完整组上可能 false、加卡后变 true。
本轮随机回放证实通用提前剪枝会从第 574 步的合法候选中删除 Neos 与 Masked HERO Furnace；该原型不得启用。
最终检查顺序优化仅在原来的完整组判定位置执行，保留这种非单调约束。
验收须包含此真实反例、月光预算失败的完整回放、非单调/数量约束、必选素材、替代素材、组恢复与正常对局消息一致。

## 2.6 可重复融合素材的失败状态复用（2026-10-06）

`SelectMixRep` 已选择的素材先分配固定角色和可重复角色，再检查是否还需补充素材。
同一已选组、剩余角色和替代素材状态会因选取顺序不同被重复访问；两个只能担当同一个固定角色的素材，
夹杂大量重复角色素材时，会产生阶乘级失败排列。简单的总数量上限不能排除这种情况。

在一次 `SelectMixRep` 调用内，对已分配组、剩余必选素材、剩余角色的有序函数身份、
剩余 min/max 和当前 sub 标志完全相同的**失败状态**做局部记忆。
tp、mg、sg、fc、sub2、contact、sumtype、chkf、fun1 在此调用内固定；g 只含 sg 中的卡。
卡片以该调用的 sg 身份列表编码，不用卡号合并不同副本。沿用条件回调只读、结果由其参数及不变对局状态决定的约定；
允许条件依赖当前素材组，不把单张素材的资格当作与组无关。不支持条件用调用次数或随机数改变合法性。

保持原来的深度优先遍历、首个成功返回和组恢复；只复用完整探索过的 false，不缓存成功，不提前检查额外约束。
至少 6 个已选素材才启用，较小组沿用原搜索；每次调用最多缓存 4,096 个失败状态，满后继续原搜索，
不拒绝素材；每次候选、每局及嵌套搜索均独立。
内容哈希限定和默认脚本指令预算不变，子模块不改。未知上游版本照旧不打补丁。
验收覆盖真实训练失败的动作/应答复现、高预算原版与默认预算修复的完整消息一致、
组相关回调、角色/替代状态、必选素材、同卡号不同副本、回调嵌套及缓存容量回退。
