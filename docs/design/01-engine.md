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
