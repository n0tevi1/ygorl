# ygorl 工程计划

> 对应设计文档：[design/README.md](design/README.md)。本文把设计拆成可独立验收的任务（ticket），给出依赖、验收标准与推进顺序。每个任务对应一个 GitHub issue（#1–#51），标题前缀 `[M0]`…`[M6]` 标记里程碑；issue 编号在下表「Issue」列。

## 1. 里程碑总览

| 里程碑 | 目标 | 交付物 | 出口条件 |
|--------|------|--------|---------|
| M0 骨架 | 能构建、能测、有环境配置规范 | pyproject / CMake / 子模块 / CI / `Environment` | CI 绿，`import ygorl` 成功 |
| M1 引擎绑定 | Python 能完整驱动一局决斗 | `ygorl.engine`、`ygorl.cards` | 随机 vs 随机 1,000 局 × 10 套牌零异常；确定性重放通过 |
| M2 向量化环境 | 批量、可编码、可课程、可快照 | `ygorl.env`、`csrc/` | 16 核吞吐基准落地；编码交叉校验通过 |
| M3 基线与评估 | 有对手、有裁判 | `ygorl.agents`、`ygorl.eval`、CLI | Greedy 显著胜 Random；对局矩阵与 Nash 可复现 |
| M4 策略训练 | 会玩，带对手预测 | `ygorl.solver`、`ygorl.nets`、`ygorl.train` | 单牌组对稳定胜 Greedy；10 套牌池不塌陷；信念头校准报告 |
| M5 数据与组牌 | 能从卡池搜出牌组 | `ygorl.data`、`ygorl.build` | 小格式端到端跑通；协同图召回 ≥ 80% meta 引擎包 |
| M6 元游戏闭环 | 产出 off-meta 报告，环境可升级 | 两阶段对手池、利用者微调、报告 | 至少一套 rogue 牌通过三元组验证 |

关键路径：M0 → M1 → M2 → M3 → M4b。M4a（求解器示范）依赖 T1.7 的 spike 结论。M5 的数据与协同图任务（T5.1–T5.4）只依赖 M1 的 cdb 加载，可与 M2–M4 并行。

## 2. 任务清单

「依赖」只列直接前置。验收标准是 issue 关闭的条件，与下文 [§4 验证方式](#4-验证方式) 一致。

### M0 骨架

| ID | 任务 | 依赖 | 验收标准 | Issue |
|----|------|------|---------|-------|
| T0.1 | 项目骨架：uv + `pyproject.toml`（scikit-build-core）+ CMake + pybind11，编译一个 hello 扩展 | — | `uv sync && uv run python -c "import ygorl._core"` 成功；`uv run pytest` 跑通空测试 | [#1](https://github.com/n0tevi1/ygorl/issues/1) |
| T0.2 | 子模块引入 edo9300/ygopro-core、ProjectIgnis/CardScripts、BabelCDB、LFLists；CMake 把核心（含其自带的 Lua 5.4 子模块）编成静态库并链接进扩展 | T0.1 | `OCG_GetVersion` 能从 Python 调到并返回 11.x | [#2](https://github.com/n0tevi1/ygorl/issues/2) |
| T0.3 | CI：GitHub Actions 在 Linux 上构建扩展并跑 pytest；缓存子模块与构建 | T0.2 | PR 上 CI 绿 | [#3](https://github.com/n0tevi1/ygorl/issues/3) |
| T0.4 | `Environment` 数据类与 `environments/<version>/` 目录规范（`pool.json`、`banlist.lflist.conf`、`meta/*.ydk`、`meta.json`、`artifacts/`）；加载/校验/版本号 | T0.1 | 单测覆盖加载与缺文件报错；`docs/environments.md` | [#4](https://github.com/n0tevi1/ygorl/issues/4) |

### M1 引擎绑定

| ID | 任务 | 依赖 | 验收标准 | Issue |
|----|------|------|---------|-------|
| T1.1 | pybind11 封装 `OCG_*`（`CoreBackend`）：创建/加卡/加载脚本/开始/process/get_message/set_response/query/销毁；card reader 与 script reader 回调走 Python 侧注册的 loader | T0.2 | 一局能推进到 `OCG_DUEL_STATUS_END`；GIL 在 process 期间释放 | [#5](https://github.com/n0tevi1/ygorl/issues/5) |
| T1.2 | `cards.cdb` 加载与位标志解码（type/attribute/race/level+scale/link marker/setcode×4），含效果串 `texts.str1..16`；卡片词表与快照 | T0.2 | 单测：抽样 20 张卡的解码字段与 YGOPRODECK 一致；多效果卡的 str 索引与脚本 `SetDescription` 对得上（抽样 50 张人工核对） | [#6](https://github.com/n0tevi1/ygorl/issues/6) |
| T1.3 | `.lflist.conf` 与 `.ydk` 解析；牌组合法性校验（禁限、3 张、40–60 主、≤15 额外、卡池归属） | T1.2 | 单测覆盖全部规则；错误信息可读 | [#7](https://github.com/n0tevi1/ygorl/issues/7) |
| T1.4 | 全部 `MSG_*` 反序列化为类型化消息（含 `SELECT_COUNTER / SELECT_DISFIELD / SORT_CARD / ROCK_PAPER_SCISSORS / ANNOUNCE_*`），**遇未知消息不抛异常**，记录并降级 | T1.1 | 消息表与 `ocgapi_constants.h` 逐项核对；未知消息路径有测试 | [#8](https://github.com/n0tevi1/ygorl/issues/8) |
| T1.5 | 单局 API `Duel(seed, env, deck_a, deck_b).run(agent_a, agent_b)` + Random agent；`tests/decks/` 放 10 套 MD meta 牌组 | T1.3, T1.4 | 随机 vs 随机 1,000 局 × 10 套牌零未处理消息、零异常 | [#9](https://github.com/n0tevi1/ygorl/issues/9) |
| T1.6 | 确定性与重放：同种子 + 同应答 → 消息字节流逐字节一致；应答日志重放到相同终局 | T1.5 | 单测 + CI | [#10](https://github.com/n0tevi1/ygorl/issues/10) |
| T1.7 | Spike：核实 ygo-combo-solver 的核心版本与 edo9300 兼容性，决定「封装二进制」还是「移植搜索到我们的核心」；评估其 arena 快照移植难度 | T0.2 | 一页结论写入 `docs/spikes/combo-solver.md`，含决定与工作量估计 | [#11](https://github.com/n0tevi1/ygorl/issues/11) |
| T1.8 | 回放格式：环境版本 + 种子 + 双方牌组 + 规则 flag + 应答日志（可选候选列表与策略概率）；存储/加载 API；导出 EDOPro 可打开的 `.yrpX` | T1.6 | 任意一局可记录并从文件完整重放到相同终局；导出的 `.yrpX` 在 EDOPro 中可回看（人工验证一次）；跨环境版本加载给出明确错误 | [#12](https://github.com/n0tevi1/ygorl/issues/12) |

### M2 向量化环境

| ID | 任务 | 依赖 | 验收标准 | Issue |
|----|------|------|---------|-------|
| T2.1 | C++ `DuelPool` 线程池：异步 `send(actions) / recv() -> batch`，每线程独立核心句柄；Python `VecDuelEnv` 与单局 `DuelEnv` | T1.5 | 多线程 1,000 局结果与单线程逐局一致；无数据竞争（TSAN 跑一次） | [#13](https://github.com/n0tevi1/ygorl/issues/13) |
| T2.2 | 观测编码（C++）：卡片 token 表 `[N_cards, F]`、全局向量、候选动作表 `[max_options, A]`（含效果 id）+ 掩码；Python 参考编码器；`docs/encoding.md` | T2.1, T1.2 | C++ 与 Python 参考编码在 10k 决策点逐元素一致 | [#14](https://github.com/n0tevi1/ygorl/issues/14) |
| T2.3 | 多选拆步与可行集：`SELECT_CARD/UNSELECT_CARD/TRIBUTE/SUM` 逐张决策 + Finish；C++ 侧计算剩余可行集（求和约束） | T2.2 | 超量/同调/仪式/连接素材场景：环境可行集 = 引擎接受集合 | [#15](https://github.com/n0tevi1/ygorl/issues/15) |
| T2.4 | 事件 token 流：全部 `MSG_*` 事件编码为 token；**响应窗口 + 放弃** token（触发类型、对手场面、LP、回合） | T2.2 | 人工构造「可抛灰流丽但放弃」场景 token 正确出现；序列长度可配 | [#16](https://github.com/n0tevi1/ygorl/issues/16) |
| T2.5 | 训练态真值输出（对手手牌/牌堆/盖卡张量），推理态关闭；接口保证 actor 拿不到 | T2.2 | 单测：推理态张量为空；训练态与引擎查询一致 | [#17](https://github.com/n0tevi1/ygorl/issues/17) |
| T2.6 | 课程开关与开局配平：单人展开（对手自动放弃响应）/ 仅手坑 / 完整；先后攻配平；「增广开局」标志位 | T2.1 | 三种模式各跑 100 局无异常；标志位出现在观测里 | [#18](https://github.com/n0tevi1/ygorl/issues/18) |
| T2.7 | 吞吐基准：决策/秒 vs 线程数，写入 `docs/benchmarks.md` | T2.2 | 16 核实测数字入库（目标 ≥ 1 万决策/秒，以实测为准） | [#19](https://github.com/n0tevi1/ygorl/issues/19) |
| T2.8 | arena 快照 `snapshot()/restore()`（移植写监视 arena）；受阻则退回「从种子重放」实现中局开局，接口不变 | T1.7, T2.1 | restore 后继续对局与重放逐字节一致；恢复耗时 < 重放的 1/5（或记录退回原因） | [#20](https://github.com/n0tevi1/ygorl/issues/20) |
| T2.9 | 分支探索 API：`fork(replay, t)` 从任意决策点分叉，对同一决策点尝试多个候选并用给定策略 rollout，比较结果；首版用确定性「重放到 t」实现，T2.8 落地后切到 `snapshot/restore`，接口不变。用途：反事实分析/调试、中局开局的数据来源、后期 PIMC 搜索基础。限制：保留同一隐藏状态（上帝视角分支），对手未知信息的重采样（确定化）留作后续 | T1.8, T2.1 | 从任意 t 分叉并按原应答继续，终局与原回放一致；CLI `ygorl branch <replay> --at t --try all --policy <agent>` 输出各候选 rollout 结果表；文档说明限制 | [#21](https://github.com/n0tevi1/ygorl/issues/21) |

M2 遗留事项（验收之外发现、需要在后续任务前解决）：

- **超过 128 个合法动作的决策**（T2.3 备注）：观测只编码前 128 个动作，主要影响 ANNOUNCE_CARD（可宣言卡名可达上千个），第 128 个之后的动作 agent 选不到（[encoding.md](encoding.md)「截断」）。需在 T4b.1 训练 ANNOUNCE_CARD 之前定方案（拆步宣言或按先验排序）。
- **C++ 步进路径上的课程模式**（T2.6 备注）：课程模式只在 Python 主机实现，`EncodedVecEnv` 对非 `full` 规格报 `NotImplementedError`；T4d.1 的课程调度前需要把代答规则移植到 C++ `Tracker`。
- **T2.7 的 16 核数字**：目前只有 4 核实测（[benchmarks.md](benchmarks.md)）。

### M3 基线与评估

| ID | 任务 | 依赖 | 验收标准 | Issue |
|----|------|------|---------|-------|
| T3.1 | `Agent.act(obs) -> action` 协议；Random、Greedy（优先发动/召唤、高打低）、PolicyAgent 壳 | T2.2 | Greedy 在 10 套牌上都能完成对局 | [#22](https://github.com/n0tevi1/ygorl/issues/22) |
| T3.2 | `Arena`：配对种子（同起手互换先后）、置信区间、并行 | T3.1 | Greedy vs Random 2,000 局显著 > 50%，报告含 CI | [#23](https://github.com/n0tevi1/ygorl/issues/23) |
| T3.3 | 对局矩阵 + nashpy Nash / alpha-rank；结果写入 `environments/<v>/artifacts/` | T3.2 | 固定种子下矩阵与 Nash 可复现 | [#24](https://github.com/n0tevi1/ygorl/issues/24) |
| T3.4 | CLI：`ygorl duel`、`ygorl arena`、`ygorl matrix`、`ygorl replay` | T3.3, T1.8 | README 示例可运行 | [#25](https://github.com/n0tevi1/ygorl/issues/25) |
| T3.5 | 信念校准评估：ECE、各头准确率/AUC 的评估脚本（供 M4c 使用） | T2.5 | 对随机预测器给出合理基线数字 | [#26](https://github.com/n0tevi1/ygorl/issues/26) |

### M4 策略训练

| ID | 任务 | 依赖 | 验收标准 | Issue |
|----|------|------|---------|-------|
| T4a.1 | `ygorl.solver`：封装 ygo-combo-solver（或移植），批量求解 meta 牌组起手展开线（含 `--fire` 手坑变体），输出示范集格式 | T1.7 | 10 套牌 × 1,000 起手的示范集；每条线经新鲜重放验证 | [#27](https://github.com/n0tevi1/ygorl/issues/27) |
| T4a.2 | BC 预热 `train/bc.py`：示范集 → 策略网络 | T4a.1, T4b.1 | 固定起手线复现率报告；未见起手的场面质量（求解器评分） | [#28](https://github.com/n0tevi1/ygorl/issues/28) |
| T4b.1 | 网络：卡片/效果编码器（结构化 ⊕ 冻结卡文本 ⊕ 冻结效果文本 ⊕ 可关 ID 嵌入）、局面 Transformer、动作打分头（点积 + 掩码 softmax） | T2.3, T5.2 | 前向 shape 单测；对随机批次 loss 可回传 | [#29](https://github.com/n0tevi1/ygorl/issues/29) |
| T4b.2 | 历史模块：事件流因果 Transformer（GTrXL 门控）+ LSTM 基线，可切换 | T2.4, T4b.1 | 两者接口一致；单测 | [#30](https://github.com/n0tevi1/ygorl/issues/30) |
| T4b.3 | 特权 critic：历史条件的 Q 头（候选动作）+ V 头；VRPO Q-boosted 优势；GAE 作对照开关 | T2.5, T4b.2 | 优势估计单测（玩具 MDP 上与解析值一致） | [#31](https://github.com/n0tevi1/ygorl/issues/31) |
| T4b.4 | PPO 自博弈训练循环：裁剪目标、熵 0.05–0.2、KL 到慢速参考/BC 先验、当前策略自博弈 + 小历史快照池 + keep-best、牌组池采样、checkpoint/日志 | T4b.3, T3.2 | 玩具环境上收敛；YGO 上单牌组对跑通 1 小时无崩溃 | [#32](https://github.com/n0tevi1/ygorl/issues/32) |
| T4b.5 | 单牌组对实验与消融：数小时内 >95% 胜 Random、稳定胜 Greedy；消融 BC 预热 / Transformer vs LSTM / VRPO vs GAE | T4b.4, T4a.2 | 三组消融曲线入 `docs/experiments/` | [#33](https://github.com/n0tevi1/ygorl/issues/33) |
| T4c.1 | 信念头：牌组类型、剩余构成（0–3 份数多头）、手牌（≥1 + 角色位）、盖卡、被响应概率；损失掩码（已公开置 1、已现份数扣除）；meta 先验初始化 + HDT 式过滤特征 | T4b.2, T2.5 | 各头准确率/AUC 高于 HDT 过滤基线；ECE 报告 | [#34](https://github.com/n0tevi1/ygorl/issues/34) |
| T4c.2 | 三通道消费（detach 输入 / 辅助损失 / 特权 critic）消融；「放弃响应」场景后验 vs 朴素规则 | T4c.1, T4b.5 | 消融胜率差与校准报告入 `docs/experiments/` | [#35](https://github.com/n0tevi1/ygorl/issues/35) |
| T4d.1 | 多牌组池：10 套 MD meta + off-meta 噪声采样；课程三阶段调度；中局开局（带标志位） | T4b.5, T2.6, T2.8, T2.9 | 每套牌胜率不塌陷（对 Greedy ≥ 基线） | [#36](https://github.com/n0tevi1/ygorl/issues/36) |
| T4d.2 | 利用者训练 `train/exploiter.py`：平台期自动启动；对主 agent 的可利用性曲线 | T4d.1 | 可利用性随训练下降的曲线 | [#37](https://github.com/n0tevi1/ygorl/issues/37) |

M4 待议的消融（不改变主线设计，结果先入 `docs/experiments/`）：

- **MaxRL 对照**：主训练循环按设计用 PPO + VRPO（T4b.4）。在单人展开课程阶段（先攻回合是否到达目标场面，二元奖励），对比 MaxRL 式目标（最大化成功概率的对数，同一起点多次 rollout 估计梯度；起点用 T2.8 快照复制，求解器示范线可作成功样本）与 PPO/VRPO 的样本效率。T4b.4 的损失函数应可插拔以便接入。

### M5 数据与组牌

| ID | 任务 | 依赖 | 验收标准 | Issue |
|----|------|------|---------|-------|
| T5.1 | 数据抓取与快照：YGOPRODECK MD 卡池（限速 20 req/s）、masterduelmeta 卡表与占比、`md.lflist.conf` 生成脚本（含人工校对流程）、Yugipedia SMW 关系 | T0.4, T1.3 | `environments/md-<v>/` 一键生成；快照进仓库 | [#38](https://github.com/n0tevi1/ygorl/issues/38) |
| T5.2 | 卡文本 + 效果文本嵌入生成（sentence-transformers，离线 `.npy`），含词表对齐 | T1.2 | 全卡池覆盖；缺文本卡有占位策略 | [#39](https://github.com/n0tevi1/ygorl/issues/39) |
| T5.3 | 脚本挖掘协同图：解析 CardScripts 的 `SetCategory / IsSetCard / LOCATION_* / IsType|Race|Attribute|Level` → 卡密节点、检索/特召边；并联 Yugipedia 关系与文本相似度 | T1.2 | 用 meta 卡表做召回检验 ≥ 80% 引擎包（不参与构图） | [#40](https://github.com/n0tevi1/ygorl/issues/40) |
| T5.4 | 引擎包枚举：协同图连通子图、可达密度排序 | T5.3 | Top-50 包人工抽检合理；含跨系列包 | [#41](https://github.com/n0tevi1/ygorl/issues/41) |
| T5.5 | 基因型与约束：引擎包份数 + 泛用槽 + 额外卡组；禁限/3 张/40–60/≤15 硬约束；变异/交叉算子 | T5.4, T1.3 | 随机采样 10k 基因型全部合法 | [#42](https://github.com/n0tevi1/ygorl/issues/42) |
| T5.6 | 漏斗第一层：求解器起手分析（最佳线存在性、卡手率、抗手坑率）作为廉价 fitness 与描述符 | T4a.1, T5.5 | 与真实首回合卡手率一致（配对检验） | [#43](https://github.com/n0tevi1/ygorl/issues/43) |
| T5.7 | 代理模型：计数向量 + 卡文本嵌入均值 → 对 meta 池胜率 + 描述符；在线更新（DSA-ME） | T5.5, T3.3 | 留出集误差 < 10 个百分点 | [#44](https://github.com/n0tevi1/ygorl/issues/44) |
| T5.8 | pyribs MAP-Elites + Dominated Novelty Search + AURORA 学习型描述符；描述符 {先攻胜率, 后攻胜率, 手坑数, combo 长度, 卡手率}；meta 距离作过滤/报告 | T5.7 | 小格式上档案覆盖率与 QD-score 曲线 | [#45](https://github.com/n0tevi1/ygorl/issues/45) |
| T5.9 | LLM emitter：以卡池与协同图为输入提出候选引擎包（不做成对协同判定） | T5.5 | 提案经约束校验的通过率；与协同图包的重叠率 | [#46](https://github.com/n0tevi1/ygorl/issues/46) |
| T5.10 | 小格式端到端：3–4 个系列约 300 张卡；精英牌组真实胜率 > 随机合法牌组；人工检查 | T5.8, T4d.1 | 报告入 `docs/experiments/` | [#47](https://github.com/n0tevi1/ygorl/issues/47) |

### M6 元游戏闭环与 off-meta 报告

| ID | 任务 | 依赖 | 验收标准 | Issue |
|----|------|------|---------|-------|
| T6.1 | 两阶段对手池：精英加入训练牌组池与对手池；最佳应对验证（Haluska & Schmid） | T5.10, T4d.2 | 精英经最佳应对后的胜率变化报告 | [#48](https://github.com/n0tevi1/ygorl/issues/48) |
| T6.2 | 指定牌组的利用者微调（目标 1 的「给定一套牌」强化）；每候选微调预算与学习曲线 | T4d.2 | 微调后对 meta 池胜率提升曲线 | [#49](https://github.com/n0tevi1/ygorl/issues/49) |
| T6.3 | 环境版本升级流程：新禁限表 → 新版本 → 热启动权重/信念头/代理/档案；文档化 | T5.1, T4c.1, T5.7 | 热启动首轮评估用时显著低于冷启动 | [#50](https://github.com/n0tevi1/ygorl/issues/50) |
| T6.4 | 报告生成：对局矩阵、Nash 混合、精英档案、off-meta 三元组榜单（EV > 50% ∧ Nash 权重 > 0 ∧ meta 距离靠前），附展开线 | T6.1 | 至少一套 rogue 牌通过三元组；报告可读 | [#51](https://github.com/n0tevi1/ygorl/issues/51) |

## 3. 推进顺序

1. **第一阶段（M0–M1）**：单人串行，T0.1 → T0.2 → T0.3 / T0.4 → T1.1 → T1.2 / T1.4 → T1.3 → T1.5 → T1.6 → T1.8；T1.7 spike 与 T1.4 并行。
2. **第二阶段（M2–M3）**：T2.1 → T2.2 → T2.3 / T2.4 / T2.5 / T2.6 并行 → T2.7；T2.8 视 T1.7 结论，T2.9 先以重放实现再切快照；M3 在 T2.2 之后即可开始。
3. **第三阶段（M4）**：先 T4b.1–T4b.4 打通 PPO，再 T4a（示范）与 T4c（信念头）并行，最后 T4d。
4. **并行支线（M5 数据）**：T5.1–T5.4 在 M1 完成后即可开工，不阻塞主线；T5.5 之后依赖 M4 的策略。
5. **第四阶段（M5 后半 + M6）**。

不在首期范围：人类/EDOPro 客户端联机、决策时 PIMC 搜索、多机 actor/learner、Rush/Speed 格式实测。

## 4. 验证方式

- **M1**：同种子 + 同应答 → 消息字节流逐字节一致；随机 vs 随机 1,000 局 × 10 套牌零未处理消息、零异常；从应答日志重放到相同终局；效果串与脚本 `SetDescription` 索引对得上（抽样 50 张多效果卡人工核对）。
- **M2**：C++ 编码与 Python 参考编码在 10k 决策点逐元素一致；多选可行集与引擎接受集合一致（超量/同调/仪式/连接素材场景）；响应窗口 token 在人工构造的「可抛灰流丽但放弃」场景中正确出现；吞吐基准（决策/秒 vs 线程数，目标 16 核 ≥ 1 万，以实测写入 docs）；快照：`snapshot/restore` 后继续对局与重放结果逐字节一致，恢复耗时 < 重放的 1/5。
- **M3**：Greedy 对 Random 显著 > 50%（配对种子 2,000 局，置信区间）；矩阵与 Nash 可复现。
- **M4a**：BC 模型在示范起手上的线复现率；对未见起手的场面质量（求解器评分）。**M4b**：对 Random / Greedy / 历史快照胜率曲线；快照池内对每个对手 ≥ 40%；有 BC 预热 vs 无、Transformer vs LSTM、VRPO vs GAE 三组消融各报一条曲线。**M4c**：牌组类型 top-1、手牌头 AUC、盖卡头 top-3、ECE；三通道消融的胜率差；「放弃响应」场景下的后验相对朴素规则的改进。**M4d**：多牌组池上每套牌胜率不塌陷；利用者对主 agent 的可利用性随训练下降。
- **M5**：协同图能重新发现 ≥ 80% 的 meta 引擎包（用 meta 卡表做召回检验，但不用它训练）；小格式上精英牌组真实胜率 > 随机合法牌组，代理误差 < 10 个百分点；生成牌组通过约束校验；求解器起手分析的卡手率与真实首回合一致。
- **M6**：热启动后首轮评估用时显著低于冷启动；off-meta 榜单中至少有一套牌经最佳应对验证后仍满足三元组，并附可读的展开线与对 meta 各牌组的胜率。
- 全程：`pytest` 覆盖解析器、编码器、约束、信念头损失掩码；CI 构建扩展并跑单测；每个里程碑在 README 记录实测数字。

## 5. 工作约定

- 每个 ticket 一个分支、一个 PR；PR 描述引用 issue 并写明验收标准如何被满足。
- 任何数字结果（吞吐、胜率、校准）写入 `docs/benchmarks.md` 或 `docs/experiments/`，并注明环境版本与提交哈希。
- 设计变更先改 `docs/design/`，再改代码；设计文档是唯一权威。
- 术语与卡片主键约定见 [CLAUDE.md](../CLAUDE.md)。
