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
| T4b.6 | 训练信号与探索：按 KL 提前停、熵、固定对手 / 入池门槛 / PFSP、批大小与轮数、快赢压力与塑形（与设计 C3 冲突）、UPGO（与 I2 冲突）、中局开局、第 1 回合 GRPO、求解器数据上的 DPO；诊断指标 `adv/<kind>` | T4b.5, T4a.2 | 每项相对同一 b2s 起点、同一批 200 局的配对差入 `docs/benchmarks.md`；采用的项先改设计 | [#61](https://github.com/n0tevi1/ygorl/issues/61) |
| T4b.7 | 扩规模：GPU 学习器与批量推理、更多 CPU 核、actor / learner 解耦、多机 actor、ID 嵌入提速（见 [scaling.md](scaling.md)） | T4b.4 | 每个选项的实测吞吐与固定时间内对 Greedy 的胜率 | [#60](https://github.com/n0tevi1/ygorl/issues/60) |
| T4e.1 | 致死搜索：对局时兜底原型 → 搜索作老师蒸馏进策略（C++ 路径快照 / 分支）→ 用信念头采样的 PIMC | T2.8, T2.9, T4b.5, T4c.1 | 兜底原型在同一批 200 局上的配对差；漏掉的斩杀数 | [#62](https://github.com/n0tevi1/ygorl/issues/62) |
| T4b.8 | 对局强度：策略对局矩阵（agent 对 agent，Nash / alpha-rank，可增量加入检查点）+ 扩规模训练（#72–#77 提速、更大网络、≥ 2 万次更新）+ 可恢复多机执行 | T3.3, T4b.7 | 矩阵可复现且增量扩展与一次构建逐格相同；强度曲线与吞吐入 `docs/benchmarks.md` | [#83](https://github.com/n0tevi1/ygorl/issues/83) |
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

验收记录（M4 / M5，已达标的任务；细节与数字见各自文档）：

- **T4c.1**（[belief-heads.md](belief-heads.md)）：`ygorl.nets.belief` + `ygorl.env.belief_prior`；随机自博弈 1,500 局上五个头均优于 HDT 过滤基线（牌组类型 top-1 0.547 → 0.863、ECE 0.31 → 0.01；手牌 auc_macro 0.879 → 0.892；盖卡 top-3 0.528 → 0.621），ECE 全表见文档。`responded` 标签为「任何发动是否被连锁」（被无效的发动常不结算，无法仅限检索 / 特召）；延伸角色位暂用粗规则，待 meta 卡表提供。
- **T5.1**（[data.md](data.md)）：`ygorl env build md-<v>` 一键生成并通过 T0.4 校验；快照 `environments/md-2026-09`（卡池 13,858、20 套 meta 占 71.7%、未对应卡 0）。禁限表 2026-09-23 与 YGOPRODECK、Yugipedia 交叉核对，207 张三方一致，已校对（`tools/crosscheck_banlist.py`，记录见 `environments/md-2026-09/review/banlist-crosscheck.md`）。TCG / OCG 环境导出未做。
- **T5.3**（[synergy.md](synergy.md)）：真实 meta 召回（md-2026-09，31 个引擎包，`tools/make_meta_packages.py`）：纯脚本图 0.935（全部边）/ 0.871（仅检索 + 特召），达到 ≥ 80%；并联 Yugipedia `archetype_support`（`ygorl.build.relations`，可选）后 1.000 / 0.968。按卡组不拆引擎的严格变体 0.750。文本相似度（T5.2）尚未并联。
- **T5.6**（[funnel.md](funnel.md)）：10 套 × 12 手，漏斗卡手率 0.425 vs 真实首回合 0.408，McNemar p = 0.625；相对大预算求解器有约 4 个百分点的单向假卡手（5/120）；每套牌 45.6–115.9 求解器进程秒，全部在 120 秒预算内。预算研究（[funnel.md](funnel.md#预算研究每套-120-秒是否合理)）：48 手 × 20 秒参照 vs 每手 2/5/10 秒直接测；120 秒/套合理（16 核约 1.15 万套/天，早停后淘汰牌组 41–54 秒），但每手 5 秒的假卡手 +6.5 pp 集中在长 combo 牌组（snake_eye/purrely +17–19 pp，与 combo 长度 Spearman 0.91），10 秒降到 +2.9 pp、成本 101 秒/套；12 手抽样 SD 约 13 pp，卡手率维度约支持 3 箱；`fire_ms` 3 秒与 10 秒一致。默认已改为每手 10 秒（设计方确认）。
- **T4a.2**（[bc.md](bc.md)，**达标**）：`train/bc.py`、`train/heuristic_demos.py`、`nets/agent.py`、`env/observer.py`。训练起手线复现 137/158、目标达成 158/158（bc60）；held-out 目标达成 41/100（bc12，求解器 62）。只用求解器示范时对 Random 0.305：根因是示范只覆盖先攻第 1 回合，策略从不进战斗阶段（诊断见 bc.md「根因分析」，无流水线 bug）。补 Greedy 第 2 回合起的战斗阶段示范（1:1，b2）后对 Random 0.795（0.734–0.845）、对 Greedy 0.415，held-out 41/100 不变，满足验收；设计 I4 已相应修改（求解器第 1 回合示范 + 启发式战斗示范，KL 先验只在第 1 回合）。PPO 侧新增可选 `PPOConfig.kl_prior_turns`（BC 先验 KL 只加在第 1 回合状态，默认关闭）。
- **T4b.5 输入**（bc.md「补救实验」，10 套牌、每臂 200 次更新、单种子、小网络）：bc12 热启动与从零相同（对 Random 0.515）；全局 BC 先验锁死「不攻击」（0.305）；b2 热启动 → 对 Random 0.925、对 Greedy 0.635，但第 1 回合展开 37→18；第 1 回合先验 2.0 保住展开（31）、对 Greedy 0.485。已知问题：一次熵崩塌（自博弈陷入循环，未查）；成对发局间快照被逐出导致的 KeyError 已修（`SelfPlaySchedule.opponent`）。
- **T4b.5 / T4b.6 进展**（[scaling.md](scaling.md)、[benchmarks.md](benchmarks.md)「PPO 步长」「BC 热启动 + PPO」）：从零 PPO 在 4 核 CPU 上 3 小时对 Greedy 只到约 0.15——规模比 ygo-agent（约 100M 局、相当于 32 块 4090 训 5 天）少约 5 个数量级；重建的 b2s 热启动后 PPO（旧步长 P4o）对 Random 0.895、对 Greedy 0.490（b2s 本身 0.735 / 0.400）；新步长从热启动出发走太远（P4n 0.845 / 0.415，进战斗率 69%）；`--target-kl 0.01`（K）0.910 / 0.550，已设为默认。熵 0.01（KE）显著变差、固定 b2s 对手（KP）无帮助、按动作类型的优势（D0）未证实卡组耗尽假设，余下消融与 GRPO / DPO 见 #61。
- **T4e.1 阶段 A**（[benchmarks.md](benchmarks.md)「致死搜索原型」）：`lethal:<agent>` 包在 K 外面对 Random 0.910 → 0.925、对 Greedy 0.550 → 0.555，在噪声以内；K 的负局不是因为漏斩杀，只为斩杀做阶段 B 不值得，阶段 B 并入 C（PIMC）再议，见 #62。
- **`.yrpX` 导出**（T1.8 补充）：现按 EDOPro 方式 LZMA 压缩（1–3 MB → 10–20 KB）；T1.8 的人工回看需确认压缩文件在客户端里能打开（`--yrpx-uncompressed` 可对照）。

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

**2026-09-25 优先级调整（设计方确认）**：M4 的 P0 是**对局强度**（agent 打得聪明），见 [#83](https://github.com/n0tevi1/ygorl/issues/83)：
策略对局矩阵作强度尺子 → 更新阶段提速（#72–#77）→ 更大网络与长训练 → 可恢复的多机执行。新卡泛化不是 P0：约 1.5 万张卡的卡池可以全部学到，
新卡以后增量学习；卡片视角（文本 / 事实 / ID 丢弃）在当前规模下没有测出强度收益（[benchmarks.md](benchmarks.md)「卡片视角消融」），
进行中的实验跑完后不再扩展。

## 4. 验证方式

**2026-10-02 RL 接手**：对局强度仍为 P0；[接手记录](spikes/rl-handoff-2026-10-02.md) 集中记录实验完成状态、旧结论更正和下一步验收。
已完成 #83 的统一外部 panel、#169 失败回放，以及 #62/#61 的回应 pilot 和完整对局验证。
公开重排器局面 +2.59pp 未转化为外部对局收益（−1.28pp，CI [−3.35,+0.79]pp）；自对弈 +1.76pp 仍不确定。
**2026-10-04 已完成下一轮**：[部署采样对齐报告](spikes/aligned-response-2026-10-04.md)。22,784 次续局拟合后，以独立种子做 6,144 局四策略测试；
ridge−control +0.79pp [−1.36,+2.94]pp，bias −0.60pp，总是激活 −0.23pp；五个校正比较均不显著，17 次脚本错误共同排除 2/128 配对。
仍不扩大该老师或 survival，也不启动 PPO 蒸馏。
老师的后续候选必须冻结后重新独立验收；当前表示的容量/正则诊断见下文。
**2026-10-04 同调修复**：[回调边界与验证报告](spikes/synchro-pruning-2026-10-04.md)。已证明旧原型会误拒一个回调改变等级的合法组合；
新实现仅接受已审查的库内回调身份，未知回调/脚本版本退回原搜索，默认预算不变。两个真实卡死点恢复；
523 次对照执行中 521 次完整一致，另外两次为同一开局的原脚本超时、新脚本完成；完整测试 1,379 passed / 8 skipped。
**随后完成融合修复**：[完整组约束顺序报告](spikes/fusion-search-2026-10-04.md)。真实 Fusion Destiny 反例否定了部分组剪枝，
最终只调整完整组判定顺序；完整测试 1,399 passed / 3 skipped（含 GPU）。原 6,144 局全部重跑，17 错误全部恢复、
6,127 原成功结果一致，保留 128 配对；ridge +0.85pp [−1.27,+2.96]pp，仍不支持扩训。
**当前回应重排器支线收尾**：[容量与噪声诊断](spikes/response-capacity-2026-10-04.md)。11 根的 704 续局重跑后，
301 错误恢复、403 原成功结果一致。原集合与恢复集合分别做配对分组的 5×4 嵌套验证，后者 32 配对/324 根/388 标签，
内层选择容量/正则后的外层局面收益仅 +0.56pp；没有足够证据推出新候选。结束这版首回应公开 ridge 迭代，不扩训、不蒸馏。
完整回应决策、critic/优势方向与其它训练研究仍未验收，不能将这条支线收尾解释为整个 RL 问题已解决。

**2026-10-04 后续 RL 对照与修复**：真实终局 MC 配方在 matched pilot 中为 −4.56pp，未采用；完整游戏 lambda 的第三种子确认未过门槛
（[终局目标报告](spikes/terminal-mc-2026-10-04.md)）。随后补齐三种子的近似等行数固定段控制，在新的 21,504 局面板中，
完整游戏−固定段仅 **+0.15pp [-1.03,+1.33]**，两者相对冻结起点均未测出稳定收益，默认 PPO 不变
（[采样对照](spikes/collection-control-2026-10-04.md)）。发现的 BC 初始化种子、额外示范身份和错误局处理问题已由 #179 修复并合并；
旧示范 10/10 牌组不符合当前 MD 环境，后续先构建合法且环境绑定的正式示范集，再推进 #88/#92。
双牌组烟测只验证修复后的链路，未完成容量或 ≥2 万更新的验收。
**2026-10-05 教师覆盖诊断**：[20 牌组试验](spikes/md-bc-data-2026-10-05.md) 126/160 手解出，5,685 步回放零失败；
修复 Extra Deck 序号匹配后，真实洗牌 120/126 条保留目标。全牌组覆盖未通过：Lunalight 最终目标 3 秒 0/8、30 秒 2/8，
中间目标 8/8 仍缺少到最终场面的续接；Maliss / Tearlaments 有随机分支不迁移。下一步修教师续接与分支，正式数据冻结后再跑 #88/#92。
批次续跑现在校验输入 manifest、互斥写入并保留已有错误；见上述报告。
survival teacher 与先后手分网的 400-update 训练和旧 panel 已结束；目前不据单种子点估计修改默认训练配置，也不把共同进化的卡组增益当作策略提升。

**2026-10-07 战术纠错与保持（#218）**：[灰流丽首轮报告](spikes/ash-correction-results-2026-10-07.md)。
1024局新采集、95条训练偏好使封存局面的自我否定概率16.11%降至1.36–2.16%，但512局完整评估未证明强度提高。
768局事后隔离支持优先检查其他决策的策略变化；不替换主策略、不修改默认奖励或mask。
三种子pure/once/rehearsal接续PPO已完成并独立审计：288更新、4718592行、33693训练终局与1152评估局均健康。
旧测试自灰概率均值12.86/1.40/0.31%，旧整局胜率66.67/68.49/67.97%；局部保持改善，整体强度差异区间仍跨0。
后续[连锁关系诊断](spikes/chain-relation-diagnosis-2026-10-07.md)：917个窗口的原始归属完整，冻结表示可读出归属，
但不等于策略正确使用。只训练动作头的三种子纠错弱于全参数版本，384局诊断未证明强度提高。
真实rollout的PPO损失干预已完成98304行、356个健康终局、六组×五种更新，工件审计通过；
概率变化依赖数据、损失组合及KL早停步数，尚不能归因于单一项；不按旧面板选择或晋升模型。
[学习动作过滤器](spikes/learned-action-filter-design-2026-10-07.md)已实现独立风险头与有界软过滤组件；
[首轮旁路实验](spikes/action-filter-pilot-2026-10-07.md)完成512健康局、420唯一窗口、57条局部标签。
context头自灰均值27.49%→11.96%，但误惩罚1个好动作且己方样本仅20，gate未通过。
history表示事后对照减少该误判；冻结候选的新1024健康局取得74条局部标签，0好动作误标，己方Ash均值16.46%→2.24%。
己方仅26窗口、合理例外不足，gate仍未通过。
[范围对照与例外检查](spikes/action-filter-exceptions-2026-10-07.md)已完成：仅Ash惩罚消除了对pass的惩罚，
但两种范围都在Lycoris致死检索反例中误压救命自灰（4.06%→0.078%、2.85%→0.055%）。
候选明确失败，尚未执行过滤或接入PPO。旧标签会拒绝该反例，暴露的是推理范围超出已核验标签覆盖；
下一步做跨效果族的分支后果、关系输入和未知范围弃权监督，不按单卡添加硬禁令。
[前序决策检查](spikes/action-sequence-diagnosis-2026-10-07.md)完成14条原生分支：先清场再检索保留Ash并获得检索卡，
先检索再自灰仅为补救。原actor在己方低LP窗口立即检索仅0.27%，对方回合窗口却有34.61%，
被推进后又以97.15%概率放行；不能把条件概率当自然败率。下一步优先从真实策略前缀收集发动前错误，
分开训练/评估避免错误与事后补救，再做跨效果族和完整对局验收。
额外7条低LP脚本终局均健康：己方三条存活路线都赢，对方回合自灰和不发动都在下次抽卡被烧死；
故该对方回合反例仅验证延缓败北，不证明胜负价值改善，不能按即刻存活直接构造全局好动作标签。
[真实前缀筛查](spikes/natural-action-outcomes-2026-10-07.md)已完成：复用1024局526067决策，
4392个完整攻击区间中479个己方受损，8075个完整效果区间中19个受损；这些不是错误率。
20个代表局面的40条原生分支健康。3个效果例子pass仍受同样伤害，来源是更早CL1；另有自伤换特召收益。
12个攻击例子中2个目标原本里侧，3个明确为公开高DEF且双方不破坏；优先核验公开关系的后果预测，
保留未知收益和终局缺失，不自动生成坏动作标签或部署masker。
[公开战斗探针](spikes/battle-outcome-probe-2026-10-07.md)完成256局精确重放、1206可用样本、9个冻结表示小头。
250测试样本上普通算术方向100%、破坏99.2%；冻结表示头方向80.4–84.0%、伤害MAE343.7–364.1LP。
训练自伤仅43/558，测试自伤召回仍低；52条组合留出无算术例外，测试例外仅3条，不证明通用效果理解。
下一步控制类别平衡和显式数值关系，排除探针拟合限制后再决定策略监督；actor/PPO未修改。
[类别平衡与关系对照](spikes/battle-relation-controls-2026-10-07.md)已完成：4组×3种子、全新256局确认。
关系输入提高自伤召回、平衡进一步提高召回但增加误报；训练方向近100%，新局仍低，剩余主要是泛化差距。
去掉冻结表示的数值关系分支探索后，六头冻结再确认全新256局，召回86.49–89.19%、0/1026误报；
平衡版94.59–96.40%召回但1.56–3.61%误报。所有六头仍漏掉4/5自伤例外，
原生重放及脚本定位为攻击后/计算前增攻、互毁后墓地烧血。下一步分离数值计算和效果残差监督，
扩充效果覆盖及不确定性验收后再做策略辅助训练；未部署masker、未更新actor、无新胜率提升证据。
[数值与上下文残差](spikes/battle-effect-residual-2026-10-07.md)已完成额外768局开发重放、4方案×3种子和512全新局。
均匀残差无伤害方向收益，平衡残差三种子均验证选回零修正，直接拼接损害普通预测；不晋升。
训练68偏离中仅17条伤害变化、验证无非算术伤害，新面板又无效果额外自伤，现有覆盖不能验收旧漏报。
下一步优先分开数值变化/非算术伤害/破坏差异监督，补跨机制原生分支挑战集；
源checkpoint未启用正文/效果文本，任何语义特征方案仍需独立证明，不能假定直接开启即获益。
[分机制原生挑战](spikes/battle-mechanism-challenge-2026-10-07.md)已补16例/46分支，7对合法响应改变同声明输入的后果。
定位公开超量素材送墓烧血关系，以及自损500换取对方1200伤害的实例；不按自伤构造坏动作标签。
9567条分量监督已导出，但富集挑战包含旧训练例，不能作为独立泛化验收。
下一步按机制族/变体隔离，并显式加入相关卡片关系和后续动作条件；仍需新前缀挑战和完整对局验证。


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


### 2026-10-05：普通洗牌教师数据推进（#88 / #83）

[展开续接与直接生成对照](spikes/teacher-continuation-2026-10-05.md)：相同预算下，前缀续接和从头搜索
解出相同的 7/14 手；Lunalight 中间前缀未提高完整展开覆盖。普通洗牌搜索恢复原有 Maliss/Tearlaments
六条失败分支，新开发种子两牌组各 15/16，1,897 步独立回放零失败，BC 编码 956 样本可复现。
已修复无成功线时 BC 起始评估悄悄换回 pseudo-shuffle 的问题。下一步针对 Lunalight 验证完整成功线引导，
再做全牌组普通洗牌覆盖；当前没有新策略提升，不关闭 #88/#92。


### 2026-10-05：完整参考线引导未过开发门槛（#88 / #83）

[完整参考线对照](spikes/lunalight-guidance-2026-10-05.md)：新 16 起手、每手每臂 30 秒，
保留引导 7/16、清空引导 5/16，净增 2 手但未达到至少 8/16。780 步回放与 native 必要响应覆盖全过，
不扩大单参考线配方；下一步先核对融合材料/改名能力的目标分解，再冻结新种子同预算对照。
正式教师数据与容量/长训练依赖仍开放；不把两臂并集当单倍预算覆盖。


### 2026-10-05：材料分解对照完成，转向可达性诊断（#83 / #88）

[新种子材料引导](spikes/material-guidance-2026-10-05.md) 为 2/16 vs 普通搜索 2/16，零净收益，
296 步独立回放与 native 响应覆盖全过。旧 12 条完整线的材料审计没有误判不可达，但距离信号很稀疏。
记录了参考线对 novelty 默认值的隐含影响；新旧两轮比例不能当相同配方的重复实验。
下一步诊断 13 个共同失败起手的可达性：找到完整线证明搜索漏解，未找到仍记 unknown；
不把追加诊断混入已结束的开发 gate。正式容量和长训练依赖仍未完成。


### 2026-10-05：找回 6 条完整线并修复环境验证边界（#83 / #88）

[可达性诊断](spikes/reachability-2026-10-05.md) 对旧 13 个失败起手花 977.94 native 秒，
找到 6 条完整 Liger 线；带环境的 expected Replay 二次回放漏传 env 曾误报 unverified，
已修复并从原候选零搜索恢复，584 步独立验证通过。旧面板至少9/16可达，其余仍未知。
已冻结新种子20×16普通洗牌覆盖：Lunalight120秒/novelty0，其它牌组30秒/novelty12，
逐牌组验收后才冻结正式教师数据；不能把诊断追加结果回填旧 gate。

2026-10-05 全牌组普通洗牌复验：265/320 solved，54 unsolved、1 unverified；Lunalight 7/16、Odion 5/16，gate 仍失败。
已定位 Sky Striker 连锁中途误报与 Primite 精确卡名声明漏路，修复后正在单独重跑全部 55 个失败起手。
保留原面板和失败结论；先完成[根因调查](spikes/solver-failure-causes-2026-10-05.md)，再决定正式教师数据协议。

2026-10-05 教师动作边界修复：完整回放始终验真，BC 默认排除合成被动收尾；旧模式显式保留用于复现。
同一开发数据从 6,060 降为 5,466 样本，原生前缀逐元素不变；全套 1,430 passed / 3 skipped（实际 ROCm）。
55 手原生修复诊断仅恢复 Sky Striker 5，剩余 54 手未解出；继续有限声明列表修复及 49 手 unknown 的独立可达性诊断。
正式容量与长训练尚未完成，见 [BC 边界报告](spikes/bc-closing-boundary-2026-10-05.md)。

**2026-10-05 健康与教师根因更新**：PR #191修复教师/开局的Lua错误和提前截断漏验；PR #192修复原生RL丢弃Lua日志。
完整测试分别1,436/1,443 passed（均3 skip）。665条教师、800实际起手、960开局重新检查通过。
四组固定容量BC与128×16成本诊断完成，见 [容量报告](spikes/bounded-capacity-2026-10-05.md)：
自主目标16/22/13/17（/160），尚无可靠容量收益。进一步定位到Greedy取消攻击后，旧记录器仍保留被撤销的攻击标签；
训练628组/验证320组完整输入冲突。已完成原480局双模式重放及 [四容量教师修复复验](spikes/teacher-cancel-2026-10-05.md)：矛盾标签归零，
自主目标18/29/23/21（/160），完整对局1,280局健康无上限，但Greedy无可信改进。选择64×1对照与128×2候选，
先做可观测短程PPO再推进长跑。剩余可达性经两套完整树核对为273可达/16限域解释/31unknown；
旧覆盖gate仍失败，#83/#92继续开放；#88的容量比较已完成，结论是未证实容量强度增益。

2026-10-05：[#90检查点登记与矩阵健康修复](spikes/training-registration-2026-10-05.md) 已实现，
最终实际ROCm1,460 passed/3 skipped。独立曲线可恢复且不重复已完成cell；异常win不再进入矩阵胜率。
[32更新双配置PPO诊断](spikes/registered-ppo-pilot-2026-10-05.md)已完成：4,108训练终局与1,440评估局均健康，
Greedy曲线64×1为37.5→32.5→47.5%，128×2为15→47.5→35%，单seed小面板不代表长期收益。
普通Arena健康与异常best晋级已由PR196修复；PR197修复KL已越界仍多走Adam；PR199修复GPU续训随机流。
最新完整实际ROCm1,499 passed/3 skipped。固定rollout[根因诊断](spikes/ppo-first-step-2026-10-05.md)显示
首步仍过冲、fresh critic通过advantage有间接作用；不能把估值相关性当实际动作信号比例。
[两容量×两学习率四臂控制](spikes/ppo-lr-control-2026-10-05.md)训练完成，评估因4局决策上限停止；
#92两万更新尚未完成。当前运行明确FP32，另行修复的[BF16作用域回归](spikes/ppo-autocast-2026-10-05.md)不改变这些producer。

### 2026-10-05：保留跨更新训练截断轨迹

[训练截断诊断](spikes/training-truncation-traces-2026-10-05.md)补齐完整GameSpec、显式动作、responses、
skip_forced与健康字段，支持跨rollout精确回放；普通健康终局不落trace。
四臂128×2高LR的历史decision_limit仍因原日志缺失而无法定位，不把新日志能力当作根因已解。
整合BF16修复后真实ROCm全量1492 passed、3 skipped（525.83s）；源码1a43104，证据已封存。

### 2026-10-05：共享参数reference EMA修复

[EMA根因](spikes/ema-shared-parameters-2026-10-05.md)：按state_dict别名重复原地更新使实际history模型
共享embedding的τ=.02变成.07763184。现在每个唯一参数/持久buffer只更新一次，旧checkpoint兼容。
5例修前失败，修后定向26通过；整合main真实ROCm全量1497 passed、3 skipped（409.86s，ef11ae7）。
该错误不能解释EMA之前的首步过冲，修复的胜率净效应未测。四臂训练全部32更新完成，
独立评估使用冻结旧EMA且已停止；下一训练使用修复版本。训练文档同时撤回用自举Q EV称critic“学好”的过度归因。

### 2026-10-05：融合素材取消循环

四臂共2,097,152训练行、7,292终局；2,040新评估中4个decision_limit使128×2低LR终点未发布。
[四局精确复现](spikes/ppo-decision-limits-2026-10-05.md)定位到目标/素材菜单往返使旧计数清零；
同一状态取消概率由update0的.022–.118升至update32的.9977–.9995，训练强化该行为的机制仍待查。
新guard累计无进展选择链取消，前32次保留；四个失败局复验全部正常终局（506–745决策），
冷回放全部一致，定向39项通过。完整实际ROCm测试1499 passed、3 skipped（287.34s），证据已封存。
已完成[固定actor critic终局校准](spikes/critic-warmup-calibration-2026-10-05.md)：32更新/1,614正常训练局，
128完整heldout局全部健康，actor逐位不变。Q终局MSE1.16326→.98823，通过预定继续条件，
但真实EV仅.0131，且MSE不及在测试标签上拟合的乐观常数基线.9811；不能由此宣称可靠critic或棋力提升。
[三种子critic权重预热策略对照](spikes/critic-warmup-control-2026-10-05.md)已完成：
288更新/4,718,592行/16,636正常训练局，9,984评估局全部正常；预热额外预算单独披露。
update32 warm−cold **+1.61pp [−2.39,+5.73]**，seed差+4.82/−1.56/+1.56pp，未过加长预算条件。
warm−initial **+4.21pp [+1.30,+7.16]**，表明本面板有学习迹象，不能据此断言预热优于cold。
下一优先级为固定actor的实际终局监督critic诊断。#92两万更新与顶尖对局目标继续开放。
[终局监督critic拟合诊断](spikes/terminal-critic-fit-2026-10-06.md)已事前登记并启动：
896新完整游戏（train512/validation128/test256）、3个critic初始化各32epochs，actor/共享主干冻结，
真实终局标签、逐游戏等权，与train拟合后冻结的常数基线比较；区分训练拟合和新发牌泛化。
缓存特征的真实GPU输出/梯度等价预检通过，独立systemd用户服务托管顺序采样/拟合/分析，
保留原子阶段报告并支持完成阶段核验与分析收尾恢复。现已完成896健康游戏及三个固定32epoch拟合：
Q训练MSE均值.1191，test却1.2276，差于train常数.9848（差+.2428，[+.1169,+.3678]），未过门槛。
这支持过拟合/预测幅度失准，不能归为完全不会拟合；符号仍有65–67%状态预测正确率，非对局胜率。
验证集在三个seed均以epoch1最好；已登记[全新256局早期critic确认](spikes/terminal-critic-confirm-2026-10-06.md)，
公共epoch1在新数据上检验，旧test不补测或改终点。独立256健康局确认已通过：
early Q MSE.897976，冻结常数.992278，差−.094302 [−.174524,−.014720]，三个seed均优于常数。
这是固定BC行为下的终局预测改善，未证明棋力；已登记并启动[早期终局critic的三seed PPO对照](spikes/terminal-critic-policy-2026-10-06.md)，
仅转移epoch0/1模型权重、六臂各32更新、新9,984局面板，以固定主终点确认实际策略收益。
全部三个实际GPU转移配对/空Adam/RNG合同及历史非研究Arena预检通过，独立用户服务托管训练/评估/分析；
该研究在70/192更新、2,560个完整评估局后因融合脚本预算停止，无完整胜率结论。
[可重复融合素材修复](spikes/fusion-repeat-memo-2026-10-06.md)复用局部失败状态，原失败完整回放与高预算参照一致，
128局新回归的消息/应答/终局全部一致，完整ROCm1,511 passed/3 skipped，修复已合并。
[新身份三seed重跑](spikes/terminal-critic-policy-restart-2026-10-06.md)于13:19 PDT由独立服务启动，
首个实际更新及评估写入正常，192更新/9,984评估局预算与原判定门槛不变；尚无新强度结论。
15:11 PDT增加独立持续监控：每30秒核验服务/进度/已完成评估文件，每10分钟或异常/完成时同步issues，
完成后独立复算原始胜负和交叉bootstrap。启动时173更新、6,400完整评估局，零截断/错误；后续状态以监控段为准。
原lambda预热对照的负主终点继续保留。

[冻结数据的advantage诊断](spikes/critic-warmup-mechanism-2026-10-05.md)显示critic项std由.31135降至.03273，
full与reward trace相关性.249→.875；这是幅度和相关性的变化，不是信号占比或棋力证据。
79.32%的行距终局>=64个learner rows，直接reward trace已近零，仍依赖critic项；长程credit assignment
和取消行为被强化的具体因果链继续开放，不因全局相关性变好而宣布解决。
[实际PPO首批审计](spikes/critic-warmup-first-rollout-2026-10-05.md)进一步确认终局奖励稀疏，
并复现共享采样RNG对异步事件顺序敏感：同seed/起始actor不代表同训练轨迹。
固定完整游戏上的相关性不直接代表实际PPO数据；三个seed均观察到幅度下降，但主终点未过门槛。
原监控父进程失联导致finalization遗漏，训练/评估子任务全部完成；已用冻结分析器恢复并独立核对bootstrap，
保留原日志/旧状态。父进程退出原因未知，持久监控与可恢复收尾仍需加固。


## 2026-10-08：恢复完整对局强度主线

[128→512续训协议](spikes/terminal-critic-continuation-2026-10-08.md)：从上一研究六个完整训练检查点恢复，
追加2,304更新；原三对手主面板与固定历史RL对手独立报告，新面板19,456局，旧BC任务保留检查18节点。
不接入masker/纠错复习/辅助损失。首次启动的initial模型格式适配错误已定位，零训练更新；修复后以新身份重启，正式启动信息以协议的执行记录及#61/#83/#92监控为准。
仍未完成#92两万更新或跨未见牌组泛化验收。
