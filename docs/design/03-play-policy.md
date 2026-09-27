# 对局策略：起点与改进

> 属于 [ygorl 设计文档](README.md)。状态：已评审通过（2026-09-22）。


**起点（可移植的设计，不直接依赖其代码）**
- **ygo-agent** `docs/`：观测（卡片 160×41、全局 23、合法动作 24×12、历史 32×14）、13 类动作 + 多选序列化 + Finish 标记、`code_list.txt` 词表、einsum 动作打分。→ 移植编码方案到 PyTorch，但网络与训练循环按下表升级。
- **cjiang1209/yugioh-agent**：edo9300 ctypes 绑定 + 快照池自博弈的新鲜参考。
- **envpool** 的 `Env<Spec>` 异步线程池模式 → 自己用 pybind11 + `std::thread` 实现同等接口，避免 Bazel。**CleanRL** 单文件 PPO 作训练循环骨架。
- nashpy / OpenSpiel `psro_v2`、`egt.alpharank`：对局矩阵 → Nash / alpha-rank。IceYGO/windbot 作可选外部基线（非首期）。

**ygo-agent 2024 设计的已知短板与对应改进**（按证据强度与优先级排序）

| # | 改进 | 取代/补充 ygo-agent 的什么 | 依据 | 成本 |
|---|------|------|------|------|
| I1 | **PPO 重调参**：熵系数 0.05–0.2（比库默认高 10 倍）、当前策略自博弈 + 小规模历史快照池（池中对手按 PFSP 抽：学习方打不过的快照更常出现）+ keep-best 快照；**每次更新 16,384 行（128 局 × 128 步，minibatch 2,048）**：更新噪声是平台期的主因（2,048 行时独立批次梯度余弦仅 +0.10） | 其对固定 checkpoint 自博弈、默认熵 | Rudolph et al. ICLR 2026（7000 次运行：调好的 PPO/MMD 可利用性低于 NFSP/PSRO/R-NaD）；Big 2 2026；Gin Rummy 2026；AlphaStar（PFSP）；本仓库三个种子对 Greedy +0.041（[benchmarks.md](../benchmarks.md)「联赛消融」）；16k 行批：三个种子平均胜率 +0.04（benchmarks.md「训练平台期」） | ≈0 |
| I2 | **特权 Q-critic + Q-boosted 优势（VRPO）**：以候选动作打分头再出一个 Q 头，用 Expected-SARSA(λ) 回溯替代 GAE；**λ = 0.5**（更依赖 critic、方差更低；0.3 与纯 critic 模式的偏差反而有害） | 其 GAE / V-trace / UPGO | Fan & Farina 2026：自博弈中 GAE 方差被随机均衡策略放大；同预算击败 PerfectDou；本仓库 λ = 0.5 三个种子平均胜率 +0.04，与 16k 行批叠加 +0.07（benchmarks.md「训练平台期」） | 中（一个 Q 头 + 合法动作期望） |
| I3 | **引擎状态快照**：把核心跑在写监视 arena 里，按代标只读、捕获首次写、只恢复脏页（ygo-combo-solver 的做法，恢复流量比重放少 15×）。**已实现的首版**（T2.8）没有做脏页跟踪：每局一个 arena 槽位，快照 / 恢复整段拷贝已用前缀（约 4–6 MiB，恢复约 0.6 ms、为重放的约 1%），已满足验收；脏页只恢复留作后续优化，见 [engine.md](../engine.md)「快照」 | 「无法克隆状态」的前提 | ygo-combo-solver 可运行代码 | 工程（移植 arena 到向量化环境） |
| I4 | **先攻展开求解器 → 示范**：用 ygo-combo-solver 离线求解各牌组起手的展开线作第 1 回合示范，第 2 回合起的战斗决策（进战斗阶段、战斗阶段内的选择）用启发式 agent（Greedy）的示范补足，两者混合做 BC 预热；KL 先验只加在求解器示范覆盖的状态（先攻第 1 回合、回合玩家自己的决策），再 RL（ExIt 思路）。只用求解器示范时策略从不进战斗阶段、对 Random 仅 0.305；补战斗示范（1:1）后 0.795，第 1 回合展开不变；KL 先验加在所有状态会把「不攻击」锁死（见 [bc.md](../bc.md)「根因分析」「补救实验」） | 其从零 RL 探索长 combo | ByteRL-BC、VGC-Bench「BC 初始化自博弈最强」、AlphaStar；Balatro PPO 复盘 | 中；求解器是唯一可规模化的展开示范来源，战斗决策的启发式示范零成本 |
| I5 | **效果级文本嵌入**：动作 token 用「该效果的描述串」（`SetDescription(aux.Stringid(code, idx))` → cdb `texts.str1..16`）而非卡级文本；冻结编码器，不依赖卡 ID | 其效果只用索引号，多效果卡不可分 | DraftFM 2026、Cardsformer、MTG CoG24 | 低 |
| I6 | **事件 token 序列 Transformer + 辅助头**：把全部 `MSG_*` 事件（抽卡、召唤、发动、伤害、**响应窗口/放弃**）编码为 token 流，因果 Transformer 替代「最近 16–32 动作 + LSTM」；附加 next-token-prediction、胜负、隐藏信息预测辅助损失 | 其 LSTM + 短历史窗 | DanLM（NTP 辅助损失大幅提升）、Metamon/AMAGO、SkyNet 胜者头、GTrXL/Memory Gym | 2–4× 训练算力 |
| I7 | **中局起始状态**：从求解器线与记录对局采样中局状态开局，并加「增广开局」标志位避免均衡偏移；先后攻配平 | 其总从空场开局 | DAGS 2026 | 低（依赖 I3） |
| I8 | **阻尼/正则化自博弈**：对慢速参考策略加 KL 项（MMD / R-NaD 简化版），胜率平台期后加一个廉价利用者 | 其无正则、无利用者 | Stratego「Ataraxos」2025（< $8k 算力）、Liar's Poker 2025、Minimax Exploiter 2024 | 低（一行 KL） |
| I9 | **非对称 critic 的正确用法**：critic 必须看到「历史 + 特权信息」，只看特权状态会有偏；特权信息只进 critic/辅助头，**绝不**直接喂 actor | 其 critic 看对手状态但未必含历史 | Baisero & Amato 2021；Informed AAC ICML 2026；imitation-gap 系列 | ≈0 |
| — | 不采用：PSRO 系列（大游戏中每轮重训策略）、学习模型搜索（MuZero/LAMIR）、LLM 直接当策略（PTCG-Bench 2026：成本高不等于更强） | | | |

**强度评估**（2026-09-25，#83）：对局强度用**策略对局矩阵**衡量。检查点、规则 agent、搜索包装的 agent 在同一批牌组配对上两两对局（两种牌组分配 × 先后攻，公共随机数），
用 Nash 混合与 alpha-rank 排名，不用单一 Elo（C7：克制关系不一定可传递）。之后训练中的检查点可以增量加入同一张矩阵，画出强度曲线。
单一对手（Greedy）的胜率只作辅助：它已接近被打饱和。

**决策时搜索**留作后期插件：PIMC/确定化搜索在 LOCM 2026 上以 51.4% 击败 ByteRL，MAPLE 把采样世界聚合进一棵树；依赖 I3、吞吐与 §4.5 的信念头（作为确定化采样器）。


网络与训练流水的落地决定见 [06-architecture.md](06-architecture.md) §5.2。
