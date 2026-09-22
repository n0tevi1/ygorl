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
| I1 | **PPO 重调参**：熵系数 0.05–0.2（比库默认高 10 倍）、当前策略自博弈 + 小规模历史快照池 + keep-best 快照 | 其对固定 checkpoint 自博弈、默认熵 | Rudolph et al. ICLR 2026（7000 次运行：调好的 PPO/MMD 可利用性低于 NFSP/PSRO/R-NaD）；Big 2 2026；Gin Rummy 2026 | ≈0 |
| I2 | **特权 Q-critic + Q-boosted 优势（VRPO）**：以候选动作打分头再出一个 Q 头，用 Expected-SARSA(λ) 回溯替代 GAE | 其 GAE / V-trace / UPGO | Fan & Farina 2026：自博弈中 GAE 方差被随机均衡策略放大；同预算击败 PerfectDou | 中（一个 Q 头 + 合法动作期望） |
| I3 | **引擎状态快照**：把核心跑在写监视 arena 里，按代标只读、捕获首次写、只恢复脏页（ygo-combo-solver 的做法，恢复流量比重放少 15×）。**已实现的首版**（T2.8）没有做脏页跟踪：每局一个 arena 槽位，快照 / 恢复整段拷贝已用前缀（约 4–6 MiB，恢复约 0.6 ms、为重放的约 1%），已满足验收；脏页只恢复留作后续优化，见 [engine.md](../engine.md)「快照」 | 「无法克隆状态」的前提 | ygo-combo-solver 可运行代码 | 工程（移植 arena 到向量化环境） |
| I4 | **先攻展开求解器 → 示范**：用 ygo-combo-solver 离线求解各牌组起手的最优展开线，BC 预热 + KL 先验，再 RL（ExIt 思路） | 其从零 RL 探索长 combo | ByteRL-BC、VGC-Bench「BC 初始化自博弈最强」、AlphaStar；Balatro PPO 复盘 | 中；求解器是唯一可规模化的示范来源 |
| I5 | **效果级文本嵌入**：动作 token 用「该效果的描述串」（`SetDescription(aux.Stringid(code, idx))` → cdb `texts.str1..16`）而非卡级文本；冻结编码器，不依赖卡 ID | 其效果只用索引号，多效果卡不可分 | DraftFM 2026、Cardsformer、MTG CoG24 | 低 |
| I6 | **事件 token 序列 Transformer + 辅助头**：把全部 `MSG_*` 事件（抽卡、召唤、发动、伤害、**响应窗口/放弃**）编码为 token 流，因果 Transformer 替代「最近 16–32 动作 + LSTM」；附加 next-token-prediction、胜负、隐藏信息预测辅助损失 | 其 LSTM + 短历史窗 | DanLM（NTP 辅助损失大幅提升）、Metamon/AMAGO、SkyNet 胜者头、GTrXL/Memory Gym | 2–4× 训练算力 |
| I7 | **中局起始状态**：从求解器线与记录对局采样中局状态开局，并加「增广开局」标志位避免均衡偏移；先后攻配平 | 其总从空场开局 | DAGS 2026 | 低（依赖 I3） |
| I8 | **阻尼/正则化自博弈**：对慢速参考策略加 KL 项（MMD / R-NaD 简化版），胜率平台期后加一个廉价利用者 | 其无正则、无利用者 | Stratego「Ataraxos」2025（< $8k 算力）、Liar's Poker 2025、Minimax Exploiter 2024 | 低（一行 KL） |
| I9 | **非对称 critic 的正确用法**：critic 必须看到「历史 + 特权信息」，只看特权状态会有偏；特权信息只进 critic/辅助头，**绝不**直接喂 actor | 其 critic 看对手状态但未必含历史 | Baisero & Amato 2021；Informed AAC ICML 2026；imitation-gap 系列 | ≈0 |
| — | 不采用：PSRO 系列（大游戏中每轮重训策略）、学习模型搜索（MuZero/LAMIR）、LLM 直接当策略（PTCG-Bench 2026：成本高不等于更强） | | | |

**决策时搜索**留作后期插件：PIMC/确定化搜索在 LOCM 2026 上以 51.4% 击败 ByteRL，MAPLE 把采样世界聚合进一棵树；依赖 I3、吞吐与 §4.5 的信念头（作为确定化采样器）。


网络与训练流水的落地决定见 [06-architecture.md](06-architecture.md) §5.2。
