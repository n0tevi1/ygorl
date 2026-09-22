# 组牌探索与 Off-meta 发现

> 属于 [ygorl 设计文档](README.md)。状态：已评审通过（2026-09-22）。

## 5.1 通用组件
- **pyribs**：MAP-Elites / CMA-MAE（炉石 QD 系列同一实验室）。
- **DSA-ME**（Zhang & Fontaine 2022）：在线深度代理模型 + MAP-Elites 的流程；EvoStone2 仅作参考（C#）。
- **Fontaine 2019** 两阶段对手池：先固定 meta 牌组，再把进化出的精英加入池做 counter 探索。
- **Q-DeckRec**（换一张卡的 MDP）作为 emitter 之一 / warm start。
- **fferegrino/card-embeddings、decksage**：从卡表共现学 card2vec（decksage 实测 E5 文本嵌入比共现好 10-18%）；用于新颖度评分与系列先验。
- Optuna successive-halving 做评估预算分配；BoTorch/Ax 调少量连续比例。


## 5.2 Off-meta 发现专用组件
调研结论：**没有任何组牌论文显式奖励「与 meta 的距离」**；直接把距离当 fitness 只会得到「怪但弱」的牌。可用的组件：
- **语义协同图（不依赖比赛卡表）**：静态挖掘 ProjectIgnis/CardScripts 的 Lua 脚本：`SetCategory(CATEGORY_SEARCH | CATEGORY_TOHAND | CATEGORY_SPECIAL_SUMMON …)`、`IsSetCard(0x…)` 系列码、`LOCATION_DECK/GRAVE/EXTRA` 过滤器、`IsType/IsRace/IsAttribute/IsLevel` 谓词 → 节点=卡密，边=「A 能从卡组检索/特召 B 类卡」。叠加 Yugipedia SMW 关系与 decksage 的 E5 文本相似度。**尚无人做过 YGO 脚本挖掘**，这是本项目的原创组件。比赛卡表共现只用于「新颖度评分」，不用于候选生成。
- **引擎包枚举**：协同图上的连通子图（检索链可达启动/延伸怪 ≥ k 条路径），按「可达密度」而非 meta 出现率排序 → 跨系列共享种族/等级的包正是 rogue 种子。
- **ygo-combo-solver 作漏斗第一层**：「最佳线存在性 + 抗手坑率（`--fire`）」替代单纯的起手概率；同时为 I4 生成示范。
- **新颖度机制**：描述符空间 = 对局衍生描述符 + **学习型描述符**（AURORA 式：动作轨迹自编码器），让真正新的打法自动获得 niche；局部竞争用 **Dominated Novelty Search**（GECCO 2025，pyribs 可接）或 NSLC，让 rogue 牌只与行为相邻者竞争。meta 距离（与 top-cut 卡表的 Jaccard / 共现嵌入距离）只作**过滤与报告维度**。
- **LLM emitter**（UrzaGPT / DraftFM 思路）：作为 pyribs 之外的候选生成器，提出协同图漏掉的包；但**不要**用 LLM 判定成对协同（2025 年研究：LLM 对负协同 F1 ≤ 0.17）。
- **「难开但强」的可见性**：fitness = 经 B 局自博弈微调后的胜率，并记录学习曲线；最佳应对训练（Haluska & Schmid）作为验证对手。
- **报告三元组**：一套 rogue 牌「有意思」当且仅当 (1) 对 meta 占比向量的期望胜率 > 50%，(2) 在增广对局矩阵中的 Nash 权重 > 0，(3) meta 距离排名靠前。三者同时满足才进报告。
- **数据现实**：TCG/OCG 站点只有卡表与 top-cut 占比，无对局矩阵；MD 只有单卡使用率/胜率差。因此「低占比高胜率」的 popularity-paradox 分析主要靠我们自己的模拟矩阵，外部数据只做种子与先验。
