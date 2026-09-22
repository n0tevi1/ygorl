# 系统架构

> 属于 [ygorl 设计文档](README.md)。状态：已评审通过（2026-09-22）。


```
                 ┌──────────────────────────────────────────────────────────┐
                 │  ygorl.build  组牌探索（外层）                             │
                 │  协同图/引擎包 → 基因型 → 约束 → 求解器起手分析 → 代理模型   │
                 │  → MAP-Elites+DNS(pyribs) → 微调+真实对局 → 矩阵/Nash/报告 │
                 └───────────────▲──────────────────────────┬───────────────┘
                                 │ 胜率/描述符/学习曲线        │ 候选牌组
┌─────────────┐  ┌───────────────┴──────────────────────────▼───────────────┐
│ ygorl.cards │  │  ygorl.nets / ygorl.train  对局策略（内层）                │
│ cdb 加载    │  │  卡片+效果 token Transformer · 事件流 Transformer          │
│ 文本/效果嵌入│─▶│  动作打分头 · 特权 Q-critic · 信念头(§4.5) · 辅助头        │
│ 禁限/格式   │  │  BC 预热(求解器示范) → PPO 自博弈(I1/I2/I8) + 课程 + 中局开局│
│ meta 卡表   │  └───────────────▲──────────────────────────┬───────────────┘
│ 协同图      │                  │ 批量观测/掩码/真值(仅训练)  │ 批量动作
│ (ygorl.data)│  ┌───────────────┴──────────────────────────▼───────────────┐
└─────────────┘  │  ygorl.env  向量化环境（C++ 线程池, envpool 风格异步 API）  │
                 │  AEC 语义 · 观测/候选动作/事件 token 编码 · 响应窗口 token   │
                 │  arena 快照(I3) · 课程开关 · 中局开局                       │
                 └───────────────▲──────────────────────────┬───────────────┘
                                 │ 类型化 Message             │ set_response
                 ┌───────────────┴──────────────────────────▼───────────────┐
                 │  ygorl.engine  pybind11 绑定 edo9300 核心 (CoreBackend)    │
                 │  + ProjectIgnis CardScripts / BabelCDB / LFLists (子模块)  │
                 │  ygorl.solver  ygo-combo-solver 封装：展开线搜索 → 示范     │
                 └──────────────────────────────────────────────────────────┘
```

## 5.1 仓库布局

```
ygorl/
├── pyproject.toml                # uv 管理；scikit-build-core 编译 C++ 扩展
├── CMakeLists.txt
├── third_party/                  # git submodule：ygopro-core(edo9300) CardScripts BabelCDB LFLists
├── csrc/                         # C++：core_backend.cpp msg_decoder.cpp duel_pool.cpp
│                                 #       obs_encoder.cpp event_tokens.cpp arena_snapshot.cpp binding.cpp
├── src/ygorl/
│   ├── engine/    duel.py messages.py backend.py           # 单局 API、类型化消息
│   ├── cards/     cdb.py features.py embeddings.py lflist.py ydk.py   # 含效果级文本 str1..16
│   ├── env/       pool.py single.py spaces.py curriculum.py # VecDuelEnv、DuelEnv、课程/中局开局
│   ├── solver/    combo_solver.py demos.py                 # ygo-combo-solver 封装、示范数据集
│   ├── agents/    base.py random_agent.py greedy.py policy.py
│   ├── nets/      encoders.py history.py heads.py actor_critic.py belief.py
│   ├── train/     bc.py ppo.py selfplay.py exploiter.py    # BC 预热、PPO(VRPO/KL)、快照池
│   ├── eval/      arena.py matchup.py meta_solve.py calibration.py
│   ├── build/     synergy_graph.py packages.py genome.py constraints.py
│   │             funnel.py surrogate.py qd.py report.py
│   ├── data/      ygoprodeck.py masterduelmeta.py yugipedia.py environment.py
│   └── cli.py
├── environments/md-2026-10/      # 版本化快照：environment.json pool.json banlist.lflist.conf meta/*.ydk meta.json artifacts/
├── docs/          encoding.md belief.md offmeta.md
└── tests/         decks/ replays/ combos/ ...
```

## 5.2 关键设计决定
- **`Environment` 一等公民**：`{format, card_pool, banlist, rule_flags, meta_decks(with share), version}`；MD 版 = `DUEL_MODE_MR5` + YGOPRODECK MD 卡池 + 手工维护的 `md.lflist.conf`。TCG/OCG 版只换 flags/表/卡表。所有产物写入 `environments/<version>/artifacts/`。
- **一个决策 = 一个 step**：所有 `MSG_SELECT_*`（含 `SELECT_COUNTER / SELECT_DISFIELD / SORT_CARD / ROCK_PAPER_SCISSORS / ANNOUNCE_*`）解码为候选列表，**绝不抛异常**；多选拆成逐张决策 + Finish。
- **观测（C++ 侧固定形状）**：卡片 token 表 `[N_cards, F]`；全局向量；候选动作表 `[max_options, A]`（含效果 id）+ 掩码；**事件 token 流**（最近 L 个 `MSG_*` 事件 + 响应窗口/放弃 token）；训练时另输出真值张量（对手手牌/牌堆/盖卡）供 critic 与信念头，推理时不输出。
- **网络（PyTorch）**：卡片编码 = 结构化特征 ⊕ 冻结卡文本向量 ⊕ 冻结效果文本向量（⊕ 可关闭的 ID 嵌入）→ 局面 Transformer；事件流因果 Transformer（GTrXL 式门控）替代 LSTM，LSTM 保留为消融基线；动作打分头（点积 + 掩码 softmax）；**Q 头**（对候选动作）+ V 头，均为历史条件的特权 critic；**信念头**（§4.5 五个目标）+ NTP/胜负辅助头。信念头输出 detach 后拼进策略输入。
- **训练流水**：(1) `solver/` 离线求解 meta 牌组起手 → 示范集；(2) `train/bc.py` BC 预热；(3) `train/ppo.py`：PPO 裁剪目标 + Q-boosted 优势（VRPO）+ 熵 0.05–0.2 + KL 到慢速参考/BC 先验；当前策略自博弈 + 小历史快照池 + keep-best；牌组按 meta 分布 + off-meta 噪声采样；课程三阶段 + 中局开局（带标志位）+ 先后攻配平；平台期启动 `exploiter.py`。
- **对局接口（目标 1）**：`Agent.act(obs) -> action` 协议；Random / Greedy / Policy；`Arena` 配对种子 + 先后互换；CLI `ygorl duel`、`ygorl arena`。人类/EDOPro 客户端联机不在首期。
- **组牌（目标 2）**：候选生成来自协同图引擎包 + LLM emitter；基因型 = 引擎包份数 + 泛用槽 + 额外卡组；硬约束；漏斗 (1) 求解器起手分析（最佳线存在性、卡手率、抗手坑率）(2) 代理模型（计数向量 + 卡文本嵌入均值 → 胜率 + 描述符，在线更新）(3) 微调 B 局后的真实对局（vs Nash 加权 meta 池 + 最佳应对）；pyribs MAP-Elites + DNS，描述符 = {先攻胜率, 后攻胜率, 手坑数, combo 长度, 卡手率} + AURORA 学习型描述符；两阶段对手池；输出精英档案、对局矩阵、Nash 混合、off-meta 三元组报告。
- **回放与分支探索**：回放是一等公民（环境版本 + 种子 + 牌组 + 规则 flag + 应答日志，可选每步候选与策略概率），任意一局可记录、重放、导出 `.yrpX` 供 EDOPro 回看。`fork(replay, t)` 从任意决策点分叉并对多个候选 rollout，用于反事实分析、中局开局（I7）与后期 PIMC 搜索；首版以确定性「重放到 t」实现，arena 快照（I3）落地后切换，接口不变。限制：分叉保留同一隐藏状态（上帝视角），对手未知信息的重采样（确定化）需重建局面，留作后续。
- **可扩展点**：`VecDuelEnv` 异步 send/recv 与 envpool 同构，可换 PufferLib 或 actor/learner 分离；`CoreBackend` 可换核心；信念头可复用为 PIMC 采样器。
