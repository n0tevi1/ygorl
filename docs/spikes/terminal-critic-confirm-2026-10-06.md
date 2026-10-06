# 早期终局critic的独立确认（2026-10-06，采样前协议）

上一研究固定epoch32泛化失败；三个初始化在预定保存节点0/1/4/8/16/32中，
validation Q MSE均以epoch1最低。现冻结**同一个公共epoch1**，用全新游戏确认这一验证集选择。
这是探索后单独确认，不修改上一研究终点，不在旧test上补测epoch1，也不继续拟合模型。

源为`terminal-critic-fit-2026-10-06`三个seed的epoch1、32完整模型；actor全部相同。
新固定BC自对弈 **256完整游戏/128配对发牌cluster，seed2026100544**，8env批次采样，
同一20牌组池、FP32、TF32关闭、完整原始rollout/明确winner/player标签/缓存特征；
实际game seed与此前train/validation/test全部互斥。任意健康错误/截断停止并留证据，不替换。
沿用原研究真实GPU核验的缓存前向实现及配置，绑定依赖driver/源checkpoint/原研究validation SHA。

全部候选共用同一新数据；主终点为**epoch1 Q_taken逐游戏等权MSE**，比较零预测及
此前仅在train拟合、已冻结的常数.13227164774014238。epoch32作为固定同状态晚期对照，
V指标及距终局>=64行结果描述性报告。未选动作Q排序和棋力仍未测。
20,000次critic seed×128deal cluster交叉bootstrap，seed2026100545，保留两局配对结构。
只有当early相对两基线的差值区间上界均<0，且三个seed相对常数均更好，才值得动作排序/策略验证。
无样本追加、节点选择或模型更新。三个模型共用训练语料，因此区间不覆盖训练数据抽样方差。

研究根`out/research/terminal-critic-confirm-2026-10-06/`；协议提交后才初始化身份/采样。
独立systemd用户服务、最多4小时；分阶段原子登记，分析收尾可重试，已完成原始数据不覆盖。
