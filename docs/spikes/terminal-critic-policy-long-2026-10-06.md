# 终局critic 128更新对照（2026-10-06，运行前协议）

[32更新研究](terminal-critic-policy-restart-2026-10-06.md)已完成且独立审计通过继续门槛。
本研究回答：critic预热优势能否保持到128更新，且warm从32到128是否继续提高棋力。
输出另存`out/research/terminal-critic-policy-long-2026-10-06/`，此前研究全部封存。

## 固定设计

继续相同三个critic初始化epoch0/epoch1权重，只转移model：每臂fresh Trainer/Adam/pool/RNG/counters。
同对actor相同、critic私有权重不同；同初始RNG/schedule不保证异步逐步同轨迹。
PPO seed **2026100556/57/58**、eval seed **2026100559**、bootstrap seed **2026100560**，均与上一轮不同。
六臂各128更新，共**768更新/12,582,912行**；保存且评估固定0/32/128节点。
同一臂从32到128连续训练，不在中间重置optimizer、pool或对局。六臂按seed顺序cold/warm串行训练。
LR1e−4、lambda .5、128env×128steps、FP32/TF32 off、2epochs×2048 minibatch、entropy/reference .05、
KL guard .01、selfplay .75、PFSP/pool8/snapshot10、torch/collect threads2，其余配置不变。
warm仍额外使用共享512游戏/162,019行、1监督epoch/80 Adam步每critic；不声称总成本匹配。

新固定64个ordered deck-pair/deal cluster，各交换牌组/先后手4局；Greedy/旧BC256/初始BC128等权。
公共initial768局，六臂各32/128两节点各768局，共**9,984新评估局**；所有候选共用这组配对输入。
20,000次交叉重采样三个训练seed及64deal cluster，另报条件deal区间。

唯一主终点update128。进一步增加训练预算须同时满足：

1. warm128−cold128交叉95% CI下界>0，且三个seed差均>0；
2. warm128−initial交叉95% CI下界>0；
3. 配对的warm128−warm32交叉95% CI下界>0。

第三项明确检验延长训练收益，不用“仍比cold好”代替学习进展。cold128−cold32也固定报告。
32节点仅用于预定增长比较和描述，不能事后选更好节点作为主结果，不能追加本面板样本求显著。
固定报告每个对手胜率，尤其Greedy；门槛通过也不代表顶尖对局或#92两万更新验收。
共享critic语料、仅三个训练seed、固定对手与训练牌组限制泛化；不计算尚未收集的新语料不确定性。

## 版本、健康与执行

产品源码保持`ceda223`及已验证native。四个研究driver在旧已完成研究副本上仅改变seeds、预算/节点、
增长比较和时间界限；身份绑定drivers、完整inputs、旧analysis与独立completion audit SHA、本协议提交。
真实GPU预检三对起始actor/reference/私有critic差异、空Adam/pool/counters及历史非研究Arena游戏。
协议先提交、预检通过、身份封存后才生成正式数据。

引擎错误/非有限值即停；每臂至少100终局后累计截断>1%停止。任何评估错误或上限停止全研究，
保留实际动作与错误，不替换、不补局、不计平。未完成训练/评估不得覆盖续写。
独立systemd用户服务whole-cgroup管理，单GPU producer与4-worker单线程CPU consumer并行，**16小时上限**。
独立监控另存`out/research/terminal-critic-policy-long-monitor-2026-10-06/`，每30秒核验，
每10分钟及完成/异常更新#61/#83/#92专属段落；完成后独立重算原始胜负/模型SHA/bootstrap及全部增长门槛。
监控只读正式研究，失败证据保留，不自动改变实验协议。
