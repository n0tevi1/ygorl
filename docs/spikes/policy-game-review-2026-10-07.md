# PPO对局行为抽查（2026-10-07）

用户要求检查具体对局，判断棋力提升是否体现为合理打牌。只读已完成128更新研究，不改变原评估。
先固定选择规则再查看详细轨迹：seed0 warm32→warm128对Greedy的输→赢、赢→输、都赢、都输各取
最小game_id，后选项跳过已选deal pair；再从seed1/2的输→赢中按相同规则各取一局。
六组各重放initial/warm32/warm128，共18局。此为按结果分层的定性样本，不能估计错误率或额外证明总体强度。

保持原牌组加载顺序、先后手、发牌/agent RNG、环境与checkpoint SHA，核对原始GameRecord所有结果字段。
记录每步原合法选项、编码mask、实际选择和概率、公开场面与自身手牌、事件和LP；观察信息不反馈给agent。
对手未知手牌/盖卡不用于事前决策判断。另保存可加载replay与EDOPro yrpX。
新增消息记录可在第二次response replay逐字节核对；原实验未封存成功局完整消息，不能声称与原轨迹逐字节相同。

评价展开是否形成有效场面、回合结束时是否浪费明显机会、战斗/伤害与资源使用、应对及失误；
区分已执行动作/引擎结果、局面层面的推断、尚未做反事实验证的更优打法。
优先用当前固定CDB和引擎脚本理解卡片效果，不把合法动作当作合理动作，也不把赢局都算好棋。
证据另存`out/research/policy-game-review-2026-10-07/`，每组同时呈现早晚模型，记录负面例子。

## 结果与证据

18局均逐字段匹配原GameRecord，新增消息日志与response replay逐字节一致，均健康完成；
另导出18份EDOPro `.yrpX`。`report.json`与`selection.json`绑定原始研究与模型SHA。
本地交互查看器：`out/research/policy-game-review-2026-10-07/review.html`，可离线打开，
左右切换checkpoint、决策编号、场面及动作概率，并提供三个重点节点跳转和EDOPro文件链接。
只显示候选方手牌与公开对方场面；原始全事件日志仅供审计，不将未知对方手牌用于事前好坏判断。
可选效果文字由viewer按当前`code << 20 | effect`格式重解码；初版trace的可选`effect_text`字段不作为依据。

下表是预先选定的定性分层样本；胜负为候选方，T为总回合数，不能据此计算泛化胜率。

| case / seed / game_id | 牌组对 | initial | warm32 | warm128 |
|---|---|---|---|---|
| 0 / 0 / 8 | Dracotail / Clown Crew | 胜T19 | 负T20 | 胜T7 |
| 1 / 0 / 7 | Radiant Typhoon Zoodiac / Elfnote Kewl Tune | 负T4 | 胜T29 | 负T8 |
| 2 / 0 / 0 | Resonators / Maliss | 负T16 | 胜T7 | 胜T5 |
| 3 / 0 / 12 | Yummy / Elfnote Kewl Tune | 负T12 | 负T8 | 负T8 |
| 4 / 1 / 16 | Dracotail / Orcust | 负T12 | 负T10 | 胜T11 |
| 5 / 2 / 21 | Yummy / Tearlaments，候选后攻 | 负T17 | 负T15 | 胜T20 |

### 有意义的改善：停手与保留攻击场面

**case0，d27是完全相同的局面。** warm32/128的d0–26动作、公开场面、事件及选项均一致。
对方Maxx “C”已生效；己方已融合出攻击表示Dracotail Gulamel（2800 ATK），另有盖放Mululu。
warm32对Ketu Dracotail继续展开的概率90.5%、结束回合0.7%；warm128变为24.6%和69.4%。
早期模型继续用Gulamel和手牌Urgula融合成守备表示Arthalion，又给对方一次抽牌；后期模型停手，
保留Gulamel，在T3/T5/T7面对空场直接攻击，T7获胜。此变化符合减少继续展开成本、保留有效威胁的判断。
不能仅凭此局断言模型普遍理解Maxx “C”：其权重变化也可能来自通常更保守的结束回合偏好。

做了单点干预：保持warm32在d27之前的实际动作和RNG消耗不变，只把d27改为end_phase，
后续仍由原warm32与Greedy正常采样。此固定续局由负T20变为胜T9（LP5600/−600）。
这是单个局面的干预证据，不是新胜率估计；没有用此数据重算正式评估。

**case2，同样能看到更直接的伤害转化。** warm128在T3保有Magnamhut/Harmonia/Druiswurm三个2500 ATK怪兽，
用Magnamhut与Firewall Dragon交换，再用另两只各打2500；之后保留场面，T5完成击杀。
warm32在T3只有两只2500，打2500后在M2继续重组场面，到T7才赢；initial到T16输。
不能简单把少展开当更优，但此处实际资源与伤害链条支持后期模型更有效地利用已有场面。

case4的warm128同样以Rindbrumm持续打伤害，后续Urgula清理较弱目标并收尾；仍有0 ATK Ash直接攻击、
先把0 ATK手坑留攻击表示并吃到伤害、随后才转守等低质量动作。case5能够在750 LP时用2100+1400攻击收尾，
但也出现下述撞守备错误。因此“赢局”不等于每一步合理。

### 仍然严重的缺陷：自我无效与ATK/DEF判断

**case1，warm128 d6：用Ash Blossom否定自己的Radiant Typhoon Vision。**
链1己方Fuwalos、链2己方Vision（选择检索MST）、链3己方Ash，随后引擎明确`ChainDisabled(chain_count=2)`。
此时Ash概率43.8%、pass56.2%，所以实际采样会经常走坏分支，虽argmax在此节点会pass。
只将d6改成pass，Vision正常检索MST，保留Ash；整局仍负T8，说明该错误不是全部败因。
不能由这一局推出greedy解码整体更好，需另立完整对照。

**case3，warm128 T3：两次攻击公开的3000 DEF守备怪兽。** d207用2200 ATK的Lollipo★Yummy Way攻击，
d208以97.0%概率选Kewl Tune Loudness War（0 ATK/3000 DEF）；无战斗破坏，己方损失800 LP。
d223又用1000 ATK Cupsy☆Yummy攻击同一目标，损失2000，LP7900→5100，对方仍8000。
第一次攻击几乎100%概率，说明不能把问题都解释成低概率采样噪声；第二次攻击23.9%、main2为76.1%。
仅把d207改为main2，下一主阶段保有7900 LP，避免该2800伤害；固定续局仍输，且变为T6结束，
不能声称局部避免伤害必然提高整局胜率。

**case5，另一个seed的赢局也有同类错误。** warm128 T18 d932以近100%概率让1600 ATK的Cooky★Yummy Way
攻击800 ATK/2400 DEF、守备表示的Fiendsmith Engraver，己方损失800 LP（1650→850），没有清掉目标。
这加强了“面对低ATK高DEF守备怪兽仍错误进攻”的怀疑，但六组按结果选例不能估计其总体发生率。

### 输入核验与结论边界

在原生HostDuel的实际policy输入上按原动作前缀重放探针：

- case1/d6事件张量中，Fuwalos及Vision连锁的相对player均为1（自己），Vision chain_count为2；
- case3/d208卡片表第27行明确记录对方怪兽position=3（表侧守备）、ATK=0、DEF=3000；
  所选目标的action card reference指向第27行，非padding或错位。

因此这两例不能归因为引擎没提供己方连锁归属或目标当前DEF。尚未证明网络内部为何没利用信息，
也未证明同类全局编码无bug；可见输入→表示/动作价值学习的薄弱环节应优先诊断。
实际probability、精确输入和局部反事实保存于`input-d*.json`、`branches/`。

总体判断：后期策略有可观察、可解释的局部进步，尤其减少无必要展开与把已有场面转成伤害；
同时战斗数值、连锁归属、采样下的坏动作控制仍不稳定，远非顶尖水平。原评估的Greedy会提供很长的
低压窗口，64.19%对Greedy胜率不应被解释成能应对强玩家的展开与互动。
下一步应建立独立的新战术局面面板（公开攻守值、己/敌连锁、Maxx “C”下继续展开成本），
先量化失误频率与完整对局代价，再比较训练/解码/教师辅助方案。不能直接全局禁止自我连锁或亏血攻击，
这些动作在其他卡片效果与局面下可能有合理用途。

后续根因诊断与验收跟踪：[issue #218](https://github.com/n0tevi1/ygorl/issues/218)，
关联训练信号 #61、强度评估 #83 与长程训练 #92。当前未据此修改训练目标或合法动作，也未启动新的训练。

验证：18局结果与新增消息重放、3次单点干预前缀及消息重放均通过；两个原生输入探针完成。
本地浏览器实际渲染两栏共同d27局面、checkpoint/步骤选择器及对应动作概率。
仅新增文档；产品源码与测试相对`ceda223`无变化，沿用该版本已完成的ROCm全套1511 passed / 3 skipped，
本次重新执行presubmit与diff检查。
