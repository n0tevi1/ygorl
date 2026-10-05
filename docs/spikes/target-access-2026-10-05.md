# 指定目标的访问路径根因（2026-10-05）

这是49手搜索诊断之后的探索性规则调查，固定 `md-2026-09` 的实际卡表与脚本、普通先攻、被动对手、
目标在回合1结束时在场。不能推广到真实对手会发动效果的对局，也不能把不可达指定怪兽等同于坏手牌。

## Blue-Eyes 1：缺少能同时留场的调整

起手：Droll、Ash、Fydraulis Harmonia、两张 Blue-Eyes White Dragon。
最初我误以为 Droll 不满足 Link-1 种族要求；实际它是魔法师，host 明确允许
**Droll → Spirit with Eyes of Blue → Blue-Eyes White Dragon**。因此不能用最初猜测作不可达证明。

沿合法分支继续：Link-1 取得 Mausoleum，但额外通常召唤只适用 Level1 LIGHT Tuner；
这手没有该类怪兽。通常召唤已用于 Droll；Link-1 需解放自身叫出青眼，不能同时保留它。
Harmonia 需要对方场上怪兽发动效果，被动对手不给事件。Ash 单独通常召唤也没有获取第二只怪兽的资源。
这条路线可做出青眼，却无法凑齐目标 Blue-Eyes Spirit Dragon 的调整与非调整青眼材料。

为避免再凭印象判断，做了两个独立 host 遍历：每个节点从原始start冷重放；以及 session snapshot 恢复分支。
两者均走完 **63,166 个决策节点、11,081 个 turn2 叶子**，全程未出现指定目标；
全部节点/动作/路径/叶子场面逐元素相同。snapshot 遍历还每200个叶子另冷重放一次，共56次。
未按同名卡/区域做去重，没有状态转置、beam 或预算裁剪。深度/节点guard只会报错，不能当完成。

排除的操作只有手牌重排和未提交素材选择的 cancel；它们不增加此手的卡片访问与召唤资源。
这不是枚举所有含无穷重排/取消的输入序列，更不是所有规则版本的数学证明。
第一次40步guard被取消循环触发，随后10,000节点guard也触发；保留两次未完成工件，不把它们当结果。
最终提高审计上限后实际遍历完成，不修改原生搜索预算或回填其失败记录。

## Odion：任务目标访问与 Primite 展开不同

检查实际主/额外卡组的全部卡片效果，Mark 的直接手牌访问入口为
**Mark (97522863)、Anubis (60411677)、Treasures (69299029)**。
主卡组分别3/1/2张，另有1张 Pot of Extravagance；这不是“有入口就必能达成目标”的充分条件判定器。

原16手中，成功的2/3/7/15有直接入口；成功的11没有直接入口，但 Pot 抽2张得到 Treasures。
其余有 Pot 的失败手分别做了引擎实际分支核对：

| 起手 | 抽1 | 抽2的第二张 | 直接访问入口 |
|---|---|---|---|
| Odion 6 | Ether Beryl | Dominus Impulse | 无 |
| Odion 10 | Lordly Lode | Ether Beryl | 无 |
| Odion 11（成功对照） | Solemn Warning | Treasures | 抽2时有 |

Pot 的 pinned 脚本要求 Main1 且没有 phase activity；不能先用 Lode 检索洗牌再开 Pot 挑换抽牌。
随机除外额外卡组不改变这些固定主卡组顶部抽牌。完整trace记录两种抽牌数、随机除外后的场面与后续合法窗口。

没有入口的起手也可能合法展开 Primite，但这与取得 Mark 不同：

- Lode 只检索 Primite，Beryl 只盖放 Primite 魔陷；本卡表的选择不通往 Mark/Anubis/Treasures。
- Lode 叫 Labradorite 后，对特殊召唤的场上怪兽有回合内效果发动限制，后续同调/连接体不能凭其场上起动/诱发效果检索或抽牌。
- 初始可用怪兽资源为 Beryl（4）与 Labradorite（6）。通常召唤额度和 Lode 的单次特召不给 Legatia（12）所需的非调整资源。
  Imduk 的额外通常召唤仅适用 World Chalice；本主卡组没有该系列。Link Spider 也不能从手牌叫等级6的 Labradorite。
- Trap/Apophis 的回合1访问没有凭空获得 Temple 的盖放即开效果：Treasures 虽在场改名为 Temple，
  不会复制原卡效果；本卡表没有原版 Temple。Apophis Serpent 的盖放即开许可要先结算其自身，不能启动刚盖下的第一张。
- 可当回合盖放发动的 Solemn Accusation 能否定自己的魔陷，但不能生成新的目标访问；
  被动对手不给 Dominus/Impermanence/Songs 等所需对方场面或效果。没有此任务中的战斗/后续自然抽牌。

因此对原剩余 **Odion 0/5/6/9/10/12/13/14**，有逐卡资源封闭的不可达规则论证；
加上先前全反应陷阱的1/4/8，11个失败都得到这个固定任务的解释。
这是人工核对 pinned 卡池/脚本的规则论证及具体引擎分支支持，**不是 Odion 全搜索树穷举**；
未来若发现遗漏的合法访问路径，必须撤回相应分类，而不能拒收反例。
原生声明漏路仍是真 bug，应保留已修复的确定性回归；它不意味着这些起手本应完成指定目标。

## 对推进方向的影响

原320手的历史产物保持不变。跨独立诊断合计273手有完整可达线，14手有上述限域规则解释，
**33手仍 unknown**；273不是某个单一配方的覆盖率，不宣布旧gate通过。
另把原320手和新160手验证集的元数据手牌逐一与真实host引擎初始手牌比较，480/480多重集一致；
规则分析没有依赖一个未经核对的采样helper自证起手。小样本入口卡频率偏低本身不证明洗牌错误。
这些证据进一步说明不能把统一每牌组50%的“单怪先攻”门槛等同于教师或实战质量。
保留失败记录，继续有限课程容量实验；后续陷阱牌组训练/评估需要完整对局、后攻与响应价值。

工件：`out/research/declaration-list-2026-10-05/` 中 `blue-eyes-01-crosscheck.json`、
两份完整树、失败guard日志，以及 `odion-access.json` 的完整卡文/脚本hash与Pot分支。
这份解释不改变训练/验证样本、目标、种子、预算或模型选择协议。

验证：仅文档变更，已测运行实现 `e0eea8b` 全套 **1,434 passed / 3 skipped，331.51 s**，
含真实 ROCm；跳过 snapshot 未启用分支与两项 opt-in 联网测试，无 hosted checks。
`target-access-validation.json` 封存审计脚本、规则卡文/脚本身份、两份完整树、失败guard记录与全套测试。
