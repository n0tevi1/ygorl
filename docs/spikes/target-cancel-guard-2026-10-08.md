# SELECT_CARD取消循环修复（2026-10-08）

关联#218、#83；表示缺口后续追踪[#239](https://github.com/n0tevi1/ygorl/issues/239)。
正式terminal-critic continuation监测发现两局训练decision_limit：
update385、seed1792055546714535986，maliss/sky-striker，第4回合；
update400、seed7153147779260123603，kewl-tune/maliss，第3回合。均到6000步。
已有训练诊断保存完整跨update动作、强制动作跳过配置、实际卡组、规则及全部responses。

在原冻结native上以Python与C++双host重建强制动作，分别核对2987/2990个显式动作、
6000条response及终局字段。两例都在SELECT_CARD选择87746184后，取消第二个对象，
返回第一个目标。分别2922/2933次cancel，没有实际局面进展，输入摘要也反复相同。
它们不是新的融合素材窗口；旧规则6只计数和mask SELECT_UNSELECT_CARD，故无法生效。
证据：`out/research/training-limit-audit-2026-10-08/{385,400}/`。

## 修复与验证计划

两类选择窗口共享已有32次无进展取消预算，Python/C++同时扩展计数和mask条件。
保留玩家、其它决策类型或实际游戏事件重置，Hint不重置；前32次取消、最后唯一出口、
所有原始actions及responses语义不变。显式旧轨迹仍可回放，不把取消从引擎合法动作中删除。

两条真实前缀变成回归fixture：第33次cancel修前仍可选，修后两host应同时mask；
核验observation逐字段相等、snapshot恢复、真实向前选择可完成，以及guard后的旧动作仍可显式重放。
另外验证SELECT_CARD与SELECT_UNSELECT_CARD混合取消共享预算及已有重置条件。

新native独立构建并核对core/Lua来源，不修改运行中的训练/监控绑定库。用冻结u256 actor从两条
真实前缀的首个被mask点继续原生对局；这是已开封异常回归，不是原训练轨迹后续，也不算强度提升。
原研究按原1%截断门槛继续观察、保留失败数据；修复库只供另行登记的新实验，不能热替换原研究。

## 验证结果

两个真实前缀在修前均于第33次cancel应被mask处失败；新库18项定向测试通过，涵盖Python/C++
逐字段编码、snapshot、共享预算、三种重置、唯一出口及显式旧动作语义。新构建与正式旧库的103个
core/Lua源文件逐字节一致，仅host修复不同；来源核验见`core-source-audit.json`。

监测随后捕获update407（seed5976641300419660348，sky-striker/kewl-tune，turn8），
也在两host中逐条复现全部6000responses，确认2784次同型取消；证据另存
`out/research/training-limit-audit-supplement-2026-10-08/407/`。

三个例子都由78397661的墓地效果触发，且第二对象只有己方场上的一张卡可选。
本地官方脚本`CardScripts/official/c78397661.lua`的`tdtg`明确用`repeat ... until fieldg~=nil`
重选两个对象：cancel只返回对象选择，并不会撤回发动。可能更应检查更早的发动决策，
但仅看回收己方卡不能宣布发动必错；需要包含后续收益的对照。guard只保证完成已经进入的选择，
不能把其强制向前动作当作正确策略监督。

三条真实前缀各以4个预先固定续行seed使用冻结u256 actor，两席selfplay；首次改变精确落在
原第33次cancel，之前所有responses匹配。12/12健康终局且动作/响应/结果独立冷回放相同：

| 原训练update | 首次guard改变的决策index | 新总决策数（4seed） | 新回合数 |
|---|---:|---|---|
| 385 | 220 | 737、350、413、329 | 14、5、7、5 |
| 400 | 198 | 399、504、408、757 | 9、10、8、18 |
| 407 | 497 | 623、620、674、674 | 12、12、14、14 |

产物`fixed-continuations/{protocol.json,report.json}`。这些是已开封回归，不代表一般截断率下降幅度，
也不是原训练策略的反事实后续（原策略跨update变化，接续模型/RNG不同）。

随后update429（seed1275104883817376365，sky-striker/lunalight，turn5）出现第4例，
双host同样核对6000responses，2845次SELECT_CARD取消；单独保存于supplement根`429/`。
按同样4seed补做接续，首次guard介入d374，4/4健康终局并冷回放通过；总决策970/976/914/788，
回合17/15/21/18。证据`fixed-continuations-fourth/`；累计16/16异常接续通过，不合并作独立棋力评估。

完整本地presubmit **1577 passed、3 skipped（556.08s）**，原生构建与日志SHA已封存在
`validation.json`、`presubmit-full.log`；跳过项为1项snapshot条件、2项需显式启用的网络测试。

## 更深的表示缺口：已验证与待验证

使用原冻结native、真实checkpoint词表及其64事件长度，逐动作重建输入并核对完整responses。
`input-audit.json`保存逐键张量SHA，包含cards/globals/actions/mask/events/event_mask：

- u256评估game149中32个素材取消窗口的完整输入相同，尽管此前选择的融合目标分别是
  19222426、87758525、93347961。事件历史不保存这些决策响应，当前观察也不含选定融合目标。
  因而该模型无法从这些输入区分这里的三个目标；不能仅归因于模型没读懂已有信息。
- 三个训练失败例中2922/2933/2784次取消窗口也各自逐字节相同。当前输入未表达已尝试多少次；
  原guard不覆盖这类窗口，mask也一直不变，确定性策略无法自行从相同输入改变动作。

这解释了不具备区别上下文的能力，但没有单独证明为何PPO把cancel偏好推高。guard修复只兜底，
不会补上目标语义，也不解决所有自我妨碍。下一步先在独立表示版本加入玩家可见的选择上下文
（已选目标/最近取消/当前段计数），做真实前缀可区分性、Python/C++一致性及隐藏信息隔离检查；
再固定预算比较上下文与无上下文模型的取消率、选择正确性、终局表现。不能把未来实际后果放进输入，
也不能只以更少决策作为训练奖励或验收。正式旧研究保持版本固定。
