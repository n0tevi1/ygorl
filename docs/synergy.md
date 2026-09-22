# 语义协同图（T5.3）

> 对应设计文档 [5.2 Off-meta 发现专用组件](design/05-deck-building.md#52-off-meta-发现专用组件)「语义协同图」，
> 工程计划 T5.3（[#40](https://github.com/n0tevi1/ygorl/issues/40)）。代码在 `src/ygorl/build/`。

协同图不依赖比赛卡表：它静态挖掘 ProjectIgnis/CardScripts 的 `official/c<password>.lua` 脚本（不跑 Lua），
把每张卡「能从哪里、把什么样的卡检索 / 特召 / 送墓」解析成过滤条件，再拿过滤条件去匹配卡片数据库，
得到以卡密（`password`）为节点的有向、带类型的图。

```python
from ygorl.build.synergy_graph import load_or_build, build_graph

g = load_or_build()                       # 首次约 15 s（单进程），之后读磁盘缓存约 1 s
g.successors(9674034, types=("search",))  # Snake-Eye Ash 能检索的卡
g.edge(62962630, 44362883, "search")      # Aluber -> Branded Fusion
g.summary()                               # 统计与解析覆盖率
g.save("graph.json.gz"); g.load("graph.json.gz")
```

命令行：`uv run python tools/build_synergy_graph.py [--out graph.json.gz] [--workers 2]`，
打印统计与代理召回（见下）。`--environment <version>` 把图限制到该环境的卡池、打上
`Environment.stamp()` 并写入 `environments/<version>/artifacts/synergy_graph.json.gz`。

## 模块

| 模块 | 作用 |
|------|------|
| `ygorl.build.lua` | 轻量 Lua 读取器：分词（去注释、长字符串）、Lua 5.4 优先级的表达式解析器、`function/if/do/repeat … end/until` 块匹配、顶层函数与赋值定位、整数常量折叠（读 `constant.lua`、`archetype_setcode_constants.lua`、`card_counter_constants.lua`）。不解析语句，按 token 模式定位感兴趣的调用再解析其表达式。 |
| `ygorl.build.filters` | 过滤条件 IR（`Pred / And / Or / Not`）、从 Lua 过滤函数编译 IR（`FilterCompiler`）、在卡片数据库上用位集求值（`CardIndex`）。 |
| `ygorl.build.scripts` | 单个脚本的事实抽取：效果与分类、目标查询（动作 + 位置 + 过滤条件）、召唤 / 素材手续、`listed_names` 等。 |
| `ygorl.build.synergy_graph` | 构图、`SynergyGraph`（邻接查询、保存 / 载入、统计、按卡池限制）、磁盘缓存、召回评估。 |

## 解析覆盖

**效果与分类。** 识别 `eN=Effect.CreateEffect(c)`、`eN=eM:Clone()`、`eN=Fusion.CreateSummonEff(...)` / `Ritual.*`，
以及 `eN:SetCategory / SetTarget / SetOperation / SetCost`。`SetCategory` 的表达式（`CATEGORY_TOHAND+CATEGORY_SEARCH`、`|`）
按常量折叠成位掩码；效果引用的函数沿脚本内调用图向下传播，这样 `s.thtg → s.thfilter` 里的调用也能知道自己属于哪个效果、带哪些分类。

**目标查询。** 以下调用的过滤参数与己方位置参数（`LOCATION_DECK|LOCATION_GRAVE` 等，常量折叠）生成查询：

| 调用 | 过滤 / 玩家 / 己方位置 / 对方位置 / 附加参数起点 |
|------|------|
| `Duel.IsExistingMatchingCard`、`Duel.IsExistingTarget` | 0 / 1 / 2 / 3 / 6 |
| `Duel.SelectMatchingCard`、`Duel.SelectTarget` | 1 / 2 / 3 / 4 / 8 |
| `Duel.GetMatchingGroup(Count)`、`Duel.GetFirstMatchingCard` | 0 / 1 / 2 / 3 / 5 |

- 己方位置为 0（只看对方场上）的查询跳过；玩家参数是 `1-tp` 时交换己方 / 对方位置。
- 调用末尾传给过滤函数的附加参数（如 `IsExistingMatchingCard(s.spfilter,tp,loc,0,1,nil,e,tp,44632120)`）
  按位置绑定到过滤函数形参，常量实参参与求值（`c:IsCode(code)`），未绑定的形参视为未知。
- 位置是局部变量等无法折叠时，退回到同一效果 `Duel.SetOperationInfo(0,CATEGORY_X,…,loc)` 中对应分类的位置。

**动作推断**（记录在 `Query.evidence`，按可信度）：

1. `filter`：过滤函数里的 `IsAbleToHand`（加入手牌）、`IsCanBeSpecialSummoned`（特召）、`IsAbleToGrave(AsCost)`（送墓）、
   `IsSSetable`（盖放）、`IsAbleToRemove`、`IsAbleToDeck` 等；
2. `hintmsg`：`Select*` 调用之前最近的 `Duel.Hint(HINT_SELECTMSG,tp,HINTMSG_*)`（`ATOHAND / SPSUMMON / TOGRAVE / SET / TOFIELD（放置）/ EQUIP` 等）；
3. `name`：过滤函数名前缀（`th* / sp* / tg* / set*`）且效果带对应分类；
4. `category`：效果分类里只有一个相关动作（`TOHAND|SEARCH`、`SPECIAL_SUMMON`、`TOGRAVE|DECKDES`、`SET`、`EQUIP`）。

**过滤谓词。** 过滤函数（`s.xxx`、局部函数、匿名函数、`aux.FilterBoolFunction(Ex)(Card.IsX, …)`、`aux.FaceupFilter`、
`aux.NecroValleyFilter`、`aux.NOT/AND/OR`、`Fusion.IsMonsterFilter`、`Synchro.NonTuner`）的所有 `return` 表达式取并集，
`and / or / not` 与嵌套的 `s.helper(c, …)`（带实参绑定，深度 ≤ 6、防递归）内联展开。可识别的谓词：

| 谓词 | 对应卡片数据库字段 |
|------|------|
| `IsSetCard / IsOriginalSetCard / Is*SetCard`（`SET_*`、十六进制、`{A,B}` 表） | `setcodes`：查询 `q` 与卡片系列码 `sc` 匹配当且仅当 `(sc & 0xfff) == (q & 0xfff)` 且 `(sc & q & 0xf000) == (q & 0xf000)`（子系列语义） |
| `IsCode / IsOriginalCode(Rule) / IsSummonCode`（数字、`id`、`CARD_*`、绑定的形参） | `password`，以及 `alias` 指向它的卡；异画卡号折回原卡 |
| `IsType / IsOriginalType`、`IsMonster / IsSpell / IsTrap / IsSpellTrap`、`IsQuickPlaySpell` 等、`IsExactType`、`IsNormalSpell/Trap` | `type` 位标志（任一位 / 全部位 / 相等） |
| `IsRace`、`IsAttribute`（及 `Original` 变体） | `race`、`attribute` 位标志 |
| `IsLevel / IsLevelBelow / IsLevelAbove / IsLevelBetween`、`IsRank*`、`IsLink*`、`HasLevel`，`c:GetLevel()<=4` 等比较 | `level`（等级只存在于非超量、非连接怪兽） |
| `IsAttack* / IsDefense*`、`c:GetAttack()>=…` | `attack` / `defense` |
| `ListsCode / ListsArchetype / ListsCodeAsMaterial / ListsArchetypeAsMaterial` | 被匹配卡脚本自身的 `s.listed_names`、`s.listed_series`、融合素材卡号（`s.material`、`Fusion.AddProcMix` 的卡号）、`s.material_setcode` |

无法解释的原子（位置判断、对其它卡的条件、局部变量）记为 `unknown`，按**过近似**处理：从合取中丢弃；
出现在析取或否定中时整个过滤视为「不受约束」。因此漏解析只会让过滤变宽，不会丢掉真实可匹配的卡。

**手续。** `Fusion.CreateSummonEff / RegisterSummonEff / SummonEffTG / SummonEffOP`（`fusfilter`、`location`，具名表或位置参数）→
从额外卡组特召融合怪兽；`Ritual.AddProcGreater / AddProcEqual / CreateProc / AddProc / Target / Operation / Add*Code` → 特召仪式怪兽
（默认手牌）；`Link / Xyz / Synchro.AddProcedure`、`Fusion.AddProcMix / AddProcMixN / AddProcMixRep` → 素材边（超量加上 `level` 条件）。

**覆盖率**（CardScripts 与 BabelCDB 为当前子模块版本；`tools/build_synergy_graph.py` 输出）：

| 项目 | 数量 |
|------|------|
| 脚本 / 在 `cards.cdb` 中的脚本 / 解析错误 | 13,461 / 13,455 / 0 |
| 识别出至少一个效果分类的脚本 | 11,232（83.5%） |
| 产生至少一条查询的脚本 | 8,817（65.5%） |
| 产生至少一条边的脚本（`script_coverage`） | 4,739（35.2%） |
| 匹配类调用 / 只涉及对方 / 动作未知 / 位置未知 | 34,615 / 4,610 / 7,592 / 142 |
| 查询 / 产生边 / 过宽（> `max_fanout`）/ 不受约束 / 动作不建边（除外、回卡组等） | 15,744 / 6,226 / 3,572 / 738 / 5,060 |

约三分之一的脚本没有查询是正常的：通常怪兽、纯数值 / 破坏 / 无效类效果本来就不指定要移动的卡。
「动作未知」主要是条件检查（「自己场上有 X 存在的场合」）与破坏 / 无效等对象选择。

## 边语义

节点是 `cards.cdb` 中所有**规范、非衍生物**卡（异画经 `CardDB.canonical` 折回原卡号），共 14,139 个；
`nodes[p] = {"categories": 效果分类位掩码, "has_script": bool}`。有向边 `A → B` 表示「A 的效果能把 B 带入可用状态」：

| 类型 | 含义（`locations` = B 所在的己方位置） | 目标范围 | 数量 |
|------|------|------|------|
| `search` | A 从卡组把 B 加入手牌，或盖放 / 放置 / 装备（`to_hand / set / place / equip` + `LOCATION_DECK`） | 主卡组卡 | 25,603 |
| `special_summon` | A 从手牌 / 卡组 / 墓地 / 除外 / 额外卡组特召 B（含融合、仪式手续） | 手牌、卡组 → 主卡组怪兽；墓地、除外 → 所有怪兽；额外卡组 → 额外卡组怪兽与灵摆怪兽 | 41,409 |
| `send_to_gy` | A 把卡组 / 手牌 / 额外卡组的 B 送去墓地 | 同上按位置 | 8,913 |
| `recover` | A 把墓地 / 除外的 B 加入手牌 | 主卡组卡 | 10,512 |
| `material` | A 可作为 B 的连接 / 超量 / 同调 / 融合素材（**方向：素材 → 额外卡组怪兽**） | 所有怪兽 | 17,366 |

合计 103,803 条边；11,210 个节点至少有一条边。每条边记录：
- `fanout`：产生它的查询在目标范围内匹配的卡数（多个查询产生同一条边时取最小值）。`fanout` 大于 `max_fanout`（默认 100）的查询
  视为泛用（如「从卡组把 1 只怪兽加入手牌」）不建边；下游可按 `1/fanout` 给边加权（T5.4 即如此）。
- `evidence`：最可信的动作证据（边数：`filter` 81,702，`procedure` 19,337，`category` 1,445，`hintmsg` 1,019，`name` 300）。
- 每个 `(src, dst, type)` 只有一条边，重复时位置取并集。

跨系列边（两端都有系列码但没有共同的基础系列码）约 20,600 条，另有约 34,300 条边至少一端无系列码（种族 / 等级 / 卡号驱动）。

**持久化。** `save/load` 为 JSON（`.gz` 后缀自动 gzip，约 0.7 MB），`format = "ygorl-synergy-graph"`、`version = 1`，
`meta` 含 `max_fanout`、CardScripts / BabelCDB 的 commit、统计数据；按环境限制后另含 `meta.environment = Environment.stamp()`。
`load_or_build()` 的磁盘缓存（`$YGORL_CACHE_DIR` 或 `~/.cache/ygorl/synergy/`）以 `ygorl/build/*.py` 源码、脚本文件、常量文件与
`cards.cdb` 为键，不会返回过期的图。

**扩展点。** Yugipedia SMW 关系（T5.1）与卡文本相似度（T5.2）目前无法离线获取；它们可作为新的边类型经
`SynergyGraph.add_edges` 并入（`save/load` 支持任意额外类型），不需要改动挖掘部分。

## 召回检验（代理）

验收标准（工程计划）：协同图召回 ≥ 80% 的 meta 引擎包，meta 卡表只用于检验、不参与构图。
**真实 meta 卡表要到 T5.1 才有（masterduelmeta / YGOPRODECK 在当前网络策略下不可达）**，
所以现在用**代理引擎包**评估，结果只能作为代理指标：

- 代理包由 `tools/make_proxy_packages.py` 从 `tools/make_test_decks.py` 的 `DECKS`（10 套测试牌组的系列核心）派生，
  写入 `tests/data/proxy_packages.json`（卡密为键，卡名只作注释）：去掉泛用卡（泛用陷阱、泛用额外卡组怪兽，手坑与泛用魔陷本来就不在 `DECKS` 里），
  跨牌组共用的引擎单独成包（Fiendsmith、Ryzeal）。共 11 个包、8–20 张 / 包。
- 定义：包的**覆盖度** = 包内成员在「包诱导子图（指定边类型）的最大弱连通分量」中的比例；覆盖度 ≥ 0.8 记为召回。
  `evaluate_recall(graph, packages, threshold=0.8, types=None)` 返回召回率、每包覆盖度、召回 / 漏召列表。

| 边类型 | 召回（覆盖度 ≥ 0.8） | 平均覆盖度 |
|------|------|------|
| 全部 | **0.909（10/11）** | 0.931 |
| 仅 `search` + `special_summon` | 0.909（10/11） | 0.923 |

各包覆盖度：Branded/Despia、Labrynth、Purrely、Ryzeal、Tenpai、Yubel 为 1.00；Fiendsmith 0.92；Snake-Eye 0.92（仅检索 / 特召 0.83）；
Voiceless Voice 0.83；Kashtira 0.80；**Tearlaments 0.77（漏召）**。未连上的成员：
- Tearlaments：Keldo / Kelbek / Mudora 与 Tearlaments 的协同来自「回到卡组 → 诱发」这类事件，不是对象过滤；
- Voiceless Voice：Dogmatika Ecclesia / Nadir Servant 自成 Dogmatika 小簇，与仪式核心没有过滤层面的边；
- Kashtira Big Bang、Kashtira Arise-Heart（泛用 7 星超量素材被 `max_fanout` 剪掉）、Promethean Princess（泛用素材 / 泛用炎属性复活）、Necroquip Princess。

`tests/test_synergy_graph.py::test_proxy_package_recall` 把代理召回 ≥ 0.8 固定为回归测试。T5.1 提供真实 meta 卡表后，
应改用真实引擎包重新评估，并把本节数字替换为真实结果。

## 局限

- **只看对象过滤。** 靠事件诱发的协同（「被送去墓地的场合」「回到卡组的场合」「被除外的场合」）、从卡组上方翻开 / 随机堆墓（`DECKDES` 不指定卡）
  以及翻开卡组上方后 `g:Filter(...)` 的效果都不产生边。Tearlaments、Kashtira 这类包因此覆盖度偏低。
- **过近似。** 位置相关的过滤（「墓地的 X 或卡组的 Y」）按两者之并；无法解析的条件被丢弃；`category` 证据可能把
  「自己墓地有 X 存在的场合」的条件检查误读成对 X 的动作（这些边的 `evidence` 标为 `category` / `name`，可按需过滤）。
- **泛用查询被剪掉。** `max_fanout = 100` 会去掉「4 星以下战士族」这类宽过滤与大多数泛用素材要求；这些关系留给 T5.2 的文本相似度与 T5.4 的密度排序处理。
- **素材附加检查未解析。** 连接 / 同调手续里 `s.lcheck` 之类的组检查函数（「包含 X 的怪兽 2 只」）不参与匹配，只看单卡素材过滤。
- **环境无关。** 图按子模块中的全部官方脚本构建；绑定环境时用 `restrict(pool, stamp)`，`fanout` 仍是在全卡池上计算的。
