# 语义协同图（T5.3）与引擎包（T5.4）

> 对应设计文档 [5.2 Off-meta 发现专用组件](design/05-deck-building.md#52-off-meta-发现专用组件)「语义协同图」「引擎包枚举」，
> 工程计划 T5.3（[#40](https://github.com/n0tevi1/ygorl/issues/40)）与 T5.4（[#41](https://github.com/n0tevi1/ygorl/issues/41)）。
> 代码在 `src/ygorl/build/`。

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
`Environment.stamp()` 并写入 `environments/<version>/artifacts/synergy_graph.json.gz`（`--out` 可改写到别处），
同时打印该环境 meta 引擎包的召回；再加 `--relations [archetype_support,archseries_related]` 则并入该环境的
Yugipedia 关系边，并分别打印不含 / 含关系边的召回（见[召回检验](#召回检验)）。

## 模块

| 模块 | 作用 |
|------|------|
| `ygorl.build.lua` | 轻量 Lua 读取器：分词（去注释、长字符串）、Lua 5.4 优先级的表达式解析器、`function/if/do/repeat … end/until` 块匹配、顶层函数与赋值定位、整数常量折叠（读 `constant.lua`、`archetype_setcode_constants.lua`、`card_counter_constants.lua`）。不解析语句，按 token 模式定位感兴趣的调用再解析其表达式。 |
| `ygorl.build.filters` | 过滤条件 IR（`Pred / And / Or / Not`）、从 Lua 过滤函数编译 IR（`FilterCompiler`）、在卡片数据库上用位集求值（`CardIndex`）。 |
| `ygorl.build.scripts` | 单个脚本的事实抽取：效果与分类、目标查询（动作 + 位置 + 过滤条件）、召唤 / 素材手续、`listed_names` 等。 |
| `ygorl.build.synergy_graph` | 构图、`SynergyGraph`（邻接查询、保存 / 载入、统计、按卡池限制）、磁盘缓存、召回评估。 |
| `ygorl.build.relations` | 可选边源：把环境的 Yugipedia 关系（`relations.json`）转成 `archetype_support` / `archseries_related` 边并入图（见下「Yugipedia 关系边」）。 |
| `ygorl.build.packages` | 引擎包枚举（T5.4）：局部生长、可达密度打分、去重、跨系列标记。 |

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
`load_or_build()` 的磁盘缓存（`$YGORL_CACHE_DIR` 或 `~/.cache/ygorl/synergy/`）以构图模块（`lua / filters / scripts / synergy_graph.py`）
源码、脚本文件、常量文件、`cards.cdb` 与 `max_fanout` 为键，不会返回过期的图；测试会话共用一次构建（`tests/conftest.py` 的 `real_graph`）。

**扩展点。** 其它关系来源作为新的边类型经 `SynergyGraph.add_edges` 并入（`save/load` 支持任意额外类型），不需要改动挖掘部分：
Yugipedia SMW 关系已接入（下一节，默认不启用），卡文本相似度（T5.2）以后同样接入。

## Yugipedia 关系边（可选）

`ygorl.build.relations` 把环境目录里的 `relations.json`（T5.1，[data.md](data.md)）转成边。**默认不启用**：`build_graph`、
`load_or_build` 与磁盘缓存得到的仍是纯脚本图，上面的统计与下面的「脚本边」召回都不含这些边；需要时显式调用：

```python
from ygorl.build.relations import add_relation_edges, load_relations

g = load_or_build().restrict(env.card_pool, stamp=env.stamp())
add_relation_edges(g, load_relations(env))                      # 默认只加 archetype_support
add_relation_edges(g, load_relations(env), ("archseries_related",))  # 另一种类型单独开启
```

对每个系列页面 `X`（成员 `M = archseries[X]`，按 Yugipedia 系列名而不是 cdb `setcode` 解析）：

| 边类型 | 边 | 默认 |
|------|------|------|
| `archetype_support` | `archetype_support[X]` 中每张**非成员**卡 `s` → `M` 中每张卡 | 是 |
| `archseries_related` | `archseries_related[X]` 中每张卡 `r` → `M` 中每张卡 | 否（需显式传入） |

- `fanout = |M|`、`locations = 0`、`evidence = "yugipedia"`；同一 `(src, dst, type)` 由多个系列产生时按 `fanout` 取最小值合并。
  图的 `meta.relations` 记录来源、抓取日期、每种类型用到 / 被上限剪掉的系列数与边数。
- **上限**：`|M| > 100`（与脚本查询的 `max_fanout` 相同）的系列跳过，理由也相同：「HERO」（157 张）、「Number」「Chaos」「Performapal」这类大页面说明不了哪些卡要一起用，
  每个还会带来上万条边。没有 `archseries` 条目的页面（「Link Monster」「Level 4 Monster Cards」这类卡片群）没有成员，不产生边。
- **成员不作为来源**（`member_sources=False`）：Yugipedia 把系列自己的成员也列在该系列的 `archetype_support` 下（14 张 Dracotail 全都「支援」Dracotail），
  按字面连边只是把系列成员关系写成稠密的完全二部图：md-2026-09 上 `archetype_support` 的 14.5 万对里 10.7 万对来自成员，而召回完全不变。
  只保留非成员来源后，这些边表达的正是脚本图可能漏掉的信息：系列外的卡支援该系列（「Sage with Eyes of Blue」→「Blue-Eyes」怪兽、
  「Nadir Servant」→「Dogmatika」卡）。`member_sources=True` 恢复字面含义。
- **`anti_support` 不建边**：它表示「克制该系列」的卡（如针对连接怪兽的卡），而图上的边在所有使用处（召回、T5.4 的包生长）都被当作
  「两张卡应该放在一起」的理由，把反支援连成边会产生错误的正协同；只在统计里记数量（`anti_support_skipped`）。
- `archseries_related` 默认关闭：它除了真实的关联（「Maiden of White」与「Blue-Eyes」）还包含与协同无关的页面
  （「Recolored counterpart」100 张、「Signature move」84 张），md-2026-09 上约 8.3 万条边里约 1.7 万条来自这两页。

md-2026-09 的卡池图（脚本边 99,368 条）加入默认的 `archetype_support` 后多 37,369 条边（581 个系列，5 个超过上限），
再加 `archseries_related` 共多 120,524 条。T5.4 的 `enumerate_packages` 默认只看脚本边类型，不受影响。

## 召回检验

验收标准（工程计划）：协同图召回 ≥ 80% 的 meta 引擎包，meta 卡表只用于检验、不参与构图。
定义：包的**覆盖度** = 包内成员在「包诱导子图（指定边类型）的最大弱连通分量」中的比例；覆盖度 ≥ 0.8 记为召回。
`evaluate_recall(graph, packages, threshold=0.8, types=None)` 返回召回率、每包覆盖度、召回 / 漏召列表。

### 真实 meta 引擎包（md-2026-09）

**引擎包的推导**（`tools/make_meta_packages.py`，规则固定、没有针对单个卡组调整）从环境的 20 套 meta 卡组
（`environments/md-2026-09/meta/*.ydk`，主卡组 + 额外卡组）得到 31 个包、共 335 张卡，写入
`environments/md-2026-09/artifacts/meta_packages.json`（卡密为键、卡名只作注释，带 `Environment.stamp()`）：

1. **已知泛用卡**全部去掉：`tests/data/generic_pool.json`（手坑、解场、泛用魔陷、泛用额外卡组怪兽）与
   `tools/make_proxy_packages.py` 的 `GENERIC`，两者都在 T5.1 之前整理、与这些卡组无关（28 张）。
2. **拆出共用引擎**（与代理包的做法一致）：按「哪些 meta 卡组用了这张卡」（签名）分组；签名含 ≥ 2 套卡组的组是共用引擎候选
   （如 Branded 与 Dracotail 共用的 Branded 卡），只出现在 1 套卡组的是该卡组自己的候选。
3. **按声明关系拆分**：每个候选组按一种只取自卡片数据库的「声明关系」（不用协同图，也不用 Yugipedia）分成连通分量：
   两张卡基础系列码相同（`setcode & 0xfff`），或一张卡的文本用引号提到另一张的卡名或其卡名中的字符串（"Dracotail" → Dracotail Faimena），
   或两张卡的文本引用了同一个字符串（都提到 "Temple of the Kings"）；卡文本里引用自身卡名的（「"X" 的效果 1 回合只能使用 1 次」）不算。
4. **≥ 3 张的分量成为包**，其余卡视为泛用：出现在 ≥ 2 套卡组、又不属于任何共用引擎的卡（`shared`，35 张：Maxx "C"、Fuwalos、
   Bystial Magnamhut、Dominus 陷阱、Golden Cloud Beast - Malong……），以及只出现在 1 套卡组、既不属于该卡组的任何系列也不与其它卡互相点名的卡
   （`unrelated`，86 张：Pot of Desires、Upstart Goblin、Solemn Strike、Garura、Firewall Dragon、Keldo / Mudora……）。
   去掉的卡按原因列在同一文件的 `excluded` 里，可逐张审阅。

第 3 步使一套卡组里并用的两个引擎成为两个包（Vanquish Soul 与 K9、Ryzeal 与 Mitsurugi、Radiant Typhoon 与 Zoodiac、Magistus 与 Fairy Tail、
Orcust 与 Sinful Spoils / Azamina、Maliss 与 @Ignister），与基因型「若干引擎包份数」的单位一致。`--per-deck` 不做这一步
（同一签名组合成一个包），得到更严格的变体（24 个包）。

**偏向与局限（必须和数字一起读）：** 声明关系与脚本过滤条件同源（系列码、点名），所以这些包天然偏向协同图擅长的「系列核心 + 点名卡」；
只靠种族 / 等级 / 事件诱发协同的卡（Keldo / Mudora、Bonfire、Preparation of Rites、Fabled Lurrie）按第 4 步被当作泛用卡去掉，不参与检验。
反过来，互相点名的泛用卡也会成包（VS K9 里的 Rank 5 泛用超量 `number`）。20 套卡组太少，频率规则抓不到只出现一次的泛用卡，所以才需要第 3、4 步。

**结果**（`tools/build_synergy_graph.py --environment md-2026-09 --relations`；图限制在 md-2026-09 卡池）：

| 边 | 引擎包 | 全部边类型 | 仅 `search` + `special_summon`（+ 关系边） |
|------|------|------|------|
| 脚本 | 31 个（默认） | **0.935（29/31）**，平均覆盖 0.957 | **0.871（27/31）**，0.926 |
| 脚本 + `archetype_support` | 31 个 | 1.000（31/31），0.989 | 0.968（30/31），0.981 |
| 脚本 + `archetype_support` + `archseries_related` | 31 个 | 1.000（31/31），0.989 | 1.000（31/31），0.989 |
| 脚本 | 24 个（`--per-deck`） | 0.750（18/24），0.899 | 0.667（16/24），0.863 |
| 脚本 + `archetype_support` | 24 个（`--per-deck`） | 0.792（19/24），0.916 | 0.750（18/24），0.906 |

**结论：在真实 meta 引擎包上，纯脚本图的召回为 0.935（全部边类型）/ 0.871（仅检索 / 特召），满足 ≥ 80% 的验收标准**；
加入 Yugipedia 支援关系后为 1.000 / 0.968。按卡组不拆引擎的严格变体为 0.750 / 0.667，不满足，原因见下。

各包覆盖度（脚本边，全部 / 仅检索 + 特召；+`archetype_support` 后）：Blue-Eyes、Clown Crew、Dracotail、Branded（共用，13 张）、Elfnote、
Fairy Tail、Fiendsmith、@Ignister、K9、Kewl Tune、Lunalight、Mitsurugi、Orcust、Azamina、Primite、Radiant Typhoon（两个）、Ryzeal、Sky Striker、
Tearlaments、Yummy、Zoodiac 均为 1.00 / 1.00；Vanquish Soul 1.00 / 0.91；Magistus 1.00 / 0.88；Odion 1.00 / 0.86；
Resonators 1.00 / **0.74**（+关系 0.79）；The Fallen & The Virtuous + Albion the Branded Dragon + Ecclesia and the Dark Dragon（`branded-shared-2`，5 套卡组共用）1.00 / **0.67**（+关系 1.00）；
HEROs 0.97 / 0.97；Maliss 0.90 / 0.90；**Branded（Branded 卡组自有部分，12 张）0.58**（+关系 1.00）；**`number` 0.20**（+关系 0.80）。漏召的原因：

- **Branded 自有部分（0.58）**：Branded Retribution / Branded Opening、Nadir Servant / Dogmatika Ecclesia、Despian Proskenion 的连接都指向
  Fallen of Albaz、Branded Fusion，而这两张在共用包 `branded-shared` 里（Dracotail 也用）。这是拆出共用引擎的副作用，不是缺边；
  Yugipedia 的 Branded / Dogmatika 支援关系直接连上了它们。
- **`number`（0.20）**：Number 104 / C104、N.As.H. Knight / CXyz N.As.Ch. Knight、Arc Rebellion 是泛用 Rank 5 超量，因为互相点名而被第 3、4 步留下（规则的误报）；
  它们之间的关系是「以 X 为素材时获得效果」（条件判断，不是对象过滤）与 Rank-Up 魔法，而素材要求「5 星怪兽 4 只」匹配 > 100 张、被 `max_fanout` 剪掉。
- **仅检索 / 特召时**：Resonators 的 4 张额外卡组同调怪兽与 Bone Archfiend、Ecclesia and the Dark Dragon、Magistus 的 Ninaruru / Endymion 只通过素材或送墓边相连，属预期。
- 未连上但包仍被召回的成员：Maliss in Underground（发动时把「Maliss」卡除外，除外类动作不建边）、Xtra HERO Infernal Devicer（「HERO 怪兽 2 只」素材匹配 > 100 张，被 `max_fanout` 剪掉）。
- **严格变体（`--per-deck`）** 另外漏掉 5 个双引擎卡组：Vanquish Soul K9 0.42、Radiant Typhoon Zoodiac 0.64、Ryzeal Mitsurugi 0.65、
  Magistus Fairy Tail 0.68、Orcust 0.70。每个引擎各自完整连通，但两个引擎之间没有过滤层面的边：它们是因为「两套都强」或共用额外卡组 /
  送墓之类的泛用手段而并用，Yugipedia 关系也不连接它们。让图覆盖这种「并用」需要对局数据或共现统计，不属于脚本挖掘的范围。

`tests/test_synergy_graph.py::test_meta_package_recall` 把上面的数字固定为回归测试（阈值略低于实测：脚本 0.9 / 0.85，+`archetype_support` 0.95 / 0.93），
`test_meta_packages_are_reproducible` 检查提交的包文件可由工具重新生成。包文件带环境戳，环境内容变化（`fingerprint` 改变）或换环境时用 `tools/make_meta_packages.py <版本>` 重新推导。

### 代理引擎包（历史）

T5.1 之前没有真实 meta 卡表，用**代理引擎包**评估（数字保留作对照；该测试 `test_proxy_package_recall` 仍在）：

- 代理包由 `tools/make_proxy_packages.py` 从 `tools/make_test_decks.py` 的 `DECKS`（10 套测试牌组的系列核心）派生，
  写入 `tests/data/proxy_packages.json`（卡密为键，卡名只作注释）：去掉泛用卡（泛用陷阱、泛用额外卡组怪兽，手坑与泛用魔陷本来就不在 `DECKS` 里），
  跨牌组共用的引擎单独成包（Fiendsmith、Ryzeal）。共 11 个包、8–20 张 / 包。

| 边类型 | 召回（覆盖度 ≥ 0.8） | 平均覆盖度 |
|------|------|------|
| 全部 | **0.909（10/11）** | 0.931 |
| 仅 `search` + `special_summon` | 0.909（10/11） | 0.923 |

各包覆盖度：Branded/Despia、Labrynth、Purrely、Ryzeal、Tenpai、Yubel 为 1.00；Fiendsmith 0.92；Snake-Eye 0.92（仅检索 / 特召 0.83）；
Voiceless Voice 0.83；Kashtira 0.80；**Tearlaments 0.77（漏召）**。未连上的成员：
- Tearlaments：Keldo / Kelbek / Mudora 与 Tearlaments 的协同来自「回到卡组 → 诱发」这类事件，不是对象过滤；
- Voiceless Voice：Dogmatika Ecclesia / Nadir Servant 自成 Dogmatika 小簇，与仪式核心没有过滤层面的边；
- Kashtira Big Bang、Kashtira Arise-Heart（泛用 7 星超量素材被 `max_fanout` 剪掉）、Promethean Princess（泛用素材 / 泛用炎属性复活）、Necroquip Princess。

代理包是人工整理的系列核心，真实 meta 包（上节）则含共用引擎、双引擎卡组与 meta 里实际的点名卡；两者结论一致：脚本图能把系列核心连起来，
漏掉的是事件诱发协同、泛用素材与没有卡片层面联系的并用。

## 局限

- **只看对象过滤。** 靠事件诱发的协同（「被送去墓地的场合」「回到卡组的场合」「被除外的场合」）、从卡组上方翻开 / 随机堆墓（`DECKDES` 不指定卡）
  以及翻开卡组上方后 `g:Filter(...)` 的效果都不产生边。Tearlaments、Kashtira 这类包因此覆盖度偏低。
- **过近似。** 位置相关的过滤（「墓地的 X 或卡组的 Y」）按两者之并；无法解析的条件被丢弃；`category` 证据可能把
  「自己墓地有 X 存在的场合」的条件检查误读成对 X 的动作（这些边的 `evidence` 标为 `category` / `name`，可按需过滤）。
- **泛用查询被剪掉。** `max_fanout = 100` 会去掉「4 星以下战士族」这类宽过滤与大多数泛用素材要求；这些关系留给 T5.2 的文本相似度与 T5.4 的密度排序处理。
- **素材附加检查未解析。** 连接 / 同调手续里 `s.lcheck` 之类的组检查函数（「包含 X 的怪兽 2 只」）不参与匹配，只看单卡素材过滤。
- **Yugipedia 关系边粒度粗。** 它只说「支援该系列」，不说怎么支援，一张支援卡连向系列全体成员；用来补连通性，不应当作检索 / 特召路径
  （T5.4 的可达打分只看 `search` / `special_summon`，默认也不读这些边）。
- **环境无关。** 图按子模块中的全部官方脚本构建；绑定环境时用 `restrict(pool, stamp)`，`fanout` 仍是在全卡池上计算的。

## 引擎包枚举（T5.4）

```python
from ygorl.build.packages import enumerate_packages, setcodes_from_db
from ygorl.cards.cdb import CardDB

pkgs = enumerate_packages(load_or_build(), setcodes=setcodes_from_db(CardDB.load()))  # 全部种子约 1 s
pkgs[0].members, pkgs[0].starters, pkgs[0].score, pkgs[0].cross_archetype
```

命令行：`uv run python tools/list_packages.py [--top 50] [--cross-only] [--json out.json]`。

**生长（连通子图）。** 边按特异性加权：`w = 类型权重 / sqrt(fanout)`（`search`、`special_summon` 为 1，`send_to_gy` 0.7，
`recover`、`material` 0.5；点名检索权重 1，匹配 16 张的系列检索 0.25），两个方向相加得到无向亲和度。
从每个有出边的卡出发，反复加入使 `aff(v,P)² / deg(v)` 最大的候选 `v`：既与包联系紧密，自身权重又大部分落在包内。
要求 `aff(v,P) ≥ 0.25` 且 `aff(v,P)/deg(v) ≥ 0.15`，上限 15 张。泛用卡（被大量宽过滤指向）过不了占比门槛。
每次只加入与包相邻的卡，所以每个包在协同图上都是（弱）连通子图。

**打分（可达密度）。** 只看包诱导子图上的 `search` + `special_summon` 边：
- `sources(v)`：包内能沿检索 / 特召链到达 `v` 的其它成员数；
- `starters`：贪心最小集合，其可达闭包覆盖全包（没有成员能到达的卡必然是启动卡）；
- `reach_mass = Σ_v min(sources(v), k)/k`（`k = 2`：能从 ≥ 2 个来源到达的成员计满分，对应设计中的「≥ k 条路径」）；
- **`score = reach_mass / |starters|`**，即每张启动卡能稳定拉出的成员数，作为排序主键；
- `density`：包内有序对 `(u, v)` 中 `v` 可从 `u` 到达的比例。一张启动卡拉满全包时 `score` 在包大小处饱和，此时按 `density` 排序。

Jaccard 重叠 > 0.6 的包只保留排名靠前者。

**跨系列包。** 成员的「组」是基础系列码 `setcode & 0xfff`（无系列码的卡自成一组）；`cross_edges` 统计两端不同组的包内边；
`archetypes` 是至少 2 名成员共有的基础系列码。满足「有跨组边」且「≥ 2 个系列」或「主系列之外 ≥ 2 名成员」时标为 `cross_archetype`。

### 结果（默认参数，当前子模块版本）

共 709 个包（去重后），其中 354 个跨系列；大小分布以 15（336 个，达到上限）与 3–7 为主；448 个只需 1 张启动卡。
**Top-50**（`tools/list_packages.py --top 50`）全部是 15 张、1 张启动卡、`score = 15` 的包，按 `density` 排序（1.00 → 0.53）：

> Dragonmaid、Fur Hire、Exosister、Kozmo、Unchained、Lunalight、Mikanko、S-Force、**Nouvelles + Recipe**、Traptrix、Artmage、
> **Springans**（含 Sprind / Tri-Brigade）、Rescue-ACE、Crystron、**Power Patron + DoomZ**、Purrely、P.U.N.K.、Speedroid + Synchron、Dragunity、
> Bujin、Superheavy Samurai、Witchcrafter、Ritual Beast、Memento、Drytron + Ursarctic（2 个，成员只重叠 2 张）、Plunder Patroll、Dream Mirror、Gold Pride、
> Super Quant（2 个）、Cubic、Sky Striker、**Assault Mode**、Fire King、Karakuri、Solfachord、Mathmech、Libromancer、Metalfoes、SPYRAL、
> Salamangreat、**Skull Guardian + Voiceless Voice**、Amazoness、Nephthys、**Gravekeeper's + Necrovalley**、Goblin Rider、Tellarknight、
> **GMX**、Altergeist（粗体为 `cross_archetype`，共 7 个）。

人工抽检：50 个包都是可以成立的检索 / 特召引擎（系列核心加上它点名或按种族 / 等级拉出的卡），没有出现由泛用卡拼成的包。
跨系列包里有真正的混合引擎：Nouvelles + Recipe；Power Patron + DoomZ；Skull Guardian + Voiceless Voice；
White Forest + Azamina + Sinful Spoils；Spright + Ryzeal；Tenpai + Sangen 加上按种族 / 等级拉入的龙族（Dora Dora、Poki Draco、Masked Dragon）。
后者正是设计文档所说「跨系列共享种族 / 等级」的 rogue 种子。也有一部分是官方拆成不同基础系列码的姊妹系列（Heraldic Beast / Heraldry、Aroma / Rose）。
`--cross-only` 可单独列出跨系列包。

与代理引擎包的对照（每个代理包与全部 709 个包的最佳重叠系数 `|P∩R| / min(|P|,|R|)`）：Labrynth、Ryzeal 1.00，Purrely 0.93，
Kashtira 0.90，Fiendsmith 0.89，Voiceless Voice 0.83，Yubel 0.82，Tearlaments 0.77，Tenpai 0.75，Snake-Eye 0.50，Branded/Despia 0.50。
后两者的代理包各含两套引擎（Snake-Eye + Sinful Spoils / Diabellstar；Branded + Despia + Albaz 融合，共 20 张），
一个 ≤ 15 张、经去重的包覆盖不全，属于预期。参数（`min_affinity ∈ {0.2, 0.25, 0.35}`、`min_share ∈ {0.08…0.2}`）扫描下
平均最佳重叠在 0.80–0.84 之间，结果对参数不敏感；默认值按原则选定，没有针对代理包调参。

**局限。** `score` 在包大小处饱和，Top-N 实际由 `density` 决定；上限 15 张会截断大系列（剩余部分作为另一个重叠 ≤ 0.6 的包出现）；
贪心生长依赖种子，同一引擎可能被拆成几个包。T5.5 的基因型（[genotype.md](genotype.md)）以「引擎包份数」为单位，这些包是它的候选集合，
不需要是互斥的划分。
