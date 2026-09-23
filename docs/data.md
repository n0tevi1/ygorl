# 数据抓取与环境快照（T5.1）

> 对应设计文档 [引擎与数据：可复用资源清单](design/07-building-blocks.md) 中的 YGOPRODECK、masterduelmeta、Yugipedia 三个数据源，
> 工程计划 T5.1（[#38](https://github.com/n0tevi1/ygorl/issues/38)）。环境目录规范见 [environments.md](environments.md)。
> 代码在 `src/ygorl/data/`，命令行入口 `ygorl env`（`src/ygorl/commands/env.py`）。

一条命令生成（或重新生成）一个 Master Duel 环境：

```bash
uv run ygorl env build md-2026-09                 # 抓取缺失的原始文件 → 解析 → 写入 environments/md-2026-09/ → 校验
uv run ygorl env build md-2026-09 --offline       # 只用已有的原始文件（来源失效、手工替换文件后用它）
uv run ygorl env check md-2026-09                 # T0.4 校验（含 meta 卡组合法性），打印摘要
```

`build` 的参数：`--root DIR`（环境根目录，默认 `$YGORL_ENVIRONMENTS` 或 `./environments`）、`--raw DIR`（原始文件目录，默认 `<环境>/raw`）、
`--refresh`（全部重新抓取）、`--since YYYY-MM-DD`（meta 窗口起点）、`--min-share F`（入选 meta 的最小份额，默认 0.01）、
`--max-decks N`（最多 N 套 meta，默认 20）、`--no-relations`（跳过 Yugipedia）、`--reviewed-by NAME`（记录禁限表已人工校对）。
`--since`、`--min-share`、`--max-decks` 不给时沿用同一版本上一次构建记在 `environment.json` 的 `sources.build` 里的值，
所以 `--offline` 重建得到相同的文件。版本号必须形如 `md-YYYY-MM[-修订]`；目前只支持构建 MD 环境（TCG / OCG 环境的 BabelCDB 导出未实现）。

**构建只能抓到「现在」的数据**：禁限状态与卡池都是抓取时刻的。历史版本只能用当时保存的原始文件离线重建，所以版本号应取抓取所在的月份。

## 抓取器与解析器

每个来源分成两半：**抓取器**（网络 → `<环境>/raw/<来源>/<文件>.json`）和**解析器**（原始文件 → 环境文件）。解析器只读原始文件、不联网，
所以来源失效时可以手工做一个同形状的文件放到原位，`--offline` 重建。抓取器只用标准库（`urllib` 走 `HTTPS_PROXY`，`ssl` 用系统 CA，
从不关闭 TLS 校验），带标识本项目的 User-Agent，按来源限速，429 / 5xx / 网络错误指数退避重试 3 次。

每个原始文件旁有 `<文件>.source.json`：请求过的 URL、抓取时间（UTC）、SHA-256、User-Agent。构建时把它们汇总进 `environment.json` 的
`sources.raw`（每个文件：首个 URL、请求数、抓取时间、SHA-256）；文件内容与记录的 SHA-256 不符或没有 `.source.json` 时记 `"manual": true`，
一眼可见哪些输入是手工替换的。

| 来源 | 请求 | 限速 | 原始文件 | 解析器需要的最少字段（手工替换时照此） | 产出 |
|------|------|------|----------|------------------------------------------|------|
| YGOPRODECK API v7 | `cardinfo.php?format=master duel`，1 次 | 10 次/秒（API 上限 20） | `ygoprodeck/cardinfo.json`（约 24 MB） | `{"data": [{"id", "name", "frameType"}]}` | `pool.json` |
| masterduelmeta | `/api/v1/cards?limit=3000&page=N&sort=_id`，约 6 次 | 1 次/秒 | `masterduelmeta/cards.json` | `[{"_id", "konamiID", "name"}]`（卡表按 `_id` 引用卡片；`konamiID` 就是卡密） | 卡表的卡片对应 |
| 同上 | 同一批响应中带 `banStatus` 的卡（即站点「禁限卡表」页的数据） | — | `masterduelmeta/banlist.json` | `[{"konamiID" 或 "name", "banStatus"}]`，`banStatus` 取 `Forbidden` / `Limited 1` / `Limited 2`（也认 `Limited`、`Semi-Limited`、`Unlimited`，或整数 `limit`） | `banlist.lflist.conf` |
| 同上 | `/api/v1/articles?limit=400&sort=-date`，1 次 | 1 次/秒 | `masterduelmeta/articles.json` | `[{"title", "date", "url"}]` | 最近一次 MD 禁限更新（标题匹配 `Master Duel … Forbidden … Limited`），默认 meta 窗口起点 |
| 同上 | `/api/v1/top-decks?sort=-created&limit=500&page=N`，翻页到窗口起点 | 1 次/秒 | `masterduelmeta/top-decks.json` | `[{"deckType": {"name"}, "created", "url", "main"/"extra": [{"card": {"_id", "name"}, "amount"}], "rankedType"/"tournamentType": {"statsWeight", "includeInStats"}}]` | `meta.json`、`meta/*.ydk` |
| Yugipedia SMW | `api.php?action=ask`，每个属性按卡片类型 × 属性分 9 片、每页 1000 条，共约 40 次 | 2 秒/次 | `yugipedia/<属性>.json` | `{"query": {"results": {<页面>: {"printouts": {"Password": [...], <属性>: [{"fulltext"} 或字符串]}}}}}`（即 `ask` 的响应形状） | `relations.json` |

Yugipedia 的 SMW 查询在偏移超过 5000 时会**静默地从 0 重新开始**，所以每个查询都按卡片类型与属性切片（每片远小于 5000 条），
抓取器发现返回的偏移与请求不符、或某片需要超过 5000 的偏移时直接报错，而不是悄悄写出残缺的数据。

原始文件不进 git（`.gitignore` 的 `environments/*/raw/`，合计约 33 MB，其中 YGOPRODECK 的完整响应约 24 MB）；进 git 的是它们生成的环境文件
与 `sources.raw` 里的 SHA-256。

## 卡片对应

卡片一律以 BabelCDB（`third_party/BabelCDB/cards.cdb`）的 8 位 `password` 为主键（`ygorl.data.cardmap.CardMapper`）：

1. 来源给的卡密（YGOPRODECK `id`、masterduelmeta `konamiID`、Yugipedia `Password`）在卡片数据库里存在 → 采用；
   它是异画（`alias` 指向同名原卡）时折回原卡密，所以 `pool.json`、禁限表、`.ydk` 里都是原卡密（异画按原卡计张数，见 `cards.legality`）；
2. 否则按英文卡名匹配：先精确，再忽略大小写、重音与标点；同名多张（不唯一）时不猜，留给人工；
3. 衍生物不作为按名匹配的目标，YGOPRODECK 列出的衍生物不进卡池。

两步都失败的卡**不会被静默丢弃**：卡池、禁限表、卡表、Yugipedia 各自的无法对应条目都写进 `review/report.md`，命令行也打印数量。

## 禁限表生成与人工校对

MD 禁限表没有上游（`third_party/LFLists` 里只有 TCG、OCG、World、GOAT、Speed、Rush 等表，没有 MD 表），由 masterduelmeta 的 `banStatus`
生成 `banlist.lflist.conf`（EDOPro 格式，表名 `YYYY.MM MD`，每行 `--卡名` 注释，异画折回原卡；同一张卡状态冲突时取更严的并在报告里列出）。
YGOPRODECK 的 `banlist_info` 没有 MD 字段，不能作第二来源。

生成的禁限表默认是**待校对**（`environment.json` 的 `review.banlist.status = "pending"`，`ygorl env check` 与 `build` 都会显示）。校对流程：

1. 打开 `review/report.md` 的「禁限表校对」一节：全部禁限卡（卡密、卡名、MD / TCG / OCG 状态、备注：异画合并、按卡名匹配、人工修正、不在卡池），
   与同根目录下上一个 `md-*` 版本相比的变化，以及 masterduelmeta 的更新公告链接（公告写明生效日期）。
2. 在游戏内「卡组 → 禁止・限制卡一览」逐条核对。报告里的「按日期统计的卡表」也是一项旁证：禁限表生效后仍有卡表违反它，说明禁限表可能有误。
3. 不一致的卡写进 `review/banlist-overrides.lflist.conf`（进 git），每行 `<password> <张数> --原因`，张数 3 表示解除限制：

   ```
   32909498 1 --游戏内为限制，masterduelmeta 未更新
   68059897 3 --游戏内未列出
   ```

4. `uv run ygorl env build <版本> --offline --reviewed-by <名字>` 重新生成：修正叠加在生成结果上，`review.banlist` 记为
   `{"status": "reviewed", "by", "date"}`。之后不带 `--reviewed-by` 重建时，禁限表内容不变则保留「已校对」，一旦变化就回到「待校对」。

## Meta 卡组

- **窗口**：从 `--since`（默认：最近一次 MD 禁限更新公告的日期）到抓取时刻的 masterduelmeta 卡表（排位上位、比赛）。
  公告日期往往早于实际生效日（维护日），窗口开头会混入旧禁限表下的卡表；报告按日期列出「卡表数 / 在新禁限表下不合法的卡表数」，
  若不合法卡表只出现在窗口开头，`build` 打印警告并给出建议的 `--since`。
- **份额**：按 masterduelmeta 自己的统计权重 `statsWeight` 加权（普通排位与比赛为 10；Dice Rally 等活动、Win Streak、特殊赛事为 0 或
  `includeInStats: false`，不计入），某卡组类型（`deckType.name`）的份额 = 该类型卡表的权重和 / 窗口内全部权重和。
- **代表卡表**：每个类型取**中心卡表（medoid）**：在该类型能完整对应到卡片数据库、且在本环境（卡池 + 禁限表 + 构筑规则）下合法的卡表中，
  选与该类型全部卡表的逐卡张数 L1 距离之和最小的一份，平局取最新的。它是真实存在的卡表（`.ydk` 头部注释写明来源 URL 与日期），
  不是拼出来的平均卡表。副卡组丢弃（MD 排位没有副卡组）。
- **入选**：份额 ≥ `--min-share` 且有合法代表卡表的类型，按份额取前 `--max-decks` 个，写成 `meta/<类型名>.ydk`；`meta.json` 另记
  每套的卡表数、合法卡表数和代表卡表 URL。份额之和 < 1，剩余视为「其它」。
- **合法性**：构建最后用 `load_environment(path, cards=CardDB.load())` 校验整个环境，任何一套 meta 卡组不合法都会失败（T0.4 的规则）。

## Yugipedia 关系（`relations.json`）

四个 SMW 属性：`Archetype support`（支援某系列）、`Anti-support`（反支援）、`Archseries related`（与某系列相关）以及 `Archseries`
（所属系列，用来把系列名解析成成员卡）。文件按属性倒排、每个系列一行：

```json
{
 "source": "Yugipedia Semantic MediaWiki (action=ask)", "retrieved": "2026-09-23", "properties": {...}, "note": "...",
 "archetype_support": {
  "Abyss Script": [4682617, 6004133, ...],
  ...
 },
 "anti_support": {...}, "archseries_related": {...}, "archseries": {...}
}
```

值是 Yugipedia 页面名（多数是系列，也有「Level 4 Monster Cards」「Link Monster」这类卡片群），只含卡池里的卡。
`relations.json` 是环境目录里的可选文件，`load_environment` 不读它，也不计入 `fingerprint`。

**尚未接入协同图。** [synergy.md](synergy.md) 的扩展点可以直接用它：对 `archetype_support[X]` 里的每张卡 `s`、`archseries[X]` 里的每张卡 `m`
加一条 `s → m` 的 `archetype_support` 边（`archseries_related` 同理），经 `SynergyGraph.add_edges` 并入；Yugipedia 系列名与 cdb `setcode`
的名字不完全一致，所以用 `archseries` 解析成员而不是 `setcode`。

## 快照 `environments/md-2026-09`

2026-09-23 抓取，`--since 2026-09-04`（公告 2026-08-31 发布、写明 2026-09-03 维护时实施；9 月 3 日的卡表仍有 2 份违反新表，4 日起为 0）。

| 项 | 数值 |
|----|------|
| 卡池 | 13 858 个卡密：YGOPRODECK 列出 13 865 张，去掉 7 张衍生物；2 张按卡名对应（Barrel Dragon 81480461 → 81480460；Mercurium the Living Quicksilver 的 YGOPRODECK 临时 id 101303084 → 22984000）；无法对应 0 张 |
| 禁限表 | 207 张：禁止 108、限制 73、准限制 26。masterduelmeta 的 231 条记录中含重复与 7 张异画（Apollousa、Droll & Lock Bird、Foolish Burial、Harpie's Feather Duster、Monster Reborn、Number 41: Bagooska、Ultimate Offering），折回原卡；无法对应 0 张；1 张不在卡池（Maliss \<Q\> Red Ransom：masterduelmeta 列为禁止，YGOPRODECK 未列入 MD 卡池）。**待人工校对** |
| Meta 窗口 | 2026-09-04 至 2026-09-23：848 份卡表，计入份额的 560 份，145 个卡组类型 |
| Meta 卡组 | 20 套，份额合计 71.7%：Dracotail 8.9%、Clown Crew 8.2%、Sky Striker 7.9%、Branded 6.1%、Resonators 5.4%、Kewl Tune 4.5%、Elfnote Kewl Tune 3.8%、Magistus Fairy Tail 3.4%、Yummy 2.9%、Lunalight 2.7%、Elfnote、Vanquish Soul K9、Radiant Typhoon Zoodiac、Ryzeal Mitsurugi、Maliss、Tearlaments、Blue-Eyes、HEROs、Odion、Orcust |
| 卡表合法性 | 窗口内 848 份卡表在生成的禁限表下全部合法；无法对应的卡 0 张。20 套代表卡表全部合法；卡池里的效果卡全部在 CardScripts 中有脚本；20 套各与 Dracotail 打一局（greedy 对 random，6 回合）无脚本错误、无未知消息 |
| Yugipedia | 10 932 张卡池卡有至少一项关系；各属性无法对应到卡片数据库的页面（多为 Rush / 动画卡或 cdb 尚未收录的新卡）：archetype_support 67、anti_support 6、archseries_related 23、archseries 104 |
| 大小 | 进 git 约 530 KB：`relations.json` 268 KB、`pool.json` 136 KB、`meta/` 84 KB，其余各数 KB |

## 代码

| 模块 | 作用 |
|------|------|
| `ygorl.data.fetch` | `Http`（限速、重试、User-Agent）、`write_raw` / `read_raw` / `provenance`（原始文件与 `.source.json`） |
| `ygorl.data.cardmap` | `CardMapper`：来源卡密 / 卡名 → 原卡密 |
| `ygorl.data.ygoprodeck` | `fetch_md_cards`、`parse_md_pool` |
| `ygorl.data.masterduelmeta` | `fetch_cards` / `fetch_articles` / `fetch_top_decks`；`parse_banlist`、`last_banlist_update`、`parse_top_decks`、`summarize_types`（份额与 medoid） |
| `ygorl.data.yugipedia` | `fetch_property`、`parse_property`、`relations` |
| `ygorl.data.build` | `BuildOptions`、`build`（抓取 → 解析 → 写入 → 校验）、校对报告 |

## 测试

`tests/test_data_sources.py`：解析器在测试里生成的小型原始文件上离线运行（真实卡密取自 `tests/decks`）：异画折叠、按卡名对应、衍生物、
无法对应的卡、禁限状态的各种写法与冲突、窗口过滤与权重、medoid 选择、Yugipedia 空结果与倒排；抓取器用假的 HTTP 对象测翻页、窗口截止、
SMW 偏移上限检测与重试 / `Retry-After`；离线构建端到端（校验、份额、人工修正、校对状态的保留与失效、参数沿用、手工替换的原始文件、
缺文件报错）、`ygorl env build / check` 命令行，以及仓库快照通过 T0.4 校验。真正联网的两个测试默认跳过，`YGORL_NETWORK_TESTS=1` 时运行。
