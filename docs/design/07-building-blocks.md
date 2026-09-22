# 引擎与数据：可复用资源清单

> 属于 [ygorl 设计文档](README.md)。状态：已评审通过（2026-09-22）。

- **edo9300/ygopro-core**：`OCG_CreateDuel / OCG_DuelNewCard / OCG_LoadScript / OCG_StartDuel / OCG_DuelProcess / OCG_DuelGetMessage / OCG_DuelSetResponse / OCG_DuelQuery*`，`OCG_VERSION 11.0`。
- **ProjectIgnis/CardScripts**（Lua 5.4）、**BabelCDB**（`cards.cdb`：`datas(id, ot, alias, setcode, type, atk, def, level, race, attribute, category)` + `texts(id, name, desc, str1..16)`；type/attribute/race 位标志，level 打包灵摆刻度，Link 的 def 存连接标记，setcode 打包 4 个 16 位系列码；`str1..16` 即效果级描述串）、**LFLists**（`.lflist.conf`：`!name` / `code count` / `$whitelist`）。
- **YGOPRODECK v7 API**：`cardinfo.php?format=master duel` 得 MD 卡池，`banlist_info`、`archetype`、`misc_info.md_rarity`；限速 20 req/s，图片须自托管。
- **masterduelmeta** 非官方 JSON `/api/v1/top-decks?limit=0`（`deckType.name`、`main/extra[{card.name, amount}]`）→ MD meta 卡表与占比；DawnbrandBots/deck-transfer-for-master-duel 做 MD ↔ `.ydk`。Konami 每次维护发布的**单卡**使用率/胜率差（masterduelmeta 转载）可做卡级先验。
- **Yugipedia SMW**（MediaWiki `ask` API）：人工维护的 `Archetype support / Anti-support / Archseries related` 关系。
- **ygo-combo-solver（96jonesa）**：ocgcore 上的展开线搜索器（NRPA + Levin 树搜索 + Iterated-Width 剪枝），写监视 arena 快照，`--no-ref --target` 从牌组 + 目标场面搜索，`--fire` 强制手坑，`--adapt` 学一个跨牌组的选择策略，所有线以新鲜重放验证并输出 `.yrp`。需核实其核心版本与 edo9300 的兼容性。
- `.ydk` = `#main / #extra / !side` 三段卡密；`.yrp/.yrpX` 回放解析器：ygoreplay/cliff、ghlin/ygopro-replay-inflate。**无公开 YGO 人类回放语料。**
