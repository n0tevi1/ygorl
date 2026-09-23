# md-2026-09 禁限表交叉核对

手写记录（`ygorl env build` 不覆盖本文件，生成的 [report.md](report.md) 链接到这里）。`banlist.lflist.conf` 由 masterduelmeta
的 `banStatus` 生成；这里用两个与 masterduelmeta 无关的来源逐卡（按卡密，异画折回原卡）核对，代替游戏内人工核对。

- 核对日期：2026-09-23；核对人：Claude（`--reviewed-by "cross-checked vs Yugipedia+YGOPRODECK (Claude)"`）
- 工具：`uv run python tools/crosscheck_banlist.py md-2026-09 --date 2026-09-03 --save environments/md-2026-09/review/crosscheck`；
  离线复现：`--from environments/md-2026-09/review/crosscheck`（本目录 `crosscheck/` 保存了原始响应及 `.source.json` 出处）。

## 来源

| 来源 | URL | 抓取时间（UTC） | 生效日期 | 禁止 / 限制 / 准限制 |
|---|---|---|---|---|
| 本环境（masterduelmeta `banStatus`） | https://www.masterduelmeta.com/api/v1/cards | 2026-09-23 00:45 | 2026-09-03（公告 2026-08-30） | 108 / 73 / 26 |
| YGOPRODECK 禁限表 API（https://ygoprodeck.com/banlist/ 的数据） | https://ygoprodeck.com/api/banlist/getBanList.php?list=Master%20Duel | 2026-09-23 03:03 | 2026-09-03 | 108 / 73 / 26 |
| Yugipedia「September 2026 Lists (Master Duel)」 | https://yugipedia.com/wiki/September_2026_Lists_(Master_Duel)（经 `api.php?action=parse&prop=wikitext`） | 2026-09-23 03:03 | 2026-09-03 | 108 / 73 / 26 |

- YGOPRODECK 以卡密给出，207 条中 204 条直接命中 BabelCDB，3 条是异画（Ultimate Offering 80604092、Harpie's Feather Duster
  18144507、Monster Reborn 83764719）折回原卡。YGOPRODECK `cardinfo.php` 的 `banlist_info` 仍只有 TCG / OCG / GOAT 字段，
  MD 表只在这个禁限表 API 里。
- Yugipedia 只给卡名：207 张受限卡全部按卡名唯一对应，其中 4 张 Maliss 卡名用全角 `＜＞`，经宽松匹配对应；另有 1 条
  「Tearlaments Havnis; Unlimited」（本次解除限制）不计入。
- 两个来源自身一致，也与 masterduelmeta 公告的变化一致：与 YGOPRODECK 2026-08-05 表相比变化 4 张——Called by the Grave
  限制 → 禁止，Dracotail Mululu、Ketu Dracotail 无限制 → 准限制，Tearlaments Havnis 限制 → 无限制。
- 是否有更新的表：YGOPRODECK `getBanListDates.php` 最新的 Master Duel 日期为 2026-09-03；Yugipedia 页面的 `next` 为空；
  masterduelmeta 2026-08-30 之后的文章里没有新的 MD 禁限公告（9 月 20 日的两篇是 TCG 9/21 表与 OCG 10/1 表，不适用于 MD）。
  截至 2026-09-23，2026-09-03 表仍是现行表。

## 差异

| 卡密 | 卡名 | 本环境 | YGOPRODECK | Yugipedia | 处理 |
|---|---|---|---|---|---|
| — | 无 | — | — | — | — |

三方 207 张完全一致（禁止 108、限制 73、准限制 26），无需在 `banlist-overrides.lflist.conf` 写修正；
`banlist.lflist.conf` 内容不变，只把 `environment.json` 的 `review.banlist` 记为已校对。

## 附注

- 68059897 Maliss <Q> Red Ransom 三方都是禁止，但 YGOPRODECK `format=master duel` 卡池没有列出它（报告里标「不在卡池」）。
  这是卡池来源的缺漏，不影响合法性（禁止卡本来就不能放进卡组）。
- 旧数据的小出入：个别新闻站写 Called by the Grave 是「无限制 → 禁止」，Yugipedia 与 YGOPRODECK 历史表都记为「限制 → 禁止」；
  只影响变化说明，不影响现行状态。
- 本次校对改动了 `environment.json`，环境 `fingerprint` 随之变化（e621b477… → 65ca28f7…）；`artifacts/meta_packages.json`
  已用 `tools/make_meta_packages.py md-2026-09` 重新生成（内容不变，只换了环境戳）。
