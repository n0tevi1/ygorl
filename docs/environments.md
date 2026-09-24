# 环境（Environment）规范

「环境」是一等公民配置：`Environment = {format, card_pool, banlist, rule_flags, meta_decks(with share), version}`
（见 [设计文档 §5.2](design/06-architecture.md)）。所有训练、评估、组牌产物都必须绑定环境版本。
代码入口：`ygorl.data.load_environment(path_or_version)`，返回不可变的 `ygorl.data.Environment`。

## 目录布局

```
environments/<version>/
├── environment.json        # 清单：版本、格式、规则 flag、玩家与构筑规则
├── pool.json               # 卡池：合法卡片密码列表
├── banlist.lflist.conf     # 禁限表（EDOPro 格式）
├── meta.json               # meta 牌组清单与占比
├── meta/*.ydk              # meta 牌组
├── relations.json          # 可选：Yugipedia 系列关系（ygorl env build 生成，见 data.md）
├── review/                 # 可选：构建与校对报告 report.md、禁限表人工修正 banlist-overrides.lflist.conf
├── raw/                    # 可选：抓取的原始文件，git 忽略（ygorl env build）
└── artifacts/              # 本版本下的产物（对局矩阵、模型、档案……），按需创建
```

MD 环境由 `ygorl env build md-YYYY-MM` 一键生成（抓取、解析、校验，见 [data.md](data.md)）；`ygorl env check <版本>` 做下面全部校验并打印摘要。
仓库里的快照：`environments/md-2026-09`（2026-09-23 抓取，禁限表已与 YGOPRODECK、Yugipedia 交叉核对，见 [data.md](data.md)）。可选文件不参与加载与 `fingerprint`。

前四个文件缺任何一个，加载时抛 `EnvironmentFileMissing`（同时是 `FileNotFoundError`），错误信息列出缺失文件名。
内容不合法抛 `EnvironmentConfigError`（`ValueError` 子类），信息包含文件路径与出错的键或行号。

`load_environment` 接受目录路径，或版本名；版本名在 `root` 参数、环境变量 `YGORL_ENVIRONMENTS`、
`./environments` 中依次查找。

## 版本号

- 格式：小写字母开头，`-` 分隔的小写字母、数字、点段，正则 `^[a-z][a-z0-9]*(?:-[a-z0-9.]+)*$`。
  约定为 `<format>-<YYYY>-<MM>[-<修订>]`，例如 `md-2026-10`、`tcg-2026-09`、`md-2026-10-b`。
- 必须等于目录名。
- 环境一经发布即视为不可变：禁限表、卡池、meta 变化时建新版本，而不是改旧目录（升级流程见 T6.3）。
- 加载时对四个必需文件和全部 meta 牌组计算 SHA-256 `fingerprint`。`Environment.stamp()` 返回
  `{"environment": <version>, "fingerprint": <sha256>}`，每个产物都应写入这一戳；读取产物时用
  `Environment.check_stamp(stamp)` 校验，版本不同或同版本内容被改过都会报错。

## `environment.json`

```json
{
  "version": "md-2026-10",
  "format": "md",
  "description": "Master Duel, 2026-10 banlist",
  "rules": { "mode": "MR5", "extra_flags": [] },
  "player": { "starting_lp": 8000, "starting_hand": 5, "draw_per_turn": 1 },
  "deck": { "main_min": 40, "main_max": 60, "extra_max": 15, "side_max": 15, "max_copies": 3 },
  "banlist_name": "2026.10 MD",
  "sources": { "pool": "YGOPRODECK API, retrieved 2026-10-01" },
  "review": { "banlist": { "status": "pending" } }
}
```

| 键 | 必需 | 含义 |
|----|------|------|
| `version` | 是 | 版本号，同目录名 |
| `format` | 是 | 格式标识：`md` / `tcg` / `ocg` / `speed` / `rush` / `goat` 等，自由字符串 |
| `rules.mode` | 否，默认 `MR5` | 核心规则预设：`MR1`–`MR5`、`SPEED`、`RUSH`、`GOAT`，对应 `DUEL_MODE_*` |
| `rules.extra_flags` | 否 | 追加的核心 flag 名，去掉 `DUEL_` 前缀，如 `TCG_SEGOC_NONPUBLIC` |
| `player` | 否 | 起始 LP、起手张数、每回合抽卡数；默认 8000 / 5 / 1 |
| `deck` | 否 | 构筑规则：主卡组 40–60、额外 ≤ 15、副卡组 ≤ 15、同名 ≤ 3 |
| `banlist_name` | 否 | 禁限表文件含多张表时选择哪一张（`!name`）；缺省取第一张 |
| `description`、`sources` | 否 | 说明与数据来源，原样保留在 `Environment.manifest`；`ygorl env build` 在 `sources` 里记原始文件出处（`raw`）、禁限更新公告、meta 窗口与构建参数（`build`） |
| `review` | 否 | 人工校对状态，如 `{"banlist": {"status": "pending" \| "reviewed", "by", "date"}}`（[data.md](data.md)） |

`rule_flags` 由 `rules.mode` 与 `extra_flags` 按位或得到，直接传给 `OCG_DuelOptions.flags`。
flag 数值来自 `ygorl.engine.constants`，该模块由 `tools/gen_constants.py` 从核心的 `ocgapi_constants.h`
生成，单测保证二者同步。MD 版 = `MR5` + MD 卡池 + MD 禁限表；TCG / OCG 版只换 flag、禁限表与卡池。

加载时的校验（违反任何一条都抛 `EnvironmentConfigError`，命令行显示为退出码 2 的错误，不会带着不合理的规则开局）：

- 类型：清单必须是 JSON 对象；`rules`、`player`、`deck` 缺省或为对象；`rules.mode` 是上表中的字符串；
  `rules.extra_flags` 是 flag 名字符串的**列表**（写成单个字符串会报 `rules.extra_flags must be a list`）；
  `description`、`banlist_name` 若给出必须是字符串；`player` / `deck` 只允许上表的键，值是非负整数且不超过 2^31−1。
- `player.starting_lp` ≥ 1（为 0 时开局即平局）。
- `player.starting_hand` ≤ `deck.main_min`，`player.draw_per_turn` ≤ `deck.main_min`：合法牌组至少有 `main_min` 张，
  起手或单次抽卡超过它会让双方在第一回合就抽空卡组。两者可以为 0。
- `deck.main_min` ≤ `deck.main_max`；`deck.max_copies` ≥ 1。
- 四个文件都必须是 UTF-8；非 UTF-8 的文件（包括禁限表与 meta `.ydk`）报错并指出文件。

## `pool.json`

```json
{ "source": "YGOPRODECK", "retrieved": "2026-10-01",
  "cards": [89631139, 14558127, { "password": 46986414, "name": "Dark Magician" }] }
```

- `cards`：卡片密码（8 位官方 `password`，1 到 99999999 的整数）列表；元素可以是整数，也可以是带 `password` 键的对象
  （其余字段作为说明，加载时忽略）。文件本身必须是 JSON 对象。
- 不允许重复，不允许为空。卡池是「本格式存在的卡」，禁限状态由禁限表决定。
- 来源：MD 卡池由 YGOPRODECK API（`format=master duel`）抓取生成，写原卡密（异画折回原卡）并去掉衍生物（[data.md](data.md)）；
  TCG / OCG 可由 BabelCDB 按 `ot` 字段导出（尚未实现）。

## `banlist.lflist.conf`

EDOPro / Project Ignis 格式，解析器为 `ygorl.cards.lflist`：

```
#[2026.10 MD]           注释（# 或 -- 开头）
!2026.10 MD             开始一张名为 "2026.10 MD" 的表
$whitelist              可选：未列出的卡一律禁止
14558127 1 --Ash Blossom & Joyous Spring
<password> <张数> [--注释]
```

张数：0 禁止、1 限制、2 准限制、3 不限（白名单用）。未列出的卡默认 3 张（白名单下为 0）。
环境内的禁限表按严格模式解析：同一张卡出现两次且张数不同、张数不在 0–3 都报错并给出行号。
TCG / OCG / GOAT 等表直接取自 `third_party/LFLists`；**MD 表没有上游**，由 masterduelmeta 数据生成后人工校对（流程见 [data.md](data.md)）。

## `meta.json` 与 `meta/*.ydk`

```json
{ "source": "masterduelmeta.com", "retrieved": "2026-10-01",
  "decks": [
    { "name": "Snake-Eye", "file": "meta/snake-eye.ydk", "share": 0.18 },
    { "name": "Yubel",     "file": "meta/yubel.ydk",     "share": 0.11 }
  ] }
```

- `name` 唯一；`file` 是相对环境目录的路径，不能跳出目录；`share` 在 [0, 1]，总和 ≤ 1（剩余份额视为「其它」）。
- `.ydk` 为 YGOPro 通用格式（`#main` / `#extra` / `!side`，每行一个密码，一张一行），解析器为 `ygorl.cards.ydk`；
  密码必须是 1 到 99999999 的整数，文件必须是 UTF-8。
- 允许 `decks` 为空列表（例如尚未整理 meta 的新环境）。
- **meta 卡组必须在本环境下合法**：卡池、禁限卡表、`deck` 规则，以及结构规则（密码在卡片数据库里存在、衍生物不能入组、
  额外卡组怪兽只能在额外卡组、同名卡含异画合计计数）。这项检查需要卡片数据库，所以只在给出时进行：
  `load_environment(path, cards=CardDB.load())` 或 `env.check_meta_decks(cards)`，不合法时 `EnvironmentConfigError` 列出每套不合法的
  meta 卡组（文件路径、名字）及其全部违规。命令行（`--env`）总是带卡片数据库加载，见 [cli.md](cli.md)；
  不带 `cards` 的 `load_environment(path)` 只做本页其余的格式校验（测试用的假卡池环境依赖这一点）。

## `artifacts/`

`Environment.artifact_path("matrix", "2026-10-02.json")` 返回 `artifacts/` 下的路径并创建父目录。
会离开 `artifacts/` 的路径（绝对路径、含 `..` 的部分、解析后指向外部的符号链接、空路径）一律抛 `EnvironmentConfigError`，
所以 `MetaGame.save(env=env, name="../x")` 之类的名字写不到环境目录之外。
产物文件应包含 `Environment.stamp()`。大型产物（模型权重等）不进 git，只提交小型报告与矩阵。

已有的产物：

| 路径 | 产生者 | 内容 |
|------|--------|------|
| `artifacts/matrix/<name>.json` | `ygorl.eval.matchup.MetaGame.save(env=env, name=...)` | 对局胜率矩阵、Nash 混合、alpha-rank（格式见 [evaluation.md](evaluation.md)） |
| `artifacts/meta_packages.json` | `tools/make_meta_packages.py <版本>` | 由 meta 卡组推导的引擎包（协同图召回检验用，规则见 [synergy.md](synergy.md#真实-meta-引擎包md-2026-09)），进 git |
| `artifacts/deck_corpus.json`、`artifacts/decks/*.ydk` | `tools/build_deck_corpus.py <版本> --fetch --smoke` | 牌组语料：全部历史卡表里在本环境下合法的每个类型至多 3 份真实卡表（含娱乐 / 活动卡组），训练牌组池与调卡组的起点（[data.md](data.md#牌组语料artifactsdeck_corpusjsonartifactsdecks)），进 git |
| `artifacts/synergy_graph.json.gz` | `tools/build_synergy_graph.py --environment <版本>` | 限制到本卡池的协同图（可加 `--relations`），约 0.7 MB，按需生成 |
