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
└── artifacts/              # 本版本下的产物（对局矩阵、模型、档案……），按需创建
```

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
  "sources": { "pool": "YGOPRODECK API, retrieved 2026-10-01" }
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
| `description`、`sources` | 否 | 说明与数据来源，原样保留在 `Environment.manifest` |

`rule_flags` 由 `rules.mode` 与 `extra_flags` 按位或得到，直接传给 `OCG_DuelOptions.flags`。
flag 数值来自 `ygorl.engine.constants`，该模块由 `tools/gen_constants.py` 从核心的 `ocgapi_constants.h`
生成，单测保证二者同步。MD 版 = `MR5` + MD 卡池 + MD 禁限表；TCG / OCG 版只换 flag、禁限表与卡池。

## `pool.json`

```json
{ "source": "YGOPRODECK", "retrieved": "2026-10-01",
  "cards": [89631139, 14558127, { "password": 46986414, "name": "Dark Magician" }] }
```

- `cards`：卡片密码（8 位官方 `password`，正整数）列表；元素可以是整数，也可以是带 `password` 键的对象
  （其余字段作为说明，加载时忽略）。
- 不允许重复，不允许为空。卡池是「本格式存在的卡」，禁限状态由禁限表决定。
- 来源：MD 卡池由 YGOPRODECK API（`format=master duel`）抓取生成；TCG / OCG 可由 BabelCDB 按 `ot` 字段导出（T5.1）。

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
TCG / OCG / GOAT 等表直接取自 `third_party/LFLists`；**MD 表没有上游**，由 masterduelmeta 数据生成后人工校对（T5.1）。

## `meta.json` 与 `meta/*.ydk`

```json
{ "source": "masterduelmeta.com", "retrieved": "2026-10-01",
  "decks": [
    { "name": "Snake-Eye", "file": "meta/snake-eye.ydk", "share": 0.18 },
    { "name": "Yubel",     "file": "meta/yubel.ydk",     "share": 0.11 }
  ] }
```

- `name` 唯一；`file` 是相对环境目录的路径，不能跳出目录；`share` 在 [0, 1]，总和 ≤ 1（剩余份额视为「其它」）。
- `.ydk` 为 YGOPro 通用格式（`#main` / `#extra` / `!side`，每行一个密码，一张一行），解析器为 `ygorl.cards.ydk`。
- 允许 `decks` 为空列表（例如尚未整理 meta 的新环境）。

## `artifacts/`

`Environment.artifact_path("matrix", "2026-10-02.json")` 返回 `artifacts/` 下的路径并创建父目录。
产物文件应包含 `Environment.stamp()`。大型产物（模型权重等）不进 git，只提交小型报告与矩阵。

已有的产物：

| 路径 | 产生者 | 内容 |
|------|--------|------|
| `artifacts/matrix/<name>.json` | `ygorl.eval.matchup.MetaGame.save(env=env, name=...)` | 对局胜率矩阵、Nash 混合、alpha-rank（格式见 [evaluation.md](evaluation.md)） |
