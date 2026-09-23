# 命令行（`ygorl`，T3.4）

`ygorl` 是 `pyproject.toml` 里登记的入口（`ygorl.cli:main`），`uv run ygorl <命令>` 运行。每个子命令一个模块
（`src/ygorl/commands/<命令>.py`，提供 `add_parser(subparsers)` 并设置 `func`），在 `ygorl.cli.COMMANDS` 里登记。
重依赖（C++ 核心、numpy、nashpy）只在命令执行时导入，`ygorl --help` 不加载规则核心（`test_help_does_not_load_the_engine`）。

| 命令 | 作用 | 底层 API |
|------|------|----------|
| `ygorl duel` | 打一局，打印胜者、终局原因、回合、LP、决策数；可存回放、导出 `.yrpX` | `Duel`（[engine.md](engine.md)）、`Replay`（[replays.md](replays.md)） |
| `ygorl replay` | 显示回放元数据；重新模拟并核对终局；导出 `.yrpX` | `Replay.load / play / to_yrpx` |
| `ygorl branch` | 从回放的某一步分叉，比较候选动作 | `fork`（[branching.md](branching.md)） |
| `ygorl arena` | 配对种子对局，agent a 对 agent b 的胜率与 Wilson 区间 | `Arena`、`merge`（[evaluation.md](evaluation.md)） |
| `ygorl matrix` | 牌组两两对局的胜率矩阵 + Nash 混合 + alpha-rank | `build_matrix`、`analyze`、`MetaGame.save` |
| `ygorl env` | `build`：抓取数据并生成 MD 环境；`check`：校验环境并打印摘要 | `ygorl.data.build.build`、`load_environment`（[data.md](data.md)） |

## 通用约定

- **牌组**：`.ydk` 文件，或目录（取其中全部 `*.ydk`，按文件名排序）。牌组名是文件名去掉扩展名，卡片以 8 位 `password` 标识
  （`.ydk` 里的 `password` 必须是 1 到 99999999 的整数，文件必须是 UTF-8，否则报错并指出文件与行号）。
- **牌组合法性检查**（`duel` / `arena` / `matrix`）：开局前检查每套牌，任何一套不合法就一局不打，退出码 2，
  打印 `ygorl <命令>: error: <文件>: deck '<名字>' is illegal ...` 并逐条列出违反的规则（`ygorl.cards.legality.validate_deck`）。
  - 给 `--env` 时按该环境检查：卡池（`pool.json`）、禁限卡表、`deck` 规则（主卡组张数上下限、额外 / 副卡组上限、同名卡上限），
    外加下面的结构规则。
  - 不给 `--env` 时只检查结构规则：`password` 在卡片数据库里存在、衍生物不能入组、额外卡组怪兽（融合 / 同调 / 超量 / 连接）
    只能在额外卡组且额外卡组只能放它们、主卡组 40–60 张、额外 / 副卡组各至多 15 张、同名卡（含异画）合计至多 3 张。
    空 `.ydk` 因主卡组 0 张而被拒。
  - 只有命令行做这项检查；库 API（`Duel(...)` 不传 `validate=True`、`Arena`、`build_matrix`）照常对任意牌组开局，便于测试与实验。
- **agent 规格** `name[:arg]`：按名字从登记表构造（`ygorl.agents.registry`），当前有 `random`、`greedy`、`policy:PATH[@greedy][@t=T]`（策略网络检查点，按温度 T 采样，默认 1；`@greedy` 取 argmax；`policy-greedy:PATH` 是 `policy:PATH@greedy` 的简写）。检查点可以是 PPO 训练的 checkpoint（[evaluation.md](evaluation.md)「策略检查点 agent」）或 BC 等导出的策略检查点（[bc.md](bc.md)），按文件的 `format` 字段区分；需要 `train` 可选依赖。`--help` 列出全部。
  `agent_factory(spec)` 返回可 pickle、带名字的 factory（`AgentSpec`），并行 Arena 的子进程按名字重建 agent，
  所以新 agent（如策略检查点）要在导入时 `register_agent`。未知规格报错并列出可用的名字。
- **环境** `--env PATH|VERSION`：目录，或 `$YGORL_ENVIRONMENTS`（默认 `./environments`）下的版本名（[environments.md](environments.md)）。
  给出时按环境的规则 flag、LP、起手、抽卡数对局，回放、Arena 报告、矩阵都带环境版本与指纹；不给时用 Master Rule 5 默认值、不绑定环境。
  `replay` / `branch` 不给 `--env` 时按回放记录的版本去环境根目录找，找不到报错并提示 `--env PATH`。
  命令行加载环境时一并用卡片数据库检查它的全部 meta 卡组（`load_environment(..., cards=...)`），不合法的 meta 卡组让任何命令以退出码 2
  拒绝该环境，错误里带 meta 卡组文件路径与违反的规则；环境清单的其它校验（LP、起手、类型等）见 [environments.md](environments.md)。
- **`--max-turns N`**（`duel` / `arena` / `matrix`）：回合上限，到达时 LP 高者胜、相等为平局（`reason=turn_limit`），默认 200。
- **输出文件**的父目录自动创建。
- **退出码**：0 成功；1 结果不健康（`replay --verify` 未到达录制的终局；`arena` / `matrix` 有对局抛异常，`arena` 另含 retry、未知消息）；
  2 用法或输入错误，打印为 `ygorl <命令>: error: ...`，不打印 traceback：参数不合法、文件不存在、未知 agent、环境不符、
  牌组不合法、`.ydk` / 环境文件格式错误（非 UTF-8、`password` 越界、字段类型不对、规则不合理），
  以及回放文件损坏（`.json.gz` 截断或不是 gzip、不是 JSON 对象、字段类型不对，如 `seed` 不是整数、`responses` 不是十六进制串、`decks` 结构不对）。

## `ygorl duel`

```
ygorl duel A.ydk B.ydk [--seed S] [--first a|b] [--agent-a AGENT] [--agent-b AGENT] [--env PATH|VERSION]
                       [--max-turns N] [--save-replay PATH] [--yrpx PATH [--yrpx-uncompressed]]
```

agent a 驾驶牌组 A，agent b 驾驶牌组 B（默认都是 `random`）。对局种子 `S`（默认 0）；agent a 的种子是 `S`，agent b 是 `S + 1`，
同样的参数必定打出同一局。`--first` 指定先攻方（默认 a）。`--save-replay` 写 `.json` / `.json.gz` 回放，`--yrpx` 导出 EDOPro 回放
（与 EDOPro 一样 LZMA 压缩，一局约 10–20 KB；加 `--yrpx-uncompressed` 写不压缩的版本，约 1–3 MB，EDOPro 两种都能读；不给 `--yrpx` 时用它报用法错误）。

```
$ uv run ygorl duel tests/decks/snake_eye.ydk tests/decks/kashtira.ydk --seed 1 --agent-a greedy --save-replay out/game.json.gz --yrpx out/game.yrpX
duel       snake_eye (greedy) vs kashtira (random): seed 1, a moves first, environment none
winner     a (snake_eye, greedy)
reason     win (lp)
turns      11
lp         a 8000, b -2000
decisions  495
replay     out/game.json.gz
yrpX       out/game.yrpX
```

`winner` 为 `a` / `b` / `draw`；`reason` 为 `DuelResult.reason`（`win`、`turn_limit`、`decision_limit`、`error` 等），
`win` 时括号里是 `MSG_WIN` 原因：`lp`（生命值归零）、`deck_out`（卡组抽空）或 `card effect 0x..`（特殊胜利）。
有 retry、未知 / 无法解码的消息、脚本错误时多一行 `warnings`。

## `ygorl replay`

```
ygorl replay PATH [--verify] [--export-yrpx OUT [--yrpx-uncompressed]] [--env PATH|VERSION]
```

只给路径时打印元数据，不需要环境也不跑核心：

```
$ uv run ygorl replay out/game.json.gz --verify --export-yrpx out/again.yrpX
replay     out/game.json.gz (ygorl-replay v1, ocgcore 11.0)
environment none
seed       1 (a moves first)
rules      flags 0x2e800, lp 8000, hand 5, draw 1, max_turns 200, max_decisions 20000, curriculum full
deck_a     snake_eye: 40 main, 15 extra, 0 side
deck_b     kashtira: 40 main, 15 extra, 0 side
responses  493 (0 agent steps recorded)
recorded   winner=a reason=win (lp) turns=11 lp=8000/-2000 decisions=495
verify     ok: replayed to the recorded end, winner=a reason=win (lp) turns=11 lp=8000/-2000 responses=493/493
yrpX       out/again.yrpX
```

- `--verify`：按应答日志重新模拟（`Replay.play`），核对胜者、终局原因、`MSG_WIN` 原因、回合、LP 与录制的 `result` 一致，
  且应答日志恰好用完（重放中途要不到应答时终局原因是 `log_exhausted`）。不一致时打印 `verify     MISMATCH (<不一致的字段>): ...`，退出码 1。
- `--export-yrpx OUT`：导出 EDOPro `.yrpX`（格式与限制见 [replays.md](replays.md)），双方名字取牌组名；默认 LZMA 压缩，`--yrpx-uncompressed` 不压缩（同 `duel`）。
- 回放绑定了环境时，`--verify` / `--export-yrpx` 需要同一版本、同一指纹的环境，否则退出码 2 并说明原因。

## `ygorl branch`

见 [branching.md](branching.md) 的「命令行」一节。用 `ygorl duel --save-replay` 录一局即可分叉：

```bash
uv run ygorl branch out/game.json.gz --at 12 --try all --policy greedy --seed 1
```

## `ygorl arena`

```
ygorl arena DECK... [--vs DECK...] [--agent-a AGENT] [--agent-b AGENT] [--games N] [--workers N] [--seed S]
                    [--confidence C] [--env PATH|VERSION] [--max-turns N] [--out PATH]
```

- **对阵**：不给 `--vs` 时双方用同一组牌，agent a 驾驶每套牌对 agent b 驾驶的每套牌（全部有序组合，含镜像）；
  给 `--vs` 时 agent a 驾驶 `DECK...`、agent b 驾驶 `--vs` 的牌（笛卡尔积）。默认 `greedy` 对 `random`。
- **局数**：`--games`（默认 200）平均分到各对阵，每个对阵 `2 × pairs` 局，`pairs = max(1, round(N / (2 × 对阵数)))`，
  所以实际局数可能与 `N` 略有出入（第一行打印实际值）。
- **种子**：第 `i` 套（a 侧）对第 `j` 套（b 侧）用 `derive_seed(S, i, j)` 作为该对阵的 Arena 种子，其下的配对规则见 [evaluation.md](evaluation.md) 的 Arena 一节。
  结果与 `--workers` 无关。
- **输出**：汇总报告（agent a 视角的胜/负/平、胜率与 `--confidence` 水平的 Wilson 区间、先攻 / 后攻分项、先攻方胜率、错误与 retry）、终局原因计数、平均回合；
  a 侧或 b 侧不止一套牌时，按 agent a 驾驶的牌、按对手的牌分别列出胜率。`--out` 写汇总报告 JSON（`ArenaReport.to_dict()`，含每局记录）。

```
$ uv run ygorl arena tests/decks/snake_eye.ydk --vs tests/decks/kashtira.ydk --games 20 --workers 2
1 deck pairing x 20 games = 20 games in 4.1s with 2 workers (environment: none)
greedy[snake_eye] vs random[kashtira]: 20 games, W/L/D 9/11/0, win rate 0.450 (95% CI 0.258-0.658); first 0.600, second 0.300; first player wins 0.650; errors 0, retries 0
reasons: {'win': 20}; mean turns 31.1
```

T3.2 的基准 `uv run python tools/arena.py --games 2000 --workers 2` 现在是 `ygorl arena tests/decks --games 2000 --workers 2` 的薄包装
（`--decks a,b` 从 `tests/decks` 按名字挑牌），种子与对阵顺序不变。注意 [benchmarks.md](benchmarks.md) 的数字测于 commit `f621a5f`，之后 T2.3 修正了多选应答（合法动作列表有细微变化），同一种子下个别对局会不同，统计结论需要重测才能确认。本页的示例输出同样来自当时的代码，逐字复现不作保证。

## `ygorl matrix`

```
ygorl matrix [DECK...] [--agent AGENT] [--games N] [--workers N] [--seed S] [--confidence C] [--alpha A]
                       [--population M] [--env PATH|VERSION] [--max-turns N] [--out PATH] [--name NAME]
```

- 同一个 agent（默认 `greedy`）驾驶双方，每两套牌打 `N` 局（向上取偶数，即 `pairs = ⌈N / 2⌉` 对配对种子，默认 20 局），
  镜像不打。每格种子由 `S` 与两套牌的名字决定，矩阵与牌组顺序无关（见 [evaluation.md](evaluation.md)）。牌组名（文件名）必须唯一，至少两套。
- 不给 `DECK` 而给 `--env` 时，用环境的 meta 卡组（名字取 `meta.json` 里的 `name`）。
- 输出胜率矩阵（行牌组对列牌组的胜率，平局算半胜）与每套牌的 Nash 混合概率、alpha-rank 质量；`--alpha`（默认 10）、`--population`（默认 50）是 alpha-rank 参数。
- **保存**：给 `--out` 写到该路径；否则给了 `--env` 就写到 `environments/<版本>/artifacts/matrix/<NAME>.json`（`--name` 默认取 agent 规格，
  其中字母、数字、`.`、`_`、`-` 以外的字符换成 `-`）；两者都没有时只打印不保存。`--name` 必须是单纯的文件名：只含字母、数字、`.`、`_`、`-`，
  不以 `.` 或 `-` 开头（因此不能含路径分隔符或 `..`），否则开局前退出码 2；`Environment.artifact_path` 本身也拒绝任何离开 `artifacts/` 的路径
  （绝对路径、`..`、指向外部的符号链接）。文件格式见 [evaluation.md](evaluation.md) 的「产物格式」，`MetaGame.load(path, env=env)` 读回。

```
$ uv run ygorl matrix tests/decks/snake_eye.ydk tests/decks/kashtira.ydk tests/decks/yubel.ydk --games 10 --workers 2 --out out/matrix.json
3 decks, 3 deck pairs x 10 games = 30 games in 5.9s with 2 workers; agent greedy, seed 0 (environment: none)
win rate of the row deck against the column deck (draws count half); alpha-rank alpha 10, population 50

deck       snake_eye  kashtira  yubel   nash  alpha_rank
snake_eye          -     0.300  0.400  0.000       0.000
kashtira       0.700         -  0.800  1.000       1.000
yubel          0.600     0.200      -  0.000       0.000

written to out/matrix.json
```

每格 10 局时区间很宽，上例只演示格式；比较牌组强弱需要足够的局数（看 JSON 里每格的 `ci_low` / `ci_high`）。

## `ygorl env`

```
ygorl env build VERSION [--root DIR] [--raw DIR] [--offline | --refresh] [--since YYYY-MM-DD] [--min-share F]
                        [--max-decks N] [--no-relations] [--reviewed-by NAME]
ygorl env check PATH|VERSION
```

- `build`：从 YGOPRODECK、masterduelmeta、Yugipedia 抓取缺失的原始文件到 `<环境>/raw/`，生成 `environments/<VERSION>/` 并按 T0.4 校验
  （含 meta 卡组合法性）；版本号形如 `md-YYYY-MM[-修订]`。来源、参数、校对流程见 [data.md](data.md)。输出卡池大小、禁限张数、meta 份额与警告
  （禁限表待校对、无法对应的卡、建议推迟的 meta 窗口起点）。抓取失败、原始文件缺失（`--offline`）、结果校验不过时退出码 2。
- `check`：带卡片数据库加载环境（与 `--env` 相同的校验），打印版本、`fingerprint` 前缀、卡池大小、禁限张数与校对状态、每套 meta 的份额。

```
$ uv run ygorl env check md-2026-09
environment md-2026-09 (md) at /path/to/ygorl/environments/md-2026-09
fingerprint 65ca28f79233e73d
pool        13858 cards
banlist     2026.09 MD: 108 forbidden, 73 limited, 26 semi-limited (review reviewed)
meta        20 decks, share 71.7%, all legal
    8.9%  Dracotail
    ...
```

## 测试

- `tests/test_cli.py`：每个命令的输出与直接调用 API 的结果一致（`duel` 对 `Duel.run`，`arena` 对 `Arena.run_many` + `merge`，`matrix` 对 `build_matrix` + `analyze`），
  回放保存与 `.yrpX` 导出（默认压缩与 `--yrpx-uncompressed`）、`--verify` 成功与截断日志后的 `MISMATCH`、环境查找与报错、目录作牌组、`--vs`、环境 meta 卡组写入 `artifacts/matrix/`，以及各种用法错误；
  不合法牌组（未知 `password`、空文件、4 张同名、额外卡组怪兽在主卡组或反之、超过 60 张、环境禁止或卡池外的卡、不合法的 meta 卡组）在三个命令下都以退出码 2 拒绝，
  损坏的回放 / `.ydk` / 环境文件与不合理的环境规则给出干净的错误，`--name` 不能带路径。
- `tests/test_data_sources.py`：`ygorl env build --offline`（小型原始文件）与 `ygorl env check`，以及用法错误（非 `md-` 版本、`--offline` 与 `--refresh` 同用）。
- `tests/test_readme.py`：逐行执行 README「快速开始」里以 `uv run ygorl` 开头的示例（在临时目录中，`tests/decks` 与 `environments` 链接到仓库，`--games` 上限 2、`--rollouts` 上限 1），
  全部退出码 0，且每个子命令至少有一个示例。
