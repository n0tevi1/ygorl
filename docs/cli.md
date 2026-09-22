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

## 通用约定

- **牌组**：`.ydk` 文件，或目录（取其中全部 `*.ydk`，按文件名排序）。牌组名是文件名去掉扩展名，卡片以 8 位 `password` 标识。
- **agent 规格** `name[:arg]`：按名字从登记表构造（`ygorl.agents.registry`），当前有 `random`、`greedy`；`--help` 列出全部。
  `agent_factory(spec)` 返回可 pickle、带名字的 factory（`AgentSpec`），并行 Arena 的子进程按名字重建 agent，
  所以新 agent（如策略检查点）要在导入时 `register_agent`。未知规格报错并列出可用的名字。
- **环境** `--env PATH|VERSION`：目录，或 `$YGORL_ENVIRONMENTS`（默认 `./environments`）下的版本名（[environments.md](environments.md)）。
  给出时按环境的规则 flag、LP、起手、抽卡数对局，回放、Arena 报告、矩阵都带环境版本与指纹；不给时用 Master Rule 5 默认值、不绑定环境。
  `replay` / `branch` 不给 `--env` 时按回放记录的版本去环境根目录找，找不到报错并提示 `--env PATH`。
- **`--max-turns N`**（`duel` / `arena` / `matrix`）：回合上限，到达时判平（`reason=turn_limit`），默认 200。
- **输出文件**的父目录自动创建。
- **退出码**：0 成功；1 结果不健康（`replay --verify` 未到达录制的终局；`arena` / `matrix` 有对局抛异常，`arena` 另含 retry、未知消息）；
  2 用法错误（参数不合法、文件不存在、未知 agent、环境不符等），打印为 `ygorl <命令>: error: ...`。

## `ygorl duel`

```
ygorl duel A.ydk B.ydk [--seed S] [--first a|b] [--agent-a AGENT] [--agent-b AGENT] [--env PATH|VERSION]
                       [--max-turns N] [--save-replay PATH] [--yrpx PATH]
```

agent a 驾驶牌组 A，agent b 驾驶牌组 B（默认都是 `random`）。对局种子 `S`（默认 0）；agent a 的种子是 `S`，agent b 是 `S + 1`，
同样的参数必定打出同一局。`--first` 指定先攻方（默认 a）。`--save-replay` 写 `.json` / `.json.gz` 回放，`--yrpx` 导出 EDOPro 回放。

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
ygorl replay PATH [--verify] [--export-yrpx OUT] [--env PATH|VERSION]
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
- `--export-yrpx OUT`：导出 EDOPro `.yrpX`（格式与限制见 [replays.md](replays.md)），双方名字取牌组名。
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
（`--decks a,b` 从 `tests/decks` 按名字挑牌），种子与对阵顺序不变，[benchmarks.md](benchmarks.md) 的数字照旧可复现。

## `ygorl matrix`

```
ygorl matrix [DECK...] [--agent AGENT] [--games N] [--workers N] [--seed S] [--confidence C] [--alpha A]
                       [--population M] [--env PATH|VERSION] [--max-turns N] [--out PATH] [--name NAME]
```

- 同一个 agent（默认 `greedy`）驾驶双方，每两套牌打 `N` 局（向上取偶数，即 `pairs = ⌈N / 2⌉` 对配对种子，默认 20 局），
  镜像不打。每格种子由 `S` 与两套牌的名字决定，矩阵与牌组顺序无关（见 [evaluation.md](evaluation.md)）。牌组名（文件名）必须唯一，至少两套。
- 不给 `DECK` 而给 `--env` 时，用环境的 meta 卡组（名字取 `meta.json` 里的 `name`）。
- 输出胜率矩阵（行牌组对列牌组的胜率，平局算半胜）与每套牌的 Nash 混合概率、alpha-rank 质量；`--alpha`（默认 10）、`--population`（默认 50）是 alpha-rank 参数。
- **保存**：给 `--out` 写到该路径；否则给了 `--env` 就写到 `environments/<版本>/artifacts/matrix/<NAME>.json`（`--name` 默认取 agent 规格）；
  两者都没有时只打印不保存。文件格式见 [evaluation.md](evaluation.md) 的「产物格式」，`MetaGame.load(path, env=env)` 读回。

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

## 测试

- `tests/test_cli.py`：每个命令的输出与直接调用 API 的结果一致（`duel` 对 `Duel.run`，`arena` 对 `Arena.run_many` + `merge`，`matrix` 对 `build_matrix` + `analyze`），
  回放保存与 `.yrpX` 导出、`--verify` 成功与截断日志后的 `MISMATCH`、环境查找与报错、目录作牌组、`--vs`、环境 meta 卡组写入 `artifacts/matrix/`，以及各种用法错误。
- `tests/test_readme.py`：逐行执行 README「快速开始」里以 `uv run ygorl` 开头的示例（在临时目录中，`--games` 上限 2、`--rollouts` 上限 1），
  全部退出码 0，且每个子命令至少有一个示例。
