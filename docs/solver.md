# 求解器示范集（`ygorl.solver`，T4a.1）

用 [ygo-combo-solver](https://github.com/96jonesa/ygo-combo-solver)（AGPL-3.0，固定提交 `e0c7221`）为 meta 牌组的起手求解展开线，
把每条线在**我们的核心**里重新跑一遍、逐个应答反解成我们的动作下标，验证通过的才写进示范集，供 BC 预热（T4a.2）与起手分析（T5.6）使用。
决策依据见 [spikes/combo-solver.md](spikes/combo-solver.md)：封装二进制、不移植搜索。

```
tools/build_combo_solver.sh     拉取求解器与 EDOPro 的 LZMA 源码，用我们的核心编译 build/combo-solver/bin/combosolver
ygorl.solver.combo_solver       找二进制、建工作目录、组装参数、带超时运行、解析 @event 与输出文件名
ygorl.solver.targets            目标场面语法（卡片密码）与终局场面检查
ygorl.solver.demo               线 → 新鲜重放 → 动作下标（验证）；示范集记录（JSONL）；iter_steps 重放出 (DecisionPoint, 动作)
ygorl.solver.batch              起手抽样、模板、单手求解（含 --fire 变体）、断点续跑与汇总
tools/solve_openings.py         批量驱动：多进程、每手时间预算、--fire 变体、可续跑
tools/verify_demos.py           复验示范集文件：每条线从零重放（不需要求解器）
ygorl.engine.replay             读取 .yrp / .yrpX（含 LZMA）、Replay.from_yrp、显式核心种子（seed_words）
```

## 构建

```bash
tools/build_combo_solver.sh          # 首次约 40 秒（4 核，含拉取）；之后无改动时立即返回；最后一行输出二进制路径
```

依赖：git、python3、patch、g++（C++17）以及 sqlite3 开发包（Debian/Ubuntu：`apt install libsqlite3-dev`），安装方式同时写在 README「快速开始」。
求解器源码**不进入本仓库**：脚本在构建目录（默认 `build/combo-solver/`，git 已忽略，可用 `YGORL_SOLVER_BUILD_DIR` 改）里按固定提交拉取：

| 输入 | 来源 |
|------|------|
| 求解器 | `96jonesa/ygo-combo-solver` @ `e0c7221802a2`（浅克隆该提交） |
| LZMA | `edo9300/edopro` @ `c250b6ab9beb` 的 `gframe/lzma`（稀疏检出） |
| 规则核心 | `third_party/ygopro-core`（含 `lua/src`）的副本 |

核心按以下顺序打补丁：我们的 `0001`（确定性遍历顺序）、`0002`（Lua 字符串哈希种子）；**不打** `0003`（我们的 Lua 分配器钩子 `ygorl_lua_alloc`，求解器自带分配器钩子）；
然后是求解器的 5 个补丁——直接取自固定提交的 `tools/fetch_solver_deps.sh` 中的 Python 段（按锚点替换：种子 + 分配器钩子、`luaL_newstate`、`OCG_DuelQueryProcessorState` 及其声明、`<algorithm>`）。
其中种子补丁与我们的 `0002` 相同（重复的同值宏定义合法）。补丁 `0001` 必须保留：它改变消息顺序，求解器与我们的核心必须逐步一致，求解器写出的线才能在我们这边重放。

编译：Lua 按 C++ 编译并强制包含 `luaconf-customize.h`（排除 `lua.c`、`luac.c` 等与 premake 相同的文件）；核心 `-fno-rtti`；LZMA 以 C 编译、`-D_7ZIP_ST`；
求解器源码在 GCC 下缺若干头文件，加 `-include cmath -include cstring -include algorithm -include cstdint`；全部 `-O2 -ffp-contract=off`（与求解器 macOS 构建相同，避免 FMA 改变 Lua 浮点）；
链接 `-lsqlite3 -lpthread`，有 ccache 时自动使用。脚本对「求解器提交、EDOPro 提交、核心提交与未提交改动、Lua 提交、补丁内容、编译器版本、脚本本身」取哈希，全部不变时直接复用二进制。

二进制查找顺序（`find_solver`）：显式路径 → `$YGORL_COMBO_SOLVER` → `build/combo-solver/bin/combosolver` → `PATH`。找不到时报错并提示构建命令。

## 一次求解

```python
from pathlib import Path
from ygorl.solver import HandJob, solve_hand, solve_fire, iter_steps

job = HandJob(deck_path=Path("tests/decks/labrynth.ydk"), hand_index=0, hand_seed=7,
              targets=("1225009", "5380979@szone:fd"), workdir=Path("out/solver/wd"), scratch=Path("out/solver/h0"),
              solve_ms=20_000)
demo = solve_hand(job)              # Demonstration：status、hand、lines（已验证）……
fire = solve_fire(job, demo, 14558127)   # --fire 变体：对手手里有 Ash Blossom
for point, action in iter_steps(demo, 0):
    ...                             # point.actions[action] 即示范动作；观测由 ygorl.env.encoding 从 point 编码
```

`solve_hand` 的步骤：

1. **起手**：`sample_hand(deck, hand_seed)` 用主机洗牌（`shuffle_deck`，与对局相同）打乱主卡组，取顶上 5 张（核心从列表末尾抽卡）；也可以用 `hand=` 直接给定。
   打乱后的顺序写成本手的 `.ydk` 交给求解器，求解器把起手放到列表末尾、其余保持原顺序，所以剩余卡组也是随机顺序。
2. **模板**（`make_template`）：一个不含应答的 `.yrpX`，只提供规则 flag、LP / 起手 / 抽卡数（来自环境或默认 MR5）、4 个核心种子字（由 `hand_seed` 展开）和对手。
   对手是 40 张 Blue-Eyes White Dragon（89631139，通常怪兽，没有任何效果），所以单人展开时对手只会放弃连锁。求解器以 `--no-ref` 模式把它当作对局模板，换上我们的卡组与起手，
   并给核心加上 `DUEL_PSEUDO_SHUFFLE`（开局后核心不再洗卡组）。
3. **工作目录**（`Workdir.create`）：`cards.cdb` 链接到 `third_party/BabelCDB/cards.cdb`；脚本按 `ygorl.paths.script_directories()` 的优先级以 `--scriptdir` 传入。
   求解器会把每个 `--scriptdir` 的直接子目录也加进来，所以 CardScripts 根目录以「只含根目录 `*.lua` 的链接视图」传入，其后依次是 `official`、`pre-release`、`pre-errata`、`goat`、`rush`、`skill`、`unofficial`，与我们的查找顺序完全一致。
4. **运行**：`combosolver template.yrpX --workdir … --scriptdir … --no-ref --deck hand.ydk --hand a|b|c|d|e --target … --solve-ms N --threads T [--seed S] --max-written K --json --outdir out/`。
   墙钟超时默认 `solve_ms + 120 s`，超时先发 SIGINT（求解器会写出已找到的线），20 秒后仍未退出则 kill。
5. **收集**：输出文件 `solution_NN_bB_aA[_alt].yrp`（按求解器评分排序：先少烧卡 B，再少动作 A）逐个转换、验证（见下节），保留前 `lines` 条互不相同的通过者；失败的记入 `rejected`。

状态 `status`：`solved`（至少一条线通过验证）、`unsolved`（预算内没找到线，即「卡手」或预算不足）、`unverified`（求解器写了线但**没有一条**通过我们的重放——这是一致性问题，批量驱动以退出码 1 报告）、
`error`（参数、目标不在卡组、求解器异常退出等，`error` 字段给出求解器报告里的 `!!` 行）；`--fire` 记录另有 `no_window`（这条线上手坑没有可发动的时点）与 `skipped`（基础起手无解）。
未解出时 `solver.best_placed / target_cards` 记录搜索最接近时放上了几张目标卡。

## 目标场面

目标是卡片密码列表，语法为求解器 `--target` 的子集：`password[@zone[:fd]]`（不接受卡名，卡片一律以密码标识）。

| zone | 含义（与求解器的判定一致，`search.cpp` 的 `BoardKey`） |
|------|------|
| `atk`（默认）/ `def` / `mzone` | 在怪兽区；只看表侧 / 里侧（`:fd`），**不区分攻守表示** |
| `szone` | 在魔陷区（含灵摆区、场地区）；`:fd` 为盖放 |
| `hand` / `grave` / `banished` | 在手牌 / 墓地 / 除外区，只计数 |

`field` 与 `extra` 被拒绝（求解器同样拒绝）。重复写同一张卡表示需要多张。异画卡按原卡（`CardDB.canonical`）计。
求解前先检查目标卡都在卡组（主 + 额外）里且张数足够，否则直接记为 `error`。

测试牌组的目标写在 [`tests/decks/solver_targets.json`](../tests/decks/solver_targets.json)（按牌组名，另含 `--fire` 用的手坑列表）：

| 牌组 | 目标 | 说明 |
|------|------|------|
| branded_despia | 87746184 | Albion the Branded Dragon（融合） |
| fiendsmith_ryzeal | 34909328 | Ryzeal Detonator（4 阶超量） |
| kashtira | 73542331 | Kashtira Shangri-Ira（7 阶超量） |
| labrynth | 1225009 + 5380979@szone:fd | Arianna the Labrynth Servant 在怪兽区 + 盖放 Welcome Labrynth |
| purrely | 52645235 | Epurrely Happiness（1 阶超量） |
| snake_eye | 48452496 | Snake-Eyes Flamberge Dragon（与 spike 相同的目标） |
| tearlaments | 92731385 | Tearlaments Kitkallos（融合） |
| tenpai | 82570174 | Sangenpai Bident Dragion（同调） |
| voiceless_voice | 10774240 | Skull Guardian, Protector of the Voiceless Voice（仪式） |
| yubel | 80453041 | Phantom of Yubel（融合） |

这些是「首回合一个可达的代表性展开终点」，不是最优终场；换成真实 meta 牌组时按环境的 meta 卡表另写目标文件（`--targets`）。

## 转换与验证（`convert_line`）

一条线是求解器 `yrp1` 里的应答序列。转换即验证：

1. 用文件自己的 4 个核心种子字、规则 flag、LP/起手/抽卡数和**按加载顺序**的双方卡组建局（`Replay.from_yrp`：`shuffle_decks=False`，`seed_words` 直接喂给核心），用 `DuelSession` 逐步推进。
2. 对每个应答：先规范化编码（`canonical_response`：核心接受 4 种选卡编码——u32/u16/u8 下标表与位图，求解器发 u8 下标表 mode 2，我们的动作模型总发 u32 的 mode 0，核心对两者的解析结果相同），
   再用 `ygorl.engine.branch.actions_for_response` 在决策状态的副本上搜索出产生这一应答的动作下标序列（多选拆步后，一个应答对应多步），逐步喂给引擎。
   找不到动作序列、对局提前结束、核心回 `MSG_RETRY`，都记为该线失败。
3. 求解器的应答用完后，主机以被动选择**补完本回合**（结束阶段、放弃连锁、否、取消、完成，没有被动选项时取第一个动作），直到第 2 回合的第一个决策。
   这样示范也包含「展开完毕后结束回合」这一步，终局场面是回合结束后的场面；`solver_responses` / `solver_steps` 标出哪些来自求解器。
4. 取双方场面摘要（`board_summary`：怪兽区、魔陷区含表示形式与超量素材、手牌、墓地、除外、额外卡组表侧、卡组张数、LP），检查目标卡全部在场（`board_summary_missing`），缺卡即失败。

因此每条写入示范集的线都满足：全部应答被消费、`MSG_RETRY` 为 0、终局场面含目标卡；且动作下标能逐字节重现应答（`iter_steps` 结束时再核对一次）。

## `--fire` 手坑变体（`solve_fire`）

对已解出的起手，取最佳示范线（已补完到回合结束），导出为 `.yrpX` 作为求解器的参考线，运行
`combosolver line.yrpX --fire <手坑密码> --fire-bake --fire-ms N`：求解器把手坑加进对手手牌，在这条线上**每个**对手能发动它的时点让对手真的发动（每个时点一次），
再从发动后的局面搜索重新到达同一终场（求解器按参考线终场的怪兽区与魔陷区整体判定：卡、表侧/里侧、素材与指示物；我们的验证仍只检查目标卡）。`--fire-bake` 把这张卡写进输出回放的头部（追加到对手卡组末尾，使它被抽进对手起手），
所以输出的线不需要额外参数就能重放；转换时对手卡组里就有这张手坑，对手发动它的那一步也是示范的一部分（`players` 为 1 的步骤）。

每个手坑一条记录（`variant = "fire"`，`fire = 密码`），`solver.windows` / `solver.converted` 为可发动时点数与成功恢复的时点数；输出的每条线对应某个成功恢复的时点。

注意（求解器行为）：`--fire` 在参考线「第 `target_at` 个应答之后、尚未处理」时取目标场面，如果参考线恰好停在最后一次召唤的应答上（求解器自己的输出就是这样），
取到的是空场面，随后的对齐检查失败（「discovery pass does not reach the board」）。先把线补完到回合结束再交给 `--fire` 可以避开这个问题，这也是转换默认补完回合的原因之一。

## 示范集格式（JSONL）

一个文件，一行一条记录（牌组 × 起手 × 变体），`format = "ygorl-demo"`，`format_version = 1`：

| 键 | 含义 |
|----|------|
| `environment` | `{"version", "fingerprint"}`；无环境时为 `null`。续跑时文件里的环境必须与本次一致 |
| `engine` | `{"ocgcore": [major, minor]}` |
| `deck` | `{"name", "main", "extra"}`：给定的卡组（抽起手之前） |
| `hand_index`, `hand_seed`, `hand` | 起手编号、洗牌种子、起手 5 张的密码 |
| `variant`, `fire` | `"plain"`（单人展开）或 `"fire"`（对手持有并发动手坑 `fire`） |
| `targets` | 目标卡，`password@zone[:fd]` |
| `status`, `error` | 见上文「一次求解」 |
| `start` | 线的起始对局：`core_seed`（4 个 u64）、`rule_flags`（含 `DUEL_PSEUDO_SHUFFLE`）、`player`（LP/起手/抽卡）、`decks.a / decks.b` 的 `main`、`extra`（**加载顺序**，a 为引擎玩家 0、先攻）；无解时为 `null` |
| `lines[]` | 已验证的线，按求解器评分排序 |
| `lines[].responses` | 每个应答（十六进制，我们的编码），求解器部分 + 补完回合部分 |
| `lines[].actions`, `lines[].players` | 每一步的动作下标与行动的引擎玩家（多选拆成多步） |
| `lines[].solver_responses`, `lines[].solver_steps` | 前多少个应答 / 步来自求解器 |
| `lines[].board` | 终局场面摘要（`turn`、双方 `lp`、`mzone`、`szone`、`hand`、`grave`、`banished`、`extra_faceup`、`deck_count`、`extra_count`） |
| `lines[].score` | 求解器评分：`burned`（送墓或除外的卡数）、`actions`（召唤与发动次数）、`alt`（`--fire-spare` 的替代目标） |
| `lines[].source` | 求解器输出文件名 |
| `rejected[]` | 未通过重放的求解器线：`{"source", "error"}` |
| `solver` | 求解器提交、`solve_ms`、`threads`、`seed`、`returncode`、`elapsed_s`、`wall_s`、`candidates`、`written`、`timed_out`；`--fire` 另有 `fire_ms`、`windows`、`converted`；未解出时 `best_placed`、`target_cards` |

读取与使用：

```python
from ygorl.solver import read_jsonl, iter_steps

for demo in read_jsonl("out/demos/solver.jsonl"):
    if demo.status != "solved":
        continue
    for point, action in iter_steps(demo, 0, env=None):   # 绑定环境的记录要传同一个环境
        ...                                                  # 例如只取 point.player == 0 的步骤做 BC
    rep = demo.replay(0)            # ygorl Replay：rep.play()、rep.to_yrpx(...)、branch.fork(rep, t)
```

示范只存应答与动作下标，不存观测：观测由重放在任何时刻重建（`iter_steps` 给出 `DecisionPoint`），编码器变了也不用重跑求解器。

## 批量驱动

```bash
tools/build_combo_solver.sh
uv run python tools/solve_openings.py tests/decks --hands 1000 --solve-ms 60000 --fire-ms 20000 --workers 4 --threads 1
uv run python tools/solve_openings.py tests/decks/snake_eye.ydk --hands 5 --solve-ms 15000 --no-fire --out out/demos/se.jsonl
```

- 第 `i` 手的洗牌种子 `hand_seed(--seed, 牌组名, i)`（SHA-256 截断，跨机器稳定）；`--first-hand` 指定起始编号，便于分片。
- 每手一个子进程作业（`multiprocessing` spawn），`--workers` 个并行，每个求解器 `--threads` 个线程（批量时 1 线程/进程吞吐最高）。
- 记录完成即追加到 `--out`（默认 `out/demos/solver.jsonl`；给 `--env` 时为 `environments/<版本>/artifacts/demos/solver.jsonl`），
  中断后重跑同一命令只补缺失的记录；被中断截断的最后一行会先被切掉。
- `--fire P` 可重复；缺省用目标文件里每套牌的 `fire` 列表，`--no-fire` 关闭。`--lines K` 每手保留 K 条线。`--keep-files` 保留每手的模板、卡组、求解器日志 `solver.log` 与输出。
- 结束时打印每套牌 × 变体的汇总（状态计数、线数、被拒线数、求解墙钟）；`--summary` 另存 JSON。存在 `error` / `unverified` 记录时退出码 1。

## 实测（小批量）与全量成本

全量验收（10 套牌 × 1,000 起手，含 `--fire`）**尚未运行**，超出本次会话。已用同一流程跑了一个真实小批量（2026-09-22，4 核，详表见 [benchmarks.md](benchmarks.md#求解器示范集小批量t4a1)）：

- 10 套测试牌组 × 5 手，`--solve-ms 20000`，`--fire 14558127 --fire-ms 10000`，4 进程 × 1 线程，墙钟 277 秒；
- plain：50 手解出 28 手（每手平均 7.1 秒），22 手在 20 秒内未解出；`--fire`：28 次运行中 9 次得到恢复线，4 次无可发动时点，45 个时点恢复 11 个；
- 写入 37 条线（1,393 个动作步），求解器输出的线**全部**通过新鲜重放（被拒 0 条），`tools/verify_demos.py` 独立复验 37/37 通过；
- 外推：同样预算下全量约 59 进程小时（4 核约 15 小时）；spike 量级预算（60 秒 / 20 秒每时点）约 135 进程小时。

复验一个示范集文件（不需要求解器二进制）：

```bash
uv run python tools/verify_demos.py out/demos/solver.jsonl      # 每条线从零重放：动作合法、无 MSG_RETRY、应答一致、终局一致且含目标卡
```

## 限制

- **开局后不再洗卡组**：求解器靠 `DUEL_PSEUDO_SHUFFLE` 固定起手，该 flag 同时让核心在对局中的「洗切卡组」变成空操作，检索后的卡组顺序与起始顺序相同。
  示范线在这一 flag 下成立并在我们的核心里验证；拿到标准对局（有洗牌）里，依赖卡组顶的效果（挖掘、翻开）可能走向不同。
- **目标需要人工指定**，且只判定目标卡（子集），不要求「最优」终场；`unsolved` 同时混有真卡手与预算不足，求解器的搜索按墙钟预算、多线程时不可复现（`--threads 1 --solver-seed S` 可复现同一次搜索）。
- 对手是白板（40 张通常怪兽），`--fire` 只在最佳一条线上测试、每个时点只发动一次；对手的其他互动（连锁顺序、多张手坑）不在范围内。
- 只支持 1 对 1、带扩展头（4 个种子字）的回放；求解器的 `--opp-hand`（不 bake）线需要额外把卡加进对手手牌，示范集只用 `--fire-bake`。
- 求解器在 Linux 上是手工构建（上游只配置 Windows / macOS），GCC 兼容性靠强制包含头文件；升级求解器提交时需重新确认补丁锚点与这些修补。
