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
ygorl.solver.blocking           阻断点（脚本事实）、终场打分、任意牌组的自动目标阶梯（见「阻断场面」）
tools/solve_blocking.py         任意牌组的阻断场面示范：多进程、自动目标、按终场阻断点选线
ygorl.solver.survival           真实对手 + 当前策略打完对手第 2 回合的终场评估器与选线（见「存活场面」）
tools/solve_survival.py         按第 2 回合后的存活选线的示范：多候选（阻断阶梯 + boss + 多条线）、配对样本、可续跑、校准报告
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

## 阻断场面：任意牌组的自动目标（`ygorl.solver.blocking`）

动机：先攻第 1 回合是策略最弱的地方（诊断见 issue #83 一线：10 套测试牌组 × 12 手里，随机探索能到目标场面的 41 手，
PPO 最好的检查点只到 7 手；自博弈先攻胜率 0.449，而 Master Duel 竞技环境先攻约 0.70）。好玩家先攻铺的是**阻断**（无效、妨害、盖放陷阱、留在手里的手坑），
不是最大的怪兽。`tests/decks/solver_targets.json` 只覆盖 10 套测试牌组、每套一张终点卡，训练池的 837 套语料牌组没有目标。

### 求解器能优化什么

读求解器 `--help`（固定提交 `e0c7221`）的结论：**它只能「到达目标卡」**，没有可打分的终场。

- 目标是卡片子集：`--target card@zone[:fd]`（场上 `atk`/`def`/`mzone`/`szone`，以及在目标时刻检查的 `hand`/`grave`/`banished`），重复即要求多张；
  另有线约束（`--summon` 第 n 次召唤、`--resolve` 至少结算某效果 n 次、`--summon-min`、`--material`、`--no-activate`、`--no-chain`、`--guard`）。
- 搜索的分数是「放上了几张目标卡」（每张 100）加地标 / 配方启发式；到达目标即停。`--optimize` 在到达后继续，但只按**代价**字典序优化（少烧卡、少动作），方向与「多铺阻断」相反。
- `--hindsight` 把搜索中召唤出的额外卡组怪兽当替代目标来学策略，但不写出这些线；`--fire` 验证的是「被手坑后能否恢复同一终场」，不是终场好坏。

所以目标函数放在我们这边：自动生成**一组**目标、各求一次，再用脚本事实给每条已验证线的终场打分，留分最高的。

### 阻断点：从脚本事实推出（不列卡名）

`ygorl.build.scripts` 的每个效果另记 `EffectFact`：`categories`、`SetType`、`SetCode`、`SetRange` 的常量值，以及 `opponent`——
效果的 target / operation / cost 调用链里有没有能取对手卡的匹配调用（`Duel.SelectTarget(tp, f, tp, 0, LOCATION_ONFIELD, …)` 这类对手区域参数非 0）或
`Duel.GetFieldGroup(tp, 0, …)`。一张卡在某处算一个**阻断点**（`interruption`），当且仅当它有一个效果同时满足：

| 条件 | 判定 |
|------|------|
| 对手回合可用 | `EFFECT_TYPE_QUICK_O` / `QUICK_F`（发动位置取 `SetRange`：怪兽区 / 魔陷区 / 场地 / 灵摆区 → `field`，手牌 → `hand`，墓地 → `grave`；没有 `SetRange` 时怪兽取怪兽区、魔陷取魔陷区）；或陷阱卡、速攻魔法的 `EFFECT_TYPE_ACTIVATE`（→ `set`，盖放后下回合可发动） |
| 能打断 | 分类含 `NEGATE` / `DISABLE` / `DISABLE_SUMMON`（记为「无效」`negate`）；或含 `DESTROY` / `REMOVE` / `TODECK` / `TOHAND` / `TOGRAVE` / `CONTROL` / `POSITION` / `RELEASE` 且 `opponent` 为真 |

每张卡每个位置最多一个（无效优先）。抽查：Ash Blossom → 手牌·无效，Infinite Impermanence → 盖放·无效，Baronne de Fleur → 场上·无效，
S:P Little Knight → 场上·妨害，Called by the Grave → 盖放·妨害；Maxx "C"、Accesscode Talker、I:P Masquerena 不算。

**终场打分**（`board_interruptions`，玩家 0）：表侧怪兽与表侧魔陷按 `field`、盖放魔陷按 `set`、手牌按 `hand`、墓地按 `grave` 计阻断点；里侧怪兽、超量素材不计。
比较键为（阻断点数，其中无效数）。

### 目标阶梯（`blocking_plan` / `solve_blocking`）

每手先按起手定**底座** `base`：起手里能盖放阻断的陷阱 / 速攻魔法要求 `@szone:fd`（至多 5 张），手坑要求 `@hand`（留着不用）。
再从卡组取**阻断件** `piece`：场上有阻断点、且不是手坑的怪兽，排序为「引擎件优先（与主卡组共享字段，或脚本的 `listed_names` / 素材与主卡组互相点名）→ 无效优先 → 额外卡组优先」。
泛用额外怪兽（Baronne、Underworld Goddess）排在系列 boss 之后，因为它们只对部分牌组可达。每手的求解尝试（每次都是完整的 `solve_hand`：求解 + 新鲜重放验证）：

1. `base + piece_i`，i 取前 `--pieces` 个（默认 3）；
2. 两个件各自解出时，`base + 两件` 再求一次（`--no-pair` 关闭）；
3. 前面都没解出而 `base` 非空时，只求 `base`（「盖陷阱、留手坑、结束」）。

所有解出的线按终场打分，留最高的一条作记录（`targets` 为它的目标）；`solver.blocking` 记 `base`、`pieces`、每次尝试（目标、状态、耗时、分数）、选中的尝试和终场阻断点。
记录格式与普通示范集相同，`tools/train_bc.py`、`tools/verify_demos.py` 直接读。

```bash
nice -n 19 uv run python tools/solve_blocking.py out/corpus/train --sample 20 --hands 8 --solve-ms 10000 --workers 12 \
  --out out/solver_teacher/pilot.jsonl --summary out/solver_teacher/pilot_summary.json
```

### 试点（2026-09-30）

语料训练池随机 20 套（`--sample 20 --sample-seed 0`）× 8 手，`--solve-ms 10000`、3 个件 + 组合、12 进程 × 1 线程、`nice -n 19`（机器负载约 80–100 / 32 核），
不绑定环境（与 `out/why/bb_lam05` 的 PPO 运行一致，默认 MR5 规则）。墙钟 **332 秒**；`verify_demos.py` 独立复验 147 条线（6,286 步）0 失败。

| 指标 | 结果 |
|------|------|
| 解出（任一尝试） | **147 / 160（92%）** |
| 其中到达场上阻断件 | 107 / 160（67%）；其余 40 手是「只有底座」（盖陷阱 / 留手坑） |
| 每手求解墙钟（各次尝试之和） | 平均 23.5 秒（11–33 秒 / 牌组）；共 538 次尝试 |
| 终场阻断点 | 平均 2.23，其中无效 1.50 |
| 未解出 | danger-dark-world-2 的 8 手（卡组没有任何阻断件、起手也没有可盖 / 可留的卡，0 次尝试）；其余 5 手分散 |

终场示例（卡名仅用于显示）：

- orcust：Enlilgirsu + 盖放 Dominus Impulse ×2、Forbidden Droplet、Infinite Impermanence（5 个，4 无效）；
- hecahands-2：Diabellstar Vengeance、Azamina Ilia Silvia + 盖放 Azamina Aphes、Sinful Spoils of Betrayal - Silvera ×2（5 个）；
- live-twin-spright：Spright Red、Spright Carrot（场上无效 ×2）+ 手里 Ash Blossom ×2；
- materiactor-3：Number F0: Utopic Draco Future、Number 3: Cicada King + 手里 Ash；
- nordic-3：Gorgon, Empress of the Evil Eyed、Serziel + 盖放 Evil Eye Retribution；
- exodia-3：Red-Eyes Dark Dragoon + 盖放 Future Silence ×2；
- goblin-biker-2：Fiendsmith's Desirae + D/D/D Wave High King Caesar（组合尝试）。

分牌组的解出率、每手耗时与全部终场在 `out/solver_teacher/pilot_summary.json`。

### 作为第 1 回合的教师：BC 先验 + 只在第 1 回合的 KL

已有的机制（[training.md](training.md)「只在第 1 回合用先验」、[bc.md](bc.md)「补救实验」(c)）直接可用：
`--bc-prior CKPT --kl-prior C --kl-prior-turns 1` 只对先攻第 1 回合、回合玩家自己的决策行计 `KL(π‖π_prior)`，后面的回合不受先验约束（全状态先验会把策略钉在「不攻击」上）。
先验的词表必须与 PPO 运行相同，所以先验**从当前策略的 actor 微调**（`train_bc.py --init-from`：沿用检查点的网络、卡片词表与事件窗口），而不是另起一个 BC 网络。

试点上的原型（CPU；hands 0–5 训练、6–7 留出，同牌组未见过的起手）：

| 策略（argmax） | 留出 40 手到达阻断目标（36 手求解器可解） | 留出步准确率 / NLL |
|------|------|------|
| Random | 4 / 40 | — |
| Greedy | 7 / 40 | — |
| PPO `bb_lam05/update_000400` | 6 / 40 | 0.382 / 1.71 |
| + BC 微调 4 轮，lr 1e-4（`bc_ft`） | 7 / 40 | 0.396 / 1.22 |
| + BC 微调 20 轮，lr 3e-4（`bc_ft20`，每轮约 3 秒） | **12 / 40** | 0.469 / 1.14 |

CPU 上的 PPO 冒烟（2 套牌、8 槽 × 16 步、3 次更新）：`--init-from update_000400.pt --bc-prior bc_ft/policy.pt --kl-prior 2 --kl-prior-turns 1` 正常运行，
`kl_prior` 只在第 1 回合的行上非零（`kl_prior_rows` 76 / 14 / 0）。

GPU 上的实际运行（先在 CPU 上生成全量示范，约 837 × 8 手 × 24 秒 / 16 进程 ≈ 2.8 小时）：

```bash
# 1. 示范（CPU）：全部训练牌组 × 8 手；hands 8–9 另生成作留出
nice -n 19 uv run --no-sync python tools/solve_blocking.py out/corpus/train --hands 8 --solve-ms 10000 --workers 16 \
  --out out/solver_teacher/train.jsonl --summary out/solver_teacher/train_summary.json
nice -n 19 uv run --no-sync python tools/solve_blocking.py out/corpus/train --sample 100 --first-hand 8 --hands 2 \
  --solve-ms 10000 --workers 16 --out out/solver_teacher/heldout.jsonl
# 2. 先验：从当前 actor 微调（CPU 即可，样本约 1.2 万）
nice -n 19 uv run --no-sync python tools/train_bc.py --train out/solver_teacher/train.jsonl --heldout out/solver_teacher/heldout.jsonl \
  --init-from out/why/bb_lam05/checkpoints/update_000400.pt --epochs 20 --lr 3e-4 --threads 8 --openings heldout \
  --out out/solver_teacher/bc_prior
# 3. PPO（GPU）：与 bb_lam05 相同的配置，从 update_000400 起步，critic 重新预热；对照组去掉 --bc-prior 三个参数
PYTHONFAULTHANDLER=1 nice -n 19 uv run --no-sync python tools/train_ppo.py out/corpus/train/*.ydk --d-model 64 --layers 1 \
  --event-length 64 --device cuda --env-threads 4 --pairings all --eval-pairs 1 --eval-pairings 100 --eval-opponents greedy \
  --keep-best-by greedy --eval-workers 6 --seed 0 --envs 128 --steps 128 --minibatch 2048 --eval-every 50 --lam 0.5 \
  --init-from out/why/bb_lam05/checkpoints/update_000400.pt --critic-warmup 50 \
  --bc-prior out/solver_teacher/bc_prior/policy.pt --kl-prior 2.0 --kl-prior-turns 1 \
  --updates 400 --out out/why/teacher_kl2
```

看什么：`metrics.jsonl` 的 `first_player_win_rate`（基线 0.449）、对 Greedy 的评估、`train_bc.py --no-train --checkpoint … --openings heldout` 的留出阻断目标到达率；
系数扫 0.5 / 2.0，另跑一个无先验对照（同样 `--init-from` 与 `--critic-warmup`）。

### 限制（阻断场面）

- 阻断点只看效果的分类、时机与取对象范围：「对手回合特殊召唤」类（Welcome Labrynth）、永续压制（Dimension Shifter）、只在对手召唤时诱发的效果不算；
  同时有些分类宽松的效果会多算（例如墓地里能除外对手卡的诱发即时效果）。每卡每处最多 1 个，不看次数限制。
- 求解器到达目标即停，线不会「顺手」再多铺；多个阻断件只靠「组合」一步去争取。只有底座的记录（40 / 160）教的是「盖陷阱、留手坑、结束回合」，
  对手里有展开件的起手这是保守的；训练时可按 `solver.blocking.attempts[chosen].label == "base"` 过滤或降权。
- 手坑以 `@hand` 留住：如果它同时是展开需要的消耗（丢弃、素材），这一件就解不出（没有回退为不留的尝试）。
- 试点不绑定环境（与 bb_lam05 一致）；绑定环境的运行加 `--env`，示范、先验与 PPO 必须同一环境。

## 存活场面：按对手第 2 回合打分选线（`ygorl.solver.survival`）

动机：阻断场面的教师（上节）把策略锚在「多铺阻断点」上（`out/why/teacher_kl2`：BC 先验 + `--kl-prior-turns 1`，阻断场面 44% 对对照 15%），
先攻胜率却没涨（0.422 对 0.443）。对局阅读（`tools/game_trace.py`）的结论：阻断点数与先攻胜率无关；能预测胜负的是第 1 回合的怪兽数、
**挺过对手第 2 回合的怪兽数**、第 3 回合的攻击；斩杀全是战斗伤害；被锚住的策略拿怪兽换盖卡和手坑、手牌打空。
所以不再数阻断点，而是**真的打一遍对手的回合**再打分。

### 评估器（`play_line` / `survival_score`）

- **在真实对局里重放示范线**：用记录自己的起点（加载顺序 = 起手在顶、核心种子字、玩家规则），清掉 `DUEL_PSEUDO_SHUFFLE`（与 `build.first_turn.replay_line` 相同），
  对手换成**真实卡组**（语料随机一套，主卡组按 `order_seed` 洗）。线的每一步在伪洗牌对局里重放、按「做了什么」（种类、卡、区域）匹配到真实对局的候选
  （`first_turn.match_action`）；第 1 回合里对手的每个决策由策略做（会发手坑）。某一步匹配不到（手坑打断、真实洗牌后检索 / 硬币不同）就记为 `deviated`，
  先攻剩下的第 1 回合交给策略打完。
- **策略打第 2 回合**：同一个检查点（默认 `out/why/bb_lam05/checkpoints/update_000400.pt`）双方都用，一个 `CheckpointAgent` 与对局 lockstep。
  第 3 回合开始时读场面（`board_features`）：先攻的怪兽数、存活数（第 1 回合末的怪兽仍在场的张数）、手牌、LP、严格阻断点；对手场上卡数。
- **配对样本**：每手 N 个对手样本（卡组、洗牌、策略种子由手的种子决定），同一手的所有候选线打同一组样本；前 `--to-end` 个样本打到终局，用于校准。
- **打分**：`survival_score` = 怪兽 + 0.5×手牌 + 6×LP/8000 + 0.5×严格阻断点 − 0.6×对手场上卡 + 2（活着）；第 2 回合被斩杀只剩对手场面项。
  权重来自试点 12,989 局的 logistic 回归（每单位系数：怪兽 0.33、存活 0.32、LP 0.26/千、手牌 0.15、严格阻断点 0.15、对手场面 −0.21；按怪兽 = 1 缩放取整；
  存活与怪兽高度相关，只留怪兽）。试点自己用的是 `PILOT_WEIGHTS`（LP 1、对手场面 −0.25、无阻断点项）。

### 候选线与选线（`solve_survival` / `tools/solve_survival.py`）

每手的求解尝试（每次都是完整求解 + 新鲜重放验证，`--lines K` 每次保留 K 条线）：阻断阶梯（底座 + 前 `--pieces` 个件、两件组合、无件可解时只求底座，
按**严格**计数定底座与件），外加 `--bosses` 个**纯 boss** 目标（与主卡组同系列 / 互相点名的额外卡组怪兽，按攻击力，不带底座）。所有互不相同的已验证线都是候选；
另用同一组样本跑一次「策略自己打第 1 回合」作参照。记录只保留平均分最高的一条线（同分看严格阻断点），格式与示范集相同（`train_bc.py` 直接读）；
`solver.survival` 记每个候选（尝试、目标、终场、旧 / 严格阻断点、评估均值）、选中的、**旧工具会选的**（各阻断尝试的第一条线里阻断点最多的）、策略参照；
每个样本一行写进 `<out>.samples.jsonl`。`--report OUT.jsonl [--weights JSON]` 汇总一个输出（可换权重重算校准与留出选线）。

### 严格阻断点（`board_interruptions(strict=True)`）

`EffectFact` 另记 `count_code`（`SetCountLimit(1, id)` 的卡名一回合一次）、`acts`（操作链里有没有无效 / 破坏 / 除外 / 送去 / 回手 / 回卡组 / 控制权 / 表示形式 /
`EFFECT_DISABLE` 之类，成本不算）、`requires`（条件或对象函数里「自己怪兽区」的匹配调用的过滤器，以及「对象是自己的 … 怪兽」型条件里 `IsExists` 的过滤器）。
严格计数：去掉 `acts` 为假（玩家锁：Droll & Lock Bird）或只抽卡（Mulcharmy Fuwalos / Purulia、Future Silence）的效果；有 `requires` 的效果只在己方场上有满足过滤器的表侧怪兽时计
（Xyz Reflect 要超量怪兽、Dragonmaid Tidying 要龙族）；同一 `count_code` 只计一次（两张 Ash 计 1）。1,484 张有旧阻断点的卡里 39 张丢掉某处、263 张带条件。
旧计数仍是默认（`strict=False`），`solve_blocking` 不变。

### 试点（2026-10-01）

与阻断试点同样的 20 套 × 8 手（`--sample 20 --sample-seed 0`），`--solve-ms 10000 --lines 3 --pieces 3 --bosses 2 --samples 8 --to-end 8`，12 进程，墙钟 20 分钟
（每手求解 30 秒 + 评估 57 秒）；输出 `out/solver_teacher/survival/pilot*.json(l)`。

| 指标 | 新选择 | 旧选择（阻断点最多） |
|------|------|------|
| 解出 | 156 / 160（boss 目标补上了 danger-dark-world-2 这类无阻断件的牌组） | 147 / 160 |
| 与旧选择不同 | 112 / 147 | — |
| 第 1 回合终场：怪兽 / 手牌 / 盖放魔陷 | 1.56 / 2.27 / 0.53 | 1.49 / 2.05 / 0.67 |
| 旧 / 严格阻断点 | 1.55 / 1.36 | 2.27 / 2.03 |
| 第 3 回合开始：怪兽 / 存活 / 手牌 / LP | 1.11 / 0.86 / 3.15 / 6,816 | 0.78 / 0.63 / 2.88 / 6,528 |

选中的标签：件 63、boss 56、底座 25、组合 12。候选平均 9.4 条 / 手；35% 的样本线没走完（大多在真实洗牌后匹配不到）。

**预测力**（12,989 局打到终局，先攻胜率 0.47）：单局分数对胜负的 AUC 0.70（Pearson 0.35）；单项 AUC：第 2 回合后怪兽 0.67、存活 0.67、LP 0.67、
严格阻断点 0.57、手牌 0.51，第 1 回合末的阻断点 0.53、怪兽 0.57；logistic 全部特征 AUC 0.73。按线平均：分数与胜率 Pearson 0.32（1,624 条线）；
同一手里分最高的线比分最低的胜率高 0.14。

**留出选线**（前 4 个样本选线，后 4 个样本上的胜率；147 手配对）：

| 选法 | 胜率 |
|------|------|
| 试点权重选的线 | 0.463（比旧选择 +0.009，95% CI −0.03…+0.05） |
| 拟合权重（现默认）选的线 | 0.473（+0.020，CI −0.01…+0.06；权重在同一批局上拟合，略乐观） |
| 旧选择（阻断点最多） | 0.449 |
| 随机候选 | 0.444 |
| 策略自己的第 1 回合 | 0.418 |

旧选择几乎等于随机候选——阻断点数确实不预测胜负；按存活选能多赢一点，但 4 个样本选线噪声大，差距在误差内。

### 全量与训练

```bash
# 1. 示范（CPU，约 837×8 手 × 87 秒 / 12 进程 ≈ 14 小时；可续跑；--to-end 2 留一点校准局）
YGORL_COMBO_SOLVER=… CUDA_VISIBLE_DEVICES= nice -n 19 uv run --no-sync python tools/solve_survival.py out/corpus/train \
  --checkpoint out/why/bb_lam05/checkpoints/update_000400.pt --hands 8 --solve-ms 10000 --lines 3 --samples 8 --to-end 2 \
  --workers 12 --out out/solver_teacher/survival/train.jsonl --summary out/solver_teacher/survival/train_summary.json
CUDA_VISIBLE_DEVICES= nice -n 19 uv run --no-sync python tools/solve_survival.py out/corpus/train --sample 100 --first-hand 8 --hands 2 \
  --checkpoint out/why/bb_lam05/checkpoints/update_000400.pt --samples 8 --workers 12 --out out/solver_teacher/survival/heldout.jsonl
# 2. 先验：从当前 actor 微调（与 teacher_kl2 的先验同样的参数）
CUDA_VISIBLE_DEVICES= nice -n 19 uv run --no-sync python tools/train_bc.py --train out/solver_teacher/survival/train.jsonl \
  --heldout out/solver_teacher/survival/heldout.jsonl --init-from out/why/bb_lam05/checkpoints/update_000400.pt \
  --epochs 20 --lr 3e-4 --threads 8 --openings heldout --out out/solver_teacher/survival/bc_prior
# 3. PPO（GPU）：与 teacher_kl2 相同的参数，只换先验
PYTHONFAULTHANDLER=1 nice -n 19 uv run --no-sync python tools/train_ppo.py out/corpus/train/*.ydk --d-model 64 --layers 1 \
  --event-length 64 --device cuda --env-threads 4 --pairings all --eval-pairs 1 --eval-pairings 100 --eval-opponents greedy \
  --keep-best-by greedy --eval-workers 6 --seed 0 --envs 128 --steps 128 --minibatch 2048 --eval-every 50 --lam 0.5 \
  --init-from out/why/bb_lam05/checkpoints/update_000400.pt --critic-warmup 50 \
  --bc-prior out/solver_teacher/survival/bc_prior/policy.pt --kl-prior 2.0 --kl-prior-turns 1 \
  --updates 400 --out out/why/survival_kl2
```

看什么：`first_player_win_rate`（teacher_kl2 0.422、对照 0.443）、`tools/game_trace.py` 的第 1 回合怪兽数与第 2 回合存活数、对 Greedy 的评估。

### 限制（存活场面）

- 评估器是**当前策略**打的第 2 回合：策略的弱点（不会解场、不会斩杀）会被当成场面的强度；策略变强后要用新检查点重跑。
- 示范线是伪洗牌下的线，真实对局里 35% 走不完（洗牌后检索顺序、硬币 / 骰子用的核心随机数随洗牌而变、对手手坑）；BC 学的仍是伪洗牌线本身。
- 8 个样本 / 线、胜负噪声大：留出选线的提升在误差内；全量训练集的选择本身带噪声。
- 严格计数的条件只认简单形式（自己怪兽区的匹配调用、「对象是自己的…」）；更复杂的条件、软一回合一次、以及效果之间的共享次数仍不看。
- 只在一手的候选线之间选；求解器到达目标即停，线不会为「多留手牌」而优化。

## 限制

- **开局后不再洗卡组**：求解器靠 `DUEL_PSEUDO_SHUFFLE` 固定起手，该 flag 同时让核心在对局中的「洗切卡组」变成空操作，检索后的卡组顺序与起始顺序相同。
  示范线在这一 flag 下成立并在我们的核心里验证；拿到标准对局（有洗牌）里，依赖卡组顶的效果（挖掘、翻开）可能走向不同。
- **目标需要人工指定**，且只判定目标卡（子集），不要求「最优」终场；`unsolved` 同时混有真卡手与预算不足，求解器的搜索按墙钟预算、多线程时不可复现（`--threads 1 --solver-seed S` 可复现同一次搜索）。
- 求解器的线可能在 SELECT_UNSELECT_CARD 里把同一张卡**选中又立刻取消**、来回多次（2026-09-23 的 BC 数据集里一条 tenpai 线来回 320 次，
  4 手 held-out tenpai 起手各 318–327 次）；这些线是合法的、能通过验证，只是冗长。BC 把这样的步对从样本中去掉（[bc.md](bc.md)）。
- 对手是白板（40 张通常怪兽），`--fire` 只在最佳一条线上测试、每个时点只发动一次；对手的其他互动（连锁顺序、多张手坑）不在范围内。
- 只支持 1 对 1、带扩展头（4 个种子字）的回放；求解器的 `--opp-hand`（不 bake）线需要额外把卡加进对手手牌，示范集只用 `--fire-bake`。
- 求解器在 Linux 上是手工构建（上游只配置 Windows / macOS），GCC 兼容性靠强制包含头文件；升级求解器提交时需重新确认补丁锚点与这些修补。
