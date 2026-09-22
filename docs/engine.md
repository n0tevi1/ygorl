# 引擎层（`ygorl.engine`）

本文说明 M1 的引擎绑定：`CoreBackend`（C++）、消息解码、动作模型、单局 API 与确定性。对应设计文档 [01-engine.md](design/01-engine.md)、[06-architecture.md §5.2](design/06-architecture.md)，任务 T1.1 / T1.4 / T1.5 / T1.6。

## 分层

```
ygorl.engine.duel      Duel(seed, env, deck_a, deck_b).run(agent_a, agent_b) / .replay(responses)
ygorl.engine.actions   决策消息 → 合法动作列表（多选拆步）→ set_response 字节
ygorl.engine.messages  OCG_DuelGetMessage 缓冲区 → 类型化 Message（绝不抛异常）
ygorl.engine.constants 由 tools/gen_constants.py 从 ocgapi_constants.h 生成
ygorl.engine.puzzle    残局：把卡直接放进指定区域开局（确定性的规则场景，T2.3 测试用）
ygorl._core            pybind11：CardDatabase / ScriptDirectory / Duel（csrc/）
third_party/ygopro-core + patches/ygopro-core/*.patch   规则核心（构建时打补丁）
```

## CoreBackend（`csrc/core_backend.{h,cpp}`、`csrc/binding.cpp`）

`ygorl._core.Duel` 是对一个 `OCG_Duel` 句柄的极薄封装：

| 方法 | 对应 C API |
|------|-----------|
| `Duel(seed[4], flags, team1, team2, cards, scripts)` | `OCG_CreateDuel` |
| `load_script(name)` | 经 script source 调 `OCG_LoadScript` |
| `new_card(team, duelist, code, con, loc, seq, pos)` | `OCG_DuelNewCard` |
| `start()` | `OCG_StartDuel` |
| `process()` | `OCG_DuelProcess`，返回 `DUEL_STATUS_*` |
| `get_message()` / `set_response(bytes)` | `OCG_DuelGetMessage` / `OCG_DuelSetResponse` |
| `query` / `query_location` / `query_field` / `query_count` | `OCG_DuelQuery*` |
| `pop_logs()` | 核心 log handler 收集的 `(type, text)` |
| `close()` / 析构 | `OCG_DestroyDuel` |

卡片数据与脚本通过两个接口提供（C++ 侧，不依赖 Python）：

- `CardSource`：回答核心的 card reader。原生实现 `CardDatabase`（由 Python 从 `cards.cdb` 灌入，之后只读）；也接受 Python 可调用对象 `code -> 12 元组 | None`。
- `ScriptSource`：回答 script reader，并负责调用 `OCG_LoadScript`。原生实现 `ScriptDirectory`（按目录列表查找，先到先得，默认顺序见 `ygorl.paths.script_directories()`）；也接受 `name -> bytes | None`。

原生实现构造完成后只读，可在 M2 线程池的多个核心句柄间共享。

**GIL 与线程**：绑定的每个方法都先释放 GIL 再获取该局的互斥锁；Python 回调在内部重新获取 GIL。这个顺序避免「一个线程持 GIL 等锁、另一个持锁等 GIL」的死锁。测试 `test_gil_released_while_core_runs` 验证核心执行 Lua 时其他线程仍可运行 Python。

**回调异常**：Lua 以 C++ 方式编译，会把任何穿过它的 C++ 异常当作 Lua 错误吞掉，所以回调在 C 边界捕获异常、暂存，由发起调用的方法（包括构造函数）在核心返回后重新抛出。

### 换核心时需要重写的部分

上层（`actions`、`duel`、agents、env）只依赖 `messages` 里的类型化消息和 `_core.Duel` 的方法集合。换成 Fluorohydride 核心时需要：

1. 重写 `csrc/core_backend.cpp`（API 形状几乎相同，但回调是全局的，需要按线程或按局分发）。
2. 重写 `messages.py` 的解码器：位置信息从 `{u8,u8,u32,u32}` 改为 4 个 u8，个别消息负载不同。
3. 重新生成 `constants.py`，重写 `actions.py` 里的应答编码（选卡、选位置等格式有差异）。
4. 重新确认 `patches/` 中的确定性补丁是否适用。

## 消息解码（`messages.py`）

`OCG_DuelGetMessage` 的缓冲区是 `[u32 长度][u8 类型][负载]` 的序列。每个 `MSG_*` 都有冻结 dataclass；布局逐条抄自核心的 `new_message(...)` 写入点。未知类型返回 `UnknownMessage`，负载截断返回 `UndecodableMessage`，两者都保留原始字节并记日志，**不抛异常**。测试脚本化核对：头文件中每个 `MSG_*` 要么有解码器，要么在 `NOT_EMITTED_BY_CORE`（`MSG_START` 等由 EDOPro 主机端生成的消息）里。

核心的三种位置编码：`loc_info`（u8 控制者、u8 区域、u32 序号、u32 表示形式/叠放序号）、`loc32`（无表示形式）、`loc8`（序号为 u8）。LP 在核心里是有符号 int32，致命伤害后会是负数。

## 动作模型（`actions.py`）

一次 `agent.act()` = 一个 step。`make_decision(msg)` 返回状态机：`actions()` 列出当前合法动作，`step(i)` 应用选择，完成后 `response` 为要 `set_response` 的字节。

| 决策 | 动作 | 说明 |
|------|------|------|
| IDLECMD / BATTLECMD | summon / spsummon / reposition / mset / sset / activate / attack / battle_phase / main2 / end_phase / shuffle | 单步；应答 `i32 (类别 \| 下标<<16)` |
| EFFECTYN / YESNO / OPTION / CHAIN / POSITION / ANNOUNCE_NUMBER / RPS | yes/no、option、chain/pass、position、number、rps | 单步 |
| SELECT_CARD / TRIBUTE / SUM | 逐张 select + finish；未选任何卡且消息可取消时另有 cancel | 只给出仍能补全为合法集合的卡：祭品数按 `release_param` 计；SUM 精确模式用子集和 DP（含每张卡的两种数值与必选卡），「≥ 且最小」模式做单调剪枝的搜索。finish 总是发显式列表（可以为空：min 为 0，或 SUM 的必选卡已足够），cancel 发 `-1`，见下文「多选可行集验证」 |
| SELECT_UNSELECT_CARD | select / unselect / finish 或 cancel | 核心本身就逐张询问 |
| SELECT_COUNTER | 每步移除一个指示物 | 共 `count` 步 |
| SORT_CARD / SORT_CHAIN | 逐张定序（首步可 default） | 剩一张时自动完成 |
| SELECT_PLACE / DISFIELD | 逐个区域 | 只列出 flag 未置位的区域；`value = 玩家<<16 \| 区域<<8 \| 序号` |
| ANNOUNCE_RACE / ATTRIB | 逐个位 | 共 `count` 步 |
| ANNOUNCE_CARD | declare（全部可宣言的卡） | 用与核心一致的 RPN opcode 解释器（`is_declarable`）从卡库筛选 |

因此 agent 永远不会触发 `MSG_RETRY`；压力测试把 retry 数作为失败条件。

## 多选可行集验证（T2.3）

SELECT_CARD / SELECT_TRIBUTE / SELECT_SUM 在核心里是**一次**应答（一组卡），环境把它拆成逐张 select + finish，并且每一步只给出还能补全成合法应答的卡（Python `actions.py` 与 C++ `csrc/host.cpp` 的 `DecisionState` 逐元素一致）。验收标准是「环境可行集 = 引擎接受集合」，由 `tests/test_feasible_sets.py` 用真实引擎检查。

**残局构造**：`ygorl.engine.puzzle.Puzzle` 用 `OCG_DuelNewCard` 把卡直接放到指定区域（手牌、怪兽区含额外怪兽区、魔陷区、额外卡组），双方起手 0 张、每回合抽 0 张，开局即玩家 0 的主要阶段 1。场景因此一两步就能走到素材选择，且与普通对局一样确定。

**做法**：从召唤/发动命令开始，对之后的每个决策：

1. SELECT_CARD / TRIBUTE / SUM：枚举所提供卡的**全部子集**（外加 `-1`），逐个作为应答发给引擎。被拒绝的应答只产生 `MSG_RETRY`、核心停在同一决策上，所以被拒的候选在同一局里接着试，只有被接受的候选需要重放一局（同一残局 + 同一应答前缀 → 同一状态）。得到的接受集合必须等于环境逐步动作能到达的完整应答集合（所有以 finish 或自动完成结束的 select 路径），Python 与 C++ 状态机都要相等，且任何路径都不能走进死路。
2. SELECT_UNSELECT_CARD：CardScripts 的同调/超量/连接手续（`proc_synchro.lua` / `proc_xyz.lua` / `proc_link.lua`）都用 `Group.SelectUnselect` 让核心逐张询问，环境只是透传；对它以及手续中途的其他决策（如 ANNOUNCE_NUMBER）遍历所有动作。连锁窗口、区域、表示形式取固定默认。
3. 每条路径回到空闲命令时，记录离开手牌/场上的卡（即实际使用的素材），与按卡片效果手工列出的合法素材集合比较。

**场景**（卡片密码见测试文件的 `CARDS`）：

| 场景 | 决策 | 要点 |
|------|------|------|
| 上级召唤 Dark Magician（46986414，暗，7 星） | SELECT_TRIBUTE | Double Coston（44436472）对暗属性怪兽算 2 个祭品：单独它、它 + 1 只、任意 2 只普通怪兽 |
| 上级召唤 Blue-Eyes White Dragon（89631139，光） | SELECT_TRIBUTE | Double Coston 只算 1 个 |
| Twin Twisters（43898403） | SELECT_CARD ×2 | 丢弃 1 张（1..1），再以 1..2 张盖放的魔法为对象 |
| Dark Magician Girl the Magician's Apprentice（2501624） | SELECT_CARD（min 0） | 从手牌特殊召唤时丢弃 0..1 张；选 0 张则召唤中止 |
| Black Luster Ritual（55761792，≥ 8） | SELECT_SUM 至少模式 | Miracle Raven（18988396，场上可作全部祭品：1 或 8）+ 手牌/场上 1–6 星 |
| Contract with the Abyss（69035382，= 8，暗属性仪式） | SELECT_SUM 精确模式 | Ritual Raven（34334692，1 或 8）+ 手牌/场上 1–6 星 |
| Number 39: Utopia / Number 61: Volcasaurus / Number 32: Shark Drake | SELECT_UNSELECT_CARD（+ ANNOUNCE_NUMBER） | Star Drawing（24610207）可当 5 星；Drake Shark（81096431）对需要 3 个以上素材的水属性超量可当 2 个素材 |
| Stardust Dragon（44508094，调整 + 非调整，8 星） | SELECT_UNSELECT_CARD | Yamatako Orochi（4632019，调整，1 或 8 星）、Tuningware（92676637，1 或 2 星） |
| Decode Talker（1861629，连接 3，2+ 效果怪兽） | SELECT_UNSELECT_CARD | Proxy Dragon（22862454）算 1 或 2，Linkuriboh（41999284）算 1，通常怪兽不可用 |

另外对随机生成的 SELECT_CARD / TRIBUTE / SUM 消息（含必选卡、min 0、双数值卡），用照抄 `playerop.cpp` 检查逻辑的纯 Python 模型比较（只取至少有一个合法应答的消息，核心不会发出无解的选择）。

**发现并修复的不一致**（Python 与 C++ 同步修改）：

- min 为 0 的 SELECT_CARD / TRIBUTE：原来「一张不选就 finish」被编码成 `-1`。核心同时接受 `-1` 和显式空列表，但含义不同：对可取消的脚本 `-1` 返回 nil（上级召唤里是取消召唤），空列表返回空卡片组。现在 finish 总是发显式列表，另给 `cancel`（`-1`）。min 为 0 时消息里的「可取消」位总是置 1（核心写的是 `cancelable || min == 0`），所以两个动作都会出现；脚本本身不可取消时二者效果相同。课程模式把 `cancel` 视为「放弃」类动作，因此 solo 模式下对手的 min 0 选卡由主机代答 `-1`。
- SELECT_SUM 的必选卡本身已满足条件时：核心接受空列表（至少模式下这是唯一合法应答，再加任何卡都多余），原实现要求至少选一张，走进死路。现在允许不选直接 finish。

**引擎行为备忘**：

- 核心对 SELECT_SUM 至少模式的检查是「各卡较大值之和 ≥ 目标，且各卡较小值之和减去最小的较小值 < 目标」。对能作全部祭品的卡（参数低 16 位为仪式怪兽等级、高 16 位为自身等级）它比规则文字宽松：Miracle Raven + Frostosaurus（6 星）会被接受。环境以核心为准。
- SELECT_SUM 精确模式由 `select_sum_check1` 按**卡片指针顺序**检查（`parse_response_cards` 按指针排序），它要求非末位的卡值严格小于剩余量。若出现数值为 0 的卡，接受与否取决于指针顺序；实际脚本没有遇到，环境按「存在一种取值使总和相等」处理。
- 上级召唤一般不会发出 min 0 的 SELECT_TRIBUTE：可以不解放时核心先问 YESNO「是否解放」，答是之后 min 为 1。
- 灵摆怪兽作为素材离场时进额外卡组（表侧），不在墓地；判断「用了哪些素材」要看离开手牌/场上的卡。
- 同一张多选消息之后核心可能还会询问别的决策（德雷克鲨在被选中后问「当作 1 个还是 2 个素材」）；这些信息不在后续的 SELECT_UNSELECT_CARD 消息里，是脚本内部状态。

运行时间：`uv run pytest tests/test_feasible_sets.py` 约 30 秒（每个接受的候选重放一局残局，一局约 10 ms）。

## 单局 API（`duel.py`）

```python
from ygorl.agents import RandomAgent
from ygorl.cards.ydk import load_ydk
from ygorl.engine.duel import Duel

a, b = load_ydk("tests/decks/snake_eye.ydk"), load_ydk("tests/decks/kashtira.ydk")
result = Duel(seed=1, env=None, deck_a=a, deck_b=b).run(RandomAgent(1), RandomAgent(2))
print(result.summary())
```

- AEC 语义：谁被问谁决策；`DecisionPoint` 含决策消息、合法动作、回合、阶段、LP 和上次决策以来的事件。**事件是核心的全知视角**，按玩家过滤属于 M2（T2.2/T2.5）。
- 引擎玩家 0 先攻；`first=1` 让 b 先攻。结果按 (a, b) 顺序报告。
- 核心在 `MSG_WIN` 之后仍会继续处理，主机（EDOPro 与我们）在第一个 `MSG_WIN` 处结束对局。胜负原因：1 = LP，2 = 卡组耗尽，0x10 以上为卡片特殊胜利。
- 回合上限 / 决策数上限（`DuelConfig.max_turns / max_decisions`）触发时 LP 高者胜，相等为平局。
- 课程模式（`DuelConfig.curriculum / learner`）让主机在学习方回合替对手作答「放弃」类决策，并过滤对手的非手牌发动；增广开局标志 `augmented_start` 随 `DecisionPoint` 下发。见 [curriculum.md](curriculum.md)。
- **洗牌在主机侧**：核心开局不洗卡组（EDOPro 由主机洗好再加卡）。`shuffle_deck(cards, seed, player)` 是 splitmix64 + 无偏 Fisher–Yates，有黄金向量测试锁定，供 M2 的 C++ 实现逐位复现。

## 确定性（R4，T1.6）

同一 `seed` + 同一应答序列 → 同一消息字节流。组成：

1. 核心 RNG 种子 = `expand_seed(seed)`（splitmix64 扩成 4 个 u64，喂给 Xoshiro256**）。
2. 主机洗牌用同一 `seed` 的独立流。
3. **补丁** `patches/ygopro-core/0001-deterministic-iteration-order.patch`：核心中若干以 `card*` / `effect*` 为键的 `unordered_set/map` 被遍历时会发消息或移除效果，遍历顺序随堆地址变化。实测未打补丁时同一进程内 60 局重放有 11 局消息顺序不同（都是效果重置时的 `MSG_CARD_HINT` / `MSG_PLAYER_HINT`），打补丁后为 0。补丁按效果 id（及 initial_id、code、type、description）或卡片 `cardid` 排序后遍历。子模块保持原样，CMake 在构建目录复制一份核心源码并按文件名顺序打补丁。
4. **补丁** `patches/ygopro-core/0002-constant-lua-string-hash-seed.patch`：Lua 5.4 默认用时钟与堆地址生成字符串哈希种子，`pairs()` 遍历字符串键表的顺序因此每局不同；补丁把 `luai_makeseed` 固定为 0（ygo-combo-solver 同样处理）。测试 `test_lua_string_keyed_pairs_order_is_stable` 用 Lua 探针脚本验证。

已知残余风险：核心里还有以原始指针为键的**有序**容器（如 `std::map<effect*, chain>`、`std::set<std::pair<effect*, tevent>>`），其遍历顺序由地址大小决定。大规模扫描（`tools/check_determinism.py`）未观察到由此导致的差异；若出现，彻底方案是每局一个地址确定的 arena 分配器，这也是 T2.8 快照需要的基础设施。

验证：`tests/test_determinism.py`（同种子字节一致、应答日志重放到同一终局、动作日志重放、截断日志、与参考流比对），以及 `uv run python tools/check_determinism.py --games 1000`。

## 向量化环境（`ygorl.env`，T2.1）

`csrc/duel_pool.{h,cpp}` 的 `DuelPool` 持有 `num_envs` 个槽位和 `num_threads` 个工作线程，槽位 `i` 固定由线程 `i % num_threads` 处理（每线程独立的核心句柄集合，为 T2.8 的每线程 arena 留好位置）。接口是 envpool 式的异步调用：

- `start(env, seed[4], flags, team1, team2, decks)`：在工作线程上建局、加载基础脚本、按给定顺序加卡、开始，并运行到第一个停点；
- `respond(env, bytes)`：设置应答并运行到下一个停点；
- `recv(min_results, timeout_ms)`：取回完成的作业 `(env, status, buffer, logs, error)`，不会等待比在途作业更多的结果；
- 停点 = 核心要求应答、对局结束，或缓冲区里出现 `MSG_WIN`（与单局一致，主机在第一个胜负处停止）。

所有调用在等待与核心运行期间释放 GIL。Python 侧 `VecDuelEnv` 为每个槽位持有一个 `DuelTracker`（与 `Duel.run` 共用同一份主机逻辑），多选的中间步骤在本地完成，只有完整应答才回到核心；`DuelEnv` 是单局的 `reset/step` 包装；`run_games(specs, agent_factory, num_envs, num_threads)` 按规格顺序返回结果。

验收：`tests/test_pool.py` 检查池化结果与逐局 `Duel.run` 完全一致、线程数不影响结果、上限在池中同样生效；`uv run python tools/check_pool.py --games 1000 --threads 4` 做 1,000 局比对。目前每个决策的消息解码与动作生成仍在 Python 中完成，吞吐受其限制，T2.2 把观测编码移到 C++ 后再做基准（T2.7）。

线程安全：`tools/tsan/check.sh` 用 `-fsanitize=thread` 另行编译 `_core`，在预加载 `libtsan` 的 Python 中以 4 线程跑池化对局并与逐局结果比对。2026-09-22 的一次运行：12 局、4 线程、结果与单线程一致，ThreadSanitizer 零报告（同一方式对一个故意制造数据竞争的库能正确报警，作为阳性对照）。

## C++ 步进环境（`EncodedVecEnv`，T2.2 / T2.7）

`VecDuelEnv` 的每个决策仍要在 Python 里解码消息、生成动作；训练用的是另一条路径：`csrc/host.{h,cpp}` 把消息解码、动作状态机（`DecisionState`，含多选拆步与可行性剪枝）、主机追踪（`Tracker`，对应 `DuelTracker`）与观测编码器（`csrc/obs_encoder.cpp`，规格见 [encoding.md](encoding.md)）全部用 C++ 重写，`tests/test_cpp_host.py` 逐元素核对它与 Python 参考实现一致。

`csrc/host_pool.{h,cpp}` 的 `HostPool` 在此之上提供异步接口（线程分配规则与 `DuelPool` 相同，由通用模板 `csrc/worker_pool.h` 实现）：

- `reset(env, spec)`：在工作线程上建局并运行到第一个决策；
- `step(env, action_index)`：应用一个动作；多选的中间步骤只更新本地状态，完整应答才送回核心；
- `recv(min_events, timeout)`：取回事件 `EncodedEvent(env_id, player, obs, result)`。`obs` 是定长 numpy 数组（`cards` 160×23、`globals` 22、`actions` 128×10、`action_mask` 128），对局结束时 `obs` 为 `None`、`result` 为终局信息（引擎玩家顺序）。
- 槽位在其作业结果被 `recv` 取走之前一直处于 busy 状态，此时再次 `reset/step` 会抛出 `RuntimeError`。
- 课程模式（[curriculum.md](curriculum.md)）目前只在 Python 主机（`Duel.run` / `VecDuelEnv`）里实现；`curriculum` 不是 `full` 或 `augmented_start=True` 的规格会被 `EncodedVecEnv.reset` 以 `NotImplementedError` 拒绝，而不是悄悄按完整对局运行。

Python 侧 `ygorl.env.encoded.EncodedVecEnv` 负责把 `GameSpec` 转成种子与加卡顺序（与 `Duel` 共用主机洗牌），`play(specs, choose)` 用于测试与基准。`tests/test_host_pool.py` 检查它与 `Duel.run` 在相同动作序列下逐局一致、线程数不影响结果、上限同样生效。吞吐基准见 [benchmarks.md](benchmarks.md)（`tools/bench_throughput.py`）。
