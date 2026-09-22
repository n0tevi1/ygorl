# 引擎层（`ygorl.engine`）

本文说明 M1 的引擎绑定：`CoreBackend`（C++）、消息解码、动作模型、单局 API 与确定性。对应设计文档 [01-engine.md](design/01-engine.md)、[06-architecture.md §5.2](design/06-architecture.md)，任务 T1.1 / T1.4 / T1.5 / T1.6。

## 分层

```
ygorl.engine.duel      Duel(seed, env, deck_a, deck_b).run(agent_a, agent_b) / .replay(responses)
ygorl.engine.actions   决策消息 → 合法动作列表（多选拆步）→ set_response 字节
ygorl.engine.messages  OCG_DuelGetMessage 缓冲区 → 类型化 Message（绝不抛异常）
ygorl.engine.constants 由 tools/gen_constants.py 从 ocgapi_constants.h 生成
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
| SELECT_CARD / TRIBUTE / SUM | 逐张 select + finish（可取消时 cancel） | 只给出仍能补全为合法集合的卡：祭品数按 `release_param` 计；SUM 精确模式用子集和 DP（含每张卡的两种数值与必选卡），「≥ 且最小」模式做单调剪枝的搜索 |
| SELECT_UNSELECT_CARD | select / unselect / finish 或 cancel | 核心本身就逐张询问 |
| SELECT_COUNTER | 每步移除一个指示物 | 共 `count` 步 |
| SORT_CARD / SORT_CHAIN | 逐张定序（首步可 default） | 剩一张时自动完成 |
| SELECT_PLACE / DISFIELD | 逐个区域 | 只列出 flag 未置位的区域；`value = 玩家<<16 \| 区域<<8 \| 序号` |
| ANNOUNCE_RACE / ATTRIB | 逐个位 | 共 `count` 步 |
| ANNOUNCE_CARD | declare（全部可宣言的卡） | 用与核心一致的 RPN opcode 解释器（`is_declarable`）从卡库筛选 |

因此 agent 永远不会触发 `MSG_RETRY`；压力测试把 retry 数作为失败条件。

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
