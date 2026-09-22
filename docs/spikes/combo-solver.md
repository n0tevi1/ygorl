# Spike：ygo-combo-solver 与 edo9300 核心的兼容性（T1.7）

日期：2026-09-22。对象：`github.com/96jonesa/ygo-combo-solver` @ `e0c7221`（README 中的上游名为 Armytille/ygo-combo-solver），AGPL-3.0。

## 结论

1. **兼容**：求解器编译自己的 edo9300 ocgcore，固定在 `5a985af`（2026-08-10）。它是本仓库核心 `122e0d0` 的祖先，只差 15 个提交，`ocgapi*.h` 无改动（API 同为 11.0），两者间只有两个提交影响规则（`EFFECT_SPSUMMON_PROC` 第三返回值、融合 `chkf` 改 64 位）。它对核心的 5 个补丁（Lua 分配器钩子、固定字符串哈希种子、`OCG_DuelQueryProcessorState`）**原样打在我们的核心上全部成功**。
2. **决定：先封装二进制，不移植搜索**。T4a.1（示范集）与 T5.6（起手分析）用我们固定的核心 + 我们的脚本从源码编译求解器，以子进程调用，输入 `.yrpX` 或 `.ydk + 起手 + 目标场面`，输出的 `.yrp` 在我们的核心里重放验证后转成示范。搜索部分（NRPA + Levin 树搜索 + Iterated-Width，约 3 万行 C++）不移植。
3. **快照（T2.8）移植机制、不移植代码**：arena 的做法清楚且可行，但它依赖全局 `operator new` 重载，在 Python 扩展里需要改成仅作用于我们 `.so` 的隐藏符号版本，另见下文。

## 实测（本环境：Linux x86-64，GCC 13）

| 项目 | 结果 |
|------|------|
| Linux 构建 | 手工 g++ 构建成功（上游只配置了 Windows / macOS）。唯一问题：若干源文件缺 `#include <cmath>` 等，GCC 下需要 `-include cmath -include cstring -include algorithm -include cstdint` |
| 依赖 | sqlite3（系统库）、EDOPro 的 `gframe/lzma`、EDOPro 式工作目录（`cards.cdb` + 脚本目录，`--scriptdir` 可覆盖） |
| 重放我们导出的 `.yrpX`（随机对局，55 回合） | 1448/1448 个应答被消费，`MSG_RETRY` 0，自检通过。说明我们的 `.yrpX` 能被 EDOPro 同源的解析器读取，且两份独立构建、各自打补丁的核心逐步一致 |
| 求解（`--deck snake_eye.ydk --hand <5 张> --no-ref --target "Snake-Eyes Flamberge Dragon@atk" --solve-ms 60000`） | 探测 10.1 s，搜索共 34.9 s，找到 48 条到达目标的线（最优：10 个动作、56 个决策），写出 16 个 `.yrp` |
| 在我们的核心里验证这 16 个 `.yrp` | 16/16：应答全部被消费，retry 0，终局场上有 Flamberge |

## 封装方案与工作量（T4a.1 / T5.6）

| 工作 | 内容 | 估计 |
|------|------|------|
| 构建 | `tools/build_combo_solver.sh`：固定求解器提交，复制我们打过补丁的核心，套用求解器的 5 个补丁（其中种子补丁我们已自带，见 `patches/ygopro-core/0002`），编译；CI 可选任务 | 1–2 天 |
| 调用封装 | `ygorl.solver.combo_solver`：生成工作目录、组装参数（`--deck/--hand/--target/--fire/--solve-ms`）、解析 `--json` 事件与输出 `.yrp`（yrp1 解析与 LZMA 解压已在 spike 中验证） | 2–3 天 |
| 示范转换与验证 | `.yrp` 应答序列在 `Duel` 中重放（关闭主机洗牌、按 yrp 中的卡组顺序加卡），逐步把应答字节映射回我们的动作下标（多选拆步后，一个应答对应多步），得到 `(观测, 动作)` 序列 | 2–3 天 |
| 批量驱动 | 10 套牌 × 1000 起手 × (是否 `--fire`)，多进程；本例每手约 45–60 s，全量约 150–350 CPU 小时，需要按起手难度设预算 | 1–2 天 |

风险：求解器以 Windows 为主要平台，GCC 兼容性需要维护一个小补丁；输出目标场面语法依赖卡名解析；`--fire`、`--adapt` 未在本 spike 中测试。

## arena 快照（T2.8）移植评估

求解器的做法（`arena.h/.cpp`，约 2000 行）：

- 把一局的全部可变内存限制在一段自控地址区：Lua 堆经 `lua_newstate` 的分配器钩子（打在 `luaL_newstate` 上），核心的 C++ 对象经全局 `operator new/delete` 重载（按线程的「当前 arena」路由）。
- 快照 = 拷贝已用区域；恢复在**同一基地址**进行，所有绝对指针无需重定位。脏页跟踪目前只做度量。
- 搜索期间停掉 Lua GC（标记会写脏几乎所有页），被放弃分支的内存由恢复回收。
- 每个工作线程一个 arena，快照不跨线程。

移植到本项目的要点：

1. 全局 `operator new` 在 Python 扩展里不能影响解释器与 PyTorch：在 `_core` 里定义**隐藏可见性**的 `operator new/delete`，只绑定本 `.so` 内（核心 + 绑定）的调用；arena 未激活时回落到 `malloc`。`Free` 按地址判断归属。
2. 与 T2.1 线程池一致：每个工作线程固定一个 arena 与一组对局，快照只在本线程恢复。
3. 「逃逸」（arena 激活时落到宿主堆的分配）必须视为错误，沿用求解器的「中毒」标记。
4. 这同时解决 `docs/engine.md` 中「以指针为键的有序容器」的残余确定性风险：arena 内地址由分配序列决定。

估计 1–2 周（含与线程池集成和测试）。若受阻，按工程计划退回「从种子重放」，接口不变。

## 复现步骤

```bash
git clone https://github.com/96jonesa/ygo-combo-solver solver     # e0c7221
# 核心：复制 third_party/ygopro-core，运行 solver/tools/fetch_solver_deps.sh 中的 Python 补丁段
# 编译：Lua（按 C++，force-include luaconf-customize.h）、核心、gframe/lzma（-D_7ZIP_ST）、solver/*.cpp
#       solver 源码加 -include cmath -include cstring -include algorithm -include cstdint；链接 -lsqlite3 -lpthread
./combosolver game.yrpX --workdir <含 cards.cdb 的目录> --scriptdir <CardScripts 目录，可重复>
```
