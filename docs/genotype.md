# 组牌基因型与约束（T5.5）

> 对应设计文档 [06-architecture.md](design/06-architecture.md)「基因型 = 引擎包份数 + 泛用槽 + 额外卡组；硬约束」
> 与 [05-deck-building.md](design/05-deck-building.md)（pyribs MAP-Elites、代理模型的输入），
> 工程计划 T5.5（[#42](https://github.com/n0tevi1/ygorl/issues/42)）。代码在 `src/ygorl/build/genotype.py`。

基因型是组牌搜索（T5.8 的 MAP-Elites、T5.9 的 LLM emitter）操作的对象。设计目标：**任何算子的输出都是合法牌组**，
搜索不需要拒绝采样，也不需要在 fitness 里惩罚非法解。

```python
import numpy as np
from ygorl.build.genotype import GenotypeSpace
from ygorl.build.packages import enumerate_packages, setcodes_from_db
from ygorl.build.synergy_graph import load_or_build
from ygorl.cards.cdb import CardDB
from ygorl.data import load_environment

db, env = CardDB.load(), load_environment("md-2026-10")   # 任一环境版本
pkgs = enumerate_packages(load_or_build(), setcodes=setcodes_from_db(db))
generic = {14558127: "hand_trap", 54693926: "board_breaker", 29301450: "extra"}   # 卡密 -> 角色
sp = GenotypeSpace.from_environment(env, db, pkgs, generic)

rng = np.random.default_rng(0)
g = sp.sample(rng)                       # 随机基因型
h = sp.mutate(g, rng, n_ops=2)           # 变异（默认从 5 种算子中均匀抽）
c = sp.crossover(g, h, rng)              # 包级交叉
deck = sp.decode(c, name="elite-17")     # ygorl.cards.ydk.Deck，可直接 to_ydk()
x = sp.vector(c)                         # int8 计数向量（代理模型 / pyribs 用）
sp.from_vector(x + rng.normal(size=len(sp)), rng)   # 任意实数向量 → 对应的合法基因型
```

`rng` 处处是 `numpy.random.Generator` 或整数种子；给定种子，采样、变异、交叉、修复的结果完全确定。

## 表示

**基因型空间 `GenotypeSpace`** 绑定一个环境（卡池、禁限表、构筑规则、`Environment.stamp()`），固定一张**卡片索引**：

- 索引 = 所有保留下来的引擎包成员 ∪ 泛用池，**主卡组卡在前、额外卡组怪兽在后，两段各按卡密升序**。
  `sp.passwords[i]` 是第 `i` 项输出到牌组里的卡密，`sp.n_main` 是主卡组段长度，`sp.is_extra`、`sp.is_generic` 是布尔掩码。
- `sp.cap[i] = min(max_copies, banlist.limit(原卡号))`，即这张卡最多能放几张（3 / 准限制 2 / 限制 1）。
- `sp.packages[p]` 是第 `p` 个引擎包（限制到本空间后的成员卡密），`sp.package_source[p]` 是它在输入包列表中的位置。
- `sp.roles`：泛用卡的角色（`hand_trap` / `board_breaker` / `staple` / `extra` 等，自由字符串），
  `sp.role_counts(g)` 返回各角色的张数，T5.8 的「手坑数」描述符直接取 `role_counts(g)["hand_trap"]`。

**基因型 `Genotype`** 只有两个字段：

| 字段 | 含义 |
|------|------|
| `packages` | 选中的引擎包编号（升序元组）。只有选中包的成员与泛用卡才**允许**出现 |
| `counts` | 只读 `int8` 计数向量，长度 = 索引长度，`counts[i]` 是第 `i` 张卡的份数 |

设计文档中的三段都落在 `counts` 上：引擎包份数 = 选中包成员的份数；泛用槽 = 泛用池卡的份数；
额外卡组 = 索引后段（选中包的额外卡组成员 + 泛用额外卡组怪兽）的份数。
**牌组只由 `counts` 决定**（`decode`：主卡组 = 前段按份数展开，额外卡组 = 后段展开，副卡组为空），
`packages` 决定哪些卡可以有份数，并为包级算子提供结构。`Genotype` 可哈希、可比较（`key()` 为字节串）。

### 构造时的过滤（硬约束的第一层）

| 规则 | 处理 |
|------|------|
| 卡池 | 原卡号在卡池中则输出原卡号；只有该异画卡号在卡池中则输出异画卡号；都不在则丢弃 |
| 异画（同名、`alias` 指向原卡） | 折叠为一个条目，份数合并计算，规则与 `ygorl.cards.legality.canonical_password` 完全一致 |
| 禁止卡、衍生物、卡片数据库里没有的卡 | 丢弃 |
| 主 / 额外卡组 | 按卡片类型：怪兽且带融合 / 同调 / 超量 / 连接位 → 额外卡组段，其它 → 主卡组段 |
| 引擎包 | 限制到上述卡之后，成员少于 `min_package_size`（默认 2）或没有主卡组成员的包丢弃；限制后成员相同的包只保留一个 |
| 可行性 | 所有主卡组卡的 `cap` 之和小于主卡组下限、保留的包少于 `min_packages`、`main_range` / `extra_size` 超出规则时抛 `ValueError` |

### 搜索空间参数（不是规则）

| 参数 | 默认 | 作用 |
|------|------|------|
| `min_packages` / `max_packages` | 1 / 3 | 选中包的数量；`max_packages` 约束采样、`add_package` 与交叉，修复只在主卡组无法凑够下限时才会超出它 |
| `main_range` | `(main_min, main_min + 5)`，即 40–45 | 主卡组张数范围，必须在规则的 40–60 之内；需要完整范围时传 `(40, 60)` |
| `extra_size` | 规则的 `extra_max`（15） | 额外卡组总是尽量填满到这个张数（额外卡组的卡不稀释抽卡） |
| `max_generic` | 12 | 随机采样时最多选多少种泛用主卡组卡 |
| `min_package_size` | 2 | 见上 |

`sp.fingerprint` 是索引、`cap`、包、角色与上述参数的 SHA-256，`sp.stamp` 是环境戳；持久化基因型用
`genotype_to_json(g)`（`{"space": fingerprint, "packages": [...], "cards": {卡密: 份数}}`），
`genotype_from_json` 会拒绝来自其它空间的基因型。

## 修复（硬约束的第二层）

所有算子都以 `repair(packages, counts, rng, main_target=None)` 结束：

1. 选中包不足 `min_packages` 时随机补包；
2. 份数裁剪到 `[0, cap]`，不允许的卡（既不是泛用卡、也不在选中包里）清零；
3. 主卡组目标张数 = `main_target`（缺省为当前张数）夹到 `main_range`；允许卡的剩余空位不够时随机加入未选中的包；
   超出则在所有份数中均匀随机去掉，不足则在所有空位中均匀随机补；
4. 额外卡组超过 `extra_size` 时随机去掉；不足时先给未出现的允许卡各补 1 张，再按空位补，直到 `extra_size` 或无卡可补。

已满足不变量的基因型原样返回，且不消耗随机数。`sp.check(g)` 列出违反的不变量
（份数越界、不允许的卡有份数、主卡组张数不在 `main_range`、额外卡组张数不等于 `min(extra_size, 可用容量)`、包数不足），
空列表即有效；有效基因型解码后必然通过 `validate_deck`。

## 算子

**随机采样 `sample(rng)`**：包数在 `[min_packages, max_packages]` 均匀取，包从全部包中无放回均匀抽；
选中包的主卡组成员各取 1..`cap` 张，额外卡组成员各 1 张；再均匀选 0..`max_generic` 种泛用主卡组卡，各 1..`cap` 张；
主卡组目标张数在 `main_range` 内均匀取，最后修复。

**变异 `mutate(g, rng, n_ops=1, ops=None)`**：每步从 `ops`（默认全部）中均匀选一个算子，然后修复：

| 算子 | 作用 | 主卡组张数 |
|------|------|------|
| `copies` | 任选一张允许的卡，改成 `0..cap` 中另一个份数（额外卡组的卡改为 0 时相当于换卡） | 可漂移（仍在 `main_range` 内） |
| `swap_generic` | 把一种泛用主卡组卡的全部份数换成一种未出现的泛用卡（份数按新卡 `cap` 截断） | 保持 |
| `add_package` | 选入一个未选中的包并按采样规则给其成员份数；已达 `max_packages` 时先随机移除一个包（换包） | 保持（修复时从其它卡中随机让出位置） |
| `drop_package` | 随机移除一个选中的包，其独有成员清零；已达 `min_packages` 时改为换包 | 保持（从剩余允许卡中补足） |
| `swap_extra` | 把一种额外卡组怪兽换成一种未出现的允许额外卡组怪兽 | — |

不能执行的算子（没有可换的卡、没有未选中的包）退化为 `copies`。

**交叉 `crossover(a, b, rng)`（包级）**：双亲都选中的包保留；只在一方的包各以 1/2 概率加入（总数不超过 `max_packages`）；
每个保留包的成员份数整体取自选中它的那一方（双方都选中时随机一方）；泛用卡逐张随机取自一方（均匀交叉）；
主卡组目标张数取自随机一方；最后修复。相同的双亲交叉得到自身。

**编码转换**：`from_vector(x, rng)` 把任意实数向量四舍五入、裁剪到 `[0, cap]`，把「有非泛用成员份数 > 0」的包视为选中，再修复——
供 pyribs 的连续 emitter（CMA-ME 等）使用；`from_deck(deck, rng)` 把现成牌组（如环境的 meta 牌组）编码进空间（空间外的卡忽略），
可作为档案的初始解。

## 数值编码（供 T5.7 代理模型与 T5.8 pyribs）

- **计数向量** `sp.vector(g)`：`int8`，长度 `len(sp)`，`[0 : n_main]` 为主卡组份数，`[n_main :]` 为额外卡组份数，
  取值 `0..cap[i]`。代理模型的「计数向量」输入就是它（可再按 `sp.passwords` 查卡文本嵌入求均值）；
  pyribs 的解向量维度 = `len(sp)`，上下界 = `[0, sp.cap]`，经 `from_vector` 解码为合法基因型。
- 索引是**空间相关**的：只在同一个 `fingerprint` 的空间内可比。跨空间 / 跨环境对齐时按 `sp.passwords` 映射到全局的
  `ygorl.cards.cdb.CardVocab` 索引。
- 包选择不在向量里：它可以从向量推断（`from_vector`），并且对牌组没有影响。

## 验收：10k 随机基因型全部合法

`tests/test_genotype.py::test_real_10k_random_genotypes_are_legal` 在默认测试集中运行（约 2 s）：用真实协同图的全部引擎包，
构造测试格式——卡池 = 排名前 60 的包的成员 + 泛用池（`tests/data/generic_pool.json`：7 张手坑、7 张解场、6 张泛用魔陷、
15 张泛用额外卡组怪兽；泛用池另含一张不在卡池里的 Ash Blossom 异画卡号，必须折叠为原卡），禁限表对每第 7 个包成员依次设为禁止 / 限制 / 准限制，
另把 Maxx "C" 与 S:P Little Knight 设为禁止、Ash Blossom 与 Dark Ruler No More 设为限制——随机采样 10,000 个基因型，
逐个用 `Environment.validate_deck` 校验。另有 2,000 步交叉 + 变异链的回归测试，以及绕过修复的反例（验证校验器确实会报错）。

`uv run python tools/sample_genotypes.py`（同一测试格式，`--environment <version>` 可换成真实环境）输出（2026-09-22，当前子模块版本）：

| 项目 | 结果 |
|------|------|
| 空间 | 卡池 910 张 → 索引 866 张（主卡组 706 / 额外卡组 160），95 个包，33 张泛用卡；`cap` 分布 3 张 780、2 张 42、1 张 44 |
| 随机采样 10,000 个 | 用时 1.50 s（150 µs / 个），校验 0.63 s；**10,000 合法、0 非法**；10,000 个基因型、10,000 套牌组两两不同 |
| 主卡组张数 | 40–45 各约 1/6；额外卡组全部 15 张；每个基因型 1 / 2 / 3 个包各约 1/3 |
| 构成 | 泛用卡平均占主卡组 35%；平均手坑 4.5、解场 5.4、泛用魔陷 5.0、泛用额外卡组 11.2 张 |
| 交叉 + 变异 10,000 步（种群 100，每步 1–3 个变异算子） | 3.89 s；**10,000 合法、0 非法** |

合成小卡表上的单测覆盖每条过滤规则（禁止卡、衍生物、卡池外、异画折叠与只有异画在卡池的情况、限制 / 准限制的 `cap`、
额外卡组成员 20 张的包）、每个算子单独连续执行 300 次、交叉 600 次、从随机噪声向量修复、自定义构筑规则
（20–30 张 / 额外 5 张 / 同名 2 张）与完整 40–60 范围。

## 局限

- **修复是均匀随机的。** 补位与裁剪不区分启动卡与其它成员，也不看协同图；随机采样得到的是「合法且有结构」的牌组，
  不是「好」牌组。质量交给漏斗与 QD 搜索（T5.6–T5.8）；需要时可以按 `Package.starters` 或边权给补位加权。
- **泛用池是临时数据。** 真实环境的泛用池应来自环境的 meta 卡表与使用率（T5.1）；`tests/data/generic_pool.json` 只是测试替身。
- **副卡组不参与搜索**（解码为空）。`max_packages` 在修复为凑够主卡组下限而补包时可能被超过（测试格式的 10k 随机采样中从未发生）。
- **包是候选集合而非划分。** 包之间可能共享成员；共享成员在交叉时取自后处理的包，删除包时只清零不再被任何选中包或泛用池覆盖的卡。
