# 未解出与拒收的根因调查（2026-10-05）

## 已观察事实与诊断协议

320 手冻结覆盖仍在完成中，原产物和二进制不变。独立诊断不回填覆盖 gate。

- Sky Striker index 5：四条候选都在双 Linkage 连锁中途首次出现 Shizuku 时停止；下一条强制送墓选择只剩 Shizuku，后续连锁将它送墓。Python 回放正确拒收，并非无解证明。调查原生 GoalCheck 缺少未结算连锁检查；保留原候选、逐步 trace，再验证稳定节点检查是否恢复完整线。
- Odion index 0：日志明确有 类型位图 14 无候选且无 default response，真实搜索分支被杀。捕获后发现实际类型为 MSG_ANNOUNCE_CARD（142），被日志 `type & 63` 显示为 14；不是 MSG_SELECT_CARD。具体为声明密码 62514770 的过滤表达式。下一步核对 host 合法候选与原生声明枚举。
- 对最终所有 unsolved/unverified 起手汇总原生日志：搜索耗时、最佳目标覆盖、状态数、turn/constraint/novelty/subset/dead-end 裁剪。日志预算耗尽不能证明预算是唯一根因。

诊断二进制单独构建，仅增加有限的空 prompt/前缀记录，先复现 Odion 0 的具体失败。
修复前先用捕获的合法 prompt 做原/修复枚举对照；修复搜索语义后对失败起手开展固定预算的独立诊断，
成功只证明原搜索漏解，仍未找到记 unknown。禁止降低目标或跳过结束回合验证。

## 实现范围（先于修复）

- 在原生 GoalCheck 判定目标匹配后，用已有 ProcessorState/LastChainLink 查询拒绝未结算连锁内的候选；搜索继续，host 结束回合检查保持严格。
- 原生 MSG_ANNOUNCE_CARD 支持 `[password, OPCODE_ISCODE]` 这个完整过滤器形状；必须在数据库中存在且满足核心默认 alias/token 规则。未知或复杂表达式继续拒绝猜测答案。
- 两个补丁随本仓库构建脚本应用到独立源码副本，补丁内容进入 build stamp；不编辑 pinned upstream checkout，也不覆盖正在运行的旧二进制。诊断记录具体实际 binary hash。
- 验证捕获的声明前缀（33 响应后出现唯一合法答案 62514770）原枚举覆盖缺口与修复后覆盖；验证 Sky Striker 5 在同起始身份下产出可完整结束回合的线。随后才对所有未解出进行单列修复后诊断。

## 修复后诊断冻结

原始覆盖最终为 265 solved / 54 unsolved / 1 unverified（Sky Striker 5），Lunalight 7/16、Odion 5/16，gate 失败。
对全部 55 个失败记录，在完全相同 start、目标、seed=2026100508、每牌组预算、novelty 与深度下重跑修复后二进制；
4 workers、nice19。独立 output/manifest，不回填 320 手。修复前后墙钟预算具有搜索调度波动，
只有捕获前缀上的确定性回归可直接归因；重试解出提供漏解证据，不将联合成功率冒充新独立通过。
同时逐手记录 starter/必需访问路径和裁剪日志；找不到完整线的记录继续 unknown，
已证明被规则禁止的起手才单独列为目标不可达，不用“像卡手”推断。
