# 未解出与拒收的根因调查（2026-10-05）

## 已观察事实与诊断协议

320 手冻结覆盖仍在完成中，原产物和二进制不变。独立诊断不回填覆盖 gate。

- Sky Striker index 5：四条候选都在双 Linkage 连锁中途首次出现 Shizuku 时停止；下一条强制送墓选择只剩 Shizuku，后续连锁将它送墓。Python 回放正确拒收，并非无解证明。调查原生 GoalCheck 缺少未结算连锁检查；保留原候选、逐步 trace，再验证稳定节点检查是否恢复完整线。
- Odion index 0：日志明确有 MSG_SELECT_CARD（14）无候选且无 default response，真实搜索分支被杀。先记录具体 prompt 和前缀，用 host action model 重放，检查是否按 card code 去重丢失多张同名卡的选择数量；在获得具体证据前不归因于手牌或预算。
- 对最终所有 unsolved/unverified 起手汇总原生日志：搜索耗时、最佳目标覆盖、状态数、turn/constraint/novelty/subset/dead-end 裁剪。日志预算耗尽不能证明预算是唯一根因。

诊断二进制单独构建，仅增加有限的空 prompt/前缀记录，先复现 Odion 0 的具体失败。
修复前先用捕获的合法 prompt 做原/修复枚举对照；修复搜索语义后对失败起手开展固定预算的独立诊断，
成功只证明原搜索漏解，仍未找到记 unknown。禁止降低目标或跳过结束回合验证。
