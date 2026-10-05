# Lunalight 失败起手可达性诊断（2026-10-05）

## 冻结协议

本轮是失败归因，不是追加上一轮样本或修改已关闭的 gate。
输入固定为 `material-guidance-2026-10-05` 两臂共同未解出的 13 个普通洗牌起手：
indices 0,1,2,3,5,6,7,8,9,10,12,13,14（hand seed 来源 2026100506）。
直接复用记录的双方牌序、核心 seed、规则及目标 Liger Dancer 54701958。

每手一次新搜索，solver seed=2026100507，120,000ms，1 native thread，深度 692，
`--no-ref --start --novelty 0 --no-serial --reenter 0`；无 recipe/backward、参考引导、prefix、fire。
显式关闭 novelty pruning 以寻找被短预算/剪枝漏掉的路线；这同时改变预算、solver seed、novelty，
只能用来证明“存在合法成功线”，不能归因为某一个开关的因果收益。
4 CPU workers、nice19、max-written=4、保留 1 条完整验证线，timeout=150s。
所有失败保留；无解只能标为 unknown。成功线须核对精确起始身份、turn2、原目标并独立重放。

结果将决定后续工作：先核对新增可达证据，再针对仍未知起手检查发动/资源路径。
不同诊断阶段独立记录成本和条件，不能回填上一轮 30 秒 gate，也不能声称穷尽搜索。

## 结果

搜索生成提交 `8495077`，验证修复 `fdea761`；当前 MD 完整环境指纹
`65ca28f79233e73d540a23c2046718ef42749ae883baa920216b94365b3e42d9`。
13 手 native 搜索共 977.94 秒，未出现 native 非零退出或 timeout。

搜索结束时原始记录为 6 unverified / 7 unsolved：所有 24 个候选（每成功手 4 个）
在二次回放被同一错误拒绝：带环境标签的 expected Replay 调用 `play()` 时漏传 env。
这是候选收集的边界 bug，不是非法动作或引擎失败；原始记录、候选及输入保持不变。
`_collect` 现显式接收 env、核对 expected 的环境指纹，并传递给二次回放；
`solve_hand` 传递已加载的同一环境。新回归测试覆盖合法环境成功与相同规则但错误指纹仍拒绝，
没有通过清除环境身份绕过校验。

修复后只重验保存文件，**零次新搜索**：

- 找回 indices **3,5,8,9,12,13**，6 条完整 Liger 线；584 步独立回放及 584/584 native 响应覆盖通过。
- indices **0,1,2,6,7,10,14** 仍 unknown；不作不可达结论。
- 修复后的 13 条记录 0 error/unverified/拒收。加上前轮已知成功的 4,11,15，旧面板至少 **9/16 可达**。
  这是不同配置和追加预算下的可达性下界，不是单个 30 秒教师的通过率。
- 验证恢复可零工作续跑；旧 `diagnostic.jsonl` SHA256 始终为
  `065ba07e8a43d7794ef75de66292635ff14d4abc02c4f58134f956a2f70aab34`。

手 3 说明不能凭直觉认定卡手：Wolf + Chick 的真实成功线使用 Yellow Marten 回收 Wolf、
Dugares 与 Perfume 等途径续接，118 个 host 动作完成最终场面；完整响应由引擎独立核验。
本轮同时改变 seed/预算/novelty，不能把找回的全部收益单独归给关闭剪枝。

## 持续推进

已经冻结 [独立全牌组普通洗牌覆盖](ordinary-coverage-2026-10-05.md)：新 seed 2026100508、
20 牌组各 16 手；Lunalight 固定 120 秒/novelty0，其它牌组30秒/novelty12，原目标不变。
所有牌组分别 ≥8/16、零验证错误后才进入正式训练/留出冻结；诊断结果不混入新面板。

## 工件与验证

`out/research/reachability-2026-10-05/` 保存 `run_diagnostic.py`、原始 `diagnostic.jsonl` 和 manifest、
全部逐手输入/命令/native 日志/候选；原始失败明确保留。
`revalidate.py` 从相同文件建立独立验证 manifest，绑定生成身份、修复后 validator 身份和原始文件哈希，
生成 `verified.jsonl`、`verified-summary.json`。native 回放覆盖日志及恢复续跑日志也保留。
生成时 Python 对应 `8495077`，验证时对应 `fdea761`，solver/core/cards/scripts 身份完全一致。
修复后原搜索 driver 的旧 manifest 会拒绝混入新实现；恢复只在新的验证输出中进行。

solver 单测 34 passed；完整 `tools/presubmit.sh --test` **1,426 passed / 3 skipped，285.24 秒**，
含实际 ROCm。三个跳过为 snapshot 编译选项和两个可选联网测试。已测运行实现 `fdea761`，
后续仅文档变更；`validation.json` 封存 592 个工件文件哈希。
