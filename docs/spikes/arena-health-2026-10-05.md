# Arena分母与keep-best健康修复（2026-10-05）

PR195修正了矩阵后，继续检查发现普通Arena仍把全部records计入胜负/和局，errors只统计exception，
忽略原生error、Lua/retry等字段。同步Trainer.evaluate又未检查健康即覆盖best.pt并pin最佳策略。
这会让异常局成为胜场/半胜场，甚至晋级污染面板选出的checkpoint。

设计先提交6206159，修前10项回归均失败：7类异常分母、真实Ash Lua错误、CLI错误退出、
错误面板创建best.pt。修复a24a7b3集中GameRecord.healthy供Arena/矩阵共用；
统计只用健康局，attempted_games保留全部尝试，原始记录/原因/计数保留。
任一基线错误或零有效局阻止整次keep-best；invalid完整报告写eval-errors.jsonl。
单独验证零局的另一基线也能阻止健康高分基线晋级，避免幸存样本选择。

定向79 passed（82.70s）及额外零局用例1 passed。初次全套在收集阶段发现外部工具还引用被删的
ERROR_REASONS；恢复兼容常量，并将compare_checkpoints已保存配对单元也迁移到健康谓词，
相关19 passed（1.82s），源2d47ad9。初次失败日志保留，不掩盖实现遗漏。

最终完整 `tools/presubmit.sh --test` **1,468 passed /11 skipped，695.75s**。
本轮明确关闭GPU给已冻结的RL实验，8个GPU用例未重跑；另1个snapshot选项与2个联网skip。
变更为CPU统计/晋级/日志，相关路径全部执行；不得将此写成实际ROCm全套通过。
在新汇总下重新读取此前8模型×160局的全部原始字段，**1,280局分数不变、零异常**。
工件封存 `out/research/arena-health-2026-10-05/validation.json`。

上限计分策略没有在这个修复中改变；研究驱动另外拒绝上限局。历史缺失健康信息不能因此被证明干净。
独立训练producer仍在原冻结worktree、源码与二进制不变。
另完成 [直接攻击反事实诊断](battle-counterfactual-2026-10-05.md)，未找到原败局的直接获胜替代线。
