# tools/scale — 规模化实验的驱动脚本（2026-10 先后手差研究）

从 `out/claude-scale/` 复制而来，供换机继续时参考；绝对路径（`/home/ya0guang/Code/ygorl/out/...`）需按新机器修改。背景与读数见 `docs/spikes/first-seat-interruption-plan-2026-10-24.md` §14.3。

| 文件 | 用途 |
|---|---|
| `run_arm.py` | 从检查点续跑一条臂：`--ckpt CKPT --out DIR --updates N --chunk 64 [--bf16] --set key=json --ppo-set key=json`；自动从 `DIR/checkpoints/latest.pt` 续跑，每 chunk 存一个不可变检查点 |
| `panel.py` | 扩展面板（旧 4 对手均值 + 强锚点 G576 + 老师）：`panel.py CKPT --workers 4` → `CKPT 同目录/panel/update_N/summary.json` |
| `panel_g576.py` | 统一口径的面板行（G576 单项；先手/后手按除 E2016 外的对手加权），新旧 summary 都能比 |
| `panel_row.py` | 旧的逐席位行打印（依赖旧目录布局） |
| `panel_watch.sh` / `colab/panel_colab.sh` | 自动评每个 64 倍数检查点（LA 臂从 u2128 续跑，分块在 2192…，脚本里有例外） |
| `colab/rulers_colab.sh` | 手坑尺子循环：`turn1_board.py` + `tools/handtrap_ruler.py` → `colab_arms/rulers.jsonl` |
| `la_ruler.sh` | 前瞻尺子循环：`tools/seat_gap_decomposition.py --class anchors --opponent G576 --seats 0 --conditions base opp` → `la_ruler/<arm>/<ckpt>/anchors_summary.json` |
| `handtrap_eval.py` / `turn1_board.py` | 旧版手坑尺子 R2 与回合 1 打断数（R2 已进仓库 `tools/handtrap_ruler.py`） |
| `handtrap_labels_graded.py` | 分级标签 + 恢复线生成（`--fire 密码 --hands a-b --max-levels --recovery-out`，需要求解器二进制） |
| `prune_recovery.py` | 多卡恢复线剪枝 |

本地启动用 `systemd-run --user --unit ygorl-ab1-<名> ...`（macOS 用 `nohup`）；两条 128×128 臂可同跑，三条会 OOM（47 GiB 分区）。
