# Warm u512 cancellation-budget confirmation (2026-10-10)

The isolated **T=1 sampling budget32 versus budget1** experiment completed all 3,072 registered games. Every source/model hash, cold replay, saved result, paired input and mask intervention was independently verified. The preregistered gate to consider a guarded training canary passes; this does not promote the default or establish learned improvement.

Study identity `03bcb8278128a5b4f19906254a91bb0f85c421f872af9ee36df891f46e9d3003`; frozen Clown-fixed runtime `885e063`. Three warm u512 models, four opponents, 32 fresh shared deal blocks with four deck/seat assignments, 1,536 games/arm. Completion event `f0d4dcf508ef6cf12906f95a0499efb3f3be0ce9862dd3820c6b744c8a7ddc29` acknowledged. All games ended naturally, with zero engine/replay failures or termination limits. Registered files remain unchanged.

## Results

| Metric | Sampling budget32 | Sampling budget1 |
|---|---:|---:|
| Primary wins (first three opponents) | 1000/1152 (86.81%) | 999/1152 (86.72%) |
| Historical opponent wins | 287/384 | 288/384 |
| All-panel wins | 1287/1536 | 1287/1536 |
| All-panel mean turns | 9.2005 | 9.1960 |
| All-panel mean decisions | 430.158 | 429.846 |
| All-panel P95 decisions | 824.50 | 821.75 |
| All-panel cancels | 265 | 65 |
| Games with ≥8 cancels | 5 | 0 |
| Maximum cancels/game | 35 | 2 |

Primary win change **−0.0868 pp**, crossed95% interval **[−0.5208,+0.3472] pp**, above the registered −2pp noninferiority margin. Each opponent's point change is within ±0.2604pp. All-panel mean-turn change **−0.00456 [−0.03906,+0.02865]**; no evidence of shorter games or stronger play. Cancellation count drops **75.47%**; all five observed ≥8-cancel games disappear. No sole-exit fallback was needed in this panel; support preservation is additionally covered by unit tests.

The P95 gate uses the complete four-opponent panel. Primary-only P95 decisions increases slightly, **868.35→870.00**; primary mean decisions also increases **0.144 [−1.209,+1.788]**. Do not claim every tail or every matchup improves. The new deal panel contains only five severe-cancel games: its absolute loop frequency is much lower than the earlier fixed panel. These populations must not be pooled into a before/after training claim.

Twenty thousand paired bootstrap draws resample three fitted training seeds and the shared 32 deal blocks jointly. A separate multiplicity-weight computation reproduces every interval. Fixed models, one deck pool and fixed opponents limit generalization. All five registered checks pass: health, primary noninferiority, per-opponent point-loss bound, ≥50% reduction of ≥8-cancel games, and no increase in overall P95 decisions/limits.

## Training integration

`TrainConfig.cancel_budget` and `EncodedVecEnv(cancel_budget=...)` accept 0/1/4/32, default **32**. `HostDuel` / `HostPool` intersect the existing encoded native mask after native undo/deduplication/fallback rules. They preserve a sole remaining exit and never enable an action. Budget32 leaves native behavior intact. Tracker counters and engine responses are unchanged; this is an observation-mask intervention, not a new legality rule.

The mask is applied before native forced-action skipping, policy/critic evaluation, sampling and rollout storage. PPO therefore sees the same mask that generated its behavior probabilities. The configuration survives checkpoint save/resume; old checkpoints default to32. This environment option applies to **both engine seats, including frozen pool opponents**. The existing inference diagnostic affects only the candidate. These are different experiments and are explicitly distinguished in the canary protocol. Exported actor checkpoints do not implicitly turn on this environment guard during ordinary inference.

Tests cover native/reference masks on real target and material cancellation windows for all four budgets, explicit engine-response parity, pooled masks, legacy/default configuration, save/resume, and stored-mask / sampled-probability consistency. The native extension is rebuilt in an isolated build directory, without modifying the root core submodule or frozen runtimes. Guard, selection-history, checkpoint, PPO and trainer suites pass.

## Next canary

`out/research/warm512-cancel-training-canary-2026-10-10/` registers two arms from **the same seed0 warm u512 checkpoint**, preserving actor/critic/optimizer/reference/pool/schedule/CPU-GPU-collector RNG and counters. Sixteen additional updates per arm (budget32 control versus budget1 for both seats), checkpoints every4 updates, health checks before optimizer, separate study/watchdog services. Active duels restart equally. No reward, argmax, precision or supervised-refresh change.

Then evaluate the starting checkpoint and both u528 endpoints on 16 fresh paired deal clusters, all four opponents, with each inference budget32/1: **1,536 games**. Unguarded evaluation of the guard-trained actor is essential to distinguish learning from guard support. Review retention before any larger continuation. Single-seed canary results cannot select a universally better recipe. No default or policy promotion is authorized by this report alone.

Evidence: `out/research/warm512-cancel-budget-review-2026-10-10/review.py`, `review.json`, `review.log`; original reports and 3,072 replay/result files under the confirmation study. Related #83, #218, #239.

## Training canary completed: no policy promotion

Both seed0 arms completed 16 updates / 262,144 new rows, with 1,869 games for
budget32 and 1,896 for budget1. Training reported zero errors and truncations.
The 1,536-game fresh panel completed with zero engine-health errors/limits;
all cold replays matched and their stored hashes were checked.

Each cell below has 256 games across the same four opponents and 16 deck-pair
clusters. “Guard” means budget1 at inference; otherwise inference uses budget32.

| Checkpoint | Inference | Win rate | Mean turns | Games with >1 repeated selection cancel |
|---|---|---:|---:|---:|
| Initial u512 | sample | 81.25% | 8.922 | 10 |
| Initial u512 | guard | 80.08% | 8.945 | 0 |
| Train budget32 u528 | sample | 81.64% | 9.121 | 15 |
| Train budget32 u528 | guard | 81.64% | 9.094 | 0 |
| Train budget1 u528 | sample | 76.56% | 9.055 | 13 |
| Train budget1 u528 | guard | 75.78% | 9.105 | 0 |

Holding inference mode fixed, budget1 training minus budget32 has win-rate
difference **−5.08 pp** (paired deck-cluster bootstrap 95% CI −11.72 to +1.95)
without the guard, and **−5.86 pp** (−12.11 to +0.78) with it. These exploratory
intervals are conditional on one training seed and a small fixed opponent/pair
panel. They do not prove a population-level regression, and do not support
claiming improvement or noninferiority. Mean turns do not show a useful decrease.

The guard mechanically removes repeated cancels, but budget1 training does not
reliably internalize that behavior: disabling the guard still produces repeat
cancels in 13/256 games, with a maximum of 32. No automatic promotion or reward
change. The next strength decision needs paired replication across training
seeds and review of divergent games; cancellation count alone is insufficient.

Evidence: `out/research/warm512-cancel-training-canary-2026-10-10/analysis.json`
and its reproducible `analyze.py`; original report, checkpoints and replays remain
unchanged. Related #83, #218, #239.
