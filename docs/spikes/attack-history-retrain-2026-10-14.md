# Bounded attack-history retraining, 2026-10-14

Status: protocol registered before collection. The preceding four-update canary
established execution health, not a strength improvement. This experiment tests
the repaired input and rollout path with successful-game evidence retained.

Artifacts: `out/research/attack-history-retrain-2026-10-14/`. The registration,
runner, evaluation code, source checkpoint, tracked runtime files, native module,
card database and scripts are hashed before launch. `identity.json` binds the
runtime commit. `report.json` is written only after both training and evaluation
pass their health/replay checks.

## Fixed experiment

- Two training seeds (`2026101401`, `2026101402`), attack history off/on per seed.
- Restore both actor and critic from seed-0 warm u128 in
  `terminal-critic-policy-long-2026-10-06`; introduce the two new history
  embedding rows as zeros. Verify all existing parameters are preserved.
- Matched fresh Adam, reference and opponent pool in both arms. This is a warm
  restart, not continuation of the historical optimizer/pool.
- 16 updates × 1,024 rows per arm/seed: **65,536 training rows total**. Eight
  native environments, 128 steps, full-batch deterministic receive order.
- CPU FP32; retain source reward, objective, lambda, learning rate, deck pool,
  event length and self-play settings. Selection history stays off and cancel
  budget stays 32 in both arms. No corrective labels from selected bad games.
- Save actual rollouts, checkpoints before and after every optimizer update,
  optimizer/reference/RNG state, and replayable finished-game traces with each
  explicit action's collection model version. Replay every completed training
  game exactly. Count unfinished slots at the fixed endpoint separately; these
  are not losses or finished-game samples.

## Evaluation and decision

Use 64 fresh deal clusters (`2026101403`), each with all four deck/start
assignments, against Greedy and the frozen source warm-u128 policy. Evaluate two
initial schemas, four final models and frozen warm-u512 as a reference: **3,584
evaluation games**. Verify the two seeds' initial model tensors are identical
within each schema before sharing the initial evaluations. Every evaluation game
must pass exact cold replay; the first game per cell also checks wrapper parity.

Primary contrast: context minus control score, averaged across the two training
seeds and opponents, paired by deal cluster. Also report initial schema effect,
difference-in-differences, each seed/opponent, length quantiles, >20-turn rate,
cancel counts, and multi-attacker training coverage. Bootstrap whole deal
clusters, retaining the paired assignments and opponent/seed measurements.
These intervals are conditional on the two training seeds; they do not establish
robustness across a population of training seeds.

A five-percentage-point improvement would be useful, but this bounded diagnostic
is **not powered to reliably detect 5 pp**. The earlier eight-cluster
training-difference SD was roughly 0.25; taking that unstable estimate literally
suggests an approximately 9 pp 80%-power detectable effect at 64 clusters.
Unclear results remain unclear. No automatic extension, checkpoint selection,
promotion, or change to the production default follows from this run.

Any unhealthy/truncated game, nonfinite parameter, identity drift, replay
mismatch, <20 GiB free space, or wall-time breach stops the pipeline and preserves
evidence. Per-arm training timeout is 30 minutes; evaluation timeout is three
hours. A separate service monitors the bound invocation, heartbeat, progress,
STOP records and completion identity. Results must be reviewed before choosing
another experiment; fresh deals here do not mean unseen cards or unseen decks.
