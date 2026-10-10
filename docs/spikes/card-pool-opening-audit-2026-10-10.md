# Opening-deal reuse: evaluation correction and critic follow-up

The card-pool follow-ups did not generate the fresh opening deals claimed in
previous reports. Exact result/response replay and new engine RNG seeds passed,
but neither check proves opening independence. Preserve the original artifacts;
withdraw their new-opening interpretation before adapting the critic.

## Mechanism and scope

`pairing_slots` explicitly pre-shuffles deck lists and sets `shuffle_decks=False`
so checkpoints, opponents and seat swaps receive common deals. Later one-off
preparation scripts copied those lists and changed only core/agent RNG seeds.
`Duel.loaded_decks()` correctly preserves their order in this configuration;
the core does not shuffle at duel start. The failure is in follow-up panel
construction and its missing coverage check, not a newly discovered core RNG bug.

| Panel | Games | Distinct ordered main-deck pairs | Meaning |
|---|---:|---:|---|
| Original coverage | 1,024 | 64 | Actual original two-draw, two-role panel |
| Pool-training canary | 768 | 64 | Reused coverage orders, new continuation RNG |
| Each training replication | 768 | 64 | Same coverage orders again |
| Retention confirmation | 2,304 | 16 | Only original draw0, facing role; four nominal draws reuse it |

All listed jobs explicitly disabled engine-side shuffling. Retention games
228–231 have distinct engine seeds but identical replay-loaded top five cards
for both decks. This is direct evidence in saved replays, not just a diagnosis
from generator code. Counts distinguish deck roles; seat swaps do not count as
new orders. Full file hashes and concrete examples are saved in
`out/research/card-pool-fresh-deals-2026-10-10/legacy-opening-audit.json`.

The original outcomes, engine health checks and conditional action interventions
remain observations of those specific states. The four nominal retention draws
must not be bootstrapped as independent opening clusters. Previously published
retention intervals are withdrawn for new-opening inference; the new panel is
not pooled with them. Seed-disjoint replication was not deal-disjoint replication.
This finding alone does not invalidate the separately generated training runs:
Trainer constructs the normal shuffle-enabled DuelConfig, and both Python and
native training hosts consume `Duel.loaded_decks()`. It also does not automatically
invalidate other experiments whose deal generation has not been audited here.

## What the completed critic audit establishes

The 2,304-game critic replay was mechanically verified, including response bytes,
actor probabilities, frozen identities and predictions. It inherits the same
16-opening restriction. Game-averaged V bias and squared terminal-return error
on the Runick slice (96 games per arm) are:

| Checkpoint arm | Bias | MSE |
|---|---:|---:|
| Start | +0.2866 | 1.1350 |
| Fixed pool | +0.5553 | 1.3432 |
| Expanded pool | +0.4529 | 1.2185 |

Predicting zero has terminal-return MSE 1. Familiar-deck V bias instead ranges
from -0.2976 to -0.2232. Thus a global calibration shift is not supported even
within these sampled trajectories. Reviewed-card presence correlates with larger
overestimation but does not prove that a lock is active or identify its cause.
Possible explanations include sparse coverage of alternate win conditions,
features that work in ordinary combat becoming misleading, and a fixed-opponent
value distribution different from the training league. These remain hypotheses;
the flawed opening coverage prevents a broad root-cause conclusion.

## Repair and replacement evaluation

`ygorl.eval.panel_deals.redeal_job` shuffles canonical card multisets explicitly.
Its key contains panel seed, target, anchor, role and draw; checkpoint, opponent
and engine seat are excluded. It preserves deck contents and extra decks and
binds the generated order plus engine/agent RNG in job metadata. Validation fails
on stale metadata, double shuffling, prior-order overlap, or different draw labels
sharing an order. Repeated opening hands can legitimately occur; their count is
reported without rejecting or resampling them. Engine-loaded deck orders are
checked before play. Regression tests query actual engine hand contents across
models, opponents and seat swaps and reproduce the old seed-only failure.

Generate future panels through the shared helper, rather than copying prepared
jobs and assigning RNG seeds:

```sh
python -m ygorl.eval.panel_deals --input old-panel.json --output panel.json \
  --panel-seed 202610109101 --prior another-prior-panel.json
```

For `card_generalization_games.run`, register `fresh_opening_deals=true` and
`prior_panels` (including the source panel); those files must also be identity
bound. The runner validates coverage before scheduling any games. Legacy replay
studies keep their explicit fixed-state semantics and original artifacts.

The replacement `card-pool-fresh-deals-2026-10-10` registers 2,304 games on **64
new ordered pairs and 64 opening-hand pairs**, with zero order overlap against
all five prior panel files. Four deals per target/anchor are shared across nine
existing checkpoints, two opponents and both seats. It uses the original frozen
runtime so that reshuffling, rather than a concurrent engine patch, is the
intervention. All full games receive exact cold replays and terminal games also
receive the same no-gradient critic audit. Limits remain non-wins and are reported
as censored for terminal critic calibration. No actor or critic optimization,
reward change or policy promotion occurs in this measurement stage.

Primary comparisons remain fixed-minus-start and expanded-minus-start facing
added decks, with familiar retention and deck/opponent breakdowns. Bootstrap
blocks now correspond to verified target/anchor/draw orders, shared across
checkpoints and seats. Only three existing training seeds and eight target lists
are covered; this is not proof of broad card understanding or elite strength.

If Runick overestimation persists on this corrected panel, the next critic-only
experiment will use separately generated training/validation/test deals, frozen
actor parameters and short validation-selected fits. The evaluated old and new
panels must not become critic training data. Previous terminal-critic work showed
that extended fitting could overfit, so lower training loss is not a promotion
criterion; subsequent matched RL continuation and retained gameplay performance
are required.

Validation at repair: 79 distinct focused tests passed across panel deals, arena,
agent matrix, duel and monitoring suites; lint and whitespace checks passed. Two
end-to-end frozen-runtime preflight games also passed actual-order checks, full
cold replay, critic response replay and actor-probability parity. The replacement
run passed all 18 registered health canaries and entered full evaluation with an
independent invocation-bound monitor. Completion results remain pending.
