# Battle-Q target replication and label averaging

The original fitted-label advantage largely disappears when the same training roots receive new continuation outcomes. This supports investigating noisy targets, but neither proves that noise is the only cause nor establishes an improved critic. Three controlled refits passed retention checks. A separate 3,568-game evaluation on freshly shuffled deals is running. No actor or critic is promoted.

## Independent target replication

`out/research/critic-battle-target-replication-2026-10-11` completed all **1,904 games**: 32 original training roots, all 117 supported actions, 16 new suffix streams (701–716), and 32 original-action/original-RNG controls. Models and predictions were frozen before outcomes. All games ended with a winner; no engine errors or limits occurred.

| Target sample | Unchanged centered action MSE | Relative-step32 centered action MSE | Unchanged conditional wins | Relative conditional wins |
|---|---:|---:|---:|---:|
| Original fitted labels | .021531 | .015286 | 69.531% | 72.461% |
| New suffixes | .017395 | .016888 | 71.289% | 72.656% |

The primary new-suffix MSE difference is **−.000507**, paired 95% interval **[−.003893, +.002956]**. The secondary conditional-win difference is **+1.367 pp**, interval **[−.781, +3.516 pp]**. Both include zero. Original fitted-label performance is descriptive, not validation. These are fixed-root private-Q interventions with unchanged continuation policies, not full-game actor strength.

The old/new centered label discrepancy is **.025321**. Estimated sampling-noise contributions are .013633 and .013881; their sum is **.027514**. The observed discrepancy is compatible with substantial sampling noise from 16 continuations per action. These estimates are uncertain and do not exclude representation, coverage, or optimization problems. We do not subtract estimated noise from fitted training error or treat small point differences as improvements.

The audit (`analyze.py`, `analysis.json`) checks runtime/input identity, final report identity, complete task grids, task/result agreement, replay hashes, recorded cold replay equality and every original control. Games were not independently rerun during the audit. Shared suffix indices are bootstrapped jointly over actions and roots, with 50,000 draws and seed 2026101105. The interval is conditional on registered roots, not a population-of-states interval.

## Controlled refits: old16, new16, pooled32

`critic-battle-label-average-fit-2026-10-11` changes only training targets: original 16-suffix means, independently resampled 16-suffix means, or pooled 32-suffix means. All arms use the same 32 roots and 117 actions, terminal-refresh initialization, 32 Adam steps, learning rate 1e-4, epsilon 1e-8 and gradient clipping at 1. Only the 66,177 Q-head parameters train.

The objective adds equal-root centered action MSE to prediction distillation on the original mixed50 corpus, with unit coefficients. The latter anchors initial predictions; it is not a new terminal-MC target. Actor, private encoder, trunk, resource MLP, V head and ridge remain frozen. Checkpoints are recorded at steps 0, 8, 16 and 32; step 32 is fixed in advance for evaluation, without choosing a winner on retention or branch outcomes.

Independent verification recomputed all 12 checkpoint retention forwards and gates and checked frozen parameters and checkpoint hashes. The old16 control exactly reproduces previous relative-fit parameters at **all four checkpoints**. The new16 and pooled32 optimizer trajectories were not independently rerun.

All three step32 arms pass broad, old, familiar, own-Runick and low-deck retention gates. Familiar MSE changes relative to the gate baseline are +0.142% (old16), +0.034% (new16), and +0.107% (pooled32). Passing retention does not demonstrate better action ranking or playing strength.

## Fresh source games and registered confirmation

`critic-battle-fresh-roots-2026-10-11` completed **128 source games**, all with ordinary wins and recorded cold replay equality. Explicitly reshuffled decks produce 128 distinct full deals and 128 distinct opening-hand pairs, with **zero full-deal overlap against 2,624 registered prior deals**. This is limited to the audited registry; it is not a guarantee against every historical game or unseen-card generalization. Existing card/deck pools and fixed learner/opponent policies remain.

Before source outcomes, root selection was fixed to eight games per broad/Runick corpus × old/mirror opponent × own/face role stratum. Games are ordered by SHA256 of `fresh-root-selection-20261011:<id>`; the first learner battle prompt with 2–6 supported actions qualifies. Selection does not use winners, Q errors or model disagreement. All eight strata provided eight roots.

`critic-battle-label-average-confirmation-2026-10-11` is running:

- **64 roots, 219 actions, 16 fresh shared suffixes (801–816), 64 original controls: 3,568 games.**
- Primary: pooled32-step32 minus old16-step32 centered action MSE, equal root weighting; 50,000 shared suffix-block bootstrap draws, seed 2026101106. Squared targets cancel in the paired loss difference.
- Secondary: conditional Q-argmax wins and new16/unchanged comparisons. No pooling of prior outcomes or checkpoint selection.
- Model predictions and source prefix/actor/selected-action parity checks are sealed before branch outcomes. Pooled32 changes 12 of 64 choices versus old16; this does not show those changes are beneficial.
- Two workers, four-hour limit, original controls before interventions, per-game artifacts, cold replay and an independent watchdog. Health failures stop the study. Existing STOP evidence is preserved.

If pooled32 improves ranking on fresh deals, the next gate is an actual public-information actor-training comparison with retention and full-game evaluation. If not, retain the current model and reassess coverage and target quality. Private-Q error improvement alone cannot justify actor promotion.

## Provenance and handoff

| Study | Identity SHA256 |
|---|---|
| Target replication | `332bb74d180419b692480b2e74fa982fd35e119c0797b61c7e587c8f46325178` |
| Label-average fit | `05869854fb744e94c061ead0966f53f47b1e3e6201025273fd729303d798101d` |
| Fresh source roots | `e6c2c4e1c53fe747b66f33f06a2e029094d4326fc04e10d26608febbb9ae38ab` |
| Fresh confirmation | `4181784071c63f6dbe82aa27845f330e1518185379917bf601f994d8eab9d7b2` |

All directories are under `out/research/`. Completion events for the first three studies were acknowledged after verification. The active confirmation service is `ygorl-critic-battle-label-average-confirmation-20261011.service`, invocation `1c9ea329845a4ce8aa59213aa376dc5a`; its sibling `-watch.service` independently monitors progress and health. Read `monitor/latest.json`, `progress.json`, `HANDOFF.md` and, when complete, run `analyze.py` before acknowledging or drawing conclusions.

Execution uses frozen worktree `codex-runick-coverage-audit` at `ae09594ec0126b5365e3c8311e9ac5c67d23f270`. Reporting is based on main `0482028` (PR #325). Frozen runtime, native engine and card scripts are unchanged. Issue #319 remains unresolved. This change adds research documentation only; no production defaults change.
