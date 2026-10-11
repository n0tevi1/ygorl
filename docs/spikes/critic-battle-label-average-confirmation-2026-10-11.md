# Fresh-deal label averaging: no demonstrated ranking improvement

The 3,568-game fresh-root confirmation is complete and verified. Doubling branch-label samples from 16 to 32 does **not** establish improvement: the primary centered-action MSE interval includes zero, and the secondary conditional-win comparison is adverse. No model is promoted. The production actor remains unchanged.

## Registered results

The comparison fixes 64 roots from independently shuffled source deals, all 219 supported actions and 16 new shared continuation streams (801–816). Step32 checkpoints and predictions were sealed before outcomes. All 3,568 games ended with a winner; all 64 original-action/original-RNG controls passed. Verification checks immutable runtime and inputs, report identity, exact task coverage, per-game task/results, replay hashes, recorded cold replay equality and checkpoint hashes. Games were not independently rerun during this audit.

| Frozen critic | Centered action MSE | Conditional Q-argmax wins |
|---|---:|---:|
| Unchanged initialization | .032131 | 69.141% |
| old16 | .031181 | 69.727% |
| new16 | .030025 | 69.629% |
| pooled32 | .030001 | 68.457% |

The **primary pooled32-minus-old16 MSE difference is −.001181**, 95% interval **[−.002624, +.000189]**. The point reduction is 3.79%, but the interval includes zero. The secondary conditional-win difference is **−1.270 percentage points**, interval **[−2.344, −.293 pp]**.

Intervals use 50,000 shared suffix-block bootstrap draws, seed2026101106, with equal weighting of all registered roots. These are conditional intervals for fixed roots and continuation policies, not intervals over the population of game states. Secondary intervals are unadjusted exploratory comparisons. The adverse secondary result is a reason to retain the current model and investigate, not a claim of established population-level degradation. new16 and unchanged comparisons also do not establish improvement. Previous training or confirmation outcomes are not pooled into these results.

This is a private-critic Q-argmax diagnostic. It does not measure deployed public-information actor strength, shorter games or unseen-card understanding. Source deal novelty is relative to the audited registry of 2,624 prior deals, with the existing card/deck pool.

## What changed, and what remains unexplained

pooled32 changes 12/64 choices versus old16: two roots have net favorable outcomes, five adverse outcomes and five equal win counts. There are 12 favorable and 25 unfavorable paired suffix outcomes, a net loss of 13/1,024. All five adverse roots occur in Runick-containing strata. Three switch from an attack to an activation; this is not a basis for a blanket activation penalty. Other activations are neutral or favorable, and target-level estimates remain noisy.

| Root | Choice change | old16 wins | pooled32 wins | Diagnostic |
|---|---|---:|---:|---|
| 53 | Light Dragon @Ignister attack → Crossout Designator | 14/16 | 10/16 | All-action centered MSE improves by .001223 despite choosing the lower observed-return action. |
| 59 | TY-PHON attack → Runick Smiting Storm | 16/16 | 13/16 | Candidate activation ranks above the attack. |
| 60 | Geri attack → Runick Flashing Fire | 10/16 | 3/16 | Predicted return margin flips from −.045742 to +.003205; observed margin is −.875. |
| 41 | Main Phase 2 → End Phase | 7/16 | 12/16 | An example of a favorable change, retained in the follow-up. |

Names come from the frozen experiment's local card database; actions and public boards come from source transcripts. These summaries do not establish a particular card-effect mechanism, missed lethal, or which later choice caused the loss. In particular, a Runick activation's battle restriction must be traced in the engine rather than assumed to cancel the current battle opportunity.

Root53 is a concrete example of objective mismatch: averaged error over every action can improve while the best-action ordering worsens. The models' identical retained non-Q parameters and exact old16 training replication make an accidental actor update an unlikely explanation for this comparison. But finite labels, state coverage, action representation and fixed continuation-policy behavior remain possible contributors. More target samples alone did not resolve the observed problem.

## Follow-up: fixed independent choice replication

**Completed:** the 780-game independent comparison repeats the aggregate adverse difference: −1.367 pp, 95% interval [−2.295, −.488 pp]. Root53's individual adverse example does not repeat. See [replication and continuation audit](critic-battle-choice-replication-2026-10-11.md) for corrections, concentration of losses and the next mechanistic check. The following records the original registration.

`critic-battle-choice-replication-2026-10-11` registered **780 games**: both frozen chosen actions at all 12 disagreement roots ×32 new suffix streams (901–932), plus 12 original controls. Selection uses every disagreement in sealed predictions, not only losing roots. No models are fitted or selected. The original other 52 roots have identical chosen actions and contribute structural zero differences, so replaying them is unnecessary for this comparison.

Primary: pooled32-minus-old16 conditional-win difference over the **original 64 roots**, using only fresh outcomes, 50,000 shared suffix-block bootstrap draws and seed2026101107. The 12-root results must not be reported as the full-panel mean without the 12/64 factor. Complete the fixed budget once; do not stop early or keep resampling for significance. Per-root changes are secondary diagnostics, not tactical ground-truth labels.

If the adverse effect repeats, prioritize ranking/decision-sensitive supervision and training-state coverage, using new training data rather than these held-out roots. If it does not, report the inconsistency and reassess suffix sensitivity. In either case, retain the current production model; any later actor integration needs its own full-game and retention gates. No threshold or action mask is being tuned on this confirmation.

## Evidence and execution

- Completed: `out/research/critic-battle-label-average-confirmation-2026-10-11/{analysis.json,analyze.py,ranking-audit.json,audit_ranking.py}`; identity `4181784071c63f6dbe82aa27845f330e1518185379917bf601f994d8eab9d7b2`.
- Completion event `1182e4c55dc451632a6ec2e135f619cfff801d1dbb70e6fdbb87e39a9e87a93c` acknowledged after verification.
- Follow-up identity `169db354e5acaa6b50ea1f79aee7a8cc55791440bc82ed0486a5522db09b09a0`; unit `ygorl-critic-battle-choice-replication-20261011.service`, invocation `4cbc57d2e8514c2abedbe8c2884ce60c`.
- Two workers, two-hour cap, original controls before interventions, cold replay, per-game artifacts and independent `-watch.service`. Read its `HANDOFF.md` and run sealed `analyze.py` after verified completion.
- Frozen runtime `codex-runick-coverage-audit` remains at `ae09594ec0126b5365e3c8311e9ac5c67d23f270`. Reporting starts from main `25dd182` (PR #326). Existing STOPs and unresolved issue #319 are preserved.

This is a documentation update backed by experiment verification and a new registered diagnostic. No production code or defaults change.
