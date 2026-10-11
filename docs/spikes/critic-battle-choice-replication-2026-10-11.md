# Independent choice replication confirms a conditional regression

All 780 registered games passed verification. The pooled32 critic's adverse choice result repeats on independent suffixes: **−1.367 percentage points**, paired 95% interval **[−2.295, −.488 pp]**, versus old16. This is the preregistered difference over the original 64 fixed roots; 52 identical-choice roots contribute zero. No production model is promoted.

The 12 disagreement roots use both frozen choices with 32 new streams (901–932), plus 12 original controls. All games ended with a winner. The audit checks identity, exact task grids, result agreement, replay hashes, recorded cold equality, all controls and model hashes. Games were not independently rerun during that audit. Bootstrap: 50,000 shared suffix blocks, seed2026101107. Earlier 16-suffix outcomes are not pooled. The interval conditions on these roots and continuation policies; it does not establish a population-wide actor-strength loss.

## Concentration and correction of the previous examples

Across the 12 changed roots there are 21 favorable and 49 unfavorable paired outcomes, a net loss of 28/2,048. Two roots improve, six worsen and four tie. Roots45 and60 contribute 23 of the 28 net losses. Their selection for the next mechanistic audit is explicitly post hoc, not a new unbiased performance panel.

| Root | old16 wins / 32 | pooled32 wins / 32 | Interpretation |
|---|---:|---:|---|
| 45: Main2 → End Phase | 29 | 19 | Main source of repeated phase-choice loss. |
| 60: Geri attack → Runick Flashing Fire | 20 | 7 | Main source of repeated activation-choice loss. |
| 59: TY-PHON attack → Runick Smiting Storm | 31 | 28 | Smaller repeated adverse difference. |
| 53: attack → Crossout Designator | 27 | 28 | Earlier 14/16 → 10/16 loss does **not** repeat. |
| 41: Main2 → End Phase | 20 | 20 | Earlier favorable difference does not repeat. |

The previous root53 example still demonstrates that sample-average error and sample action ranking can disagree, but it must not be treated as a verified tactical mistake. The independently repeated aggregate regression is stronger evidence than that individual anecdote. Likewise, ending the phase or activating a Runick card is not intrinsically an error.

## Mechanistic observations, not yet root-cause proof

In root45's 32 Main2 branches, the learner immediately ends the turn in 31 cases; in one it sets a spell first. For both root choices, the first next-turn optional effect is accepted 13/32 times and declined 19/32. In paired trial901, however, one branch accepts and the other declines it. An extra decision consumes a different random draw, so a shared seed does not ensure the same later sampled choice. Equal marginal counts do not establish identical distributions, state histories or downstream behavior. The current inference remains valid for the sampled continuation protocol; the tactical interpretation requires more evidence.

In root60 trial903, the activation branch resolves Flashing Fire and **still attacks in the current battle phase**. It then encounters a different defensive sequence from the opponent. Thus “the activation immediately prevents the current attack” is contradicted by the recorded trace. Later battle restrictions, resource changes and subsequent policy choices remain possible contributors; none is assigned sole responsibility yet.

The original 32-root training set contains 16 attack/phase roots, 12 activation+attack/phase roots and four phase-only roots, for 40 attack, 13 activation, 32 Main2 and 32 End Phase action candidates. Flashing Fire, Smiting Storm and Crossout are present. This is not simply absence of those card IDs or action categories. Coverage of their particular contexts, target noise, learned ranking and continuation behavior still need investigation. Coverage counts are descriptive, not causal evidence.

## Exact replay with additional diagnostics

**Completed and verified:** all130 diagnostic reruns exactly match prior results, engine responses and decision transcripts. At root45's aligned next-turn effect, 31/32 pairs have identical hashes for every native observation component and identical actor probabilities; 14 of those pairs sample different choices. Trial930, which sets a spell in Main2, has changed cards/globals/events and a .048155 maximum probability difference. This is evidence of RNG-position divergence on otherwise equal actor inputs, not proof of complete engine-state equivalence or a faulty simulator. The previous fixed-root win estimates remain recorded; their tactical interpretation must be qualified.

At root60, a battle-phase prompt occurs on turn13 in 0/32 activation branches and 8/32 attack branches. This is consistent with the Runick restriction and differing subsequent states, but does not isolate the restriction's contribution to final wins. It is not a count of all legal attack opportunities.

`critic-battle-continuation-audit-restart-2026-10-11` replays all **130 existing tasks** for roots45/60: both choices ×32 prior suffixes plus two originals. It records actor probabilities, native observation component hashes when available, public board/spells, phase and LP. Every result, engine response and decision transcript must match the prior replay exactly; it adds no independent statistical evidence and does not fit models.

The initial diagnostic run stopped in its canaries because instrumentation assumed the legacy `PolicyAgent` opponent exposed `.host`. The fix checks that interface and explicitly records native-observation unavailability for legacy agents while retaining their probabilities. It does not change either policy. A new directory preserves the original STOP and failure evidence. Both restarted original controls pass prior-response/transcript, original and cold replay checks before the remaining diagnostic tasks.

The next decision is whether the main failures trace to context-dependent action ordering, changed history inputs, or behavior of the fixed continuation policy. Do not add masks or train on these held-out roots. Use separate training data for any subsequent ranking/coverage intervention and reserve actual public-actor full-game evaluation for its own gate.

## Evidence and handoff

- Replication: `out/research/critic-battle-choice-replication-2026-10-11/{analysis.json,analyze.py,training-coverage-audit.json}`; identity `169db354e5acaa6b50ea1f79aee7a8cc55791440bc82ed0486a5522db09b09a0`. Completion event `dab4701e1f428a8396a71c7dab9df4805c6db63560742c07eed30580cfa28b9d` acknowledged after verification.
- Diagnostic restart identity `cf1bc32f149c8b226a3f85a1a212a7d2abbefc770967ed88e518c4781cf3d855`; service `ygorl-critic-battle-continuation-audit-restart-20261011.service`, invocation `7ed25fe251654ee18f60a7ba4bd1fca0`. Two workers, two-hour cap and independent `-watch.service`.
- Original diagnostic STOP event `8aece4a59bf6b70e7700cb73c2a7ccdc1169e39a8384e65818621f4cb34154f7` acknowledged with the recovery explanation. Neither its STOP nor previous failures are cleared.
- Read the restart's `HANDOFF.md`; run `analyze.py` after completion to verify all130 prior replays and summarize aligned distributions. Missing native observations must remain explicit.

Frozen execution remains at `codex-runick-coverage-audit` commit `ae09594ec0126b5365e3c8311e9ac5c67d23f270`. Reporting starts from main `f839d1d` (PR #327). No production code/default changes; issue #319 remains unresolved.

## Completed: phase-boundary RNG coupling diagnostic

All **257 games** passed immutable-input, full task-grid, per-game result/replay hash, cold replay and original-control verification. All ended with a winner. There is no missing boundary in either mode. The earlier phase-ending disadvantage does not persist in this fresh sequential sample; resetting the suffix streams at the aligned boundary removes every observed continuation difference.

| Coupling, 64 trials each | Main2 wins | End Phase wins | End-minus-Main2 | Different winners | Identical semantic suffixes |
|---|---:|---:|---:|---:|---:|
| Sequential draws | 52 | 54 | +3.125 pp | 20/64 | 0/64 |
| Boundary reseed | 54 | 54 | 0 pp | 0/64 | 64/64 |

Both modes have **64/64 identical actor-observation pairs at the boundary**. Semantic suffix comparison removes only the decision index, which differs because Main2 inserts an extra prompt; players, turns, options and chosen actions must still match. The boundary arm then reproduces the same entire continuation, not merely the same final winner. These are new trials1001–1064, separate from previous samples.

**Interpretation correction:** the earlier negative phase-choice estimates must not be taught as a stable tactical preference for Main2. On these sampled paths, an extra decision shifts RNG consumption, and finite sequential suffix panels can produce substantially different estimated returns for continuations that coincide under aligned randomness. This is Monte Carlo target/measurement noise, not an engine correctness failure. Boundary reseeding changes coupling rather than the policies' intended sampling distributions. It does not establish equivalence for every possible Main2 continuation: the earlier trial930 actually set a spell and changed state. Nor does this single selected-root diagnostic invalidate the recorded 64-root estimates or establish general model improvement.

The implication for future supervision is to measure target uncertainty and distinguish materially changed continuations from RNG-offset differences before treating small action-value gaps as reliable ranking targets. Do not force all Main2/End actions equal or apply a blanket phase mask.

`critic-phase-rng-boundary-2026-10-11` registered **257 games** on root45: 64 new trials1001–1064 × both phase choices × two continuation RNG couplings, plus one original control. Sequential coupling is the current evaluator. Boundary coupling reseeds both agents immediately before the learner's first prompt on a later turn, using independent SHA256-derived trial/seat seeds. Both arms record that boundary's inputs, probabilities and state. Policies and action distributions are unchanged; the intended random-seed ensemble marginals are preserved, while finite sampled paths can change.

All trials remain in the denominator, including differing boundary states or failure to reach a boundary. Compare outcome divergence and semantic suffix equality, especially when actor observations coincide. This is a post-hoc single-root mechanism experiment, not another model-promotion test. Do not interpret it as a general tactical label or train on the held-out root. No optional stopping or additional significance retries.

Identity: `b91c33d5be0158f187b8795f0affcb5e86155b3cf15db18fba7711c52f8cf3e5`; service `ygorl-critic-phase-rng-boundary-20261011.service`, invocation `c2896d9e49954c7fafb2eb1b61e3866d`. Two workers, two-hour cap, cold replay, original control and independent watchdog. The diagnostic completion event `3a13da7d7d2876034c87d10405f6b6751132380ab5c0004355b4c9ffb823f1bd` was acknowledged after full verification. Supplemental results are in the completed diagnostic's `analysis.json` and `mechanism-summary.json`.

The phase-boundary completion event `ae383d118f2b75bcd75258c5c967fa233c1988365171c6c2e88f5a8797b1ca65` was acknowledged after verification. Its own `analysis.json` records all257 games and paired suffix comparisons. The audit did not independently rerun games beyond their recorded cold replay checks.

## Next: isolate Runick continuation differences

`critic-runick-rng-boundary-2026-10-11` runs **257 games** on root60: 64 new trials1101–1164 × activation/attack × sequential/boundary coupling, plus one original control. The tested choices are Flashing Fire activation (action0) and Geri attack (action1). Report **activate-minus-attack** wins separately for each coupling, input agreement and suffix divergence. The worker is byte-identical to the completed phase study; it reseeds both agents before the first learner prompt on a later turn. Models remain frozen.

The purpose is to check whether adverse Runick outcomes remain after controlling RNG alignment. Reseeding does not equalize different states, and a remaining gap would not by itself isolate battle restrictions from resource changes or subsequent policy behavior. Do not pool previous trials, tune on the outcomes, stop early or infer population-level strength from this post-hoc root. No fitting, action masking or promotion.

Identity `537c6f0efc30a55931ea71ee87fc205a1a5f70d93b3513e082721e94b4e6daa0`; service `ygorl-critic-runick-rng-boundary-20261011.service`, invocation `52df931f7b0e4a458e8f739536fc1787`. Two workers, two-hour cap, original canary, per-game cold replay and independent watchdog. Sealed `analyze.py`, with the supplementary `HANDOFF.md`, defines completion handling. Reporting base is main `f72a465` (PR #329); frozen runtime and previous STOP evidence are unchanged.

## Follow-up completed: Runick RNG control and optimizer interference

The Runick boundary experiment is complete: activation wins18/64 versus attack32/64 with sequential coupling and30/64 with boundary coupling. Unlike root45, boundary actor inputs differ in every pair. The adverse conditional choice survives RNG alignment, but the battle restriction is not isolated as the sole cause.

[The interference report](critic-ranking-interference-2026-10-11.md) records exact optimizer replay, controlled loss removal, parameter-path interventions and a new prospective collection. Other training-root updates counteract the existing Runick example and retention anchor; context-only fitting repairs this local ranking but remains an unpromoted hypothesis awaiting independent roots. It also documents a corrected secondary legal-argmax reporting bug in the loss-intervention script.
