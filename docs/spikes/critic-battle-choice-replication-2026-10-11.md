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

`critic-battle-continuation-audit-restart-2026-10-11` replays all **130 existing tasks** for roots45/60: both choices ×32 prior suffixes plus two originals. It records actor probabilities, native observation component hashes when available, public board/spells, phase and LP. Every result, engine response and decision transcript must match the prior replay exactly; it adds no independent statistical evidence and does not fit models.

The initial diagnostic run stopped in its canaries because instrumentation assumed the legacy `PolicyAgent` opponent exposed `.host`. The fix checks that interface and explicitly records native-observation unavailability for legacy agents while retaining their probabilities. It does not change either policy. A new directory preserves the original STOP and failure evidence. Both restarted original controls pass prior-response/transcript, original and cold replay checks before the remaining diagnostic tasks.

The next decision is whether the main failures trace to context-dependent action ordering, changed history inputs, or behavior of the fixed continuation policy. Do not add masks or train on these held-out roots. Use separate training data for any subsequent ranking/coverage intervention and reserve actual public-actor full-game evaluation for its own gate.

## Evidence and handoff

- Replication: `out/research/critic-battle-choice-replication-2026-10-11/{analysis.json,analyze.py,training-coverage-audit.json}`; identity `169db354e5acaa6b50ea1f79aee7a8cc55791440bc82ed0486a5522db09b09a0`. Completion event `dab4701e1f428a8396a71c7dab9df4805c6db63560742c07eed30580cfa28b9d` acknowledged after verification.
- Diagnostic restart identity `cf1bc32f149c8b226a3f85a1a212a7d2abbefc770967ed88e518c4781cf3d855`; service `ygorl-critic-battle-continuation-audit-restart-20261011.service`, invocation `7ed25fe251654ee18f60a7ba4bd1fca0`. Two workers, two-hour cap and independent `-watch.service`.
- Original diagnostic STOP event `8aece4a59bf6b70e7700cb73c2a7ccdc1169e39a8384e65818621f4cb34154f7` acknowledged with the recovery explanation. Neither its STOP nor previous failures are cleared.
- Read the restart's `HANDOFF.md`; run `analyze.py` after completion to verify all130 prior replays and summarize aligned distributions. Missing native observations must remain explicit.

Frozen execution remains at `codex-runick-coverage-audit` commit `ae09594ec0126b5365e3c8311e9ac5c67d23f270`. Reporting starts from main `f839d1d` (PR #327). No production code/default changes; issue #319 remains unresolved.
