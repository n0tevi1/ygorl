# Combat policy handoff: first action versus continuation

The six selected earlier-win examples do not share a single failure mode. Two
need only a different first action to win immediately in every tested suffix;
two remain difficult even after that first action. The next diagnostic varies
how much of the demonstrated sequence is provided before returning control.

Root: `out/research/combat-policy-handoff-2026-10-09/`.
Identity: `ba915b2267c92068de9d6453e2f9cb9617f1459edb02cb8093e9076ea9e85a0f`.
All192 runs are complete and healthy, with no cutoffs. Independent `review.py`
checks per-job records, source hashes, full panel coverage and paired suffix
IDs. All original actor prefixes match, and the six trial0 control results
exactly reproduce their frozen original games.

The first action is fixed to either the original chosen action or a demonstrated
alternative; all subsequent candidate and opponent decisions use their original
policies. Trial0 preserves both actor RNG streams. Trials1–15 reset only the
suffix RNG, paired across arms, on the same realized game state. These are six
selected states, not192 independent deals or a general strength benchmark.
The original root action is fixed even in the resampled suffix controls.

| Candidate / opponent / game | Same-turn wins: original → forced /16 | Eventual wins: original → forced /16 | Mean turns: original → forced |
| --- | ---: | ---: | ---: |
| cold512 / historical /88 | 4→11 | 14→16 | 7.3125→5.6250 |
| cold512 / historical /100 | 0→16 | 10→16 | 12.3750→9.0000 |
| warm512 / Greedy /0 | 0→1 | 16→16 | 11.0000→10.8750 |
| warm512 / Greedy /48 | 0→9 | 16→16 | 13.1250→11.8750 |
| warm512 / historical /124 | 0→16 | 6→16 | 12.3125→11.0000 |
| seed1 cold256 / historical /107 | 0→0 | 14→16 | 12.0000→11.1875 |

All except the last use seed0. No paired win→loss transitions were observed;
loss→win counts are2,6,0,0,10,2 respectively. Do not pool these repeated selected
states to claim a population win-rate gain or independent-sample significance.

## Implications for the root cause

For historical124 and100, the original policy can complete the line after one
changed decision. Its first-action choice is therefore a concrete bottleneck in
these tested states. For warm0 and seed1cold107, changing that action alone is
insufficient for reliable same-turn completion. This supports sequence-level
review instead of labeling only the opening action.

The undiscounted terminal objective is indifferent between earlier and later
wins **when both paths win**, but that is not the entire explanation. Under
fresh suffix randomness, postponement loses10/16 times in historical124 and6/16
in historical100, whereas their forced-action arms win16/16. On these fixed
states, an earlier finish also avoids actual loss risk. Neither this conditional
result nor a teacher's return is a universal Q target or hidden-state guarantee.

## Next: controlled demonstration prefixes

Root: `out/research/combat-prefix-handoff-2026-10-09/`.
Identity: `d87ece65b1be677286272256588541ac6ef7566dcda4f9da290e897d3d255817`.
Study the three incomplete cases: warm/Greedy0 and48, seed1cold/historical107.
Seven prefix budgets (1,2,4,8,16,32,64 meaningful candidate choices) ×eight
paired suffix streams ×three states =168 runs. The original first action is
replaced; each following taught choice must obey the native policy mask. A
singleton legal choice advances the teacher but does not consume this budget.
The teacher is used only on the root turn. Afterwards the original policy acts.

Every decision still consumes the original actor's sample, even if overridden;
this keeps actor RNG progression explicit. One-choice controls must exactly
match the preceding study's result and same-turn candidate action/probability
trace. Record each additional override with both probabilities and public board.
Two CPU workers, three-hour cap,4096 further-decision cap, independent monitor.
The two first real preflights passed before launch.

In the warm0/trial0 preflight, one taught choice still wins only on turn11, but
two meaningful choices win on turn9. The second intervention is the attack-trigger
confirmation for card25072579 (Prototype Sky Striker Ace - Amatsu): the original
actor chooses `no` with99.27% probability, while the teacher chooses `yes`.
The installed script/text permits destroying one own Sky Striker Ace and one
opposing card when an attack involves Amatsu. This is a specific effect-use
failure in this selected state, not a reason to always answer yes or attack.
Wait for the complete prefix comparison before estimating how consistently this
intervention helps. No training labels, reward change or checkpoint promotion
has been made from these selected examples.

## Prefix study completed

All168 prefix runs completed, including an important harmful partial-prefix
counterexample. See [prefix results and fresh coverage](combat-prefix-handoff-2026-10-09.md)
for full counts, control verification and the new768-game cohort.
