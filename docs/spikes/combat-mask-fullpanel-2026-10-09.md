# Full-panel combat mask audit

The corrected mask-constrained probe adds one earlier-win candidate, for six
windows total. It still finds none in the tested long-game windows. These are
conditional heuristic continuations, not guaranteed wins against arbitrary
responses, general strength estimates, or automatic teaching labels.

Root: `out/research/combat-mask-fullpanel-2026-10-09/`.
Identity: `d1127941cda49e66aa617b92eed9acd1ef4776a2e92368f8a707f4fa822e9f06`.
Independent `review.py` checks all768 source hashes/results, exact engine
responses and complete replay steps after snapshot restoration. All651 windows
are probed (450 battle exits plus201 skipped entries),796 continuations per arm.
Completion is healthy and acknowledged; original artifacts remain intact.

| Chooser with encoded policy mask | Winning continuations /796 | Distinct winning windows |
| --- | ---: | ---: |
| Legacy target context | 10 | 6 |
| Correct attacker context, printed stats | 8 | 5 |
| Correct attacker context, public current stats | 8 | 5 |

All six are battle-exit windows. Across the32 games lasting >=16 turns, the
combined92 windows yield zero conditional same-turn wins in this bounded search.
This is not proof no lethal exists or that long-game behavior is good. The
original five shorter-game examples remain; the sixth is seed1 cold u256 versus
historical RL, game107 decision373: probe wins on turn7, original wins on turn13.

## A counterexample to blanket bad-attack labels

In the new case, the candidate has a face-up2400 ATK monster versus2500 ATK,
4800 LP versus1050. The successful legacy continuation attacks, chains the
candidate's hand effect56651978, and develops additional monsters. The original
attacker is destroyed and loses100 LP; a2500 attacker then trades with the
opposing2500 monster, allowing a2000 direct attack to win.

A preflight using the **original active opponent** reproduces this win. All34
candidate mask vectors in that branch agree between the native policy host and
Python encoder, and every original pre-intervention actor choice matches.
The original policy gives72.58% to leaving battle and0.0185% to the forced attack.
The two properly primed heuristics miss this line: their target/selection choices
differ. Correct combat context is necessary for faithful diagnostics, but Greedy's
local ATK/DEF comparison is not a complete effect planner or a safe masker.

This example supports the user's earlier concern: an initially losing attack
can be part of a useful sequence. Do not train a universal self-damage penalty.
The continuation remains conditional on realized hidden state and this opponent.

## Closing the native/Python support gap

`out/research/combat-mask-parity-2026-10-09/`, identity
`e52f83570580c1c852f45b54a7057cd372e61a469589ccc5b728d2529cc90a2d`,
rechecks all six positive windows, all initial actions and three modes under
both passive and original opponents:78 trials. Before each candidate action,
compare the entire native host and Python encoder mask vectors. Require passive
outcomes/winning lines to match the full-panel source and original seeded actor
prefixes to reproduce. The original opponent retains its state and RNG.
All78 trials completed, with2,100 full candidate-mask vectors equal and zero
mask violations. Passive and original-opponent wins per13 trials are10/8/8
(legacy/context/public-current). All six have at least one active-opponent win;
the sixth remains legacy-only. `reviewed-completion.json` verifies coverage and
per-job records. No population or untested-state parity claim follows.

## Objective boundary

The running continuation config has `ppo.gamma=1.0` and `turn_discount=1.0`.
With terminal win/loss rewards, an eventual win has the same undiscounted terminal
return whether reached now or later. All six selected original games eventually
win, so their faster alternatives alone do not establish a terminal-return
improvement. This is a concrete objective limitation for learning to finish
quickly, not proof that it fully causes the observed action probabilities.
Opponent risk, critic errors and policy optimization still matter.

Keep this formal experiment unchanged while it completes. The read-only
`critic-readout.py` reconstructed the six original actor prefixes and evaluated
the complete checkpoint model on the native observation/privileged input.
It verifies actor probabilities against the original policy and binds checkpoint
and script hashes. For warm/historical124, the immediate2400-versus2200 attack
has Q0.1221, versus Q0.1419 for main2; the actor gives them1.96% and92.89%.
For warm/Greedy48, attack22623509 has Q0.4510, above main2's0.4197, yet probabilities
are3.14% versus89.87%. Thus the cases do not support one universal explanation
that the actor simply chooses the highest Q. These readouts are evidence to
investigate, not a causal attribution. A finite successful
heuristic continuation must not be substituted for the learned Q target: Q
estimates continuation under the current policy, not under this heuristic.


## Next: hand the continuation back to the original policy

Registered root: `out/research/combat-policy-handoff-2026-10-09/`, identity
`ba915b2267c92068de9d6453e2f9cb9617f1459edb02cb8093e9076ea9e85a0f`.
Compare the original root action with one demonstrated first action, then give
all subsequent decisions back to the original policy and opponent. Six fixed
states ×16 suffix trials ×two arms =192 runs. Trial0 preserves original RNG
and must reproduce the entire original result in the control arm; other trials
use paired suffix RNG across arms. Keep4096-decision cutoffs separate from wins.
The forced action is chosen from successful original-opponent traces, preferring
the public-current chooser and then the shortest line, with legacy fallback.

This separates a missed first action from inability to complete the line. It
also prevents treating Q under the original policy as if it were Q under a
heuristic teacher. The fixed hidden states and selected action choice mean this
is a conditional diagnostic, not a general strength evaluation. No reward or
mask defaults change; formal PPO continues independently.
