# Optional summon reentry defeats the within-selection cancel budget

The fresh inference panel found a candidate decision-limit loop despite budget-1
cancellation. Do not promote argmax + cancel-budget-1 as a complete loop solution.
The panel remains frozen and continues under its original protocol: limits remain
non-wins in the denominator and make completion unhealthy; engine/replay failures
would STOP. There is no new Lua error in this incident.

Evidence: `out/research/summon-reentry-audit-2026-10-10/`.
Source: `policy-inference-clown-restart-2026-10-10/games/2815.json`, identity
`71c5afbc608ab29c9fe35455f06a9a0aa86b8aa4437cc168774f57750030bc22`.
Original replay SHA256:
`51f59aed0b2b3eab263882286cd6275d131e0870af90b7fc0bd0c98f46b1fb46`.
Monitor event `70809c643bd0c77a32d480a40682f4e8aa43cf3864aedd9b776e7380bb35b51b`
was acknowledged after inspection; its failure evidence was retained.

## Reproduced cause

Seed-1 cold u512 against initial-128x2, game 127 / deal 31, reaches 6,000 decisions
on turn 3 with LP 3800/6700. It makes 823 candidate cancellations, but the measured
consecutive-selection cancel maximum is **zero**, and budget-1 intervenes zero
times. Both a fresh policy run and cold engine replay reproduce every original
action, response and result. There are no retries, Lua errors or decoding errors.

At decisions 233–239 the candidate starts Chaos Angel (22850702), selects Ghost
Belle (73642296) and Fabled Lurrie (97651498), then repeatedly deselects and cancels.
Decision 240 returns to the same idle menu and starts the same summon. The exact
seven-decision cycle repeats until the cap. Cold capture verifies identical root
observation tensors and only SelectUnselectCard/Hint messages within the attempt,
then SelectIdleCmd on return: no material movement, payment or chain progress.

The Python and native trackers reset their selection counters at the idle menu.
This is consistent implementation of a *within-selection* guard, not a parity bug.
Each optional summon is abandoned once; no individual selection exceeds budget 1.
Extending that counter blindly would risk removing an essential summon exit.

The learned decisions also conflict. The root puts 37.096% on Chaos Angel, its
highest-probability command. After choosing levels 3 and 1, the remaining level-6
Bystial Magnamhut (33854624) completes a legal level-10 summon. The policy assigns
only **2.7507%** to selecting it and **97.2493%** to undoing the tuner selection.
At the final material prompt it assigns **94.8871%** to cancel. Returning to the
unchanged menu restores exactly the original command distribution, so argmax
reenters deterministically. This directly demonstrates command/material-policy
inconsistency; it does not by itself distinguish missing selected-goal features
from insufficient terminal credit or inadequate training coverage.

## Isolated interventions

All runs use the same frozen repaired source/native runtime and checkpoints. Only
the candidate inference action restriction, or one explicitly forced diagnostic
action, changes. Every completed run cold-replays exactly without engine errors.

| Candidate policy on the selected failing deal | Outcome | Turns | Decisions | Cancels |
|---|---|---:|---:|---:|
| Original argmax + cancel-budget-1 | decision limit, non-win | 3 (censored) | 6000 | 823 |
| Add canceled-summon retry budget 1 | win | 9 | 557 | 2 |
| Add canceled-summon retry budget 4 | win | 9 | 599 | 8 |
| Original policy; force only material choice 236 | win | 7 | 397 | see replay |
| Original T=1 sampled paired game 2814 | win | 9 | 763 | unchanged baseline |

Budget 1 suppresses the abandoned Chaos Angel command at 240. The policy then
tries and abandons Baronne de Fleur, which is also suppressed at the unchanged
menu at 247; it chooses end phase and continues. Budget 4 allows four aborts per
command before doing the same. This stops wasted work without teaching selection.

Forcing the single material choice at 236 instead successfully summons Chaos
Angel (place/position follows immediately). It demonstrates an available successful
procedure, not an oracle proof that summoning it is globally optimal. No victory
or turn-count conclusion can be generalized from this selected failure. In
particular, the censored 3-turn baseline must not be treated as faster play.

Predetermined regression controls cover all 16 panel cells, game IDs 0 and 64:
**32/32 healthy**, **zero new guard interventions**, exact full action/response/
result equality to unmodified argmax + cancel-budget-1. Pending original panel
rows were independently run with the same frozen original policy; their provenance
is recorded in `controls-comparison.json`. These establish no changes on those
controls, not safety for every legal material revision or a fresh strength gain.

## Opt-in implementation and remaining work

`tools/summon_reentry.py` is an experimental candidate-only inference helper. It
recognizes only idle/spsummon -> material selection -> explicit cancel -> identical
idle, with no intervening event outside the narrow selection/Hint whitelist.
It allows the first attempt and cancellation, intersects the encoded mask, and
preserves the sole available exit. Other actions, events, players, turns, phases,
LP or root tensors invalidate history. Equal tensors alone are not used to infer
engine-state identity. A legal retry could still choose better materials, so this
is a heuristic restriction and **is not enabled in training, evaluation defaults,
engine legality, or the running registered panel**.

Integration must call `guard.observe(point)` for every player's decision, then
`guard.restrict(point, candidate_observation)` once at candidate decisions, and
`guard.on_decision(point, chosen_index)` for every action. Use a fresh instance per
duel. The recorded engine regression verifies native/reference parity, budgets
1/4, retained end-phase exit, and successful completion with the remaining material.
Safety tests cover real/uncertain progress, target cancellation, changed context,
changed observations, finished selections, and the sole-exit fallback. The guard,
existing cancel budget and selection-history suites pass **42 tests**.

Next: finish and review the unchanged 4,096-game inference panel, including this
limit, before a new paired promotion study. For learning, add successful material
completion and legitimate revision counterexamples to the selected-goal/credit
evaluation described in [the seed-2 review](seed-2-cold-u256-2026-10-10.md). A model
should learn to finish the legal summon, not merely learn to abandon it. Existing
opt-in history representation and a restricted inference policy are different
interventions and need separate matched training/retention/strength validation.
