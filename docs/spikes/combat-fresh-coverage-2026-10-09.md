# Fresh combat coverage and active-opponent validation

The missed-finish signal recurs on new deals: six conditional same-turn winning
windows in five games, all of which the original candidate eventually won.
The search still does not explain the long-game tail. These are diagnostic
continuations, not evidence that the trained policy has improved.

## Coverage and integrity

Root: `out/research/combat-fresh-coverage-2026-10-09/`.
Identity: `d6aaa4de26e4476e00b8655af2c5fea1d4ce70d2299ea21a2920b6f45ffed259`.
Independent `review.py` verified all 768 baseline/result bindings, replay hashes,
and complete response/step equality after snapshot branches. Each baseline also
passed cold replay and an independently sampled unwrapped-policy rerun. All
games completed without engine health failures. Completion was acknowledged.

The preregistered panel uses 32 shared new deal blocks, four seat/deck
assignments, three checkpoints and two opponents. Engine seeds are disjoint
from the earlier panel; the 20-deck pool is unchanged. Deals shared across
checkpoints are correlated observations, not independent training replications.

| Candidate | Opponent | Wins / 128 | Mean turns | Windows | Positive windows |
| --- | --- | ---: | ---: | ---: | ---: |
| seed0 warm512 | Greedy | 112 | 9.0234 | 98 | 0 |
| seed0 warm512 | historical RL | 90 | 8.6016 | 109 | 1 |
| seed0 cold512 | Greedy | 103 | 9.5703 | 130 | 0 |
| seed0 cold512 | historical RL | 91 | 9.0547 | 125 | 0 |
| seed1 cold256 | Greedy | 94 | 10.8281 | 103 | 3 |
| seed1 cold256 | historical RL | 73 | 9.5781 | 129 | 2 |

Overall mean is 9.4427 turns. Different deals and a mixture of checkpoints mean
this is not a longitudinal training-progress estimate. Use the fixed paired
checkpoint evaluations for that question.

All 694 battle-exit/skipped-entry windows were probed: 415 main2 exits and 279
end-phase exits. There were 872 continuations per chooser mode, 2,616 total.
Legacy/context/public-current modes found 10/10/11 successful branches; their
union is six windows. No winning branch had to be excluded for changing turn.
Branches enforce the encoded policy mask and use a passive opponent, at most
four initial choices and 128 subsequent decisions. A negative result is not
proof that no winning line exists; a positive is not a hidden-information-safe
instruction to the actor.

## Positive cases

| Candidate / opponent / game | Decision | Opportunity turn | Original win turn |
| --- | ---: | ---: | ---: |
| seed0 warm512 / historical / 72 | 466 | 5 | 7 |
| seed1 cold256 / Greedy / 65 | 402 | 8 | 12 |
| seed1 cold256 / Greedy / 65 | 499 | 10 | 12 |
| seed1 cold256 / Greedy / 68 | 534 | 13 | 15 |
| seed1 cold256 / historical / 52 | 490 | 9 | 10 |
| seed1 cold256 / historical / 76 | 325 | 7 | 8 |

Game 65 contributes two windows, not two independent games. `result.winner`
already identifies deck/agent A or B; it must not be converted a second time
using `first`. All five candidates won their baseline game (`winner == 0`).

The earlier panel found six windows in six games. The fresh panel corroborates
the existence of missed earlier finishes beyond those selected examples, but
does not establish their population prevalence or generalization to new decks.

## Long games remain unresolved

There are 58 games lasting at least 16 turns, containing 160 examined windows
and 194 continuations per mode. None yielded a same-turn win. The earlier panel
had 32 such games, 92 windows and no positive. This finite search cannot
attribute the long tail to an absence of lethal opportunities. Development,
battle entry, target selection and earlier strategic choices remain candidates;
do not shorten the objective or label every battle exit as wrong from this data.

## Original-opponent validation passed

Root: `out/research/combat-fresh-validation-2026-10-09/`.
Identity: `e0553001b186d774c40c519d55b9376d162e82dbbe7d0876ee7bf52d3afbd83a`.
All six positive windows, all 14 probed initial choices, three continuation
modes and passive/original opponent give 84 registered branches. Exact actor
prefix/RNG reproduction, native-versus-Python full-mask equality and native
mask support are required. Passive winning outcomes/lines must match source
records; same-turn wins require the terminal turn to equal the root turn.

All 84 branches passed independent review. All six windows remain positive
under the original opponent, with legacy/context/public-current branch wins
10/10/11 out of 14, matching the passive totals. All 2,302 complete native mask
vectors equal the Python encoder vectors; no unsupported action or engine
health failure occurred. Exact actor prefixes and passive winning lines match.
The source is `resolved-panel.json`, which binds the new baseline records, not
the initial panel that lacked their completed results. These realized-state
checks do not establish robustness over unknown opponent cards or responses.

## Fresh first-action handoff

Root: `out/research/combat-fresh-handoff-2026-10-09/`.
Identity: `2b124d1a589955340d5bb3547018b186c722ee4a0732b96d436d49edea7b82e0`.
Six windows × 16 paired suffix streams × original/force-first arms = 192 trials.
Select the first action from a validated original-opponent winning branch,
prioritizing public-current mode then shortest trace. Both original actors
play after that first action. Trial0 preserves their RNG and must reproduce
original full-game results in the control arm; the other streams share suffix
seeds across arms. The review groups by game **and decision**, preserving the
two windows from game65 without merging them into one state.

Four real preflight branches passed prefix, mask and health checks, covering
both candidate seats. They already include an adverse intervention: warm72,
trial0, wins on turn7 unassisted but loses on turn8 after forcing the first
attack and handing control back. The complete heuristic continuation wins on
turn5. In contrast, cold65's turn8 window wins immediately after the first
action alone. Preserve `preflight/games/1.json`; a validated complete winning
line is not evidence that its first action helps the current continuation
policy. The full 192-branch run and independent watchdog have started.

Measure same-turn wins, eventual wins and paired win-to-loss reversals, retaining
the previously observed harmful partial-prefix example. This distinguishes a
poor first choice from inability to finish the subsequent sequence before
correction training. No pooling of repeated states as independent games. Formal
PPO, rewards and policy defaults are unchanged.
