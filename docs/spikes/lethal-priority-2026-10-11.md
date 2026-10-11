# Public damage priority for combat search

The combat diagnostic now has an opt-in `lethal_first=True` mode. It prioritizes a legal, face-up attack-position target whose current ATK difference would exhaust the opponent's LP. The previous heuristic always selected the strongest beatable target, even when attacking a weaker monster would finish the game. This change only proposes search continuations; card effects and opponent responses can prevent nominal lethal damage.

Development was synchronized with GitHub main `7b323f1`, independently verified by fetch and the GitHub commits API. The new worktree is `codex-lethal-heuristic`. The ongoing critic coverage experiment remains pinned to its registered runtime; its code was not changed mid-run. No newer main commit was present at verification.

The heuristic requires a public-current-stat provider, matches physical card instances, ignores facedown targets, does not invent piercing damage, uses the opponent's LP from either seat, and operates after policy-mask filtering. Missing current attacker information retains the baseline behavior. Default Greedy, actor inference, training reward and existing combat-probe behavior are unchanged.

## Verified behavior and limits

An engine regression uses only normal monsters: a 2000 ATK attacker against 1800 and 1200 ATK targets, with the defender at 500 LP. On turn four, the baseline attacks the stronger target, deals 200 damage and reaches the turn limit. The new priority attacks the weaker target and wins that turn. Both runs have no engine errors. Unit tests additionally cover defense/hidden targets, insufficient damage, masked-out targets, missing current information and mirrored seats.

`out/research/lethal-priority-replay-2026-10-11/` evaluates the same six previously exposed missed-finish windows as the old combat mask-parity study. All 13 supported initial-action branches are tested with passive and original active opponents, for both current-stat baseline and lethal priority: 52 tasks. These are selected diagnostic states, not independent games or a population win-rate test.

- All 26 baseline continuation traces, final events and results exactly reproduce the original study on main. Prefix actor choices, root board/LP and 1,396 native/Python action-mask comparisons pass. No engine health failures or mask violations occur.
- Both modes win this turn in 8/13 branches against the passive opponent and 8/13 against the original opponent. No paired trace changes: this new target-priority rule does not trigger in these examples and provides no additional measured wins there.
- The executed source is archived with its exact original hash; `format-verification.json` verifies AST equality with the formatted implementation. `analyze.py` verifies stored results and baseline traces against the earlier report. The original reports remain intact.

The earlier heuristic search already found conditional winning lines where the actor exited battle. Repairing the target priority does not teach the actor to start those attacks or complete those sequences. A successful branch against the realized opponent and hidden state is also not a proof against all legal responses. In particular, `LethalAgent`'s historical passive-opponent/event-filter search must not be treated as a public-information guarantee or automatically promoted as the default actor.

## Next comparison

Use a bounded ensemble of the existing search modes plus the new priority; retain effect-dependent and initially losing-attack lines instead of masking them. Measure trigger coverage on a prespecified broader set of battle prompts. Compare unchanged policy, first-action-only guidance, and guidance through the verified combat sequence under matched active-opponent continuations. Preserve failures and report eventual wins as well as earlier wins, regression cases, mask health and decision overhead. Freeze the trigger/search budget before generating fresh evaluation deals. Keep privileged branch validation separate from an eventual public-information inference interface; a hidden-state search result is not an actor feature or an automatic training label.

Current validation: 50 combat/agent/action-outcome tests pass, including the engine regression. Issues #62, #83 and #105 remain open; this is a tested research heuristic, not a claim that missed wins or long games are solved.
