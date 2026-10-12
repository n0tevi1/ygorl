# Cached lethal-line guards

The next implementation audit found a correctness gap in the opt-in `LethalAgent`: its cached-line and shadow checks compared only the player, prompt type and action kinds. Two `select` rows could refer to different cards or different copies of the same card without invalidating a saved response index. The same weak signature also ignored LP and phase changes.

The signature now binds the decision index, player, turn, turn player, phase, LP, immutable decision payload, complete ordered actions and undo exclusions. A mismatch immediately abandons the cached line. Search also refuses to start from a mismatching shadow prompt, and the existing desynchronization path closes the shadow before forwarding a response. These checks compare available prompt data; they do not establish equivalence of every hidden engine state.

The rollout terminal check now explicitly requires the terminal turn to equal the search turn. A synthetic atomic engine advance reproduced the old acceptance of a next-turn win without an intervening prompt. A separate real-engine deckout test confirms the ordinary prompt-boundary path already stopped correctly. **No naturally occurring cross-turn false positive was found:** the archived full-panel and fresh-coverage reports contain zero flagged cases among 2,388 and 2,616 diagnostic probes respectively. This guard is not an explanation for the measured long games.

The module's strict-mode documentation is corrected: rejecting own draws, random events and opponent chains does not prove a line works using only public information or against active responses. The shadow contains actual hidden state and the search opponent is passive. No default actor, reward, training checkpoint or teacher-label export is changed.

## Regression evidence

Artifacts: `out/research/lethal-line-guards-2026-10-12/`.

- On main `e23000d`, eight prompt-change tests execute the wrong cached choice; the synthetic terminal-turn test also fails. The real-engine deckout and existing tests pass. The retained pre-fix log records nine failures and five passes.
- After the fix, all 29 lethal/combat tests pass, including refusal to forward a response into a mismatching shadow. Existing engine examples still find genuine current-turn wins.
- A bounded compatibility run fixes four deck pairs, two wrapper seats and four seeds before execution. Both old and new wrappers use the same random inner agent, opponent, search budgets and 2,000-decision cap: 16 games, eight paired comparisons. All complete without engine errors, cutoffs or shadow desynchronization. Re-read full transcripts match exactly for every pair.
- Each arm performs 417 searches and 4,925 bounded internal continuations, finds ten lines, supplies 176 actions and abandons six lines. These totals count search work, not independent games. No additional training data is collected; this is compatibility evidence, not a strength estimate.

The earlier [prefix intervention study](combat-fresh-prefix-2026-10-09.md) already tested one-step and multi-step assistance. It should not be repeated as a new experiment. It found heterogeneous failures and harmful partial prefixes, including a successful teacher trace containing self-Ash. The [public damage priority](lethal-priority-2026-10-11.md) still requires broader trigger coverage and active-opponent validation before any actor-facing use. These guard fixes remove implementation hazards but do not establish that a heuristic or critic improves play. Issues #62, #83 and #105 remain open.
