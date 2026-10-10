# Combat probe context and continuation audit

The diagnostic driver lost the forced attacker's identity before asking Greedy
to choose its target. `tools/combat_probe.py` repairs this research-only context
handoff and adds an optional public-current-stat sensitivity arm. It changes no
training, evaluation-opponent, reward or mask defaults.

## Rechecked phase-entry audit

Root: `out/research/longgame-phase-entry-audit-2026-10-09/`, identity
`f2adfd64b072006dab679d018b35639e40814e40e75da50adc3f2d75e2670855`.
Independent `review.py` verifies all768 results, input hashes, complete original
engine responses and replay steps. There are201 skipped-battle-entry windows in
141 games. Among32 games lasting >=16 turns,13 games contain28 windows; the
registered first-two-window limit probes20 of them. None of those20 passive
continuations wins that turn. This is a finite negative result, not no-lethal proof.

`window-context-audit.json` supplies descriptive public-board counts:
90/201 entry windows have no face-up attack-position monster;115/201 have
nonpositive total face-up attack-position ATK, including those90. Available battle
entry is therefore a noisy error signal. Do not penalize it automatically.

Likewise, summing every monster's ATK overstates remaining attacks. In the earlier
battle-exit audit, seed1 cold u256/historical games54 and92 show field totals7200
and1700 against2000 and900 LP, but remaining legal attackers have only100 and0
ATK respectively. These sums are descriptive, not damage guarantees.

## Actual diagnostic defect and tests

After `session.act(forced_attack)`, a fresh Greedy has no `_attacker`. Its next
SelectCard enters generic selection instead of the attack-target heuristic.
`CombatProbe.prime` initializes that context before the forced action. A second
optional mode reads only face-up current ATK/DEF, matched by controller, zone,
sequence and card code; facedown targets retain the fixed unknown-stat heuristic.
Printed stats remain the unmodified Greedy baseline.

Seven regression cases pass: natural/forced target equivalence (and reproduction
of the old wrong generic selection), current-stat reversal, two hidden-stat
invariance cases, separate instances with the same card code, mask/index preservation and empty-mask rejection before chooser mutation. A real frozen
game also passes preflight and exact original-transcript restoration.

This helper does not solve card effects, hidden information, or exhaustive combat.
In particular, canceling an apparently unfavorable attack can itself miss a
winning effect interaction. Better context does not make Greedy an oracle.

## Full-window sensitivity completed

Root: `out/research/combat-probe-sensitivity-2026-10-09/`, identity
`3d52fc9c96618ca3860ac6009b3b7d421d9d853f93f0a4333e5cf0ff3c0874bb`.
All768 original games remain byte-equivalent in engine responses and equal in
replay steps/results after snapshot probes. Independent `review.py` binds the
raw report hash and checks coverage. Monitor completion is healthy and acknowledged.

Unlike the earlier >=16-turn/first-two filter, this study probes all651 windows
(450 battle exits plus201 skipped entries), up to four initial actions each.
Each arm has796 continuations, capped at128 further decisions against a passive
opponent on the realized engine state.

| Arm | Winning continuations /796 | Distinct winning windows |
| --- | ---: | ---: |
| Legacy fresh Greedy | 8 | 5 |
| Forced-attacker context, printed stats | 7 | 3 |
| Forced-attacker context, public current stats | 9 | 5 |

The union is five windows, all battle exits in five shorter games. No conditional
same-turn wins appear in the92 combined windows of >=16-turn games. Every
positive passes the existing event filter, which **does not establish hidden-state
robustness or a public-information winning guarantee**. All five original games
already won, two turns later. Discovery here cannot be attributed solely to the
context fix: the old chooser also finds five windows once the scope is expanded.
These selected cases do not estimate overall missed-win prevalence.

| Candidate / opponent | Game | Decision | Probe turn | Original win turn |
| --- | ---: | ---: | ---: | ---: |
| seed0 warm512 / Greedy | 0 | 493 | 9 | 11 |
| seed0 warm512 / Greedy | 48 | 522 | 11 | 13 |
| seed0 warm512 / historical RL | 124 | 591 | 11 | 13 |
| seed0 cold512 / historical RL | 88 | 275 | 5 | 7 |
| seed0 cold512 / historical RL | 100 | 386 | 9 | 11 |

## Completed follow-up: original opponent responses

Root: `out/research/combat-opponent-response-2026-10-09/`, identity
`ae0ab47a239d130b46c2069fec9a41cc455445d5242af5bddfa8b310ddc7befb`.
All initial actions and all three modes in those five windows are tested against
both passive and original active opponents:72 trials, two CPU workers, three-hour
cap, independent heartbeat/watchdog. Both original actors replay from original
seeds and must reproduce every pre-intervention choice; the opponent retains its
state/RNG. Save policy probabilities, branch choices/events, public boards and
mask-coverage violations. Passive outcomes and winning lines must reproduce the
prior study. Successful real-game preflight precedes launch.

Preflight also caught a result-finalization mistake in this new driver:
`tracker.result.turns` is populated only by `session.result()` after termination.
It was corrected before launch; the failed preflight explanation is retained.
Incomplete branches have a null terminal result. The source study is untouched.

Active-opponent success would still be conditional on this opponent and realized
hidden state. No automatic training-label generation or deployment.

All72 trials completed with healthy prefixes and engine continuations; the passive
winning lines reproduce the source. Original-opponent and passive win counts
match for every mode, but exact traces and LP can differ. An additional audit
found that Greedy sometimes cancels an attack through a row the policy masks:
the engine allows that action, while the training policy cannot choose it.
For public-current mode, only5 of9 winning lines remain wholly mask-supported,
covering3 windows (warm Greedy48, warm historical124, cold historical100).
Across **all modes**, including the legacy chooser, all5 windows have at least
one mask-supported active-opponent win. The narrower three-window count must
not be reported as the union. Neither total is a public-information guarantee.

A particularly direct example is warm/historical124: current2400 ATK attacks
public2200 ATK for the opponent's last200 LP. The policy assigns92.89% to leaving
battle and1.96% to that attack. Greedy48 has two direct1850 attacks versus2225 LP,
yet leaving battle has89.87% probability. These are concrete selected failures
to finish the duel, not evidence that every battle exit is bad.

`masked_action` now filters candidate actions before the chooser runs and maps
the result back to the engine index, covering truncated masks as well as undo
rows. This avoids advancing chooser state/RNG with a rejected guess. The
matched72-trial mask-constrained follow-up is registered at
`out/research/combat-mask-continuation-2026-10-09/`, identity
`56699d7f3b2001e539dbfc623d88764b10d65bb9dc272aa1ed1cefa252650b5b`.
It requires zero candidate mask violations and keeps every original comparison;
no previous result is overwritten.

## Repeated selection remains open

New seed1 cold512 alerts were acknowledged and checked against exact original
and cold replays. Greedy game147 has15 cancellations and wins in13 turns;
old256x2 game65 reaches32 cancellations and then advances, winning in12 turns.
Historical RL game200 has5 cancellations in turn7 and32 in turn8, then loses:
37 total does not violate a per-selection32 cap. All have zero engine-health
errors. The cap bounds the loop but the policy still repeats it.

Greedy147's400 LP loss attacked a facedown target. Its later-revealed1200 DEF
must not be treated as information the policy had when attacking. These signals
need context, not automatic bad-action labels. Keep #83/#218/#239 open.
