# All-action sweep: shorter games can mean faster losses

All992 full continuations and30 original trial0 controls passed independent
review, with no health errors or cutoffs. Root-action choices affect both win
probability and duration, but the selected long-game windows do not reveal
one universal early-finish fix. Do not optimize raw turn length independently
of winning, or train from an in-sample best action without confirmation.

Root: `out/research/combat-root-action-sweep-2026-10-09/`.
Identity: `dfc32df86674d86cd4e405a2f834364bf055c0d0c53b4ab4e53a8189995f4e1e`.
Completion was acknowledged after `review.py` checked every result, action,
source binding and exact original full-game control.

The30 windows contain97 native-supported actions. Six previously positive
windows use16 paired suffix streams;24 selected long-game windows use8.
Every action is followed by the original learned policy and original opponent,
not the heuristic teacher. Root action and original root RNG consumption are
fixed, and suffix seeds are paired across arms.

## Earlier positive windows

The direct Heavy Borger effect and previously identified successful attacks
reproduce their earlier results. Alternative initial attacks can have sharply
different outcomes despite all being native-mask-supported. At cold65/499,
the previously validated first attack wins16/16, but another attack wins3/16.

At warm72/466, initiating with Spright Elf instead of the prior teacher's
chosen attacker gives16/16 eventual wins and11/16 same-turn finishes, versus
the original12/16 wins and no immediate finishes. Mean turns change7.250→5.750,
with no observed paired win-to-loss reversal. The prior teacher first action
gives14/16 eventual wins and5/16 immediate finishes, with two paired harms.
The Elf attack has root probability0.0096%. This candidate was selected from
the sweep and needs new suffix confirmation before any superiority claim.

## Long-game windows

There are70 root actions and560 branches across24 long-game windows; none
finishes on the root turn. Among46 non-control alternatives,32 shorten mean
duration, but18 of those also reduce wins in these small samples. These are
descriptive counts of selected interventions, not population error rates.

| State | Candidate intervention | Original → intervention wins | Mean turns | Paired wins lost |
| --- | --- | ---: | ---: | ---: |
| seed0 cold512 / Greedy5 | end phase | 1→5 /8 | 17.125→16.000 | 0 |
| seed0 warm512 / Greedy50 | set Dominus Purge | 1→4 /8 | 15.125→15.375 | 0 |
| seed1 cold256 / Greedy38 | attack | 6→1 /8 | 15.000→9.375 | 5 |
| seed1 cold256 / historical47 | attack | 2→6 /8 | 12.500→11.750 | 1 |
| seed1 cold256 / historical47 | activate | 2→4 /8 | 12.500→11.000 | 1 |
| seed1 cold256 / historical109 | attack | 5→7 /8 | 13.125→11.000 | 0 |

Greedy38 forces a2500 attacker into a public2800 attacker. The large duration
reduction accompanies lost games; the original policy strongly prefers main2
(99.83%). Conversely, historical109's300 attacker into500 improves sampled
wins. This does not establish why the latter works or justify an ATK-only
rule: follow-up needs full outcomes and replayed sequences.

Not every observed poor choice reflects the actor's dominant preference.
At Greedy50, the sampled end-phase action has probability0.54%, while setting
the trap has57.77% and battle entry41.69%. At historical47, the sampled main2
has11.82%, whereas attack has75.40%. Separately inspect stochastic bad tails
and persistently misranked actions; do not infer a need to eliminate exploration
from a selected bad trajectory.

The original full games were chosen for lasting at least16 turns. Their
resampled controls can finish earlier: this is conditional selection on an
original trajectory, not an estimate of the general turn distribution. All
effects remain conditional on fixed deals, hidden states and fitted actors.

## Registered independent suffix confirmation

Completed: see [independent confirmation and full-game sampling comparison](combat-root-confirmation-2026-10-09.md).

Root: `out/research/combat-root-confirmation-2026-10-09/`.
Identity: `921fed2204470124908e620d3d1ab65de76f84ea3c62b443e6973cc94a1179b7`.
Six selected states,14 state/action pairs,64 new suffix streams each and six
exact original calibration games give902 branches. The selection is frozen:

- Warm72: original main2, prior teacher attack, alternative Elf attack.
- Cold/Greedy5: main2 versus end phase.
- Warm/Greedy50: end phase versus trap set.
- Cold1/Greedy38: main2 versus attack, retaining harmful shortening as a control.
- Cold1/historical47: main2 versus activation versus attack.
- Cold1/historical109: main2 versus attack.

Trial IDs100–163 yield RNG seeds disjoint from the exploratory1–15 streams.
Original root sampling is consumed equally, then both actors' suffix RNG is
reset. New streams are paired across arms and states; these are not new deals
or hidden-state generalization. Analyze all contrasts, including harms, with
conditional paired intervals and turns among pairs won in both arms. Do not
pool states or claim multiplicity-adjusted superiority.

Save full replays for the six trial0 controls and14 trial100 branches; all
other branches retain results and candidate same-turn action/probability
traces. Independent review requires the calibration response/step transcripts
to match the original records exactly. A real calibration replay and one new
suffix preflight passed. Two CPU workers, a three-hour cap, saved per-branch
results and an independent watchdog completed the902-branch study. Independent
review passed all records and calibration transcripts. Formal training and policy
defaults are unchanged; the long-game root cause remains open.
