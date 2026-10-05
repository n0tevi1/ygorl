# Ordinary-shuffle teacher continuation (2026-10-05)

Related: #88, #83; follows the failed all-deck coverage gate in
[the MD teacher pilot](md-bc-data-2026-10-05.md). This is a development diagnostic,
not a formal held-out evaluation or evidence of improved agent strength.

## Protocol frozen before the searches

Sources are the eight saved Lunalight Perfume Dancer lines and the six saved
Maliss/Tearlaments lines that failed ordinary-shuffle transfer in the previous
pilot. Preserve source hashes. Those six are a failure-conditioned sample;
never report their recovery rate as the overall deck's success probability.

For each of these 14 starting hands compare two searches, 28 jobs total:

1. From-start search in an ordinary-shuffle duel.
2. The same search supplied with the longest transferable source prefix,
   stopping at the solver's last response (before passive turn closing) or
   immediately before the first semantic mismatch. Only complete core responses
   are exported; partial host selections are discarded.

Both arms use the exact same recorded deck order, core seed, opponent and MD
rules with `DUEL_PSEUDO_SHUFFLE` cleared. Final targets are unchanged: Liger
Dancer for Lunalight; the original targets for Maliss/Tearlaments. Both use
`--no-plan` (no learned reference repertoire), 30,000 ms search, 20,000 ms reserved
for the finisher, one thread, seed `2026100503`, at most four written candidates,
and depth `12*(main+extra)+32`. Use four CPU workers under nice 19 and a 60 s
per-process timeout. Run in paired hand order, alternating which arm is first.
Report actual elapsed time too: native prelude/finisher work may exceed the
nominal search budget. Prefix-generation cost is extra/amortized from the prior
pilot; this is not an equal-total-compute claim.

The native `--approach` mechanism can backtrack the supplied line and solves on
that file's own replay header. Thus every approach and candidate must match the
intended start (both ordered decks, seed, flags and player rules). Reject a
candidate of a different start even if it reaches the target. Independently
replay every accepted line, require turn 2 and the original final target.
Record all failures, native warnings, rejected candidates and timeouts.

No changing targets, seeds, budgets, sample membership or search controls after
looking at results. After these development searches, decide whether the method
merits a new independent-seed coverage test. Do not generate or tune on formal
held-out hands in this experiment. No large-network training yet.

## Implementation

`SolveRequest.start` uses the recorded start instead of constructing a synthetic
pseudo-shuffle hand. `continue_opening` checks environment/legality, derives an
ordinary-shuffle prefix, verifies exported start identities, and filters and
independently replays each resulting candidate. Intermediate-only completions
never become final-target demonstrations. Native logs, commands and replay
inputs are retained for auditing; the result is ordinary `Demonstration` JSON.

Artifacts: `out/research/teacher-continuation-2026-10-05/`.
Generation: `cd13d26` for the paired experiment; `559d971` for direct ordinary generation.


## First paired result and follow-up protocol

At generation commit `cd13d26`, both arms solve exactly the same hands:
Lunalight **1/8**, Maliss **2/2**, Tearlaments **4/4**, with zero error/unverified
records and zero rejected candidates. Supplying the current intermediate prefix
provides no observed coverage gain. All six previously failed shuffle branches
are now solved to their original targets in ordinary duels, including in the
from-start arm. These are development recoveries, not a strength evaluation.
The Lunalight intermediate-prefix hypothesis has not passed its coverage gate.

The native source accepts `--no-ref --start`, even though its README's flag
table emphasizes `--deck`. This permits ordinary-shuffle generation directly
from an empty recorded opening, without any solved source line or reference
repertoire. Add this as opt-in `solve_openings.py --ordinary-shuffle --no-fire`.
Preserve the exact start even on unsolved records; otherwise the existing BC
opening evaluator reconstructs those hands with pseudo-shuffle and changes the
rules for just the unsuccessful cases. Fix that before evaluating this data.

**Independent development check, specified before running:** Maliss and
Tearlaments, 16 hands each (indices 0–15), base and solver seed `2026100504`,
30,000 ms, one thread, four workers, one line, 60 s timeout, no fire. Use the
original target cards, ordinary shuffling, no reference, and the solver's default
finisher split. This is a new recipe, not an extension of the preceding paired
comparison. Pass the narrow two-deck coverage check only with >=8/16 solved in
each deck, zero engine/replay failures, exact candidate-start identity and
ordinary flags throughout. Audit all raw results and the BC encoding path.
Do not claim it satisfies the still-failing 20-deck gate or generalizes to all
random shuffles. Seeds are for development only; formal held-out remains unused.


## Independent check results and decision

The **narrow two-deck check passes**: Maliss 15/16 and Tearlaments 15/16,
30/32 solved, two unsolved, zero error/unverified and zero rejected candidates.
Search wall time is 69.8 s (four CPU workers); per-job solver wall times total
253.43 s. All **1,897 action steps** independently replay with ordinary shuffle
flags and exact recorded starts, to the original targets at turn 2. Core seed
words and initial ordered decks match the specified hand sampler for all 32
records, including the two unsolved hands.

BC preprocessing yields **956 non-forced samples**. Exclusions are 473 forced
steps, four select/unselect toggles and one undo; no beyond-128 action is dropped.
Two independent encoding passes produce elementwise-identical observation and
action arrays, and the environment/vocabulary/window-checked save/load roundtrip
is exact. `ordinary-encoded-development.npz` is labeled development-only.
Zero-work CLI resume preserves the raw file hash and reports all 32 records.

The earlier paired experiment has **14 accepted lines / 912 verified steps**
(counting both arms). All 14 hand outcomes agree, and all seven solved pairs
have byte-identical response sequences. Thus the observed recovery belongs to
ordinary from-start search, with no measured benefit from these prefixes. The
successful jobs finish before the finisher approach stage. Lunalight remains
1/8; the intermediate-prefix recipe is not promoted. The native logs show the
failed prefix searches did enter the approach stage, so this is not a missing
argument or a silently unused approach on those jobs.

Generation and replay fixes are ready to use. This does **not** finish the
20-deck teacher coverage gate, #88's capacity comparison, or #92's long RL run.
No new policy is trained/promoted. The new seed is an independent development
check, not the formal BC held-out set. Ordinary replay verification here checks
the same concrete shuffle realization; it does not establish transfer to every
possible draw/mill or remove the perfect-information teacher's access to deck
order. Passive synthetic opponents remain opening exercises, not full MD games.

**Next:** test guidance from successful *complete* Lunalight lines against an
unguided ordinary-shuffle control, keeping the Liger target and matched budgets.
Do not keep scaling the failed intermediate-prefix recipe or substitute Perfume
Dancer as the final target. Then run a fresh all-deck ordinary-generation coverage
check, freeze formal train/held-out inputs and compare network capacities.

Validation: 71 BC/solver/funnel tests pass. Full presubmit of final implementation
`559d971`: **1,424 passed / 3 skipped** in 297.11 s, including real ROCm tests.
Skips: one snapshot build-option test and two opt-in network tests. No hosted
GitHub checks. `validation.json` seals the source revisions and artifact hashes.
