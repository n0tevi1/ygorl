# Long-game battle-exit audit

This finite probe did not verify a missed same-turn win in the tested long-game
battle-exit windows. It does not prove no winning line exists, and available
attacks must not be turned into automatic error labels or penalties.

Root: `out/research/longgame-battle-audit-2026-10-09/`.
Identity: `c5a9c9ddb3472d0d85e3546dd08caddeec0be9b52a549f7ac711f0d2d2fc53b3`.
All768 frozen confirmation controls were replayed without policy resampling.
`review.py` independently checks coverage, results, hashes and every original
engine response and step after diagnostic snapshot restoration.

| Scope | Games | Games with battle-exit signal | Exit windows | Probed windows | Continuations | Conditional same-turn wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full panel | 768 | 292 | 450 | 40 | 48 | 0 |
| Games lasting >=16 turns | 32 | 23 | 64 | 40 | 48 | 0 |

A signal means the candidate chose main2/end_phase while the encoded mask still
allowed an attack. Probes cover only the first two such windows per long game,
at most four different initial attacks, then at most128 decisions of Greedy
continuation against a passive opponent. No exhaustive search, no active-opponent
winning guarantee, and no claim of public-information safety. Shorter-game
windows were counted but not searched. Existing event filtering does not remove
all dependence on realized hidden state.

## Completion contract repaired without rewriting evidence

The driver completed successfully but omitted the monitor-required `healthy`
field from its report. The independent monitor correctly rejected completion
and delivered an attention event. This was a report-schema defect, not a failed
duel; no run was restarted or replaced.

The original `report.json`, alert and monitor config remain preserved. Only after
independently checking all768 records and full transcripts, `reviewed-completion.json`
adds a positive health assertion, binds the original report hash and supplies
the audit results. The monitor was pointed to this separate proof and restarted
to load the config. It subsequently reported verified completion with no alerts.
Both attention and completion were acknowledged. The successor explicitly emits
the required health field. The earlier takeover note's provisional total482 was
corrected to450; the table above is the independently summed result.

## Next: ending the turn without entering battle

The prior probe cannot detect a candidate ending its main phase despite having
a legal battle_phase action, because no battle-exit window then occurs.
The next study covers this separate trigger on the same768 frozen inputs:
`out/research/longgame-phase-entry-audit-2026-10-09/`, identity
`f2adfd64b072006dab679d018b35639e40814e40e75da50adc3f2d75e2670855`.

For >=16-turn games, probe the first two skipped-entry windows: force battle
entry, then the same bounded Greedy/passive continuation. Count all windows and
retain all originals; do not change seeds to seek a positive result. A real-game
preflight passed. The CPU-only two-worker service and independent watchdog are
running, with a three-hour cap and exact transcript/identity checks. Conditional
wins, if found, require further review before being used as teaching examples.
Formal training, rewards and policy/mask defaults remain unchanged.
