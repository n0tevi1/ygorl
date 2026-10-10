# Fresh 4,096-game inference confirmation: no policy promotion

The frozen repaired panel completed all 4,096 games. Independent review verifies
the registered source identities, every result/replay hash, all cold replays, and
that every cancel intervention only removes encoded cancel actions while keeping
an available action. **Evaluation is valid but not fully healthy:** one argmax
game hits the 6,000-decision limit. It remains a non-win in every analysis.

Evidence: `out/research/policy-inference-clown-restart-2026-10-10/`, especially
`reviewed-completion.json`, `aggregate-counts.json`, and the frozen `review.py`.
Identity: `71c5afbc608ab29c9fe35455f06a9a0aa86b8aa4437cc168774f57750030bc22`.
The older Clown Crew Lua-error study retains its original STOP and is not pooled
into this panel. No checkpoint, engine, mask or result was replaced mid-study.

Four fixed u512 candidates (seed 0/1, cold/warm), four opponents, 32 shared deal
blocks, four seat/deck assignments and two inference modes produce 4,096 games.
Deal seeds are disjoint from the previous two sampling panels. The three primary
opponents are weighted equally, as are the fixed candidates; historical RL and
all-four pooling are secondary. Twenty thousand paired bootstrap draws resample
shared deal blocks jointly across cells. These intervals condition on the fitted
models; they are not independent training-seed inference or adjusted cellwise tests.

| Panel | T=1 sample wins | Argmax + cancel-budget-1 wins | Paired delta, 95% deal interval |
|---|---:|---:|---:|
| Primary three | 1297/1536 (84.44%) | 1312/1536 (85.42%) | +0.98 pp [−1.82, +3.71] |
| Historical RL | 320/512 (62.50%) | 352/512 (68.75%) | +6.25 pp [−0.20, +12.50] |
| All four, secondary | 1617/2048 (78.96%) | 1664/2048 (81.25%) | +2.29 pp [−0.44, +4.98] |

The primary strength interval crosses zero. Mean recorded turns fall from
**9.184 to 8.770** (paired −0.413, interval [−0.783, −0.051]); this includes a
censored turn-3 limit and must not be interpreted as all games naturally finishing
sooner. Mean primary decisions are 444.35 versus 436.29. Lower means do not erase
the severe deterministic tail failure.

Across all four opponents, sampled play has zero limits and 344 candidate cancels;
argmax + budget 1 has one limit and 923 cancels. **823** of the latter occur in
job 2815: canceled optional summon attempts return to idle and restart, so the
within-selection cancel budget never intervenes. See the independently reproduced
[summon reentry diagnosis and opt-in mitigation](summon-reentry-2026-10-10.md).
That mitigation's successful selected-case rerun does not replace job 2815 here.

The pipeline service ended successfully after writing its complete report. The
monitor correctly refuses to mark completion healthy because the limit is retained;
its resulting identity/health alert is not a second Lua crash or incomplete panel.
Keep the original report, errors marker, incident acknowledgment and monitor state.

Decision: **keep default sampling**, retain argmax and both cancel/reentry guards
as experimental candidates. There is neither a demonstrated primary strength gain
nor a clean health pass to support promotion. A future guarded-policy comparison
must be separately registered with fresh deals and report strength, natural
termination, procedural loops and retained legal revisions together.
