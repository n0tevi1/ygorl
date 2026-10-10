# Complete-game terminal policy-credit pilot

The first paired pilot does **not** support replacing the default policy credit.
Terminal credit has lower observed Greedy win rate and slightly longer games;
the paired intervals include zero. Fewer evaluation cancellations are not
established improvement: training contains more cancellations and severe repeated
inputs. Both arms remain research controls; production defaults are unchanged.

## Verified scope and evaluation

Study: `out/research/selection-terminal-policy-pilot-2026-10-09/`, identity
`65ca92ae4e8f7e7272bd0d9c671b2995522aebd123c01a1a292a434de7130d7a`.
One common seed250 legacy checkpoint; training seed265; 24 updates ×16 complete
selfplay games per arm. Recursive comparison confirms identical initial learner
states, including actor, critic, Adam and EMA reference. Only policy credit
changes from VRPO lambda=.5 to terminal outcome minus the same expected-Q
baseline, with whole-rollout standard normalization. Critic targets stay
lambda=.5. No reward, mask or card-specific rule changes.

All 768 training games and 384 evaluation games pass health checks, with no
truncations. Evaluation uses 32 common deck/deal blocks ×4 assignments/starts.
The common initial actor is evaluated once.

| Endpoint | Wins /128 | Win rate | Mean turns | P90 turns | Cancels / opportunities | Games with >=8 cancels without public progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial | 94 | 73.4375% | 10.2344 | 15 | 6 /63 | 0 |
| VRPO final | 93 | 72.6563% | 10.2500 | 17 | 7 /59 | 0 |
| Terminal final | 86 | 67.1875% | 10.5781 | 16 | 5 /45 | 0 |

Terminal minus VRPO: win rate **-5.4688 percentage points**, paired 95% bootstrap
interval **[-12.5000, +1.5625]**; mean turns **+0.3281**, interval
**[-1.0234, +1.4844]**. There are 17 terminal-only wins and 24 VRPO-only wins.
Both cancellation counts and opportunities differ; the observed per-opportunity
rates (11.11% versus 11.86%) also do not justify a success claim.

Bootstrap resamples the 32 shared blocks 20,000 times (seed20261009266), not
128 independent games. These intervals describe evaluation uncertainty for
these endpoints; they do not include training-seed uncertainty. Initial-to-final
win changes also have intervals spanning zero. The panel has no severe repeat
cases, limiting what its cancellation metric can establish. Raw outcomes,
accepted actions and engine responses are retained for all games.

## Training and repeated-input audit

| Metric | VRPO | Terminal |
| --- | ---: | ---: |
| Complete games | 384 | 384 |
| Learner rows | 60,274 | 57,054 |
| Summed reported training step seconds | 234.66 | 256.47 |
| Mean selfplay turns | 9.2161 | 9.4818 |
| Chosen cancel rows | 24 | 87 |
| Repeated actor-input groups (>=2 cancels within a game) | 5 | 10 |
| Maximum same-input cancel count | 3 | 31 |
| Updates stopped early by KL guard | 11 /24 | 6 /24 |
| Mean reported gradient norm | 0.8156 | 0.8489 |

Training trajectories diverge after updates; counts are descriptive, not paired
counterfactual outcomes. Equal games and updates are not equal rows or compute.
Reported step times exclude the complete service/evaluation runtime and do not
establish an infrastructure speed difference. Identical actor inputs can hide
private pending choices, so these groups are investigation signals, not automatic
bad-action labels. The exact largest count is **31** cancel rows; the initial
chat update described it as 30 before counting the saved row list.

Terminal credit assigns the same normalized advantage to repeated cancel and
same-input exit in these groups. In a winning u36 episode, two cancels and the
exit all receive +1.51733; actual recorded cancellation probability rises
12.3647% →19.9078%. This is consistent with reinforcing the sampled repetitions,
but the full update includes all other rows, critic, regularizers and Adam.
A contrary winning u42 episode has eight cancels and exit all at +1.13049,
yet cancellation probability **falls** 92.7216% →92.0776%. Positive row credit
alone therefore does not establish the aggregate gradient direction.

In the losing u40 episode, the 31-cancel group and its exit all receive -0.91151;
actual cancellation probability still rises 96.8257% →97.0372%. At the behavior
policy, the summed derivative of just those cancel/exit policy losses with
respect to a shared cancel logit is +0.01438 (favoring a decrease), but this is
not the full PPO/Adam update. All 15 groups and their contrary directions are
reported, not only these examples. CPU pre-update probabilities match recorded
behavior probabilities within 1e-5; post-update probes load actual saved states.

On the same saved rollouts, average within-rollout raw advantage standard
deviation is about 0.0932–0.0956 for short returns versus 0.8840–0.8849 for
terminal returns. Both objectives normalize their used advantages. This raw
spread is **not** a policy-gradient variance estimate or proof that noise caused
the win difference; terminal has fewer KL early stops and a comparable mean
reported gradient norm. A winning continuation does not label every preceding
action useful. Undiscounted win-only rewards also contain no direct preference
for fewer harmless steps if those steps preserve winning chances.

Independent audit artifacts: `analyze.py`, `audited-analysis.json`,
`probe-repeats.py`, `repeat-probes.json`. Saved minibatch counts/acceptance agree
with metrics; complete-game layouts, health and common evaluation jobs are
checked. The [earlier same-input credit analysis](selection-exit-credit-2026-10-09.md)
remains valid; correcting the short-return sign on selected winning exits alone
was insufficient to establish better play.

## Fixed-budget replication completed

The two additional seeds266/267 completed with all health and credit audits.
Neither establishes a strength benefit; the candidate remains unpromoted and
further identical terminal-credit trials are not queued. See the
[replication report and cancellation-budget follow-up](selection-terminal-policy-replication-2026-10-09.md)
for per-seed results, shared-panel uncertainty and the next experiment.
