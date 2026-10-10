# Cancellation-budget stress results and confirmation

Budget1 is the next confirmation candidate, not an adopted default. It sharply
reduces cancellations in selected high-repeat games and has no observed lost
wins on this fresh panel, but does not materially shorten ordinary games.
These are inference-mask effects, not evidence of learned strategic improvement.

## Audited experiment

Root: `out/research/cancel-budget-stress-restart-2026-10-09/`.
Identity: `1cd40984ef6f5eadb17617690b4b76febdbb122738f7bb8eb2bffb8e1ab3064b`.
All 1,128 registered games completed without health failures. The independent
`review.py` / `completion-review.json` checks exact panel coverage, unique records,
record/replay hashes, cold replay results, unwrapped baseline equality and every
saved mask intervention. Removed actions are cancellations only, a legal action
always remains, and the chosen action is allowed. The 26 historical controls
passed exact outcomes/responses and all non-probability step fields before any
intervention. The earlier recording-only STOP remains preserved; see the
[recovery account](selection-terminal-policy-replication-2026-10-09.md).

Cancellation counts are candidate-only and cover material/target selection.
A budget bounds the tracked selection cancellation sequence, not all cancellations
in a game. Different selection sequences can each use the budget. Other forms
of repetition are not established to be eliminated.

## Selected stress cases

All26 inputs with >=8 candidate material cancellations in the frozen completed
seed0 warm u256/u512 and seed1 cold u256 cells were included, irrespective of
outcome or repeated-input classification. This is selected-case evidence.

| Budget | Wins /26 | Total cancels | Mean decisions | Mean turns | Mask interventions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 21 | 609 | 525.92 | 8.6154 | 0 |
| 4 | 22 | 110 | 473.54 | 8.3077 | 29 |
| 1 | 22 | 28 | 449.85 | 8.1923 | 28 |
| 0 | 22 | 0 | 462.62 | 8.8077 | 64 |

Budget1 removes95.4% of observed cancellations and14.5% of decisions here.
Relative to32, it gains one win with no lost wins. Budgets4 and0 each gain two
and lose one. These totals cannot estimate population win rates or prove that
all removed cancellations were strategically useless.

## Fresh panel

Fixed seed0 warm u512 candidate, 32 shared new deal blocks, two deck assignments,
two starts, Greedy and historical RL: 256 inputs per budget. Do not mix these
rates with the selected stress cases.

| Budget | Wins /256 | Greedy wins /128 | Historical wins /128 | Cancels | Mean decisions | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 175 | 99 | 76 | 68 | 418.63 | 8.6953 |
| 4 | 176 | 99 | 77 | 29 | 417.52 | 8.6953 |
| 1 | 177 | 99 | 78 | 15 | 414.63 | 8.6523 |
| 0 | 176 | 98 | 78 | 0 | 418.90 | 8.7109 |

Budget1 has two gained wins and no lost wins, +0.7813 percentage points, but only
15 mask interventions. Its mean turn difference is -0.0430 and decision difference
-4.0078. The registered shared-deal bootstrap gives intervals [0,+1.9531] points,
[-0.1134,0] turns and [-11.0042,+0.0352] decisions. With so few discordant outcomes,
a bootstrap lower bound of zero is **not** evidence of population noninferiority
or absence of tactical harm. It cannot resample failure modes absent from this
panel. This was also the panel used to choose budget1 from three alternatives.

Budget0 gains two historical-RL wins but loses one Greedy win (game107), with
that game lengthening by8 turns. At the first divergence after16 common decisions,
the baseline cancels and budget0 selects card875572. The same initial change
against historical RL leads to a gained win. Accepted-action divergence is
established, but changed later sampling alignment prevents attributing the final
winner to the immediate tactical quality of the forced selection.

There were no sole-exit fallbacks in these games; tests, not this panel, cover
that exception. The known alternate-target challenge still admits its first
cancel with budgets1/4/32 and blocks it with0. It proves a legal route can be lost,
not that this route is always necessary or optimal. No full cancellation ban is
adopted. Complete healthy games do not by themselves establish good selections.

## Registered next step

`out/research/cancel-budget-confirmation-2026-10-09/`, identity
`fd440b0f121cfd26d84486b486f21e47ea18dc4d315a74d37e3f9fec63723d42`.
Budget1 versus32 on three fixed candidates: seed0 warm u512, seed0 cold u512 and
seed1 cold u256. Each faces Greedy and historical RL on32 new shared deal blocks
with both assignments/starts: 768 inputs, 1,536 games. Panel seed20261009271;
20,000 shared-block bootstrap draws, seed20261009272. Report every candidate and
opponent separately. These checkpoints and shared deals are correlated; do not
treat them as independent evidence about a training population.

Preparation freezes full Python/tool/native/card/script/deck/environment identities
and checkpoint hashes. A separate two-budget preflight passed. The study runs
with two workers, a three-hour cap, per-game durable artifacts, ten-second
heartbeat and an independent watchdog. All768 baselines precede the restricted
runs. Health, identity or replay failure preserves STOP and exits without replacing
games. Formal training and policy/mask defaults remain unchanged.

On completion, independently audit all results, review newly lost games and
selection exceptions, and decide whether a configurable deployment/training
trial is justified. Budget1 remains a hypothesis; win retention and tactical
selection checks matter more than making the cancellation metric zero.
