# Card-pool expansion canary: no established gain after 16 updates

From the same seed0 warm-u512 checkpoint, both arms trained 16 PPO updates
(262,144 rows each). Fixed retained the 20-deck pool; expanded added legal
Labrynth, Kashtira, Tenpai Dragon and Runick lists. Full state restoration was
checked before training: actor, critic, optimizer, reference, snapshot pool and
RNG were identical, with only the registered deck list changed. Text features,
reward, inference and cancellation budget stayed fixed. This tests adaptation
to those lists, **not** zero-shot generalization to unseen cards or families.

All 768 fresh paired evaluation games were healthy, with no limits and exact
cold replay matches. Frozen inputs, eight training checkpoints, all saved result
records and replay hashes were independently verified. Neither arm is promoted.

## Results

Every cell contains 64 games: four target decks, two anchors, two deals, both
seats and two fixed opponents. These are weak-opponent diagnostics. The three
candidate policies use identical evaluation deals and agent RNG seeds.

| Candidate role / target group | Start u512 | Fixed u528 | Expanded u528 |
|---|---:|---:|---:|
| Pilot familiar decks | 61/64 | 63/64 | 63/64 |
| Face familiar decks | 41/64 | 47/64 | 43/64 |
| Pilot added decks | 57/64 | 63/64 | 56/64 |
| Face added decks | 50/64 | 41/64 | 47/64 |
| Total | 209/256 | 214/256 | 209/256 |

Expanded minus fixed when piloting added decks is -10.94 percentage points;
registered paired cluster bootstrap interval [-20.31, -3.13] points. When facing
added decks it is +9.38 points, interval [-6.25, +25.00]. Familiar own-deck
performance is tied; facing familiar decks is -6.25 points, interval
[-21.88, +10.94]. These are conditional single-seed, multiple-endpoint pilot
intervals. They do not establish a general harm from diversity. Expanded is only
one win below the starting actor on added own decks; much of the relative gap
comes from the fixed arm improving. Do not call this catastrophic forgetting.

Mean turns (start / fixed / expanded): familiar-own 8.02 / 8.36 / 8.63,
familiar-facing 11.30 / 10.36 / 10.00, added-own 12.94 / 12.94 / 12.98,
added-facing 12.73 / 15.00 / 14.25. There is no broad shortening. Facing Runick,
wins are 12/16 / 7/16 / 8/16 and mean turns 17.13 / 26.13 / 26.31.
Short games alone remain an inappropriate objective for a battle-lock/deck-out
matchup; report wins, termination reason and legal action availability together.

## Root-cause checks and remaining uncertainty

New decks were actually dealt: 581/1,839 completed expanded-arm training games
involved an added deck (31.6%), versus zero of 1,869 fixed-arm games. The expected
share under uniform cross-pairs is 172/552 = 31.2%. Completed games with the learner
piloting Kashtira / Labrynth / Runick / Tenpai were 164 / 110 / 126 / 124. These
counts include correlated paired seats and are not independent sample sizes.
Engine errors and truncations were zero in both arms. Missing deck ingestion,
an invalid list, failed restoration or corrupted evaluation is not supported by
the evidence. This is still a short exposure per new deck.

The eight added-own cases that fixed won and expanded lost first diverge at a
chain/pass decision under matching recorded public inputs; there is also one
opposite outcome. The changed choice and later loss are not enough to label that
first choice tactically wrong: subsequent policy and sampled trajectories also
change. First-divergence records are retained for counterfactual investigation.
A larger per-update KL is not a sufficient explanation either: mean KL to the
moving reference was 0.0356 expanded versus 0.0493 fixed; these values are not
KL to the starting actor. The causal explanation for the role-specific gap
remains unresolved. Check replication before redesigning rewards or masking cards.

Repeated selection cancels occur in 2 / 1 / 2 evaluation games; four games overall
cross the eight-repeat warning threshold. Self-Ash selections/opportunities are
2/102, 0/78 and 1/96. Expanded's one self-Ash repeats a starting-actor case involving
its own Mulcharmy Fuwalos (probabilities 8.88% versus 8.28%). These tiny correlated
counts do not establish that either issue is fixed or that expansion caused it.
Neither loops nor self-Ash occurs in the eight added-own discordant losses, so
those warning signals do not explain that particular performance gap.

## Independent replication

Register training seeds 1 and 2 before reading their outcomes. Repeat the same
16-update fixed/expanded intervention and evaluate start/fixed/expanded on a
fresh common panel (768 games per seed, 1,536 total), disjoint from discovery
opening seeds. Report each seed, own/facing roles, familiar retention and Runick
separately; seed0 remains discovery evidence and is not pooled into confirmation.
With only two replication seeds, even a pooled interval remains exploratory.

Use full restoration checks, checkpoints every four updates, pre-optimizer
engine-health gates, exact cold replays and independent progress monitoring.
Keep failed artifacts and STOP markers. No text switch, mutation, blanket masks
or automatic policy promotion. A positive result would still require unseen-list
and unseen-family tests before claiming broader card intelligence.

Evidence: `out/research/card-pool-training-canary-2026-10-10/{analysis.json,
training-exposure.json,first-divergences.json}`. Replication registration and
frozen runtime bindings: `out/research/card-pool-training-replication-2026-10-10/`.
Related: #36, #83, #105, #218; prior coverage/lock report in PR #284.
