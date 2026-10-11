# Low-deck gains confirm, but resource-only correction harms retention

The [fixed-mixture fit](critic-lowdeck-mixture-2026-10-10.md) completed independent
confirmation on 768 healthy games: 512 broad and 256 Runick, with 192 openings
disjoint from 1,856 prior openings. All games have eligible idle samples.
Independent verification checks replay/cache hashes, opening/panel identities
and exactly reproduces all four prediction groups and reported metrics. The
completion event was acknowledged after verification.

## Separate confirmation panels

Q/V errors are equal-game MSE on terminal returns in [-1,+1]. All estimators
score identical frozen-actor outcomes; these are not gameplay improvements.

| Estimator | Broad | Broad familiar | Runick | Facing Runick | Operating Runick |
|---|---:|---:|---:|---:|---:|
| Frozen base | 0.56773 | 0.51372 | 1.07963 | 1.26001 | 0.89926 |
| Original conditioned control | 0.50968 | 0.43694 | 0.96974 | 1.20614 | 0.73334 |
| 25% new-data mass | 0.49289 | 0.45027 | 0.82507 | 0.84014 | 0.81000 |
| 50% new-data mass | 0.50084 | 0.46394 | 0.80417 | 0.79757 | 0.81076 |

Differences versus the original conditioned control, with 20,000 stratified
paired-opening bootstrap replicates preserving opponent/seat clusters:

| Slice | Mixed25 difference [95% interval] | Mixed50 difference [95% interval] |
|---|---|---|
| Broad | -0.01679 [-0.04414, +0.00951] | -0.00885 [-0.04176, +0.02258] |
| Broad familiar | +0.01333 [+0.00426, +0.02248] | +0.02701 [+0.01369, +0.04094] |
| Runick | -0.14467 [-0.22476, -0.06510] | -0.16557 [-0.26434, -0.07115] |
| Facing Runick | -0.36600 [-0.51936, -0.21738] | -0.40857 [-0.58952, -0.23350] |
| Operating Runick | +0.07666 [+0.02175, +0.13689] | +0.07742 [+0.00799, +0.15007] |

Intervals are descriptive and unadjusted. Both own-Runick and familiar-retention
regressions replicate. Broad improvement versus the current control remains
uncertain, even though both mixtures improve over the older frozen base. The
validation retention gates did not guarantee retention on fresh openings.
Neither mixture is adopted, and no winner is selected from this confirmation.

## Low-deck benefit is real but incomplete

In the Runick panel, 73 games / 38 openings reach own deck size <=5 at a sampled
idle state. The first qualifying state per game is scored:

| Estimator | V MSE | Signed V bias |
|---|---:|---:|
| Frozen base | 2.01613 | +0.86422 |
| Original conditioned control | 1.76726 | +0.86991 |
| Mixed25 | 0.93751 | +0.27132 |
| Mixed50 | 0.82420 | +0.17370 |

MSE differences versus control are -0.82975 [-1.06408, -0.57615] and -0.94307
[-1.20628, -0.65445]. Bias falls but remains positive; the near-zero validation
bias does not carry over exactly. This is improved terminal prediction, not
proof of accurate values for every low-deck state.

Mean lengths are 11.115 turns on the broad panel and 23.633 on Runick. The
unchanged actor wins 352/512 and 144/256 respectively. These outcomes are shared
by all scored critics and do not establish progress toward shorter games.

## Why investigate the correction's state dependence?

The base critic already sees state context. However, the added linear residual
only sees 13 resource counts and an opponent indicator. It cannot independently
condition its correction on the full frozen card/board representation.

A post-confirmation algebraic audit decomposes MSE change as
`2 * old_error * correction + correction^2`, averaged equally per game. For
Mixed25, the alignment term is -0.52475 facing Runick but +0.04261 operating it;
the latter also pays +0.03404 correction magnitude, totaling +0.07666 harm.
Mixed50 likewise has positive alignment and magnitude terms for own Runick.
Thus the correction is not merely failing to remove a constant average bias:
on this subgroup it systematically adds error. This is a diagnostic
decomposition, not proof that missing context is the sole cause.

## Frozen-context pilot and next confirmation

`critic-context-mixture-fit-2026-10-10` appends the existing frozen public actor
context to the same resource/opponent features. It uses the same data, mixture
weights and ridge grid. No role/target labels or new private observations enter
the residual, and actor/base parameters remain frozen. A new prospective
own-Runick retention gate joins the existing broad/old/familiar/low-deck gates;
this does not retroactively redefine the previous experiment's success.

All twelve coefficient fits and metrics were independently reproduced. A check
also verifies that altering role/target/outcome metadata leaves feature tensors
unchanged. Context25 fails familiar retention at every ridge value and is not
advanced. Context50 has two eligible candidates and selects lambda=0.001:

| Validation metric | Original control | Resource Mixed50 | Context50 |
|---|---:|---:|---:|
| Broad Q/V MSE | 0.58013 | 0.54551 | 0.56260 |
| Broad familiar Q/V MSE | 0.46763 | 0.45128 | 0.47696 |
| Runick Q/V MSE | 0.87981 | 0.74549 | 0.66532 |
| Operating Runick Q/V MSE | 0.67532 | 0.73615 | 0.61273 |
| First-low-deck V MSE | 1.56449 | 1.00361 | 0.69206 |

Familiar error is nearly at the 2% selection allowance, not a demonstrated
retention success. This pilot adds capacity and reuses historical validation;
only fresh confirmation can test its selected result. No prior confirmation
outcomes enter fitting or ridge selection.

`critic-context-mixture-confirmation-2026-10-10` is running with independent
monitoring: 512 broad +256 Runick games, 192 new openings disjoint from all
2,048 prior openings. It scores frozen base, original control, same-data
resource Mixed50 and Context50. Primary contrasts are Context50 versus both
controls, reported separately by panel, role and target, including familiar
retention and first-low-deck uncertainty. No automatic PPO promotion;
learner-policy-shift and matched RL continuation remain later requirements.

Evidence under `out/research/`: the mixture confirmation's `analysis.json`,
`completion-verification.json` and `correction-audit.json`; context pilot
registration, all candidates and exact verification; new confirmation identity
and watchdog handoff. All sealed evidence is preserved.
