# Role coverage confirmation: aggregate gains hide facing-Runick regression

The [equal-budget role experiment](critic-role-coverage-fit-2026-10-10.md)
completed independent confirmation on 512 healthy games / 128 new openings.
All 14 checkpoint forwards and aggregate/role metrics reproduce exactly; actor
hashes remain unchanged. No critic is promoted to RL.

## Confirmed tradeoff

The following scores average metrics over the three auxiliary initializations.
All models score the same fixed actor's states, with equal weight per game.

| Critic | Overall Q/V MSE | Face-role MSE | Own-role MSE | Facing Runick V bias |
|---|---:|---:|---:|---:|
| Face-only MSE | 0.49546 | 0.63721 | 0.35370 | +0.50980 |
| Balanced MSE | 0.48179 | 0.66107 | 0.30250 | +0.65271 |
| Balanced calibration | 0.48146 | 0.65903 | 0.30388 | +0.64058 |

For balanced MSE minus face-only MSE, a 20,000-replicate paired opening-deal
bootstrap gives these descriptive, unadjusted 95% intervals:

| Slice | MSE change | Interval |
|---|---:|---:|
| Overall | -0.01367 | [-0.03043, +0.00349] |
| Facing targets | +0.02385 | [+0.00320, +0.04494] |
| Operating targets | -0.05120 | [-0.07777, -0.02295] |
| Facing Runick | +0.18482 | [+0.08843, +0.29044] |

The resampling preserves target/anchor/role strata and keeps opponent/seat games
from one opening together. Initializations are averaged before resampling, not
counted as extra independent games. Facing Runick has only 32 games / eight
openings; inference remains limited to this actor, deck-list and opponent panel.
The overall interval includes no improvement while the face-role regression is
clear on this panel. Conditional mean calibration does not remove the tradeoff.

At the first eligible state with own deck <=5 and LP ahead, 12 facing-Runick
games have mean terminal return **-0.8333**. Face-only mean V is +0.23980, versus
balanced +0.45053 and balanced-calibration +0.42867. V MSE worsens from 1.51654
to 1.98406 / 1.93721. These are small, trajectory-conditioned diagnostic slices;
they do not justify a universal LP penalty.

## Mechanism: changed return prior and conflicting updates

A post-confirmation audit examined the same three matched initializations. Its
analysis plan was recorded before the offset and gradient measurements, but it
is exploratory and was motivated by these confirmation results.

The equal-game training mean terminal return changes from **+0.32031** in the
face-only corpus to **+0.55469** in the balanced corpus. Within the latter,
face-role return is +0.29688 and own-role return +0.81250. Adding own-role data
therefore also changes the outcome distribution substantially.

For the middle initialization, mean balanced-minus-face V on the balanced
**training** data is +0.11752. On confirmation, V rises by +0.11028 for the
face role and +0.12590 for the own role. A constant Q/V offset computed on
training data accounts for much of the net aggregate MSE change: arithmetic
offset effect -0.01303, remaining effect -0.00061, actual change -0.01364. This
is an additive error decomposition, not a new calibrated model or a claim that
all individual state changes are constant. Arithmetic intermediates may lie
outside [-1,1]. No confirmation outcomes fit the offset.

A direct optimizer probe supports a training mechanism, beyond correlation:

- Start from the face-only selected checkpoint, middle initialization.
- Compute exact equal-game MSE gradients separately for own-role training,
  face-role training, and facing-Runick training in the balanced corpus.
- Own-role versus facing-Runick gradient cosine is **-0.81365**. Face-role
  versus facing-Runick cosine is +0.65401.
- One fresh Adam step on own-role examples alone increases facing-Runick
  training MSE **0.88972 → 0.89516**, and mean V **0.40116 → 0.40520**.
  A face-role-only step lowers its MSE to 0.88589; a facing-Runick-only step
  lowers it to 0.88382. Each starts from identical weights and empty optimizer
  state (LR 1e-5, epsilon 1e-5, clipping 0.5); actor parameters remain fixed.

These measurements show cross-role interference at the tested checkpoint and
explain why more role coverage can worsen a previously overestimated matchup.
They do not establish that representation is sufficient, that priors explain
every error, or that the same effect holds throughout PPO training.

## Next bounded objective pilot

`critic-group-loss-pilot-2026-10-10` uses the same balanced training/validation,
architecture and 32-update budget. A single registered initialization compares
ordinary MSE against a smooth maximum of the sixteen target/role group MSEs,
at temperatures 0.25 and 0.5. The objective is
`tau * (logsumexp(group_mse / tau) - log(group_count))`.
It gives higher-error groups more weight without a Runick-specific rule or
inference-time group label. Lower aggregate error alone remains insufficient.

`tools/critic_group_probe.py` implements the objective and exact two-pass
derivative. Tests compare the accumulated gradient with direct autograd under
unequal group/game weights, verify the ordinary-MSE control and reject invalid
temperatures. Together with resource/corpus tests, 13 focused tests pass.

All three selected fits chose update 32 and their metrics/minima were independently
reproduced. The ordinary-MSE control reproduces the preceding same-seed balanced
MSE result to below 2e-9 aggregate validation MSE difference. Validation shows
another tradeoff:

| Objective | Overall Q/V MSE | Facing Runick Q/V MSE | Facing Runick V bias | Operating Runick Q/V MSE |
|---|---:|---:|---:|---:|
| Ordinary MSE | 0.45560 | 1.02809 | +0.34998 | 0.68540 |
| Group maximum, tau 0.25 | 0.47725 | 0.95192 | +0.14640 | 0.77548 |
| Group maximum, tau 0.5 | 0.46786 | 0.96678 | +0.20192 | 0.75031 |

Reducing facing-Runick bias is possible, but these objectives sacrifice aggregate
and own-Runick error. This is not an accepted fix. All arms use the same balanced
validation selection criterion; none is chosen or modified using test outcomes.
`critic-group-loss-confirmation-2026-10-10` completed 512 new games on 128
openings, disjoint from 832 prior openings. It evaluates all three selected
candidates, unchanged source and a fixed prior resource/calibration reference.
Report roles, target lists and low-deck retention separately. This remains a
single-initialization pilot; no inference defaults, actor parameters or rewards
change.

The [confirmation and resource-readout follow-up](critic-resource-readout-2026-10-10.md)
reports a confirmed facing-Runick improvement with an own-role regression,
representation diagnostics, and a new frozen Q/V residual confirmation.

Artifacts: `critic-role-coverage-confirmation-2026-10-10` retains independent
verification, predictions and paired analysis; `critic-role-shift-audit-2026-10-10`
retains the exploratory plan, script, input hashes and mechanism results.
The group-loss pilot retains registration, immutable input identity,
checkpoints and independent monitoring under `out/research/`.
