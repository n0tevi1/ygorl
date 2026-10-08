# Learner precision pilot: verified launch, 2026-10-08

The paired pilot is running on one NVIDIA L4. This report records execution
evidence, not a BF16 speed or strength result. Implementation PR #242 merged as
`df5ad9890acc81cdf0969b286d3bc87f89a85c2d`; the frozen experiment source is
`16f8c66379c7311b188a17febe0541b6a31c669f`.

## First durable training evidence

The first arm, seed-0 FP32 learner, resumed its registered warm-u128 checkpoint
with all saved fields equal after the declared platform/config migration.
The CUDA worker reports NVIDIA L4, PyTorch `2.14.0+cu130`, CUDA 13.0 and Python
3.11.17. Actual linear outputs during collection/bootstrap and learning were
FP32, with autocast disabled in both scopes.

The locally downloaded u132 checkpoint and its exact metric/game-log prefix
passed the independent snapshot validator:

| Quantity | New work since u128 |
|---|---:|
| Updates | 4 |
| Collected training rows | 65,536 |
| Accepted / evaluated minibatches | 64 / 64 |
| KL early stops | 0 |
| Completed games | 461 |
| Truncations / engine errors | 0 / 0 |
| Median learner / collection time per update | 7.685 / 25.76 seconds |

These four observations include startup effects and are not the speed endpoint.
BF16 is scheduled after this first 32-update FP32 arm. The complete protocol
contains six arms and 6,912 fixed-panel evaluation games; no automatic promotion
is enabled. Checkpoint SHA256:
`b8703a4012a57394eeeba2b1e92deb53eff01a877ae5f2e9cf54cbc309e0cd26`.

## Observer startup failure and recovery

The first behavior observer exited before consuming cells because it expected
`pipeline-status.json`, while the new supervisor writes `watch/latest.json`.
Its exception is retained as `behavior/startup-failure.json`. A relative symlink
now connects these interfaces; `behavior/startup-recovery.json` records the fix.
Only the failed observer was restarted. No training/evaluation input changed,
no scored game was replaced, and no frozen Python/native source was edited.

Recovery was checked against fresh output: two completed cells were visible,
13 cold replays had completed, two were running, and no observer health alert
was reported at that check. Ash/self-damage signals remain diagnostic observations,
not automatic mistakes or population-rate estimates.

## Services and evidence

All service names below end in `-20261008.service`:

| Prefix | Invocation ID |
|---|---|
| `ygorl-learner-precision` | `5ef1f40f6ba74fa2b1e3ac31f40e2b7c` |
| `ygorl-learner-precision-eval` | `2ae66199e12448db83992f2251c9c305` |
| `ygorl-learner-precision-watch` | `14747ea266f54b5a92d059f2b86e295c` |
| `ygorl-learner-precision-behavior` (restarted) | `5f25306620ef4c07ad134f3834dd6641` |

Evidence root: `out/research/learner-precision-pilot-2026-10-08/`.
`first-verified-progress.json` preserves the validation, restore and dtype proof;
`launch.json`, `identity.json`, `registration.json`, `package.json` and
`preflight.json` bind source and inputs. The initial `services.json` predates
the observer restart; the invocation above and recovery record supersede its
observer entry. Registration SHA256:
`392fa3e82c2bb1737bd59917181730ae4fa81274f96940468469dd6b113a41e8`.

[Live operational comment](https://github.com/n0tevi1/ygorl/issues/60#issuecomment-6071330913)
and [strength/recovery tracking](https://github.com/n0tevi1/ygorl/issues/83#issuecomment-6071344395)
are separate from the frozen formal self-play study's issue blocks. The latter
study continues under its original source and configuration.
