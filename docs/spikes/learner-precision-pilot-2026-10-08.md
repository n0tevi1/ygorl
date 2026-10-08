# Learner-only BF16 pilot — preregistration, 2026-10-08

## Question and fixed intervention

The infrastructure audit found a possible BF16 learner speed benefit without
establishing learning quality. This pilot compares FP32 collection with either
FP32 or BF16 learner autocast. It never changes the frozen terminal-critic study.
The new `learner_precision` option defaults to `inherit` for checkpoint backwards
compatibility; only this pilot explicitly sets `fp32` or `bf16`. Keep autocast
weight caching disabled across Adam steps and the pre-step KL guard enabled.

Three paired starts use original `terminal-critic-policy-long-2026-10-06`
seed-0/1/2 **warm-u128** full checkpoints. Within each pair restore identical
actor, critic, reference, Adam, pool, CPU/collector RNG and schedule. On first
ROCm-to-CUDA migration, use CUDA seed `2026100820 + seed_id` for both arms; on
recovery preserve saved CUDA RNG. Native unfinished duels are not restored and
the migration is not claimed to reproduce an uninterrupted ROCm trajectory.

Train **32 new updates per arm**, u128→u160, 16,384 rows/update: 192 updates and
3,145,728 new rows total. Retain LR, minibatches, epochs, loss coefficients,
reference EMA, deck distribution and league behavior. Both arms use the same
latest source/native patches. Allow only these execution overrides: relocated
deck paths, `device=cuda`, `bf16=False`, the registered learner precision,
`log_games=True`, `register_every=0`, `register_matrix=None`, `eval_every=0`,
`checkpoint_every=0` (the worker owns verified snapshots). Record the final config;
no automatic hyperparameter or budget changes after observing results.

Run one L4 VM at a time to leave the current local GPU experiment running.
Counterbalance order: seed0 FP32/BF16, seed1 BF16/FP32, seed2 FP32/BF16. Limits:
six wall-clock hours, 12 compute units and eight allocations, whichever comes
first. Do not allocate a replacement until the previous owned session is confirmed
absent or stopped. A budget interruption is incomplete evidence, not a failed
learning arm or permission to select a different seed.

## Recovery and monitoring contract

Controller inputs: `registration.json` holds run_id, origin_update=128,
target_update=160, snapshot_every=4, lease_seconds=300, limits and ordered jobs
(`job_id`, `seed`, `learner_precision`, checkpoint path/SHA). `package.json` binds
source archive, per-file manifest and bootstrap hashes. Source includes the driver
files; no unregistered code edits after upload. Derive and bind each checkpoint's
environment stamp and deck hashes. Reject mismatched resumed precision/config.

Every four updates publish an immutable checkpoint plus exact metric prefix,
cumulative complete-game log and manifests. Download and validate hashes and
counter boundaries before promoting `durable/<job_id>/latest.json`. A separate
precision manifest binds the game log and study/job/source identity to the base
checkpoint manifest. Resume only the newest locally verified generation, fencing
older workers with a 300-second lease. Capture generation/recovery counts and
recomputed work; recovery cost belongs in wall-clock efficiency reporting.

Health errors or nonfinite metrics/parameters stop before further training; any
native error before an update prevents that update. Stop if cumulative new training
truncations exceed 1% after 100 completed games. Worker failure is investigated,
not treated as VM preemption. Preserve failed logs, partial downloads and traces.
Controller expiry and independent VM expiry bound orphan cost. Only owned sessions
may be released. Completed arms publish `completed/<job_id>.json` for the local
read-only evaluator. No probe model is automatically promoted.

An independent local supervisor checks controller/evaluator service state with a
startup grace period, writes a shared STOP on unexpected exit, and records progress
every 30 seconds. Completed services may exit normally. It maintains one issue
comment at ten-minute intervals or terminal transitions; a publication failure is
recorded and retried without aborting healthy training. Final efficiency reporting
separates accepted/evaluated minibatch counts and flags unequal work. Report only
measured setup/transfer/checkpoint/recovery costs; missing timing is not zero cost.

## Evaluation and interpretation

Use a new fixed evaluation panel, seed **2026100822**: 64 ordered deck/deal
clusters × four seat/deck variants = 256 games per opponent, on the same 20
`md-2026-09` legal decks. Opponents: Greedy, old BC256, and the immutable completed
seed0 cold-u512 checkpoint. All candidates share the exact panel; register the
opponent SHA256s before evaluation. Evaluate the three starting checkpoints and
six u160 endpoints in FP32: 6,912 games total. Any evaluation error/truncation
invalidates the affected cell and stops scoring; preserve the partial artifact.

Primary quality contrast: BF16 minus FP32 u160 equal-weight opponent win rate,
with paired differences and seed × deal-cluster bootstrap (20,000 replicates,
seed **2026100823**). Report each seed/opponent and each arm's change from its own
u128. This is a small pilot, not proof of long-run equivalence. A candidate for
larger validation needs a nominal 95% lower bound above −2 percentage points,
no seed losing more than five points, healthy runs and no unresolved worsening
behavior signal. Failure to resolve the interval is inconclusive, not equivalence.

Efficiency: actual collected rows/second and total elapsed time, learner/collection
times, accepted/evaluated minibatches, KL stops and recovery/setup/checkpoint cost
separately. A useful speed signal is at least 10% lower median learner time at
comparable accepted work; do not call earlier KL stopping an infrastructure win.

Behavior: turn/decision distributions, >20-turn tails, cancellation counts and
deck-outs; reuse fixed-index plus explicitly selected-tail cold replay audits for
loops/self-interference. These are diagnostic signals, not automatic error labels.
Keep selected tails separate from frequency denominators. Existing behavior
monitor can consume the same evaluation-cell schema, without publishing over the
formal study's issue block. No shortened-game reward or action mask is introduced.

## Launch gates

Commit design and implementation, pass configuration/CLI/checkpoint and real GPU
precision/cache regressions and the full applicable suite, then freeze a source
manifest and registration before allocation. Verify on the remote first worker
that collection runs FP32 and learner autocast matches the assigned arm; record
hardware and installed versions. Launch local evaluation/behavior supervision
against immutable completed artifacts. Update #60/#83 with actual service/session
identity, first real progress and recovery evidence; a launched process alone is
not evidence of completed optimizer work.

## Implementation and local validation

The four drivers are `tools/precision_colab.py`, `precision_worker.py`,
`precision_evaluate.py`, and `precision_watch.py`. The observer uses
`tools/behavior_watch.py` with a separate output root and without `--publish`,
so it does not replace the formal continuation study's issue block.
The precision supervisor maintains its own comment on #60; #83 tracks strength
and recovery interpretation. A controller process restart intentionally refuses
automatic reuse of an existing study identity; inspect the retained artifacts
before any manual recovery. Confirmed VM preemption within the live controller
uses verified checkpoints automatically.

Local presubmit passed **1,605 tests**, with three explicit skips (one unsupported
snapshot fixture and two opt-in network checks). The additional evaluator and
supervisor checks were developed while that suite ran; the final focused driver
and checkpoint run covers 36 tests. Real registered starting checkpoints also
passed a separate evaluator identity creation/resume smoke check without playing
or scoring evaluation games. Runtime evidence belongs under
`out/research/learner-precision-pilot-2026-10-08/`; source, registration, service
identities, hardware precision proof and verified generations remain distinct
artifacts.
