# Independent learner precision (2026-10-08)

## Implementation design

The PPO learner dominates the current training cost. A bounded fixed-rollout benchmark found lower update
latency with BF16, but it did not establish end-to-end speed or policy strength. Separate learner precision
from collection so a controlled training comparison can leave action sampling and bootstrap inference in FP32.

Add `TrainConfig.learner_precision` / `--learner-precision` with `inherit` (default), `fp32`, and `bf16`.
`inherit` uses the existing `bf16` setting, preserving historical configs and checkpoints. Explicit values only
override the complete `PPOLearner.update` autocast scope, including critic-only warmup, reference/prior inference,
and all optimizer minibatches. Collection, including bootstrap evaluation and overlap collection, continues to
use `bf16`. Parameters remain FP32. Learner autocast must retain `cache_enabled=False` because Adam changes the
weights within the scope. Precision is serialized in config/checkpoints and recorded by `bench_train.py`.

Validation must observe actual GPU forward dtypes during collection, bootstrap, and updates for inherited and
explicit settings (including the converse BF16 collection / FP32 learner); preserve per-Adam fresh-weight
checks; exercise config/CLI/checkpoint round trips and legacy checkpoints missing the field. This change does
not itself promote BF16 into the formal training recipe. Compare wall time and completed minibatches as well
as held-out strength/behavior before promotion; KL early stopping can otherwise make less work appear faster.
