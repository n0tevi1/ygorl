# Repaired-environment critic continuation

The October 8 continuation stopped on a Lua legality error after 1,702 committed
updates. Its incomplete cold/warm comparison remains historical evidence. PR #247
corrects prospective fusion immunity and host error handling. The independent
recovery audit (`activation-health-canary-2026-10-09/independent-audit.json`) verifies
16 FP32 updates, 262,144 rows, 1,853 healthy wins, no errors/truncations, and the u310
checkpoint. Turns: mean 9.1133, median 9, P95 15, P99 18, maximum 31; seven games
exceed 20 turns. This is a health gate, not evidence of improved playing strength.

Run a separately registered comparison from **all six original u128 checkpoints**
(three paired seeds, cold/warm), through u256 and u512, under one fixed repaired
implementation. Keep FP32, learning configuration, optimizer/reference/pool state,
and restored RNG. Active duels reset on resume. Canonicalize only absent config
fields using current documented defaults, then verify all restored state exactly.
Do not mix completed old-environment arms with repaired arms as one unchanged study.

Retain the existing primary three fixed opponents, separate historical-RL opponent,
critic-retention checks, 64 paired deal clusters with four variants, and 20,000 crossed
bootstrap replicates. Use fresh evaluation seed **2026100904**. The 19,456 evaluation
games include the initial model and all u128/u256/u512 nodes. The 2,304 additional
training updates and advance criteria are unchanged; no BF16 or cancellation-budget
promotion is included. Bind the latest failed study/STOP and independent recovery
audit in the new identity. Repeat six-model GPU restore and independent statistical
and retention preflight; never copy an old preflight report.

Persist actual native response sequences plus complete game specs for every healthy
training game with >20 turns or >1,000 decisions, plus the deterministic seed-mod-64
sample. Label selection reasons; the enriched sample is not an unbiased population
estimate. These sequences replay the actual changing-policy game without pretending
one frozen endpoint reproduces it. Successful games do not retain explicit policy
step indices, so native responses are the replay authority. Error/truncation traces
remain separate and fatal-health checks still precede optimization. Audit selected
long games and normal samples for loops, self-obstruction and missed wins.

A completion report is published only after the existing independent study auditor
passes. A persistent response monitor binds the service invocation, checks raw
failures as well as committed progress, and dispatches into this thread for action.
Retain service metadata after exit and require successful termination before monitor
completion; an empty terminal InvocationID alone is not a restart. Include retention
progress in liveness checks. CPU evaluation can overlap GPU training; assess Colab
parallelism only with checkpoint and artifact-identity guarantees preserved.
