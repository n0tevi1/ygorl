# Precision controller recovery: operational amendment

The first apparent VM preemption recovered successfully from u156. The replacement
then stopped responding to a polling RPC while it remained listed on the server.
The controller failed closed and attempted name-based cleanup. The native worker
and model did not report a failure. The RPC's underlying cause remains unknown.

Further audit found that the CLI dropped the first local session name while its
endpoint remained assigned remotely under `[?]`. Thus the earlier "confirmed
preemption" interpretation was too strong. Name-based stop returns success for
a missing name without unassigning the endpoint. CLI session synchronization can
also turn an authentication failure into an empty list. Use the installed client's
direct assignment API and propagate failures. Prove ownership from session-created
history before unassigning the exact endpoint, then verify server absence. Install
an independent endpoint-based expiry timer as well. Do not modify account files or
touch unrelated machines; refuse new allocation while any assignment is present.
The new direct-API check during this investigation returned no active endpoints;
this observation does not retroactively prove that the earlier cleanup succeeded.

Implement a separate local recovery controller, importing the original frozen
controller and continuing to upload its original package. Do not change training,
evaluation, native code, study identity, registration, completed cells or seeds.
An explicit audit file binds the original STOP, last transport failure, study
identity and recovery implementation hash. No generic STOP bypass is allowed.

Under the existing exclusive controller lock, validate all saved job identities,
completed endpoints, newest durable prefixes and complete evaluation artifacts.
Reject partial evaluation cells. Confirm prior controller/evaluator services are
inactive and all previously owned Colab sessions absent. Archive the original
STOP only after admission and preserve it permanently. Use new service names and
record a recovery manifest before resuming. No changes to the frozen imports are
permitted; evaluator/behavior/supervisor run from their original worktree.

Retain the original absolute deadline and starting compute balance, and count all
previous allocations, including ambiguous attempted allocations. Allocate only
unused session names; choose generation and log serials above all existing
artifacts. Completed jobs are skipped only after verification; a verified final
snapshot without a completion marker can be finalized locally without training.
Resume unfinished jobs from their latest verified full checkpoint and RNG.

Transport recovery remains distinct from worker failure. On an unresponsive RPC,
try one short read-only diagnostic to preserve status/logs if possible. Confirm
termination of the owned instance before replacing it; a server listing alone is
not proof a worker is healthy. A diagnostic worker failure halts the experiment.
If no diagnostic is obtainable, record the uncertainty explicitly, recover only
verified work, and charge every replacement against the original budgets.
Never retry a reported worker/health error as infrastructure trouble. Preserve
the terminal controller failure after cleanup instead of hiding it behind a
session-release status. The original six-hour/12-unit/eight-allocation limits and
incomplete-evidence interpretation remain in force.

Validation covers unchanged budgets, monotonically increasing identifiers,
completed-job reuse, rejected checkpoint/cell tampering, STOP admission,
confirmed termination before replacement and preserved failure status. Then run
full presubmit, commit the amendment, and verify real resumed optimizer work and
local checkpoint receipt. Do not claim the remaining arms complete at launch.
