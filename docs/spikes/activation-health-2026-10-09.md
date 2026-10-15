# Activation health repair and bounded recovery

The stopped October 8 critic continuation remains immutable. Its committed metrics
omit one rejected rollout at seed 2 / cold / update 294; the Lua failure was fatal
before the optimizer. The first differing menu under the correction is decision
410, where the invalid Verte copy activation is removed. Its preceding 410 actions,
responses, and actor observations are identical. No cost is paid by the corrected menu.

Chaos Angel's immunity incorrectly borrowed chain 0's player/type while evaluating
prospective effects. The pinned override uses the queried effect's prospective
context unless chain 0 actually refers to it. Legal activation and subsequent loss
of usable materials are separate cases: the regression injects immunity at the
resolution boundary and verifies an empty fusion operation resolves without error;
its paired unchanged-state case successfully summons a Fusion Monster.

The Python host also used to continue after logging Lua errors. It now stops at the
same boundary as the native host, with matching responses and terminal fields.

## Recovery protocol

First run a new, bounded **16-update health canary** from the retained seed-2 cold
`health-stop.pt` (u294). Preserve learner, optimizer, pool, schedule and RNG state;
active duels are not checkpointed and are reset on resume. Keep FP32 and all training
hyperparameters. Bind source checkpoint, implementation, patched scripts, protocol,
and failed-study STOP hashes before launch. Save a health-stop checkpoint and stop
before any optimizer step fed by an unhealthy rollout. Retain separate error logs.
Stop if cumulative new-game truncations exceed 1% after 100 games. Checkpoint every
update so a process interruption loses at most one committed update.

This canary is recovery validation, not a continuation of the old preregistered
comparison and not evidence of playing-strength improvement. Acceptance requires
16 committed updates, finite metrics, no health errors, and no truncation-limit
breach. After acceptance, register the next paired fixed-environment experiment;
do not combine old and repaired training endpoints as one unchanged study.

## Response monitor

`tools/study_watch.py` reads immutable study identity, STOP, raw failing rollouts,
service state, heartbeat and progress. A stopped service remains an open incident.
The monitor delivers a clearly labeled automated message to this existing T3
thread, preserving model selection and the `auto` runtime mode. Dispatch command
IDs survive retries; delivery is independently verified by reading the message
back. It retries ambiguous failures and reminds after 30 minutes without an
acknowledgment. Delivery, acknowledgment and completion are distinct records.
A user service with `Restart=on-failure` supervises the monitor; completion exits
only after a bound healthy report and acknowledgment. `Restart=always` is invalid
for this lifecycle: it relaunches acknowledged, successfully completed monitors.
Local T3 tokens live only in memory and expire after 15 minutes. No email or other recipient is used.

An end-to-end stopped-study message was dispatched and observed in this conversation
on October 9, then explicitly acknowledged. Evidence is under
`out/research/verte-health-stop-2026-10-09/response-monitor/`.

### Monitor lifecycle recovery (2026-10-15)

The October 15 alert burst came from a systemd query disconnect after repeated
successful watcher exits. Inventory found 196 watch units, 192 configured with
`Restart=always`, and 5,362,399 cumulative restarts. The repeated restarts are
confirmed; the precise cause of the message-bus disconnect is not established.
Live identity/report/health checks passed for 184 study watchers; these were
stopped and their current incidents acknowledged. Eleven failed/unhealthy study
watchers were retained, including their STOP and rollout evidence. One different
watcher without a `--monitor` argument was outside this retirement procedure.
This does not represent new experiment runs or new scientific results.

Launch future study monitors through the checked-in launcher:

```sh
python3 tools/study_watch.py --monitor /absolute/study/monitor \
  --launch-unit ygorl-study-name-watch.service
```

The launcher uses the current interpreter and this script, `Type=notify`,
`NotifyAccess=main`, a 120-second watchdog and `Restart=on-failure` with a
10-second retry delay. A normal acknowledged completion exits successfully and
stays stopped; crashes/watchdog failures can restart. It never starts the study
itself. Existing transient units must be explicitly retired or replaced; changing
this source file does not change their systemd properties.

Acknowledgment accepts an exact current or archived incident ID. An old warning
can therefore be acknowledged after a recovery notification without overwriting
or acknowledging the newer incident. Neither acknowledgment nor service retirement
clears STOPs or changes scientific reports. Evidence and retirement checks:
`out/maintenance/monitor-restart-storm-2026-10-15/`.

Recovery also acknowledged 83 archived inspection alerts by exact ID after their
current recovery events had been verified. A live launcher canary against an
already completed and acknowledged study exited and its transient unit was
collected; its journal showed no repeat start after the restart interval. The
post-collection default service properties are not the original launch settings.
