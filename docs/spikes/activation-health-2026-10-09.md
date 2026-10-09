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
A restarting user service supervises the monitor; completion exits only after a
bound healthy report and acknowledgment. Local T3 tokens live only in memory and
expire after 15 minutes. No email or other recipient is used.

An end-to-end stopped-study message was dispatched and observed in this conversation
on October 9, then explicitly acknowledged. Evidence is under
`out/research/verte-health-stop-2026-10-09/response-monitor/`.
