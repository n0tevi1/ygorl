# Activation legality and health incidents

Activation checks and resolution are distinct. A legal activation can lose all usable
cards before resolution; that must resolve according to the card's rules, without
retroactively rejecting the activation or manufacturing a replacement target.

The October 9 failure instead happens inside one activation: Chaos Angel's immunity
predicate reads chain 0 even while checking a prospective Verte Anaconda effect.
Before the chain exists it allows immune Synchro Monsters as fusion materials; after
activation it excludes them, leaving the spell-selection group empty. Paying LP is
not the cause in the recorded failure. The official fusion ruling requires a usable
material combination at activation:
<https://www.db.yugioh-card.com/yugiohdb/faq_search.action?fid=23510&ope=5&request_locale=ja>.

Apply a content-pinned, in-memory correction to the reviewed Chaos Angel script.
Use immutable chain player/type only when that chain belongs to the queried effect;
otherwise use the prospective effect's activation type and player. Preserve actual
chain information when the handler changes controller or type. Do not patch Verte
with a silent empty-group return: that would hide an invalid cost payment and leave
the legality mismatch intact. Verify the recorded failure and valid fusion cases,
including a legal activation whose materials become unavailable before resolution.

Training health errors remain fatal and retain their failing replay/checkpoint.
The Python tracker, like the native host, must stop before decoding the message
buffer from a failed Lua call, rather than continuing with only a logged error.
Uncommitted failing rollouts must appear separately from committed metric counters.
A stopped study is an unresolved incident, not successful monitor completion.
Publication alone is not acknowledgment; monitoring must retain pending incidents
and verify delivery/acknowledgment separately. Repaired training receives a new
identity; previous STOP evidence and results are immutable.

A completed service can lose its `InvocationID` when systemd discards inactive
runtime metadata. An empty terminal ID is not evidence of a new invocation.
Completion requires a bound healthy report, no STOP or raw failures, and a clean
service exit (inactive/dead or active/exited, success, status 0). A nonempty changed
ID remains an incident even with a healthy report; missing identity while running
or before valid completion also remains an incident. Retain unit state after exit
for future runs where possible. Do not finalize merely because a report appeared
while the process is still running.

A payment can itself remove prospective targets. In the October 10 Clown Crew
Malabarisme incident, the tribute filter excluded the released monster but counted
its attached Equip Card; paying the cost removed both, before target selection.
The content-pinned correction excludes equips attached to that particular tribute
candidate. It preserves hand tributes and field tributes with an independent target.
It does not add a silent nil-target return to conceal the illegal cost, or reject
legal activations whose target subsequently leaves during a chain. Regression tests
cover the recorded Python/native failure, unchanged prefix observations, surviving
choices, and target loss at resolution. This is a scoped dependency correction,
not a general solver for every effect that can change target availability on payment.
Repaired evaluations also receive a fresh identity and rerun both treatment arms;
never silently splice repaired games into a stopped comparison.
