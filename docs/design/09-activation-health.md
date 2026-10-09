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
