# Clown Crew tribute / target health incident (2026-10-10)

The broader policy-inference comparison stopped at job 1679 with a real Lua error,
not a cancellation-budget failure. The original study remains stopped and its
partial records are not a completed performance comparison.

Evidence: `out/research/policy-inference-broad-confirmation-2026-10-10/` (identity
`47b95caa92f63adb656c045b0893f830ceb7c2a785a68b08bfabc407d26fb589`), event
`2e80596ba8ef2992818bee4c41049666d943d21f4e8f301be637371f601306a0`.
Exact reproduction and follow-up artifacts: `out/research/clown-target-health-2026-10-10/`.

## Root cause

The seed-0 warm u512 candidate faced initial-128x2, pair 17 / game 71, in the
argmax-budget1 arm. The opponent activated the GY effect of Clown Crew Malabarisme
(83232904) at decision 391. At 392 it tributed the last field monster (70088809).
That monster's equip, Tails of the Fairy Tails (82119326), immediately left by game
rules. The cost filter excluded the tribute itself but incorrectly counted its
equip as a remaining return-to-hand target. Target selection was empty; resolution
called `tc:IsRelateToEffect(e)` on nil. The original 395 decisions reproduce the
same error, turn 6, LP 2850/4850. This happens during payment, before another effect
can respond; it is distinct from legal activation followed by target loss in a chain.
The candidate's earlier budget interventions were at 302 and 341, not this payment.

## Correction and scope

Add a SHA-pinned in-memory override for the reviewed `c83232904.lua` source
(`7fbfd6cee3ed1d90c78fd02c42d70bf47436176dc75ca793281ca71e1996b4fd`). The cost
predicate excludes cards equipped to the proposed tribute. Shared CardScripts,
formal training, checkpoint parameters, cancellation defaults and the native binary
are not edited. Unknown script revisions receive no override.

Recorded replay regression proves identical menus, engine responses and observation
bytes before decision 392. The fixed payment menu removes only 70088809 and retains
all three hand choices (82159583 twice, 91800273). All three choices finish healthy
with independently seeded random continuations. Python and native hosts reproduce
the same original error. A controlled operation-boundary intervention verifies both
normal return-to-hand and a legitimate target leaving before resolution. Board probes
cover either controller and the presence/absence of an independent field target;
they enumerate matching cards outside an active effect, while the full replay tests
real targeting eligibility. These are health/legality regressions, not strength results.

The next comparison must rerun the complete 4,096-game paired panel in an isolated
runtime derived from the original frozen runtime plus this correction. Keep the old
STOP and all partial results. Both arms use the same repaired engine; do not exclude
the failing deal or claim a policy benefit from the crash fix. Run the failed policy
case and all 32 cell/arm smoke games first, then restart with its own independent
monitor and immutable source/runtime/data/checkpoint identity. Existing formal
training retains its original runtime identity.
