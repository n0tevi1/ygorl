"""In-action decisions of the Solo player re-anchored on the current board (design 10 「识别与执行的系统化」).

The shadow duel (:mod:`ygorl.selfplay.observe`) answers MD's questions inside an action as long as it stays in
lockstep with MD; when it does not (a desync, no shadow at all in the opponent's turn), the questions below are
answerable from the board MD shows *now*: each rebuilds a fresh reconstruction from the current snapshot, brings the
core to a decision point that asks the same question, and lets the policy score it, averaged over ``samples``
completions of the hidden cards (as :func:`ygorl.advisor.advise.advise` does for the command points).

- :func:`attack_target`: the battle command's ``attack`` with the attacker MD is asked about, then the core's own
  target selection (``SelectCard`` over the opponent's monsters). Exact up to the hidden cards.
- :func:`summon_position`: a one-shot effect registered by the board script Special Summons the card MD asks the
  position for, so the core asks its own ``SelectPosition``. Approximation: the board before the Summon, without
  the effect that summons it (its cost and the rest of its resolution are not modelled).
- :func:`respond`: MD's "activate in response?" prompt. ygopro-core cannot start mid-chain and the reconstruction
  starts in the advisor's own Main Phase 1, so the core is driven by the host to the same turn player and phase:
  the advisor ends its turn (for an opponent-turn prompt), the opponent's turn is played by a fixed script (go to
  the Battle Phase, attack once with its first attacker, Main Phase 2, End Phase), and the core offers the
  advisor its open-timing chain windows on the way. The window of MD's phase offering the most of the advisor's
  activations (only the card MD offers, when memory lists it) is scored: activate vs pass. Approximation: the
  event that opened MD's window (the opponent's activation or Summon, the chain so far) is not reproduced; the
  window is the open timing of the same phase.

Choices are recorded with what they averaged (:meth:`Choice.record`); errors are notes, never raised to the player.
The effect choice of a card with several effects is not here: the policy's activate action names its tab
(:func:`ygorl.selfplay.observe.option_of_description`).
"""

from __future__ import annotations

import random
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from ygorl.advisor.advise import GLOBAL_PHASE, GLOBAL_TURN, GLOBAL_WENT_FIRST
from ygorl.advisor.reconstruct import DECLINE, CompletedBoard, OpponentPrior, build, complete
from ygorl.advisor.snapshot import BoardSnapshot
from ygorl.cards.ydk import Deck
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import advance
from ygorl.engine.tracker import DecisionPoint
from ygorl.selfplay.observe import POSITION_BUTTONS
from ygorl.selfplay.shadow import ShadowDuel

MD_PHASES = ("draw", "standby", "main1", "battle", "main2", "end")  # = ygorl.md.memory.PHASES
BATTLE_STEPS = C.PHASE_BATTLE_START | C.PHASE_BATTLE_STEP | C.PHASE_DAMAGE | C.PHASE_DAMAGE_CAL | C.PHASE_BATTLE
PHASE_BITS = {"draw": C.PHASE_DRAW, "standby": C.PHASE_STANDBY, "main1": C.PHASE_MAIN1, "battle": BATTLE_STEPS,
              "main2": C.PHASE_MAIN2, "end": C.PHASE_END}  # fmt: skip
OWN_TURN_PHASES = ("battle", "end")  # phases of the advisor's own turn the respond rebuild can reach
WINDOWS = (M.SelectChain, M.SelectEffectYn)  # open timings: activate in response, or an optional trigger
MAX_HOST_STEPS = 600  # decisions the host answers before giving up on reaching MD's window
SUMMON_SCRIPT = """
local ygorl_tc=Debug.AddCard({password},0,0,{location},0,{position})
local ygorl_e=Effect.CreateEffect(ygorl_tc)
ygorl_e:SetType(EFFECT_TYPE_FIELD+EFFECT_TYPE_CONTINUOUS)
ygorl_e:SetRange({location})
ygorl_e:SetCode(EVENT_PHASE_START+PHASE_MAIN1)
ygorl_e:SetOperation(function(e) e:Reset() Duel.SpecialSummon(ygorl_tc,0,0,0,true,true,POS_FACEUP) end)
ygorl_tc:RegisterEffect(ygorl_e)
"""  # the monster's own effect: summon locks that only let a card's own effects through (Ra's Disciple) allow it


@dataclass(frozen=True)
class Clock:
    """The real game's turn, first player and phase (a rebuild starts at turn 1), restored in the observation."""

    turn: int
    went_first: bool
    phase: int  # core PHASE_* bit


Scorer = Callable[[object, DecisionPoint, Clock], np.ndarray]  # (core, point, clock) -> one logit per action


def policy_scorer(policy, cards, starting_lp: int = 8000) -> Scorer:
    """The network's logits on a rebuilt decision point (``policy``: :func:`ygorl.train.checkpoint.load_actor`)."""

    def score(core, point: DecisionPoint, clock: Clock) -> np.ndarray:
        import torch

        from ygorl.env.observer import PointObserver
        from ygorl.nets import collate

        obs = PointObserver(cards, policy.vocab, policy.event_length, starting_lp=starting_lp).encode(point, core)
        g = obs["globals"]
        g[GLOBAL_WENT_FIRST] = int(clock.went_first)
        g[GLOBAL_TURN] = min(clock.turn, 999)
        g[GLOBAL_PHASE] = (clock.phase & -clock.phase).bit_length()
        with torch.no_grad():
            return policy.net(collate([obs])).logits[0].float().numpy()

    return score


def _probs(scorer: Scorer | None, core, point: DecisionPoint, clock: Clock, indices) -> dict[int, float]:
    """Softmax of the policy over ``indices`` (uniform without a scorer)."""
    indices = list(indices)
    if scorer is None:
        return {i: 1.0 / len(indices) for i in indices}
    logits = np.asarray(scorer(core, point, clock), dtype=np.float64)
    z = np.array([logits[i] if i < logits.shape[0] else -np.inf for i in indices])
    if not np.isfinite(z).any():
        return {i: 1.0 / len(indices) for i in indices}
    e = np.exp(z - z[np.isfinite(z)].max())
    e /= e.sum()
    return dict(zip(indices, e.tolist(), strict=True))


@dataclass
class Choice:
    """A rebuilt decision: the option picked and its probability averaged over the samples that reached it."""

    question: str  # "target" | "position" | "respond"
    pick: object = None  # the option (target: opponent Monster Zone sequence or "direct"; position: button index;
    # respond: 0 activate / 1 pass); None: no sample reached the question
    probs: dict = field(default_factory=dict)
    reached: int = 0
    samples: int = 0
    notes: list[str] = field(default_factory=list)
    shadow: ShadowDuel | None = None  # a rebuild stepped past the choice, to continue the lockstep with
    detail: dict = field(default_factory=dict)

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def finish(self, sums: Mapping, prefer=None) -> Choice:
        if self.reached:
            self.probs = {k: v / self.reached for k, v in sums.items()}
            if self.probs:
                self.pick = max(self.probs, key=lambda k: (self.probs[k], k == prefer))
        return self

    def record(self) -> dict:
        return {"question": self.question, "pick": self.pick, "reached": self.reached, "samples": self.samples,
                "probs": {str(k): round(float(v), 4) for k, v in self.probs.items()}, "notes": self.notes[:4],
                **self.detail}  # fmt: skip


# --- the attack target -------------------------------------------------------------------------------------------


def attack_target(snapshot: BoardSnapshot, own_deck: Deck, prior: OpponentPrior, rule_flags: int, attacker: int, *,
                  cards, scorer: Scorer | None = None, samples: int = 4, seed: int = 0, scripts=None) -> Choice:  # fmt: skip
    """The policy's target for an attack by the advisor's monster in Monster Zone ``attacker`` (a Battle Phase
    snapshot). ``Choice.shadow``: the first sample's shadow stepped past the target, at MD's next question."""
    choice = Choice("target", samples=samples)
    rng = random.Random(seed)
    sums: dict = defaultdict(float)
    asked: list[tuple[ShadowDuel, dict[int, int]]] = []
    for _ in range(samples):
        try:
            sh = ShadowDuel.from_snapshot(snapshot, own_deck, prior, rule_flags, cards=cards, scripts=scripts,
                                          rng=rng.randrange(1 << 30), seed=rng.randrange(1, 1 << 62))  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - a sample the core cannot hold
            choice.note(f"rebuild: {type(exc).__name__}: {exc}")
            continue
        point = sh.point
        if point is None or not isinstance(point.decision, M.SelectBattleCmd):
            choice.note("the rebuild is not at the battle command")
            continue
        index = next((i for i, a in enumerate(point.actions) if a.kind == "attack" and a.card is not None
                      and a.card.loc.location == C.LOCATION_MZONE and a.card.loc.sequence == attacker), None)  # fmt: skip
        if index is None:
            choice.note(f"the rebuild offers no attack from Monster Zone {attacker}")
            continue
        clock = Clock(snapshot.turn, snapshot.went_first, point.phase or C.PHASE_BATTLE_START)
        auto = sh.step(index)
        point = sh.point
        targets = {}
        if point is not None and point.player == 0 and isinstance(point.decision, M.SelectCard):
            for i in point.choosable:
                a = point.actions[i]
                if a.kind == "select" and a.card is not None and a.card.loc.controller == 1:
                    targets.setdefault(a.card.loc.sequence, i)
        choice.reached += 1
        if targets:
            p = _probs(scorer, sh.rec.core, point, clock, targets.values())
            for seq, i in targets.items():
                sums[seq] += p[i]
            asked.append((sh, targets))
        else:  # one target (answered by the shadow) or a direct attack
            took = [s for s in auto if s.player == 0 and s.decision == "SelectCard" and s.action.card is not None]
            sums[took[-1].action.card.loc.sequence if took else "direct"] += 1.0
    choice.finish(sums)
    for sh, targets in asked:
        if choice.pick in targets:
            try:
                sh.step(targets[choice.pick])
                choice.shadow = sh
            except Exception as exc:  # noqa: BLE001
                choice.note(f"stepping the target: {type(exc).__name__}: {exc}")
            break
    return choice


# --- the position of a Summon --------------------------------------------------------------------------------------


def _take(board: CompletedBoard, password: int, cards: Mapping) -> int:
    """Remove one copy of ``password`` from the advisor's completed board (hand, Deck, Extra Deck, GY, banished, in
    that order) and return the location the summon script adds it to (the Extra Deck or the hand)."""
    mine = board.cards[0]
    for loc in (C.LOCATION_HAND, C.LOCATION_GRAVE, C.LOCATION_REMOVED):
        k = next((i for i, c in enumerate(mine) if c.location == loc and c.password == password), None)
        if k is not None:
            del mine[k]
            break
    else:
        for pile in (board.decks[0], board.extras[0]):
            if password in pile:
                pile.remove(password)
                break
    card = cards.get(password)
    return C.LOCATION_EXTRA if card is not None and card.is_extra_deck else C.LOCATION_HAND


def summon_position(snapshot: BoardSnapshot, own_deck: Deck, prior: OpponentPrior, rule_flags: int, password: int, *,
                    cards, scorer: Scorer | None = None, samples: int = 4, seed: int = 0, scripts=None,
                    clock: Clock | None = None) -> Choice:  # fmt: skip
    """The policy's position for the advisor's monster ``password`` being Special Summoned: the button of MD's
    position dialog (:data:`ygorl.selfplay.observe.POSITION_BUTTONS`: 0 face-up Attack, 1 face-up Defense)."""
    choice = Choice("position", samples=samples, detail={"card": password})
    rng = random.Random(seed)
    clock = clock or Clock(snapshot.turn, snapshot.went_first, PHASE_BITS.get(snapshot.phase, C.PHASE_MAIN1))
    sums: dict = defaultdict(float)

    def asks(point: DecisionPoint) -> bool:
        return point.player == 0 and isinstance(point.decision, M.SelectPosition) and point.decision.code == password

    for _ in range(samples):
        try:
            board = complete(snapshot, own_deck, prior, cards, rng)
            where = _take(board, password, cards)
            script = SUMMON_SCRIPT.format(password=password, location=where, position=C.POS_FACEDOWN_DEFENSE)
            rec = build(board, rule_flags, cards=cards, scripts=scripts, seed=rng.randrange(1, 1 << 62),
                        extra_script=script, until=asks)  # fmt: skip
        except Exception as exc:  # noqa: BLE001
            choice.note(f"rebuild: {type(exc).__name__}: {exc}")
            continue
        point = rec.point
        if not asks(point):
            choice.note("the core asked no position (no free zone, or a fixed position)")
            continue
        buttons = {b: i for i in point.choosable for b, pos in enumerate(POSITION_BUTTONS)
                   if point.actions[i].kind == "position" and point.actions[i].value == pos}  # fmt: skip
        if not buttons:
            choice.note("the core offers neither face-up position")
            continue
        choice.reached += 1
        p = _probs(scorer, rec.core, point, clock, buttons.values())
        for b, i in buttons.items():
            sums[b] += p[i]
    return choice.finish(sums, prefer=0)


# --- activate in response? ---------------------------------------------------------------------------------------


def _host(point: DecisionPoint, theirs: bool, phase: str, attacks: int) -> int:
    """The host's answer on the way to MD's window (see the module docstring)."""
    acts = point.actions
    kinds = [acts[i].kind for i in point.choosable]

    def first(*names) -> int | None:
        return next((i for i in point.choosable if acts[i].kind in names), None)

    decision = point.decision
    if isinstance(decision, M.SelectIdleCmd):
        if point.player == 0 and not theirs and phase == "battle" and "battle_phase" in kinds:
            return first("battle_phase")
        if point.player == 1 and point.phase == C.PHASE_MAIN1 and "battle_phase" in kinds:
            return first("battle_phase")  # the opponent goes through its Battle Phase: the most open timings
        return first("end_phase", "main2") if first("end_phase", "main2") is not None else point.choosable[0]
    if isinstance(decision, M.SelectBattleCmd):
        if point.player == 1 and phase == "battle" and attacks < 1 and "attack" in kinds:
            return first("attack")
        if point.player == 1 and phase == "main2" and "main2" in kinds:
            return first("main2")
        got = first("end_phase", "main2")
        return got if got is not None else point.choosable[0]
    if point.player == 1 and not isinstance(decision, (*WINDOWS, M.SelectYesNo)):
        got = next((i for i in point.choosable if acts[i].kind not in DECLINE), None)  # e.g. its attack target
        return got if got is not None else point.choosable[0]
    got = first(*DECLINE)
    return got if got is not None else point.choosable[0]


def respond(snapshot: BoardSnapshot, own_deck: Deck, prior: OpponentPrior, rule_flags: int, *, theirs: bool,
            phase: str, clock: Clock, offered: tuple[int, ...] = (), cards, scorer: Scorer | None = None,
            samples: int = 4, seed: int = 0, scripts=None) -> Choice:  # fmt: skip
    """Activate (pick 0) or pass (1) at MD's prompt in ``phase`` of the opponent's turn (``theirs``) or of the
    advisor's own; ``snapshot``: the advisor's turn before it (:func:`situation`); ``offered``: passwords of the
    cards MD offers, when memory lists them (else any activation of the advisor's counts)."""
    choice = Choice("respond", samples=samples, detail={"phase": phase, "theirs": theirs})
    if phase not in PHASE_BITS or (not theirs and phase not in OWN_TURN_PHASES):
        choice.note(f"no rebuild reaches the advisor's own {phase} window (a prompt inside an action)")
        return choice
    rng = random.Random(seed)
    mask = PHASE_BITS[phase]
    turn_player = 1 if theirs else 0
    sums: dict = defaultdict(float)
    cards_seen: dict[int, float] = defaultdict(float)
    for _ in range(samples):
        try:
            board = complete(snapshot, own_deck, prior, cards, rng)
            rec = build(board, rule_flags, cards=cards, scripts=scripts, seed=rng.randrange(1, 1 << 62))
        except Exception as exc:  # noqa: BLE001
            choice.note(f"rebuild: {type(exc).__name__}: {exc}")
            continue
        tracker, core = rec.tracker, rec.core
        best = None  # (activations offered, p activate, p pass, card probabilities)
        attacks, started = 0, False
        try:
            for _ in range(MAX_HOST_STEPS):
                point = None if tracker.done else tracker.point()
                if point is None:
                    break
                if point.turn_player == turn_player:
                    started = True
                elif started:
                    break  # the turn of MD's prompt is over
                here = started and bool(point.phase & mask)
                if started and not here and point.phase > mask:
                    break  # past MD's phase
                if here and point.player == 0 and isinstance(point.decision, WINDOWS):
                    acts = [i for i in point.choosable if point.actions[i].kind not in DECLINE]
                    if offered:
                        acts = [i for i in acts if point.actions[i].card is not None
                                and point.actions[i].card.code in offered]  # fmt: skip
                    if acts and (best is None or len(acts) > best[0]):
                        p = _probs(scorer, core, point, Clock(clock.turn, clock.went_first, point.phase),
                                   point.choosable)  # fmt: skip
                        by_card: dict[int, float] = defaultdict(float)
                        for i in acts:
                            card = point.actions[i].card
                            by_card[card.code if card is not None else 0] += p[i]
                        p_act = sum(p[i] for i in acts)
                        p_pass = sum(p[i] for i in point.choosable if point.actions[i].kind in DECLINE)
                        best = (len(acts), p_act, p_pass, by_card)
                index = _host(point, theirs, phase, attacks)
                attacks += point.player == 1 and point.actions[index].kind == "attack"
                response = tracker.act(index)
                if response is not None and not tracker.done:
                    advance(core, tracker, response)
        except Exception as exc:  # noqa: BLE001
            choice.note(f"host: {type(exc).__name__}: {exc}")
        if best is None:
            choice.note(f"no window of the advisor's in the {'opponent' if theirs else 'own'} {phase} offered an "
                        f"activation{' of the offered card' if offered else ''}")  # fmt: skip
            continue
        choice.reached += 1
        _, p_act, p_pass, by_card = best
        total = p_act + p_pass or 1.0
        sums[0] += p_act / total
        sums[1] += p_pass / total
        for code, q in by_card.items():
            cards_seen[code] += q / total
    choice.finish(sums, prefer=1)
    if choice.reached and cards_seen:
        choice.detail["card"] = max(cards_seen, key=cards_seen.get)
    return choice


# --- the board MD shows now, for any phase -----------------------------------------------------------------------


@dataclass(frozen=True)
class Situation:
    """The current board as a reconstruction snapshot, with what the snapshot cannot say (whose turn, which phase)."""

    snapshot: BoardSnapshot  # the advisor's own turn: the current one, or the one before the opponent's
    phase: str  # MD's phase (MD_PHASES)
    theirs: bool  # the opponent's turn
    clock: Clock  # the real turn, first player and phase


def situation(raw, passwords: Mapping[int, int], attacked: frozenset[int] = frozenset(),
              stats: Mapping[int, tuple[int, int]] | None = None) -> Situation:  # fmt: skip
    """The board of ``raw`` (a :class:`ygorl.md.memory.RawDuel`) in any turn and phase. The reconstruction only
    starts in the advisor's own Main / Battle Phase, so outside them the snapshot is read as the advisor's Main Phase
    1; in the opponent's turn it is the advisor's turn before (the opponent's hand one card smaller and its Deck one
    larger: the rebuild's own draw gives it back). ``stats``: the face-up monsters' ATK / DEF MD shows
    (:func:`ygorl.md.memory.to_snapshot`)."""
    from ygorl.md.memory import to_snapshot

    phase = MD_PHASES[raw.phase] if 0 <= raw.phase < len(MD_PHASES) else "main1"
    theirs = raw.turn_player != raw.me
    base = phase if not theirs and phase in ("main1", "battle", "main2") else "main1"
    snap = to_snapshot(replace(raw, turn_player=raw.me, phase=MD_PHASES.index(base)), passwords, attacked,
                       stats=stats)  # fmt: skip
    turn, went_first = snap.turn, snap.went_first != theirs  # to_snapshot read the turn as the advisor's
    if theirs:
        op = snap.sides[1]
        hand = next((k for k, c in enumerate(op.cards) if c.location == "hand" and not c.known), None)
        if hand is not None:
            op = replace(op, cards=op.cards[:hand] + op.cards[hand + 1 :], deck_count=op.deck_count + 1)
        before = max(1, turn - 1)
        snap = replace(snap, turn=before, went_first=went_first if before > 1 else True, normal_summoned=False,
                       sides=(snap.sides[0], op)).check()  # fmt: skip
    return Situation(snap, phase, theirs, Clock(turn, went_first, PHASE_BITS.get(phase, C.PHASE_MAIN1)))


def in_limbo(before, now, passwords: Mapping[int, int], cards: Mapping) -> set[int]:
    """Passwords of the advisor's monsters that left their pile (hand, Deck, Extra Deck, GY, banished) since
    ``before`` and are not on the field ``now`` (both :class:`ygorl.md.memory.RawDuel`): a monster being summoned."""
    from collections import Counter

    def own(raw, field: bool) -> Counter:
        pl = raw.players[raw.me]
        out = Counter(c.cid for pile in pl.piles.values() for c in pile)
        if field:
            out.update(c.cid for c in pl.slots if c is not None)
        return out

    gone = own(before, False) - own(now, False)
    gone -= Counter(c.cid for c in now.players[now.me].slots if c is not None) - Counter(
        c.cid for c in before.players[before.me].slots if c is not None)  # fmt: skip
    out = set()
    for cid in gone:
        card = cards.get(passwords.get(cid, 0))
        if card is not None and card.is_monster:
            out.add(passwords[cid])
    return out


def summoned_card(req, action, passwords: Mapping[int, int], shadow: ShadowDuel | None = None, *, before=None,
                  now=None, cards: Mapping | None = None, listed: tuple[int, ...] = ()) -> int:  # fmt: skip
    """The password of the monster MD asks the position for (0: unknown): the card MD's dialog names (a mix entry of
    kind 5), the shadow's own position question, the card of the policy's Special Summon, the one monster of the
    advisor's that left its pile since the policy's action (``before``) without reaching the field ``now``, or the
    one monster MD listed in this action (``listed``: MD card ids of the last list, e.g. the Disciples to summon)."""
    for kind, value in getattr(req, "params", ()) or ():
        if kind == 5 and value in passwords:
            return passwords[value]
    point = shadow.point if shadow is not None else None
    if point is not None and isinstance(point.decision, M.SelectPosition):
        return point.decision.code
    if action is not None and action.kind == "spsummon" and action.card is not None:
        return action.card.code
    if before is not None and now is not None and cards is not None:
        limbo = in_limbo(before, now, passwords, cards)
        if len(limbo) == 1:
            return next(iter(limbo))
    names = {passwords.get(cid, 0) for cid in listed}
    if len(names) == 1 and cards is not None:
        (password,) = names
        card = cards.get(password)
        if card is not None and card.is_monster:
            return password
    return 0


class Decider:
    """The rebuilt decisions for the Solo player (:mod:`ygorl.selfplay.loop`): MD memory in, a :class:`Choice` out."""

    def __init__(self, rule_flags: int, cards, passwords: Mapping[int, int], prior: OpponentPrior, *,
                 scorer: Scorer | None = None, samples: int = 4, scripts=None,
                 stats: Callable[[object], Mapping[int, tuple[int, int]]] | None = None) -> None:  # fmt: skip
        """``stats``: raw -> uid -> (ATK, DEF) MD shows for the face-up monsters (the client's views)."""
        self.rule_flags, self.cards, self.passwords, self.prior = rule_flags, cards, passwords, prior
        self.scorer, self.samples, self.scripts = scorer, samples, scripts
        self.stats = stats

    def _situation(self, raw, attacked: frozenset[int] = frozenset()) -> Situation:
        return situation(raw, self.passwords, attacked, self.stats(raw) if self.stats is not None else None)

    def _deck(self, raw) -> Deck:
        from ygorl.md.memory import own_deck

        return own_deck(raw, self.passwords, self.cards)

    def _run(self, question: str, fn) -> Choice:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - a decision the rebuild cannot make: the caller's rule decides
            return Choice(question, notes=[f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=2)])

    def attack_target(self, raw, attacker: int, attacked: frozenset[int] = frozenset()) -> Choice:
        def run():
            sit = self._situation(raw, attacked)
            return attack_target(sit.snapshot, self._deck(raw), self.prior, self.rule_flags, attacker, cards=self.cards,
                                 scorer=self.scorer, samples=self.samples, seed=random.randrange(1 << 30),
                                 scripts=self.scripts)  # fmt: skip

        return self._run("target", run)

    def position(self, raw, req=None, action=None, shadow: ShadowDuel | None = None, before=None,
                 listed: tuple[int, ...] = ()) -> Choice:  # fmt: skip
        """``action``, ``before``, ``listed``: the policy's last action, the board it was taken on and the last list
        MD showed in it (to tell the monster, :func:`summoned_card`)."""

        def run():
            password = summoned_card(req, action, self.passwords, shadow, before=before, now=raw, cards=self.cards,
                                     listed=listed)  # fmt: skip
            if not password:
                return Choice("position", notes=["the summoned monster is not known"])
            sit = self._situation(raw)
            return summon_position(sit.snapshot, self._deck(raw), self.prior, self.rule_flags, password,
                                   cards=self.cards, scorer=self.scorer, samples=self.samples,
                                   seed=random.randrange(1 << 30), scripts=self.scripts, clock=sit.clock)  # fmt: skip

        return self._run("position", run)

    def respond(self, raw, req=None, offered: tuple[int, ...] | None = None) -> Choice:
        """``offered``: passwords of the cards MD's prompt offers (the client's list); else the request's."""

        def run():
            sit = self._situation(raw)
            nonlocal offered
            if offered is None:
                offered = tuple(self.passwords[c.cid] for c in getattr(req, "candidates", ()) or ()
                                if c.cid in self.passwords)  # fmt: skip
            return respond(sit.snapshot, self._deck(raw), self.prior, self.rule_flags, theirs=sit.theirs,
                           phase=sit.phase, clock=sit.clock, offered=offered, cards=self.cards, scorer=self.scorer,
                           samples=self.samples, seed=random.randrange(1 << 30), scripts=self.scripts)  # fmt: skip

        return self._run("respond", run)


__all__ = ["Choice", "Clock", "Decider", "Situation", "attack_target", "policy_scorer", "respond", "situation",
           "summon_position", "summoned_card"]  # fmt: skip
