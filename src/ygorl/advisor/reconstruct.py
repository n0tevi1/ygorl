"""Put a :class:`BoardSnapshot` into the core so the policy can be asked about it (design 10, 「重建」).

Two steps:

- :func:`complete`: fill in what the advisor cannot see, one sample per call. The opponent's hidden cards (hand,
  Set cards, face-down banished cards, deck, Extra Deck) are drawn from one meta deck, chosen by share among the
  meta decks consistent with the opponent's public cards (the HDT filter of design 04); the advisor's own deck
  and Extra Deck are its deck list minus the cards already seen.
- :func:`build`: a core duel holding the completed board, run to the advisor's first idle decision. The cards are
  placed by a generated Lua script (``Debug.AddCard``: Xyz materials, face-up Pendulum cards in the Extra Deck and
  counters included), served next to the card scripts. The advisor is engine player 0 and the duel starts in its
  turn-1 Main Phase 1 with no draws; ``DUEL_ATTACK_FIRST_TURN`` lifts the turn-1 battle ban when the snapshot is
  later in the game. The real turn number and who went first are restored in the observation
  (:func:`ygorl.advisor.advise.observe`).

The turn so far (design 10「重建能还原什么」) is applied by the same script when the reconstruction's Main Phase 1
starts (the core clears per-turn state at the turn's start): summoned / Set / repositioned this turn as card statuses
(position changes, "Set this turn" Traps), monsters that attacked cannot attack, Normal Summons used, additional
Tribute Summons granted (as Ancient Chant grants them), the ATK / DEF the game shows for face-up monsters (a lasting
change by the difference to what the reconstruction computes: equips and other effects it cannot hold), and
once-per-turn limits spent: an effect used by a card on the field, the activation of a face-up Spell / Trap
activated this turn, the cards activated this turn that have left the field, and the activation of a Spell / Trap
whose lasting effect is on the player (``player_effects``) while no copy is face-up on the field (it was activated
this turn). A spent limit is applied only when it is unambiguous: the card has one once-per-turn limit there (or all
its limits share one count code); otherwise the core keeps offering it and the real game's refusal is the fallback.

Lost on the way (design 10「重建的已知失真」): the event history, which of several once-per-turn effects of one card
was used, other effects lasting the turn, effect states such as equips and targets (their ATK / DEF change is kept,
not its source), and anything mid-chain. Only the advisor's own Main Phase is supported.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ygorl import _core
from ygorl.advisor.snapshot import LOCATIONS, BoardSnapshot, SnapshotCard
from ygorl.cards.ydk import Deck
from ygorl.data.environment import PlayerRules
from ygorl.engine import constants as C
from ygorl.engine import messages as M
from ygorl.engine.duel import DecisionPoint, DuelConfig, DuelTracker, default_cards, default_scripts, expand_seed

BOARD_SCRIPT = "ygorl_advisor_board.lua"
MAX_PRE_DECISIONS = 32  # decisions answered by the host before the idle point (standby-phase triggers, ...)
DECLINE = ("pass", "no", "cancel", "finish")  # how the host answers them: decline when it can, else the first option


class ReconstructError(ValueError):
    """A snapshot the core cannot be put into the asked state."""


@dataclass(frozen=True)
class OpponentPrior:
    """Meta decks with their shares, the opponent's possible decks."""

    decks: tuple[Deck, ...]
    shares: tuple[float, ...]

    @classmethod
    def from_environment(cls, env) -> OpponentPrior:
        if not env.meta_decks:
            raise ReconstructError(f"environment {env.version} has no meta decks to sample the opponent from")
        return cls(tuple(m.deck for m in env.meta_decks), tuple(m.share for m in env.meta_decks))

    def consistent(self, seen: Counter[int]) -> list[int]:
        """Indices of the decks holding every card seen (with multiplicity), the HDT filter."""
        out = []
        for i, deck in enumerate(self.decks):
            counts = deck.counts()
            if all(counts[p] >= n for p, n in seen.items()):
                out.append(i)
        return out

    def pick(self, seen: Counter[int], rng: random.Random) -> tuple[int, bool]:
        """(deck index, consistent?) drawn by share among the consistent decks, or among all when none is."""
        pool = self.consistent(seen)
        ok = bool(pool)
        if not ok:
            pool = list(range(len(self.decks)))
        return rng.choices(pool, weights=[self.shares[i] for i in pool])[0], ok


@dataclass
class Placed:
    """One card of a completed board, in core terms."""

    password: int
    location: int
    sequence: int = 0
    position: int = C.POS_FACEDOWN_DEFENSE
    owner: int | None = None  # engine player; None = the controller
    materials: tuple[int, ...] = ()
    counters: tuple[tuple[int, int], ...] = ()
    status: int = 0  # core STATUS_* bits for this turn (summoned / Set / repositioned)
    attacked: bool = False  # declared an attack this turn
    spend: str = ""  # once-per-turn use to spend: "effect" (an effect where it is), "activation" (its activation)
    attack: int | None = None  # a face-up monster's ATK / DEF in the real game (None = its own values)
    defense: int | None = None


@dataclass
class CompletedBoard:
    """A snapshot with every hidden card filled in (one sample)."""

    snapshot: BoardSnapshot
    cards: tuple[list[Placed], list[Placed]] = field(default_factory=lambda: ([], []))
    decks: tuple[list[int], list[int]] = field(default_factory=lambda: ([], []))
    extras: tuple[list[int], list[int]] = field(default_factory=lambda: ([], []))
    opponent_deck: int = -1  # index into the prior's decks
    consistent: bool = True  # the opponent deck was consistent with the public cards
    notes: list[str] = field(default_factory=list)


def _draw(pool: Counter[int], rng: random.Random, accept: Callable[[int], bool] = lambda p: True,
          fallback: Sequence[int] = ()) -> int:  # fmt: skip
    """One card out of ``pool`` (removed from it), preferring cards that ``accept``; else from ``fallback``."""
    items = [p for p, n in pool.items() for _ in range(n)]
    good = [p for p in items if accept(p)]
    choice = good or items
    if choice:
        p = rng.choice(choice)
        pool[p] -= 1
        if not pool[p]:
            del pool[p]
        return p
    good = [p for p in fallback if accept(p)] or list(fallback)
    if not good:
        raise ReconstructError("no card left to fill a hidden slot")
    return rng.choice(good)


def _from_deck(side: int, snapshot: BoardSnapshot, cards: Mapping) -> Counter[int]:
    """Known cards that left ``side``'s deck: owned by it and not tokens (Xyz materials included)."""
    out: Counter[int] = Counter()
    for card in snapshot.sides[side].cards:
        for p in (card.password, *card.materials):
            info = cards.get(p)
            if p and (card.owner is None or card.owner == side) and not (info is not None and info.type & C.TYPE_TOKEN):
                out[p] += 1
    return out


_SUMMON_STATUS = {"normal": C.STATUS_SUMMON_TURN, "special": C.STATUS_SPSUMMON_TURN,
                  "flip": C.STATUS_FLIP_SUMMON_TURN, "set": C.STATUS_SET_TURN}  # fmt: skip


def turn_state(card: SnapshotCard) -> tuple[int, bool, str]:
    """(core status bits, attacked, once-per-turn use to spend) of an advisor card for this turn."""
    status = _SUMMON_STATUS.get(card.summoned, 0)
    if card.location == "mzone" and card.position_locked and not status:
        status = C.STATUS_FORM_CHANGED  # locked without a summon this turn: repositioned (or attacked)
    if card.location == "szone" and card.set_this_turn:
        status |= C.STATUS_SET_TURN
    spend = ""
    if card.card_activated:
        spend = "activation"  # the activation itself raised the effect flag: no other effect can be told apart
    elif card.effect_used:
        spend = "effect"
    return status, card.attacked, spend


def _placed(card: SnapshotCard, password: int, own: bool = False) -> Placed:
    # snapshot sides are (advisor, opponent) and the advisor is engine player 0, so owners map one to one
    status, attacked, spend = turn_state(card) if own else (0, False, "")
    return Placed(password, card.core_location, card.sequence, card.core_position, card.owner, card.materials,
                  card.counters, status, attacked, spend, card.attack, card.defense)  # fmt: skip


def complete(snapshot: BoardSnapshot, own_deck: Deck, prior: OpponentPrior, cards: Mapping, rng: random.Random
             ) -> CompletedBoard:  # fmt: skip
    """One sample of the hidden cards (see the module docstring)."""
    if snapshot.turn_player != 0:
        raise ReconstructError("only the advisor's own turn is supported")
    board = CompletedBoard(snapshot)

    def is_extra(p: int) -> bool:
        card = cards.get(p)
        return card is not None and card.is_extra_deck

    def is_main_monster(p: int) -> bool:
        card = cards.get(p)
        return card is not None and card.is_monster and not card.is_extra_deck

    def is_spell_trap(p: int) -> bool:
        card = cards.get(p)
        return card is not None and (card.is_spell or card.is_trap)

    # the advisor: its deck list minus what is already out of the deck
    me = snapshot.sides[0]
    seen = _from_deck(0, snapshot, cards)
    main_left = Counter(own_deck.main) - seen
    extra_left = Counter(own_deck.extra) - Counter({p: n for p, n in seen.items() if is_extra(p)})
    board.cards[0].extend(_placed(c, c.password, own=True) for c in me.cards)
    deck = [p for p, n in main_left.items() for _ in range(n)]
    rng.shuffle(deck)
    if len(deck) < me.deck_count:
        board.notes.append(f"own deck: {me.deck_count} cards in the game, {len(deck)} left in the list; padded")
        deck += [rng.choice(own_deck.main) for _ in range(me.deck_count - len(deck))]
    board.decks[0].extend(deck[: me.deck_count])
    extra = [p for p, n in extra_left.items() for _ in range(n)]
    rng.shuffle(extra)
    if len(extra) < me.extra_count:
        board.notes.append(f"own Extra Deck: {me.extra_count} cards in the game, {len(extra)} left in the list")
    board.extras[0].extend(extra[: me.extra_count])

    # the opponent: one meta deck consistent with its public cards, minus those cards
    op = snapshot.sides[1]
    op_seen = _from_deck(1, snapshot, cards)
    index, board.consistent = prior.pick(op_seen, rng)
    board.opponent_deck = index
    chosen = prior.decks[index]
    if not board.consistent:
        board.notes.append("no meta deck holds every public opponent card; the opponent deck is a share-weighted guess")
    main_left = Counter(chosen.main) - op_seen
    extra_left = Counter(chosen.extra) - op_seen
    for card in op.cards:
        if card.known:
            board.cards[1].append(_placed(card, card.password))
            continue
        if card.location == "mzone":
            accept = is_main_monster
        elif card.location == "szone":
            accept = is_spell_trap
        else:
            accept = lambda p: True  # noqa: E731
        board.cards[1].append(_placed(card, _draw(main_left, rng, accept, chosen.main)))
    deck = [p for p, n in main_left.items() for _ in range(n)]
    rng.shuffle(deck)
    deck += [rng.choice(chosen.main) for _ in range(op.deck_count - len(deck))]
    board.decks[1].extend(deck[: op.deck_count])
    extra = [p for p, n in extra_left.items() for _ in range(n)]
    rng.shuffle(extra)
    if chosen.extra:
        extra += [rng.choice(chosen.extra) for _ in range(op.extra_count - len(extra))]
    board.extras[1].extend(extra[: op.extra_count])
    return board


CARD_ADVANCE = 52112003
SETUP_LP_MARGIN = 8000  # LP added for the setup, taken back at the start of Main Phase 1 (see TURN_STATE_LUA)

# Applied at the start of the reconstruction's Main Phase 1 (the turn start clears per-turn state): statuses, attack
# bans, Normal Summons used, additional Tribute Summons granted and spent once-per-turn limits of player 0.
TURN_STATE_LUA = """
local function ygorl_spend(c,where,activation)
  local keys,pick={},{}
  for _,e in ipairs({c:GetOwnEffects()}) do
    local _,max,code,flag,hopt=e:GetCountLimit()
    local act=e:IsHasType(EFFECT_TYPE_ACTIVATE)
    if max and max>0 and act==activation and (act or e:GetRange()&where~=0) then
      local k=code~=0 and (code..":"..flag..":"..hopt) or tostring(e)
      if not keys[k] then keys[k]=true pick[#pick+1]=e end
    end
  end
  if #pick==1 then pick[1]:UseCountLimit(0) end
end
local function ygorl_where(c)
  local loc=c:GetLocation()
  if loc==LOCATION_SZONE then return LOCATION_SZONE|LOCATION_FZONE|LOCATION_PZONE end
  return loc
end
local ygorl_e=Effect.GlobalEffect()
ygorl_e:SetType(EFFECT_TYPE_FIELD|EFFECT_TYPE_CONTINUOUS)
ygorl_e:SetCode(EVENT_PHASE_START|PHASE_MAIN1)
ygorl_e:SetOperation(function(e)
  -- the real LP: the setup ran with SETUP_LP_MARGIN more, as it replays the turn's Draw / Standby Phases (Burning
  -- Land's Standby damage was dealt again, and at 300 LP the rebuilt duel ended before Main Phase 1)
  Duel.SetLP(0,YGORL_TURN.lp[1])
  Duel.SetLP(1,YGORL_TURN.lp[2])
  for _,t in ipairs(YGORL_TURN.cards) do
    local c,status,attacked,spend=t[1],t[2],t[3],t[4]
    if status~=0 then c:SetStatus(status,true) end
    if attacked then
      local a=Effect.CreateEffect(c)
      a:SetType(EFFECT_TYPE_SINGLE)
      a:SetCode(EFFECT_CANNOT_ATTACK)
      a:SetReset(RESET_EVENT|RESETS_STANDARD|RESET_PHASE|PHASE_END)
      c:RegisterEffect(a)
    end
    if spend=="effect" then ygorl_spend(c,ygorl_where(c),false) end
    if spend=="activation" then ygorl_spend(c,0,true) end
  end
  for _,t in ipairs(YGORL_TURN.activated) do
    local c=Duel.GetMatchingGroup(Card.IsCode,0,0xff,0,nil,t[1]):GetFirst()
    if c then ygorl_spend(c,t[2],(t[2]&LOCATION_ONFIELD)==0 and c:IsSpellTrap()) end
    -- "you can only activate 1 <card> per turn" across a card's effects is a player flag keyed by the card's id in
    -- the scripts (Ancient Chant's Graveyard effect checks it): the activation this turn set it
    if not Duel.HasFlagEffect(0,t[1]) then Duel.RegisterFlagEffect(0,t[1],RESET_PHASE|PHASE_END,0,1) end
  end
  for _,code in ipairs(YGORL_TURN.effects) do  -- a Spell / Trap's lasting effect on the player, the card gone:
    local g=Duel.GetMatchingGroup(Card.IsCode,0,0xff,0,nil,code)  -- it was activated this turn
    local c=g:GetFirst()
    local up=g:IsExists(function(x) return x:IsOnField() and x:IsFaceup() end,1,nil)
    if c and c:IsSpellTrap() and not up then ygorl_spend(c,0,true) end
  end
  local extra=YGORL_TURN.extra
  local used=YGORL_TURN.summons
  if YGORL_TURN.advance then  -- Card Advance granted one: no card grants another (IsPlayerCanAdditionalTributeSummon)
    Duel.RegisterFlagEffect(0,CARD_CARD_ADVANCE,RESET_PHASE|PHASE_END,0,1)
  end
  if extra>0 and used>1 then extra=math.max(0,extra-(used-1)) used=1 end
  for _=1,math.min(used,1) do Duel.IncreaseSummonedCount() end
  if extra>0 then
    local s=Effect.GlobalEffect()
    s:SetType(EFFECT_TYPE_FIELD)
    s:SetCode(EFFECT_EXTRA_SUMMON_COUNT)
    s:SetTargetRange(LOCATION_HAND,0)
    s:SetValue(function() return 0x1 end)  -- not the literal: proc_workaround would raise Card Advance's flag
    s:SetReset(RESET_PHASE|PHASE_END)
    Duel.RegisterEffect(s,0)
    local s2=s:Clone()
    s2:SetCode(EFFECT_EXTRA_SET_COUNT)
    Duel.RegisterEffect(s2,0)
  end
  -- ATK / DEF the real game shows (equips and other continuous effects the board cannot hold), as the difference
  -- to what the reconstruction computes once everything above is in place; it stays while the card does
  for _,t in ipairs(YGORL_TURN.stats) do
    local c=t[1]
    if c:IsLocation(LOCATION_MZONE) and c:IsFaceup() then
      for _,st in ipairs({{t[2],EFFECT_UPDATE_ATTACK,c:GetAttack()},{t[3],EFFECT_UPDATE_DEFENSE,c:GetDefense()}}) do
        local want,code,now=st[1],st[2],st[3]
        if want and want~=now and not (code==EFFECT_UPDATE_DEFENSE and c:IsType(TYPE_LINK)) then
          local u=Effect.CreateEffect(c)
          u:SetType(EFFECT_TYPE_SINGLE)
          u:SetCode(code)
          u:SetProperty(EFFECT_FLAG_CANNOT_DISABLE|EFFECT_FLAG_UNCOPYABLE)
          u:SetValue(want-now)
          u:SetReset(RESET_EVENT|RESETS_STANDARD)
          c:RegisterEffect(u)
        end
      end
    end
  end
  e:Reset()
end)
Duel.RegisterEffect(ygorl_e,0)
"""


def board_script(board: CompletedBoard) -> str:
    """The Lua script that puts ``board`` into a fresh duel (``Debug.AddCard`` per card) and, at the start of its
    Main Phase 1, the turn so far (``TURN_STATE_LUA``)."""
    snap = board.snapshot
    # MD lets Card Advance grant the additional Tribute Summon unless Card Advance itself already did (recorded duels:
    # accepted after Ancient Chant's grant, refused after its own); ygopro's workaround flag blocks it after any
    advance = CARD_ADVANCE in {pw for pw, _ in snap.player_effects} or any(p == CARD_ADVANCE for p, _ in snap.activated)
    lines = ["-- generated by ygorl.advisor.reconstruct",
             f"YGORL_TURN={{cards={{}},activated={{}},stats={{}},summons={snap.summons_used},"
             f"lp={{{snap.sides[0].lp},{snap.sides[1].lp}}},"
             f"extra={snap.extra_tribute_summons},advance={'true' if advance else 'false'},"
             f"effects={{{','.join(str(pw) for pw in sorted({pw for pw, _ in snap.player_effects}))}}}}}"]  # fmt: skip
    for pw, where in snap.activated:
        lines.append(f"table.insert(YGORL_TURN.activated,{{{pw},{LOCATIONS[where]}}})")
    for player in (0, 1):
        for card in board.cards[player]:
            owner = player if card.owner is None else card.owner
            proc = "true" if card.location == C.LOCATION_MZONE else "false"
            lines.append(f"local c=Debug.AddCard({card.password},{owner},{player},{card.location},{card.sequence},"
                         f"{card.position},{proc})")  # fmt: skip
            if card.status or card.attacked or card.spend:
                lines.append(f"if c then table.insert(YGORL_TURN.cards,{{c,{card.status},"
                             f"{'true' if card.attacked else 'false'},\"{card.spend}\"}}) end")  # fmt: skip
            if card.attack is not None or card.defense is not None:
                shown = ",".join("nil" if v is None else str(v) for v in (card.attack, card.defense))
                lines.append(f"if c then table.insert(YGORL_TURN.stats,{{c,{shown}}}) end")
            for t, n in card.counters:
                lines.append(f"if c then Debug.PreAddCounter(c,{t},{n}) end")
            for m in card.materials:  # AddCard on an occupied Monster Zone attaches an Xyz material
                lines.append(f"Debug.AddCard({m},{owner},{player},{C.LOCATION_MZONE},{card.sequence},"
                             f"{C.POS_FACEUP_ATTACK})")  # fmt: skip
        for p in board.decks[player]:
            lines.append(f"Debug.AddCard({p},{player},{player},{C.LOCATION_DECK},0,{C.POS_FACEDOWN_DEFENSE})")
        for p in board.extras[player]:
            lines.append(f"Debug.AddCard({p},{player},{player},{C.LOCATION_EXTRA},0,{C.POS_FACEDOWN_DEFENSE})")
    return "\n".join(lines) + "\n" + TURN_STATE_LUA


@dataclass
class Reconstruction:
    """A live core duel holding a completed board, stopped at the advisor's idle decision."""

    board: CompletedBoard
    core: _core.Duel
    tracker: DuelTracker
    point: DecisionPoint
    warnings: list[str]


def build(board: CompletedBoard, rule_flags: int, *, cards=None, scripts=None, seed: int = 1, extra_script: str = "",
          until: Callable[[DecisionPoint], bool] | None = None) -> Reconstruction:  # fmt: skip
    """Start a duel holding ``board`` and run it to the advisor's first idle decision: Main Phase 1, or for a
    Battle Phase snapshot the battle command reached by taking the core's own ``battle_phase`` action there.

    ``extra_script``: Lua appended to the board script (:mod:`ygorl.selfplay.decide` registers a one-shot effect
    there to make the core ask a question of its own, e.g. a Summon's position); ``until``: stop instead at the
    first setup decision it accepts (that decision is not answered)."""
    cards = cards if cards is not None else default_cards()
    base = scripts if scripts is not None else default_scripts()
    snap = board.snapshot
    if snap.turn > 1:
        rule_flags |= C.DUEL_ATTACK_FIRST_TURN
    script = (board_script(board) + extra_script).encode()

    def read(name: str) -> bytes | None:
        return script if name == BOARD_SCRIPT else base.read(name)

    # set up with a margin (the Draw / Standby Phases replayed on the way to Main Phase 1 can deal damage the real
    # game dealt already); TURN_STATE_LUA sets the real LP at the start of Main Phase 1
    lp = (snap.sides[0].lp + SETUP_LP_MARGIN, snap.sides[1].lp + SETUP_LP_MARGIN)
    core = _core.Duel(expand_seed(seed), rule_flags, (lp[0], 0, 0), (lp[1], 0, 0), cards.to_core(), read)
    for name in ("constant.lua", "utility.lua", BOARD_SCRIPT):
        if not core.load_script(name):
            raise ReconstructError(f"failed to load {name}: {' | '.join(str(t) for _, t in core.pop_logs())}")
    core.start()
    config = DuelConfig(rule_flags=rule_flags, player=PlayerRules(max(lp), 0, 0), shuffle_decks=False)
    tracker = DuelTracker(config, 0, cards)
    tracker.lp = list(lp)
    warnings = list(board.notes)
    run = True  # False while a multi-select is half answered (the core still waits for the response)
    for _ in range(MAX_PRE_DECISIONS + 1):
        while run:
            status = core.process()
            tracker.on_buffer(core.get_message(), status, core.pop_logs())
            if tracker.done or tracker.awaiting:
                break
        if tracker.done:
            raise ReconstructError(f"the duel ended while being set up ({tracker.result.reason})")
        point = tracker.point()
        if point is None:
            raise ReconstructError(f"the duel stopped while being set up ({tracker.result.reason})")
        if until is not None and until(point):
            warnings += [f"script error: {e}" for e in tracker.result.script_errors]
            return Reconstruction(board, core, tracker, point, warnings)
        if point.player == 0 and isinstance(point.decision, M.SelectBattleCmd) and snap.phase == "battle":
            warnings += [f"script error: {e}" for e in tracker.result.script_errors]
            return Reconstruction(board, core, tracker, point, warnings)
        if point.player == 0 and isinstance(point.decision, M.SelectIdleCmd) and point.phase == C.PHASE_MAIN1:
            if snap.phase != "battle":
                warnings += [f"script error: {e}" for e in tracker.result.script_errors]
                return Reconstruction(board, core, tracker, point, warnings)
            index = next((i for i, a in enumerate(point.actions) if a.kind == "battle_phase"), None)
            if index is None:
                raise ReconstructError("the reconstruction cannot enter the Battle Phase")
            response = tracker.act(index)
            run = response is not None
            if run:
                core.set_response(response)
            continue
        index = next((i for i, a in enumerate(point.actions) if a.kind in DECLINE), 0)
        warnings.append(f"answered {point.decision.name} of player {point.player} before Main Phase 1: "
                        f"{point.actions[index].kind}")  # fmt: skip
        response = tracker.act(index)
        run = response is not None
        if run:
            core.set_response(response)
    raise ReconstructError(f"no idle decision after {MAX_PRE_DECISIONS} setup decisions")


__all__ = ["BOARD_SCRIPT", "CompletedBoard", "OpponentPrior", "Placed", "Reconstruction", "ReconstructError",
           "TURN_STATE_LUA", "board_script", "build", "complete", "turn_state"]  # fmt: skip
