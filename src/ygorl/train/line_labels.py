"""Solver hand-trap labels on the learner's chain windows of scripted-line games (docs/training.md §8.3).

The graded labels (``out/claude-scale/handtrap_labels_graded.py``) say, for a solved ``plain`` demonstration (a deck
and a hand), how much the combo's best reachable board drops if the responder fires Ash Blossom (:data:`FIRE`) at
its k-th Ash-able chain window along the line: ``values[k]`` in ``[0, best_without]`` (0: the combo fully recovers,
a bait; ``best_without``: nothing is left, the starter). A window is counted along the solver's replay with the
passive responder holding Ash, and the learner answers the same line with its own deck, so its k-th window need
not be the solver's k-th: the windows are identified instead by the effect they answer and by how many times the
responder has answered that effect in the game (:func:`ygorl.train.lines.chain_effect`; ``Rollout.line_effect`` /
``line_occurrence``). :func:`window_identities` replays a line exactly as the dataset builder does (the plain
start with Ash appended to the responder's deck, the combo seat on the line, the responder passing) and records
that identity for every window; :func:`build_window_table` caches the result as JSON next to the graded file so a
training run replays nothing once the cache is complete.

:class:`LineLabels` joins the labels to a :class:`~ygorl.train.lines.LineLibrary` and turns a rollout into a per-row
bonus for PPO (``PPOConfig.line_bonus_coef``): at a labelled window with an Ash candidate, with ``u_k = values[k] /
best_without`` and ``m_k = max(u_j, j > k)`` (0 at the last window), the gain of firing is ``g_k = u_k - m_k`` when
``u_k < m_k`` (a later window hurts more: hold) and ``u_k`` otherwise (this window is as damaging as any later one,
ties included: fire now); the taken action gets ``g_k`` if it is the Ash activation and ``-g_k`` otherwise. Rows
without a label get 0. :meth:`LineLabels.ce_targets` turns the same windows into supervised targets instead
(``PPOConfig.line_ce_coef``): the activation where ``g_k > 0``, the passive answer otherwise, weighted by ``|g_k|``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from ygorl.cards.cdb import CardVocab
from ygorl.data.environment import Environment
from ygorl.engine.constants import MSG_SELECT_CHAIN
from ygorl.engine.duel import DuelSession, default_cards
from ygorl.env.encoding import ACTION_KINDS
from ygorl.env.observer import PointObserver
from ygorl.solver.demo import PASSIVE_KINDS, Demonstration, read_jsonl
from ygorl.train.bc import passive_action
from ygorl.train.lines import LINE_PLAYER, LineLibrary, chain_effect

FIRE = 14558127  # Ash Blossom & Joyous Spring: the hand trap the graded labels are about
CACHE_FORMAT = "ygorl.line-windows.v1"
_RESPONDER = 1 - LINE_PLAYER


def read_graded(path: str | Path, fire: int = FIRE) -> dict[tuple[str, int], dict]:
    """The graded records of hand trap ``fire`` in ``path`` by ``(deck name, hand index)``; a record of 0 windows has
    no labels. A file may grade several hand traps (one record per hand and card, the card in ``"fire"``; a record
    without it is Ash's, as the first graded runs wrote them); the other cards' records are left out."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if int(r.get("fire", FIRE)) == fire:
                    out[(r["deck"], int(r["hand_index"]))] = r
    return out


def fire_start(demo: Demonstration, fire: int = FIRE) -> Demonstration:
    """``demo`` with the responder's deck holding ``fire`` on top of the passive deck, as the fire records start."""
    start = json.loads(json.dumps(demo.start))
    start["decks"]["b"]["main"].append(fire)
    return replace(demo, start=start)


def window_identities(demo: Demonstration, *, cards=None, env: Environment | None = None, event_length: int = 64,
                      fire: int = FIRE) -> list[tuple[int, int]]:  # fmt: skip
    """``(effect password or 0, occurrence)`` of every chain window at which the responder could activate ``fire``
    along ``demo.lines[0]``, in order: the combo seat plays the line, the responder (holding ``fire``) passes at
    every prompt. Occurrences count the responder's MSG_SELECT_CHAIN prompts with a real choice (two or more
    choosable rows) per effect, exactly as the collector does on the learner's rows."""
    cards = cards if cards is not None else default_cards()
    vocab = CardVocab.from_db(cards)
    line = demo.lines[0]
    ours = iter(i for i, p in zip(line.actions, line.players) if p == LINE_PLAYER)
    session = DuelSession(fire_start(demo, fire).replay(None).duel(env, cards=cards))
    observer = PointObserver(cards, vocab, event_length)
    counts: dict[int, int] = {}
    out: list[tuple[int, int]] = []
    try:
        while not session.done and session.tracker.turn < 2:
            point = session.point
            if point is None:
                break
            if point.player == LINE_PLAYER:
                idx = next(ours, None)
                if idx is None:
                    break
            else:
                idx = passive_action(point)
                if point.decision.TYPE == MSG_SELECT_CHAIN and len(point.choosable) > 1:
                    obs = observer.encode(point, session.core)
                    effect = chain_effect(obs)
                    counts[effect] = counts.get(effect, 0) + 1
                    if any(a.card is not None and a.card.code == fire for a in point.actions):
                        out.append((vocab.password(effect) if effect >= CardVocab.FIRST_INDEX else 0, counts[effect]))
            observer.on_decision(point, idx)
            session.act(idx)
    finally:
        session.close()
    return out


def fire_windows(fire_demo: Demonstration, *, cards=None, env: Environment | None = None,
                 fire: int = FIRE) -> dict[int, int]:  # fmt: skip
    """Window -> index of the first line of the ``--fire`` record ``fire_demo`` that fires ``fire`` there: the
    windows its lines recover. A window is numbered as the graded labels number it (``values[k]``): the k-th
    responder decision along the line with a ``fire`` activation among its actions, 1-based. The record starts
    where its lines do (the solver bakes ``fire`` into the responder's deck)."""
    cards = cards if cards is not None else default_cards()
    demo = fire_demo
    if demo.start is None:
        raise ValueError("the fire record has no start position")
    out: dict[int, int] = {}
    for n, ln in enumerate(demo.lines):
        session = DuelSession(demo.replay(None).duel(env, cards=cards))
        window = 0
        try:
            for idx in ln.actions:
                point = session.point
                if point is None:
                    break
                if point.player == _RESPONDER and any(
                    a.card is not None and a.card.code == fire for a in point.actions
                ):
                    window += 1
                    a = point.actions[idx]
                    if a.card is not None and a.card.code == fire:
                        out.setdefault(window, n)
                        break
                session.act(idx)
        finally:
            session.close()
    return out


def default_cache(graded: str | Path) -> Path:
    return Path(graded).with_suffix(".windows.json")


def build_window_table(graded: str | Path, demos: Iterable[str | Path], *, cards=None, env: Environment | None = None,
                       event_length: int = 64, cache: str | Path | None = None, fire: int = FIRE,
                       log: Callable[[str], None] | None = None) -> dict[str, dict]:  # fmt: skip
    """Window identities of every graded hand with windows whose solved plain record is in ``demos``, keyed
    ``"<deck>#<hand_index>"``: ``{"windows": <solver's count>, "ids": [[password, occurrence], ...]}``. Read from
    ``cache`` (default: the graded file with the suffix ``.windows.json``) when it has the hand and was built with
    the same ``event_length`` and ``fire``; the hands it lacks are replayed and the cache rewritten."""
    cache = default_cache(graded) if cache is None else Path(cache)
    labels = read_graded(graded, fire)
    hands: dict[str, dict] = {}
    if cache.exists():
        data = json.loads(cache.read_text())
        if (data.get("format"), data.get("event_length"), data.get("fire")) == (CACHE_FORMAT, event_length, fire):
            hands = data["hands"]
    table: dict[str, dict] = {}
    replayed = 0
    for path in demos:
        for demo in read_jsonl(path):
            key = (demo.deck["name"], demo.hand_index)
            rec = labels.get(key)
            if demo.variant != "plain" or demo.status != "solved" or not demo.lines or rec is None:
                continue
            name = f"{key[0]}#{key[1]}"
            if not rec.get("windows"):
                continue
            entry = hands.get(name)
            if entry is None:
                ids = window_identities(demo, cards=cards, env=env, event_length=event_length, fire=fire)
                entry = hands[name] = {"windows": len(ids), "ids": [list(x) for x in ids]}
                replayed += 1
            table[name] = entry
    if replayed:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"format": CACHE_FORMAT, "event_length": event_length, "fire": fire,
                                     "hands": hands}, separators=(",", ":")) + "\n")  # fmt: skip
    if log is not None:
        log(f"line labels: {len(table)} hands with windows ({replayed} replayed, cache {cache})")
    return table


@dataclass(frozen=True)
class WindowLabel:
    u: float  # values[k] / best_without
    later: float  # max u over the later windows of the hand (0 at the last one)

    @property
    def fire_bonus(self) -> float:
        """Gain of firing here: ``u - later`` when a later window hurts more (negative: hold), else ``u`` itself
        (firing at the earliest window that is as damaging as any later one, including ties, is right; 0 when
        nothing is at stake). The strict ``u - later`` alone made every tied window a hold and taught over-holding."""
        return self.u - self.later if self.u < self.later else self.u


class LineLabels:
    """The labels of a library's lines keyed by ``(line id, effect vocab index, occurrence)``, and the per-row PPO
    bonus of a rollout (:meth:`bonus`)."""

    def __init__(self, table: dict[tuple[int, int, int], WindowLabel], fire_index: int, hands: int = 0) -> None:
        self.table = table
        self.fire_index = fire_index  # vocab index of the hand trap: an action row with this card is the activation
        self.hands = hands  # labelled hands joined to a line of the library

    @classmethod
    def load(cls, graded: str | Path, library: LineLibrary, vocab: CardVocab, *, cards=None,
             env: Environment | None = None, event_length: int = 64, cache: str | Path | None = None,
             log: Callable[[str], None] | None = None) -> LineLabels:  # fmt: skip
        """Join the graded labels of ``graded`` (window identities replayed from the library's files, see
        :func:`build_window_table`) to the library's lines."""
        files = sorted({s.source for s in library.scripts if s.source})
        windows = build_window_table(
            graded, files, cards=cards, env=env, event_length=event_length, cache=cache, log=log
        )
        labels = read_graded(graded)
        table: dict[tuple[int, int, int], WindowLabel] = {}
        hands = 0
        for script in library.scripts:
            rec = labels.get((script.deck.name, script.hand_index))
            entry = windows.get(script.name)
            if rec is None or entry is None:
                continue
            best = float(rec.get("best_without") or 0)
            values = {int(k): float(v) for k, v in rec.get("values", {}).items()}
            n = min(int(rec["windows"]), len(entry["ids"]))
            if best <= 0 or any(k not in values for k in range(1, n + 1)):
                continue
            u = [values[k] / best for k in range(1, n + 1)]
            for k, (password, occurrence) in enumerate(entry["ids"][:n], 1):
                effect = vocab.index(password) if password else 0
                later = max(u[k:], default=0.0)
                table[(script.id, effect, occurrence)] = WindowLabel(u[k - 1], later)
            hands += 1
        if log is not None:
            log(f"line labels: {len(table)} windows of {hands} hands joined to {len(library)} lines")
        return cls(table, vocab.index(FIRE), hands)

    def __len__(self) -> int:
        return len(self.table)

    def _windows(self, ro):
        """``(t, b, label, ash)`` of every labelled window row with a choosable ``fire`` candidate, ``ash`` the bool
        candidate mask of the activation rows, plus the rollout's ``(codes, mask)`` (None when nothing applies)."""
        ids, effect, occurrence = ro.line_id, ro.line_effect, ro.line_occurrence
        if ids is None or effect is None or occurrence is None or not self.table or not isinstance(ro.obs, dict):
            return
        rows = ((ids >= 0) & (occurrence > 0)).nonzero().tolist()
        if not rows:
            return
        B = ro.shape[1]
        codes = ro.obs["actions"][..., 2].cpu()  # [T * B, A] card vocab index of every candidate row
        mask = ro.action_mask.cpu()
        width = min(codes.shape[1], mask.shape[-1])
        for t, b in rows:
            label = self.table.get((int(ids[t, b]), int(effect[t, b]), int(occurrence[t, b])))
            if label is None:
                continue
            ash = (codes[t * B + b, :width] == self.fire_index) & mask[t, b, :width]
            if bool(ash.any()):
                yield t, b, label, ash

    @torch.no_grad()
    def bonus(self, ro) -> tuple[Tensor, dict[str, float]]:
        """``[T, B]`` bonus of the taken action at every labelled window with a choosable ``fire`` candidate (0
        elsewhere) and the stats: ``line_labelled_rows``, ``line_fire_rows`` (the taken action was the activation),
        ``line_bonus_mean`` (over the labelled rows)."""
        T, B = ro.shape
        bonus = torch.zeros(T, B)
        stats = {"line_labelled_rows": 0, "line_fire_rows": 0, "line_bonus_mean": 0.0}
        taken = ro.actions.cpu()
        n = fired = 0
        for t, b, label, ash in self._windows(ro):
            fire = bool(ash[int(taken[t, b])])
            bonus[t, b] = label.fire_bonus if fire else -label.fire_bonus
            n += 1
            fired += fire
        stats["line_labelled_rows"], stats["line_fire_rows"] = n, fired
        stats["line_bonus_mean"] = float(bonus.sum()) / n if n else 0.0
        return bonus, stats

    @torch.no_grad()
    def ce_targets(self, ro) -> tuple[Tensor, Tensor, dict[str, float]]:
        """The supervised targets of ``PPOConfig.line_ce_coef`` at the windows :meth:`bonus` uses: ``[T, B]`` long
        candidate index (-1 elsewhere) -- the first Ash activation row when the gain ``g_k`` is positive, else the row's
        passive answer (the first choosable candidate of a kind in :data:`~ygorl.solver.demo.PASSIVE_KINDS` order;
        a window without one is skipped) -- and ``[T, B]`` float weight ``|g_k|`` (0 elsewhere). Stats:
        ``line_ce_rows`` (rows with a target), ``line_ce_fire_targets`` (of which the target is the activation)."""
        T, B = ro.shape
        target = torch.full((T, B), -1, dtype=torch.long)
        weight = torch.zeros(T, B)
        stats = {"line_ce_rows": 0, "line_ce_fire_targets": 0}
        kinds = passive = None
        for t, b, label, ash in self._windows(ro):
            if label.fire_bonus > 0:
                idx = int(ash.nonzero()[0, 0])
                stats["line_ce_fire_targets"] += 1
            else:
                if kinds is None:
                    kinds = ro.obs["actions"][..., 0].cpu()  # [T * B, A] action kind + 1 (docs/encoding.md)
                    passive = [ACTION_KINDS.index(k) + 1 for k in PASSIVE_KINDS]
                legal = ro.action_mask[t, b, : ash.shape[0]].cpu() & ~ash
                row = kinds[t * B + b, : ash.shape[0]]
                idx = next((int(i) for k in passive for i in (legal & (row == k)).nonzero()[:, 0].tolist()), -1)
                if idx < 0:
                    continue
            target[t, b] = idx
            weight[t, b] = abs(label.fire_bonus)
            stats["line_ce_rows"] += 1
        return target, weight, stats


__all__ = ["CACHE_FORMAT", "FIRE", "LineLabels", "WindowLabel", "build_window_table", "default_cache", "fire_start",
           "fire_windows", "read_graded", "window_identities"]  # fmt: skip
