"""Behaviour cloning warm start from solver demonstrations (T4a.2, design I4); see docs/bc.md.

The demonstration set of ``ygorl.solver`` (docs/solver.md) stores action indices only. Each line is
replayed in a fresh duel and every decision of the deck under study (engine player 0) is encoded with
the Python reference encoders (:class:`~ygorl.env.observer.PointObserver`: the same four board arrays
and event window ``EncodedVecEnv`` produces), giving ``(observation, demonstrated action row)`` pairs.
:func:`train_bc` fits :class:`~ygorl.nets.PolicyNet` by cross-entropy over the legal candidates (the
network masks the illegal rows). The checkpoint (:func:`ygorl.nets.agent.save_checkpoint`) is the PPO
initialisation and the KL reference policy of I8; ``PolicyAgent`` plays it through
:class:`~ygorl.nets.agent.NetPolicy` (``make_agent("policy:PATH")``).

Evaluation (the acceptance of T4a.2):

- :func:`step_accuracy`: teacher-forced top-1 agreement with the demonstrated actions;
- :func:`play_opening`: free-running play of turn 1 from a record's start position (the solver's duel:
  same seed words, deck order and passive opponent) until turn 2, scored like the solver scores a
  line — do the target cards stand on the final board (:func:`~ygorl.solver.targets.board_summary_missing`);
- :func:`opening_report`: line reproduction on the training hands and field quality on held-out hands.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch

from ygorl.cards.cdb import CardVocab
from ygorl.cards.ydk import Deck
from ygorl.data.environment import Environment
from ygorl.engine.constants import DUEL_PSEUDO_SHUFFLE
from ygorl.engine.duel import DuelConfig, DuelSession, default_cards, expand_seed
from ygorl.engine.replay import Replay
from ygorl.env.encoding import MAX_OPTIONS
from ygorl.env.events import DEFAULT_EVENT_LENGTH
from ygorl.env.observer import PointObserver
from ygorl.nets.batch import OBS_KEYS, to_tensors
from ygorl.nets.policy import PolicyNet
from ygorl.solver.batch import PASSIVE_OPPONENT, _order_with_hand, sample_hand
from ygorl.solver.demo import PASSIVE_KINDS, DemoError, Demonstration
from ygorl.solver.targets import board_summary, board_summary_missing, parse_targets

DEMO_PLAYER = 0  # engine player of the deck under study in every demonstration (it moves first)
MAX_OPENING_STEPS = 300  # free-running turn 1: the agent's steps before the host closes the turn passively


# ------------------------------------------------------------------ dataset


@dataclass
class BCData:
    """Stacked observations (``OBS_KEYS``, leading sample dimension) and the demonstrated action rows.

    Only decisions of :data:`DEMO_PLAYER` with at least two legal actions are kept (a forced step has zero
    loss); ``meta[i]`` says where sample ``i`` comes from.
    """

    obs: dict[str, np.ndarray]
    actions: np.ndarray  # [N] int64, row of the candidate-action table
    meta: list[dict] = field(default_factory=list)  # deck, hand_index, variant, fire, line, step, solver (vs closing)
    skipped: Counter = field(default_factory=Counter)  # forced steps, actions beyond the 128 encoded rows

    def __len__(self) -> int:
        return len(self.actions)

    def batch(self, idx: np.ndarray, device: torch.device | str | None = None) -> tuple[dict, torch.Tensor]:
        """Samples ``idx`` as tensors, with the padding rows no sample of the batch uses cut off (:func:`trim_padding`)."""
        return to_tensors(trim_padding({k: v[idx] for k, v in self.obs.items()}), device), torch.as_tensor(self.actions[idx])

    def subset(self, keep: np.ndarray) -> BCData:
        keep = np.asarray(keep)
        idx = np.flatnonzero(keep) if keep.dtype == bool else keep
        return BCData({k: v[idx] for k, v in self.obs.items()}, self.actions[idx], [self.meta[i] for i in idx],
                      Counter(self.skipped))  # fmt: skip


def trim_padding(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop the trailing card / action / event rows that are padding in every sample of a batch.

    Exact for :class:`PolicyNet`: padding card rows are masked out of attention, actions do not attend
    to each other, and the history is causal with the valid tokens first (docs/nets.md). The logits
    come back ``[B, A']`` with ``A'`` the widest legal action set of the batch. On turn-1 demonstrations
    (on average 60 of 160 card rows, 4 of 128 action rows and 38 of 128 event tokens in use) a training
    step takes about 40% of the untrimmed time.
    """
    out = dict(obs)

    def used(mask: np.ndarray) -> int:
        cols = np.flatnonzero(mask.any(0))
        return int(cols[-1]) + 1 if len(cols) else 1

    n = used(obs["cards"][..., 0] > 0)
    out["cards"] = obs["cards"][:, :n]
    a = used(obs["action_mask"] != 0)
    out["actions"], out["action_mask"] = obs["actions"][:, :a], obs["action_mask"][:, :a]
    if "event_mask" in obs:
        t = used(obs["event_mask"] != 0)
        out["events"], out["event_mask"] = obs["events"][:, :t], obs["event_mask"][:, :t]
    return out


@dataclass
class DemoStep:
    """One step of a replayed line."""

    step: int  # index into DemoLine.actions
    player: int  # engine player who decided
    action: int  # demonstrated action index
    n_legal: int
    kind: str  # Action.kind of the demonstrated action
    card: tuple | None  # (password, location) of the action's card, None without one
    obs: dict[str, np.ndarray] | None = field(default=None, repr=False)  # encoded for the decisions asked for


def line_steps(demo: Demonstration, line: int, vocab: CardVocab | None = None, *, cards=None, scripts=None,
               env: Environment | None = None, event_length: int = DEFAULT_EVENT_LENGTH,
               player: int | None = DEMO_PLAYER) -> list[DemoStep]:  # fmt: skip
    """Replay ``demo.lines[line]`` and describe every step (both players').

    With a ``vocab``, the non-forced decisions of ``player`` (at least two legal actions) carry their
    observation. Every step feeds the event history, exactly as the C++ environment does.
    """
    cards = cards if cards is not None else default_cards()
    ln = demo.lines[line]
    kwargs = {k: v for k, v in (("cards", cards), ("scripts", scripts)) if v is not None}
    session = DuelSession(demo.replay(line).duel(env, **kwargs))
    observer = PointObserver(cards, vocab, event_length) if vocab is not None else None
    out: list[DemoStep] = []
    try:
        for step, idx in enumerate(ln.actions):
            point = session.point
            if point is None:
                raise DemoError(f"step {step}: the duel stopped before the line ended")
            act = point.actions[idx]
            card = (act.card.code, act.card.loc) if act.card is not None else None
            obs = None
            if observer is not None:
                observer.observe(point)
                if point.player == player and len(point.actions) > 1:
                    obs = observer.encode(point, session.core)
            out.append(DemoStep(step, point.player, idx, len(point.actions), act.kind, card, obs))
            session.act(idx)
    finally:
        session.close()
    return out


def undone_steps(steps: Sequence[DemoStep]) -> set[int]:
    """Steps that cancel out: a ``select`` of a card immediately followed by the same player's ``unselect`` of it.

    The solver's lines sometimes toggle a card in and out of a SELECT_UNSELECT_CARD selection (one tenpai
    line does it 320 times while choosing synchro materials); each pair returns to the same decision
    state, so it demonstrates nothing and is left out of the training samples and of the line a
    reproduction must match (docs/bc.md).
    """
    undone: set[int] = set()
    for a, b in zip(steps, steps[1:]):
        if (a.kind == "select" and b.kind == "unselect" and a.player == b.player and a.card == b.card
                and a.step not in undone):  # fmt: skip
            undone.update((a.step, b.step))
    return undone


def build_dataset(demos: Iterable[Demonstration], vocab: CardVocab, *, cards=None, scripts=None,
                  env: Environment | None = None, event_length: int = DEFAULT_EVENT_LENGTH,
                  player: int = DEMO_PLAYER) -> BCData:  # fmt: skip
    """Every verified line of every solved record (plain and ``--fire``) as training samples.

    Kept: the decisions of ``player`` with at least two legal actions whose demonstrated row is encoded
    (< 128), minus select/unselect toggles (:func:`undone_steps`); ``skipped`` counts the rest.
    """
    columns: dict[str, list[np.ndarray]] = {}
    actions: list[int] = []
    meta: list[dict] = []
    skipped: Counter = Counter()
    for demo in demos:
        if demo.status != "solved":
            continue
        for li, ln in enumerate(demo.lines):
            steps = line_steps(demo, li, vocab, cards=cards, scripts=scripts, env=env, event_length=event_length,
                               player=player)  # fmt: skip
            undone = undone_steps(steps)
            for st in steps:
                if st.player != player:
                    continue
                if st.step in undone:
                    skipped["toggle"] += 1
                elif st.obs is None:
                    skipped["forced"] += 1
                elif st.action >= MAX_OPTIONS:
                    skipped["beyond_128"] += 1
                else:
                    for k in OBS_KEYS:
                        if k in st.obs:
                            columns.setdefault(k, []).append(st.obs[k])
                    actions.append(st.action)
                    meta.append({"deck": demo.deck["name"], "hand_index": demo.hand_index, "variant": demo.variant,
                                 "fire": demo.fire, "line": li, "step": st.step, "n_legal": st.n_legal,
                                 "solver": st.step < ln.solver_steps})  # fmt: skip
    if not actions:
        raise ValueError("no training samples: no solved demonstration with a decision of the deck under study")
    obs = {k: np.stack(v) for k, v in columns.items()}
    return BCData(obs, np.asarray(actions, dtype=np.int64), meta, skipped)


# ------------------------------------------------------------------ training


@dataclass
class BCConfig:
    epochs: int = 12  # held-out NLL rises after about 6 epochs on the 2026-09-23 set (docs/bc.md)
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    warmup_steps: int = 100  # linear warm-up, then cosine decay to 0
    seed: int = 0


def bc_loss(net: PolicyNet, obs: dict, target: torch.Tensor, label_smoothing: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean cross-entropy over the legal candidates, logits). Illegal rows carry ``MASKED_LOGIT``: probability 0.

    With label smoothing the smoothed mass goes to the legal candidates only.
    """
    logits = net(obs).logits
    logp = torch.log_softmax(logits, -1)
    nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
    if label_smoothing > 0:
        mask = obs["action_mask"]
        uniform = -(logp * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        nll = (1 - label_smoothing) * nll + label_smoothing * uniform
    return nll.mean(), logits


def _lr_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def train_bc(net: PolicyNet, data: BCData, cfg: BCConfig, *, eval_sets: dict[str, BCData] | None = None,
             log: Callable[[dict], None] | None = None) -> list[dict]:  # fmt: skip
    """Fit ``net`` to ``data`` (AdamW, warm-up + cosine); one history entry per epoch."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    per_epoch = math.ceil(len(data) / cfg.batch_size)
    total = per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, total, cfg.warmup_steps))
    history = []
    for epoch in range(cfg.epochs):
        net.train()
        t0 = time.time()
        perm = rng.permutation(len(data))
        loss_sum = correct = 0.0
        for b in range(per_epoch):
            idx = perm[b * cfg.batch_size : (b + 1) * cfg.batch_size]
            obs, target = data.batch(idx)
            loss, logits = bc_loss(net, obs, target, cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(-1) == target).sum().item()
        entry = {"epoch": epoch + 1, "loss": loss_sum / len(data), "accuracy": correct / len(data),
                 "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}  # fmt: skip
        for name, ev in (eval_sets or {}).items():
            if len(ev):
                acc = step_accuracy(net, ev)
                entry[f"{name}_loss"], entry[f"{name}_accuracy"] = acc["nll"], acc["accuracy"]
        history.append(entry)
        if log is not None:
            log(entry)
    net.eval()
    return history


@torch.no_grad()
def step_accuracy(net: PolicyNet, data: BCData, batch_size: int = 256) -> dict:
    """Teacher-forced agreement: top-1 accuracy, mean NLL, and the uniform-guess accuracy for scale."""
    was_training = net.training
    net.eval()
    correct = nll = 0.0
    for b in range(0, len(data), batch_size):
        idx = np.arange(b, min(len(data), b + batch_size))
        obs, target = data.batch(idx)
        logp = net(obs).log_probs()
        correct += (logp.argmax(-1) == target).sum().item()
        nll -= logp.gather(1, target.unsqueeze(1)).sum().item()
    net.train(was_training)
    n = max(1, len(data))
    uniform = float(np.mean([1.0 / min(m["n_legal"], MAX_OPTIONS) for m in data.meta])) if data.meta else 0.0
    return {"samples": len(data), "accuracy": correct / n, "nll": nll / n, "uniform_accuracy": uniform}


# ------------------------------------------------------------------ free-running openings


def start_replay(demo: Demonstration, config: DuelConfig | None = None) -> Replay:
    """The duel a record's lines start from; rebuilt from the deck and hand seed when no line was solved.

    The rebuild matches the solver's setup (docs/solver.md): the host shuffle of ``hand_seed`` with the
    opening hand moved to the end of the list, the seed words of ``hand_seed``, the rule flags plus
    ``DUEL_PSEUDO_SHUFFLE`` and the passive opponent.
    """
    if demo.start is not None and demo.lines:
        rep = demo.replay(0)
        rep.responses = []
        return rep
    config = config or DuelConfig()
    deck = Deck(tuple(demo.deck["main"]), tuple(demo.deck["extra"]), (), demo.deck["name"])
    hand, order = sample_hand(deck, demo.hand_seed, config.player.starting_hand)
    if list(hand) != list(demo.hand):
        raise ValueError(f"{demo.deck['name']} hand {demo.hand_index}: the hand seed does not give the recorded hand")
    main = _order_with_hand(replace(deck, main=tuple(order)), hand)
    decks = {"a": {"name": deck.name, "main": main, "extra": list(deck.extra), "side": []},
             "b": {"name": "opponent", "main": list(PASSIVE_OPPONENT.main), "extra": [], "side": []}}  # fmt: skip
    return Replay(seed=0, first=0, rule_flags=config.rule_flags | DUEL_PSEUDO_SHUFFLE, player=asdict(config.player),
                  shuffle_decks=False, decks=decks, responses=[], environment=demo.environment,
                  seed_words=expand_seed(demo.hand_seed))  # fmt: skip


@dataclass
class OpeningResult:
    actions: list[int]  # the agent's action indices (player 0), in order
    reached: bool  # every target card on the final board
    placed: int  # target cards on the final board
    targets: int
    capped: bool  # the agent hit the step cap and the host closed the turn
    board: dict = field(repr=False, default_factory=dict)


def passive_action(point) -> int:
    """The host's closing answer (end phase, pass, no, cancel, finish; else the first action), as in convert_line."""
    kinds = [a.kind for a in point.actions]
    return next((kinds.index(k) for k in PASSIVE_KINDS if k in kinds), 0)


def play_opening(replay: Replay, agent, targets: Sequence[str], *, env: Environment | None = None, cards=None,
                 scripts=None, max_steps: int = MAX_OPENING_STEPS, player: int = DEMO_PLAYER) -> OpeningResult:  # fmt: skip
    """Let ``agent`` play turn 1 for ``player`` from ``replay``'s start; the opponent answers passively.

    ``agent`` follows the Agent protocol (``act``, optional ``observe(point, core)``). Play stops at the
    first decision of turn 2 (the demonstrations end there too); the final board is scored against
    ``targets`` as the solver's lines are.
    """
    cards = cards if cards is not None else default_cards()
    kwargs = {k: v for k, v in (("cards", cards), ("scripts", scripts)) if v is not None}
    session = DuelSession(replay.duel(env, **kwargs))
    tracker = session.tracker
    observe = getattr(agent, "observe", None)
    actions: list[int] = []
    capped = False
    try:
        while not session.done and tracker.turn < 2:
            point = session.point
            if point is None:
                break
            if observe is not None:
                observe(point, session.core)
            if point.player == player and len(actions) < max_steps:
                idx = agent.act(point)
                actions.append(idx)
            else:
                capped = capped or point.player == player
                idx = passive_action(point)
            session.act(idx)
        board = board_summary(session.core, tracker.turn, (tracker.lp[0], tracker.lp[1]))
    finally:
        session.close()
    parsed = parse_targets(targets)
    missing = board_summary_missing(board, parsed, cards, player)
    return OpeningResult(actions, not missing, len(parsed) - len(missing), len(parsed), capped, board)


def demo_player_actions(demo: Demonstration, player: int = DEMO_PLAYER, *, cards=None, scripts=None,
                        env: Environment | None = None) -> list[list[int]]:  # fmt: skip
    """Each line's action indices of ``player`` without select/unselect toggles (what a reproduction must match)."""
    out = []
    for li in range(len(demo.lines)):
        steps = line_steps(demo, li, None, cards=cards, scripts=scripts, env=env, player=player)
        undone = undone_steps(steps)
        out.append([st.action for st in steps if st.player == player and st.step not in undone])
    return out


def opening_report(demos: Sequence[Demonstration], agent_factory: Callable[[int], object], *,
                   env: Environment | None = None, cards=None, scripts=None,
                   max_steps: int = MAX_OPENING_STEPS) -> dict:  # fmt: skip
    """Free-running turn 1 on the start position of every plain record (solved or not).

    Returns totals and per-deck rows: ``reached`` (targets on the final board), ``placed`` (target cards
    placed / target cards), ``reproduced`` (the agent's actions equal one of the record's lines) and the
    solver's own result on the same hands (``solver_solved``).
    """
    config = DuelConfig.from_environment(env) if env is not None else DuelConfig()
    rows: dict[str, Counter] = {}
    per_hand = []
    for i, demo in enumerate(d for d in demos if d.variant == "plain" and d.status in ("solved", "unsolved")):
        res = play_opening(start_replay(demo, config), agent_factory(i), demo.targets, env=env, cards=cards,
                           scripts=scripts, max_steps=max_steps)  # fmt: skip
        lines = demo_player_actions(demo, cards=cards, scripts=scripts, env=env)
        reproduced = any(res.actions == ln for ln in lines)
        prefix = max((_common_prefix(res.actions, ln) / len(ln) for ln in lines if ln), default=0.0)
        solved = demo.status == "solved"
        for key in (demo.deck["name"], "total"):
            row = rows.setdefault(key, Counter())
            row["hands"] += 1
            row["solver_solved"] += solved
            row["reached"] += res.reached
            row["reached_on_solved"] += res.reached and solved
            row["reached_on_unsolved"] += res.reached and not solved
            row["placed"] += res.placed
            row["target_cards"] += res.targets
            row["reproduced"] += reproduced
            row["prefix"] += prefix
            row["capped"] += res.capped
            row["steps"] += len(res.actions)
        per_hand.append({"deck": demo.deck["name"], "hand_index": demo.hand_index, "solver": demo.status,
                         "reached": res.reached, "placed": res.placed, "targets": res.targets,
                         "reproduced": reproduced, "prefix": round(prefix, 3), "steps": len(res.actions),
                         "capped": res.capped})  # fmt: skip
    return {"totals": {k: _rates(v) for k, v in rows.items()}, "hands": per_hand}


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _rates(row: Counter) -> dict:
    n = row["hands"] or 1
    solved = row["solver_solved"]
    return {"hands": row["hands"], "solver_solved": solved, "solver_solve_rate": solved / n,
            "reached": row["reached"], "reach_rate": row["reached"] / n,
            "reach_rate_on_solved": row["reached_on_solved"] / solved if solved else None,
            "reached_on_unsolved": row["reached_on_unsolved"],
            "placed_fraction": row["placed"] / (row["target_cards"] or 1),
            "reproduced": row["reproduced"], "reproduction_rate": row["reproduced"] / n,
            "mean_prefix": row["prefix"] / n, "capped": row["capped"], "mean_steps": row["steps"] / n}  # fmt: skip


def hand_overlap(train: Iterable[Demonstration], heldout: Iterable[Demonstration]) -> int:
    """Held-out plain records whose opening hand (as a multiset, same deck) also occurs in the training records."""
    seen = {(d.deck["name"], tuple(sorted(d.hand))) for d in train}
    return sum((d.deck["name"], tuple(sorted(d.hand))) in seen for d in heldout if d.variant == "plain")


__all__ = ["BCConfig", "BCData", "DEMO_PLAYER", "MAX_OPENING_STEPS", "OpeningResult", "bc_loss", "build_dataset",
           "DemoStep", "demo_player_actions", "hand_overlap", "line_steps", "opening_report", "passive_action", "play_opening",
           "start_replay", "step_accuracy", "train_bc", "trim_padding", "undone_steps"]  # fmt: skip
