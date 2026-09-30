"""The training config's command line: every :class:`TrainConfig` / :class:`PPOConfig` field is one flag.

Each option is declared once. The config dataclasses give the field's type and default (or the base config a tool
passes, e.g. tools/bench_train.py's smaller rollout); :data:`FLAGS` adds what only the command line needs: the flag
name, help, negation, choices. Three functions use the table::

    add_arguments(parser, base)        # argparse groups, defaults taken from ``base`` (default TrainConfig())
    from_args(args, decks, base)       # parsed namespace -> TrainConfig
    to_argv(cfg, base)                 # TrainConfig -> the flags that rebuild it (fields equal to base left out)

The network switches set keys of ``TrainConfig.net`` (NetConfig overrides): only keys whose value differs from
``base.net`` (or NetConfig's default) are written. Decks are the tools' positional arguments, not flags. A new config
field needs a row here; tests/test_train_cli.py fails until it has one.
"""

from __future__ import annotations

import argparse
import typing
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class Flag:
    flag: str  # "--envs"; argparse's dest follows from it (args.envs)
    keys: tuple[str, ...]  # fields it sets, all to the same value: "num_envs", "ppo.lam", "net.board_layers"
    help: str | None = None
    negate: bool = False  # a switch that sets its bool field(s) to False ("--no-privileged")
    choices: tuple[str, ...] | None = None
    metavar: str | None = None
    zero_none: bool = False  # 0 on the command line means None ("off")
    listing: str | None = None  # tuple fields: "csv" (one comma-separated value) or "append" (repeatable flag)
    path: bool = False  # stored as an absolute path

    @property
    def dest(self) -> str:
        return self.flag[2:].replace("-", "_")


def _f(flag: str, keys: str | tuple[str, ...], help: str | None = None, **kw) -> Flag:
    return Flag(flag, (keys,) if isinstance(keys, str) else keys, help, **kw)


# (argparse group title, flags); None = the parser's own options
FLAGS: tuple[tuple[str | None, tuple[Flag, ...]], ...] = (
    (None, (
        _f("--pairings", "pairings", "deck pairings to sample (cross: distinct decks only)",
           choices=("all", "cross", "mirror")),
        _f("--deck-pool", "deck_pool", "evolved decks added while training runs (ygorl-deck-pool JSON, re-read every "
           "--deck-pool-every updates)", metavar="MANIFEST", path=True),
        _f("--deck-pool-every", "deck_pool_every", "updates between manifest re-reads"),
        _f("--evolved-share", "evolved_share", "share of deals with an evolved deck"),
        _f("--evolved-power", "evolved_power",
           "evolved decks are drawn with weight x (1 - p) ** power, p the policy's score piloting it"),
        _f("--env", "env", "environment (rules; stamped into checkpoints)", metavar="PATH|VERSION"),
        _f("--log-games", "log_games", "append one record per finished game (decks, first player, winner, turns, "
           "reason, update, environment) to RUN/games.jsonl.gz"),
    )),
    ("environment and rollout", (
        _f("--envs", "num_envs", "environment slots = rollout columns B"),
        _f("--env-threads", "env_threads", "C++ worker threads"),
        _f("--steps", "steps", "rows per column per rollout, T"),
        _f("--min-batch", "min_batch", "ready decisions per forward pass (default: --envs // 2)"),
        _f("--event-length", "event_length", "event tokens per observation"),
        _f("--keep-forced", "skip_forced",
           "also make rows of decisions with a single legal action (default: played in C++, no rows)", negate=True),
        _f("--max-turns", "max_turns", "turn limit of a duel (default: the environment's)"),
        _f("--max-decisions", "max_decisions", "decision limit of a duel (default: the environment's)"),
        _f("--stall-timeout", "stall_timeout", "seconds without any environment event before the run stops with a "
           "checkpoint; 0 = wait forever", metavar="SECONDS", zero_none=True),
    )),
    ("network", (
        _f("--d-model", "net.d_model"),
        _f("--layers", ("net.board_layers", "net.history_layers"), "board and history Transformer layers"),
        _f("--history", "net.history", choices=("transformer", "lstm", "none")),
        _f("--no-id-embedding", "net.id_embedding", "drop the per-card ID embedding", negate=True),
        _f("--text-dir", "text_dir", "card feature directory: frozen text tables and/or card_facts.npz (docs/nets.md)"),
        _f("--card-facts", "net.card_facts", "use card_facts.npz in --text-dir (experimental)"),
        _f("--no-text", ("net.card_text", "net.effect_text"), "ignore the text tables in --text-dir", negate=True),
        _f("--id-dropout", "net.id_dropout", "training: drop each card's ID embedding"),
        _f("--separate-critic", "shared_backbone", "critic gets its own trunk", negate=True),
        _f("--critic-deck-order", "critic_deck_order",
           "the privileged critic also sees both players' next 10 draws (docs/encoding.md)"),
        _f("--no-privileged", "privileged_critic", "non-privileged critic (ablation)", negate=True),
        _f("--privileged-dim", "privileged_dim", "width of the critic's privileged (opponent ground truth) features"),
        _f("--critic-hidden", "critic_hidden", "hidden width of the critic heads"),
    )),
    ("PPO", (
        _f("--objective", "ppo.objective", "policy objective (registered: ppo_clip, pg)"),
        _f("--estimator", "ppo.estimator", choices=("vrpo", "gae")),
        _f("--vrpo-mode", "ppo.vrpo_mode",
           "VRPO advantage: 'return' (Q-boosted with lambda returns) or 'critic'", choices=("return", "critic")),
        _f("--gamma", "ppo.gamma", "discount"),
        _f("--lam", "ppo.lam", "lambda of the advantage estimate"),
        _f("--critic-lam", "ppo.critic_lam",
           "lambda of the critic's Q / V targets only (default: --lam); 1.0 = game results"),
        _f("--clip", "ppo.clip", "PPO ratio clip"),
        _f("--entropy", "ppo.entropy_coef", "entropy coefficient (design: 0.05-0.2)"),
        _f("--kl-ref", "ppo.kl_ref_coef", "KL coefficient to the EMA reference"),
        _f("--ema", "ppo.reference_ema", "reference EMA rate per update"),
        _f("--q-coef", "ppo.q_coef", "Q-head loss coefficient"),
        _f("--v-coef", "ppo.v_coef", "V-head loss coefficient"),
        _f("--lr", "ppo.lr"),
        _f("--adam-eps", "ppo.adam_eps"),
        _f("--max-grad-norm", "ppo.max_grad_norm"),
        _f("--epochs", "ppo.epochs"),
        _f("--minibatch", "ppo.minibatch_size"),
        _f("--adv-norm", "ppo.adv_norm", "advantage normalization over the whole rollout",
           choices=("none", "standard", "scale")),
        _f("--bc-prior", "bc_prior", "policy (e.g. BC) or PPO checkpoint used as a KL prior; same card vocab",
           metavar="CKPT"),
        _f("--kl-prior", "ppo.kl_prior_coef", "KL coefficient to the BC prior"),
        _f("--kl-prior-turns", "ppo.kl_prior_turns",
           "apply the prior KL only to the turn player's decisions up to this turn (0: all)"),
        _f("--target-kl", "ppo.target_kl",
           "stop an update's remaining minibatches once one exceeds 1.5x this approx_kl; 0 = off", zero_none=True),
        _f("--init-from", "init_from",
           "initialize the actor from a policy (e.g. BC) or PPO checkpoint with the same network config",
           metavar="CKPT"),
        _f("--critic-warmup", "critic_warmup", "train only the critic (policy frozen) for up to N updates at the "
           "start, until its Q explained variance averages --critic-warmup-ev over 5 updates; for warm starts whose "
           "critic is fresh (0 = off)", metavar="N"),
        _f("--critic-warmup-ev", "critic_warmup_ev", "Q explained variance that ends the critic warm-up",
           metavar="EV"),
        _f("--turn-discount", "turn_discount", "speed pressure: a decided game's terminal reward is +/- G ** turns "
           "(1.0 = off; a diagnostic arm, design T6 vs C3)", metavar="G"),
    )),
    ("league and evaluation", (
        _f("--selfplay-fraction", "selfplay_fraction", "deals against the current policy; the rest against the pool"),
        _f("--pool-size", "pool_size"),
        _f("--pin", "pin_opponents",
           "keep this policy / PPO checkpoint in the opponent pool for the whole run (repeatable)",
           metavar="CKPT", listing="append"),
        _f("--pinned-share", "pinned_share", "share of pool games against pinned opponents"),
        _f("--snapshot-every", "snapshot_every"),
        _f("--pool-sampling", "pool_sampling",
           "pfsp (design I1): draw pool opponents by (1 - learner win rate) ** --pfsp-power",
           choices=("uniform", "pfsp")),
        _f("--pfsp-power", "pfsp_power"),
        _f("--snapshot-min-win-rate", "snapshot_min_win_rate",
           "a due snapshot joins the pool only if the learner scored above this against the pool"),
        _f("--snapshot-min-games", "snapshot_min_games"),
        _f("--checkpoint-every", "checkpoint_every"),
        _f("--eval-every", "eval_every"),
        _f("--eval-pairs", "eval_pairs", "paired seeds per deck pairing and baseline"),
        _f("--eval-pairings", "eval_pairings",
           "evaluate on a fixed sample of this many training pairings (0 = all; for large deck pools)"),
        _f("--eval-opponents", "eval_opponents", "comma-separated baseline agents", listing="csv"),
        _f("--keep-best-by", "keep_best_by", "the --eval-opponents baseline that picks best.pt"),
        _f("--eval-workers", "eval_workers"),
        _f("--eval-greedy-policy", "eval_greedy_policy",
           "evaluate the argmax policy (policy-greedy) instead of sampling"),
        _f("--seed", "seed"),
        _f("--device", "device", "PyTorch device of the learner and acting network (cpu, cuda)"),
        _f("--overlap", "overlap_collect",
           "experimental: collect the next rollout while updating (one update stale)"),
        _f("--bf16", "bf16", "experimental: bf16 autocast on a GPU"),
        _f("--torch-threads", "torch_threads", "PyTorch threads during updates"),
        _f("--collect-threads", "collect_threads", "PyTorch threads while collecting"),
    )),
)  # fmt: skip


def add_arguments(parser: argparse.ArgumentParser, base=None) -> None:
    """Add every config flag to ``parser`` (groups as in :data:`FLAGS`), with defaults from ``base``."""
    base = _base(base)
    for title, flags in FLAGS:
        group = parser if title is None else parser.add_argument_group(title)
        for f in flags:
            default, tp = _get(base, f.keys[0]), _type(f.keys[0])
            if tp is bool:
                group.add_argument(f.flag, action="store_true", default=default != f.negate, help=f.help)
                continue
            kw = {"choices": f.choices, "metavar": f.metavar}
            if f.listing == "append":
                kw.update(action="append", default=list(default))
            elif f.listing == "csv":
                kw.update(default=",".join(default))
            else:
                kw.update(type=tp, default=0 if default is None and f.zero_none else default)
            shown = kw["default"]
            if shown not in (None, []) and "(default" not in (f.help or ""):
                kw["help"] = f"{f.help} (default {shown})" if f.help else f"default {shown}"
            else:
                kw["help"] = f.help
            group.add_argument(f.flag, **kw)


def from_args(args: argparse.Namespace, decks, base=None):
    """The :class:`TrainConfig` of parsed ``args`` (from a parser :func:`add_arguments` built) and ``decks``."""
    base = _base(base)
    values = {}
    for f in _flags():
        value = getattr(args, f.dest)
        if f.negate:
            value = not value
        elif f.zero_none and value == 0:
            value = None
        elif f.listing == "csv":
            value = tuple(s for s in value.split(",") if s)
        elif f.listing == "append":
            value = tuple(value)
        elif f.path and value is not None:
            value = str(Path(value).resolve())
        values.update(dict.fromkeys(f.keys, value))
    top = {k: v for k, v in values.items() if "." not in k}
    ppo = {k[4:]: v for k, v in values.items() if k.startswith("ppo.")}
    net = dict(base.net)
    for key, value in values.items():
        if key.startswith("net.") and value != _get(base, key):
            net[key[4:]] = value
    return replace(base, **top, decks=tuple(str(d) for d in decks), ppo=replace(base.ppo, **ppo), net=net)


def to_argv(cfg, base=None) -> list[str]:
    """Flags that rebuild ``cfg`` through :func:`from_args` (with the same decks); fields equal to ``base``'s are
    left out. Raises ValueError for a value the command line cannot express (e.g. different board / history
    layers, a net key without a flag)."""
    base = _base(base)
    argv = []
    for f in _flags():
        value = _get(cfg, f.keys[0])
        if any(_get(cfg, k) != value for k in f.keys[1:]):
            raise ValueError(f"{f.flag} sets {', '.join(f.keys)} to one value; the config has different ones")
        if value == _get(base, f.keys[0]) and all(_get(base, k) == value for k in f.keys[1:]):
            continue
        if isinstance(value, bool):
            if value is f.negate:
                raise ValueError(f"{f.flag} cannot set {f.keys[0]} to {value}")
            argv.append(f.flag)
        elif value is None:
            if not f.zero_none:
                raise ValueError(f"{f.flag} cannot set {f.keys[0]} to None")
            argv += [f.flag, "0"]
        elif f.listing == "append":
            for item in value:
                argv += [f.flag, item]
        else:
            argv += [f.flag, ",".join(value) if f.listing == "csv" else str(value)]
    flagged = {k[4:] for f in _flags() for k in f.keys if k.startswith("net.")}
    if extra := {k for k in cfg.net.keys() - flagged if cfg.net[k] != _get(base, "net." + k)}:
        raise ValueError(f"net keys without a flag: {sorted(extra)}")
    return argv


# ------------------------------------------------------------------------------------------ internals


def _flags() -> list[Flag]:
    return [f for _, flags in FLAGS for f in flags]


def _base(base):
    if base is not None:
        return base
    from ygorl.train.trainer import TrainConfig

    return TrainConfig()


def _classes() -> dict[str, type]:
    from ygorl.nets.config import NetConfig
    from ygorl.train.ppo import PPOConfig
    from ygorl.train.trainer import TrainConfig

    return {"": TrainConfig, "ppo": PPOConfig, "net": NetConfig}


def _get(cfg, key: str):
    """``cfg``'s value of a flag key; a net key missing from ``cfg.net`` has NetConfig's default."""
    owner, _, name = key.rpartition(".")
    if owner == "net":
        if name in cfg.net:
            return cfg.net[name]
        default = next(f.default for f in fields(_classes()["net"]) if f.name == name)
        assert default is not MISSING, name
        return default
    return getattr(cfg.ppo if owner == "ppo" else cfg, name)


def _type(key: str) -> type:
    """The scalar type of a key's field: ``X | None`` -> X, ``tuple[X, ...]`` -> X, ``Literal["a", ...]`` -> str."""
    owner, _, name = key.rpartition(".")
    tp = typing.get_type_hints(_classes()[owner])[name]
    while typing.get_args(tp):
        args = [a for a in typing.get_args(tp) if a is not type(None) and a is not Ellipsis]
        tp = type(args[0]) if typing.get_origin(tp) is typing.Literal else args[0]
    return tp
