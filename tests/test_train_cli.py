"""The training config's command line (ygorl.train.cli, #121): every TrainConfig / PPOConfig field is one flag that
round-trips, the historical flag names keep their meaning, and tools/train_ppo.py builds its config through it."""

import argparse
import importlib.util
from dataclasses import fields, replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from ygorl.train import cli  # noqa: E402
from ygorl.train.ppo import PPOConfig  # noqa: E402
from ygorl.train.trainer import TrainConfig  # noqa: E402

DECKS = ("a.ydk", "b.ydk")
# string fields without choices: a valid non-default value each
STRINGS = {"env": "v1", "text_dir": "feats", "bc_prior": "bc.pt", "init_from": "init.pt", "device": "cuda:1",
           "keep_best_by": "random", "ppo.objective": "pg", "deck_pool": str(Path("/tmp/pool.json").resolve())}  # fmt: skip
TUPLES = {"eval_opponents": ("random", "greedy", "policy:x.pt"), "pin_opponents": ("a.pt", "b.pt")}


def parse(argv, base=None):
    p = argparse.ArgumentParser()
    cli.add_arguments(p, base)
    return cli.from_args(p.parse_args(argv), DECKS, base)


def flag_keys() -> list[str]:
    return [k for f in cli._flags() for k in f.keys]


def config_keys() -> list[str]:
    top = [f.name for f in fields(TrainConfig) if f.name not in ("decks", "ppo", "net")]
    return top + ["ppo." + f.name for f in fields(PPOConfig)]


def perturbed(key: str):
    """A config with ``key`` set to a value other than its default (and every other field at its default)."""
    base = TrainConfig(decks=DECKS)
    flag = next(f for f in cli._flags() if key in f.keys)
    default = cli._get(base, key)
    if key in STRINGS:
        value = STRINGS[key]
    elif key in TUPLES:
        value = TUPLES[key]
    elif flag.choices:
        value = next(c for c in flag.choices if c != default)
    else:
        value = {bool: lambda: not default, int: lambda: (default or 0) + 3,
                 float: lambda: (default or 0.0) + 0.25}[cli._type(key)]()  # fmt: skip
    values = {k: value for k in flag.keys}
    if key == "ppo.critic_target":
        base = replace(base, complete_games=True)
    owner = key.partition(".")[0]
    if owner == "net":
        return replace(base, net={**base.net, **{k[4:]: v for k, v in values.items()}})
    if owner == "ppo":
        return replace(base, ppo=replace(base.ppo, **{k[4:]: v for k, v in values.items()}))
    return replace(base, **values)


def test_every_config_field_has_exactly_one_flag():
    keys = [k for k in flag_keys() if not k.startswith("net.")]
    assert sorted(keys) == sorted(config_keys())
    assert len(set(flag_keys())) == len(flag_keys())
    net_fields = {f.name for f in fields(cli._classes()["net"])}
    assert {k[4:] for k in flag_keys() if k.startswith("net.")} <= net_fields


@pytest.mark.parametrize("key", config_keys() + [k for k in flag_keys() if k.startswith("net.")])
def test_every_field_round_trips_through_the_command_line(key):
    cfg = perturbed(key)
    argv = cli.to_argv(cfg)
    assert argv, key
    assert parse(argv) == cfg
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg


def test_none_values_round_trip_as_zero():
    cfg = TrainConfig(decks=DECKS, stall_timeout=None, ppo=PPOConfig(target_kl=None))
    assert cli.to_argv(cfg) == ["--stall-timeout", "0", "--target-kl", "0"]
    assert parse(cli.to_argv(cfg)) == cfg


def test_no_flags_give_the_default_config():
    assert parse([]) == TrainConfig(decks=DECKS)
    assert cli.to_argv(TrainConfig(decks=DECKS)) == []


def test_historical_flags_keep_their_meaning():
    """Flag names used by run scripts (out/**/run*.sh), including those that do not match their field."""
    cfg = parse(["--envs", "64", "--entropy", "0.1", "--no-privileged", "--separate-critic", "--keep-forced",
                 "--lam", "0.7", "--critic-lam", "1.0", "--critic-deck-order", "--deck-pool", "pool.json",
                 "--evolved-share", "0.5", "--layers", "2", "--no-text", "--no-id-embedding", "--d-model", "96",
                 "--eval-opponents", "greedy", "--keep-best-by", "greedy", "--pin", "x.pt", "--pin", "y.pt",
                 "--target-kl", "0", "--minibatch", "512", "--kl-ref", "0.1", "--ema", "0.05", "--kl-prior", "0.2"])  # fmt: skip
    assert (cfg.num_envs, cfg.privileged_critic, cfg.shared_backbone, cfg.skip_forced) == (64, False, False, False)
    assert cfg.critic_deck_order and cfg.deck_pool == str(Path("pool.json").resolve()) and cfg.evolved_share == 0.5
    assert cfg.eval_opponents == ("greedy",) and cfg.pin_opponents == ("x.pt", "y.pt")
    assert cfg.net == {**TrainConfig().net, "board_layers": 2, "history_layers": 2, "card_text": False,
                       "effect_text": False, "id_embedding": False, "d_model": 96}  # fmt: skip
    p = cfg.ppo
    assert (p.entropy_coef, p.lam, p.critic_lam, p.target_kl, p.minibatch_size) == (0.1, 0.7, 1.0, None, 512)
    assert (p.kl_ref_coef, p.reference_ema, p.kl_prior_coef) == (0.1, 0.05, 0.2)


def test_parser_defaults_come_from_the_base_config():
    base = TrainConfig(num_envs=32, ppo=PPOConfig(minibatch_size=256), eval_every=0)
    cfg = parse(["--steps", "8"], base)
    assert cfg == replace(base, decks=DECKS, steps=8)
    assert cli.to_argv(cfg, base) == ["--steps", "8"]


def test_to_argv_refuses_what_the_command_line_cannot_say():
    with pytest.raises(ValueError, match="--layers"):
        cli.to_argv(TrainConfig(net={**TrainConfig().net, "board_layers": 3}))
    with pytest.raises(ValueError, match="gate_bias"):
        cli.to_argv(TrainConfig(net={**TrainConfig().net, "gate_bias": 1.0}))


def test_train_tool_passes_overlap_and_bf16():
    """--overlap and --bf16 used to be parsed and dropped."""
    spec = importlib.util.spec_from_file_location("train_ppo", Path(__file__).parents[1] / "tools" / "train_ppo.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    args = tool.build_parser().parse_args(["d.ydk", "--overlap", "--bf16", "--out", "run", "--updates", "3"])
    cfg = tool.config_from_args(args, ["d.ydk"])
    assert cfg.overlap_collect and cfg.bf16 and cfg.decks == ("d.ydk",)
    assert (args.out, args.updates) == (Path("run"), 3)
