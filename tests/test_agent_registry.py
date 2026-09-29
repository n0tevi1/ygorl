"""Agent registry: build agents from command-line specs (`name` or `name:arg`)."""

import pytest

from ygorl.agents import RandomAgent, available_agents, make_agent, register_agent


class Point:
    def __init__(self, n):
        self.actions = list(range(n))


def test_random_is_registered_and_seeded():
    assert "random" in available_agents()
    a, b = make_agent("random", seed=7), make_agent("random", seed=7)
    assert isinstance(a, RandomAgent)
    assert [a.act(Point(10)) for _ in range(20)] == [b.act(Point(10)) for _ in range(20)]
    with pytest.raises(ValueError, match="random takes no argument"):
        make_agent("random:x")


def test_unknown_agent_lists_the_available_ones():
    with pytest.raises(ValueError, match=r"unknown agent 'nope'.*random"):
        make_agent("nope")


def test_register_agent_with_an_argument():
    calls = []

    def factory(arg, seed):
        calls.append((arg, seed))
        return RandomAgent(seed)

    register_agent("test-ckpt", factory, "a test entry taking a path")
    try:
        assert isinstance(make_agent("test-ckpt:runs/a.pt", seed=3), RandomAgent)
        make_agent("test-ckpt", seed=4)
        assert calls == [("runs/a.pt", 3), (None, 4)]
        assert available_agents()["test-ckpt"] == "a test entry taking a path"
        with pytest.raises(ValueError, match="already registered"):
            register_agent("test-ckpt", factory)
    finally:
        register_agent("test-ckpt", None)
    assert "test-ckpt" not in available_agents()


def test_agent_factory_is_a_named_picklable_factory():
    import pickle

    from ygorl.agents import agent_factory, agent_name

    factory = agent_factory("greedy")
    assert agent_name(factory) == "greedy"
    clone = pickle.loads(pickle.dumps(factory))
    assert agent_name(clone) == "greedy"
    a, b = factory(5), clone(5)
    assert type(a).__name__ == "GreedyAgent" and type(a) is type(b)
    with pytest.raises(ValueError, match="unknown agent 'nope'"):
        agent_factory("nope")  # a bad spec fails when the factory is made, not in a worker
    with pytest.raises(ValueError, match="random takes no argument"):
        agent_factory("random:x")


def test_parse_plain_and_argument_specs():
    from ygorl.agents.registry import parse_spec

    p = parse_spec("greedy")
    assert (p.spec, p.kind, p.arg, p.checkpoint, p.inner, p.is_policy, p.policy) == (
        "greedy", "greedy", None, None, None, False, None)  # fmt: skip
    assert parse_spec("random:x").arg == "x"  # the factory rejects it, the parser does not
    with pytest.raises(ValueError, match=r"unknown agent 'nope'.*random"):
        parse_spec("nope:x")


@pytest.mark.parametrize(
    "spec, checkpoint, greedy, temperature",
    [
        ("policy:out/p.pt", "out/p.pt", False, 1.0),
        ("policy:out/p.pt@greedy", "out/p.pt", True, 1.0),
        ("policy:out/p.pt@t=0.5", "out/p.pt", False, 0.5),
        ("policy:out/p.pt@t=2@greedy", "out/p.pt", True, 2.0),
        ("policy-greedy:out/p.pt", "out/p.pt", True, 1.0),
        ("policy:C:/runs/p.pt@t=0.25", "C:/runs/p.pt", False, 0.25),  # only the first ':' splits
    ],
)
def test_parse_policy_sampling_suffixes(spec, checkpoint, greedy, temperature):
    from ygorl.agents.registry import parse_spec

    p = parse_spec(spec)
    assert (p.checkpoint, p.greedy, p.temperature, p.is_policy, p.inner) == (checkpoint, greedy, temperature, True,
                                                                              None)  # fmt: skip
    assert p.policy is p and p.spec == spec


def test_a_path_may_contain_at(tmp_path):
    from ygorl.agents.registry import parse_spec

    ckpt = tmp_path / "run@3.pt"
    ckpt.write_bytes(b"")
    p = parse_spec(f"policy:{ckpt}@greedy")
    assert p.checkpoint == str(ckpt) and p.greedy
    with pytest.raises(ValueError, match="unknown policy option 'bogus'"):
        parse_spec("policy:out/p.pt@bogus")  # not a file: the suffix must be an option


@pytest.mark.parametrize(
    "spec, message",
    [
        ("policy", "policy needs a checkpoint"),
        ("policy:", "policy needs a checkpoint"),
        ("policy-greedy", "policy needs a checkpoint"),
        ("policy:p.pt@t=x", "bad policy temperature 't=x'"),
        ("policy:p.pt@t=0", "must be positive"),
        ("lethal", "lethal needs an inner agent"),
        ("lethal:", "lethal needs an inner agent"),
        ("lethal:nope", "unknown agent 'nope'"),
        ("lethal:policy", "policy needs a checkpoint"),
    ],
)
def test_parse_errors(spec, message):
    from ygorl.agents.registry import parse_spec

    with pytest.raises(ValueError, match=message):
        parse_spec(spec)
    with pytest.raises(ValueError, match=message):
        make_agent(spec)  # the same errors when building


def test_parse_wrappers():
    from ygorl.agents.registry import parse_spec

    p = parse_spec("lethal:policy:out/p.pt@t=0.5")
    assert (p.kind, p.arg, p.checkpoint, p.is_policy) == ("lethal", "policy:out/p.pt@t=0.5", None, False)
    assert p.inner == parse_spec("policy:out/p.pt@t=0.5")
    assert (p.policy.checkpoint, p.policy.temperature) == ("out/p.pt", 0.5)
    twice = parse_spec("lethal:lethal:policy-greedy:p.pt")
    assert twice.inner.inner.kind == "policy-greedy" and twice.policy.greedy
    assert parse_spec("lethal:greedy").policy is None

    register_agent("test-wrap", lambda arg, seed: make_agent(arg, seed), wraps=True)
    try:
        assert parse_spec("test-wrap:lethal:random").inner.inner.kind == "random"
        assert isinstance(make_agent("test-wrap:random", seed=1), RandomAgent)
    finally:
        register_agent("test-wrap", None)


def test_policy_spec_round_trips():
    from ygorl.agents.registry import parse_spec, policy_spec

    assert policy_spec("out/p.pt") == "policy:out/p.pt"
    for greedy, temperature in ((False, 1.0), (True, 1.0), (False, 0.3), (True, 2.0)):
        p = parse_spec(policy_spec("out/p.pt", greedy=greedy, temperature=temperature))
        assert (p.checkpoint, p.greedy, p.temperature) == ("out/p.pt", greedy, temperature)
    with pytest.raises(ValueError, match="must be positive"):
        policy_spec("out/p.pt", temperature=0)


def test_parse_named_spec():
    from ygorl.agents.registry import parse_named_spec

    assert parse_named_spec("greedy") == ("greedy", "greedy")
    assert parse_named_spec("g2=greedy") == ("g2", "greedy")
    assert parse_named_spec("policy:out/p.pt@t=0.5") == ("policy:out/p.pt@t=0.5", "policy:out/p.pt@t=0.5")
    assert parse_named_spec("best=policy:out/p.pt@t=1") == ("best", "policy:out/p.pt@t=1")
    with pytest.raises(ValueError, match="empty spec"):
        parse_named_spec("x=")
    for bad in ("=greedy", ".x=greedy", "a/b=greedy"):
        with pytest.raises(ValueError, match="agent name"):
            parse_named_spec(bad)
