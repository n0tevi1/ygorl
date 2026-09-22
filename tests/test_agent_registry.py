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
