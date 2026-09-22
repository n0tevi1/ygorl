"""Agents that play duels: the Agent protocol, baseline agents and the name registry."""

from ygorl.agents.base import Agent, AgentFactory, agent_name
from ygorl.agents.greedy import GreedyAgent
from ygorl.agents.policy import PolicyAgent
from ygorl.agents.random_agent import RandomAgent
from ygorl.agents.registry import AgentSpec, agent_factory, available_agents, make_agent, register_agent

AGENTS: dict[str, AgentFactory] = {"random": RandomAgent, "greedy": GreedyAgent}
"""Baseline agents by name (for tools and the CLI); each value is a picklable factory ``seed -> agent``."""

__all__ = ["AGENTS", "Agent", "AgentFactory", "AgentSpec", "GreedyAgent", "PolicyAgent", "RandomAgent", "agent_factory",
           "agent_name", "available_agents", "make_agent", "register_agent"]  # fmt: skip
