"""Agents that play duels: the Agent protocol and baseline agents."""

from ygorl.agents.base import Agent, AgentFactory, agent_name
from ygorl.agents.greedy import GreedyAgent
from ygorl.agents.policy import PolicyAgent
from ygorl.agents.random_agent import RandomAgent

AGENTS: dict[str, AgentFactory] = {"random": RandomAgent, "greedy": GreedyAgent}
"""Baseline agents by name (for tools and the CLI); each value is a picklable factory ``seed -> agent``."""

__all__ = ["AGENTS", "Agent", "AgentFactory", "GreedyAgent", "PolicyAgent", "RandomAgent", "agent_name"]
