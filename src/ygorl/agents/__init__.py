"""Agents that play duels: the Agent protocol and baseline agents."""

from ygorl.agents.base import Agent, AgentFactory, agent_name
from ygorl.agents.greedy import GreedyAgent
from ygorl.agents.policy import PolicyAgent
from ygorl.agents.random_agent import RandomAgent

__all__ = ["Agent", "AgentFactory", "GreedyAgent", "PolicyAgent", "RandomAgent", "agent_name"]
