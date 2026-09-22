"""Agents that play duels: the Agent protocol, baseline agents and the name registry."""

from ygorl.agents.base import Agent
from ygorl.agents.random_agent import RandomAgent
from ygorl.agents.registry import available_agents, make_agent, register_agent

__all__ = ["Agent", "RandomAgent", "available_agents", "make_agent", "register_agent"]
