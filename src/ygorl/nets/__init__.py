"""Policy networks (M4, T4b.1 / T4b.2): card / effect encoders, board Transformer, event history, action head.

Requires the ``train`` extra (PyTorch). Specification and design choices: docs/nets.md.
"""

from ygorl.nets.batch import collate, to_tensors
from ygorl.nets.config import NetConfig
from ygorl.nets.history import (
    HistoryEncoder,
    HistoryOutput,
    HistoryState,
    LSTMHistory,
    LSTMState,
    TransformerHistory,
    TransformerState,
)
from ygorl.nets.policy import Features, PolicyNet, PolicyOutput, count_parameters
from ygorl.nets.text import TextFeatures

__all__ = [
    "Features",
    "HistoryEncoder",
    "HistoryOutput",
    "HistoryState",
    "LSTMHistory",
    "LSTMState",
    "NetConfig",
    "PolicyNet",
    "PolicyOutput",
    "TextFeatures",
    "TransformerHistory",
    "TransformerState",
    "collate",
    "count_parameters",
    "to_tensors",
]
