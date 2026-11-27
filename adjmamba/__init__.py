"""Bidirectional dynamic-graph Mamba package."""

from .block import BidirectionalDynamicGraphBlock
from .config import AdjMambaConfig, PoolingMode
from .graph import DynamicGraphConstructor
from .mamba import ConditionalMambaBlock
from .model import AdjMamba
from .pooling import AttentionPool, pool_features
from .scan import parallel_scan

__all__ = [
    "AdjMamba",
    "AdjMambaConfig",
    "AttentionPool",
    "BidirectionalDynamicGraphBlock",
    "ConditionalMambaBlock",
    "DynamicGraphConstructor",
    "PoolingMode",
    "parallel_scan",
    "pool_features",
]


if __name__ == "__main__":
    print("adjmamba package exports:", ", ".join(__all__))
