"""
Execution algorithms for sophisticated order execution.

This module contains various execution algorithms that minimize market impact
and optimize execution quality for large orders.

Algorithms:
- BaseAlgorithm: Abstract base class for all execution algorithms
- TWAP: Time-Weighted Average Price execution
- VWAP: Volume-Weighted Average Price execution
- Iceberg: Hidden order execution with visible quantity limits
- SmartRouter: Multi-exchange routing for optimal execution
"""

from .base_algorithm import BaseAlgorithm
from .iceberg import IcebergAlgorithm
from .smart_router import SmartOrderRouter
from .twap import TWAPAlgorithm
from .vwap import VWAPAlgorithm

__all__ = [
    "BaseAlgorithm",
    "IcebergAlgorithm",
    "SmartOrderRouter",
    "TWAPAlgorithm",
    "VWAPAlgorithm",
]
