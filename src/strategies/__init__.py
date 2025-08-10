"""
Strategy framework for trading bot.

This module provides the foundation for all trading strategies with unified interface,
factory pattern for dynamic instantiation, and configuration management.

CRITICAL: All strategy implementations must inherit from BaseStrategy.
"""

# MANDATORY: Import from P-011
# MANDATORY: Import from P-001
from src.core.types import StrategyConfig, StrategyMetrics, StrategyStatus, StrategyType
from src.strategies.base import BaseStrategy
from src.strategies.config import StrategyConfigurationManager
from src.strategies.factory import StrategyFactory

__all__ = [
    # Base strategy interface
    "BaseStrategy",
    # Factory and management
    "StrategyFactory",
    "StrategyConfigurationManager",
    # Types
    "StrategyConfig",
    "StrategyStatus",
    "StrategyMetrics",
    "StrategyType",
]

# Version information
__version__ = "1.0.0"
__author__ = "Trading Bot Team"
__description__ = "Strategy framework for algorithmic trading"
