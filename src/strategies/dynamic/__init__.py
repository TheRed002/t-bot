"""
Dynamic and Adaptive Strategies Module - REFACTORED Day 13

This module implements adaptive strategies that adjust to market conditions using
enhanced service layer architecture with centralized indicator calculations,
state persistence, and comprehensive metrics tracking.

Refactoring improvements:
- Service layer integration for all strategies
- Centralized technical indicator calculations
- Enhanced state persistence and recovery
- Comprehensive metrics and monitoring
- Removed direct data access patterns

CRITICAL: All strategies follow the perfect architecture from Day 12.
"""

# Refactored implementations (enhanced with service layer)
from .adaptive_momentum import AdaptiveMomentumStrategy
from .volatility_breakout import VolatilityBreakoutStrategy

__all__ = [
    "AdaptiveMomentumStrategy",
    "VolatilityBreakoutStrategy",
]
