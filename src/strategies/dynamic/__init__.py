"""
Dynamic and Adaptive Strategies Module

This module implements adaptive strategies that adjust to market conditions.
These strategies integrate with market regime detection and adaptive risk management.

CRITICAL: All strategies must inherit from BaseStrategy and follow the exact interface.
"""

from .adaptive_momentum import AdaptiveMomentumStrategy
from .volatility_breakout import VolatilityBreakoutStrategy

__all__ = ["AdaptiveMomentumStrategy", "VolatilityBreakoutStrategy"]
