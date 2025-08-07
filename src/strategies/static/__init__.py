"""
Static trading strategies.

This module contains static trading strategies that have been implemented in P-012.
"""

# Import strategy classes implemented in P-012
from .mean_reversion import MeanReversionStrategy
from .trend_following import TrendFollowingStrategy  
from .breakout import BreakoutStrategy

__all__ = [
    # Static strategies (implemented in P-012)
    "MeanReversionStrategy",
    "TrendFollowingStrategy", 
    "BreakoutStrategy"
]

# Version information
__version__ = "1.0.0"
__description__ = "Static trading strategies for algorithmic trading" 