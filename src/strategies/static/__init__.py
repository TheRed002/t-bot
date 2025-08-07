"""
Static trading strategies.

This module contains static trading strategies that have been implemented in P-012 and P-013A.
"""

# Import strategy classes implemented in P-012
from .mean_reversion import MeanReversionStrategy
from .trend_following import TrendFollowingStrategy  
from .breakout import BreakoutStrategy

# Import arbitrage strategy classes implemented in P-013A
from .cross_exchange_arbitrage import CrossExchangeArbitrageStrategy
from .triangular_arbitrage import TriangularArbitrageStrategy
from .arbitrage_scanner import ArbitrageOpportunity

__all__ = [
    # Static strategies (implemented in P-012)
    "MeanReversionStrategy",
    "TrendFollowingStrategy", 
    "BreakoutStrategy",
    
    # Arbitrage strategies (implemented in P-013A)
    "CrossExchangeArbitrageStrategy",
    "TriangularArbitrageStrategy", 
    "ArbitrageOpportunity"
]

# Version information
__version__ = "1.0.0"
__description__ = "Static trading strategies for algorithmic trading including arbitrage strategies" 