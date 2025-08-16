"""
Static trading strategies.

This module contains static trading strategies that have been implemented in P-012, P-013A, and P-013B.
"""

# Import strategy classes implemented in P-012
from .arbitrage_scanner import ArbitrageOpportunity
from .breakout import BreakoutStrategy

# Import arbitrage strategy classes implemented in P-013A
from .cross_exchange_arbitrage import CrossExchangeArbitrageStrategy
from .inventory_manager import InventoryManager

# Import market making strategy classes implemented in P-013B
from .market_making import MarketMakingStrategy
from .mean_reversion import MeanReversionStrategy
from .spread_optimizer import SpreadOptimizer
from .trend_following import TrendFollowingStrategy
from .triangular_arbitrage import TriangularArbitrageStrategy

__all__ = [
    "ArbitrageOpportunity",
    "BreakoutStrategy",
    # Arbitrage strategies (implemented in P-013A)
    "CrossExchangeArbitrageStrategy",
    "InventoryManager",
    # Market making strategies (implemented in P-013B)
    "MarketMakingStrategy",
    # Static strategies (implemented in P-012)
    "MeanReversionStrategy",
    "SpreadOptimizer",
    "TrendFollowingStrategy",
    "TriangularArbitrageStrategy",
]

# Version information
__version__ = "1.0.0"
__description__ = "Static trading strategies for algorithmic trading including arbitrage and market making strategies"
