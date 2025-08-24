"""
Backtesting Framework (P-013C).

This module provides comprehensive backtesting capabilities including:
- Historical market replay engine
- Walk-forward analysis
- Monte Carlo simulations
- Performance attribution
- Risk-adjusted metrics calculation

CRITICAL: This module integrates with P-001 (types, exceptions, config),
P-002A (error handling), P-007A (utils), and P-008/P-009 (risk management).
"""

from .analysis import MonteCarloAnalyzer, WalkForwardAnalyzer
from .attribution import PerformanceAttributor
from .data_replay import DataReplayManager, ReplayMode
from .engine import BacktestConfig, BacktestEngine, BacktestResult
from .metrics import BacktestMetrics, MetricsCalculator
from .simulator import SimulationConfig, TradeSimulator

# Note: BacktestService import disabled temporarily due to service dependencies

__all__ = [
    "BacktestConfig",
    # Core engine
    "BacktestEngine",
    # Metrics
    "BacktestMetrics",
    "BacktestResult",
    # Data replay
    "DataReplayManager",
    "MetricsCalculator",
    "MonteCarloAnalyzer",
    # Attribution
    "PerformanceAttributor",
    "ReplayMode",
    "SimulationConfig",
    # Simulation
    "TradeSimulator",
    # Analysis
    "WalkForwardAnalyzer",
    # Service Layer (temporarily disabled)
    # "BacktestService",
    # "BacktestRequest",
]

# Version information
__version__ = "1.0.0"
__author__ = "Trading Bot Framework"
__description__ = "Advanced Backtesting Framework with ML Support (P-013C)"
