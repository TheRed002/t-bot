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


# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "MonteCarloAnalyzer":
        from .analysis import MonteCarloAnalyzer

        return MonteCarloAnalyzer
    elif name == "WalkForwardAnalyzer":
        from .analysis import WalkForwardAnalyzer

        return WalkForwardAnalyzer
    elif name == "PerformanceAttributor":
        from .attribution import PerformanceAttributor

        return PerformanceAttributor
    elif name == "DataReplayManager":
        from .data_replay import DataReplayManager

        return DataReplayManager
    elif name == "ReplayMode":
        from .data_replay import ReplayMode

        return ReplayMode
    elif name == "BacktestConfig":
        from .engine import BacktestConfig

        return BacktestConfig
    elif name == "BacktestEngine":
        from .engine import BacktestEngine

        return BacktestEngine
    elif name == "BacktestResult":
        from .engine import BacktestResult

        return BacktestResult
    elif name == "BacktestMetrics":
        from .metrics import BacktestMetrics

        return BacktestMetrics
    elif name == "MetricsCalculator":
        from .metrics import MetricsCalculator

        return MetricsCalculator
    elif name == "SimulationConfig":
        from .simulator import SimulationConfig

        return SimulationConfig
    elif name == "TradeSimulator":
        from .simulator import TradeSimulator

        return TradeSimulator
    elif name == "BacktestRequest":
        from .service import BacktestRequest

        return BacktestRequest
    elif name == "BacktestService":
        from .service import BacktestService

        return BacktestService
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestRequest",
    "BacktestResult",
    "BacktestService",
    "DataReplayManager",
    "MetricsCalculator",
    "MonteCarloAnalyzer",
    "PerformanceAttributor",
    "ReplayMode",
    "SimulationConfig",
    "TradeSimulator",
    "WalkForwardAnalyzer",
]

# Version information
__version__ = "1.0.0"
__author__ = "Trading Bot Framework"
__description__ = "Advanced Backtesting Framework with ML Support (P-013C)"
