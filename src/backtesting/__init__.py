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
_LAZY_IMPORTS = {
    "MonteCarloAnalyzer": ("analysis", "MonteCarloAnalyzer"),
    "WalkForwardAnalyzer": ("analysis", "WalkForwardAnalyzer"),
    "PerformanceAttributor": ("attribution", "PerformanceAttributor"),
    "DataReplayManager": ("data_replay", "DataReplayManager"),
    "ReplayMode": ("data_replay", "ReplayMode"),
    "BacktestConfig": ("engine", "BacktestConfig"),
    "BacktestEngine": ("engine", "BacktestEngine"),
    "BacktestResult": ("engine", "BacktestResult"),
    "BacktestMetrics": ("metrics", "BacktestMetrics"),
    "MetricsCalculator": ("metrics", "MetricsCalculator"),
    "SimulationConfig": ("simulator", "SimulationConfig"),
    "TradeSimulator": ("simulator", "TradeSimulator"),
    "BacktestRequest": ("service", "BacktestRequest"),
    "BacktestService": ("service", "BacktestService"),
    "BacktestController": ("controller", "BacktestController"),
    "BacktestRepository": ("repository", "BacktestRepository"),
    "BacktestFactory": ("factory", "BacktestFactory"),
    "get_backtest_service": ("di_registration", "get_backtest_service"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = __import__(f"{__name__}.{module_name}", fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BacktestConfig",
    "BacktestController",
    "BacktestEngine",
    "BacktestFactory",
    "BacktestMetrics",
    "BacktestRepository",
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
    "get_backtest_service",
]

# Version information
__version__ = "1.0.0"
__author__ = "Trading Bot Framework"
__description__ = "Advanced Backtesting Framework with ML Support (P-013C)"
