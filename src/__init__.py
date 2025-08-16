"""
Trading Bot Framework - Main Package

This package contains the core trading bot framework implementation.
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Team"
__description__ = "Comprehensive algorithmic trading platform"

# Core framework imports
from .core.config import Config, DatabaseConfig, SecurityConfig
from .core.exceptions import (
    DataError,
    ExchangeError,
    ExecutionError,
    ModelError,
    RiskManagementError,
    SecurityError,
    StateConsistencyError,
    TradingBotError,
    ValidationError,
)
from .core.logging import (
    PerformanceMonitor,
    correlation_context,
    get_logger,
    get_secure_logger,
    log_async_performance,
    log_performance,
    setup_logging,
)
from .core.types import (
    MarketData,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderType,
    Position,
    Signal,
    SignalDirection,
    TradingMode,
)

__all__ = [
    # Configuration
    "Config",
    "DataError",
    "DatabaseConfig",
    "ExchangeError",
    "ExecutionError",
    "MarketData",
    "ModelError",
    "OrderRequest",
    "OrderResponse",
    "OrderSide",
    "OrderType",
    "PerformanceMonitor",
    "Position",
    "RiskManagementError",
    "SecurityConfig",
    "SecurityError",
    "Signal",
    "SignalDirection",
    "StateConsistencyError",
    # Exceptions
    "TradingBotError",
    # Types
    "TradingMode",
    "ValidationError",
    "correlation_context",
    # Logging
    "get_logger",
    "get_secure_logger",
    "log_async_performance",
    "log_performance",
    "setup_logging",
]
