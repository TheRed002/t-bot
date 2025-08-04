"""
Trading Bot Framework - Main Package

This package contains the core trading bot framework implementation.
"""

__version__ = "2.0.0"
__author__ = "Trading Bot Team"
__description__ = "Comprehensive algorithmic trading platform"

# Core framework imports
from .core.types import (
    TradingMode, SignalDirection, OrderSide, OrderType,
    Signal, MarketData, OrderRequest, OrderResponse, Position
)

from .core.config import Config, DatabaseConfig, SecurityConfig

from .core.exceptions import (
    TradingBotError, ExchangeError, RiskManagementError,
    ValidationError, ExecutionError, ModelError, DataError,
    StateConsistencyError, SecurityError
)

from .core.logging import (
    get_logger, setup_logging, log_performance, log_async_performance,
    get_secure_logger, PerformanceMonitor, correlation_context
)

__all__ = [
    # Types
    "TradingMode", "SignalDirection", "OrderSide", "OrderType",
    "Signal", "MarketData", "OrderRequest", "OrderResponse", "Position",
    
    # Configuration
    "Config", "DatabaseConfig", "SecurityConfig",
    
    # Exceptions
    "TradingBotError", "ExchangeError", "RiskManagementError",
    "ValidationError", "ExecutionError", "ModelError", "DataError",
    "StateConsistencyError", "SecurityError",
    
    # Logging
    "get_logger", "setup_logging", "log_performance", "log_async_performance",
    "get_secure_logger", "PerformanceMonitor", "correlation_context",
] 