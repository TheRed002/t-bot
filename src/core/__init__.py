"""
Core Framework Package

This package contains the core framework components including types, configuration,
exceptions, and logging systems.
"""

# Core framework exports
from .types import (
    TradingMode, SignalDirection, OrderSide, OrderType,
    Signal, MarketData, OrderRequest, OrderResponse, Position,
    ValidationLevel, ValidationResult, QualityLevel, DriftType,
    IngestionMode, PipelineStatus, ProcessingStep, StorageMode,
    NewsSentiment, SocialSentiment
)

from .config import Config, DatabaseConfig, SecurityConfig

from .exceptions import (
    TradingBotError, ExchangeError, RiskManagementError,
    ValidationError, ExecutionError, ModelError, DataError,
    StateConsistencyError, SecurityError
)

from .logging import (
    get_logger, setup_logging, log_performance, log_async_performance,
    get_secure_logger, PerformanceMonitor, correlation_context
)

__all__ = [
    # Types
    "TradingMode", "SignalDirection", "OrderSide", "OrderType",
    "Signal", "MarketData", "OrderRequest", "OrderResponse", "Position",
    "ValidationLevel", "ValidationResult", "QualityLevel", "DriftType",
    "IngestionMode", "PipelineStatus", "ProcessingStep", "StorageMode",
    "NewsSentiment", "SocialSentiment",
    
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