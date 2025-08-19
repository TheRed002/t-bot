"""
Core Framework Package

This package contains the core framework components including types, configuration,
exceptions, and logging systems.
"""

# Core framework exports
from .config import Config, DatabaseConfig
from .exceptions import (
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
from .logging import (
    PerformanceMonitor,
    correlation_context,
    get_logger,
    get_secure_logger,
    log_async_performance,
    log_performance,
    setup_logging,
)
from .types import (
    DriftType,
    ErrorPattern,
    IngestionMode,
    MarketData,
    NewsSentiment,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderType,
    PipelineStatus,
    Position,
    ProcessingStep,
    QualityLevel,
    Signal,
    SignalDirection,
    SocialSentiment,
    StorageMode,
    TradingMode,
    ValidationLevel,
    ValidationResult,
)

__all__ = [
    # Configuration
    "Config",
    "DataError",
    "DatabaseConfig",
    "DriftType",
    # Error handling types (P-002A)
    "ErrorPattern",
    "ExchangeError",
    "ExecutionError",
    "IngestionMode",
    "MarketData",
    "ModelError",
    "NewsSentiment",
    "OrderRequest",
    "OrderResponse",
    "OrderSide",
    "OrderType",
    "PerformanceMonitor",
    "PipelineStatus",
    "Position",
    "ProcessingStep",
    "QualityLevel",
    "RiskManagementError",
    "SecurityError",
    "Signal",
    "SignalDirection",
    "SocialSentiment",
    "StateConsistencyError",
    "StorageMode",
    # Exceptions
    "TradingBotError",
    # Types
    "TradingMode",
    "ValidationError",
    "ValidationLevel",
    "ValidationResult",
    "correlation_context",
    # Logging
    "get_logger",
    "get_secure_logger",
    "log_async_performance",
    "log_performance",
    "setup_logging",
]
