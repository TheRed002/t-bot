"""
Core Framework Package

This package contains the core framework components including types, configuration,
exceptions, and logging systems.
"""

# Core framework exports
from .config import Config, DatabaseConfig, SecurityConfig
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
    # Types
    "TradingMode",
    "SignalDirection",
    "OrderSide",
    "OrderType",
    "Signal",
    "MarketData",
    "OrderRequest",
    "OrderResponse",
    "Position",
    "ValidationLevel",
    "ValidationResult",
    "QualityLevel",
    "DriftType",
    "IngestionMode",
    "PipelineStatus",
    "ProcessingStep",
    "StorageMode",
    "NewsSentiment",
    "SocialSentiment",
    # Configuration
    "Config",
    "DatabaseConfig",
    "SecurityConfig",
    # Exceptions
    "TradingBotError",
    "ExchangeError",
    "RiskManagementError",
    "ValidationError",
    "ExecutionError",
    "ModelError",
    "DataError",
    "StateConsistencyError",
    "SecurityError",
    # Logging
    "get_logger",
    "setup_logging",
    "log_performance",
    "log_async_performance",
    "get_secure_logger",
    "PerformanceMonitor",
    "correlation_context",
    # Error handling types (P-002A)
    "ErrorPattern",
]
