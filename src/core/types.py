"""
Core type definitions for the trading bot framework.

This module re-exports all types from the types submodules for backward compatibility.
All new code should import from src.core.types directly, which will use the organized submodules.
"""

# Import and re-export base types explicitly
from .types.base import (
    AlertSeverity,
    BaseValidatedModel,
    ConfigDict,
    ConnectionType,
    ExchangeType,
    FinancialBaseModel,
    MarketType,
    RequestType,
    TradingMode,
    ValidationLevel,
    ValidationResult,
)

# Import and re-export bot types
from .types.bot import BotConfiguration, BotPriority, BotState, BotStatus, BotType

# Import and re-export data types
from .types.data import (
    DriftType,
    ErrorPattern,
    FeatureSet,
    IngestionMode,
    MLMarketData,
    PipelineStatus,
    PredictionResult,
    ProcessingStep,
    QualityLevel,
    StorageMode,
)

# Import and re-export execution types
from .types.execution import (
    ExecutionAlgorithm,
    ExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
)

# Import and re-export market types
from .types.market import (
    ExchangeGeneralInfo,
    ExchangeInfo,
    ExchangeStatus,
    MarketData,
    OrderBook,
    OrderBookLevel,
    Ticker,
    Trade,
)

# Import and re-export risk types
from .types.risk import (
    AllocationStrategy,
    CapitalAllocation,
    CapitalMetrics,
    CapitalProtection,
    CircuitBreakerEvent,
    CircuitBreakerStatus,
    CircuitBreakerType,
    CurrencyExposure,
    EmergencyAction,
    ExchangeAllocation,
    FundFlow,
    PortfolioState,
    PositionLimits,
    PositionSizeMethod,
    RiskAlert,
    RiskLevel,
    RiskLimits,
    RiskMetrics,
    WithdrawalRule,
)

# Import and re-export strategy types
from .types.strategy import (
    MarketRegime,
    RegimeChangeEvent,
    Signal,
    SignalDirection,
    StrategyConfig,
    StrategyMetrics,
    StrategyPerformance,
    StrategyState,
    StrategyStatus,
    StrategyType,
)

# Import and re-export trading types
from .types.trading import (
    Balance,
    Order,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
    TimeInForce,
    TradeFill,
)

# Re-export all types for backward compatibility
__all__ = [
    "AlertSeverity",
    "AllocationStrategy",
    "Balance",
    "BaseValidatedModel",
    "BotConfiguration",
    "BotPriority",
    "BotState",
    "BotStatus",
    "BotType",
    "CapitalAllocation",
    "CapitalMetrics",
    "CapitalProtection",
    "CircuitBreakerEvent",
    "CircuitBreakerStatus",
    "CircuitBreakerType",
    "ConfigDict",
    "ConnectionType",
    "CurrencyExposure",
    "DriftType",
    "EmergencyAction",
    "ErrorPattern",
    "ExchangeAllocation",
    "ExchangeGeneralInfo",
    "ExchangeInfo",
    "ExchangeStatus",
    "ExchangeType",
    "ExecutionAlgorithm",
    "ExecutionInstruction",
    "ExecutionResult",
    "ExecutionStatus",
    "FeatureSet",
    "FinancialBaseModel",
    "FundFlow",
    "IngestionMode",
    "MLMarketData",
    "MarketData",
    "MarketRegime",
    "MarketType",
    "Order",
    "OrderBook",
    "OrderBookLevel",
    "OrderRequest",
    "OrderResponse",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PipelineStatus",
    "PortfolioState",
    "Position",
    "PositionLimits",
    "PositionSide",
    "PositionSizeMethod",
    "PositionStatus",
    "PredictionResult",
    "ProcessingStep",
    "QualityLevel",
    "RegimeChangeEvent",
    "RequestType",
    "RiskAlert",
    "RiskLevel",
    "RiskLimits",
    "RiskMetrics",
    "Signal",
    "SignalDirection",
    "StorageMode",
    "StrategyConfig",
    "StrategyMetrics",
    "StrategyPerformance",
    "StrategyState",
    "StrategyStatus",
    "StrategyType",
    "Ticker",
    "TimeInForce",
    "Trade",
    "TradeFill",
    "TradingMode",
    "ValidationLevel",
    "ValidationResult",
    "WithdrawalRule",
]
