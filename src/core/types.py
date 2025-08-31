"""
Core type definitions for the trading bot framework.

This module re-exports all types from the types submodules for backward compatibility.
All new code should import from src.core.types directly, which will use the organized submodules.
"""

# Import and re-export base types explicitly
from .types.base import (
    BaseValidatedModel,
    ConfigDict,
    ConnectionType,
    ExchangeType,
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
    DataQualityMetrics,
    DataSource,
    DataValidationResult,
    DriftType,
    ErrorPattern,
    IngestionMode,
    NewsSentiment,
    PipelineStatus,
    ProcessingStep,
    QualityLevel,
    SocialSentiment,
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
    ExchangeInfo,
    ExchangeStatus,
    Kline,
    MarketData,
    MarketDepth,
    OrderBook,
    Ticker,
    Trade,
)

# Import and re-export risk types
from .types.risk import (
    DrawdownInfo,
    PortfolioRisk,
    PositionRisk,
    RiskAlert,
    RiskConfig,
    RiskLevel,
    RiskMetrics,
    VaRResult,
)

# Import and re-export strategy types
from .types.strategy import (
    Signal,
    SignalDirection,
    StrategyConfig,
    StrategyPerformance,
    StrategyState,
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
    # Trading types
    "Balance",
    # Base types
    "BaseValidatedModel",
    # Bot types
    "BotConfiguration",
    "BotPriority",
    "BotState",
    "BotStatus",
    "BotType",
    "ConfigDict",
    "ConnectionType",
    # Data types
    "DataQualityMetrics",
    "DataSource",
    "DataValidationResult",
    # Risk types
    "DrawdownInfo",
    "DriftType",
    "ErrorPattern",
    # Market types
    "ExchangeInfo",
    "ExchangeStatus",
    "ExchangeType",
    # Execution types
    "ExecutionAlgorithm",
    "ExecutionInstruction",
    "ExecutionResult",
    "ExecutionStatus",
    "IngestionMode",
    "Kline",
    "MarketData",
    "MarketDepth",
    "MarketType",
    "NewsSentiment",
    "Order",
    "OrderBook",
    "OrderRequest",
    "OrderResponse",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PipelineStatus",
    "PortfolioRisk",
    "Position",
    "PositionRisk",
    "PositionSide",
    "PositionStatus",
    "ProcessingStep",
    "QualityLevel",
    "RequestType",
    "RiskAlert",
    "RiskConfig",
    "RiskLevel",
    "RiskMetrics",
    # Strategy types
    "Signal",
    "SignalDirection",
    "SocialSentiment",
    "StorageMode",
    "StrategyConfig",
    "StrategyPerformance",
    "StrategyState",
    "Ticker",
    "TimeInForce",
    "Trade",
    "TradeFill",
    "TradingMode",
    "VaRResult",
    "ValidationLevel",
    "ValidationResult",
]
