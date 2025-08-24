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
    TimeInForce,
    TradeFill,
)

# Re-export all types for backward compatibility
__all__ = [
    # Base types
    "BaseValidatedModel",
    "ConfigDict",
    "ConnectionType",
    "ExchangeType",
    "MarketType",
    "RequestType",
    "TradingMode",
    "ValidationLevel",
    "ValidationResult",
    # Bot types
    "BotConfiguration",
    "BotPriority",
    "BotState",
    "BotStatus",
    "BotType",
    # Data types
    "DataQualityMetrics",
    "DataSource",
    "DataValidationResult",
    "DriftType",
    "ErrorPattern",
    "IngestionMode",
    "NewsSentiment",
    "PipelineStatus",
    "ProcessingStep",
    "QualityLevel",
    "SocialSentiment",
    "StorageMode",
    # Execution types
    "ExecutionAlgorithm",
    "ExecutionInstruction",
    "ExecutionResult",
    "ExecutionStatus",
    # Market types
    "ExchangeInfo",
    "ExchangeStatus",
    "Kline",
    "MarketData",
    "MarketDepth",
    "OrderBook",
    "Ticker",
    "Trade",
    # Risk types
    "DrawdownInfo",
    "PortfolioRisk",
    "PositionRisk",
    "RiskAlert",
    "RiskConfig",
    "RiskLevel",
    "RiskMetrics",
    "VaRResult",
    # Strategy types
    "Signal",
    "SignalDirection",
    "StrategyConfig",
    "StrategyPerformance",
    "StrategyState",
    # Trading types
    "Balance",
    "Order",
    "OrderRequest",
    "OrderResponse",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "PositionSide",
    "TimeInForce",
    "TradeFill",
]
