"""
Core type definitions for the T-Bot trading system.

This module re-exports all types for backward compatibility.
Types are now organized into domain-specific modules for better maintainability.
"""

# Base types and common enums
# State types
from enum import Enum

from .base import (
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

# Bot management types
from .bot import (
    BotConfiguration,
    BotMetrics,
    BotPriority,
    BotState,
    BotStatus,
    BotType,
    ResourceAllocation,
    ResourceType,
)

# Data pipeline types
from .data import (
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

# Execution types
from .execution import (
    ExecutionAlgorithm,
    ExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
    SlippageMetrics,
    SlippageType,
)

# Market data types
from .market import (
    ExchangeGeneralInfo,
    ExchangeInfo,
    ExchangeStatus,
    MarketData,
    OrderBook,
    OrderBookLevel,
    Ticker,
)

# Risk management types
from .risk import (
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
    PortfolioMetrics,
    PortfolioState,
    PositionLimits,
    PositionSizeMethod,
    RiskAlert,
    RiskLevel,
    RiskLimits,
    RiskMetrics,
    WithdrawalRule,
)

# Strategy types
from .strategy import (
    MarketRegime,
    NewsSentiment,
    RegimeChangeEvent,
    SocialSentiment,
    StrategyConfig,
    StrategyMetrics,
    StrategyStatus,
    StrategyType,
)

# Import BotEvent and BotEventType from events module
from ..events import BotEvent, BotEventType

# Trading types
from .trading import (
    ArbitrageOpportunity,
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
    Signal,
    SignalDirection,
    TimeInForce,
    Trade,
    TradeState,
)


class StateType(str, Enum):
    """State type enumeration for type safety."""

    BOT_STATE = "bot_state"
    POSITION_STATE = "position_state"
    ORDER_STATE = "order_state"
    PORTFOLIO_STATE = "portfolio_state"
    RISK_STATE = "risk_state"
    STRATEGY_STATE = "strategy_state"
    MARKET_STATE = "market_state"
    TRADE_STATE = "trade_state"
    EXECUTION = "execution"
    SYSTEM_STATE = "system_state"
    CAPITAL_STATE = "capital_state"


class StatePriority(str, Enum):
    """State operation priority levels."""

    CRITICAL = "critical"  # Trading operations, risk limits
    HIGH = "high"  # Order management, position updates
    MEDIUM = "medium"  # Strategy updates, configuration
    LOW = "low"  # Metrics, historical data


# Export all for convenience
__all__ = [
    "AlertSeverity",
    "AllocationStrategy",
    "ArbitrageOpportunity",
    "Balance",
    "BaseValidatedModel",
    "BotConfiguration",
    "BotEvent",
    "BotEventType",
    "BotMetrics",
    "BotPriority",
    "BotState",
    # Bot
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
    # Market
    "ExchangeStatus",
    "ExchangeType",
    # Execution
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
    "NewsSentiment",
    "Order",
    "OrderBook",
    "OrderBookLevel",
    "OrderRequest",
    "OrderResponse",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PipelineStatus",
    "PortfolioMetrics",
    "PortfolioState",
    "Position",
    "PositionLimits",
    "PositionSide",
    "PositionSizeMethod",
    "PositionStatus",
    "PredictionResult",
    "ProcessingStep",
    # Data
    "QualityLevel",
    "RegimeChangeEvent",
    "RequestType",
    "ResourceAllocation",
    "ResourceType",
    # Risk
    "RiskAlert",
    "RiskLevel",
    "RiskLimits",
    "RiskMetrics",
    "Signal",
    # Trading
    "SignalDirection",
    "SlippageMetrics",
    "SlippageType",
    "SocialSentiment",
    "StorageMode",
    "StrategyConfig",
    "StrategyMetrics",
    "StrategyStatus",
    # Strategy
    "StrategyType",
    "Ticker",
    "TimeInForce",
    "Trade",
    "TradeState",
    # State types
    "StatePriority",
    "StateType",
    # Base
    "TradingMode",
    "ValidationLevel",
    "ValidationResult",
    "WithdrawalRule",
]
