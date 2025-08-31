"""
Core type definitions for the T-Bot trading system.

This module re-exports all types for backward compatibility.
Types are now organized into domain-specific modules for better maintainability.
"""

# Base types and common enums
from .base import (
    AlertSeverity,
    ConnectionType,
    ExchangeType,
    MarketType,
    RequestType,
    TradingMode,
    ValidationLevel,
    ValidationResult,
)

# Bot management types
from .bot import (
    BotConfiguration,
    BotEvent,
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
from .market import ExchangeInfo, ExchangeStatus, MarketData, OrderBook, OrderBookLevel, Ticker

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

# Export all for convenience
__all__ = [
    "AlertSeverity",
    "AllocationStrategy",
    "ArbitrageOpportunity",
    "Balance",
    "BotConfiguration",
    "BotEvent",
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
    "ConnectionType",
    "CurrencyExposure",
    "DriftType",
    "EmergencyAction",
    "ErrorPattern",
    "ExchangeAllocation",
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
    # Base
    "TradingMode",
    "ValidationLevel",
    "ValidationResult",
    "WithdrawalRule",
]
