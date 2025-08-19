"""
Core type definitions for the T-Bot trading system.

This module re-exports all types for backward compatibility.
Types are now organized into domain-specific modules for better maintainability.
"""

# Base types and common enums
from .base import (
    TradingMode,
    ExchangeType,
    RequestType,
    ConnectionType,
    ValidationLevel,
    ValidationResult
)

# Trading types
from .trading import (
    SignalDirection,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    TradeState,
    Signal,
    OrderRequest,
    OrderResponse,
    Order,
    Position,
    Trade,
    Balance
)

# Market data types
from .market import (
    ExchangeStatus,
    MarketData,
    Ticker,
    OrderBookLevel,
    OrderBook,
    ExchangeInfo
)

# Strategy types
from .strategy import (
    StrategyType,
    StrategyStatus,
    MarketRegime,
    NewsSentiment,
    SocialSentiment,
    StrategyConfig,
    StrategyMetrics,
    RegimeChangeEvent
)

# Risk management types
from .risk import (
    RiskLevel,
    PositionSizeMethod,
    CircuitBreakerStatus,
    CircuitBreakerType,
    AllocationStrategy,
    RiskMetrics,
    PositionLimits,
    CircuitBreakerEvent,
    CapitalAllocation,
    FundFlow,
    CapitalMetrics,
    CurrencyExposure,
    ExchangeAllocation,
    WithdrawalRule,
    CapitalProtection
)

# Execution types
from .execution import (
    ExecutionAlgorithm,
    ExecutionStatus,
    SlippageType,
    ExecutionInstruction,
    ExecutionResult,
    SlippageMetrics
)

# Bot management types
from .bot import (
    BotStatus,
    BotType,
    BotPriority,
    ResourceType,
    BotConfiguration,
    BotMetrics,
    BotState,
    ResourceAllocation,
    BotEvent
)

# Data pipeline types
from .data import (
    QualityLevel,
    DriftType,
    IngestionMode,
    PipelineStatus,
    ProcessingStep,
    StorageMode,
    ErrorPattern
)

# Export all for convenience
__all__ = [
    # Base
    'TradingMode',
    'ExchangeType',
    'RequestType',
    'ConnectionType',
    'ValidationLevel',
    'ValidationResult',
    
    # Trading
    'SignalDirection',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'TimeInForce',
    'TradeState',
    'Signal',
    'OrderRequest',
    'OrderResponse',
    'Order',
    'Position',
    'Trade',
    'Balance',
    
    # Market
    'ExchangeStatus',
    'MarketData',
    'Ticker',
    'OrderBookLevel',
    'OrderBook',
    'ExchangeInfo',
    
    # Strategy
    'StrategyType',
    'StrategyStatus',
    'MarketRegime',
    'NewsSentiment',
    'SocialSentiment',
    'StrategyConfig',
    'StrategyMetrics',
    'RegimeChangeEvent',
    
    # Risk
    'RiskLevel',
    'PositionSizeMethod',
    'CircuitBreakerStatus',
    'CircuitBreakerType',
    'AllocationStrategy',
    'RiskMetrics',
    'PositionLimits',
    'CircuitBreakerEvent',
    'CapitalAllocation',
    'FundFlow',
    'CapitalMetrics',
    'CurrencyExposure',
    'ExchangeAllocation',
    'WithdrawalRule',
    'CapitalProtection',
    
    # Execution
    'ExecutionAlgorithm',
    'ExecutionStatus',
    'SlippageType',
    'ExecutionInstruction',
    'ExecutionResult',
    'SlippageMetrics',
    
    # Bot
    'BotStatus',
    'BotType',
    'BotPriority',
    'ResourceType',
    'BotConfiguration',
    'BotMetrics',
    'BotState',
    'ResourceAllocation',
    'BotEvent',
    
    # Data
    'QualityLevel',
    'DriftType',
    'IngestionMode',
    'PipelineStatus',
    'ProcessingStep',
    'StorageMode',
    'ErrorPattern'
]