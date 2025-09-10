"""Database models package."""

# Import base first to avoid circular dependencies
# Import analytics models
from .analytics import (
    AnalyticsOperationalMetrics,
    AnalyticsPortfolioMetrics,
    AnalyticsPositionMetrics,
    AnalyticsRiskMetrics,
    AnalyticsStrategyMetrics,
)

# Import audit models (minimal dependencies)
from .audit import (
    CapitalAuditLog,
    ExecutionAuditLog,
    PerformanceAuditLog,
    RiskAuditLog,
)

# Import backtesting models
from .backtesting import BacktestResult, BacktestRun, BacktestTrade
from .base import Base

# Import bot models before dependent models
from .bot import Bot, BotLog, Signal, Strategy
from .bot_instance import BotInstance

# Import capital management models
from .capital import (
    CapitalAllocationDB,
    CurrencyExposureDB,
    ExchangeAllocationDB,
    FundFlowDB,
)

# Import data models (minimal dependencies)
from .data import (
    DataPipelineRecord,
    DataQualityRecord,
    FeatureRecord,
)

# Import market data models
from .market_data import MarketDataRecord

# Import ML models (moderate dependencies)
from .ml import MLModelMetadata, MLPrediction, MLTrainingJob

# Import optimization models
from .optimization import (
    OptimizationObjectiveDB,
    OptimizationResult,
    OptimizationRun,
    ParameterSet,
)

# Import risk management models (depends on bot_instance)
from .risk import (
    CircuitBreakerConfig,
    CircuitBreakerEvent,
    RiskConfiguration,
    RiskViolation,
)

# Import state models
from .state import StateBackup, StateCheckpoint, StateHistory, StateMetadata, StateSnapshot

# Import system models (depends on user)
from .system import (
    Alert,
    AlertRule,
    AuditLog,
    BalanceSnapshot,
    EscalationPolicy,
    PerformanceMetrics,
)

# Import trading models (depends on bot/strategy)
from .trading import Order, OrderFill, Position, Trade

# Import independent models first (no foreign key dependencies)
from .user import User

__all__ = [
    # System models
    "Alert",
    "AlertRule",
    "AuditLog",
    "BalanceSnapshot",
    "Base",
    # Bot models
    "Bot",
    "BotInstance",
    "BotLog",
    # Capital management models
    "CapitalAllocationDB",
    # Audit models
    "CapitalAuditLog",
    "CurrencyExposureDB",
    "DataPipelineRecord",
    "DataQualityRecord",
    "EscalationPolicy",
    "ExchangeAllocationDB",
    "ExecutionAuditLog",
    # Data models
    "FeatureRecord",
    "FundFlowDB",
    # ML models
    "MLModelMetadata",
    "MLPrediction",
    "MLTrainingJob",
    # Market data models
    "MarketDataRecord",
    # Trading models
    "Order",
    "OrderFill",
    "PerformanceAuditLog",
    "PerformanceMetrics",
    "Position",
    "RiskAuditLog",
    "Signal",
    "StateBackup",
    "StateCheckpoint",
    "StateHistory",
    "StateMetadata",
    # State models
    "StateSnapshot",
    "Strategy",
    "Trade",
    # User models
    "User",
    # Analytics models
    "AnalyticsOperationalMetrics",
    "AnalyticsPortfolioMetrics",
    "AnalyticsPositionMetrics",
    "AnalyticsRiskMetrics",
    "AnalyticsStrategyMetrics",
    # Risk management models
    "CircuitBreakerConfig",
    "CircuitBreakerEvent",
    "RiskConfiguration",
    "RiskViolation",
    # Backtesting models
    "BacktestRun",
    "BacktestResult",
    "BacktestTrade",
    # Optimization models
    "OptimizationObjectiveDB",
    "OptimizationResult",
    "OptimizationRun",
    "ParameterSet",
]
