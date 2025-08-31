"""Database models package."""

# Import base first to avoid circular dependencies
# Import audit models
from .audit import (
    CapitalAuditLog,
    ExecutionAuditLog,
    PerformanceAuditLog,
    RiskAuditLog,
)
from .base import Base

# Import bot models
from .bot import Bot, BotLog, Signal, Strategy
from .bot_instance import BotInstance

# Import capital management models
from .capital import (
    CapitalAllocationDB,
    CurrencyExposureDB,
    ExchangeAllocationDB,
    FundFlowDB,
)

# Import data models
from .data import (
    DataPipelineRecord,
    DataQualityRecord,
    FeatureRecord,
)

# Import market data models
from .market_data import MarketDataRecord

# Import ML models
from .ml import MLModelMetadata, MLPrediction, MLTrainingJob

# Import state models
from .state import StateBackup, StateCheckpoint, StateHistory, StateMetadata, StateSnapshot

# Import system models
from .system import (
    Alert,
    AlertRule,
    AuditLog,
    BalanceSnapshot,
    EscalationPolicy,
    PerformanceMetrics,
)

# Import trading models
from .trading import Order, OrderFill, Position, Trade

# Import user models
from .user import User

# Import analytics models
from .analytics import (
    AnalyticsOperationalMetrics,
    AnalyticsPortfolioMetrics,
    AnalyticsPositionMetrics,
    AnalyticsRiskMetrics,
    AnalyticsStrategyMetrics,
)

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
]
