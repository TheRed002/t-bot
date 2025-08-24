"""
Database models package - imports consolidated models from models/ directory.

This module serves as the main entry point for all database models,
importing them from the structured models/ directory to avoid duplication.
"""

# Import all models from the models package
from .models.audit import (
    CapitalAuditLog,
    ExecutionAuditLog,
    PerformanceAuditLog,
    RiskAuditLog,
)
from .models.base import Base
from .models.bot import Bot, BotLog, Signal, Strategy
from .models.bot_instance import BotInstance
from .models.capital import (
    CapitalAllocationDB,
    CurrencyExposureDB,
    ExchangeAllocationDB,
    FundFlowDB,
)
from .models.data import (
    DataPipelineRecord,
    DataQualityRecord,
    FeatureRecord,
)
from .models.market_data import MarketDataRecord
from .models.ml import MLModelMetadata, MLPrediction, MLTrainingJob
from .models.state import StateBackup, StateCheckpoint, StateHistory, StateMetadata, StateSnapshot
from .models.system import Alert, AuditLog, BalanceSnapshot, PerformanceMetrics
from .models.trading import Order, OrderFill, Position, Trade
from .models.user import User

__all__ = [
    # System models
    "Alert",
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
]
