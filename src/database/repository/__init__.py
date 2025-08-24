"""Database repository package."""

from .audit import (
    CapitalAuditLogRepository,
    ExecutionAuditLogRepository,
    PerformanceAuditLogRepository,
    RiskAuditLogRepository,
)
from .base import BaseRepository, RepositoryInterface
from .bot import BotLogRepository, BotRepository, SignalRepository, StrategyRepository
from .bot_instance import BotInstanceRepository
from .capital import (
    CapitalAllocationRepository,
    CurrencyExposureRepository,
    ExchangeAllocationRepository,
    FundFlowRepository,
)
from .data import DataPipelineRepository, DataQualityRepository, FeatureRepository
from .market_data import MarketDataRepository
from .ml import (
    MLModelMetadataRepository,
    MLPredictionRepository,
    MLRepository,
    MLTrainingJobRepository,
)
from .state import (
    StateBackupRepository,
    StateCheckpointRepository,
    StateHistoryRepository,
    StateMetadataRepository,
    StateSnapshotRepository,
)
from .system import (
    AlertRepository,
    AuditLogRepository,
    BalanceSnapshotRepository,
    PerformanceMetricsRepository,
)
from .trading import OrderFillRepository, OrderRepository, PositionRepository, TradeRepository
from .user import UserRepository

__all__ = [
    # Base
    "BaseRepository",
    "RepositoryInterface",
    # User
    "UserRepository",
    # Bot
    "BotRepository",
    "BotInstanceRepository",
    "BotLogRepository",
    "SignalRepository",
    "StrategyRepository",
    # Trading
    "OrderRepository",
    "OrderFillRepository",
    "PositionRepository",
    "TradeRepository",
    # Audit
    "CapitalAuditLogRepository",
    "ExecutionAuditLogRepository",
    "PerformanceAuditLogRepository",
    "RiskAuditLogRepository",
    # Capital
    "CapitalAllocationRepository",
    "CurrencyExposureRepository",
    "ExchangeAllocationRepository",
    "FundFlowRepository",
    # Data
    "DataPipelineRepository",
    "DataQualityRepository",
    "FeatureRepository",
    # Market Data
    "MarketDataRepository",
    # ML
    "MLRepository",
    "MLPredictionRepository",
    "MLModelMetadataRepository",
    "MLTrainingJobRepository",
    # State
    "StateBackupRepository",
    "StateCheckpointRepository",
    "StateHistoryRepository",
    "StateMetadataRepository",
    "StateSnapshotRepository",
    # System
    "AlertRepository",
    "AuditLogRepository",
    "BalanceSnapshotRepository",
    "PerformanceMetricsRepository",
]
