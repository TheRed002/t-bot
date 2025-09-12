"""Database repository package."""

from .audit import (
    CapitalAuditLogRepository,
    ExecutionAuditLogRepository,
    PerformanceAuditLogRepository,
    RiskAuditLogRepository,
)
from .base import DatabaseRepository
from .bot import BotLogRepository, BotRepository, SignalRepository, StrategyRepository
from .bot_instance import BotInstanceRepository
from .capital import (
    CapitalAllocationRepository,
    CurrencyExposureRepository,
    ExchangeAllocationRepository,
    FundFlowRepository,
)

# DatabaseRepository now imported from .base
from .data import DataPipelineRepository, DataQualityRepository, FeatureRepository
from .market_data import MarketDataRepository
from .ml import (
    MLModelMetadataRepository,
    MLPredictionRepository,
    MLRepository,
    MLTrainingJobRepository,
)
from .risk import (
    PortfolioRepository,
    PortfolioRepositoryImpl,
    RiskMetricsRepository,
    RiskMetricsRepositoryImpl,
)
from .service_repository import DatabaseServiceRepository
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
    # System
    "AlertRepository",
    "AuditLogRepository",
    "BalanceSnapshotRepository",
    # Base
    "DatabaseRepository",
    "BotInstanceRepository",
    "BotLogRepository",
    # Bot
    "BotRepository",
    # Capital
    "CapitalAllocationRepository",
    # Audit
    "CapitalAuditLogRepository",
    "CurrencyExposureRepository",
    # Data
    "DataPipelineRepository",
    "DataQualityRepository",
    "DatabaseServiceRepository",
    "ExchangeAllocationRepository",
    "ExecutionAuditLogRepository",
    "FeatureRepository",
    "FundFlowRepository",
    "MLModelMetadataRepository",
    "MLPredictionRepository",
    # ML
    "MLRepository",
    "MLTrainingJobRepository",
    # Market Data
    "MarketDataRepository",
    "OrderFillRepository",
    # Trading
    "OrderRepository",
    "PerformanceAuditLogRepository",
    "PerformanceMetricsRepository",
    # Risk Management
    "PortfolioRepository",
    "PortfolioRepositoryImpl",
    "PositionRepository",
    "RiskAuditLogRepository",
    "RiskMetricsRepository",
    "RiskMetricsRepositoryImpl",
    "SignalRepository",
    # State
    "StateBackupRepository",
    "StateCheckpointRepository",
    "StateHistoryRepository",
    "StateMetadataRepository",
    "StateSnapshotRepository",
    "StrategyRepository",
    "TradeRepository",
    # User
    "UserRepository",
]
