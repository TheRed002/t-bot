"""Unit of Work pattern for database transactions."""

from contextlib import contextmanager

from sqlalchemy.orm import Session, sessionmaker

from src.core.logging import get_logger

# Import all repositories
from src.database.repository import (
    AlertRepository,
    AuditLogRepository,
    BalanceSnapshotRepository,
    BotInstanceRepository,
    BotLogRepository,
    BotRepository,
    CapitalAllocationRepository,
    CapitalAuditLogRepository,
    CurrencyExposureRepository,
    DataPipelineRepository,
    DataQualityRepository,
    ExchangeAllocationRepository,
    ExecutionAuditLogRepository,
    FeatureRepository,
    FundFlowRepository,
    MarketDataRepository,
    MLModelMetadataRepository,
    MLPredictionRepository,
    MLRepository,
    MLTrainingJobRepository,
    OrderFillRepository,
    OrderRepository,
    PerformanceAuditLogRepository,
    PerformanceMetricsRepository,
    PositionRepository,
    RiskAuditLogRepository,
    SignalRepository,
    StateBackupRepository,
    StateCheckpointRepository,
    StateHistoryRepository,
    StateMetadataRepository,
    StateSnapshotRepository,
    StrategyRepository,
    TradeRepository,
    UserRepository,
)

logger = get_logger(__name__)


class UnitOfWork:
    """
    Unit of Work pattern for managing database transactions.

    This eliminates duplication of session management and ensures
    consistent transaction handling across the application.
    
    Note: Does not inherit from BaseComponent to avoid circular dependencies
    and maintain clean separation of concerns.
    """

    def __init__(self, session_factory: sessionmaker):
        """
        Initialize Unit of Work.

        Args:
            session_factory: SQLAlchemy session factory
        """
        self.session_factory = session_factory
        self.session: Session | None = None
        self._logger = logger

        # Core repositories
        self.users: UserRepository | None = None
        self.bots: BotRepository | None = None
        self.bot_instances: BotInstanceRepository | None = None
        self.bot_logs: BotLogRepository | None = None
        self.strategies: StrategyRepository | None = None
        self.signals: SignalRepository | None = None

        # Trading repositories
        self.orders: OrderRepository | None = None
        self.positions: PositionRepository | None = None
        self.trades: TradeRepository | None = None
        self.fills: OrderFillRepository | None = None

        # Audit repositories
        self.capital_audit_logs: CapitalAuditLogRepository | None = None
        self.execution_audit_logs: ExecutionAuditLogRepository | None = None
        self.performance_audit_logs: PerformanceAuditLogRepository | None = None
        self.risk_audit_logs: RiskAuditLogRepository | None = None

        # Capital management repositories
        self.capital_allocations: CapitalAllocationRepository | None = None
        self.fund_flows: FundFlowRepository | None = None
        self.currency_exposures: CurrencyExposureRepository | None = None
        self.exchange_allocations: ExchangeAllocationRepository | None = None

        # Data repositories
        self.features: FeatureRepository | None = None
        self.data_quality: DataQualityRepository | None = None
        self.data_pipelines: DataPipelineRepository | None = None
        self.market_data: MarketDataRepository | None = None

        # ML repositories
        self.ml: MLRepository | None = None
        self.ml_predictions: MLPredictionRepository | None = None
        self.ml_models: MLModelMetadataRepository | None = None
        self.ml_training_jobs: MLTrainingJobRepository | None = None

        # State management repositories
        self.state_snapshots: StateSnapshotRepository | None = None
        self.state_checkpoints: StateCheckpointRepository | None = None
        self.state_history: StateHistoryRepository | None = None
        self.state_metadata: StateMetadataRepository | None = None
        self.state_backups: StateBackupRepository | None = None

        # System repositories
        self.alerts: AlertRepository | None = None
        self.audit_logs: AuditLogRepository | None = None
        self.performance_metrics: PerformanceMetricsRepository | None = None
        self.balance_snapshots: BalanceSnapshotRepository | None = None

    def __enter__(self):
        """Enter context manager."""
        self.session = self.session_factory()

        # Initialize core repositories
        self.users = UserRepository(self.session)
        self.bots = BotRepository(self.session)
        self.bot_instances = BotInstanceRepository(self.session)
        self.bot_logs = BotLogRepository(self.session)
        self.strategies = StrategyRepository(self.session)
        self.signals = SignalRepository(self.session)

        # Initialize trading repositories
        self.orders = OrderRepository(self.session)
        self.positions = PositionRepository(self.session)
        self.trades = TradeRepository(self.session)
        self.fills = OrderFillRepository(self.session)

        # Initialize audit repositories
        self.capital_audit_logs = CapitalAuditLogRepository(self.session)
        self.execution_audit_logs = ExecutionAuditLogRepository(self.session)
        self.performance_audit_logs = PerformanceAuditLogRepository(self.session)
        self.risk_audit_logs = RiskAuditLogRepository(self.session)

        # Initialize capital management repositories
        self.capital_allocations = CapitalAllocationRepository(self.session)
        self.fund_flows = FundFlowRepository(self.session)
        self.currency_exposures = CurrencyExposureRepository(self.session)
        self.exchange_allocations = ExchangeAllocationRepository(self.session)

        # Initialize data repositories
        self.features = FeatureRepository(self.session)
        self.data_quality = DataQualityRepository(self.session)
        self.data_pipelines = DataPipelineRepository(self.session)
        self.market_data = MarketDataRepository(self.session)

        # Initialize ML repositories
        self.ml = MLRepository(self.session)
        self.ml_predictions = MLPredictionRepository(self.session)
        self.ml_models = MLModelMetadataRepository(self.session)
        self.ml_training_jobs = MLTrainingJobRepository(self.session)

        # Initialize state management repositories
        self.state_snapshots = StateSnapshotRepository(self.session)
        self.state_checkpoints = StateCheckpointRepository(self.session)
        self.state_history = StateHistoryRepository(self.session)
        self.state_metadata = StateMetadataRepository(self.session)
        self.state_backups = StateBackupRepository(self.session)

        # Initialize system repositories
        self.alerts = AlertRepository(self.session)
        self.audit_logs = AuditLogRepository(self.session)
        self.performance_metrics = PerformanceMetricsRepository(self.session)
        self.balance_snapshots = BalanceSnapshotRepository(self.session)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if exc_type:
            self.rollback()
        else:
            try:
                self.commit()
            except Exception as e:
                self._logger.error(f"Error committing transaction: {e}")
                self.rollback()
                raise

        self.close()

    def commit(self):
        """Commit transaction."""
        if self.session:
            try:
                self.session.commit()
                self._logger.debug("Transaction committed")
            except Exception as e:
                self._logger.error(f"Commit failed: {e}")
                raise

    def rollback(self):
        """Rollback transaction."""
        if self.session:
            try:
                self.session.rollback()
                self._logger.debug("Transaction rolled back")
            except Exception as e:
                self._logger.error(f"Rollback failed: {e}")

    def close(self):
        """Close session."""
        if self.session:
            self.session.close()
            self.session = None

            # Clear repository references
            # Core repositories
            self.users = None
            self.bots = None
            self.bot_instances = None
            self.bot_logs = None
            self.strategies = None
            self.signals = None

            # Trading repositories
            self.orders = None
            self.positions = None
            self.trades = None
            self.fills = None

            # Audit repositories
            self.capital_audit_logs = None
            self.execution_audit_logs = None
            self.performance_audit_logs = None
            self.risk_audit_logs = None

            # Capital management repositories
            self.capital_allocations = None
            self.fund_flows = None
            self.currency_exposures = None
            self.exchange_allocations = None

            # Data repositories
            self.features = None
            self.data_quality = None
            self.data_pipelines = None
            self.market_data = None

            # State management repositories
            self.state_snapshots = None
            self.state_checkpoints = None
            self.state_history = None
            self.state_metadata = None
            self.state_backups = None

            # System repositories
            self.alerts = None
            self.audit_logs = None
            self.performance_metrics = None
            self.balance_snapshots = None

    def refresh(self, entity):
        """Refresh entity from database."""
        if self.session:
            self.session.refresh(entity)

    def flush(self):
        """Flush pending changes."""
        if self.session:
            self.session.flush()

    @contextmanager
    def savepoint(self):
        """Create a savepoint."""
        if not self.session:
            raise RuntimeError("No active session")

        savepoint = self.session.begin_nested()
        try:
            yield savepoint
            savepoint.commit()
        except Exception:
            savepoint.rollback()
            raise


class AsyncUnitOfWork(UnitOfWork):
    """Async version of Unit of Work."""

    async def __aenter__(self):
        """Async enter."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit."""
        return self.__exit__(exc_type, exc_val, exc_tb)

    async def commit_async(self):
        """Async commit."""
        self.commit()

    async def rollback_async(self):
        """Async rollback."""
        self.rollback()


class UnitOfWorkFactory:
    """Factory for creating Unit of Work instances."""

    def __init__(self, session_factory: sessionmaker):
        """
        Initialize factory.

        Args:
            session_factory: SQLAlchemy session factory
        """
        self.session_factory = session_factory
        self._logger = logger

    def create(self) -> UnitOfWork:
        """Create new Unit of Work."""
        return UnitOfWork(self.session_factory)

    def create_async(self) -> AsyncUnitOfWork:
        """Create new async Unit of Work."""
        return AsyncUnitOfWork(self.session_factory)

    @contextmanager
    def transaction(self):
        """Create Unit of Work in transaction context."""
        uow = self.create()
        with uow:
            yield uow


# Example usage patterns
class DatabaseService:
    """Example service using Unit of Work."""

    def __init__(self, uow_factory: UnitOfWorkFactory):
        """Initialize with UoW factory."""
        self.uow_factory = uow_factory
        self._logger = logger

    async def place_order_with_position(self, order_data: dict, position_data: dict):
        """Place order and create position in single transaction."""
        uow = self.uow_factory.create_async()
        async with uow:
            # Create order
            order = await uow.orders.create(order_data)

            # Create position
            position_data["entry_order_id"] = order.id
            position = await uow.positions.create(position_data)

            # Update order with position reference
            order.position_id = position.id
            await uow.orders.update(order)

            # Transaction commits automatically on context exit
            return order, position

    async def close_position_with_trade(self, position_id: str, exit_price: float):
        """Close position and record trade."""
        uow = self.uow_factory.create_async()
        async with uow:
            # Close position
            success = await uow.positions.close_position(position_id, exit_price)

            if success:
                # Get position
                position = await uow.positions.get(position_id)

                # Create trade record
                trade = await uow.trades.create_from_position(
                    position,
                    exit_order=None,  # Would get from actual exit order
                )

                # Update bot statistics
                bot = await uow.bots.get(position.bot_id)
                bot.total_trades += 1
                if trade.pnl > 0:
                    bot.winning_trades += 1
                else:
                    bot.losing_trades += 1
                bot.total_pnl += trade.pnl
                await uow.bots.update(bot)

                return trade

            return None
