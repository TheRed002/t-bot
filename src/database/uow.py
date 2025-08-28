"""Unit of Work pattern for database transactions."""

from contextlib import asynccontextmanager, contextmanager

from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import Config
from src.core.exceptions import (
    DatabaseError,
    DatabaseQueryError,
)
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

# Import error handling from P-002A
from src.error_handling.error_handler import ErrorHandler

# Import utils from P-007A
from src.utils.decorators import retry

logger = get_logger(__name__)


class UnitOfWork:
    """
    Unit of Work pattern for managing database transactions.

    This eliminates duplication of session management and ensures
    consistent transaction handling across the application.

    Note: Does not inherit from BaseComponent to avoid circular dependencies
    and maintain clean separation of concerns.
    """

    def __init__(self, session_factory: sessionmaker, config: Config | None = None):
        """
        Initialize Unit of Work.

        Args:
            session_factory: SQLAlchemy session factory
            config: Configuration object (optional)
        """
        self.session_factory = session_factory
        self.session: Session | None = None
        self._logger = logger
        self.config = config
        self.error_handler = ErrorHandler(config) if config else None

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

    @retry(max_attempts=3, delay=0.5)
    def commit(self):
        """Commit transaction with retry logic."""
        if self.session:
            try:
                self.session.commit()
                self._logger.debug("Transaction committed")
            except (IntegrityError, OperationalError) as e:
                self._logger.warning(f"Database transaction error: {e}")
                self.rollback()
                if self.error_handler and self.config:
                    # Handle deadlock and connection issues
                    error_context = self.error_handler.create_error_context(
                        error=e,
                        component="unit_of_work",
                        operation="commit",
                        details={"error_type": type(e).__name__},
                    )
                    # For sync methods, we can't await the async error handler
                    # Log the error context for monitoring
                    self._logger.error(
                        "Transaction commit failed",
                        error_id=error_context.error_id,
                        severity=error_context.severity.value,
                        component=error_context.component,
                        operation=error_context.operation,
                    )
                raise DatabaseQueryError(
                    "Database transaction failed",
                    suggested_action="Check data constraints and retry",
                ) from e
            except Exception as e:
                self._logger.error(f"Commit failed: {e}")
                self.rollback()
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
            try:
                self.session.close()
            finally:
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
        except (IntegrityError, OperationalError) as e:
            self.logger.warning(f"Database transaction error in savepoint: {e}")
            savepoint.rollback()
            raise DatabaseQueryError(
                "Database transaction failed",
                suggested_action="Check data constraints and database connectivity",
            ) from e
        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy error in savepoint: {e}")
            savepoint.rollback()
            raise DatabaseError(
                "Database operation failed",
                suggested_action="Check database state and retry",
            ) from e
        except Exception as e:
            self._logger.error(f"Unexpected error in savepoint: {e}")
            savepoint.rollback()
            raise DatabaseError(
                "Critical database error",
                suggested_action="Check system state and contact support",
            ) from e


class AsyncUnitOfWork:
    """
    Async Unit of Work pattern for managing database transactions.

    This provides proper async/await support for database operations.
    """

    def __init__(self, async_session_factory):
        """
        Initialize Async Unit of Work.

        Args:
            async_session_factory: Async SQLAlchemy session factory
        """
        self.async_session_factory = async_session_factory
        self.session: AsyncSession | None = None
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

    async def __aenter__(self):
        """Async enter context manager."""
        self.session = self.async_session_factory()

        # Initialize all repositories with async session
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

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit context manager with guaranteed resource cleanup."""
        commit_error = None
        rollback_error = None
        close_error = None

        if exc_type:
            # Exception occurred, try to rollback
            try:
                await self.rollback()
            except Exception as e:
                rollback_error = e
                self._logger.critical(f"Failed to rollback transaction: {e}")
        else:
            # No exception, try to commit
            try:
                await self.commit()
            except Exception as e:
                commit_error = e
                self._logger.error(f"Error committing transaction: {e}")
                # Try to rollback after commit failure
                try:
                    await self.rollback()
                except Exception as rollback_e:
                    rollback_error = rollback_e
                    self._logger.critical(f"Failed to rollback after commit error: {rollback_e}")

        # Always try to close the session, regardless of previous errors
        try:
            await self.close()
        except Exception as e:
            close_error = e
            self._logger.error(f"Error closing session: {e}")
            # If close fails, try to invalidate the session
            if self.session:
                try:
                    await self.session.invalidate()
                except Exception as invalidate_error:
                    self._logger.critical(f"Failed to invalidate session: {invalidate_error}")

        # Raise the most critical error
        if rollback_error and exc_type:
            # Original exception + rollback failure
            raise DatabaseError(
                "Critical: Transaction rollback failed after exception",
                suggested_action="Check database connectivity and state",
            ) from rollback_error
        elif commit_error:
            # Commit failure is critical
            raise DatabaseError(
                "Transaction commit failed",
                suggested_action="Check database state and retry",
            ) from commit_error
        elif close_error:
            # Close failure might leave connections hanging
            raise DatabaseError(
                "Session close failed - potential resource leak",
                suggested_action="Monitor connection pool usage",
            ) from close_error

    async def commit(self):
        """Async commit transaction."""
        if self.session:
            try:
                await self.session.commit()
                self._logger.debug("Transaction committed")
            except Exception as e:
                self._logger.error(f"Commit failed: {e}")
                raise

    async def rollback(self):
        """Async rollback transaction."""
        if self.session:
            try:
                await self.session.rollback()
                self._logger.debug("Transaction rolled back")
            except Exception as e:
                self._logger.error(f"Rollback failed: {e}")
                # Re-raise to ensure the error is not swallowed
                raise DatabaseError(
                    "Transaction rollback failed",
                    suggested_action="Check database connectivity and state",
                ) from e

    async def close(self):
        """Async close session with guaranteed cleanup."""
        if self.session:
            try:
                await self.session.close()
            finally:
                # Always clear references, even if close fails
                self.session = None

                # Clear all repository references
                self.users = None
                self.bots = None
                self.bot_instances = None
                self.bot_logs = None
                self.strategies = None
                self.signals = None
                self.orders = None
                self.positions = None
                self.trades = None
                self.fills = None
                self.capital_audit_logs = None
                self.execution_audit_logs = None
                self.performance_audit_logs = None
                self.risk_audit_logs = None
                self.capital_allocations = None
                self.fund_flows = None
                self.currency_exposures = None
                self.exchange_allocations = None
                self.features = None
                self.data_quality = None
                self.data_pipelines = None
                self.market_data = None
                self.ml = None
                self.ml_predictions = None
                self.ml_models = None
                self.ml_training_jobs = None
                self.state_snapshots = None
                self.state_checkpoints = None
                self.state_history = None
                self.state_metadata = None
                self.state_backups = None
                self.alerts = None
                self.audit_logs = None
                self.performance_metrics = None
                self.balance_snapshots = None

    async def refresh(self, entity):
        """Async refresh entity from database."""
        if self.session:
            await self.session.refresh(entity)

    async def flush(self):
        """Async flush pending changes."""
        if self.session:
            await self.session.flush()

    @asynccontextmanager
    async def savepoint(self):
        """Create an async savepoint."""
        if not self.session:
            raise RuntimeError("No active session")

        async with self.session.begin_nested() as savepoint:
            try:
                yield savepoint
            except (IntegrityError, OperationalError) as e:
                self._logger.warning(f"Database transaction error in savepoint: {e}")
                await savepoint.rollback()
                raise DatabaseQueryError(
                    "Database transaction failed",
                    suggested_action="Check data constraints and database connectivity",
                ) from e
            except SQLAlchemyError as e:
                self._logger.error(f"SQLAlchemy error in savepoint: {e}")
                await savepoint.rollback()
                raise DatabaseError(
                    "Database operation failed",
                    suggested_action="Check database state and retry",
                ) from e
            except Exception as e:
                self._logger.error(f"Unexpected error in savepoint: {e}")
                await savepoint.rollback()
                raise DatabaseError(
                    "Critical database error",
                    suggested_action="Check system state and contact support",
                ) from e


class UnitOfWorkFactory:
    """Factory for creating Unit of Work instances."""

    def __init__(self, session_factory: sessionmaker, async_session_factory=None):
        """
        Initialize factory.

        Args:
            session_factory: SQLAlchemy session factory for sync operations
            async_session_factory: Async SQLAlchemy session factory for async operations
        """
        self.session_factory = session_factory
        self.async_session_factory = async_session_factory
        self._logger = logger

    def create(self) -> UnitOfWork:
        """Create new Unit of Work."""
        return UnitOfWork(self.session_factory)

    def create_async(self) -> AsyncUnitOfWork:
        """Create new async Unit of Work."""
        if not self.async_session_factory:
            raise RuntimeError("Async session factory not configured")
        return AsyncUnitOfWork(self.async_session_factory)

    @contextmanager
    def transaction(self):
        """Create Unit of Work in transaction context."""
        uow = self.create()
        with uow:
            yield uow

    @asynccontextmanager
    async def async_transaction(self):
        """Create async Unit of Work in transaction context."""
        uow = self.create_async()
        async with uow:
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
