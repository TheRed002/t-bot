"""Unit of Work pattern for database transactions with service layer integration."""

from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING

from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import Config
from src.core.exceptions import (
    DatabaseError,
    DatabaseQueryError,
)
from src.core.logging import get_logger
from src.database.interfaces import UnitOfWorkFactoryInterface

# Import repository classes for UoW
from src.database.repository.audit import (
    CapitalAuditLogRepository,
    ExecutionAuditLogRepository,
    PerformanceAuditLogRepository,
    RiskAuditLogRepository,
)
from src.database.repository.bot import (
    BotLogRepository,
    BotRepository,
    SignalRepository,
    StrategyRepository,
)
from src.database.repository.bot_instance import BotInstanceRepository
from src.database.repository.capital import (
    CapitalAllocationRepository,
    CurrencyExposureRepository,
    ExchangeAllocationRepository,
    FundFlowRepository,
)
from src.database.repository.data import (
    DataPipelineRepository,
    DataQualityRepository,
    FeatureRepository,
)
from src.database.repository.market_data import MarketDataRepository
from src.database.repository.ml import (
    MLModelMetadataRepository,
    MLPredictionRepository,
    MLRepository,
    MLTrainingJobRepository,
)
from src.database.repository.state import (
    StateBackupRepository,
    StateCheckpointRepository,
    StateHistoryRepository,
    StateMetadataRepository,
    StateSnapshotRepository,
)
from src.database.repository.system import (
    AlertRepository,
    AuditLogRepository,
    BalanceSnapshotRepository,
    PerformanceMetricsRepository,
)
from src.database.repository.trading import (
    OrderFillRepository,
    OrderRepository,
    PositionRepository,
    TradeRepository,
)
from src.database.repository.user import UserRepository

# Import services instead of repositories for business logic
if TYPE_CHECKING:
    from src.database.services.trading_service import TradingService

# Import error handling from P-002A
from src.error_handling.error_handler import ErrorHandler

# Import utils from P-007A
from src.utils.decorators import retry

logger = get_logger(__name__)


class UnitOfWork:
    """
    Unit of Work pattern for managing database transactions with service layer integration.

    This eliminates duplication of session management and ensures
    consistent transaction handling across the application through services.
    Controllers should interact with UoW services, not repositories directly.

    Note: Does not inherit from BaseComponent to avoid circular dependencies
    and maintain clean separation of concerns.
    """

    def __init__(self, session_factory: sessionmaker, config: Config | None = None, dependency_injector=None):
        """
        Initialize Unit of Work with injected session factory and service layer.

        Args:
            session_factory: Injected SQLAlchemy session factory
            config: Optional configuration object
            dependency_injector: Optional dependency injector for service creation
        """
        self.session_factory = session_factory
        self.session: Session | None = None
        self._logger = logger
        self.config = config
        self.error_handler = ErrorHandler(config) if config else None
        self._dependency_injector = dependency_injector

        # Services layer - controllers interact with these, not repositories
        self.trading_service: TradingService | None = None

        # Repository access for service layer only - not exposed to controllers
        self._repositories: dict[str, any] = {}
        
        # Flag to prevent direct repository access
        self._repositories_hidden = False

    def __enter__(self):
        """Enter context manager and initialize services."""
        self.session = self.session_factory()

        # Initialize services using dependency injection pattern
        if self._dependency_injector:
            self._create_services_via_di()
        else:
            self._create_services_direct()

        # Hide repositories from controllers - only services should be accessible
        self._hide_repositories()
        return self

    def _create_services_via_di(self):
        """Create services using dependency injection with repository creation."""
        try:
            # Create repositories internally for service layer
            self._create_internal_repositories()

            # Create services that use repositories
            try:
                self.trading_service = self._create_service("TradingService")
            except Exception as e:
                self._logger.warning(f"Failed to create TradingService via DI: {e}")
                self.trading_service = None

        except Exception as e:
            self._logger.warning(f"Failed to create services via DI: {e}")
            # Fallback to direct creation
            self._create_services_direct()

    def _create_service(self, service_name: str):
        """Create service using dependency injection or direct instantiation."""
        try:
            # Try to resolve from dependency injector first
            return self._dependency_injector.resolve(service_name)
        except (ImportError, AttributeError, KeyError, TypeError) as e:
            # Fallback handled by _create_services_direct
            logger.debug(f"DI resolution failed for {service_name}, using direct instantiation: {e}")
            raise

    def _create_services_direct(self):
        """Create services directly without dependency injection."""
        # Create repositories internally for service layer
        self._create_internal_repositories()

        # Create services that use repositories
        try:
            from src.database.services.trading_service import TradingService

            self.trading_service = TradingService(
                order_repo=self._repositories["orders"],
                position_repo=self._repositories["positions"],
                trade_repo=self._repositories["trades"],
            )
        except ImportError:
            self._logger.warning("TradingService not found, skipping service creation")
            self.trading_service = None

    def _create_internal_repositories(self):
        """Create repositories internally for service layer use only."""
        # Import repositories here to avoid circular imports
        from src.database.repository.trading import (
            OrderRepository,
            PositionRepository,
            TradeRepository,
        )

        # Create repositories for internal service use - not exposed to controllers
        self._repositories["orders"] = OrderRepository(self.session)
        self._repositories["positions"] = PositionRepository(self.session)
        self._repositories["trades"] = TradeRepository(self.session)
        
    def _hide_repositories(self) -> None:
        """Hide repositories from controllers to enforce service layer pattern."""
        self._repositories_hidden = True
        # Remove any accidentally exposed repository attributes
        for attr_name in list(self.__dict__.keys()):
            if attr_name.endswith('_repo') or (attr_name in [
                'users', 'bots', 'bot_instances', 'bot_logs', 'strategies', 'signals',
                'orders', 'positions', 'trades', 'fills', 'capital_audit_logs',
                'execution_audit_logs', 'performance_audit_logs', 'risk_audit_logs',
                'capital_allocations', 'fund_flows', 'currency_exposures', 
                'exchange_allocations', 'features', 'data_quality', 'data_pipelines',
                'market_data', 'state_snapshots', 'state_checkpoints', 'state_history',
                'state_metadata', 'state_backups', 'alerts', 'audit_logs', 
                'performance_metrics', 'balance_snapshots'
            ]):
                delattr(self, attr_name)
                
    def __getattr__(self, name):
        """Prevent direct repository access by controllers."""
        if getattr(self, '_repositories_hidden', False) and (name.endswith('_repo') or name in [
            'users', 'bots', 'bot_instances', 'bot_logs', 'strategies', 'signals',
            'orders', 'positions', 'trades', 'fills', 'capital_audit_logs',
            'execution_audit_logs', 'performance_audit_logs', 'risk_audit_logs',
            'capital_allocations', 'fund_flows', 'currency_exposures', 
            'exchange_allocations', 'features', 'data_quality', 'data_pipelines',
            'market_data', 'state_snapshots', 'state_checkpoints', 'state_history',
            'state_metadata', 'state_backups', 'alerts', 'audit_logs', 
            'performance_metrics', 'balance_snapshots'
        ]):
            raise AttributeError(
                f"Direct repository access is not allowed. Use service layer instead. "
                f"Attempted to access: {name}"
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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
                        e,
                        "unit_of_work",
                        "commit",
                        error_type=type(e).__name__,
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
        """Close session with guaranteed resource cleanup."""
        session = None
        try:
            session = self.session
            if session:
                session.close()
        except Exception as e:
            self._logger.warning(f"Error closing session: {e}")
            # If close fails, try to invalidate to prevent resource leaks
            if session:
                try:
                    session.invalidate()
                except Exception as invalidate_error:
                    self._logger.error(f"Session invalidate failed: {invalidate_error}")
                    raise
        finally:
            # Always clear references even if close operations fail
            self.session = None
            
            # Clear service references
            self.trading_service = None
            
            # Clear internal repository references
            self._repositories.clear()

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
            self._logger.warning(f"Database transaction error in savepoint: {e}")
            savepoint.rollback()
            raise DatabaseQueryError(
                "Database transaction failed",
                suggested_action="Check data constraints and database connectivity",
            ) from e
        except SQLAlchemyError as e:
            self._logger.error(f"SQLAlchemy error in savepoint: {e}")
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

    def __init__(self, async_session_factory, dependency_injector=None):
        """
        Initialize Async Unit of Work with injected session factory.

        Args:
            async_session_factory: Injected async SQLAlchemy session factory
            dependency_injector: Optional dependency injector for repository creation
        """
        self.async_session_factory = async_session_factory
        self.session: AsyncSession | None = None
        self._logger = logger
        self._dependency_injector = dependency_injector

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

        # Initialize repositories using factory pattern with dependency injection
        if self._dependency_injector:
            # Use dependency injection for repository creation
            await self._create_repositories_via_di()
        else:
            # Fallback to direct instantiation
            self._create_repositories_direct()

        return self

    async def _create_repositories_via_di(self):
        """Create repositories using dependency injection."""
        try:
            # Core repositories
            self.users = self._create_repository(UserRepository)
            self.bots = self._create_repository(BotRepository)
            self.bot_instances = self._create_repository(BotInstanceRepository)
            self.bot_logs = self._create_repository(BotLogRepository)
            self.strategies = self._create_repository(StrategyRepository)
            self.signals = self._create_repository(SignalRepository)

            # Trading repositories
            self.orders = self._create_repository(OrderRepository)
            self.positions = self._create_repository(PositionRepository)
            self.trades = self._create_repository(TradeRepository)
            self.fills = self._create_repository(OrderFillRepository)

            # Audit repositories
            self.capital_audit_logs = self._create_repository(CapitalAuditLogRepository)
            self.execution_audit_logs = self._create_repository(ExecutionAuditLogRepository)
            self.performance_audit_logs = self._create_repository(PerformanceAuditLogRepository)
            self.risk_audit_logs = self._create_repository(RiskAuditLogRepository)

            # Capital management repositories
            self.capital_allocations = self._create_repository(CapitalAllocationRepository)
            self.fund_flows = self._create_repository(FundFlowRepository)
            self.currency_exposures = self._create_repository(CurrencyExposureRepository)
            self.exchange_allocations = self._create_repository(ExchangeAllocationRepository)

            # Data repositories
            self.features = self._create_repository(FeatureRepository)
            self.data_quality = self._create_repository(DataQualityRepository)
            self.data_pipelines = self._create_repository(DataPipelineRepository)
            self.market_data = self._create_repository(MarketDataRepository)

            # ML repositories
            self.ml = self._create_repository(MLRepository)
            self.ml_predictions = self._create_repository(MLPredictionRepository)
            self.ml_models = self._create_repository(MLModelMetadataRepository)
            self.ml_training_jobs = self._create_repository(MLTrainingJobRepository)

            # State management repositories
            self.state_snapshots = self._create_repository(StateSnapshotRepository)
            self.state_checkpoints = self._create_repository(StateCheckpointRepository)
            self.state_history = self._create_repository(StateHistoryRepository)
            self.state_metadata = self._create_repository(StateMetadataRepository)
            self.state_backups = self._create_repository(StateBackupRepository)

            # System repositories
            self.alerts = self._create_repository(AlertRepository)
            self.audit_logs = self._create_repository(AuditLogRepository)
            self.performance_metrics = self._create_repository(PerformanceMetricsRepository)
            self.balance_snapshots = self._create_repository(BalanceSnapshotRepository)

        except Exception as e:
            self._logger.warning(f"Failed to create repositories via DI: {e}")
            # Fallback to direct creation
            self._create_repositories_direct()

    def _create_repository(self, repository_class):
        """Create repository using dependency injection or direct instantiation."""
        try:
            # Try to resolve from dependency injector first
            repo_name = repository_class.__name__
            return self._dependency_injector.resolve(repo_name)
        except (ImportError, AttributeError, KeyError, TypeError) as e:
            # Fallback to direct instantiation on DI resolution failures
            logger.debug(f"DI resolution failed for {repository_class.__name__}, " f"using direct instantiation: {e}")
            return repository_class(self.session)

    def _create_repositories_direct(self):
        """Create repositories directly without dependency injection."""
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
                    raise

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


class UnitOfWorkFactory(UnitOfWorkFactoryInterface):
    """Factory for creating Unit of Work instances."""

    def __init__(
        self,
        session_factory: sessionmaker,
        async_session_factory=None,
        dependency_injector=None,
    ):
        """
        Initialize factory with injected session factories.

        Args:
            session_factory: Injected SQLAlchemy session factory for sync operations
            async_session_factory: Injected async SQLAlchemy session factory for async operations
            dependency_injector: Optional dependency injector for UoW creation
        """
        self.session_factory = session_factory
        self.async_session_factory = async_session_factory
        self._dependency_injector = dependency_injector
        self._logger = logger

    def create(self) -> UnitOfWork:
        """Create new Unit of Work."""
        return UnitOfWork(self.session_factory, dependency_injector=self._dependency_injector)

    def create_async(self) -> AsyncUnitOfWork:
        """Create new async Unit of Work."""
        if not self.async_session_factory:
            raise RuntimeError("Async session factory not configured")
        return AsyncUnitOfWork(self.async_session_factory, dependency_injector=self._dependency_injector)

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

    def configure_dependencies(self, dependency_injector) -> None:
        """Configure dependency injection for created UoW instances."""
        self._dependency_injector = dependency_injector
        self._logger.info("Dependencies configured for UnitOfWorkFactory")


# Example usage patterns
class UnitOfWorkExample:
    """Example demonstrating Unit of Work usage patterns."""

    def __init__(self, uow_factory: UnitOfWorkFactory):
        """Initialize with UoW factory."""
        self.uow_factory = uow_factory
        self._logger = logger

    async def example_transaction(self, entity_data: dict):
        """Example transaction pattern."""
        uow = self.uow_factory.create_async()
        async with uow:
            # Example: Create related entities in single transaction
            # This is a template - actual business logic should be in services

            # Step 1: Create primary entity
            primary_entity = await uow.bots.create(entity_data)

            # Step 2: Create related entities
            # (Implementation would depend on specific business requirements)

            # Transaction commits automatically on context exit
            return primary_entity

    async def example_multi_repository_operation(self, entity_id: str):
        """Example multi-repository operation."""
        uow = self.uow_factory.create_async()
        async with uow:
            # Example: Update multiple related entities
            # This is a template - actual business logic should be in services

            # Step 1: Get primary entity
            primary_entity = await uow.bots.get(entity_id)

            if primary_entity:
                # Step 2: Update related entities
                # (Implementation would depend on specific business requirements)
                await uow.bots.update(primary_entity)

                return True

            return False
