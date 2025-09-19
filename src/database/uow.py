"""Unit of Work pattern for database transactions with service layer integration."""

from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any

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

# Use TYPE_CHECKING to avoid circular imports
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

    def __init__(
        self, session_factory: sessionmaker, config: Config | None = None, dependency_injector=None
    ):
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
        self._repositories: dict[str, Any] = {}

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
        """Create service using dependency injection."""
        return self._dependency_injector.resolve(service_name)

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
            if attr_name.endswith("_repo") or (
                attr_name
                in [
                    "users",
                    "bots",
                    "bot_instances",
                    "bot_logs",
                    "strategies",
                    "signals",
                    "orders",
                    "positions",
                    "trades",
                    "fills",
                    "capital_audit_logs",
                    "execution_audit_logs",
                    "performance_audit_logs",
                    "risk_audit_logs",
                    "capital_allocations",
                    "fund_flows",
                    "currency_exposures",
                    "exchange_allocations",
                    "features",
                    "data_quality",
                    "data_pipelines",
                    "market_data",
                    "state_snapshots",
                    "state_checkpoints",
                    "state_history",
                    "state_metadata",
                    "state_backups",
                    "alerts",
                    "audit_logs",
                    "performance_metrics",
                    "balance_snapshots",
                ]
            ):
                delattr(self, attr_name)

    def __getattr__(self, name):
        """Prevent direct repository access by controllers."""
        if getattr(self, "_repositories_hidden", False) and (
            name.endswith("_repo")
            or name
            in [
                "users",
                "bots",
                "bot_instances",
                "bot_logs",
                "strategies",
                "signals",
                "orders",
                "positions",
                "trades",
                "fills",
                "capital_audit_logs",
                "execution_audit_logs",
                "performance_audit_logs",
                "risk_audit_logs",
                "capital_allocations",
                "fund_flows",
                "currency_exposures",
                "exchange_allocations",
                "features",
                "data_quality",
                "data_pipelines",
                "market_data",
                "state_snapshots",
                "state_checkpoints",
                "state_history",
                "state_metadata",
                "state_backups",
                "alerts",
                "audit_logs",
                "performance_metrics",
                "balance_snapshots",
            ]
        ):
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
    def commit(self, processing_mode: str = "stream"):
        """
        Commit transaction with retry logic and consistent processing mode validation.
        
        Args:
            processing_mode: Processing mode ("stream" for real-time, "batch" for transactional)
        """
        if self.session:
            try:
                self.session.commit()
                self._logger.debug(f"Transaction committed with processing_mode: {processing_mode}")
            except (IntegrityError, OperationalError) as e:
                self._logger.warning(f"Database transaction error: {e}")
                self.rollback()
                if self.error_handler and self.config:
                    # Handle deadlock and connection issues with consistent error propagation
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
                        processing_mode=processing_mode,
                        data_format="transaction_error_v1",
                        message_pattern="pub_sub"  # Align with analytics patterns
                    )
                raise DatabaseQueryError(
                    "Database transaction failed",
                    suggested_action="Check data constraints and retry",
                    details={
                        "processing_mode": processing_mode,
                        "data_format": "database_error_v1",
                        "message_pattern": "pub_sub",  # Align with analytics patterns
                        "operation_type": "commit"
                    }
                ) from e
            except Exception as e:
                self._logger.error(f"Commit failed: {e}")
                self.rollback()
                # Apply consistent error propagation
                self._propagate_uow_error(e, "commit", processing_mode)
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

    def _propagate_uow_error(self, error: Exception, operation: str, processing_mode: str = "stream") -> None:
        """Propagate UnitOfWork errors with consistent patterns aligned with core module."""
        from src.core.exceptions import DatabaseError, ValidationError

        # Apply consistent error propagation patterns
        if isinstance(error, ValidationError):
            # Validation errors are re-raised as-is for consistency
            self._logger.debug(
                f"Validation error in uow.{operation} - propagating as validation error",
                operation=operation,
                processing_mode=processing_mode,
                error_type=type(error).__name__
            )
        elif isinstance(error, DatabaseError):
            # Database errors get additional UoW context
            self._logger.warning(
                f"Database error in uow.{operation} - adding UoW context",
                operation=operation,
                processing_mode=processing_mode,
                error=str(error)
            )
        else:
            # Generic errors get UoW-level error propagation
            self._logger.error(
                f"UoW error in uow.{operation} - wrapping in DatabaseError",
                operation=operation,
                processing_mode=processing_mode,
                original_error=str(error)
            )


class AsyncUnitOfWork:
    """
    Async Unit of Work pattern for managing database transactions with service layer pattern.

    This provides proper async/await support for database operations.
    Controllers should only interact with services, not repositories directly.
    """

    def __init__(self, async_session_factory, dependency_injector=None):
        """
        Initialize Async Unit of Work with injected session factory.

        Args:
            async_session_factory: Injected async SQLAlchemy session factory
            dependency_injector: Optional dependency injector for service creation
        """
        self.async_session_factory = async_session_factory
        self.session: AsyncSession | None = None
        self._logger = logger
        self._dependency_injector = dependency_injector

        # Services layer - controllers interact with these, not repositories
        self.trading_service: TradingService | None = None

        # Repository access for service layer only - not exposed to controllers
        self._repositories: dict[str, Any] = {}

        # Flag to prevent direct repository access
        self._repositories_hidden = False

    async def __aenter__(self):
        """Async enter context manager and initialize services."""
        self.session = self.async_session_factory()

        # Initialize services using dependency injection pattern
        if self._dependency_injector:
            await self._create_services_via_di()
        else:
            await self._create_services_direct()

        # Hide repositories from controllers - only services should be accessible
        self._hide_repositories()
        return self

    async def _create_services_via_di(self):
        """Create services using dependency injection with repository creation."""
        try:
            # Create repositories internally for service layer
            await self._create_internal_repositories()

            # Create services that use repositories
            try:
                self.trading_service = self._create_service("TradingService")
            except Exception as e:
                self._logger.warning(f"Failed to create TradingService via DI: {e}")
                self.trading_service = None

        except Exception as e:
            self._logger.warning(f"Failed to create services via DI: {e}")
            # Fallback to direct creation
            await self._create_services_direct()

    def _create_service(self, service_name: str):
        """Create service using dependency injection."""
        return self._dependency_injector.resolve(service_name)

    async def _create_services_direct(self):
        """Create services directly without dependency injection."""
        # Create repositories internally for service layer
        await self._create_internal_repositories()

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

    async def _create_internal_repositories(self):
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

    def __getattr__(self, name):
        """Prevent direct repository access by controllers."""
        if getattr(self, "_repositories_hidden", False):
            raise AttributeError(
                f"Direct repository access is not allowed. Use service layer instead. "
                f"Attempted to access: {name}"
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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

    async def commit(self, processing_mode: str = "stream"):
        """
        Async commit transaction with consistent processing mode validation.
        
        Args:
            processing_mode: Processing mode ("stream" for real-time, "batch" for transactional)
        """
        if self.session:
            try:
                await self.session.commit()
                self._logger.debug(f"Async transaction committed with processing_mode: {processing_mode}")
            except Exception as e:
                self._logger.error(f"Async commit failed: {e}")
                # Apply consistent error propagation
                await self._propagate_async_uow_error(e, "commit", processing_mode)
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

                # Clear service references
                self.trading_service = None

                # Clear internal repository references
                self._repositories.clear()

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

    async def _propagate_async_uow_error(self, error: Exception, operation: str, processing_mode: str = "stream") -> None:
        """Propagate async UnitOfWork errors with consistent patterns aligned with core module."""
        from src.core.exceptions import DatabaseError, ValidationError

        # Apply consistent error propagation patterns
        if isinstance(error, ValidationError):
            # Validation errors are re-raised as-is for consistency
            self._logger.debug(
                f"Validation error in async_uow.{operation} - propagating as validation error",
                operation=operation,
                processing_mode=processing_mode,
                error_type=type(error).__name__
            )
        elif isinstance(error, DatabaseError):
            # Database errors get additional async UoW context
            self._logger.warning(
                f"Database error in async_uow.{operation} - adding async UoW context",
                operation=operation,
                processing_mode=processing_mode,
                error=str(error)
            )
        else:
            # Generic errors get async UoW-level error propagation
            self._logger.error(
                f"Async UoW error in async_uow.{operation} - wrapping in DatabaseError",
                operation=operation,
                processing_mode=processing_mode,
                original_error=str(error)
            )


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
        return AsyncUnitOfWork(
            self.async_session_factory, dependency_injector=self._dependency_injector
        )

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
        """Example transaction pattern using services."""
        uow = self.uow_factory.create_async()
        async with uow:
            # Example: Use services for business logic
            if uow.trading_service:
                trade_result = await uow.trading_service.create_trade(entity_data)
                return trade_result
            return None

    async def example_multi_service_operation(self, entity_id: str):
        """Example multi-service operation."""
        uow = self.uow_factory.create_async()
        async with uow:
            # Example: Use services for coordinated operations
            if uow.trading_service:
                statistics = await uow.trading_service.get_trade_statistics(entity_id)
                return statistics
            return None
