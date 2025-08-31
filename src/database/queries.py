"""
Database query utilities for the trading bot framework.

This module provides common database operations with type safety,
bulk operations, and query builders for complex filtering.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for data persistence.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, TypeVar

from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.exc import (
    SQLAlchemyError,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.base import BaseComponent

# Import core components from P-001
from src.core.exceptions import DatabaseError, DataError, ValidationError
from src.core.logging import PerformanceMonitor, correlation_context

# Import database models
from src.database.models import (
    Alert,
    AuditLog,
    BalanceSnapshot,
    Base,
    BotInstance,
    CapitalAllocationDB,
    CurrencyExposureDB,
    DataPipelineRecord,
    DataQualityRecord,
    ExchangeAllocationDB,
    FeatureRecord,
    FundFlowDB,
    MarketDataRecord,
    PerformanceMetrics,
    Position,
    Trade,
    User,
)

# Import error handling from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import NetworkDisconnectionRecovery

# Import constants from utils
from src.utils.constants import TIMEOUTS

# Import utils from P-007A
from src.utils.decorators import UnifiedDecorator, time_execution

# Use ValidationFramework for validation
from src.utils.validation import ValidationFramework

# Create log_performance decorator for compatibility
log_performance = UnifiedDecorator.enhance(log=True, monitor=True)


# Database query timeout constants - use config values with fallbacks
def get_query_config(config: dict[str, Any] | None = None) -> dict[str, int]:
    """Get query configuration with fallbacks to constants."""
    return {
        "QUERY_CACHE_TTL_LONG": (
            config.get("query_cache_ttl_long", TIMEOUTS.get("QUERY_CACHE_TTL_LONG", 300)) if config else 300
        ),
        "QUERY_CACHE_TTL_SHORT": (
            config.get("query_cache_ttl_short", TIMEOUTS.get("QUERY_CACHE_TTL_SHORT", 60)) if config else 60
        ),
        "QUERY_CACHE_TTL_REALTIME": (
            config.get("query_cache_ttl_realtime", TIMEOUTS.get("QUERY_CACHE_TTL_REALTIME", 30)) if config else 30
        ),
        "QUERY_TIMEOUT_DEFAULT": (
            config.get("query_timeout_default", TIMEOUTS.get("QUERY_TIMEOUT_DEFAULT", 30)) if config else 30
        ),
    }


T = TypeVar("T", bound=Base)


class DatabaseQueries(BaseComponent):
    """Database query utilities with common CRUD operations."""

    def __init__(self, session: AsyncSession, config: dict[str, Any] | None = None):
        # Initialize BaseComponent
        super().__init__()

        self.session = session
        self.config = config or {}

        # Get query configuration with proper fallbacks
        self.query_config = get_query_config(self.config)

        # Always initialize error handler, use empty config if none provided
        # This prevents None errors during error handling
        try:
            from src.core.config import Config

            if not config:
                # Create a minimal config for error handler
                default_config = Config()
                self.error_handler = ErrorHandler(default_config)
            else:
                self.error_handler = ErrorHandler(config)
        except Exception as e:
            # If Config class is not available, error handler features will be limited
            self.logger.warning(f"Failed to initialize error handler: {e}")
            self.error_handler = None

    @asynccontextmanager
    async def _acquire_session(self):
        """Yield a fresh AsyncSession managed by the global connection manager.

        Always use a short-lived session scoped to the operation to guarantee
        proper cleanup and avoid leaking connections in tests.
        """
        from src.database.connection import get_async_session

        async with get_async_session() as temp_session:
            yield temp_session

    # Generic CRUD operations
    @time_execution
    @log_performance
    async def create(self, model_instance: T) -> T:
        """Create a new record."""
        with correlation_context.correlation_context():
            async with self._acquire_session() as session:
                session_closed = False
                try:
                    # Validate financial data if it's a Trade model
                    if hasattr(model_instance, "price") and hasattr(model_instance, "quantity"):
                        # Use ValidationFramework for proper validation
                        model_instance.price = ValidationFramework.validate_price(model_instance.price)
                        model_instance.quantity = ValidationFramework.validate_quantity(model_instance.quantity)

                        # Round to proper precision for financial calculations
                        # Keep as Decimal to maintain precision
                        model_instance.price = Decimal(str(model_instance.price)).quantize(Decimal("0.00000001"))
                        model_instance.quantity = Decimal(str(model_instance.quantity)).quantize(Decimal("0.00000001"))

                    session.add(model_instance)
                    await session.flush()
                    await session.commit()
                    return model_instance
                except ValidationError as e:
                    # ValidationError from utils validators - don't retry these
                    self.logger.error("Validation failed", error=str(e))
                    raise  # Re-raise validation errors as-is
                except SQLAlchemyError as e:
                    try:
                        await session.rollback()
                    except SQLAlchemyError as rollback_error:
                        self.logger.critical(
                            "CRITICAL: Transaction rollback failed",
                            error=str(rollback_error),
                            original_error=str(e),
                        )
                        # Try to close and invalidate session after rollback failure
                        try:
                            await session.close()
                            session_closed = True
                        except SQLAlchemyError as close_error:
                            self.logger.error(f"Session close failed: {close_error}")
                            try:
                                session.invalidate()
                            except SQLAlchemyError as invalidate_error:
                                self.logger.critical(f"Session invalidate failed: {invalidate_error}")
                                raise
                        raise DatabaseError("Transaction rollback failed") from rollback_error

                    # Use ErrorHandler for sophisticated error management if
                    # available
                    if self.error_handler:
                        error_context = self.error_handler.create_error_context(
                            e,
                            "database_queries",
                            "create_record",
                            model_type=type(model_instance).__name__,
                        )

                        # Use recovery scenario for database operations
                        recovery_config = getattr(self, "config", {})
                        if hasattr(recovery_config, "to_dict"):
                            recovery_config = recovery_config.to_dict()
                        recovery_scenario = NetworkDisconnectionRecovery(recovery_config)

                        handled = await self.error_handler.handle_error(e, error_context, recovery_scenario)
                        if handled:
                            # Retry the operation after error handling
                            try:
                                session.add(model_instance)
                                await session.commit()
                                return model_instance
                            except SQLAlchemyError as retry_error:
                                self.logger.error(f"Retry operation failed: {retry_error}")
                                # Fall through to raise original error

                    self.logger.error("Database create operation failed", error=str(e))
                    raise DataError(f"Failed to create record: {e!s}")
                finally:
                    # Ensure session is closed if not already closed
                    if not session_closed and session:
                        try:
                            await session.close()
                        except SQLAlchemyError as close_error:
                            self.logger.warning(f"Error closing session: {close_error}")
                            try:
                                session.invalidate()
                            except SQLAlchemyError as invalidate_error:
                                self.logger.critical(f"Session invalidate failed: {invalidate_error}")

    @time_execution
    @log_performance
    async def get_by_id(self, model_class: type[T], record_id: str) -> T | None:
        """Get a record by ID."""
        with PerformanceMonitor(f"get_by_id_{model_class.__name__}"):
            try:
                async with self._acquire_session() as session:
                    result = await session.execute(select(model_class).where(model_class.id == record_id))
                    return result.scalar_one_or_none()
            except SQLAlchemyError as e:
                # Use ErrorHandler for sophisticated error management if
                # available
                if self.error_handler:
                    error_context = self.error_handler.create_error_context(
                        e,
                        "database_queries",
                        "get_by_id",
                        model_type=model_class.__name__,
                        record_id=record_id,
                    )

                    # Use recovery scenario for database operations
                    recovery_config = getattr(self, "config", {})
                    if hasattr(recovery_config, "to_dict"):
                        recovery_config = recovery_config.to_dict()
                    recovery_scenario = NetworkDisconnectionRecovery(recovery_config)

                    handled = await self.error_handler.handle_error(e, error_context, recovery_scenario)
                    if handled:
                        # Retry the operation after error handling
                        try:
                            result = await session.execute(select(model_class).where(model_class.id == record_id))
                            return result.scalar_one_or_none()
                        except SQLAlchemyError as retry_error:
                            self.logger.error(f"Retry operation failed: {retry_error}")
                            # Fall through to raise original error

                self.logger.error("Database get_by_id operation failed", error=str(e))
                raise DataError(f"Failed to get record by ID: {e!s}")

    async def get_all(self, model_class: type[T], limit: int | None = None, offset: int = 0) -> list[T]:
        """Get all records with optional pagination."""
        try:
            async with self._acquire_session() as session:
                query = select(model_class)
                if limit:
                    query = query.limit(limit).offset(offset)
                result = await session.execute(query)
                return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Database get_all operation failed", error=str(e))
            raise DataError(f"Failed to get all records: {e!s}")

    async def update(self, model_instance: T) -> T:
        """Update an existing record."""
        session = None
        try:
            session = self.session
            if not session:
                raise DataError("Database session not available")
            await session.commit()
            await session.refresh(model_instance)
            return model_instance
        except SQLAlchemyError as e:
            if session:
                try:
                    await session.rollback()
                except SQLAlchemyError as rollback_error:
                    self.logger.critical(f"Rollback failed during update: {rollback_error}")
                    # Try to close and invalidate session after rollback failure
                    try:
                        await session.close()
                    except SQLAlchemyError as close_error:
                        self.logger.error(f"Session close failed: {close_error}")
                        try:
                            session.invalidate()
                        except SQLAlchemyError as invalidate_error:
                            self.logger.critical(f"Session invalidate failed: {invalidate_error}")
                            raise
            self.logger.error("Database update operation failed", error=str(e))
            raise DataError(f"Failed to update record: {e!s}")

    async def delete(self, model_instance: T) -> bool:
        """Delete a record."""
        session = None
        try:
            session = self.session
            if not session:
                raise DataError("Database session not available")
            session.delete(model_instance)
            await session.commit()
            return True
        except SQLAlchemyError as e:
            if session:
                try:
                    await session.rollback()
                except SQLAlchemyError as rollback_error:
                    self.logger.critical(f"Rollback failed during delete: {rollback_error}")
                    # Try to close and invalidate session after rollback failure
                    try:
                        await session.close()
                    except SQLAlchemyError as close_error:
                        self.logger.error(f"Session close failed: {close_error}")
                        try:
                            session.invalidate()
                        except SQLAlchemyError as invalidate_error:
                            self.logger.critical(f"Session invalidate failed: {invalidate_error}")
                            raise
            self.logger.error("Database delete operation failed", error=str(e))
            raise DataError(f"Failed to delete record: {e!s}")

    async def bulk_create(self, model_instances: list[T]) -> list[T]:
        """Create multiple records in bulk."""
        session = None
        try:
            session = self.session
            if not session:
                raise DataError("Database session not available")
            session.add_all(model_instances)
            await session.commit()
            for instance in model_instances:
                await session.refresh(instance)
            return model_instances
        except SQLAlchemyError as e:
            if session:
                try:
                    await session.rollback()
                except SQLAlchemyError as rollback_error:
                    self.logger.critical(f"Rollback failed during bulk_create: {rollback_error}")
                    # Try to close and invalidate session after rollback failure
                    try:
                        await session.close()
                    except SQLAlchemyError as close_error:
                        self.logger.error(f"Session close failed: {close_error}")
                        try:
                            session.invalidate()
                        except SQLAlchemyError as invalidate_error:
                            self.logger.critical(f"Session invalidate failed: {invalidate_error}")
                            raise
            self.logger.error("Database bulk_create operation failed", error=str(e))
            raise DataError(f"Failed to bulk create records: {e!s}")

    async def bulk_update(self, model_class: type[T], updates: list[dict[str, Any]], id_field: str = "id") -> int:
        """Update multiple records in bulk."""
        session = None
        try:
            session = self.session
            if not session:
                raise DataError("Database session not available")
            updated_count = 0
            for update_data in updates:
                record_id = update_data.pop(id_field)
                stmt = update(model_class).where(getattr(model_class, id_field) == record_id).values(**update_data)
                result = await session.execute(stmt)
                updated_count += result.rowcount

            await session.commit()
            return updated_count
        except SQLAlchemyError as e:
            if session:
                try:
                    await session.rollback()
                except SQLAlchemyError as rollback_error:
                    self.logger.critical(f"Rollback failed during bulk_update: {rollback_error}")
                    # Try to close and invalidate session after rollback failure
                    try:
                        await session.close()
                    except SQLAlchemyError as close_error:
                        self.logger.error(f"Session close failed: {close_error}")
                        try:
                            session.invalidate()
                        except SQLAlchemyError as invalidate_error:
                            self.logger.critical(f"Session invalidate failed: {invalidate_error}")
                            raise
            self.logger.error("Database bulk_update operation failed", error=str(e))
            raise DataError(f"Failed to bulk update records: {e!s}")

    # User-specific queries
    async def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        try:
            async with self._acquire_session() as session:
                result = await session.execute(select(User).where(User.username == username))
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get user by username", error=str(e))
            raise DataError(f"Failed to get user by username: {e!s}")

    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email."""
        try:
            async with self._acquire_session() as session:
                result = await session.execute(select(User).where(User.email == email))
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get user by email", error=str(e))
            raise DataError(f"Failed to get user by email: {e!s}")

    async def get_active_users(self) -> list[User]:
        """Get all active users."""
        try:
            result = await self.session.execute(select(User).where(User.is_active))
            return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get active users", error=str(e))
            raise DataError(f"Failed to get active users: {e!s}")

    # Bot instance queries
    async def get_bot_instances_by_user(self, user_id: str) -> list[BotInstance]:
        """Get all bot instances for a user."""
        try:
            result = await self.session.execute(
                select(BotInstance).where(BotInstance.user_id == user_id).options(selectinload(BotInstance.user))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get bot instances by user", error=str(e))
            raise DataError(f"Failed to get bot instances by user: {e!s}")

    async def get_bot_instance_by_name(self, user_id: str, name: str) -> BotInstance | None:
        """Get bot instance by name for a specific user."""
        try:
            result = await self.session.execute(
                select(BotInstance)
                .where(and_(BotInstance.user_id == user_id, BotInstance.name == name))
                .options(selectinload(BotInstance.user))
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get bot instance by name", error=str(e))
            raise DataError(f"Failed to get bot instance by name: {e!s}")

    async def get_running_bots(self) -> list[BotInstance]:
        """Get all running bot instances."""
        try:
            result = await self.session.execute(
                select(BotInstance).where(BotInstance.status == "running").options(selectinload(BotInstance.user))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get running bots", error=str(e))
            raise DataError(f"Failed to get running bots: {e!s}")

    # Trade queries
    @time_execution
    @log_performance
    async def get_trades_by_bot(self, bot_id: str, limit: int | None = None, offset: int = 0) -> list[Trade]:
        """Get trades for a specific bot."""
        try:
            async with self._acquire_session() as session:
                query = select(Trade).where(Trade.bot_id == bot_id).order_by(desc(Trade.created_at))
                if limit:
                    query = query.limit(limit).offset(offset)
                result = await session.execute(query)
                return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get trades by bot", error=str(e))
            raise DataError(f"Failed to get trades by bot: {e!s}")

    async def get_trades_by_symbol(
        self, symbol: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[Trade]:
        """Get trades for a specific symbol within a time range."""
        try:
            async with self._acquire_session() as session:
                query = select(Trade).where(Trade.symbol == symbol)

                if start_time:
                    query = query.where(Trade.created_at >= start_time)
                if end_time:
                    query = query.where(Trade.created_at <= end_time)

                query = query.order_by(desc(Trade.created_at))
                result = await session.execute(query)
                return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get trades by symbol", error=str(e))
            raise DataError(f"Failed to get trades by symbol: {e!s}")

    async def get_trades_by_date_range(self, start_time: datetime, end_time: datetime) -> list[Trade]:
        """Get all trades within a date range."""
        try:
            async with self._acquire_session() as session:
                result = await session.execute(
                    select(Trade)
                    .where(and_(Trade.created_at >= start_time, Trade.created_at <= end_time))
                    .order_by(desc(Trade.created_at))
                )
                return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get trades by date range", error=str(e))
            raise DataError(f"Failed to get trades by date range: {e!s}")

    # Position queries
    async def get_positions_by_bot(self, bot_id: str) -> list[Position]:
        """Get all positions for a specific bot."""
        try:
            async with self._acquire_session() as session:
                result = await session.execute(
                    select(Position).where(Position.bot_id == bot_id).order_by(desc(Position.updated_at))
                )
                return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get positions by bot", error=str(e))
            raise DataError(f"Failed to get positions by bot: {e!s}")

    async def get_open_positions(self) -> list[Position]:
        """Get all open positions (not closed)."""
        try:
            result = await self.session.execute(
                select(Position).where(Position.closed_at.is_(None)).order_by(desc(Position.updated_at))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get open positions", error=str(e))
            raise DataError(f"Failed to get open positions: {e!s}")

    # Balance queries
    @time_execution
    @log_performance
    async def get_latest_balance_snapshot(self, user_id: str, exchange: str, currency: str) -> BalanceSnapshot | None:
        """Get the latest balance snapshot for a user, exchange, and currency."""
        try:
            async with self._acquire_session() as session:
                result = await session.execute(
                    select(BalanceSnapshot)
                    .where(
                        and_(
                            BalanceSnapshot.user_id == user_id,
                            BalanceSnapshot.exchange == exchange,
                            BalanceSnapshot.currency == currency,
                        )
                    )
                    .order_by(desc(BalanceSnapshot.created_at))
                    .limit(1)
                )
                return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get latest balance snapshot", error=str(e))
            raise DataError(f"Failed to get latest balance snapshot: {e!s}")

    # Performance metrics queries
    async def get_performance_metrics_by_bot(
        self, bot_id: str, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> list[PerformanceMetrics]:
        """Get performance metrics for a bot within a date range."""
        try:
            query = select(PerformanceMetrics).where(PerformanceMetrics.bot_id == bot_id)

            if start_date:
                query = query.where(PerformanceMetrics.metric_date >= start_date)
            if end_date:
                query = query.where(PerformanceMetrics.metric_date <= end_date)

            query = query.order_by(desc(PerformanceMetrics.metric_date))
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get performance metrics by bot", error=str(e))
            raise DataError(f"Failed to get performance metrics by bot: {e!s}")

    # Alert queries
    async def get_unread_alerts_by_user(self, user_id: str) -> list[Alert]:
        """Get unread alerts for a user."""
        try:
            result = await self.session.execute(
                select(Alert).where(and_(Alert.user_id == user_id, not Alert.is_read)).order_by(desc(Alert.timestamp))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get unread alerts by user", error=str(e))
            raise DataError(f"Failed to get unread alerts by user: {e!s}")

    async def get_alerts_by_severity(self, severity: str, limit: int | None = None) -> list[Alert]:
        """Get alerts by severity level."""
        try:
            query = select(Alert).where(Alert.severity == severity).order_by(desc(Alert.timestamp))
            if limit:
                query = query.limit(limit)
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get alerts by severity", error=str(e))
            raise DataError(f"Failed to get alerts by severity: {e!s}")

    # Audit log queries
    async def get_audit_logs_by_user(self, user_id: str, limit: int | None = None) -> list[AuditLog]:
        """Get audit logs for a user."""
        try:
            query = select(AuditLog).where(AuditLog.user_id == user_id).order_by(desc(AuditLog.timestamp))
            if limit:
                query = query.limit(limit)
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error("Failed to get audit logs by user", error=str(e))
            raise DataError(f"Failed to get audit logs by user: {e!s}")

    # Aggregation queries
    @time_execution
    @log_performance
    async def get_total_pnl_by_bot(
        self, bot_id: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> Decimal:
        """Get total P&L for a bot within a time range."""
        try:
            async with self._acquire_session() as session:
                query = select(func.sum(Trade.pnl)).where(Trade.bot_id == bot_id)

                if start_time:
                    query = query.where(Trade.created_at >= start_time)
                if end_time:
                    query = query.where(Trade.created_at <= end_time)

                result = await session.execute(query)
                total_pnl = result.scalar()
                return total_pnl or Decimal("0")
        except SQLAlchemyError as e:
            self.logger.error("Failed to get total P&L by bot", error=str(e))
            raise DataError(f"Failed to get total P&L by bot: {e!s}")

    async def get_trade_count_by_bot(
        self, bot_id: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> int:
        """Get trade count for a bot within a time range."""
        try:
            async with self._acquire_session() as session:
                query = select(func.count(Trade.id)).where(Trade.bot_id == bot_id)

                if start_time:
                    query = query.where(Trade.created_at >= start_time)
                if end_time:
                    query = query.where(Trade.created_at <= end_time)

                result = await session.execute(query)
                return result.scalar() or 0
        except SQLAlchemyError as e:
            self.logger.error("Failed to get trade count by bot", error=str(e))
            raise DataError(f"Failed to get trade count by bot: {e!s}")

    async def get_win_rate_by_bot(
        self, bot_id: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> Decimal:
        """Get win rate for a bot within a time range."""
        try:
            async with self._acquire_session() as session:
                # Get winning trades count
                winning_query = select(func.count(Trade.id)).where(and_(Trade.bot_id == bot_id, Trade.pnl > 0))

                if start_time:
                    winning_query = winning_query.where(Trade.created_at >= start_time)
                if end_time:
                    winning_query = winning_query.where(Trade.created_at <= end_time)

                winning_result = await session.execute(winning_query)
                winning_trades = winning_result.scalar() or 0

                # Get total trades count
                total_query = select(func.count(Trade.id)).where(Trade.bot_id == bot_id)

                if start_time:
                    total_query = total_query.where(Trade.created_at >= start_time)
                if end_time:
                    total_query = total_query.where(Trade.created_at <= end_time)

                total_result = await session.execute(total_query)
                total_trades = total_result.scalar() or 0

            if total_trades > 0:
                return Decimal(str(winning_trades)) / Decimal(str(total_trades))
            else:
                return Decimal("0.0")

        except SQLAlchemyError as e:
            self.logger.error("Failed to get win rate by bot", error=str(e))
            raise DataError(f"Failed to get win rate by bot: {e!s}")

    # Data export utilities
    async def export_trades_to_csv_data(
        self, bot_id: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Export trades to CSV format data."""
        try:
            async with self._acquire_session() as session:
                query = select(Trade).where(Trade.bot_id == bot_id)

                if start_time:
                    query = query.where(Trade.created_at >= start_time)
                if end_time:
                    query = query.where(Trade.created_at <= end_time)

                query = query.order_by(desc(Trade.created_at))
                result = await session.execute(query)
                trades = result.scalars().all()

                # Convert to CSV format
                csv_data = []
                for trade in trades:
                    csv_data.append(
                        {
                            "id": str(trade.id),
                            "created_at": trade.created_at.isoformat(),
                            "symbol": trade.symbol,
                            "side": trade.side,
                            "quantity": str(trade.quantity),
                            "entry_price": str(trade.entry_price),
                            "exit_price": str(trade.exit_price),
                            "fees": str(trade.fees),
                            "pnl": str(trade.pnl) if trade.pnl else "0",
                            "pnl_percentage": str(trade.pnl_percentage) if trade.pnl_percentage else "0",
                        }
                    )

                return csv_data

        except SQLAlchemyError as e:
            self.logger.error("Failed to export trades to CSV data", error=str(e))
            raise DataError(f"Failed to export trades to CSV data: {e!s}")

    # Health check
    async def health_check(self) -> bool:
        """Check database health with a simple query."""
        try:
            async with self._acquire_session() as session:
                result = await session.execute(select(func.count(User.id)))
                count = result.scalar()
                return count is not None
        except SQLAlchemyError as e:
            self.logger.error("Database health check failed", error=str(e))
            return False

    # Capital Management Queries

    async def get_capital_allocations_by_strategy(
        self, strategy_id: str, limit: int | None = None, offset: int = 0
    ) -> list[CapitalAllocationDB]:
        """Get capital allocations for a specific strategy."""
        query = (
            select(CapitalAllocationDB)
            .where(CapitalAllocationDB.strategy_id == strategy_id)
            .order_by(CapitalAllocationDB.created_at.desc())
        )

        if limit:
            query = query.limit(limit).offset(offset)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_capital_allocations_by_exchange(
        self, exchange: str, limit: int | None = None, offset: int = 0
    ) -> list[CapitalAllocationDB]:
        """Get capital allocations for a specific exchange."""
        query = (
            select(CapitalAllocationDB)
            .where(CapitalAllocationDB.exchange == exchange)
            .order_by(CapitalAllocationDB.created_at.desc())
        )

        if limit:
            query = query.limit(limit).offset(offset)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_fund_flows_by_reason(
        self,
        reason: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> list[FundFlowDB]:
        """Get fund flows by reason within time range."""
        query = select(FundFlowDB).where(FundFlowDB.reason == reason)

        if start_time:
            query = query.where(FundFlowDB.timestamp >= start_time)
        if end_time:
            query = query.where(FundFlowDB.timestamp <= end_time)

        query = query.order_by(FundFlowDB.timestamp.desc())

        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_fund_flows_by_currency(
        self,
        currency: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> list[FundFlowDB]:
        """Get fund flows by currency within time range."""
        query = select(FundFlowDB).where(FundFlowDB.currency == currency)

        if start_time:
            query = query.where(FundFlowDB.timestamp >= start_time)
        if end_time:
            query = query.where(FundFlowDB.timestamp <= end_time)

        query = query.order_by(FundFlowDB.timestamp.desc())

        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_currency_exposure_by_currency(self, currency: str) -> CurrencyExposureDB | None:
        """Get currency exposure for a specific currency."""
        query = select(CurrencyExposureDB).where(CurrencyExposureDB.currency == currency)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_exchange_allocation_by_exchange(self, exchange: str) -> ExchangeAllocationDB | None:
        """Get exchange allocation for a specific exchange."""
        query = select(ExchangeAllocationDB).where(ExchangeAllocationDB.exchange == exchange)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_total_capital_allocated(
        self, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> Decimal:
        """Get total capital allocated within time range."""
        query = select(func.sum(CapitalAllocationDB.allocated_amount))

        if start_time:
            query = query.where(CapitalAllocationDB.created_at >= start_time)
        if end_time:
            query = query.where(CapitalAllocationDB.created_at <= end_time)

        result = await self.session.execute(query)
        total = result.scalar()
        return total or Decimal("0")

    async def get_total_fund_flows(
        self, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> Decimal:
        """Get total fund flows within time range."""
        query = select(func.sum(FundFlowDB.amount))

        if start_time:
            query = query.where(FundFlowDB.timestamp >= start_time)
        if end_time:
            query = query.where(FundFlowDB.timestamp <= end_time)

        result = await self.session.execute(query)
        total = result.scalar()
        return total or Decimal("0")

    async def bulk_create_capital_allocations(
        self, allocations: list[CapitalAllocationDB]
    ) -> list[CapitalAllocationDB]:
        """Create multiple capital allocations in bulk."""
        return await self.bulk_create(allocations)

    async def bulk_update_capital_allocations(self, updates: list[dict[str, Any]]) -> int:
        """Update multiple capital allocations in bulk."""
        return await self.bulk_update(CapitalAllocationDB, updates, "id")

    async def bulk_create_fund_flows(self, flows: list[FundFlowDB]) -> list[FundFlowDB]:
        """Create multiple fund flows in bulk."""
        return await self.bulk_create(flows)

    # Data Management Queries
    async def create_market_data_record(self, market_data: MarketDataRecord) -> MarketDataRecord:
        """Create a new market data record."""
        return await self.create(market_data)

    async def bulk_create_market_data_records(self, records: list[MarketDataRecord]) -> list[MarketDataRecord]:
        """Create multiple market data records in bulk."""
        return await self.bulk_create(records)

    async def get_market_data_records(
        self,
        symbol: str,
        exchange: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> list[MarketDataRecord]:
        """Get market data records for a symbol and exchange within time range."""
        query = select(MarketDataRecord).where(
            and_(MarketDataRecord.symbol == symbol, MarketDataRecord.exchange == exchange)
        )

        if start_time:
            query = query.where(MarketDataRecord.data_timestamp >= start_time)
        if end_time:
            query = query.where(MarketDataRecord.data_timestamp <= end_time)

        query = query.order_by(MarketDataRecord.data_timestamp.desc())

        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_market_data_by_quality(
        self,
        min_quality_score: Decimal,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[MarketDataRecord]:
        """Get market data records above quality threshold."""
        # Note: Quality scoring will be implemented when quality_score field is
        # added to MarketDataRecord
        query = select(MarketDataRecord)

        if start_time:
            query = query.where(MarketDataRecord.data_timestamp >= start_time)
        if end_time:
            query = query.where(MarketDataRecord.data_timestamp <= end_time)

        query = query.order_by(MarketDataRecord.data_timestamp.desc())

        result = await self.session.execute(query)
        return result.scalars().all()

    async def delete_old_market_data(self, cutoff_date: datetime) -> int:
        """Delete market data records older than cutoff date."""
        query = select(MarketDataRecord).where(MarketDataRecord.data_timestamp < cutoff_date)

        result = await self.session.execute(query)
        old_records = result.scalars().all()

        if old_records:
            for record in old_records:
                self.session.delete(record)
            await self.session.commit()
            return len(old_records)

        return 0

    async def create_feature_record(self, feature: FeatureRecord) -> FeatureRecord:
        """Create a new feature record."""
        return await self.create(feature)

    async def bulk_create_feature_records(self, features: list[FeatureRecord]) -> list[FeatureRecord]:
        """Create multiple feature records in bulk."""
        return await self.bulk_create(features)

    async def get_feature_records(
        self,
        symbol: str,
        feature_type: str | None = None,
        feature_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[FeatureRecord]:
        """Get feature records for a symbol with optional filtering."""
        query = select(FeatureRecord).where(FeatureRecord.symbol == symbol)

        if feature_type:
            query = query.where(FeatureRecord.feature_type == feature_type)
        if feature_name:
            query = query.where(FeatureRecord.feature_name == feature_name)
        if start_time:
            query = query.where(FeatureRecord.calculation_timestamp >= start_time)
        if end_time:
            query = query.where(FeatureRecord.calculation_timestamp <= end_time)

        query = query.order_by(FeatureRecord.calculation_timestamp.desc())

        result = await self.session.execute(query)
        return result.scalars().all()

    async def create_data_quality_record(self, quality_record: DataQualityRecord) -> DataQualityRecord:
        """Create a new data quality record."""
        return await self.create(quality_record)

    async def get_data_quality_records(
        self,
        symbol: str | None = None,
        data_source: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[DataQualityRecord]:
        """Get data quality records with optional filtering."""
        query = select(DataQualityRecord)

        if symbol:
            query = query.where(DataQualityRecord.symbol == symbol)
        if data_source:
            query = query.where(DataQualityRecord.data_source == data_source)
        if start_time:
            query = query.where(DataQualityRecord.quality_check_timestamp >= start_time)
        if end_time:
            query = query.where(DataQualityRecord.quality_check_timestamp <= end_time)

        query = query.order_by(DataQualityRecord.quality_check_timestamp.desc())

        result = await self.session.execute(query)
        return result.scalars().all()

    async def create_data_pipeline_record(self, pipeline_record: DataPipelineRecord) -> DataPipelineRecord:
        """Create a new data pipeline record."""
        return await self.create(pipeline_record)

    async def update_data_pipeline_status(
        self,
        execution_id: str,
        status: str,
        stage: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Update data pipeline execution status."""
        update_data = {"status": status, "updated_at": datetime.now(timezone.utc)}

        if stage:
            update_data["stage"] = stage
        if error_message:
            update_data["last_error"] = error_message
            update_data["error_count"] = DataPipelineRecord.error_count + 1

        query = update(DataPipelineRecord).where(DataPipelineRecord.execution_id == execution_id).values(**update_data)

        result = await self.session.execute(query)
        await self.session.commit()
        return result.rowcount > 0

    async def get_data_pipeline_records(
        self,
        pipeline_name: str | None = None,
        status: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[DataPipelineRecord]:
        """Get data pipeline records with optional filtering."""
        query = select(DataPipelineRecord)

        if pipeline_name:
            query = query.where(DataPipelineRecord.pipeline_name == pipeline_name)
        if status:
            query = query.where(DataPipelineRecord.status == status)
        if start_time:
            query = query.where(DataPipelineRecord.execution_timestamp >= start_time)
        if end_time:
            query = query.where(DataPipelineRecord.execution_timestamp <= end_time)

        query = query.order_by(DataPipelineRecord.execution_timestamp.desc())

        result = await self.session.execute(query)
        return result.scalars().all()
