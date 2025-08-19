"""
Database query utilities for the trading bot framework.

This module provides common database operations with type safety,
bulk operations, and query builders for complex filtering.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for data persistence.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any, TypeVar

from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

# Import core components from P-001
from src.core.exceptions import DataError
from src.core.logging import PerformanceMonitor, correlation_context, get_logger

# Import error handling from P-002A
from src.error_handling.error_handler import ErrorHandler

# Import utils from P-007A
from src.utils.decorators import cache_result, log_performance, time_execution, timeout
from src.utils.validators import validate_decimal, validate_positive_number

# Import database models
from .models import (
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

logger = get_logger(__name__)

T = TypeVar("T", bound=Base)


class DatabaseQueries:
    """Database query utilities with common CRUD operations."""

    def __init__(self, session: AsyncSession, config: dict[str, Any] | None = None):
        self.session = session
        if config:
            self.error_handler = ErrorHandler(config)
        else:
            # TODO: Initialize with default config in production
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
                try:
                    # Validate financial data if it's a Trade model
                    if hasattr(model_instance, "price") and hasattr(model_instance, "quantity"):
                        # Use proper utils validator signatures
                        model_instance.price = validate_decimal(model_instance.price)
                        validate_positive_number(model_instance.price, "price")
                        model_instance.quantity = validate_decimal(model_instance.quantity)
                        validate_positive_number(model_instance.quantity, "quantity")

                        # Round to proper precision for financial calculations
                        # Keep as Decimal to maintain precision
                        model_instance.price = Decimal(str(model_instance.price)).quantize(
                            Decimal("0.00000001")
                        )
                        model_instance.quantity = Decimal(str(model_instance.quantity)).quantize(
                            Decimal("0.00000001")
                        )

                    session.add(model_instance)
                    await session.flush()
                    await session.commit()
                    return model_instance
                except SQLAlchemyError as e:
                    try:
                        await session.rollback()
                    except Exception:
                        pass

                    # Use ErrorHandler for sophisticated error management if
                    # available
                    if self.error_handler:
                        error_context = self.error_handler.create_error_context(
                            error=e,
                            component="database_queries",
                            operation="create_record",
                            details={"model_type": type(model_instance).__name__},
                        )

                        handled = await self.error_handler.handle_error(error_context)
                        if handled:
                            # Retry the operation after error handling
                            try:
                                session.add(model_instance)
                                await session.commit()
                                return model_instance
                            except SQLAlchemyError:
                                pass  # Fall through to raise original error

                    logger.error("Database create operation failed", error=str(e))
                    raise DataError(f"Failed to create record: {e!s}") from e

    @time_execution
    @cache_result(ttl_seconds=300)  # Cache for 5 minutes
    @log_performance
    async def get_by_id(self, model_class: type[T], record_id: str) -> T | None:
        """Get a record by ID."""
        with PerformanceMonitor(f"get_by_id_{model_class.__name__}"):
            try:
                async with self._acquire_session() as session:
                    result = await session.execute(
                        select(model_class).where(model_class.id == record_id)
                    )
                    return result.scalar_one_or_none()
            except SQLAlchemyError as e:
                # Use ErrorHandler for sophisticated error management if
                # available
                if self.error_handler:
                    error_context = self.error_handler.create_error_context(
                        error=e,
                        component="database_queries",
                        operation="get_by_id",
                        details={"model_type": model_class.__name__, "record_id": record_id},
                    )

                    handled = await self.error_handler.handle_error(error_context)
                    if handled:
                        # Retry the operation after error handling
                        try:
                            result = await self.session.execute(
                                select(model_class).where(model_class.id == record_id)
                            )
                            return result.scalar_one_or_none()
                        except SQLAlchemyError:
                            pass  # Fall through to raise original error

                logger.error("Database get_by_id operation failed", error=str(e))
                raise DataError(f"Failed to get record by ID: {e!s}") from e

    async def get_all(
        self, model_class: type[T], limit: int | None = None, offset: int = 0
    ) -> list[T]:
        """Get all records with optional pagination."""
        try:
            async with self._acquire_session() as session:
                query = select(model_class)
                if limit:
                    query = query.limit(limit).offset(offset)
                result = await session.execute(query)
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Database get_all operation failed", error=str(e))
            raise DataError(f"Failed to get all records: {e!s}") from e

    async def update(self, model_instance: T) -> T:
        """Update an existing record."""
        try:
            await self.session.commit()
            await self.session.refresh(model_instance)
            return model_instance
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Database update operation failed", error=str(e))
            raise DataError(f"Failed to update record: {e!s}") from e

    async def delete(self, model_instance: T) -> bool:
        """Delete a record."""
        try:
            await self.session.delete(model_instance)
            await self.session.commit()
            return True
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Database delete operation failed", error=str(e))
            raise DataError(f"Failed to delete record: {e!s}") from e

    async def bulk_create(self, model_instances: list[T]) -> list[T]:
        """Create multiple records in bulk."""
        try:
            self.session.add_all(model_instances)
            await self.session.commit()
            for instance in model_instances:
                await self.session.refresh(instance)
            return model_instances
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Database bulk_create operation failed", error=str(e))
            raise DataError(f"Failed to bulk create records: {e!s}") from e

    async def bulk_update(
        self, model_class: type[T], updates: list[dict[str, Any]], id_field: str = "id"
    ) -> int:
        """Update multiple records in bulk."""
        try:
            updated_count = 0
            for update_data in updates:
                record_id = update_data.pop(id_field)
                stmt = (
                    update(model_class)
                    .where(getattr(model_class, id_field) == record_id)
                    .values(**update_data)
                )
                result = await self.session.execute(stmt)
                updated_count += result.rowcount

            await self.session.commit()
            return updated_count
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Database bulk_update operation failed", error=str(e))
            raise DataError(f"Failed to bulk update records: {e!s}") from e

    # User-specific queries
    async def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        try:
            async with self._acquire_session() as session:
                result = await session.execute(select(User).where(User.username == username))
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error("Failed to get user by username", error=str(e))
            raise DataError(f"Failed to get user by username: {e!s}") from e

    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email."""
        try:
            async with self._acquire_session() as session:
                result = await session.execute(select(User).where(User.email == email))
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error("Failed to get user by email", error=str(e))
            raise DataError(f"Failed to get user by email: {e!s}") from e

    async def get_active_users(self) -> list[User]:
        """Get all active users."""
        try:
            result = await self.session.execute(select(User).where(User.is_active))
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get active users", error=str(e))
            raise DataError(f"Failed to get active users: {e!s}") from e

    # Bot instance queries
    async def get_bot_instances_by_user(self, user_id: str) -> list[BotInstance]:
        """Get all bot instances for a user."""
        try:
            result = await self.session.execute(
                select(BotInstance)
                .where(BotInstance.user_id == user_id)
                .options(selectinload(BotInstance.user))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get bot instances by user", error=str(e))
            raise DataError(f"Failed to get bot instances by user: {e!s}") from e

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
            logger.error("Failed to get bot instance by name", error=str(e))
            raise DataError(f"Failed to get bot instance by name: {e!s}") from e

    async def get_running_bots(self) -> list[BotInstance]:
        """Get all running bot instances."""
        try:
            result = await self.session.execute(
                select(BotInstance)
                .where(BotInstance.status == "running")
                .options(selectinload(BotInstance.user))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get running bots", error=str(e))
            raise DataError(f"Failed to get running bots: {e!s}") from e

    # Trade queries
    @time_execution
    @cache_result(ttl_seconds=60)  # Cache for 1 minute for frequent queries
    @log_performance
    @timeout(30)
    async def get_trades_by_bot(
        self, bot_id: str, limit: int | None = None, offset: int = 0
    ) -> list[Trade]:
        """Get trades for a specific bot."""
        try:
            async with self._acquire_session() as session:
                query = select(Trade).where(Trade.bot_id == bot_id).order_by(desc(Trade.timestamp))
                if limit:
                    query = query.limit(limit).offset(offset)
                result = await session.execute(query)
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get trades by bot", error=str(e))
            raise DataError(f"Failed to get trades by bot: {e!s}") from e

    async def get_trades_by_symbol(
        self, symbol: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[Trade]:
        """Get trades for a specific symbol within a time range."""
        try:
            query = select(Trade).where(Trade.symbol == symbol)

            if start_time:
                query = query.where(Trade.timestamp >= start_time)
            if end_time:
                query = query.where(Trade.timestamp <= end_time)

            query = query.order_by(desc(Trade.timestamp))
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get trades by symbol", error=str(e))
            raise DataError(f"Failed to get trades by symbol: {e!s}") from e

    async def get_trades_by_date_range(
        self, start_time: datetime, end_time: datetime
    ) -> list[Trade]:
        """Get all trades within a date range."""
        try:
            result = await self.session.execute(
                select(Trade)
                .where(and_(Trade.timestamp >= start_time, Trade.timestamp <= end_time))
                .order_by(desc(Trade.timestamp))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get trades by date range", error=str(e))
            raise DataError(f"Failed to get trades by date range: {e!s}") from e

    # Position queries
    async def get_positions_by_bot(self, bot_id: str) -> list[Position]:
        """Get all positions for a specific bot."""
        try:
            async with self._acquire_session() as session:
                result = await session.execute(
                    select(Position)
                    .where(Position.bot_id == bot_id)
                    .order_by(desc(Position.updated_at))
                )
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get positions by bot", error=str(e))
            raise DataError(f"Failed to get positions by bot: {e!s}") from e

    async def get_open_positions(self) -> list[Position]:
        """Get all open positions (not closed)."""
        try:
            result = await self.session.execute(
                select(Position)
                .where(Position.closed_at.is_(None))
                .order_by(desc(Position.updated_at))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get open positions", error=str(e))
            raise DataError(f"Failed to get open positions: {e!s}") from e

    # Balance queries
    @time_execution
    # Cache for 30 seconds for real-time balance queries
    @cache_result(ttl_seconds=30)
    @log_performance
    @timeout(15)
    async def get_latest_balance_snapshot(
        self, user_id: str, exchange: str, currency: str
    ) -> BalanceSnapshot | None:
        """Get the latest balance snapshot for a user, exchange, and currency."""
        try:
            result = await self.session.execute(
                select(BalanceSnapshot)
                .where(
                    and_(
                        BalanceSnapshot.user_id == user_id,
                        BalanceSnapshot.exchange == exchange,
                        BalanceSnapshot.currency == currency,
                    )
                )
                .order_by(desc(BalanceSnapshot.timestamp))
                .limit(1)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error("Failed to get latest balance snapshot", error=str(e))
            raise DataError(f"Failed to get latest balance snapshot: {e!s}") from e

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
            logger.error("Failed to get performance metrics by bot", error=str(e))
            raise DataError(f"Failed to get performance metrics by bot: {e!s}") from e

    # Alert queries
    async def get_unread_alerts_by_user(self, user_id: str) -> list[Alert]:
        """Get unread alerts for a user."""
        try:
            result = await self.session.execute(
                select(Alert)
                .where(and_(Alert.user_id == user_id, not Alert.is_read))
                .order_by(desc(Alert.timestamp))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get unread alerts by user", error=str(e))
            raise DataError(f"Failed to get unread alerts by user: {e!s}") from e

    async def get_alerts_by_severity(self, severity: str, limit: int | None = None) -> list[Alert]:
        """Get alerts by severity level."""
        try:
            query = select(Alert).where(Alert.severity == severity).order_by(desc(Alert.timestamp))
            if limit:
                query = query.limit(limit)
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get alerts by severity", error=str(e))
            raise DataError(f"Failed to get alerts by severity: {e!s}") from e

    # Audit log queries
    async def get_audit_logs_by_user(
        self, user_id: str, limit: int | None = None
    ) -> list[AuditLog]:
        """Get audit logs for a user."""
        try:
            query = (
                select(AuditLog)
                .where(AuditLog.user_id == user_id)
                .order_by(desc(AuditLog.timestamp))
            )
            if limit:
                query = query.limit(limit)
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get audit logs by user", error=str(e))
            raise DataError(f"Failed to get audit logs by user: {e!s}") from e

    # Aggregation queries
    @time_execution
    @cache_result(ttl_seconds=120)  # Cache for 2 minutes for PnL calculations
    @log_performance
    @timeout(45)
    async def get_total_pnl_by_bot(
        self, bot_id: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> Decimal:
        """Get total P&L for a bot within a time range."""
        try:
            query = select(func.sum(Trade.pnl)).where(Trade.bot_id == bot_id)

            if start_time:
                query = query.where(Trade.timestamp >= start_time)
            if end_time:
                query = query.where(Trade.timestamp <= end_time)

            result = await self.session.execute(query)
            total_pnl = result.scalar()
            return total_pnl or Decimal("0")
        except SQLAlchemyError as e:
            logger.error("Failed to get total P&L by bot", error=str(e))
            raise DataError(f"Failed to get total P&L by bot: {e!s}") from e

    async def get_trade_count_by_bot(
        self, bot_id: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> int:
        """Get trade count for a bot within a time range."""
        try:
            query = select(func.count(Trade.id)).where(Trade.bot_id == bot_id)

            if start_time:
                query = query.where(Trade.timestamp >= start_time)
            if end_time:
                query = query.where(Trade.timestamp <= end_time)

            result = await self.session.execute(query)
            return result.scalar() or 0
        except SQLAlchemyError as e:
            logger.error("Failed to get trade count by bot", error=str(e))
            raise DataError(f"Failed to get trade count by bot: {e!s}") from e

    async def get_win_rate_by_bot(
        self, bot_id: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> float:
        """Get win rate for a bot within a time range."""
        try:
            # Get winning trades count
            winning_query = select(func.count(Trade.id)).where(
                and_(Trade.bot_id == bot_id, Trade.pnl > 0)
            )

            if start_time:
                winning_query = winning_query.where(Trade.timestamp >= start_time)
            if end_time:
                winning_query = winning_query.where(Trade.timestamp <= end_time)

            winning_result = await self.session.execute(winning_query)
            winning_trades = winning_result.scalar() or 0

            # Get total trades count
            total_query = select(func.count(Trade.id)).where(Trade.bot_id == bot_id)

            if start_time:
                total_query = total_query.where(Trade.timestamp >= start_time)
            if end_time:
                total_query = total_query.where(Trade.timestamp <= end_time)

            total_result = await self.session.execute(total_query)
            total_trades = total_result.scalar() or 0

            if total_trades > 0:
                return winning_trades / total_trades
            else:
                return 0.0

        except SQLAlchemyError as e:
            logger.error("Failed to get win rate by bot", error=str(e))
            raise DataError(f"Failed to get win rate by bot: {e!s}") from e

    # Data export utilities
    async def export_trades_to_csv_data(
        self, bot_id: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Export trades to CSV format data."""
        try:
            query = select(Trade).where(Trade.bot_id == bot_id)

            if start_time:
                query = query.where(Trade.timestamp >= start_time)
            if end_time:
                query = query.where(Trade.timestamp <= end_time)

            query = query.order_by(desc(Trade.timestamp))
            result = await self.session.execute(query)
            trades = result.scalars().all()

            # Convert to CSV format
            csv_data = []
            for trade in trades:
                csv_data.append(
                    {
                        "id": trade.id,
                        "timestamp": trade.timestamp.isoformat(),
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "order_type": trade.order_type,
                        "quantity": float(trade.quantity),
                        "price": float(trade.price),
                        "executed_price": float(trade.executed_price),
                        "fee": float(trade.fee),
                        "pnl": float(trade.pnl) if trade.pnl else 0.0,
                        "status": trade.status,
                    }
                )

            return csv_data

        except SQLAlchemyError as e:
            logger.error("Failed to export trades to CSV data", error=str(e))
            raise DataError(f"Failed to export trades to CSV data: {e!s}") from e

    # Health check
    async def health_check(self) -> bool:
        """Check database health with a simple query."""
        try:
            result = await self.session.execute(select(func.count(User.id)))
            count = result.scalar()
            return count is not None
        except SQLAlchemyError as e:
            logger.error("Database health check failed", error=str(e))
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

    async def get_exchange_allocation_by_exchange(
        self, exchange: str
    ) -> ExchangeAllocationDB | None:
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

    async def bulk_create_market_data_records(
        self, records: list[MarketDataRecord]
    ) -> list[MarketDataRecord]:
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
            query = query.where(MarketDataRecord.timestamp >= start_time)
        if end_time:
            query = query.where(MarketDataRecord.timestamp <= end_time)

        query = query.order_by(MarketDataRecord.timestamp.desc())

        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_market_data_by_quality(
        self,
        min_quality_score: float,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[MarketDataRecord]:
        """Get market data records above quality threshold."""
        query = select(MarketDataRecord).where(MarketDataRecord.quality_score >= min_quality_score)

        if start_time:
            query = query.where(MarketDataRecord.timestamp >= start_time)
        if end_time:
            query = query.where(MarketDataRecord.timestamp <= end_time)

        query = query.order_by(MarketDataRecord.timestamp.desc())

        result = await self.session.execute(query)
        return result.scalars().all()

    async def delete_old_market_data(self, cutoff_date: datetime) -> int:
        """Delete market data records older than cutoff date."""
        query = select(MarketDataRecord).where(MarketDataRecord.timestamp < cutoff_date)

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

    async def bulk_create_feature_records(
        self, features: list[FeatureRecord]
    ) -> list[FeatureRecord]:
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

    async def create_data_quality_record(
        self, quality_record: DataQualityRecord
    ) -> DataQualityRecord:
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

    async def create_data_pipeline_record(
        self, pipeline_record: DataPipelineRecord
    ) -> DataPipelineRecord:
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
        update_data = {"status": status, "updated_at": datetime.utcnow()}

        if stage:
            update_data["stage"] = stage
        if error_message:
            update_data["last_error"] = error_message
            update_data["error_count"] = DataPipelineRecord.error_count + 1

        query = (
            update(DataPipelineRecord)
            .where(DataPipelineRecord.execution_id == execution_id)
            .values(**update_data)
        )

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
