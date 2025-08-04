"""
Database query utilities for the trading bot framework.

This module provides common database operations with type safety,
bulk operations, and query builders for complex filtering.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for data persistence.
"""

from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy import select, update, delete, and_, or_, func, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.exc import SQLAlchemyError

# Import core components from P-001
from src.core.exceptions import DataError, ValidationError
from src.core.logging import get_logger

# Import database models
from .models import (
    Base, User, BotInstance, Trade, Position, BalanceSnapshot,
    StrategyConfig, MLModel, PerformanceMetrics, Alert, AuditLog
)

logger = get_logger(__name__)

T = TypeVar('T', bound=Base)


class DatabaseQueries:
    """Database query utilities with common CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    # Generic CRUD operations
    async def create(self, model_instance: T) -> T:
        """Create a new record."""
        try:
            self.session.add(model_instance)
            await self.session.commit()
            await self.session.refresh(model_instance)
            return model_instance
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Database create operation failed", error=str(e))
            raise DataError(f"Failed to create record: {str(e)}")
    
    async def get_by_id(self, model_class: type[T], record_id: str) -> Optional[T]:
        """Get a record by ID."""
        try:
            result = await self.session.execute(
                select(model_class).where(model_class.id == record_id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error("Database get_by_id operation failed", error=str(e))
            raise DataError(f"Failed to get record by ID: {str(e)}")
    
    async def get_all(self, model_class: type[T], limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """Get all records with optional pagination."""
        try:
            query = select(model_class)
            if limit:
                query = query.limit(limit).offset(offset)
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Database get_all operation failed", error=str(e))
            raise DataError(f"Failed to get all records: {str(e)}")
    
    async def update(self, model_instance: T) -> T:
        """Update an existing record."""
        try:
            await self.session.commit()
            await self.session.refresh(model_instance)
            return model_instance
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Database update operation failed", error=str(e))
            raise DataError(f"Failed to update record: {str(e)}")
    
    async def delete(self, model_instance: T) -> bool:
        """Delete a record."""
        try:
            await self.session.delete(model_instance)
            await self.session.commit()
            return True
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Database delete operation failed", error=str(e))
            raise DataError(f"Failed to delete record: {str(e)}")
    
    async def bulk_create(self, model_instances: List[T]) -> List[T]:
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
            raise DataError(f"Failed to bulk create records: {str(e)}")
    
    async def bulk_update(self, model_class: type[T], updates: List[Dict[str, Any]], id_field: str = "id") -> int:
        """Update multiple records in bulk."""
        try:
            updated_count = 0
            for update_data in updates:
                record_id = update_data.pop(id_field)
                stmt = update(model_class).where(getattr(model_class, id_field) == record_id).values(**update_data)
                result = await self.session.execute(stmt)
                updated_count += result.rowcount
            
            await self.session.commit()
            return updated_count
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error("Database bulk_update operation failed", error=str(e))
            raise DataError(f"Failed to bulk update records: {str(e)}")
    
    # User-specific queries
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            result = await self.session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error("Failed to get user by username", error=str(e))
            raise DataError(f"Failed to get user by username: {str(e)}")
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            result = await self.session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error("Failed to get user by email", error=str(e))
            raise DataError(f"Failed to get user by email: {str(e)}")
    
    async def get_active_users(self) -> List[User]:
        """Get all active users."""
        try:
            result = await self.session.execute(
                select(User).where(User.is_active == True)
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get active users", error=str(e))
            raise DataError(f"Failed to get active users: {str(e)}")
    
    # Bot instance queries
    async def get_bot_instances_by_user(self, user_id: str) -> List[BotInstance]:
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
            raise DataError(f"Failed to get bot instances by user: {str(e)}")
    
    async def get_bot_instance_by_name(self, user_id: str, name: str) -> Optional[BotInstance]:
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
            raise DataError(f"Failed to get bot instance by name: {str(e)}")
    
    async def get_running_bots(self) -> List[BotInstance]:
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
            raise DataError(f"Failed to get running bots: {str(e)}")
    
    # Trade queries
    async def get_trades_by_bot(self, bot_id: str, limit: Optional[int] = None, offset: int = 0) -> List[Trade]:
        """Get trades for a specific bot."""
        try:
            query = select(Trade).where(Trade.bot_id == bot_id).order_by(desc(Trade.timestamp))
            if limit:
                query = query.limit(limit).offset(offset)
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get trades by bot", error=str(e))
            raise DataError(f"Failed to get trades by bot: {str(e)}")
    
    async def get_trades_by_symbol(self, symbol: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[Trade]:
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
            raise DataError(f"Failed to get trades by symbol: {str(e)}")
    
    async def get_trades_by_date_range(self, start_time: datetime, end_time: datetime) -> List[Trade]:
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
            raise DataError(f"Failed to get trades by date range: {str(e)}")
    
    # Position queries
    async def get_positions_by_bot(self, bot_id: str) -> List[Position]:
        """Get all positions for a specific bot."""
        try:
            result = await self.session.execute(
                select(Position)
                .where(Position.bot_id == bot_id)
                .order_by(desc(Position.updated_at))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get positions by bot", error=str(e))
            raise DataError(f"Failed to get positions by bot: {str(e)}")
    
    async def get_open_positions(self) -> List[Position]:
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
            raise DataError(f"Failed to get open positions: {str(e)}")
    
    # Balance queries
    async def get_latest_balance_snapshot(self, user_id: str, exchange: str, currency: str) -> Optional[BalanceSnapshot]:
        """Get the latest balance snapshot for a user, exchange, and currency."""
        try:
            result = await self.session.execute(
                select(BalanceSnapshot)
                .where(and_(
                    BalanceSnapshot.user_id == user_id,
                    BalanceSnapshot.exchange == exchange,
                    BalanceSnapshot.currency == currency
                ))
                .order_by(desc(BalanceSnapshot.timestamp))
                .limit(1)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error("Failed to get latest balance snapshot", error=str(e))
            raise DataError(f"Failed to get latest balance snapshot: {str(e)}")
    
    # Performance metrics queries
    async def get_performance_metrics_by_bot(self, bot_id: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[PerformanceMetrics]:
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
            raise DataError(f"Failed to get performance metrics by bot: {str(e)}")
    
    # Alert queries
    async def get_unread_alerts_by_user(self, user_id: str) -> List[Alert]:
        """Get unread alerts for a user."""
        try:
            result = await self.session.execute(
                select(Alert)
                .where(and_(Alert.user_id == user_id, Alert.is_read == False))
                .order_by(desc(Alert.timestamp))
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get unread alerts by user", error=str(e))
            raise DataError(f"Failed to get unread alerts by user: {str(e)}")
    
    async def get_alerts_by_severity(self, severity: str, limit: Optional[int] = None) -> List[Alert]:
        """Get alerts by severity level."""
        try:
            query = select(Alert).where(Alert.severity == severity).order_by(desc(Alert.timestamp))
            if limit:
                query = query.limit(limit)
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get alerts by severity", error=str(e))
            raise DataError(f"Failed to get alerts by severity: {str(e)}")
    
    # Audit log queries
    async def get_audit_logs_by_user(self, user_id: str, limit: Optional[int] = None) -> List[AuditLog]:
        """Get audit logs for a user."""
        try:
            query = select(AuditLog).where(AuditLog.user_id == user_id).order_by(desc(AuditLog.timestamp))
            if limit:
                query = query.limit(limit)
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error("Failed to get audit logs by user", error=str(e))
            raise DataError(f"Failed to get audit logs by user: {str(e)}")
    
    # Aggregation queries
    async def get_total_pnl_by_bot(self, bot_id: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Decimal:
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
            raise DataError(f"Failed to get total P&L by bot: {str(e)}")
    
    async def get_trade_count_by_bot(self, bot_id: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> int:
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
            raise DataError(f"Failed to get trade count by bot: {str(e)}")
    
    async def get_win_rate_by_bot(self, bot_id: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> float:
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
            raise DataError(f"Failed to get win rate by bot: {str(e)}")
    
    # Data export utilities
    async def export_trades_to_csv_data(self, bot_id: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
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
                csv_data.append({
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
                    "status": trade.status
                })
            
            return csv_data
            
        except SQLAlchemyError as e:
            logger.error("Failed to export trades to CSV data", error=str(e))
            raise DataError(f"Failed to export trades to CSV data: {str(e)}")
    
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