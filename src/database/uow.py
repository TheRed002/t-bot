"""Unit of Work pattern for database transactions."""

from typing import Optional
from contextlib import contextmanager
from sqlalchemy.orm import Session, sessionmaker

from src.database.repository.trading import (
    OrderRepository,
    PositionRepository,
    TradeRepository,
    OrderFillRepository
)
from src.database.repository.bot import BotRepository, StrategyRepository, SignalRepository
from src.core.logging import get_logger

logger = get_logger(__name__)


class UnitOfWork:
    """
    Unit of Work pattern for managing database transactions.
    
    This eliminates duplication of session management and ensures
    consistent transaction handling across the application.
    """
    
    def __init__(self, session_factory: sessionmaker):
        """
        Initialize Unit of Work.
        
        Args:
            session_factory: SQLAlchemy session factory
        """
        self.session_factory = session_factory
        self.session: Optional[Session] = None
        self._logger = logger
        
        # Repositories
        self.orders: Optional[OrderRepository] = None
        self.positions: Optional[PositionRepository] = None
        self.trades: Optional[TradeRepository] = None
        self.fills: Optional[OrderFillRepository] = None
        self.bots: Optional[BotRepository] = None
        self.strategies: Optional[StrategyRepository] = None
        self.signals: Optional[SignalRepository] = None
    
    def __enter__(self):
        """Enter context manager."""
        self.session = self.session_factory()
        
        # Initialize repositories
        self.orders = OrderRepository(self.session)
        self.positions = PositionRepository(self.session)
        self.trades = TradeRepository(self.session)
        self.fills = OrderFillRepository(self.session)
        self.bots = BotRepository(self.session)
        self.strategies = StrategyRepository(self.session)
        self.signals = SignalRepository(self.session)
        
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
            self.orders = None
            self.positions = None
            self.trades = None
            self.fills = None
            self.bots = None
            self.strategies = None
            self.signals = None
    
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
        self.uow_factory = uow_factory
    
    async def place_order_with_position(
        self,
        order_data: dict,
        position_data: dict
    ):
        """Place order and create position in single transaction."""
        with self.uow_factory.transaction() as uow:
            # Create order
            order = await uow.orders.create(order_data)
            
            # Create position
            position_data['entry_order_id'] = order.id
            position = await uow.positions.create(position_data)
            
            # Update order with position reference
            order.position_id = position.id
            await uow.orders.update(order)
            
            # Transaction commits automatically on context exit
            return order, position
    
    async def close_position_with_trade(
        self,
        position_id: str,
        exit_price: float
    ):
        """Close position and record trade."""
        with self.uow_factory.transaction() as uow:
            # Close position
            success = await uow.positions.close_position(position_id, exit_price)
            
            if success:
                # Get position
                position = await uow.positions.get(position_id)
                
                # Create trade record
                trade = await uow.trades.create_from_position(
                    position,
                    exit_order=None  # Would get from actual exit order
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