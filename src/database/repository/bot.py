"""Bot-specific repository implementations."""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from src.database.repository.base import BaseRepository
from src.database.models.bot import Bot, Strategy, Signal, BotLog
from src.core.logging import get_logger

logger = get_logger(__name__)


class BotRepository(BaseRepository[Bot]):
    """Repository for Bot entities."""
    
    def __init__(self, session: Session):
        super().__init__(session, Bot)
    
    async def get_active_bots(self) -> List[Bot]:
        """Get all active bots."""
        return await self.get_all(
            filters={'status': ['RUNNING', 'PAUSED']},
            order_by='name'
        )
    
    async def get_running_bots(self) -> List[Bot]:
        """Get running bots."""
        return await self.get_all(filters={'status': 'RUNNING'})
    
    async def get_bot_by_name(self, name: str) -> Optional[Bot]:
        """Get bot by name."""
        return await self.get_by(name=name)
    
    async def start_bot(self, bot_id: str) -> bool:
        """Start a bot."""
        bot = await self.get(bot_id)
        if bot and bot.status in ('STOPPED', 'PAUSED'):
            bot.status = 'INITIALIZING'
            await self.update(bot)
            return True
        return False
    
    async def stop_bot(self, bot_id: str) -> bool:
        """Stop a bot."""
        bot = await self.get(bot_id)
        if bot and bot.status in ('RUNNING', 'PAUSED', 'ERROR'):
            bot.status = 'STOPPING'
            await self.update(bot)
            return True
        return False
    
    async def pause_bot(self, bot_id: str) -> bool:
        """Pause a bot."""
        bot = await self.get(bot_id)
        if bot and bot.status == 'RUNNING':
            bot.status = 'PAUSED'
            await self.update(bot)
            return True
        return False
    
    async def update_bot_status(self, bot_id: str, status: str) -> bool:
        """Update bot status."""
        bot = await self.get(bot_id)
        if bot:
            bot.status = status
            await self.update(bot)
            return True
        return False
    
    async def update_bot_metrics(
        self,
        bot_id: str,
        metrics: Dict[str, Any]
    ) -> bool:
        """Update bot performance metrics."""
        bot = await self.get(bot_id)
        if bot:
            for key, value in metrics.items():
                if hasattr(bot, key):
                    setattr(bot, key, value)
            await self.update(bot)
            return True
        return False
    
    async def get_bot_performance(
        self,
        bot_id: str
    ) -> Dict[str, Any]:
        """Get bot performance metrics."""
        bot = await self.get(bot_id)
        if not bot:
            return {}
        
        return {
            'total_trades': bot.total_trades,
            'winning_trades': bot.winning_trades,
            'losing_trades': bot.losing_trades,
            'win_rate': bot.win_rate,
            'total_pnl': bot.total_pnl,
            'average_pnl': bot.average_pnl,
            'allocated_capital': bot.allocated_capital,
            'current_balance': bot.current_balance,
            'roi': ((bot.current_balance - bot.allocated_capital) / bot.allocated_capital * 100)
                   if bot.allocated_capital > 0 else 0
        }


class StrategyRepository(BaseRepository[Strategy]):
    """Repository for Strategy entities."""
    
    def __init__(self, session: Session):
        super().__init__(session, Strategy)
    
    async def get_active_strategies(
        self,
        bot_id: Optional[str] = None
    ) -> List[Strategy]:
        """Get active strategies."""
        filters = {'status': 'ACTIVE'}
        if bot_id:
            filters['bot_id'] = bot_id
        
        return await self.get_all(filters=filters)
    
    async def get_strategies_by_bot(self, bot_id: str) -> List[Strategy]:
        """Get all strategies for a bot."""
        return await self.get_all(filters={'bot_id': bot_id})
    
    async def get_strategy_by_name(
        self,
        bot_id: str,
        name: str
    ) -> Optional[Strategy]:
        """Get strategy by name within a bot."""
        return await self.get_by(bot_id=bot_id, name=name)
    
    async def activate_strategy(self, strategy_id: str) -> bool:
        """Activate a strategy."""
        strategy = await self.get(strategy_id)
        if strategy and strategy.status in ('INACTIVE', 'PAUSED'):
            strategy.status = 'ACTIVE'
            await self.update(strategy)
            return True
        return False
    
    async def deactivate_strategy(self, strategy_id: str) -> bool:
        """Deactivate a strategy."""
        strategy = await self.get(strategy_id)
        if strategy and strategy.status == 'ACTIVE':
            strategy.status = 'INACTIVE'
            await self.update(strategy)
            return True
        return False
    
    async def update_strategy_params(
        self,
        strategy_id: str,
        params: Dict[str, Any]
    ) -> bool:
        """Update strategy parameters."""
        strategy = await self.get(strategy_id)
        if strategy:
            strategy.params.update(params)
            await self.update(strategy)
            return True
        return False
    
    async def update_strategy_metrics(
        self,
        strategy_id: str,
        metrics: Dict[str, Any]
    ) -> bool:
        """Update strategy performance metrics."""
        strategy = await self.get(strategy_id)
        if strategy:
            for key, value in metrics.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
            await self.update(strategy)
            return True
        return False


class SignalRepository(BaseRepository[Signal]):
    """Repository for Signal entities."""
    
    def __init__(self, session: Session):
        super().__init__(session, Signal)
    
    async def get_unexecuted_signals(
        self,
        strategy_id: Optional[str] = None
    ) -> List[Signal]:
        """Get unexecuted signals."""
        filters = {'executed': False}
        if strategy_id:
            filters['strategy_id'] = strategy_id
        
        return await self.get_all(
            filters=filters,
            order_by='-created_at'
        )
    
    async def get_signals_by_strategy(
        self,
        strategy_id: str,
        limit: int = 100
    ) -> List[Signal]:
        """Get signals for a strategy."""
        return await self.get_all(
            filters={'strategy_id': strategy_id},
            order_by='-created_at',
            limit=limit
        )
    
    async def get_recent_signals(
        self,
        hours: int = 24,
        strategy_id: Optional[str] = None
    ) -> List[Signal]:
        """Get recent signals."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        query = self.session.query(Signal).filter(
            Signal.created_at >= since
        )
        
        if strategy_id:
            query = query.filter(Signal.strategy_id == strategy_id)
        
        return query.order_by(Signal.created_at.desc()).all()
    
    async def mark_signal_executed(
        self,
        signal_id: str,
        order_id: str,
        execution_time: float
    ) -> bool:
        """Mark signal as executed."""
        signal = await self.get(signal_id)
        if signal:
            signal.executed = True
            signal.order_id = order_id
            signal.execution_time = execution_time
            await self.update(signal)
            return True
        return False
    
    async def update_signal_outcome(
        self,
        signal_id: str,
        outcome: str,
        pnl: Optional[float] = None
    ) -> bool:
        """Update signal outcome."""
        signal = await self.get(signal_id)
        if signal:
            signal.outcome = outcome
            if pnl is not None:
                signal.pnl = pnl
            await self.update(signal)
            return True
        return False
    
    async def get_signal_statistics(
        self,
        strategy_id: str,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get signal statistics for a strategy."""
        query = self.session.query(Signal).filter(
            Signal.strategy_id == strategy_id
        )
        
        if since:
            query = query.filter(Signal.created_at >= since)
        
        signals = query.all()
        
        if not signals:
            return {
                'total_signals': 0,
                'executed_signals': 0,
                'successful_signals': 0,
                'execution_rate': 0,
                'success_rate': 0,
                'average_execution_time': 0
            }
        
        executed = [s for s in signals if s.executed]
        successful = [s for s in executed if s.outcome == 'SUCCESS']
        execution_times = [s.execution_time for s in executed if s.execution_time]
        
        return {
            'total_signals': len(signals),
            'executed_signals': len(executed),
            'successful_signals': len(successful),
            'execution_rate': (len(executed) / len(signals)) * 100,
            'success_rate': (len(successful) / len(executed)) * 100 if executed else 0,
            'average_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0
        }


class BotLogRepository(BaseRepository[BotLog]):
    """Repository for BotLog entities."""
    
    def __init__(self, session: Session):
        super().__init__(session, BotLog)
    
    async def get_logs_by_bot(
        self,
        bot_id: str,
        level: Optional[str] = None,
        limit: int = 100
    ) -> List[BotLog]:
        """Get logs for a bot."""
        filters = {'bot_id': bot_id}
        if level:
            filters['level'] = level
        
        return await self.get_all(
            filters=filters,
            order_by='-created_at',
            limit=limit
        )
    
    async def get_error_logs(
        self,
        bot_id: Optional[str] = None,
        hours: int = 24
    ) -> List[BotLog]:
        """Get error logs."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        query = self.session.query(BotLog).filter(
            BotLog.level.in_(['ERROR', 'CRITICAL']),
            BotLog.created_at >= since
        )
        
        if bot_id:
            query = query.filter(BotLog.bot_id == bot_id)
        
        return query.order_by(BotLog.created_at.desc()).all()
    
    async def log_event(
        self,
        bot_id: str,
        level: str,
        message: str,
        category: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> BotLog:
        """Log an event."""
        log = BotLog(
            bot_id=bot_id,
            level=level,
            message=message,
            category=category,
            context=context
        )
        
        return await self.create(log)
    
    async def cleanup_old_logs(
        self,
        days: int = 30
    ) -> int:
        """Delete logs older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        deleted = self.session.query(BotLog).filter(
            BotLog.created_at < cutoff
        ).delete()
        
        self.session.flush()
        return deleted