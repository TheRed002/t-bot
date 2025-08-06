"""
Base strategy interface for all trading strategies.

This module provides the abstract base class that ALL strategies must inherit from.
The interface defines the contract that all strategy implementations must follow.

CRITICAL: All strategy implementations (P-012, P-013A-E, P-019) MUST inherit from this exact interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime

# MANDATORY: Import from P-001
from src.core.types import (
    Signal, MarketData, Position, StrategyConfig, 
    StrategyStatus, StrategyMetrics
)
from src.core.exceptions import ValidationError, RiskManagementError
from src.core.logging import get_logger

# MANDATORY: Import from P-007A  
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_signal

# MANDATORY: Import from P-008+ - Use risk management
from src.risk_management.base import BaseRiskManager

# MANDATORY: Import from P-003+ - Use exchange interfaces
from src.exchanges.base import BaseExchange

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """Base strategy interface that ALL strategies must inherit from."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy with configuration.
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = StrategyConfig(**config)
        self.name: str = self.config.name  # Use config name for unique identification
        self.version: str = "1.0.0"
        self.status: StrategyStatus = StrategyStatus.STOPPED
        self.metrics: StrategyMetrics = StrategyMetrics()
        self._risk_manager: Optional[BaseRiskManager] = None
        self._exchange: Optional[BaseExchange] = None
        
        logger.info("Strategy initialized", strategy=self.name, config=self.config.model_dump())
    
    @abstractmethod
    async def _generate_signals_impl(self, data: MarketData) -> List[Signal]:
        """Internal signal generation implementation.
        
        Args:
            data: Market data for signal generation
            
        Returns:
            List of trading signals
        """
        pass
    
    @time_execution
    async def generate_signals(self, data: MarketData) -> List[Signal]:
        """Generate trading signals from market data.
        
        MANDATORY: All implementations must:
        1. Validate input data
        2. Return empty list on errors (graceful degradation)
        3. Apply confidence thresholds
        4. Log signal generation events
        
        Args:
            data: Market data for signal generation
            
        Returns:
            List of trading signals
        """
        try:
            return await self._generate_signals_impl(data)
        except Exception as e:
            logger.error("Signal generation failed", strategy=self.name, error=str(e))
            return []  # Graceful degradation
    
    @abstractmethod
    async def validate_signal(self, signal: Signal) -> bool:
        """Validate signal before execution.
        
        MANDATORY: Check signal confidence, direction, timestamp
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size for signal.
        
        MANDATORY: Integrate with risk management
        
        Args:
            signal: Signal for position sizing
            
        Returns:
            Position size as Decimal
        """
        pass
    
    @abstractmethod
    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine if position should be closed.
        
        MANDATORY: Check stop loss, take profit, time exits
        
        Args:
            position: Current position
            data: Current market data
            
        Returns:
            True if position should be closed, False otherwise
        """
        pass
    
    # Standard methods that can be overridden
    async def pre_trade_validation(self, signal: Signal) -> bool:
        """Pre-trade validation hook.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        if not await self.validate_signal(signal):
            logger.warning("Signal validation failed", strategy=self.name, signal=signal.model_dump())
            return False
        
        if self._risk_manager:
            risk_valid = await self._risk_manager.validate_signal(signal)
            if not risk_valid:
                logger.warning("Risk validation failed", strategy=self.name, signal=signal.model_dump())
                return False
        
        return True
    
    async def post_trade_processing(self, trade_result: Any) -> None:
        """Post-trade processing hook.
        
        Args:
            trade_result: Result of trade execution
        """
        # Update metrics
        self.metrics.total_trades += 1
        
        # TODO: Remove in production - Debug logging
        logger.debug("Post-trade processing", strategy=self.name, trade_result=trade_result)
        
        # Log trade result
        if hasattr(trade_result, 'pnl') and trade_result.pnl:
            if trade_result.pnl > 0:
                self.metrics.winning_trades += 1
            else:
                self.metrics.losing_trades += 1
            
            self.metrics.total_pnl += trade_result.pnl
            
            # Update win rate
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
        
        self.metrics.last_updated = datetime.now()
    
    def set_risk_manager(self, risk_manager: BaseRiskManager) -> None:
        """Set risk manager for strategy.
        
        Args:
            risk_manager: Risk manager instance
        """
        self._risk_manager = risk_manager
        logger.info("Risk manager set", strategy=self.name)
    
    def set_exchange(self, exchange: BaseExchange) -> None:
        """Set exchange for strategy.
        
        Args:
            exchange: Exchange instance
        """
        self._exchange = exchange
        logger.info("Exchange set", strategy=self.name, exchange=exchange.__class__.__name__)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "config": self.config.model_dump(),
            "metrics": self.metrics.model_dump()
        }
    
    async def start(self) -> None:
        """Start the strategy."""
        self.status = StrategyStatus.STARTING
        logger.info("Starting strategy", strategy=self.name)
        
        try:
            await self._on_start()
            self.status = StrategyStatus.RUNNING
            logger.info("Strategy started successfully", strategy=self.name)
        except Exception as e:
            self.status = StrategyStatus.ERROR
            logger.error("Failed to start strategy", strategy=self.name, error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the strategy."""
        self.status = StrategyStatus.STOPPED
        logger.info("Stopping strategy", strategy=self.name)
        
        try:
            await self._on_stop()
            logger.info("Strategy stopped successfully", strategy=self.name)
        except Exception as e:
            logger.error("Error stopping strategy", strategy=self.name, error=str(e))
            raise
    
    async def pause(self) -> None:
        """Pause the strategy."""
        if self.status == StrategyStatus.RUNNING:
            self.status = StrategyStatus.PAUSED
            logger.info("Strategy paused", strategy=self.name)
    
    async def resume(self) -> None:
        """Resume the strategy."""
        if self.status == StrategyStatus.PAUSED:
            self.status = StrategyStatus.RUNNING
            logger.info("Strategy resumed", strategy=self.name)
    
    # Optional lifecycle hooks that can be overridden
    async def _on_start(self) -> None:
        """Called when strategy starts. Override for custom initialization."""
        pass
    
    async def _on_stop(self) -> None:
        """Called when strategy stops. Override for custom cleanup."""
        pass
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update strategy configuration.
        
        Args:
            new_config: New configuration dictionary
        """
        old_config = self.config.model_dump()
        self.config = StrategyConfig(**new_config)
        logger.info("Strategy config updated", strategy=self.name, 
                    old_config=old_config, new_config=self.config.model_dump())
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the strategy.
        
        Returns:
            Performance summary dictionary
        """
        return {
            "strategy_name": self.name,
            "status": self.status.value,
            "total_trades": self.metrics.total_trades,
            "winning_trades": self.metrics.winning_trades,
            "losing_trades": self.metrics.losing_trades,
            "win_rate": self.metrics.win_rate,
            "total_pnl": float(self.metrics.total_pnl),
            "sharpe_ratio": self.metrics.sharpe_ratio,
            "max_drawdown": self.metrics.max_drawdown,
            "last_updated": self.metrics.last_updated.isoformat()
        } 