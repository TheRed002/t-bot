"""
Strategy factory for dynamic strategy instantiation and management.

This module provides the factory pattern for creating and managing strategy instances.
It supports dynamic strategy creation, hot-swapping, and resource management.
"""

from typing import Dict, Any, Optional, Type, List
import importlib
import asyncio
from datetime import datetime

# MANDATORY: Import from P-001
from src.core.types import StrategyConfig, StrategyStatus, StrategyType
from src.core.exceptions import ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_strategy_config

# MANDATORY: Import from P-008+
from src.risk_management.base import BaseRiskManager

# MANDATORY: Import from P-003+
from src.exchanges.base import BaseExchange

# MANDATORY: Import from P-011
from src.strategies.base import BaseStrategy

logger = get_logger(__name__)


class StrategyFactory:
    """Factory for creating and managing strategy instances."""
    
    def __init__(self):
        """Initialize strategy factory."""
        self._strategies: Dict[str, BaseStrategy] = {}
        self._strategy_classes: Dict[str, Type[BaseStrategy]] = {}
        self._risk_manager: Optional[BaseRiskManager] = None
        self._exchange: Optional[BaseExchange] = None
        
        # Register built-in strategy classes
        self._register_builtin_strategies()
        
        logger.info("Strategy factory initialized")
    
    def _register_builtin_strategies(self) -> None:
        """Register built-in strategy classes."""
        # TODO: Register actual strategy classes when they are implemented
        # self._register_strategy_class("mean_reversion", MeanReversionStrategy)
        # self._register_strategy_class("trend_following", TrendFollowingStrategy)
        # self._register_strategy_class("breakout", BreakoutStrategy)
        
        logger.info("Built-in strategies registered")
    
    def _register_strategy_class(self, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """Register a strategy class.
        
        Args:
            name: Strategy name
            strategy_class: Strategy class to register
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise ValidationError(f"Strategy class must inherit from BaseStrategy: {strategy_class}")
        
        self._strategy_classes[name] = strategy_class
        logger.info("Strategy class registered", name=name, class_name=strategy_class.__name__)
    
    @time_execution
    def create_strategy(self, strategy_name: str, config: Dict[str, Any]) -> BaseStrategy:
        """Create a strategy instance.
        
        Args:
            strategy_name: Name of the strategy to create
            config: Strategy configuration
            
        Returns:
            Strategy instance
            
        Raises:
            ValidationError: If strategy creation fails
        """
        try:
            # Validate configuration
            validate_strategy_config(config)
            
            # Get strategy class
            strategy_class = self._get_strategy_class(strategy_name)
            if not strategy_class:
                raise ValidationError(f"Unknown strategy: {strategy_name}")
            
            # Create strategy instance
            strategy = strategy_class(config)
            
            # Set dependencies if available
            if self._risk_manager:
                strategy.set_risk_manager(self._risk_manager)
            
            if self._exchange:
                strategy.set_exchange(self._exchange)
            
            # Store strategy instance
            self._strategies[strategy.name] = strategy
            
            logger.info("Strategy created successfully", 
                       strategy_name=strategy_name, 
                       strategy_instance=strategy.name)
            
            return strategy
            
        except Exception as e:
            logger.error("Failed to create strategy", 
                        strategy_name=strategy_name, 
                        error=str(e))
            raise ValidationError(f"Failed to create strategy {strategy_name}: {str(e)}")
    
    def _get_strategy_class(self, strategy_name: str) -> Optional[Type[BaseStrategy]]:
        """Get strategy class by name.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy class or None if not found
        """
        # Check registered classes first
        if strategy_name in self._strategy_classes:
            return self._strategy_classes[strategy_name]
        
        # Try dynamic import
        try:
            module_name = f"src.strategies.static.{strategy_name.lower()}"
            class_name = f"{strategy_name.title()}Strategy"
            
            module = importlib.import_module(module_name)
            strategy_class = getattr(module, class_name)
            
            # Register for future use
            self._register_strategy_class(strategy_name, strategy_class)
            
            return strategy_class
            
        except (ImportError, AttributeError) as e:
            logger.warning("Strategy class not found", 
                          strategy_name=strategy_name, 
                          error=str(e))
            return None
    
    @time_execution
    async def start_strategy(self, strategy_name: str) -> bool:
        """Start a strategy.
        
        Args:
            strategy_name: Name of the strategy to start
            
        Returns:
            True if strategy started successfully, False otherwise
        """
        try:
            strategy = self._strategies.get(strategy_name)
            if not strategy:
                logger.error("Strategy not found", strategy_name=strategy_name)
                return False
            
            await strategy.start()
            return True
            
        except Exception as e:
            logger.error("Failed to start strategy", 
                        strategy_name=strategy_name, 
                        error=str(e))
            return False
    
    @time_execution
    async def stop_strategy(self, strategy_name: str) -> bool:
        """Stop a strategy.
        
        Args:
            strategy_name: Name of the strategy to stop
            
        Returns:
            True if strategy stopped successfully, False otherwise
        """
        try:
            strategy = self._strategies.get(strategy_name)
            if not strategy:
                logger.error("Strategy not found", strategy_name=strategy_name)
                return False
            
            await strategy.stop()
            return True
            
        except Exception as e:
            logger.error("Failed to stop strategy", 
                        strategy_name=strategy_name, 
                        error=str(e))
            return False
    
    @time_execution
    async def restart_strategy(self, strategy_name: str) -> bool:
        """Restart a strategy.
        
        Args:
            strategy_name: Name of the strategy to restart
            
        Returns:
            True if strategy restarted successfully, False otherwise
        """
        try:
            # Stop strategy
            await self.stop_strategy(strategy_name)
            
            # Wait a bit before restarting
            await asyncio.sleep(1)
            
            # Start strategy
            return await self.start_strategy(strategy_name)
            
        except Exception as e:
            logger.error("Failed to restart strategy", 
                        strategy_name=strategy_name, 
                        error=str(e))
            return False
    
    def get_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """Get strategy instance by name.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy instance or None if not found
        """
        return self._strategies.get(strategy_name)
    
    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """Get all strategy instances.
        
        Returns:
            Dictionary of strategy instances
        """
        return self._strategies.copy()
    
    def get_strategy_status(self, strategy_name: str) -> Optional[StrategyStatus]:
        """Get strategy status.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy status or None if not found
        """
        strategy = self._strategies.get(strategy_name)
        return strategy.status if strategy else None
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get strategy performance summary.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Performance summary or None if not found
        """
        strategy = self._strategies.get(strategy_name)
        return strategy.get_performance_summary() if strategy else None
    
    def set_risk_manager(self, risk_manager: BaseRiskManager) -> None:
        """Set risk manager for all strategies.
        
        Args:
            risk_manager: Risk manager instance
        """
        self._risk_manager = risk_manager
        
        # Update existing strategies
        for strategy in self._strategies.values():
            strategy.set_risk_manager(risk_manager)
        
        logger.info("Risk manager set for all strategies")
    
    def set_exchange(self, exchange: BaseExchange) -> None:
        """Set exchange for all strategies.
        
        Args:
            exchange: Exchange instance
        """
        self._exchange = exchange
        
        # Update existing strategies
        for strategy in self._strategies.values():
            strategy.set_exchange(exchange)
        
        logger.info("Exchange set for all strategies")
    
    @time_execution
    async def hot_swap_strategy(self, strategy_name: str, new_config: Dict[str, Any]) -> bool:
        """Hot swap a strategy with new configuration.
        
        Args:
            strategy_name: Name of the strategy to swap
            new_config: New configuration
            
        Returns:
            True if swap successful, False otherwise
        """
        try:
            strategy = self._strategies.get(strategy_name)
            if not strategy:
                logger.error("Strategy not found for hot swap", strategy_name=strategy_name)
                return False
            
            # Store current status
            current_status = strategy.status
            
            # Stop strategy if running
            if strategy.status == StrategyStatus.RUNNING:
                await strategy.stop()
            
            # Update configuration
            strategy.update_config(new_config)
            
            # Restart if it was running
            if current_status == StrategyStatus.RUNNING:
                await strategy.start()
            
            logger.info("Strategy hot swapped successfully", 
                       strategy_name=strategy_name)
            return True
            
        except Exception as e:
            logger.error("Failed to hot swap strategy", 
                        strategy_name=strategy_name, 
                        error=str(e))
            return False
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy.
        
        Args:
            strategy_name: Name of the strategy to remove
            
        Returns:
            True if strategy removed successfully, False otherwise
        """
        try:
            strategy = self._strategies.pop(strategy_name, None)
            if strategy:
                logger.info("Strategy removed", strategy_name=strategy_name)
                return True
            else:
                logger.warning("Strategy not found for removal", strategy_name=strategy_name)
                return False
                
        except Exception as e:
            logger.error("Failed to remove strategy", 
                        strategy_name=strategy_name, 
                        error=str(e))
            return False
    
    @time_execution
    async def shutdown_all_strategies(self) -> None:
        """Shutdown all strategies."""
        logger.info("Shutting down all strategies")
        
        shutdown_tasks = []
        for strategy_name, strategy in self._strategies.items():
            if strategy.status in [StrategyStatus.RUNNING, StrategyStatus.PAUSED]:
                task = asyncio.create_task(strategy.stop())
                shutdown_tasks.append((strategy_name, task))
        
        # Wait for all strategies to stop
        for strategy_name, task in shutdown_tasks:
            try:
                await task
                logger.info("Strategy shutdown completed", strategy_name=strategy_name)
            except Exception as e:
                logger.error("Failed to shutdown strategy", 
                            strategy_name=strategy_name, 
                            error=str(e))
        
        logger.info("All strategies shutdown completed")
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of all strategies.
        
        Returns:
            Strategy summary dictionary
        """
        summary = {
            "total_strategies": len(self._strategies),
            "running_strategies": 0,
            "stopped_strategies": 0,
            "error_strategies": 0,
            "strategies": {}
        }
        
        for name, strategy in self._strategies.items():
            summary["strategies"][name] = {
                "status": strategy.status.value,
                "version": strategy.version,
                "symbols": strategy.config.symbols,
                "timeframe": strategy.config.timeframe,
                "performance": strategy.get_performance_summary()
            }
            
            if strategy.status == StrategyStatus.RUNNING:
                summary["running_strategies"] += 1
            elif strategy.status == StrategyStatus.STOPPED:
                summary["stopped_strategies"] += 1
            elif strategy.status == StrategyStatus.ERROR:
                summary["error_strategies"] += 1
        
        return summary 