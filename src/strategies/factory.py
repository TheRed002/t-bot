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

# MANDATORY: Import from P-013 - Dynamic strategies
from src.strategies.static.mean_reversion import MeanReversionStrategy
from src.strategies.static.trend_following import TrendFollowingStrategy
from src.strategies.static.breakout import BreakoutStrategy
from src.strategies.dynamic.adaptive_momentum import AdaptiveMomentumStrategy
from src.strategies.dynamic.volatility_breakout import VolatilityBreakoutStrategy

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
        # Static strategies
        self._register_strategy_class("mean_reversion", MeanReversionStrategy)
        self._register_strategy_class(
            "trend_following", TrendFollowingStrategy)
        self._register_strategy_class("breakout", BreakoutStrategy)

        # Dynamic strategies (P-013)
        self._register_strategy_class(
            "adaptive_momentum", AdaptiveMomentumStrategy)
        self._register_strategy_class(
            "volatility_breakout",
            VolatilityBreakoutStrategy)

        logger.info("Built-in strategies registered",
                    static_count=3,
                    dynamic_count=2)

    def _register_strategy_class(
            self,
            name: str,
            strategy_class: Type[BaseStrategy]) -> None:
        """Register a strategy class.

        Args:
            name: Strategy name
            strategy_class: Strategy class to register
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise ValidationError(
                f"Strategy class must inherit from BaseStrategy: {strategy_class}")

        self._strategy_classes[name] = strategy_class
        logger.info(
            "Strategy class registered",
            name=name,
            class_name=strategy_class.__name__)

    @time_execution
    def create_strategy(self, strategy_name: str,
                        config: Dict[str, Any]) -> BaseStrategy:
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
            self._strategies[strategy_name] = strategy

            logger.info("Strategy created successfully",
                        name=strategy_name,
                        class_name=strategy_class.__name__)

            return strategy

        except Exception as e:
            logger.error("Strategy creation failed",
                         strategy_name=strategy_name, error=str(e))
            raise ValidationError(f"Strategy creation failed: {str(e)}")

    def _get_strategy_class(
            self, strategy_name: str) -> Optional[Type[BaseStrategy]]:
        """Get strategy class by name.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy class or None if not found
        """
        # First check if already registered
        if strategy_name in self._strategy_classes:
            return self._strategy_classes[strategy_name]

        # Try dynamic import from static strategies
        try:
            module = importlib.import_module(
                f"src.strategies.static.{strategy_name}")
            strategy_class = getattr(
                module, f"{
                    strategy_name.title().replace(
                        '_', '')}Strategy", None)
            if strategy_class and issubclass(strategy_class, BaseStrategy):
                return strategy_class
        except (ImportError, AttributeError):
            pass

        # Try dynamic import from dynamic strategies
        try:
            module = importlib.import_module(
                f"src.strategies.dynamic.{strategy_name}")
            strategy_class = getattr(
                module, f"{
                    strategy_name.title().replace(
                        '_', '')}Strategy", None)
            if strategy_class and issubclass(strategy_class, BaseStrategy):
                return strategy_class
        except (ImportError, AttributeError):
            pass

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
            strategy = self.get_strategy(strategy_name)
            if not strategy:
                logger.error("Strategy not found", strategy_name=strategy_name)
                return False

            # Start strategy
            await strategy.start()

            logger.info(
                "Strategy started successfully",
                strategy_name=strategy_name)
            return True

        except Exception as e:
            logger.error("Strategy start failed",
                         strategy_name=strategy_name, error=str(e))
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
            strategy = self.get_strategy(strategy_name)
            if not strategy:
                logger.error("Strategy not found", strategy_name=strategy_name)
                return False

            # Stop strategy
            await strategy.stop()

            logger.info(
                "Strategy stopped successfully",
                strategy_name=strategy_name)
            return True

        except Exception as e:
            logger.error("Strategy stop failed",
                         strategy_name=strategy_name, error=str(e))
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
            if not await self.stop_strategy(strategy_name):
                return False

            # Wait a moment for cleanup
            await asyncio.sleep(1)

            # Start strategy
            if not await self.start_strategy(strategy_name):
                return False

            logger.info(
                "Strategy restarted successfully",
                strategy_name=strategy_name)
            return True

        except Exception as e:
            logger.error("Strategy restart failed",
                         strategy_name=strategy_name, error=str(e))
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

    def get_strategy_status(
            self,
            strategy_name: str) -> Optional[StrategyStatus]:
        """Get strategy status.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy status or None if not found
        """
        strategy = self.get_strategy(strategy_name)
        return strategy.status if strategy else None

    def get_strategy_performance(
            self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get strategy performance metrics.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Performance metrics or None if not found
        """
        strategy = self.get_strategy(strategy_name)
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
    async def hot_swap_strategy(
            self, strategy_name: str, new_config: Dict[str, Any]) -> bool:
        """Hot-swap a strategy with new configuration.

        Args:
            strategy_name: Name of the strategy to swap
            new_config: New configuration

        Returns:
            True if swap successful, False otherwise
        """
        try:
            # Validate new configuration
            validate_strategy_config(new_config)

            # Get current strategy
            current_strategy = self.get_strategy(strategy_name)
            if not current_strategy:
                logger.error(
                    "Strategy not found for hot swap",
                    strategy_name=strategy_name)
                return False

            # Update the existing strategy's configuration
            current_strategy.update_config(new_config)

            logger.info("Strategy hot-swapped successfully",
                        strategy_name=strategy_name,
                        class_name=current_strategy.__class__.__name__)

            return True

        except Exception as e:
            logger.error("Strategy hot-swap failed",
                         strategy_name=strategy_name, error=str(e))
            return False

    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy.

        Args:
            strategy_name: Name of the strategy to remove

        Returns:
            True if strategy removed successfully, False otherwise
        """
        try:
            if strategy_name not in self._strategies:
                logger.warning(
                    "Strategy not found for removal",
                    strategy_name=strategy_name)
                return False

            # Remove strategy instance
            del self._strategies[strategy_name]

            logger.info(
                "Strategy removed successfully",
                strategy_name=strategy_name)
            return True

        except Exception as e:
            logger.error("Strategy removal failed",
                         strategy_name=strategy_name, error=str(e))
            return False

    @time_execution
    async def shutdown_all_strategies(self) -> None:
        """Shutdown all strategies."""
        try:
            shutdown_tasks = []

            for strategy_name, strategy in self._strategies.items():
                try:
                    await strategy.stop()
                    shutdown_tasks.append(strategy_name)
                except Exception as e:
                    logger.error("Failed to shutdown strategy",
                                 strategy_name=strategy_name, error=str(e))

            # Clear all strategies
            self._strategies.clear()

            logger.info("All strategies shutdown",
                        shutdown_count=len(shutdown_tasks),
                        total_count=len(self._strategies))

        except Exception as e:
            logger.error("Strategy shutdown failed", error=str(e))

    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of all strategies.

        Returns:
            Dictionary with strategy summary information
        """
        running_count = 0
        stopped_count = 0
        error_count = 0

        for strategy in self._strategies.values():
            if strategy.status == StrategyStatus.RUNNING:
                running_count += 1
            elif strategy.status == StrategyStatus.STOPPED:
                stopped_count += 1
            elif strategy.status == StrategyStatus.ERROR:
                error_count += 1

        summary = {
            'total_strategies': len(self._strategies),
            'running_strategies': running_count,
            'stopped_strategies': stopped_count,
            'error_strategies': error_count,
            'strategies': {}
        }

        for name, strategy in self._strategies.items():
            strategy_info = {
                'name': name,
                'class': strategy.__class__.__name__,
                'status': strategy.status.value,
                'type': strategy.config.strategy_type.value if hasattr(
                    strategy.config,
                    'strategy_type') else 'unknown'}
            summary['strategies'][name] = strategy_info

        return summary
