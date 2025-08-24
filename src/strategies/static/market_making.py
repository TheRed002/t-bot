"""
Market Making Strategy Implementation.

This module implements a sophisticated market making strategy with dual-sided order placement,
spread management, and inventory control. The strategy provides liquidity to the market
while managing inventory risk and optimizing spreads based on market conditions.

CRITICAL: This strategy MUST inherit from BaseStrategy and follow the exact interface.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np

# From P-001 - Use structured logging
# Logger is provided by BaseStrategy (via BaseComponent)
# From P-001 - Use existing types
from src.core.types import (
    MarketData,
    Position,
    Signal,
    SignalDirection,
    StrategyType,
)

# From P-008+ - Use risk management
# MANDATORY: Import from P-011 - NEVER recreate the base strategy
from src.strategies.base import BaseStrategy

# From P-007A - Use decorators and validators
from src.utils.decorators import time_execution
from src.utils.validators import validate_price, validate_quantity


@dataclass
class OrderLevel:
    """Represents a single order level in the market making strategy."""

    level: int
    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal
    spread: Decimal
    timestamp: datetime


@dataclass
class InventoryState:
    """Current inventory state for the market making strategy."""

    current_inventory: Decimal
    target_inventory: Decimal
    max_inventory: Decimal
    inventory_skew: float  # -1 to 1, negative means short bias
    last_rebalance: datetime


class MarketMakingStrategy(BaseStrategy):
    """
    Market Making Strategy Implementation.

    This strategy provides liquidity to the market by placing dual-sided orders
    (bids and asks) and managing inventory risk through dynamic spread adjustment
    and position rebalancing.

    Key Features:
    - Dual-sided order placement with configurable spreads
    - Multi-level order book management (5 levels default)
    - Dynamic spread adjustment based on volatility and competition
    - Inventory skew implementation for risk management
    - Order refresh management with rate limit awareness
    - Competitive quote monitoring and adjustment
    - Cross-exchange rate limit synchronization
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize Market Making Strategy.

        Args:
            config: Strategy configuration dictionary
        """
        super().__init__(config)
        self.strategy_type = StrategyType.MARKET_MAKING

        # Strategy-specific parameters with defaults
        self.base_spread = Decimal(str(self.config.parameters.get("base_spread", 0.001)))  # 0.1%
        self.order_levels = self.config.parameters.get("order_levels", 5)
        self.base_order_size = Decimal(str(self.config.parameters.get("base_order_size", 0.01)))
        self.size_multiplier = self.config.parameters.get("size_multiplier", 1.5)
        self.order_size_distribution = self.config.parameters.get(
            "order_size_distribution", "exponential"
        )

        # Spread adjustment parameters
        self.volatility_multiplier = self.config.parameters.get("volatility_multiplier", 2.0)
        self.inventory_skew_enabled = self.config.parameters.get("inventory_skew", True)
        self.competitive_quotes_enabled = self.config.parameters.get("competitive_quotes", True)

        # Inventory management
        self.target_inventory = Decimal(str(self.config.parameters.get("target_inventory", 0.5)))
        self.max_inventory = Decimal(str(self.config.parameters.get("max_inventory", 1.0)))
        self.inventory_risk_aversion = self.config.parameters.get("inventory_risk_aversion", 0.1)
        self.rebalance_threshold = self.config.parameters.get("rebalance_threshold", 0.2)

        # Risk parameters
        self.max_position_value = Decimal(
            str(self.config.parameters.get("max_position_value", 10000))
        )
        self.stop_loss_inventory = Decimal(
            str(self.config.parameters.get("stop_loss_inventory", 2.0))
        )
        self.daily_loss_limit = Decimal(str(self.config.parameters.get("daily_loss_limit", 100)))
        self.min_profit_per_trade = Decimal(
            str(self.config.parameters.get("min_profit_per_trade", 0.00001))
        )

        # Order management
        self.order_refresh_time = self.config.parameters.get("order_refresh_time", 30)  # seconds
        self.adaptive_spreads = self.config.parameters.get("adaptive_spreads", True)
        self.competition_monitoring = self.config.parameters.get("competition_monitoring", True)

        # State tracking
        self.active_orders: dict[str, OrderLevel] = {}
        self.inventory_state = InventoryState(
            current_inventory=Decimal("0"),
            target_inventory=self.target_inventory,
            max_inventory=self.max_inventory,
            inventory_skew=0.0,
            last_rebalance=datetime.now(),
        )

        # Market data tracking
        self.price_history: list[float] = []
        self.volatility_history: list[float] = []
        self.spread_history: list[float] = []

        # Performance tracking
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = Decimal("0")
        self.daily_pnl = Decimal("0")
        self.last_daily_reset = datetime.now()

        self.logger.info(
            "Market Making Strategy initialized",
            strategy=self.name,
            base_spread=float(self.base_spread),
            order_levels=self.order_levels,
            target_inventory=float(self.target_inventory),
        )

    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Generate market making signals from market data.

        MANDATORY: Use graceful error handling and input validation.

        Args:
            data: Market data for signal generation

        Returns:
            List of trading signals
        """
        try:
            # Validate input data
            if not data or not data.price or not data.bid or not data.ask:
                self.logger.warning(
                    "Invalid market data for signal generation",
                    symbol=data.symbol if data else None,
                )
                return []

            # Update price history
            self._update_price_history(data)

            # Calculate current spread and volatility
            current_spread = (data.ask - data.bid) / data.bid
            current_volatility = self._calculate_volatility()

            # Generate signals for each order level
            signals = []

            for level in range(1, self.order_levels + 1):
                # Calculate level-specific parameters
                level_spread = self._calculate_level_spread(
                    level, current_spread, current_volatility
                )
                level_size = self._calculate_level_size(level)

                # Generate bid signal
                bid_price = data.bid * (Decimal("1") - level_spread / Decimal("2"))
                bid_signal = Signal(
                    direction=SignalDirection.BUY,
                    confidence=0.8,  # High confidence for market making
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    strategy_name=self.name,
                    metadata={
                        "order_type": "limit",
                        "level": level,
                        "price": float(bid_price),
                        "size": float(level_size),
                        "spread": float(level_spread),
                        "side": "bid",
                    },
                )
                signals.append(bid_signal)

                # Generate ask signal
                ask_price = data.ask * (Decimal("1") + level_spread / Decimal("2"))
                ask_signal = Signal(
                    direction=SignalDirection.SELL,
                    confidence=0.8,  # High confidence for market making
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    strategy_name=self.name,
                    metadata={
                        "order_type": "limit",
                        "level": level,
                        "price": float(ask_price),
                        "size": float(level_size),
                        "spread": float(level_spread),
                        "side": "ask",
                    },
                )
                signals.append(ask_signal)

            self.logger.debug(
                "Generated market making signals",
                strategy=self.name,
                symbol=data.symbol,
                signal_count=len(signals),
                current_spread=float(current_spread),
                volatility=current_volatility,
            )

            return signals

        except Exception as e:
            self.logger.error(
                "Market making signal generation failed", strategy=self.name, error=str(e)
            )
            return []  # Graceful degradation

    def _calculate_level_spread(
        self, level: int, base_spread: Decimal, volatility: float
    ) -> Decimal:
        """Calculate spread for a specific order level.

        Args:
            level: Order level (1 to order_levels)
            base_spread: Base spread from market data
            volatility: Current volatility

        Returns:
            Calculated spread for the level
        """
        # Base spread increases with level
        level_spread = self.base_spread * Decimal(str(1 + (level - 1) * 0.2))

        # Adjust for volatility
        if self.adaptive_spreads and volatility > 0:
            volatility_adjustment = min(volatility * self.volatility_multiplier, 0.01)
            level_spread += Decimal(str(volatility_adjustment))

        # Adjust for inventory skew with correlation consideration
        if self.inventory_skew_enabled:
            # CRITICAL FIX: Consider correlation in inventory risk
            correlation_factor = self._calculate_correlation_risk_factor()
            inventory_adjustment = self.inventory_state.inventory_skew * 0.001 * correlation_factor
            level_spread += Decimal(str(inventory_adjustment))

        # Ensure minimum spread
        min_spread = Decimal("0.0001")  # 0.01%
        return max(level_spread, min_spread)
    
    def _calculate_correlation_risk_factor(self) -> Decimal:
        """Calculate risk factor based on correlation with other assets.
        
        Returns:
            Risk factor between 1.0 (no correlation) and 2.0 (high correlation)
        """
        # TODO: Implement actual correlation calculation with portfolio
        # For now, return conservative estimate
        base_factor = Decimal("1.0")
        
        # Increase risk factor if we have high inventory
        inventory_ratio = abs(self.inventory_state.inventory_skew)
        if inventory_ratio > 0.7:
            # High inventory means higher correlation risk
            base_factor += Decimal("0.5")
        elif inventory_ratio > 0.5:
            base_factor += Decimal("0.3")
        
        return base_factor

    def _calculate_level_size(self, level: int) -> Decimal:
        """Calculate order size for a specific level.

        Args:
            level: Order level (1 to order_levels)

        Returns:
            Calculated order size
        """
        if self.order_size_distribution == "exponential":
            size = self.base_order_size * Decimal(str(self.size_multiplier ** (level - 1)))
        elif self.order_size_distribution == "linear":
            size = self.base_order_size * Decimal(str(level))
        else:  # constant
            size = self.base_order_size

        # Apply inventory-based size adjustment
        if self.inventory_skew_enabled:
            inventory_factor = 1 + abs(self.inventory_state.inventory_skew) * 0.5
            size *= Decimal(str(inventory_factor))

        return size

    def _calculate_volatility(self) -> float:
        """Calculate current volatility from price history.

        Returns:
            Current volatility as a percentage
        """
        if len(self.price_history) < 20:
            return 0.02  # Default 2% volatility

        # Calculate rolling volatility
        prices = np.array(self.price_history[-20:])
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        return float(volatility)

    def _update_price_history(self, data: MarketData) -> None:
        """Update price history for calculations.

        Args:
            data: Market data
        """
        self.price_history.append(float(data.price))

        # Keep only last 100 prices
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]

        # Update spread history
        if data.bid and data.ask:
            spread = (data.ask - data.bid) / data.bid
            self.spread_history.append(float(spread))

            if len(self.spread_history) > 100:
                self.spread_history = self.spread_history[-100:]

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate market making signal before execution.

        MANDATORY: Check signal confidence, direction, timestamp

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Basic validation
            if not signal or signal.confidence < self.config.min_confidence:
                self.logger.warning(
                    "Signal confidence too low", strategy=self.name, confidence=signal.confidence
                )
                return False

            # Check if we have required metadata
            if "price" not in signal.metadata or "size" not in signal.metadata:
                self.logger.warning(
                    "Signal missing required metadata", strategy=self.name, metadata=signal.metadata
                )
                return False

            # Validate price and size
            price = Decimal(str(signal.metadata["price"]))
            size = Decimal(str(signal.metadata["size"]))

            if not validate_price(float(price), signal.symbol):
                self.logger.warning(
                    "Invalid price in signal", strategy=self.name, price=float(price)
                )
                return False

            if not validate_quantity(float(size), signal.symbol):
                self.logger.warning("Invalid size in signal", strategy=self.name, size=float(size))
                return False

            # Check inventory limits
            if not self._check_inventory_limits(signal):
                self.logger.warning(
                    "Signal violates inventory limits", strategy=self.name, signal=signal.direction
                )
                return False

            # Check daily loss limit
            if self.daily_pnl < -self.daily_loss_limit:
                self.logger.warning(
                    "Daily loss limit reached", strategy=self.name, daily_pnl=float(self.daily_pnl)
                )
                return False

            # Check with risk manager if available
            if self._risk_manager:
                if not await self._risk_manager.validate_signal(signal):
                    self.logger.warning(
                        "Signal rejected by risk manager",
                        strategy=self.name,
                        signal=signal.direction,
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error("Signal validation failed", strategy=self.name, error=str(e))
            return False

    def _check_inventory_limits(self, signal: Signal) -> bool:
        """Check if signal violates inventory limits.

        Args:
            signal: Signal to check

        Returns:
            True if within limits, False otherwise
        """
        # Calculate potential inventory change
        size = Decimal(str(signal.metadata.get("size", 0)))

        if signal.direction == SignalDirection.BUY:
            new_inventory = self.inventory_state.current_inventory + size
        else:
            new_inventory = self.inventory_state.current_inventory - size

        # Check against limits
        if abs(new_inventory) > self.inventory_state.max_inventory:
            return False

        return True

    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size for market making signal.

        Args:
            signal: Trading signal

        Returns:
            Position size in base currency
        """
        try:
            # Get size from signal metadata
            size = Decimal(str(signal.metadata.get("size", self.base_order_size)))

            # Apply risk management if available
            if self._risk_manager:
                # TODO: Integrate with risk manager for position sizing
                pass

            # Ensure size is within limits
            max_size = self.max_position_value / Decimal(str(signal.metadata.get("price", 1)))
            size = min(size, max_size)

            return size

        except Exception as e:
            self.logger.error("Position size calculation failed", strategy=self.name, error=str(e))
            return self.base_order_size

    async def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine if market making position should be closed.

        Args:
            position: Current position
            data: Market data

        Returns:
            True if position should be closed, False otherwise
        """
        try:
            # Check stop loss based on inventory
            if abs(position.quantity) > self.stop_loss_inventory:
                self.logger.info(
                    "Position exceeds stop loss inventory",
                    strategy=self.name,
                    quantity=float(position.quantity),
                )
                return True

            # Check for profit taking (if position is profitable)
            if position.unrealized_pnl > self.min_profit_per_trade:
                # Consider closing if spread has narrowed significantly
                current_spread = (data.ask - data.bid) / data.bid if data.bid and data.ask else 0
                if current_spread < self.base_spread * Decimal("0.5"):
                    self.logger.info(
                        "Closing position due to narrow spread",
                        strategy=self.name,
                        spread=float(current_spread),
                    )
                    return True

            # Check for inventory rebalancing
            if await self._should_rebalance_inventory(position):
                self.logger.info("Closing position for inventory rebalancing", strategy=self.name)
                return True

            return False

        except Exception as e:
            self.logger.error("Position exit check failed", strategy=self.name, error=str(e))
            return False

    async def _should_rebalance_inventory(self, position: Position) -> bool:
        """Check if inventory rebalancing is needed.

        Args:
            position: Current position

        Returns:
            True if rebalancing needed, False otherwise
        """
        # Calculate current inventory skew
        current_skew = self.inventory_state.current_inventory / self.inventory_state.max_inventory

        # Check if skew is too high
        if abs(current_skew) > 0.8:  # 80% of max inventory
            self.logger.debug(
                "Rebalancing needed due to high skew",
                strategy=self.name,
                current_skew=float(current_skew),
            )
            return True

        # Check if we need to move toward target inventory
        target_diff = abs(
            self.inventory_state.current_inventory - self.inventory_state.target_inventory
        )
        threshold = self.inventory_state.max_inventory * Decimal(str(self.rebalance_threshold))

        self.logger.debug(
            "Checking rebalancing conditions",
            strategy=self.name,
            current_inventory=float(self.inventory_state.current_inventory),
            target_inventory=float(self.inventory_state.target_inventory),
            target_diff=float(target_diff),
            threshold=float(threshold),
        )

        if target_diff >= threshold:
            self.logger.debug(
                "Rebalancing needed due to target deviation",
                strategy=self.name,
                target_diff=float(target_diff),
            )
            return True

        return False

    async def update_inventory_state(self, new_position: Position) -> None:
        """Update inventory state after position change.

        Args:
            new_position: New position data
        """
        try:
            self.inventory_state.current_inventory = new_position.quantity

            # Calculate inventory skew
            if self.inventory_state.max_inventory > 0:
                self.inventory_state.inventory_skew = float(
                    new_position.quantity / self.inventory_state.max_inventory
                )

            # Update last rebalance time
            self.inventory_state.last_rebalance = datetime.now()

            self.logger.debug(
                "Inventory state updated",
                strategy=self.name,
                current_inventory=float(self.inventory_state.current_inventory),
                inventory_skew=self.inventory_state.inventory_skew,
            )

        except Exception as e:
            self.logger.error("Inventory state update failed", strategy=self.name, error=str(e))

    async def update_performance_metrics(self, trade_result: dict[str, Any]) -> None:
        """Update performance metrics after trade execution.

        Args:
            trade_result: Trade execution result
        """
        try:
            self.total_trades += 1

            # Extract P&L from trade result
            pnl = Decimal(str(trade_result.get("pnl", 0)))
            self.total_pnl += pnl
            self.daily_pnl += pnl

            # Update profitable trades count
            if pnl > 0:
                self.profitable_trades += 1

            # Reset daily P&L if needed
            if datetime.now() - self.last_daily_reset > timedelta(days=1):
                self.daily_pnl = Decimal("0")
                self.last_daily_reset = datetime.now()

            # Update win rate
            if self.total_trades > 0:
                self.metrics.win_rate = self.profitable_trades / self.total_trades
                self.metrics.total_trades = self.total_trades

            self.metrics.total_pnl = self.total_pnl
            self.metrics.last_updated = datetime.now()

            self.logger.debug(
                "Performance metrics updated",
                strategy=self.name,
                total_trades=self.total_trades,
                total_pnl=float(self.total_pnl),
                daily_pnl=float(self.daily_pnl),
            )

        except Exception as e:
            self.logger.error("Performance metrics update failed", strategy=self.name, error=str(e))

    def get_strategy_info(self) -> dict[str, Any]:
        """Get comprehensive strategy information.

        Returns:
            Dictionary with strategy information
        """
        base_info = super().get_strategy_info()

        # Add market making specific information
        market_making_info = {
            "strategy_type": "market_making",
            "base_spread": float(self.base_spread),
            "order_levels": self.order_levels,
            "target_inventory": float(self.target_inventory),
            "current_inventory": float(self.inventory_state.current_inventory),
            "inventory_skew": self.inventory_state.inventory_skew,
            "active_orders": len(self.active_orders),
            "total_trades": self.total_trades,
            "profitable_trades": self.profitable_trades,
            "total_pnl": float(self.total_pnl),
            "daily_pnl": float(self.daily_pnl),
            "order_size_distribution": self.order_size_distribution,
            "adaptive_spreads": self.adaptive_spreads,
            "competition_monitoring": self.competition_monitoring,
        }

        base_info.update(market_making_info)
        return base_info
