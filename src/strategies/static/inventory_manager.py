"""
Inventory Manager for Market Making Strategy.

This module implements inventory risk management for the market making strategy,
including target inventory maintenance, inventory skew adjustments, position
rebalancing triggers, and emergency inventory liquidation procedures.

CRITICAL: This integrates with the market making strategy and risk management framework.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.exceptions import RiskManagementError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import OrderRequest, OrderSide, OrderType, Position
from src.strategies.dependencies import StrategyServiceContainer

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution

# MANDATORY: Import from P-008+


class InventoryManager:
    """
    Inventory Manager for Market Making Strategy.

    This class manages inventory risk for market making operations, including:
    - Target inventory maintenance (50% of max position default)
    - Inventory risk aversion adjustments
    - Position rebalancing triggers
    - Currency hedging for multi-asset market making
    - Inventory-based spread skewing
    - Emergency inventory liquidation procedures
    """

    def __init__(self, config: dict[str, Any], services: "StrategyServiceContainer | None" = None):
        """
        Initialize Inventory Manager.

        Args:
            config: Configuration dictionary with inventory parameters
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.InventoryManager")

        # Inventory parameters
        self.target_inventory = Decimal(str(config.get("target_inventory", 0.5)))
        self.max_inventory = Decimal(str(config.get("max_inventory", 1.0)))
        self.min_inventory = Decimal(str(config.get("min_inventory", -1.0)))
        self.inventory_risk_aversion = config.get("inventory_risk_aversion", 0.1)

        # Rebalancing parameters
        self.rebalance_threshold = config.get("rebalance_threshold", 0.2)  # 20% deviation
        self.rebalance_frequency_hours = config.get("rebalance_frequency_hours", 4)
        self.max_rebalance_size = Decimal(str(config.get("max_rebalance_size", 0.5)))

        # Emergency parameters
        self.emergency_threshold = config.get("emergency_threshold", 0.8)  # 80% of max
        self.emergency_liquidation_enabled = config.get("emergency_liquidation_enabled", True)

        # Trading symbol
        self.symbol = config.get("symbol", "BTCUSDT")
        
        # State tracking
        self.current_inventory = Decimal("0")
        self.inventory_skew = 0.0  # -1 to 1
        self.last_rebalance = datetime.now(timezone.utc)
        self.rebalance_count = 0
        self.emergency_count = 0

        # Performance tracking
        self.total_rebalance_cost = Decimal("0")
        self.total_emergency_cost = Decimal("0")

        self.logger.info(
            "Inventory Manager initialized",
            target_inventory=float(self.target_inventory),
            max_inventory=float(self.max_inventory),
            rebalance_threshold=self.rebalance_threshold,
        )

    @time_execution
    async def update_inventory(self, position: Position) -> None:
        """
        Update current inventory based on position change.

        Args:
            position: Current position data
        """
        try:
            self.current_inventory = position.quantity

            # Calculate inventory skew (-1 to 1)
            if self.max_inventory > 0:
                self.inventory_skew = float(position.quantity / self.max_inventory)
            else:
                self.inventory_skew = 0.0

            self.logger.debug(
                "Inventory updated",
                current_inventory=float(self.current_inventory),
                inventory_skew=self.inventory_skew,
                position_quantity=float(position.quantity),
            )

        except Exception as e:
            self.logger.error("Inventory update failed", error=str(e))
            raise RiskManagementError(f"Inventory update failed: {e!s}")

    @time_execution
    async def should_rebalance(self) -> bool:
        """
        Check if inventory rebalancing is needed.

        Returns:
            True if rebalancing is needed, False otherwise
        """
        try:
            # Check if we're within rebalancing threshold
            inventory_deviation = abs(self.current_inventory - self.target_inventory)
            threshold = self.max_inventory * Decimal(str(self.rebalance_threshold))

            if inventory_deviation >= threshold:
                self.logger.info(
                    "Inventory rebalancing needed",
                    current_inventory=float(self.current_inventory),
                    target_inventory=float(self.target_inventory),
                    deviation=float(inventory_deviation),
                    threshold=float(threshold),
                )
                return True

            # Check if we've exceeded max inventory
            if abs(self.current_inventory) > self.max_inventory:
                self.logger.warning(
                    "Inventory exceeds maximum limit",
                    current_inventory=float(self.current_inventory),
                    max_inventory=float(self.max_inventory),
                )
                return True

            # Check time-based rebalancing
            time_since_rebalance = datetime.now(timezone.utc) - self.last_rebalance
            if time_since_rebalance > timedelta(hours=self.rebalance_frequency_hours):
                self.logger.info(
                    "Time-based rebalancing triggered",
                    hours_since_rebalance=time_since_rebalance.total_seconds() / 3600,
                )
                return True

            return False

        except Exception as e:
            self.logger.error("Rebalancing check failed", error=str(e))
            return False

    @time_execution
    async def calculate_rebalance_orders(self, current_price: Decimal) -> list[OrderRequest]:
        """
        Calculate rebalancing orders to move inventory toward target.

        Args:
            current_price: Current market price

        Returns:
            List of rebalancing orders
        """
        try:
            orders: list[Any] = []

            # Calculate required inventory change
            required_change = self.target_inventory - self.current_inventory

            # Limit rebalancing size
            if abs(required_change) > self.max_rebalance_size:
                if required_change > 0:
                    required_change = self.max_rebalance_size
                else:
                    required_change = -self.max_rebalance_size

            if abs(required_change) < Decimal("0.001"):  # Minimum change threshold
                self.logger.debug(
                    "Rebalancing change too small", required_change=float(required_change)
                )
                return orders

            # Create rebalancing order
            if required_change > 0:
                # Need to buy to increase inventory
                order = OrderRequest(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=abs(required_change),
                    price=None,  # Market order
                    client_order_id=f"rebalance_{datetime.now(timezone.utc).timestamp()}",
                )
                orders.append(order)

                self.logger.info(
                    "Created rebalancing buy order",
                    quantity=float(abs(required_change)),
                    current_inventory=float(self.current_inventory),
                    target_inventory=float(self.target_inventory),
                )

            elif required_change < 0:
                # Need to sell to decrease inventory
                order = OrderRequest(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=abs(required_change),
                    price=None,  # Market order
                    client_order_id=f"rebalance_{datetime.now(timezone.utc).timestamp()}",
                )
                orders.append(order)

                self.logger.info(
                    "Created rebalancing sell order",
                    quantity=float(abs(required_change)),
                    current_inventory=float(self.current_inventory),
                    target_inventory=float(self.target_inventory),
                )

            return orders

        except Exception as e:
            self.logger.error("Rebalancing order calculation failed", error=str(e))
            return []

    @time_execution
    async def should_emergency_liquidate(self) -> bool:
        """
        Check if emergency liquidation is needed.

        Returns:
            True if emergency liquidation is needed, False otherwise
        """
        try:
            if not self.emergency_liquidation_enabled:
                return False

            # Check if inventory exceeds emergency threshold
            emergency_limit = self.max_inventory * Decimal(str(self.emergency_threshold))

            if abs(self.current_inventory) > emergency_limit:
                self.logger.warning(
                    "Emergency liquidation threshold exceeded",
                    current_inventory=float(self.current_inventory),
                    emergency_limit=float(emergency_limit),
                )
                return True

            # Check for rapid inventory accumulation
            # TBD: Implement time-based emergency detection for rapid inventory changes

            return False

        except Exception as e:
            self.logger.error("Emergency liquidation check failed", error=str(e))
            return False

    @time_execution
    async def calculate_emergency_orders(self, current_price: Decimal) -> list[OrderRequest]:
        """
        Calculate emergency liquidation orders.

        Args:
            current_price: Current market price

        Returns:
            List of emergency liquidation orders
        """
        try:
            orders: list[Any] = []

            if self.current_inventory > 0:
                # Need to sell entire inventory
                order = OrderRequest(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=self.current_inventory,
                    price=None,  # Market order
                    client_order_id=f"emergency_{datetime.now(timezone.utc).timestamp()}",
                )
                orders.append(order)

                self.logger.warning(
                    "Created emergency liquidation sell order",
                    quantity=float(self.current_inventory),
                    current_inventory=float(self.current_inventory),
                )

            elif self.current_inventory < 0:
                # Need to buy to cover short position
                order = OrderRequest(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=abs(self.current_inventory),
                    price=None,  # Market order
                    client_order_id=f"emergency_{datetime.now(timezone.utc).timestamp()}",
                )
                orders.append(order)

                self.logger.warning(
                    "Created emergency liquidation buy order",
                    quantity=float(abs(self.current_inventory)),
                    current_inventory=float(self.current_inventory),
                )

            return orders

        except Exception as e:
            self.logger.error("Emergency order calculation failed", error=str(e))
            return []

    @time_execution
    async def calculate_spread_adjustment(self, base_spread: Decimal) -> Decimal:
        """
        Calculate spread adjustment based on inventory skew.

        Args:
            base_spread: Base spread to adjust

        Returns:
            Adjusted spread
        """
        try:
            # Calculate inventory-based adjustment
            inventory_adjustment = self.inventory_skew * self.inventory_risk_aversion

            # Apply adjustment to spread
            adjusted_spread = base_spread * Decimal(str(1 + inventory_adjustment))

            # Ensure minimum spread
            min_spread = Decimal("0.0001")  # 0.01%
            adjusted_spread = max(adjusted_spread, min_spread)

            self.logger.debug(
                "Spread adjustment calculated",
                base_spread=float(base_spread),
                inventory_skew=self.inventory_skew,
                adjustment=inventory_adjustment,
                adjusted_spread=float(adjusted_spread),
            )

            return adjusted_spread

        except Exception as e:
            self.logger.error("Spread adjustment calculation failed", error=str(e))
            return base_spread

    @time_execution
    async def calculate_size_adjustment(self, base_size: Decimal) -> Decimal:
        """
        Calculate order size adjustment based on inventory.

        Args:
            base_size: Base order size

        Returns:
            Adjusted order size
        """
        try:
            # Increase size when inventory is skewed (to rebalance)
            inventory_factor = 1 + abs(self.inventory_skew) * 0.5

            # Apply adjustment
            adjusted_size = base_size * Decimal(str(inventory_factor))

            # Ensure minimum size
            min_size = base_size * Decimal("0.1")  # 10% of base size
            adjusted_size = max(adjusted_size, min_size)

            self.logger.debug(
                "Size adjustment calculated",
                base_size=float(base_size),
                inventory_skew=self.inventory_skew,
                inventory_factor=inventory_factor,
                adjusted_size=float(adjusted_size),
            )

            return adjusted_size

        except Exception as e:
            self.logger.error("Size adjustment calculation failed", error=str(e))
            return base_size

    @time_execution
    async def record_rebalance(self, cost: Decimal) -> None:
        """
        Record rebalancing cost and update statistics.

        Args:
            cost: Cost of rebalancing (can be negative for profit)
        """
        try:
            self.total_rebalance_cost += cost
            self.rebalance_count += 1
            self.last_rebalance = datetime.now(timezone.utc)

            self.logger.info(
                "Rebalancing recorded",
                cost=float(cost),
                total_cost=float(self.total_rebalance_cost),
                rebalance_count=self.rebalance_count,
            )

        except Exception as e:
            self.logger.error("Rebalancing record failed", error=str(e))

    @time_execution
    async def record_emergency(self, cost: Decimal) -> None:
        """
        Record emergency liquidation cost and update statistics.

        Args:
            cost: Cost of emergency liquidation (can be negative for profit)
        """
        try:
            self.total_emergency_cost += cost
            self.emergency_count += 1

            self.logger.warning(
                "Emergency liquidation recorded",
                cost=float(cost),
                total_cost=float(self.total_emergency_cost),
                emergency_count=self.emergency_count,
            )

        except Exception as e:
            self.logger.error("Emergency liquidation record failed", error=str(e))

    def get_inventory_summary(self) -> dict[str, Any]:
        """
        Get comprehensive inventory summary.

        Returns:
            Dictionary with inventory information
        """
        try:
            return {
                "current_inventory": float(self.current_inventory),
                "target_inventory": float(self.target_inventory),
                "max_inventory": float(self.max_inventory),
                "min_inventory": float(self.min_inventory),
                "inventory_skew": self.inventory_skew,
                "rebalance_threshold": self.rebalance_threshold,
                "last_rebalance": self.last_rebalance.isoformat(),
                "rebalance_count": self.rebalance_count,
                "emergency_count": self.emergency_count,
                "total_rebalance_cost": float(self.total_rebalance_cost),
                "total_emergency_cost": float(self.total_emergency_cost),
                "inventory_risk_aversion": self.inventory_risk_aversion,
                "emergency_liquidation_enabled": self.emergency_liquidation_enabled,
            }

        except Exception as e:
            self.logger.error("Inventory summary generation failed", error=str(e))
            return {}

    @time_execution
    async def validate_inventory_limits(self, new_position: Position) -> bool:
        """
        Validate that new position doesn't violate inventory limits.

        Args:
            new_position: Proposed new position

        Returns:
            True if within limits, False otherwise
        """
        try:
            # Check against max inventory
            if abs(new_position.quantity) > self.max_inventory:
                self.logger.warning(
                    "Position violates max inventory limit",
                    position_quantity=float(new_position.quantity),
                    max_inventory=float(self.max_inventory),
                )
                return False

            # Check against min inventory
            if new_position.quantity < self.min_inventory:
                self.logger.warning(
                    "Position violates min inventory limit",
                    position_quantity=float(new_position.quantity),
                    min_inventory=float(self.min_inventory),
                )
                return False

            return True

        except Exception as e:
            self.logger.error("Inventory limit validation failed", error=str(e))
            return False
