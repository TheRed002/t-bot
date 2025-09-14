"""
Trade Execution Simulator for Backtesting.

This module provides realistic trade execution simulation including
order book modeling, market impact, and latency simulation.
"""

import asyncio
import random
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, Field

from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.core.types import (
    ExecutionAlgorithm,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.utils.backtesting_decorators import trading_operation
from src.utils.decimal_utils import to_decimal
from src.utils.financial_calculations import calculate_daily_returns, calculate_market_impact
from src.utils.initialization_helpers import log_configuration_initialization

if TYPE_CHECKING:
    from src.core.types import MarketData as CoreMarketData

logger = get_logger(__name__)


class SimulationConfig(BaseModel):
    """Configuration for trade simulation."""

    # Core simulation parameters
    initial_capital: Decimal = Field(
        default_factory=lambda: to_decimal("10000"), description="Initial capital"
    )
    commission_rate: Decimal = Field(
        default_factory=lambda: to_decimal("0.001"), description="Commission rate"
    )
    slippage_rate: Decimal = Field(
        default_factory=lambda: to_decimal("0.0005"), description="Slippage rate"
    )
    enable_shorting: bool = Field(default=False, description="Enable short selling")
    max_positions: int = Field(default=5, description="Maximum open positions")

    # Advanced simulation features
    enable_partial_fills: bool = Field(default=True, description="Enable partial order fills")
    enable_market_impact: bool = Field(
        default=True, description="Model market impact of large orders"
    )
    enable_latency: bool = Field(
        default=True, description="Simulate network and processing latency"
    )
    latency_ms: tuple[int, int] = Field(
        default=(10, 100), description="Min and max latency in milliseconds"
    )
    market_impact_factor: Decimal = Field(
        default_factory=lambda: to_decimal("0.0001"), description="Market impact per unit volume"
    )
    liquidity_factor: Decimal = Field(
        default_factory=lambda: to_decimal("0.1"),
        description="Available liquidity as fraction of volume",
    )
    rejection_probability: Decimal = Field(
        default_factory=lambda: to_decimal("0.01"), description="Probability of order rejection"
    )


class SimulatedOrder(BaseModel):
    """Extended order representation for simulation tracking."""

    request: OrderRequest
    order_id: str
    timestamp: datetime
    filled_quantity: Decimal = Field(default_factory=lambda: to_decimal("0"))
    status: OrderStatus = OrderStatus.PENDING
    average_fill_price: Decimal | None = None
    fill_time: datetime | None = None
    execution_fees: Decimal = Field(default_factory=lambda: to_decimal("0"))
    execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    algorithm_params: dict[str, Any] = Field(default_factory=dict)


class TradeSimulator:
    """
    Realistic trade execution simulator.

    Simulates:
    - Order book dynamics
    - Partial fills
    - Market impact
    - Execution latency
    - Order rejections
    """

    def __init__(self, config: SimulationConfig, slippage_model: Any = None):
        """
        Initialize trade simulator.

        Args:
            config: Simulation configuration
            slippage_model: Optional slippage model for realistic slippage calculation
        """
        self.config = config
        self.slippage_model = slippage_model
        self._order_book: dict[str, dict[str, list[SimulatedOrder]]] = {}
        self._executed_trades: list[dict[str, Any]] = []
        self._pending_orders: dict[str, SimulatedOrder] = {}

        log_configuration_initialization("TradeSimulator", config, logger_instance=logger)

    @trading_operation(operation="simulate_order_execution")
    async def execute_order(
        self,
        order_request: OrderRequest,
        market_data: pd.Series,
        order_book_depth: pd.DataFrame | None = None,
        execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET,
        algorithm_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Simulate order execution.

        Args:
            order_request: Order request to execute
            market_data: Current market data (OHLCV)
            order_book_depth: Optional order book depth data
            execution_algorithm: Algorithm to use for execution
            algorithm_params: Algorithm-specific parameters

        Returns:
            Execution result with fill details
        """
        # Create simulated order

        simulated_order = SimulatedOrder(
            request=order_request,
            order_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            execution_algorithm=execution_algorithm,
            algorithm_params=algorithm_params or {},
        )

        # Simulate latency
        if self.config.enable_latency:
            await self._simulate_latency()

        # Check for rejection
        if self._should_reject_order():
            return self._create_rejection_result(simulated_order, "Random rejection simulation")

        # Apply execution algorithm modifications
        if execution_algorithm != ExecutionAlgorithm.MARKET:
            result = await self._execute_with_algorithm(
                simulated_order, market_data, order_book_depth
            )
        else:
            # Determine execution based on order type
            if order_request.order_type == OrderType.MARKET:
                result = await self._execute_market_order(
                    simulated_order, market_data, order_book_depth
                )
            elif order_request.order_type == OrderType.LIMIT:
                result = await self._execute_limit_order(
                    simulated_order, market_data, order_book_depth
                )
            else:
                result = await self._execute_stop_order(simulated_order, market_data)

        # Record execution
        if result["status"] == "filled" or result["status"] == "partial":
            self._executed_trades.append(result)

        return result

    async def _simulate_latency(self) -> None:
        """Simulate network and processing latency."""
        min_latency, max_latency = self.config.latency_ms
        latency = random.uniform(min_latency, max_latency) / 1000.0
        await asyncio.sleep(latency)

    def _should_reject_order(self) -> bool:
        """Determine if order should be rejected."""
        return random.random() < self.config.rejection_probability

    def _create_rejection_result(self, order: SimulatedOrder, reason: str) -> dict[str, Any]:
        """Create rejection result."""
        return {
            "order_id": order.order_id,
            "symbol": order.request.symbol,
            "status": "rejected",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc),
            "filled_size": to_decimal("0"),
            "average_price": None,
        }

    async def _execute_market_order(
        self,
        order: SimulatedOrder,
        market_data: pd.Series,
        order_book: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Execute market order with slippage and market impact."""
        # Base execution price
        if order.request.side == OrderSide.BUY:
            base_price = to_decimal(market_data["high"])  # Worse price for buyer
        else:
            base_price = to_decimal(market_data["low"])  # Worse price for seller

        # Apply market impact
        if self.config.enable_market_impact:
            impact = self._calculate_market_impact(
                order.request.quantity, to_decimal(market_data["volume"])
            )
            if order.request.side == OrderSide.BUY:
                execution_price = base_price * (to_decimal("1") + impact)
            else:
                execution_price = base_price * (to_decimal("1") - impact)
        else:
            execution_price = base_price

        # Determine fill size (considering liquidity)
        if self.config.enable_partial_fills:
            available_liquidity = to_decimal(market_data["volume"]) * to_decimal(
                self.config.liquidity_factor
            )
            filled_size = min(order.request.quantity, available_liquidity)
            status = "filled" if filled_size == order.request.quantity else "partial"
        else:
            filled_size = order.request.quantity
            status = "filled"

        return {
            "order_id": order.order_id,
            "symbol": order.request.symbol,
            "status": status,
            "side": order.request.side.value,
            "type": order.request.order_type.value,
            "requested_size": order.request.quantity,
            "filled_size": filled_size,
            "average_price": execution_price,
            "timestamp": datetime.now(timezone.utc),
            "slippage": abs(execution_price - base_price) / base_price,
        }

    async def _execute_limit_order(
        self,
        order: SimulatedOrder,
        market_data: pd.Series,
        order_book: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Execute limit order checking price conditions."""
        current_price = to_decimal(market_data["close"])

        # Check if limit price is met
        can_execute = False
        if order.request.price is None:
            # Should not happen for limit orders - use proper core exception
            raise ValidationError(
                "Limit order missing price",
                field_name="price",
                field_value=None,
                error_code="ORDER_001",
            )

        if order.request.side == OrderSide.BUY:
            # Buy limit: execute if market price <= limit price
            can_execute = current_price <= order.request.price
            execution_price = min(order.request.price, current_price)
        else:
            # Sell limit: execute if market price >= limit price
            can_execute = current_price >= order.request.price
            execution_price = max(order.request.price, current_price)

        if not can_execute:
            # Add to pending orders
            self._pending_orders[order.order_id] = order
            return {
                "order_id": order.order_id,
                "symbol": order.request.symbol,
                "status": "pending",
                "side": order.request.side.value,
                "type": order.request.order_type.value,
                "message": "Limit price not met",
                "timestamp": datetime.now(timezone.utc),
            }

        # Execute the order
        if self.config.enable_partial_fills:
            available_liquidity = to_decimal(market_data["volume"]) * to_decimal(
                self.config.liquidity_factor
            )
            filled_size = min(order.request.quantity, available_liquidity)
            status = "filled" if filled_size == order.request.quantity else "partial"
        else:
            filled_size = order.request.quantity
            status = "filled"

        return {
            "order_id": order.order_id,
            "symbol": order.request.symbol,
            "status": status,
            "side": order.request.side.value,
            "type": order.request.order_type.value,
            "requested_size": order.request.quantity,
            "filled_size": filled_size,
            "average_price": execution_price,
            "limit_price": order.request.price,
            "timestamp": datetime.now(timezone.utc),
        }

    async def _execute_stop_order(
        self, order: SimulatedOrder, market_data: pd.Series
    ) -> dict[str, Any]:
        """Execute stop-loss or take-profit order."""
        current_price = to_decimal(market_data["close"])

        # Check if stop is triggered
        triggered = False
        stop_price = order.request.stop_price or order.request.price
        if stop_price is None:
            raise ValidationError(
                "Stop order missing price",
                field_name="stop_price",
                field_value=None,
                error_code="ORDER_002",
            )

        if order.request.order_type == OrderType.STOP_LOSS:
            if order.request.side == OrderSide.SELL:
                # Stop-loss sell: trigger if price <= stop price
                triggered = current_price <= stop_price
            else:
                # Stop-loss buy (for short positions): trigger if price >= stop price
                triggered = current_price >= stop_price
        elif order.request.order_type == OrderType.TAKE_PROFIT:
            if order.request.side == OrderSide.SELL:
                # Take-profit sell: trigger if price >= target price
                triggered = current_price >= stop_price
            else:
                # Take-profit buy: trigger if price <= target price
                triggered = current_price <= stop_price

        if not triggered:
            self._pending_orders[order.order_id] = order
            return {
                "order_id": order.order_id,
                "symbol": order.request.symbol,
                "status": "pending",
                "side": order.request.side.value,
                "type": order.request.order_type.value,
                "message": "Stop price not triggered",
                "timestamp": datetime.now(timezone.utc),
            }

        # Execute as market order once triggered
        # Create a new market order request
        market_order = SimulatedOrder(
            request=OrderRequest(
                symbol=order.request.symbol,
                side=order.request.side,
                order_type=OrderType.MARKET,
                quantity=order.request.quantity,
                time_in_force=order.request.time_in_force,
                client_order_id=order.request.client_order_id,
                metadata=order.request.metadata,
            ),
            order_id=order.order_id,
            timestamp=order.timestamp,
        )
        return await self._execute_market_order(market_order, market_data)

    async def _execute_with_algorithm(
        self,
        order: SimulatedOrder,
        market_data: pd.Series,
        order_book: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Execute order using specific execution algorithm."""
        algorithm = order.execution_algorithm
        params = order.algorithm_params

        if algorithm == ExecutionAlgorithm.TWAP:
            # Time-Weighted Average Price execution
            return await self._execute_twap(order, market_data, params)
        elif algorithm == ExecutionAlgorithm.VWAP:
            # Volume-Weighted Average Price execution
            return await self._execute_vwap(order, market_data, params)
        elif algorithm == ExecutionAlgorithm.ICEBERG:
            # Iceberg order execution
            return await self._execute_iceberg(order, market_data, params)
        else:
            # Default to market execution
            return await self._execute_market_order(order, market_data, order_book)

    async def _execute_twap(
        self, order: SimulatedOrder, market_data: pd.Series, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate TWAP execution with multiple child orders."""
        # Extract parameters
        num_slices = params.get("max_slices", 5)

        # Calculate slice size
        slice_size = order.request.quantity / to_decimal(num_slices)

        # Simulate execution across time slices with varying prices
        total_filled = to_decimal("0")
        total_cost = to_decimal("0")

        base_price = to_decimal(market_data["close"])

        for _ in range(num_slices):
            # Simulate price movement during execution
            price_variation = to_decimal(random.uniform(-0.001, 0.001))  # 0.1% variation
            slice_price = base_price * (to_decimal("1") + price_variation)

            # Apply market impact for this slice
            impact = self._calculate_market_impact(slice_size, to_decimal(market_data["volume"]))
            if order.request.side == OrderSide.BUY:
                execution_price = slice_price * (to_decimal("1") + impact)
            else:
                execution_price = slice_price * (to_decimal("1") - impact)

            total_filled += slice_size
            total_cost += slice_size * execution_price

        average_price = total_cost / total_filled if total_filled > 0 else base_price

        return {
            "order_id": order.order_id,
            "symbol": order.request.symbol,
            "status": "filled",
            "side": order.request.side.value,
            "type": order.request.order_type.value,
            "algorithm": order.execution_algorithm.value,
            "requested_size": order.request.quantity,
            "filled_size": total_filled,
            "average_price": average_price,
            "timestamp": datetime.now(timezone.utc),
            "num_slices": num_slices,
            "slippage": abs(average_price - base_price) / base_price,
        }

    async def _execute_vwap(
        self, order: SimulatedOrder, market_data: pd.Series, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate VWAP execution based on volume profile."""
        # For simplicity, simulate with volume-based execution
        base_price = to_decimal(market_data["close"])
        volume = to_decimal(market_data["volume"])

        # Calculate execution price based on VWAP
        # In real VWAP, we'd use volume profile, but here we simulate
        vwap_price = (
            to_decimal(market_data["high"]) + to_decimal(market_data["low"]) + base_price
        ) / to_decimal("3")

        # Apply market impact
        impact = self._calculate_market_impact(order.request.quantity, volume)
        if order.request.side == OrderSide.BUY:
            execution_price = vwap_price * (to_decimal("1") + impact)
        else:
            execution_price = vwap_price * (to_decimal("1") - impact)

        return {
            "order_id": order.order_id,
            "symbol": order.request.symbol,
            "status": "filled",
            "side": order.request.side.value,
            "type": order.request.order_type.value,
            "algorithm": order.execution_algorithm.value,
            "requested_size": order.request.quantity,
            "filled_size": order.request.quantity,
            "average_price": execution_price,
            "vwap_price": vwap_price,
            "timestamp": datetime.now(timezone.utc),
            "slippage": abs(execution_price - base_price) / base_price,
        }

    async def _execute_iceberg(
        self, order: SimulatedOrder, market_data: pd.Series, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate Iceberg order with hidden quantity."""
        # Extract parameters
        visible_percentage = params.get("visible_percentage", 0.1)  # 10% visible by default

        # Calculate visible and hidden quantities
        visible_qty = order.request.quantity * to_decimal(visible_percentage)
        hidden_qty = order.request.quantity - visible_qty

        # Execute visible portion first
        visible_result = await self._execute_market_order(
            SimulatedOrder(
                request=OrderRequest(
                    symbol=order.request.symbol,
                    side=order.request.side,
                    order_type=OrderType.MARKET,
                    quantity=visible_qty,
                    time_in_force=order.request.time_in_force,
                ),
                order_id=order.order_id + "_visible",
                timestamp=order.timestamp,
            ),
            market_data,
        )

        # Execute hidden portion with less market impact
        hidden_result = await self._execute_market_order(
            SimulatedOrder(
                request=OrderRequest(
                    symbol=order.request.symbol,
                    side=order.request.side,
                    order_type=OrderType.MARKET,
                    quantity=hidden_qty,
                    time_in_force=order.request.time_in_force,
                ),
                order_id=order.order_id + "_hidden",
                timestamp=order.timestamp,
            ),
            market_data,
        )

        # Combine results
        total_filled = to_decimal(visible_result["filled_size"]) + to_decimal(
            hidden_result["filled_size"]
        )
        total_cost = to_decimal(visible_result["filled_size"]) * to_decimal(
            visible_result["average_price"]
        ) + to_decimal(hidden_result["filled_size"]) * to_decimal(hidden_result["average_price"])
        average_price = (
            total_cost / total_filled if total_filled > 0 else to_decimal(market_data["close"])
        )

        return {
            "order_id": order.order_id,
            "symbol": order.request.symbol,
            "status": "filled",
            "side": order.request.side.value,
            "type": order.request.order_type.value,
            "algorithm": order.execution_algorithm.value,
            "requested_size": order.request.quantity,
            "filled_size": total_filled,
            "average_price": average_price,
            "visible_size": visible_qty,
            "hidden_size": hidden_qty,
            "timestamp": datetime.now(timezone.utc),
            "slippage": abs(average_price - to_decimal(market_data["close"]))
            / to_decimal(market_data["close"]),
        }

    def _calculate_market_impact(self, order_size: Decimal, volume: Decimal) -> Decimal:
        """Calculate market impact based on order size."""
        if self.slippage_model:
            # Use sophisticated slippage model from execution module

            # Create a mock market data object for slippage calculation
            mock_market_data = CoreMarketData(
                symbol="",  # Not needed for impact calculation
                timestamp=datetime.now(timezone.utc),
                close=to_decimal("1"),  # Normalized
                volume=to_decimal(volume),
                high=to_decimal("1.01"),
                low=to_decimal("0.99"),
                open=to_decimal("1"),
                exchange="mock",
            )

            # Calculate slippage using the model
            if hasattr(self.slippage_model, "calculate_slippage"):
                slippage_result = self.slippage_model.calculate_slippage(
                    order_size=order_size,
                    market_data=mock_market_data,
                    order_side=OrderSide.BUY,  # Doesn't matter for impact
                )
                return slippage_result.market_impact_pct / to_decimal("100")
            else:
                # Model doesn't have expected interface, fall back
                logger.warning("Slippage model doesn't have calculate_slippage method")
                return self.config.market_impact_factor
        else:
            # Use shared market impact calculation
            base_impact = calculate_market_impact(order_size, volume)
            return base_impact * self.config.market_impact_factor

    async def check_pending_orders(self, market_data: dict[str, pd.Series]) -> list[dict[str, Any]]:
        """
        Check and execute pending orders if conditions are met.

        Args:
            market_data: Current market data for all symbols

        Returns:
            List of execution results
        """
        results = []

        for order_id, order in list(self._pending_orders.items()):
            if order.request.symbol in market_data:
                # Re-check order conditions
                if order.request.order_type == OrderType.LIMIT:
                    result = await self._execute_limit_order(
                        order, market_data[order.request.symbol]
                    )
                else:
                    result = await self._execute_stop_order(
                        order, market_data[order.request.symbol]
                    )

                # Remove from pending if executed
                if result["status"] in ["filled", "partial", "rejected"]:
                    del self._pending_orders[order_id]
                    results.append(result)

        return results

    def calculate_execution_costs(self, trades: list[dict[str, Any]]) -> dict[str, Decimal]:
        """
        Calculate total execution costs from trades.

        Args:
            trades: List of executed trades

        Returns:
            Dictionary with cost breakdown
        """
        total_slippage = to_decimal("0")
        total_impact = to_decimal("0")
        total_volume = to_decimal("0")

        for trade in trades:
            if trade.get("status") in ["filled", "partial"]:
                filled_size = to_decimal(trade.get("filled_size", 0))
                slippage = to_decimal(trade.get("slippage", 0))

                total_volume += filled_size
                total_slippage += slippage * filled_size

                # Estimate market impact from slippage
                if self.config.enable_market_impact:
                    total_impact += slippage * filled_size * to_decimal("0.5")  # Rough estimate

        avg_slippage = total_slippage / total_volume if total_volume > 0 else to_decimal("0")

        return {
            "total_volume": total_volume,
            "average_slippage": avg_slippage,
            "total_slippage_cost": total_slippage,
            "estimated_market_impact": total_impact,
            "total_execution_cost": total_slippage + total_impact,
        }

    def get_execution_statistics(self) -> dict[str, Any]:
        """Get execution statistics from simulation."""
        if not self._executed_trades:
            return {
                "total_orders": 0,
                "filled_orders": 0,
                "partial_fills": 0,
                "rejected_orders": 0,
                "fill_rate": 0.0,
            }

        filled = len([t for t in self._executed_trades if t["status"] == "filled"])
        partial = len([t for t in self._executed_trades if t["status"] == "partial"])
        rejected = len([t for t in self._executed_trades if t["status"] == "rejected"])
        total = len(self._executed_trades)

        return {
            "total_orders": total,
            "filled_orders": filled,
            "partial_fills": partial,
            "rejected_orders": rejected,
            "pending_orders": len(self._pending_orders),
            "fill_rate": (filled + partial) / total if total > 0 else 0.0,
            "rejection_rate": rejected / total if total > 0 else 0.0,
        }

    def cleanup(self) -> None:
        """Cleanup simulator resources."""
        try:
            # Clear all internal state
            self._order_book.clear()
            self._executed_trades.clear()
            self._pending_orders.clear()

            logger.info("TradeSimulator cleanup completed")
        except Exception as e:
            logger.error(f"TradeSimulator cleanup error: {e}")
            # Don't re-raise cleanup errors to avoid masking original issues

    async def get_simulation_results(self) -> dict[str, Any]:
        """Get current simulation results."""
        return {
            "executed_trades": self._executed_trades.copy(),
            "execution_stats": self.get_execution_statistics(),
        }

    def _calculate_daily_returns(self, equity_curve: list[dict[str, Any]]) -> list[float]:
        """Calculate daily returns from equity curve."""
        return calculate_daily_returns(equity_curve)
