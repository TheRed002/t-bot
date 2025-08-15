"""
Trade Execution Simulator for Backtesting.

This module provides realistic trade execution simulation including
order book modeling, market impact, and latency simulation.
"""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.exceptions import SimulationError
from src.core.logging import get_logger
from src.core.types import OrderSide, OrderType
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class SimulationConfig(BaseModel):
    """Configuration for trade simulation."""

    enable_partial_fills: bool = Field(
        default=True, description="Enable partial order fills"
    )
    enable_market_impact: bool = Field(
        default=True, description="Model market impact of large orders"
    )
    enable_latency: bool = Field(
        default=True, description="Simulate network and processing latency"
    )
    latency_ms: Tuple[int, int] = Field(
        default=(10, 100), description="Min and max latency in milliseconds"
    )
    market_impact_factor: Decimal = Field(
        default=Decimal("0.0001"), description="Market impact per unit volume"
    )
    liquidity_factor: Decimal = Field(
        default=Decimal("0.1"), description="Available liquidity as fraction of volume"
    )
    rejection_probability: float = Field(
        default=0.01, description="Probability of order rejection"
    )


class Order(BaseModel):
    """Order representation for simulation."""

    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    price: Decimal
    size: Decimal
    timestamp: datetime
    filled: Decimal = Decimal("0")
    status: str = "pending"
    fill_price: Optional[Decimal] = None
    fill_time: Optional[datetime] = None


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

    def __init__(self, config: SimulationConfig):
        """
        Initialize trade simulator.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self._order_book: Dict[str, Dict[str, List[Order]]] = {}
        self._executed_trades: List[Dict[str, Any]] = []
        self._pending_orders: Dict[str, Order] = {}
        
        logger.info("TradeSimulator initialized", config=config.model_dump())

    @time_execution
    async def execute_order(
        self,
        order: Order,
        market_data: pd.Series,
        order_book_depth: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Simulate order execution.

        Args:
            order: Order to execute
            market_data: Current market data (OHLCV)
            order_book_depth: Optional order book depth data

        Returns:
            Execution result with fill details
        """
        try:
            # Simulate latency
            if self.config.enable_latency:
                await self._simulate_latency()

            # Check for rejection
            if self._should_reject_order():
                return self._create_rejection_result(order, "Random rejection simulation")

            # Determine execution based on order type
            if order.type == OrderType.MARKET:
                result = await self._execute_market_order(order, market_data, order_book_depth)
            elif order.type == OrderType.LIMIT:
                result = await self._execute_limit_order(order, market_data, order_book_depth)
            else:
                result = await self._execute_stop_order(order, market_data)

            # Record execution
            if result["status"] == "filled" or result["status"] == "partial":
                self._executed_trades.append(result)

            return result

        except Exception as e:
            logger.error(f"Order execution failed", order_id=order.id, error=str(e))
            raise SimulationError(f"Order execution failed: {str(e)}")

    async def _simulate_latency(self) -> None:
        """Simulate network and processing latency."""
        min_latency, max_latency = self.config.latency_ms
        latency = random.uniform(min_latency, max_latency) / 1000.0
        await asyncio.sleep(latency)

    def _should_reject_order(self) -> bool:
        """Determine if order should be rejected."""
        return random.random() < self.config.rejection_probability

    def _create_rejection_result(self, order: Order, reason: str) -> Dict[str, Any]:
        """Create rejection result."""
        return {
            "order_id": order.id,
            "symbol": order.symbol,
            "status": "rejected",
            "reason": reason,
            "timestamp": datetime.now(),
            "filled_size": Decimal("0"),
            "average_price": None,
        }

    async def _execute_market_order(
        self,
        order: Order,
        market_data: pd.Series,
        order_book: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Execute market order with slippage and market impact."""
        # Base execution price
        if order.side == OrderSide.BUY:
            base_price = Decimal(str(market_data["high"]))  # Worse price for buyer
        else:
            base_price = Decimal(str(market_data["low"]))  # Worse price for seller

        # Apply market impact
        if self.config.enable_market_impact:
            impact = self._calculate_market_impact(order.size, market_data["volume"])
            if order.side == OrderSide.BUY:
                execution_price = base_price * (Decimal("1") + impact)
            else:
                execution_price = base_price * (Decimal("1") - impact)
        else:
            execution_price = base_price

        # Determine fill size (considering liquidity)
        if self.config.enable_partial_fills:
            available_liquidity = Decimal(str(market_data["volume"])) * self.config.liquidity_factor
            filled_size = min(order.size, available_liquidity)
            status = "filled" if filled_size == order.size else "partial"
        else:
            filled_size = order.size
            status = "filled"

        return {
            "order_id": order.id,
            "symbol": order.symbol,
            "status": status,
            "side": order.side.value,
            "type": order.type.value,
            "requested_size": float(order.size),
            "filled_size": float(filled_size),
            "average_price": float(execution_price),
            "timestamp": datetime.now(),
            "slippage": float(abs(execution_price - base_price) / base_price),
        }

    async def _execute_limit_order(
        self,
        order: Order,
        market_data: pd.Series,
        order_book: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Execute limit order checking price conditions."""
        current_price = Decimal(str(market_data["close"]))
        
        # Check if limit price is met
        can_execute = False
        if order.side == OrderSide.BUY:
            # Buy limit: execute if market price <= limit price
            can_execute = current_price <= order.price
            execution_price = min(order.price, current_price)
        else:
            # Sell limit: execute if market price >= limit price
            can_execute = current_price >= order.price
            execution_price = max(order.price, current_price)

        if not can_execute:
            # Add to pending orders
            self._pending_orders[order.id] = order
            return {
                "order_id": order.id,
                "symbol": order.symbol,
                "status": "pending",
                "side": order.side.value,
                "type": order.type.value,
                "message": "Limit price not met",
                "timestamp": datetime.now(),
            }

        # Execute the order
        if self.config.enable_partial_fills:
            available_liquidity = Decimal(str(market_data["volume"])) * self.config.liquidity_factor
            filled_size = min(order.size, available_liquidity)
            status = "filled" if filled_size == order.size else "partial"
        else:
            filled_size = order.size
            status = "filled"

        return {
            "order_id": order.id,
            "symbol": order.symbol,
            "status": status,
            "side": order.side.value,
            "type": order.type.value,
            "requested_size": float(order.size),
            "filled_size": float(filled_size),
            "average_price": float(execution_price),
            "limit_price": float(order.price),
            "timestamp": datetime.now(),
        }

    async def _execute_stop_order(
        self, order: Order, market_data: pd.Series
    ) -> Dict[str, Any]:
        """Execute stop-loss or take-profit order."""
        current_price = Decimal(str(market_data["close"]))
        
        # Check if stop is triggered
        triggered = False
        if order.type == OrderType.STOP_LOSS:
            if order.side == OrderSide.SELL:
                # Stop-loss sell: trigger if price <= stop price
                triggered = current_price <= order.price
            else:
                # Stop-loss buy (for short positions): trigger if price >= stop price
                triggered = current_price >= order.price
        elif order.type == OrderType.TAKE_PROFIT:
            if order.side == OrderSide.SELL:
                # Take-profit sell: trigger if price >= target price
                triggered = current_price >= order.price
            else:
                # Take-profit buy: trigger if price <= target price
                triggered = current_price <= order.price

        if not triggered:
            self._pending_orders[order.id] = order
            return {
                "order_id": order.id,
                "symbol": order.symbol,
                "status": "pending",
                "side": order.side.value,
                "type": order.type.value,
                "message": "Stop price not triggered",
                "timestamp": datetime.now(),
            }

        # Execute as market order once triggered
        order.type = OrderType.MARKET  # Convert to market order
        return await self._execute_market_order(order, market_data)

    def _calculate_market_impact(self, order_size: Decimal, volume: float) -> Decimal:
        """Calculate market impact based on order size."""
        if volume == 0:
            return self.config.market_impact_factor * Decimal("10")  # High impact
        
        # Impact increases with order size relative to volume
        relative_size = order_size / Decimal(str(volume))
        impact = self.config.market_impact_factor * relative_size
        
        # Cap maximum impact
        return min(impact, Decimal("0.05"))  # Max 5% impact

    async def check_pending_orders(
        self, market_data: Dict[str, pd.Series]
    ) -> List[Dict[str, Any]]:
        """
        Check and execute pending orders if conditions are met.

        Args:
            market_data: Current market data for all symbols

        Returns:
            List of execution results
        """
        results = []
        
        for order_id, order in list(self._pending_orders.items()):
            if order.symbol in market_data:
                # Re-check order conditions
                if order.type == OrderType.LIMIT:
                    result = await self._execute_limit_order(
                        order, market_data[order.symbol]
                    )
                else:
                    result = await self._execute_stop_order(
                        order, market_data[order.symbol]
                    )
                
                # Remove from pending if executed
                if result["status"] in ["filled", "partial", "rejected"]:
                    del self._pending_orders[order_id]
                    results.append(result)
        
        return results

    def calculate_execution_costs(
        self, trades: List[Dict[str, Any]]
    ) -> Dict[str, Decimal]:
        """
        Calculate total execution costs from trades.

        Args:
            trades: List of executed trades

        Returns:
            Dictionary with cost breakdown
        """
        total_slippage = Decimal("0")
        total_impact = Decimal("0")
        total_volume = Decimal("0")
        
        for trade in trades:
            if trade.get("status") in ["filled", "partial"]:
                filled_size = Decimal(str(trade.get("filled_size", 0)))
                slippage = Decimal(str(trade.get("slippage", 0)))
                
                total_volume += filled_size
                total_slippage += slippage * filled_size
                
                # Estimate market impact from slippage
                if self.config.enable_market_impact:
                    total_impact += slippage * filled_size * Decimal("0.5")  # Rough estimate
        
        avg_slippage = total_slippage / total_volume if total_volume > 0 else Decimal("0")
        
        return {
            "total_volume": total_volume,
            "average_slippage": avg_slippage,
            "total_slippage_cost": total_slippage,
            "estimated_market_impact": total_impact,
            "total_execution_cost": total_slippage + total_impact,
        }

    def get_execution_statistics(self) -> Dict[str, Any]:
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