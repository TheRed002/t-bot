"""
Real-time Trading Analytics Engine.

This module provides comprehensive real-time analytics for trading operations,
including live P&L tracking, risk monitoring, performance attribution, and
institutional-grade analytics with WebSocket integration.

Key Features:
- Real-time P&L calculation and attribution
- Live risk metrics (VaR, Expected Shortfall, concentration risk)
- Trade execution quality analysis with slippage tracking
- Position and portfolio-level analytics
- WebSocket integration for live dashboard updates
- Redis caching for high-performance data access
- Stress testing and scenario analysis capabilities
"""

import asyncio
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import Any

import numpy as np
import pandas as pd
import websockets
from scipy import stats

from src.analytics.types import (
    AnalyticsAlert,
    AnalyticsConfiguration,
    PerformanceAttribution,
    PortfolioMetrics,
    PositionMetrics,
    StrategyMetrics,
    TimeSeries,
    TradeAnalytics,
)
from src.core.base.component import BaseComponent
from src.core.exceptions import DataError, RiskCalculationError
from src.core.types import AlertSeverity
from src.core.types.trading import Order, Position, Trade
from src.utils.datetime_utils import get_current_utc_timestamp
from src.utils.decimal_utils import to_decimal


class RealtimeAnalyticsEngine(BaseComponent):
    """
    Real-time analytics engine for trading operations.

    Provides institutional-grade real-time analytics including:
    - Live P&L tracking with attribution
    - Position monitoring and exposure analysis
    - Strategy performance metrics calculation
    - Risk metrics (VaR, drawdown, volatility)
    - Trade execution quality analysis
    """

    def __init__(self, config: AnalyticsConfiguration, metrics_collector=None):
        """
        Initialize real-time analytics engine.

        Args:
            config: Analytics configuration
            metrics_collector: Optional metrics collector (injected)
        """
        super().__init__()
        self.config = config
        # Use dependency injection - do not create dependencies directly
        self.metrics_collector = metrics_collector

        if self.metrics_collector is None:
            from src.core.exceptions import ComponentError

            raise ComponentError(
                "metrics_collector must be injected via dependency injection",
                component="RealtimeAnalyticsEngine",
                operation="__init__",
                context={"missing_dependency": "metrics_collector"},
            )

        # Set decimal precision context for financial calculations
        getcontext().prec = 28
        getcontext().rounding = ROUND_HALF_UP

        # Data storage
        self._positions: dict[str, Position] = {}
        self._trades: deque = deque(maxlen=10000)  # Keep last 10k trades
        self._orders: dict[str, Order] = {}
        self._price_cache: dict[str, Decimal] = {}
        self._benchmark_prices: dict[str, deque] = defaultdict(lambda: deque(maxlen=252))

        # Time series storage
        self._portfolio_timeseries: dict[str, TimeSeries] = {}
        self._position_timeseries: dict[str, dict[str, TimeSeries]] = defaultdict(dict)
        self._strategy_timeseries: dict[str, dict[str, TimeSeries]] = defaultdict(dict)

        # Analytics state
        self._portfolio_value_history: deque = deque(maxlen=252)  # 1 year daily
        self._intraday_values: deque = deque(maxlen=1440)  # 24 hours of minute data
        self._daily_returns: deque = deque(maxlen=252)
        self._intraday_returns: deque = deque(maxlen=1440)
        self._pnl_cache: dict[str, dict[str, Decimal]] = defaultdict(dict)
        self._last_portfolio_value: Decimal | None = None
        self._last_update: datetime | None = None

        # Advanced performance tracking
        self._strategy_performance: dict[str, dict[str, Any]] = defaultdict(dict)
        self._position_analytics: dict[str, dict[str, Any]] = defaultdict(dict)
        self._execution_analytics: dict[str, dict[str, Any]] = defaultdict(dict)
        self._slippage_tracker: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._market_impact_tracker: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # P&L Attribution components
        self._sector_pnl: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self._currency_pnl: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self._strategy_pnl: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self._alpha_beta_decomposition: dict[str, dict[str, Decimal]] = defaultdict(dict)

        # Risk monitoring
        self._var_history: deque = deque(maxlen=252)
        self._volatility_surface: dict[str, dict[str, Any]] = defaultdict(dict)
        self._correlation_cache: dict[str, dict[str, Decimal]] = defaultdict(dict)

        # Transaction cost analysis
        self._tca_data: deque = deque(maxlen=1000)
        self._liquidity_metrics: dict[str, dict[str, Any]] = defaultdict(dict)

        # Alert management
        self._active_alerts: dict[str, AnalyticsAlert] = {}
        self._alert_history: deque = deque(maxlen=1000)
        self._alert_thresholds: dict[str, dict[str, Any]] = {
            "max_portfolio_drawdown": {"threshold": Decimal("0.05"), "enabled": True},
            "position_concentration": {"threshold": Decimal("0.20"), "enabled": True},
            "daily_var_breach": {"threshold": Decimal("0.02"), "enabled": True},
            "execution_cost_spike": {"threshold": Decimal("0.005"), "enabled": True},
        }

        # WebSocket integration for live dashboard updates
        self._websocket_clients: set = set()
        self._websocket_server = None
        self._broadcast_queue: asyncio.Queue = asyncio.Queue()

        # Cache client injection (Redis or other caching mechanism)
        self._cache_client: Any = None
        self._cache_enabled: bool = False

        # Stress testing and scenario analysis
        self._stress_scenarios: dict[str, dict[str, Any]] = {
            "market_crash_2008": {"equity_shock": -0.25, "vol_spike": 2.0},
            "flash_crash": {"equity_shock": -0.10, "vol_spike": 3.0},
            "covid_crash": {"equity_shock": -0.30, "vol_spike": 2.5},
            "interest_rate_shock": {"rate_change": 0.02, "duration_shock": -0.15},
        }

        # Calculation intervals
        self._calculation_tasks: set[asyncio.Task] = set()
        self._running = False

        self.logger.info("RealtimeAnalyticsEngine initialized with enhanced features")

    def set_cache_client(self, cache_client: Any) -> None:
        """
        Inject cache client dependency.

        Args:
            cache_client: Cache client implementation (Redis, etc.)
        """
        self._cache_client = cache_client
        self.logger.debug("Cache client injected")

    def set_websocket_manager(self, websocket_manager: Any) -> None:
        """
        Inject WebSocket manager dependency.

        Args:
            websocket_manager: WebSocket manager implementation
        """
        self._websocket_manager = websocket_manager
        self.logger.debug("WebSocket manager injected")

    async def start(self) -> None:
        """Start the real-time analytics engine."""
        if self._running:
            self.logger.warning("Analytics engine already running")
            return

        self._running = True

        # Initialize Redis caching if configured
        await self._initialize_cache()

        # Start WebSocket server for live updates
        await self._start_websocket_server()

        # Start calculation tasks
        tasks = [
            self._portfolio_analytics_loop(),
            self._position_analytics_loop(),
            self._strategy_analytics_loop(),
            self._risk_monitoring_loop(),
            self._alert_processing_loop(),
            self._websocket_broadcast_loop(),
            self._cache_sync_loop(),
        ]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._calculation_tasks.add(task)
            task.add_done_callback(self._calculation_tasks.discard)

        self.logger.info("Real-time analytics engine started")

    async def stop(self) -> None:
        """Stop the real-time analytics engine."""
        self._running = False

        # Close WebSocket server
        if self._websocket_server:
            self._websocket_server.close()
            await self._websocket_server.wait_closed()
            self.logger.info("WebSocket server closed")

        # Close all WebSocket client connections with proper timeout handling
        if self._websocket_clients:
            disconnect_tasks = []
            for client in list(self._websocket_clients):
                try:
                    # Use timeout to prevent hanging
                    await asyncio.wait_for(client.close(), timeout=5.0)
                    disconnect_tasks.append(client)
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout closing WebSocket client")
                except Exception as e:
                    self.logger.warning(f"Error closing WebSocket client: {e}")
            self._websocket_clients.clear()
            self.logger.info(f"Closed {len(disconnect_tasks)} WebSocket connections")

        # Cancel all tasks
        for task in self._calculation_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._calculation_tasks:
            await asyncio.gather(*self._calculation_tasks, return_exceptions=True)

        self.logger.info("Real-time analytics engine stopped")

    def update_position(self, position: Position) -> None:
        """
        Update position data and trigger analytics recalculation.

        Args:
            position: Updated position data
        """
        position_key = f"{position.exchange}:{position.symbol}"
        self._positions[position_key] = position

        # Use asyncio.gather for coordinated task execution to prevent race conditions
        if self._running:
            task = asyncio.create_task(self._update_position_coordinated(position))
            self._calculation_tasks.add(task)
            task.add_done_callback(self._calculation_tasks.discard)

    async def _update_position_coordinated(self, position: Position) -> None:
        """Coordinated position update to prevent race conditions."""
        try:
            # Run position and portfolio analytics concurrently with proper error handling
            await asyncio.gather(
                self._calculate_position_analytics(position),
                self._calculate_portfolio_analytics(),
                return_exceptions=True,
            )
        except Exception as e:
            self.logger.error(f"Error in coordinated position update: {e}")

    def update_trade(self, trade: Trade) -> None:
        """
        Update trade data and calculate trade analytics.

        Args:
            trade: New trade data
        """
        self._trades.append(trade)

        # Use coordinated task execution to prevent race conditions
        if self._running:
            task = asyncio.create_task(self._update_trade_coordinated(trade))
            self._calculation_tasks.add(task)
            task.add_done_callback(self._calculation_tasks.discard)

    async def _update_trade_coordinated(self, trade: Trade) -> None:
        """Coordinated trade update to prevent race conditions."""
        try:
            # Prepare tasks list
            tasks = [self._calculate_trade_analytics(trade)]

            # Add strategy performance update if applicable
            if hasattr(trade, "strategy") and trade.metadata.get("strategy"):
                strategy = trade.metadata["strategy"]
                tasks.append(self._update_strategy_performance(strategy, trade))

            # Run all tasks concurrently with proper error handling
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Error in coordinated trade update: {e}")

    def update_order(self, order: Order) -> None:
        """
        Update order data and execution analytics.

        Args:
            order: Updated order data
        """
        self._orders[order.order_id] = order

        # Calculate execution quality metrics with proper task management
        if self._running:
            task = asyncio.create_task(self._calculate_execution_analytics(order))
            self._calculation_tasks.add(task)
            task.add_done_callback(self._calculation_tasks.discard)

    def update_price(self, symbol: str, price: Decimal) -> None:
        """
        Update price data for real-time P&L calculation.

        Args:
            symbol: Trading symbol
            price: Current price
        """
        old_price = self._price_cache.get(symbol)
        self._price_cache[symbol] = price

        # Calculate price return if we have previous price
        if old_price and old_price > 0:
            price_return = (price - old_price) / old_price
            self._update_price_return_analytics(symbol, price_return)

        # Trigger coordinated real-time updates to prevent race conditions
        if self._running:
            task = asyncio.create_task(self._update_price_coordinated(symbol))
            self._calculation_tasks.add(task)
            task.add_done_callback(self._calculation_tasks.discard)

    async def _update_price_coordinated(self, symbol: str) -> None:
        """Coordinated price update to prevent race conditions."""
        try:
            # Run P&L and risk updates concurrently with proper error handling
            await asyncio.gather(
                self._update_realtime_pnl(symbol),
                self._update_real_time_risk_metrics(),
                return_exceptions=True,
            )
        except Exception as e:
            self.logger.error(f"Error in coordinated price update: {e}")

    async def get_portfolio_metrics(self) -> PortfolioMetrics | None:
        """
        Get current portfolio metrics.

        Returns:
            Current portfolio metrics or None if not available
        """
        return await self._calculate_portfolio_metrics()

    async def get_position_metrics(self, symbol: str | None = None) -> list[PositionMetrics]:
        """
        Get position metrics.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of position metrics
        """
        metrics = []
        positions = self._positions

        if symbol:
            positions = {k: v for k, v in positions.items() if symbol in k}

        for position in positions.values():
            pos_metrics = await self._calculate_position_metrics(position)
            if pos_metrics:
                metrics.append(pos_metrics)

        return metrics

    async def get_strategy_metrics(self, strategy: str | None = None) -> list[StrategyMetrics]:
        """
        Get strategy performance metrics.

        Args:
            strategy: Optional strategy filter

        Returns:
            List of strategy metrics
        """
        if strategy:
            strategies = [strategy] if strategy in self._strategy_performance else []
        else:
            strategies = list(self._strategy_performance.keys())

        metrics = []
        for strat_name in strategies:
            strat_metrics = await self._calculate_strategy_metrics(strat_name)
            if strat_metrics:
                metrics.append(strat_metrics)

        return metrics

    async def get_active_alerts(self) -> list[AnalyticsAlert]:
        """
        Get active analytics alerts.

        Returns:
            List of active alerts
        """
        return list(self._active_alerts.values())

    async def get_trade_analytics(
        self, trade_id: str | None = None, symbol: str | None = None
    ) -> list[TradeAnalytics]:
        """
        Get detailed trade analytics with execution quality metrics.

        Args:
            trade_id: Optional specific trade ID
            symbol: Optional symbol filter

        Returns:
            List of trade analytics
        """
        analytics = []

        for trade in self._trades:
            if trade_id and trade.trade_id != trade_id:
                continue
            if symbol and trade.symbol != symbol:
                continue

            trade_analysis = await self._calculate_detailed_trade_analytics(trade)
            if trade_analysis:
                analytics.append(trade_analysis)

        return analytics

    async def get_performance_attribution(
        self, period_days: int = 1
    ) -> PerformanceAttribution | None:
        """
        Get performance attribution analysis.

        Args:
            period_days: Attribution period in days

        Returns:
            Performance attribution or None
        """
        return await self._calculate_performance_attribution(period_days)

    async def get_execution_quality_metrics(self, symbol: str | None = None) -> dict[str, Any]:
        """
        Get execution quality metrics including slippage and market impact.

        Args:
            symbol: Optional symbol filter

        Returns:
            Execution quality metrics
        """
        return await self._calculate_execution_quality_metrics(symbol)

    async def get_position_analytics(self) -> dict[str, Any]:
        """Get detailed position analytics with consistent error handling."""
        try:
            position_analytics = {}
            for symbol, position in self._positions.items():
                # Use consistent attribute access pattern
                if position.is_open():
                    # Get current price from price cache
                    current_price = self._price_cache.get(position.symbol, position.entry_price)
                    position_value = position.quantity * current_price

                    # Calculate unrealized PnL with consistent pattern
                    unrealized_pnl = position.calculate_pnl(current_price)

                    position_analytics[symbol] = {
                        "value": position_value,
                        "quantity": position.quantity,
                        "entry_price": position.entry_price,
                        "current_price": current_price,
                        "unrealized_pnl": unrealized_pnl,
                        "side": position.side.value,
                        "exchange": position.exchange,
                    }
            return position_analytics
        except Exception as e:
            self.logger.error(f"Error getting position analytics: {e}")
            # Use consistent error propagation pattern from data pipeline
            raise DataError(
                "Failed to calculate position analytics",
                error_code="ANALYTICS_001",
                data_type="position_analytics",
                context={"error": str(e), "positions_count": len(self._positions)},
            )

    async def get_portfolio_risk_metrics(self) -> dict[str, Any]:
        """Get portfolio risk metrics."""
        try:
            return {
                "var_95": self._var_history[-1] if self._var_history else Decimal("0"),
                "volatility": self._calculate_portfolio_volatility(),
                "max_drawdown": self._calculate_max_drawdown(),
            }
        except Exception as e:
            self.logger.error(f"Error getting portfolio risk metrics: {e}")
            raise RiskCalculationError(
                "Failed to calculate portfolio risk metrics",
                error_code="ANALYTICS_002",
                calculation_type="portfolio_risk",
                context={"error": str(e)},
            )

    def _calculate_portfolio_volatility(self) -> Decimal:
        """Calculate portfolio volatility."""
        if len(self._daily_returns) < 20:
            return Decimal("0")
        returns = [float(r) for r in self._daily_returns]  # Convert to float for numpy calculations
        volatility = np.std(returns) * np.sqrt(252)
        return to_decimal(volatility)

    def _calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown."""
        if len(self._portfolio_value_history) < 2:
            return Decimal("0")
        values = [
            float(v) for v in self._portfolio_value_history
        ]  # Convert to float for numpy calculations
        peak = values[0]
        max_dd = 0.0
        for value in values[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        return to_decimal(max_dd)

    async def _get_current_portfolio_value(self) -> Decimal:
        """Get current portfolio value."""
        try:
            total_value = Decimal("0")
            for position in self._positions.values():
                if hasattr(position, "size") and hasattr(position, "current_price"):
                    total_value += getattr(position, "size", Decimal("0")) * getattr(
                        position, "current_price", Decimal("1")
                    )
            return total_value
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            raise RiskCalculationError(
                "Failed to calculate portfolio value",
                error_code="ANALYTICS_003",
                calculation_type="portfolio_value",
                context={"error": str(e)},
            )

    async def _portfolio_analytics_loop(self) -> None:
        """Background loop for portfolio analytics calculation."""
        while self._running:
            try:
                await self._calculate_portfolio_analytics()
                await asyncio.sleep(60)  # Update every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in portfolio analytics loop: {e}")
                # Don't re-raise here as this is a background task
                await asyncio.sleep(5)

    async def _position_analytics_loop(self) -> None:
        """Background loop for position analytics calculation."""
        while self._running:
            try:
                for position in self._positions.values():
                    await self._calculate_position_analytics(position)
                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in position analytics loop: {e}")
                await asyncio.sleep(5)

    async def _strategy_analytics_loop(self) -> None:
        """Background loop for strategy analytics calculation."""
        while self._running:
            try:
                for strategy in self._strategy_performance.keys():
                    await self._calculate_strategy_metrics(strategy)
                await asyncio.sleep(60)  # Update every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in strategy analytics loop: {e}")
                await asyncio.sleep(5)

    async def _risk_monitoring_loop(self) -> None:
        """Background loop for risk monitoring and alerts."""
        while self._running:
            try:
                await self._check_risk_thresholds()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _alert_processing_loop(self) -> None:
        """Background loop for alert processing and cleanup."""
        while self._running:
            try:
                await self._process_alerts()
                await asyncio.sleep(30)  # Process every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(5)

    async def _calculate_portfolio_analytics(self) -> None:
        """Calculate comprehensive portfolio analytics."""
        try:
            now = get_current_utc_timestamp()

            # Calculate total portfolio value
            total_value = Decimal("0")
            total_unrealized_pnl = Decimal("0")
            total_realized_pnl = Decimal("0")
            positions_count = 0

            for position in self._positions.values():
                if position.is_open():
                    # Get current price for mark-to-market
                    current_price = self._price_cache.get(position.symbol)
                    if current_price:
                        market_value = position.quantity * current_price
                        total_value += market_value

                        # Calculate unrealized P&L
                        unrealized = position.calculate_pnl(current_price)
                        total_unrealized_pnl += unrealized

                    total_realized_pnl += position.realized_pnl
                    positions_count += 1

            # Calculate returns if we have previous value
            daily_return = None
            if self._last_portfolio_value and self._last_portfolio_value > 0:
                daily_return = (
                    total_value - self._last_portfolio_value
                ) / self._last_portfolio_value
                self._daily_returns.append(daily_return)

            # Update portfolio value history
            self._portfolio_value_history.append(
                {
                    "timestamp": now,
                    "value": total_value,
                    "unrealized_pnl": total_unrealized_pnl,
                    "realized_pnl": total_realized_pnl,
                }
            )

            # Calculate risk metrics
            volatility = None
            sharpe_ratio = None
            # max_drawdown = None  # Reserved for future use

            if len(self._daily_returns) > 5:
                returns_array = np.array(list(self._daily_returns))
                volatility = np.std(returns_array) * np.sqrt(252)  # Annualized

                if volatility > 0:
                    mean_return = np.mean(returns_array) * 252  # Annualized
                    excess_return = mean_return - to_decimal(float(self.config.risk_free_rate))
                    sharpe_ratio = excess_return / volatility

                # Calculate max drawdown
                if len(self._portfolio_value_history) > 1:
                    values = [p["value"] for p in self._portfolio_value_history]
                    peak = values[0]
                    max_dd = Decimal("0")

                    for value in values[1:]:
                        if value > peak:
                            peak = value
                        else:
                            dd = (peak - value) / peak
                            if dd > max_dd:
                                max_dd = dd

                    # max_drawdown = max_dd  # Variable kept for potential future use

            # Store current values for next calculation
            self._last_portfolio_value = total_value
            self._last_update = now

            # Update metrics
            self.metrics_collector.set_gauge("analytics_portfolio_value", float(total_value))
            self.metrics_collector.set_gauge(
                "analytics_portfolio_pnl", float(total_unrealized_pnl + total_realized_pnl)
            )
            if volatility:
                self.metrics_collector.set_gauge("analytics_portfolio_volatility", volatility)
            if sharpe_ratio:
                self.metrics_collector.set_gauge("analytics_portfolio_sharpe", sharpe_ratio)

        except Exception as e:
            self.logger.error(f"Error calculating portfolio analytics: {e}")
            raise

    async def _calculate_position_analytics(self, position: Position) -> None:
        """Calculate analytics for individual position."""
        try:
            current_price = self._price_cache.get(position.symbol)
            if not current_price:
                return

            # Calculate basic metrics
            market_value = position.quantity * current_price
            unrealized_pnl = position.calculate_pnl(current_price)
            unrealized_pnl_percent = unrealized_pnl / (position.quantity * position.entry_price)

            # Calculate duration
            duration_hours = None
            if position.opened_at:
                duration = get_current_utc_timestamp() - position.opened_at
                duration_hours = duration.total_seconds() / 3600

            # Store position analytics
            position_key = f"{position.exchange}:{position.symbol}"
            self._position_analytics[position_key] = {
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_percent": unrealized_pnl_percent,
                "duration_hours": duration_hours,
                "current_price": current_price,
                "last_update": get_current_utc_timestamp(),
            }

            # Update metrics
            self.metrics_collector.set_gauge(
                "analytics_position_pnl",
                float(unrealized_pnl),
                labels={"symbol": position.symbol, "exchange": position.exchange},
            )

        except Exception as e:
            self.logger.error(f"Error calculating position analytics: {e}")
            raise

    async def _calculate_trade_analytics(self, trade: Trade) -> None:
        """Calculate analytics for executed trade."""
        try:
            # Basic trade metrics
            trade_value = trade.quantity * trade.price

            # Calculate fees as percentage of trade value
            fee_percent = (trade.fee / trade_value) * 100 if trade_value > 0 else Decimal("0")

            # Update trade metrics
            self.metrics_collector.observe_histogram("analytics_trade_value", float(trade_value))
            self.metrics_collector.observe_histogram(
                "analytics_trade_fee_percent", float(fee_percent)
            )

        except Exception as e:
            self.logger.error(f"Error calculating trade analytics: {e}")
            raise

    async def _calculate_execution_analytics(self, order: Order) -> None:
        """Calculate execution quality analytics."""
        try:
            if not order.is_filled():
                return

            # Calculate execution metrics
            execution_time = None
            if order.created_at and order.updated_at:
                execution_time = (order.updated_at - order.created_at).total_seconds()

            # Update execution metrics
            if execution_time:
                self.metrics_collector.observe_histogram(
                    "analytics_execution_time_seconds",
                    execution_time,
                    labels={"exchange": order.exchange, "order_type": order.order_type.value},
                )

        except Exception as e:
            self.logger.error(f"Error calculating execution analytics: {e}")
            raise

    async def _update_strategy_performance(self, strategy: str, trade: Trade) -> None:
        """Update strategy performance metrics."""
        try:
            if strategy not in self._strategy_performance:
                self._strategy_performance[strategy] = {
                    "trades": deque(maxlen=1000),
                    "total_pnl": Decimal("0"),
                    "total_volume": Decimal("0"),
                    "winning_trades": 0,
                    "losing_trades": 0,
                }

            perf = self._strategy_performance[strategy]
            perf["trades"].append(trade)

            # Calculate P&L if trade is completed
            if hasattr(trade, "pnl"):
                pnl = getattr(trade, "pnl", Decimal("0"))
                perf["total_pnl"] += pnl

                if pnl > 0:
                    perf["winning_trades"] += 1
                elif pnl < 0:
                    perf["losing_trades"] += 1

            # Update volume
            trade_value = trade.quantity * trade.price
            perf["total_volume"] += trade_value

            # Update strategy metrics
            self.metrics_collector.set_gauge(
                "analytics_strategy_pnl", float(perf["total_pnl"]), labels={"strategy": strategy}
            )

            total_trades = perf["winning_trades"] + perf["losing_trades"]
            if total_trades > 0:
                win_rate = (perf["winning_trades"] / total_trades) * 100
                self.metrics_collector.set_gauge(
                    "analytics_strategy_win_rate", win_rate, labels={"strategy": strategy}
                )

        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {e}")
            raise

    async def _update_realtime_pnl(self, symbol: str) -> None:
        """Update real-time P&L for positions with updated prices."""
        try:
            updated_positions = [
                pos for key, pos in self._positions.items() if symbol in key and pos.is_open()
            ]

            for position in updated_positions:
                await self._calculate_position_analytics(position)

            # Trigger portfolio update if any positions updated
            if updated_positions:
                await self._calculate_portfolio_analytics()

        except Exception as e:
            self.logger.error(f"Error updating real-time P&L: {e}")
            raise

    async def _check_risk_thresholds(self) -> None:
        """Check risk thresholds and generate alerts."""
        try:
            # Check portfolio drawdown
            if len(self._portfolio_value_history) > 1:
                current_value = self._portfolio_value_history[-1]["value"]
                peak_value = max(p["value"] for p in self._portfolio_value_history)

                if peak_value > 0:
                    current_drawdown = (peak_value - current_value) / peak_value
                    threshold = self.config.alert_thresholds.get("max_drawdown", Decimal("0.05"))

                    if current_drawdown > threshold:
                        await self._generate_alert(
                            "drawdown_breach",
                            AlertSeverity.HIGH,
                            "Maximum Drawdown Threshold Breached",
                            f"Current drawdown: {current_drawdown:.2%}, Threshold: {threshold:.2%}",
                            current_value=current_drawdown,
                            threshold_value=threshold,
                        )

        except Exception as e:
            self.logger.error(f"Error checking risk thresholds: {e}")

    async def _generate_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        current_value: Decimal | None = None,
        threshold_value: Decimal | None = None,
    ) -> None:
        """Generate analytics alert."""
        try:
            # Check if alert already exists
            if alert_id in self._active_alerts:
                return

            alert = AnalyticsAlert(
                id=alert_id,
                timestamp=get_current_utc_timestamp(),
                severity=severity,
                title=title,
                message=message,
                metric_name=alert_id,
                current_value=current_value,
                threshold_value=threshold_value,
            )

            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)

            # Update alert metrics
            self.metrics_collector.increment_counter(
                "analytics_alerts_generated", labels={"severity": severity.value, "type": alert_id}
            )

            self.logger.warning(f"Generated alert: {title} - {message}")

        except Exception as e:
            self.logger.error(f"Error generating alert: {e}")

    async def _process_alerts(self) -> None:
        """Process and clean up alerts."""
        try:
            now = get_current_utc_timestamp()
            expired_alerts = []

            # Mark old alerts as resolved
            for alert_id, alert in self._active_alerts.items():
                if (now - alert.timestamp).total_seconds() > 3600:  # 1 hour
                    expired_alerts.append(alert_id)

            for alert_id in expired_alerts:
                if alert_id in self._active_alerts:
                    alert = self._active_alerts[alert_id]
                    alert.resolved = True
                    alert.resolved_timestamp = now
                    del self._active_alerts[alert_id]

        except Exception as e:
            self.logger.error(f"Error processing alerts: {e}")

    async def _calculate_portfolio_metrics(self) -> PortfolioMetrics | None:
        """Calculate and return current portfolio metrics."""
        try:
            if not self._portfolio_value_history:
                return None

            latest = self._portfolio_value_history[-1]

            # Calculate returns
            daily_return = None
            mtd_return = None
            ytd_return = None

            if len(self._daily_returns) > 0:
                daily_return = to_decimal(self._daily_returns[-1] * 100)  # Convert to percentage

            if len(self._daily_returns) > 30:
                mtd_return = to_decimal(np.mean(list(self._daily_returns)[-30:]) * 30 * 100)

            if len(self._daily_returns) > 250:
                ytd_return = to_decimal(np.mean(list(self._daily_returns)) * 252 * 100)

            # Calculate risk metrics
            volatility = None
            sharpe_ratio = None
            # max_drawdown = None  # Reserved for future use

            if len(self._daily_returns) > 5:
                returns_array = np.array(list(self._daily_returns))
                volatility = to_decimal(np.std(returns_array) * np.sqrt(252) * 100)

                if volatility and volatility > 0:
                    mean_return = np.mean(returns_array) * 252
                    excess_return = mean_return - to_decimal(self.config.risk_free_rate)
                    sharpe_ratio = excess_return / (volatility / Decimal("100"))

            # Calculate portfolio composition
            total_value = latest["value"]
            positions_count = len([p for p in self._positions.values() if p.is_open()])
            active_strategies = len(self._strategy_performance)

            return PortfolioMetrics(
                timestamp=latest["timestamp"],
                total_value=latest["value"],
                cash=Decimal("0"),  # TBD: Cash tracking integration with portfolio service
                invested_capital=total_value,
                unrealized_pnl=latest["unrealized_pnl"],
                realized_pnl=latest["realized_pnl"],
                total_pnl=latest["unrealized_pnl"] + latest["realized_pnl"],
                daily_return=daily_return,
                mtd_return=mtd_return,
                ytd_return=ytd_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=None,  # Will be calculated in future versions
                positions_count=positions_count,
                active_strategies=active_strategies,
            )

        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return None

    async def _calculate_position_metrics(self, position: Position) -> PositionMetrics | None:
        """Calculate metrics for individual position."""
        try:
            current_price = self._price_cache.get(position.symbol)
            if not current_price:
                return None

            # position_key = f"{position.exchange}:{position.symbol}"  # Reserved for analytics lookup
            # analytics = self._position_analytics.get(position_key, {})  # For future use

            market_value = position.quantity * current_price
            unrealized_pnl = position.calculate_pnl(current_price)
            unrealized_pnl_percent = (
                unrealized_pnl / (position.quantity * position.entry_price)
            ) * 100

            # Calculate portfolio weight
            total_portfolio_value = self._last_portfolio_value or Decimal("1")
            weight = (market_value / total_portfolio_value) * 100

            # Calculate duration
            duration_hours = None
            if position.opened_at:
                duration = get_current_utc_timestamp() - position.opened_at
                duration_hours = to_decimal(duration.total_seconds() / 3600)

            return PositionMetrics(
                timestamp=get_current_utc_timestamp(),
                symbol=position.symbol,
                exchange=position.exchange,
                side=position.side.value,
                quantity=position.quantity,
                entry_price=position.entry_price,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_percent=unrealized_pnl_percent,
                realized_pnl=position.realized_pnl,
                total_pnl=unrealized_pnl + position.realized_pnl,
                weight=weight,
                duration_hours=duration_hours,
            )

        except Exception as e:
            self.logger.error(f"Error calculating position metrics: {e}")
            return None

    async def _calculate_strategy_metrics(self, strategy: str) -> StrategyMetrics | None:
        """Calculate metrics for strategy performance."""
        try:
            if strategy not in self._strategy_performance:
                return None

            perf = self._strategy_performance[strategy]

            # Calculate basic metrics
            total_trades = perf["winning_trades"] + perf["losing_trades"]
            win_rate = None
            if total_trades > 0:
                win_rate = to_decimal((perf["winning_trades"] / total_trades) * 100)

            # Calculate returns and risk metrics from trade history
            if len(perf["trades"]) > 0:
                # Extract returns from trades
                trade_returns = []
                for trade in perf["trades"]:
                    if hasattr(trade, "pnl") and hasattr(trade, "value"):
                        pnl = getattr(trade, "pnl", Decimal("0"))
                        value = getattr(trade, "value", trade.quantity * trade.price)
                        if value > 0:
                            trade_return = pnl / value
                            trade_returns.append(trade_return)

                # Calculate risk metrics
                volatility = None
                sharpe_ratio = None
                if len(trade_returns) > 2:
                    returns_array = np.array(trade_returns)
                    volatility = to_decimal(np.std(returns_array) * np.sqrt(252) * 100)

                    if volatility and volatility > 0:
                        mean_return = np.mean(returns_array) * 252
                        excess_return = mean_return - to_decimal(self.config.risk_free_rate)
                        sharpe_ratio = excess_return / (volatility / Decimal("100"))

            return StrategyMetrics(
                timestamp=get_current_utc_timestamp(),
                strategy_name=strategy,
                total_pnl=perf["total_pnl"],
                unrealized_pnl=Decimal(
                    "0"
                ),  # TBD: Calculate from open positions using current prices
                realized_pnl=perf["total_pnl"],
                total_return=Decimal(
                    "0"
                ),  # TBD: Calculate based on allocated capital from capital management service
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                total_trades=total_trades,
                winning_trades=perf["winning_trades"],
                losing_trades=perf["losing_trades"],
                capital_allocated=Decimal(
                    "100000"
                ),  # TBD: Integration with capital allocator service
                capital_utilized=perf["total_volume"],
                utilization_rate=Decimal("0"),  # TBD: Calculate capital utilization rate
                active_positions=0,  # TBD: Count active positions by strategy from position tracking
            )

        except Exception as e:
            self.logger.error(f"Error calculating strategy metrics: {e}")
            return None

    def _update_price_return_analytics(self, symbol: str, price_return: Decimal) -> None:
        """Update price return analytics for volatility and correlation tracking."""
        try:
            # Store price return for volatility calculation
            if symbol not in self._volatility_surface:
                self._volatility_surface[symbol] = defaultdict(lambda: deque(maxlen=252))  # type: ignore

            self._volatility_surface[symbol]["returns"].append(price_return)

            # Calculate rolling volatility
            returns = list(self._volatility_surface[symbol]["returns"])
            if len(returns) >= 20:  # Need minimum observations
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                self._volatility_surface[symbol]["volatility"] = to_decimal(volatility)

        except Exception as e:
            self.logger.error(f"Error updating price return analytics: {e}")

    async def _update_real_time_risk_metrics(self) -> None:
        """Update real-time risk metrics including VaR and correlations."""
        try:
            if len(self._intraday_returns) < 20:
                return

            # Calculate intraday VaR
            returns = np.array(list(self._intraday_returns))
            var_95 = np.percentile(returns, 5)  # 95% VaR
            var_99 = np.percentile(returns, 1)  # 99% VaR

            # Store VaR metrics
            self._var_history.append(
                {
                    "timestamp": get_current_utc_timestamp(),
                    "var_95": to_decimal(var_95),
                    "var_99": to_decimal(var_99),
                    "volatility": to_decimal(np.std(returns)),
                }
            )

            # Update correlation matrix if we have position data
            await self._update_correlation_matrix()

        except Exception as e:
            self.logger.error(f"Error updating real-time risk metrics: {e}")

    async def _update_correlation_matrix(self) -> None:
        """Update real-time correlation matrix for position risk."""
        try:
            symbols = [pos.symbol for pos in self._positions.values() if pos.is_open()]

            if len(symbols) < 2:
                return

            # Calculate correlation matrix from price returns
            returns_data = {}
            for symbol in symbols:
                if (
                    symbol in self._volatility_surface
                    and "returns" in self._volatility_surface[symbol]
                ):
                    returns_data[symbol] = list(self._volatility_surface[symbol]["returns"])

            if len(returns_data) >= 2:
                # Ensure all return series have same length
                min_length = min(len(returns) for returns in returns_data.values())
                if min_length >= 20:
                    df_returns = pd.DataFrame(
                        {symbol: returns[-min_length:] for symbol, returns in returns_data.items()}
                    )

                    correlation_matrix = df_returns.corr()

                    # Store correlations
                    for symbol1 in symbols:
                        for symbol2 in symbols:
                            if (
                                symbol1 != symbol2
                                and symbol1 in correlation_matrix.index
                                and symbol2 in correlation_matrix.columns
                            ):
                                corr_value = correlation_matrix.loc[symbol1, symbol2]
                                if not np.isnan(corr_value):
                                    self._correlation_cache[symbol1][symbol2] = to_decimal(
                                        corr_value
                                    )

        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {e}")

    async def _calculate_detailed_trade_analytics(self, trade: Trade) -> TradeAnalytics | None:
        """Calculate comprehensive trade analytics including execution quality."""
        try:
            # Basic trade metrics
            # trade_value = trade.quantity * trade.price  # For future transaction cost analysis
            fees = getattr(trade, "fee", Decimal("0"))

            # Calculate execution quality metrics
            slippage = await self._calculate_trade_slippage(trade)
            market_impact = await self._calculate_market_impact(trade)
            timing_cost = await self._calculate_timing_cost(trade)

            # Calculate risk-adjusted metrics
            position_key = f"{trade.exchange}:{trade.symbol}"
            position = self._positions.get(position_key)

            risk_adjusted_return = None
            sharpe_contribution = None
            var_impact = None

            if position:
                # Calculate position's contribution to portfolio risk
                portfolio_value = self._last_portfolio_value or Decimal("1")
                position_weight = (position.quantity * trade.price) / portfolio_value

                # Estimate VaR impact (simplified)
                if trade.symbol in self._volatility_surface:
                    volatility = self._volatility_surface[trade.symbol].get(
                        "volatility", Decimal("0")
                    )
                    var_impact = position_weight * volatility

            # Performance attribution factors
            attribution_factors = {}
            if hasattr(trade, "strategy"):
                strategy = getattr(trade, "strategy", "unknown")
                attribution_factors["strategy"] = strategy

            # Calculate alpha/beta if we have benchmark data
            alpha_contribution = None
            benchmark_return = None

            return TradeAnalytics(
                trade_id=trade.trade_id,
                timestamp=trade.timestamp,
                symbol=trade.symbol,
                exchange=trade.exchange,
                strategy=getattr(trade, "strategy", "unknown"),
                side=trade.side.value,
                quantity=trade.quantity,
                entry_price=trade.price,
                realized_pnl=getattr(trade, "pnl", None),
                fees=fees,
                slippage=slippage,
                market_impact=market_impact,
                timing_cost=timing_cost,
                risk_adjusted_return=risk_adjusted_return,
                sharpe_contribution=sharpe_contribution,
                var_impact=var_impact,
                alpha_contribution=alpha_contribution,
                benchmark_return=benchmark_return,
                attribution_factors=attribution_factors,
            )

        except Exception as e:
            self.logger.error(f"Error calculating detailed trade analytics: {e}")
            return None

    async def _calculate_trade_slippage(self, trade: Trade) -> Decimal | None:
        """Calculate trade slippage against market prices."""
        try:
            # Get market price at time of trade execution
            current_price = self._price_cache.get(trade.symbol)
            if not current_price:
                return None

            # Calculate slippage as difference from mid-market price
            # This is a simplified calculation - in practice would use bid/ask spreads
            slippage = abs(trade.price - current_price) / current_price

            # Store slippage for tracking
            self._slippage_tracker[trade.symbol].append(slippage)

            return to_decimal(slippage * 10000)  # Return in basis points

        except Exception as e:
            self.logger.error(f"Error calculating trade slippage: {e}")
            return None

    async def _calculate_market_impact(self, trade: Trade) -> Decimal | None:
        """Calculate market impact of trade execution."""
        try:
            # This is a simplified market impact model
            # In practice would use more sophisticated models like Almgren-Chriss

            trade_value = trade.quantity * trade.price

            # Estimate daily volume (simplified - would use historical data)
            estimated_daily_volume = trade_value * Decimal(
                "100"
            )  # Assume trade is 1% of daily volume

            if estimated_daily_volume > 0:
                participation_rate = trade_value / estimated_daily_volume

                # Simple square-root market impact model
                volatility = Decimal("0.02")  # 2% default volatility
                if trade.symbol in self._volatility_surface:
                    volatility = self._volatility_surface[trade.symbol].get(
                        "volatility", volatility
                    )

                market_impact = volatility * (participation_rate ** Decimal("0.5"))

                # Store for tracking
                self._market_impact_tracker[trade.symbol].append(market_impact)

                return to_decimal(market_impact * 10000)  # Return in basis points

            return None

        except Exception as e:
            self.logger.error(f"Error calculating market impact: {e}")
            return None

    async def _calculate_timing_cost(self, trade: Trade) -> Decimal | None:
        """Calculate timing cost of trade execution."""
        try:
            # Timing cost is the difference between decision price and execution price
            # This would typically require storing decision timestamps and prices

            # Simplified implementation - would need more sophisticated timing tracking
            if hasattr(trade, "decision_price") and hasattr(trade, "decision_time"):
                decision_price = trade.decision_price
                timing_cost = abs(trade.price - decision_price) / decision_price
                return to_decimal(timing_cost * 10000)  # Basis points

            return None

        except Exception as e:
            self.logger.error(f"Error calculating timing cost: {e}")
            return None

    async def _calculate_performance_attribution(
        self, period_days: int
    ) -> PerformanceAttribution | None:
        """Calculate comprehensive performance attribution."""
        try:
            now = get_current_utc_timestamp()
            period_start = now - timedelta(days=period_days)

            if not self._portfolio_value_history or len(self._portfolio_value_history) < 2:
                return None

            # Find portfolio values for the period
            period_values = [
                v for v in self._portfolio_value_history if v["timestamp"] >= period_start
            ]

            if len(period_values) < 2:
                return None

            # Calculate total return
            start_value = period_values[0]["value"]
            end_value = period_values[-1]["value"]
            total_return = (end_value - start_value) / start_value

            # Strategy attribution
            strategy_attribution = {}
            for strategy, perf in self._strategy_performance.items():
                if perf["total_pnl"] != Decimal("0"):
                    strategy_contribution = perf["total_pnl"] / start_value
                    strategy_attribution[strategy] = strategy_contribution

            # Sector attribution (simplified - would need sector mapping)
            sector_allocation = {}
            for position in self._positions.values():
                if position.is_open():
                    # Simplified sector assignment
                    sector = self._get_position_sector(position.symbol)
                    position_value = position.quantity * self._price_cache.get(
                        position.symbol, position.entry_price
                    )
                    position_return = (
                        self._price_cache.get(position.symbol, position.entry_price)
                        - position.entry_price
                    ) / position.entry_price

                    if sector not in sector_allocation:
                        sector_allocation[sector] = Decimal("0")
                    sector_allocation[sector] += (position_value / start_value) * position_return

            # Factor attribution (simplified)
            factor_attribution: dict[str, Decimal] = {}

            # Calculate tracking error and information ratio
            benchmark_return = None  # Would need benchmark data
            active_return = None
            tracking_error = None
            information_ratio = None

            if benchmark_return:
                active_return = total_return - benchmark_return

                # Calculate tracking error from historical active returns
                if len(self._daily_returns) >= 30:
                    # Simplified - would need actual benchmark returns
                    tracking_error = to_decimal(np.std(list(self._daily_returns)[-30:]))

                    if tracking_error and tracking_error > 0:
                        information_ratio = active_return / tracking_error

            return PerformanceAttribution(
                timestamp=now,
                period_start=period_start,
                period_end=now,
                total_return=to_decimal(total_return),
                benchmark_return=benchmark_return,
                active_return=active_return,
                strategy_attribution=strategy_attribution,
                sector_allocation=sector_allocation,
                factor_attribution=factor_attribution,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
            )

        except Exception as e:
            self.logger.error(f"Error calculating performance attribution: {e}")
            return None

    async def _calculate_execution_quality_metrics(
        self, symbol: str | None = None
    ) -> dict[str, Any]:
        """Calculate comprehensive execution quality metrics."""
        try:
            metrics: dict[str, Any] = {
                "overall": {},
                "by_symbol": {},
                "by_strategy": {},
                "time_series": {},
            }

            # Calculate overall execution metrics
            if self._slippage_tracker:
                all_slippage: list[float] = []
                for slippage_list in self._slippage_tracker.values():
                    all_slippage.extend(slippage_list)

                if all_slippage:
                    metrics["overall"]["avg_slippage_bps"] = np.mean(all_slippage) * 10000
                    metrics["overall"]["median_slippage_bps"] = np.median(all_slippage) * 10000
                    metrics["overall"]["slippage_volatility"] = np.std(all_slippage) * 10000

            # Market impact metrics
            if self._market_impact_tracker:
                all_impact: list[float] = []
                for impact_list in self._market_impact_tracker.values():
                    all_impact.extend(impact_list)

                if all_impact:
                    metrics["overall"]["avg_market_impact_bps"] = np.mean(all_impact) * 10000
                    metrics["overall"]["median_market_impact_bps"] = np.median(all_impact) * 10000

            # Symbol-specific metrics
            symbols = [symbol] if symbol else list(self._slippage_tracker.keys())

            for sym in symbols:
                symbol_metrics = {}

                if self._slippage_tracker.get(sym):
                    slippage_data = list(self._slippage_tracker[sym])
                    symbol_metrics["avg_slippage_bps"] = np.mean(slippage_data) * 10000
                    symbol_metrics["slippage_std_bps"] = np.std(slippage_data) * 10000

                if self._market_impact_tracker.get(sym):
                    impact_data = list(self._market_impact_tracker[sym])
                    symbol_metrics["avg_market_impact_bps"] = np.mean(impact_data) * 10000
                    symbol_metrics["impact_std_bps"] = np.std(impact_data) * 10000

                if symbol_metrics:
                    metrics["by_symbol"][sym] = symbol_metrics

            # Strategy-specific metrics (would need strategy mapping)
            # This would require tracking trades by strategy

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating execution quality metrics: {e}")
            return {}

    def _get_position_sector(self, symbol: str) -> str:
        """Get sector for position (simplified implementation)."""
        # This would typically use external data or mapping
        # Simplified sector assignment based on symbol patterns
        if "BTC" in symbol or "ETH" in symbol:
            return "Cryptocurrency"
        elif "USD" in symbol or "EUR" in symbol:
            return "Foreign Exchange"
        else:
            return "Other"

    # Enhanced WebSocket Integration Methods

    async def _initialize_cache(self) -> None:
        """Initialize Redis cache connection if configured."""
        try:
            # Cache client must be injected - no direct instantiation
            if self._cache_client:
                self._cache_enabled = True
                self.logger.info("Cache client available for analytics")
            else:
                self.logger.info("No cache client injected - running without caching")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Redis cache: {e}")
            self._cache_enabled = False

    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for live analytics updates."""
        try:
            if hasattr(self.config, "websocket_port") and self.config.websocket_port:

                async def handle_websocket(websocket, path):
                    self._websocket_clients.add(websocket)
                    client_addr = getattr(websocket, "remote_address", "unknown")
                    self.logger.info(f"New WebSocket client connected: {client_addr}")

                    try:
                        # Send initial connection confirmation
                        welcome_msg = {
                            "type": "connection_established",
                            "timestamp": get_current_utc_timestamp().isoformat(),
                            "client_id": id(websocket),
                        }
                        await websocket.send(json.dumps(welcome_msg))

                        # Set up heartbeat task
                        # Set up heartbeat task with proper cleanup
                        heartbeat_task = asyncio.create_task(self._websocket_heartbeat(websocket))

                        try:
                            # Wait for connection to close or timeout
                            await asyncio.wait_for(
                                websocket.wait_closed(),
                                timeout=getattr(
                                    self.config, "websocket_timeout", 300
                                ),  # 5 min default
                            )
                        except asyncio.TimeoutError:
                            self.logger.info(f"WebSocket client {client_addr} connection timed out")
                            await websocket.close()
                        finally:
                            # Properly cancel and wait for heartbeat task cleanup
                            heartbeat_task.cancel()
                            try:
                                await heartbeat_task
                            except asyncio.CancelledError:
                                pass

                    except websockets.exceptions.ConnectionClosed:
                        self.logger.info(f"WebSocket client {client_addr} disconnected normally")
                    except Exception as e:
                        self.logger.warning(f"WebSocket client {client_addr} error: {e}")
                        try:
                            await websocket.close()
                        except Exception as e:
                            self.logger.debug(f"WebSocket close error during cleanup: {e}")
                            # Continue cleanup regardless of close error
                    finally:
                        self._websocket_clients.discard(websocket)
                        self.logger.debug(f"WebSocket client {client_addr} cleanup completed")

                # Configure WebSocket server with enhanced timeouts and limits
                self._websocket_server = await websockets.serve(
                    handle_websocket,
                    "localhost",
                    self.config.websocket_port,
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,  # Wait 10 seconds for pong
                    close_timeout=10,  # Wait 10 seconds for close
                    max_size=1024 * 1024,  # 1MB max message size
                    max_queue=32,  # Limit message queue to prevent memory issues
                    compression=None,  # Disable compression for low latency
                )
                self.logger.info(f"WebSocket server started on port {self.config.websocket_port}")
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")

    async def _websocket_broadcast_loop(self) -> None:
        """Background loop for broadcasting analytics updates to WebSocket clients."""
        last_broadcast_time = 0
        broadcast_interval = getattr(
            self.config, "websocket_broadcast_interval", 1.0
        )  # Default 1 second
        max_message_size = getattr(
            self.config, "websocket_max_message_size", 1024 * 1024
        )  # 1MB default

        while self._running:
            try:
                # Skip broadcast if no clients connected (backpressure)
                if not self._websocket_clients:
                    await asyncio.sleep(broadcast_interval)
                    continue

                current_time = asyncio.get_event_loop().time()

                # Rate limiting - only broadcast at configured intervals
                if current_time - last_broadcast_time < broadcast_interval:
                    await asyncio.sleep(0.1)  # Small sleep to prevent busy waiting
                    continue

                # Get latest analytics data with timeout to prevent blocking
                try:
                    analytics_tasks = [
                        asyncio.wait_for(self.get_portfolio_metrics(), timeout=5.0),
                        asyncio.wait_for(self.get_position_analytics(), timeout=5.0),
                        asyncio.wait_for(self.get_portfolio_risk_metrics(), timeout=5.0),
                    ]

                    portfolio_metrics, position_metrics, risk_metrics = await asyncio.gather(
                        *analytics_tasks, return_exceptions=True
                    )

                    # Handle any exceptions from data gathering
                    if isinstance(portfolio_metrics, Exception):
                        portfolio_metrics = None
                    if isinstance(position_metrics, Exception):
                        position_metrics = None
                    if isinstance(risk_metrics, Exception):
                        risk_metrics = None

                except asyncio.TimeoutError:
                    self.logger.warning("Analytics data gathering timed out, skipping broadcast")
                    await asyncio.sleep(broadcast_interval)
                    continue

                # Create broadcast message
                message = {
                    "timestamp": get_current_utc_timestamp().isoformat(),
                    "type": "analytics_update",
                    "data": {
                        "portfolio": portfolio_metrics,
                        "positions": position_metrics,
                        "risk": risk_metrics,
                        "alerts": [alert.dict() for alert in self._active_alerts.values()],
                    },
                }

                # Serialize and check message size
                try:
                    message_str = json.dumps(message, default=str)
                    if len(message_str.encode("utf-8")) > max_message_size:
                        self.logger.warning(
                            f"Broadcast message too large ({len(message_str)} bytes), truncating"
                        )
                        # Create a smaller message with essential data only
                        message = {
                            "timestamp": get_current_utc_timestamp().isoformat(),
                            "type": "analytics_update_lite",
                            "data": {
                                "portfolio": {
                                    "total_value": getattr(portfolio_metrics, "total_value", 0)
                                    if portfolio_metrics
                                    else 0,
                                    "total_pnl": getattr(portfolio_metrics, "total_pnl", 0)
                                    if portfolio_metrics
                                    else 0,
                                },
                                "positions_count": len(position_metrics) if position_metrics else 0,
                                "alerts_count": len(self._active_alerts),
                            },
                        }
                        message_str = json.dumps(message, default=str)

                except Exception as e:
                    self.logger.error(f"Error serializing broadcast message: {e}")
                    continue

                # Broadcast to all connected clients
                await self._broadcast_to_clients(message_str)
                last_broadcast_time = current_time

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in WebSocket broadcast loop: {e}")
                await asyncio.sleep(broadcast_interval)  # Use configured interval for error backoff

    async def _websocket_heartbeat(self, websocket) -> None:
        """Send periodic heartbeat to WebSocket client."""
        try:
            while not websocket.closed:
                heartbeat_msg = {
                    "type": "heartbeat",
                    "timestamp": get_current_utc_timestamp().isoformat(),
                }
                await websocket.send(json.dumps(heartbeat_msg))
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
        except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception as e:
            self.logger.warning(f"Heartbeat error for WebSocket client: {e}")

    async def _broadcast_to_clients(self, message: str) -> None:
        """Broadcast message to all WebSocket clients with proper error handling and backpressure."""
        if not self._websocket_clients:
            return

        # Use a copy of the set to avoid modification during iteration
        clients_to_send = set(self._websocket_clients)
        disconnected_clients = set()

        # Implement backpressure handling with timeouts
        async def send_to_client_with_backpressure(client):
            try:
                # Add timeout to prevent hanging on slow clients
                await asyncio.wait_for(client.send(message), timeout=2.0)
                return client, None
            except asyncio.TimeoutError:
                self.logger.warning("Client send timeout - removing slow client")
                return client, "send_timeout"
            except websockets.exceptions.ConnectionClosed:
                return client, "connection_closed"
            except Exception as e:
                self.logger.warning(f"Error sending to WebSocket client: {e}")
                return client, str(e)

        # Limit concurrent sends to prevent resource exhaustion
        max_concurrent_sends = 50  # Reasonable limit for concurrent WebSocket sends

        # Process clients in batches if there are many
        client_list = list(clients_to_send)
        for i in range(0, len(client_list), max_concurrent_sends):
            batch = client_list[i : i + max_concurrent_sends]
            send_tasks = [send_to_client_with_backpressure(client) for client in batch]

            if send_tasks:
                try:
                    # Use timeout for the entire batch
                    results = await asyncio.wait_for(
                        asyncio.gather(*send_tasks, return_exceptions=True), timeout=5.0
                    )

                    # Process results and collect failed clients
                    for result in results:
                        if isinstance(result, tuple):
                            client, error = result
                            if error:
                                disconnected_clients.add(client)
                        elif isinstance(result, Exception):
                            self.logger.error(f"Unexpected error in broadcast batch: {result}")

                except asyncio.TimeoutError:
                    self.logger.error(f"Broadcast batch timeout for {len(send_tasks)} clients")
                    # Mark all clients in this batch as problematic
                    disconnected_clients.update(batch)
                except Exception as e:
                    self.logger.error(f"Critical error in WebSocket broadcast batch: {e}")
                    disconnected_clients.update(batch)

        # Remove disconnected clients from active set
        if disconnected_clients:
            self._websocket_clients -= disconnected_clients
            self.logger.debug(f"Removed {len(disconnected_clients)} disconnected WebSocket clients")

    async def _cache_sync_loop(self) -> None:
        """Background loop for syncing analytics data to Redis cache."""
        while self._running and self._cache_enabled:
            try:
                # Cache key metrics for fast access
                portfolio_metrics = await self.get_portfolio_metrics()
                if portfolio_metrics and self._cache_client:
                    await self._cache_client.set(
                        "analytics:portfolio:latest",
                        json.dumps(portfolio_metrics, default=str),
                        expire=60,  # 1-minute expiry
                    )

                # Cache position metrics
                position_metrics = await self.get_position_analytics()
                if position_metrics and self._cache_client:
                    await self._cache_client.set(
                        "analytics:positions:latest",
                        json.dumps(position_metrics, default=str),
                        expire=60,
                    )

                await asyncio.sleep(5)  # Update cache every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cache sync loop: {e}")
                await asyncio.sleep(5)

    # Advanced Analytics Methods

    async def calculate_stress_test_scenarios(self) -> dict[str, Any]:
        """
        Calculate portfolio performance under various stress scenarios.

        Returns:
            Dictionary containing stress test results for different scenarios
        """
        try:
            results = {}
            current_portfolio_value = await self._get_current_portfolio_value()

            if current_portfolio_value is None:
                return {}

            for scenario_name, scenario_params in self._stress_scenarios.items():
                scenario_result = {
                    "scenario_name": scenario_name,
                    "current_value": current_portfolio_value,
                    "stressed_value": 0.0,
                    "absolute_loss": 0.0,
                    "percentage_loss": 0.0,
                    "position_impacts": {},
                }

                total_stressed_value = Decimal("0")

                # Apply stress to each position
                for symbol, position in self._positions.items():
                    position_value = position.size * position.current_price

                    # Apply scenario-specific shocks
                    if "equity_shock" in scenario_params:
                        shock = Decimal(str(scenario_params["equity_shock"]))
                        stressed_value = position_value * (1 + shock)
                    else:
                        stressed_value = position_value

                    total_stressed_value += stressed_value

                    scenario_result["position_impacts"][symbol] = {
                        "original_value": position_value,
                        "stressed_value": float(stressed_value),
                        "impact": float(stressed_value - position_value),
                    }

                scenario_result["stressed_value"] = float(total_stressed_value)
                scenario_result["absolute_loss"] = float(
                    total_stressed_value - current_portfolio_value
                )
                scenario_result["percentage_loss"] = float(
                    (total_stressed_value - current_portfolio_value) / current_portfolio_value * 100
                )

                results[scenario_name] = scenario_result

            return {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "scenarios": results,
                "summary": {
                    "worst_case_scenario": (
                        min(results.keys(), key=lambda k: results[k]["percentage_loss"])
                        if results
                        else None
                    ),
                    "worst_case_loss": min(
                        (r["percentage_loss"] for r in results.values()), default=0.0
                    ),
                    "average_loss": (
                        np.mean([r["percentage_loss"] for r in results.values()])
                        if results
                        else 0.0
                    ),
                },
            }

        except Exception as e:
            self.logger.error(f"Error in stress test calculation: {e}")
            return {}

    async def calculate_concentration_risk(self) -> dict[str, Any]:
        """
        Calculate concentration risk metrics for the portfolio.

        Returns:
            Dictionary containing concentration risk analysis
        """
        try:
            if not self._positions:
                return {}

            total_portfolio_value = await self._get_current_portfolio_value()
            if total_portfolio_value is None or total_portfolio_value == 0:
                return {}

            # Calculate position weights
            position_weights = {}
            sector_weights: dict[str, float] = defaultdict(float)
            currency_weights: dict[str, float] = defaultdict(float)

            for symbol, position in self._positions.items():
                position_value = position.size * position.current_price
                weight = float(position_value / total_portfolio_value)
                position_weights[symbol] = weight

                # Aggregate by sector
                sector = self._get_position_sector(symbol)
                sector_weights[sector] += weight

                # Aggregate by currency (simplified)
                currency = symbol[-3:] if len(symbol) > 3 else "USD"
                currency_weights[currency] += weight

            # Calculate concentration metrics
            position_weights_list = list(position_weights.values())

            # Herfindahl-Hirschman Index (HHI)
            hhi = sum(w**2 for w in position_weights_list)

            # Maximum single position weight
            max_position_weight = max(position_weights_list) if position_weights_list else 0

            # Top 5 positions concentration
            top_5_weight = sum(sorted(position_weights_list, reverse=True)[:5])

            # Effective number of positions
            effective_positions = 1 / hhi if hhi > 0 else 0

            return {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "position_concentration": {
                    "herfindahl_index": hhi,
                    "effective_positions": effective_positions,
                    "max_position_weight": max_position_weight,
                    "top_5_concentration": top_5_weight,
                    "position_weights": position_weights,
                },
                "sector_concentration": dict(sector_weights),
                "currency_concentration": dict(currency_weights),
                "risk_flags": {
                    "high_concentration": max_position_weight > 0.20,
                    "very_high_concentration": max_position_weight > 0.30,
                    "low_diversification": effective_positions < 5,
                },
            }

        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {e}")
            return {}

    async def calculate_advanced_var(
        self, confidence_levels: list[float] | None = None
    ) -> dict[str, Any]:
        """
        Calculate advanced Value at Risk metrics using multiple methodologies.

        Args:
            confidence_levels: List of confidence levels (default: [0.95, 0.99])

        Returns:
            Dictionary containing VaR calculations using different methods
        """
        try:
            confidence_levels = confidence_levels or [0.95, 0.99]

            if len(self._intraday_returns) < 30:
                return {"error": "Insufficient return data for VaR calculation"}

            returns = np.array(list(self._intraday_returns))
            current_portfolio_value = await self._get_current_portfolio_value()

            if current_portfolio_value is None:
                return {"error": "Unable to get current portfolio value"}

            var_results = {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "portfolio_value": float(current_portfolio_value),
                "methodologies": {},
            }

            for confidence in confidence_levels:
                alpha = 1 - confidence

                # Historical VaR
                historical_var = np.percentile(returns, alpha * 100)

                # Parametric VaR (assuming normal distribution)
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                z_score = stats.norm.ppf(alpha)
                parametric_var = mean_return + z_score * std_return

                # Modified VaR (Cornish-Fisher expansion)
                skewness = stats.skew(returns)
                kurtosis = stats.kurtosis(returns)

                cornish_fisher_z = (
                    z_score
                    + (z_score**2 - 1) * skewness / 6
                    + (z_score**3 - 3 * z_score) * kurtosis / 24
                    - (2 * z_score**3 - 5 * z_score) * skewness**2 / 36
                )
                modified_var = mean_return + cornish_fisher_z * std_return

                # Expected Shortfall (CVaR)
                tail_returns = returns[returns <= historical_var]
                expected_shortfall = (
                    np.mean(tail_returns) if len(tail_returns) > 0 else historical_var
                )

                var_results["methodologies"][f"{confidence:.0%}"] = {
                    "historical_var": {
                        "percentage": historical_var,
                        "dollar_amount": float(historical_var * current_portfolio_value),
                    },
                    "parametric_var": {
                        "percentage": parametric_var,
                        "dollar_amount": float(parametric_var * current_portfolio_value),
                    },
                    "modified_var": {
                        "percentage": modified_var,
                        "dollar_amount": float(modified_var * current_portfolio_value),
                    },
                    "expected_shortfall": {
                        "percentage": expected_shortfall,
                        "dollar_amount": float(expected_shortfall * current_portfolio_value),
                    },
                }

            # Calculate VaR breach statistics
            var_95_breaches = sum(
                1
                for r in returns
                if r < var_results["methodologies"]["95%"]["historical_var"]["percentage"]
            )
            breach_rate = var_95_breaches / len(returns)

            var_results["breach_analysis"] = {
                "var_95_breaches": var_95_breaches,
                "total_observations": len(returns),
                "breach_rate": breach_rate,
                "expected_breach_rate": 0.05,
                "model_accuracy": "Good" if 0.03 <= breach_rate <= 0.07 else "Poor",
            }

            return var_results

        except Exception as e:
            self.logger.error(f"Error calculating advanced VaR: {e}")
            return {"error": str(e)}

    async def generate_real_time_dashboard_data(self) -> dict[str, Any]:
        """
        Generate comprehensive data for real-time analytics dashboard.

        Returns:
            Dictionary containing all dashboard data
        """
        try:
            # Get all analytics components
            portfolio_metrics = await self.get_portfolio_metrics()
            position_analytics = await self.get_position_analytics()
            risk_metrics = await self.get_portfolio_risk_metrics()
            performance_attribution = await self.calculate_performance_attribution()
            execution_quality = await self._calculate_execution_quality_metrics()
            stress_test_results = await self.calculate_stress_test_scenarios()
            concentration_analysis = await self.calculate_concentration_risk()
            advanced_var = await self.calculate_advanced_var()

            dashboard_data = {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "status": "active",
                "last_update": self._last_update.isoformat() if self._last_update else None,
                # Core metrics
                "portfolio": portfolio_metrics,
                "positions": position_analytics,
                "risk": risk_metrics,
                # Advanced analytics
                "performance_attribution": (
                    performance_attribution.dict() if performance_attribution else {}
                ),
                "execution_quality": execution_quality,
                "stress_testing": stress_test_results,
                "concentration_risk": concentration_analysis,
                "advanced_var": advanced_var,
                # Real-time alerts
                "active_alerts": [alert.dict() for alert in self._active_alerts.values()],
                "alert_summary": {
                    "total_alerts": len(self._active_alerts),
                    "critical_alerts": len(
                        [
                            a
                            for a in self._active_alerts.values()
                            if a.severity == AlertSeverity.CRITICAL
                        ]
                    ),
                    "high_alerts": len(
                        [
                            a
                            for a in self._active_alerts.values()
                            if a.severity == AlertSeverity.HIGH
                        ]
                    ),
                },
                # Market summary
                "market_summary": {
                    "total_symbols": len(self._positions),
                    "active_strategies": len(
                        set(p.strategy for p in self._positions.values() if hasattr(p, "strategy"))
                    ),
                    "total_trades_today": len(
                        [
                            t
                            for t in self._trades
                            if t.entry_time.date() == datetime.now(timezone.utc).date()
                        ]
                    ),
                },
                # Performance summary
                "performance_summary": {
                    "daily_pnl": (
                        float(portfolio_metrics.get("daily_pnl", 0)) if portfolio_metrics else 0
                    ),
                    "unrealized_pnl": (
                        float(portfolio_metrics.get("unrealized_pnl", 0))
                        if portfolio_metrics
                        else 0
                    ),
                    "realized_pnl": (
                        float(portfolio_metrics.get("realized_pnl", 0)) if portfolio_metrics else 0
                    ),
                    "total_return": (
                        float(portfolio_metrics.get("total_return", 0)) if portfolio_metrics else 0
                    ),
                },
            }

            # Cache the dashboard data if Redis is available with timeout
            if self._cache_enabled and self._cache_client:
                try:
                    await asyncio.wait_for(
                        self._cache_client.set(
                            "analytics:dashboard:latest",
                            json.dumps(dashboard_data, default=str),
                            expire=30,  # 30-second expiry for fresh data
                        ),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Dashboard cache operation timed out")
                except Exception as cache_error:
                    self.logger.error(f"Error caching dashboard data: {cache_error}")

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            return {"error": str(e), "timestamp": get_current_utc_timestamp().isoformat()}
