"""
Backtesting Engine - Core implementation for historical strategy testing.

This module provides the main backtesting engine that replays historical market data
and simulates strategy execution with realistic trading conditions.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.backtesting.utils import convert_market_records_to_dataframe
from src.core.exceptions import TradingBotError
from src.core.logging import get_logger
from src.core.types import (
    ExecutionAlgorithm,
    OrderRequest,
    OrderSide,
    OrderType,
    Signal,
    SignalDirection,
    TimeInForce,
)
from src.data.services.data_service import DataService
from src.data.types import DataRequest
from src.error_handling.decorators import with_circuit_breaker, with_error_context, with_retry
from src.execution import ExecutionEngine, OrderManager
from src.execution.types import ExecutionInstruction
from src.risk_management.base import BaseRiskManager
from src.strategies.interfaces import BaseStrategyInterface
from src.utils.decimal_utils import safe_decimal
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class BacktestConfig(BaseModel):
    """Configuration for backtesting engine."""

    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: Decimal = Field(
        default_factory=lambda: safe_decimal("10000"), description="Initial capital for backtesting"
    )
    symbols: list[str] = Field(..., description="List of symbols to backtest")
    exchange: str = Field(default="binance", description="Exchange to use for backtesting")
    timeframe: str = Field(default="1h", description="Timeframe for backtesting")
    commission: Decimal = Field(
        default_factory=lambda: safe_decimal("0.001"), description="Commission rate (0.1%)"
    )
    slippage: Decimal = Field(
        default_factory=lambda: safe_decimal("0.0005"), description="Slippage rate (0.05%)"
    )
    enable_shorting: bool = Field(default=False, description="Enable short selling")
    max_open_positions: int = Field(default=5, description="Maximum number of open positions")
    use_tick_data: bool = Field(
        default=False, description="Use tick data for more accurate simulation"
    )
    warm_up_period: int = Field(default=100, description="Number of candles for indicator warm-up")

    # Execution configuration
    execution_algorithm: ExecutionAlgorithm = Field(
        default=ExecutionAlgorithm.TWAP, description="Default execution algorithm"
    )
    execution_time_horizon_minutes: int = Field(
        default=5, description="Execution time horizon in minutes"
    )

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, v: datetime, info) -> datetime:
        """Validate that end date is after start date."""
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("End date must be after start date")
        return v

    @field_validator("commission", "slippage")
    @classmethod
    def validate_rates(cls, v: Decimal) -> Decimal:
        """Validate commission and slippage rates."""
        # Rates can be zero (no commission/slippage) but must be non-negative and <= 10%
        if v < 0:
            raise ValueError("Rate must be between 0 and 0.1 (10%)")
        if v > Decimal("0.1"):
            raise ValueError("Rate must be between 0 and 0.1 (10%)")
        return v


class BacktestResult(BaseModel):
    """Results from a backtest run."""

    # Performance metrics
    total_return: Decimal = Field(..., description="Total return percentage")
    annual_return: Decimal = Field(..., description="Annualized return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown: Decimal = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Win rate percentage")

    # Trade statistics
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    avg_win: Decimal = Field(..., description="Average winning trade amount")
    avg_loss: Decimal = Field(..., description="Average losing trade amount")
    profit_factor: float = Field(..., description="Profit factor")

    # Risk metrics
    volatility: float = Field(..., description="Strategy volatility")
    var_95: Decimal = Field(..., description="Value at Risk (95%)")
    cvar_95: Decimal = Field(..., description="Conditional Value at Risk (95%)")

    # Additional data
    equity_curve: list[dict[str, Any]] = Field(..., description="Equity curve over time")
    trades: list[dict[str, Any]] = Field(..., description="List of all trades")
    daily_returns: list[float] = Field(..., description="Daily return series")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BacktestEngine:
    """
    Main backtesting engine for strategy evaluation.

    This engine provides realistic market simulation with support for:
    - Historical data replay
    - Commission and slippage modeling
    - Risk management integration
    - Performance tracking
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy: BaseStrategyInterface,
        risk_manager: BaseRiskManager | None = None,
        data_service: DataService | None = None,
        execution_engine: ExecutionEngine | None = None,
        order_manager: OrderManager | None = None,
    ):
        """
        Initialize backtesting engine.

        Args:
            config: Backtest configuration
            strategy: Strategy to backtest
            risk_manager: Optional risk manager for position sizing
            data_service: Optional data service for market data access
            execution_engine: Optional execution engine for realistic order execution
            order_manager: Optional order manager for order lifecycle management
        """
        self.config = config
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.data_service = data_service
        self.execution_engine = execution_engine
        self.order_manager = order_manager

        # Internal state
        self._capital = config.initial_capital
        self._positions: dict[str, dict[str, Any]] = {}
        self._trades: list[dict[str, Any]] = []
        self._equity_curve: list[dict[str, Any]] = []
        self._current_time: datetime | None = None
        self._market_data: dict[str, pd.DataFrame] = {}

        logger.info(
            "BacktestEngine initialized",
            config=config.model_dump(),
            strategy=strategy.name,
            strategy_type=(
                strategy.strategy_type.value if hasattr(strategy, "strategy_type") else "unknown"
            ),
        )

    @time_execution
    @with_error_context(component="backtesting", operation="run_backtest")
    @with_retry(max_attempts=3)
    async def run(self) -> BacktestResult:
        """
        Run the backtest.

        Returns:
            BacktestResult containing performance metrics and trade data
        """
        logger.info("Starting backtest", start=self.config.start_date, end=self.config.end_date)

        # Load historical data
        await self._load_historical_data()

        # Initialize strategy
        await self._initialize_strategy()

        # Run simulation
        await self._run_simulation()

        # Calculate metrics
        result = await self._calculate_results()

        logger.info("Backtest completed", total_trades=len(self._trades))
        return result

    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    async def _load_historical_data(self) -> None:
        """Load historical market data for all symbols."""
        logger.info("Loading historical data", symbols=self.config.symbols)

        for symbol in self.config.symbols:
            if self.data_service:
                # Load using DataService
                data = await self._load_from_data_service(symbol)
            else:
                # Load from default source (placeholder)
                data = await self._load_default_data(symbol)

            self._market_data[symbol] = data
            logger.debug(f"Loaded data for {symbol}", rows=len(data))

    @with_error_context(component="data_loading", operation="load_symbol_data")
    async def _load_from_data_service(self, symbol: str) -> pd.DataFrame:
        """Load data using DataService."""
        if not self.data_service:
            raise TradingBotError("Data service not configured", "BACKTEST_005")

        # Create data request
        data_request = DataRequest(
            symbol=symbol,
            exchange=self.config.exchange,
            start_time=self.config.start_date,
            end_time=self.config.end_date,
            limit=None,
            cache_ttl=3600,
            use_cache=True,
        )

        # Get market data records
        records = await self.data_service.get_market_data(data_request)

        if not records:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame using shared utility
        return convert_market_records_to_dataframe(records)

    async def _load_default_data(self, symbol: str) -> pd.DataFrame:
        """Load default/sample data for testing."""
        # Generate synthetic data for testing
        import numpy as np

        dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=self.config.timeframe,
        )

        # Generate random walk price data
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate valid OHLC data
        open_prices = prices * (1 + np.random.normal(0, 0.001, len(dates)))
        close_prices = prices

        # Ensure high and low are valid relative to open and close
        high_offsets = np.abs(np.random.normal(0, 0.005, len(dates)))
        low_offsets = np.abs(np.random.normal(0, 0.005, len(dates)))

        # High is the maximum of open, close, and a price above the base price
        high_prices = np.maximum(np.maximum(open_prices, close_prices), prices * (1 + high_offsets))

        # Low is the minimum of open, close, and a price below the base price
        low_prices = np.minimum(np.minimum(open_prices, close_prices), prices * (1 - low_offsets))

        df = pd.DataFrame(
            {
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": np.random.uniform(1000, 10000, len(dates)),
            },
            index=dates,
        )

        return df

    async def _initialize_strategy(self) -> None:
        """Initialize the strategy with warm-up data."""
        logger.info("Initializing strategy", warm_up_period=self.config.warm_up_period)

        # Start the strategy if it has a start method
        if hasattr(self.strategy, "start") and callable(self.strategy.start):
            try:
                await self.strategy.start()
            except Exception as e:
                logger.warning(
                    f"Strategy {self.strategy.name} start() failed: {e}",
                    strategy=self.strategy.name,
                    error=str(e),
                )
        else:
            logger.debug(f"Strategy {self.strategy.name} has no start() method")

        # Prepare strategy for backtesting if it supports the interface
        if hasattr(self.strategy, "prepare_for_backtest"):
            try:
                await self.strategy.prepare_for_backtest(
                    {
                        "start_date": self.config.start_date,
                        "end_date": self.config.end_date,
                        "symbols": self.config.symbols,
                        "initial_capital": float(self.config.initial_capital),
                        "timeframe": self.config.timeframe,
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Strategy {self.strategy.name} prepare_for_backtest() failed: {e}",
                    strategy=self.strategy.name,
                    error=str(e),
                )

        # Strategy warm-up: process historical data without generating trades
        for symbol, data in self._market_data.items():
            if len(data) > self.config.warm_up_period:
                # Process warm-up data for indicator initialization
                # This allows the strategy to build up its internal state
                logger.debug(
                    f"Warming up strategy with {self.config.warm_up_period} periods for {symbol}"
                )

    async def _run_simulation(self) -> None:
        """Run the main simulation loop."""
        logger.info("Running simulation")

        # Get all timestamps
        all_timestamps = set()
        for data in self._market_data.values():
            all_timestamps.update(data.index[self.config.warm_up_period :])

        timestamps = sorted(all_timestamps)

        # Process each timestamp
        for timestamp in timestamps:
            self._current_time = timestamp

            # Update market data
            current_data = self._get_current_market_data(timestamp)

            # Generate signals
            signals = await self._generate_signals(current_data)

            # Execute trades
            await self._execute_trades(signals, current_data)

            # Update positions
            self._update_positions(current_data)

            # Record equity
            self._record_equity()

            # Check risk limits
            if self.risk_manager:
                await self._check_risk_limits()

    def _get_current_market_data(self, timestamp: datetime) -> dict[str, pd.Series]:
        """Get market data for current timestamp."""
        current_data = {}
        for symbol, data in self._market_data.items():
            if timestamp in data.index:
                current_data[symbol] = data.loc[timestamp]
        return current_data

    async def _generate_signals(self, market_data: dict[str, pd.Series]) -> dict[str, Signal]:
        """Generate trading signals from strategy."""
        from src.core.types import MarketData as CoreMarketData

        all_signals = {}

        for symbol, data in market_data.items():
            # Convert pandas Series to MarketData object for strategy
            if self._current_time is None:
                continue
            market_data_obj = CoreMarketData(
                symbol=symbol,
                timestamp=self._current_time,
                close=safe_decimal(data["close"]),
                volume=safe_decimal(data["volume"]),
                high=safe_decimal(data["high"]),
                low=safe_decimal(data["low"]),
                open=safe_decimal(data["open"]),
                exchange=self.config.exchange,
            )

            # Generate signals using the strategy's method with error handling
            try:
                signals = await self.strategy.generate_signals(market_data_obj)
            except Exception as e:
                logger.error(
                    "Strategy signal generation failed",
                    error=str(e),
                    strategy=self.strategy.name,
                    symbol=symbol,
                )
                signals = []  # Graceful degradation

            # Store the first signal for this symbol if any
            if signals:
                all_signals[symbol] = signals[0]  # Take the first signal

        return all_signals

    async def _execute_trades(
        self,
        signals: dict[str, Signal],
        market_data: dict[str, pd.Series],
    ) -> None:
        """Execute trades based on signals."""
        for symbol, signal in signals.items():
            if symbol not in market_data:
                continue

            price = float(market_data[symbol]["close"])

            # Use ExecutionEngine if available, otherwise use simple execution
            if self.execution_engine and self.order_manager:
                await self._execute_with_engine(symbol, signal, market_data[symbol])
            else:
                # Fallback to simple execution
                # Apply slippage based on signal direction
                if signal.direction == SignalDirection.BUY:
                    execution_price = price * (1 + float(self.config.slippage))
                    await self._open_position(symbol, execution_price, signal.direction)
                elif signal.direction == SignalDirection.SELL:
                    # Check if we have a position to close
                    if symbol in self._positions:
                        execution_price = price * (1 - float(self.config.slippage))
                        await self._close_position(symbol, execution_price)
                    elif self.config.enable_shorting:
                        # Open short position if shorting is enabled
                        execution_price = price * (1 - float(self.config.slippage))
                        await self._open_position(symbol, execution_price, signal.direction)

    @with_error_context(component="backtesting", operation="execute_with_engine")
    async def _execute_with_engine(
        self,
        symbol: str,
        signal: Signal,
        market_data_row: pd.Series,
    ) -> None:
        """Execute trades using the ExecutionEngine for realistic simulation."""
        from src.core.types import MarketData as CoreMarketData

        # Convert market data to MarketData object
        if self._current_time is None:
            return
        market_data = CoreMarketData(
            symbol=symbol,
            timestamp=self._current_time,
            close=safe_decimal(market_data_row["close"]),
            volume=safe_decimal(market_data_row["volume"]),
            high=safe_decimal(market_data_row["high"]),
            low=safe_decimal(market_data_row["low"]),
            open=safe_decimal(market_data_row["open"]),
            exchange=self.config.exchange,
        )

        # Determine order side based on signal and current position
        order_side = OrderSide.BUY if signal.direction == SignalDirection.BUY else OrderSide.SELL

        # Calculate position size
        if self.risk_manager:
            position_size = await self.risk_manager.calculate_position_size(
                signal, self._capital, market_data.close
            )
        else:
            # Default position sizing
            position_size = self._capital / safe_decimal(self.config.max_open_positions)

        # Create order request
        order_request = OrderRequest(
            symbol=symbol,
            side=order_side,
            order_type=OrderType.MARKET,  # Use market orders for backtesting
            quantity=position_size / market_data.close,  # Convert value to quantity
            time_in_force=TimeInForce.GTC,
            metadata={
                "signal_strength": signal.strength,
                "strategy": self.strategy.name,
                "backtest": True,
            },
        )

        # Create execution instruction
        execution_instruction = ExecutionInstruction(
            order=order_request,
            algorithm=self.config.execution_algorithm,
            strategy_name=self.strategy.name,
            time_horizon_minutes=self.config.execution_time_horizon_minutes,
            max_slippage_bps=safe_decimal(
                float(self.config.slippage) * 10000
            ),  # Convert to basis points
        )

        try:
            # Execute through ExecutionEngine
            if self.execution_engine is None:
                raise ValueError("ExecutionEngine is None")
            result = await self.execution_engine.execute_order(
                instruction=execution_instruction,
                market_data=market_data,
                strategy_name=self.strategy.name,
            )

            # Validate execution result structure
            if not hasattr(result, "status"):
                logger.error("Execution result missing status field", result=result)
                raise ValueError("Invalid execution result: missing status")

            # Process execution result
            if result.status == "completed":
                # Validate required fields for completed orders
                required_fields = ["average_price", "total_quantity", "total_cost"]
                for field in required_fields:
                    if not hasattr(result, field):
                        logger.error(f"Execution result missing {field}", result=result)
                        raise ValueError(f"Invalid execution result: missing {field}")

                # Update position tracking
                if order_side == OrderSide.BUY:
                    self._positions[symbol] = {
                        "entry_time": self._current_time,
                        "entry_price": float(result.average_price),
                        "size": float(result.total_quantity),
                        "side": order_side,
                    }
                else:
                    # Close position
                    if symbol in self._positions:
                        position = self._positions[symbol]
                        # Calculate P&L
                        if position["side"] == OrderSide.BUY:
                            pnl = (
                                float(result.average_price) - position["entry_price"]
                            ) * position["size"]
                        else:
                            pnl = (
                                position["entry_price"] - float(result.average_price)
                            ) * position["size"]

                        # Record trade
                        self._trades.append(
                            {
                                "symbol": symbol,
                                "entry_time": position["entry_time"],
                                "exit_time": self._current_time,
                                "entry_price": position["entry_price"],
                                "exit_price": float(result.average_price),
                                "size": position["size"],
                                "pnl": pnl - float(result.total_cost),  # Include execution costs
                                "side": position["side"].value,
                                "execution_cost": float(result.total_cost),
                            }
                        )

                        # Remove position
                        del self._positions[symbol]

                # Update capital with execution costs
                self._capital -= safe_decimal(result.total_cost)

        except ValueError as e:
            # Validation errors should not fall back
            logger.error(
                "Execution validation failed",
                error=str(e),
                symbol=symbol,
                signal=signal,
            )
            raise  # Re-raise validation errors
        except AttributeError as e:
            # Missing attributes in result
            logger.error(
                "Execution result structure error",
                error=str(e),
                symbol=symbol,
                signal=signal,
            )
            # Fall back to simple execution
            price = float(market_data_row["close"])
            if signal.direction == SignalDirection.BUY:
                execution_price = price * (1 + float(self.config.slippage))
                await self._open_position(symbol, execution_price, signal.direction)
            elif signal.direction == SignalDirection.SELL and symbol in self._positions:
                execution_price = price * (1 - float(self.config.slippage))
                await self._close_position(symbol, execution_price)
        except Exception as e:
            logger.error(
                "Execution engine failed in backtest",
                error=str(e),
                symbol=symbol,
                signal=signal,
                error_type=type(e).__name__,
            )
            # Fall back to simple execution
            price = float(market_data_row["close"])
            if signal.direction == SignalDirection.BUY:
                execution_price = price * (1 + float(self.config.slippage))
                await self._open_position(symbol, execution_price, signal.direction)
            elif signal.direction == SignalDirection.SELL and symbol in self._positions:
                execution_price = price * (1 - float(self.config.slippage))
                await self._close_position(symbol, execution_price)

    async def _open_position(self, symbol: str, price: float, signal: SignalDirection) -> None:
        """Open a new position."""
        if symbol in self._positions:
            return  # Position already exists

        if len(self._positions) >= self.config.max_open_positions:
            return  # Max positions reached

        # Calculate position size
        if self.risk_manager:
            from src.core.types import Signal as CoreSignal

            # Create a signal for position sizing
            if self._current_time is None:
                return
            sizing_signal = CoreSignal(
                symbol=symbol,
                direction=signal,
                strength=0.8,
                timestamp=self._current_time,
                source="backtest",
            )
            position_size = await self.risk_manager.calculate_position_size(
                sizing_signal, self._capital, safe_decimal(price)
            )
        else:
            # Default position sizing (equal weight)
            position_size = self._capital / safe_decimal(self.config.max_open_positions)

        # Calculate commission
        commission = position_size * self.config.commission

        # Create position
        self._positions[symbol] = {
            "entry_time": self._current_time,
            "entry_price": price,
            "size": float(position_size),
            "side": OrderSide.BUY if signal == SignalDirection.BUY else OrderSide.SELL,
        }

        # Update capital (deduct position size AND commission)
        self._capital -= position_size + commission

        logger.debug(
            "Opened position",
            symbol=symbol,
            price=price,
            size=position_size,
        )

    async def _close_position(self, symbol: str, price: float) -> None:
        """Close an existing position."""
        if symbol not in self._positions:
            return

        position = self._positions[symbol]

        # Calculate P&L
        if position["side"] == OrderSide.BUY:
            # For long: profit = (exit_price - entry_price) * size
            pnl = (price - position["entry_price"]) * position["size"]
        else:
            # For short: profit = (entry_price - exit_price) * size
            pnl = (position["entry_price"] - price) * position["size"]

        # Apply commission
        commission = safe_decimal(position["size"]) * self.config.commission
        pnl_after_commission = safe_decimal(pnl) - commission

        # Update capital
        self._capital += safe_decimal(position["size"]) + pnl_after_commission

        # Record trade
        self._trades.append(
            {
                "symbol": symbol,
                "entry_time": position["entry_time"],
                "exit_time": self._current_time,
                "entry_price": position["entry_price"],
                "exit_price": price,
                "size": position["size"],
                "pnl": float(pnl_after_commission),
                "side": position["side"].value,
            }
        )

        # Remove position
        del self._positions[symbol]

        logger.debug(
            "Closed position",
            symbol=symbol,
            price=price,
            pnl=pnl_after_commission,
        )

    def _update_positions(self, market_data: dict[str, pd.Series]) -> None:
        """Update position values with current prices."""
        for symbol, position in self._positions.items():
            if symbol in market_data:
                current_price = float(market_data[symbol]["close"])
                position["current_price"] = current_price

                # Calculate unrealized P&L
                if position["side"] == OrderSide.BUY:
                    position["unrealized_pnl"] = (
                        (current_price - position["entry_price"])
                        * position["size"]
                        / position["entry_price"]
                    )
                else:
                    position["unrealized_pnl"] = (
                        (position["entry_price"] - current_price)
                        * position["size"]
                        / position["entry_price"]
                    )

    def _record_equity(self) -> None:
        """Record current equity value."""
        # Calculate total equity
        total_equity = float(self._capital)

        for position in self._positions.values():
            total_equity += position["size"] + position.get("unrealized_pnl", 0)

        self._equity_curve.append({"timestamp": self._current_time, "equity": total_equity})

    async def _check_risk_limits(self) -> None:
        """Check and enforce risk limits."""
        # Check drawdown limits (basic check even without risk manager)
        if self._equity_curve:
            peak = max(e["equity"] for e in self._equity_curve)
            current = self._equity_curve[-1]["equity"]
            drawdown = (peak - current) / peak

            if drawdown > 0.2:  # 20% max drawdown
                logger.warning("Max drawdown exceeded, closing all positions")
                await self._close_all_positions()

        # Additional risk checks if risk manager is available
        if self.risk_manager:
            # Could add more sophisticated risk checks here
            pass

    async def _close_all_positions(self) -> None:
        """Close all open positions."""
        for symbol in list(self._positions.keys()):
            if symbol in self._market_data:
                current_data = self._market_data[symbol].loc[self._current_time]
                price = float(current_data["close"])
                await self._close_position(symbol, price)

    async def _calculate_results(self) -> BacktestResult:
        """Calculate final backtest results."""
        logger.info("Calculating backtest results")

        # Close any remaining positions
        await self._close_all_positions()

        # Calculate returns
        initial_equity = float(self.config.initial_capital)
        final_equity = self._equity_curve[-1]["equity"] if self._equity_curve else initial_equity
        total_return = safe_decimal((final_equity - initial_equity) / initial_equity * 100)

        # Calculate daily returns
        equity_df = pd.DataFrame(self._equity_curve)
        if not equity_df.empty:
            equity_df.set_index("timestamp", inplace=True)
            daily_equity = equity_df.resample("D").last()
            daily_returns = daily_equity["equity"].pct_change().dropna().tolist()
        else:
            daily_returns = []

        # Calculate trade statistics
        winning_trades = [t for t in self._trades if t["pnl"] > 0]
        losing_trades = [t for t in self._trades if t["pnl"] <= 0]

        # Calculate metrics
        from .metrics import MetricsCalculator

        calculator = MetricsCalculator()
        metrics = calculator.calculate_all(
            equity_curve=self._equity_curve,
            trades=self._trades,
            daily_returns=daily_returns,
            initial_capital=float(self.config.initial_capital),
        )

        # Get strategy-specific backtest metrics if available
        strategy_metrics = {}
        if hasattr(self.strategy, "get_backtest_metrics"):
            try:
                strategy_metrics = await self.strategy.get_backtest_metrics()
            except Exception as e:
                logger.error(
                    "Failed to get strategy backtest metrics",
                    error=str(e),
                    strategy=self.strategy.name,
                )

        return BacktestResult(
            total_return=total_return,
            annual_return=metrics.get("annual_return", safe_decimal("0")),
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            sortino_ratio=metrics.get("sortino_ratio", 0.0),
            max_drawdown=metrics.get("max_drawdown", safe_decimal("0")),
            win_rate=metrics.get("win_rate", 0.0),
            total_trades=len(self._trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=metrics.get("avg_win", safe_decimal("0")),
            avg_loss=metrics.get("avg_loss", safe_decimal("0")),
            profit_factor=metrics.get("profit_factor", 0.0),
            volatility=metrics.get("volatility", 0.0),
            var_95=metrics.get("var_95", safe_decimal("0")),
            cvar_95=metrics.get("cvar_95", safe_decimal("0")),
            equity_curve=self._equity_curve,
            trades=self._trades,
            daily_returns=daily_returns,
            metadata={
                "config": self.config.model_dump(),
                "strategy": self.strategy.name,
                "strategy_type": (
                    self.strategy.strategy_type.value
                    if hasattr(self.strategy, "strategy_type")
                    else "unknown"
                ),
                "backtest_duration": str(self.config.end_date - self.config.start_date),
                "strategy_metrics": strategy_metrics,
            },
        )
