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

from src.core.exceptions import BacktestError
from src.core.logging import get_logger
from src.core.types import OrderSide, SignalDirection
from src.database.manager import DatabaseManager
from src.error_handling.decorators import with_retry
from src.risk_management import RiskManager
from src.strategies.base import BaseStrategy
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class BacktestConfig(BaseModel):
    """Configuration for backtesting engine."""

    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: Decimal = Field(
        default=Decimal("10000"), description="Initial capital for backtesting"
    )
    symbols: list[str] = Field(..., description="List of symbols to backtest")
    timeframe: str = Field(default="1h", description="Timeframe for backtesting")
    commission: Decimal = Field(default=Decimal("0.001"), description="Commission rate (0.1%)")
    slippage: Decimal = Field(default=Decimal("0.0005"), description="Slippage rate (0.05%)")
    enable_shorting: bool = Field(default=False, description="Enable short selling")
    max_open_positions: int = Field(default=5, description="Maximum number of open positions")
    use_tick_data: bool = Field(
        default=False, description="Use tick data for more accurate simulation"
    )
    warm_up_period: int = Field(default=100, description="Number of candles for indicator warm-up")

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
        if v < 0 or v > Decimal("0.1"):
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
        strategy: BaseStrategy,
        risk_manager: RiskManager | None = None,
        db_manager: DatabaseManager | None = None,
    ):
        """
        Initialize backtesting engine.

        Args:
            config: Backtest configuration
            strategy: Strategy to backtest
            risk_manager: Optional risk manager for position sizing
            db_manager: Optional database manager for data access
        """
        self.config = config
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.db_manager = db_manager

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
            strategy=strategy.__class__.__name__,
        )

    @time_execution
    @with_retry(max_attempts=3)
    async def run(self) -> BacktestResult:
        """
        Run the backtest.

        Returns:
            BacktestResult containing performance metrics and trade data
        """
        try:
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

        except Exception as e:
            logger.error("Backtest failed", error=str(e))
            raise BacktestError(f"Backtest failed: {e!s}")

    async def _load_historical_data(self) -> None:
        """Load historical market data for all symbols."""
        logger.info("Loading historical data", symbols=self.config.symbols)

        for symbol in self.config.symbols:
            if self.db_manager:
                # Load from database
                data = await self._load_from_database(symbol)
            else:
                # Load from default source (placeholder)
                data = await self._load_default_data(symbol)

            self._market_data[symbol] = data
            logger.debug(f"Loaded data for {symbol}", rows=len(data))

    async def _load_from_database(self, symbol: str) -> pd.DataFrame:
        """Load data from database."""
        if not self.db_manager:
            raise BacktestError("Database manager not configured")

        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp
        """

        rows = await self.db_manager.fetch_all(
            query, symbol, self.config.start_date, self.config.end_date
        )

        df = pd.DataFrame(rows)
        df.set_index("timestamp", inplace=True)
        return df

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

        df = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.001, len(dates))),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                "close": prices,
                "volume": np.random.uniform(1000, 10000, len(dates)),
            },
            index=dates,
        )

        return df

    async def _initialize_strategy(self) -> None:
        """Initialize the strategy with warm-up data."""
        logger.info("Initializing strategy", warm_up_period=self.config.warm_up_period)

        # Provide warm-up data to strategy
        for symbol, data in self._market_data.items():
            if len(data) > self.config.warm_up_period:
                warm_up_data = data.iloc[: self.config.warm_up_period]
                await self.strategy.initialize(symbol, warm_up_data)

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

    async def _generate_signals(
        self, market_data: dict[str, pd.Series]
    ) -> dict[str, SignalDirection]:
        """Generate trading signals from strategy."""
        signals = {}

        for symbol, _data in market_data.items():
            # Get historical data up to current point
            hist_data = self._market_data[symbol].loc[: self._current_time]

            # Generate signal
            signal = await self.strategy.generate_signal(symbol, hist_data)
            signals[symbol] = signal

        return signals

    async def _execute_trades(
        self,
        signals: dict[str, SignalDirection],
        market_data: dict[str, pd.Series],
    ) -> None:
        """Execute trades based on signals."""
        for symbol, signal in signals.items():
            if symbol not in market_data:
                continue

            price = float(market_data[symbol]["close"])

            # Apply slippage
            if signal == SignalDirection.BUY:
                execution_price = price * (1 + float(self.config.slippage))
                await self._open_position(symbol, execution_price, signal)
            elif signal == SignalDirection.SELL:
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
            position_size = await self.risk_manager.calculate_position_size(
                symbol, Decimal(str(price)), self._capital
            )
        else:
            # Default position sizing (equal weight)
            position_size = self._capital / Decimal(self.config.max_open_positions)

        # Apply commission
        commission = position_size * self.config.commission
        position_size -= commission

        # Create position
        self._positions[symbol] = {
            "entry_time": self._current_time,
            "entry_price": price,
            "size": float(position_size),
            "side": OrderSide.BUY if signal == SignalDirection.BUY else OrderSide.SELL,
        }

        # Update capital
        self._capital -= Decimal(str(position_size))

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
            pnl = (price - position["entry_price"]) * position["size"] / position["entry_price"]
        else:
            pnl = (position["entry_price"] - price) * position["size"] / position["entry_price"]

        # Apply commission
        commission = Decimal(str(position["size"])) * self.config.commission
        pnl_after_commission = Decimal(str(pnl)) - commission

        # Update capital
        self._capital += Decimal(str(position["size"])) + pnl_after_commission

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
        if not self.risk_manager:
            return

        # Check drawdown limits
        if self._equity_curve:
            peak = max(e["equity"] for e in self._equity_curve)
            current = self._equity_curve[-1]["equity"]
            drawdown = (peak - current) / peak

            if drawdown > 0.2:  # 20% max drawdown
                logger.warning("Max drawdown exceeded, closing all positions")
                await self._close_all_positions()

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
        total_return = Decimal(str((final_equity - initial_equity) / initial_equity * 100))

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

        return BacktestResult(
            total_return=total_return,
            annual_return=metrics.get("annual_return", Decimal("0")),
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            sortino_ratio=metrics.get("sortino_ratio", 0.0),
            max_drawdown=metrics.get("max_drawdown", Decimal("0")),
            win_rate=metrics.get("win_rate", 0.0),
            total_trades=len(self._trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=metrics.get("avg_win", Decimal("0")),
            avg_loss=metrics.get("avg_loss", Decimal("0")),
            profit_factor=metrics.get("profit_factor", 0.0),
            volatility=metrics.get("volatility", 0.0),
            var_95=metrics.get("var_95", Decimal("0")),
            cvar_95=metrics.get("cvar_95", Decimal("0")),
            equity_curve=self._equity_curve,
            trades=self._trades,
            daily_returns=daily_returns,
            metadata={
                "config": self.config.model_dump(),
                "strategy": self.strategy.__class__.__name__,
                "backtest_duration": str(self.config.end_date - self.config.start_date),
            },
        )
