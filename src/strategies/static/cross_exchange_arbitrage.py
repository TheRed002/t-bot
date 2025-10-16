"""
Cross-exchange arbitrage strategy implementation.

This module implements cross-exchange arbitrage by detecting price differences
between the same asset on different exchanges and executing simultaneous trades
to capture the spread.

CRITICAL: This strategy requires ultra-low latency execution and careful
rate limit management across multiple exchanges.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.exceptions import ArbitrageError, ValidationError

# Logger is provided by BaseStrategy (via BaseComponent)
# From P-001 - Use existing types
from src.core.types import (
    MarketData,
    Position,
    Signal,
    SignalDirection,
    StrategyStatus,
    StrategyType,
)

# MANDATORY: Import from P-011 - NEVER recreate the base strategy
from src.strategies.base import BaseStrategy
from src.strategies.dependencies import StrategyServiceContainer
from src.utils.constants import GLOBAL_FEE_STRUCTURE, GLOBAL_MINIMUM_AMOUNTS, PRECISION_LEVELS
from src.utils.decimal_utils import round_to_precision

# From P-007A - Use decorators and validators
from src.utils.decorators import log_errors, time_execution
from src.utils.formatters import format_currency, format_percentage
from src.utils.validators import ValidationFramework

# From P-008+ - Use risk management

# From P-003+ - Use exchange interfaces


class CrossExchangeArbitrageStrategy(BaseStrategy):
    """
    Cross-exchange arbitrage strategy for detecting and executing price differences.

    This strategy monitors the same asset across multiple exchanges and executes
    simultaneous buy/sell orders when profitable spreads are detected.
    """

    def __init__(self, config: dict, services: "StrategyServiceContainer"):
        """Initialize cross-exchange arbitrage strategy.

        Args:
            config: Strategy configuration dictionary
            services: Service container for dependencies
        """
        super().__init__(config, services)
        # Note: name, version, status are set by BaseStrategy

        # Strategy-specific configuration
        self.min_profit_threshold = Decimal(
            str(config.get("min_profit_threshold", "0.001"))
        )  # 0.1%
        self.max_execution_time = config.get("max_execution_time", 500)  # milliseconds
        self.exchanges = config.get("exchanges", ["binance", "okx", "coinbase"])
        self.symbols = config.get("symbols", ["BTCUSDT", "ETHUSDT"])
        self.latency_threshold = config.get("latency_threshold", 100)  # milliseconds
        self.slippage_limit = Decimal(str(config.get("slippage_limit", "0.0005")))  # 0.05%

        # State tracking
        self.active_arbitrages: dict[str, dict] = {}
        self.exchange_prices: dict[str, dict[str, MarketData]] = {}
        self.last_opportunity_check = datetime.now(timezone.utc)

        self.logger.info(
            "Cross-exchange arbitrage strategy initialized",
            strategy=self.name,
            exchanges=self.exchanges,
            symbols=self.symbols,
            min_profit_threshold=self.min_profit_threshold,
        )

    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        return StrategyType.ARBITRAGE

    @property
    def name(self) -> str:
        """Get the strategy name."""
        # BaseStrategy sets self.name as an attribute, this property allows getting it
        return getattr(self, "_name_attr", "cross_exchange_arbitrage")

    @name.setter
    def name(self, value: str) -> None:
        """Set the strategy name."""
        self._name_attr = value

    @property
    def version(self) -> str:
        """Get the strategy version."""
        # BaseStrategy sets self.version as an attribute, this property allows getting it
        return getattr(self, "_version_attr", "1.0.0")

    @version.setter
    def version(self, value: str) -> None:
        """Set the strategy version."""
        self._version_attr = value

    @property
    def status(self) -> StrategyStatus:
        """Get the current strategy status."""
        # BaseStrategy sets self.status as an attribute, this property allows getting it
        return getattr(self, "_status_attr", StrategyStatus.STOPPED)

    @status.setter
    def status(self, value: StrategyStatus) -> None:
        """Set the strategy status."""
        self._status_attr = value

    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """
        Generate arbitrage signals based on cross-exchange price differences.

        Args:
            data: Market data from one exchange

        Returns:
            List of arbitrage signals
        """
        try:
            # Update price data for the exchange
            exchange_name = data.metadata.get("exchange", "unknown")
            if exchange_name not in self.exchange_prices:
                self.exchange_prices[exchange_name] = {}

            self.exchange_prices[exchange_name][data.symbol] = data

            # Check for arbitrage opportunities across all exchanges
            signals = await self._detect_arbitrage_opportunities(data.symbol)

            # Log cross-exchange arbitrage opportunities for monitoring
            if signals:
                self.logger.debug(
                    "Arbitrage signals generated",
                    strategy=self.name,
                    symbol=data.symbol,
                    signal_count=len(signals),
                    signals=[s.direction.value for s in signals],
                )

            return signals

        except Exception as e:
            self.logger.error(
                "Arbitrage signal generation failed",
                strategy=self.name,
                symbol=data.symbol,
                error=str(e),
            )
            return []  # Graceful degradation

    async def _detect_arbitrage_opportunities(self, symbol: str) -> list[Signal]:
        """
        Detect arbitrage opportunities for a given symbol across exchanges.

        Args:
            symbol: Trading symbol to check

        Returns:
            List of arbitrage signals
        """
        signals: list[Any] = []

        try:
            # Get all available prices for this symbol
            symbol_prices = {}
            for exchange in self.exchanges:
                if exchange in self.exchange_prices and symbol in self.exchange_prices[exchange]:
                    symbol_prices[exchange] = self.exchange_prices[exchange][symbol]

            if len(symbol_prices) < 2:
                return signals  # Need at least 2 exchanges

            # Find best bid and ask across exchanges
            best_bid_exchange = None
            best_bid_price = Decimal("0")
            best_ask_exchange = None
            best_ask_price = Decimal("inf")

            for exchange, market_data in symbol_prices.items():
                if market_data.bid and market_data.bid > best_bid_price:
                    best_bid_price = market_data.bid
                    best_bid_exchange = exchange

                if market_data.ask and market_data.ask < best_ask_price:
                    best_ask_price = market_data.ask
                    best_ask_exchange = exchange

            # Check if arbitrage opportunity exists
            if (
                best_bid_exchange
                and best_ask_exchange
                and best_bid_exchange != best_ask_exchange
                and best_bid_price > best_ask_price
            ):
                # Calculate potential profit
                spread = best_bid_price - best_ask_price
                spread_percentage = (spread / best_ask_price) * 100

                # Account for fees and slippage
                estimated_fees = self._calculate_total_fees(best_ask_price, best_bid_price)
                net_profit = spread - estimated_fees
                net_profit_percentage = (net_profit / best_ask_price) * 100

                # Check if profit meets threshold
                # Check if profit meets threshold - use Decimal comparison
                threshold_percentage = self.min_profit_threshold * Decimal("100")
                if net_profit_percentage >= threshold_percentage:
                    # Validate execution time constraints
                    if await self._validate_execution_timing(symbol):
                        # Create arbitrage signal
                        signal = Signal(
                            signal_id="test_signal_1",
                            strategy_id="test_strategy_1",
                            strategy_name="test_strategy",
                            symbol=symbol,
                            direction=SignalDirection.BUY,  # Buy on lower price exchange
                            # Scale strength with profit
                            strength=min(Decimal("0.9"), net_profit_percentage / Decimal("200")),
                            timestamp=datetime.now(timezone.utc),
                            source=self.name,
                            metadata={
                                "arbitrage_type": "cross_exchange",
                                "buy_exchange": best_ask_exchange,
                                "sell_exchange": best_bid_exchange,
                                "buy_price": str(best_ask_price),
                                "sell_price": str(best_bid_price),
                                "spread_percentage": str(spread_percentage),
                                "net_profit_percentage": str(net_profit_percentage),
                                "estimated_fees": str(estimated_fees),
                                "execution_timeout": self.max_execution_time,
                            },
                        )

                        signals.append(signal)

                        self.logger.info(
                            "Arbitrage opportunity detected",
                            strategy=self.name,
                            symbol=symbol,
                            buy_exchange=best_ask_exchange,
                            sell_exchange=best_bid_exchange,
                            spread_percentage=float(spread_percentage),
                            net_profit_percentage=float(net_profit_percentage),
                        )

        except Exception as e:
            self.logger.error(
                "Arbitrage opportunity detection failed",
                strategy=self.name,
                symbol=symbol,
                error=str(e),
            )

        return signals

    @log_errors
    def _calculate_total_fees(self, buy_price: Decimal, sell_price: Decimal) -> Decimal:
        """
        Calculate total fees for arbitrage execution using proper validation and formatting.

        Args:
            buy_price: Price to buy at
            sell_price: Price to sell at

        Returns:
            Total estimated fees

        Raises:
            ValidationError: If prices are invalid
            ArbitrageError: If fee calculation fails
        """
        try:
            # Validate input prices using utils
            ValidationFramework.validate_price(buy_price)
            ValidationFramework.validate_price(sell_price)

            # Get fee structure from constants and convert to Decimal
            maker_fee_rate = Decimal(str(GLOBAL_FEE_STRUCTURE.get("maker_fee", 0.001)))  # 0.1%
            taker_fee_rate = Decimal(str(GLOBAL_FEE_STRUCTURE.get("taker_fee", 0.001)))  # 0.1%

            # Calculate fees using proper rounding
            buy_fees = round_to_precision(buy_price * taker_fee_rate, PRECISION_LEVELS["fee"])
            sell_fees = round_to_precision(sell_price * taker_fee_rate, PRECISION_LEVELS["fee"])

            # Calculate slippage cost (as percentage of prices, not spread)
            buy_slippage = round_to_precision(
                buy_price * self.slippage_limit, PRECISION_LEVELS["price"]
            )
            sell_slippage = round_to_precision(
                sell_price * self.slippage_limit, PRECISION_LEVELS["price"]
            )
            slippage_cost = buy_slippage + sell_slippage

            # Calculate total fees
            total_fees = buy_fees + sell_fees + slippage_cost

            # Validate final result
            # total_fees is already a Decimal from calculations

            self.logger.debug(
                "Fee calculation completed",
                strategy=self.name,
                buy_price=format_currency(buy_price),
                sell_price=format_currency(sell_price),
                buy_fees=format_currency(buy_fees),
                sell_fees=format_currency(sell_fees),
                slippage_cost=format_currency(slippage_cost),
                total_fees=format_currency(total_fees),
            )

            return total_fees

        except Exception as e:
            self.logger.error(
                "Fee calculation failed",
                strategy=self.name,
                buy_price=float(buy_price),
                sell_price=float(sell_price),
                error=str(e),
            )
            raise ArbitrageError(f"Fee calculation failed: {e!s}")

    async def _validate_execution_timing(self, symbol: str) -> bool:
        """
        Validate that execution timing constraints are met.

        Args:
            symbol: Trading symbol

        Returns:
            True if timing is valid, False otherwise
        """
        try:
            # Check if we have recent price data
            current_time = datetime.now(timezone.utc)
            max_age = timedelta(milliseconds=self.latency_threshold)

            for exchange in self.exchanges:
                if exchange in self.exchange_prices and symbol in self.exchange_prices[exchange]:
                    price_data = self.exchange_prices[exchange][symbol]
                    if current_time - price_data.timestamp > max_age:
                        self.logger.warning(
                            "Price data too old for arbitrage",
                            strategy=self.name,
                            exchange=exchange,
                            symbol=symbol,
                            age_ms=(current_time - price_data.timestamp).total_seconds() * 1000,
                        )
                        return False

            # Check if we have too many active arbitrages
            active_count = len(
                [a for a in self.active_arbitrages.values() if a.get("symbol") == symbol]
            )

            max_arbitrages = self.config.parameters.get("max_open_arbitrages", 5)
            if active_count >= max_arbitrages:
                self.logger.warning(
                    "Too many active arbitrages",
                    strategy=self.name,
                    symbol=symbol,
                    active_count=active_count,
                    max_allowed=max_arbitrages,
                )
                return False

            return True

        except Exception as e:
            self.logger.error(
                "Execution timing validation failed",
                strategy=self.name,
                symbol=symbol,
                error=str(e),
            )
            return False

    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate arbitrage signal before execution.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Basic signal validation
            if not signal or signal.direction not in [SignalDirection.BUY, SignalDirection.SELL]:
                return False

            # Check confidence threshold
            if signal.strength < self.config.min_confidence:
                return False

            # Validate arbitrage-specific metadata
            metadata = signal.metadata
            required_fields = [
                "arbitrage_type",
                "buy_exchange",
                "sell_exchange",
                "buy_price",
                "sell_price",
                "net_profit_percentage",
            ]

            for field in required_fields:
                if field not in metadata:
                    self.logger.warning(
                        "Missing required metadata field",
                        strategy=self.name,
                        field=field,
                        signal_id=signal.timestamp,
                    )
                    return False

            # Validate profit threshold - use Decimal comparison
            net_profit = Decimal(str(metadata.get("net_profit_percentage", 0)))
            threshold_percentage = self.min_profit_threshold * Decimal("100")
            if net_profit < threshold_percentage:
                return False

            # Validate exchanges are different
            if metadata.get("buy_exchange") == metadata.get("sell_exchange"):
                return False

            return True

        except Exception as e:
            self.logger.error("Signal validation failed", strategy=self.name, error=str(e))
            return False

    @log_errors
    def get_position_size(self, signal: Signal) -> Decimal:
        """
        Calculate position size for arbitrage signal using risk management components.

        Args:
            signal: Arbitrage signal

        Returns:
            Position size in base currency

        Raises:
            ArbitrageError: If position size calculation fails
        """
        try:
            # Validate signal using utils
            if not signal:
                raise ArbitrageError("Invalid signal for position sizing")

            if signal.strength < 0.0 or signal.strength > 1.0:
                raise ValidationError(f"Invalid signal confidence: {signal.strength}")

            # Get configuration parameters
            total_capital = Decimal(str(self.config.parameters.get("total_capital", 10000)))
            risk_per_trade = self.config.parameters.get("risk_per_trade", 0.02)
            max_position_size = self.config.parameters.get("max_position_size", 0.1)

            # Calculate base position size using simple percentage method
            base_size = total_capital * Decimal(str(risk_per_trade))

            # Apply maximum position size limit
            max_size = total_capital * Decimal(str(max_position_size))
            if base_size > max_size:
                base_size = max_size

            # CRITICAL FIX: Ensure position size doesn't exceed 10% of total capital
            absolute_max = total_capital * Decimal("0.1")  # 10% hard limit
            if base_size > absolute_max:
                self.logger.warning(
                    "Position size exceeded absolute limit",
                    requested_size=base_size,
                    absolute_max=absolute_max,
                )
                base_size = absolute_max

            # Scale by arbitrage-specific factors
            metadata = signal.metadata
            profit_potential = Decimal(str(metadata.get("net_profit_percentage", 0))) / Decimal(
                "100"
            )
            profit_potential_pct = float(profit_potential * 100)
            if profit_potential_pct < 0.0 or profit_potential_pct > 1000.0:
                raise ValidationError(f"Invalid profit potential: {profit_potential_pct}%")

            # Apply arbitrage-specific adjustments
            arbitrage_multiplier = min(
                Decimal("2.0"), profit_potential * Decimal("10")
            )  # Scale with profit
            confidence_multiplier = Decimal(str(signal.strength))

            # Calculate final position size with proper validation
            position_size = round_to_precision(
                base_size * confidence_multiplier * arbitrage_multiplier,
                PRECISION_LEVELS["position"],
            )

            # Apply minimum position size from constants
            min_size = Decimal(str(GLOBAL_MINIMUM_AMOUNTS.get("position", 0.001)))
            if position_size < min_size:
                position_size = min_size

            # Validate final result
            ValidationFramework.validate_quantity(position_size)

            self.logger.debug(
                "Position size calculated",
                strategy=self.name,
                base_size=format_currency(base_size),
                profit_potential=format_percentage(profit_potential * 100),
                confidence=format_percentage(signal.strength * 100),
                final_size=format_currency(position_size),
            )

            return position_size

        except Exception as e:
            self.logger.error(
                "Position size calculation failed",
                strategy=self.name,
                signal_confidence=signal.strength if signal else None,
                error=str(e),
            )
            raise ArbitrageError(f"Position size calculation failed: {e!s}")

    async def should_exit(self, position: Position, data: MarketData) -> bool:
        """
        Determine if arbitrage position should be closed.

        Args:
            position: Current position
            data: Latest market data

        Returns:
            True if position should be closed, False otherwise
        """
        try:
            # Check if this is an arbitrage position
            if "arbitrage_type" not in position.metadata:
                return False  # Not an arbitrage position

            # Check execution timeout
            execution_timeout = position.metadata.get("execution_timeout", self.max_execution_time)
            position_age = (datetime.now(timezone.utc) - position.opened_at).total_seconds() * 1000

            if position_age > execution_timeout:
                self.logger.info(
                    "Arbitrage position timeout",
                    strategy=self.name,
                    symbol=position.symbol,
                    age_ms=position_age,
                    timeout_ms=execution_timeout,
                )
                return True

            # Check if arbitrage opportunity still exists
            metadata = position.metadata
            buy_exchange = metadata.get("buy_exchange")
            sell_exchange = metadata.get("sell_exchange")

            if buy_exchange and sell_exchange:
                # Check current spread
                current_spread = await self._get_current_spread(
                    position.symbol, buy_exchange, sell_exchange
                )

                if current_spread <= 0:
                    self.logger.info(
                        "Arbitrage opportunity closed",
                        strategy=self.name,
                        symbol=position.symbol,
                        current_spread=current_spread,
                    )
                    return True

            return False

        except Exception as e:
            self.logger.error(
                "Exit condition check failed",
                strategy=self.name,
                symbol=position.symbol,
                error=str(e),
            )
            return False

    async def _get_current_spread(
        self, symbol: str, buy_exchange: str, sell_exchange: str
    ) -> Decimal:
        """
        Get current spread between exchanges.

        Args:
            symbol: Trading symbol
            buy_exchange: Exchange to buy from
            sell_exchange: Exchange to sell to

        Returns:
            Current spread (positive if profitable)
        """
        try:
            buy_price = None
            sell_price = None

            # Get current prices
            if (
                buy_exchange in self.exchange_prices
                and symbol in self.exchange_prices[buy_exchange]
            ):
                buy_price = self.exchange_prices[buy_exchange][symbol].ask

            if (
                sell_exchange in self.exchange_prices
                and symbol in self.exchange_prices[sell_exchange]
            ):
                sell_price = self.exchange_prices[sell_exchange][symbol].bid

            if buy_price and sell_price:
                return sell_price - buy_price

            return Decimal("0")

        except Exception as e:
            self.logger.error(
                "Spread calculation failed", strategy=self.name, symbol=symbol, error=str(e)
            )
            return Decimal("0")

    async def post_trade_processing(self, trade_result: dict[str, Any]) -> None:
        """Process completed arbitrage trade."""
        try:
            await super().post_trade_processing(trade_result)
            # Add strategy-specific post-processing logic here
            self.logger.info(
                "Cross-exchange arbitrage trade completed",
                strategy=self.name,
                trade_result=trade_result
            )
        except Exception as e:
            self.logger.error(
                "Error in post-trade processing",
                strategy=self.name,
                error=str(e)
            )

    # Helper methods for accessing data through data service

    async def _process_trade_result(self, trade_result: dict[str, Any]) -> None:
        """
        Process completed arbitrage trade.

        Args:
            trade_result: Trade execution result
        """
        try:
            # Update metrics
            self.metrics.total_trades += 1

            # Calculate P&L
            if "pnl" in trade_result:
                self.metrics.total_pnl += Decimal(str(trade_result["pnl"]))

                if trade_result["pnl"] > 0:
                    self.metrics.winning_trades += 1
                else:
                    self.metrics.losing_trades += 1

            # Update win rate
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades

            # Log trade result
            self.logger.info(
                "Arbitrage trade completed",
                strategy=self.name,
                symbol=trade_result.get("symbol"),
                pnl=float(trade_result.get("pnl", 0)),
                execution_time_ms=trade_result.get("execution_time_ms", 0),
            )

        except Exception as e:
            self.logger.error("Post-trade processing failed", strategy=self.name, error=str(e))
