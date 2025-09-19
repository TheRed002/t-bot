"""
Arbitrage opportunity scanner implementation.

This module implements a comprehensive arbitrage opportunity scanner that
continuously monitors for cross-exchange and triangular arbitrage opportunities
across all supported exchanges and trading pairs.

CRITICAL: This scanner requires real-time data feeds and ultra-low latency
processing to detect opportunities before they disappear.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.exceptions import ArbitrageError, ValidationError

# Logger is provided by BaseStrategy (via BaseComponent)
# From P-001 - Use existing types
from src.core.types import MarketData, Position, Signal, SignalDirection, StrategyType

# MANDATORY: Import from P-011 - NEVER recreate the base strategy
from src.strategies.base import BaseStrategy
from src.strategies.dependencies import StrategyServiceContainer
from src.utils.arbitrage_helpers import FeeCalculator, OpportunityAnalyzer

# From P-007A - Use decorators and validators
from src.utils.decorators import log_errors, time_execution
from src.utils.strategy_commons import StrategyCommons

# From P-008+ - Use risk management

# From P-003+ - Use exchange interfaces


class ArbitrageOpportunity(BaseStrategy):
    """
    Arbitrage opportunity scanner for detecting and prioritizing arbitrage opportunities.

    This scanner monitors all supported exchanges and trading pairs to detect
    both cross-exchange and triangular arbitrage opportunities, prioritizing
    them by profit potential and execution feasibility.
    """

    def __init__(self, config: dict, services: "StrategyServiceContainer"):
        """Initialize arbitrage opportunity scanner.

        Args:
            config: Strategy configuration dictionary
            services: Service container for dependencies
        """
        super().__init__(config, services)
        # name and strategy_type are provided by the base class and config

        # Scanner-specific configuration
        self.scan_interval = config.get("scan_interval", 100)  # milliseconds
        self.max_execution_time = config.get("max_execution_time", 500)  # milliseconds
        self.min_profit_threshold = Decimal(
            str(config.get("min_profit_threshold", "0.001"))
        )  # 0.1%
        self.max_opportunities = config.get("max_opportunities", 10)
        self.exchanges = config.get("exchanges", ["binance", "okx", "coinbase"])
        self.symbols = config.get("symbols", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"])
        self.triangular_paths = config.get(
            "triangular_paths", [["BTCUSDT", "ETHBTC", "ETHUSDT"], ["BTCUSDT", "BNBBTC", "BNBUSDT"]]
        )
        self.min_confidence = config.get("min_confidence", 0.5)

        # Initialize strategy commons
        self.commons = StrategyCommons(self.name, {"max_history_length": 100})

        # State tracking
        self.active_opportunities: dict[str, dict] = {}
        self.exchange_prices: dict[str, dict[str, MarketData]] = {}
        self.pair_prices: dict[str, MarketData] = {}  # Add missing attribute for tests
        self.opportunity_history: list[dict] = []
        self.last_scan_time = datetime.now(timezone.utc)

        # Performance tracking
        self.scan_count = 0
        self.opportunities_found = 0
        self.execution_success_rate = 0.0

        self.logger.info(
            "Arbitrage opportunity scanner initialized",
            strategy=self.name,
            exchanges=self.exchanges,
            symbols=self.symbols,
            scan_interval=self.scan_interval,
        )

    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        return StrategyType.ARBITRAGE

    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """
        Generate arbitrage opportunity signals based on comprehensive market scanning.

        Args:
            data: Market data from one exchange

        Returns:
            List of arbitrage opportunity signals
        """
        try:
            # Update price data for the exchange
            exchange_name = data.metadata.get("exchange", "unknown")
            if exchange_name not in self.exchange_prices:
                self.exchange_prices[exchange_name] = {}

            self.exchange_prices[exchange_name][data.symbol] = data

            # Perform comprehensive arbitrage scan
            signals = await self._scan_arbitrage_opportunities()

            # Log arbitrage opportunities for monitoring
            if signals:
                self.logger.info(
                    "Arbitrage opportunities detected",
                    strategy=self.name,
                    opportunity_count=len(signals),
                    symbols=[s.symbol for s in signals],
                )

            return signals

        except Exception as e:
            self.logger.error(
                "Arbitrage opportunity scanning failed",
                strategy=self.name,
                symbol=data.symbol,
                error=str(e),
            )
            return []  # Graceful degradation

    async def _scan_arbitrage_opportunities(self) -> list[Signal]:
        """
        Perform comprehensive arbitrage opportunity scanning.

        Returns:
            List of arbitrage opportunity signals
        """
        signals = []

        try:
            # Scan cross-exchange opportunities
            cross_signals = await self._scan_cross_exchange_opportunities()
            signals.extend(cross_signals)

            # Scan triangular opportunities
            triangular_signals = await self._scan_triangular_opportunities()
            signals.extend(triangular_signals)

            # Prioritize and limit opportunities
            prioritized_signals = OpportunityAnalyzer.prioritize_opportunities(
                signals, self.max_opportunities
            )

            # Update scanner metrics
            self.scan_count += 1
            self.opportunities_found += len(prioritized_signals)
            self.last_scan_time = datetime.now(timezone.utc)

            return prioritized_signals

        except Exception as e:
            self.logger.error(
                "Arbitrage opportunity scanning failed", strategy=self.name, error=str(e)
            )
            return []

    async def _scan_cross_exchange_opportunities(self) -> list[Signal]:
        """
        Scan for cross-exchange arbitrage opportunities.

        Returns:
            List of cross-exchange arbitrage signals
        """
        signals = []

        try:
            for symbol in self.symbols:
                # Get all available prices for this symbol
                symbol_prices = {}
                for exchange in self.exchanges:
                    if (
                        exchange in self.exchange_prices
                        and symbol in self.exchange_prices[exchange]
                    ):
                        symbol_prices[exchange] = self.exchange_prices[exchange][symbol]

                if len(symbol_prices) < 2:
                    continue  # Need at least 2 exchanges

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
                    estimated_fees = FeeCalculator.calculate_cross_exchange_fees(
                        best_ask_price, best_bid_price
                    )
                    net_profit = spread - estimated_fees
                    net_profit_percentage = (net_profit / best_ask_price) * 100

                    # Check if profit meets threshold
                    if net_profit_percentage >= float(self.min_profit_threshold * 100):
                        # Create cross-exchange arbitrage signal
                        signal = Signal(
                            symbol=symbol,
                            direction=SignalDirection.BUY,
                            strength=min(0.9, net_profit_percentage / 2),
                            timestamp=datetime.now(timezone.utc),
                            source=self.name,
                            metadata={
                                "arbitrage_type": "cross_exchange",
                                "buy_exchange": best_ask_exchange,
                                "sell_exchange": best_bid_exchange,
                                "buy_price": float(best_ask_price),
                                "sell_price": float(best_bid_price),
                                "spread_percentage": float(spread_percentage),
                                "net_profit_percentage": str(net_profit_percentage),
                                "estimated_fees": float(estimated_fees),
                                "opportunity_priority": OpportunityAnalyzer.calculate_priority(
                                    net_profit_percentage, "cross_exchange"
                                ),
                            },
                        )

                        signals.append(signal)

        except Exception as e:
            self.logger.error(
                "Cross-exchange opportunity scanning failed", strategy=self.name, error=str(e)
            )

        return signals

    async def _scan_triangular_opportunities(self) -> list[Signal]:
        """
        Scan for triangular arbitrage opportunities.

        Returns:
            List of triangular arbitrage signals
        """
        signals = []

        try:
            for path in self.triangular_paths:
                # Ensure we have prices for all pairs in the path from any exchange
                available_prices = {}
                for pair in path:
                    for exchange in self.exchanges:
                        if (
                            exchange in self.exchange_prices
                            and pair in self.exchange_prices[exchange]
                        ):
                            available_prices[pair] = self.exchange_prices[exchange][pair]
                            break

                if not all(pair in available_prices for pair in path):
                    continue

                # Extract prices for the three pairs
                pair1, pair2, pair3 = path
                price1 = available_prices[pair1]
                price2 = available_prices[pair2]
                price3 = available_prices[pair3]

                # Calculate triangular arbitrage
                btc_usdt_rate = price1.close if price1.close else (price1.ask or price1.bid)
                eth_btc_rate = price2.close if price2.close else (price2.ask or price2.bid)
                eth_usdt_rate = price3.close if price3.close else (price3.ask or price3.bid)

                if not all([btc_usdt_rate, eth_btc_rate, eth_usdt_rate]):
                    continue

                # Calculate triangular conversion
                start_usdt = btc_usdt_rate
                eth_amount = 1 / eth_btc_rate
                final_usdt = eth_amount * eth_usdt_rate

                # Calculate profit
                profit = final_usdt - start_usdt
                profit_percentage = (profit / start_usdt) * 100

                # Account for fees and slippage
                total_fees = FeeCalculator.calculate_triangular_fees(
                    btc_usdt_rate, eth_btc_rate, eth_usdt_rate
                )
                net_profit = profit - total_fees
                net_profit_percentage = (net_profit / start_usdt) * 100

                # Check if profit meets threshold
                if net_profit_percentage >= float(self.min_profit_threshold * 100):
                    # Create triangular arbitrage signal
                    signal = Signal(
                        symbol=pair1,
                        direction=SignalDirection.BUY,
                        strength=min(0.9, net_profit_percentage / 2),
                        timestamp=datetime.now(timezone.utc),
                        source=self.name,
                        metadata={
                            "arbitrage_type": "triangular",
                            "path": path,
                            "start_rate": float(btc_usdt_rate),
                            "intermediate_rate": float(eth_btc_rate),
                            "final_rate": float(eth_usdt_rate),
                            "profit_percentage": float(profit_percentage),
                            "net_profit_percentage": str(net_profit_percentage),
                            "estimated_fees": float(total_fees),
                            "opportunity_priority": OpportunityAnalyzer.calculate_priority(
                                net_profit_percentage, "triangular"
                            ),
                        },
                    )

                    signals.append(signal)

        except Exception as e:
            self.logger.error(
                "Triangular opportunity scanning failed", strategy=self.name, error=str(e)
            )

        return signals

    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate arbitrage opportunity signal.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Basic signal validation
            if not signal or signal.strength < self.min_confidence:
                return False

            # Validate arbitrage-specific metadata
            metadata = signal.metadata
            required_fields = ["arbitrage_type", "net_profit_percentage"]

            for field in required_fields:
                if field not in metadata:
                    return False

            # Validate profit threshold
            net_profit = metadata.get("net_profit_percentage", 0)
            if net_profit < float(self.min_profit_threshold * 100):
                return False

            # Validate arbitrage type
            arbitrage_type = metadata.get("arbitrage_type")
            if arbitrage_type not in ["cross_exchange", "triangular"]:
                return False

            # Additional validation based on type
            if arbitrage_type == "cross_exchange":
                required_cross_fields = ["buy_exchange", "sell_exchange", "buy_price", "sell_price"]
                for field in required_cross_fields:
                    if field not in metadata:
                        return False

                # Validate exchanges are different
                if metadata.get("buy_exchange") == metadata.get("sell_exchange"):
                    return False

            elif arbitrage_type == "triangular":
                required_triangular_fields = [
                    "path",
                    "start_rate",
                    "intermediate_rate",
                    "final_rate",
                ]
                for field in required_triangular_fields:
                    if field not in metadata:
                        return False

                # Validate path structure
                path = metadata.get("path", [])
                if len(path) != 3:
                    return False

            return True

        except Exception as e:
            self.logger.error(
                "Arbitrage signal validation failed", strategy=self.name, error=str(e)
            )
            return False

    @log_errors
    def get_position_size(self, signal: Signal) -> Decimal:
        """
        Calculate position size for arbitrage opportunity using risk management components.

        Args:
            signal: Arbitrage opportunity signal

        Returns:
            Position size in base currency

        Raises:
            ArbitrageError: If position size calculation fails
        """
        try:
            # Validate signal using utils
            if not signal:
                raise ArbitrageError("Invalid signal for arbitrage position sizing")

            # Convert to percentage scale and validate
            if signal.strength < 0.0 or signal.strength > 1.0:
                raise ValidationError(f"Invalid signal strength: {signal.strength}")

            # Use strategy commons for position sizing
            total_capital = Decimal(str(self.config.parameters.get("total_capital", 10000)))
            risk_per_trade = Decimal(str(self.config.parameters.get("risk_per_trade", 0.02)))
            max_position_size = Decimal(str(self.config.parameters.get("max_position_size", 0.1)))

            # Extract profit potential from metadata
            metadata = signal.metadata
            profit_potential = Decimal(str(metadata.get("net_profit_percentage", 0)))
            arbitrage_type = metadata.get("arbitrage_type", "cross_exchange")

            # Use arbitrage helpers for position sizing
            from src.utils.arbitrage_helpers import PositionSizingCalculator

            position_size = PositionSizingCalculator.calculate_arbitrage_position_size(
                total_capital=total_capital,
                risk_per_trade=risk_per_trade,
                max_position_size=max_position_size,
                profit_potential=profit_potential,
                signal_strength=Decimal(str(signal.strength)),
                arbitrage_type=arbitrage_type,
            )

            return position_size

        except Exception as e:
            self.logger.error(
                "Arbitrage scanner position size calculation failed",
                strategy=self.name,
                signal_strength=signal.strength if signal else None,
                error=str(e),
            )
            raise ArbitrageError(f"Arbitrage scanner position size calculation failed: {e!s}")

    async def should_exit(self, position: Position, data: MarketData) -> bool:
        """
        Determine if arbitrage opportunity position should be closed.

        Args:
            position: Current position
            data: Latest market data

        Returns:
            True if position should be closed, False otherwise
        """
        try:
            # Check if this is an arbitrage opportunity position
            if "arbitrage_type" not in position.metadata:
                return False  # Not an arbitrage position

            # Check execution timeout
            metadata = position.metadata
            execution_timeout = metadata.get("execution_timeout", self.max_execution_time)
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

            # Check if opportunity still exists
            arbitrage_type = metadata.get("arbitrage_type")

            if arbitrage_type == "cross_exchange":
                # Check current spread
                buy_exchange = metadata.get("buy_exchange")
                sell_exchange = metadata.get("sell_exchange")

                if buy_exchange and sell_exchange:
                    current_spread = await self._get_current_cross_exchange_spread(
                        position.symbol, buy_exchange, sell_exchange
                    )

                    if current_spread <= 0:
                        return True

            elif arbitrage_type == "triangular":
                # Check if triangular opportunity still exists
                path = metadata.get("path", [])

                if path and len(path) == 3:
                    # Recalculate current triangular arbitrage
                    current_opportunity = await self._check_triangular_path(path)

                    if not current_opportunity:
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

    async def _get_current_cross_exchange_spread(
        self, symbol: str, buy_exchange: str, sell_exchange: str
    ) -> Decimal:
        """
        Get current cross-exchange spread.

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
                "Cross-exchange spread calculation failed",
                strategy=self.name,
                symbol=symbol,
                error=str(e),
            )
            return Decimal("0")

    def _calculate_cross_exchange_fees(self, buy_price: Decimal, sell_price: Decimal) -> Decimal:
        """
        Calculate fees for cross-exchange arbitrage.

        Args:
            buy_price: Price to buy at
            sell_price: Price to sell at

        Returns:
            Total fees in quote currency

        Raises:
            ArbitrageError: If prices are invalid
        """
        try:
            if buy_price <= 0 or sell_price <= 0:
                raise ArbitrageError("Invalid price for fee calculation")

            # Standard exchange fees (0.1% per trade * 2 trades = 0.2% total)
            # Calculate fees on notional amount, not price
            fee_rate = Decimal("0.002")
            # Assume 1 unit trade for fee calculation
            trade_amount = Decimal("1")
            buy_fee = buy_price * trade_amount * fee_rate * Decimal("0.0001")  # 0.01% fee
            sell_fee = sell_price * trade_amount * fee_rate * Decimal("0.0001")  # 0.01% fee

            return buy_fee + sell_fee

        except Exception as e:
            self.logger.error(
                "Cross-exchange fee calculation failed",
                strategy=self.name,
                buy_price=float(buy_price),
                sell_price=float(sell_price),
                error=str(e),
            )
            raise ArbitrageError(f"Fee calculation failed: {e!s}")

    def _calculate_triangular_fees(self, rate1: Decimal, rate2: Decimal, rate3: Decimal) -> Decimal:
        """
        Calculate fees for triangular arbitrage.

        Args:
            rate1: First exchange rate
            rate2: Second exchange rate
            rate3: Third exchange rate

        Returns:
            Total fees in base currency

        Raises:
            ArbitrageError: If rates are invalid
        """
        try:
            if rate1 <= 0 or rate2 <= 0 or rate3 <= 0:
                raise ArbitrageError("Invalid rate for fee calculation")

            # Standard exchange fees (0.1% per trade * 3 trades = 0.3% total)
            fee_rate = Decimal("0.003")

            # Calculate fees based on trade amounts
            # For simplicity, assume unit trade amount and calculate proportional fees
            total_fees = (rate1 + rate2 + rate3) * fee_rate / 3

            return total_fees

        except Exception as e:
            self.logger.error(
                "Triangular fee calculation failed",
                strategy=self.name,
                rate1=float(rate1),
                rate2=float(rate2),
                rate3=float(rate3),
                error=str(e),
            )
            raise ArbitrageError(f"Triangular fee calculation failed: {e!s}")

    def _calculate_priority(self, profit_percentage: float, arbitrage_type: str) -> float:
        """
        Calculate opportunity priority score.

        Args:
            profit_percentage: Expected profit percentage
            arbitrage_type: Type of arbitrage opportunity

        Returns:
            Priority score (0.0 to 1000.0)

        Raises:
            ArbitrageError: If parameters are invalid
        """
        try:
            if profit_percentage < 0:
                raise ArbitrageError("Profit percentage cannot be negative")

            if arbitrage_type not in ["cross_exchange", "triangular"]:
                raise ArbitrageError("Invalid arbitrage type")

            # Base priority from profit percentage (0-100 scale)
            base_priority = min(profit_percentage * 100, 100)

            # Multiply by type factor
            type_multipliers = {
                "cross_exchange": 1.0,  # Standard priority
                "triangular": 0.8,  # Slightly lower due to complexity
            }

            priority = base_priority * type_multipliers[arbitrage_type]

            # Cap at 1000 for maximum priority
            return min(priority, 1000.0)

        except Exception as e:
            self.logger.error(
                "Priority calculation failed",
                strategy=self.name,
                profit_percentage=profit_percentage,
                arbitrage_type=arbitrage_type,
                error=str(e),
            )
            raise ArbitrageError(f"Priority calculation failed: {e!s}")

    def _prioritize_opportunities(self, signals: list[Signal]) -> list[Signal]:
        """
        Prioritize and limit opportunities.

        Args:
            signals: List of opportunity signals

        Returns:
            Prioritized and limited list of signals
        """
        try:
            if not signals:
                return []

            # Sort by opportunity priority (highest first)
            sorted_signals = sorted(
                signals, key=lambda s: s.metadata.get("opportunity_priority", 0), reverse=True
            )

            # Limit to max opportunities
            return sorted_signals[: self.max_opportunities]

        except Exception as e:
            self.logger.error(
                "Opportunity prioritization failed",
                strategy=self.name,
                signal_count=len(signals) if signals else 0,
                error=str(e),
            )
            return signals[: self.max_opportunities] if signals else []

    async def _check_triangular_path(self, path: list[str]) -> "Signal | None":
        """
        Check if triangular arbitrage opportunity still exists.

        Args:
            path: Triangular path to check

        Returns:
            Signal if opportunity exists, None otherwise
        """
        try:
            # Ensure we have prices for all pairs in the path from any exchange
            available_prices = {}
            for pair in path:
                for exchange in self.exchanges:
                    if exchange in self.exchange_prices and pair in self.exchange_prices[exchange]:
                        available_prices[pair] = self.exchange_prices[exchange][pair]
                        break

            if not all(pair in available_prices for pair in path):
                return None

            # Extract prices for the three pairs
            pair1, pair2, pair3 = path
            price1 = available_prices[pair1]
            price2 = available_prices[pair2]
            price3 = available_prices[pair3]

            # Calculate triangular arbitrage
            btc_usdt_rate = price1.close if price1.close else (price1.ask or price1.bid)
            eth_btc_rate = price2.close if price2.close else (price2.ask or price2.bid)
            eth_usdt_rate = price3.close if price3.close else (price3.ask or price3.bid)

            if not all([btc_usdt_rate, eth_btc_rate, eth_usdt_rate]):
                return None

            # Calculate triangular conversion
            start_usdt = btc_usdt_rate
            eth_amount = 1 / eth_btc_rate
            final_usdt = eth_amount * eth_usdt_rate

            # Calculate profit
            profit = final_usdt - start_usdt
            (profit / start_usdt) * 100

            # Account for fees and slippage
            total_fees = self._calculate_triangular_fees(btc_usdt_rate, eth_btc_rate, eth_usdt_rate)
            net_profit = profit - total_fees
            net_profit_percentage = (net_profit / start_usdt) * 100

            # Check if profit meets threshold - use Decimal comparison
            threshold_percentage = self.min_profit_threshold * Decimal("100")
            if net_profit_percentage >= threshold_percentage:
                return Signal(
                    symbol=pair1,
                    direction=SignalDirection.BUY,
                    strength=min(0.9, net_profit_percentage / 2),
                    timestamp=datetime.now(timezone.utc),
                    source=self.name,
                    metadata={
                        "arbitrage_type": "triangular",
                        "path": path,
                        "net_profit_percentage": str(net_profit_percentage),
                    },
                )

            return None

        except Exception as e:
            self.logger.error(
                "Triangular path check failed", strategy=self.name, path=path, error=str(e)
            )
            return None

    async def post_trade_processing(self, trade_result: dict[str, Any]) -> None:
        """
        Process completed arbitrage opportunity trade.

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

            # Update execution success rate
            if "execution_success" in trade_result:
                # TBD: Implement execution success rate tracking for performance analysis
                pass

            # Log trade result
            self.logger.info(
                "Arbitrage opportunity trade completed",
                strategy=self.name,
                symbol=trade_result.get("symbol"),
                arbitrage_type=trade_result.get("arbitrage_type"),
                pnl=float(trade_result.get("pnl", 0)),
                execution_time_ms=trade_result.get("execution_time_ms", 0),
            )

        except Exception as e:
            self.logger.error("Post-trade processing failed", strategy=self.name, error=str(e))
