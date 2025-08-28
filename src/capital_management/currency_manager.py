"""
Currency Manager Implementation (P-010A)

This module implements multi-currency capital management with base currency
standardization, cross-currency exposure monitoring, and currency hedging.

Key Features:
- Base currency standardization (USDT default)
- Cross-currency exposure monitoring and hedging
- Currency conversion optimization
- Exchange rate risk management
- Multi-asset portfolio currency hedging
- Currency exposure limits and controls

Author: Trading Bot Framework
Version: 1.0.0
"""

import statistics
from datetime import datetime, timezone
from decimal import Decimal

# Remove direct exchange dependency - use protocols
from typing import Any, Protocol

from src.core.base.service import TransactionalService
from src.core.exceptions import ServiceError, ValidationError

# MANDATORY: Import from P-001
from src.core.types.capital import (
    CapitalCurrencyExposure as CurrencyExposure,
    CapitalFundFlow as FundFlow,
)


class ExchangeDataProtocol(Protocol):
    """Protocol for exchange data operations."""

    async def fetch_tickers(self) -> dict[str, Any]: ...
    async def fetch_order_book(self, symbol: str, limit: int = 50) -> dict[str, Any]: ...
    async def fetch_status(self) -> dict[str, Any]: ...


from src.utils.decorators import time_execution
from src.utils.formatters import format_currency
from src.utils.validators import ValidationFramework

# Import service interfaces
from .interfaces import AbstractCurrencyManagementService

# MANDATORY: Use structured logging from src.core.logging for all capital
# management operations

# From P-003+ - MANDATORY: Use existing exchange interfaces

# From P-002A - MANDATORY: Use error handling

# From P-007A - MANDATORY: Use decorators and validators


class CurrencyManager(AbstractCurrencyManagementService, TransactionalService):
    """
    Multi-currency capital management system.

    This class manages currency exposures, handles currency conversions,
    and implements hedging strategies to minimize exchange rate risk.
    """

    def __init__(
        self,
        exchange_data_service: Any = None,
        correlation_id: str | None = None,
    ) -> None:
        """
        Initialize the currency manager service.

        Args:
            exchange_data_service: Exchange data service for market data (injected)
            correlation_id: Request correlation ID for tracing
        """
        super().__init__(
            name="CurrencyManagerService",
            correlation_id=correlation_id,
        )
        self._exchange_data_service = exchange_data_service

        # Currency tracking
        self.currency_exposures: dict[str, CurrencyExposure] = {}
        self.exchange_rates: dict[str, Decimal] = {}
        self.base_currency = "USDT"  # Default, will be loaded from config

        # Hedging tracking
        self.hedge_positions: dict[str, Decimal] = {}
        self.hedging_threshold = 0.1  # Default, will be loaded from config
        self.hedge_ratio = 0.5  # Default, will be loaded from config

        # Historical exchange rates for volatility calculation
        self.rate_history: dict[str, list[tuple[datetime, Decimal]]] = {}

        # Configuration will be loaded in _do_start
        self.capital_config: dict[str, Any] = {}

    async def _do_start(self) -> None:
        """Start the currency manager service."""
        try:
            # Resolve exchange data service if not injected
            if not self._exchange_data_service:
                try:
                    self._exchange_data_service = self.resolve_dependency("ExchangeDataService")
                except Exception as e:
                    self._logger.warning(
                        f"ExchangeDataService not available via DI: {e}, will use fallback rates"
                    )

            # Load configuration from ConfigService
            await self._load_configuration()

            # Initialize supported currencies
            self._initialize_currencies()

            self._logger.info(
                "Currency manager service started",
                base_currency=self.base_currency,
                supported_currencies=self.capital_config.get(
                    "supported_currencies", ["USDT", "BTC", "ETH"]
                ),
                hedging_enabled=self.capital_config.get("hedging_enabled", False),
            )
        except Exception as e:
            self._logger.error(f"Failed to start CurrencyManager service: {e}")
            raise ServiceError(f"CurrencyManager startup failed: {e}") from e

    async def _load_configuration(self) -> None:
        """Load configuration from ConfigService."""
        try:
            config_service = self.resolve_dependency("ConfigService")
            self.capital_config = config_service.get_config_value("capital_management", {})

            # Update settings from config
            self.base_currency = self.capital_config.get("base_currency", "USDT")
            self.hedging_threshold = self.capital_config.get("hedging_threshold", 0.1)
            self.hedge_ratio = self.capital_config.get("hedge_ratio", 0.5)
        except Exception as e:
            self._logger.warning(f"Failed to load configuration: {e}, using defaults")
            # Use defaults
            self.capital_config = {
                "base_currency": "USDT",
                "hedging_threshold": 0.1,
                "hedge_ratio": 0.5,
                "supported_currencies": ["USDT", "BTC", "ETH"],
                "hedging_enabled": False,
            }
            self.base_currency = "USDT"
            self.hedging_threshold = 0.1
            self.hedge_ratio = 0.5

    @time_execution
    async def update_currency_exposures(
        self, balances: dict[str, dict[str, Decimal]]
    ) -> dict[str, CurrencyExposure]:
        """
        Update currency exposures based on current balances.

        Args:
            balances: Dictionary of exchange balances by currency

        Returns:
            Dict[str, CurrencyExposure]: Updated currency exposures
        """
        try:
            # Update exchange rates
            await self._update_exchange_rates()

            # Validate currencies are supported
            supported_currencies = self.capital_config.get(
                "supported_currencies", ["USDT", "BTC", "ETH"]
            )
            for _exchange, exchange_balances in balances.items():
                for currency, amount in exchange_balances.items():
                    if currency not in supported_currencies:
                        raise ValidationError(f"Unsupported currency: {currency}")

            # Calculate total exposures by currency
            total_exposures = {}
            for _exchange, exchange_balances in balances.items():
                for currency, amount in exchange_balances.items():
                    if currency not in total_exposures:
                        total_exposures[currency] = Decimal("0")
                    total_exposures[currency] += amount

            # Calculate base currency equivalents and percentages
            total_base_value = Decimal("0")
            base_equivalents = {}

            for currency, amount in total_exposures.items():
                if currency == self.base_currency:
                    base_equivalent = amount
                else:
                    rate = self.exchange_rates.get(f"{currency}/{self.base_currency}", Decimal("1"))
                    base_equivalent = amount * rate

                base_equivalents[currency] = base_equivalent
                total_base_value += base_equivalent

            # Create currency exposure objects
            exposures = {}
            for currency, amount in total_exposures.items():
                base_equivalent = base_equivalents[currency]
                exposure_percentage = (
                    float(base_equivalent / total_base_value) if total_base_value > 0 else 0.0
                )

                # Determine if hedging is required
                hedging_enabled = getattr(self.capital_config, "hedging_enabled", False)
                hedging_required = (
                    hedging_enabled
                    and currency != self.base_currency
                    and exposure_percentage > self.hedging_threshold
                )

                # Calculate hedge amount
                hedge_amount = Decimal("0")
                if hedging_required:
                    hedge_amount = base_equivalent * Decimal(str(self.hedge_ratio))

                exposure = CurrencyExposure(
                    currency=currency,
                    total_exposure=amount,
                    base_currency_equivalent=base_equivalent,
                    exposure_percentage=exposure_percentage,
                    hedging_required=hedging_required,
                    hedge_amount=hedge_amount,
                    timestamp=datetime.now(timezone.utc),
                )

                exposures[currency] = exposure
                self.currency_exposures[currency] = exposure

            self.logger.info(
                "Currency exposures updated",
                total_currencies=len(exposures),
                total_base_value=format_currency(float(total_base_value)),
                hedging_required=sum(1 for e in exposures.values() if e.hedging_required),
            )

            return exposures

        except Exception as e:
            self.logger.error("Failed to update currency exposures", error=str(e))
            raise

    @time_execution
    async def calculate_hedging_requirements(self) -> dict[str, Decimal]:
        """
        Calculate hedging requirements for currency exposures.

        Returns:
            Dict[str, Decimal]: Required hedge amounts by currency
        """
        try:
            hedging_requirements = {}

            for currency, exposure in self.currency_exposures.items():
                if exposure.hedging_required:
                    # Calculate required hedge amount
                    required_hedge = exposure.base_currency_equivalent * Decimal(
                        str(self.hedge_ratio)
                    )
                    current_hedge = self.hedge_positions.get(currency, Decimal("0"))

                    hedge_delta = required_hedge - current_hedge
                    if abs(hedge_delta) > Decimal("0.01"):  # Minimum hedge amount
                        hedging_requirements[currency] = hedge_delta

            self.logger.info(
                "Hedging requirements calculated",
                currencies_to_hedge=len(hedging_requirements),
                total_hedge_amount=format_currency(float(sum(hedging_requirements.values()))),
            )

            return hedging_requirements

        except Exception as e:
            self.logger.error("Failed to calculate hedging requirements", error=str(e))
            raise

    @time_execution
    async def execute_currency_conversion(
        self, from_currency: str, to_currency: str, amount: Decimal, exchange: str
    ) -> FundFlow:
        """
        Execute currency conversion between currencies.

        Args:
            from_currency: Source currency
            to_currency: Target currency
            amount: Amount to convert
            exchange: Exchange to use for conversion

        Returns:
            FundFlow: Conversion flow record
        """
        try:
            # Validate inputs
            if not ValidationFramework.validate_quantity(float(amount)):
                raise ValidationError(f"Invalid currency conversion amount: {amount}")

            supported_currencies = self.capital_config.get(
                "supported_currencies", ["USDT", "BTC", "ETH"]
            )
            if from_currency not in supported_currencies:
                raise ValidationError(f"Unsupported source currency: {from_currency}")

            if to_currency not in supported_currencies:
                raise ValidationError(f"Unsupported target currency: {to_currency}")

            # Get exchange rate
            if from_currency == to_currency:
                rate = Decimal("1")
            else:
                rate = self.exchange_rates.get(f"{from_currency}/{to_currency}")
                if not rate:
                    # Try reverse rate
                    reverse_rate = self.exchange_rates.get(f"{to_currency}/{from_currency}")
                    if reverse_rate:
                        rate = Decimal("1") / reverse_rate
                    else:
                        raise ValidationError(
                            f"No exchange rate available for {from_currency}/{to_currency}"
                        )

            # Calculate converted amount
            converted_amount = amount * rate

            # Create fund flow record
            flow = FundFlow(
                from_strategy=None,
                to_strategy=None,
                from_exchange=exchange,
                to_exchange=exchange,
                amount=amount,
                currency=from_currency,
                converted_amount=converted_amount,
                exchange_rate=rate,
                reason="currency_conversion",
                timestamp=datetime.now(timezone.utc),
            )

            self.logger.info(
                "Currency conversion executed",
                from_currency=from_currency,
                to_currency=to_currency,
                amount=format_currency(float(amount), from_currency),
                converted_amount=format_currency(float(converted_amount), to_currency),
                exchange_rate=float(rate),
                exchange=exchange,
            )

            return flow

        except Exception as e:
            self.logger.error(
                "Currency conversion failed",
                from_currency=from_currency,
                to_currency=to_currency,
                amount=format_currency(float(amount), from_currency),
                error=str(e),
            )
            raise

    @time_execution
    async def optimize_currency_allocation(
        self, target_allocations: dict[str, Decimal]
    ) -> dict[str, Decimal]:
        """
        Optimize currency allocation to minimize conversion costs.

        Args:
            target_allocations: Target currency allocations

        Returns:
            Dict[str, Decimal]: Optimized allocation amounts
        """
        try:
            # Get current exposures
            current_exposures = {
                curr: exp.total_exposure for curr, exp in self.currency_exposures.items()
            }

            # Calculate required changes
            required_changes = {}
            for currency, target_amount in target_allocations.items():
                current_amount = current_exposures.get(currency, Decimal("0"))
                change = target_amount - current_amount
                if abs(change) > Decimal("0.01"):  # Minimum change threshold
                    required_changes[currency] = change

            # Optimize conversions to minimize costs
            optimized_changes = await self._optimize_conversions(required_changes)

            self.logger.info(
                "Currency allocation optimized",
                currencies_optimized=len(optimized_changes),
                total_conversion_cost=format_currency(
                    float(sum(abs(change) for change in optimized_changes.values()))
                ),
            )

            return optimized_changes

        except Exception as e:
            self.logger.error("Failed to optimize currency allocation", error=str(e))
            raise

    @time_execution
    async def get_currency_risk_metrics(self) -> dict[str, dict[str, float]]:
        """
        Calculate currency risk metrics.

        Returns:
            Dict[str, float]: Risk metrics by currency
        """
        try:
            risk_metrics = {}

            for currency, exposure in self.currency_exposures.items():
                if currency == self.base_currency:
                    continue

                # Calculate volatility from historical rates
                rate_history = self.rate_history.get(f"{currency}/{self.base_currency}", [])
                if len(rate_history) > 1:
                    # Last 30 data points
                    rates = [float(rate) for _, rate in rate_history[-30:]]
                    try:
                        volatility = statistics.stdev(rates) if len(rates) > 1 else 0.0
                    except (statistics.StatisticsError, ValueError):
                        volatility = 0.0
                else:
                    volatility = 0.0

                # Calculate correlation with base currency (simplified)
                correlation = 0.0  # TODO: Implement correlation calculation

                # Calculate VaR (simplified)
                var_95 = exposure.base_currency_equivalent * Decimal(str(volatility * 1.645))

                risk_metrics[currency] = {
                    "volatility": volatility,
                    "correlation": correlation,
                    "var_95": float(var_95),
                    "exposure_pct": exposure.exposure_percentage,
                    "hedging_required": exposure.hedging_required,
                }

            return risk_metrics

        except Exception as e:
            self.logger.error("Failed to calculate currency risk metrics", error=str(e))
            raise

    def _initialize_currencies(self) -> None:
        """Initialize supported currencies with default exposures."""
        supported_currencies = getattr(
            self.capital_config, "supported_currencies", ["USDT", "BTC", "ETH"]
        )
        for currency in supported_currencies:
            exposure = CurrencyExposure(
                currency=currency,
                total_exposure=Decimal("0"),
                base_currency_equivalent=Decimal("0"),
                exposure_percentage=0.0,
                hedging_required=False,
                hedge_amount=Decimal("0"),
                timestamp=datetime.now(timezone.utc),
            )
            self.currency_exposures[currency] = exposure

    async def _update_exchange_rates(self) -> None:
        """Update exchange rates from exchange data service."""
        try:
            rates_updated = 0

            # Fetch real rates from exchange data service if available
            if self._exchange_data_service:
                try:
                    # Use service method to get ticker data
                    if hasattr(self._exchange_data_service, "get_tickers"):
                        tickers = await self._exchange_data_service.get_tickers()
                    elif hasattr(self._exchange_data_service, "fetch_tickers"):
                        tickers = await self._exchange_data_service.fetch_tickers()
                    else:
                        # Service doesn't have expected methods
                        tickers = {}

                    for symbol, ticker in tickers.items():
                        if isinstance(ticker, dict) and ticker.get("last"):
                            rate = Decimal(str(ticker["last"]))
                            self.exchange_rates[symbol] = rate

                            # Store in history for volatility calculation
                            if symbol not in self.rate_history:
                                self.rate_history[symbol] = []

                            self.rate_history[symbol].append((datetime.now(timezone.utc), rate))

                            # Keep only last 100 data points
                            if len(self.rate_history[symbol]) > 100:
                                self.rate_history[symbol] = self.rate_history[symbol][-100:]

                            rates_updated += 1

                except Exception as e:
                    self.logger.warning(f"Failed to fetch rates from exchange data service: {e}")

            # Fallback to hardcoded rates if no service available (for testing)
            if rates_updated == 0:
                self.logger.warning("No exchange rates fetched, using fallback rates")
                fallback_rates = {
                    "BTC/USDT": Decimal("45000"),
                    "ETH/USDT": Decimal("3000"),
                    "USDC/USDT": Decimal("1.0"),
                    "BUSD/USDT": Decimal("1.0"),
                    "ETH/BTC": Decimal("0.0667"),
                    "USDC/BTC": Decimal("0.000022"),
                    "BUSD/BTC": Decimal("0.000022"),
                }

                for pair, rate in fallback_rates.items():
                    self.exchange_rates[pair] = rate
                    rates_updated += 1

            self.logger.debug(f"Updated {rates_updated} exchange rates")

        except Exception as e:
            self.logger.error("Failed to update exchange rates", error=str(e))

    async def _optimize_conversions(
        self, required_changes: dict[str, Decimal]
    ) -> dict[str, Decimal]:
        """
        Optimize currency conversions to minimize costs.

        Args:
            required_changes: Required changes by currency

        Returns:
            Dict[str, Decimal]: Optimized changes
        """
        try:
            optimized_changes = {}

            # Simple optimization: prioritize conversions with better rates
            # In production, this would use more sophisticated algorithms

            for currency, change in required_changes.items():
                if currency == self.base_currency:
                    optimized_changes[currency] = change
                    continue

                # Check if we can use existing balances instead of converting
                current_exposure = self.currency_exposures.get(currency)
                if current_exposure and current_exposure.total_exposure > 0:
                    # Use existing balance if possible
                    if change > 0:  # Need more of this currency
                        available = current_exposure.total_exposure
                        if available >= change:
                            # No conversion needed
                            optimized_changes[currency] = Decimal("0")
                            continue
                        else:
                            # Partial conversion needed
                            optimized_changes[currency] = change - available
                    else:
                        # Selling this currency - no conversion needed
                        optimized_changes[currency] = change
                else:
                    # No existing balance, conversion required
                    optimized_changes[currency] = change

            return optimized_changes

        except Exception as e:
            self.logger.error("Failed to optimize conversions", error=str(e))
            raise

    async def get_total_base_value(self) -> Decimal:
        """Get total portfolio value in base currency."""
        total_value = Decimal("0")
        for exposure in self.currency_exposures.values():
            total_value += exposure.base_currency_equivalent
        return total_value

    async def get_currency_exposure(self, currency: str) -> CurrencyExposure | None:
        """Get current exposure for a specific currency."""
        return self.currency_exposures.get(currency)

    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> Decimal | None:
        """Get current exchange rate between currencies."""
        if from_currency == to_currency:
            return Decimal("1")

        # Try direct rate
        rate = self.exchange_rates.get(f"{from_currency}/{to_currency}")
        if rate:
            return rate

        # Try reverse rate
        reverse_rate = self.exchange_rates.get(f"{to_currency}/{from_currency}")
        if reverse_rate:
            return Decimal("1") / reverse_rate

        return None

    async def update_hedge_position(self, currency: str, hedge_amount: Decimal) -> None:
        """Update hedge position for a currency."""
        self.hedge_positions[currency] = hedge_amount

        self.logger.info(
            "Hedge position updated",
            currency=currency,
            hedge_amount=format_currency(float(hedge_amount)),
        )

    async def get_hedging_summary(self) -> dict[str, Any]:
        """Get summary of current hedging status."""
        total_hedge_value = sum(self.hedge_positions.values())
        currencies_hedged = len(
            [curr for curr, exp in self.currency_exposures.items() if exp.hedging_required]
        )

        return {
            "total_hedge_value": float(total_hedge_value),
            "currencies_hedged": currencies_hedged,
            "hedging_enabled": self.capital_config.hedging_enabled,
            "hedging_threshold": self.hedging_threshold,
            "hedge_ratio": self.hedge_ratio,
            "hedge_positions": {
                curr: float(amount) for curr, amount in self.hedge_positions.items()
            },
        }
