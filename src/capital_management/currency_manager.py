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

from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import TYPE_CHECKING, Any

from src.capital_management.constants import (
    DEFAULT_BASE_CURRENCY,
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_CONVERSION_FEE_RATE,
    DEFAULT_HEDGE_RATIO,
    DEFAULT_HEDGING_THRESHOLD,
    DEFAULT_MAX_RATE_HISTORY,
    DEFAULT_SUPPORTED_CURRENCIES,
    MAX_RATE_HISTORY_PER_SYMBOL,
    MIN_CHANGE_THRESHOLD,
    MIN_HEDGE_AMOUNT,
    RATE_CALCULATION_LOOKBACK_DAYS,
    RATE_HISTORY_MAX_AGE_DAYS,
    DEFAULT_VaR_CONFIDENCE_MULTIPLIER,
)
from src.capital_management.interfaces import AbstractCurrencyManagementService
from src.core.base.service import TransactionalService
from src.core.exceptions import ServiceError, ValidationError
from src.core.types.capital import (
    CapitalCurrencyExposure as CurrencyExposure,
    CapitalFundFlow as FundFlow,
)
from src.utils.capital_config import (
    get_supported_currencies,
    load_capital_config,
    resolve_config_service,
)
from src.utils.capital_validation import (
    validate_capital_amount,
    validate_supported_currencies,
)
from src.utils.decimal_utils import safe_decimal_conversion
from src.utils.decorators import time_execution
from src.utils.formatters import format_currency
from src.utils.interfaces import ValidationServiceInterface

# Set decimal context for financial precision
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

if TYPE_CHECKING:
    from src.risk_management import RiskService


class CurrencyManager(AbstractCurrencyManagementService, TransactionalService):
    """
    Multi-currency capital management system.

    This class manages currency exposures, handles currency conversions,
    and implements hedging strategies to minimize exchange rate risk.
    """

    def __init__(
        self,
        exchange_data_service: Any = None,
        validation_service: ValidationServiceInterface | None = None,
        risk_service: "RiskService | None" = None,
        correlation_id: str | None = None,
    ) -> None:
        """
        Initialize the currency manager service.

        Args:
            exchange_data_service: Exchange data service for market data (injected)
            validation_service: Validation service for input validation (injected)
            risk_service: Risk service for VaR and risk calculations (injected)
            correlation_id: Request correlation ID for tracing
        """
        super().__init__(
            name="CurrencyManagerService",
            correlation_id=correlation_id,
        )
        self._exchange_data_service = exchange_data_service
        self.validation_service = validation_service
        self.risk_service = risk_service

        self.currency_exposures: dict[str, CurrencyExposure] = {}
        self.exchange_rates: dict[str, Decimal] = {}
        self.base_currency = DEFAULT_BASE_CURRENCY

        self.hedge_positions: dict[str, Decimal] = {}
        self.hedging_threshold = DEFAULT_HEDGING_THRESHOLD  # Default, will be loaded from config
        self.hedge_ratio = DEFAULT_HEDGE_RATIO  # Default, will be loaded from config

        self.rate_history: dict[str, list[tuple[datetime, Decimal]]] = {}

        self._cleanup_required = False

        self.capital_config: dict[str, Any] = {}

    async def _do_start(self) -> None:
        """Start the currency manager service."""
        try:
            try:
                await self._load_configuration()
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
                raise ServiceError(f"Configuration loading failed: {e}") from e

            try:
                self._initialize_currencies()
            except Exception as e:
                self.logger.error(f"Failed to initialize currencies: {e}")
                raise ServiceError(f"Currency initialization failed: {e}") from e

            self.logger.info(
                "Currency manager service started",
                base_currency=self.base_currency,
                supported_currencies=self.capital_config.get(
                    "supported_currencies", DEFAULT_SUPPORTED_CURRENCIES
                ),
                hedging_enabled=self.capital_config.get("hedging_enabled", False),
            )
        except Exception as e:
            self.logger.error(f"Failed to start CurrencyManager service: {e}")
            raise ServiceError(f"CurrencyManager startup failed: {e}") from e

    async def _do_stop(self) -> None:
        """Stop the currency manager service and clean up resources."""
        from src.utils.service_utils import safe_service_shutdown

        await safe_service_shutdown(
            service_name="CurrencyManager",
            cleanup_func=self.cleanup_resources,
            service_logger=self.logger,
        )

    async def _load_configuration(self) -> None:
        """Load configuration from ConfigService."""
        resolved_config_service = resolve_config_service(self)
        self.capital_config = load_capital_config(resolved_config_service)

        self.base_currency = self.capital_config.get("base_currency", DEFAULT_BASE_CURRENCY)
        self.hedging_threshold = self.capital_config.get(
            "hedging_threshold", DEFAULT_HEDGING_THRESHOLD
        )
        self.hedge_ratio = self.capital_config.get("hedge_ratio", DEFAULT_HEDGE_RATIO)
        self.supported_currencies = get_supported_currencies(self.capital_config)
        self.hedging_enabled = self.capital_config.get("hedging_enabled", False)

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
            await self._update_exchange_rates()
            await self._validate_currencies(balances)

            total_exposures = self._calculate_total_exposures(balances)
            base_equivalents, total_base_value = self._calculate_base_equivalents(total_exposures)
            exposures = self._create_currency_exposures(
                total_exposures, base_equivalents, total_base_value
            )

            self._log_exposure_update(exposures, total_base_value)
            return exposures

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error("Failed to update currency exposures", error=str(e))
            raise ServiceError(f"Currency exposure update failed: {e}") from e

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
                exposure_percentage = safe_decimal_conversion(exposure.exposure_percentage)
                if exposure_percentage > self.hedging_threshold:
                    required_hedge = exposure.base_currency_equivalent * Decimal(
                        str(self.hedge_ratio)
                    )
                    current_hedge = self.hedge_positions.get(currency, Decimal("0"))

                    hedge_delta = required_hedge - current_hedge
                    if abs(hedge_delta) > MIN_HEDGE_AMOUNT:
                        hedging_requirements[currency] = hedge_delta

            self.logger.info(
                "Hedging requirements calculated",
                currencies_to_hedge=len(hedging_requirements),
                total_hedge_amount=format_currency(sum(hedging_requirements.values())),
            )

            return hedging_requirements

        except Exception as e:
            self.logger.error("Failed to calculate hedging requirements", error=str(e))
            raise ServiceError(f"Hedging requirements calculation failed: {e}") from e

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
            validate_capital_amount(amount, "conversion amount", component="CurrencyManager")

            supported_currencies = get_supported_currencies(self.capital_config)
            validate_supported_currencies(from_currency, supported_currencies, "CurrencyManager")
            validate_supported_currencies(to_currency, supported_currencies, "CurrencyManager")

            if from_currency == to_currency:
                rate = Decimal("1")
                fee_rate = Decimal("0")
                fee_amount = Decimal("0")
            else:
                rate_value = self.exchange_rates.get(f"{from_currency}/{to_currency}")
                if rate_value is None:
                    reverse_rate = self.exchange_rates.get(f"{to_currency}/{from_currency}")
                    if reverse_rate is not None:
                        rate = Decimal("1") / reverse_rate
                    else:
                        raise ValidationError(
                            f"No exchange rate available for {from_currency}/{to_currency}"
                        )
                else:
                    rate = rate_value

                fee_rate = DEFAULT_CONVERSION_FEE_RATE
                fee_amount = amount * rate * fee_rate

            converted_amount = amount * rate
            final_converted_amount = converted_amount - fee_amount
            flow = FundFlow(
                from_strategy=None,
                to_strategy=None,
                from_exchange=exchange,
                to_exchange=exchange,
                amount=amount,
                currency=from_currency,
                converted_amount=final_converted_amount,
                exchange_rate=rate,
                reason="currency_conversion",
                timestamp=datetime.now(timezone.utc),
                metadata={"fee_rate": fee_rate, "gross_converted_amount": converted_amount},
                fees=fee_amount,
                fee_amount=fee_amount,
            )

            self.logger.info(
                "Currency conversion executed",
                from_currency=from_currency,
                to_currency=to_currency,
                amount=format_currency(amount, from_currency),
                converted_amount=format_currency(converted_amount, to_currency),
                exchange_rate=rate,
                exchange=exchange,
            )

            return flow

        except Exception as e:
            self.logger.error(
                "Currency conversion failed",
                from_currency=from_currency,
                to_currency=to_currency,
                amount=format_currency(amount, from_currency),
                error=str(e),
            )
            raise ServiceError(f"Currency conversion failed: {e}") from e

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
            current_exposures = {
                curr: exp.total_exposure for curr, exp in self.currency_exposures.items()
            }

            required_changes = {}
            for currency, target_amount in target_allocations.items():
                current_amount = current_exposures.get(currency, Decimal("0"))
                change = target_amount - current_amount
                if abs(change) > MIN_CHANGE_THRESHOLD:
                    required_changes[currency] = change

            optimized_changes = await self._optimize_conversions(required_changes)

            self.logger.info(
                "Currency allocation optimized",
                currencies_optimized=len(optimized_changes),
                total_conversion_cost=format_currency(
                    sum(abs(change) for change in optimized_changes.values())
                ),
            )

            return optimized_changes

        except Exception as e:
            self.logger.error("Failed to optimize currency allocation", error=str(e))
            raise ServiceError(f"Currency allocation optimization failed: {e}") from e

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

                rate_history = self.rate_history.get(f"{currency}/{self.base_currency}", [])

                if self.risk_service:
                    try:
                        volatility = await self._calculate_currency_volatility_via_risk_service(
                            currency, rate_history
                        )
                        correlation = Decimal("0.0")
                        var_95 = await self._calculate_currency_var_via_risk_service(
                            exposure.base_currency_equivalent, rate_history
                        )
                    except Exception as risk_error:
                        self.logger.warning(
                            f"Risk service calculation failed for {currency}: {risk_error}"
                        )
                        volatility, var_95 = self._fallback_risk_calculation(
                            currency, exposure, rate_history
                        )
                        correlation = Decimal("0.0")
                else:
                    volatility, var_95 = self._fallback_risk_calculation(
                        currency, exposure, rate_history
                    )
                    correlation = Decimal("0.0")

                risk_metrics[currency] = {
                    "volatility": float(volatility),
                    "correlation": float(correlation),
                    "var_95": float(var_95),
                    "exposure_pct": float(exposure.exposure_percentage),
                    "hedging_required": float(exposure.hedging_required),
                }

            return risk_metrics

        except Exception as e:
            self.logger.error("Failed to calculate currency risk metrics", error=str(e))
            raise ServiceError(f"Currency risk metrics calculation failed: {e}") from e

    def _initialize_currencies(self) -> None:
        """Initialize supported currencies with default exposures."""
        supported_currencies = self.capital_config.get(
            "supported_currencies", DEFAULT_SUPPORTED_CURRENCIES
        )
        for currency in supported_currencies:
            exposure = CurrencyExposure(
                currency=currency,
                total_exposure=Decimal("0"),
                base_currency_equivalent=Decimal("0"),
                exposure_percentage=Decimal("0.0"),
                hedging_required=False,
                hedge_amount=Decimal("0"),
                timestamp=datetime.now(timezone.utc),
            )
            self.currency_exposures[currency] = exposure

    async def _update_exchange_rates(self) -> None:
        """Update exchange rates from exchange data service."""
        tickers = None
        try:
            rates_updated = 0

            # Fetch real rates from exchange data service if available
            if self._exchange_data_service:
                try:
                    # Use service method to get ticker data with proper async context management
                    import asyncio

                    try:
                        if hasattr(self._exchange_data_service, "get_tickers"):
                            tickers = await asyncio.wait_for(
                                self._exchange_data_service.get_tickers(),
                                timeout=DEFAULT_CONNECTION_TIMEOUT,
                            )
                        elif hasattr(self._exchange_data_service, "fetch_tickers"):
                            tickers = await asyncio.wait_for(
                                self._exchange_data_service.fetch_tickers(),
                                timeout=DEFAULT_CONNECTION_TIMEOUT,
                            )
                        else:
                            tickers = {}
                    except (asyncio.TimeoutError, ConnectionError) as e:
                        self.logger.warning(f"Exchange data service request failed: {e}")
                        tickers = {}
                    except Exception as e:
                        self.logger.warning(f"Unexpected error fetching tickers: {e}")
                        tickers = {}

                    if tickers:
                        for symbol, ticker in tickers.items():
                            if isinstance(ticker, dict) and ticker.get("last"):
                                rate = safe_decimal_conversion(ticker["last"])
                                self.exchange_rates[symbol] = rate

                                if symbol not in self.rate_history:
                                    self.rate_history[symbol] = []

                                self.rate_history[symbol].append((datetime.now(timezone.utc), rate))

                                if len(self.rate_history[symbol]) > DEFAULT_MAX_RATE_HISTORY:
                                    self.rate_history[symbol] = self.rate_history[symbol][
                                        -DEFAULT_MAX_RATE_HISTORY:
                                    ]

                                rates_updated += 1

                except Exception as e:
                    self.logger.warning(f"Failed to fetch rates from exchange data service: {e}")

            if rates_updated == 0:
                self.logger.warning("No exchange rates fetched, using fallback rates from config")
                fallback_rates = self.capital_config.get(
                    "fallback_exchange_rates",
                    {
                        "BTC/USDT": "45000",
                        "ETH/USDT": "3000",
                        "USDC/USDT": "1.0",
                        "BUSD/USDT": "1.0",
                        "ETH/BTC": "0.0667",
                        "USDC/BTC": "0.000022",
                        "BUSD/BTC": "0.000022",
                    },
                )

                for pair, rate_str in fallback_rates.items():
                    self.exchange_rates[pair] = safe_decimal_conversion(rate_str)
                    rates_updated += 1

            self.logger.info(f"Updated {rates_updated} exchange rates")

            max_history_exceeded = any(
                len(history) > MAX_RATE_HISTORY_PER_SYMBOL for history in self.rate_history.values()
            )
            if max_history_exceeded:
                self._cleanup_required = True

        except Exception as e:
            self.logger.error("Failed to update exchange rates", error=str(e))
        finally:
            tickers = None
            if self._cleanup_required:
                try:
                    await self._cleanup_rate_history()
                except Exception as cleanup_error:
                    self.logger.warning(f"Rate history cleanup failed: {cleanup_error}")
                finally:
                    self._cleanup_required = False

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

            for currency, change in required_changes.items():
                if currency == self.base_currency:
                    optimized_changes[currency] = change
                    continue

                current_exposure = self.currency_exposures.get(currency)
                if current_exposure and current_exposure.total_exposure > 0:
                    if change > 0:
                        available = current_exposure.total_exposure
                        if available >= change:
                            # No conversion needed
                            optimized_changes[currency] = Decimal("0")
                            continue
                        else:
                            optimized_changes[currency] = change - available
                    else:
                        optimized_changes[currency] = change
                else:
                    optimized_changes[currency] = change

            return optimized_changes

        except Exception as e:
            self.logger.error("Failed to optimize conversions", error=str(e))
            raise ServiceError(f"Currency conversion optimization failed: {e}") from e

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

        rate = self.exchange_rates.get(f"{from_currency}/{to_currency}")
        if rate:
            return rate

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
            hedge_amount=format_currency(hedge_amount),
        )

    async def get_hedging_summary(self) -> dict[str, Any]:
        """Get summary of current hedging status."""
        total_hedge_value = sum(self.hedge_positions.values())
        currencies_hedged = len(
            [curr for curr, exp in self.currency_exposures.items() if exp.hedging_required]
        )

        return {
            "total_hedge_value": total_hedge_value,
            "currencies_hedged": currencies_hedged,
            "hedging_enabled": self.capital_config.get("hedging_enabled", False),
            "hedging_threshold": self.hedging_threshold,
            "hedge_ratio": self.hedge_ratio,
            "hedge_positions": {curr: amount for curr, amount in self.hedge_positions.items()},
        }

    async def _cleanup_rate_history(self) -> None:
        """Clean up rate history to prevent memory leaks."""
        try:
            for symbol in self.rate_history:
                if len(self.rate_history[symbol]) > MAX_RATE_HISTORY_PER_SYMBOL:
                    self.rate_history[symbol] = self.rate_history[symbol][
                        -MAX_RATE_HISTORY_PER_SYMBOL:
                    ]

            self.logger.info("Rate history cleanup completed")
        except Exception as e:
            self.logger.warning(f"Rate history cleanup failed: {e}")

    async def cleanup_resources(self) -> None:
        """Clean up all resources to prevent memory leaks with proper async handling."""
        from src.utils.capital_resources import async_cleanup_resources, get_resource_manager

        try:
            resource_manager = get_resource_manager()

            # Clean rate history using resource manager - make it async-safe
            async def clean_rate_history():
                self.rate_history = resource_manager.clean_time_based_data(
                    self.rate_history,
                    max_age_days=RATE_HISTORY_MAX_AGE_DAYS,
                    max_items_per_key=MAX_RATE_HISTORY_PER_SYMBOL,
                )

            # Clear old currency exposures (keep only recent ones) - make it async-safe
            async def clean_currency_exposures():
                current_time = datetime.now(timezone.utc)
                old_exposures = [
                    curr
                    for curr, exp in self.currency_exposures.items()
                    if (current_time - exp.timestamp).days > 7
                ]
                for curr in old_exposures:
                    del self.currency_exposures[curr]

            # Use common cleanup utility to reduce duplication
            await async_cleanup_resources(
                clean_rate_history(), clean_currency_exposures(), logger_instance=self.logger
            )

            self.logger.info("Resource cleanup completed")
        except Exception as e:
            self.logger.warning(f"Resource cleanup failed: {e}")
        finally:
            # Mark cleanup as not required
            self._cleanup_required = False

    # Risk integration helper methods

    async def _calculate_currency_volatility_via_risk_service(
        self, currency: str, rate_history: list[tuple[datetime, Decimal]]
    ) -> Decimal:
        """Calculate currency volatility using proper utils functions."""
        try:
            from src.utils.math_utils import calculate_volatility

            # Extract just the rates for volatility calculation
            if len(rate_history) > 1:
                rates = [
                    rate for _, rate in rate_history[-RATE_CALCULATION_LOOKBACK_DAYS:]
                ]  # Last N data points
                if len(rates) > 1:
                    # Use proper volatility calculation from utils
                    volatility = calculate_volatility(rates)
                    return volatility
            return Decimal("0.0")
        except Exception as e:
            self.logger.warning(f"Volatility calculation failed: {e}")
            return Decimal("0.0")

    async def _calculate_currency_var_via_risk_service(
        self, exposure_amount: Decimal, rate_history: list[tuple[datetime, Decimal]]
    ) -> Decimal:
        """Calculate currency VaR using proper utils functions."""
        try:
            from src.utils.math_utils import calculate_var

            # Extract returns from rate history for VaR calculation
            if len(rate_history) > 1:
                rates = [rate for _, rate in rate_history]
                if len(rates) > 1:
                    # Calculate returns from rates
                    returns = []
                    for i in range(1, len(rates)):
                        if rates[i-1] > 0:
                            return_val = (rates[i] - rates[i-1]) / rates[i-1]
                            returns.append(return_val)

                    if returns:
                        # Use proper VaR calculation from utils (95% confidence level)
                        var_return = calculate_var(returns, Decimal("0.95"))
                        # Apply to exposure amount
                        var_95 = exposure_amount * abs(var_return)
                        return var_95

            return Decimal("0.0")
        except Exception as e:
            self.logger.warning(f"VaR calculation failed: {e}")
            return Decimal("0.0")

    def _fallback_risk_calculation(
        self,
        currency: str,
        exposure: CurrencyExposure,
        rate_history: list[tuple[datetime, Decimal]],
    ) -> tuple[Decimal, Decimal]:
        """Fallback risk calculation when RiskService is not available."""
        try:
            from src.utils.math_utils import calculate_volatility, calculate_var

            volatility = Decimal("0.0")
            var_95 = Decimal("0.0")

            if len(rate_history) > 1:
                # Last N data points
                rates = [rate for _, rate in rate_history[-RATE_CALCULATION_LOOKBACK_DAYS:]]
                if len(rates) > 1:
                    # Use proper volatility calculation from utils
                    volatility = calculate_volatility(rates)

                    # Calculate returns for VaR
                    returns = []
                    for i in range(1, len(rates)):
                        if rates[i-1] > 0:
                            return_val = (rates[i] - rates[i-1]) / rates[i-1]
                            returns.append(return_val)

                    if returns:
                        # Use proper VaR calculation from utils (95% confidence level)
                        var_return = calculate_var(returns, Decimal("0.95"))
                        var_95 = exposure.base_currency_equivalent * abs(var_return)
                    else:
                        var_95 = exposure.base_currency_equivalent * (
                            volatility * DEFAULT_VaR_CONFIDENCE_MULTIPLIER
                        )

            return volatility, var_95
        except Exception:
            return Decimal("0.0"), Decimal("0.0")

    async def _validate_currencies(self, balances: dict[str, dict[str, Decimal]]) -> None:
        """Validate all currencies against supported list."""
        supported_currencies = get_supported_currencies(self.capital_config)
        for _exchange, exchange_balances in balances.items():
            for currency, _amount in exchange_balances.items():
                validate_supported_currencies(currency, supported_currencies, "CurrencyManager")

    def _calculate_total_exposures(
        self, balances: dict[str, dict[str, Decimal]]
    ) -> dict[str, Decimal]:
        """Calculate total exposures across all exchanges."""
        total_exposures = {}
        for _exchange, exchange_balances in balances.items():
            for currency, amount in exchange_balances.items():
                if currency not in total_exposures:
                    total_exposures[currency] = Decimal("0")
                total_exposures[currency] += amount
        return total_exposures

    def _calculate_base_equivalents(
        self, total_exposures: dict[str, Decimal]
    ) -> tuple[dict[str, Decimal], Decimal]:
        """Calculate base currency equivalents for all exposures."""
        base_equivalents = {}
        total_base_value = Decimal("0")

        for currency, amount in total_exposures.items():
            if currency == self.base_currency:
                base_equivalent = amount
            else:
                rate = self.exchange_rates.get(f"{currency}/{self.base_currency}", Decimal("1"))
                base_equivalent = amount * rate

            base_equivalents[currency] = base_equivalent
            total_base_value += base_equivalent

        return base_equivalents, total_base_value

    def _create_currency_exposures(
        self,
        total_exposures: dict[str, Decimal],
        base_equivalents: dict[str, Decimal],
        total_base_value: Decimal,
    ) -> dict[str, CurrencyExposure]:
        """Create currency exposure objects."""
        exposures = {}
        hedging_enabled = self.capital_config.get("hedging_enabled", False)

        for currency, amount in total_exposures.items():
            base_equivalent = base_equivalents[currency]
            exposure_percentage = (
                base_equivalent / total_base_value if total_base_value > 0 else Decimal("0.0")
            )

            hedging_required = (
                hedging_enabled
                and currency != self.base_currency
                and exposure_percentage > self.hedging_threshold
            )

            hedge_amount = Decimal("0")
            if hedging_required:
                hedge_amount = base_equivalent * safe_decimal_conversion(self.hedge_ratio)

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

        return exposures

    def _log_exposure_update(
        self, exposures: dict[str, CurrencyExposure], total_base_value: Decimal
    ) -> None:
        """Log currency exposure update results."""
        self.logger.info(
            "Currency exposures updated",
            total_currencies=len(exposures),
            total_base_value=format_currency(total_base_value),
            hedging_required=sum(1 for e in exposures.values() if e.hedging_required),
        )
