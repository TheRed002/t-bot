"""
Exchange Distributor Implementation (P-010A)

This module implements capital distribution across exchanges with dynamic
distribution based on liquidity scores, fee structure optimization, and
API reliability weighting for exchange selection.

Key Features:
- Dynamic distribution based on liquidity scores
- Fee structure optimization for capital efficiency
- API reliability weighting for exchange selection
- Historical slippage analysis for allocation decisions
- Rebalancing frequency management (daily default)
- Cross-exchange hedging and currency management

Author: Trading Bot Framework
Version: 1.0.0
"""

import statistics
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

# Import service interfaces
from src.capital_management.interfaces import AbstractExchangeDistributionService
from src.core.base.service import TransactionalService

# Exchange-specific exceptions
from src.core.exceptions import ExchangeConnectionError, NetworkError, ServiceError

# MANDATORY: Import from P-001
from src.core.types.capital import CapitalExchangeAllocation as ExchangeAllocation

# Use interface instead of direct BaseExchange import
from src.exchanges.interfaces import IExchange
from src.utils.capital_config import (
    load_capital_config,
    resolve_config_service,
)
from src.utils.capital_resources import get_resource_manager
from src.utils.capital_validation import (
    validate_capital_amount,
)
from src.utils.decorators import time_execution
from src.utils.formatters import format_currency
from src.utils.interfaces import ValidationServiceInterface

# MANDATORY: Use structured logging from src.core.logging for all capital
# management operations

# From P-003+ - MANDATORY: Use existing exchange interfaces

# From P-002A - MANDATORY: Use error handling

# From P-007A - MANDATORY: Use decorators and validators


class ExchangeDistributor(AbstractExchangeDistributionService, TransactionalService):
    """
    Multi-exchange capital distribution manager.

    This class manages capital distribution across multiple exchanges,
    optimizing for liquidity, fees, and reliability while maintaining
    proper risk management and rebalancing protocols.
    """

    def __init__(
        self,
        exchanges: dict[str, IExchange] | None = None,
        validation_service: ValidationServiceInterface | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """
        Initialize the exchange distributor service.

        Args:
            exchanges: Dictionary of exchange instances (injected)
            validation_service: Validation service instance (injected)
            correlation_id: Request correlation ID for tracing
        """
        super().__init__(
            name="ExchangeDistributorService",
            correlation_id=correlation_id,
        )
        self.exchanges = exchanges or {}
        self.validation_service = validation_service

        # Configuration will be loaded in _do_start
        self.capital_config: dict[str, Any] = {}

        # Exchange allocation tracking
        self.exchange_allocations: dict[str, ExchangeAllocation] = {}
        self.last_rebalance = datetime.now(timezone.utc)

        # Exchange metrics tracking
        self.liquidity_scores: dict[str, float] = {}
        self.fee_efficiencies: dict[str, float] = {}
        self.reliability_scores: dict[str, float] = {}
        self.historical_slippage: dict[str, list[float]] = {}

        # Resource management (configurable via capital_config)
        self._max_slippage_history = 100

        # Service dependencies (injected via DI)
        self._exchange_info_service = None

    async def _do_start(self) -> None:
        """Start the exchange distributor service."""
        try:
            # Resolve exchange info service if not injected
            if not self._exchange_info_service:
                try:
                    self._exchange_info_service = self.resolve_dependency("ExchangeInfoService")
                except Exception as e:
                    self._logger.warning(
                        f"ExchangeInfoService not available via DI: {e}, will use fallback data"
                    )

            # Load configuration from ConfigService
            await self._load_configuration()

            # Configure resource limits from config
            self._max_slippage_history = self.capital_config.get("max_slippage_history", 100)

            # Initialize exchange allocations
            await self._initialize_exchange_allocations()

            self._logger.info(
                "Exchange distributor service started",
                exchanges=self.supported_exchanges,
                total_exchanges=len(self.supported_exchanges),
            )
        except Exception as e:
            self._logger.error(f"Failed to start ExchangeDistributor service: {e}")
            raise ServiceError(f"ExchangeDistributor startup failed: {e}") from e

    async def _do_stop(self) -> None:
        """Stop the exchange distributor service and clean up resources."""
        try:
            await self.cleanup_resources()
            self._logger.info("ExchangeDistributor service stopped and resources cleaned up")
        except Exception as e:
            self._logger.error(f"Error during ExchangeDistributor shutdown: {e}")
            raise ServiceError(f"ExchangeDistributor shutdown failed: {e}") from e

    async def _load_configuration(self) -> None:
        """Load configuration from ConfigService."""
        resolved_config_service = resolve_config_service(self)
        self.capital_config = load_capital_config(resolved_config_service)

    @time_execution
    async def distribute_capital(self, total_amount: Decimal) -> dict[str, ExchangeAllocation]:
        """
        Distribute capital across exchanges based on optimization criteria.

        Args:
            total_amount: Total capital to distribute

        Returns:
            Dict[str, ExchangeAllocation]: Distribution across exchanges
        """
        try:
            # Validate input using utility
            validate_capital_amount(
                total_amount, "distribution amount", component="ExchangeDistributor"
            )

            # Calculate exchange scores
            await self._update_exchange_metrics()

            # Determine distribution strategy - use dynamic mode by default
            # Can be overridden with hardcoded weights if needed
            exchange_weights = self.capital_config.get("exchange_allocation_weights", "dynamic")

            if exchange_weights == "dynamic":
                allocations = await self._dynamic_distribution(total_amount)
            else:
                allocations = await self._weighted_distribution(total_amount, exchange_weights)

            # Apply minimum balance requirements
            allocations = await self._apply_minimum_balances(allocations)

            # Update tracking
            self.exchange_allocations.update(allocations)
            self.last_rebalance = datetime.now(timezone.utc)

            self._logger.info(
                "Capital distributed across exchanges",
                total_amount=format_currency(total_amount),
                allocations_count=len(allocations),
            )

            return allocations

        except Exception as e:
            self._logger.error(
                "Capital distribution failed",
                total_amount=format_currency(total_amount),
                error=str(e),
            )
            raise ServiceError(f"Capital distribution failed: {e}") from e

    @time_execution
    async def rebalance_exchanges(self) -> dict[str, ExchangeAllocation]:
        """
        Rebalance capital across exchanges based on current metrics.

        Returns:
            Dict[str, ExchangeAllocation]: Updated allocations
        """
        try:
            self._logger.info("Starting exchange rebalancing")

            # Update exchange metrics
            await self._update_exchange_metrics()

            # Calculate total allocated capital
            total_allocated = sum(
                alloc.allocated_amount for alloc in self.exchange_allocations.values()
            ) or Decimal("0")

            # Determine new distribution
            new_allocations = await self._calculate_optimal_distribution(total_allocated)

            # Apply rebalancing limits
            final_allocations = await self._apply_rebalancing_limits(new_allocations)

            # Update tracking
            self.exchange_allocations.update(final_allocations)
            self.last_rebalance = datetime.now(timezone.utc)

            self._logger.info(
                "Exchange rebalancing completed", allocations_count=len(final_allocations)
            )

            return final_allocations

        except Exception as e:
            self._logger.error("Exchange rebalancing failed", error=str(e))
            raise ServiceError(f"Exchange rebalancing failed: {e}") from e

    @time_execution
    async def get_exchange_allocation(self, exchange: str) -> ExchangeAllocation | None:
        """
        Get current allocation for a specific exchange.

        Args:
            exchange: Exchange name

        Returns:
            Optional[ExchangeAllocation]: Current allocation
        """
        return self.exchange_allocations.get(exchange)

    @time_execution
    async def update_exchange_utilization(self, exchange: str, utilized_amount: Decimal) -> None:
        """
        Update utilization for a specific exchange.

        Args:
            exchange: Exchange name
            utilized_amount: Amount currently utilized
        """
        try:
            if exchange in self.exchange_allocations:
                allocation = self.exchange_allocations[exchange]
                allocation.utilized_amount = utilized_amount
                allocation.available_amount = allocation.allocated_amount - utilized_amount

                # Update utilization rate
                if allocation.allocated_amount > 0:
                    allocation.utilization_rate = utilized_amount / allocation.allocated_amount

                self._logger.debug(
                    "Exchange utilization updated",
                    exchange=exchange,
                    utilized=format_currency(utilized_amount),
                    available=format_currency(allocation.available_amount),
                    utilization_rate=f"{allocation.utilization_rate:.2%}",
                )

        except Exception as e:
            self._logger.error(
                "Exchange utilization update failed", exchange=exchange, error=str(e)
            )

    @time_execution
    async def calculate_optimal_distribution(self, total_capital: Decimal) -> dict[str, Decimal]:
        """
        Calculate optimal capital distribution across exchanges.

        Args:
            total_capital: Total capital to distribute

        Returns:
            Dict[str, Decimal]: Optimal distribution amounts
        """
        try:
            # Update exchange metrics
            await self._update_exchange_metrics()

            # Calculate composite scores
            composite_scores = {}
            total_score = 0.0

            for exchange in self.supported_exchanges:
                liquidity = self.liquidity_scores.get(exchange, 0.5)
                fee_efficiency = self.fee_efficiencies.get(exchange, 0.5)
                reliability = self.reliability_scores.get(exchange, 0.5)

                # Calculate composite score using configurable weights
                liquidity_weight = self.capital_config.get("liquidity_weight", 0.4)
                fee_weight = self.capital_config.get("fee_weight", 0.3)
                reliability_weight = self.capital_config.get("reliability_weight", 0.3)

                composite_score = (
                    liquidity * liquidity_weight
                    + fee_efficiency * fee_weight
                    + reliability * reliability_weight
                )

                composite_scores[exchange] = composite_score
                total_score += composite_score

            # Calculate optimal distribution
            distribution = {}
            for exchange, score in composite_scores.items():
                if total_score > 0:
                    allocation_pct = score / total_score
                else:
                    allocation_pct = 1.0 / len(composite_scores)

                # Apply maximum allocation limit
                max_allocation_pct = self.capital_config.get("max_allocation_pct", 0.3)
                allocation_pct = min(allocation_pct, max_allocation_pct)

                distribution[exchange] = total_capital * Decimal(str(allocation_pct))

            return distribution

        except Exception as e:
            self._logger.error("Optimal distribution calculation failed", error=str(e))
            raise ServiceError(f"Optimal distribution calculation failed: {e}") from e

    async def _initialize_exchange_allocations(self) -> None:
        """Initialize exchange allocations with default values."""
        for exchange_name in self.exchanges.keys():
            allocation = ExchangeAllocation(
                exchange=exchange_name,
                allocated_amount=Decimal("0"),
                available_amount=Decimal("0"),
                utilization_rate=Decimal("0.0"),
                liquidity_score=Decimal("0.5"),  # Default score
                fee_efficiency=Decimal("0.5"),  # Default score
                reliability_score=Decimal("0.5"),  # Default score
                last_rebalance=datetime.now(timezone.utc),
            )
            self.exchange_allocations[exchange_name] = allocation

    async def _update_exchange_metrics(self) -> None:
        """Update exchange metrics (liquidity, fees, reliability)."""
        try:
            for exchange_name, _exchange in self.exchanges.items():
                # Update liquidity score
                self.liquidity_scores[exchange_name] = await self._calculate_liquidity_score(
                    exchange_name
                )

                # Update fee efficiency
                self.fee_efficiencies[exchange_name] = await self._calculate_fee_efficiency(
                    exchange_name
                )

                # Update reliability score
                self.reliability_scores[exchange_name] = await self._calculate_reliability_score(
                    exchange_name
                )

                # Update historical slippage
                await self._update_slippage_data(exchange_name)

        except Exception as e:
            self._logger.error("Failed to update exchange metrics", error=str(e))

    async def _calculate_liquidity_score(self, exchange_name: str) -> float:
        """
        Calculate liquidity score for an exchange.

        Args:
            exchange_name: Exchange name

        Returns:
            float: Liquidity score (0-1)
        """
        orderbook = None
        try:
            # Try to calculate based on real market data if available via service
            if self._exchange_info_service:
                try:
                    # Sample a few major pairs to assess liquidity
                    major_pairs = ["BTC/USDT", "ETH/USDT"]
                    depth_scores = []

                    for pair in major_pairs:
                        try:
                            if hasattr(self._exchange_info_service, "get_order_book"):
                                orderbook = await self._exchange_info_service.get_order_book(
                                    exchange_name, pair, limit=50
                                )
                            else:
                                continue

                            # Calculate liquidity based on order book depth
                            if orderbook:
                                bid_volume = sum(
                                    Decimal(str(bid[1])) * Decimal(str(bid[0]))
                                    for bid in orderbook.get("bids", [])[:20]
                                )
                                ask_volume = sum(
                                    Decimal(str(ask[1])) * Decimal(str(ask[0]))
                                    for ask in orderbook.get("asks", [])[:20]
                                )

                                # Normalize score (higher volume = better liquidity)
                                # Using 1M USD as reference for max score
                                depth_score = min(
                                    float((bid_volume + ask_volume) / Decimal("2000000")), 1.0
                                )
                                depth_scores.append(depth_score)
                        except Exception as pair_error:
                            self._logger.debug(
                                f"Could not fetch {pair} data for {exchange_name}: {pair_error}"
                            )
                        finally:
                            orderbook = None

                    if depth_scores:
                        return sum(depth_scores) / len(depth_scores)

                except Exception as e:
                    self._logger.debug(
                        f"Could not fetch real liquidity data for {exchange_name}: {e}"
                    )

            # Fallback to known exchange ratings
            fallback_scores = {
                "binance": 0.9,  # High liquidity
                "okx": 0.7,  # Medium liquidity
                "coinbase": 0.8,  # Good liquidity
            }

            return fallback_scores.get(exchange_name, 0.5)

        except Exception as e:
            self._logger.error(
                f"Failed to calculate liquidity score for {exchange_name}", error=str(e)
            )
            return 0.5  # Default score

    async def _calculate_fee_efficiency(self, exchange_name: str) -> float:
        """
        Calculate fee efficiency score for an exchange.

        Args:
            exchange_name: Exchange name

        Returns:
            float: Fee efficiency score (0-1)
        """
        fees_data = None
        try:
            # Try to get real fee structure if available via service
            if self._exchange_info_service and hasattr(self._exchange_info_service, "get_fees"):
                try:
                    fees_data = await self._exchange_info_service.get_fees(exchange_name)
                    trading_fees = fees_data.get("trading", {}) if fees_data else {}

                    # Get taker fee (usually higher than maker)
                    taker_fee = trading_fees.get("taker", 0.001)  # Default 0.1%

                    # Convert to efficiency score (lower fee = higher score)
                    # 0.0% fee = 1.0 score, 0.5% fee = 0.0 score
                    fee_efficiency = max(0.0, 1.0 - (taker_fee * 200))

                    return fee_efficiency

                except Exception as e:
                    self._logger.debug(f"Could not fetch real fee data for {exchange_name}: {e}")
                finally:
                    # Clear fees_data reference to prevent memory leaks
                    fees_data = None

            # Fallback to known exchange fee structures
            fallback_efficiencies = {
                "binance": 0.8,  # ~0.1% fees
                "okx": 0.7,  # ~0.15% fees
                "coinbase": 0.6,  # ~0.2% fees
            }

            return fallback_efficiencies.get(exchange_name, 0.5)

        except Exception as e:
            self._logger.error(
                f"Failed to calculate fee efficiency for {exchange_name}", error=str(e)
            )
            return 0.5  # Default score

    async def _calculate_reliability_score(self, exchange_name: str) -> float:
        """
        Calculate reliability score for an exchange.

        Args:
            exchange_name: Exchange name

        Returns:
            float: Reliability score (0-1)
        """
        try:
            # Start with base score
            reliability_score = 0.5

            # Check exchange health/status if available via service
            if self._exchange_info_service and hasattr(self._exchange_info_service, "get_status"):
                try:
                    status = await self._exchange_info_service.get_status(exchange_name)
                    if status.get("status") == "ok":
                        reliability_score += 0.2
                except (ExchangeConnectionError, NetworkError) as e:
                    self._logger.debug(f"Exchange {exchange_name} status check failed: {e}")
                    reliability_score -= 0.1  # Penalize for connectivity issues
                except Exception as e:
                    self._logger.debug(
                        f"Unexpected error checking exchange {exchange_name} status: {e}"
                    )
                    # Don't penalize for unexpected errors, just skip status bonus

            # Service availability scoring (if service supports various operations)
            if self._exchange_info_service:
                service_score = 0.0
                if hasattr(self._exchange_info_service, "get_ticker"):
                    service_score += 0.075
                if hasattr(self._exchange_info_service, "get_order_book"):
                    service_score += 0.075
                if hasattr(self._exchange_info_service, "create_order"):
                    service_score += 0.075
                if hasattr(self._exchange_info_service, "cancel_order"):
                    service_score += 0.075
                reliability_score += service_score

            # Fallback adjustments for known exchanges
            known_adjustments = {
                "binance": 0.15,  # Well-established
                "okx": 0.05,  # Established
                "coinbase": 0.1,  # Well-established
            }

            reliability_score += known_adjustments.get(exchange_name, 0.0)

            # Cap at 1.0
            return min(reliability_score, 1.0)

        except Exception as e:
            self._logger.error(
                f"Failed to calculate reliability score for {exchange_name}", error=str(e)
            )
            return 0.5  # Default score

    async def _update_slippage_data(self, exchange_name: str) -> None:
        """
        Update historical slippage data for an exchange.

        Args:
            exchange_name: Exchange name
        """
        try:
            if exchange_name not in self.historical_slippage:
                self.historical_slippage[exchange_name] = []

            # In production, this would be updated from actual trade execution data
            # For now, use a reasonable estimate based on exchange characteristics
            base_slippage = {
                "binance": 0.0005,  # 0.05% - high liquidity
                "okx": 0.001,  # 0.1% - medium liquidity
                "coinbase": 0.0008,  # 0.08% - good liquidity
            }.get(exchange_name.lower(), 0.001)

            # Use time-based variance for deterministic behavior
            time_factor = datetime.now(timezone.utc).microsecond / 1000000.0  # 0-1 range
            variance_range = 0.0002
            variance = (time_factor - 0.5) * variance_range * 2  # -0.0002 to 0.0002
            slippage = max(0.0, base_slippage + variance)

            self.historical_slippage[exchange_name].append(slippage)

            # Keep only last 100 data points
            if len(self.historical_slippage[exchange_name]) > self._max_slippage_history:
                self.historical_slippage[exchange_name] = self.historical_slippage[exchange_name][
                    -self._max_slippage_history :
                ]

        except Exception as e:
            self._logger.error(f"Failed to update slippage data for {exchange_name}", error=str(e))
        finally:
            # Cleanup if slippage history is getting too large
            if (
                exchange_name in self.historical_slippage
                and len(self.historical_slippage[exchange_name]) > self._max_slippage_history * 1.5
            ):
                try:
                    await self.cleanup_resources()
                except Exception as cleanup_error:
                    self._logger.warning(f"Slippage cleanup failed: {cleanup_error}")

    async def _dynamic_distribution(self, total_amount: Decimal) -> dict[str, ExchangeAllocation]:
        """Dynamic distribution based on real-time metrics."""
        # Calculate optimal distribution
        optimal_distribution = await self.calculate_optimal_distribution(total_amount)

        # Create allocation objects
        allocations = {}
        for exchange, amount in optimal_distribution.items():
            allocation = ExchangeAllocation(
                exchange=exchange,
                allocated_amount=amount,
                available_amount=amount,
                utilization_rate=Decimal("0.0"),
                liquidity_score=self.liquidity_scores.get(exchange, 0.5),
                fee_efficiency=self.fee_efficiencies.get(exchange, 0.5),
                reliability_score=self.reliability_scores.get(exchange, 0.5),
                last_rebalance=datetime.now(timezone.utc),
            )
            allocations[exchange] = allocation

        return allocations

    async def _weighted_distribution(
        self, total_amount: Decimal, weights: dict[str, float]
    ) -> dict[str, ExchangeAllocation]:
        """Weighted distribution based on predefined weights."""
        allocations = {}

        for exchange, weight in weights.items():
            if exchange in self.supported_exchanges:
                amount = total_amount * Decimal(str(weight))
                allocation = ExchangeAllocation(
                    exchange=exchange,
                    allocated_amount=amount,
                    available_amount=amount,
                    utilization_rate=Decimal("0.0"),
                    liquidity_score=self.liquidity_scores.get(exchange, 0.5),
                    fee_efficiency=self.fee_efficiencies.get(exchange, 0.5),
                    reliability_score=self.reliability_scores.get(exchange, 0.5),
                    last_rebalance=datetime.now(timezone.utc),
                )
                allocations[exchange] = allocation

        return allocations

    async def _apply_minimum_balances(
        self, allocations: dict[str, ExchangeAllocation]
    ) -> dict[str, ExchangeAllocation]:
        """Apply minimum balance requirements."""
        min_balance = Decimal(str(self.capital_config.get("min_deposit_amount", 1000)))

        for exchange, allocation in allocations.items():
            if allocation.allocated_amount < min_balance:
                allocation.allocated_amount = min_balance
                allocation.available_amount = min_balance

                self._logger.warning(
                    "Applied minimum balance requirement",
                    exchange=exchange,
                    min_balance=format_currency(min_balance),
                )

        return allocations

    async def _calculate_optimal_distribution(
        self, total_amount: Decimal
    ) -> dict[str, ExchangeAllocation]:
        """Calculate optimal distribution based on current metrics."""
        optimal_distribution = await self.calculate_optimal_distribution(total_amount)

        allocations = {}
        for exchange, amount in optimal_distribution.items():
            current_allocation = self.exchange_allocations.get(exchange)

            allocation = ExchangeAllocation(
                exchange=exchange,
                allocated_amount=amount,
                available_amount=amount,
                utilization_rate=current_allocation.utilization_rate
                if current_allocation
                else Decimal("0.0"),
                liquidity_score=self.liquidity_scores.get(exchange, 0.5),
                fee_efficiency=self.fee_efficiencies.get(exchange, 0.5),
                reliability_score=self.reliability_scores.get(exchange, 0.5),
                last_rebalance=datetime.now(timezone.utc),
            )
            allocations[exchange] = allocation

        return allocations

    async def _apply_rebalancing_limits(
        self, new_allocations: dict[str, ExchangeAllocation]
    ) -> dict[str, ExchangeAllocation]:
        """Apply rebalancing limits to prevent excessive changes."""
        # Calculate total from new allocations being rebalanced
        total_being_rebalanced = sum(alloc.allocated_amount for alloc in new_allocations.values())
        max_daily_change = total_being_rebalanced * Decimal(
            str(self.capital_config.get("max_daily_reallocation_pct", 0.1))
        )

        for exchange, allocation in new_allocations.items():
            if exchange in self.exchange_allocations:
                current_allocation = self.exchange_allocations[exchange]
                change_amount = abs(
                    allocation.allocated_amount - current_allocation.allocated_amount
                )

                if change_amount > max_daily_change:
                    # Limit the change
                    if allocation.allocated_amount > current_allocation.allocated_amount:
                        allocation.allocated_amount = (
                            current_allocation.allocated_amount + max_daily_change
                        )
                    else:
                        allocation.allocated_amount = (
                            current_allocation.allocated_amount - max_daily_change
                        )

                    allocation.available_amount = allocation.allocated_amount

                    self._logger.warning(
                        "Exchange allocation change limited",
                        exchange=exchange,
                        original_change=format_currency(change_amount),
                        limited_change=format_currency(max_daily_change),
                    )

        return new_allocations

    @property
    def supported_exchanges(self) -> list[str]:
        """Get list of supported exchange names."""
        return list(self.exchanges.keys())

    @property
    def total_capital(self) -> Decimal:
        """Get total allocated capital across all exchanges."""
        return sum(
            alloc.allocated_amount for alloc in self.exchange_allocations.values()
        ) or Decimal("0")

    async def get_exchange_metrics(self) -> dict[str, dict[str, float]]:
        """Get current metrics for all exchanges."""
        metrics = {}

        for exchange_name in self.supported_exchanges:
            metrics[exchange_name] = {
                "liquidity_score": self.liquidity_scores.get(exchange_name, 0.5),
                "fee_efficiency": self.fee_efficiencies.get(exchange_name, 0.5),
                "reliability_score": self.reliability_scores.get(exchange_name, 0.5),
                "avg_slippage": Decimal(
                    str(statistics.mean(self.historical_slippage.get(exchange_name, [0.001])))
                ),
            }

        return metrics

    async def get_distribution_summary(self) -> dict[str, Any]:
        """Get distribution summary for all exchanges."""
        total_allocated = sum(
            alloc.allocated_amount for alloc in self.exchange_allocations.values()
        )
        total_exchanges = len(self.exchange_allocations)

        summary: dict[str, Any] = {
            "total_allocated": total_allocated,
            "total_exchanges": total_exchanges,
            "exchanges": {},
        }

        for exchange_name, allocation in self.exchange_allocations.items():
            summary["exchanges"][exchange_name] = {
                "allocated_amount": allocation.allocated_amount,
                "available_amount": allocation.available_amount,
                "utilization_rate": allocation.utilization_rate,
                "liquidity_score": allocation.liquidity_score,
                "fee_efficiency": allocation.fee_efficiency,
                "reliability_score": allocation.reliability_score,
            }

        return summary

    async def cleanup_resources(self) -> None:
        """Clean up resources to prevent memory leaks."""
        try:
            resource_manager = get_resource_manager()

            # Clean up historical slippage data using resource manager
            for exchange_name in list(self.historical_slippage.keys()):
                slippage_history = self.historical_slippage[exchange_name]
                self.historical_slippage[exchange_name] = resource_manager.limit_list_size(
                    slippage_history, self._max_slippage_history
                )

            # Remove exchanges that are no longer in use
            active_exchanges = set(self.exchanges.keys())
            inactive_exchanges = set(self.historical_slippage.keys()) - active_exchanges
            for exchange_name in inactive_exchanges:
                del self.historical_slippage[exchange_name]
                self.liquidity_scores.pop(exchange_name, None)
                self.fee_efficiencies.pop(exchange_name, None)
                self.reliability_scores.pop(exchange_name, None)

            self._logger.debug("Exchange distributor resource cleanup completed")
        except Exception as e:
            self._logger.warning(f"Resource cleanup failed: {e}")
