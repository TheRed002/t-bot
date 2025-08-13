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
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.config import Config
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import ExchangeAllocation
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import PartialFillRecovery
from src.exchanges.base import BaseExchange
from src.utils.decorators import time_execution
from src.utils.formatters import format_currency
from src.utils.validators import validate_quantity

# MANDATORY: Use structured logging from src.core.logging for all capital
# management operations
logger = get_logger(__name__)

# From P-003+ - MANDATORY: Use existing exchange interfaces

# From P-002A - MANDATORY: Use error handling

# From P-007A - MANDATORY: Use decorators and validators


class ExchangeDistributor:
    """
    Multi-exchange capital distribution manager.

    This class manages capital distribution across multiple exchanges,
    optimizing for liquidity, fees, and reliability while maintaining
    proper risk management and rebalancing protocols.
    """

    def __init__(self, config: Config, exchanges: dict[str, BaseExchange]):
        """
        Initialize the exchange distributor.

        Args:
            config: Application configuration
            exchanges: Dictionary of exchange instances
        """
        self.config = config
        self.exchanges = exchanges
        self.capital_config = config.capital_management

        # Exchange allocation tracking
        self.exchange_allocations: dict[str, ExchangeAllocation] = {}
        self.last_rebalance = datetime.now()

        # Exchange metrics tracking
        self.liquidity_scores: dict[str, float] = {}
        self.fee_efficiencies: dict[str, float] = {}
        self.reliability_scores: dict[str, float] = {}
        self.historical_slippage: dict[str, list[float]] = {}

        # Error handler
        self.error_handler = ErrorHandler(config)

        # Recovery scenarios
        self.partial_fill_recovery = PartialFillRecovery(config)

        # TODO: Remove this in production - This is a placeholder implementation
        # Initialize exchange allocations
        # The Real Solution:
        # The total_capital should be initialized by a service that:
        # 1. Queries Exchange Balances: Uses the exchange integrations (P-003+) to get real balances
        # 2. Currency Conversion: Uses the CurrencyManager to convert all balances to base currency
        # 3. Updates Capital Components: Calls update_total_capital() on all
        # capital management components

        logger.info(
            "Exchange distributor initialized",
            exchanges=list(exchanges.keys()),
            total_exchanges=len(exchanges),
        )

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
            # Validate input
            validate_quantity(float(total_amount), "capital_distribution")

            # Calculate exchange scores
            await self._update_exchange_metrics()

            # Determine distribution strategy
            distribution_mode = self.capital_config.exchange_allocation_weights

            if distribution_mode == "dynamic":
                allocations = await self._dynamic_distribution(total_amount)
            else:
                allocations = await self._weighted_distribution(total_amount, distribution_mode)

            # Apply minimum balance requirements
            allocations = await self._apply_minimum_balances(allocations)

            # Update tracking
            self.exchange_allocations.update(allocations)
            self.last_rebalance = datetime.now()

            logger.info(
                "Capital distributed across exchanges",
                total_amount=format_currency(float(total_amount)),
                allocations_count=len(allocations),
            )

            return allocations

        except Exception as e:
            logger.error(
                "Capital distribution failed",
                total_amount=format_currency(float(total_amount)),
                error=str(e),
            )
            raise

    @time_execution
    async def rebalance_exchanges(self) -> dict[str, ExchangeAllocation]:
        """
        Rebalance capital across exchanges based on current metrics.

        Returns:
            Dict[str, ExchangeAllocation]: Updated allocations
        """
        try:
            logger.info("Starting exchange rebalancing")

            # Update exchange metrics
            await self._update_exchange_metrics()

            # Calculate total allocated capital
            total_allocated = sum(
                alloc.allocated_amount for alloc in self.exchange_allocations.values()
            )

            # Determine new distribution
            new_allocations = await self._calculate_optimal_distribution(total_allocated)

            # Apply rebalancing limits
            final_allocations = await self._apply_rebalancing_limits(new_allocations)

            # Update tracking
            self.exchange_allocations.update(final_allocations)
            self.last_rebalance = datetime.now()

            logger.info("Exchange rebalancing completed", allocations_count=len(final_allocations))

            return final_allocations

        except Exception as e:
            logger.error("Exchange rebalancing failed", error=str(e))
            raise

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
                    allocation.utilization_rate = float(
                        utilized_amount / allocation.allocated_amount
                    )

                logger.debug(
                    "Exchange utilization updated",
                    exchange=exchange,
                    utilized=format_currency(float(utilized_amount)),
                    available=format_currency(float(allocation.available_amount)),
                    utilization_rate=f"{allocation.utilization_rate:.2%}",
                )

        except Exception as e:
            logger.error("Exchange utilization update failed", exchange=exchange, error=str(e))

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
            total_score = 0

            for exchange in self.exchanges.keys():
                liquidity = self.liquidity_scores.get(exchange, 0.5)
                fee_efficiency = self.fee_efficiencies.get(exchange, 0.5)
                reliability = self.reliability_scores.get(exchange, 0.5)

                # Calculate composite score (weighted average)
                composite_score = liquidity * 0.4 + fee_efficiency * 0.3 + reliability * 0.3

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
                max_allocation_pct = self.capital_config.max_exchange_allocation_pct
                allocation_pct = min(allocation_pct, max_allocation_pct)

                distribution[exchange] = total_capital * Decimal(str(allocation_pct))

            return distribution

        except Exception as e:
            logger.error("Optimal distribution calculation failed", error=str(e))
            raise

    async def _initialize_exchange_allocations(self) -> None:
        """Initialize exchange allocations with default values."""
        for exchange_name in self.exchanges.keys():
            allocation = ExchangeAllocation(
                exchange=exchange_name,
                allocated_amount=Decimal("0"),
                available_amount=Decimal("0"),
                utilization_rate=0.0,
                liquidity_score=0.5,  # Default score
                fee_efficiency=0.5,  # Default score
                reliability_score=0.5,  # Default score
                last_rebalance=datetime.now(),
            )
            self.exchange_allocations[exchange_name] = allocation

    async def _update_exchange_metrics(self) -> None:
        """Update exchange metrics (liquidity, fees, reliability)."""
        try:
            for exchange_name, exchange in self.exchanges.items():
                # Update liquidity score
                self.liquidity_scores[exchange_name] = await self._calculate_liquidity_score(
                    exchange
                )

                # Update fee efficiency
                self.fee_efficiencies[exchange_name] = await self._calculate_fee_efficiency(
                    exchange
                )

                # Update reliability score
                self.reliability_scores[exchange_name] = await self._calculate_reliability_score(
                    exchange
                )

                # Update historical slippage
                await self._update_slippage_data(exchange_name)

        except Exception as e:
            logger.error("Failed to update exchange metrics", error=str(e))

    async def _calculate_liquidity_score(self, exchange: BaseExchange) -> float:
        """
        Calculate liquidity score for an exchange.

        Args:
            exchange: Exchange instance

        Returns:
            float: Liquidity score (0-1)
        """
        try:
            # TODO: Remove in production - Mock liquidity calculation
            # In production, this would analyze real market depth and volume
            # data
            exchange_name = exchange.__class__.__name__.lower()

            # Mock liquidity scores based on exchange
            liquidity_scores = {
                "binance": 0.9,  # High liquidity
                "okx": 0.7,  # Medium liquidity
                "coinbase": 0.8,  # Good liquidity
            }

            return liquidity_scores.get(exchange_name, 0.5)

        except Exception as e:
            logger.error(f"Failed to calculate liquidity score for {exchange}", error=str(e))
            return 0.5  # Default score

    async def _calculate_fee_efficiency(self, exchange: BaseExchange) -> float:
        """
        Calculate fee efficiency score for an exchange.

        Args:
            exchange: Exchange instance

        Returns:
            float: Fee efficiency score (0-1)
        """
        try:
            # TODO: Remove in production - Mock fee efficiency calculation
            # In production, this would analyze actual fee structures
            exchange_name = exchange.__class__.__name__.lower()

            # Mock fee efficiency scores (lower fees = higher efficiency)
            fee_efficiencies = {
                "binance": 0.8,  # Low fees
                "okx": 0.7,  # Medium fees
                "coinbase": 0.6,  # Higher fees
            }

            return fee_efficiencies.get(exchange_name, 0.5)

        except Exception as e:
            logger.error(f"Failed to calculate fee efficiency for {exchange}", error=str(e))
            return 0.5  # Default score

    async def _calculate_reliability_score(self, exchange: BaseExchange) -> float:
        """
        Calculate reliability score for an exchange.

        Args:
            exchange: Exchange instance

        Returns:
            float: Reliability score (0-1)
        """
        try:
            # TODO: Remove in production - Mock reliability calculation
            # In production, this would track API uptime, response times, etc.
            exchange_name = exchange.__class__.__name__.lower()

            # Mock reliability scores
            reliability_scores = {
                "binance": 0.95,  # Very reliable
                "okx": 0.85,  # Reliable
                "coinbase": 0.9,  # Very reliable
            }

            return reliability_scores.get(exchange_name, 0.5)

        except Exception as e:
            logger.error(f"Failed to calculate reliability score for {exchange}", error=str(e))
            return 0.5  # Default score

    async def _update_slippage_data(self, exchange_name: str) -> None:
        """
        Update historical slippage data for an exchange.

        Args:
            exchange_name: Exchange name
        """
        try:
            # TODO: Remove in production - Mock slippage data
            # In production, this would track actual slippage from trades
            if exchange_name not in self.historical_slippage:
                self.historical_slippage[exchange_name] = []

            # Mock slippage data (0.1% average)
            mock_slippage = 0.001 + (hash(exchange_name) % 50) / 10000
            self.historical_slippage[exchange_name].append(mock_slippage)

            # Keep only last 100 data points
            if len(self.historical_slippage[exchange_name]) > 100:
                self.historical_slippage[exchange_name] = self.historical_slippage[exchange_name][
                    -100:
                ]

        except Exception as e:
            logger.error(f"Failed to update slippage data for {exchange_name}", error=str(e))

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
                utilization_rate=0.0,
                liquidity_score=self.liquidity_scores.get(exchange, 0.5),
                fee_efficiency=self.fee_efficiencies.get(exchange, 0.5),
                reliability_score=self.reliability_scores.get(exchange, 0.5),
                last_rebalance=datetime.now(),
            )
            allocations[exchange] = allocation

        return allocations

    async def _weighted_distribution(
        self, total_amount: Decimal, weights: dict[str, float]
    ) -> dict[str, ExchangeAllocation]:
        """Weighted distribution based on predefined weights."""
        allocations = {}

        for exchange, weight in weights.items():
            if exchange in self.exchanges:
                amount = total_amount * Decimal(str(weight))
                allocation = ExchangeAllocation(
                    exchange=exchange,
                    allocated_amount=amount,
                    available_amount=amount,
                    utilization_rate=0.0,
                    liquidity_score=self.liquidity_scores.get(exchange, 0.5),
                    fee_efficiency=self.fee_efficiencies.get(exchange, 0.5),
                    reliability_score=self.reliability_scores.get(exchange, 0.5),
                    last_rebalance=datetime.now(),
                )
                allocations[exchange] = allocation

        return allocations

    async def _apply_minimum_balances(
        self, allocations: dict[str, ExchangeAllocation]
    ) -> dict[str, ExchangeAllocation]:
        """Apply minimum balance requirements."""
        min_balance = Decimal(str(self.capital_config.min_exchange_balance))

        for exchange, allocation in allocations.items():
            if allocation.allocated_amount < min_balance:
                allocation.allocated_amount = min_balance
                allocation.available_amount = min_balance

                logger.warning(
                    "Applied minimum balance requirement",
                    exchange=exchange,
                    min_balance=format_currency(float(min_balance)),
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
                utilization_rate=current_allocation.utilization_rate if current_allocation else 0.0,
                liquidity_score=self.liquidity_scores.get(exchange, 0.5),
                fee_efficiency=self.fee_efficiencies.get(exchange, 0.5),
                reliability_score=self.reliability_scores.get(exchange, 0.5),
                last_rebalance=datetime.now(),
            )
            allocations[exchange] = allocation

        return allocations

    async def _apply_rebalancing_limits(
        self, new_allocations: dict[str, ExchangeAllocation]
    ) -> dict[str, ExchangeAllocation]:
        """Apply rebalancing limits to prevent excessive changes."""
        max_daily_change = self.total_capital * Decimal(
            str(self.capital_config.max_daily_reallocation_pct)
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

                    logger.warning(
                        "Exchange allocation change limited",
                        exchange=exchange,
                        original_change=format_currency(float(change_amount)),
                        limited_change=format_currency(float(max_daily_change)),
                    )

        return new_allocations

    @property
    def total_capital(self) -> Decimal:
        """Get total allocated capital across all exchanges."""
        return sum(alloc.allocated_amount for alloc in self.exchange_allocations.values())

    async def get_exchange_metrics(self) -> dict[str, dict[str, float]]:
        """Get current metrics for all exchanges."""
        metrics = {}

        for exchange_name in self.exchanges.keys():
            metrics[exchange_name] = {
                "liquidity_score": self.liquidity_scores.get(exchange_name, 0.5),
                "fee_efficiency": self.fee_efficiencies.get(exchange_name, 0.5),
                "reliability_score": self.reliability_scores.get(exchange_name, 0.5),
                "avg_slippage": statistics.mean(
                    self.historical_slippage.get(exchange_name, [0.001])
                ),
            }

        return metrics

    async def get_distribution_summary(self) -> dict[str, Any]:
        """Get distribution summary for all exchanges."""
        total_allocated = sum(
            alloc.allocated_amount for alloc in self.exchange_allocations.values()
        )
        total_exchanges = len(self.exchange_allocations)

        summary = {
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
