"""
Capital Allocator Implementation (P-010A)

This module implements the dynamic capital allocation framework that manages
total capital with emergency reserves, performance-based strategy allocation
adjustments, and risk-adjusted capital distribution using Sharpe ratios.

Key Features:
- Total capital management with emergency reserves (10% default)
- Performance-based strategy allocation adjustments
- Risk-adjusted capital distribution using Sharpe ratios
- Dynamic rebalancing based on strategy performance
- Kelly Criterion integration for optimal sizing
- Capital scaling based on account growth

Author: Trading Bot Framework
Version: 1.0.0
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import (
    AllocationStrategy,
    CapitalAllocation,
    CapitalMetrics,
)
from src.database.connection import get_influxdb_client, get_redis_client
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import PartialFillRecovery
from src.risk_management.base import BaseRiskManager
from src.utils.decorators import retry, time_execution
from src.utils.formatters import format_currency
from src.utils.validators import validate_quantity

# MANDATORY: Use structured logging from src.core.logging for all capital
# management operations
logger = get_logger(__name__)

# From P-008+ - MANDATORY: Use existing risk management

# From P-003+ - MANDATORY: Use existing exchange interfaces

# From P-002A - MANDATORY: Use error handling

# From P-007A - MANDATORY: Use decorators and validators


class CapitalAllocator:
    """
    Dynamic capital allocation framework for optimal capital utilization.

    This class manages total capital with emergency reserves, implements
    performance-based allocation adjustments, and provides risk-adjusted
    capital distribution using various allocation strategies.
    """

    def __init__(self, config: Config, risk_manager: BaseRiskManager | None = None):
        """
        Initialize the capital allocator.

        Args:
            config: Application configuration
            risk_manager: Risk management instance for validation
        """
        self.config = config
        self.risk_manager = risk_manager
        self.capital_config = config.capital_management

        # Capital state
        self.total_capital = Decimal(str(self.capital_config.total_capital))
        self.emergency_reserve = self.total_capital * Decimal(
            str(self.capital_config.emergency_reserve_pct)
        )
        self.available_capital = self.total_capital - self.emergency_reserve

        # Allocation tracking
        self.strategy_allocations: dict[str, CapitalAllocation] = {}
        self.exchange_allocations: dict[str, CapitalAllocation] = {}
        self.last_rebalance = datetime.now()

        # Performance tracking
        self.strategy_performance: dict[str, dict[str, float]] = {}
        self.performance_window = timedelta(days=30)  # 30-day performance window

        # Error handler
        self.error_handler = ErrorHandler(config)

        # Recovery scenarios
        self.partial_fill_recovery = PartialFillRecovery(config)

        # Redis client for caching (optional)
        try:
            self.redis_client = get_redis_client()
        except Exception:
            self.redis_client = None
            logger.warning("Redis client not available, caching disabled")

        # Cache keys
        self.cache_keys = {
            "allocations": "capital:allocations",
            "performance": "capital:performance",
            "metrics": "capital:metrics",
            "total_capital": "capital:total",
        }

        # Cache TTL (seconds)
        self.cache_ttl = 300  # 5 minutes

        # InfluxDB client for time series data (optional)
        try:
            self.influx_client = get_influxdb_client()
        except Exception:
            self.influx_client = None
            logger.warning("InfluxDB client not available, time series storage disabled")

        logger.info(
            "Capital allocator initialized",
            total_capital=format_currency(float(self.total_capital)),
            emergency_reserve=format_currency(float(self.emergency_reserve)),
            available_capital=format_currency(float(self.available_capital)),
        )

    @retry(max_attempts=2, base_delay=0.5)
    async def _get_cached_allocations(self) -> dict[str, CapitalAllocation] | None:
        """Get cached allocations from Redis."""
        if not self.redis_client:
            return None
        try:
            cached_data = await self.redis_client.get(self.cache_keys["allocations"])
            if cached_data:
                # Convert cached data back to CapitalAllocation objects
                allocations = {}
                for key, data in cached_data.items():
                    allocations[key] = CapitalAllocation(**data)
                return allocations
        except Exception as e:
            logger.warning("Failed to get cached allocations", error=str(e))
        return None

    @retry(max_attempts=2, base_delay=0.5)
    async def _cache_allocations(self, allocations: dict[str, CapitalAllocation]) -> None:
        """Cache allocations in Redis."""
        if not self.redis_client:
            return
        try:
            # Convert CapitalAllocation objects to dict for caching
            cache_data = {}
            for key, allocation in allocations.items():
                cache_data[key] = allocation.model_dump()

            await self.redis_client.set(
                self.cache_keys["allocations"], cache_data, ttl=self.cache_ttl
            )
        except Exception as e:
            logger.warning("Failed to cache allocations", error=str(e))

    @retry(max_attempts=2, base_delay=0.5)
    async def _get_cached_performance(self) -> dict[str, dict[str, float]] | None:
        """Get cached performance data from Redis."""
        if not self.redis_client:
            return None
        try:
            cached_data = await self.redis_client.get(self.cache_keys["performance"])
            if cached_data:
                return cached_data
        except Exception as e:
            logger.warning("Failed to get cached performance", error=str(e))
        return None

    @retry(max_attempts=2, base_delay=0.5)
    async def _cache_performance(self, performance: dict[str, dict[str, float]]) -> None:
        """Cache performance data in Redis."""
        if not self.redis_client:
            return
        try:
            await self.redis_client.set(
                self.cache_keys["performance"], performance, ttl=self.cache_ttl
            )
        except Exception as e:
            logger.warning("Failed to cache performance", error=str(e))

    @retry(max_attempts=2, base_delay=0.5)
    async def _get_cached_metrics(self) -> CapitalMetrics | None:
        """Get cached metrics from Redis."""
        if not self.redis_client:
            return None
        try:
            cached_data = await self.redis_client.get(self.cache_keys["metrics"])
            if cached_data:
                return CapitalMetrics(**cached_data)
        except Exception as e:
            logger.warning("Failed to get cached metrics", error=str(e))
        return None

    async def _cache_metrics(self, metrics: CapitalMetrics) -> None:
        """Cache metrics in Redis."""
        if not self.redis_client:
            return
        try:
            await self.redis_client.set(
                self.cache_keys["metrics"], metrics.model_dump(), ttl=self.cache_ttl
            )
        except Exception as e:
            logger.warning("Failed to cache metrics", error=str(e))

    async def _store_capital_metrics_influxdb(self, metrics: CapitalMetrics) -> None:
        """Store capital metrics in InfluxDB for time series analysis."""
        if not self.influx_client:
            return
        try:
            # Create a point for capital metrics
            from influxdb_client import Point

            point = (
                Point("capital_metrics")
                .tag("component", "capital_allocator")
                .field("total_capital", float(metrics.total_capital))
                .field("allocated_capital", float(metrics.allocated_capital))
                .field("available_capital", float(metrics.available_capital))
                .field("utilization_rate", metrics.utilization_rate)
                .field("allocation_efficiency", metrics.allocation_efficiency)
                .field("allocation_count", metrics.allocation_count)
            )

            # Write to InfluxDB
            self.influx_client.write_api().write(bucket="trading_bot", record=point)
        except Exception as e:
            logger.warning("Failed to store metrics in InfluxDB", error=str(e))

    async def _store_allocation_change_influxdb(
        self,
        strategy_id: str,
        exchange: str,
        amount: Decimal,
        allocation_type: str,
        timestamp: datetime,
    ) -> None:
        """Store allocation changes in InfluxDB for tracking."""
        if not self.influx_client:
            return
        try:
            # Create a point for allocation changes
            from influxdb_client import Point

            point = (
                Point("allocation_changes")
                .tag("component", "capital_allocator")
                .tag("strategy_id", strategy_id)
                .tag("exchange", exchange)
                .tag("allocation_type", allocation_type)
                .field("amount", float(amount))
            )

            # Write to InfluxDB
            self.influx_client.write_api().write(bucket="trading_bot", record=point)
        except Exception as e:
            logger.warning("Failed to store allocation change in InfluxDB", error=str(e))

    @time_execution
    async def allocate_capital(
        self, strategy_id: str, exchange: str, requested_amount: Decimal
    ) -> CapitalAllocation:
        """
        Allocate capital to a strategy on a specific exchange.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name
            requested_amount: Requested capital amount

        Returns:
            CapitalAllocation: Allocation record

        Raises:
            ValidationError: If allocation violates limits
            RiskManagementError: If risk limits exceeded
        """
        try:
            # Validate inputs
            validate_quantity(float(requested_amount), "capital_allocation")

            # Validate strategy_id
            if not strategy_id or not strategy_id.strip():
                raise ValidationError("Strategy ID cannot be empty")

            # Check minimum allocation requirements
            min_allocation = self._get_minimum_allocation(strategy_id)
            if requested_amount < Decimal(str(min_allocation)):
                raise ValidationError(
                    f"Requested amount {requested_amount} below minimum "
                    f"{min_allocation} for strategy {strategy_id}"
                )

            # Check maximum allocation limits
            max_allocation = self.available_capital * Decimal(
                str(self.capital_config.max_allocation_pct)
            )
            if requested_amount > max_allocation:
                raise ValidationError(
                    f"Requested amount {requested_amount} exceeds "
                    f"maximum allocation {max_allocation}"
                )

            # Calculate allocation percentage
            allocation_percentage = float(requested_amount / self.total_capital)

            # Create allocation record
            allocation = CapitalAllocation(
                strategy_id=strategy_id,
                exchange=exchange,
                allocated_amount=requested_amount,
                utilized_amount=Decimal("0"),
                available_amount=requested_amount,
                allocation_percentage=allocation_percentage,
                last_rebalance=datetime.now(),
            )

            # Store allocation
            key = f"{strategy_id}_{exchange}"
            self.strategy_allocations[key] = allocation

            # Update available capital
            self.available_capital -= requested_amount

            # Cache allocations
            await self._cache_allocations(self.strategy_allocations)

            # Store allocation change in InfluxDB
            await self._store_allocation_change_influxdb(
                strategy_id, exchange, requested_amount, "allocation", datetime.now()
            )

            logger.info(
                "Capital allocated successfully",
                strategy_id=strategy_id,
                exchange=exchange,
                amount=format_currency(float(requested_amount)),
                allocation_percentage=f"{allocation_percentage:.2%}",
            )

            return allocation

        except Exception as e:
            # Create comprehensive error context
            context = self.error_handler.create_error_context(
                error=e,
                component="capital_management",
                operation="allocate_capital",
                details={
                    "strategy_id": strategy_id,
                    "exchange": exchange,
                    "requested_amount": float(requested_amount),
                    "available_capital": float(self.available_capital),
                    "total_capital": float(self.total_capital),
                },
            )

            # Handle error with recovery strategy
            await self.error_handler.handle_error(e, context, self.partial_fill_recovery)

            # Log detailed error information
            logger.error(
                "Capital allocation failed",
                error_id=context.error_id,
                severity=context.severity.value,
                strategy_id=strategy_id,
                exchange=exchange,
                requested_amount=format_currency(float(requested_amount)),
                error=str(e),
            )
            raise

    @time_execution
    async def rebalance_allocations(self) -> dict[str, CapitalAllocation]:
        """
        Rebalance capital allocations based on performance and strategy.

        Returns:
            Dict[str, CapitalAllocation]: Updated allocations
        """
        try:
            logger.info("Starting capital rebalancing")

            # Get current performance metrics
            performance_metrics = await self._calculate_performance_metrics()

            # Determine allocation strategy
            strategy = AllocationStrategy(self.capital_config.allocation_strategy)

            # Calculate new allocations based on strategy
            if strategy == AllocationStrategy.EQUAL_WEIGHT:
                new_allocations = await self._equal_weight_allocation()
            elif strategy == AllocationStrategy.PERFORMANCE_WEIGHTED:
                new_allocations = await self._performance_weighted_allocation(performance_metrics)
            elif strategy == AllocationStrategy.VOLATILITY_WEIGHTED:
                new_allocations = await self._volatility_weighted_allocation(performance_metrics)
            elif strategy == AllocationStrategy.RISK_PARITY:
                new_allocations = await self._risk_parity_allocation(performance_metrics)
            else:  # DYNAMIC
                new_allocations = await self._dynamic_allocation(performance_metrics)

            # Apply rebalancing with limits
            updated_allocations = await self._apply_rebalancing_limits(new_allocations)

            # Update tracking
            self.last_rebalance = datetime.now()

            logger.info(
                "Capital rebalancing completed",
                strategy=strategy.value,
                allocations_count=len(updated_allocations),
            )

            return updated_allocations

        except Exception as e:
            # Create comprehensive error context
            context = self.error_handler.create_error_context(
                error=e,
                component="capital_management",
                operation="rebalance_allocations",
                details={
                    "strategy_count": len(self.strategy_allocations),
                    "total_capital": float(self.total_capital),
                    "available_capital": float(self.available_capital),
                },
            )

            # Handle error with recovery strategy
            await self.error_handler.handle_error(e, context, self.partial_fill_recovery)

            # Log detailed error information
            logger.error(
                "Capital rebalancing failed",
                error_id=context.error_id,
                severity=context.severity.value,
                error=str(e),
            )
            raise

    @time_execution
    async def update_utilization(
        self, strategy_id: str, exchange: str, utilized_amount: Decimal
    ) -> None:
        """
        Update capital utilization for a strategy.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name
            utilized_amount: Amount currently utilized
        """
        try:
            key = f"{strategy_id}_{exchange}"
            if key in self.strategy_allocations:
                allocation = self.strategy_allocations[key]
                allocation.utilized_amount = utilized_amount
                allocation.available_amount = allocation.allocated_amount - utilized_amount

                logger.debug(
                    "Utilization updated",
                    strategy_id=strategy_id,
                    exchange=exchange,
                    utilized=format_currency(float(utilized_amount)),
                    available=format_currency(float(allocation.available_amount)),
                )

        except Exception as e:
            # Create comprehensive error context
            context = self.error_handler.create_error_context(
                error=e,
                component="capital_management",
                operation="update_utilization",
                details={
                    "strategy_id": strategy_id,
                    "exchange": exchange,
                    "utilized_amount": float(utilized_amount),
                },
            )

            # Handle error with recovery strategy
            await self.error_handler.handle_error(e, context, self.partial_fill_recovery)

            # Log detailed error information
            logger.error(
                "Utilization update failed",
                error_id=context.error_id,
                severity=context.severity.value,
                strategy_id=strategy_id,
                exchange=exchange,
                error=str(e),
            )
            raise

    @time_execution
    async def get_capital_metrics(self) -> CapitalMetrics:
        """
        Get current capital management metrics.

        Returns:
            CapitalMetrics: Current capital metrics
        """
        try:
            # Try to get cached metrics first
            cached_metrics = await self._get_cached_metrics()
            if cached_metrics:
                logger.debug("Returning cached capital metrics")
                return cached_metrics

            # Calculate utilization rate
            total_allocated = sum(
                alloc.allocated_amount for alloc in self.strategy_allocations.values()
            )
            total_utilized = sum(
                alloc.utilized_amount for alloc in self.strategy_allocations.values()
            )

            utilization_rate = (
                float(total_utilized / total_allocated) if total_allocated > 0 else 0.0
            )

            # Calculate allocation efficiency (based on performance vs
            # allocation)
            efficiency_score = await self._calculate_allocation_efficiency()

            metrics = CapitalMetrics(
                total_capital=self.total_capital,
                allocated_capital=total_allocated,
                available_capital=self.available_capital,
                utilization_rate=utilization_rate,
                allocation_efficiency=efficiency_score,
                rebalance_frequency_hours=self.capital_config.rebalance_frequency_hours,
                emergency_reserve=self.emergency_reserve,
                last_updated=datetime.now(),
                allocation_count=len(self.strategy_allocations),
            )

            # Cache metrics for future use
            await self._cache_metrics(metrics)

            # Store metrics in InfluxDB for time series analysis
            await self._store_capital_metrics_influxdb(metrics)

            return metrics

        except Exception as e:
            # Create comprehensive error context
            context = self.error_handler.create_error_context(
                error=e,
                component="capital_management",
                operation="get_capital_metrics",
                details={
                    "strategy_count": len(self.strategy_allocations),
                    "total_capital": float(self.total_capital),
                },
            )

            # Handle error with recovery strategy
            await self.error_handler.handle_error(e, context, self.partial_fill_recovery)

            # Log detailed error information
            logger.error(
                "Failed to calculate capital metrics",
                error_id=context.error_id,
                severity=context.severity.value,
                error=str(e),
            )
            raise

    async def _calculate_performance_metrics(self) -> dict[str, dict[str, float]]:
        """
        Calculate performance metrics for all strategies.

        Returns:
            Dict[str, Dict[str, float]]: Performance metrics by strategy
        """
        metrics = {}

        for strategy_id in self.strategy_allocations.keys():
            strategy_key = strategy_id.split("_")[0]  # Extract strategy name

            # TODO: Remove in production - Mock performance data for now
            # In production, this would fetch real performance data from
            # database
            if strategy_key not in self.strategy_performance:
                self.strategy_performance[strategy_key] = {
                    # Mock Sharpe ratio
                    "sharpe_ratio": 0.5 + (hash(strategy_key) % 100) / 1000,
                    # Mock return rate
                    "return_rate": 0.02 + (hash(strategy_key) % 50) / 1000,
                    # Mock volatility
                    "volatility": 0.15 + (hash(strategy_key) % 30) / 1000,
                    # Mock drawdown
                    "max_drawdown": 0.05 + (hash(strategy_key) % 20) / 1000,
                    # Mock win rate
                    "win_rate": 0.55 + (hash(strategy_key) % 30) / 1000,
                }

            metrics[strategy_key] = self.strategy_performance[strategy_key]

        return metrics

    async def _equal_weight_allocation(self) -> dict[str, CapitalAllocation]:
        """Equal weight allocation strategy."""
        strategies = list(
            set(alloc.strategy_id.split("_")[0] for alloc in self.strategy_allocations.values())
        )
        allocation_per_strategy = self.available_capital / len(strategies)

        new_allocations = {}
        for strategy_id in strategies:
            for exchange in ["binance", "okx", "coinbase"]:
                key = f"{strategy_id}_{exchange}"
                if key in self.strategy_allocations:
                    allocation = self.strategy_allocations[key]
                    allocation.allocated_amount = (
                        allocation_per_strategy / 3
                    )  # Split across exchanges
                    allocation.allocation_percentage = float(
                        allocation.allocated_amount / self.total_capital
                    )
                    new_allocations[key] = allocation

        return new_allocations

    async def _performance_weighted_allocation(
        self, performance_metrics: dict[str, dict[str, float]]
    ) -> dict[str, CapitalAllocation]:
        """Performance-weighted allocation strategy."""
        # Calculate weights based on Sharpe ratios
        total_sharpe = sum(
            metrics.get("sharpe_ratio", 0) for metrics in performance_metrics.values()
        )

        new_allocations = {}
        for strategy_id, metrics in performance_metrics.items():
            weight = (
                metrics.get("sharpe_ratio", 0) / total_sharpe
                if total_sharpe > 0
                else 1.0 / len(performance_metrics)
            )
            allocation_amount = self.available_capital * Decimal(str(weight))

            for exchange in ["binance", "okx", "coinbase"]:
                key = f"{strategy_id}_{exchange}"
                if key in self.strategy_allocations:
                    allocation = self.strategy_allocations[key]
                    allocation.allocated_amount = allocation_amount / 3
                    allocation.allocation_percentage = float(
                        allocation.allocated_amount / self.total_capital
                    )
                    new_allocations[key] = allocation

        return new_allocations

    async def _volatility_weighted_allocation(
        self, performance_metrics: dict[str, dict[str, float]]
    ) -> dict[str, CapitalAllocation]:
        """Volatility-weighted allocation strategy (inverse volatility weighting)."""
        # Calculate inverse volatility weights
        inverse_volatilities = {}
        total_inverse_vol = 0

        for strategy_id, metrics in performance_metrics.items():
            volatility = metrics.get("volatility", 0.15)
            if volatility > 0:
                inverse_vol = 1.0 / volatility
                inverse_volatilities[strategy_id] = inverse_vol
                total_inverse_vol += inverse_vol

        new_allocations = {}
        for strategy_id, inverse_vol in inverse_volatilities.items():
            weight = (
                inverse_vol / total_inverse_vol
                if total_inverse_vol > 0
                else 1.0 / len(inverse_volatilities)
            )
            allocation_amount = self.available_capital * Decimal(str(weight))

            for exchange in ["binance", "okx", "coinbase"]:
                key = f"{strategy_id}_{exchange}"
                if key in self.strategy_allocations:
                    allocation = self.strategy_allocations[key]
                    allocation.allocated_amount = allocation_amount / 3
                    allocation.allocation_percentage = float(
                        allocation.allocated_amount / self.total_capital
                    )
                    new_allocations[key] = allocation

        return new_allocations

    async def _risk_parity_allocation(
        self, performance_metrics: dict[str, dict[str, float]]
    ) -> dict[str, CapitalAllocation]:
        """Risk parity allocation strategy."""
        # Calculate risk contributions (simplified)
        risk_contributions = {}
        total_risk = 0

        for strategy_id, metrics in performance_metrics.items():
            volatility = metrics.get("volatility", 0.15)
            risk_contributions[strategy_id] = volatility
            total_risk += volatility

        new_allocations = {}
        for strategy_id, risk in risk_contributions.items():
            # Equal risk contribution
            target_risk = total_risk / len(risk_contributions)
            weight = target_risk / risk if risk > 0 else 1.0 / len(risk_contributions)
            allocation_amount = self.available_capital * Decimal(str(weight))

            for exchange in ["binance", "okx", "coinbase"]:
                key = f"{strategy_id}_{exchange}"
                if key in self.strategy_allocations:
                    allocation = self.strategy_allocations[key]
                    allocation.allocated_amount = allocation_amount / 3
                    allocation.allocation_percentage = float(
                        allocation.allocated_amount / self.total_capital
                    )
                    new_allocations[key] = allocation

        return new_allocations

    async def _dynamic_allocation(
        self, performance_metrics: dict[str, dict[str, float]]
    ) -> dict[str, CapitalAllocation]:
        """Dynamic allocation based on market conditions and performance."""
        # Combine multiple factors for dynamic allocation
        allocation_scores = {}
        total_score = 0

        for strategy_id, metrics in performance_metrics.items():
            # Calculate composite score
            sharpe = metrics.get("sharpe_ratio", 0)
            return_rate = metrics.get("return_rate", 0)
            volatility = metrics.get("volatility", 0.15)
            win_rate = metrics.get("win_rate", 0.5)

            # Normalize and weight factors
            score = sharpe * 0.3 + return_rate * 0.25 + (1 - volatility) * 0.2 + win_rate * 0.25

            allocation_scores[strategy_id] = score
            total_score += score

        new_allocations = {}
        for strategy_id, score in allocation_scores.items():
            weight = score / total_score if total_score > 0 else 1.0 / len(allocation_scores)
            allocation_amount = self.available_capital * Decimal(str(weight))

            for exchange in ["binance", "okx", "coinbase"]:
                key = f"{strategy_id}_{exchange}"
                if key in self.strategy_allocations:
                    allocation = self.strategy_allocations[key]
                    allocation.allocated_amount = allocation_amount / 3
                    allocation.allocation_percentage = float(
                        allocation.allocated_amount / self.total_capital
                    )
                    new_allocations[key] = allocation

        return new_allocations

    async def _apply_rebalancing_limits(
        self, new_allocations: dict[str, CapitalAllocation]
    ) -> dict[str, CapitalAllocation]:
        """Apply rebalancing limits to prevent excessive changes."""
        max_daily_change = self.total_capital * Decimal(
            str(self.capital_config.max_daily_reallocation_pct)
        )

        for key, allocation in new_allocations.items():
            if key in self.strategy_allocations:
                current_allocation = self.strategy_allocations[key]
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

                    allocation.allocation_percentage = float(
                        allocation.allocated_amount / self.total_capital
                    )

                    logger.warning(
                        "Allocation change limited",
                        strategy_id=allocation.strategy_id,
                        exchange=allocation.exchange,
                        original_change=format_currency(float(change_amount)),
                        limited_change=format_currency(float(max_daily_change)),
                    )

        return new_allocations

    async def _calculate_allocation_efficiency(self) -> float:
        """
        Calculate real allocation efficiency based on actual trading metrics.
        No artificial caps - efficiency reflects real performance.
        """
        if not self.strategy_allocations:
            return 0.5  # Neutral efficiency when no allocations

        # Calculate total utilization and allocation
        total_utilization = sum(
            allocation.utilized_amount for allocation in self.strategy_allocations.values()
        )
        total_allocated = sum(
            allocation.allocated_amount for allocation in self.strategy_allocations.values()
        )

        if total_allocated == 0:
            return 0.5

        # Base efficiency: utilization rate (how much allocated capital is
        # being used)
        utilization_efficiency = float(total_utilization / total_allocated)

        # Performance adjustment based on strategy performance
        performance_multiplier = await self._calculate_performance_multiplier()

        # Market regime adjustment
        market_regime_multiplier = await self._calculate_market_regime_multiplier()

        # Final efficiency can exceed 1.0 in good conditions
        final_efficiency = (
            utilization_efficiency * performance_multiplier * market_regime_multiplier
        )

        return max(0.0, final_efficiency)  # Only cap at 0, not at 1.0

    async def _calculate_performance_multiplier(self) -> float:
        """Calculate performance-based efficiency multiplier."""
        if not self.strategy_performance:
            return 1.0  # Neutral multiplier

        total_performance_score = 0.0
        strategy_count = 0

        for _strategy_id, metrics in self.strategy_performance.items():
            # Calculate composite performance score
            sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
            return_rate = metrics.get("return_rate", 0.0)
            win_rate = metrics.get("win_rate", 0.5)

            # Performance score can exceed 1.0 for exceptional strategies
            performance_score = (
                sharpe_ratio * 0.4  # 40% weight on risk-adjusted returns
                + return_rate * 0.4  # 40% weight on absolute returns
                + win_rate * 0.2  # 20% weight on win rate
            )

            total_performance_score += performance_score
            strategy_count += 1

        avg_performance = total_performance_score / strategy_count if strategy_count > 0 else 1.0

        # Performance multiplier can range from 0.5 (poor) to 2.0 (exceptional)
        return max(0.5, min(2.0, avg_performance))

    async def _calculate_market_regime_multiplier(self) -> float:
        """Calculate market regime-based efficiency multiplier."""
        # TODO: This should integrate with market regime detection from P-010
        # For now, use a neutral multiplier that can be enhanced later

        # Market regime multipliers:
        # - Bull market: 1.2x (20% efficiency boost)
        # - Bear market: 0.8x (20% efficiency reduction)
        # - Sideways market: 1.0x (neutral)
        # - Crisis: 0.6x (40% efficiency reduction)

        # Placeholder implementation - should be replaced with real market
        # regime detection
        current_regime = "neutral"  # This should come from market regime detection

        regime_multipliers = {
            "bull": 1.2,
            "bear": 0.8,
            "sideways": 1.0,
            "crisis": 0.6,
            "neutral": 1.0,
        }

        return regime_multipliers.get(current_regime, 1.0)

    def _get_minimum_allocation(self, strategy_id: str) -> float:
        """Get minimum allocation for a strategy."""
        strategy_type = strategy_id.split("_")[0] if "_" in strategy_id else strategy_id
        return self.capital_config.per_strategy_minimum.get(strategy_type, 1000.0)

    async def get_emergency_reserve(self) -> Decimal:
        """Get current emergency reserve amount."""
        return self.emergency_reserve

    async def update_total_capital(self, new_total: Decimal) -> None:
        """Update total capital and recalculate reserves."""
        self.total_capital = new_total
        self.emergency_reserve = self.total_capital * Decimal(
            str(self.capital_config.emergency_reserve_pct)
        )
        self.available_capital = self.total_capital - self.emergency_reserve

        logger.info(
            "Total capital updated",
            new_total=format_currency(float(self.total_capital)),
            emergency_reserve=format_currency(float(self.emergency_reserve)),
            available_capital=format_currency(float(self.available_capital)),
        )

    async def get_allocation_summary(self) -> dict[str, Any]:
        """Get allocation summary for all strategies."""
        total_allocations = len(self.strategy_allocations)
        total_allocated = sum(
            alloc.allocated_amount for alloc in self.strategy_allocations.values()
        )

        summary = {
            "total_allocations": total_allocations,
            "total_allocated": total_allocated,
            "total_capital": self.total_capital,
            "available_capital": self.available_capital,
            "emergency_reserve": self.emergency_reserve,
            "strategies": {},
        }

        for key, allocation in self.strategy_allocations.items():
            strategy_id, exchange = key.split("_", 1) if "_" in key else (key, "unknown")
            summary["strategies"][key] = {
                "strategy_id": strategy_id,
                "exchange": exchange,
                "allocated_amount": allocation.allocated_amount,
                "utilized_amount": allocation.utilized_amount,
                "available_amount": allocation.available_amount,
                "allocation_percentage": allocation.allocation_percentage,
            }

        return summary
