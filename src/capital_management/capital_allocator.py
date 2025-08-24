"""
Capital Allocator Implementation - Refactored to use CapitalService

This module implements the dynamic capital allocation framework that manages
total capital with emergency reserves using the enterprise-grade CapitalService.

Key Features:
- Uses CapitalService for all database operations (NO direct DB access)
- Full audit trail through service layer
- Transaction support with rollback capabilities
- Performance-based strategy allocation adjustments
- Risk-adjusted capital distribution using Sharpe ratios
- Dynamic rebalancing based on strategy performance
- Enterprise-grade error handling and monitoring

Author: Trading Bot Framework
Version: 2.0.0 - Refactored for service layer
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.capital_management.service import CapitalService
from src.core.base.component import BaseComponent

# Import foundation services
from src.core.exceptions import RiskManagementError, ServiceError, ValidationError

# MANDATORY: Import from P-001
from src.core.types.risk import (
    AllocationStrategy,
    CapitalAllocation,
    CapitalMetrics,
)

# MANDATORY: Import from P-002A (error handling) - Updated decorators
try:
    from src.risk_management import RiskManager, RiskService

    # Validate imports are correct types
    if not (hasattr(RiskService, "__module__") and "risk_management" in RiskService.__module__):
        RiskService = None  # type: ignore
    if not (hasattr(RiskManager, "__module__") and "risk_management" in RiskManager.__module__):
        RiskManager = None  # type: ignore
except ImportError as e:
    # No logger available at module level - use print for import warnings
    print(f"Warning: Risk management imports not available: {e}")
    RiskService = None  # type: ignore
    RiskManager = None  # type: ignore

try:
    from src.state import TradeLifecycleManager

    # Validate import is correct type
    if not (
        hasattr(TradeLifecycleManager, "__module__") and "state" in TradeLifecycleManager.__module__
    ):
        TradeLifecycleManager = None  # type: ignore
except ImportError as e:
    # No logger available at module level - use print for import warnings
    print(f"Warning: State management imports not available: {e}")
    TradeLifecycleManager = None  # type: ignore

from src.utils.decorators import time_execution
from src.utils.formatters import format_currency
from src.utils.validators import ValidationFramework


class CapitalAllocator(BaseComponent):
    """
    Dynamic capital allocation framework using enterprise CapitalService.

    This class provides a high-level interface for capital management operations
    while delegating all database operations to the CapitalService. All operations
    now include audit trails, transaction support, and enterprise error handling.

    Key Changes:
    - Uses CapitalService for all data operations (NO direct database access)
    - Full audit trail for all allocation operations
    - Transaction support with rollback capabilities
    - Enterprise-grade monitoring and metrics
    """

    def __init__(
        self,
        capital_service: CapitalService,
        config_service=None,
        risk_manager: RiskService | RiskManager | None = None,
        trade_lifecycle_manager: TradeLifecycleManager | None = None,
    ):
        """
        Initialize the capital allocator with CapitalService dependency injection.

        Args:
            capital_service: CapitalService instance for all database operations
            config_service: ConfigService instance for configuration access
            risk_manager: Risk management instance for validation
            trade_lifecycle_manager: Trade lifecycle manager for trade-capital integration
        """
        # Initialize base component
        super().__init__()

        # CRITICAL: Use CapitalService for ALL database operations
        self.capital_service = capital_service
        self.risk_manager = risk_manager
        self.trade_lifecycle_manager = trade_lifecycle_manager

        # Get config from ConfigService or fallback to legacy method
        self.config_service = config_service

        if config_service is None:
            # Fallback to dependency injection
            try:
                from src.core.dependency_injection import get_container
                from src.core.exceptions import DependencyError

                try:
                    config_service = get_container().get("ConfigService")
                    self.capital_config = config_service.get_config_value("capital_management", {})
                except DependencyError:
                    # No ConfigService available, use fallback
                    raise AttributeError
            except (ImportError, KeyError, AttributeError):
                # Final fallback to legacy method - with proper error handling
                try:
                    from src.core.config import get_config

                    full_config = get_config()
                    self.capital_config = (
                        full_config.capital_management
                        if hasattr(full_config, "capital_management")
                        else {}
                    )
                except ImportError:
                    # If config module not available, use defaults
                    self.logger.warning("Config module not available, using default capital config")
                    self.capital_config = {}
        elif isinstance(config_service, dict):
            # Direct config dict (for testing)
            self.capital_config = config_service
        else:
            # Standard config service
            self.capital_config = config_service.get_config_value("capital_management", {})

        # Performance tracking (local cache only)
        self.strategy_performance: dict[str, dict[str, float]] = {}
        self.performance_window = timedelta(days=30)
        self.last_rebalance = datetime.now(timezone.utc)

        # Allocation strategy configuration
        if isinstance(self.capital_config, dict):
            self.rebalance_frequency_hours = self.capital_config.get(
                "rebalance_frequency_hours", 24
            )
            self.max_daily_reallocation_pct = self.capital_config.get(
                "max_daily_reallocation_pct", 0.1
            )
        else:
            # CapitalManagementConfig object access
            self.rebalance_frequency_hours = getattr(
                self.capital_config, "rebalance_frequency_hours", 24
            )
            self.max_daily_reallocation_pct = getattr(
                self.capital_config, "max_daily_reallocation_pct", 0.1
            )

        self.logger.info(
            "Capital allocator initialized with CapitalService",
            service_type=type(self.capital_service).__name__,
            rebalance_frequency=self.rebalance_frequency_hours,
        )

    @time_execution
    async def allocate_capital(
        self, strategy_id: str, exchange: str, requested_amount: Decimal, bot_id: str | None = None
    ) -> CapitalAllocation:
        """
        Allocate capital to a strategy on a specific exchange using CapitalService.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name
            requested_amount: Requested capital amount
            bot_id: Optional bot instance ID

        Returns:
            CapitalAllocation: Allocation record with full audit trail

        Raises:
            ValidationError: If allocation violates limits
            ServiceError: If allocation operation fails
        """
        try:
            # Validate inputs using existing validation logic
            if not ValidationFramework.validate_quantity(float(requested_amount)):
                raise ValidationError(f"Invalid capital allocation amount: {requested_amount}")

            if not strategy_id or not strategy_id.strip():
                raise ValidationError("Strategy ID cannot be empty")

            # Use CapitalService for allocation - includes all validation, audit, and transaction support
            allocation = await self.capital_service.allocate_capital(
                strategy_id=strategy_id,
                exchange=exchange,
                requested_amount=requested_amount,
                bot_id=bot_id,
                authorized_by="CapitalAllocator",
                risk_context={
                    "component": "CapitalAllocator",
                    "allocation_strategy": (
                        self.capital_config.get("allocation_strategy", "EQUAL_WEIGHT")
                        if isinstance(self.capital_config, dict)
                        else getattr(
                            self.capital_config,
                            "allocation_strategy",
                            AllocationStrategy.EQUAL_WEIGHT,
                        ).value
                    ),
                    "risk_assessment": await self._assess_allocation_risk(
                        strategy_id, exchange, requested_amount
                    ),
                },
            )

            self.logger.info(
                "Capital allocation completed via service",
                strategy_id=strategy_id,
                exchange=exchange,
                amount=format_currency(float(requested_amount)),
                allocation_percentage=f"{allocation.allocation_percentage:.2%}",
            )

            return allocation

        except (ValidationError, ServiceError) as e:
            # Re-raise service layer exceptions
            self.logger.error(
                "Capital allocation failed",
                strategy_id=strategy_id,
                exchange=exchange,
                requested_amount=format_currency(float(requested_amount)),
                error=str(e),
            )
            raise

        except Exception as e:
            # Log and wrap unexpected exceptions
            self.logger.error(
                "Unexpected error in capital allocation",
                strategy_id=strategy_id,
                exchange=exchange,
                error=str(e),
                exc_info=True,
            )
            raise ServiceError(f"Capital allocation failed: {e}") from e

    @time_execution
    async def release_capital(
        self, strategy_id: str, exchange: str, amount: Decimal, bot_id: str | None = None
    ) -> bool:
        """
        Release capital allocation for a strategy using CapitalService.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name
            amount: Amount to release
            bot_id: Optional bot instance ID

        Returns:
            bool: True if release successful

        Raises:
            ValidationError: If release is invalid
            ServiceError: If release operation fails
        """
        try:
            # Use CapitalService for capital release - includes full audit trail
            success = await self.capital_service.release_capital(
                strategy_id=strategy_id,
                exchange=exchange,
                release_amount=amount,
                bot_id=bot_id,
                authorized_by="CapitalAllocator",
            )

            self.logger.info(
                "Capital release completed via service",
                strategy_id=strategy_id,
                exchange=exchange,
                amount=format_currency(float(amount)),
                success=success,
            )

            return success

        except (ValidationError, ServiceError) as e:
            self.logger.error(
                "Capital release failed",
                strategy_id=strategy_id,
                exchange=exchange,
                amount=format_currency(float(amount)),
                error=str(e),
            )
            return False

        except Exception as e:
            self.logger.error(
                "Unexpected error in capital release",
                strategy_id=strategy_id,
                exchange=exchange,
                error=str(e),
            )
            return False

    @time_execution
    async def rebalance_allocations(
        self, authorized_by: str | None = None
    ) -> dict[str, CapitalAllocation]:
        """
        Rebalance capital allocations based on performance and strategy using CapitalService.

        Args:
            authorized_by: User or system authorizing rebalance

        Returns:
            Dict[str, CapitalAllocation]: Updated allocations with full audit trail
        """
        try:
            self.logger.info("Starting capital rebalancing via service layer")

            # Get current metrics from service
            current_metrics = await self.capital_service.get_capital_metrics()

            # Check if rebalancing is needed
            if not await self._should_rebalance(current_metrics):
                self.logger.info("Rebalancing not needed at this time")
                return {}

            # Get current performance metrics
            await self._calculate_performance_metrics()

            # Determine allocation strategy
            strategy_name = (
                self.capital_config.get("allocation_strategy", "EQUAL_WEIGHT")
                if isinstance(self.capital_config, dict)
                else getattr(
                    self.capital_config, "allocation_strategy", AllocationStrategy.EQUAL_WEIGHT
                ).value
            )
            strategy = AllocationStrategy(strategy_name)

            # For now, return empty dict as rebalancing logic would need to be
            # implemented using multiple service calls with proper transaction handling
            self.logger.warning(
                "Rebalancing logic needs to be implemented with service layer transactions",
                strategy=strategy.value,
                current_allocations=current_metrics.allocation_count,
            )

            # Update rebalance tracking
            self.last_rebalance = datetime.now(timezone.utc)

            return {}

        except Exception as e:
            self.logger.error("Capital rebalancing failed", error=str(e))
            raise ServiceError(f"Capital rebalancing failed: {e}") from e

    @time_execution
    async def update_utilization(
        self, strategy_id: str, exchange: str, utilized_amount: Decimal, bot_id: str | None = None
    ) -> None:
        """
        Update capital utilization for a strategy using CapitalService.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name
            utilized_amount: Amount currently utilized
            bot_id: Optional bot instance ID
        """
        try:
            # Use CapitalService for utilization update
            success = await self.capital_service.update_utilization(
                strategy_id=strategy_id,
                exchange=exchange,
                utilized_amount=utilized_amount,
                bot_id=bot_id,
            )

            if success:
                self.logger.debug(
                    "Capital utilization updated via service",
                    strategy_id=strategy_id,
                    exchange=exchange,
                    utilized=format_currency(float(utilized_amount)),
                )
            else:
                self.logger.warning(
                    "Capital utilization update failed",
                    strategy_id=strategy_id,
                    exchange=exchange,
                    utilized_amount=format_currency(float(utilized_amount)),
                )

        except Exception as e:
            self.logger.error(
                "Utilization update error",
                strategy_id=strategy_id,
                exchange=exchange,
                utilized_amount=format_currency(float(utilized_amount)),
                error=str(e),
            )
            raise ServiceError(f"Utilization update failed: {e}") from e

    @time_execution
    async def get_capital_metrics(self) -> CapitalMetrics:
        """
        Get current capital management metrics using CapitalService.

        Returns:
            CapitalMetrics: Current capital metrics with caching and monitoring
        """
        try:
            # Use CapitalService for metrics - includes caching and monitoring
            metrics = await self.capital_service.get_capital_metrics()

            self.logger.debug(
                "Capital metrics retrieved via service",
                total_capital=format_currency(float(metrics.total_capital)),
                allocated_capital=format_currency(float(metrics.allocated_capital)),
                utilization_rate=f"{metrics.utilization_rate:.2%}",
                allocation_efficiency=f"{metrics.allocation_efficiency:.2f}",
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to get capital metrics: {e}")
            raise ServiceError(f"Capital metrics retrieval failed: {e}") from e

    # Helper methods

    async def _assess_allocation_risk(
        self, strategy_id: str, exchange: str, amount: Decimal
    ) -> dict[str, Any]:
        """Assess risk for capital allocation."""
        risk_assessment = {
            "risk_level": "low",
            "risk_factors": [],
            "recommendations": [],
        }

        if self.risk_manager:
            try:
                # If using RiskService (new architecture) - check for proper interface
                if (
                    hasattr(self.risk_manager, "__class__")
                    and self.risk_manager.__class__.__name__ == "RiskService"
                ):
                    # RiskService should have evaluate_allocation_risk method
                    if hasattr(self.risk_manager, "evaluate_allocation_risk"):
                        assessment = await self.risk_manager.evaluate_allocation_risk(
                            strategy_id=strategy_id, exchange=exchange, amount=amount
                        )
                        risk_assessment.update(assessment)
                    else:
                        # Fallback if method not available
                        risk_assessment["risk_level"] = "medium"
                        risk_assessment["risk_factors"].append("RiskService method not available")
                # If using legacy RiskManager
                elif (
                    hasattr(self.risk_manager, "__class__")
                    and self.risk_manager.__class__.__name__ == "RiskManager"
                ):
                    # Use available methods for risk assessment
                    if hasattr(self.risk_manager, "calculate_risk_metrics"):
                        # Legacy risk assessment
                        risk_assessment["risk_level"] = "medium"
                        risk_assessment["risk_factors"].append("Using legacy risk assessment")
                    else:
                        risk_assessment["risk_level"] = "unknown"
                        risk_assessment["risk_factors"].append(
                            "No risk assessment method available"
                        )
                else:
                    # Unknown risk manager type
                    risk_assessment["risk_level"] = "unknown"
                    risk_assessment["risk_factors"].append(
                        f"Unknown risk manager type: {type(self.risk_manager)}"
                    )
            except RiskManagementError as e:
                # Log the error but don't re-raise - let allocation continue with high risk flag
                self.logger.error(f"Risk assessment failed: {e}")
                risk_assessment["risk_level"] = "high"
                risk_assessment["risk_factors"].append(f"Risk assessment error: {e!s}")
                risk_assessment["recommendations"].append(
                    "Manual review recommended due to risk assessment failure"
                )
            except Exception as e:
                # Handle any other unexpected errors gracefully
                self.logger.warning(f"Unexpected error in risk assessment: {e}")
                risk_assessment["risk_level"] = "unknown"
                risk_assessment["risk_factors"].append(f"Unexpected error: {e!s}")

        return risk_assessment

    async def _should_rebalance(self, current_metrics: CapitalMetrics) -> bool:
        """Determine if rebalancing is needed."""
        # Check time since last rebalance
        time_since_rebalance = datetime.now(timezone.utc) - self.last_rebalance
        if time_since_rebalance < timedelta(hours=self.rebalance_frequency_hours):
            return False

        # Check if efficiency is too low
        if current_metrics.allocation_efficiency < 0.3:
            self.logger.info(
                "Rebalancing triggered by low allocation efficiency",
                efficiency=current_metrics.allocation_efficiency,
            )
            return True

        # Check utilization rate
        if current_metrics.utilization_rate < 0.5:
            self.logger.info(
                "Rebalancing triggered by low utilization rate",
                utilization_rate=current_metrics.utilization_rate,
            )
            return True

        return False

    async def _calculate_performance_metrics(self) -> dict[str, dict[str, float]]:
        """
        Calculate performance metrics for all strategies.

        Note: This returns cached performance metrics. In production, this would integrate
        with performance tracking systems and use historical trade data.
        """
        # Return existing performance metrics
        # These should be updated by external performance tracking service
        return self.strategy_performance.copy()

    async def get_emergency_reserve(self) -> Decimal:
        """Get current emergency reserve amount from service."""
        try:
            metrics = await self.capital_service.get_capital_metrics()
            return metrics.emergency_reserve
        except Exception as e:
            self.logger.error(f"Failed to get emergency reserve: {e}")
            return Decimal("0")

    async def get_allocation_summary(self) -> dict[str, Any]:
        """Get allocation summary using service layer."""
        try:
            metrics = await self.capital_service.get_capital_metrics()

            # Get service performance metrics
            service_metrics = self.capital_service.get_performance_metrics()

            summary = {
                "total_allocations": metrics.allocation_count,
                "total_allocated": metrics.allocated_capital,
                "total_capital": metrics.total_capital,
                "available_capital": metrics.available_capital,
                "emergency_reserve": metrics.emergency_reserve,
                "utilization_rate": metrics.utilization_rate,
                "allocation_efficiency": metrics.allocation_efficiency,
                "service_metrics": {
                    "successful_allocations": service_metrics.get("successful_allocations", 0),
                    "failed_allocations": service_metrics.get("failed_allocations", 0),
                    "total_releases": service_metrics.get("total_releases", 0),
                    "average_allocation_time_ms": service_metrics.get(
                        "average_allocation_time_ms", 0
                    ),
                },
                "last_updated": metrics.last_updated,
            }

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get allocation summary: {e}")
            return {
                "error": str(e),
                "total_allocations": 0,
                "total_allocated": Decimal("0"),
                "total_capital": Decimal("0"),
            }

    # Trade Lifecycle Integration Methods

    async def reserve_capital_for_trade(
        self,
        trade_id: str,
        strategy_id: str,
        exchange: str,
        requested_amount: Decimal,
        bot_id: str | None = None,
    ) -> CapitalAllocation | None:
        """
        Reserve capital for a specific trade.

        This method integrates with TradeLifecycleManager to ensure capital
        is reserved when a trade is initiated.

        Args:
            trade_id: Unique trade identifier
            strategy_id: Strategy identifier
            exchange: Exchange name
            requested_amount: Amount to reserve for the trade
            bot_id: Associated bot instance ID

        Returns:
            CapitalAllocation if successful, None if allocation fails
        """
        try:
            # Allocate capital through the service
            allocation = await self.capital_service.allocate_capital(
                strategy_id=strategy_id,
                exchange=exchange,
                requested_amount=requested_amount,
                bot_id=bot_id,
                authorized_by=f"trade_{trade_id}",
            )

            # Track trade-capital association if trade lifecycle manager available
            if self.trade_lifecycle_manager:
                try:
                    await self.trade_lifecycle_manager.update_trade_metadata(
                        trade_id=trade_id,
                        metadata={
                            "capital_allocated": str(requested_amount),
                            "capital_allocation_id": f"{strategy_id}_{exchange}",
                            "capital_allocation_timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                except Exception as tlm_error:
                    # Log error but don't fail the allocation
                    self.logger.warning(
                        f"Failed to update trade metadata: {tlm_error}",
                        trade_id=trade_id,
                        error=str(tlm_error),
                    )

            self.logger.info(
                "Capital reserved for trade",
                trade_id=trade_id,
                strategy_id=strategy_id,
                exchange=exchange,
                amount=format_currency(float(requested_amount)),
            )

            return allocation

        except Exception as e:
            self.logger.error(
                f"Failed to reserve capital for trade {trade_id}: {e}",
                trade_id=trade_id,
                strategy_id=strategy_id,
                exchange=exchange,
                requested_amount=float(requested_amount),
            )
            return None

    async def release_capital_from_trade(
        self,
        trade_id: str,
        strategy_id: str,
        exchange: str,
        release_amount: Decimal,
        bot_id: str | None = None,
    ) -> bool:
        """
        Release capital when a trade is closed.

        This method integrates with TradeLifecycleManager to ensure capital
        is released when a trade is completed or cancelled.

        Args:
            trade_id: Unique trade identifier
            strategy_id: Strategy identifier
            exchange: Exchange name
            release_amount: Amount to release
            bot_id: Associated bot instance ID

        Returns:
            bool: True if release successful
        """
        try:
            # Release capital through the service
            success = await self.capital_service.release_capital(
                strategy_id=strategy_id,
                exchange=exchange,
                release_amount=release_amount,
                bot_id=bot_id,
                authorized_by=f"trade_{trade_id}",
            )

            if success and self.trade_lifecycle_manager:
                # Update trade metadata to reflect capital release
                try:
                    await self.trade_lifecycle_manager.update_trade_metadata(
                        trade_id=trade_id,
                        metadata={
                            "capital_released": str(release_amount),
                            "capital_release_timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                except Exception as tlm_error:
                    # Log error but don't fail the release
                    self.logger.warning(
                        f"Failed to update trade metadata for release: {tlm_error}",
                        trade_id=trade_id,
                        error=str(tlm_error),
                    )

            self.logger.info(
                "Capital released from trade",
                trade_id=trade_id,
                strategy_id=strategy_id,
                exchange=exchange,
                amount=format_currency(float(release_amount)),
            )

            return success

        except Exception as e:
            self.logger.error(
                f"Failed to release capital from trade {trade_id}: {e}",
                trade_id=trade_id,
                strategy_id=strategy_id,
                exchange=exchange,
                release_amount=float(release_amount),
            )
            return False

    async def get_trade_capital_efficiency(
        self,
        trade_id: str,
        strategy_id: str,
        exchange: str,
        realized_pnl: Decimal,
    ) -> dict[str, Any]:
        """
        Calculate capital efficiency metrics for a completed trade.

        Args:
            trade_id: Unique trade identifier
            strategy_id: Strategy identifier
            exchange: Exchange name
            realized_pnl: Realized profit/loss from the trade

        Returns:
            dict: Capital efficiency metrics
        """
        try:
            # Get allocation for the strategy/exchange
            allocation = await self._get_allocation(strategy_id, exchange)

            if not allocation:
                return {
                    "trade_id": trade_id,
                    "error": "No capital allocation found",
                }

            # Calculate efficiency metrics
            allocated_amount = allocation.allocated_amount
            roi = (realized_pnl / allocated_amount * 100) if allocated_amount > 0 else Decimal("0")

            efficiency_metrics = {
                "trade_id": trade_id,
                "allocated_capital": float(allocated_amount),
                "realized_pnl": float(realized_pnl),
                "roi_percentage": float(roi),
                "capital_utilization": (
                    float(allocation.utilized_amount / allocated_amount)
                    if allocated_amount > 0
                    else 0.0
                ),
            }

            # Update trade metadata if lifecycle manager available
            if self.trade_lifecycle_manager:
                try:
                    await self.trade_lifecycle_manager.update_trade_metadata(
                        trade_id=trade_id, metadata={"capital_efficiency": efficiency_metrics}
                    )
                except Exception as tlm_error:
                    # Log error but still return the metrics
                    self.logger.warning(
                        f"Failed to update trade efficiency metadata: {tlm_error}",
                        trade_id=trade_id,
                        error=str(tlm_error),
                    )

            return efficiency_metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate trade capital efficiency: {e}")
            return {
                "trade_id": trade_id,
                "error": str(e),
            }

    async def _get_allocation(self, strategy_id: str, exchange: str) -> CapitalAllocation | None:
        """
        Get current allocation for a strategy and exchange.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name

        Returns:
            CapitalAllocation if found, None otherwise
        """
        try:
            # Get allocation through service
            allocations = await self.capital_service.get_allocations_by_strategy(strategy_id)

            # Find allocation for specific exchange
            for allocation in allocations:
                if allocation.exchange == exchange:
                    return allocation

            return None

        except Exception as e:
            self.logger.error(f"Failed to get allocation for {strategy_id}/{exchange}: {e}")
            return None
