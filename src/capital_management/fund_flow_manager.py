"""
Fund Flow Manager Implementation (P-010A)

This module implements deposit/withdrawal management with minimum capital
requirements, withdrawal rules enforcement, and auto-compounding features.

Key Features:
- Minimum capital requirements per strategy validation
- Withdrawal rules enforcement (profit-only, minimum maintenance)
- Auto-compounding of profits (weekly default)
- Performance-based withdrawal permissions
- Emergency capital preservation procedures
- Capital flow audit trail and reporting

Author: Trading Bot Framework
Version: 2.0.0
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

# Remove direct infrastructure dependencies
# Use service protocols for time series and caching
from typing import Any, Protocol

# Import service interfaces
from src.capital_management.interfaces import AbstractFundFlowManagementService
from src.core.base.service import TransactionalService
from src.core.exceptions import ServiceError, ValidationError

# MANDATORY: Import from P-001
from src.core.types.capital import (
    CapitalFundFlow as FundFlow,
    ExtendedCapitalProtection as CapitalProtection,
    ExtendedWithdrawalRule as WithdrawalRule,
)
from src.utils.capital_config import (
    load_capital_config,
    resolve_config_service,
    validate_config_values,
)
from src.utils.capital_resources import get_resource_manager
from src.utils.capital_validation import (
    validate_capital_amount,
    validate_withdrawal_request,
)
from src.utils.decorators import time_execution
from src.utils.formatters import format_currency
from src.utils.interfaces import ValidationServiceInterface


class TimeSeriesServiceProtocol(Protocol):
    """Protocol for time series data storage."""

    async def write_point(
        self, measurement: str, tags: dict[str, str], fields: dict[str, Any]
    ) -> None: ...


class CacheServiceProtocol(Protocol):
    """Protocol for caching operations."""

    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: Any, ttl: int) -> None: ...


# MANDATORY: Use structured logging from src.core.logging for all capital
# management operations

# From P-002A - MANDATORY: Use error handling

# From P-007A - MANDATORY: Use decorators and validators


class FundFlowManager(AbstractFundFlowManagementService, TransactionalService):
    """
    Deposit/withdrawal management system.

    This class manages capital flows, enforces withdrawal rules,
    and implements auto-compounding features for optimal capital growth.
    """

    def __init__(
        self,
        cache_service: CacheServiceProtocol | None = None,
        time_series_service: TimeSeriesServiceProtocol | None = None,
        validation_service: ValidationServiceInterface | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """
        Initialize the fund flow manager service.

        Args:
            correlation_id: Request correlation ID for tracing
        """
        super().__init__(
            name="FundFlowManagerService",
            correlation_id=correlation_id,
        )

        # Inject service dependencies
        self._cache_service = cache_service
        self._time_series_service = time_series_service
        self.validation_service = validation_service

        # Configuration will be loaded in _do_start
        self.config: dict[str, Any] = {}

        # Flow tracking
        self.fund_flows: list[FundFlow] = []
        self.withdrawal_rules: dict[str, WithdrawalRule] = {}
        self.capital_protection: CapitalProtection | None = None  # Will be initialized in _do_start

        # Resource management
        self._max_flow_history = 1000  # Configurable via capital_config

        # Performance tracking
        self.strategy_performance: dict[str, dict[str, Decimal]] = {}
        self.total_profit = Decimal("0")
        self.locked_profit = Decimal("0")

        # Capital management integration
        self.capital_allocator = None
        self.total_capital = Decimal("0")

        # Auto-compounding tracking
        self.last_compound_date = datetime.now(timezone.utc)
        self.compound_schedule: dict[str, Any] = {}  # Will be initialized in _do_start

        # Cache keys
        self.cache_keys = {
            "fund_flows": "fund:flows",
            "summary": "fund:summary",
            "withdrawal_rules": "fund:withdrawal_rules",
        }

        # Cache TTL (seconds) - configurable via capital_config
        self.cache_ttl = 300

    async def _do_start(self) -> None:
        """Start the fund flow manager service."""
        try:
            # Resolve services if not injected
            if not self._cache_service:
                try:
                    self._cache_service = self.resolve_dependency("CacheService")
                except Exception as e:
                    self._logger.warning(f"CacheService not available via DI: {e}")

            if not self._time_series_service:
                try:
                    self._time_series_service = self.resolve_dependency("TimeSeriesService")
                except Exception as e:
                    self._logger.warning(f"TimeSeriesService not available via DI: {e}")

            # Load configuration
            await self._load_configuration()

            # Initialize withdrawal rules
            self._initialize_withdrawal_rules()

            # Initialize capital protection
            self._initialize_capital_protection()

            # Initialize compound schedule and load resource limits from config
            self.compound_schedule = self._calculate_compound_schedule()
            self._max_flow_history = self.config.get("max_flow_history", 1000)
            self.cache_ttl = self.config.get("cache_ttl_seconds", 300)

            self._logger.info(
                "Fund flow manager service started",
                auto_compound_enabled=self.config.get("auto_compound_enabled", True),
                profit_threshold=format_currency(
                    Decimal(str(self.config.get("profit_threshold", 1000)))
                ),
            )
        except Exception as e:
            self._logger.error(f"Failed to start FundFlowManager service: {e}")
            raise ServiceError(f"FundFlowManager startup failed: {e}") from e

    async def _load_configuration(self) -> None:
        """Load configuration from ConfigService."""
        resolved_config_service = resolve_config_service(self)
        self.config = load_capital_config(resolved_config_service)
        self.config = validate_config_values(self.config)

    async def _do_stop(self) -> None:
        """Stop the fund flow manager service and clean up resources."""
        try:
            await self.cleanup_resources()
            self._logger.info("FundFlowManager service stopped and resources cleaned up")
        except Exception as e:
            self._logger.error(f"Error during FundFlowManager shutdown: {e}")
            raise ServiceError(f"FundFlowManager shutdown failed: {e}") from e

    def _initialize_capital_protection(self) -> None:
        """Initialize capital protection with current config."""
        total_capital = self.config.get("total_capital", 100000)
        self.capital_protection = CapitalProtection(
            protection_id="fund_flow_protection",
            enabled=True,
            min_capital_threshold=Decimal(str(total_capital * 0.1)),
            stop_trading_threshold=Decimal(str(total_capital * 0.05)),
            reduce_size_threshold=Decimal(str(total_capital * 0.2)),
            size_reduction_factor=0.5,
            max_daily_loss=Decimal(
                str(total_capital * self.config.get("max_daily_loss_pct", 0.05))
            ),
            max_weekly_loss=Decimal(
                str(total_capital * self.config.get("max_weekly_loss_pct", 0.10))
            ),
            max_monthly_loss=Decimal(
                str(total_capital * self.config.get("max_monthly_loss_pct", 0.20))
            ),
            emergency_threshold=Decimal(str(total_capital * 0.02)),
            emergency_reserve_pct=self.config.get("emergency_reserve_pct", 0.1),
            max_daily_loss_pct=self.config.get("max_daily_loss_pct", 0.05),
            max_weekly_loss_pct=self.config.get("max_weekly_loss_pct", 0.10),
            max_monthly_loss_pct=self.config.get("max_monthly_loss_pct", 0.20),
            profit_lock_pct=self.config.get("profit_lock_pct", 0.5),
            auto_compound_enabled=self.config.get("auto_compound_enabled", True),
        )

    async def _cache_fund_flows(self, flows: list[FundFlow]) -> None:
        """Cache fund flows via cache service."""
        if not self._cache_service:
            return
        cache_data = None
        try:
            # Convert FundFlow objects to dict for caching
            cache_data = [
                flow.model_dump() if hasattr(flow, "model_dump") else flow.__dict__
                for flow in flows
            ]
            await self._cache_service.set(self.cache_keys["fund_flows"], cache_data, self.cache_ttl)
        except Exception as e:
            self._logger.warning("Failed to cache fund flows", error=str(e))
        finally:
            cache_data = None

    async def _get_cached_fund_flows(self) -> list[FundFlow] | None:
        """Get cached fund flows via cache service."""
        if not self._cache_service:
            return None
        cached_data = None
        flows = None
        try:
            cached_data = await self._cache_service.get(self.cache_keys["fund_flows"])
            if cached_data:
                # Convert cached data back to FundFlow objects
                flows = [FundFlow(**data) for data in cached_data]
                return flows
        except Exception as e:
            self._logger.warning("Failed to get cached fund flows", error=str(e))
        finally:
            cached_data = None
            if flows and len(flows) == 0:
                flows = None
        return None

    async def _store_fund_flow_time_series(self, flow: FundFlow) -> None:
        """Store fund flow in time series service for analysis."""
        if not self._time_series_service:
            return
        try:
            # Prepare time series data
            tags = {
                "component": "fund_flow_manager",
                "currency": getattr(flow, "currency", "USDT"),
                "reason": flow.reason,
                "from_exchange": flow.from_exchange or "none",
                "to_exchange": flow.to_exchange or "none",
            }
            fields = {
                "amount": float(flow.amount),
            }

            # Write to time series service
            await self._time_series_service.write_point("fund_flows", tags, fields)
        except Exception as e:
            self._logger.warning("Failed to store fund flow in time series", error=str(e))

    @time_execution
    async def process_deposit(
        self, amount: Decimal, currency: str = "USDT", exchange: str = "binance"
    ) -> FundFlow:
        """
        Process a deposit request.

        Args:
            amount: Deposit amount
            currency: Currency of deposit
            exchange: Target exchange

        Returns:
            FundFlow: Deposit flow record
        """
        try:
            # Validate inputs using utilities
            min_deposit = Decimal(str(self.config.get("min_deposit_amount", 100)))
            validate_capital_amount(amount, "deposit amount", min_deposit, "FundFlowManager")

            # Create fund flow record
            flow = FundFlow(
                from_strategy=None,
                to_strategy=None,
                from_exchange=None,
                to_exchange=exchange,
                amount=amount,
                currency=currency,
                reason="deposit",
                timestamp=datetime.now(timezone.utc),
            )

            # Add to flow history with size management
            self.fund_flows.append(flow)

            if len(self.fund_flows) > self._max_flow_history:
                self.fund_flows = self.fund_flows[-self._max_flow_history :]

            # Cache fund flows with proper resource management
            try:
                await self._cache_fund_flows(self.fund_flows)
            except Exception as cache_error:
                self._logger.warning(f"Cache operation failed: {cache_error}")

            # Store fund flow in time series with proper resource management
            try:
                await self._store_fund_flow_time_series(flow)
            except Exception as ts_error:
                self._logger.warning(f"Time series operation failed: {ts_error}")

            self._logger.info(
                "Deposit processed successfully",
                amount=format_currency(amount, currency),
                exchange=exchange,
            )

            return flow

        except Exception as e:
            # Log error information
            self._logger.error(
                "Deposit processing failed",
                amount=format_currency(amount, currency),
                exchange=exchange,
                error=str(e),
            )
            raise

    @time_execution
    async def process_withdrawal(
        self,
        amount: Decimal,
        currency: str = "USDT",
        exchange: str = "binance",
        reason: str = "withdrawal",
    ) -> FundFlow:
        """
        Process a withdrawal request with rule validation.

        Args:
            amount: Withdrawal amount
            currency: Currency of withdrawal
            exchange: Source exchange
            reason: Withdrawal reason

        Returns:
            FundFlow: Withdrawal flow record

        Raises:
            ValidationError: If withdrawal violates rules
        """
        try:
            # Validate inputs using utilities
            min_withdrawal = Decimal(str(self.config.get("min_withdrawal_amount", 100)))
            validate_withdrawal_request(
                amount, currency, exchange, min_withdrawal, "FundFlowManager"
            )

            # Validate withdrawal rules
            await self._validate_withdrawal_rules(amount, currency)

            # Check maximum withdrawal percentage
            if self.total_capital > Decimal("0"):
                max_withdrawal = self.total_capital * Decimal(
                    str(self.config.get("max_withdrawal_pct", 0.2))
                )
                if amount > max_withdrawal:
                    raise ValidationError(
                        f"Withdrawal amount {amount} exceeds maximum {max_withdrawal}"
                    )
            else:
                self._logger.warning(
                    "Total capital not set, skipping withdrawal percentage validation"
                )

            # Check cooldown period
            await self._check_withdrawal_cooldown()

            # Create fund flow record
            flow = FundFlow(
                from_strategy=None,
                to_strategy=None,
                from_exchange=exchange,
                to_exchange=None,
                amount=amount,
                currency=currency,
                reason=reason,
                timestamp=datetime.now(timezone.utc),
            )

            # Add to flow history with size management
            self.fund_flows.append(flow)

            # Maintain flow history size limit to prevent memory leaks
            if len(self.fund_flows) > self._max_flow_history:
                self.fund_flows = self.fund_flows[-self._max_flow_history :]

            self._logger.info(
                "Withdrawal processed successfully",
                amount=format_currency(amount, currency),
                exchange=exchange,
                reason=reason,
            )

            return flow

        except Exception as e:
            # Log error information
            self._logger.error(
                "Withdrawal processing failed",
                amount=format_currency(amount, currency),
                exchange=exchange,
                error=str(e),
            )
            raise

    @time_execution
    async def process_strategy_reallocation(
        self, from_strategy: str, to_strategy: str, amount: Decimal, reason: str = "reallocation"
    ) -> FundFlow:
        """
        Process capital reallocation between strategies.

        Args:
            from_strategy: Source strategy
            to_strategy: Target strategy
            amount: Reallocation amount
            reason: Reallocation reason

        Returns:
            FundFlow: Reallocation flow record
        """
        try:
            # Validate inputs
            if amount <= 0:
                raise ValidationError(f"Invalid reallocation amount: {amount}")

            # Check maximum daily reallocation
            if self.total_capital > Decimal("0"):
                max_daily_reallocation = self.total_capital * Decimal(
                    str(self.config.get("max_daily_reallocation_pct", 0.1))
                )
                daily_reallocation = await self._get_daily_reallocation_amount()

                if daily_reallocation + amount > max_daily_reallocation:
                    raise ValidationError(
                        f"Reallocation would exceed daily limit. Current: {daily_reallocation}, "
                        f"Requested: {amount}, Limit: {max_daily_reallocation}"
                    )
            else:
                self._logger.warning(
                    "Total capital not set, skipping reallocation limit validation"
                )

            # Create fund flow record
            flow = FundFlow(
                from_strategy=from_strategy,
                to_strategy=to_strategy,
                from_exchange=None,
                to_exchange=None,
                amount=amount,
                reason=reason,
                timestamp=datetime.now(timezone.utc),
            )

            # Add to flow history with size management
            self.fund_flows.append(flow)

            # Maintain flow history size limit to prevent memory leaks
            if len(self.fund_flows) > self._max_flow_history:
                self.fund_flows = self.fund_flows[-self._max_flow_history :]

            self._logger.info(
                "Strategy reallocation processed",
                from_strategy=from_strategy,
                to_strategy=to_strategy,
                amount=format_currency(amount),
                reason=reason,
            )

            return flow

        except Exception as e:
            self._logger.error(
                "Strategy reallocation failed",
                from_strategy=from_strategy,
                to_strategy=to_strategy,
                amount=format_currency(amount),
                error=str(e),
            )
            raise ServiceError(f"Strategy reallocation failed: {e}") from e

    @time_execution
    async def process_auto_compound(self) -> FundFlow | None:
        """
        Process auto-compounding of profits.

        Returns:
            Optional[FundFlow]: Compound flow record if applicable
        """
        try:
            if not self.config.get("auto_compound_enabled", True):
                return None

            # Check if it's time to compound
            if not self._should_compound():
                return None

            # Calculate compound amount
            compound_amount = await self._calculate_compound_amount()

            if compound_amount <= Decimal("0"):
                return None

            # Create compound flow record
            flow = FundFlow(
                from_strategy=None,
                to_strategy=None,
                from_exchange=None,
                to_exchange=None,
                amount=compound_amount,
                reason="auto_compound",
                timestamp=datetime.now(timezone.utc),
            )

            # Add to flow history with size management
            self.fund_flows.append(flow)

            # Maintain flow history size limit to prevent memory leaks
            if len(self.fund_flows) > self._max_flow_history:
                self.fund_flows = self.fund_flows[-self._max_flow_history :]

            # Update tracking
            self.last_compound_date = datetime.now(timezone.utc)
            self.locked_profit += compound_amount

            self._logger.info(
                "Auto-compound processed",
                compound_amount=format_currency(compound_amount),
                total_locked_profit=format_currency(self.locked_profit),
            )

            return flow

        except Exception as e:
            self._logger.error("Auto-compound processing failed", error=str(e))
            raise ServiceError(f"Auto-compound processing failed: {e}") from e

    @time_execution
    async def update_performance(
        self, strategy_id: str, performance_metrics: dict[str, float]
    ) -> None:
        """
        Update performance metrics for a strategy.

        Args:
            strategy_id: Strategy identifier
            performance_metrics: Performance metrics
        """
        try:
            self.strategy_performance[strategy_id] = performance_metrics

            # Update total profit
            if "total_pnl" in performance_metrics:
                self.total_profit = Decimal(str(performance_metrics["total_pnl"]))

            self._logger.debug(
                "Performance updated",
                strategy_id=strategy_id,
                total_profit=format_currency(self.total_profit),
            )

        except Exception as e:
            self._logger.error("Performance update failed", strategy_id=strategy_id, error=str(e))

    @time_execution
    async def get_flow_history(self, days: int = 30) -> list[FundFlow]:
        """
        Get fund flow history for the specified period.

        Args:
            days: Number of days to look back

        Returns:
            List[FundFlow]: Flow history
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

            recent_flows = [flow for flow in self.fund_flows if flow.timestamp >= cutoff_date]

            return recent_flows

        except Exception as e:
            self._logger.error("Failed to get flow history", error=str(e))
            raise ServiceError(f"Failed to get flow history: {e}") from e

    @time_execution
    async def get_flow_summary(self, days: int = 30) -> dict[str, Any]:
        """
        Get summary of fund flows for the specified period.

        Args:
            days: Number of days to look back

        Returns:
            Dict[str, Any]: Flow summary
        """
        try:
            flows = await self.get_flow_history(days)

            # Calculate summaries
            total_deposits = sum(flow.amount for flow in flows if flow.reason == "deposit")
            total_withdrawals = sum(flow.amount for flow in flows if flow.reason == "withdrawal")
            total_reallocations = sum(
                flow.amount for flow in flows if flow.reason == "reallocation"
            )
            total_compounds = sum(flow.amount for flow in flows if flow.reason == "auto_compound")

            # Group by currency
            currency_flows = {}
            for flow in flows:
                currency = getattr(flow, "currency", "USDT")
                if currency not in currency_flows:
                    currency_flows[currency] = {
                        "deposits": Decimal("0"),
                        "withdrawals": Decimal("0"),
                        "reallocations": Decimal("0"),
                        "compounds": Decimal("0"),
                    }

                if flow.reason == "deposit":
                    currency_flows[currency]["deposits"] += flow.amount
                elif flow.reason == "withdrawal":
                    currency_flows[currency]["withdrawals"] += flow.amount
                elif flow.reason == "reallocation":
                    currency_flows[currency]["reallocations"] += flow.amount
                elif flow.reason == "auto_compound":
                    currency_flows[currency]["compounds"] += flow.amount

            summary = {
                "period_days": days,
                "total_flows": len(flows),
                "total_deposits": total_deposits,
                "total_withdrawals": total_withdrawals,
                "total_reallocations": total_reallocations,
                "total_compounds": total_compounds,
                "net_flow": total_deposits - total_withdrawals,
                "currency_flows": {
                    curr: {k: v for k, v in stats.items()} for curr, stats in currency_flows.items()
                },
            }

            return summary

        except Exception as e:
            self._logger.error("Failed to get flow summary", error=str(e))
            raise ServiceError(f"Failed to get flow summary: {e}") from e

    def _initialize_withdrawal_rules(self) -> None:
        """Initialize withdrawal rules from configuration."""
        # Check if withdrawal_rules is a dict
        withdrawal_rules_config = self.config.get("withdrawal_rules", {})
        if not isinstance(withdrawal_rules_config, dict):
            self._logger.warning("No withdrawal rules configured")
            return

        for rule_name, rule_config in withdrawal_rules_config.items():
            # Ensure rule_config is a dict
            if not isinstance(rule_config, dict):
                self._logger.warning(f"Invalid config for withdrawal rule {rule_name}")
                continue

            rule = WithdrawalRule(
                name=rule_name,
                description=rule_config.get("description", ""),
                enabled=rule_config.get("enabled", True),
                threshold=rule_config.get("threshold"),
                min_amount=(
                    Decimal(str(rule_config.get("min_amount", 0)))
                    if rule_config.get("min_amount")
                    else None
                ),
                max_percentage=rule_config.get("max_percentage"),
                cooldown_hours=rule_config.get("cooldown_hours"),
            )
            self.withdrawal_rules[rule_name] = rule

    async def _validate_withdrawal_rules(self, amount: Decimal, currency: str) -> None:
        """
        Validate withdrawal against all enabled rules.

        Args:
            amount: Withdrawal amount
            currency: Currency of withdrawal

        Raises:
            ValidationError: If any rule is violated
        """
        try:
            for rule_name, rule in self.withdrawal_rules.items():
                if not rule.enabled:
                    continue

                if rule_name == "profit_only":
                    if self.total_profit <= Decimal("0"):
                        raise ValidationError("Withdrawal not allowed: No profits available")

                elif rule_name == "maintain_minimum":
                    # Check if withdrawal would violate minimum capital
                    # requirements
                    remaining_capital = self.total_capital - amount
                    min_required = await self._calculate_minimum_capital_required()
                    if remaining_capital < min_required:
                        raise ValidationError(
                            f"Withdrawal would violate minimum capital requirement. "
                            f"Remaining: {remaining_capital}, Required: {min_required}"
                        )

                elif rule_name == "performance_based":
                    if rule.threshold:
                        # Check if performance meets threshold
                        performance_ok = await self._check_performance_threshold(rule.threshold)
                        if not performance_ok:
                            raise ValidationError(
                                f"Withdrawal not allowed: Performance below "
                                f"threshold {rule.threshold}"
                            )

                # Check minimum amount rule
                if rule.min_amount and amount < rule.min_amount:
                    raise ValidationError(
                        f"Withdrawal amount {amount} below minimum {rule.min_amount}"
                    )

                # Check maximum percentage rule
                if rule.max_percentage and self.total_capital > Decimal("0"):
                    max_amount = self.total_capital * Decimal(str(rule.max_percentage))
                    if amount > max_amount:
                        raise ValidationError(
                            f"Withdrawal amount {amount} exceeds maximum {max_amount} "
                            f"({rule.max_percentage:.1%} of total capital)"
                        )

        except Exception as e:
            self._logger.error("Withdrawal rule validation failed", error=str(e))
            raise ServiceError(f"Withdrawal rule validation failed: {e}") from e

    async def _check_withdrawal_cooldown(self) -> None:
        """
        Check if withdrawal is allowed based on cooldown period.

        Raises:
            ValidationError: If cooldown period not met
        """
        try:
            # Find the most recent withdrawal
            recent_withdrawals = [flow for flow in self.fund_flows if flow.reason == "withdrawal"]

            if recent_withdrawals:
                last_withdrawal = max(recent_withdrawals, key=lambda f: f.timestamp)
                cooldown_hours = self.config.get("fund_flow_cooldown_minutes", 60) / 60

                if datetime.now(timezone.utc) - last_withdrawal.timestamp < timedelta(
                    hours=cooldown_hours
                ):
                    raise ValidationError(
                        f"Withdrawal not allowed: Cooldown period not met. "
                        f"Last withdrawal: {last_withdrawal.timestamp}"
                    )

        except Exception as e:
            self._logger.error("Withdrawal cooldown check failed", error=str(e))
            raise ServiceError(f"Withdrawal cooldown check failed: {e}") from e

    async def _calculate_minimum_capital_required(self) -> Decimal:
        """Calculate minimum capital required for all strategies."""
        try:
            total_minimum = Decimal("0")

            per_strategy_minimum = self.config.get("per_strategy_minimum", {})
            if not isinstance(per_strategy_minimum, dict):
                return Decimal("1000")  # Default minimum

            for strategy_type, min_amount in per_strategy_minimum.items():
                # Check if this strategy type is active
                active_strategies = [
                    strategy_id
                    for strategy_id in self.strategy_performance.keys()
                    if strategy_type in strategy_id.lower()
                ]

                if active_strategies:
                    total_minimum += Decimal(str(min_amount))

            return total_minimum

        except Exception as e:
            self._logger.error("Failed to calculate minimum capital required", error=str(e))
            return Decimal("1000")  # Default minimum

    async def _check_performance_threshold(self, threshold: float) -> bool:
        """
        Check if overall performance meets threshold.

        Args:
            threshold: Performance threshold

        Returns:
            bool: True if performance meets threshold
        """
        try:
            if not self.strategy_performance:
                return False

            # Calculate overall performance
            total_return = Decimal("0")
            strategy_count = 0

            for _strategy_id, metrics in self.strategy_performance.items():
                if "total_pnl" in metrics and "initial_capital" in metrics:
                    initial_capital = Decimal(str(metrics["initial_capital"]))
                    if initial_capital > 0:
                        return_rate = Decimal(str(metrics["total_pnl"])) / initial_capital
                        total_return += return_rate
                        strategy_count += 1

            if strategy_count > 0:
                avg_return = total_return / strategy_count
                return float(avg_return) >= threshold

            return False

        except Exception as e:
            self._logger.error("Failed to check performance threshold", error=str(e))
            return False

    async def _get_daily_reallocation_amount(self) -> Decimal:
        """Get total reallocation amount for today."""
        try:
            today = datetime.now(timezone.utc).date()
            today_flows = [
                flow
                for flow in self.fund_flows
                if flow.reason == "reallocation" and flow.timestamp.date() == today
            ]

            return sum(flow.amount for flow in today_flows) or Decimal("0")

        except Exception as e:
            self._logger.error("Failed to get daily reallocation amount", error=str(e))
            return Decimal("0")

    def _should_compound(self) -> bool:
        """Check if it's time to compound profits."""
        try:
            if not self.config.get("auto_compound_enabled", True):
                return False

            # Check frequency
            frequency = self.config.get("auto_compound_frequency", "weekly")
            if frequency == "weekly":
                days_since_last = (datetime.now(timezone.utc) - self.last_compound_date).days
                return days_since_last >= 7
            elif frequency == "monthly":
                days_since_last = (datetime.now(timezone.utc) - self.last_compound_date).days
                return days_since_last >= 30
            else:
                return False

        except Exception as e:
            self._logger.error("Failed to check compound timing", error=str(e))
            return False

    async def _calculate_compound_amount(self) -> Decimal:
        """Calculate amount to compound based on profits."""
        try:
            profit_threshold = Decimal(str(self.config.get("profit_threshold", 1000)))
            if self.total_profit <= profit_threshold:
                return Decimal("0")

            # Calculate compound amount (profit above threshold)
            compound_amount = self.total_profit - profit_threshold

            # Apply profit lock percentage
            locked_amount = compound_amount * Decimal(str(self.config.get("profit_lock_pct", 0.5)))

            return locked_amount

        except Exception as e:
            self._logger.error("Failed to calculate compound amount", error=str(e))
            return Decimal("0")

    def _calculate_compound_schedule(self) -> dict[str, Any]:
        """Calculate compound schedule based on frequency."""
        try:
            frequency = self.config.get("auto_compound_frequency", "weekly")
            schedule = {
                "frequency": frequency,
                "next_compound": self.last_compound_date
                + timedelta(days=7 if frequency == "weekly" else 30),
                "enabled": self.config.get("auto_compound_enabled", True),
            }

            return schedule

        except Exception as e:
            self._logger.error("Failed to calculate compound schedule", error=str(e))
            return {}

    async def update_total_capital(self, total_capital: Decimal) -> None:
        """
        Update total capital from actual account balances.

        Args:
            total_capital: Current total capital across all exchanges
        """
        try:
            self.total_capital = total_capital
            self._logger.info("Total capital updated", total_capital=format_currency(total_capital))
        except Exception as e:
            self._logger.error("Failed to update total capital", error=str(e))
            raise ServiceError(f"Failed to update total capital: {e}") from e

    async def get_total_capital(self) -> Decimal:
        """Get current total capital."""
        return self.total_capital

    async def get_capital_protection_status(self) -> dict[str, Any]:
        """Get current capital protection status."""
        try:
            if not self.capital_protection:
                return {}

            status = {
                "emergency_reserve_pct": self.capital_protection.emergency_reserve_pct,
                "max_daily_loss_pct": self.capital_protection.max_daily_loss_pct,
                "max_weekly_loss_pct": self.capital_protection.max_weekly_loss_pct,
                "max_monthly_loss_pct": self.capital_protection.max_monthly_loss_pct,
                "profit_lock_pct": self.capital_protection.profit_lock_pct,
                "total_profit": float(self.total_profit),
                "locked_profit": float(self.locked_profit),
                "auto_compound_enabled": self.capital_protection.auto_compound_enabled,
                "next_compound_date": self.compound_schedule.get(
                    "next_compound", datetime.now(timezone.utc)
                ),
                # Protection is active if there's profit
                "protection_active": self.total_profit > 0 or self.locked_profit > 0,
            }

            return status

        except Exception as e:
            self._logger.error("Failed to get capital protection status", error=str(e))
            raise ServiceError(f"Failed to get capital protection status: {e}") from e

    async def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for all strategies."""
        try:
            total_pnl = Decimal("0.0")

            # Calculate total PnL from strategy performance
            for _strategy_id, metrics in self.strategy_performance.items():
                if isinstance(metrics, dict):
                    pnl_value = metrics.get("pnl", Decimal("0.0"))
                    if isinstance(pnl_value, int | float):
                        total_pnl += Decimal(str(pnl_value))
                    elif isinstance(pnl_value, Decimal):
                        total_pnl += pnl_value
                else:
                    if isinstance(metrics, int | float):
                        total_pnl += Decimal(str(metrics))
                    elif isinstance(metrics, Decimal):
                        total_pnl += metrics

            summary: dict[str, Any] = {
                "total_pnl": total_pnl,
                "total_profit": self.total_profit,
                "locked_profit": self.locked_profit,
                "strategy_count": len(self.strategy_performance),
                "strategies": {},
            }

            for strategy_id, metrics in self.strategy_performance.items():
                if isinstance(metrics, dict):
                    pnl_value = metrics.get("pnl", Decimal("0.0"))
                    if isinstance(pnl_value, int | float):
                        pnl_value = Decimal(str(pnl_value))
                    performance_score = metrics.get("performance_score", Decimal("0.0"))
                    if isinstance(performance_score, int | float):
                        performance_score = Decimal(str(performance_score))

                    summary["strategies"][strategy_id] = {
                        "pnl": pnl_value,
                        "performance_score": performance_score,
                        "last_updated": metrics.get("last_updated", datetime.now(timezone.utc)),
                    }
                else:
                    # Handle case where metrics might be a single value
                    if isinstance(metrics, int | float):
                        pnl_value = Decimal(str(metrics))
                    elif isinstance(metrics, Decimal):
                        pnl_value = metrics
                    else:
                        pnl_value = Decimal("0.0")

                    summary["strategies"][strategy_id] = {
                        "pnl": pnl_value,
                        "performance_score": Decimal("0.0"),
                        "last_updated": datetime.now(timezone.utc),
                    }

            return summary

        except Exception as e:
            self._logger.error("Failed to get performance summary", error=str(e))
            raise ServiceError(f"Failed to get performance summary: {e}") from e

    def set_capital_allocator(self, capital_allocator: Any) -> None:
        """
        Set the capital allocator for integration.

        Args:
            capital_allocator: CapitalAllocator instance
        """
        self.capital_allocator = capital_allocator
        self._logger.info("Capital allocator integration established")

    def _validate_config(self) -> None:
        """Validate and set default configuration values."""
        # Set defaults for any missing config attributes
        config_defaults = {
            "total_capital": 100000,
            "emergency_reserve_pct": 0.1,
            "max_daily_loss_pct": 0.05,
            "max_weekly_loss_pct": 0.10,
            "max_monthly_loss_pct": 0.20,
            "profit_lock_pct": 0.5,
            "auto_compound_enabled": True,
            "auto_compound_frequency": "weekly",
            "profit_threshold": 1000,
            "min_deposit_amount": 100,
            "min_withdrawal_amount": 100,
            "max_withdrawal_pct": 0.2,
            "max_daily_reallocation_pct": 0.1,
            "fund_flow_cooldown_minutes": 60,
            "max_flow_history": 1000,
            "cache_ttl_seconds": 300,
            "withdrawal_rules": {},
            "per_strategy_minimum": {},
        }

        for key, default_value in config_defaults.items():
            if key not in self.config:
                self.config[key] = default_value
                if hasattr(self, "_logger"):
                    self._logger.warning(f"Config missing {key}, using default: {default_value}")

    async def cleanup_resources(self) -> None:
        """Clean up resources to prevent memory leaks."""
        try:
            resource_manager = get_resource_manager()

            # Clean fund flows using resource manager
            self.fund_flows = resource_manager.clean_fund_flows(self.fund_flows, max_age_days=30)

            # Clean performance data using resource manager
            self.strategy_performance = resource_manager.clean_performance_data(
                self.strategy_performance, max_age_days=30
            )

            self._logger.debug("Fund flow manager resource cleanup completed")
        except Exception as e:
            self._logger.warning(f"Resource cleanup failed: {e}")
