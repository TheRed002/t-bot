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

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import Any

from src.capital_management.constants import (
    COMPOUND_FREQUENCY_DAYS,
    DEFAULT_BASE_CURRENCY,
    DEFAULT_CACHE_OPERATION_TIMEOUT,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_CLEANUP_TIMEOUT,
    DEFAULT_EXCHANGE,
    DEFAULT_FUND_FLOW_COOLDOWN_MINUTES,
    DEFAULT_MAX_FLOW_HISTORY,
    DEFAULT_MAX_MONTHLY_LOSS_PCT,
    DEFAULT_MAX_QUEUE_SIZE,
    DEFAULT_MAX_WEEKLY_LOSS_PCT,
    DEFAULT_PERFORMANCE_WINDOW_DAYS,
    DEFAULT_PROFIT_THRESHOLD,
    DEFAULT_TIME_SERIES_TIMEOUT,
    DEFAULT_TOTAL_CAPITAL,
    EMERGENCY_RESERVE_PCT,
    EMERGENCY_THRESHOLD_PCT,
    FINANCIAL_DECIMAL_PRECISION,
    MAX_CONCURRENT_OPERATIONS,
    MAX_DAILY_LOSS_PCT,
    MAX_DAILY_REALLOCATION_PCT,
    MAX_WITHDRAWAL_PCT,
    MIN_DEPOSIT_AMOUNT,
    MIN_WITHDRAWAL_AMOUNT,
    PROFIT_LOCK_PCT,
)
from src.capital_management.interfaces import (
    AbstractFundFlowManagementService,
    CapitalAllocatorProtocol,
)
from src.core.base.service import TransactionalService
from src.core.exceptions import ServiceError, ValidationError
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
from src.utils.capital_validation import (
    validate_capital_amount,
    validate_withdrawal_request,
)
from src.utils.decimal_utils import safe_decimal_conversion
from src.utils.decorators import time_execution
from src.utils.formatters import format_currency

# Set decimal context for financial precision
getcontext().prec = FINANCIAL_DECIMAL_PRECISION
getcontext().rounding = ROUND_HALF_UP


class FundFlowManager(AbstractFundFlowManagementService, TransactionalService):
    """
    Deposit/withdrawal management system.

    This class manages capital flows, enforces withdrawal rules,
    and implements auto-compounding features for optimal capital growth.
    """

    def __init__(
        self,
        cache_service: Any | None = None,
        time_series_service: Any | None = None,
        validation_service: Any | None = None,
        capital_allocator: CapitalAllocatorProtocol | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """
        Initialize the fund flow manager service.

        Args:
            cache_service: Cache service for flow data storage
            time_series_service: Time series service for metrics storage
            validation_service: Validation service for data validation
            capital_allocator: Capital allocator service for integration
            correlation_id: Request correlation ID for tracing
        """
        super().__init__(
            name="FundFlowManagerService",
            correlation_id=correlation_id,
        )

        self._cache_service = cache_service
        self._time_series_service = time_series_service
        self.validation_service = validation_service
        self.capital_allocator = capital_allocator

        self.config: dict[str, Any] = {}
        self.fund_flows: list[FundFlow] = []
        self.withdrawal_rules: dict[str, WithdrawalRule] = {}
        self.capital_protection: CapitalProtection | None = None
        self._max_flow_history = DEFAULT_MAX_FLOW_HISTORY
        self.strategy_performance: dict[str, dict[str, Decimal]] = {}
        self.total_profit = Decimal("0")
        self.locked_profit = Decimal("0")
        self.total_capital = Decimal("0")
        self.last_compound_date = datetime.now(timezone.utc)
        self.compound_schedule: dict[str, Any] = {}
        self.cache_keys = {
            "fund_flows": "fund:flows",
            "summary": "fund:summary",
            "withdrawal_rules": "fund:withdrawal_rules",
        }
        self.cache_ttl = DEFAULT_CACHE_TTL_SECONDS
        self._cache_operations_queue: asyncio.Queue | None = None
        self._time_series_queue: asyncio.Queue | None = None
        self._max_queue_size = DEFAULT_MAX_QUEUE_SIZE
        self._processing_semaphore: asyncio.Semaphore | None = None

    async def _do_start(self) -> None:
        """Start the fund flow manager service."""
        try:
            try:
                await self._load_configuration()
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
                raise ServiceError(f"Configuration loading failed: {e}") from e

            try:
                self._initialize_withdrawal_rules()
            except Exception as e:
                self.logger.error(f"Failed to initialize withdrawal rules: {e}")
                raise ServiceError(f"Withdrawal rules initialization failed: {e}") from e

            try:
                self._initialize_capital_protection()
            except Exception as e:
                self.logger.error(f"Failed to initialize capital protection: {e}")
                raise ServiceError(f"Capital protection initialization failed: {e}") from e

            self.compound_schedule = self._calculate_compound_schedule()
            self._max_flow_history = self.config.get("max_flow_history", DEFAULT_MAX_FLOW_HISTORY)
            self.cache_ttl = self.config.get("cache_ttl_seconds", DEFAULT_CACHE_TTL_SECONDS)

            import asyncio

            self._max_queue_size = self.config.get("max_queue_size", DEFAULT_MAX_QUEUE_SIZE)
            self._cache_operations_queue = asyncio.Queue(maxsize=self._max_queue_size)
            self._time_series_queue = asyncio.Queue(maxsize=self._max_queue_size)
            self._processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_OPERATIONS)

            self.logger.info(
                "Fund flow manager service started",
                auto_compound_enabled=self.config.get("auto_compound_enabled", True),
                profit_threshold=format_currency(
                    safe_decimal_conversion(
                        self.config.get("profit_threshold", DEFAULT_PROFIT_THRESHOLD)
                    )
                ),
            )
        except Exception as e:
            self.logger.error(f"Failed to start FundFlowManager service: {e}")
            raise ServiceError(f"FundFlowManager startup failed: {e}") from e

    async def _load_configuration(self) -> None:
        """Load configuration from ConfigService."""
        resolved_config_service = resolve_config_service(self)
        self.config = load_capital_config(resolved_config_service)
        self.config = validate_config_values(self.config)

    async def _do_stop(self) -> None:
        """Stop the fund flow manager service and clean up resources."""
        from src.utils.service_utils import safe_service_shutdown

        await safe_service_shutdown(
            service_name="FundFlowManager",
            cleanup_func=self.cleanup_resources,
            service_logger=self.logger,
        )

    def _initialize_capital_protection(self) -> None:
        """Initialize capital protection with current config."""
        total_capital = self.config.get("total_capital", DEFAULT_TOTAL_CAPITAL)
        self.capital_protection = CapitalProtection(
            protection_id="fund_flow_protection",
            enabled=True,
            min_capital_threshold=total_capital * EMERGENCY_RESERVE_PCT,
            stop_trading_threshold=total_capital * MAX_DAILY_LOSS_PCT,
            reduce_size_threshold=total_capital * MAX_WITHDRAWAL_PCT,
            size_reduction_factor=PROFIT_LOCK_PCT,
            max_daily_loss=total_capital
            * safe_decimal_conversion(self.config.get("max_daily_loss_pct", MAX_DAILY_LOSS_PCT)),
            max_weekly_loss=total_capital
            * safe_decimal_conversion(
                self.config.get("max_weekly_loss_pct", DEFAULT_MAX_WEEKLY_LOSS_PCT)
            ),
            max_monthly_loss=total_capital
            * safe_decimal_conversion(
                self.config.get("max_monthly_loss_pct", DEFAULT_MAX_MONTHLY_LOSS_PCT)
            ),
            emergency_threshold=total_capital * EMERGENCY_THRESHOLD_PCT,
            emergency_reserve_pct=self.config.get("emergency_reserve_pct", EMERGENCY_RESERVE_PCT),
            max_daily_loss_pct=self.config.get("max_daily_loss_pct", MAX_DAILY_LOSS_PCT),
            max_weekly_loss_pct=self.config.get("max_weekly_loss_pct", DEFAULT_MAX_WEEKLY_LOSS_PCT),
            max_monthly_loss_pct=self.config.get(
                "max_monthly_loss_pct", DEFAULT_MAX_MONTHLY_LOSS_PCT
            ),
            profit_lock_pct=self.config.get("profit_lock_pct", PROFIT_LOCK_PCT),
            auto_compound_enabled=self.config.get("auto_compound_enabled", True),
        )

    async def _cache_fund_flows(self, flows: list[FundFlow]) -> None:
        """Cache fund flows via cache service with backpressure handling."""
        if not self._cache_service or not self._cache_operations_queue:
            return

        async def _cache_operation():
            async with self._processing_semaphore:  # Limit concurrent operations
                cache_data = None
                try:
                    # Convert FundFlow objects to dict for caching
                    cache_data = [
                        flow.model_dump() if hasattr(flow, "model_dump") else flow.__dict__
                        for flow in flows
                    ]
                    # Use timeout to prevent hanging
                    import asyncio

                    await asyncio.wait_for(
                        self._cache_service.set(
                            self.cache_keys["fund_flows"], cache_data, self.cache_ttl
                        ),
                        timeout=DEFAULT_CACHE_OPERATION_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Cache operation timed out")
                except Exception as e:
                    self.logger.warning("Failed to cache fund flows", error=str(e))
                finally:
                    cache_data = None

        try:
            # Add to queue for processing, drop if queue is full (backpressure)
            self._cache_operations_queue.put_nowait(_cache_operation)
        except asyncio.QueueFull:
            self.logger.warning(
                "Cache operations queue full, dropping cache request (backpressure)"
            )
        except Exception as e:
            self.logger.warning(f"Failed to queue cache operation: {e}")

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
            self.logger.warning("Failed to get cached fund flows", error=str(e))
        finally:
            cached_data = None
            if not flows:
                flows = None
        return None

    async def _store_fund_flow_time_series(self, flow: FundFlow) -> None:
        """Store fund flow in time series service for analysis with backpressure handling."""
        if not self._time_series_service or not self._time_series_queue:
            return

        async def _time_series_operation():
            async with self._processing_semaphore:  # Limit concurrent operations
                try:
                    # Prepare time series data
                    tags = {
                        "component": "fund_flow_manager",
                        "currency": getattr(flow, "currency", DEFAULT_BASE_CURRENCY),
                        "reason": flow.reason,
                        "from_exchange": flow.from_exchange or "none",
                        "to_exchange": flow.to_exchange or "none",
                    }
                    fields = {
                        "amount": str(flow.amount),
                    }

                    # Write to time series service with timeout
                    import asyncio

                    await asyncio.wait_for(
                        self._time_series_service.write_point("fund_flows", tags, fields),
                        timeout=DEFAULT_TIME_SERIES_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Time series operation timed out")
                except Exception as e:
                    self.logger.warning("Failed to store fund flow in time series", error=str(e))

        try:
            # Add to queue for processing, drop if queue is full (backpressure)
            self._time_series_queue.put_nowait(_time_series_operation)
        except asyncio.QueueFull:
            self.logger.warning("Time series queue full, dropping write request (backpressure)")
        except Exception as e:
            self.logger.warning(f"Failed to queue time series operation: {e}")

    @time_execution
    async def process_deposit(
        self,
        amount: Decimal,
        currency: str = DEFAULT_BASE_CURRENCY,
        exchange: str = DEFAULT_EXCHANGE,
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
            min_deposit = safe_decimal_conversion(
                self.config.get("min_deposit_amount", MIN_DEPOSIT_AMOUNT)
            )
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
                self.logger.warning(f"Cache operation failed: {cache_error}")

            # Store fund flow in time series with proper resource management
            try:
                await self._store_fund_flow_time_series(flow)
            except Exception as ts_error:
                self.logger.warning(f"Time series operation failed: {ts_error}")

            self.logger.info(
                "Deposit processed successfully",
                amount=format_currency(amount, currency),
                exchange=exchange,
            )

            return flow

        except Exception as e:
            # Log error information
            self.logger.error(
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
        currency: str = DEFAULT_BASE_CURRENCY,
        exchange: str = DEFAULT_EXCHANGE,
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
            min_withdrawal = safe_decimal_conversion(
                self.config.get("min_withdrawal_amount", MIN_WITHDRAWAL_AMOUNT)
            )
            validate_withdrawal_request(
                amount, currency, exchange, min_withdrawal, "FundFlowManager"
            )

            # Validate withdrawal rules
            await self._validate_withdrawal_rules(amount, currency)

            # Check maximum withdrawal percentage
            if self.total_capital > Decimal("0"):
                max_withdrawal = self.total_capital * safe_decimal_conversion(
                    self.config.get("max_withdrawal_pct", MAX_WITHDRAWAL_PCT)
                )
                if amount > max_withdrawal:
                    raise ValidationError(
                        f"Withdrawal amount {amount} exceeds maximum {max_withdrawal}"
                    )
            else:
                self.logger.warning(
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

            self.logger.info(
                "Withdrawal processed successfully",
                amount=format_currency(amount, currency),
                exchange=exchange,
                reason=reason,
            )

            return flow

        except Exception as e:
            # Log error information
            self.logger.error(
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
            from src.utils.service_utils import validate_positive_amount

            validate_positive_amount(amount, "amount", "reallocation")

            # Check maximum daily reallocation
            if self.total_capital > Decimal("0"):
                max_daily_reallocation = self.total_capital * safe_decimal_conversion(
                    self.config.get("max_daily_reallocation_pct", MAX_DAILY_REALLOCATION_PCT)
                )
                daily_reallocation = await self._get_daily_reallocation_amount()

                if daily_reallocation + amount > max_daily_reallocation:
                    raise ValidationError(
                        f"Reallocation would exceed daily limit. Current: {daily_reallocation}, "
                        f"Requested: {amount}, Limit: {max_daily_reallocation}"
                    )
            else:
                self.logger.warning("Total capital not set, skipping reallocation limit validation")

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

            self.logger.info(
                "Strategy reallocation processed",
                from_strategy=from_strategy,
                to_strategy=to_strategy,
                amount=format_currency(amount),
                reason=reason,
            )

            return flow

        except Exception as e:
            self.logger.error(
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

            self.logger.info(
                "Auto-compound processed",
                compound_amount=format_currency(compound_amount),
                total_locked_profit=format_currency(self.locked_profit),
            )

            return flow

        except Exception as e:
            self.logger.error("Auto-compound processing failed", error=str(e))
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
            # Convert float values to Decimal for internal storage
            self.strategy_performance[strategy_id] = {
                key: safe_decimal_conversion(value) for key, value in performance_metrics.items()
            }

            # Update total profit
            if "total_pnl" in performance_metrics:
                self.total_profit = safe_decimal_conversion(performance_metrics["total_pnl"])

            self.logger.info(
                "Performance updated",
                strategy_id=strategy_id,
                total_profit=format_currency(self.total_profit),
            )

        except Exception as e:
            self.logger.error("Performance update failed", strategy_id=strategy_id, error=str(e))

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
            self.logger.error("Failed to get flow history", error=str(e))
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
                currency = getattr(flow, "currency", DEFAULT_BASE_CURRENCY)
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
            self.logger.error("Failed to get flow summary", error=str(e))
            raise ServiceError(f"Failed to get flow summary: {e}") from e

    def _initialize_withdrawal_rules(self) -> None:
        """Initialize withdrawal rules from configuration."""
        # Check if withdrawal_rules is a dict
        withdrawal_rules_config = self.config.get("withdrawal_rules", {})
        if not isinstance(withdrawal_rules_config, dict):
            self.logger.warning("No withdrawal rules configured")
            return

        for rule_name, rule_config in withdrawal_rules_config.items():
            # Ensure rule_config is a dict
            if not isinstance(rule_config, dict):
                self.logger.warning(f"Invalid config for withdrawal rule {rule_name}")
                continue

            rule = WithdrawalRule(
                name=rule_name,
                description=rule_config.get("description", ""),
                enabled=rule_config.get("enabled", True),
                threshold=rule_config.get("threshold"),
                min_amount=(
                    safe_decimal_conversion(rule_config.get("min_amount", 0))
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
                        performance_ok = await self._check_performance_threshold(
                            safe_decimal_conversion(rule.threshold)
                        )
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
                    max_amount = self.total_capital * safe_decimal_conversion(rule.max_percentage)
                    if amount > max_amount:
                        raise ValidationError(
                            f"Withdrawal amount {amount} exceeds maximum {max_amount} "
                            f"({rule.max_percentage:.1%} of total capital)"
                        )

        except ValidationError:
            # Re-raise ValidationError without wrapping
            raise
        except Exception as e:
            self.logger.error("Withdrawal rule validation failed", error=str(e))
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
                cooldown_hours = (
                    self.config.get(
                        "fund_flow_cooldown_minutes", DEFAULT_FUND_FLOW_COOLDOWN_MINUTES
                    )
                    / 60
                )

                if datetime.now(timezone.utc) - last_withdrawal.timestamp < timedelta(
                    hours=cooldown_hours
                ):
                    raise ValidationError(
                        f"Withdrawal not allowed: Cooldown period not met. "
                        f"Last withdrawal: {last_withdrawal.timestamp}"
                    )

        except ValidationError:
            # Re-raise ValidationError as-is for test compatibility
            raise
        except Exception as e:
            self.logger.error("Withdrawal cooldown check failed", error=str(e))
            raise ServiceError(f"Withdrawal cooldown check failed: {e}") from e

    async def _calculate_minimum_capital_required(self) -> Decimal:
        """Calculate minimum capital required for all strategies."""
        try:
            total_minimum = Decimal("0")

            per_strategy_minimum = self.config.get("per_strategy_minimum", {})
            if not isinstance(per_strategy_minimum, dict):
                return DEFAULT_PROFIT_THRESHOLD  # Default minimum

            for strategy_type, min_amount in per_strategy_minimum.items():
                # Check if this strategy type is active
                active_strategies = [
                    strategy_id
                    for strategy_id in self.strategy_performance.keys()
                    if strategy_type in strategy_id.lower()
                ]

                if active_strategies:
                    total_minimum += safe_decimal_conversion(min_amount)

            # If no minimum calculated but we have strategy performance, use default
            if total_minimum == Decimal("0") and self.strategy_performance:
                return DEFAULT_PROFIT_THRESHOLD

            # If no strategy performance at all, return default minimum
            if total_minimum == Decimal("0"):
                return DEFAULT_PROFIT_THRESHOLD

            return total_minimum

        except Exception as e:
            self.logger.error("Failed to calculate minimum capital required", error=str(e))
            return DEFAULT_PROFIT_THRESHOLD  # Default minimum

    async def _check_performance_threshold(self, threshold: Decimal) -> bool:
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

            # Convert threshold to Decimal to ensure proper comparison
            threshold_decimal = safe_decimal_conversion(threshold)

            # Calculate overall performance
            total_return = Decimal("0")
            strategy_count = 0

            for _strategy_id, metrics in self.strategy_performance.items():
                if "total_pnl" in metrics and "initial_capital" in metrics:
                    initial_capital = safe_decimal_conversion(metrics["initial_capital"])
                    if initial_capital > 0:
                        return_rate = (
                            safe_decimal_conversion(metrics["total_pnl"]) / initial_capital
                        )
                        total_return += return_rate
                        strategy_count += 1

            if strategy_count > 0:
                avg_return = total_return / strategy_count
                return avg_return >= threshold_decimal

            return False

        except Exception as e:
            self.logger.error("Failed to check performance threshold", error=str(e))
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
            self.logger.error("Failed to get daily reallocation amount", error=str(e))
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
                return days_since_last >= COMPOUND_FREQUENCY_DAYS
            else:
                return False

        except Exception as e:
            self.logger.error("Failed to check compound timing", error=str(e))
            return False

    async def _calculate_compound_amount(self) -> Decimal:
        """Calculate amount to compound based on profits."""
        try:
            profit_threshold = Decimal(
                str(self.config.get("profit_threshold", DEFAULT_PROFIT_THRESHOLD))
            )
            if self.total_profit <= profit_threshold:
                return Decimal("0")

            # Calculate compound amount (profit above threshold)
            compound_amount = self.total_profit - profit_threshold

            # Apply profit lock percentage
            locked_amount = compound_amount * Decimal(
                str(self.config.get("profit_lock_pct", PROFIT_LOCK_PCT))
            )

            return locked_amount

        except Exception as e:
            self.logger.error("Failed to calculate compound amount", error=str(e))
            return Decimal("0")

    def _calculate_compound_schedule(self) -> dict[str, Any]:
        """Calculate compound schedule based on frequency."""
        try:
            frequency = self.config.get("auto_compound_frequency", "weekly")
            schedule = {
                "frequency": frequency,
                "next_compound": self.last_compound_date
                + timedelta(days=7 if frequency == "weekly" else COMPOUND_FREQUENCY_DAYS),
                "enabled": self.config.get("auto_compound_enabled", True),
            }

            return schedule

        except Exception as e:
            self.logger.error("Failed to calculate compound schedule", error=str(e))
            return {}

    async def update_total_capital(self, total_capital: Decimal) -> None:
        """
        Update total capital from actual account balances.

        Args:
            total_capital: Current total capital across all exchanges
        """
        try:
            self.total_capital = total_capital
            self.logger.info("Total capital updated", total_capital=format_currency(total_capital))
        except Exception as e:
            self.logger.error("Failed to update total capital", error=str(e))
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
                "total_profit": str(self.total_profit),
                "locked_profit": str(self.locked_profit),
                "auto_compound_enabled": self.capital_protection.auto_compound_enabled,
                "next_compound_date": self.compound_schedule.get(
                    "next_compound", datetime.now(timezone.utc)
                ),
                # Protection is active if there's profit
                "protection_active": self.total_profit > 0 or self.locked_profit > 0,
            }

            return status

        except Exception as e:
            self.logger.error("Failed to get capital protection status", error=str(e))
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
                        total_pnl += safe_decimal_conversion(pnl_value)
                    elif isinstance(pnl_value, Decimal):
                        total_pnl += pnl_value
                else:
                    if isinstance(metrics, int | float):
                        total_pnl += safe_decimal_conversion(metrics)
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
                        pnl_value = safe_decimal_conversion(pnl_value)
                    performance_score = metrics.get("performance_score", Decimal("0.0"))
                    if isinstance(performance_score, int | float):
                        performance_score = safe_decimal_conversion(performance_score)

                    summary["strategies"][strategy_id] = {
                        "pnl": pnl_value,
                        "performance_score": performance_score,
                        "last_updated": metrics.get("last_updated", datetime.now(timezone.utc)),
                    }
                else:
                    # Handle case where metrics might be a single value
                    if isinstance(metrics, int | float):
                        pnl_value = safe_decimal_conversion(metrics)
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
            self.logger.error("Failed to get performance summary", error=str(e))
            raise ServiceError(f"Failed to get performance summary: {e}") from e

    def _validate_config(self) -> None:
        """Validate and set default configuration values."""
        # Set defaults for any missing config attributes
        config_defaults = {
            "total_capital": DEFAULT_TOTAL_CAPITAL,
            "emergency_reserve_pct": EMERGENCY_RESERVE_PCT,
            "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
            "max_weekly_loss_pct": float(DEFAULT_MAX_WEEKLY_LOSS_PCT),
            "max_monthly_loss_pct": float(DEFAULT_MAX_MONTHLY_LOSS_PCT),
            "profit_lock_pct": PROFIT_LOCK_PCT,
            "auto_compound_enabled": True,
            "auto_compound_frequency": "weekly",
            "profit_threshold": DEFAULT_PROFIT_THRESHOLD,
            "min_deposit_amount": MIN_DEPOSIT_AMOUNT,
            "min_withdrawal_amount": MIN_WITHDRAWAL_AMOUNT,
            "max_withdrawal_pct": MAX_WITHDRAWAL_PCT,
            "max_daily_reallocation_pct": MAX_DAILY_REALLOCATION_PCT,
            "fund_flow_cooldown_minutes": DEFAULT_FUND_FLOW_COOLDOWN_MINUTES,
            "max_flow_history": DEFAULT_MAX_FLOW_HISTORY,
            "cache_ttl_seconds": DEFAULT_CACHE_TTL_SECONDS,
            "withdrawal_rules": {},
            "per_strategy_minimum": {},
        }

        for key, default_value in config_defaults.items():
            if key not in self.config:
                self.config[key] = default_value
                if hasattr(self, "_logger"):
                    self.logger.warning(f"Config missing {key}, using default: {default_value}")

    async def cleanup_resources(self) -> None:
        """Clean up resources to prevent memory leaks with proper async handling."""
        from src.utils.capital_resources import async_cleanup_resources, get_resource_manager

        try:
            resource_manager = get_resource_manager()

            # Define cleanup tasks to run concurrently
            async def clean_fund_flows():
                """Clean fund flows data."""
                self.fund_flows = resource_manager.clean_fund_flows(
                    self.fund_flows, max_age_days=DEFAULT_PERFORMANCE_WINDOW_DAYS
                )

            async def clean_performance_data():
                """Clean strategy performance data."""
                self.strategy_performance = resource_manager.clean_performance_data(
                    self.strategy_performance, max_age_days=DEFAULT_PERFORMANCE_WINDOW_DAYS
                )

            async def clean_async_queues():
                """Clean up async queues and pending operations."""
                import asyncio

                try:
                    # Process remaining items in queues with timeout
                    if self._cache_operations_queue:
                        remaining_cache_ops = []
                        while not self._cache_operations_queue.empty():
                            try:
                                op = self._cache_operations_queue.get_nowait()
                                remaining_cache_ops.append(op)
                            except asyncio.QueueEmpty:
                                break

                        if remaining_cache_ops:
                            try:
                                await asyncio.wait_for(
                                    asyncio.gather(
                                        *[op() for op in remaining_cache_ops],
                                        return_exceptions=True,
                                    ),
                                    timeout=DEFAULT_CLEANUP_TIMEOUT,
                                )
                            except asyncio.TimeoutError:
                                self.logger.warning("Cache operations cleanup timed out")

                    if self._time_series_queue:
                        remaining_ts_ops = []
                        while not self._time_series_queue.empty():
                            try:
                                op = self._time_series_queue.get_nowait()
                                remaining_ts_ops.append(op)
                            except asyncio.QueueEmpty:
                                break

                        if remaining_ts_ops:
                            try:
                                await asyncio.wait_for(
                                    asyncio.gather(
                                        *[op() for op in remaining_ts_ops], return_exceptions=True
                                    ),
                                    timeout=DEFAULT_CLEANUP_TIMEOUT,
                                )
                            except asyncio.TimeoutError:
                                self.logger.warning("Time series operations cleanup timed out")

                except Exception as e:
                    self.logger.warning(f"Queue cleanup failed: {e}")

            # Use common cleanup utility to reduce duplication
            await async_cleanup_resources(
                clean_fund_flows(),
                clean_performance_data(),
                clean_async_queues(),
                logger_instance=self.logger,
            )

            self.logger.info("Fund flow manager resource cleanup completed")
        except Exception as e:
            self.logger.warning(f"Resource cleanup failed: {e}")
