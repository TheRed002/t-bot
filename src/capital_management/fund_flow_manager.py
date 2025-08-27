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
from typing import Any

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import ValidationError

# MANDATORY: Import from P-001
from src.core.types.capital import (
    CapitalFundFlow as FundFlow,
    ExtendedCapitalProtection as CapitalProtection,
    ExtendedWithdrawalRule as WithdrawalRule,
)

# Database connections will be provided through service layer
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import PartialFillRecovery
from src.utils.decorators import time_execution
from src.utils.formatters import format_currency
from src.utils.validators import ValidationFramework

# Conditional import for InfluxDB
try:
    from influxdb_client import Point
except ImportError:
    Point = None  # Handle case where influxdb_client is not installed

# MANDATORY: Use structured logging from src.core.logging for all capital
# management operations

# From P-002A - MANDATORY: Use error handling

# From P-007A - MANDATORY: Use decorators and validators


class FundFlowManager(BaseComponent):
    """
    Deposit/withdrawal management system.

    This class manages capital flows, enforces withdrawal rules,
    and implements auto-compounding features for optimal capital growth.
    """

    def __init__(self, config: Config, error_handler: ErrorHandler | None = None):
        """
        Initialize the fund flow manager.

        Args:
            config: Application configuration
            error_handler: Optional error handler instance (uses DI if not provided)
        """
        super().__init__()  # Initialize BaseComponent

        # Validate config
        if not hasattr(config, "capital_management"):
            raise ValidationError("Missing capital_management configuration")

        self.config = config.capital_management

        # Validate required config attributes with defaults
        self._validate_config()

        # Flow tracking
        self.fund_flows: list[FundFlow] = []
        self.withdrawal_rules: dict[str, WithdrawalRule] = {}
        self.capital_protection = CapitalProtection(
            protection_id="fund_flow_protection",
            enabled=True,
            min_capital_threshold=Decimal(str(self.config.total_capital * 0.1)),
            stop_trading_threshold=Decimal(str(self.config.total_capital * 0.05)),
            reduce_size_threshold=Decimal(str(self.config.total_capital * 0.2)),
            size_reduction_factor=0.5,
            max_daily_loss=Decimal(str(self.config.total_capital * self.config.max_daily_loss_pct)),
            max_weekly_loss=Decimal(
                str(self.config.total_capital * self.config.max_weekly_loss_pct)
            ),
            max_monthly_loss=Decimal(
                str(self.config.total_capital * self.config.max_monthly_loss_pct)
            ),
            emergency_threshold=Decimal(str(self.config.total_capital * 0.02)),
            emergency_reserve_pct=self.config.emergency_reserve_pct,
            max_daily_loss_pct=self.config.max_daily_loss_pct,
            max_weekly_loss_pct=self.config.max_weekly_loss_pct,
            max_monthly_loss_pct=self.config.max_monthly_loss_pct,
            profit_lock_pct=self.config.profit_lock_pct,
            auto_compound_enabled=self.config.auto_compound_enabled,
        )

        # Performance tracking
        self.strategy_performance: dict[str, dict[str, float]] = {}
        self.total_profit = Decimal("0")
        self.locked_profit = Decimal("0")

        # Capital management integration
        self.capital_allocator = None  # Will be set by CapitalAllocator integration
        # Total capital will be updated from actual account balances
        self.total_capital = Decimal("0")

        # Auto-compounding tracking
        self.last_compound_date = datetime.now(timezone.utc)
        self.compound_schedule = self._calculate_compound_schedule()

        # Error handler - use dependency injection or provided instance
        if error_handler:
            self.error_handler = error_handler
        else:
            try:
                from src.core.dependency_injection import get_container

                self.error_handler = get_container().get("ErrorHandler")
            except (ImportError, KeyError):
                # Fallback to creating instance if DI not available
                self.error_handler = ErrorHandler(config)

        # Recovery scenarios
        self.partial_fill_recovery = PartialFillRecovery(config)

        # Database clients will be injected through service layer
        self.redis_client = None
        self.influx_client = None
        self.logger.info("Database connections will be managed by service layer")

        # Cache keys
        self.cache_keys = {
            "fund_flows": "fund:flows",
            "summary": "fund:summary",
            "withdrawal_rules": "fund:withdrawal_rules",
        }

        # Cache TTL (seconds)
        self.cache_ttl = 300  # 5 minutes

        # Initialize withdrawal rules
        self._initialize_withdrawal_rules()

        self.logger.info(
            "Fund flow manager initialized",
            auto_compound_enabled=self.config.auto_compound_enabled,
            profit_threshold=format_currency(float(self.config.profit_threshold)),
        )

    async def _cache_fund_flows(self, flows: list[FundFlow]) -> None:
        """Cache fund flows in Redis."""
        if not self.redis_client:
            return
        try:
            # Convert FundFlow objects to dict for caching
            cache_data = [flow.model_dump() for flow in flows]
            await self.redis_client.set(
                self.cache_keys["fund_flows"], cache_data, ttl=self.cache_ttl
            )
        except Exception as e:
            self.logger.warning("Failed to cache fund flows", error=str(e))

    async def _get_cached_fund_flows(self) -> list[FundFlow] | None:
        """Get cached fund flows from Redis."""
        if not self.redis_client:
            return None
        try:
            cached_data = await self.redis_client.get(self.cache_keys["fund_flows"])
            if cached_data:
                # Convert cached data back to FundFlow objects
                flows = [FundFlow(**data) for data in cached_data]
                return flows
        except Exception as e:
            self.logger.warning("Failed to get cached fund flows", error=str(e))
        return None

    async def _store_fund_flow_influxdb(self, flow: FundFlow) -> None:
        """Store fund flow in InfluxDB for time series analysis."""
        if not self.influx_client or Point is None:
            return
        try:
            # Create a point for fund flows
            point = (
                Point("fund_flows")
                .tag("component", "fund_flow_manager")
                .tag("currency", flow.currency)
                .tag("reason", flow.reason)
                .tag("from_exchange", flow.from_exchange or "none")
                .tag("to_exchange", flow.to_exchange or "none")
                .field("amount", float(flow.amount))
            )

            # Write to InfluxDB
            self.influx_client.write_api().write(bucket="trading_bot", record=point)
        except Exception as e:
            self.logger.warning("Failed to store fund flow in InfluxDB", error=str(e))

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
            # Validate inputs
            if not ValidationFramework.validate_quantity(float(amount)):
                raise ValidationError(f"Invalid deposit amount: {amount}")

            if amount < Decimal(str(self.config.min_deposit_amount)):
                raise ValidationError(
                    f"Deposit amount {amount} below minimum {self.config.min_deposit_amount}"
                )

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

            # Add to flow history
            self.fund_flows.append(flow)

            # Cache fund flows
            await self._cache_fund_flows(self.fund_flows)

            # Store fund flow in InfluxDB
            await self._store_fund_flow_influxdb(flow)

            self.logger.info(
                "Deposit processed successfully",
                amount=format_currency(float(amount), currency),
                exchange=exchange,
            )

            return flow

        except Exception as e:
            # Create comprehensive error context
            from src.error_handling.context import ErrorContext

            context = ErrorContext.from_exception(
                error=e,
                component="capital_management",
                operation="process_deposit",
                details={
                    "amount": float(amount),
                    "currency": currency,
                    "exchange": exchange,
                    "total_capital": float(self.total_capital),
                },
            )

            # Handle error with recovery strategy
            await self.error_handler.handle_error(e, context, self.partial_fill_recovery)

            # Log detailed error information
            self.logger.error(
                "Deposit processing failed",
                error_id=context.error_id,
                severity=context.severity.value,
                amount=format_currency(float(amount), currency),
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
            # Validate inputs
            if not ValidationFramework.validate_quantity(float(amount)):
                raise ValidationError(f"Invalid withdrawal amount: {amount}")

            # Check minimum withdrawal amount
            if amount < Decimal(str(self.config.min_withdrawal_amount)):
                raise ValidationError(
                    f"Withdrawal amount {amount} below minimum {self.config.min_withdrawal_amount}"
                )

            # Validate withdrawal rules
            await self._validate_withdrawal_rules(amount, currency)

            # Check maximum withdrawal percentage
            if self.total_capital > Decimal("0"):
                max_withdrawal = self.total_capital * Decimal(str(self.config.max_withdrawal_pct))
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

            # Add to flow history
            self.fund_flows.append(flow)

            self.logger.info(
                "Withdrawal processed successfully",
                amount=format_currency(float(amount), currency),
                exchange=exchange,
                reason=reason,
            )

            return flow

        except Exception as e:
            # Create comprehensive error context
            from src.error_handling.context import ErrorContext

            context = ErrorContext.from_exception(
                error=e,
                component="capital_management",
                operation="process_withdrawal",
                details={
                    "amount": float(amount),
                    "currency": currency,
                    "exchange": exchange,
                    "reason": reason,
                    "total_capital": float(self.total_capital),
                },
            )

            # Handle error with recovery strategy
            await self.error_handler.handle_error(e, context, self.partial_fill_recovery)

            # Log detailed error information
            self.logger.error(
                "Withdrawal processing failed",
                error_id=context.error_id,
                severity=context.severity.value,
                amount=format_currency(float(amount), currency),
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
            if not ValidationFramework.validate_quantity(float(amount)):
                raise ValidationError(f"Invalid reallocation amount: {amount}")

            # Check maximum daily reallocation
            if self.total_capital > Decimal("0"):
                max_daily_reallocation = self.total_capital * Decimal(
                    str(self.config.max_daily_reallocation_pct)
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

            # Add to flow history
            self.fund_flows.append(flow)

            self.logger.info(
                "Strategy reallocation processed",
                from_strategy=from_strategy,
                to_strategy=to_strategy,
                amount=format_currency(float(amount)),
                reason=reason,
            )

            return flow

        except Exception as e:
            self.logger.error(
                "Strategy reallocation failed",
                from_strategy=from_strategy,
                to_strategy=to_strategy,
                amount=format_currency(float(amount)),
                error=str(e),
            )
            raise

    @time_execution
    async def process_auto_compound(self) -> FundFlow | None:
        """
        Process auto-compounding of profits.

        Returns:
            Optional[FundFlow]: Compound flow record if applicable
        """
        try:
            if not self.config.auto_compound_enabled:
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

            # Add to flow history
            self.fund_flows.append(flow)

            # Update tracking
            self.last_compound_date = datetime.now(timezone.utc)
            self.locked_profit += compound_amount

            self.logger.info(
                "Auto-compound processed",
                compound_amount=format_currency(float(compound_amount)),
                total_locked_profit=format_currency(float(self.locked_profit)),
            )

            return flow

        except Exception as e:
            self.logger.error("Auto-compound processing failed", error=str(e))
            raise

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

            self.logger.debug(
                "Performance updated",
                strategy_id=strategy_id,
                total_profit=format_currency(float(self.total_profit)),
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
            raise

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
                "total_deposits": float(total_deposits),
                "total_withdrawals": float(total_withdrawals),
                "total_reallocations": float(total_reallocations),
                "total_compounds": float(total_compounds),
                "net_flow": float(total_deposits - total_withdrawals),
                "currency_flows": {
                    curr: {k: float(v) for k, v in stats.items()}
                    for curr, stats in currency_flows.items()
                },
            }

            return summary

        except Exception as e:
            self.logger.error("Failed to get flow summary", error=str(e))
            raise

    def _initialize_withdrawal_rules(self) -> None:
        """Initialize withdrawal rules from configuration."""
        # Check if withdrawal_rules is a dict
        withdrawal_rules_config = self.config.withdrawal_rules
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
            self.logger.error("Withdrawal rule validation failed", error=str(e))
            raise

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
                cooldown_hours = self.config.fund_flow_cooldown_minutes / 60

                if datetime.now(timezone.utc) - last_withdrawal.timestamp < timedelta(
                    hours=cooldown_hours
                ):
                    raise ValidationError(
                        f"Withdrawal not allowed: Cooldown period not met. "
                        f"Last withdrawal: {last_withdrawal.timestamp}"
                    )

        except Exception as e:
            self.logger.error("Withdrawal cooldown check failed", error=str(e))
            raise

    async def _calculate_minimum_capital_required(self) -> Decimal:
        """Calculate minimum capital required for all strategies."""
        try:
            total_minimum = Decimal("0")

            per_strategy_minimum = getattr(self.config, "per_strategy_minimum", {})
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
            self.logger.error("Failed to calculate minimum capital required", error=str(e))
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

            return sum(flow.amount for flow in today_flows)

        except Exception as e:
            self.logger.error("Failed to get daily reallocation amount", error=str(e))
            return Decimal("0")

    def _should_compound(self) -> bool:
        """Check if it's time to compound profits."""
        try:
            if not self.config.auto_compound_enabled:
                return False

            # Check frequency
            if self.config.auto_compound_frequency == "weekly":
                days_since_last = (datetime.now(timezone.utc) - self.last_compound_date).days
                return days_since_last >= 7
            elif self.config.auto_compound_frequency == "monthly":
                days_since_last = (datetime.now(timezone.utc) - self.last_compound_date).days
                return days_since_last >= 30
            else:
                return False

        except Exception as e:
            self.logger.error("Failed to check compound timing", error=str(e))
            return False

    async def _calculate_compound_amount(self) -> Decimal:
        """Calculate amount to compound based on profits."""
        try:
            if self.total_profit <= self.config.profit_threshold:
                return Decimal("0")

            # Calculate compound amount (profit above threshold)
            compound_amount = self.total_profit - self.config.profit_threshold

            # Apply profit lock percentage
            locked_amount = compound_amount * Decimal(str(self.config.profit_lock_pct))

            return locked_amount

        except Exception as e:
            self.logger.error("Failed to calculate compound amount", error=str(e))
            return Decimal("0")

    def _calculate_compound_schedule(self) -> dict[str, Any]:
        """Calculate compound schedule based on frequency."""
        try:
            schedule = {
                "frequency": self.config.auto_compound_frequency,
                "next_compound": self.last_compound_date
                + timedelta(days=7 if self.config.auto_compound_frequency == "weekly" else 30),
                "enabled": self.config.auto_compound_enabled,
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
            self.logger.info(
                "Total capital updated", total_capital=format_currency(float(total_capital))
            )
        except Exception as e:
            self.logger.error("Failed to update total capital", error=str(e))
            raise

    async def get_total_capital(self) -> Decimal:
        """Get current total capital."""
        return self.total_capital

    async def get_capital_protection_status(self) -> dict[str, Any]:
        """Get current capital protection status."""
        try:
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
            self.logger.error("Failed to get capital protection status", error=str(e))
            raise

    async def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for all strategies."""
        try:
            total_pnl = 0.0

            # Calculate total PnL from strategy performance
            for strategy_id, metrics in self.strategy_performance.items():
                if isinstance(metrics, dict):
                    total_pnl += metrics.get("pnl", 0.0)
                else:
                    # Handle case where metrics might be a single value
                    total_pnl += (
                        float(metrics) if isinstance(metrics, int | float | Decimal) else 0.0
                    )

            summary = {
                "total_pnl": total_pnl,
                "total_profit": float(self.total_profit),
                "locked_profit": float(self.locked_profit),
                "strategy_count": len(self.strategy_performance),
                "strategies": {},
            }

            for strategy_id, metrics in self.strategy_performance.items():
                if isinstance(metrics, dict):
                    summary["strategies"][strategy_id] = {
                        "pnl": metrics.get("pnl", 0.0),
                        "performance_score": metrics.get("performance_score", 0.0),
                        "last_updated": metrics.get("last_updated", datetime.now(timezone.utc)),
                    }
                else:
                    # Handle case where metrics might be a single value
                    summary["strategies"][strategy_id] = {
                        "pnl": (
                            float(metrics) if isinstance(metrics, int | float | Decimal) else 0.0
                        ),
                        "performance_score": 0.0,
                        "last_updated": datetime.now(timezone.utc),
                    }

            return summary

        except Exception as e:
            self.logger.error("Failed to get performance summary", error=str(e))
            raise

    def set_capital_allocator(self, capital_allocator) -> None:
        """
        Set the capital allocator for integration.

        Args:
            capital_allocator: CapitalAllocator instance
        """
        self.capital_allocator = capital_allocator
        self.logger.info("Capital allocator integration established")

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
            "withdrawal_rules": {},
            "per_strategy_minimum": {},
        }

        for key, default_value in config_defaults.items():
            if not hasattr(self.config, key):
                setattr(self.config, key, default_value)
                self.logger.warning(f"Config missing {key}, using default: {default_value}")
