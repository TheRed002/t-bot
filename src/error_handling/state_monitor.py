"""
State consistency monitor for cross-system state validation.

This module provides cross-system state consistency checking, automatic state
reconciliation procedures, state corruption detection, and real-time state
validation alerts.

CRITICAL: This module integrates with P-001 core framework and P-002 database
for state persistence and will be used by all subsequent prompts.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.utils.decimal_utils import to_decimal

# MANDATORY: Import from P-001 core framework
# MANDATORY: Import from P-007A utils framework
from src.utils.decorators import retry, time_execution


@dataclass
class StateValidationResult:
    """Result of state validation check."""

    is_consistent: bool
    discrepancies: list[dict[str, Any]] = field(default_factory=list)
    validation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    component: str = ""
    severity: str = "low"  # low, medium, high, critical


class StateMonitor:
    """Monitors and validates state consistency across system components."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(self.__class__.__module__)
        # Create default state monitoring config if not present
        self.state_monitoring_config = getattr(
            config,
            "error_handling",
            {
                "state_validation_frequency": 60,
                "max_state_drift_tolerance": 0.01,
                "state_history_retention_days": 7,
            },
        )
        self.validation_frequency = self.state_monitoring_config.get(
            "state_validation_frequency", 60
        )
        self.consistency_checks = [
            "portfolio_balance_sync",
            "position_quantity_sync",
            "order_status_sync",
            "risk_limit_compliance",
        ]
        self.reconciliation_config = self.state_monitoring_config
        self.auto_reconcile = self.reconciliation_config.get("auto_reconciliation_enabled", True)
        self.max_discrepancy = self.reconciliation_config.get("max_discrepancy_threshold", 0.01)
        self.force_sync_threshold = 0.05  # Default threshold

        # State tracking
        self.last_validation_results: dict[str, StateValidationResult] = {}
        self.state_history: list[StateValidationResult] = []
        self.reconciliation_attempts: dict[str, int] = {}

    def _safe_to_decimal(self, value: Any, field_name: str = "value") -> Decimal:
        """Safely convert value to Decimal with validation."""
        if value is None:
            raise ValidationError(f"{field_name} cannot be None for decimal conversion")

        try:
            return to_decimal(value)
        except (ValueError, TypeError, ValidationError) as e:
            self.logger.error(
                "Failed to convert value to Decimal",
                field_name=field_name,
                value=value,
                error=str(e),
            )
            raise ValidationError(f"Invalid {field_name} for decimal conversion: {value}")

    @time_execution
    @retry(max_attempts=2)
    async def validate_state_consistency(self, component: str = "all") -> StateValidationResult:
        """Validate state consistency for specified component or all components."""

        self.logger.info("Starting state consistency validation", component=component)

        discrepancies = []
        is_consistent = True
        severity = "low"

        if component == "all":
            # Validate all components
            for check in self.consistency_checks:
                try:
                    check_result = await self._perform_consistency_check(check)
                    if not check_result["is_consistent"]:
                        is_consistent = False
                        discrepancies.extend(check_result["discrepancies"])
                        severity = max(severity, check_result["severity"])

                except Exception as e:
                    self.logger.error("State consistency check failed", check=check, error=str(e))
                    is_consistent = False
                    discrepancies.append(
                        {"check": check, "error": str(e), "type": "validation_error"}
                    )
                    severity = "critical"
        else:
            # Validate specific component
            try:
                check_result = await self._perform_consistency_check(component)
                if not check_result["is_consistent"]:
                    is_consistent = False
                    discrepancies.extend(check_result["discrepancies"])
                    severity = check_result["severity"]

            except Exception as e:
                self.logger.error(
                    "Component state validation failed", component=component, error=str(e)
                )
                is_consistent = False
                discrepancies.append(
                    {"component": component, "error": str(e), "type": "validation_error"}
                )
                severity = "critical"

        result = StateValidationResult(
            is_consistent=is_consistent,
            discrepancies=discrepancies,
            component=component,
            severity=severity,
        )

        # Store result
        self.last_validation_results[component] = result
        self.state_history.append(result)

        # Keep only last 1000 validation results
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]

        self.logger.info(
            "State consistency validation completed",
            component=component,
            is_consistent=is_consistent,
            discrepancy_count=len(discrepancies),
            severity=severity,
        )

        return result

    async def _perform_consistency_check(self, check_name: str) -> dict[str, Any]:
        """Perform a specific consistency check."""

        if check_name == "portfolio_balance_sync":
            return await self._check_portfolio_balance_sync()
        elif check_name == "position_quantity_sync":
            return await self._check_position_quantity_sync()
        elif check_name == "order_status_sync":
            return await self._check_order_status_sync()
        elif check_name == "risk_limit_compliance":
            return await self._check_risk_limit_compliance()
        else:
            self.logger.warning("Unknown consistency check", check_name=check_name)
            return {"is_consistent": True, "discrepancies": [], "severity": "low"}

    async def _check_portfolio_balance_sync(self) -> dict[str, Any]:
        """Check if portfolio balances are synchronized across systems."""
        # Using centralized decimal utilities from utils module

        try:
            discrepancies = []
            is_consistent = True
            severity = "low"

            # Use service layer for data access
            try:
                from src.database.service import DatabaseService
                from src.exchanges.factory import ExchangeFactory

                db_service = DatabaseService(self.config)
                await db_service.initialize()

                # Get balances from database via service
                db_balances = {}
                balance_records = await db_service.get_active_balances()
                for balance in balance_records:
                    key = f"{balance['exchange']}:{balance['currency']}"
                    db_balances[key] = {
                        "available": self._safe_to_decimal(
                            balance["available"], "balance.available"
                        ),
                        "locked": self._safe_to_decimal(balance["locked"], "balance.locked"),
                        "total": self._safe_to_decimal(balance["available"], "balance.available")
                        + self._safe_to_decimal(balance["locked"], "balance.locked"),
                    }

                # Get balances from Redis cache via service
                cache_balances = {}
                cached_balance_records = await db_service.get_cached_balances()
                for balance in cached_balance_records:
                    key = f"{balance['exchange']}:{balance['currency']}"
                    cache_balances[key] = {
                        "available": self._safe_to_decimal(
                            balance.get("available", 0), "cache.available"
                        ),
                        "locked": self._safe_to_decimal(balance.get("locked", 0), "cache.locked"),
                        "total": self._safe_to_decimal(balance.get("total", 0), "cache.total"),
                    }

                # Get balances from exchanges (if connected)
                exchange_balances = {}
                try:
                    for exchange_name in getattr(self.config, "exchange", {}).get(
                        "enabled_exchanges", []
                    ):
                        exchange = ExchangeFactory.create(exchange_name, self.config)
                        if hasattr(exchange, "get_account_balance"):
                            balance = await exchange.get_account_balance()
                            for currency, amounts in balance.items():
                                key = f"{exchange_name}:{currency}"
                                exchange_balances[key] = {
                                    "available": self._safe_to_decimal(
                                        amounts.get("free", 0), "exchange.available"
                                    ),
                                    "locked": self._safe_to_decimal(
                                        amounts.get("used", 0), "exchange.locked"
                                    ),
                                    "total": self._safe_to_decimal(
                                        amounts.get("total", 0), "exchange.total"
                                    ),
                                }
                except Exception as e:
                    self.logger.warning(f"Could not fetch exchange balances: {e}")

                # Compare balances across sources
                all_keys = (
                    set(db_balances.keys())
                    | set(cache_balances.keys())
                    | set(exchange_balances.keys())
                )

                for key in all_keys:
                    db_val = db_balances.get(key, {"total": self._safe_to_decimal("0", "default")})
                    cache_val = cache_balances.get(
                        key, {"total": self._safe_to_decimal("0", "default")}
                    )
                    exchange_val = exchange_balances.get(
                        key, {"total": self._safe_to_decimal("0", "default")}
                    )

                    # Check for discrepancies (with tolerance for rounding)
                    tolerance = self._safe_to_decimal("0.00000001", "tolerance")

                    if db_val["total"] > 0 or cache_val["total"] > 0 or exchange_val["total"] > 0:
                        max_diff = max(
                            abs(db_val["total"] - cache_val["total"]),
                            (
                                abs(db_val["total"] - exchange_val["total"])
                                if exchange_val["total"] > 0
                                else self._safe_to_decimal("0", "default")
                            ),
                            (
                                abs(cache_val["total"] - exchange_val["total"])
                                if exchange_val["total"] > 0
                                else self._safe_to_decimal("0", "default")
                            ),
                        )

                        if max_diff > tolerance:
                            discrepancy = {
                                "type": "balance_mismatch",
                                "key": key,
                                "db_balance": str(db_val["total"]),
                                "cache_balance": str(cache_val["total"]),
                                "exchange_balance": (
                                    str(exchange_val["total"])
                                    if exchange_val["total"] > 0
                                    else "N/A"
                                ),
                                "max_difference": str(max_diff),
                            }
                            discrepancies.append(discrepancy)

                            # Determine severity based on difference
                            if max_diff > self._safe_to_decimal("1.0", "severity_threshold"):
                                severity = "critical"
                                is_consistent = False
                            elif max_diff > self._safe_to_decimal("0.1", "severity_threshold"):
                                severity = "high" if severity != "critical" else severity
                                is_consistent = False
                            elif max_diff > self._safe_to_decimal("0.01", "severity_threshold"):
                                severity = (
                                    "medium" if severity not in ["critical", "high"] else severity
                                )

            except Exception as e:
                self.logger.warning(f"Error during balance comparison: {e}")
                discrepancies.append({"type": "comparison_error", "error": str(e)})
                severity = "high"
                is_consistent = False

            self.logger.info(
                f"Portfolio balance sync check completed. Found {len(discrepancies)} discrepancies"
            )

            return {
                "is_consistent": is_consistent,
                "discrepancies": discrepancies,
                "severity": severity,
            }

        except Exception as e:
            self.logger.error("Portfolio balance sync check failed", error=str(e))
            return {
                "is_consistent": False,
                "discrepancies": [{"error": str(e), "type": "balance_sync_error"}],
                "severity": "high",
            }
        finally:
            # Clean up service connections
            try:
                if "db_service" in locals():
                    await db_service.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup database service: {e}")

    async def _check_position_quantity_sync(self) -> dict[str, Any]:
        """Check if position quantities are synchronized across systems."""
        # Using centralized decimal utilities from utils module

        try:
            discrepancies = []
            is_consistent = True
            severity = "low"

            # Use service layer for data access
            try:
                from src.database.service import DatabaseService
                from src.exchanges.factory import ExchangeFactory
                from src.risk_management.service import RiskManagementService

                db_service = DatabaseService(self.config)
                risk_service = RiskManagementService(self.config)
                await db_service.initialize()
                await risk_service.initialize()

                # Get positions from database via service
                db_positions = {}
                position_records = await db_service.get_open_positions()
                for position in position_records:
                    key = f"{position['exchange']}:{position['symbol']}:{position['side']}"
                    db_positions[key] = {
                        "quantity": self._safe_to_decimal(
                            position["quantity"], "position.quantity"
                        ),
                        "entry_price": self._safe_to_decimal(
                            position["entry_price"], "position.entry_price"
                        ),
                    }

                # Get positions from cache via service
                cache_positions = {}
                cached_position_records = await db_service.get_cached_positions()
                for position in cached_position_records:
                    if position.get("status") == "open":
                        key = f"{position['exchange']}:{position['symbol']}:{position['side']}"
                        cache_positions[key] = {
                            "quantity": self._safe_to_decimal(
                                position.get("quantity", 0), "cache.position.quantity"
                            ),
                            "entry_price": self._safe_to_decimal(
                                position.get("entry_price", 0), "cache.position.entry_price"
                            ),
                        }
            except Exception as e:
                self.logger.error(f"Error initializing data sources for position sync: {e}")
                db_positions = {}
                cache_positions = {}

            # Get positions from exchanges (if connected)
            exchange_positions = {}
            try:
                for exchange_name in getattr(self.config, "exchange", {}).get(
                    "enabled_exchanges", []
                ):
                    exchange = ExchangeFactory.create(exchange_name, self.config)
                    if hasattr(exchange, "get_positions"):
                        positions = await exchange.get_positions()
                        for pos in positions:
                            key = f"{exchange_name}:{pos.symbol}:{pos.side}"
                            exchange_positions[key] = {
                                "quantity": self._safe_to_decimal(
                                    pos.quantity, "exchange.position.quantity"
                                ),
                                "entry_price": self._safe_to_decimal(
                                    pos.entry_price, "exchange.position.entry_price"
                                ),
                            }
            except Exception as e:
                self.logger.warning(f"Could not fetch exchange positions: {e}")

            # Get positions from risk management system via service
            risk_positions = {}
            try:
                risk_position_records = await risk_service.get_current_positions()
                for pos_key, pos_data in risk_position_records.items():
                    risk_positions[pos_key] = {
                        "quantity": self._safe_to_decimal(
                            pos_data.get("quantity", 0), "risk.position.quantity"
                        ),
                        "entry_price": self._safe_to_decimal(
                            pos_data.get("entry_price", 0), "risk.position.entry_price"
                        ),
                    }
            except Exception as e:
                self.logger.warning(f"Could not fetch risk positions: {e}")

            # Compare positions across sources
            all_keys = (
                set(db_positions.keys())
                | set(cache_positions.keys())
                | set(exchange_positions.keys())
                | set(risk_positions.keys())
            )

            for key in all_keys:
                db_pos = db_positions.get(key, {"quantity": self._safe_to_decimal("0", "default")})
                cache_pos = cache_positions.get(
                    key, {"quantity": self._safe_to_decimal("0", "default")}
                )
                exchange_pos = exchange_positions.get(key, {"quantity": Decimal("0")})
                risk_pos = risk_positions.get(key, {"quantity": Decimal("0")})

                # Check for quantity discrepancies
                quantities = [
                    db_pos["quantity"],
                    cache_pos["quantity"],
                    (
                        exchange_pos["quantity"]
                        if exchange_pos["quantity"] > 0
                        else db_pos["quantity"]
                    ),
                    risk_pos["quantity"] if risk_pos["quantity"] > 0 else db_pos["quantity"],
                ]

                max_qty = max(quantities)
                min_qty = (
                    min(q for q in quantities if q > 0)
                    if any(q > 0 for q in quantities)
                    else Decimal("0")
                )
                qty_diff = max_qty - min_qty

                tolerance = Decimal("0.00000001")

                if qty_diff > tolerance and max_qty > 0:
                    discrepancy = {
                        "type": "position_quantity_mismatch",
                        "key": key,
                        "db_quantity": str(db_pos["quantity"]),
                        "cache_quantity": str(cache_pos["quantity"]),
                        "exchange_quantity": (
                            str(exchange_pos["quantity"]) if exchange_pos["quantity"] > 0 else "N/A"
                        ),
                        "risk_quantity": (
                            str(risk_pos["quantity"]) if risk_pos["quantity"] > 0 else "N/A"
                        ),
                        "max_difference": str(qty_diff),
                    }
                    discrepancies.append(discrepancy)

                    # Determine severity based on difference
                    if qty_diff > Decimal("1.0"):
                        severity = "critical"
                        is_consistent = False
                    elif qty_diff > Decimal("0.1"):
                        severity = "high" if severity != "critical" else severity
                        is_consistent = False
                    elif qty_diff > Decimal("0.01"):
                        severity = "medium" if severity not in ["critical", "high"] else severity

            self.logger.info(
                f"Position quantity sync check completed. Found {len(discrepancies)} discrepancies"
            )

            return {
                "is_consistent": is_consistent,
                "discrepancies": discrepancies,
                "severity": severity,
            }

        except Exception as e:
            self.logger.error("Position quantity sync check failed", error=str(e))
            return {
                "is_consistent": False,
                "discrepancies": [{"error": str(e), "type": "position_sync_error"}],
                "severity": "high",
            }
        finally:
            # Clean up service connections
            try:
                if "db_service" in locals():
                    await db_service.cleanup()
                if "risk_service" in locals():
                    await risk_service.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup services during position sync: {e}")

    async def _check_order_status_sync(self) -> dict[str, Any]:
        """Check if order statuses are synchronized across systems."""

        try:
            # TODO: Implement actual order status synchronization check
            # This will be implemented in P-020 (Order Management and Execution
            # Engine)

            # Simulate order status check
            discrepancies: list[dict[str, Any]] = []
            is_consistent = True
            severity = "low"

            # TODO: Compare order statuses from:
            # - Database (P-002)
            # - Exchange APIs (P-003+)
            # - Redis cache (P-002)
            # - Execution engine (P-020)

            self.logger.info("Order status sync check completed")

            return {
                "is_consistent": is_consistent,
                "discrepancies": discrepancies,
                "severity": severity,
            }

        except Exception as e:
            self.logger.error("Order status sync check failed", error=str(e))
            return {
                "is_consistent": False,
                "discrepancies": [{"error": str(e), "type": "order_sync_error"}],
                "severity": "high",
            }

    async def _check_risk_limit_compliance(self) -> dict[str, Any]:
        """Check if risk limits are being complied with."""
        # Using centralized decimal utilities from utils module

        try:
            discrepancies = []
            is_consistent = True
            severity = "low"

            # Use service layer for data access
            try:
                from src.database.service import DatabaseService
                from src.risk_management.service import RiskManagementService

                db_service = DatabaseService(self.config)
                risk_service = RiskManagementService(self.config)
                await db_service.initialize()
                await risk_service.initialize()

                # Get current risk metrics via service
                risk_metrics = await risk_service.get_current_risk_metrics()

                # Check position size limits via service
                open_positions = await db_service.get_open_positions_with_prices()
                max_position_size = Decimal(
                    str(getattr(self.config, "risk", {}).get("max_position_size", 1000000))
                )

                for position in open_positions:
                    position_value = Decimal(str(position["quantity"])) * Decimal(
                        str(position["current_price"] or position["entry_price"])
                    )

                    if position_value > max_position_size:
                        discrepancy = {
                            "type": "position_size_limit_exceeded",
                            "symbol": position["symbol"],
                            "position_value": str(position_value),
                            "max_allowed": str(max_position_size),
                            "excess": str(position_value - max_position_size),
                        }
                        discrepancies.append(discrepancy)
                        severity = "critical"
                        is_consistent = False

                # Check portfolio exposure limits
                total_exposure = Decimal(str(risk_metrics.get("total_exposure", 0)))
                max_exposure = Decimal(
                    str(getattr(self.config, "risk", {}).get("max_portfolio_exposure", 100000))
                )

                if total_exposure > max_exposure:
                    discrepancy = {
                        "type": "portfolio_exposure_limit_exceeded",
                        "total_exposure": str(total_exposure),
                        "max_allowed": str(max_exposure),
                        "excess": str(total_exposure - max_exposure),
                    }
                    discrepancies.append(discrepancy)
                    severity = "critical"
                    is_consistent = False

                # Check leverage limits
                current_leverage = Decimal(str(risk_metrics.get("current_leverage", 1)))
                max_leverage = Decimal(
                    str(getattr(self.config, "risk", {}).get("max_leverage", 10))
                )

                if current_leverage > max_leverage:
                    discrepancy = {
                        "type": "leverage_limit_exceeded",
                        "current_leverage": str(current_leverage),
                        "max_allowed": str(max_leverage),
                        "excess": str(current_leverage - max_leverage),
                    }
                    discrepancies.append(discrepancy)
                    severity = "critical"
                    is_consistent = False

            except Exception as e:
                self.logger.error(f"Error initializing risk management components: {e}")
                # Continue with default values

            # Check stop loss compliance via service
            positions_without_stop = await db_service.count_positions_without_stop_loss()

            if positions_without_stop > 0:
                discrepancy = {
                    "type": "stop_loss_missing",
                    "positions_without_stop_loss": positions_without_stop,
                    "severity": "high",
                }
                discrepancies.append(discrepancy)
                if severity not in ["critical"]:
                    severity = "high"
                is_consistent = False

            # Check maximum drawdown limits
            current_drawdown = Decimal(str(risk_metrics.get("current_drawdown", 0)))
            max_drawdown = Decimal(str(getattr(self.config, "risk", {}).get("max_drawdown", 0.20)))

            if abs(current_drawdown) > max_drawdown:
                discrepancy = {
                    "type": "max_drawdown_exceeded",
                    "current_drawdown": str(current_drawdown),
                    "max_allowed": str(max_drawdown),
                    "excess": str(abs(current_drawdown) - max_drawdown),
                }
                discrepancies.append(discrepancy)
                severity = "critical"
                is_consistent = False

            # Check daily loss limit
            daily_loss = Decimal(str(risk_metrics.get("daily_pnl", 0)))
            max_daily_loss = Decimal(
                str(getattr(self.config, "risk", {}).get("max_daily_loss", 1000))
            )

            if daily_loss < 0 and abs(daily_loss) > max_daily_loss:
                discrepancy = {
                    "type": "daily_loss_limit_exceeded",
                    "daily_loss": str(daily_loss),
                    "max_allowed": str(max_daily_loss),
                    "excess": str(abs(daily_loss) - max_daily_loss),
                }
                discrepancies.append(discrepancy)
                severity = "critical"
                is_consistent = False

            # Check correlation limits
            correlation_risk = Decimal(str(risk_metrics.get("correlation_risk", 0)))
            max_correlation = Decimal(
                str(getattr(self.config, "risk", {}).get("max_correlation_risk", 0.80))
            )

            if correlation_risk > max_correlation:
                discrepancy = {
                    "type": "correlation_risk_exceeded",
                    "correlation_risk": str(correlation_risk),
                    "max_allowed": str(max_correlation),
                    "excess": str(correlation_risk - max_correlation),
                }
                discrepancies.append(discrepancy)
                if severity not in ["critical", "high"]:
                    severity = "medium"

            self.logger.info(
                f"Risk limit compliance check completed. Found {len(discrepancies)} violations"
            )

            return {
                "is_consistent": is_consistent,
                "discrepancies": discrepancies,
                "severity": severity,
            }

        except Exception as e:
            self.logger.error("Risk limit compliance check failed", error=str(e))
            return {
                "is_consistent": False,
                "discrepancies": [{"error": str(e), "type": "risk_compliance_error"}],
                "severity": "critical",
            }
        finally:
            # Clean up service connections
            try:
                if "db_service" in locals():
                    await db_service.cleanup()
                if "risk_service" in locals():
                    await risk_service.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup services during risk compliance check: {e}")

    async def reconcile_state(self, component: str, discrepancies: list[dict[str, Any]]) -> bool:
        """Attempt to reconcile state discrepancies."""

        self.logger.info(
            "Attempting state reconciliation",
            component=component,
            discrepancy_count=len(discrepancies),
        )

        if not self.auto_reconcile:
            self.logger.info("Auto-reconciliation disabled", component=component)
            return False

        reconciliation_attempts = self.reconciliation_attempts.get(component, 0)
        max_attempts = 3

        if reconciliation_attempts >= max_attempts:
            self.logger.warning(
                "Max reconciliation attempts reached",
                component=component,
                attempts=reconciliation_attempts,
            )
            return False

        self.reconciliation_attempts[component] = reconciliation_attempts + 1

        try:
            if component == "portfolio_balance_sync":
                success = await self._reconcile_portfolio_balances(discrepancies)
            elif component == "position_quantity_sync":
                success = await self._reconcile_position_quantities(discrepancies)
            elif component == "order_status_sync":
                success = await self._reconcile_order_statuses(discrepancies)
            elif component == "risk_limit_compliance":
                success = await self._reconcile_risk_limits(discrepancies)
            else:
                self.logger.warning("Unknown reconciliation component", component=component)
                return False

            if success:
                self.logger.info("State reconciliation successful", component=component)
                # Reset reconciliation attempts on success
                self.reconciliation_attempts[component] = 0
            else:
                self.logger.warning("State reconciliation failed", component=component)

            return success

        except Exception as e:
            self.logger.error("State reconciliation error", component=component, error=str(e))
            return False

    async def _reconcile_portfolio_balances(self, discrepancies: list[dict[str, Any]]) -> bool:
        """Reconcile portfolio balance discrepancies."""
        # Using centralized decimal utilities from utils module

        try:
            self.logger.info("Reconciling portfolio balances", discrepancy_count=len(discrepancies))

            # Use service layer for data access
            try:
                from src.database.service import DatabaseService
                from src.exchanges.factory import ExchangeFactory

                db_service = DatabaseService(self.config)
                await db_service.initialize()

                reconciled_count = 0

                for discrepancy in discrepancies:
                    if discrepancy.get("type") != "balance_mismatch":
                        continue

                    key = discrepancy.get("key", "")
                    if not key:
                        continue

                    parts = key.split(":")
                    if len(parts) < 2:
                        continue

                    exchange_name = parts[0]
                    currency = parts[1]

                    # Get truth from exchange (most reliable source)
                    try:
                        exchange = ExchangeFactory.create(exchange_name, self.config)
                        if hasattr(exchange, "get_account_balance"):
                            balance = await exchange.get_account_balance()
                            if currency in balance:
                                true_balance = {
                                    "available": Decimal(str(balance[currency].get("free", 0))),
                                    "locked": Decimal(str(balance[currency].get("used", 0))),
                                    "total": Decimal(str(balance[currency].get("total", 0))),
                                }

                                # Update database via service
                                await db_service.update_balance(
                                    exchange=exchange_name,
                                    currency=currency,
                                    available=str(true_balance["available"]),
                                    locked=str(true_balance["locked"]),
                                )

                                # Update cache via service
                                await db_service.update_balance_cache(
                                    exchange=exchange_name,
                                    currency=currency,
                                    balance_data={
                                        "available": str(true_balance["available"]),
                                        "locked": str(true_balance["locked"]),
                                        "total": str(true_balance["total"]),
                                        "updated_at": datetime.now(timezone.utc).isoformat(),
                                    },
                                )

                                # Update metrics via service
                                await db_service.record_balance_reconciliation(
                                    exchange=exchange_name,
                                    currency=currency,
                                    balance_data={
                                        "available": str(true_balance["available"]),
                                        "locked": str(true_balance["locked"]),
                                        "total": str(true_balance["total"]),
                                    },
                                )

                                reconciled_count += 1
                                self.logger.info(
                                    f"Reconciled balance for {key}",
                                    true_balance=str(true_balance["total"]),
                                )

                    except Exception as e:
                        self.logger.error(f"Failed to reconcile balance for {key}: {e}")
                        continue

            except Exception as e:
                self.logger.error(f"Error during reconciliation setup: {e}")
                reconciled_count = 0

            success = reconciled_count == len(
                [d for d in discrepancies if d.get("type") == "balance_mismatch"]
            )

            if success:
                self.logger.info(
                    f"Successfully reconciled all {reconciled_count} balance discrepancies"
                )
            else:
                self.logger.warning(
                    f"Reconciled {reconciled_count} out of {len(discrepancies)} balance discrepancies"
                )

            return success

        except Exception as e:
            self.logger.error(f"Failed to reconcile portfolio balances: {e}")
            return False
        finally:
            # Clean up service connections
            try:
                if "db_service" in locals():
                    await db_service.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup services during balance reconciliation: {e}")

    async def _reconcile_position_quantities(self, discrepancies: list[dict[str, Any]]) -> bool:
        """Reconcile position quantity discrepancies."""

        try:
            self.logger.info(
                "Reconciling position quantities", discrepancy_count=len(discrepancies)
            )

            # Use service layer for data access
            try:
                from src.database.service import DatabaseService
                from src.exchanges.factory import ExchangeFactory
                from src.risk_management.service import RiskManagementService

                db_service = DatabaseService(self.config)
                risk_service = RiskManagementService(self.config)
                await db_service.initialize()
                await risk_service.initialize()

                reconciled_count = 0

                for discrepancy in discrepancies:
                    if discrepancy.get("type") != "position_quantity_mismatch":
                        continue

                    key = discrepancy.get("key", "")
                    if not key:
                        continue

                    parts = key.split(":")
                    if len(parts) < 3:
                        continue

                    exchange_name = parts[0]
                    symbol = parts[1]
                    side = parts[2]

                    # Get truth from exchange
                    try:
                        exchange = ExchangeFactory.create(exchange_name, self.config)
                        if hasattr(exchange, "get_positions"):
                            positions = await exchange.get_positions()

                            true_position = None
                            for pos in positions:
                                if pos.symbol == symbol and pos.side == side:
                                    true_position = pos
                                    break

                            if true_position:
                                # Update database via service
                                await db_service.update_position(
                                    exchange=exchange_name,
                                    symbol=symbol,
                                    side=side,
                                    quantity=str(true_position.quantity),
                                    current_price=str(true_position.current_price),
                                )

                                # Update cache via service
                                await db_service.update_position_cache(
                                    exchange=exchange_name,
                                    symbol=symbol,
                                    side=side,
                                    position_data={
                                        "quantity": str(true_position.quantity),
                                        "entry_price": str(true_position.entry_price),
                                        "current_price": str(true_position.current_price),
                                        "status": "open",
                                        "updated_at": datetime.now(timezone.utc).isoformat(),
                                    },
                                )

                                # Update risk management system via service
                                await risk_service.update_position(
                                    symbol=symbol,
                                    quantity=true_position.quantity,
                                    side=side,
                                    exchange=exchange_name,
                                )

                                reconciled_count += 1
                                self.logger.info(
                                    f"Reconciled position for {key}",
                                    true_quantity=str(true_position.quantity),
                                )
                            else:
                                # Position closed on exchange, update local state via service
                                await db_service.close_position(
                                    exchange=exchange_name, symbol=symbol, side=side
                                )

                                # Remove from cache via service
                                await db_service.remove_position_from_cache(
                                    exchange=exchange_name, symbol=symbol, side=side
                                )

                                reconciled_count += 1
                                self.logger.info(f"Marked position {key} as closed")

                    except Exception as e:
                        self.logger.error(f"Failed to reconcile position for {key}: {e}")
                        continue

            except Exception as e:
                self.logger.error(f"Error during position reconciliation setup: {e}")
                reconciled_count = 0

            success = reconciled_count == len(
                [d for d in discrepancies if d.get("type") == "position_quantity_mismatch"]
            )

            if success:
                self.logger.info(
                    f"Successfully reconciled all {reconciled_count} position discrepancies"
                )
            else:
                self.logger.warning(
                    f"Reconciled {reconciled_count} out of {len(discrepancies)} position discrepancies"
                )

            return success

        finally:
            # Clean up service connections
            try:
                if "db_service" in locals():
                    await db_service.cleanup()
                if "risk_service" in locals():
                    await risk_service.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup services during position reconciliation: {e}")

    async def _reconcile_order_statuses(self, discrepancies: list[dict[str, Any]]) -> bool:
        """Reconcile order status discrepancies."""

        try:
            self.logger.info("Reconciling order statuses", discrepancy_count=len(discrepancies))

            # Use service layer for data access
            try:
                from src.database.service import DatabaseService
                from src.exchanges.factory import ExchangeFactory
                from src.execution.service import ExecutionService

                db_service = DatabaseService(self.config)
                execution_service = ExecutionService(self.config)
                await db_service.initialize()
                await execution_service.initialize()

                reconciled_count = 0

                for discrepancy in discrepancies:
                    if discrepancy.get("type") not in [
                        "order_status_mismatch",
                        "order_filled_quantity_mismatch",
                    ]:
                        continue

                    order_id = discrepancy.get("order_id")
                    if not order_id:
                        continue

                    # Determine exchange from order_id or discrepancy data
                    exchange_name = None
                    for key in ["db_status", "cache_status", "exchange_status", "execution_status"]:
                        if key in discrepancy and discrepancy[key] != "N/A":
                            # Try to get exchange from database
                            async with db_manager.get_session() as session:
                                result = await session.execute(
                                    "SELECT exchange FROM orders WHERE order_id = :order_id",
                                    {"order_id": order_id},
                                )
                                row = result.first()
                                if row:
                                    exchange_name = row.exchange
                                    break

                    if not exchange_name:
                        self.logger.warning(f"Could not determine exchange for order {order_id}")
                        continue

                    # Get truth from exchange
                    try:
                        exchange = ExchangeFactory.create(exchange_name, self.config)
                        if hasattr(exchange, "get_order"):
                            true_order = await exchange.get_order(order_id)

                            if true_order:
                                # Update database via service
                                await db_service.update_order_status(
                                    order_id=order_id,
                                    status=true_order.status,
                                    filled_quantity=str(true_order.filled_quantity),
                                    remaining_quantity=str(true_order.remaining_quantity),
                                )

                                # Update cache via service
                                await db_service.update_order_cache(
                                    order_id=order_id,
                                    order_data={
                                        "order_id": order_id,
                                        "status": true_order.status,
                                        "filled_quantity": str(true_order.filled_quantity),
                                        "remaining_quantity": str(true_order.remaining_quantity),
                                        "exchange": exchange_name,
                                        "symbol": true_order.symbol,
                                        "updated_at": datetime.now(timezone.utc).isoformat(),
                                    },
                                )

                                # Update execution engine via service
                                await execution_service.update_order_status(
                                    order_id=order_id,
                                    status=true_order.status,
                                    filled_quantity=true_order.filled_quantity,
                                    remaining_quantity=true_order.remaining_quantity,
                                )

                                reconciled_count += 1
                                self.logger.info(
                                    f"Reconciled order {order_id}",
                                    true_status=true_order.status,
                                    filled=str(true_order.filled_quantity),
                                )
                            else:
                                # Order not found on exchange, mark as cancelled/expired via service
                                await db_service.cancel_order(order_id)

                                # Update cache via service
                                await db_service.update_order_cache_status(order_id, "cancelled")

                                reconciled_count += 1
                                self.logger.info(f"Marked order {order_id} as cancelled")

                    except Exception as e:
                        self.logger.error(f"Failed to reconcile order {order_id}: {e}")
                        continue

            except Exception as e:
                self.logger.error(f"Error during order reconciliation setup: {e}")
                reconciled_count = 0

            relevant_discrepancies = [
                d
                for d in discrepancies
                if d.get("type") in ["order_status_mismatch", "order_filled_quantity_mismatch"]
            ]
            success = reconciled_count == len(relevant_discrepancies)

            if success:
                self.logger.info(
                    f"Successfully reconciled all {reconciled_count} order discrepancies"
                )
            else:
                self.logger.warning(
                    f"Reconciled {reconciled_count} out of {len(relevant_discrepancies)} order discrepancies"
                )

            return success

        finally:
            # Clean up service connections
            try:
                if "db_service" in locals():
                    await db_service.cleanup()
                if "execution_service" in locals():
                    await execution_service.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup services during order reconciliation: {e}")

    async def _reconcile_risk_limits(self, discrepancies: list[dict[str, Any]]) -> bool:
        """Reconcile risk limit compliance issues."""

        try:
            self.logger.info("Reconciling risk limits", discrepancy_count=len(discrepancies))

            # Use service layer for data access
            from src.database.service import DatabaseService
            from src.execution.service import ExecutionService
            from src.risk_management.service import RiskManagementService

            db_service = DatabaseService(self.config)
            execution_service = ExecutionService(self.config)
            risk_service = RiskManagementService(self.config)
            await db_service.initialize()
            await execution_service.initialize()
            await risk_service.initialize()

            reconciled_count = 0
            critical_actions_taken = []

            for discrepancy in discrepancies:
                discrepancy_type = discrepancy.get("type")

                try:
                    if discrepancy_type == "position_size_limit_exceeded":
                        symbol = discrepancy.get("symbol")
                        excess = Decimal(discrepancy.get("excess", "0"))

                        if symbol and excess > 0:
                            await execution_service.cancel_orders_by_symbol(symbol)
                            await risk_service.reduce_position(symbol=symbol, amount=excess)
                            critical_actions_taken.append(
                                f"Reduced position for {symbol} by {excess}"
                            )
                            reconciled_count += 1

                    elif discrepancy_type == "portfolio_exposure_limit_exceeded":
                        excess = Decimal(discrepancy.get("excess", "0"))
                        await execution_service.cancel_all_orders()
                        await risk_service.reduce_portfolio_exposure(excess)
                        critical_actions_taken.append(f"Reduced portfolio exposure by {excess}")
                        reconciled_count += 1

                    elif discrepancy_type == "leverage_limit_exceeded":
                        current_leverage = Decimal(discrepancy.get("current_leverage", "1"))
                        max_leverage = Decimal(discrepancy.get("max_allowed", "1"))
                        reduction_factor = max_leverage / current_leverage
                        await risk_service.adjust_leverage(reduction_factor)
                        critical_actions_taken.append(
                            f"Reduced leverage from {current_leverage} to {max_leverage}"
                        )
                        reconciled_count += 1

                    elif discrepancy_type == "stop_loss_missing":
                        positions_without_stop = discrepancy.get("positions_without_stop_loss", 0)
                        updated_positions = await db_service.add_missing_stop_losses()
                        critical_actions_taken.append(
                            f"Added stop losses to {updated_positions} positions"
                        )
                        reconciled_count += 1

                    elif discrepancy_type == "max_drawdown_exceeded":
                        await risk_service.activate_emergency_shutdown("Max drawdown exceeded")
                        critical_actions_taken.append(
                            "Activated emergency shutdown due to max drawdown"
                        )
                        reconciled_count += 1

                    elif discrepancy_type == "daily_loss_limit_exceeded":
                        await risk_service.halt_trading("Daily loss limit exceeded")
                        await execution_service.cancel_all_orders()
                        critical_actions_taken.append("Halted trading due to daily loss limit")
                        reconciled_count += 1

                    elif discrepancy_type == "correlation_risk_exceeded":
                        excess = Decimal(discrepancy.get("excess", "0"))
                        await risk_service.reduce_correlation_risk(excess)
                        critical_actions_taken.append(f"Reduced correlation risk by {excess}")
                        reconciled_count += 1

                except Exception as e:
                    self.logger.error(f"Failed to reconcile {discrepancy_type}: {e}")
                    continue

            # Log all critical actions
            if critical_actions_taken:
                self.logger.critical(
                    "Risk limit reconciliation actions taken", actions=critical_actions_taken
                )
                for action in critical_actions_taken:
                    await risk_service.send_alert(
                        level="critical", message=f"Risk reconciliation: {action}"
                    )

            success = reconciled_count == len(discrepancies)
            if success:
                self.logger.info(
                    f"Successfully reconciled all {reconciled_count} risk limit violations"
                )
            else:
                self.logger.warning(
                    f"Reconciled {reconciled_count} out of {len(discrepancies)} risk limit violations"
                )

            return success

        except Exception as e:
            self.logger.error("Risk limit reconciliation failed", error=str(e))
            return False
        finally:
            # Clean up service connections
            try:
                if "db_service" in locals():
                    await db_service.cleanup()
                if "execution_service" in locals():
                    await execution_service.cleanup()
                if "risk_service" in locals():
                    await risk_service.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup services during risk reconciliation: {e}")

    async def start_monitoring(self):
        """Start continuous state monitoring."""

        self.logger.info("Starting state monitoring")

        while True:
            try:
                # Validate all components
                result = await self.validate_state_consistency("all")

                if not result.is_consistent:
                    self.logger.warning(
                        "State inconsistency detected",
                        discrepancy_count=len(result.discrepancies),
                        severity=result.severity,
                    )

                    # Attempt reconciliation for each component with
                    # discrepancies
                    for discrepancy in result.discrepancies:
                        component = discrepancy.get("component", "unknown")
                        if component != "unknown":
                            await self.reconcile_state(component, [discrepancy])

                # Wait for next validation cycle
                await asyncio.sleep(self.validation_frequency)

            except Exception as e:
                self.logger.error("State monitoring error", error=str(e))
                await asyncio.sleep(self.validation_frequency)

    def get_state_summary(self) -> dict[str, Any]:
        """Get summary of current state monitoring status."""

        summary = {
            "last_validation_results": {},
            "reconciliation_attempts": self.reconciliation_attempts.copy(),
            "total_validations": len(self.state_history),
            "recent_inconsistencies": 0,
        }

        # Add last validation results
        for component, result in self.last_validation_results.items():
            summary["last_validation_results"][component] = {
                "is_consistent": result.is_consistent,
                "discrepancy_count": len(result.discrepancies),
                "severity": result.severity,
                "validation_time": result.validation_time.isoformat(),
            }

        # Count recent inconsistencies (last 24 hours)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        for result in self.state_history:
            if not result.is_consistent and result.validation_time > recent_cutoff:
                summary["recent_inconsistencies"] += 1

        return summary

    def get_state_history(self, hours: int = 24) -> list[StateValidationResult]:
        """Get state validation history for the specified time period."""

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [result for result in self.state_history if result.validation_time > cutoff]
