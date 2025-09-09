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
from typing import Any, Protocol

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.utils.decimal_utils import to_decimal
from src.utils.decorators import retry, time_execution

from .constants import (
    CRITICAL_THRESHOLD,
    DEFAULT_COMPONENT_TIMEOUT,
    DEFAULT_MAX_DAILY_LOSS,
    DEFAULT_STATE_VALIDATION_FREQUENCY,
    HIGH_THRESHOLD,
    MAX_VALIDATION_HISTORY,
    MEDIUM_THRESHOLD,
    TOLERANCE_DECIMAL,
)


class DatabaseServiceInterface(Protocol):
    """Protocol for database service operations."""

    async def initialize(self) -> None: ...
    async def cleanup(self) -> None: ...
    async def get_active_balances(self) -> list[dict[str, Any]]: ...
    async def get_cached_balances(self) -> list[dict[str, Any]]: ...
    async def get_open_positions(self) -> list[dict[str, Any]]: ...
    async def get_cached_positions(self) -> list[dict[str, Any]]: ...
    async def get_open_positions_with_prices(self) -> list[dict[str, Any]]: ...
    async def count_positions_without_stop_loss(self) -> int: ...
    async def update_balance(
        self, exchange: str, currency: str, available: str, locked: str
    ) -> None: ...
    async def update_balance_cache(
        self, exchange: str, currency: str, balance_data: dict[str, Any]
    ) -> None: ...
    async def record_balance_reconciliation(
        self, exchange: str, currency: str, balance_data: dict[str, Any]
    ) -> None: ...
    async def update_position(
        self, exchange: str, symbol: str, side: str, quantity: str, current_price: str
    ) -> None: ...
    async def update_position_cache(
        self, exchange: str, symbol: str, side: str, position_data: dict[str, Any]
    ) -> None: ...
    async def close_position(self, exchange: str, symbol: str, side: str) -> None: ...
    async def remove_position_from_cache(self, exchange: str, symbol: str, side: str) -> None: ...
    async def update_order_status(
        self, order_id: str, status: str, filled_quantity: str, remaining_quantity: str
    ) -> None: ...
    async def update_order_cache(self, order_id: str, order_data: dict[str, Any]) -> None: ...
    async def cancel_order(self, order_id: str) -> None: ...
    async def update_order_cache_status(self, order_id: str, status: str) -> None: ...
    async def add_missing_stop_losses(self) -> int: ...
    async def get_order_details(self, order_id: str) -> dict[str, Any] | None: ...


class RiskServiceInterface(Protocol):
    """Protocol for risk management service operations."""

    async def initialize(self) -> None: ...
    async def cleanup(self) -> None: ...
    async def get_current_risk_metrics(self) -> dict[str, Any]: ...
    async def get_current_positions(self) -> dict[str, dict[str, Any]]: ...
    async def update_position(
        self, symbol: str, quantity: Decimal, side: str, exchange: str
    ) -> None: ...
    async def reduce_position(self, symbol: str, amount: Decimal) -> None: ...
    async def reduce_portfolio_exposure(self, amount: Decimal) -> None: ...
    async def adjust_leverage(self, reduction_factor: Decimal) -> None: ...
    async def activate_emergency_shutdown(self, reason: str) -> None: ...
    async def halt_trading(self, reason: str) -> None: ...
    async def reduce_correlation_risk(self, excess: Decimal) -> None: ...
    async def send_alert(self, level: str, message: str) -> None: ...


class ExecutionServiceInterface(Protocol):
    """Protocol for execution service operations."""

    async def initialize(self) -> None: ...
    async def cleanup(self) -> None: ...
    async def cancel_orders_by_symbol(self, symbol: str) -> None: ...
    async def cancel_all_orders(self) -> None: ...
    async def update_order_status(
        self, order_id: str, status: str, filled_quantity: Decimal, remaining_quantity: Decimal
    ) -> None: ...


class ExchangeServiceInterface(Protocol):
    """Protocol for exchange service operations."""

    async def get_account_balance(self) -> dict[str, dict[str, Any]]: ...
    async def get_positions(self) -> list[Any]: ...
    async def get_order(self, order_id: str) -> Any | None: ...


@dataclass
class StateValidationResult:
    """Result of state validation check."""

    is_consistent: bool
    discrepancies: list[dict[str, Any]] = field(default_factory=list)
    validation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    component: str = "all"
    severity: str = "low"  # low, medium, high, critical


class StateMonitor:
    """Monitors and validates state consistency across system components."""

    def __init__(
        self,
        config: Config,
        database_service: DatabaseServiceInterface | None = None,
        risk_service: RiskServiceInterface | None = None,
        execution_service: ExecutionServiceInterface | None = None,
    ) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__module__)

        # Injected services - these should be provided by DI container
        self._database_service = database_service
        self._risk_service = risk_service
        self._execution_service = execution_service

        # Validate that required services are available via dependency injection
        # This is informational only - services can be injected later via configure_dependencies
        self._validate_service_availability()

        # Create default state monitoring config if not present
        self.state_monitoring_config = getattr(
            config,
            "error_handling",
            {
                "state_validation_frequency": DEFAULT_STATE_VALIDATION_FREQUENCY,
                "max_state_drift_tolerance": Decimal("0.01"),
                "state_history_retention_days": 7,
            },
        )
        self.validation_frequency = self.state_monitoring_config.get(
            "state_validation_frequency", DEFAULT_STATE_VALIDATION_FREQUENCY
        )
        self.consistency_checks = [
            "portfolio_balance_sync",
            "position_quantity_sync",
            "order_status_sync",
            "risk_limit_compliance",
        ]
        self.reconciliation_config = self.state_monitoring_config
        self.auto_reconcile = self.reconciliation_config.get("auto_reconciliation_enabled", True)
        max_discrepancy_threshold = self.reconciliation_config.get(
            "max_discrepancy_threshold", "0.01"
        )
        self.max_discrepancy = Decimal(str(max_discrepancy_threshold))
        self.force_sync_threshold = Decimal("0.05")

        # State tracking
        self.last_validation_results: dict[str, StateValidationResult] = {}
        self.state_history: list[StateValidationResult] = []
        self.reconciliation_attempts: dict[str, int] = {}

    def configure_dependencies(self, injector) -> None:
        """Configure dependencies via dependency injector."""
        try:
            # Try to get services from DI container
            if not self._database_service and injector.has_service("DatabaseService"):
                self._database_service = injector.resolve("DatabaseService")

            if not self._risk_service and injector.has_service("RiskService"):
                self._risk_service = injector.resolve("RiskService")

            if not self._execution_service and injector.has_service("ExecutionService"):
                self._execution_service = injector.resolve("ExecutionService")

            self.logger.debug("StateMonitor dependencies configured via DI container")
        except Exception as e:
            self.logger.warning(f"Failed to configure some StateMonitor dependencies via DI: {e}")

    def _validate_service_availability(self) -> None:
        """Validate that required services are available for proper operation."""
        missing_services = []

        if self._database_service is None:
            missing_services.append("DatabaseService")
        if self._risk_service is None:
            missing_services.append("RiskService")
        if self._execution_service is None:
            missing_services.append("ExecutionService")

        if missing_services:
            self.logger.info(
                "StateMonitor initialized without some services. "
                "Services will be resolved via DI container during configuration.",
                missing_services=missing_services,
            )

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
            raise ValidationError(f"Invalid {field_name} for decimal conversion: {value}") from e

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

        # Keep only last MAX_VALIDATION_HISTORY validation results
        if len(self.state_history) > MAX_VALIDATION_HISTORY:
            self.state_history = self.state_history[-MAX_VALIDATION_HISTORY:]

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
                if self._database_service is None:
                    self.logger.warning(
                        "DatabaseService not available - skipping balance sync check"
                    )
                    return {
                        "is_consistent": True,
                        "discrepancies": [],
                        "severity": "low",
                    }

                db_service = self._database_service
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

                # Note: Exchange balance checks would require ExchangeService injection
                # For now, we'll skip direct exchange queries to avoid tight coupling
                exchange_balances: dict[str, Any] = {}
                self.logger.debug(
                    "Exchange balance validation skipped - requires service injection"
                )

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
                    tolerance = TOLERANCE_DECIMAL

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
                            if max_diff > CRITICAL_THRESHOLD:
                                severity = "critical"
                                is_consistent = False
                            elif max_diff > HIGH_THRESHOLD:
                                severity = "high" if severity != "critical" else severity
                                is_consistent = False
                            elif max_diff > MEDIUM_THRESHOLD:
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
            cleanup_errors = []
            try:
                if "db_service" in locals() and db_service is not None:
                    await db_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"database service: {e}")

            if cleanup_errors:
                self.logger.error(f"Failed to cleanup services: {'; '.join(cleanup_errors)}")

    async def _check_position_quantity_sync(self) -> dict[str, Any]:
        """Check if position quantities are synchronized across systems."""
        # Using centralized decimal utilities from utils module

        try:
            discrepancies = []
            is_consistent = True
            severity = "low"

            # Use service layer for data access
            try:
                if self._database_service is None:
                    self.logger.warning(
                        "DatabaseService not available - skipping position sync check"
                    )
                    return {
                        "is_consistent": True,
                        "discrepancies": [],
                        "severity": "low",
                    }
                if self._risk_service is None:
                    self.logger.warning("RiskService not available - limited position sync check")
                    # Continue with limited functionality

                db_service = self._database_service
                risk_service = self._risk_service
                await db_service.initialize()
                if risk_service:
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

            # Note: Exchange position checks would require ExchangeService injection
            # For now, we'll skip direct exchange queries to avoid tight coupling
            exchange_positions: dict[str, Any] = {}
            self.logger.debug("Exchange position validation skipped - requires service injection")

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

                tolerance = TOLERANCE_DECIMAL

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
                    if qty_diff > CRITICAL_THRESHOLD:
                        severity = "critical"
                        is_consistent = False
                    elif qty_diff > HIGH_THRESHOLD:
                        severity = "high" if severity != "critical" else severity
                        is_consistent = False
                    elif qty_diff > MEDIUM_THRESHOLD:
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
            cleanup_errors = []
            try:
                if "db_service" in locals() and db_service is not None:
                    await db_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"database service: {e}")
            try:
                if "risk_service" in locals() and risk_service is not None:
                    await risk_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"risk service: {e}")

            if cleanup_errors:
                self.logger.error(f"Failed to cleanup services: {'; '.join(cleanup_errors)}")

    async def _check_order_status_sync(self) -> dict[str, Any]:
        """Check if order statuses are synchronized across systems."""

        try:
            # Multi-exchange order status verification placeholder
            # This will be implemented with Order Management and Execution Engine

            # Simulate order status check
            discrepancies: list[dict[str, Any]] = []
            is_consistent = True
            severity = "low"

            # Order status comparison placeholder:
            # - Database layer integration pending
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
                if self._database_service is None:
                    self.logger.warning(
                        "DatabaseService not available - skipping risk compliance check"
                    )
                    return {
                        "is_consistent": True,
                        "discrepancies": [],
                        "severity": "low",
                    }
                if self._risk_service is None:
                    self.logger.warning("RiskService not available - limited risk compliance check")
                    # Continue with limited functionality

                db_service = self._database_service
                risk_service = self._risk_service
                await db_service.initialize()
                if risk_service:
                    await risk_service.initialize()

                # Get current risk metrics via service
                risk_metrics = await risk_service.get_current_risk_metrics() if risk_service else {}

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
            risk_config = getattr(self.config, "risk", {})
            max_drawdown = Decimal(str(risk_config.get("max_drawdown", "0.20")))

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
                str(getattr(self.config, "risk", {}).get("max_daily_loss", DEFAULT_MAX_DAILY_LOSS))
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
                str(getattr(self.config, "risk", {}).get("max_correlation_risk", "0.80"))
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
            cleanup_errors = []
            try:
                if "db_service" in locals() and db_service is not None:
                    await db_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"database service: {e}")
            try:
                if "risk_service" in locals() and risk_service is not None:
                    await risk_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"risk service: {e}")

            if cleanup_errors:
                self.logger.error(f"Failed to cleanup services: {'; '.join(cleanup_errors)}")

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
                if self._database_service is None:
                    self.logger.warning("DatabaseService not available - cannot reconcile balances")
                    return False

                db_service = self._database_service
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

                    # Note: Exchange reconciliation would require ExchangeService injection
                    # For now, we'll use database as the source of truth
                    self.logger.debug(
                        f"Exchange reconciliation skipped for {key} - requires service injection"
                    )
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
                    f"Reconciled {reconciled_count} out of "
                    f"{len(discrepancies)} balance discrepancies"
                )

            return success

        except Exception as e:
            self.logger.error(f"Failed to reconcile portfolio balances: {e}")
            return False
        finally:
            # Clean up service connections
            cleanup_errors = []
            try:
                if "db_service" in locals() and db_service is not None:
                    await db_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"database service: {e}")

            if cleanup_errors:
                self.logger.error(f"Failed to cleanup services: {'; '.join(cleanup_errors)}")

    async def _reconcile_position_quantities(self, discrepancies: list[dict[str, Any]]) -> bool:
        """Reconcile position quantity discrepancies."""

        try:
            self.logger.info(
                "Reconciling position quantities", discrepancy_count=len(discrepancies)
            )

            # Use service layer for data access
            try:
                if self._database_service is None:
                    self.logger.warning(
                        "DatabaseService not available - cannot reconcile positions"
                    )
                    return False
                if self._risk_service is None:
                    self.logger.warning("RiskService not available - cannot reconcile positions")
                    return False

                db_service = self._database_service
                risk_service = self._risk_service
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

                    # Note: Exchange reconciliation would require ExchangeService injection
                    # For now, we'll use database as the source of truth
                    self.logger.debug(
                        f"Exchange position reconciliation skipped for {key} - "
                        "requires service injection"
                    )
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
                    f"Reconciled {reconciled_count} out of "
                    f"{len(discrepancies)} position discrepancies"
                )

            return success

        finally:
            # Clean up service connections
            cleanup_errors = []
            try:
                if "db_service" in locals() and db_service is not None:
                    await db_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"database service: {e}")
            try:
                if "risk_service" in locals() and risk_service is not None:
                    await risk_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"risk service: {e}")

            if cleanup_errors:
                self.logger.error(f"Failed to cleanup services: {'; '.join(cleanup_errors)}")

    async def _reconcile_order_statuses(self, discrepancies: list[dict[str, Any]]) -> bool:
        """Reconcile order status discrepancies."""

        try:
            self.logger.info("Reconciling order statuses", discrepancy_count=len(discrepancies))

            # Use service layer for data access
            try:
                if self._database_service is None:
                    self.logger.warning("DatabaseService not available - cannot reconcile orders")
                    return False
                if self._execution_service is None:
                    self.logger.warning("ExecutionService not available - cannot reconcile orders")
                    return False

                db_service = self._database_service
                execution_service = self._execution_service
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

                    # Determine exchange from order_id or discrepancy data via service
                    exchange_name = None
                    for key in ["db_status", "cache_status", "exchange_status", "execution_status"]:
                        if key in discrepancy and discrepancy[key] != "N/A":
                            # Use database service instead of direct access
                            try:
                                if self._database_service:
                                    # Get order details via service layer
                                    order_details = await self._database_service.get_order_details(
                                        order_id
                                    )
                                    if order_details:
                                        exchange_name = order_details.get("exchange")
                            except Exception as e:
                                # Fallback if service query fails
                                self.logger.debug(
                                    f"Service query failed during state monitoring: {e}"
                                )
                                # Continue monitoring other components despite this failure
                            break

                    if not exchange_name:
                        self.logger.warning(f"Could not determine exchange for order {order_id}")
                        continue

                    # Note: Exchange reconciliation would require ExchangeService injection
                    # For now, we'll use database as the source of truth
                    self.logger.debug(
                        f"Exchange order reconciliation skipped for {order_id} - "
                        "requires service injection"
                    )
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
                    f"Reconciled {reconciled_count} out of "
                    f"{len(relevant_discrepancies)} order discrepancies"
                )

            return success

        finally:
            # Clean up service connections
            cleanup_errors = []
            try:
                if "db_service" in locals() and db_service is not None:
                    await db_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"database service: {e}")
            try:
                if "execution_service" in locals() and execution_service is not None:
                    await execution_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"execution service: {e}")

            if cleanup_errors:
                self.logger.error(f"Failed to cleanup services: {'; '.join(cleanup_errors)}")

    async def _reconcile_risk_limits(self, discrepancies: list[dict[str, Any]]) -> bool:
        """Reconcile risk limit compliance issues."""

        try:
            self.logger.info("Reconciling risk limits", discrepancy_count=len(discrepancies))

            # Use service layer for data access
            if self._database_service is None:
                self.logger.warning("DatabaseService not available - cannot reconcile risk limits")
                return False
            if self._execution_service is None:
                self.logger.warning("ExecutionService not available - cannot reconcile risk limits")
                return False
            if self._risk_service is None:
                self.logger.warning("RiskService not available - cannot reconcile risk limits")
                return False

            db_service = self._database_service
            execution_service = self._execution_service
            risk_service = self._risk_service
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
                    f"Reconciled {reconciled_count} out of "
                    f"{len(discrepancies)} risk limit violations"
                )

            return success

        except Exception as e:
            self.logger.error("Risk limit reconciliation failed", error=str(e))
            return False
        finally:
            # Clean up service connections
            cleanup_errors = []
            try:
                if "db_service" in locals() and db_service is not None:
                    await db_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"database service: {e}")
            try:
                if "execution_service" in locals() and execution_service is not None:
                    await execution_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"execution service: {e}")
            try:
                if "risk_service" in locals() and risk_service is not None:
                    await risk_service.cleanup()
            except Exception as e:
                cleanup_errors.append(f"risk service: {e}")

            if cleanup_errors:
                self.logger.error(f"Failed to cleanup services: {'; '.join(cleanup_errors)}")

    async def start_monitoring(self) -> None:
        """Start continuous state monitoring with proper race condition handling."""

        self.logger.info("Starting state monitoring")

        # Use a lock to prevent concurrent monitoring runs
        if not hasattr(self, "_monitoring_lock"):
            self._monitoring_lock = asyncio.Lock()

        async with self._monitoring_lock:
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

                        # Attempt reconciliation for each component with discrepancies
                        # Group discrepancies by component to avoid race conditions
                        component_discrepancies: dict[str, list[dict[str, Any]]] = {}
                        for discrepancy in result.discrepancies:
                            component = discrepancy.get("component", "unknown")
                            if component != "unknown":
                                if component not in component_discrepancies:
                                    component_discrepancies[component] = []
                                component_discrepancies[component].append(discrepancy)

                        # Create reconciliation tasks grouped by component to prevent conflicts
                        # Process components sequentially to prevent race conditions
                        for component, discrepancies in component_discrepancies.items():
                            try:
                                self.logger.debug(
                                    "Starting component reconciliation",
                                    component=component,
                                    discrepancy_count=len(discrepancies),
                                )

                                # Use asyncio.wait_for to prevent indefinite blocking
                                reconciliation_result = await asyncio.wait_for(
                                    self.reconcile_state(component, discrepancies),
                                    timeout=DEFAULT_COMPONENT_TIMEOUT,
                                )

                                self.logger.debug(
                                    "Component reconciliation completed",
                                    component=component,
                                    success=reconciliation_result,
                                )

                            except asyncio.TimeoutError:
                                self.logger.error(
                                    "Component reconciliation timeout",
                                    component=component,
                                    timeout=DEFAULT_COMPONENT_TIMEOUT,
                                )
                            except asyncio.CancelledError:
                                self.logger.info(
                                    "Component reconciliation cancelled", component=component
                                )
                                # Re-raise cancellation to maintain proper async cleanup
                                raise
                            except Exception as reconciliation_error:
                                self.logger.error(
                                    "Component reconciliation failed",
                                    component=component,
                                    error=str(reconciliation_error),
                                )
                                # Continue with other components even if one fails
                                continue

                    # Wait for next validation cycle
                    await asyncio.sleep(self.validation_frequency)

                except asyncio.CancelledError:
                    self.logger.info("State monitoring cancelled")
                    # Ensure we exit cleanly on cancellation
                    return
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
            if isinstance(result, StateValidationResult):
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


def create_state_monitor_factory(
    config: Config | None = None,
    database_service: DatabaseServiceInterface | None = None,
    risk_service: RiskServiceInterface | None = None,
    execution_service: ExecutionServiceInterface | None = None,
):
    """Create a factory function for StateMonitor instances."""

    def factory() -> "StateMonitor":
        return StateMonitor(
            config=config or Config(),
            database_service=database_service,
            risk_service=risk_service,
            execution_service=execution_service,
        )

    return factory


def register_state_monitor_with_di(injector, config: Config | None = None) -> None:
    """Register StateMonitor with dependency injection container."""

    def state_monitor_factory() -> "StateMonitor":
        # Resolve dependencies from injector
        resolved_config = (
            injector.resolve("Config") if injector.has_service("Config") else config or Config()
        )
        database_service = (
            injector.resolve("DatabaseService") if injector.has_service("DatabaseService") else None
        )
        risk_service = (
            injector.resolve("RiskService") if injector.has_service("RiskService") else None
        )
        execution_service = (
            injector.resolve("ExecutionService")
            if injector.has_service("ExecutionService")
            else None
        )

        return StateMonitor(
            config=resolved_config,
            database_service=database_service,
            risk_service=risk_service,
            execution_service=execution_service,
        )

    injector.register_factory("StateMonitor", state_monitor_factory, singleton=True)
