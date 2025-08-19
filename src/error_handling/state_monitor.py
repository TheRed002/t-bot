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
from typing import Any

from src.core.config import Config

# MANDATORY: Import from P-001 core framework
from src.core.logging import get_logger

# MANDATORY: Import from P-007A utils framework
from src.utils.decorators import retry, time_execution

logger = get_logger(__name__)


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
        self.state_monitoring_config = config.error_handling
        self.validation_frequency = self.state_monitoring_config.state_validation_frequency
        self.consistency_checks = [
            "portfolio_balance_sync",
            "position_quantity_sync",
            "order_status_sync",
            "risk_limit_compliance",
        ]
        self.reconciliation_config = self.state_monitoring_config
        self.auto_reconcile = self.reconciliation_config.auto_reconciliation_enabled
        self.max_discrepancy = self.reconciliation_config.max_discrepancy_threshold
        self.force_sync_threshold = 0.05  # Default threshold

        # State tracking
        self.last_validation_results: dict[str, StateValidationResult] = {}
        self.state_history: list[StateValidationResult] = []
        self.reconciliation_attempts: dict[str, int] = {}

    @time_execution
    @retry(max_attempts=2)
    async def validate_state_consistency(self, component: str = "all") -> StateValidationResult:
        """Validate state consistency for specified component or all components."""

        logger.info("Starting state consistency validation", component=component)

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
                    logger.error("State consistency check failed", check=check, error=str(e))
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
                logger.error("Component state validation failed", component=component, error=str(e))
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

        logger.info(
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
            logger.warning("Unknown consistency check", check_name=check_name)
            return {"is_consistent": True, "discrepancies": [], "severity": "low"}

    async def _check_portfolio_balance_sync(self) -> dict[str, Any]:
        """Check if portfolio balances are synchronized across systems."""
        from decimal import Decimal

        from src.database.manager import DatabaseManager
        from src.database.redis_client import RedisClient
        from src.exchanges.factory import ExchangeFactory

        try:
            discrepancies = []
            is_consistent = True
            severity = "low"

            # Initialize data sources
            db_manager = DatabaseManager(self.config)
            redis_client = RedisClient(self.config)

            # Get balances from database
            db_balances = {}
            async with db_manager.get_session() as session:
                result = await session.execute(
                    "SELECT currency, available, locked, exchange FROM balances WHERE is_active = true"
                )
                for row in result:
                    key = f"{row.exchange}:{row.currency}"
                    db_balances[key] = {
                        "available": Decimal(str(row.available)),
                        "locked": Decimal(str(row.locked)),
                        "total": Decimal(str(row.available)) + Decimal(str(row.locked)),
                    }

            # Get balances from Redis cache
            cache_balances = {}
            cache_keys = await redis_client.keys("balance:*")
            for key in cache_keys:
                balance_data = await redis_client.get(key)
                if balance_data:
                    parts = key.split(":")
                    if len(parts) >= 3:
                        cache_key = f"{parts[1]}:{parts[2]}"
                        cache_balances[cache_key] = {
                            "available": Decimal(str(balance_data.get("available", 0))),
                            "locked": Decimal(str(balance_data.get("locked", 0))),
                            "total": Decimal(str(balance_data.get("total", 0))),
                        }

            # Get balances from exchanges (if connected)
            exchange_balances = {}
            try:
                for exchange_name in self.config.exchanges.keys():
                    exchange = ExchangeFactory.create(exchange_name, self.config)
                    if hasattr(exchange, "get_account_balance"):
                        balance = await exchange.get_account_balance()
                        for currency, amounts in balance.items():
                            key = f"{exchange_name}:{currency}"
                            exchange_balances[key] = {
                                "available": Decimal(str(amounts.get("free", 0))),
                                "locked": Decimal(str(amounts.get("used", 0))),
                                "total": Decimal(str(amounts.get("total", 0))),
                            }
            except Exception as e:
                logger.warning(f"Could not fetch exchange balances: {e}")

            # Compare balances across sources
            all_keys = (
                set(db_balances.keys()) | set(cache_balances.keys()) | set(exchange_balances.keys())
            )

            for key in all_keys:
                db_val = db_balances.get(key, {"total": Decimal("0")})
                cache_val = cache_balances.get(key, {"total": Decimal("0")})
                exchange_val = exchange_balances.get(key, {"total": Decimal("0")})

                # Check for discrepancies (with tolerance for rounding)
                tolerance = Decimal("0.00000001")

                if db_val["total"] > 0 or cache_val["total"] > 0 or exchange_val["total"] > 0:
                    max_diff = max(
                        abs(db_val["total"] - cache_val["total"]),
                        (
                            abs(db_val["total"] - exchange_val["total"])
                            if exchange_val["total"] > 0
                            else Decimal("0")
                        ),
                        (
                            abs(cache_val["total"] - exchange_val["total"])
                            if exchange_val["total"] > 0
                            else Decimal("0")
                        ),
                    )

                    if max_diff > tolerance:
                        discrepancy = {
                            "type": "balance_mismatch",
                            "key": key,
                            "db_balance": str(db_val["total"]),
                            "cache_balance": str(cache_val["total"]),
                            "exchange_balance": (
                                str(exchange_val["total"]) if exchange_val["total"] > 0 else "N/A"
                            ),
                            "max_difference": str(max_diff),
                        }
                        discrepancies.append(discrepancy)

                        # Determine severity based on difference
                        if max_diff > Decimal("1.0"):
                            severity = "critical"
                            is_consistent = False
                        elif max_diff > Decimal("0.1"):
                            severity = "high" if severity != "critical" else severity
                            is_consistent = False
                        elif max_diff > Decimal("0.01"):
                            severity = (
                                "medium" if severity not in ["critical", "high"] else severity
                            )

            logger.info(
                f"Portfolio balance sync check completed. Found {len(discrepancies)} discrepancies"
            )

            return {
                "is_consistent": is_consistent,
                "discrepancies": discrepancies,
                "severity": severity,
            }

        except Exception as e:
            logger.error("Portfolio balance sync check failed", error=str(e))
            return {
                "is_consistent": False,
                "discrepancies": [{"error": str(e), "type": "balance_sync_error"}],
                "severity": "high",
            }

    async def _check_position_quantity_sync(self) -> dict[str, Any]:
        """Check if position quantities are synchronized across systems."""
        from decimal import Decimal

        from src.database.manager import DatabaseManager
        from src.database.redis_client import RedisClient
        from src.exchanges.factory import ExchangeFactory
        from src.risk_management.base import RiskManager

        try:
            discrepancies = []
            is_consistent = True
            severity = "low"

            # Initialize data sources
            db_manager = DatabaseManager(self.config)
            redis_client = RedisClient(self.config)
            risk_manager = RiskManager(self.config)

            # Get positions from database
            db_positions = {}
            async with db_manager.get_session() as session:
                result = await session.execute(
                    "SELECT symbol, quantity, side, exchange, entry_price FROM positions WHERE status = 'open'"
                )
                for row in result:
                    key = f"{row.exchange}:{row.symbol}:{row.side}"
                    db_positions[key] = {
                        "quantity": Decimal(str(row.quantity)),
                        "entry_price": Decimal(str(row.entry_price)),
                    }

            # Get positions from Redis cache
            cache_positions = {}
            position_keys = await redis_client.keys("position:*")
            for key in position_keys:
                position_data = await redis_client.get(key)
                if position_data and position_data.get("status") == "open":
                    parts = key.split(":")
                    if len(parts) >= 4:
                        cache_key = f"{parts[1]}:{parts[2]}:{parts[3]}"
                        cache_positions[cache_key] = {
                            "quantity": Decimal(str(position_data.get("quantity", 0))),
                            "entry_price": Decimal(str(position_data.get("entry_price", 0))),
                        }

            # Get positions from exchanges (if connected)
            exchange_positions = {}
            try:
                for exchange_name in self.config.exchanges.keys():
                    exchange = ExchangeFactory.create(exchange_name, self.config)
                    if hasattr(exchange, "get_positions"):
                        positions = await exchange.get_positions()
                        for pos in positions:
                            key = f"{exchange_name}:{pos.symbol}:{pos.side}"
                            exchange_positions[key] = {
                                "quantity": Decimal(str(pos.quantity)),
                                "entry_price": Decimal(str(pos.entry_price)),
                            }
            except Exception as e:
                logger.warning(f"Could not fetch exchange positions: {e}")

            # Get positions from risk management system
            risk_positions = {}
            try:
                risk_state = await risk_manager.get_current_state()
                for pos_key, pos_data in risk_state.get("positions", {}).items():
                    risk_positions[pos_key] = {
                        "quantity": Decimal(str(pos_data.get("quantity", 0))),
                        "entry_price": Decimal(str(pos_data.get("entry_price", 0))),
                    }
            except Exception as e:
                logger.warning(f"Could not fetch risk positions: {e}")

            # Compare positions across sources
            all_keys = (
                set(db_positions.keys())
                | set(cache_positions.keys())
                | set(exchange_positions.keys())
                | set(risk_positions.keys())
            )

            for key in all_keys:
                db_pos = db_positions.get(key, {"quantity": Decimal("0")})
                cache_pos = cache_positions.get(key, {"quantity": Decimal("0")})
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

            logger.info(
                f"Position quantity sync check completed. Found {len(discrepancies)} discrepancies"
            )

            return {
                "is_consistent": is_consistent,
                "discrepancies": discrepancies,
                "severity": severity,
            }

        except Exception as e:
            logger.error("Position quantity sync check failed", error=str(e))
            return {
                "is_consistent": False,
                "discrepancies": [{"error": str(e), "type": "position_sync_error"}],
                "severity": "high",
            }

    async def _check_order_status_sync(self) -> dict[str, Any]:
        """Check if order statuses are synchronized across systems."""

        try:
            # TODO: Implement actual order status synchronization check
            # This will be implemented in P-020 (Order Management and Execution
            # Engine)

            # Simulate order status check
            discrepancies = []
            is_consistent = True
            severity = "low"

            # TODO: Compare order statuses from:
            # - Database (P-002)
            # - Exchange APIs (P-003+)
            # - Redis cache (P-002)
            # - Execution engine (P-020)

            logger.info("Order status sync check completed")

            return {
                "is_consistent": is_consistent,
                "discrepancies": discrepancies,
                "severity": severity,
            }

        except Exception as e:
            logger.error("Order status sync check failed", error=str(e))
            return {
                "is_consistent": False,
                "discrepancies": [{"error": str(e), "type": "order_sync_error"}],
                "severity": "high",
            }

    async def _check_risk_limit_compliance(self) -> dict[str, Any]:
        """Check if risk limits are being complied with."""
        from decimal import Decimal

        from src.database.manager import DatabaseManager
        from src.risk_management.base import RiskManager
        from src.risk_management.portfolio_limits import PortfolioLimits

        try:
            discrepancies = []
            is_consistent = True
            severity = "low"

            # Initialize risk management components
            risk_manager = RiskManager(self.config)
            portfolio_limits = PortfolioLimits(self.config)
            db_manager = DatabaseManager(self.config)

            # Get current risk metrics
            risk_state = await risk_manager.get_current_state()
            risk_metrics = risk_state.get("metrics", {})

            # Check position size limits
            async with db_manager.get_session() as session:
                result = await session.execute(
                    """SELECT symbol, quantity, side, exchange, entry_price, current_price
                       FROM positions WHERE status = 'open'"""
                )

                for row in result:
                    position_value = Decimal(str(row.quantity)) * Decimal(
                        str(row.current_price or row.entry_price)
                    )
                    max_position_size = Decimal(str(self.config.risk_management.max_position_size))

                    if position_value > max_position_size:
                        discrepancy = {
                            "type": "position_size_limit_exceeded",
                            "symbol": row.symbol,
                            "position_value": str(position_value),
                            "max_allowed": str(max_position_size),
                            "excess": str(position_value - max_position_size),
                        }
                        discrepancies.append(discrepancy)
                        severity = "critical"
                        is_consistent = False

            # Check portfolio exposure limits
            total_exposure = Decimal(str(risk_metrics.get("total_exposure", 0)))
            max_exposure = Decimal(str(self.config.risk_management.max_portfolio_exposure))

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
            max_leverage = Decimal(str(self.config.risk_management.max_leverage))

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

            # Check stop loss compliance
            async with db_manager.get_session() as session:
                result = await session.execute(
                    """SELECT COUNT(*) as count FROM positions 
                       WHERE status = 'open' AND stop_loss_price IS NULL"""
                )
                positions_without_stop = result.scalar()

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
            max_drawdown = Decimal(str(self.config.risk_management.max_drawdown))

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
            max_daily_loss = Decimal(str(self.config.risk_management.max_daily_loss))

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
            max_correlation = Decimal(str(self.config.risk_management.max_correlation_risk))

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

            logger.info(
                f"Risk limit compliance check completed. Found {len(discrepancies)} violations"
            )

            return {
                "is_consistent": is_consistent,
                "discrepancies": discrepancies,
                "severity": severity,
            }

        except Exception as e:
            logger.error("Risk limit compliance check failed", error=str(e))
            return {
                "is_consistent": False,
                "discrepancies": [{"error": str(e), "type": "risk_compliance_error"}],
                "severity": "critical",
            }

    async def reconcile_state(self, component: str, discrepancies: list[dict[str, Any]]) -> bool:
        """Attempt to reconcile state discrepancies."""

        logger.info(
            "Attempting state reconciliation",
            component=component,
            discrepancy_count=len(discrepancies),
        )

        if not self.auto_reconcile:
            logger.info("Auto-reconciliation disabled", component=component)
            return False

        reconciliation_attempts = self.reconciliation_attempts.get(component, 0)
        max_attempts = 3

        if reconciliation_attempts >= max_attempts:
            logger.warning(
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
                logger.warning("Unknown reconciliation component", component=component)
                return False

            if success:
                logger.info("State reconciliation successful", component=component)
                # Reset reconciliation attempts on success
                self.reconciliation_attempts[component] = 0
            else:
                logger.warning("State reconciliation failed", component=component)

            return success

        except Exception as e:
            logger.error("State reconciliation error", component=component, error=str(e))
            return False

    async def _reconcile_portfolio_balances(self, discrepancies: list[dict[str, Any]]) -> bool:
        """Reconcile portfolio balance discrepancies."""
        from decimal import Decimal

        from src.database.influxdb_client import InfluxDBClientWrapper
        from src.database.manager import DatabaseManager
        from src.database.redis_client import RedisClient
        from src.exchanges.factory import ExchangeFactory

        try:
            logger.info("Reconciling portfolio balances", discrepancy_count=len(discrepancies))

            db_manager = DatabaseManager(self.config)
            redis_client = RedisClient(self.config)
            influx_client = InfluxDBClientWrapper(self.config)

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

                            # Update database
                            async with db_manager.get_session() as session:
                                await session.execute(
                                    """UPDATE balances 
                                       SET available = :available, locked = :locked, 
                                           updated_at = NOW() 
                                       WHERE exchange = :exchange AND currency = :currency""",
                                    {
                                        "available": float(true_balance["available"]),
                                        "locked": float(true_balance["locked"]),
                                        "exchange": exchange_name,
                                        "currency": currency,
                                    },
                                )
                                await session.commit()

                            # Update Redis cache
                            cache_key = f"balance:{exchange_name}:{currency}"
                            await redis_client.set(
                                cache_key,
                                {
                                    "available": str(true_balance["available"]),
                                    "locked": str(true_balance["locked"]),
                                    "total": str(true_balance["total"]),
                                    "updated_at": datetime.now(timezone.utc).isoformat(),
                                },
                                expiry=300,  # 5 minutes cache
                            )

                            # Update InfluxDB metrics
                            await influx_client.write_point(
                                measurement="balance_reconciliation",
                                tags={"exchange": exchange_name, "currency": currency},
                                fields={
                                    "available": float(true_balance["available"]),
                                    "locked": float(true_balance["locked"]),
                                    "total": float(true_balance["total"]),
                                },
                            )

                            reconciled_count += 1
                            logger.info(
                                f"Reconciled balance for {key}",
                                true_balance=str(true_balance["total"]),
                            )

                except Exception as e:
                    logger.error(f"Failed to reconcile balance for {key}: {e}")
                    continue

            success = reconciled_count == len(
                [d for d in discrepancies if d.get("type") == "balance_mismatch"]
            )

            if success:
                logger.info(f"Successfully reconciled all {reconciled_count} balance discrepancies")
            else:
                logger.warning(
                    f"Reconciled {reconciled_count} out of {len(discrepancies)} balance discrepancies"
                )

            return success

        except Exception as e:
            logger.error("Portfolio balance reconciliation failed", error=str(e))
            return False

    async def _reconcile_position_quantities(self, discrepancies: list[dict[str, Any]]) -> bool:
        """Reconcile position quantity discrepancies."""

        from src.database.manager import DatabaseManager
        from src.database.redis_client import RedisClient
        from src.exchanges.factory import ExchangeFactory
        from src.risk_management.base import RiskManager

        try:
            logger.info("Reconciling position quantities", discrepancy_count=len(discrepancies))

            db_manager = DatabaseManager(self.config)
            redis_client = RedisClient(self.config)
            risk_manager = RiskManager(self.config)

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
                            # Update database
                            async with db_manager.get_session() as session:
                                await session.execute(
                                    """UPDATE positions 
                                       SET quantity = :quantity, 
                                           current_price = :current_price,
                                           updated_at = NOW() 
                                       WHERE exchange = :exchange 
                                         AND symbol = :symbol 
                                         AND side = :side 
                                         AND status = 'open'""",
                                    {
                                        "quantity": float(true_position.quantity),
                                        "current_price": float(true_position.current_price),
                                        "exchange": exchange_name,
                                        "symbol": symbol,
                                        "side": side,
                                    },
                                )
                                await session.commit()

                            # Update Redis cache
                            cache_key = f"position:{exchange_name}:{symbol}:{side}"
                            await redis_client.set(
                                cache_key,
                                {
                                    "quantity": str(true_position.quantity),
                                    "entry_price": str(true_position.entry_price),
                                    "current_price": str(true_position.current_price),
                                    "status": "open",
                                    "updated_at": datetime.now(timezone.utc).isoformat(),
                                },
                                expiry=300,
                            )

                            # Update risk management system
                            await risk_manager.update_position(
                                symbol=symbol,
                                quantity=true_position.quantity,
                                side=side,
                                exchange=exchange_name,
                            )

                            reconciled_count += 1
                            logger.info(
                                f"Reconciled position for {key}",
                                true_quantity=str(true_position.quantity),
                            )
                        else:
                            # Position closed on exchange, update local state
                            async with db_manager.get_session() as session:
                                await session.execute(
                                    """UPDATE positions 
                                       SET status = 'closed', 
                                           closed_at = NOW() 
                                       WHERE exchange = :exchange 
                                         AND symbol = :symbol 
                                         AND side = :side 
                                         AND status = 'open'""",
                                    {"exchange": exchange_name, "symbol": symbol, "side": side},
                                )
                                await session.commit()

                            # Remove from cache
                            cache_key = f"position:{exchange_name}:{symbol}:{side}"
                            await redis_client.delete(cache_key)

                            reconciled_count += 1
                            logger.info(f"Marked position {key} as closed")

                except Exception as e:
                    logger.error(f"Failed to reconcile position for {key}: {e}")
                    continue

            success = reconciled_count == len(
                [d for d in discrepancies if d.get("type") == "position_quantity_mismatch"]
            )

            if success:
                logger.info(
                    f"Successfully reconciled all {reconciled_count} position discrepancies"
                )
            else:
                logger.warning(
                    f"Reconciled {reconciled_count} out of {len(discrepancies)} position discrepancies"
                )

            return success

        except Exception as e:
            logger.error("Position quantity reconciliation failed", error=str(e))
            return False

    async def _reconcile_order_statuses(self, discrepancies: list[dict[str, Any]]) -> bool:
        """Reconcile order status discrepancies."""
        from src.database.manager import DatabaseManager
        from src.database.redis_client import RedisClient
        from src.exchanges.factory import ExchangeFactory
        from src.execution.order_manager import OrderManager

        try:
            logger.info("Reconciling order statuses", discrepancy_count=len(discrepancies))

            db_manager = DatabaseManager(self.config)
            redis_client = RedisClient(self.config)
            order_manager = OrderManager(self.config)

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
                    logger.warning(f"Could not determine exchange for order {order_id}")
                    continue

                # Get truth from exchange
                try:
                    exchange = ExchangeFactory.create(exchange_name, self.config)
                    if hasattr(exchange, "get_order"):
                        true_order = await exchange.get_order(order_id)

                        if true_order:
                            # Update database
                            async with db_manager.get_session() as session:
                                await session.execute(
                                    """UPDATE orders 
                                       SET status = :status, 
                                           filled_quantity = :filled,
                                           remaining_quantity = :remaining,
                                           updated_at = NOW() 
                                       WHERE order_id = :order_id""",
                                    {
                                        "status": true_order.status,
                                        "filled": float(true_order.filled_quantity),
                                        "remaining": float(true_order.remaining_quantity),
                                        "order_id": order_id,
                                    },
                                )
                                await session.commit()

                            # Update Redis cache
                            cache_key = f"order:{order_id}"
                            await redis_client.set(
                                cache_key,
                                {
                                    "order_id": order_id,
                                    "status": true_order.status,
                                    "filled_quantity": str(true_order.filled_quantity),
                                    "remaining_quantity": str(true_order.remaining_quantity),
                                    "exchange": exchange_name,
                                    "symbol": true_order.symbol,
                                    "updated_at": datetime.now(timezone.utc).isoformat(),
                                },
                                expiry=300,
                            )

                            # Update execution engine
                            await order_manager.update_order_status(
                                order_id=order_id,
                                status=true_order.status,
                                filled_quantity=true_order.filled_quantity,
                                remaining_quantity=true_order.remaining_quantity,
                            )

                            reconciled_count += 1
                            logger.info(
                                f"Reconciled order {order_id}",
                                true_status=true_order.status,
                                filled=str(true_order.filled_quantity),
                            )
                        else:
                            # Order not found on exchange, mark as cancelled/expired
                            async with db_manager.get_session() as session:
                                await session.execute(
                                    """UPDATE orders 
                                       SET status = 'cancelled', 
                                           updated_at = NOW() 
                                       WHERE order_id = :order_id 
                                         AND status IN ('new', 'open', 'partially_filled')""",
                                    {"order_id": order_id},
                                )
                                await session.commit()

                            # Update cache
                            cache_key = f"order:{order_id}"
                            cached = await redis_client.get(cache_key)
                            if cached:
                                cached["status"] = "cancelled"
                                await redis_client.set(cache_key, cached, expiry=300)

                            reconciled_count += 1
                            logger.info(f"Marked order {order_id} as cancelled")

                except Exception as e:
                    logger.error(f"Failed to reconcile order {order_id}: {e}")
                    continue

            relevant_discrepancies = [
                d
                for d in discrepancies
                if d.get("type") in ["order_status_mismatch", "order_filled_quantity_mismatch"]
            ]
            success = reconciled_count == len(relevant_discrepancies)

            if success:
                logger.info(f"Successfully reconciled all {reconciled_count} order discrepancies")
            else:
                logger.warning(
                    f"Reconciled {reconciled_count} out of {len(relevant_discrepancies)} order discrepancies"
                )

            return success

        except Exception as e:
            logger.error("Order status reconciliation failed", error=str(e))
            return False

    async def _reconcile_risk_limits(self, discrepancies: list[dict[str, Any]]) -> bool:
        """Reconcile risk limit compliance issues."""
        from decimal import Decimal

        from src.database.manager import DatabaseManager
        from src.execution.order_manager import OrderManager
        from src.risk_management.base import RiskManager
        from src.risk_management.emergency_controls import EmergencyControls

        try:
            logger.info("Reconciling risk limits", discrepancy_count=len(discrepancies))

            risk_manager = RiskManager(self.config)
            emergency_controls = EmergencyControls(self.config)
            order_manager = OrderManager(self.config)
            db_manager = DatabaseManager(self.config)

            reconciled_count = 0
            critical_actions_taken = []

            for discrepancy in discrepancies:
                discrepancy_type = discrepancy.get("type")

                try:
                    if discrepancy_type == "position_size_limit_exceeded":
                        # Reduce position size
                        symbol = discrepancy.get("symbol")
                        excess = Decimal(discrepancy.get("excess", "0"))

                        if symbol and excess > 0:
                            # Cancel pending orders for this symbol
                            await order_manager.cancel_orders_by_symbol(symbol)

                            # Reduce position by excess amount
                            await risk_manager.reduce_position(symbol=symbol, amount=excess)

                            critical_actions_taken.append(
                                f"Reduced position for {symbol} by {excess}"
                            )
                            reconciled_count += 1

                    elif discrepancy_type == "portfolio_exposure_limit_exceeded":
                        # Reduce overall exposure
                        excess = Decimal(discrepancy.get("excess", "0"))

                        # Cancel all pending orders
                        await order_manager.cancel_all_orders()

                        # Reduce positions proportionally
                        await risk_manager.reduce_portfolio_exposure(excess)

                        critical_actions_taken.append(f"Reduced portfolio exposure by {excess}")
                        reconciled_count += 1

                    elif discrepancy_type == "leverage_limit_exceeded":
                        # Reduce leverage
                        current_leverage = Decimal(discrepancy.get("current_leverage", "1"))
                        max_leverage = Decimal(discrepancy.get("max_allowed", "1"))

                        # Calculate reduction needed
                        reduction_factor = max_leverage / current_leverage

                        # Reduce all positions proportionally
                        await risk_manager.adjust_leverage(reduction_factor)

                        critical_actions_taken.append(
                            f"Reduced leverage from {current_leverage} to {max_leverage}"
                        )
                        reconciled_count += 1

                    elif discrepancy_type == "stop_loss_missing":
                        # Add stop losses to positions
                        positions_without_stop = discrepancy.get("positions_without_stop_loss", 0)

                        async with db_manager.get_session() as session:
                            # Get positions without stop loss
                            result = await session.execute(
                                """SELECT position_id, symbol, entry_price, side 
                                   FROM positions 
                                   WHERE status = 'open' AND stop_loss_price IS NULL"""
                            )

                            for row in result:
                                # Calculate stop loss price (2% from entry)
                                stop_loss_pct = Decimal("0.02")
                                entry_price = Decimal(str(row.entry_price))

                                if row.side == "long":
                                    stop_loss_price = entry_price * (Decimal("1") - stop_loss_pct)
                                else:
                                    stop_loss_price = entry_price * (Decimal("1") + stop_loss_pct)

                                # Update position with stop loss
                                await session.execute(
                                    """UPDATE positions 
                                       SET stop_loss_price = :stop_loss 
                                       WHERE position_id = :position_id""",
                                    {
                                        "stop_loss": float(stop_loss_price),
                                        "position_id": row.position_id,
                                    },
                                )

                            await session.commit()

                        critical_actions_taken.append(
                            f"Added stop losses to {positions_without_stop} positions"
                        )
                        reconciled_count += 1

                    elif discrepancy_type == "max_drawdown_exceeded":
                        # Activate emergency controls
                        await emergency_controls.activate_emergency_shutdown(
                            "Max drawdown exceeded"
                        )

                        critical_actions_taken.append(
                            "Activated emergency shutdown due to max drawdown"
                        )
                        reconciled_count += 1

                    elif discrepancy_type == "daily_loss_limit_exceeded":
                        # Stop trading for the day
                        await risk_manager.halt_trading("Daily loss limit exceeded")

                        # Cancel all pending orders
                        await order_manager.cancel_all_orders()

                        critical_actions_taken.append("Halted trading due to daily loss limit")
                        reconciled_count += 1

                    elif discrepancy_type == "correlation_risk_exceeded":
                        # Reduce correlated positions
                        excess = Decimal(discrepancy.get("excess", "0"))

                        await risk_manager.reduce_correlation_risk(excess)

                        critical_actions_taken.append(f"Reduced correlation risk by {excess}")
                        reconciled_count += 1

                except Exception as e:
                    logger.error(f"Failed to reconcile {discrepancy_type}: {e}")
                    continue

            # Log all critical actions
            if critical_actions_taken:
                logger.critical(
                    "Risk limit reconciliation actions taken", actions=critical_actions_taken
                )

                # Send alerts
                for action in critical_actions_taken:
                    await risk_manager.send_alert(
                        level="critical", message=f"Risk reconciliation: {action}"
                    )

            success = reconciled_count == len(discrepancies)

            if success:
                logger.info(f"Successfully reconciled all {reconciled_count} risk limit violations")
            else:
                logger.warning(
                    f"Reconciled {reconciled_count} out of {len(discrepancies)} risk limit violations"
                )

            return success

        except Exception as e:
            logger.error("Risk limit reconciliation failed", error=str(e))
            return False

    async def start_monitoring(self):
        """Start continuous state monitoring."""

        logger.info("Starting state monitoring")

        while True:
            try:
                # Validate all components
                result = await self.validate_state_consistency("all")

                if not result.is_consistent:
                    logger.warning(
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
                logger.error("State monitoring error", error=str(e))
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
