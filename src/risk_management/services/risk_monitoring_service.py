"""
Risk Monitoring Service Implementation.

This service handles real-time risk monitoring through dependency injection,
following proper service layer patterns.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from src.core.base.service import BaseService
from src.core.types import RiskLevel, RiskMetrics
from src.utils.decimal_utils import to_decimal

if TYPE_CHECKING:
    from src.database.service import DatabaseService
    from src.state import StateService


class RiskAlert:
    """Risk alert model."""

    def __init__(
        self,
        alert_id: str,
        alert_type: str,
        severity: str,
        message: str,
        details: dict[str, Any],
        timestamp: datetime | None = None,
    ):
        self.alert_id = alert_id
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.details = details
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.acknowledged = False


class RiskMonitoringService(BaseService):
    """Service for real-time risk monitoring and alerting."""

    def __init__(
        self,
        database_service: "DatabaseService",
        state_service: "StateService",
        config=None,
        correlation_id: str | None = None,
    ):
        """
        Initialize risk monitoring service.

        Args:
            database_service: Database service for data access
            state_service: State service for state management
            config: Application configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="RiskMonitoringService",
            config=config.__dict__ if config else {},
            correlation_id=correlation_id,
        )

        self.database_service = database_service
        self.state_service = state_service
        self.config = config

        # Monitoring state
        self._is_monitoring = False
        self._monitoring_task: asyncio.Task | None = None
        self._alerts: list[RiskAlert] = []
        self._emergency_stop_active = False

        # Default thresholds
        self._thresholds = {
            "var_warning": to_decimal("0.05"),  # 5% VaR warning
            "var_critical": to_decimal("0.10"),  # 10% VaR critical
            "drawdown_warning": to_decimal("0.10"),  # 10% drawdown warning
            "drawdown_critical": to_decimal("0.20"),  # 20% drawdown critical
            "sharpe_minimum": to_decimal("0.5"),  # Minimum Sharpe ratio
        }

    async def start_monitoring(self, interval: int = 60) -> None:
        """
        Start continuous risk monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self._is_monitoring:
            self._logger.warning("Risk monitoring already active")
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))

        self._logger.info(f"Started risk monitoring with {interval}s interval")

    async def stop_monitoring(self) -> None:
        """Stop continuous risk monitoring with proper timeout handling."""
        self._is_monitoring = False
        task = None

        try:
            if self._monitoring_task:
                task = self._monitoring_task
                task.cancel()
                try:
                    # Wait for task to be cancelled with timeout
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    self._logger.info("Monitoring task cancelled")
                except Exception as e:
                    self._logger.warning(f"Error cancelling monitoring task: {e}")
                finally:
                    self._monitoring_task = None
                    task = None

        finally:
            # Ensure task reference is cleared
            task = None

        self._logger.info("Stopped risk monitoring")

    async def check_emergency_conditions(self, metrics: RiskMetrics) -> bool:
        """
        Check if emergency stop conditions are met.

        Args:
            metrics: Current risk metrics

        Returns:
            True if emergency stop should be triggered
        """
        emergency_conditions = []

        # Check extreme drawdown
        if metrics.current_drawdown > self._thresholds["drawdown_critical"]:
            emergency_conditions.append(f"Extreme drawdown: {metrics.current_drawdown:.2%}")

        # Check extreme VaR
        if metrics.var_1d and metrics.portfolio_value > to_decimal("0"):
            var_pct = metrics.var_1d / metrics.portfolio_value
            if var_pct > self._thresholds["var_critical"]:
                emergency_conditions.append(f"Extreme VaR: {var_pct:.2%}")

        # Check risk level
        if metrics.risk_level == RiskLevel.CRITICAL:
            emergency_conditions.append("Risk level is CRITICAL")

        if emergency_conditions:
            await self._trigger_emergency_stop(emergency_conditions)
            return True

        return False

    async def monitor_metrics(self, metrics: RiskMetrics) -> None:
        """
        Monitor risk metrics and generate alerts.

        Args:
            metrics: Risk metrics to monitor
        """
        try:
            # Check VaR thresholds
            await self._check_var_thresholds(metrics)

            # Check drawdown thresholds
            await self._check_drawdown_thresholds(metrics)

            # Check Sharpe ratio
            await self._check_sharpe_ratio(metrics)

            # Check risk level changes
            await self._check_risk_level_changes(metrics)

            # Check portfolio concentration
            await self._check_portfolio_concentration(metrics)

        except Exception as e:
            self._logger.error(f"Error monitoring risk metrics: {e}")

    async def get_active_alerts(self, limit: int | None = None) -> list[RiskAlert]:
        """
        Get active risk alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of active alerts
        """
        # Filter for recent, unacknowledged alerts
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        active_alerts = [alert for alert in self._alerts if not alert.acknowledged and alert.timestamp > cutoff_time]

        active_alerts.sort(key=lambda x: x.timestamp, reverse=True)

        if limit:
            active_alerts = active_alerts[:limit]

        return active_alerts

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge a risk alert.

        Args:
            alert_id: ID of alert to acknowledge

        Returns:
            True if alert was found and acknowledged
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self._logger.info(f"Risk alert acknowledged: {alert_id}")
                return True

        return False

    async def set_threshold(self, threshold_name: str, value) -> None:
        """
        Set a risk monitoring threshold.

        Args:
            threshold_name: Name of threshold to set
            value: Threshold value
        """
        if threshold_name in self._thresholds:
            self._thresholds[threshold_name] = to_decimal(value)
            self._logger.info(f"Updated threshold {threshold_name} = {value}")
        else:
            self._logger.warning(f"Unknown threshold: {threshold_name}")

    async def _monitoring_loop(self, interval: int) -> None:
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                # Get latest metrics from state
                latest_metrics = await self._get_latest_metrics()
                if latest_metrics:
                    await self.monitor_metrics(latest_metrics)

                # Check emergency conditions
                if latest_metrics:
                    await self.check_emergency_conditions(latest_metrics)

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(min(interval, 30))  # Don't spam on errors

    async def _check_var_thresholds(self, metrics: RiskMetrics) -> None:
        """Check VaR thresholds and generate alerts."""
        if not metrics.var_1d or not metrics.portfolio_value:
            return

        var_pct = metrics.var_1d / metrics.portfolio_value

        if var_pct > self._thresholds["var_critical"]:
            await self._create_alert(
                "var_critical",
                "CRITICAL",
                f"VaR exceeds critical threshold: {var_pct:.2%}",
                {"var_pct": float(var_pct), "threshold": float(self._thresholds["var_critical"])},
            )
        elif var_pct > self._thresholds["var_warning"]:
            await self._create_alert(
                "var_warning",
                "HIGH",
                f"VaR exceeds warning threshold: {var_pct:.2%}",
                {"var_pct": float(var_pct), "threshold": float(self._thresholds["var_warning"])},
            )

    async def _check_drawdown_thresholds(self, metrics: RiskMetrics) -> None:
        """Check drawdown thresholds and generate alerts."""
        if not metrics.current_drawdown:
            return

        drawdown = metrics.current_drawdown

        if drawdown > self._thresholds["drawdown_critical"]:
            await self._create_alert(
                "drawdown_critical",
                "CRITICAL",
                f"Drawdown exceeds critical threshold: {drawdown:.2%}",
                {
                    "drawdown": float(drawdown),
                    "threshold": float(self._thresholds["drawdown_critical"]),
                },
            )
        elif drawdown > self._thresholds["drawdown_warning"]:
            await self._create_alert(
                "drawdown_warning",
                "HIGH",
                f"Drawdown exceeds warning threshold: {drawdown:.2%}",
                {
                    "drawdown": float(drawdown),
                    "threshold": float(self._thresholds["drawdown_warning"]),
                },
            )

    async def _check_sharpe_ratio(self, metrics: RiskMetrics) -> None:
        """Check Sharpe ratio and generate alerts."""
        if metrics.sharpe_ratio is None:
            return

        sharpe = to_decimal(metrics.sharpe_ratio)

        if sharpe < self._thresholds["sharpe_minimum"]:
            await self._create_alert(
                "sharpe_low",
                "MEDIUM",
                f"Sharpe ratio below minimum: {sharpe:.3f}",
                {
                    "sharpe_ratio": float(sharpe),
                    "minimum": float(self._thresholds["sharpe_minimum"]),
                },
            )

    async def _check_risk_level_changes(self, metrics: RiskMetrics) -> None:
        """Check for risk level changes."""
        try:
            # Get previous risk level from state
            previous_level = await self._get_previous_risk_level()

            if previous_level and previous_level != metrics.risk_level:
                await self._create_alert(
                    "risk_level_change",
                    "MEDIUM",
                    f"Risk level changed from {previous_level.value} to {metrics.risk_level.value}",
                    {"previous_level": previous_level.value, "new_level": metrics.risk_level.value},
                )

            # Store current risk level
            await self.state_service.set_state(
                "risk",
                "current_level",
                {
                    "risk_level": metrics.risk_level.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except Exception as e:
            self._logger.error(f"Error checking risk level changes: {e}")

    async def _check_portfolio_concentration(self, metrics: RiskMetrics) -> None:
        """Check portfolio concentration risk."""
        if metrics.correlation_risk and metrics.correlation_risk > to_decimal("0.7"):
            await self._create_alert(
                "concentration_high",
                "MEDIUM",
                f"High portfolio concentration: {metrics.correlation_risk:.2%}",
                {"concentration": float(metrics.correlation_risk)},
            )

    async def _create_alert(self, alert_type: str, severity: str, message: str, details: dict[str, Any]) -> None:
        """Create and store a risk alert."""
        alert_id = f"{alert_type}_{datetime.now(timezone.utc).timestamp()}"

        alert = RiskAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details,
        )

        self._alerts.append(alert)

        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-500:]

        self._logger.warning(f"Risk alert created: {message}", extra=details)

    async def _trigger_emergency_stop(self, conditions: list[str]) -> None:
        """Trigger emergency stop."""
        if self._emergency_stop_active:
            return

        self._emergency_stop_active = True
        reason = f"Emergency conditions: {'; '.join(conditions)}"

        # Create critical alert
        await self._create_alert(
            "emergency_stop",
            "CRITICAL",
            f"Emergency stop triggered: {reason}",
            {"conditions": conditions, "reason": reason},
        )

        # Store emergency state
        await self.state_service.set_state(
            "risk",
            "emergency_stop",
            {
                "active": True,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "conditions": conditions,
            },
        )

        self._logger.critical("EMERGENCY STOP TRIGGERED", reason=reason, conditions=conditions)

    async def _get_latest_metrics(self) -> RiskMetrics | None:
        """Get latest risk metrics from state."""
        try:
            metrics_data = await self.state_service.get_state("risk", "latest_metrics")
            if not metrics_data:
                return None

            # Convert back to RiskMetrics object
            # This is a simplified version - in practice you'd have proper serialization
            return None  # Metrics deserialization not implemented - returns None for safety

        except Exception as e:
            self._logger.error(f"Error getting latest metrics: {e}")
            return None

    async def _get_previous_risk_level(self) -> RiskLevel | None:
        """Get previous risk level from state."""
        try:
            level_data = await self.state_service.get_state("risk", "current_level")
            if level_data and "risk_level" in level_data:
                return RiskLevel(level_data["risk_level"])
            return None
        except Exception as e:
            self._logger.error(f"Error getting previous risk level: {e}")
            return None
