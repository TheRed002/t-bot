"""
Risk Monitoring Service Implementation.

This service handles real-time risk monitoring through dependency injection,
following proper service layer patterns.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from src.core.base.service import BaseService
from src.core.types import RiskLevel, RiskMetrics
from src.utils.decimal_utils import format_decimal, to_decimal
from src.utils.messaging_patterns import (
    BoundaryValidator,
    MessagePattern,
    MessageType,
    MessagingCoordinator,
    ProcessingParadigmAligner,
)
from src.utils.risk_monitoring import (
    get_unified_risk_monitor,
)

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
        messaging_coordinator: MessagingCoordinator | None = None,
        config=None,
        correlation_id: str | None = None,
    ):
        """
        Initialize risk monitoring service.

        Args:
            database_service: Database service for data access
            state_service: State service for state management
            messaging_coordinator: Messaging coordinator for consistent data flow
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
        self._messaging_coordinator = messaging_coordinator or MessagingCoordinator(
            "RiskMonitoringService"
        )

        # Use centralized monitoring
        self._unified_monitor = get_unified_risk_monitor()

        # Monitoring state
        self._is_monitoring = False
        self._monitoring_task: asyncio.Task | None = None
        self._alerts: list[RiskAlert] = []
        self._emergency_stop_active = False

        # Default thresholds - delegate to centralized monitor
        self._thresholds = self._unified_monitor.get_thresholds()

        # Register consistent message handlers for risk events
        self._setup_message_handlers()

    async def start_monitoring(self, interval: int = 60) -> None:
        """
        Start continuous risk monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self._is_monitoring:
            self.logger.warning("Risk monitoring already active")
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))

        self.logger.info(f"Started risk monitoring with {interval}s interval")

    async def stop_monitoring(self) -> None:
        """Stop continuous risk monitoring with proper timeout handling."""
        self._is_monitoring = False

        if not self._monitoring_task:
            self.logger.info("Risk monitoring already stopped")
            return

        try:
            # Cancel the monitoring task
            self._monitoring_task.cancel()

            # Use asyncio.wait_for with proper timeout handling
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=5.0)
            except asyncio.CancelledError:
                self.logger.info("Monitoring task cancelled successfully")
            except asyncio.TimeoutError:
                self.logger.warning("Monitoring task cancellation timed out")
            except Exception as e:
                self.logger.warning(f"Error during monitoring task cancellation: {e}")

        finally:
            # Always clear the task reference
            self._monitoring_task = None
            self.logger.info("Stopped risk monitoring")

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
        Monitor risk metrics using centralized monitoring utilities.

        Args:
            metrics: Risk metrics to monitor
        """
        try:
            # Validate metrics at boundary
            metrics_data = {
                "portfolio_value": str(metrics.portfolio_value),
                "current_drawdown": str(metrics.current_drawdown),
                "var_1d": str(metrics.var_1d) if metrics.var_1d else None,
                "risk_level": metrics.risk_level.value,
            }
            BoundaryValidator.validate_monitoring_to_error_boundary(
                {
                    "component": "RiskMonitoringService",
                    "error_type": "metrics_validation",
                    "severity": "info",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **metrics_data,
                }
            )

            # Use batch processing for consistent alignment with error_handling module
            await self._messaging_coordinator.batch_process(
                batch_id=f"risk_metrics_{datetime.now(timezone.utc).timestamp()}",
                data=metrics_data,
                source="RiskMonitoringService",
            )

            # Delegate to centralized monitoring (replaces local duplicate logic)
            await self._unified_monitor.monitor_metrics(metrics)

            # Only handle risk level changes locally as it involves state management
            await self._check_risk_level_changes(metrics)

        except Exception as e:
            self.logger.error(f"Error monitoring risk metrics: {e}")

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
        active_alerts = [
            alert
            for alert in self._alerts
            if not alert.acknowledged and alert.timestamp > cutoff_time
        ]

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
                self.logger.info(f"Risk alert acknowledged: {alert_id}")
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
            self.logger.info(f"Updated threshold {threshold_name} = {value}")
        else:
            self.logger.warning(f"Unknown threshold: {threshold_name}")

    async def _monitoring_loop(self, interval: int) -> None:
        """Main monitoring loop with proper connection resource management."""
        try:
            while self._is_monitoring:
                try:
                    # Use async context manager for resource-safe monitoring
                    async with self._monitoring_context():
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
                    self.logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(min(interval, 30))  # Don't spam on errors

        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in monitoring loop: {e}")
        finally:
            # Ensure cleanup on exit
            await self._cleanup_monitoring_resources()

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
                {
                    "var_pct": format_decimal(var_pct),
                    "threshold": format_decimal(self._thresholds["var_critical"]),
                },
            )
        elif var_pct > self._thresholds["var_warning"]:
            await self._create_alert(
                "var_warning",
                "HIGH",
                f"VaR exceeds warning threshold: {var_pct:.2%}",
                {
                    "var_pct": format_decimal(var_pct),
                    "threshold": format_decimal(self._thresholds["var_warning"]),
                },
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
                    "drawdown": format_decimal(drawdown),
                    "threshold": format_decimal(self._thresholds["drawdown_critical"]),
                },
            )
        elif drawdown > self._thresholds["drawdown_warning"]:
            await self._create_alert(
                "drawdown_warning",
                "HIGH",
                f"Drawdown exceeds warning threshold: {drawdown:.2%}",
                {
                    "drawdown": format_decimal(drawdown),
                    "threshold": format_decimal(self._thresholds["drawdown_warning"]),
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
                    "sharpe_ratio": format_decimal(sharpe),
                    "minimum": format_decimal(self._thresholds["sharpe_minimum"]),
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
            from src.core.types import StateType

            await self.state_service.set_state(
                state_type=StateType.RISK_STATE,
                state_id="current_level",
                state_data={
                    "risk_level": metrics.risk_level.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                source_component="RiskMonitoringService",
                reason="Risk level change detected",
            )

        except Exception as e:
            self.logger.error(f"Error checking risk level changes: {e}")

    async def _check_portfolio_concentration(self, metrics: RiskMetrics) -> None:
        """Check portfolio concentration risk."""
        if metrics.correlation_risk and metrics.correlation_risk > to_decimal("0.7"):
            await self._create_alert(
                "concentration_high",
                "MEDIUM",
                f"High portfolio concentration: {metrics.correlation_risk:.2%}",
                {"concentration": format_decimal(metrics.correlation_risk)},
            )

    async def _create_alert(
        self, alert_type: str, severity: str, message: str, details: dict[str, Any]
    ) -> None:
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

        self.logger.warning(f"Risk alert created: {message}", extra=details)

    async def _trigger_emergency_stop(self, conditions: list[str]) -> None:
        """Trigger emergency stop using centralized utilities."""
        if self._emergency_stop_active:
            return

        self._emergency_stop_active = True
        reason = f"Emergency conditions: {'; '.join(conditions)}"

        # Use centralized emergency stop trigger
        try:
            await self._unified_monitor.trigger_emergency_stop(reason)
        except Exception as e:
            self.logger.error(f"Error triggering centralized emergency stop: {e}")

        # Create critical alert (local backup)
        await self._create_alert(
            "emergency_stop",
            "CRITICAL",
            f"Emergency stop triggered: {reason}",
            {"conditions": conditions, "reason": reason},
        )

        # Store emergency state
        from src.core.types import StateType

        await self.state_service.set_state(
            state_type=StateType.RISK_STATE,
            state_id="emergency_stop",
            state_data={
                "active": True,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "conditions": conditions,
            },
            source_component="RiskMonitoringService",
            reason=f"Emergency stop triggered: {reason}",
        )

        self.logger.critical("EMERGENCY STOP TRIGGERED", reason=reason, conditions=conditions)

    async def _get_latest_metrics(self) -> RiskMetrics | None:
        """Get latest risk metrics from state."""
        try:
            from src.core.types import StateType

            metrics_data = await self.state_service.get_state(
                StateType.RISK_STATE, "latest_metrics"
            )
            if not metrics_data:
                return None

            # Convert back to RiskMetrics object
            # This is a simplified version - in practice you'd have proper serialization
            return None  # Metrics deserialization not implemented - returns None for safety

        except Exception as e:
            self.logger.error(f"Error getting latest metrics: {e}")
            return None

    async def _get_previous_risk_level(self) -> RiskLevel | None:
        """Get previous risk level from state."""
        try:
            from src.core.types import StateType

            level_data = await self.state_service.get_state(StateType.RISK_STATE, "current_level")
            if level_data and "risk_level" in level_data:
                return RiskLevel(level_data["risk_level"])
            return None
        except Exception as e:
            self.logger.error(f"Error getting previous risk level: {e}")
            return None

    async def get_risk_summary(self) -> dict[str, Any]:
        """
        Get comprehensive risk summary.

        Returns:
            Dictionary with risk summary information
        """
        try:
            active_alerts = await self.get_active_alerts(limit=10)

            # Get emergency stop status
            from src.core.types import StateType

            emergency_state = await self.state_service.get_state(
                StateType.RISK_STATE, "emergency_stop"
            )
            emergency_active = emergency_state and emergency_state.get("active", False)

            # Get current risk level
            current_level = await self._get_previous_risk_level() or RiskLevel.LOW

            summary = {
                "active_alerts_count": len(active_alerts),
                "monitoring_active": self._is_monitoring,
                "emergency_stop_active": emergency_active,
                "current_risk_level": current_level.value,
                "alert_types": {
                    alert_type: len([a for a in active_alerts if a.alert_type == alert_type])
                    for alert_type in set(alert.alert_type for alert in active_alerts)
                },
                "thresholds": {name: str(value) for name, value in self._thresholds.items()},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self.logger.info(
                "Risk summary generated",
                alert_count=len(active_alerts),
                monitoring_active=self._is_monitoring,
            )

            return summary

        except Exception as e:
            self.logger.error(f"Error generating risk summary: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def _setup_message_handlers(self) -> None:
        """Setup consistent message handlers for risk events using standardized patterns."""
        from src.utils.messaging_patterns import (
            MessageHandler,
            StandardMessage,
        )

        class RiskEventHandler(MessageHandler):
            """Handler for risk events using consistent pub/sub pattern."""

            def __init__(self, monitoring_service):
                self.monitoring_service = monitoring_service

            async def handle(self, message: StandardMessage) -> StandardMessage | None:
                """Handle risk events with consistent data transformation."""
                try:
                    if (
                        message.pattern == MessagePattern.PUB_SUB
                        and message.message_type == MessageType.EVENT
                    ):
                        # Apply consistent data transformation
                        event_data = message.data

                        # Process different risk event types
                        if event_data.get("event_type") == "threshold_breach":
                            await self.monitoring_service._handle_threshold_breach(event_data)
                        elif event_data.get("event_type") == "emergency_condition":
                            await self.monitoring_service._handle_emergency_condition(event_data)
                        elif event_data.get("event_type") == "risk_level_change":
                            await self.monitoring_service._handle_risk_level_change(event_data)

                    return None  # Pub/sub doesn't require response

                except Exception as e:
                    self.monitoring_service._logger.error(f"Risk event handler error: {e}")
                    return None

        # Register the handler for pub/sub risk events
        risk_handler = RiskEventHandler(self)
        self._messaging_coordinator.register_handler(MessagePattern.PUB_SUB, risk_handler)

    async def _handle_threshold_breach(self, event_data: dict) -> None:
        """Handle threshold breach events using consistent patterns."""
        try:
            # Use consistent data transformation from messaging patterns
            transformed_data = self._messaging_coordinator._apply_data_transformation(event_data)

            # Create alert using boundary validation
            from src.utils.messaging_patterns import BoundaryValidator

            BoundaryValidator.validate_database_entity(transformed_data, "create")

            await self._create_alert(
                alert_type=transformed_data.get("alert_type", "threshold_breach"),
                severity=transformed_data.get("severity", "WARNING"),
                message=transformed_data.get("message", "Threshold breach detected"),
                details=transformed_data,
            )

        except Exception as e:
            self.logger.error(f"Error handling threshold breach: {e}")

    async def _handle_emergency_condition(self, event_data: dict) -> None:
        """Handle emergency condition events using consistent patterns."""
        try:
            # Apply consistent data transformation
            transformed_data = self._messaging_coordinator._apply_data_transformation(event_data)

            # Trigger emergency procedures
            conditions = transformed_data.get("conditions", [])
            if conditions:
                await self._trigger_emergency_stop(conditions)

        except Exception as e:
            self.logger.error(f"Error handling emergency condition: {e}")

    async def _handle_risk_level_change(self, event_data: dict) -> None:
        """Handle risk level change events using consistent patterns."""
        try:
            # Apply consistent data transformation
            transformed_data = self._messaging_coordinator._apply_data_transformation(event_data)

            # Update risk level state consistently
            from src.core.types import StateType

            await self.state_service.set_state(
                state_type=StateType.RISK_STATE,
                state_id="current_level",
                state_data={
                    "risk_level": transformed_data.get("new_level", "LOW"),
                    "previous_level": transformed_data.get("previous_level", "LOW"),
                    "timestamp": transformed_data.get("timestamp"),
                    "reason": transformed_data.get("reason", "Risk level change"),
                },
                source_component="RiskMonitoringService",
                reason="Risk level update via batch processing",
            )

        except Exception as e:
            self.logger.error(f"Error handling risk level change: {e}")

    async def publish_risk_event(self, event_type: str, event_data: dict) -> None:
        """Publish risk events using consistent pub/sub pattern aligned with monitoring module."""
        try:
            # Use standardized event constants
            from src.core.event_constants import RiskEvents

            # Map event types to constants
            event_name_map = {
                "limit_exceeded": RiskEvents.LIMIT_EXCEEDED,
                "margin_call": RiskEvents.MARGIN_CALL,
                "circuit_breaker_triggered": RiskEvents.CIRCUIT_BREAKER_TRIGGERED,
                "exposure_warning": RiskEvents.EXPOSURE_WARNING,
            }

            standardized_event_type = event_name_map.get(event_type, f"risk.{event_type}")

            # Validate at risk_management -> monitoring boundary
            boundary_data = {
                "component": "RiskMonitoringService",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_mode": "async",
                "message_pattern": "pub_sub",
                "data_format": "risk_event_v1",
                **event_data,
            }
            BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)

            # Apply consistent data transformation matching monitoring module
            transformed_data = self._messaging_coordinator._apply_data_transformation(event_data)

            # Align processing modes for consistency with monitoring module
            aligned_data = ProcessingParadigmAligner.align_processing_modes(
                source_mode="async", target_mode="batch", data=transformed_data
            )

            # Use pub/sub pattern consistent with monitoring module
            await self._messaging_coordinator.publish(
                topic=standardized_event_type,
                data=aligned_data,
                source=self.name,
                metadata={
                    "service": "risk_monitoring",
                    "event_type": standardized_event_type,
                    "message_pattern": "pub_sub",
                    "boundary_validation": "enabled",
                    "processing_paradigm": "batch_aligned",
                },
            )

        except Exception as e:
            self.logger.error(f"Error publishing risk event {event_type}: {e}")

    @asynccontextmanager
    async def _monitoring_context(self) -> AsyncIterator[None]:
        """Async context manager for monitoring operations with proper resource management."""
        connection_resources = []

        try:
            # Initialize any connection resources (placeholder for WebSocket connections)
            # In a real implementation, this would set up WebSocket connections to exchanges

            yield

        except Exception as e:
            self.logger.error(f"Error in monitoring context: {e}")
            raise
        finally:
            # Clean up any connection resources
            for resource in connection_resources:
                try:
                    if hasattr(resource, "close"):
                        await resource.close()
                except Exception as cleanup_error:
                    self.logger.warning(f"Error cleaning up connection resource: {cleanup_error}")

    async def _cleanup_monitoring_resources(self) -> None:
        """Clean up monitoring resources and connections."""
        try:
            # Clean up alerts to prevent memory leaks
            if len(self._alerts) > 1000:
                self._alerts = self._alerts[-100:]  # Keep only recent alerts

            # Reset emergency stop flag if needed
            self._emergency_stop_active = False

            self.logger.info("Monitoring resources cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Error cleaning up monitoring resources: {e}")
