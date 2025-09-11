"""Risk monitoring using centralized utilities to eliminate duplication."""

import asyncio
from typing import Any

from src.core.base.component import BaseComponent
from src.core.types.risk import RiskAlert, RiskMetrics
from src.utils.decimal_utils import format_decimal
from src.utils.decorators import UnifiedDecorator as dec
from src.utils.messaging_patterns import (
    BoundaryValidator,
    ErrorPropagationMixin,
    MessagingCoordinator,
)
from src.utils.risk_monitoring import get_unified_risk_monitor

# Removed: RiskObserver classes - now using centralized utilities


class RiskMonitor(BaseComponent, ErrorPropagationMixin):
    """
    Legacy risk monitor that delegates to centralized utilities.

    This maintains backward compatibility while using the centralized
    UnifiedRiskMonitor for actual functionality.
    Uses ErrorPropagationMixin for consistent error handling across modules.
    """

    def __init__(self, messaging_coordinator: MessagingCoordinator | None = None) -> None:
        """Initialize risk monitor using centralized utilities."""
        super().__init__()
        self._unified_monitor = get_unified_risk_monitor()
        self._monitoring_task: asyncio.Task | None = None
        self._running = False
        self._metrics_history: list[RiskMetrics] = []
        self._observers: list[Any] = []
        self._messaging_coordinator = messaging_coordinator or MessagingCoordinator("RiskMonitor")

        # Initialize logger from BaseComponent
        self._logger = self.logger

    # Removed: _set_default_thresholds - now handled by UnifiedRiskMonitor

    # Observer management now delegated to UnifiedRiskMonitor
    def add_observer(self, observer) -> None:
        """Add observer (legacy compatibility)."""
        self.logger.warning("Using legacy add_observer - consider migrating to UnifiedRiskMonitor")
        # Could add compatibility layer here if needed

    def remove_observer(self, observer) -> None:
        """Remove observer (legacy compatibility)."""
        self.logger.warning(
            "Using legacy remove_observer - consider migrating to UnifiedRiskMonitor"
        )
        # Could add compatibility layer here if needed

    @dec.enhance(log=True, monitor=True)
    async def monitor_metrics(self, metrics: RiskMetrics) -> None:
        """
        Monitor risk metrics using centralized utilities.

        Args:
            metrics: Current risk metrics
        """
        # Align processing modes for consistent data flow
        _ = {
            "portfolio_value": str(metrics.portfolio_value),
            "risk_level": metrics.risk_level.value,
            "timestamp": metrics.timestamp.isoformat() if metrics.timestamp else None,
        }

        # Use stream processing pattern for alignment with execution module
        from src.risk_management.data_transformer import RiskDataTransformer

        # Transform metrics for stream processing consistency
        _ = RiskDataTransformer.transform_risk_metrics_to_event_data(
            metrics, {"monitor_id": str(id(metrics))}
        )

        # Delegate to centralized monitor
        await self._unified_monitor.monitor_metrics(metrics)

    # Removed: volatility calculation methods - now handled by UnifiedRiskMonitor

    async def monitor_portfolio(self, portfolio_data: dict[str, Any]) -> None:
        """
        Monitor portfolio-level risks using centralized utilities.

        Args:
            portfolio_data: Portfolio data
        """
        # Validate portfolio data at boundary
        BoundaryValidator.validate_database_entity(portfolio_data, "monitor_portfolio")

        # Use stream processing pattern for alignment with execution module
        from src.risk_management.data_transformer import RiskDataTransformer

        # Transform portfolio data for consistent stream processing
        _ = RiskDataTransformer.transform_for_pub_sub(
            event_type="risk.portfolio_monitoring",
            data=portfolio_data,
            metadata={"monitor_id": str(id(portfolio_data))},
        )

        # Delegate to centralized monitor
        await self._unified_monitor.monitor_portfolio(portfolio_data)

    def set_threshold(self, key: str, value) -> None:
        """
        Set monitoring threshold using centralized utilities.

        Args:
            key: Threshold key
            value: Threshold value
        """
        from src.utils.decimal_utils import to_decimal

        self._unified_monitor.set_threshold(key, to_decimal(str(value)))

    def get_thresholds(self) -> dict[str, Any]:
        """Get current thresholds from centralized monitor."""
        return {k: format_decimal(v) for k, v in self._unified_monitor.get_thresholds().items()}

    async def start_monitoring(self, interval: int = 60) -> None:
        """
        Start continuous monitoring (legacy compatibility).

        Args:
            interval: Monitoring interval in seconds
        """
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        self.logger.info(f"Started legacy risk monitoring with {interval}s interval")

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring with proper resource cleanup."""
        self._running = False
        task = None

        try:
            if self._monitoring_task:
                task = self._monitoring_task
                try:
                    task.cancel()
                    # Wait for task to be cancelled with timeout
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    self.logger.info("Monitoring task cancelled")
                except Exception as e:
                    self.logger.warning(f"Error cancelling monitoring task: {e}")
                finally:
                    self._monitoring_task = None
                    task = None

            # Clean up resources
            await self._cleanup_resources()

            self.logger.info("Stopped risk monitoring")

        finally:
            # Ensure task reference is cleared
            task = None

    async def _monitoring_loop(self, interval: int) -> None:
        """Continuous monitoring loop."""
        while self._running:
            try:
                # This would fetch current metrics from the system
                # For now, it's a placeholder
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")

    async def _cleanup_resources(self) -> None:
        """Clean up all resources used by the risk monitor."""
        try:
            # Clean up metrics history with size limit
            if len(self._metrics_history) > 1000:
                self._metrics_history = self._metrics_history[-100:]  # Keep last 100 metrics

            # Clean up observer alerts
            for observer in self._observers:
                if hasattr(observer, "alerts") and hasattr(observer.alerts, "__len__"):
                    if len(observer.alerts) > 1000:  # Prevent excessive memory usage
                        observer.alerts = observer.alerts[-100:]  # Keep last 100 alerts

            self.logger.info("Risk monitor resources cleaned up")

        except Exception as e:
            self.logger.error(f"Error cleaning up risk monitor resources: {e}")

    def get_alerts(self) -> list[RiskAlert]:
        """Get recent risk alerts from centralized monitor."""
        alerting_observer = self._unified_monitor._observers.get("alerting_observer")
        if alerting_observer and hasattr(alerting_observer, "get_recent_alerts"):
            return alerting_observer.get_recent_alerts()
        return []
