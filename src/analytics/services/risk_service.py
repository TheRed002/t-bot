"""
Risk Service.

This service provides a proper service layer implementation for risk monitoring,
following service layer patterns and using dependency injection.
"""

from typing import Any

from src.analytics.interfaces import RiskServiceProtocol
from src.analytics.risk.risk_monitor import RiskMonitor
from src.analytics.types import AnalyticsConfiguration, RiskMetrics
from src.core.base.service import BaseService
from src.core.exceptions import ComponentError, ValidationError
from src.core.types import Position


class RiskService(BaseService, RiskServiceProtocol):
    """
    Service layer implementation for risk monitoring.

    This service acts as a facade over the RiskMonitor,
    providing proper service layer abstraction and dependency injection.
    """

    def __init__(
        self,
        config: AnalyticsConfiguration,
        risk_monitor: RiskMonitor | None = None,
    ):
        """
        Initialize the risk service.

        Args:
            config: Analytics configuration
            risk_monitor: Injected risk monitor engine (optional)
        """
        super().__init__()
        self.config = config

        # Use dependency injection - risk_monitor must be injected
        if risk_monitor is None:
            raise ComponentError(
                "risk_monitor must be injected via dependency injection",
                component="RiskService",
                operation="__init__",
                context={"missing_dependency": "risk_monitor"},
            )

        self._monitor = risk_monitor

        self.logger.info("RiskService initialized")

    async def start(self) -> None:
        """Start the risk service."""
        try:
            await self._monitor.start()
            self.logger.info("Risk service started")
        except Exception as e:
            raise ComponentError(
                f"Failed to start risk service: {e}",
                component="RiskService",
                operation="start",
            ) from e

    async def stop(self) -> None:
        """Stop the risk service."""
        try:
            await self._monitor.stop()
            self.logger.info("Risk service stopped")
        except Exception as e:
            self.logger.error(f"Error stopping risk service: {e}")

    def update_position(self, position: Position) -> None:
        """
        Update position data for risk monitoring.

        Args:
            position: Position to update

        Raises:
            ValidationError: If position data is invalid
            ComponentError: If update fails
        """
        if not isinstance(position, Position):
            raise ValidationError(
                "Invalid position parameter",
                field_name="position",
                field_value=type(position),
                expected_type="Position",
            )

        try:
            self._monitor.update_position(position)
            self.logger.debug(f"Position updated for risk monitoring: {position.symbol}")
        except Exception as e:
            raise ComponentError(
                f"Failed to update position for risk monitoring: {e}",
                component="RiskService",
                operation="update_position",
                context={"symbol": position.symbol},
            ) from e

    async def get_risk_metrics(self) -> RiskMetrics:
        """
        Get current risk metrics.

        Returns:
            Current risk metrics

        Raises:
            ComponentError: If retrieval fails
        """
        try:
            return await self._monitor.get_risk_metrics()
        except Exception as e:
            raise ComponentError(
                f"Failed to get risk metrics: {e}",
                component="RiskService",
                operation="get_risk_metrics",
            ) from e

    async def calculate_var(
        self,
        confidence_level: float,
        time_horizon: int,
        method: str,
    ) -> dict[str, float]:
        """
        Calculate Value at Risk.

        Args:
            confidence_level: VaR confidence level
            time_horizon: Time horizon in days
            method: Calculation method

        Returns:
            VaR calculation results

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If calculation fails
        """
        if not 0 < confidence_level < 1:
            raise ValidationError(
                "Invalid confidence_level parameter",
                field_name="confidence_level",
                field_value=confidence_level,
                validation_rule="must be between 0 and 1",
            )

        if time_horizon <= 0:
            raise ValidationError(
                "Invalid time_horizon parameter",
                field_name="time_horizon",
                field_value=time_horizon,
                validation_rule="must be positive",
            )

        try:
            return await self._monitor.calculate_var(confidence_level, time_horizon, method)
        except Exception as e:
            raise ComponentError(
                f"Failed to calculate VaR: {e}",
                component="RiskService",
                operation="calculate_var",
                context={
                    "confidence_level": confidence_level,
                    "time_horizon": time_horizon,
                    "method": method,
                },
            ) from e

    async def run_stress_test(
        self, scenario_name: str, scenario_params: dict[str, Any]
    ) -> dict[str, float]:
        """
        Run stress test scenario.

        Args:
            scenario_name: Stress test scenario name
            scenario_params: Scenario parameters

        Returns:
            Stress test results

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If stress test fails
        """
        if not isinstance(scenario_name, str) or not scenario_name:
            raise ValidationError(
                "Invalid scenario_name parameter",
                field_name="scenario_name",
                field_value=scenario_name,
                expected_type="non-empty str",
            )

        try:
            return await self._monitor.run_stress_test(scenario_name, scenario_params)
        except Exception as e:
            raise ComponentError(
                f"Failed to run stress test: {e}",
                component="RiskService",
                operation="run_stress_test",
                context={"scenario": scenario_name},
            ) from e

    async def calculate_advanced_var_methodologies(self) -> dict[str, Any]:
        """
        Calculate advanced VaR methodologies.

        Returns:
            Advanced VaR results

        Raises:
            ComponentError: If calculation fails
        """
        try:
            return await self._monitor.calculate_advanced_var_methodologies()
        except Exception as e:
            raise ComponentError(
                f"Failed to calculate advanced VaR methodologies: {e}",
                component="RiskService",
                operation="calculate_advanced_var_methodologies",
            ) from e

    async def execute_comprehensive_stress_test(self) -> dict[str, Any]:
        """
        Execute comprehensive stress test.

        Returns:
            Comprehensive stress test results

        Raises:
            ComponentError: If stress test fails
        """
        try:
            return await self._monitor.execute_comprehensive_stress_test()
        except Exception as e:
            raise ComponentError(
                f"Failed to execute comprehensive stress test: {e}",
                component="RiskService",
                operation="execute_comprehensive_stress_test",
            ) from e

    async def create_real_time_risk_dashboard(self) -> dict[str, Any]:
        """
        Create real-time risk dashboard.

        Returns:
            Risk dashboard data

        Raises:
            ComponentError: If dashboard creation fails
        """
        try:
            return await self._monitor.create_real_time_risk_dashboard()
        except Exception as e:
            raise ComponentError(
                f"Failed to create real-time risk dashboard: {e}",
                component="RiskService",
                operation="create_real_time_risk_dashboard",
            ) from e

    def store_risk_metrics(self, risk_metrics: RiskMetrics) -> None:
        """
        Store risk metrics for analytics.

        Args:
            risk_metrics: Risk metrics to store

        Raises:
            ValidationError: If risk metrics data is invalid
            ComponentError: If storage fails
        """
        if not risk_metrics:
            raise ValidationError(
                "Invalid risk metrics parameter",
                field_name="risk_metrics",
                field_value=risk_metrics,
                expected_type="RiskMetrics",
            )

        try:
            self._monitor.store_risk_metrics(risk_metrics)
            self.logger.debug("Risk metrics stored for analytics")
        except Exception as e:
            raise ComponentError(
                f"Failed to store risk metrics: {e}",
                component="RiskService",
                operation="store_risk_metrics",
            ) from e
