"""
Strategy Dependencies - Service Container for Strategy Module

This module provides the dependency injection container for strategies module.
All strategies must use this container to access other modules through service layers.

CRITICAL: This is required for production-grade financial trading systems.
Direct module access is prohibited for safety and testability.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

# Import service interfaces to avoid circular imports
if TYPE_CHECKING:
    from src.analytics.service import AnalyticsService
    from src.capital_management.service import CapitalManagementService
    from src.data.services.data_service import DataService
    from src.execution.service import ExecutionService
    from src.ml.service import MLService
    from src.monitoring.services import MonitoringService
    from src.optimization.service import OptimizationService
    from src.risk_management.service import RiskManagementService
    from src.state.state_service import StateService

from src.core.base import BaseComponent
from src.core.logging import get_logger


@dataclass
class StrategyServiceContainer:
    """Container for all services required by strategies.
    
    This container ensures strategies access other modules through proper service layers
    rather than importing them directly. This is critical for:
    - Testability (mock services can be injected)
    - Maintainability (service interfaces can evolve)
    - Safety (business logic validation in service layers)
    - Compliance (audit trails and monitoring)
    
    MANDATORY: All strategy implementations must use this container.
    """

    # Core trading services
    risk_service: Optional["RiskManagementService"] = None
    data_service: Optional["DataService"] = None
    execution_service: Optional["ExecutionService"] = None

    # Supporting services
    monitoring_service: Optional["MonitoringService"] = None
    state_service: Optional["StateService"] = None
    capital_service: Optional["CapitalManagementService"] = None
    ml_service: Optional["MLService"] = None
    analytics_service: Optional["AnalyticsService"] = None
    optimization_service: Optional["OptimizationService"] = None

    def __post_init__(self):
        """Validate container after initialization."""
        self.logger = get_logger(__name__)

        # Check critical services are present
        critical_services = ["risk_service", "data_service", "execution_service"]
        missing_services = []

        for service_name in critical_services:
            if getattr(self, service_name) is None:
                missing_services.append(service_name)

        if missing_services:
            self.logger.warning(
                "Critical services missing from container",
                missing_services=missing_services
            )

    def is_ready(self) -> bool:
        """Check if container has all critical services."""
        critical_services = [
            self.risk_service,
            self.data_service,
            self.execution_service
        ]
        return all(service is not None for service in critical_services)

    def get_service_status(self) -> dict[str, bool]:
        """Get status of all services in container."""
        return {
            "risk_service": self.risk_service is not None,
            "data_service": self.data_service is not None,
            "execution_service": self.execution_service is not None,
            "monitoring_service": self.monitoring_service is not None,
            "state_service": self.state_service is not None,
            "capital_service": self.capital_service is not None,
            "ml_service": self.ml_service is not None,
            "analytics_service": self.analytics_service is not None,
            "optimization_service": self.optimization_service is not None,
        }


class StrategyServiceContainerBuilder(BaseComponent):
    """Builder for creating properly configured StrategyServiceContainer.
    
    This builder ensures all services are properly initialized and configured
    before being injected into strategies.
    """

    def __init__(self):
        super().__init__()
        self._container = StrategyServiceContainer()

    def with_risk_service(self, risk_service: "RiskManagementService") -> "StrategyServiceContainerBuilder":
        """Add risk management service to container."""
        self._container.risk_service = risk_service
        self.logger.debug("Risk service added to container")
        return self

    def with_data_service(self, data_service: "DataService") -> "StrategyServiceContainerBuilder":
        """Add data service to container."""
        self._container.data_service = data_service
        self.logger.debug("Data service added to container")
        return self

    def with_execution_service(self, execution_service: "ExecutionService") -> "StrategyServiceContainerBuilder":
        """Add execution service to container."""
        self._container.execution_service = execution_service
        self.logger.debug("Execution service added to container")
        return self

    def with_monitoring_service(self, monitoring_service: "MonitoringService") -> "StrategyServiceContainerBuilder":
        """Add monitoring service to container."""
        self._container.monitoring_service = monitoring_service
        self.logger.debug("Monitoring service added to container")
        return self

    def with_state_service(self, state_service: "StateService") -> "StrategyServiceContainerBuilder":
        """Add state service to container."""
        self._container.state_service = state_service
        self.logger.debug("State service added to container")
        return self

    def with_capital_service(self, capital_service: "CapitalManagementService") -> "StrategyServiceContainerBuilder":
        """Add capital management service to container."""
        self._container.capital_service = capital_service
        self.logger.debug("Capital service added to container")
        return self


    def with_ml_service(self, ml_service: "MLService") -> "StrategyServiceContainerBuilder":
        """Add ML service to container."""
        self._container.ml_service = ml_service
        self.logger.debug("ML service added to container")
        return self

    def with_analytics_service(self, analytics_service: "AnalyticsService") -> "StrategyServiceContainerBuilder":
        """Add analytics service to container."""
        self._container.analytics_service = analytics_service
        self.logger.debug("Analytics service added to container")
        return self

    def with_optimization_service(self, optimization_service: "OptimizationService") -> "StrategyServiceContainerBuilder":
        """Add optimization service to container."""
        self._container.optimization_service = optimization_service
        self.logger.debug("Optimization service added to container")
        return self

    def build(self) -> StrategyServiceContainer:
        """Build and validate the service container."""
        if not self._container.is_ready():
            missing_services = [
                name for name, available in self._container.get_service_status().items()
                if not available and name in ["risk_service", "data_service", "execution_service"]
            ]
            self.logger.warning(
                "Building container with missing critical services",
                missing_services=missing_services
            )

        self.logger.info("Strategy service container built", status=self._container.get_service_status())
        return self._container


def create_strategy_service_container(
    risk_service: Optional["RiskManagementService"] = None,
    data_service: Optional["DataService"] = None,
    execution_service: Optional["ExecutionService"] = None,
    monitoring_service: Optional["MonitoringService"] = None,
    state_service: Optional["StateService"] = None,
    capital_service: Optional["CapitalManagementService"] = None,
    ml_service: Optional["MLService"] = None,
    analytics_service: Optional["AnalyticsService"] = None,
    optimization_service: Optional["OptimizationService"] = None,
) -> StrategyServiceContainer:
    """Factory function to create a StrategyServiceContainer with all services.
    
    Args:
        risk_service: Risk management service instance
        data_service: Data service instance
        execution_service: Execution service instance
        monitoring_service: Monitoring service instance
        state_service: State service instance
        capital_service: Capital management service instance
        ml_service: ML service instance
        analytics_service: Analytics service instance
        optimization_service: Optimization service instance
    
    Returns:
        Configured StrategyServiceContainer
    """
    builder = StrategyServiceContainerBuilder()

    if risk_service:
        builder.with_risk_service(risk_service)
    if data_service:
        builder.with_data_service(data_service)
    if execution_service:
        builder.with_execution_service(execution_service)
    if monitoring_service:
        builder.with_monitoring_service(monitoring_service)
    if state_service:
        builder.with_state_service(state_service)
    if capital_service:
        builder.with_capital_service(capital_service)
    if ml_service:
        builder.with_ml_service(ml_service)
    if analytics_service:
        builder.with_analytics_service(analytics_service)
    if optimization_service:
        builder.with_optimization_service(optimization_service)

    return builder.build()
