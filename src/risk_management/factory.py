"""
Risk Management Factory for creating and managing risk components.

This factory provides a centralized way to create and configure risk management
components, supporting both legacy components and the new RiskService architecture.

It handles:
- Dependency injection for services
- Configuration management
- Migration assistance from legacy to new architecture
- Component lifecycle management
"""

from typing import TYPE_CHECKING

from src.core.config.main import Config, get_config
from src.core.dependency_injection import get_container
from src.core.exceptions import DependencyError
from src.core.logging import get_logger

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from src.database.service import DatabaseService
    from src.monitoring.metrics import MetricsCollector
    from src.state import StateService

from .position_sizing import PositionSizer
from .risk_manager import RiskManager
from .risk_metrics import RiskCalculator
from .service import RiskService

logger = get_logger(__name__)


class RiskManagementFactory:
    """
    Factory for creating risk management components.

    Provides methods to create both legacy components and the new RiskService,
    with automatic dependency injection and configuration management.
    """

    def __init__(
        self,
        config: Config | None = None,
        database_service: "DatabaseService | None" = None,
        state_service: "StateService | None" = None,
        metrics_collector: "MetricsCollector | None" = None,
    ):
        """
        Initialize risk management factory.

        Args:
            config: Application configuration
            database_service: Database service for data access
            state_service: State service for state management
        """
        if config is None:
            try:
                config_service = get_container().get("ConfigService")
                # Create legacy Config object from ConfigService
                config = Config()
                config.risk = config_service.get_risk_config()
                config.database = config_service.get_database_config()
                config.exchange = config_service.get_exchange_config()
            except (KeyError, AttributeError):
                # Fallback to legacy method for backward compatibility
                config = get_config()

        self.config = config
        self.database_service = database_service
        self.state_service = state_service
        self.metrics_collector = metrics_collector
        self._risk_service_instance: RiskService | None = None

        logger.info(
            "RiskManagementFactory initialized",
            has_database_service=database_service is not None,
            has_state_service=state_service is not None,
        )

    def create_risk_service(
        self, correlation_id: str | None = None, force_recreate: bool = False
    ) -> RiskService:
        """
        Create or get the RiskService instance (singleton pattern).

        Args:
            correlation_id: Request correlation ID
            force_recreate: Force creation of new instance

        Returns:
            RiskService instance

        Raises:
            DependencyError: If required services are not available
        """
        if self._risk_service_instance is None or force_recreate:
            if not self.database_service:
                raise DependencyError("DatabaseService is required for RiskService")

            if not self.state_service:
                raise DependencyError("StateService is required for RiskService")

            # Try to get metrics collector if not provided
            metrics_collector = self.metrics_collector
            if metrics_collector is None:
                try:
                    metrics_collector = get_container().get("MetricsCollectorProtocol")
                except Exception:
                    pass  # Will fallback in RiskService
            
            self._risk_service_instance = RiskService(
                database_service=self.database_service,
                state_service=self.state_service,
                config=self.config,
                correlation_id=correlation_id,
                metrics_collector=metrics_collector,
            )

            logger.info(
                "RiskService created",
                correlation_id=correlation_id,
                force_recreate=force_recreate,
            )

        return self._risk_service_instance

    def create_legacy_risk_manager(self) -> RiskManager:
        """
        Create a legacy RiskManager with optional service integration.

        DEPRECATED: Use create_risk_service() instead.

        Returns:
            RiskManager instance
        """
        logger.warning("Creating DEPRECATED RiskManager - consider using RiskService instead")

        return RiskManager(
            config=self.config,
            database_service=self.database_service,
            state_service=self.state_service,
        )

    def create_legacy_position_sizer(self) -> PositionSizer:
        """
        Create a legacy PositionSizer.

        DEPRECATED: Use RiskService.calculate_position_size() instead.

        Returns:
            PositionSizer instance
        """
        logger.warning(
            "Creating DEPRECATED PositionSizer - use RiskService.calculate_position_size() instead"
        )

        return PositionSizer(
            config=self.config,
            database_service=self.database_service,
        )

    def create_legacy_risk_calculator(self) -> RiskCalculator:
        """
        Create a legacy RiskCalculator.

        DEPRECATED: Use RiskService.calculate_risk_metrics() instead.

        Returns:
            RiskCalculator instance
        """
        logger.warning(
            "Creating DEPRECATED RiskCalculator - use RiskService.calculate_risk_metrics() instead"
        )

        return RiskCalculator(
            config=self.config,
            database_service=self.database_service,
        )

    def get_recommended_component(self) -> RiskService | RiskManager:
        """
        Get the recommended risk management component based on available services.

        Returns:
            RiskService if all dependencies are available, otherwise RiskManager
        """
        try:
            if self.database_service and self.state_service:
                logger.info("Creating recommended RiskService")
                return self.create_risk_service()
            else:
                missing_services = []
                if not self.database_service:
                    missing_services.append("DatabaseService")
                if not self.state_service:
                    missing_services.append("StateService")

                logger.warning(
                    f"Missing services {missing_services} - falling back to legacy RiskManager"
                )
                return self.create_legacy_risk_manager()

        except Exception as e:
            logger.error(f"Failed to create recommended component: {e}")
            logger.warning("Falling back to legacy RiskManager")
            return self.create_legacy_risk_manager()

    def validate_dependencies(self) -> dict[str, bool]:
        """
        Validate the availability of dependencies.

        Returns:
            Dictionary with dependency validation results
        """
        validation = {
            "config": self.config is not None,
            "database_service": self.database_service is not None,
            "state_service": self.state_service is not None,
            "risk_service_available": (
                self.database_service is not None and self.state_service is not None
            ),
        }

        logger.debug("Dependency validation", **validation)
        return validation

    async def start_services(self) -> None:
        """Start all managed services."""
        if self._risk_service_instance:
            await self._risk_service_instance.start()
            logger.info("RiskService started")

    async def stop_services(self) -> None:
        """Stop all managed services."""
        if self._risk_service_instance:
            await self._risk_service_instance.stop()
            logger.info("RiskService stopped")

    def get_migration_guide(self) -> dict[str, str]:
        """
        Get migration guide from legacy components to RiskService.

        Returns:
            Dictionary with migration mappings
        """
        return {
            "RiskManager.calculate_position_size()": "RiskService.calculate_position_size()",
            "RiskManager.validate_signal()": "RiskService.validate_signal()",
            "RiskManager.validate_order()": "RiskService.validate_order()",
            "RiskManager.calculate_risk_metrics()": "RiskService.calculate_risk_metrics()",
            "RiskManager.emergency_stop()": "RiskService.trigger_emergency_stop()",
            "PositionSizer.calculate_position_size()": "RiskService.calculate_position_size()",
            "RiskCalculator.calculate_risk_metrics()": "RiskService.calculate_risk_metrics()",
            # New capabilities in RiskService
            "New: Real-time monitoring": "RiskService background monitoring",
            "New: Risk alerts": "RiskService.get_risk_alerts()",
            "New: Portfolio metrics": "RiskService.get_portfolio_metrics()",
            "New: Risk summary": "RiskService.get_risk_summary()",
            "New: Emergency controls": "RiskService.trigger_emergency_stop() / reset_emergency_stop()",
            "New: Enhanced caching": "Built into RiskService methods",
            "New: Circuit breakers": "Built into RiskService operations",
            "New: State management": "Integrated with StateService",
            "New: Database integration": "Integrated with DatabaseService",
        }


# Global factory instance for convenience
_global_factory: RiskManagementFactory | None = None


def get_risk_factory(
    config: Config | None = None,
    database_service: "DatabaseService | None" = None,
    state_service: "StateService | None" = None,
) -> RiskManagementFactory:
    """
    Get or create global risk management factory.

    Args:
        config: Application configuration
        database_service: Database service
        state_service: State service

    Returns:
        RiskManagementFactory instance
    """
    global _global_factory

    if _global_factory is None:
        _global_factory = RiskManagementFactory(
            config=config,
            database_service=database_service,
            state_service=state_service,
        )

    return _global_factory


def create_risk_service(
    config: Config | None = None,
    database_service: "DatabaseService | None" = None,
    state_service: "StateService | None" = None,
    correlation_id: str | None = None,
) -> RiskService:
    """
    Convenience function to create RiskService.

    Args:
        config: Application configuration
        database_service: Database service
        state_service: State service
        correlation_id: Request correlation ID

    Returns:
        RiskService instance
    """
    if not database_service:
        raise DependencyError("DatabaseService is required for RiskService")

    if not state_service:
        raise DependencyError("StateService is required for RiskService")

    if config is None:
        try:
            config_service = get_container().get("ConfigService")
            # Create legacy Config object from ConfigService
            config = Config()
            config.risk = config_service.get_risk_config()
            config.database = config_service.get_database_config()
            config.exchange = config_service.get_exchange_config()
        except (KeyError, AttributeError):
            # Fallback to legacy method for backward compatibility
            config = get_config()

    return RiskService(
        database_service=database_service,
        state_service=state_service,
        config=config,
        correlation_id=correlation_id,
    )


def create_recommended_risk_component(
    config: Config | None = None,
    database_service: "DatabaseService | None" = None,
    state_service: "StateService | None" = None,
) -> RiskService | RiskManager:
    """
    Convenience function to create the recommended risk component.

    Args:
        config: Application configuration
        database_service: Database service
        state_service: State service

    Returns:
        RiskService if dependencies are available, otherwise RiskManager
    """
    factory = get_risk_factory(config, database_service, state_service)
    return factory.get_recommended_component()
