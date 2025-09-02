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
from src.core.dependency_injection import DependencyInjector, get_container
from src.core.exceptions import DependencyError
from src.core.logging import get_logger

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from src.database.service import DatabaseService
    from src.monitoring.metrics import MetricsCollector
    from src.state import StateService

from .controller import RiskManagementController
from .interfaces import RiskManagementFactoryInterface, RiskServiceInterface
from .position_sizing import PositionSizer
from .risk_manager import RiskManager
from .risk_metrics import RiskCalculator
from .service import RiskService

logger = get_logger(__name__)


class RiskManagementFactory(RiskManagementFactoryInterface):
    """
    Factory for creating risk management components.

    Provides methods to create both legacy components and the new RiskService,
    with automatic dependency injection and configuration management.
    """

    def __init__(
        self,
        injector: DependencyInjector | None = None,
        config: Config | None = None,
        database_service: "DatabaseService | None" = None,
        state_service: "StateService | None" = None,
        metrics_collector: "MetricsCollector | None" = None,
    ):
        """
        Initialize risk management factory.

        Args:
            injector: Dependency injector instance
            config: Application configuration
            database_service: Database service for data access
            state_service: State service for state management
            metrics_collector: Metrics collector service
        """
        # Initialize dependency injector
        if injector is None:
            from .di_registration import configure_risk_management_dependencies

            injector = configure_risk_management_dependencies()

        self.injector = injector

        # Register explicit dependencies if provided
        if config is not None:
            self.injector.register_factory("Config", lambda: config, singleton=True)
        if database_service is not None:
            self.injector.register_factory("DatabaseService", lambda: database_service, singleton=True)
        if state_service is not None:
            self.injector.register_factory("StateService", lambda: state_service, singleton=True)
        if metrics_collector is not None:
            self.injector.register_factory("MetricsCollector", lambda: metrics_collector, singleton=True)

        # Fallback config resolution for backward compatibility
        if config is None:
            try:
                config_service = get_container().get("ConfigService")
                # Create legacy Config object from ConfigService
                config = Config()
                config.risk = config_service.get_risk_config()
                config.database = config_service.get_database_config()
                config.exchange = config_service.get_exchange_config()
                self.injector.register_factory("Config", lambda: config, singleton=True)
            except (KeyError, AttributeError):
                # Fallback to legacy method for backward compatibility
                config = get_config()
                self.injector.register_factory("Config", lambda: config, singleton=True)

        logger.info(
            "RiskManagementFactory initialized with dependency injection",
            has_injector=injector is not None,
            has_config_service=self.injector.has_service("Config"),
            has_database_service=self.injector.has_service("DatabaseService"),
            has_state_service=self.injector.has_service("StateService"),
        )

    def create_risk_service(self, correlation_id: str | None = None) -> RiskServiceInterface:
        """
        Create RiskService instance using dependency injection.

        Args:
            correlation_id: Request correlation ID

        Returns:
            RiskServiceInterface instance

        Raises:
            DependencyError: If required services are not available
        """
        try:
            # Use dependency injection to create the service
            risk_service = self.injector.resolve("RiskService")

            # Set correlation ID if provided
            if correlation_id and hasattr(risk_service, "correlation_id"):
                risk_service.correlation_id = correlation_id

            logger.info(
                "RiskService created via dependency injection",
                correlation_id=correlation_id,
            )

            return risk_service

        except Exception as e:
            logger.error(f"Failed to create RiskService via dependency injection: {e}")
            raise DependencyError(f"Failed to create RiskService: {e}") from e

    def create_legacy_risk_manager(self) -> RiskManager:
        """
        Create a legacy RiskManager with optional service integration.

        DEPRECATED: Use create_risk_service() instead.

        Returns:
            RiskManager instance
        """
        logger.warning("Creating DEPRECATED RiskManager - consider using RiskService instead")

        try:
            config = self.injector.resolve("Config")
            database_service = (
                self.injector.resolve("DatabaseService") if self.injector.has_service("DatabaseService") else None
            )
            state_service = self.injector.resolve("StateService") if self.injector.has_service("StateService") else None
        except Exception as e:
            logger.error(f"Failed to resolve services from DI container: {e}")
            raise DependencyError("Failed to resolve required dependencies for legacy RiskManager") from e

        return RiskManager(
            config=config,
            database_service=database_service,
            state_service=state_service,
        )

    def create_legacy_position_sizer(self) -> PositionSizer:
        """
        Create a legacy PositionSizer.

        DEPRECATED: Use RiskService.calculate_position_size() instead.

        Returns:
            PositionSizer instance
        """
        logger.warning("Creating DEPRECATED PositionSizer - use RiskService.calculate_position_size() instead")

        try:
            config = self.injector.resolve("Config")
            database_service = (
                self.injector.resolve("DatabaseService") if self.injector.has_service("DatabaseService") else None
            )
        except Exception as e:
            logger.error(f"Failed to resolve services from DI container: {e}")
            raise DependencyError("Failed to resolve required dependencies for legacy PositionSizer") from e

        return PositionSizer(
            config=config,
            database_service=database_service,
        )

    def create_legacy_risk_calculator(self) -> RiskCalculator:
        """
        Create a legacy RiskCalculator.

        DEPRECATED: Use RiskService.calculate_risk_metrics() instead.

        Returns:
            RiskCalculator instance
        """
        logger.warning("Creating DEPRECATED RiskCalculator - use RiskService.calculate_risk_metrics() instead")

        try:
            config = self.injector.resolve("Config")
            database_service = (
                self.injector.resolve("DatabaseService") if self.injector.has_service("DatabaseService") else None
            )
        except Exception as e:
            logger.error(f"Failed to resolve services from DI container: {e}")
            raise DependencyError("Failed to resolve required dependencies for legacy RiskCalculator") from e

        return RiskCalculator(
            config=config,
            database_service=database_service,
        )

    def create_risk_management_controller(self, correlation_id: str | None = None) -> RiskManagementController:
        """
        Create a new risk management controller using dependency injection.

        This is the RECOMMENDED approach for new implementations.

        Args:
            correlation_id: Request correlation ID

        Returns:
            RiskManagementController instance

        Raises:
            DependencyError: If required services are not available
        """
        try:
            # Get services from DI injector using service locator pattern
            position_sizing_service = self.injector.resolve("PositionSizingService")
            risk_validation_service = self.injector.resolve("RiskValidationService")
            risk_metrics_service = self.injector.resolve("RiskMetricsService")
            risk_monitoring_service = self.injector.resolve("RiskMonitoringService")

            controller = RiskManagementController(
                position_sizing_service=position_sizing_service,
                risk_validation_service=risk_validation_service,
                risk_metrics_service=risk_metrics_service,
                risk_monitoring_service=risk_monitoring_service,
                correlation_id=correlation_id,
            )

            logger.info(
                "RiskManagementController created via dependency injection",
                correlation_id=correlation_id,
            )

            return controller

        except Exception as e:
            logger.error(f"Failed to create RiskManagementController via dependency injection: {e}")
            raise DependencyError(f"Failed to create RiskManagementController: {e}") from e

    def get_recommended_component(self) -> RiskService | RiskManager:
        """
        Get the recommended risk management component based on available services.

        Returns:
            RiskService if all dependencies are available, otherwise RiskManager
        """
        try:
            # Check if required services are available in DI container
            has_database = self.injector.has_service("DatabaseService")
            has_state = self.injector.has_service("StateService")

            if has_database and has_state:
                logger.info("Creating recommended RiskService via dependency injection")
                return self.create_risk_service()
            else:
                missing_services = []
                if not has_database:
                    missing_services.append("DatabaseService")
                if not has_state:
                    missing_services.append("StateService")

                logger.warning(
                    f"Missing services {missing_services} in DI container - " "falling back to legacy RiskManager"
                )
                return self.create_legacy_risk_manager()

        except Exception as e:
            logger.error(f"Failed to create recommended component: {e}")
            logger.warning("Falling back to legacy RiskManager")
            return self.create_legacy_risk_manager()

    def validate_dependencies(self) -> dict[str, bool]:
        """
        Validate the availability of dependencies via dependency injection.

        Returns:
            Dictionary with dependency validation results
        """
        validation = {
            "injector_available": self.injector is not None,
            "config": self.injector.has_service("Config") if self.injector else False,
            "database_service": self.injector.has_service("DatabaseService") if self.injector else False,
            "state_service": self.injector.has_service("StateService") if self.injector else False,
            "risk_service_available": (self.injector.has_service("RiskService") if self.injector else False),
            "position_sizing_service": self.injector.has_service("PositionSizingService") if self.injector else False,
            "risk_validation_service": self.injector.has_service("RiskValidationService") if self.injector else False,
            "risk_metrics_service": self.injector.has_service("RiskMetricsService") if self.injector else False,
            "risk_monitoring_service": self.injector.has_service("RiskMonitoringService") if self.injector else False,
        }

        logger.info("Dependency validation via DI container", **validation)
        return validation

    async def start_services(self) -> None:
        """Start all managed services via dependency injection."""
        try:
            risk_service = self.injector.resolve("RiskService")
            if hasattr(risk_service, "start"):
                await risk_service.start()
                logger.info("RiskService started via DI")
        except Exception as e:
            logger.warning(f"Failed to start RiskService via DI: {e}")

    async def stop_services(self) -> None:
        """Stop all managed services via dependency injection."""
        try:
            risk_service = self.injector.resolve("RiskService")
            if hasattr(risk_service, "stop"):
                await risk_service.stop()
                logger.info("RiskService stopped via DI")
        except Exception as e:
            logger.warning(f"Failed to stop RiskService via DI: {e}")

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
            "New: Emergency controls": ("RiskService.trigger_emergency_stop() / reset_emergency_stop()"),
            "New: Enhanced caching": "Built into RiskService methods",
            "New: Circuit breakers": "Built into RiskService operations",
            "New: State management": "Integrated with StateService",
            "New: Database integration": "Integrated with DatabaseService",
            # RECOMMENDED: Controller pattern with service delegation
            "RECOMMENDED: Use RiskManagementController": ("factory.create_risk_management_controller()"),
            "RECOMMENDED: Service delegation": "Controller -> Service -> Repository pattern",
            "RECOMMENDED: Dependency injection": "All services injected through constructor",
        }


# Global factory instance for convenience
_global_factory: RiskManagementFactory | None = None


def get_risk_factory(
    injector: DependencyInjector | None = None,
    config: Config | None = None,
    database_service: "DatabaseService | None" = None,
    state_service: "StateService | None" = None,
) -> RiskManagementFactory:
    """
    Get or create global risk management factory using dependency injection.

    Args:
        injector: Dependency injector instance
        config: Application configuration (for backward compatibility)
        database_service: Database service (for backward compatibility)
        state_service: State service (for backward compatibility)

    Returns:
        RiskManagementFactory instance
    """
    global _global_factory

    if _global_factory is None:
        _global_factory = RiskManagementFactory(
            injector=injector,
            config=config,
            database_service=database_service,
            state_service=state_service,
        )

    return _global_factory


def create_risk_service(
    injector: DependencyInjector | None = None,
    config: Config | None = None,
    database_service: "DatabaseService | None" = None,
    state_service: "StateService | None" = None,
    correlation_id: str | None = None,
) -> RiskServiceInterface:
    """
    Convenience function to create RiskService using dependency injection.

    Args:
        injector: Dependency injector instance
        config: Application configuration (for backward compatibility)
        database_service: Database service (for backward compatibility)
        state_service: State service (for backward compatibility)
        correlation_id: Request correlation ID

    Returns:
        RiskServiceInterface instance
    """
    factory = get_risk_factory(
        injector=injector,
        config=config,
        database_service=database_service,
        state_service=state_service,
    )
    return factory.create_risk_service(correlation_id=correlation_id)


def create_recommended_risk_component(
    injector: DependencyInjector | None = None,
    config: Config | None = None,
    database_service: "DatabaseService | None" = None,
    state_service: "StateService | None" = None,
) -> RiskService | RiskManager:
    """
    Convenience function to create the recommended risk component.

    DEPRECATED: Use create_risk_management_controller() for new implementations.

    Args:
        injector: Dependency injector instance
        config: Application configuration (for backward compatibility)
        database_service: Database service (for backward compatibility)
        state_service: State service (for backward compatibility)

    Returns:
        RiskService if dependencies are available, otherwise RiskManager
    """
    factory = get_risk_factory(injector, config, database_service, state_service)
    return factory.get_recommended_component()


def create_risk_management_controller(
    injector: DependencyInjector | None = None,
    config: Config | None = None,
    database_service: "DatabaseService | None" = None,
    state_service: "StateService | None" = None,
    correlation_id: str | None = None,
) -> RiskManagementController:
    """
    Convenience function to create the RECOMMENDED risk management controller.

    This follows proper controller->service->repository patterns with dependency injection.

    Args:
        injector: Dependency injector instance
        config: Application configuration (for backward compatibility)
        database_service: Database service (for backward compatibility)
        state_service: State service (for backward compatibility)
        correlation_id: Request correlation ID

    Returns:
        RiskManagementController instance
    """
    factory = get_risk_factory(injector, config, database_service, state_service)
    return factory.create_risk_management_controller(correlation_id)
