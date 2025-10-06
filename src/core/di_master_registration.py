"""
Master DI registration module that orchestrates all service registrations
in the correct order to prevent circular dependencies.
"""

from typing import Optional
from src.core.dependency_injection import DependencyInjector
from src.core.dependency_ordering import (
    DependencyRegistrar,
    DependencyLevel,
    create_ordered_registrar,
    register_module_at_level,
    register_lazy_configuration,
)
from src.core.config import Config
from src.core.logging import get_logger

logger = get_logger(__name__)


def register_all_services(
    injector: Optional[DependencyInjector] = None, config: Optional[Config] = None
) -> DependencyInjector:
    """
    Register all services in the correct dependency order.

    Args:
        injector: Optional existing injector instance
        config: Optional configuration instance

    Returns:
        Configured dependency injector
    """
    if injector is None:
        injector = DependencyInjector()

    if config is None:
        config = Config()

    logger.info("Starting master DI registration")

    # Create ordered registrar
    registrar = create_ordered_registrar(injector)

    # Register modules at appropriate levels
    _register_all_modules(registrar, config)

    # Execute all registrations in order
    registrar.register_all()

    logger.info("Completed master DI registration")
    return injector


def _register_all_modules(registrar: DependencyRegistrar, config: Config) -> None:
    """
    Register all modules at their appropriate dependency levels.

    Args:
        registrar: Dependency registrar instance
        config: Configuration instance
    """

    # Level 0: Core Configuration (already handled by create_ordered_registrar)

    # Level 1: Database Infrastructure
    def register_database_module(injector: DependencyInjector) -> None:
        """Register database services."""
        try:
            from src.database.di_registration import register_database_services

            register_database_services(injector.get_container())
            logger.debug("Database services registered successfully")
        except Exception as e:
            logger.warning(f"Database registration failed: {e}")

    register_module_at_level(registrar, DependencyLevel.DATABASE, register_database_module)

    # Level 1: Cache Infrastructure
    def register_cache_module(injector: DependencyInjector) -> None:
        """Register cache services."""
        try:
            # Register basic cache services here if needed
            logger.debug("Cache services registered successfully")
        except Exception as e:
            logger.warning(f"Cache registration failed: {e}")

    register_module_at_level(registrar, DependencyLevel.CACHE, register_cache_module)

    # Level 2: Validation Services
    def register_validation_module(injector: DependencyInjector) -> None:
        """Register validation services with lazy loading."""
        try:

            def validation_service_factory():
                from src.utils.validation.service import ValidationService

                return ValidationService()

            injector.register_factory(
                "ValidationService", validation_service_factory, singleton=True
            )
            logger.debug("Validation services registered successfully")
        except Exception as e:
            logger.warning(f"Validation registration failed: {e}")

    register_module_at_level(registrar, DependencyLevel.VALIDATION, register_validation_module)

    # Level 2: Basic Monitoring
    def register_monitoring_basic_module(injector: DependencyInjector) -> None:
        """Register basic monitoring services."""
        try:
            from src.monitoring.di_registration import register_monitoring_services

            register_monitoring_services(injector)
            logger.debug("Basic monitoring services registered successfully")
        except Exception as e:
            logger.warning(f"Basic monitoring registration failed: {e}")

    register_module_at_level(
        registrar, DependencyLevel.MONITORING_BASIC, register_monitoring_basic_module
    )

    # Level 3: State Services
    def register_state_module(injector: DependencyInjector) -> None:
        """Register state services with lazy loading."""
        try:
            from src.state.di_registration import register_state_services

            register_state_services(injector.get_container())
            logger.debug("State services registered successfully")
        except Exception as e:
            logger.warning(f"State registration failed: {e}")

    register_module_at_level(registrar, DependencyLevel.STATE_BASIC, register_state_module)

    # Level 3: Data Services
    def register_data_module(injector: DependencyInjector) -> None:
        """Register data services with lazy loading."""
        try:
            from src.data.di_registration import register_data_services

            register_data_services(injector)
            logger.debug("Data services registered successfully")
        except Exception as e:
            logger.warning(f"Data registration failed: {e}")

    register_module_at_level(registrar, DependencyLevel.DATA_BASIC, register_data_module)

    # Level 3: Exchange Services
    def register_exchange_module(injector: DependencyInjector) -> None:
        """Register exchange services."""
        try:
            from src.exchanges.di_registration import register_exchange_services

            register_exchange_services(injector.get_container())
            logger.debug("Exchange services registered successfully")
        except Exception as e:
            logger.warning(f"Exchange registration failed: {e}")

    register_module_at_level(registrar, DependencyLevel.EXCHANGE_BASIC, register_exchange_module)

    # Level 4: Risk Management
    def register_risk_module(injector: DependencyInjector) -> None:
        """Register risk management services."""
        try:
            from src.risk_management.di_registration import register_risk_management_services

            register_risk_management_services(injector.get_container())
            logger.debug("Risk management services registered successfully")
        except Exception as e:
            logger.warning(f"Risk management registration failed: {e}")

    register_module_at_level(registrar, DependencyLevel.RISK_MANAGEMENT, register_risk_module)

    # Level 4: Capital Management
    def register_capital_module(injector: DependencyInjector) -> None:
        """Register capital management services."""
        try:
            from src.capital_management.di_registration import register_capital_management_services

            register_capital_management_services(injector.get_container())
            logger.debug("Capital management services registered successfully")
        except Exception as e:
            logger.warning(f"Capital management registration failed: {e}")

    register_module_at_level(registrar, DependencyLevel.CAPITAL_MANAGEMENT, register_capital_module)

    # Level 4: ML Services
    def register_ml_module(injector: DependencyInjector) -> None:
        """Register ML services."""
        try:
            from src.ml.di_registration import register_ml_services

            register_ml_services(injector)
            logger.debug("ML services registered successfully")
        except Exception as e:
            logger.warning(f"ML registration failed: {e}")

    register_module_at_level(registrar, DependencyLevel.ML_SERVICES, register_ml_module)

    # Level 5: Execution Services
    def register_execution_module(injector: DependencyInjector) -> None:
        """Register execution services."""
        try:
            from src.execution.di_registration import register_execution_module

            register_execution_module(injector.get_container(), config)
            logger.debug("Execution services registered successfully")
        except Exception as e:
            logger.warning(f"Execution registration failed: {e}")

    register_module_at_level(registrar, DependencyLevel.EXECUTION, register_execution_module)

    # Level 5: Strategies
    def register_strategies_module(injector: DependencyInjector) -> None:
        """Register strategy services."""
        try:
            from src.strategies.di_registration import register_strategies_dependencies

            register_strategies_dependencies(injector.get_container())
            logger.debug("Strategy services registered successfully")
        except Exception as e:
            logger.warning(f"Strategy registration failed: {e}")

    register_module_at_level(registrar, DependencyLevel.STRATEGIES, register_strategies_module)

    # Level 6: Bot Management
    def register_bot_management_module(injector: DependencyInjector) -> None:
        """Register bot management services."""
        try:
            from src.bot_management.di_registration import register_bot_management_services

            register_bot_management_services(injector)
            logger.debug("Bot management services registered successfully")
        except Exception as e:
            logger.warning(f"Bot management registration failed: {e}")

    register_module_at_level(
        registrar, DependencyLevel.BOT_MANAGEMENT, register_bot_management_module
    )

    # Level 2: Error Handling (Basic) - Must be before exchanges to provide SecurityRateLimiter
    def register_error_handling_module(injector: DependencyInjector) -> None:
        """Register error handling services."""
        try:
            from src.error_handling.di_registration import register_error_handling_services

            register_error_handling_services(injector, config)
            logger.debug("Error handling services registered successfully")
        except Exception as e:
            logger.warning(f"Error handling registration failed: {e}")

    register_module_at_level(
        registrar, DependencyLevel.ERROR_HANDLING_BASIC, register_error_handling_module
    )

    # Level 7: Analytics
    def register_analytics_module(injector: DependencyInjector) -> None:
        """Register analytics services."""
        try:
            from src.analytics.di_registration import register_analytics_services

            register_analytics_services(injector.get_container())
            logger.debug("Analytics services registered successfully")
        except Exception as e:
            logger.warning(f"Analytics registration failed: {e}")

    register_module_at_level(registrar, DependencyLevel.ANALYTICS, register_analytics_module)

    # Level 7: Web Interface
    def register_web_interface_module(injector: DependencyInjector) -> None:
        """Register web interface services."""
        try:
            from src.web_interface.di_registration import register_web_interface_services

            register_web_interface_services(injector.get_container())
            logger.debug("Web interface services registered successfully")
        except Exception as e:
            logger.warning(f"Web interface registration failed: {e}")

    register_module_at_level(
        registrar, DependencyLevel.WEB_INTERFACE, register_web_interface_module
    )

    # Add lazy configurations for cross-service dependencies
    def configure_cross_service_dependencies(injector: DependencyInjector) -> None:
        """Configure cross-service dependencies after all services are registered."""
        logger.debug("Configuring cross-service dependencies")

        # Services that support dependency configuration
        configurable_services = [
            "StateService",
            "ExecutionService",
            "RiskService",
            "ErrorHandlingService",
            "BotService",
        ]

        for service_name in configurable_services:
            try:
                service = injector.get_optional(service_name)
                if service and hasattr(service, "configure_dependencies"):
                    service.configure_dependencies(injector)
                    logger.debug(f"Configured dependencies for {service_name}")
            except Exception as e:
                logger.debug(f"Could not configure dependencies for {service_name}: {e}")

    register_lazy_configuration(registrar, configure_cross_service_dependencies)


def get_master_injector() -> DependencyInjector:
    """
    Get a fully configured master injector with all services registered.

    Returns:
        Configured dependency injector
    """
    return register_all_services()


if __name__ == "__main__":
    # Test the master registration
    injector = register_all_services()
    logger.info("Master DI registration test completed successfully")
