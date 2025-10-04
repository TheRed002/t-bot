"""
Dependency ordering system to prevent circular dependencies during DI registration.

This module provides a structured approach to registering services in the correct order
to avoid circular dependency issues during DI container setup.
"""

from typing import Any, Dict, List, Optional, Callable
from enum import IntEnum
from src.core.logging import get_logger
from src.core.dependency_injection import DependencyInjector

logger = get_logger(__name__)


class DependencyLevel(IntEnum):
    """
    Dependency levels for ordering service registration.

    Lower numbers are registered first, higher numbers depend on lower levels.
    """

    # Level 0: Foundation - No dependencies
    CORE_CONFIG = 0  # Configuration, logging, basic utilities
    CORE_TYPES = 1  # Type definitions, enums, constants

    # Level 1: Infrastructure - Minimal dependencies
    DATABASE = 10  # Database connections, basic repositories
    CACHE = 11  # Redis, memory cache
    SECURITY = 12  # Authentication, encryption, basic security

    # Level 2: Base Services - Infrastructure dependencies only
    VALIDATION = 20  # Input validation, schema validation
    MONITORING_BASIC = 21  # Basic metrics, logging
    ERROR_HANDLING_BASIC = 22  # Basic error handling, no complex dependencies

    # Level 3: Core Services - Base service dependencies
    STATE_BASIC = 30  # Basic state management
    DATA_BASIC = 31  # Basic data services
    EXCHANGE_BASIC = 32  # Basic exchange connections

    # Level 4: Business Services - Core service dependencies
    RISK_MANAGEMENT = 40  # Risk calculation, position sizing
    CAPITAL_MANAGEMENT = 41  # Capital allocation, fund management
    ML_SERVICES = 42  # ML models, predictions

    # Level 5: Execution Services - Business service dependencies
    EXECUTION = 50  # Order execution, trade management
    STRATEGIES = 51  # Trading strategies

    # Level 6: Orchestration - Execution service dependencies
    BOT_MANAGEMENT = 60  # Bot coordination, lifecycle
    MONITORING_FULL = 61  # Full monitoring with all dependencies
    ERROR_HANDLING_FULL = 62  # Full error handling with all dependencies

    # Level 7: Application - All other dependencies
    WEB_INTERFACE = 70  # Web UI, APIs
    ANALYTICS = 71  # Analysis, reporting
    BACKTESTING = 72  # Backtesting engine


class DependencyRegistrar:
    """
    Manages ordered dependency registration to prevent circular dependencies.
    """

    def __init__(self, injector: DependencyInjector):
        """
        Initialize dependency registrar.

        Args:
            injector: Dependency injector instance
        """
        self.injector = injector
        self.registration_functions: Dict[DependencyLevel, List[Callable]] = {}
        self.lazy_configurations: List[Callable] = []
        self.logger = logger

    def register_at_level(self, level: DependencyLevel, registration_func: Callable) -> None:
        """
        Register a function to be called at a specific dependency level.

        Args:
            level: Dependency level for registration
            registration_func: Function to call for registration
        """
        if level not in self.registration_functions:
            self.registration_functions[level] = []
        self.registration_functions[level].append(registration_func)

    def add_lazy_configuration(self, config_func: Callable) -> None:
        """
        Add a lazy configuration function to be executed after all registrations.

        Args:
            config_func: Configuration function to execute later
        """
        self.lazy_configurations.append(config_func)

    def register_all(self) -> None:
        """
        Register all services in dependency order.
        """
        self.logger.info("Starting ordered dependency registration")

        # Register services level by level
        for level in sorted(DependencyLevel):
            if level in self.registration_functions:
                self.logger.debug(f"Registering services at level {level.name}")

                for registration_func in self.registration_functions[level]:
                    try:
                        registration_func(self.injector)
                        self.logger.debug(f"Successfully registered function at level {level.name}")
                    except Exception as e:
                        self.logger.error(f"Failed to register at level {level.name}: {e}")
                        # Continue with other registrations

        # Execute lazy configurations after all registrations
        self.logger.debug("Executing lazy configurations")
        for config_func in self.lazy_configurations:
            try:
                config_func(self.injector)
            except Exception as e:
                self.logger.warning(f"Failed to execute lazy configuration: {e}")

        # Execute any container-level lazy configurations
        self.injector.execute_lazy_configurations()

        self.logger.info("Completed ordered dependency registration")


def create_ordered_registrar(injector: Optional[DependencyInjector] = None) -> DependencyRegistrar:
    """
    Create a dependency registrar with standard ordering.

    Args:
        injector: Optional injector instance

    Returns:
        Configured dependency registrar
    """
    if injector is None:
        injector = DependencyInjector()

    registrar = DependencyRegistrar(injector)

    # Register core dependency functions
    _register_core_dependencies(registrar)

    return registrar


def _register_core_dependencies(registrar: DependencyRegistrar) -> None:
    """
    Register core dependency registration functions.

    Args:
        registrar: Dependency registrar instance
    """

    # Level 0: Core Configuration
    def register_core_config(injector: DependencyInjector) -> None:
        """Register core configuration services."""
        from src.core.config import Config

        if not injector.has_service("Config"):
            injector.register_singleton("Config", Config())

    registrar.register_at_level(DependencyLevel.CORE_CONFIG, register_core_config)

    # Level 1: Database (Infrastructure)
    def register_database_lazy(injector: DependencyInjector) -> None:
        """Register database services with lazy loading."""
        from src.database.di_registration import register_database_services

        try:
            register_database_services(injector.get_container())
        except Exception as e:
            logger.warning(f"Database registration failed: {e}")

    registrar.register_at_level(DependencyLevel.DATABASE, register_database_lazy)

    # Level 2: Validation (Base Services)
    def register_validation_lazy(injector: DependencyInjector) -> None:
        """Register validation services with lazy loading."""

        def validation_service_factory():
            from src.utils.validation.service import ValidationService

            return ValidationService()

        injector.register_factory("ValidationService", validation_service_factory, singleton=True)

    registrar.register_at_level(DependencyLevel.VALIDATION, register_validation_lazy)

    # Add lazy configuration for cross-service dependencies
    def configure_cross_dependencies(injector: DependencyInjector) -> None:
        """Configure cross-service dependencies after all services are registered."""
        logger.debug("Configuring cross-service dependencies")

        # Configure services that support dependency configuration
        for service_name in ["StateService", "ExecutionService", "RiskService"]:
            try:
                service = injector.get_optional(service_name)
                if service and hasattr(service, "configure_dependencies"):
                    service.configure_dependencies(injector)
            except Exception as e:
                logger.debug(f"Could not configure dependencies for {service_name}: {e}")

    registrar.add_lazy_configuration(configure_cross_dependencies)


# Convenience functions for module registration
def register_module_at_level(
    registrar: DependencyRegistrar,
    level: DependencyLevel,
    registration_func: Callable[[DependencyInjector], None],
) -> None:
    """
    Register a module's services at a specific dependency level.

    Args:
        registrar: Dependency registrar instance
        level: Dependency level for registration
        registration_func: Module registration function
    """
    registrar.register_at_level(level, registration_func)


def register_lazy_configuration(
    registrar: DependencyRegistrar, config_func: Callable[[DependencyInjector], None]
) -> None:
    """
    Register a lazy configuration function.

    Args:
        registrar: Dependency registrar instance
        config_func: Configuration function
    """
    registrar.add_lazy_configuration(config_func)
