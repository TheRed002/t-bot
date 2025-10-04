"""
Dependency injection registration for error handling services.

This module provides functions to register all error handling services
with the dependency injection container, ensuring proper lifetime management
and avoiding circular dependencies.
"""

from typing import Any

from src.core.config import Config
from src.core.logging import get_logger

logger = get_logger(__name__)


def register_error_handling_services(injector, config: Config | None = None) -> None:
    """
    Register all error handling services with the dependency injection container.

    Args:
        injector: The dependency injection container (DependencyInjector or DependencyContainer)
        config: Optional configuration to pass to services
    """
    logger.info("Registering error handling services with DI container")

    # Determine if injector is DependencyInjector or DependencyContainer
    # DependencyInjector has register_factory, DependencyContainer has register
    has_register_factory = hasattr(injector, 'register_factory')

    # Get the actual container for service resolution
    # DependencyInjector.get_container() returns DependencyContainer
    # DependencyContainer is passed directly
    container = injector.get_container() if has_register_factory else injector

    def _register_service(name: str, factory, singleton: bool = True):
        """Helper to register with correct method based on container type."""
        if has_register_factory:
            injector.register_factory(name, factory, singleton=singleton)
        else:
            # DependencyContainer uses register() directly
            injector.register(name, factory, singleton=singleton)

    try:
        # Register SecuritySanitizer and SecurityRateLimiter first to avoid circular deps

        def security_sanitizer_factory():
            try:
                from src.error_handling.security_sanitizer import get_security_sanitizer

                return get_security_sanitizer()
            except Exception as e:
                logger.error(f"Failed to create SecuritySanitizer: {e}")
                raise

        def security_rate_limiter_factory():
            try:
                from src.error_handling.security_rate_limiter import get_security_rate_limiter

                return get_security_rate_limiter()
            except Exception as e:
                logger.error(f"Failed to create SecurityRateLimiter: {e}")
                raise

        _register_service("SecuritySanitizer", security_sanitizer_factory, singleton=True)
        _register_service("SecurityRateLimiter", security_rate_limiter_factory, singleton=True)
        logger.debug("Registered security components")

        # Register ErrorHandler as singleton with proper error handling

        def error_handler_factory():
            try:
                from src.error_handling.error_handler import ErrorHandler

                resolved_config = (
                    container.get("Config") if container.has("Config") else config or Config()
                )

                # Try to resolve security components
                sanitizer = None
                rate_limiter = None

                try:
                    sanitizer = container.get("SecuritySanitizer")
                except Exception as e:
                    logger.debug(f"Failed to resolve SecuritySanitizer: {e}")

                try:
                    rate_limiter = container.get("SecurityRateLimiter")
                except Exception as e:
                    logger.debug(f"Failed to resolve SecurityRateLimiter: {e}")

                instance = ErrorHandler(
                    resolved_config, sanitizer=sanitizer, rate_limiter=rate_limiter
                )

                # Don't configure dependencies during factory creation to avoid circular deps
                # Dependencies will be configured after all services are registered

                return instance
            except Exception as e:
                logger.error(f"Failed to create ErrorHandler: {e}")
                raise

        _register_service("ErrorHandler", error_handler_factory, singleton=True)
        logger.debug("Registered ErrorHandler")

        # Register ErrorContextFactory using simplified factory pattern

        def context_factory_factory():
            try:
                from src.error_handling.context import ErrorContextFactory

                # Use dependency injection for factory creation
                return ErrorContextFactory(dependency_container=injector)
            except Exception as e:
                logger.warning(f"Failed to create ErrorContextFactory with DI: {e}")
                # Fallback to basic factory
                try:
                    from src.error_handling.context import ErrorContextFactory

                    return ErrorContextFactory()
                except Exception as fallback_error:
                    error_msg = f"Failed to create ErrorContextFactory: {fallback_error}"
                    logger.error(error_msg)
                    raise

        _register_service("ErrorContextFactory", context_factory_factory, singleton=True)
        logger.debug("Registered ErrorContextFactory with simplified factory pattern")

        # Register GlobalErrorHandler as singleton with dependency injection

        def global_handler_factory():
            try:
                from src.error_handling.global_handler import GlobalErrorHandler

                resolved_config = (
                    container.get("Config") if container.has("Config") else config or Config()
                )
                context_factory = (
                    container.get("ErrorContextFactory")
                    if container.has("ErrorContextFactory")
                    else None
                )
                instance = GlobalErrorHandler(resolved_config, context_factory)

                # Don't configure dependencies during factory creation to avoid circular deps
                # Dependencies will be configured after all services are registered

                return instance
            except Exception as e:
                logger.error(f"Failed to create GlobalErrorHandler: {e}")
                raise

        _register_service("GlobalErrorHandler", global_handler_factory, singleton=True)
        logger.debug("Registered GlobalErrorHandler")

        # Register ErrorPatternAnalytics as singleton

        def pattern_analytics_factory():
            try:
                from src.error_handling.pattern_analytics import ErrorPatternAnalytics

                resolved_config = (
                    container.get("Config") if container.has("Config") else config or Config()
                )
                instance = ErrorPatternAnalytics(resolved_config)

                # Don't configure dependencies during factory creation to avoid circular deps
                # Dependencies will be configured after all services are registered

                return instance
            except Exception as e:
                logger.error(f"Failed to create ErrorPatternAnalytics: {e}")
                raise

        _register_service("ErrorPatternAnalytics", pattern_analytics_factory, singleton=True)
        logger.debug("Registered ErrorPatternAnalytics")

        # Register StateMonitor as singleton - this depends on other services

        def state_monitor_factory():
            try:
                from src.error_handling.state_monitor import StateMonitor

                resolved_config = (
                    container.get("Config") if container.has("Config") else config or Config()
                )

                # Try to resolve dependencies - these are optional for StateMonitor
                state_data_service = None
                risk_service = None
                execution_service = None

                try:
                    state_data_service = (
                        container.get("StateDataService") if container.has("StateDataService") else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve StateDataService for StateMonitor: {e}")

                try:
                    risk_service = (
                        container.get("RiskService") if container.has("RiskService") else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve RiskService for StateMonitor: {e}")

                try:
                    execution_service = (
                        container.get("ExecutionService")
                        if container.has("ExecutionService")
                        else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve ExecutionService for StateMonitor: {e}")

                instance = StateMonitor(
                    config=resolved_config,
                    state_data_service=state_data_service,
                    risk_service=risk_service,
                    execution_service=execution_service,
                )

                # Don't configure dependencies during factory creation to avoid circular deps
                # Dependencies will be configured after all services are registered

                return instance
            except Exception as e:
                logger.error(f"Failed to create StateMonitor: {e}")
                raise

        _register_service("StateMonitor", state_monitor_factory, singleton=True)
        logger.debug("Registered StateMonitor")

        # Register ErrorHandlingService as singleton with all dependencies

        def service_factory():
            try:
                from src.error_handling.service import ErrorHandlingService

                resolved_config = (
                    container.get("Config") if container.has("Config") else config or Config()
                )

                # Resolve required dependencies - these should be available
                error_handler = None
                global_handler = None
                pattern_analytics = None
                state_monitor = None

                try:
                    error_handler = (
                        container.get("ErrorHandler") if container.has("ErrorHandler") else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve ErrorHandler: {e}")

                try:
                    global_handler = (
                        container.get("GlobalErrorHandler")
                        if container.has("GlobalErrorHandler")
                        else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve GlobalErrorHandler: {e}")

                try:
                    pattern_analytics = (
                        container.get("ErrorPatternAnalytics")
                        if container.has("ErrorPatternAnalytics")
                        else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve ErrorPatternAnalytics: {e}")

                try:
                    state_monitor = (
                        container.get("StateMonitor") if container.has("StateMonitor") else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve StateMonitor: {e}")

                instance = ErrorHandlingService(
                    config=resolved_config,
                    error_handler=error_handler,
                    global_handler=global_handler,
                    pattern_analytics=pattern_analytics,
                    state_monitor=state_monitor,
                )

                # Don't configure dependencies during factory creation to avoid circular deps
                # Dependencies will be configured after all services are registered

                return instance
            except Exception as e:
                logger.error(f"Failed to create ErrorHandlingService: {e}")
                raise

        _register_service("ErrorHandlingService", service_factory, singleton=True)
        logger.debug("Registered ErrorHandlingService")

        # Register ErrorHandlerFactory using simplified pattern
        try:
            from src.error_handling.factory import ErrorHandlerFactory

            def error_handler_factory_factory():
                try:
                    # Create factory with dependency injection
                    factory = ErrorHandlerFactory(dependency_container=injector)
                    _register_error_handlers(factory)
                    return factory
                except Exception as e:
                    logger.warning(f"Failed to create ErrorHandlerFactory with DI: {e}")
                    # Fallback to basic factory
                    factory = ErrorHandlerFactory()
                    _register_error_handlers(factory)
                    return factory

            _register_service("ErrorHandlerFactory", error_handler_factory_factory, singleton=True)

            # Set class-level dependency container
            ErrorHandlerFactory.set_dependency_container(injector)
            logger.debug("Registered ErrorHandlerFactory with dependency injection")

        except Exception as e:
            logger.warning(f"Failed to register ErrorHandlerFactory: {e}")

        # Register ErrorHandlerChain factory using simplified pattern
        try:
            from src.error_handling.factory import ErrorHandlerChain

            def error_chain_factory():
                try:
                    return ErrorHandlerChain.create_default_chain(dependency_container=injector)
                except Exception as e:
                    logger.warning(f"Failed to create default chain with DI: {e}")
                    # Fallback to empty chain
                    return ErrorHandlerChain([], None)

            _register_service("ErrorHandlerChain", error_chain_factory, singleton=False)
            logger.debug("Registered ErrorHandlerChain factory with dependency injection")
        except Exception as e:
            logger.warning(f"Failed to register ErrorHandlerChain factory: {e}")

        logger.info("Successfully registered all error handling services and factories")

        # Configure dependencies after all services are registered to avoid circular deps
        _configure_service_dependencies(injector)

        # Set up global error handler instance
        _setup_global_error_handler(injector)

    except Exception as e:
        logger.error(f"Failed to register error handling services: {e}")
        raise


def _configure_service_dependencies(injector) -> None:
    """Configure service dependencies after all services are registered."""
    try:
        logger.debug("Configuring error handling service dependencies")

        # Get container for service resolution
        container = injector.get_container() if hasattr(injector, 'get_container') else injector

        # Configure dependencies in dependency order (avoid circular refs)
        service_names = [
            "ErrorHandler",  # First - has minimal dependencies
            "ErrorPatternAnalytics",  # Second - depends only on security components
            "ErrorContextFactory",  # Third - has minimal dependencies
            "GlobalErrorHandler",  # Fourth - depends on ErrorContextFactory
            "StateMonitor",  # Fifth - optional dependencies only
            "ErrorHandlingService",  # Sixth - depends on all core components
        ]

        configured_services = set()

        for service_name in service_names:
            try:
                if container.has(service_name) and service_name not in configured_services:
                    service_instance = container.get(service_name)
                    if hasattr(service_instance, "configure_dependencies"):
                        service_instance.configure_dependencies(injector)
                        logger.debug(f"Configured dependencies for {service_name}")
                        configured_services.add(service_name)
                    else:
                        logger.debug(
                            f"Service {service_name} doesn't support dependency configuration"
                        )
                        configured_services.add(service_name)
            except Exception as e:
                logger.warning(f"Failed to configure dependencies for {service_name}: {e}")
                # Continue with other services

        logger.debug(
            f"Error handling service dependencies configured successfully: {configured_services}"
        )

    except Exception as e:
        logger.error(f"Failed to configure service dependencies: {e}")


def get_error_handling_service(injector) -> Any:
    """
    Get the ErrorHandlingService from the DI container.

    Args:
        injector: The dependency injection container

    Returns:
        ErrorHandlingService instance
    """
    return injector.get("ErrorHandlingService")


def _setup_global_error_handler(injector) -> None:
    """Set up the global error handler instance."""
    try:
        from src.error_handling import set_global_error_handler

        # Get container for service resolution
        container = injector.get_container() if hasattr(injector, 'get_container') else injector

        if container.has("GlobalErrorHandler"):
            global_handler = container.get("GlobalErrorHandler")
            set_global_error_handler(global_handler)
            logger.debug("Global error handler instance configured")
        else:
            logger.warning("GlobalErrorHandler service not available for global setup")
    except Exception as e:
        logger.error(f"Failed to setup global error handler: {e}")


def _register_error_handlers(factory) -> None:
    """Register all available error handlers with the factory."""
    try:
        from src.error_handling.handlers import (
            DatabaseErrorHandler,
            DataValidationErrorHandler,
            NetworkErrorHandler,
            RateLimitErrorHandler,
            ValidationErrorHandler,
        )

        # Register handlers with configurations
        handlers = [
            ("network", NetworkErrorHandler, {"max_retries": 3, "base_delay": 1.0}),
            ("rate_limit", RateLimitErrorHandler, {}),
            ("database", DatabaseErrorHandler, {}),
            ("validation", ValidationErrorHandler, {}),
            ("data_validation", DataValidationErrorHandler, {}),
        ]

        for handler_name, handler_class, config in handlers:
            try:
                factory.register(handler_name, handler_class, config)
                logger.debug(f"Registered error handler: {handler_name}")
            except Exception as e:
                logger.warning(f"Failed to register error handler '{handler_name}': {e}")

    except Exception as e:
        logger.warning(f"Failed to register error handlers: {e}")


def configure_error_handling_di(injector, config: Config | None = None) -> None:
    """
    Configure dependency injection for the entire error handling module.

    This function should be called during application startup to register
    all error handling services with the DI container.

    Args:
        injector: The dependency injection container
        config: Optional configuration to pass to services
    """
    try:
        register_error_handling_services(injector, config)
        logger.info("Error handling dependency injection configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure error handling DI: {e}")
        raise
