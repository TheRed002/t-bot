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
        injector: The dependency injection container
        config: Optional configuration to pass to services
    """
    logger.info("Registering error handling services with DI container")

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

        injector.register_factory("SecuritySanitizer", security_sanitizer_factory, singleton=True)
        injector.register_factory(
            "SecurityRateLimiter", security_rate_limiter_factory, singleton=True
        )
        logger.debug("Registered security components")

        # Register ErrorHandler as singleton with proper error handling
        def error_handler_factory():
            try:
                from src.error_handling.error_handler import ErrorHandler

                resolved_config = (
                    injector.resolve("Config")
                    if injector.has_service("Config")
                    else config or Config()
                )

                # Try to resolve security components
                sanitizer = None
                rate_limiter = None

                try:
                    sanitizer = injector.resolve("SecuritySanitizer")
                except Exception as e:
                    logger.debug(f"Failed to resolve SecuritySanitizer: {e}")

                try:
                    rate_limiter = injector.resolve("SecurityRateLimiter")
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

        injector.register_factory("ErrorHandler", error_handler_factory, singleton=True)
        logger.debug("Registered ErrorHandler")

        # Register ErrorContextFactory using simplified factory pattern
        def context_factory_factory():
            try:
                from src.error_handling.context import ErrorContextFactory

                # Use service locator pattern for factory creation
                factory = ErrorContextFactory(dependency_container=injector)
                return factory
            except Exception as e:
                logger.warning(f"Failed to create ErrorContextFactory with DI: {e}")
                # Fallback to basic factory without dependency injection
                try:
                    from src.error_handling.context import ErrorContextFactory

                    return ErrorContextFactory()
                except Exception as fallback_error:
                    logger.error(f"Failed to create ErrorContextFactory fallback: {fallback_error}")
                    raise

        injector.register_factory("ErrorContextFactory", context_factory_factory, singleton=True)
        logger.debug("Registered ErrorContextFactory with simplified factory pattern")

        # Register GlobalErrorHandler as singleton with dependency injection
        def global_handler_factory():
            try:
                from src.error_handling.global_handler import GlobalErrorHandler

                resolved_config = (
                    injector.resolve("Config")
                    if injector.has_service("Config")
                    else config or Config()
                )
                context_factory = (
                    injector.resolve("ErrorContextFactory")
                    if injector.has_service("ErrorContextFactory")
                    else None
                )
                instance = GlobalErrorHandler(resolved_config, context_factory)

                # Don't configure dependencies during factory creation to avoid circular deps
                # Dependencies will be configured after all services are registered

                return instance
            except Exception as e:
                logger.error(f"Failed to create GlobalErrorHandler: {e}")
                raise

        injector.register_factory("GlobalErrorHandler", global_handler_factory, singleton=True)
        logger.debug("Registered GlobalErrorHandler")

        # Register ErrorPatternAnalytics as singleton
        def pattern_analytics_factory():
            try:
                from src.error_handling.pattern_analytics import ErrorPatternAnalytics

                resolved_config = (
                    injector.resolve("Config")
                    if injector.has_service("Config")
                    else config or Config()
                )
                instance = ErrorPatternAnalytics(resolved_config)

                # Don't configure dependencies during factory creation to avoid circular deps
                # Dependencies will be configured after all services are registered

                return instance
            except Exception as e:
                logger.error(f"Failed to create ErrorPatternAnalytics: {e}")
                raise

        injector.register_factory(
            "ErrorPatternAnalytics", pattern_analytics_factory, singleton=True
        )
        logger.debug("Registered ErrorPatternAnalytics")

        # Register StateMonitor as singleton - this depends on other services
        def state_monitor_factory():
            try:
                from src.error_handling.state_monitor import StateMonitor

                resolved_config = (
                    injector.resolve("Config")
                    if injector.has_service("Config")
                    else config or Config()
                )

                # Try to resolve dependencies - these are optional for StateMonitor
                database_service = None
                risk_service = None
                execution_service = None

                try:
                    database_service = (
                        injector.resolve("DatabaseService")
                        if injector.has_service("DatabaseService")
                        else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve DatabaseService for StateMonitor: {e}")

                try:
                    risk_service = (
                        injector.resolve("RiskService")
                        if injector.has_service("RiskService")
                        else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve RiskService for StateMonitor: {e}")

                try:
                    execution_service = (
                        injector.resolve("ExecutionService")
                        if injector.has_service("ExecutionService")
                        else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve ExecutionService for StateMonitor: {e}")

                instance = StateMonitor(
                    config=resolved_config,
                    database_service=database_service,
                    risk_service=risk_service,
                    execution_service=execution_service,
                )

                # Don't configure dependencies during factory creation to avoid circular deps
                # Dependencies will be configured after all services are registered

                return instance
            except Exception as e:
                logger.error(f"Failed to create StateMonitor: {e}")
                raise

        injector.register_factory("StateMonitor", state_monitor_factory, singleton=True)
        logger.debug("Registered StateMonitor")

        # Register ErrorHandlingService as singleton with all dependencies
        def service_factory():
            try:
                from src.error_handling.service import ErrorHandlingService

                resolved_config = (
                    injector.resolve("Config")
                    if injector.has_service("Config")
                    else config or Config()
                )

                # Resolve required dependencies - these should be available
                error_handler = None
                global_handler = None
                pattern_analytics = None
                state_monitor = None

                try:
                    error_handler = (
                        injector.resolve("ErrorHandler")
                        if injector.has_service("ErrorHandler")
                        else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve ErrorHandler: {e}")

                try:
                    global_handler = (
                        injector.resolve("GlobalErrorHandler")
                        if injector.has_service("GlobalErrorHandler")
                        else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve GlobalErrorHandler: {e}")

                try:
                    pattern_analytics = (
                        injector.resolve("ErrorPatternAnalytics")
                        if injector.has_service("ErrorPatternAnalytics")
                        else None
                    )
                except Exception as e:
                    logger.debug(f"Failed to resolve ErrorPatternAnalytics: {e}")

                try:
                    state_monitor = (
                        injector.resolve("StateMonitor")
                        if injector.has_service("StateMonitor")
                        else None
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

        injector.register_factory("ErrorHandlingService", service_factory, singleton=True)
        logger.debug("Registered ErrorHandlingService")

        # Register ErrorHandlingServiceAdapter as the primary service interface
        def service_adapter_factory():
            try:
                from src.error_handling.service_adapter import ErrorHandlingServiceAdapter

                resolved_config = (
                    injector.resolve("Config")
                    if injector.has_service("Config")
                    else config or Config()
                )

                error_handling_service = injector.resolve("ErrorHandlingService")
                
                return ErrorHandlingServiceAdapter(
                    config=resolved_config,
                    error_handling_service=error_handling_service,
                )
            except Exception as e:
                logger.error(f"Failed to create ErrorHandlingServiceAdapter: {e}")
                raise

        injector.register_factory("ErrorHandlingServiceAdapter", service_adapter_factory, singleton=True)
        logger.debug("Registered ErrorHandlingServiceAdapter")

        # Register ErrorHandlerFactory and configure handlers using simplified pattern
        try:
            from src.error_handling.factory import ErrorHandlerFactory

            # Use service locator pattern for factory registration
            def error_handler_factory_factory():
                try:
                    # Create factory with dependency injection using service locator
                    factory = ErrorHandlerFactory(dependency_container=injector)

                    # Register available handlers with the factory
                    _register_error_handlers(factory)

                    return factory
                except Exception as e:
                    logger.warning(f"Failed to create ErrorHandlerFactory with DI: {e}")
                    # Fallback to basic factory without DI
                    factory = ErrorHandlerFactory()
                    _register_error_handlers(factory)
                    return factory

            injector.register_factory(
                "ErrorHandlerFactory", error_handler_factory_factory, singleton=True
            )

            # Configure class-level container using service locator pattern
            try:
                ErrorHandlerFactory.set_dependency_container(injector)
                logger.debug("Registered ErrorHandlerFactory with service locator pattern")
            except Exception as e:
                logger.debug(f"Failed to set dependency container on factory: {e}")
                # Continue without class-level DI

        except Exception as e:
            logger.warning(f"Failed to register ErrorHandlerFactory: {e}")

        # Register ErrorHandlerChain factory using simplified pattern
        try:
            from src.error_handling.factory import ErrorHandlerChain

            def error_chain_factory():
                try:
                    # Use service locator pattern for chain creation
                    return ErrorHandlerChain.create_default_chain(dependency_container=injector)
                except Exception as e:
                    logger.warning(f"Failed to create default chain with DI: {e}")
                    # Fallback to basic chain without dependency injection
                    try:
                        return ErrorHandlerChain([], None)  # Empty chain as fallback
                    except Exception as fallback_error:
                        logger.error(f"Failed to create fallback chain: {fallback_error}")
                        # Return None to indicate factory failure
                        return None

            injector.register_factory("ErrorHandlerChain", error_chain_factory, singleton=False)
            logger.debug("Registered ErrorHandlerChain factory with service locator pattern")
        except Exception as e:
            logger.warning(f"Failed to register ErrorHandlerChain factory: {e}")

        logger.info("Successfully registered all error handling services and factories")

        # Configure dependencies after all services are registered to avoid circular deps
        try:
            _configure_service_dependencies(injector)
        except Exception as e:
            logger.warning(f"Failed to configure some service dependencies: {e}")

        # Set up global error handler instance
        try:
            _setup_global_error_handler(injector)
        except Exception as e:
            logger.warning(f"Failed to setup global error handler: {e}")

    except Exception as e:
        logger.error(f"Failed to register error handling services: {e}")
        raise


def _configure_service_dependencies(injector) -> None:
    """Configure service dependencies after all services are registered."""
    try:
        logger.debug("Configuring error handling service dependencies")

        # Configure dependencies in dependency order (avoid circular refs)
        service_names = [
            "ErrorHandler",  # First - has minimal dependencies
            "ErrorPatternAnalytics",  # Second - depends only on security components
            "ErrorContextFactory",  # Third - has minimal dependencies
            "GlobalErrorHandler",  # Fourth - depends on ErrorContextFactory
            "StateMonitor",  # Fifth - optional dependencies only
            "ErrorHandlingService",  # Sixth - depends on all core components
            "ErrorHandlingServiceAdapter",  # Last - depends on ErrorHandlingService
        ]

        configured_services = set()

        for service_name in service_names:
            try:
                if injector.has_service(service_name) and service_name not in configured_services:
                    service_instance = injector.resolve(service_name)
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
    return injector.resolve("ErrorHandlingService")


def _setup_global_error_handler(injector) -> None:
    """Set up the global error handler instance."""
    try:
        from src.error_handling import set_global_error_handler

        if injector.has_service("GlobalErrorHandler"):
            global_handler = injector.resolve("GlobalErrorHandler")
            set_global_error_handler(global_handler)
            logger.debug("Global error handler instance configured")
        else:
            logger.warning("GlobalErrorHandler service not available for global setup")
    except Exception as e:
        logger.error(f"Failed to setup global error handler: {e}")


def _register_error_handlers(factory) -> None:
    """Register all available error handlers with the factory using service locator pattern."""
    try:
        from src.error_handling.handlers import (
            DatabaseErrorHandler,
            DataValidationErrorHandler,
            NetworkErrorHandler,
            RateLimitErrorHandler,
            ValidationErrorHandler,
        )

        # Register handlers with proper dependency injection configurations
        handler_registry = {
            "network": {
                "class": NetworkErrorHandler,
                "config": {"max_retries": 3, "base_delay": 1.0}
            },
            "rate_limit": {
                "class": RateLimitErrorHandler,
                "config": {}
            },
            "database": {
                "class": DatabaseErrorHandler,
                "config": {}
            },
            "validation": {
                "class": ValidationErrorHandler,
                "config": {}
            },
            "data_validation": {
                "class": DataValidationErrorHandler,
                "config": {}
            },
        }

        for handler_name, handler_info in handler_registry.items():
            try:
                factory.register(handler_name, handler_info["class"], handler_info["config"])
                logger.debug(f"Registered error handler: {handler_name}")
            except Exception as e:
                logger.warning(f"Failed to register error handler '{handler_name}': {e}")
                # Continue with other handlers

    except Exception as e:
        logger.warning(f"Failed to register some error handlers: {e}")


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
