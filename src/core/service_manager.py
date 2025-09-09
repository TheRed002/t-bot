"""
Service Manager for centralized service management and dependency resolution.

This module provides a centralized service registry that eliminates circular
dependencies by managing service lifecycle and dependency injection.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from src.core.base.interfaces import DIContainer
from src.core.exceptions import DependencyError, ServiceError

logger = logging.getLogger(__name__)


class ServiceManager:
    """
    Centralized service manager for dependency resolution and lifecycle management.

    This manager eliminates circular dependencies by:
    1. Registering all services with the DI container
    2. Managing service startup order based on dependencies
    3. Providing centralized service access
    4. Handling service shutdown gracefully
    """

    def __init__(self, injector: DIContainer) -> None:
        """Initialize the service manager."""
        self._injector = injector
        self._services: dict[str, Any] = {}
        self._service_configs: dict[str, dict[str, Any]] = {}
        self._startup_order: list[str] = []
        self._running_services: set[str] = set()
        self._logger = logger
        self._initialized = False

    def register_service(
        self,
        service_name: str,
        service_class: type,
        config: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
        singleton: bool = True,
    ) -> None:
        """
        Register a service with the manager.

        Args:
            service_name: Unique service identifier
            service_class: Service class to instantiate
            config: Service configuration
            dependencies: List of service dependencies
            singleton: Whether service should be singleton
        """
        self._logger.info(f"Registering service: {service_name}")

        # Store service configuration
        self._service_configs[service_name] = {
            "class": service_class,
            "config": config or {},
            "dependencies": dependencies or [],
            "singleton": singleton,
        }

        # Register factory with DI container - factory should handle its own dependencies
        def service_factory():
            instance = self._create_service(service_name)
            # Configure dependencies if the instance supports it
            if hasattr(instance, "configure_dependencies"):
                try:
                    instance.configure_dependencies(self._injector)
                except Exception as e:
                    self._logger.warning(
                        f"Failed to configure dependencies for {service_name}: {e}"
                    )
            return instance

        try:
            self._injector.register_factory(service_name, service_factory, singleton=singleton)
        except Exception as e:
            self._logger.error(f"Failed to register service factory for {service_name}: {e}")
            raise ServiceError(f"Service registration failed: {service_name}") from e

    def _create_service(self, service_name: str) -> Any:
        """Create a service instance with resolved dependencies."""
        if service_name in self._services:
            return self._services[service_name]

        config = self._service_configs[service_name]
        self._logger.debug(f"Creating service: {service_name}")

        try:
            # Resolve dependencies
            try:
                resolved_deps = self._resolve_dependencies(service_name, config["dependencies"])
            except DependencyError as e:
                self._logger.error(f"Failed to resolve dependencies for {service_name}: {e}")
                raise ServiceError(f"Dependency resolution failed: {service_name}") from e

            # Create service instance
            service_instance = self._instantiate_service(config, resolved_deps)

            # Configure service
            self._configure_service_instance(service_instance)

            # Store service if singleton
            if config["singleton"]:
                self._services[service_name] = service_instance

            return service_instance

        except Exception as e:
            self._logger.error(f"Failed to create service {service_name}: {e}")
            raise ServiceError(f"Service creation failed: {service_name}") from e

    def _resolve_dependencies(self, service_name: str, dependencies: list[str]) -> dict[str, Any]:
        """Resolve service dependencies."""
        resolved_deps = {}
        for dep_name in dependencies:
            if dep_name not in self._service_configs:
                raise DependencyError(
                    f"Unknown dependency: {dep_name}",
                    dependency_name=dep_name,
                    error_code="DEP_001",
                    suggested_action="Register the dependency service first",
                    context={
                        "requesting_service": service_name,
                        "available_services": list(self._service_configs.keys()),
                    },
                )
            try:
                resolved_deps[dep_name] = self._injector.resolve(dep_name)
            except Exception as e:
                raise DependencyError(
                    f"Failed to resolve dependency '{dep_name}' for service '{service_name}'",
                    dependency_name=dep_name,
                    error_code="DEP_002",
                    suggested_action="Check dependency service registration and startup order",
                    context={"requesting_service": service_name, "original_error": str(e)},
                ) from e
        return resolved_deps

    def _instantiate_service(self, config: dict[str, Any], resolved_deps: dict[str, Any]) -> Any:
        """Instantiate service with proper constructor arguments."""
        service_class = config["class"]
        service_config = config["config"]

        if not hasattr(service_class, "__init__"):
            return service_class()

        constructor_args = self._build_constructor_args(
            service_class, service_config, resolved_deps
        )
        return service_class(**constructor_args)

    def _build_constructor_args(
        self, service_class: type, service_config: dict[str, Any], resolved_deps: dict[str, Any]
    ) -> dict[str, Any]:
        """Build constructor arguments based on service signature."""
        import inspect

        sig = inspect.signature(service_class.__init__)
        params = list(sig.parameters.keys())[1:]  # Skip 'self'

        constructor_args = {}
        param_mapping = self._get_parameter_mapping()

        # Map parameters to dependencies
        for param in params:
            if self._should_skip_parameter(param):
                continue

            if param in param_mapping:
                dep_name = param_mapping[param]
                if dep_name and dep_name in resolved_deps:
                    constructor_args[param] = resolved_deps[dep_name]
            elif param in resolved_deps:
                constructor_args[param] = resolved_deps[param]

        # Add service config if expected
        if "config" in params and "config" in service_config:
            constructor_args["config"] = service_config["config"]

        return constructor_args

    def _get_parameter_mapping(self) -> dict[str, str | None]:
        """Get mapping of parameter names to dependency names."""
        return {
            "DatabaseService": "DatabaseService",
            "StateService": "StateService",
            "ConfigService": "ConfigService",
            "ValidationService": "ValidationService",
            "Config": "Config",
            "correlation_id": None,  # Skip correlation_id
            # Legacy support (will be removed)
            "database_service": "DatabaseService",
            "state_service": "StateService",
            "config_service": "ConfigService",
            "validation_service": "ValidationService",
            "config": "Config",
        }

    def _should_skip_parameter(self, param: str) -> bool:
        """Check if parameter should be skipped during dependency injection."""
        skip_params = {"correlation_id"}
        return param in skip_params

    def _configure_service_instance(self, service_instance: Any) -> None:
        """Configure service instance after creation."""
        # Set dependency container if service supports it
        if hasattr(service_instance, "configure_dependencies"):
            try:
                service_instance.configure_dependencies(self._injector)
            except Exception as e:
                self._logger.warning(f"Failed to configure dependencies for service: {e}")
        elif hasattr(service_instance, "_dependency_container"):
            service_instance._dependency_container = self._injector

    def _calculate_startup_order(self) -> list[str]:
        """Calculate the correct service startup order based on dependencies."""
        # Topological sort of services based on dependencies
        visited = set()
        temp_visited = set()
        order = []

        def visit(service_name: str):
            if service_name in temp_visited:
                raise ServiceError(f"Circular dependency detected involving {service_name}")

            if service_name in visited:
                return

            temp_visited.add(service_name)

            # Visit dependencies first
            config = self._service_configs[service_name]
            for dep in config["dependencies"]:
                if dep in self._service_configs:
                    visit(dep)

            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)

        # Visit all services
        for service_name in self._service_configs:
            if service_name not in visited:
                visit(service_name)

        return order

    async def start_all_services(self) -> None:
        """Start all registered services in dependency order."""
        if self._initialized:
            self._logger.warning("Services already initialized")
            return

        try:
            # Calculate startup order
            self._startup_order = self._calculate_startup_order()
            self._logger.info(f"Service startup order: {self._startup_order}")

            # Start services in order
            for service_name in self._startup_order:
                await self._start_service(service_name)

            self._initialized = True
            self._logger.info("All services started successfully")

        except Exception as e:
            self._logger.error(f"Service startup failed: {e}")
            await self.stop_all_services()
            raise ServiceError(f"Service startup failed: {e}") from e

    async def _start_service(self, service_name: str) -> None:
        """Start a specific service."""
        if service_name in self._running_services:
            return

        self._logger.info(f"Starting service: {service_name}")

        try:
            # Get service instance (creates if needed)
            service = self._injector.resolve(service_name)

            # Start service if it has a start method
            if hasattr(service, "start"):
                await service.start()
            elif hasattr(service, "_do_start"):
                await service._do_start()

            self._running_services.add(service_name)
            self._logger.info(f"Service started: {service_name}")

        except Exception as e:
            self._logger.error(f"Failed to start service {service_name}: {e}")
            raise ServiceError(f"Failed to start {service_name}") from e

    async def stop_all_services(self) -> None:
        """Stop all running services in reverse order."""
        if not self._running_services:
            return

        self._logger.info("Stopping all services")

        # Stop in reverse order
        for service_name in reversed(self._startup_order):
            if service_name in self._running_services:
                await self._stop_service(service_name)

        # Clear all service references to prevent resource leaks
        try:
            self._running_services.clear()
            self._services.clear()
            self._service_configs.clear()
        except Exception as e:
            self._logger.warning(
                "Error clearing service references during shutdown (continuing)",
                error=str(e),
                error_type=type(e).__name__,
            )

        self._initialized = False
        self._logger.info("All services stopped")

    async def _stop_service(self, service_name: str) -> None:
        """Stop a specific service."""
        if service_name not in self._running_services:
            return

        self._logger.info(f"Stopping service: {service_name}")

        try:
            service = self._services.get(service_name)
            if service:
                # Stop service if it has a stop method
                if hasattr(service, "stop"):
                    await service.stop()
                elif hasattr(service, "_do_stop"):
                    await service._do_stop()

            self._running_services.discard(service_name)
            self._logger.info(f"Service stopped: {service_name}")

        except Exception as e:
            self._logger.error(
                "Failed to stop service (continuing shutdown)",
                service=service_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            # Continue stopping other services

    def get_service(self, service_name: str) -> Any:
        """
        Get a service instance by name.

        Args:
            service_name: Name of the service

        Returns:
            Service instance

        Raises:
            ServiceError: If service not found or not started
        """
        if not self._initialized:
            raise ServiceError("Services not initialized. Call start_all_services() first.")

        try:
            return self._injector.resolve(service_name)
        except KeyError:
            raise ServiceError(f"Service not found: {service_name}") from None

    def is_service_running(self, service_name: str) -> bool:
        """Check if a service is currently running."""
        return service_name in self._running_services

    def get_running_services(self) -> list[str]:
        """Get list of currently running services."""
        return list(self._running_services)

    async def restart_service(self, service_name: str) -> None:
        """Restart a specific service."""
        if service_name not in self._service_configs:
            raise ServiceError(f"Service not registered: {service_name}")

        self._logger.info(f"Restarting service: {service_name}")

        # Stop service
        if service_name in self._running_services:
            await self._stop_service(service_name)

        # Start service
        await self._start_service(service_name)

    async def health_check_all(self) -> dict[str, Any]:
        """Perform health check on all running services using consistent patterns."""
        health_status = {}

        # Use consistent batch processing for health checks with proper error handling
        health_tasks = []
        for service_name in self._running_services:
            task = asyncio.create_task(self._check_service_health(service_name))
            health_tasks.append((service_name, task))

        # Wait for all health checks with timeout and consistent error handling using gather
        if health_tasks:
            task_list = [task for _, task in health_tasks]
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*task_list, return_exceptions=True),
                    timeout=30.0
                )

                # Process results with proper error handling
                for i, (service_name, _) in enumerate(health_tasks):
                    result = results[i] if i < len(results) else None

                    if isinstance(result, Exception):
                        health_status[service_name] = {
                            "status": "error",
                            "error": str(result),
                            "error_type": type(result).__name__,
                            "timestamp": datetime.now().isoformat(),
                        }
                    elif result:
                        health_status[service_name] = self._normalize_health_status(result)
                    else:
                        health_status[service_name] = {
                            "status": "unknown",
                            "error": "No result returned",
                            "timestamp": datetime.now().isoformat(),
                        }

            except asyncio.TimeoutError:
                # Handle timeout for all tasks
                for service_name, task in health_tasks:
                    if not task.done():
                        task.cancel()
                    health_status[service_name] = {
                        "status": "timeout",
                        "error": "Health check timed out",
                        "timestamp": datetime.now().isoformat(),
                    }

        # Consistent status aggregation using core health status patterns
        overall_status = self._aggregate_health_status(health_status)

        return {
            "overall_status": overall_status,
            "services": health_status,
            "running_count": len(self._running_services),
            "total_registered": len(self._service_configs),
            "timestamp": datetime.now().isoformat(),
        }

    async def _check_service_health(self, service_name: str) -> dict[str, Any]:
        """Check health of a single service with consistent error handling patterns."""
        try:
            service = self._services.get(service_name)
            if service and hasattr(service, "health_check"):
                result = await service.health_check()
                # Apply consistent health check result transformation
                return self._normalize_health_status(result)
            elif service and hasattr(service, "get_health_status"):
                # Check for core health status method
                result = await service.get_health_status()
                return self._normalize_health_status(result)
            else:
                return {
                    "status": "running",
                    "details": "No health check method available",
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat(),
            }

    def _normalize_health_status(self, status_result: Any) -> dict[str, Any]:
        """Normalize health status to consistent format across all services."""
        base_result = {"timestamp": datetime.now().isoformat()}

        if isinstance(status_result, dict):
            base_result.update(status_result)
            # Ensure status field exists
            if "status" not in base_result:
                base_result["status"] = "unknown"
            return base_result

        # Handle HealthStatus enum from core.base.interfaces
        if hasattr(status_result, "value"):
            base_result["status"] = status_result.value.lower()
            return base_result

        # Handle string status
        if isinstance(status_result, str):
            base_result["status"] = status_result.lower()
            return base_result

        # Handle boolean (True = healthy, False = unhealthy)
        if isinstance(status_result, bool):
            base_result["status"] = "healthy" if status_result else "unhealthy"
            return base_result

        # Default case
        base_result["status"] = "unknown"
        base_result["details"] = str(status_result)
        return base_result

    def _aggregate_health_status(self, service_statuses: dict[str, dict[str, Any]]) -> str:
        """Aggregate individual service health statuses using consistent logic."""
        if not service_statuses:
            return "unknown"

        # Define status priority (higher number = worse status)
        status_priority = {
            "healthy": 1,
            "running": 2,
            "degraded": 3,
            "timeout": 4,
            "unhealthy": 5,
            "error": 6,
            "unknown": 7,
        }

        # Find the worst status across all services
        worst_priority = 0
        worst_status = "healthy"

        for service_status in service_statuses.values():
            status = service_status.get("status", "unknown")
            priority = status_priority.get(status, 7)
            if priority > worst_priority:
                worst_priority = priority
                worst_status = status

        return worst_status


# Global service manager instance - initialized lazily
_service_manager: ServiceManager | None = None


def get_service_manager(injector_instance: Any = None) -> ServiceManager:
    """Get the global service manager instance.
    
    Args:
        injector_instance: Optional injector instance to avoid circular dependency
    """
    global _service_manager
    if _service_manager is None:
        # Use provided injector or get from global state
        if injector_instance is None:
            from src.core.dependency_injection import get_global_injector
            injector_instance = get_global_injector()

        _service_manager = ServiceManager(injector_instance)

        # Register core infrastructure factories with DI container
        _register_core_infrastructure_factories(injector_instance, _service_manager)

    return _service_manager


def _register_core_infrastructure_factories(injector: Any, service_manager: ServiceManager) -> None:
    """Register core infrastructure factories to avoid circular dependencies."""
    try:
        from src.core.memory_manager import create_memory_manager_factory

        memory_factory = create_memory_manager_factory(config=None)
        injector.register_factory("MemoryManager", memory_factory, singleton=True)
    except ImportError as e:
        service_manager._logger.warning(f"MemoryManager factory not available: {e}")

    try:
        from src.core.caching.cache_manager import create_cache_manager_factory

        cache_factory = create_cache_manager_factory(config=None)
        injector.register_factory("CacheManager", cache_factory, singleton=True)
    except ImportError as e:
        service_manager._logger.warning(f"CacheManager factory not available: {e}")


def register_core_services(config: Any) -> None:
    """
    Register all core services with the service manager.

    This function sets up the proper dependency chain to avoid circular imports.
    """
    from src.core.dependency_injection import get_global_injector
    injector = get_global_injector()

    # Get service manager with injector to avoid circular dependency
    service_manager = get_service_manager(injector)

    # Register basic config first if provided
    if config:
        injector.register_singleton("Config", config)

    # Register core infrastructure services with minimal dependencies
    try:
        # Use factory pattern for ConfigService - SINGLETON (shared configuration state)
        def config_service_factory():
            from src.core.config.service import ConfigService
            # Use service locator pattern for dependency resolution
            try:
                # Try to resolve dependencies from container
                dependencies = {}
                # ConfigService has minimal dependencies, create with available config
                service = ConfigService()
                return service
            except Exception:
                # Fallback to direct creation
                return ConfigService()

        injector.register_factory("ConfigService", config_service_factory, singleton=True)

        # Also register interface for ConfigService if available
        try:
            from src.core.base.interfaces import ServiceComponent
            injector.register_interface(ServiceComponent, config_service_factory, singleton=True)
        except ImportError:
            pass  # Interface not available
    except ImportError:
        pass  # Service may not exist yet

    try:
        # Use factory pattern for ValidationService - SINGLETON (stateless validators, shared cache)
        def validation_service_factory():
            from src.utils.validation.service import ValidationService
            # Use service locator pattern for dependency resolution
            try:
                # ValidationService typically has minimal dependencies
                dependencies = {}
                # Try to resolve ConfigService if available
                try:
                    config_service = injector.resolve("ConfigService")
                    dependencies["config_service"] = config_service
                except Exception:
                    pass

                return ValidationService(**dependencies)
            except Exception:
                # Fallback to direct creation
                return ValidationService()

        injector.register_factory("ValidationService", validation_service_factory, singleton=True)
    except ImportError:
        pass  # Service may not exist yet

    try:
        # Use factory pattern for DatabaseService with proper dependency injection
        def database_service_factory():
            from src.database.service import DatabaseService

            # Use service locator pattern for dependency resolution
            try:
                dependencies = {}
                # Try to resolve common dependencies
                try:
                    dependencies["config_service"] = injector.resolve("ConfigService")
                except Exception:
                    if config:
                        dependencies["config"] = config

                try:
                    dependencies["error_service"] = injector.resolve("ErrorHandlingService")
                except Exception:
                    pass  # Optional dependency

                return DatabaseService(**dependencies)
            except Exception:
                # Fallback to direct creation with available config
                return DatabaseService(config=config)

        # DatabaseService - SINGLETON (connection pooling, shared database state)
        injector.register_factory("DatabaseService", database_service_factory, singleton=True)

        # Also register as ServiceComponent interface
        try:
            from src.core.base.interfaces import ServiceComponent
            injector.register_interface(ServiceComponent, database_service_factory, singleton=True)
        except ImportError:
            pass
    except ImportError:
        pass  # Service may not exist yet

    try:
        # Use factory pattern for StateService with proper dependency injection
        def state_service_factory():
            from src.state.state_service import StateService

            # Use service locator pattern for dependency resolution
            try:
                dependencies = {}
                # Try to resolve required dependencies
                try:
                    dependencies["database_service"] = injector.resolve("DatabaseService")
                except Exception:
                    pass  # Service will handle missing database

                try:
                    dependencies["config_service"] = injector.resolve("ConfigService")
                except Exception:
                    if config:
                        dependencies["config"] = config

                return StateService(**dependencies)
            except Exception:
                # Fallback to direct creation
                return StateService(config=config)

        # StateService - SINGLETON (shared state management)
        injector.register_factory("StateService", state_service_factory, singleton=True)
    except ImportError:
        pass  # Service may not exist yet

    # Register ErrorHandlingService as core infrastructure
    try:
        # Use factory pattern for ErrorHandlingService
        def error_service_factory():
            from src.error_handling.service import ErrorHandlingService

            # Use service locator pattern for dependency resolution
            try:
                dependencies = {}
                # Try to resolve ConfigService dependency
                try:
                    dependencies["config_service"] = injector.resolve("ConfigService")
                except Exception:
                    if config:
                        dependencies["config"] = config

                return ErrorHandlingService(**dependencies)
            except Exception:
                # Fallback to direct creation
                return ErrorHandlingService(config=config)

        # ErrorHandlingService - SINGLETON (shared error handling context and patterns)
        injector.register_factory("ErrorHandlingService", error_service_factory, singleton=True)

        # Also register error handling components with DI container
        try:
            from src.error_handling.di_registration import configure_error_handling_di
            configure_error_handling_di(injector, config)
        except ImportError:
            pass

    except ImportError:
        pass  # Service may not exist yet


def register_business_services(config: Any) -> None:
    """Register business logic services using factory patterns."""
    from src.core.dependency_injection import get_global_injector
    injector = get_global_injector()

    # Register business services with factory patterns and proper dependency injection
    business_service_configs = [
        ("CapitalService", "src.capital_management.service", ["DatabaseService", "ValidationService", "ErrorHandlingService"]),
        ("ExecutionService", "src.execution.service", ["DatabaseService", "ValidationService", "ErrorHandlingService"]),
        ("RiskService", "src.risk_management.service", ["DatabaseService", "StateService", "ValidationService", "ErrorHandlingService"]),
        ("StrategyService", "src.strategies.service", ["ValidationService", "ErrorHandlingService"]),
        ("MLService", "src.ml.service", ["DatabaseService", "ValidationService", "ErrorHandlingService"]),
        ("BotService", "src.bot_management.service", ["DatabaseService", "StateService", "RiskService", "ExecutionService", "StrategyService", "CapitalService", "ValidationService", "ErrorHandlingService"]),
    ]

    for service_name, module_path, deps in business_service_configs:
        try:
            def create_service_factory(svc_name, mod_path, dependencies):
                def service_factory():
                    module = __import__(mod_path, fromlist=[svc_name])
                    service_class = getattr(module, svc_name)

                    # Use service locator pattern for dependency resolution
                    try:
                        resolved_deps = {}
                        for dep_name in dependencies:
                            try:
                                # Try to resolve from dependency injector
                                dependency = injector.resolve(dep_name)
                                # Convert service name to parameter name (e.g., DatabaseService -> database_service)
                                param_name = dep_name.lower().replace("service", "") + ("_service" if dep_name.endswith("Service") else "")
                                resolved_deps[param_name] = dependency
                            except Exception:
                                pass  # Dependency not available, service will handle gracefully

                        # Add config if available and not already provided
                        if config and "config" not in resolved_deps:
                            resolved_deps["config"] = config

                        return service_class(**resolved_deps)
                    except Exception:
                        # Fallback to direct creation with config
                        return service_class(config=config)

                return service_factory

            factory = create_service_factory(service_name, module_path, deps)

            # Determine correct lifetime based on service type
            # Most business services should be singleton for shared state
            is_singleton = service_name in {
                "CapitalService",    # SINGLETON - shared capital state
                "ExecutionService",  # SINGLETON - shared execution context
                "RiskService",      # SINGLETON - shared risk monitoring
                "StrategyService",  # SINGLETON - shared strategy registry
                "MLService",        # SINGLETON - shared model cache
                "BotService"        # SINGLETON - shared bot management
            }

            injector.register_factory(service_name, factory, singleton=is_singleton)

        except (ImportError, AttributeError) as e:
            from src.core.logging import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Business service {service_name} not available: {e}")
            continue


def register_application_services(config: Any) -> None:
    """Register application-level services using factory patterns."""
    from src.core.dependency_injection import get_global_injector
    injector = get_global_injector()

    # Register BacktestService with factory pattern and proper dependency injection
    try:
        def backtest_service_factory():
            from src.backtesting.service import BacktestService

            # Resolve dependencies from injector
            dependencies = [
                "DatabaseService",
                "ExecutionService",
                "RiskService",
                "StrategyService",
                "CapitalService",
                "MLService"
            ]

            resolved_deps = {}
            for dep_name in dependencies:
                try:
                    resolved_deps[dep_name.lower().replace("service", "")] = injector.resolve(dep_name)
                except Exception:
                    pass  # Dependency not available, service will handle gracefully

            # Create service with dependency injection
            if resolved_deps:
                return BacktestService(**resolved_deps, config=config)
            else:
                return BacktestService(config=config)

        # BacktestService - TRANSIENT (each backtest should be isolated)
        injector.register_factory("BacktestService", backtest_service_factory, singleton=False)

    except ImportError as e:
        from src.core.logging import get_logger
        logger = get_logger(__name__)
        logger.warning(f"BacktestService not available: {e}")


async def initialize_all_services(config: Any) -> ServiceManager:
    """
    Initialize all services in the correct order.

    Args:
        config: Application configuration

    Returns:
        Initialized service manager
    """
    logger.info("Initializing all services...")

    # Get injector first to avoid circular dependencies
    from src.core.dependency_injection import get_global_injector
    injector = get_global_injector()
    service_manager = get_service_manager(injector)

    try:
        # Register all services
        register_core_services(config)
        register_business_services(config)
        register_application_services(config)

        # Start all services
        await service_manager.start_all_services()

        logger.info("All services initialized successfully")
        return service_manager

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        await service_manager.stop_all_services()
        raise


async def shutdown_all_services() -> None:
    """Shutdown all services gracefully."""
    logger.info("Shutting down all services...")

    # Get existing service manager without creating new one
    global _service_manager
    if _service_manager is not None:
        await _service_manager.stop_all_services()
        _service_manager = None  # Clear global reference

    logger.info("All services shut down")
