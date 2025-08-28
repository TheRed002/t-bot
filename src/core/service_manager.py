"""
Service Manager for centralized service management and dependency resolution.

This module provides a centralized service registry that eliminates circular
dependencies by managing service lifecycle and dependency injection.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from src.core.dependency_injection import injector
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

    def __init__(self) -> None:
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

        # Register factory with DI container
        def service_factory():
            return self._create_service(service_name)

        self._injector.register_factory(service_name, service_factory, singleton=singleton)

    def _create_service(self, service_name: str) -> Any:
        """Create a service instance with resolved dependencies."""
        if service_name in self._services:
            return self._services[service_name]

        config = self._service_configs[service_name]
        self._logger.debug(f"Creating service: {service_name}")

        try:
            # Resolve dependencies
            resolved_deps = self._resolve_dependencies(service_name, config["dependencies"])

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
                raise DependencyError(f"Unknown dependency: {dep_name}")
            resolved_deps[dep_name] = self._injector.resolve(dep_name)
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
        if hasattr(service_instance, "_dependency_container"):
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
            self._logger.error(f"Error clearing service references: {e}")

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
            self._logger.error(f"Failed to stop service {service_name}: {e}")
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

        # Use consistent batch processing for health checks
        health_tasks = []
        for service_name in self._running_services:
            task = asyncio.create_task(self._check_service_health(service_name))
            health_tasks.append((service_name, task))

        # Wait for all health checks with timeout
        for service_name, task in health_tasks:
            try:
                health_status[service_name] = await asyncio.wait_for(task, timeout=30.0)
            except asyncio.TimeoutError:
                health_status[service_name] = {
                    "status": "timeout",
                    "error": "Health check timed out",
                }
            except Exception as e:
                health_status[service_name] = {"status": "error", "error": str(e)}

        # Consistent status aggregation
        healthy_statuses = ["healthy", "running"]
        overall_status = (
            "healthy"
            if all(status.get("status") in healthy_statuses for status in health_status.values())
            else "degraded"
        )

        return {
            "overall_status": overall_status,
            "services": health_status,
            "running_count": len(self._running_services),
            "total_registered": len(self._service_configs),
            "timestamp": datetime.now().isoformat(),
        }

    async def _check_service_health(self, service_name: str) -> dict[str, Any]:
        """Check health of a single service with consistent error handling."""
        try:
            service = self._services.get(service_name)
            if service and hasattr(service, "health_check"):
                result = await service.health_check()
                # Ensure consistent health check result format
                if isinstance(result, dict):
                    return result
                else:
                    return {"status": "healthy", "details": result}
            else:
                return {"status": "running", "details": "No health check method available"}
        except Exception as e:
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}


# Global service manager instance
service_manager = ServiceManager()


def get_service_manager() -> ServiceManager:
    """Get the global service manager instance."""
    return service_manager


def register_core_services(config: Any) -> None:
    """
    Register all core services with the service manager.

    This function sets up the proper dependency chain to avoid circular imports.
    """
    from src.core.config.service import ConfigService
    from src.database.service import DatabaseService
    from src.state.state_service import StateService
    from src.utils.validation.service import ValidationService

    # Register services in dependency order

    # 1. Configuration service (no dependencies)
    service_manager.register_service(
        "ConfigService",
        ConfigService,
        config={"config": config} if config else {},
        dependencies=[],
    )

    # 2. Validation service (no dependencies)
    service_manager.register_service(
        "ValidationService",
        ValidationService,
        dependencies=[],
    )

    # 3. Database service (depends on config and validation)
    service_manager.register_service(
        "DatabaseService",
        DatabaseService,
        config={"config": config} if config else {},
        dependencies=["ConfigService", "ValidationService"],
    )

    # 4. State service (depends on database)
    service_manager.register_service(
        "StateService",
        StateService,
        config={"config": config} if config else {},
        dependencies=["DatabaseService"],
    )


def register_business_services(config: Any) -> None:
    """Register business logic services."""
    from src.bot_management.service import BotService
    from src.capital_management.service import CapitalService
    from src.execution.service import ExecutionService
    from src.ml.service import MLService
    from src.risk_management.service import RiskService
    from src.strategies.service import StrategyService

    # 5. Capital management service
    service_manager.register_service(
        "CapitalService",
        CapitalService,
        config={"config": config} if config else {},
        dependencies=["DatabaseService"],
    )

    # 6. Execution service
    service_manager.register_service(
        "ExecutionService",
        ExecutionService,
        config={"config": config} if config else {},
        dependencies=["DatabaseService"],
    )

    # 7. Risk management service
    service_manager.register_service(
        "RiskService",
        RiskService,
        config={"config": config} if config else {},
        dependencies=["DatabaseService", "StateService"],
    )

    # 8. Strategy service (minimal dependencies)
    service_manager.register_service(
        "StrategyService",
        StrategyService,
        config={"config": config} if config else {},
        dependencies=[],
    )

    # 9. ML service
    service_manager.register_service(
        "MLService",
        MLService,
        config={"config": config} if config else {},
        dependencies=[],
    )

    # 10. Bot service (depends on all other business services)
    service_manager.register_service(
        "BotService",
        BotService,
        config={"config": config} if config else {},
        dependencies=[
            "DatabaseService",
            "StateService",
            "RiskService",
            "ExecutionService",
            "StrategyService",
            "CapitalService",
        ],
    )


def register_application_services(config: Any) -> None:
    """Register application-level services."""
    from src.backtesting.service import BacktestService

    # 11. Backtesting service (depends on all business services)
    service_manager.register_service(
        "BacktestService",
        BacktestService,
        config={"config": config} if config else {},
        dependencies=[
            "DatabaseService",
            "ExecutionService",
            "RiskService",
            "StrategyService",
            "CapitalService",
            "MLService",
        ],
    )


async def initialize_all_services(config: Any) -> ServiceManager:
    """
    Initialize all services in the correct order.

    Args:
        config: Application configuration

    Returns:
        Initialized service manager
    """
    logger.info("Initializing all services...")

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
    await service_manager.stop_all_services()
    logger.info("All services shut down")
