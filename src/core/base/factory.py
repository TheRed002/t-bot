"""
Base factory implementation for the factory pattern.

This module provides the foundation for object creation patterns
in the trading bot system, implementing registration systems,
dependency injection, and type-safe object creation.
"""

import inspect
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    get_type_hints,
)

from src.core.base.component import BaseComponent
from src.core.base.interfaces import FactoryComponent, HealthStatus
from src.core.exceptions import (
    CreationError,
    RegistrationError,
)
from src.core.types.base import ConfigDict

# Type variables for factory operations
T = TypeVar("T")  # Product type
P = TypeVar("P")  # Protocol type


class CreatorFunction(Protocol, Generic[T]):
    """Protocol for creator functions."""

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Create instance of type T."""
        ...


class BaseFactory(BaseComponent, FactoryComponent, Generic[T]):
    """
    Base factory implementing the factory pattern.

    Provides:
    - Type-safe object creation
    - Registration system for creators
    - Dependency injection integration
    - Configuration-based creation
    - Lifecycle management of created objects
    - Creation monitoring and metrics
    - Singleton and multiton patterns
    - Creator validation and testing

    Example:
        ```python
        # Create a factory for trading strategies
        strategy_factory = BaseFactory[TradingStrategy](
            name="StrategyFactory", product_type=TradingStrategy
        )

        # Register creators
        strategy_factory.register("mean_reversion", MeanReversionStrategy)
        strategy_factory.register("momentum", MomentumStrategy)

        # Create instances
        strategy = strategy_factory.create("mean_reversion", config={"lookback": 20})
        ```
    """

    def __init__(
        self,
        product_type: type[T],
        name: str | None = None,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize base factory.

        Args:
            product_type: Type of objects this factory creates
            name: Factory name for identification
            config: Factory configuration
            correlation_id: Request correlation ID
        """
        super().__init__(name, config, correlation_id)

        self._product_type = product_type
        self._creators: dict[str, CreatorFunction[T]] = {}
        self._creator_configs: dict[str, dict[str, Any]] = {}
        self._creator_metadata: dict[str, dict[str, Any]] = {}

        # Singleton management
        self._singletons: dict[str, T] = {}
        self._singleton_names: set[str] = set()
        self._creation_lock = threading.RLock()

        # Dependency injection
        self._dependency_container: Any | None = None
        self._auto_inject = True

        # Creation tracking
        self._creation_metrics: dict[str, Any] = {
            "total_creations": 0,
            "successful_creations": 0,
            "failed_creations": 0,
            "creation_times": {},
            "last_creation_time": None,
            "active_instances": 0,
        }

        # Validation settings
        self._validate_creators = True
        self._validate_products = True

        self._logger.debug(
            "Factory initialized",
            factory=self._name,
            product_type=product_type.__name__,
        )

    @property
    def product_type(self) -> type[T]:
        """Get product type managed by this factory."""
        return self._product_type

    @property
    def creation_metrics(self) -> dict[str, Any]:
        """Get creation performance metrics."""
        return self._creation_metrics.copy()

    # Registration Management
    def register(
        self,
        name: str,
        creator: type[T] | CreatorFunction[T] | Callable[..., T],
        config: dict[str, Any] | None = None,
        singleton: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Register creator function or class for given name.

        Args:
            name: Unique name for this creator
            creator: Creator function, class, or callable
            config: Default configuration for created instances
            singleton: Whether to create singleton instances
            metadata: Additional metadata about the creator

        Raises:
            RegistrationError: If registration fails or name conflicts
        """
        with self._creation_lock:
            if name in self._creators:
                raise RegistrationError(
                    f"Creator '{name}' already registered in factory {self._name}"
                )

            try:
                # Validate creator
                if self._validate_creators:
                    self._validate_creator(name, creator)

                # Store creator and configuration
                self._creators[name] = creator
                self._creator_configs[name] = config or {}
                self._creator_metadata[name] = metadata or {}

                if singleton:
                    self._singleton_names.add(name)

                self._logger.info(
                    "Creator registered",
                    factory=self._name,
                    creator_name=name,
                    creator_type=type(creator).__name__,
                    singleton=singleton,
                )

            except Exception as e:
                raise RegistrationError(
                    f"Failed to register creator '{name}' in factory {self._name}: {e}"
                ) from e

    def unregister(self, name: str) -> None:
        """
        Unregister creator function.

        Args:
            name: Name of creator to unregister

        Raises:
            RegistrationError: If creator not found
        """
        with self._creation_lock:
            if name not in self._creators:
                raise RegistrationError(f"Creator '{name}' not found in factory {self._name}")

            try:
                # Clean up resources
                if name in self._singletons:
                    instance = self._singletons[name]
                    if hasattr(instance, "cleanup"):
                        instance.cleanup()
                    del self._singletons[name]

                # Remove creator
                del self._creators[name]
                self._creator_configs.pop(name, None)
                self._creator_metadata.pop(name, None)
                self._singleton_names.discard(name)

                self._logger.info(
                    "Creator unregistered",
                    factory=self._name,
                    creator_name=name,
                )

            except Exception as e:
                raise RegistrationError(
                    f"Failed to unregister creator '{name}' in factory {self._name}: {e}"
                ) from e

    def update_creator_config(self, name: str, config: dict[str, Any]) -> None:
        """
        Update default configuration for a registered creator.

        Args:
            name: Name of creator
            config: New default configuration

        Raises:
            RegistrationError: If creator not found
        """
        with self._creation_lock:
            if name not in self._creators:
                raise RegistrationError(f"Creator '{name}' not found in factory {self._name}")

            self._creator_configs[name].update(config)

            self._logger.debug(
                "Creator configuration updated",
                factory=self._name,
                creator_name=name,
                config=config,
            )

    # Object Creation
    def create(
        self, name: str, *args: Any, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> T:
        """
        Create instance using registered creator.

        Args:
            name: Name of registered creator
            *args: Arguments to pass to creator
            config: Configuration to pass to creator
            **kwargs: Keyword arguments to pass to creator

        Returns:
            Created instance of type T

        Raises:
            CreationError: If creation fails
            RegistrationError: If creator not found
        """
        start_time = datetime.now(timezone.utc)

        with self._creation_lock:
            if name not in self._creators:
                raise RegistrationError(f"Creator '{name}' not found in factory {self._name}")

            # Check singleton pattern
            if name in self._singleton_names:
                if name in self._singletons:
                    self._logger.debug(
                        "Returning existing singleton instance",
                        factory=self._name,
                        creator_name=name,
                    )
                    return self._singletons[name]

            try:
                self._logger.debug(
                    "Creating instance",
                    factory=self._name,
                    creator_name=name,
                )

                # Prepare creation parameters
                creation_config = self._creator_configs[name].copy()
                if config:
                    creation_config.update(config)

                # Inject configuration if supported
                if creation_config:
                    kwargs.setdefault("config", creation_config)

                # Perform dependency injection
                if self._auto_inject and self._dependency_container:
                    kwargs = self._inject_dependencies(name, kwargs)

                # Execute creator
                creator = self._creators[name]
                instance = self._execute_creator(creator, *args, **kwargs)

                # Validate created instance
                if self._validate_products:
                    self._validate_product(name, instance)

                # Store singleton if needed
                if name in self._singleton_names:
                    self._singletons[name] = instance

                # Record successful creation
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                self._record_creation_success(name, execution_time)

                self._logger.info(
                    "Instance created successfully",
                    factory=self._name,
                    creator_name=name,
                    instance_type=type(instance).__name__,
                    execution_time_seconds=execution_time,
                )

                return instance

            except Exception as e:
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                self._record_creation_failure(name, execution_time, e)

                self._logger.error(
                    "Instance creation failed",
                    factory=self._name,
                    creator_name=name,
                    execution_time_seconds=execution_time,
                    error=str(e),
                    error_type=type(e).__name__,
                )

                raise CreationError(
                    f"Failed to create instance using creator '{name}' in factory {self._name}: {e}"
                ) from e

    def create_batch(self, requests: list[dict[str, Any]]) -> list[T]:
        """
        Create multiple instances in batch.

        Args:
            requests: List of creation requests, each containing:
                     {'name': str, 'args': list, 'kwargs': dict}

        Returns:
            List of created instances

        Raises:
            CreationError: If any creation fails
        """
        instances = []

        try:
            for i, request in enumerate(requests):
                name = request.get("name")
                args = request.get("args", [])
                kwargs = request.get("kwargs", {})

                if not name:
                    raise CreationError(f"Missing creator name in request {i}")

                instance = self.create(name, *args, **kwargs)
                instances.append(instance)

            self._logger.info(
                "Batch creation completed",
                factory=self._name,
                count=len(instances),
            )

            return instances

        except Exception as e:
            # Cleanup any created instances on failure
            for instance in instances:
                if hasattr(instance, "cleanup"):
                    try:
                        instance.cleanup()
                    except Exception as e:
                        # Log cleanup errors for debugging but don't fail the overall cleanup
                        self.logger.debug(f"Failed to cleanup factory instance: {e}")

            raise CreationError(f"Batch creation failed in factory {self._name}: {e}") from e

    # Creator Execution
    def _execute_creator(
        self, creator: type[T] | CreatorFunction[T] | Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute creator with proper error handling.

        Args:
            creator: Creator function or class
            *args: Creation arguments
            **kwargs: Creation keyword arguments

        Returns:
            Created instance
        """
        try:
            # Handle class constructors
            if inspect.isclass(creator):
                return creator(*args, **kwargs)

            # Handle callable objects
            elif callable(creator):
                return creator(*args, **kwargs)

            else:
                raise CreationError(f"Invalid creator type: {type(creator).__name__}")

        except Exception as e:
            raise CreationError(f"Creator execution failed: {e}") from e

    # Dependency Injection
    def _inject_dependencies(self, creator_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Inject dependencies into creation parameters.

        Args:
            creator_name: Name of creator
            kwargs: Current keyword arguments

        Returns:
            Updated keyword arguments with injected dependencies
        """
        if not self._dependency_container:
            return kwargs

        try:
            creator = self._creators[creator_name]

            # Get creator signature
            if inspect.isclass(creator):
                signature = inspect.signature(creator.__init__)
            else:
                signature = inspect.signature(creator)

            # Get type hints
            type_hints = get_type_hints(creator)

            # Inject dependencies based on parameter types
            for param_name, _param in signature.parameters.items():
                if param_name not in kwargs and param_name != "self" and param_name in type_hints:
                    param_type = type_hints[param_name]

                    try:
                        dependency = self._dependency_container.resolve(param_type)
                        kwargs[param_name] = dependency

                        self._logger.debug(
                            "Dependency injected",
                            factory=self._name,
                            creator=creator_name,
                            parameter=param_name,
                            dependency_type=param_type.__name__,
                        )

                    except Exception as e:
                        # Dependency injection is optional, log for debugging
                        self.logger.debug(f"Failed to inject dependency {param_name}: {e}")
                        continue

            return kwargs

        except Exception as e:
            self._logger.warning(
                "Dependency injection failed",
                factory=self._name,
                creator=creator_name,
                error=str(e),
            )
            return kwargs

    # Validation
    def _validate_creator(
        self, name: str, creator: type[T] | CreatorFunction[T] | Callable[..., T]
    ) -> None:
        """
        Validate creator function or class.

        Args:
            name: Creator name
            creator: Creator to validate

        Raises:
            RegistrationError: If creator is invalid
        """
        try:
            # Check if creator is callable
            if not callable(creator):
                raise RegistrationError(f"Creator '{name}' is not callable")

            # Test creator with dummy parameters if possible
            if inspect.isclass(creator):
                # Check if class has required constructor
                signature = inspect.signature(creator.__init__)
                required_params = [
                    p
                    for p in signature.parameters.values()
                    if p.default == inspect.Parameter.empty and p.name != "self"
                ]

                # Allow creators with no required parameters to be tested
                if not required_params:
                    try:
                        test_instance = creator()
                        if not isinstance(test_instance, self._product_type):
                            raise RegistrationError(f"Creator '{name}' produces wrong type")

                        # Cleanup test instance
                        if hasattr(test_instance, "cleanup"):
                            test_instance.cleanup()

                    except Exception as e:
                        self._logger.warning(
                            "Creator validation test failed",
                            factory=self._name,
                            creator=name,
                            error=str(e),
                        )

        except Exception as e:
            raise RegistrationError(f"Creator validation failed for '{name}': {e}") from e

    def _validate_product(self, creator_name: str, instance: Any) -> None:
        """
        Validate created product instance.

        Args:
            creator_name: Name of creator that produced instance
            instance: Created instance to validate

        Raises:
            CreationError: If instance is invalid
        """
        if not isinstance(instance, self._product_type):
            raise CreationError(
                f"Creator '{creator_name}' produced instance of type "
                f"{type(instance).__name__}, expected {self._product_type.__name__}"
            )

    # Information and Management
    def list_registered(self) -> list[str]:
        """
        List all registered creator names.

        Returns:
            List of registered creator names
        """
        return list(self._creators.keys())

    def is_registered(self, name: str) -> bool:
        """
        Check if creator is registered.

        Args:
            name: Creator name to check

        Returns:
            True if creator is registered
        """
        return name in self._creators

    def get_creator_info(self, name: str) -> dict[str, Any] | None:
        """
        Get information about registered creator.

        Args:
            name: Creator name

        Returns:
            Creator information dictionary or None if not found
        """
        if name not in self._creators:
            return None

        creator = self._creators[name]

        return {
            "name": name,
            "creator_type": type(creator).__name__,
            "is_class": inspect.isclass(creator),
            "is_singleton": name in self._singleton_names,
            "config": self._creator_configs.get(name, {}),
            "metadata": self._creator_metadata.get(name, {}),
            "signature": str(inspect.signature(creator)),
        }

    def get_all_creator_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all registered creators."""
        return {name: self.get_creator_info(name) for name in self._creators.keys()}

    # Singleton Management
    def get_singleton(self, name: str) -> T | None:
        """
        Get singleton instance if it exists.

        Args:
            name: Creator name

        Returns:
            Singleton instance or None
        """
        return self._singletons.get(name)

    def clear_singletons(self) -> None:
        """Clear all singleton instances."""
        with self._creation_lock:
            for name, instance in self._singletons.items():
                if hasattr(instance, "cleanup"):
                    try:
                        instance.cleanup()
                    except Exception as e:
                        self._logger.warning(
                            "Singleton cleanup failed",
                            factory=self._name,
                            singleton=name,
                            error=str(e),
                        )

            self._singletons.clear()

            self._logger.info(
                "All singletons cleared",
                factory=self._name,
            )

    def reset_singleton(self, name: str) -> None:
        """
        Reset specific singleton instance.

        Args:
            name: Creator name
        """
        with self._creation_lock:
            if name in self._singletons:
                instance = self._singletons[name]
                if hasattr(instance, "cleanup"):
                    try:
                        instance.cleanup()
                    except Exception as e:
                        self._logger.warning(
                            "Singleton cleanup failed",
                            factory=self._name,
                            singleton=name,
                            error=str(e),
                        )

                del self._singletons[name]

                self._logger.info(
                    "Singleton reset",
                    factory=self._name,
                    singleton=name,
                )

    # Health Check
    async def _health_check_internal(self) -> HealthStatus:
        """Factory-specific health check."""
        try:
            # Check if any creators are registered
            if not self._creators:
                return HealthStatus.DEGRADED  # Factory has no creators

            # Test a few creators if possible
            test_creators = list(self._creators.items())[:3]  # Test up to 3

            for name, _creator in test_creators:
                try:
                    # Try to get creator info
                    info = self.get_creator_info(name)
                    if not info:
                        return HealthStatus.DEGRADED

                except Exception as e:
                    self.logger.debug(f"Failed to check creator info for {name}: {e}")
                    return HealthStatus.DEGRADED

            # Check creation error rate
            total_creations = self._creation_metrics["total_creations"]
            failed_creations = self._creation_metrics["failed_creations"]

            if total_creations > 0:
                error_rate = failed_creations / total_creations
                if error_rate > 0.1:  # More than 10% errors
                    return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error(
                "Factory health check failed",
                factory=self._name,
                error=str(e),
            )
            return HealthStatus.UNHEALTHY

    # Metrics
    def _record_creation_success(self, creator_name: str, execution_time: float) -> None:
        """Record successful creation metrics."""
        self._creation_metrics["total_creations"] += 1
        self._creation_metrics["successful_creations"] += 1
        self._creation_metrics["last_creation_time"] = datetime.now(timezone.utc)
        self._creation_metrics["active_instances"] += 1

        # Track per-creator timing
        if creator_name not in self._creation_metrics["creation_times"]:
            self._creation_metrics["creation_times"][creator_name] = []

        times = self._creation_metrics["creation_times"][creator_name]
        times.append(execution_time)

        # Keep only last 100 timing records per creator
        if len(times) > 100:
            times.pop(0)

    def _record_creation_failure(
        self, creator_name: str, execution_time: float, error: Exception
    ) -> None:
        """Record failed creation metrics."""
        self._creation_metrics["total_creations"] += 1
        self._creation_metrics["failed_creations"] += 1
        self._creation_metrics["last_creation_time"] = datetime.now(timezone.utc)

    def get_metrics(self) -> dict[str, Any]:
        """Get combined component and factory metrics."""
        metrics = super().get_metrics()
        metrics.update(self.creation_metrics)
        metrics.update(
            {
                "registered_creators": len(self._creators),
                "singleton_instances": len(self._singletons),
            }
        )
        return metrics

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        super().reset_metrics()
        self._creation_metrics = {
            "total_creations": 0,
            "successful_creations": 0,
            "failed_creations": 0,
            "creation_times": {},
            "last_creation_time": None,
            "active_instances": 0,
        }

    # Configuration
    def configure_validation(
        self, validate_creators: bool = True, validate_products: bool = True
    ) -> None:
        """
        Configure validation settings.

        Args:
            validate_creators: Enable creator validation
            validate_products: Enable product validation
        """
        self._validate_creators = validate_creators
        self._validate_products = validate_products

        self._logger.info(
            "Validation configured",
            factory=self._name,
            validate_creators=validate_creators,
            validate_products=validate_products,
        )

    def configure_dependency_injection(
        self,
        container: Any | None = None,
        auto_inject: bool = True,
    ) -> None:
        """
        Configure dependency injection.

        Args:
            container: Dependency injection container
            auto_inject: Enable automatic dependency injection
        """
        self._dependency_container = container
        self._auto_inject = auto_inject

        self._logger.info(
            "Dependency injection configured",
            factory=self._name,
            has_container=container is not None,
            auto_inject=auto_inject,
        )

    # Lifecycle Management
    async def _do_stop(self) -> None:
        """Factory cleanup on shutdown."""
        # Clear all singletons
        self.clear_singletons()

        # Clear registrations
        self._creators.clear()
        self._creator_configs.clear()
        self._creator_metadata.clear()
        self._singleton_names.clear()

        self._logger.debug("Factory stopped and cleaned up", factory=self._name)
