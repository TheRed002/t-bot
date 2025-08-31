"""Factory for creating error handlers without direct imports."""

from typing import Any, ClassVar, Protocol, runtime_checkable

from src.core.base.factory import BaseFactory
from src.core.exceptions import CreationError
from src.error_handling.base import ErrorHandlerBase


@runtime_checkable
class ErrorHandlerProtocol(Protocol):
    """Protocol for error handlers."""

    def can_handle(self, error: Exception) -> bool:
        """Check if this handler can handle the given error."""
        ...

    async def handle(self, error: Exception, context: dict[str, Any] | None = None) -> Any:
        """Handle the error and return recovery action."""
        ...

    async def process(self, error: Exception, context: dict[str, Any] | None = None) -> Any:
        """Process the error through the chain."""
        ...


class ErrorHandlerFactory(BaseFactory[ErrorHandlerProtocol]):
    """
    Factory to create handlers without direct imports.

    This breaks circular dependencies by allowing modules to register
    handlers at startup and create them on demand.

    Uses dependency injection for complex creation logic.
    """

    # Legacy compatibility class variables
    _handlers: ClassVar[dict[str, type[ErrorHandlerBase]]] = {}
    _configurations: ClassVar[dict[str, dict[str, Any]]] = {}
    _dependency_container: ClassVar[Any] = None

    def __init__(self, dependency_container: Any | None = None):
        """Initialize factory with dependency injection."""
        super().__init__(product_type=ErrorHandlerProtocol, name="ErrorHandlerFactory")

        if dependency_container:
            self.configure_dependencies(dependency_container)
            ErrorHandlerFactory._dependency_container = dependency_container

    @classmethod
    def register(
        cls,
        error_type: str,
        handler_class: type[ErrorHandlerBase],
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a handler class for a given error type.

        Args:
            error_type: Identifier for the error type
            handler_class: The handler class to register
            config: Optional configuration for the handler
        """
        cls._handlers[error_type] = handler_class
        if config:
            cls._configurations[error_type] = config

        # Also register with new factory system if instance exists
        try:
            # This assumes there's a global factory instance
            # In practice, this would be managed by the DI container
            pass
        except Exception as e:
            # Legacy registration continues to work - log warning but continue
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Factory registration to global instance failed: {e}")
            # Continue with legacy registration

    @classmethod
    def create(
        cls, error_type: str, next_handler: ErrorHandlerBase | None = None, **kwargs
    ) -> ErrorHandlerProtocol:
        """
        Create a handler instance with dependency injection support.

        Args:
            error_type: Type of error handler to create
            next_handler: Next handler in chain
            **kwargs: Additional configuration

        Returns:
            Handler instance conforming to ErrorHandlerProtocol

        Raises:
            ValueError: If handler type is not registered
        """
        if error_type not in cls._handlers:
            raise ValueError(f"Unknown handler type: {error_type}")

        handler_class = cls._handlers[error_type]

        # Merge configurations
        config = cls._configurations.get(error_type, {}).copy()
        config.update(kwargs)

        # Add next_handler to config if handler accepts it
        import inspect

        sig = inspect.signature(handler_class.__init__)
        if "next_handler" in sig.parameters:
            config["next_handler"] = next_handler

        # Use dependency injection if available
        if cls._dependency_container:
            try:
                cls._inject_common_dependencies(config)
            except Exception as e:
                # Continue without injection if it fails - log but don't fail creation
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Dependency injection failed during error handler creation: {e}")
                # Continue with creation without injection

        try:
            instance = handler_class(**config)
            # Ensure instance conforms to protocol
            if not isinstance(instance, ErrorHandlerProtocol):
                raise CreationError(
                    f"Handler '{error_type}' does not conform to ErrorHandlerProtocol"
                )
            return instance
        except Exception as e:
            raise CreationError(f"Failed to create error handler '{error_type}': {e}") from e

    @classmethod
    def list_handlers(cls) -> list[str]:
        """List all registered handler types."""
        return list(cls._handlers.keys())

    @classmethod
    def set_dependency_container(cls, container: Any) -> None:
        """Set dependency injection container for factory."""
        cls._dependency_container = container

    @classmethod
    def _inject_common_dependencies(cls, config: dict[str, Any]) -> None:
        """Inject common dependencies into config via service locator pattern."""
        if not cls._dependency_container:
            raise ValueError("Dependency container required for service creation")

        container = cls._dependency_container

        # Use service locator pattern for dependency injection
        # Only inject dependencies that the handler actually needs
        service_mappings = {
            "Config": "config",
            "SecuritySanitizer": "sanitizer",
            "SecurityRateLimiter": "rate_limiter",
        }

        for service_name, config_key in service_mappings.items():
            # Only inject if the parameter is missing and service is available
            if config_key not in config:
                if hasattr(container, "has_service") and container.has_service(service_name):
                    # Use service locator to get the service
                    service = container.resolve(service_name)
                    if service is not None:
                        config[config_key] = service
                else:
                    # Required dependencies must be available
                    if service_name in ["SecuritySanitizer", "SecurityRateLimiter"]:
                        raise ValueError(
                            f"Required service {service_name} not registered in DI container"
                        )

    @classmethod
    def clear(cls) -> None:
        """Clear all registered handlers (useful for testing)."""
        cls._handlers.clear()
        cls._configurations.clear()
        cls._dependency_container = None


class ErrorHandlerChain(BaseFactory[ErrorHandlerProtocol]):
    """
    Manages a chain of error handlers.

    This class builds and manages the chain of responsibility for
    error handling. Now uses dependency injection for handler creation.
    """

    def __init__(self, handlers: list[str] | None = None, dependency_container: Any | None = None):
        """
        Initialize the error handler chain.

        Args:
            handlers: List of handler type names to build the chain
            dependency_container: Optional dependency container for injection
        """
        super().__init__(product_type=ErrorHandlerProtocol, name="ErrorHandlerChain")

        self.chain: ErrorHandlerBase | None = None

        if dependency_container:
            self.configure_dependencies(dependency_container)

        if handlers:
            self.build_chain(handlers)

    def build_chain(self, handler_types: list[str]) -> None:
        """
        Build the handler chain from a list of handler types using factory.

        Args:
            handler_types: List of handler type names
        """
        if not handler_types:
            return

        # Build chain in reverse order using dependency injection
        self.chain = None
        for handler_type in reversed(handler_types):
            try:
                # Use dependency injection if available
                if self._dependency_container:
                    ErrorHandlerFactory.set_dependency_container(self._dependency_container)

                handler = ErrorHandlerFactory.create(handler_type, next_handler=self.chain)
                self.chain = handler
            except Exception as e:
                # Log error but continue with simplified chain
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to create handler '{handler_type}' in chain: {e}")
                # Skip this handler and continue

    async def handle(self, error: Exception, context: dict[str, Any] | None = None) -> Any:
        """
        Handle an error through the chain.

        Args:
            error: The exception to handle
            context: Optional context information

        Returns:
            Recovery action or raises if unhandled
        """
        if not self.chain:
            raise error

        return await self.chain.process(error, context)

    def add_handler(self, handler_type: str) -> None:
        """
        Add a handler to the beginning of the chain using factory pattern.

        Args:
            handler_type: Type of handler to add
        """
        try:
            # Ensure factory uses dependency injection
            if self._dependency_container:
                ErrorHandlerFactory.set_dependency_container(self._dependency_container)

            new_handler = ErrorHandlerFactory.create(handler_type, next_handler=self.chain)
            self.chain = new_handler
        except Exception as e:
            # Log error but don't fail the operation
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to add handler '{handler_type}' to chain: {e}")
            raise

    @classmethod
    def create_default_chain(cls, dependency_container: Any | None = None) -> "ErrorHandlerChain":
        """
        Create a default error handler chain with dependency injection.

        Args:
            dependency_container: Optional dependency container

        Returns:
            Configured error handler chain
        """
        # Use simplified factory pattern with fewer dependencies
        default_handlers: list[str] = [
            "network",  # Network errors first
            "database",  # Database errors
            "validation",  # Validation errors
        ]

        try:
            # Set up dependency injection for factory
            if dependency_container:
                ErrorHandlerFactory.set_dependency_container(dependency_container)

            return cls(default_handlers, dependency_container)
        except Exception as e:
            # Fallback to basic chain without dependency injection
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to create default chain with DI, using basic chain: {e}")
            return cls(default_handlers, None)
