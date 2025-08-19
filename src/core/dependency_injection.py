"""Dependency injection system to break circular dependencies."""

from typing import Dict, Any, Type, Optional, Callable, TypeVar, Union
from functools import wraps
import inspect
from threading import Lock

from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class DependencyContainer:
    """Container for managing dependencies."""
    
    def __init__(self):
        """Initialize dependency container."""
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._lock = Lock()
        self._logger = logger
    
    def register(
        self,
        name: str,
        service: Union[Any, Callable],
        singleton: bool = False
    ) -> None:
        """
        Register a service or factory.
        
        Args:
            name: Service name
            service: Service instance or factory function
            singleton: Whether to treat as singleton
        """
        with self._lock:
            if callable(service) and not inspect.isclass(service):
                # Register as factory
                self._factories[name] = service
                if singleton:
                    self._singletons[name] = None
            else:
                # Register as service
                self._services[name] = service
                if singleton:
                    self._singletons[name] = service
            
            self._logger.debug(f"Registered service: {name} (singleton={singleton})")
    
    def register_class(
        self,
        name: str,
        cls: Type[T],
        *args,
        singleton: bool = False,
        **kwargs
    ) -> None:
        """
        Register a class for lazy instantiation.
        
        Args:
            name: Service name
            cls: Class to instantiate
            *args: Positional arguments for instantiation
            singleton: Whether to treat as singleton
            **kwargs: Keyword arguments for instantiation
        """
        def factory():
            return cls(*args, **kwargs)
        
        self.register(name, factory, singleton=singleton)
    
    def get(self, name: str) -> Any:
        """
        Get a service by name.
        
        Args:
            name: Service name
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service not found
        """
        with self._lock:
            # Check if it's a direct service
            if name in self._services:
                return self._services[name]
            
            # Check if it's a singleton that's already created
            if name in self._singletons and self._singletons[name] is not None:
                return self._singletons[name]
            
            # Check if it's a factory
            if name in self._factories:
                instance = self._factories[name]()
                
                # Cache if singleton
                if name in self._singletons:
                    self._singletons[name] = instance
                
                return instance
            
            raise KeyError(f"Service '{name}' not registered")
    
    def has(self, name: str) -> bool:
        """Check if service is registered."""
        return name in self._services or name in self._factories
    
    def clear(self) -> None:
        """Clear all registered services."""
        with self._lock:
            self._services.clear()
            self._factories.clear()
            self._singletons.clear()


class DependencyInjector:
    """
    Dependency injector for automatic dependency resolution.
    
    This eliminates circular dependencies and manual wiring.
    """
    
    _instance: Optional['DependencyInjector'] = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize dependency injector."""
        if not hasattr(self, '_initialized'):
            self._container = DependencyContainer()
            self._logger = logger
            self._initialized = True
    
    def register(
        self,
        name: str = None,
        singleton: bool = False
    ):
        """
        Decorator to register a service.
        
        Args:
            name: Service name (defaults to class name)
            singleton: Whether to treat as singleton
        """
        def decorator(cls_or_func):
            service_name = name or cls_or_func.__name__
            
            if inspect.isclass(cls_or_func):
                # Register class
                self._container.register_class(
                    service_name,
                    cls_or_func,
                    singleton=singleton
                )
            else:
                # Register function/instance
                self._container.register(
                    service_name,
                    cls_or_func,
                    singleton=singleton
                )
            
            return cls_or_func
        
        return decorator
    
    def inject(self, func: Callable) -> Callable:
        """
        Decorator to inject dependencies into function.
        
        Dependencies are resolved by parameter name.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            
            # Resolve dependencies
            for param_name, param in sig.parameters.items():
                # Skip if already provided
                if param_name in kwargs:
                    continue
                
                # Try to resolve from container
                if self._container.has(param_name):
                    kwargs[param_name] = self._container.get(param_name)
                elif param.annotation != param.empty:
                    # Try to resolve by type annotation
                    type_name = param.annotation.__name__
                    if self._container.has(type_name):
                        kwargs[param_name] = self._container.get(type_name)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    def resolve(self, name: str) -> Any:
        """
        Resolve a dependency by name.
        
        Args:
            name: Service name
            
        Returns:
            Service instance
        """
        return self._container.get(name)
    
    def register_service(
        self,
        name: str,
        service: Any,
        singleton: bool = False
    ) -> None:
        """
        Register a service directly.
        
        Args:
            name: Service name
            service: Service instance or factory
            singleton: Whether to treat as singleton
        """
        self._container.register(name, service, singleton=singleton)
    
    def register_factory(
        self,
        name: str,
        factory: Callable,
        singleton: bool = False
    ) -> None:
        """
        Register a factory function.
        
        Args:
            name: Service name
            factory: Factory function
            singleton: Whether to treat as singleton
        """
        self._container.register(name, factory, singleton=singleton)
    
    def has_service(self, name: str) -> bool:
        """Check if service is registered."""
        return self._container.has(name)
    
    def clear(self) -> None:
        """Clear all registered services."""
        self._container.clear()
    
    @classmethod
    def get_instance(cls) -> 'DependencyInjector':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Global injector instance
injector = DependencyInjector.get_instance()


# Decorators for convenience
def injectable(name: str = None, singleton: bool = False):
    """Decorator to mark a class as injectable."""
    return injector.register(name=name, singleton=singleton)


def inject(func: Callable) -> Callable:
    """Decorator to inject dependencies into a function."""
    return injector.inject(func)


# Example usage
@injectable(singleton=True)
class ConfigService:
    """Example configuration service."""
    
    def __init__(self):
        self.config = {}
    
    def get(self, key: str) -> Any:
        return self.config.get(key)


@injectable()
class DatabaseService:
    """Example database service."""
    
    @inject
    def __init__(self, ConfigService: ConfigService):
        self.config_service = ConfigService
        self.connection_string = ConfigService.get('db_url')


@injectable()
class TradingService:
    """Example trading service."""
    
    @inject
    def __init__(
        self,
        DatabaseService: DatabaseService,
        ConfigService: ConfigService
    ):
        self.db = DatabaseService
        self.config = ConfigService
    
    @inject
    def execute_trade(self, OrderService):
        """Method with injected dependency."""
        # OrderService is injected at runtime
        return OrderService.create_order()


# Service locator pattern support
class ServiceLocator:
    """Service locator for easy access to services."""
    
    def __init__(self, injector: DependencyInjector):
        self._injector = injector
    
    def __getattr__(self, name: str) -> Any:
        """Get service by attribute access."""
        try:
            return self._injector.resolve(name)
        except KeyError:
            raise AttributeError(f"Service '{name}' not found")


# Global service locator
services = ServiceLocator(injector)