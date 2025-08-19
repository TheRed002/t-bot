"""Factory for creating error handlers without direct imports."""

from typing import Dict, Type, Optional, Any
from src.error_handling.base import ErrorHandlerBase


class ErrorHandlerFactory:
    """
    Factory to create handlers without direct imports.
    
    This breaks circular dependencies by allowing modules to register
    handlers at startup and create them on demand.
    """
    
    _handlers: Dict[str, Type[ErrorHandlerBase]] = {}
    _configurations: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls,
        error_type: str,
        handler_class: Type[ErrorHandlerBase],
        config: Optional[Dict[str, Any]] = None
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
    
    @classmethod
    def create(
        cls,
        error_type: str,
        next_handler: Optional[ErrorHandlerBase] = None,
        **kwargs
    ) -> ErrorHandlerBase:
        """
        Create a handler instance.
        
        Args:
            error_type: Type of error handler to create
            next_handler: Next handler in chain
            **kwargs: Additional configuration
            
        Returns:
            Handler instance
            
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
        if 'next_handler' in sig.parameters:
            config['next_handler'] = next_handler
        
        return handler_class(**config)
    
    @classmethod
    def list_handlers(cls) -> list:
        """List all registered handler types."""
        return list(cls._handlers.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered handlers (useful for testing)."""
        cls._handlers.clear()
        cls._configurations.clear()


class ErrorHandlerChain:
    """
    Manages a chain of error handlers.
    
    This class builds and manages the chain of responsibility for
    error handling.
    """
    
    def __init__(self, handlers: Optional[list] = None):
        """
        Initialize the error handler chain.
        
        Args:
            handlers: List of handler type names to build the chain
        """
        self.chain = None
        
        if handlers:
            self.build_chain(handlers)
    
    def build_chain(self, handler_types: list) -> None:
        """
        Build the handler chain from a list of handler types.
        
        Args:
            handler_types: List of handler type names
        """
        if not handler_types:
            return
        
        # Build chain in reverse order
        self.chain = None
        for handler_type in reversed(handler_types):
            self.chain = ErrorHandlerFactory.create(
                handler_type,
                next_handler=self.chain
            )
    
    def handle(self, error: Exception, context: Optional[Dict] = None) -> Any:
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
        
        return self.chain.process(error, context)
    
    def add_handler(self, handler_type: str) -> None:
        """
        Add a handler to the beginning of the chain.
        
        Args:
            handler_type: Type of handler to add
        """
        new_handler = ErrorHandlerFactory.create(
            handler_type,
            next_handler=self.chain
        )
        self.chain = new_handler
    
    @classmethod
    def create_default_chain(cls) -> 'ErrorHandlerChain':
        """
        Create a default error handler chain.
        
        Returns:
            Configured error handler chain
        """
        default_handlers = [
            'rate_limit',     # Check rate limits first
            'network',        # Then network errors
            'database',       # Database errors
            'validation',     # Validation errors
            'data_validation' # Data validation
        ]
        
        return cls(default_handlers)