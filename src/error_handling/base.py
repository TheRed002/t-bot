"""Base classes for error handling using Chain of Responsibility pattern."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from datetime import datetime
import traceback

from src.core.logging import get_logger

logger = get_logger(__name__)


class ErrorHandlerBase(ABC):
    """
    Base class for all error handlers using Chain of Responsibility pattern.
    
    Each handler can either handle an error or pass it to the next handler
    in the chain.
    """
    
    def __init__(self, next_handler: Optional['ErrorHandlerBase'] = None):
        """
        Initialize error handler with optional next handler in chain.
        
        Args:
            next_handler: Next handler in the chain
        """
        self.next_handler = next_handler
        self._logger = get_logger(self.__class__.__module__)
    
    @abstractmethod
    def can_handle(self, error: Exception) -> bool:
        """
        Check if this handler can handle the given error.
        
        Args:
            error: Exception to check
            
        Returns:
            True if this handler can handle the error
        """
        pass
    
    @abstractmethod
    def handle(self, error: Exception, context: Optional[Dict] = None) -> Any:
        """
        Handle the error and return recovery action.
        
        Args:
            error: Exception to handle
            context: Optional context information
            
        Returns:
            Recovery action or result
        """
        pass
    
    def process(self, error: Exception, context: Optional[Dict] = None) -> Any:
        """
        Process the error through the chain.
        
        Args:
            error: Exception to process
            context: Optional context information
            
        Returns:
            Recovery action or raises the error if unhandled
        """
        if self.can_handle(error):
            self._logger.debug(f"Handling {type(error).__name__} with {self.__class__.__name__}")
            return self.handle(error, context)
        elif self.next_handler:
            return self.next_handler.process(error, context)
        else:
            # End of chain, error not handled
            self._logger.warning(f"No handler found for {type(error).__name__}")
            raise error
    
    def set_next(self, handler: 'ErrorHandlerBase') -> 'ErrorHandlerBase':
        """
        Set the next handler in the chain.
        
        Args:
            handler: Next handler
            
        Returns:
            The handler for chaining
        """
        self.next_handler = handler
        return handler


class ErrorContext:
    """Standard error context information."""
    
    def __init__(
        self,
        error: Exception,
        module: Optional[str] = None,
        function: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize error context.
        
        Args:
            error: The exception
            module: Module where error occurred
            function: Function where error occurred
            **kwargs: Additional context
        """
        self.error = error
        self.error_type = type(error).__name__
        self.error_message = str(error)
        self.timestamp = datetime.utcnow()
        self.module = module
        self.function = function
        self.traceback = traceback.format_exc()
        self.additional = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'error_type': self.error_type,
            'error_message': self.error_message,
            'module': self.module,
            'function': self.function,
            'traceback': self.traceback,
            **self.additional
        }