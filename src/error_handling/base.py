"""Base classes for error handling using Chain of Responsibility pattern."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.core.logging import get_logger

# ErrorContext import at top for proper module organization

logger = get_logger(__name__)


class ErrorHandlerBase(ABC):
    """
    Base class for all error handlers using Chain of Responsibility pattern.

    Each handler can either handle an error or pass it to the next handler
    in the chain.
    """

    def __init__(self, next_handler: Optional["ErrorHandlerBase"] = None):
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
    async def handle(self, error: Exception, context: dict[str, Any] | None = None) -> Any:
        """
        Handle the error and return recovery action.

        Args:
            error: Exception to handle
            context: Optional context information

        Returns:
            Recovery action or result
        """
        pass

    async def process(self, error: Exception, context: dict[str, Any] | None = None) -> Any:
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
            return await self.handle(error, context)
        elif self.next_handler:
            return await self.next_handler.process(error, context)
        else:
            # End of chain, error not handled
            self._logger.warning(f"No handler found for {type(error).__name__}")
            raise error

    def set_next(self, handler: "ErrorHandlerBase") -> "ErrorHandlerBase":
        """
        Set the next handler in the chain.

        Args:
            handler: Next handler

        Returns:
            The handler for chaining
        """
        self.next_handler = handler
        return handler


__all__ = ["ErrorHandlerBase"]
