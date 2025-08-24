"""Base classes and mixins for the T-Bot trading system."""

from src.core.logging import get_logger


class LoggerMixin:
    """Mixin that provides logger to any class."""

    @property
    def logger(self):
        """Get or create logger for this instance."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger


class BaseComponent(LoggerMixin):
    """Base class for all components with logging and common functionality."""

    def __init__(self):
        """Initialize base component."""
        super().__init__()
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized

    def initialize(self) -> None:
        """Initialize the component. Override in subclasses."""
        self._initialized = True
        self.logger.debug(f"{self.__class__.__name__} initialized")

    def cleanup(self) -> None:
        """Cleanup the component. Override in subclasses."""
        self._initialized = False
        self.logger.debug(f"{self.__class__.__name__} cleaned up")
