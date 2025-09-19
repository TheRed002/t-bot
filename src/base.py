"""Base classes and mixins for the T-Bot trading system."""

# Import get_logger with robust fallback
def _get_logger_function():
    """Get the logger function with fallback."""
    try:
        from src.core.logging import get_logger
        return get_logger
    except ImportError:
        # Fallback for testing or import issues
        import logging
        return logging.getLogger

get_logger = _get_logger_function()


class LoggerMixin:
    """Mixin that provides logger to any class."""

    @property
    def logger(self):
        """Get or create logger for this instance."""
        if not hasattr(self, "_logger"):
            # Import get_logger locally to avoid module loading issues
            try:
                from src.core.logging import get_logger
            except ImportError:
                import logging
                get_logger = logging.getLogger
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
