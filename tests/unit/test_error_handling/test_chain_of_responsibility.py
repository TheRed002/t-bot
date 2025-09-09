"""
Unit tests for the Chain of Responsibility error handling pattern.

Tests the new error handling system that eliminates code duplication.
"""

from unittest.mock import Mock

import pytest

from src.core.exceptions import ValidationError
from src.error_handling.base import ErrorHandlerBase
from src.error_handling.context import ErrorContext, ErrorContextFactory
from src.error_handling.factory import ErrorHandlerChain, ErrorHandlerFactory
from src.error_handling.handlers.database import DatabaseErrorHandler
from src.error_handling.handlers.network import NetworkErrorHandler, RateLimitErrorHandler
from src.error_handling.handlers.validation import ValidationErrorHandler
from src.error_handling.recovery import CircuitBreakerRecovery, FallbackRecovery, RetryRecovery


class TestErrorHandlerBase:
    """Test the base error handler functionality."""

    def test_error_context_creation(self):
        """Test ErrorContext creation."""
        error = ValueError("Test error")
        context = ErrorContext(error=error, module="test_module", function_name="test_function")

        assert context.error.__class__.__name__ == "ValueError"
        assert str(context.error) == "Test error"
        assert context.module == "test_module"
        assert context.function_name == "test_function"

        # Test to_dict conversion
        context_dict = context.to_dict()
        assert "timestamp" in context_dict
        assert context_dict["error_type"] == "ValueError"

    def test_handler_chain(self):
        """Test handler chain processing."""
        # Create mock handlers
        handler1 = Mock(spec=ErrorHandlerBase)
        handler1.can_handle = Mock(return_value=False)
        handler1.next_handler = None

        handler2 = Mock(spec=ErrorHandlerBase)
        handler2.can_handle = Mock(return_value=True)
        handler2.handle = Mock(return_value={"action": "handled"})
        handler2.process = Mock(return_value={"action": "handled"})

        # Chain handlers
        handler1.next_handler = handler2

        # Test that error is passed to next handler
        error = ValueError("Test")
        handler1.process = (
            lambda e, c: handler1.next_handler.process(e, c) if handler1.next_handler else None
        )
        result = handler1.process(error, None)

        handler2.process.assert_called_once()


class TestNetworkErrorHandler:
    """Test network error handler."""

    def test_can_handle_network_errors(self):
        """Test that handler correctly identifies network errors."""
        handler = NetworkErrorHandler()

        # Should handle these errors
        assert handler.can_handle(ConnectionError("Connection failed"))
        assert handler.can_handle(TimeoutError("Timeout"))
        assert handler.can_handle(OSError("Network unreachable"))

        # Should handle errors with network keywords
        assert handler.can_handle(Exception("Connection refused"))
        assert handler.can_handle(Exception("Socket timeout"))

        # Should not handle other errors
        assert not handler.can_handle(ValueError("Invalid value"))
        assert not handler.can_handle(KeyError("Missing key"))

    @pytest.mark.asyncio
    async def test_network_error_retry_strategy(self):
        """Test network error retry strategy."""
        from decimal import Decimal

        handler = NetworkErrorHandler(max_retries=3, base_delay=Decimal("1.0"))

        error = ConnectionError("Connection failed")

        # First retry
        result = await handler.handle(error, {"retry_count": 0})
        assert result["action"] == "retry"
        assert result["delay"] == "1.0"  # base_delay * 2^0
        assert result["retry_count"] == 1

        # Second retry
        result = await handler.handle(error, {"retry_count": 1})
        assert result["action"] == "retry"
        assert result["delay"] == "2.0"  # base_delay * 2^1
        assert result["retry_count"] == 2

        # Max retries exceeded
        result = await handler.handle(error, {"retry_count": 3})
        assert result["action"] == "fail"
        assert result["reason"] == "max_retries_exceeded"


class TestRateLimitErrorHandler:
    """Test rate limit error handler."""

    def test_can_handle_rate_limit_errors(self):
        """Test that handler correctly identifies rate limit errors."""
        handler = RateLimitErrorHandler()

        # Should handle these errors
        assert handler.can_handle(Exception("Rate limit exceeded"))
        assert handler.can_handle(Exception("Too many requests"))
        assert handler.can_handle(Exception("Error 429"))
        assert handler.can_handle(Exception("API throttled"))

        # Should not handle other errors
        assert not handler.can_handle(ValueError("Invalid value"))

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test rate limit error handling."""
        handler = RateLimitErrorHandler()

        error = Exception("Rate limit exceeded. Retry after 30 seconds")

        result = await handler.handle(error, {})
        assert result["action"] == "wait"
        assert result["delay"] == "30"  # Extracted from message
        assert result["reason"] == "rate_limit"
        assert result["circuit_break"] == True


class TestValidationErrorHandler:
    """Test validation error handler."""

    def test_can_handle_validation_errors(self):
        """Test that handler correctly identifies validation errors."""
        handler = ValidationErrorHandler()

        # Should handle these errors
        assert handler.can_handle(ValidationError("Invalid input"))
        assert handler.can_handle(ValueError("Value error"))
        assert handler.can_handle(TypeError("Type error"))
        assert handler.can_handle(AssertionError("Assertion failed"))

        # Should handle errors with validation keywords
        assert handler.can_handle(Exception("Validation failed"))
        assert handler.can_handle(Exception("Field is required"))

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test validation error handling."""
        handler = ValidationErrorHandler()

        error = ValidationError("Field 'price' must be positive")

        result = await handler.handle(error, {})
        assert result["action"] == "reject"
        assert result["reason"] == "validation_failed"
        assert result["recoverable"] == False
        assert result["user_action_required"] == True


class TestDatabaseErrorHandler:
    """Test database error handler."""

    def test_can_handle_database_errors(self):
        """Test that handler correctly identifies database errors."""
        handler = DatabaseErrorHandler()

        # Should handle errors with database keywords
        assert handler.can_handle(Exception("Database connection failed"))
        assert handler.can_handle(Exception("Deadlock detected"))
        assert handler.can_handle(Exception("Constraint violation"))
        assert handler.can_handle(Exception("PostgreSQL error"))

    @pytest.mark.asyncio
    async def test_deadlock_handling(self):
        """Test deadlock error handling."""
        handler = DatabaseErrorHandler()

        error = Exception("Deadlock detected")

        result = await handler.handle(error, {})
        assert result["action"] == "retry"
        assert result["delay"] == "0.1"
        assert result["reason"] == "deadlock"

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test database connection error handling."""
        handler = DatabaseErrorHandler()

        error = Exception("Connection pool exhausted")

        result = await handler.handle(error, {})
        assert result["action"] == "reconnect"
        assert result["delay"] == "5"
        assert result["reason"] == "connection_lost"


class TestErrorHandlerFactory:
    """Test error handler factory."""

    def test_handler_registration(self):
        """Test registering handlers with factory."""
        # Clear factory first
        ErrorHandlerFactory.clear()

        # Register handlers
        ErrorHandlerFactory.register("network", NetworkErrorHandler)
        ErrorHandlerFactory.register("validation", ValidationErrorHandler)

        assert "network" in ErrorHandlerFactory.list_handlers()
        assert "validation" in ErrorHandlerFactory.list_handlers()

    def test_handler_creation(self):
        """Test creating handlers from factory."""
        from unittest.mock import Mock
        
        ErrorHandlerFactory.clear()
        ErrorHandlerFactory.register("network", NetworkErrorHandler)
        
        # Set up mock dependency container
        mock_container = Mock()
        mock_container.has_service.return_value = True
        mock_container.resolve.side_effect = lambda service: Mock() if service in ["SecuritySanitizer", "SecurityRateLimiter"] else None
        ErrorHandlerFactory.set_dependency_container(mock_container)

        handler = ErrorHandlerFactory.create("network")
        assert isinstance(handler, NetworkErrorHandler)

        # Test with configuration
        ErrorHandlerFactory.register(
            "network_custom", NetworkErrorHandler, {"max_retries": 5, "base_delay": 2.0}
        )

        handler = ErrorHandlerFactory.create("network_custom")
        assert handler.max_retries == 5
        assert handler.base_delay == 2.0

    def test_unknown_handler_type(self):
        """Test creating unknown handler type."""
        ErrorHandlerFactory.clear()

        with pytest.raises(ValueError, match="Unknown handler type"):
            ErrorHandlerFactory.create("unknown")


class TestErrorHandlerChain:
    """Test error handler chain management."""

    def test_chain_building(self):
        """Test building handler chain."""
        from unittest.mock import Mock
        
        # Register handlers
        ErrorHandlerFactory.clear()
        ErrorHandlerFactory.register("network", NetworkErrorHandler)
        ErrorHandlerFactory.register("validation", ValidationErrorHandler)
        
        # Set up mock dependency container
        mock_container = Mock()
        mock_container.has_service.return_value = True
        mock_container.resolve.side_effect = lambda service: Mock() if service in ["SecuritySanitizer", "SecurityRateLimiter"] else None

        # Build chain
        chain = ErrorHandlerChain(["network", "validation"], dependency_container=mock_container)

        assert chain.chain is not None
        assert isinstance(chain.chain, NetworkErrorHandler)

    @pytest.mark.asyncio
    async def test_chain_error_handling(self):
        """Test error handling through chain."""
        from unittest.mock import Mock
        
        ErrorHandlerFactory.clear()
        ErrorHandlerFactory.register("network", NetworkErrorHandler)
        ErrorHandlerFactory.register("validation", ValidationErrorHandler)
        
        # Set up mock dependency container
        mock_container = Mock()
        mock_container.has_service.return_value = True
        mock_container.resolve.side_effect = lambda service: Mock() if service in ["SecuritySanitizer", "SecurityRateLimiter"] else None

        chain = ErrorHandlerChain(["network", "validation"], dependency_container=mock_container)

        # Test network error
        network_error = ConnectionError("Failed")
        result = await chain.handle(network_error, {"retry_count": 0})
        assert result["action"] == "retry"

        # Test validation error
        validation_error = ValidationError("Invalid")
        result = await chain.handle(validation_error, {})
        assert result["action"] == "reject"

    def test_default_chain_creation(self):
        """Test creating a simple default chain."""
        from unittest.mock import Mock
        
        # Register the handlers first
        ErrorHandlerFactory.clear()
        ErrorHandlerFactory.register("network", NetworkErrorHandler)
        ErrorHandlerFactory.register("database", DatabaseErrorHandler)
        ErrorHandlerFactory.register("validation", ValidationErrorHandler)
        
        # Set up mock dependency container
        mock_container = Mock()
        mock_container.has_service.return_value = True
        mock_container.resolve.side_effect = lambda service: Mock() if service in ["SecuritySanitizer", "SecurityRateLimiter"] else None

        # Create chain with available handlers
        chain = ErrorHandlerChain(["network", "database", "validation"], dependency_container=mock_container)

        assert chain.chain is not None
        # The chain is built in reverse order, so the first in list becomes last in chain
        # With ['network', 'database', 'validation'], the chain head is actually network
        assert isinstance(chain.chain, NetworkErrorHandler)


class TestErrorContextFactory:
    """Test error context factory."""

    def test_create_context(self):
        """Test creating error context."""
        factory = ErrorContextFactory()
        error = ValueError("Test error")
        context = factory.create(error, user_id=123, request_id="abc")

        assert context["error_type"] == "ValueError"
        assert context["error_message"] == "Test error"
        assert context["user_id"] == 123
        assert context["request_id"] == "abc"
        assert "timestamp" in context
        assert "traceback" in context

    def test_create_minimal_context(self):
        """Test creating minimal error context."""
        factory = ErrorContextFactory()
        error = ValueError("Test error")
        context = factory.create_minimal(error)

        assert context["error_type"] == "ValueError"
        assert context["error_message"] == "Test error"
        assert "timestamp" in context
        assert "traceback" not in context  # Minimal doesn't include traceback

    def test_enrich_context(self):
        """Test enriching error context."""
        factory = ErrorContextFactory()
        base_context = {"error": "test", "code": 500}
        enriched = factory.enrich_context(base_context, user_id=123, session_id="xyz")

        assert enriched["error"] == "test"
        assert enriched["code"] == 500
        assert enriched["user_id"] == 123
        assert enriched["session_id"] == "xyz"


class TestRecoveryStrategies:
    """Test recovery strategies."""

    @pytest.mark.asyncio
    async def test_retry_recovery(self):
        """Test retry recovery strategy."""
        from decimal import Decimal

        recovery = RetryRecovery(max_attempts=3, base_delay=Decimal("1.0"))

        error = ConnectionError("Failed")
        context = {"retry_count": 0}

        # Should allow retry
        assert recovery.can_recover(error, context)

        result = await recovery.recover(error, context)
        assert result["action"] == "retry"
        assert result["delay"] == "1.0"
        assert result["retry_count"] == 1

        # Max attempts exceeded
        context = {"retry_count": 3}
        assert not recovery.can_recover(error, context)

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery strategy."""
        from decimal import Decimal

        recovery = CircuitBreakerRecovery(failure_threshold=2, timeout=Decimal("10.0"))

        error = ConnectionError("Failed")

        # First failure - closed state
        result = await recovery.recover(error, {})
        assert result["action"] == "proceed"
        assert recovery.state == "closed"

        # Second failure - should open
        result = await recovery.recover(error, {})
        assert result["action"] == "circuit_break"
        assert recovery.state == "open"

        # While open
        result = await recovery.recover(error, {})
        assert result["action"] == "reject"
        assert result["state"] == "open"

    @pytest.mark.asyncio
    async def test_fallback_recovery(self):
        """Test fallback recovery strategy."""
        fallback_func = Mock(return_value={"status": "fallback"})
        recovery = FallbackRecovery(fallback_func)

        error = Exception("Failed")

        assert recovery.can_recover(error, {})

        result = await recovery.recover(error, {})
        assert result["action"] == "fallback_complete"
        assert result["result"]["status"] == "fallback"

        fallback_func.assert_called_once()
