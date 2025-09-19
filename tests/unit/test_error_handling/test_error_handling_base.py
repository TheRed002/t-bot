"""
Tests for error handling base classes.

Testing Chain of Responsibility pattern implementation and base error handler functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.error_handling.base import ErrorHandlerBase


class ConcreteErrorHandler(ErrorHandlerBase):
    """Concrete implementation of ErrorHandlerBase for testing."""
    
    def __init__(self, can_handle_func=None, handle_func=None, next_handler=None):
        super().__init__(next_handler)
        self._can_handle_func = can_handle_func
        self._handle_func = handle_func
    
    def can_handle(self, error: Exception) -> bool:
        if self._can_handle_func:
            return self._can_handle_func(error)
        return isinstance(error, ValueError)
    
    async def handle(self, error: Exception, context=None):
        if self._handle_func:
            return await self._handle_func(error, context)
        return "handled_by_concrete"


class TestErrorHandlerBase:
    """Test base error handler class."""

    def test_error_handler_base_is_abstract(self):
        """Test that ErrorHandlerBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ErrorHandlerBase()

    def test_concrete_handler_initialization_without_next(self):
        """Test concrete handler initialization without next handler."""
        handler = ConcreteErrorHandler()
        
        assert handler.next_handler is None
        assert hasattr(handler, '_logger')

    def test_concrete_handler_initialization_with_next(self):
        """Test concrete handler initialization with next handler."""
        next_handler = ConcreteErrorHandler()
        handler = ConcreteErrorHandler(next_handler=next_handler)
        
        assert handler.next_handler is next_handler

    def test_can_handle_abstract_method(self):
        """Test that can_handle is properly implemented in concrete class."""
        handler = ConcreteErrorHandler()
        
        assert handler.can_handle(ValueError("test")) is True
        assert handler.can_handle(RuntimeError("test")) is False

    @pytest.mark.asyncio
    async def test_handle_abstract_method(self):
        """Test that handle is properly implemented in concrete class."""
        handler = ConcreteErrorHandler()
        
        result = await handler.handle(ValueError("test"))
        assert result == "handled_by_concrete"

    @pytest.mark.asyncio
    async def test_handle_with_context(self):
        """Test handle method with context parameter."""
        async def handle_func(error, context):
            return f"handled_with_context_{context['key']}"
        
        handler = ConcreteErrorHandler(handle_func=handle_func)
        
        result = await handler.handle(ValueError("test"), context={"key": "value"})
        assert result == "handled_with_context_value"

    @pytest.mark.asyncio
    async def test_process_success_single_handler(self):
        """Test processing error with single handler that can handle it."""
        handler = ConcreteErrorHandler()
        
        with patch.object(handler, '_logger') as mock_logger:
            result = await handler.process(ValueError("test"))
            
            assert result == "handled_by_concrete"
            mock_logger.debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_success_with_context(self):
        """Test processing error with context."""
        handler = ConcreteErrorHandler()
        
        result = await handler.process(ValueError("test"), context={"test": "context"})
        assert result == "handled_by_concrete"

    @pytest.mark.asyncio
    async def test_process_chain_to_next_handler(self):
        """Test processing error that chains to next handler."""
        # First handler can't handle RuntimeError
        second_handler = ConcreteErrorHandler(
            can_handle_func=lambda e: isinstance(e, RuntimeError),
            handle_func=AsyncMock(return_value="handled_by_second")
        )
        first_handler = ConcreteErrorHandler(next_handler=second_handler)
        
        with patch.object(first_handler, '_logger'):
            with patch.object(second_handler, '_logger') as mock_logger:
                result = await first_handler.process(RuntimeError("test"))
                
                assert result == "handled_by_second"
                mock_logger.debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_no_handler_raises_error(self):
        """Test processing error when no handler can handle it."""
        handler = ConcreteErrorHandler()
        error = RuntimeError("unhandled error")
        
        with patch.object(handler, '_logger') as mock_logger:
            with pytest.raises(RuntimeError, match="unhandled error"):
                await handler.process(error)
            
            mock_logger.warning.assert_called_once_with(
                f"No handler found for {type(error).__name__}"
            )

    @pytest.mark.asyncio
    async def test_process_chain_no_handler_raises_error(self):
        """Test processing error in chain when no handler can handle it."""
        second_handler = ConcreteErrorHandler()  # Only handles ValueError
        first_handler = ConcreteErrorHandler(next_handler=second_handler)  # Only handles ValueError
        error = TypeError("unhandled error")
        
        with patch.object(second_handler, '_logger') as mock_logger:
            with pytest.raises(TypeError, match="unhandled error"):
                await first_handler.process(error)
            
            mock_logger.warning.assert_called_once_with(
                f"No handler found for {type(error).__name__}"
            )

    def test_set_next_handler(self):
        """Test setting next handler in chain."""
        first_handler = ConcreteErrorHandler()
        second_handler = ConcreteErrorHandler()
        
        result = first_handler.set_next(second_handler)
        
        assert first_handler.next_handler is second_handler
        assert result is second_handler

    def test_set_next_handler_chaining(self):
        """Test handler chaining with set_next."""
        first_handler = ConcreteErrorHandler()
        second_handler = ConcreteErrorHandler()
        third_handler = ConcreteErrorHandler()
        
        first_handler.set_next(second_handler).set_next(third_handler)
        
        assert first_handler.next_handler is second_handler
        assert second_handler.next_handler is third_handler
        assert third_handler.next_handler is None

    @pytest.mark.asyncio
    async def test_complex_chain_processing(self):
        """Test complex chain processing with multiple handlers."""
        # Third handler handles TypeError
        third_handler = ConcreteErrorHandler(
            can_handle_func=lambda e: isinstance(e, TypeError),
            handle_func=AsyncMock(return_value="handled_by_third")
        )
        
        # Second handler handles RuntimeError
        second_handler = ConcreteErrorHandler(
            can_handle_func=lambda e: isinstance(e, RuntimeError),
            handle_func=AsyncMock(return_value="handled_by_second"),
            next_handler=third_handler
        )
        
        # First handler handles ValueError
        first_handler = ConcreteErrorHandler(next_handler=second_handler)
        
        # Test each handler in the chain
        result1 = await first_handler.process(ValueError("test1"))
        assert result1 == "handled_by_concrete"
        
        result2 = await first_handler.process(RuntimeError("test2"))
        assert result2 == "handled_by_second"
        
        result3 = await first_handler.process(TypeError("test3"))
        assert result3 == "handled_by_third"

    def test_logger_initialization(self):
        """Test that logger is properly initialized."""
        handler = ConcreteErrorHandler()
        
        assert hasattr(handler, '_logger')
        assert handler._logger is not None

    @pytest.mark.asyncio
    async def test_custom_can_handle_function(self):
        """Test custom can_handle function."""
        def custom_can_handle(error):
            return "custom" in str(error)
        
        handler = ConcreteErrorHandler(can_handle_func=custom_can_handle)
        
        assert handler.can_handle(ValueError("custom error")) is True
        assert handler.can_handle(ValueError("regular error")) is False

    @pytest.mark.asyncio
    async def test_custom_handle_function(self):
        """Test custom handle function."""
        async def custom_handle(error, context):
            return f"custom_handled_{type(error).__name__}"
        
        handler = ConcreteErrorHandler(handle_func=custom_handle)
        
        result = await handler.handle(ValueError("test"))
        assert result == "custom_handled_ValueError"

    @pytest.mark.asyncio
    async def test_handle_with_none_context(self):
        """Test handle method with None context."""
        handler = ConcreteErrorHandler()
        
        result = await handler.handle(ValueError("test"), context=None)
        assert result == "handled_by_concrete"


class TestErrorHandlerBaseExports:
    """Test module exports."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from src.error_handling.base import __all__
        
        expected_exports = ["ErrorHandlerBase"]
        
        for export in expected_exports:
            assert export in __all__

    def test_error_handler_base_import(self):
        """Test ErrorHandlerBase can be imported."""
        from src.error_handling.base import ErrorHandlerBase
        
        assert ErrorHandlerBase is not None
        assert hasattr(ErrorHandlerBase, 'can_handle')
        assert hasattr(ErrorHandlerBase, 'handle')
        assert hasattr(ErrorHandlerBase, 'process')
        assert hasattr(ErrorHandlerBase, 'set_next')