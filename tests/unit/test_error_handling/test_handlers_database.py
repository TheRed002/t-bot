"""
Tests for database error handler.

Testing database error detection, handling strategies, and sanitization.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.core.exceptions import DatabaseConnectionError, DatabaseError, DatabaseQueryError
from src.error_handling.handlers.database import DatabaseErrorHandler
from src.error_handling.security_sanitizer import SensitivityLevel


class TestDatabaseErrorHandler:
    """Test database error handler."""

    @pytest.fixture
    def mock_sanitizer(self):
        """Create mock sanitizer."""
        sanitizer = Mock()
        sanitizer.sanitize_error_message.return_value = "sanitized_message"
        return sanitizer

    @pytest.fixture
    def handler(self, mock_sanitizer):
        """Create database error handler with mock sanitizer."""
        return DatabaseErrorHandler(sanitizer=mock_sanitizer)

    def test_initialization_without_sanitizer(self):
        """Test initialization without providing sanitizer."""
        with patch('src.error_handling.security_validator.get_security_sanitizer') as mock_get:
            mock_sanitizer = Mock()
            mock_get.return_value = mock_sanitizer
            
            handler = DatabaseErrorHandler()
            
            assert handler.sanitizer is mock_sanitizer
            mock_get.assert_called_once()

    def test_initialization_with_sanitizer(self, mock_sanitizer):
        """Test initialization with provided sanitizer."""
        handler = DatabaseErrorHandler(sanitizer=mock_sanitizer)
        
        assert handler.sanitizer is mock_sanitizer

    def test_initialization_with_next_handler(self, mock_sanitizer):
        """Test initialization with next handler."""
        next_handler = Mock()
        handler = DatabaseErrorHandler(next_handler=next_handler, sanitizer=mock_sanitizer)
        
        assert handler.next_handler is next_handler

    def test_can_handle_core_database_exceptions(self, handler):
        """Test handling core database exceptions."""
        assert handler.can_handle(DatabaseError("test")) is True
        assert handler.can_handle(DatabaseConnectionError("test")) is True
        assert handler.can_handle(DatabaseQueryError("test")) is True

    def test_can_handle_sqlalchemy_exceptions(self, handler):
        """Test handling SQLAlchemy-like exceptions by name."""
        # Create mock exception classes with specific names
        class OperationalError(Exception):
            pass
        
        class IntegrityError(Exception):
            pass
        
        class DataError(Exception):
            pass
        
        class DatabaseError(Exception):
            pass
        
        class InterfaceError(Exception):
            pass
        
        class InternalError(Exception):
            pass
        
        class ProgrammingError(Exception):
            pass
        
        class NotSupportedError(Exception):
            pass
        
        assert handler.can_handle(OperationalError("Test error")) is True
        assert handler.can_handle(IntegrityError("Test error")) is True
        assert handler.can_handle(DataError("Test error")) is True
        assert handler.can_handle(DatabaseError("Test error")) is True
        assert handler.can_handle(InterfaceError("Test error")) is True
        assert handler.can_handle(InternalError("Test error")) is True
        assert handler.can_handle(ProgrammingError("Test error")) is True
        assert handler.can_handle(NotSupportedError("Test error")) is True

    def test_can_handle_database_keywords(self, handler):
        """Test handling errors with database keywords."""
        assert handler.can_handle(Exception("Database connection failed")) is True
        assert handler.can_handle(Exception("Connection pool exhausted")) is True
        assert handler.can_handle(Exception("Transaction rollback error")) is True
        assert handler.can_handle(Exception("Deadlock detected")) is True
        assert handler.can_handle(Exception("Lock timeout")) is True
        assert handler.can_handle(Exception("Constraint violation")) is True
        assert handler.can_handle(Exception("Duplicate key error")) is True
        assert handler.can_handle(Exception("PostgreSQL error")) is True
        assert handler.can_handle(Exception("MySQL connection error")) is True
        assert handler.can_handle(Exception("SQLite database locked")) is True
        assert handler.can_handle(Exception("Redis connection refused")) is True

    def test_can_handle_non_database_errors(self, handler):
        """Test not handling non-database errors."""
        assert handler.can_handle(ValueError("test")) is False
        assert handler.can_handle(TypeError("test")) is False
        assert handler.can_handle(RuntimeError("test")) is False
        assert handler.can_handle(Exception("generic error")) is False

    @pytest.mark.asyncio
    async def test_handle_deadlock_error(self, handler, mock_sanitizer):
        """Test handling deadlock errors."""
        error = Exception("Deadlock detected in transaction")
        
        with patch.object(handler, '_logger') as mock_logger:
            result = await handler.handle(error)
            
            expected = {
                "action": "retry",
                "delay": "0.1",
                "reason": "deadlock",
                "max_retries": 3,
                "sanitized_error": "sanitized_message",
            }
            
            assert result == expected
            mock_sanitizer.sanitize_error_message.assert_called_with(
                str(error), SensitivityLevel.MEDIUM
            )
            mock_logger.warning.assert_called_once_with(
                "Database deadlock detected: sanitized_message"
            )

    @pytest.mark.asyncio
    async def test_handle_connection_error(self, handler, mock_sanitizer):
        """Test handling connection errors."""
        error = Exception("Connection to database lost")
        
        with patch.object(handler, '_logger') as mock_logger:
            result = await handler.handle(error)
            
            expected = {
                "action": "reconnect",
                "delay": "5",
                "reason": "connection_lost",
                "sanitized_error": "sanitized_message",
            }
            
            assert result == expected
            mock_sanitizer.sanitize_error_message.assert_called_once_with(
                str(error), SensitivityLevel.HIGH
            )
            mock_logger.error.assert_called_once_with(
                "Database connection error: sanitized_message"
            )

    @pytest.mark.asyncio
    async def test_handle_connection_pool_error(self, handler, mock_sanitizer):
        """Test handling connection pool errors."""
        error = Exception("Connection pool exhausted")
        
        with patch.object(handler, '_logger') as mock_logger:
            result = await handler.handle(error)
            
            expected = {
                "action": "reconnect",
                "delay": "5",
                "reason": "connection_lost",
                "sanitized_error": "sanitized_message",
            }
            
            assert result == expected
            mock_sanitizer.sanitize_error_message.assert_called_once_with(
                str(error), SensitivityLevel.HIGH
            )

    @pytest.mark.asyncio
    async def test_handle_connection_closed_error(self, handler, mock_sanitizer):
        """Test handling closed connection errors."""
        error = Exception("Connection is closed")
        
        result = await handler.handle(error)
        
        expected = {
            "action": "reconnect",
            "delay": "5",
            "reason": "connection_lost",
            "sanitized_error": "sanitized_message",
        }
        
        assert result == expected

    @pytest.mark.asyncio
    async def test_handle_constraint_violation(self, handler, mock_sanitizer):
        """Test handling constraint violations."""
        error = Exception("Constraint violation: unique key")
        
        with patch.object(handler, '_logger') as mock_logger:
            result = await handler.handle(error)
            
            expected = {
                "action": "reject",
                "reason": "constraint_violation",
                "sanitized_error": "sanitized_message",
                "recoverable": False,
            }
            
            assert result == expected
            mock_sanitizer.sanitize_error_message.assert_called_with(
                str(error), SensitivityLevel.MEDIUM
            )
            mock_logger.error.assert_called_once_with(
                "Database constraint violation: sanitized_message"
            )

    @pytest.mark.asyncio
    async def test_handle_duplicate_key_error(self, handler, mock_sanitizer):
        """Test handling duplicate key errors."""
        error = Exception("Duplicate key value violates unique constraint")
        
        result = await handler.handle(error)
        
        expected = {
            "action": "reject",
            "reason": "constraint_violation",
            "sanitized_error": "sanitized_message",
            "recoverable": False,
        }
        
        assert result == expected

    @pytest.mark.asyncio
    async def test_handle_unique_constraint_error(self, handler, mock_sanitizer):
        """Test handling unique constraint errors."""
        error = Exception("Unique constraint failed")
        
        result = await handler.handle(error)
        
        expected = {
            "action": "reject",
            "reason": "constraint_violation",
            "sanitized_error": "sanitized_message",
            "recoverable": False,
        }
        
        assert result == expected

    @pytest.mark.asyncio
    async def test_handle_lock_timeout_error(self, handler, mock_sanitizer):
        """Test handling lock timeout errors."""
        error = Exception("Lock wait timeout exceeded")
        
        with patch.object(handler, '_logger') as mock_logger:
            result = await handler.handle(error)
            
            expected = {
                "action": "retry",
                "delay": "2",
                "reason": "lock_timeout",
                "max_retries": 2,
                "sanitized_error": "sanitized_message",
            }
            
            assert result == expected
            mock_sanitizer.sanitize_error_message.assert_called_with(
                str(error), SensitivityLevel.MEDIUM
            )
            mock_logger.warning.assert_called_once_with(
                "Database lock timeout: sanitized_message"
            )

    @pytest.mark.asyncio
    async def test_handle_default_database_error(self, handler, mock_sanitizer):
        """Test handling default database errors."""
        error = DatabaseError("Generic database error")
        
        with patch.object(handler, '_logger') as mock_logger:
            result = await handler.handle(error)
            
            expected = {
                "action": "fail",
                "reason": "database_error",
                "sanitized_error": "sanitized_message",
                "requires_manual_intervention": True,
            }
            
            assert result == expected
            mock_sanitizer.sanitize_error_message.assert_called_once_with(
                str(error), SensitivityLevel.HIGH
            )
            mock_logger.error.assert_called_once_with(
                "Database error: sanitized_message"
            )

    @pytest.mark.asyncio
    async def test_handle_with_context(self, handler, mock_sanitizer):
        """Test handling error with context."""
        error = Exception("Database deadlock")
        context = {"transaction_id": "tx123", "table": "users"}
        
        result = await handler.handle(error, context)
        
        # Context should not affect the result for database errors
        expected = {
            "action": "retry",
            "delay": "0.1",
            "reason": "deadlock",
            "max_retries": 3,
            "sanitized_error": "sanitized_message",
        }
        
        assert result == expected

    @pytest.mark.asyncio
    async def test_handle_case_insensitive_matching(self, handler, mock_sanitizer):
        """Test that error message matching is case insensitive."""
        error_upper = Exception("DEADLOCK DETECTED")
        error_mixed = Exception("Connection POOL Exhausted")
        
        result_upper = await handler.handle(error_upper)
        result_mixed = await handler.handle(error_mixed)
        
        assert result_upper["reason"] == "deadlock"
        assert result_mixed["reason"] == "connection_lost"

    def test_sanitizer_integration(self, mock_sanitizer):
        """Test sanitizer integration."""
        handler = DatabaseErrorHandler(sanitizer=mock_sanitizer)
        
        assert handler.sanitizer is mock_sanitizer

    @pytest.mark.asyncio
    async def test_multiple_keyword_matching(self, handler, mock_sanitizer):
        """Test error with multiple matching keywords."""
        # Should match deadlock first (most specific)
        error = Exception("Database deadlock in connection pool")
        
        result = await handler.handle(error)
        
        # Should prioritize deadlock handling
        assert result["reason"] == "deadlock"
        assert result["action"] == "retry"

    @pytest.mark.asyncio  
    async def test_sanitization_levels(self, handler, mock_sanitizer):
        """Test different sanitization levels are used."""
        # High sensitivity for connection errors
        connection_error = Exception("Connection failed")
        await handler.handle(connection_error)
        mock_sanitizer.sanitize_error_message.assert_called_with(
            str(connection_error), SensitivityLevel.HIGH
        )
        
        mock_sanitizer.reset_mock()
        
        # Medium sensitivity for deadlocks
        deadlock_error = Exception("Deadlock detected")
        await handler.handle(deadlock_error)
        mock_sanitizer.sanitize_error_message.assert_called_with(
            str(deadlock_error), SensitivityLevel.MEDIUM
        )