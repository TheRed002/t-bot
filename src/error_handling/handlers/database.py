"""Database error handlers."""

from typing import Any, Optional, Dict

from src.error_handling.base import ErrorHandlerBase


class DatabaseErrorHandler(ErrorHandlerBase):
    """Handler for database-related errors."""
    
    def can_handle(self, error: Exception) -> bool:
        """Check if this is a database error."""
        # Check for SQLAlchemy errors
        error_type_name = type(error).__name__
        db_error_types = [
            'OperationalError', 'IntegrityError', 'DataError',
            'DatabaseError', 'InterfaceError', 'InternalError',
            'ProgrammingError', 'NotSupportedError'
        ]
        
        # Check error message for database keywords
        error_msg = str(error).lower()
        db_keywords = [
            'database', 'connection pool', 'transaction',
            'deadlock', 'lock', 'constraint', 'duplicate',
            'postgresql', 'mysql', 'sqlite', 'redis'
        ]
        
        return (
            error_type_name in db_error_types or
            any(keyword in error_msg for keyword in db_keywords)
        )
    
    def handle(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Handle database error with appropriate recovery strategy.
        
        Args:
            error: The database error
            context: Optional context
            
        Returns:
            Recovery action dictionary
        """
        error_msg = str(error).lower()
        
        # Handle deadlock with immediate retry
        if 'deadlock' in error_msg:
            self._logger.warning(f"Database deadlock detected: {error}")
            return {
                'action': 'retry',
                'delay': 0.1,  # Small delay
                'reason': 'deadlock',
                'max_retries': 3
            }
        
        # Handle connection issues
        if any(word in error_msg for word in ['connection', 'pool', 'closed']):
            self._logger.error(f"Database connection error: {error}")
            return {
                'action': 'reconnect',
                'delay': 5,
                'reason': 'connection_lost'
            }
        
        # Handle constraint violations
        if any(word in error_msg for word in ['constraint', 'duplicate', 'unique']):
            self._logger.error(f"Database constraint violation: {error}")
            return {
                'action': 'reject',
                'reason': 'constraint_violation',
                'error': str(error),
                'recoverable': False
            }
        
        # Handle lock timeouts
        if 'lock' in error_msg and 'timeout' in error_msg:
            self._logger.warning(f"Database lock timeout: {error}")
            return {
                'action': 'retry',
                'delay': 2,
                'reason': 'lock_timeout',
                'max_retries': 2
            }
        
        # Default database error handling
        self._logger.error(f"Database error: {error}")
        return {
            'action': 'fail',
            'reason': 'database_error',
            'error': str(error),
            'requires_manual_intervention': True
        }