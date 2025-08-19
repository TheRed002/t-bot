"""Validation error handlers."""

from typing import Any, Optional, Dict

from src.error_handling.base import ErrorHandlerBase
from src.core.exceptions import ValidationError


class ValidationErrorHandler(ErrorHandlerBase):
    """Handler for validation errors."""
    
    def can_handle(self, error: Exception) -> bool:
        """Check if this is a validation error."""
        validation_errors = (
            ValidationError,
            ValueError,
            TypeError,
            AssertionError,
        )
        
        # Check for validation-related keywords
        error_msg = str(error).lower()
        validation_keywords = [
            'validation', 'invalid', 'must be', 'should be',
            'required', 'missing', 'format', 'type error'
        ]
        
        return (
            isinstance(error, validation_errors) or
            any(keyword in error_msg for keyword in validation_keywords)
        )
    
    def handle(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Handle validation error.
        
        Validation errors typically should not be retried as they
        indicate incorrect input that needs to be fixed.
        
        Args:
            error: The validation error
            context: Optional context
            
        Returns:
            Recovery action dictionary
        """
        self._logger.error(f"Validation error: {error}")
        
        # Extract field information if available
        field = self._extract_field_name(error, context)
        
        return {
            'action': 'reject',
            'reason': 'validation_failed',
            'error': str(error),
            'field': field,
            'recoverable': False,
            'user_action_required': True
        }
    
    def _extract_field_name(self, error: Exception, context: Optional[Dict]) -> Optional[str]:
        """Try to extract field name from error or context."""
        # Check context first
        if context and 'field' in context:
            return context['field']
        
        # Try to parse from error message
        import re
        error_str = str(error)
        
        # Look for patterns like "field 'price' is invalid"
        match = re.search(r"field\s+['\"]?(\w+)['\"]?", error_str, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for "price must be"
        match = re.search(r"(\w+)\s+must\s+be", error_str, re.IGNORECASE)
        if match:
            return match.group(1)
        
        return None


class DataValidationErrorHandler(ErrorHandlerBase):
    """Handler for data validation errors."""
    
    def can_handle(self, error: Exception) -> bool:
        """Check if this is a data validation error."""
        error_msg = str(error).lower()
        data_keywords = [
            'data', 'schema', 'json', 'parse', 'decode',
            'malformed', 'corrupt'
        ]
        
        return any(keyword in error_msg for keyword in data_keywords)
    
    def handle(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Handle data validation error.
        
        Args:
            error: The data validation error
            context: Optional context
            
        Returns:
            Recovery action dictionary
        """
        self._logger.error(f"Data validation error: {error}")
        
        # Check if we can request fresh data
        if context and context.get('can_refresh', False):
            return {
                'action': 'refresh',
                'reason': 'data_validation_failed',
                'error': str(error)
            }
        
        return {
            'action': 'reject',
            'reason': 'invalid_data',
            'error': str(error),
            'recoverable': False
        }