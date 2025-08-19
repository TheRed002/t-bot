"""Error context factory for standardized error contexts."""

import inspect
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

from src.core.logging import get_logger

logger = get_logger(__name__)


class ErrorContextFactory:
    """Factory for creating standardized error contexts."""
    
    @staticmethod
    def create(
        error: Exception,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create standardized error context.
        
        This method automatically captures:
        - Timestamp
        - Module and function where error occurred
        - Full traceback
        - Error type and message
        - Any additional context provided
        
        Args:
            error: The exception
            **kwargs: Additional context to include
            
        Returns:
            Standardized error context dictionary
        """
        # Get caller frame information
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            module = caller_frame.f_globals.get('__name__', 'unknown')
            function = caller_frame.f_code.co_name
            line = caller_frame.f_lineno
            filename = caller_frame.f_code.co_filename
        else:
            module = 'unknown'
            function = 'unknown'
            line = 0
            filename = 'unknown'
        
        # Build context
        context = {
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'module': module,
            'function': function,
            'line': line,
            'filename': filename,
            'traceback': traceback.format_exc(),
        }
        
        # Add any additional context
        context.update(kwargs)
        
        return context
    
    @staticmethod
    def create_from_frame(
        error: Exception,
        frame: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create error context from a specific frame.
        
        Args:
            error: The exception
            frame: Stack frame to use for context
            **kwargs: Additional context
            
        Returns:
            Error context dictionary
        """
        if frame:
            module = frame.f_globals.get('__name__', 'unknown')
            function = frame.f_code.co_name
            line = frame.f_lineno
            filename = frame.f_code.co_filename
        else:
            module = 'unknown'
            function = 'unknown'
            line = 0
            filename = 'unknown'
        
        context = {
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'module': module,
            'function': function,
            'line': line,
            'filename': filename,
            'traceback': traceback.format_exc(),
        }
        
        context.update(kwargs)
        return context
    
    @staticmethod
    def create_minimal(error: Exception) -> Dict[str, Any]:
        """
        Create minimal error context (for logging with less overhead).
        
        Args:
            error: The exception
            
        Returns:
            Minimal error context
        """
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
        }
    
    @staticmethod
    def enrich_context(
        base_context: Dict[str, Any],
        **additional
    ) -> Dict[str, Any]:
        """
        Enrich existing context with additional information.
        
        Args:
            base_context: Existing context
            **additional: Additional fields to add
            
        Returns:
            Enriched context
        """
        enriched = base_context.copy()
        enriched.update(additional)
        return enriched