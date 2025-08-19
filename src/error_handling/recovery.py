"""Recovery strategies using Protocol for standardized interfaces."""

from typing import Protocol, Any, Dict, Optional
from abc import abstractmethod
import asyncio
import time

from src.core.logging import get_logger

logger = get_logger(__name__)


class RecoveryStrategy(Protocol):
    """Protocol for all recovery strategies."""
    
    def can_recover(self, error: Exception, context: Dict) -> bool:
        """Check if this strategy can handle the error."""
        ...
    
    def recover(self, error: Exception, context: Dict) -> Any:
        """Execute recovery logic."""
        ...
    
    @property
    def max_attempts(self) -> int:
        """Maximum recovery attempts."""
        ...


class RetryRecovery:
    """Retry recovery strategy with exponential backoff."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        """
        Initialize retry recovery.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay between retries
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
        """
        self._max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self._logger = get_logger(__name__)
    
    def can_recover(self, error: Exception, context: Dict) -> bool:
        """Check if retry is appropriate."""
        # Check if we've exceeded max attempts
        current_attempts = context.get('retry_count', 0)
        if current_attempts >= self._max_attempts:
            return False
        
        # Check for non-retryable errors
        non_retryable = [
            'validation', 'authentication', 'permission',
            'not found', 'invalid'
        ]
        error_msg = str(error).lower()
        
        return not any(keyword in error_msg for keyword in non_retryable)
    
    def recover(self, error: Exception, context: Dict) -> Dict[str, Any]:
        """Execute retry with exponential backoff."""
        retry_count = context.get('retry_count', 0)
        
        # Calculate delay
        delay = min(
            self.base_delay * (self.exponential_base ** retry_count),
            self.max_delay
        )
        
        self._logger.info(
            f"Retrying after {delay}s (attempt {retry_count + 1}/{self._max_attempts})"
        )
        
        return {
            'action': 'retry',
            'delay': delay,
            'retry_count': retry_count + 1
        }
    
    @property
    def max_attempts(self) -> int:
        """Maximum recovery attempts."""
        return self._max_attempts


class CircuitBreakerRecovery:
    """Circuit breaker recovery strategy."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        half_open_requests: int = 1
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            timeout: Time before attempting half-open state
            half_open_requests: Requests allowed in half-open state
        """
        self._max_attempts = 1  # Circuit breaker doesn't retry
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_requests = half_open_requests
        
        # Circuit breaker state
        self.state = 'closed'  # closed, open, half_open
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_count = 0
        
        self._logger = get_logger(__name__)
    
    def can_recover(self, error: Exception, context: Dict) -> bool:
        """Check if circuit breaker can handle this."""
        # Circuit breaker can handle any error
        return True
    
    def recover(self, error: Exception, context: Dict) -> Dict[str, Any]:
        """Execute circuit breaker logic."""
        current_time = time.time()
        
        if self.state == 'closed':
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                self._logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
                return {
                    'action': 'circuit_break',
                    'state': 'open',
                    'wait_time': self.timeout
                }
            
            return {'action': 'proceed'}
        
        elif self.state == 'open':
            if self.last_failure_time and \
               (current_time - self.last_failure_time) >= self.timeout:
                self.state = 'half_open'
                self.half_open_count = 0
                self._logger.info("Circuit breaker entering half-open state")
                return {'action': 'test', 'state': 'half_open'}
            
            return {
                'action': 'reject',
                'state': 'open',
                'remaining_time': self.timeout - (current_time - self.last_failure_time)
            }
        
        elif self.state == 'half_open':
            self.half_open_count += 1
            
            if context.get('success', False):
                # Success in half-open state, close circuit
                self.state = 'closed'
                self.failure_count = 0
                self._logger.info("Circuit breaker closed after successful test")
                return {'action': 'proceed', 'state': 'closed'}
            
            if self.half_open_count >= self.half_open_requests:
                # Failed in half-open, reopen circuit
                self.state = 'open'
                self.last_failure_time = current_time
                self._logger.warning("Circuit breaker reopened after half-open test failure")
                return {
                    'action': 'circuit_break',
                    'state': 'open',
                    'wait_time': self.timeout
                }
            
            return {'action': 'test', 'state': 'half_open'}
        
        return {'action': 'reject', 'reason': 'unknown_state'}
    
    @property
    def max_attempts(self) -> int:
        """Maximum recovery attempts."""
        return self._max_attempts


class FallbackRecovery:
    """Fallback to alternative implementation."""
    
    def __init__(self, fallback_function: callable, max_attempts: int = 1):
        """
        Initialize fallback recovery.
        
        Args:
            fallback_function: Function to call as fallback
            max_attempts: Maximum fallback attempts
        """
        self.fallback_function = fallback_function
        self._max_attempts = max_attempts
        self._logger = get_logger(__name__)
    
    def can_recover(self, error: Exception, context: Dict) -> bool:
        """Check if fallback is available."""
        return self.fallback_function is not None
    
    def recover(self, error: Exception, context: Dict) -> Any:
        """Execute fallback function."""
        self._logger.info(f"Using fallback for {type(error).__name__}")
        
        try:
            if asyncio.iscoroutinefunction(self.fallback_function):
                # Handle async fallback
                return {
                    'action': 'fallback',
                    'async': True,
                    'function': self.fallback_function
                }
            else:
                # Execute sync fallback
                result = self.fallback_function(**context.get('args', {}))
                return {
                    'action': 'fallback_complete',
                    'result': result
                }
        except Exception as e:
            self._logger.error(f"Fallback failed: {e}")
            return {
                'action': 'fallback_failed',
                'error': str(e)
            }
    
    @property
    def max_attempts(self) -> int:
        """Maximum recovery attempts."""
        return self._max_attempts