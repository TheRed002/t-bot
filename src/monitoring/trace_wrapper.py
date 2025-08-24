"""
Trace wrapper module for monitoring abstraction.

This module provides a clean abstraction layer for tracing functionality,
ensuring that the rest of the codebase doesn't directly import from opentelemetry.
"""

from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.trace.status import Status, StatusCode
    
    # Re-export commonly used trace components
    __all__ = ['trace', 'Status', 'StatusCode']
    
except ImportError:
    # Fallback for when OpenTelemetry is not available
    class MockStatus:
        def __init__(self, status_code, description=None):
            self.status_code = status_code
            self.description = description
    
    class MockStatusCode:
        OK = "ok"
        ERROR = "error"
        UNSET = "unset"
    
    class MockTrace:
        def Status(self, status_code, description=None):
            return MockStatus(status_code, description)
        
        StatusCode = MockStatusCode()
    
    trace = MockTrace()
    Status = MockStatus
    StatusCode = MockStatusCode()
    
    __all__ = ['trace', 'Status', 'StatusCode']