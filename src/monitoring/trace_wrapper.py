"""
Trace wrapper module for monitoring abstraction.

This module provides a clean abstraction layer for tracing functionality,
ensuring that the rest of the codebase doesn't directly import from opentelemetry.
"""

import functools


# Define fallback classes first
class MockStatus:
    def __init__(self, status_code: str, description: str | None = None) -> None:
        self.status_code = status_code
        self.description = description


class MockStatusCode:
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


class MockTrace:
    def __init__(self) -> None:
        pass

    def status(self, status_code: str, description: str | None = None) -> MockStatus:
        return MockStatus(status_code, description)

    @property
    def status_code(self) -> MockStatusCode:
        return MockStatusCode()


# Try to import from OpenTelemetry
try:
    from opentelemetry import trace as otel_trace_module
    from opentelemetry.trace.status import Status as OtelStatus, StatusCode as OtelStatusCode

    def trace(operation_name: str, *args, **kwargs):
        """Trace decorator using OpenTelemetry."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*func_args, **func_kwargs):
                tracer = otel_trace_module.get_tracer(__name__)
                with tracer.start_as_current_span(operation_name):
                    return func(*func_args, **func_kwargs)
            return wrapper
        return decorator

    Status = OtelStatus
    StatusCode = OtelStatusCode

except ImportError:
    # Use mock implementations
    def trace(operation_name: str, *args, **kwargs):
        """Mock trace decorator."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*func_args, **func_kwargs):
                return func(*func_args, **func_kwargs)
            return wrapper
        return decorator

    Status = MockStatus  # type: ignore[misc,assignment]
    StatusCode = MockStatusCode()  # type: ignore[misc,assignment]

__all__ = ["Status", "StatusCode", "trace"]
