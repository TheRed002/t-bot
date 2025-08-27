"""
Trace wrapper module for monitoring abstraction.

This module provides a clean abstraction layer for tracing functionality,
ensuring that the rest of the codebase doesn't directly import from opentelemetry.
"""


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
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace.status import Status as OtelStatus, StatusCode as OtelStatusCode

    # Use the real implementations
    trace = otel_trace
    Status = OtelStatus
    StatusCode = OtelStatusCode

except ImportError:
    # Use mock implementations
    trace = MockTrace()  # type: ignore[assignment]
    Status = MockStatus  # type: ignore[misc,assignment]
    StatusCode = MockStatusCode()  # type: ignore[misc,assignment]

__all__ = ["Status", "StatusCode", "trace"]
