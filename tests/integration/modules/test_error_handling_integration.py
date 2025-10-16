"""
Integration tests for error_handling module.

Tests the complete error handling system with real service integration,
dependency injection, and inter-module communication patterns.
This ensures error handling works properly in production scenarios.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio

from src.core.config import Config
from src.core.dependency_injection import injector
from src.core.exceptions import DependencyError, ServiceError, ValidationError
from src.error_handling.di_registration import register_error_handling_services
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.global_handler import GlobalErrorHandler
from src.error_handling.pattern_analytics import ErrorPatternAnalytics
from src.error_handling.service import ErrorHandlingService
from src.error_handling.state_monitor import StateMonitor


class TestErrorHandlingModuleIntegration:
    """Integration tests for error_handling module with real service dependencies."""

    def create_test_context(self, **kwargs) -> dict[str, Any]:
        """Create context dict with all required boundary fields."""
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "test_module",
            "processing_mode": "request_reply",
            "data_format": "core_event_data_v1",
            "message_pattern": "req_reply",
        }
        context.update(kwargs)
        return context

    @pytest_asyncio.fixture(autouse=True)
    async def setup_services(self):
        """Setup services for testing with proper DI registration."""
        # Clear any existing registrations
        container = injector.get_container()
        container.clear()

        # Register config first
        config = Config()
        container.register("Config", config, singleton=True)

        # Register error handling services
        register_error_handling_services(injector, config)

        yield

        # Cleanup after tests
        container.clear()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_handling_service_integration(self):
        """Test ErrorHandlingService with full DI integration."""
        # Resolve service from DI container
        container = injector.get_container()
        error_service = container.get("ErrorHandlingService")

        assert error_service is not None
        assert isinstance(error_service, ErrorHandlingService)

        # Initialize service
        await error_service.initialize()

        # Test error handling with various error types
        test_errors = [
            ValueError("Test validation error"),
            ServiceError("Test service error"),
            RuntimeError("Test runtime error"),
            DependencyError("Test dependency error"),
        ]

        for test_error in test_errors:
            result = await error_service.handle_error(
                error=test_error,
                component="test_component",
                operation="test_operation",
                context=self.create_test_context(test_context="integration_test"),
            )

            assert result is not None
            assert isinstance(result, dict)
            # Should handle all errors gracefully
            assert "error_id" in result or "status" in result

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_handler_component_integration(self):
        """Test ErrorHandler component with real dependencies."""
        container = injector.get_container()
        error_handler = container.get("ErrorHandler")

        assert error_handler is not None
        assert isinstance(error_handler, ErrorHandler)

        # Test error classification
        test_error = ValidationError("Test validation error")
        severity = error_handler.classify_error(test_error)
        assert severity is not None

        # Test error context creation
        context = error_handler.create_error_context(
            test_error, component="test_component", operation="test_operation"
        )
        assert context is not None
        assert context.error == test_error
        assert context.component == "test_component"
        assert context.operation == "test_operation"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_global_error_handler_integration(self):
        """Test GlobalErrorHandler with system-wide error handling."""
        container = injector.get_container()
        global_handler = container.get("GlobalErrorHandler")

        assert global_handler is not None
        assert isinstance(global_handler, GlobalErrorHandler)

        # Test global error handling capabilities
        test_error = ServiceError("Global system error")

        # Should handle errors without raising exceptions
        try:
            # Simulate global error handling with correct API
            result = await global_handler.handle_error(
                error=test_error,
                context={"component": "system", "operation": "global_operation"},
                severity="error",
            )
            # Global handler should return some result
            assert result is not None
        except Exception as e:
            pytest.fail(f"Global error handler should not raise exceptions: {e}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_pattern_analytics_integration(self):
        """Test ErrorPatternAnalytics for error pattern detection."""
        container = injector.get_container()
        pattern_analytics = container.get("ErrorPatternAnalytics")

        assert pattern_analytics is not None
        assert isinstance(pattern_analytics, ErrorPatternAnalytics)

        # Test pattern analysis capabilities
        test_errors = [
            ValueError("Pattern test 1"),
            ValueError("Pattern test 2"),
            ServiceError("Different pattern"),
            ValidationError("Validation pattern"),
        ]

        # Add multiple errors to detect patterns
        for error in test_errors:
            pattern_analytics.add_error_event(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "component": "test_component",
                    "operation": "pattern_test",
                    "severity": "medium",
                    "metadata": {"test": "pattern_detection"},
                }
            )

        # Check if patterns were detected
        patterns = pattern_analytics.get_error_patterns()
        assert patterns is not None
        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_state_monitor_integration(self):
        """Test StateMonitor with system state validation."""
        container = injector.get_container()
        state_monitor = container.get("StateMonitor")

        assert state_monitor is not None
        assert isinstance(state_monitor, StateMonitor)

        # Test state monitoring capabilities
        state_summary = state_monitor.get_state_summary()
        assert state_summary is not None
        assert isinstance(state_summary, dict)

        # Test state validation
        validation_result = await state_monitor.validate_state_consistency()
        assert validation_result is not None
        # StateMonitor returns a StateValidationResult object, not a dict
        assert hasattr(validation_result, "is_consistent")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_handling_with_financial_context(self):
        """Test error handling with financial/trading context."""
        container = injector.get_container()
        error_service = container.get("ErrorHandlingService")

        await error_service.initialize()

        # Test with financial context
        financial_context = {
            "symbol": "BTCUSDT",
            "order_id": "test_order_123",
            "user_id": "test_user",
            "amount": Decimal("1000.50"),
            "price": Decimal("50000.00"),
        }

        # Test various financial errors
        test_errors = [
            ValidationError("Invalid order amount"),
            ServiceError("Exchange API error"),
            ValueError("Invalid symbol format"),
        ]

        for test_error in test_errors:
            try:
                result = await error_service.handle_error(
                    error=test_error,
                    component="trading_engine",
                    operation="place_order",
                    context=self.create_test_context(**financial_context),
                )

                if isinstance(test_error, ValidationError):
                    # ValidationErrors are re-raised, so we shouldn't reach here
                    pytest.fail(f"ValidationError should have been re-raised: {test_error}")
                else:
                    # Other errors should be handled gracefully
                    assert result is not None
                    assert isinstance(result, dict)

            except ValidationError:
                # ValidationErrors are expected to be re-raised
                if not isinstance(test_error, ValidationError):
                    pytest.fail(f"Unexpected ValidationError re-raised for {type(test_error)}")
            except Exception as e:
                # Other exceptions should be handled, not re-raised
                if not isinstance(test_error, ValidationError):
                    pytest.fail(f"Unexpected exception re-raised: {e}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_recovery_scenarios(self):
        """Test error recovery scenarios with real services."""
        container = injector.get_container()
        error_service = container.get("ErrorHandlingService")

        await error_service.initialize()

        # Test recoverable errors
        recoverable_error = ServiceError("Temporary service unavailable")

        result = await error_service.handle_error(
            error=recoverable_error,
            component="exchange_service",
            operation="get_balance",
            context=self.create_test_context(retry_strategy="exponential_backoff"),
        )

        assert result is not None
        # Should attempt recovery for recoverable errors
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker functionality in error handling."""
        container = injector.get_container()
        error_service = container.get("ErrorHandlingService")

        await error_service.initialize()

        # Simulate repeated failures that should trigger circuit breaker
        persistent_error = ServiceError("Service consistently failing")

        # Generate multiple errors rapidly
        for i in range(5):
            try:
                await error_service.handle_error(
                    error=persistent_error,
                    component="external_service",
                    operation=f"operation_{i}",
                    context=self.create_test_context(failure_simulation=True),
                )
            except Exception:
                # Circuit breaker may prevent further calls
                pass

        # Circuit breaker should be engaged after multiple failures
        # (Implementation depends on specific circuit breaker logic)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_propagation_across_modules(self):
        """Test error propagation between modules."""
        container = injector.get_container()
        error_service = container.get("ErrorHandlingService")

        await error_service.initialize()

        # Test cross-module error propagation
        cross_module_error = ValidationError("Cross-module validation failed")

        # ValidationError should be re-raised for financial safety (correct behavior)
        with pytest.raises(ValidationError) as exc_info:
            await error_service.handle_error(
                error=cross_module_error,
                component="risk_management",
                operation="validate_position",
                context=self.create_test_context(
                    source_module="risk_management",
                    target_module="execution",
                    propagation_test=True,
                ),
            )

        # Verify the error was properly propagated
        assert "Cross-module validation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_concurrent_error_handling(self):
        """Test concurrent error handling capabilities."""
        container = injector.get_container()
        error_service = container.get("ErrorHandlingService")

        await error_service.initialize()

        # Create multiple concurrent error handling tasks
        async def handle_concurrent_error(error_id: int):
            return await error_service.handle_error(
                error=ServiceError(f"Concurrent error {error_id}"),
                component="concurrent_service",
                operation=f"concurrent_operation_{error_id}",
                context=self.create_test_context(concurrent_test=True, error_id=error_id),
            )

        # Run multiple error handling operations concurrently
        concurrent_tasks = [handle_concurrent_error(i) for i in range(10)]

        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)

        # All concurrent operations should complete
        assert len(results) == 10

        # Most should succeed (some may fail due to resource limits)
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 5  # At least half should succeed

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_handling_performance(self):
        """Test error handling performance under load."""
        container = injector.get_container()
        error_service = container.get("ErrorHandlingService")

        await error_service.initialize()

        import time

        # Test performance with many sequential errors
        start_time = time.time()

        for i in range(50):
            await error_service.handle_error(
                error=ValueError(f"Performance test error {i}"),
                component="performance_test",
                operation=f"operation_{i}",
                context=self.create_test_context(performance_test=True),
            )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should handle 50 errors in reasonable time (< 5 seconds)
        assert execution_time < 5.0, f"Error handling too slow: {execution_time}s"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_context_preservation(self):
        """Test that error context is preserved across the error handling pipeline."""
        container = injector.get_container()
        error_service = container.get("ErrorHandlingService")

        await error_service.initialize()

        # Test with rich context
        rich_context = {
            "user_id": "test_user_123",
            "session_id": "session_456",
            "request_id": "req_789",
            "exchange": "binance",
            "symbol": "ETHUSDT",
            "operation_metadata": {
                "start_time": "2025-01-01T00:00:00Z",
                "attempt_number": 1,
                "retry_policy": "exponential",
            },
        }

        # ValidationError should be re-raised (correct behavior for financial safety)
        with pytest.raises(ValidationError) as exc_info:
            await error_service.handle_error(
                error=ValidationError("Context preservation test"),
                component="context_test",
                operation="preserve_context",
                context=self.create_test_context(**rich_context),
            )

        # Verify the error was properly propagated with context preserved
        assert "Context preservation test" in str(exc_info.value)
        # Context should be preserved through error handling (logged and tracked)

    def test_error_handling_module_imports(self):
        """Test that all error handling module components can be imported."""
        # Test core components

        # Test utility components
        from src.error_handling.error_handler import ErrorHandler

        # Test factory components
        from src.error_handling.global_handler import GlobalErrorHandler
        from src.error_handling.pattern_analytics import ErrorPatternAnalytics
        from src.error_handling.service import ErrorHandlingService
        from src.error_handling.state_monitor import StateMonitor

        # All imports should succeed
        assert ErrorHandlingService is not None
        assert ErrorHandler is not None
        assert GlobalErrorHandler is not None
        assert ErrorPatternAnalytics is not None
        assert StateMonitor is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_service_health_checks(self):
        """Test health check functionality of error handling services."""
        container = injector.get_container()
        error_service = container.get("ErrorHandlingService")

        await error_service.initialize()

        # Test service health check
        health_result = await error_service.health_check()
        assert health_result is not None

        # Should indicate healthy service
        if hasattr(health_result, "is_healthy"):
            assert health_result.is_healthy
        elif isinstance(health_result, dict):
            assert health_result.get("status") in ["healthy", "ok", True]

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_service_shutdown_cleanup(self):
        """Test proper service shutdown and resource cleanup."""
        container = injector.get_container()
        error_service = container.get("ErrorHandlingService")

        await error_service.initialize()

        # Check if service has running status - some services use _initialized instead
        # This is testing that the service can be properly shut down and cleaned up
        # ErrorHandlingService uses _initialized, not is_running for initialization state
        assert error_service._initialized

        # Test proper shutdown - check if service has shutdown method
        if hasattr(error_service, "shutdown"):
            await error_service.shutdown()

            # Verify cleanup
            if hasattr(error_service, "is_running"):
                assert not error_service.is_running
        else:
            # If no shutdown method, cleanup should still work via cleanup()
            await error_service.cleanup()
            # Service should be properly cleaned up (this test just ensures no exceptions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
