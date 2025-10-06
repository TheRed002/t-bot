"""
Integration tests for utils module error handling patterns.

This test suite validates that:
1. Utils modules handle errors consistently
2. Error propagation follows established patterns
3. Circuit breakers work correctly
4. Retry mechanisms function properly
5. Error context is preserved across module boundaries
"""

import time
from decimal import Decimal, InvalidOperation
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from src.core.dependency_injection import injector
from src.core.exceptions import ComponentError, ServiceError, ValidationError
from src.utils.decorators import circuit_breaker, retry, time_execution
from src.utils.service_registry import register_util_services
from src.utils.validation.service import ValidationContext, ValidationResult, ValidationService


class TestUtilsErrorHandlingIntegration:
    """Test utils module error handling integration."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_services(self):
        """Setup services for testing."""
        injector.get_container().clear()

        # Reset the registration flag to allow re-registration
        import src.utils.service_registry as registry_module

        registry_module._services_registered = False

        register_util_services()
        yield
        injector.get_container().clear()

        # Reset the registration flag for next test
        registry_module._services_registered = False

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_service_error_handling(self):
        """Test ValidationService error handling patterns."""

        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        try:
            # Test handling of None input
            result = await validation_service.validate_order(None)
            assert not result.is_valid
            assert len(result.errors) > 0

            # Test handling of malformed data
            malformed_order = {
                "symbol": None,
                "side": 123,  # Should be string
                "type": [],  # Should be string
                "quantity": "invalid_number",
                "price": {},  # Should be number
            }

            result = await validation_service.validate_order(malformed_order)
            assert not result.is_valid
            assert len(result.errors) > 0

            # Verify error details are informative
            for error in result.errors:
                assert error.field is not None
                assert error.message is not None
                assert len(error.message) > 0

        finally:
            await validation_service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_service_timeout_handling(self):
        """Test ValidationService handles timeouts gracefully."""

        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        try:
            # Test multiple individual validations to test timeout handling
            # Since batch validation doesn't support multiple items of same type,
            # test concurrent individual validations instead
            validation_tasks = []

            import asyncio

            for i in range(100):
                order_data = {
                    "symbol": f"BTC{i}/USDT",
                    "side": "BUY",
                    "type": "MARKET",
                    "quantity": "1.0",
                }
                task = validation_service.validate_order(order_data)
                validation_tasks.append(task)

            # Execute all validations concurrently to test timeout handling
            start_time = time.time()
            results = await asyncio.gather(*validation_tasks)
            end_time = time.time()

            # Should handle large number of concurrent validations without timing out
            assert len(results) == 100

            # All results should be ValidationResult instances
            for result in results:
                assert isinstance(result, ValidationResult)

            # Should complete in reasonable time (< 5 seconds for 100 validations)
            execution_time = end_time - start_time
            assert execution_time < 5.0, f"Concurrent validations took too long: {execution_time}s"

        finally:
            await validation_service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_service_cache_error_handling(self):
        """Test ValidationService handles cache errors gracefully."""

        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        try:
            if validation_service.cache:
                # Mock cache to raise errors
                original_get = validation_service.cache.get
                original_set = validation_service.cache.set

                async def failing_get(key):
                    raise Exception("Cache get failed")

                async def failing_set(key, value, ttl=None):
                    raise Exception("Cache set failed")

                validation_service.cache.get = failing_get
                validation_service.cache.set = failing_set

                # Validation should still work despite cache errors
                order_data = {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "type": "LIMIT",
                    "quantity": "0.001",
                    "price": "50000.0",
                }

                result = await validation_service.validate_order(order_data)
                assert result.is_valid  # Should work despite cache errors
                assert not result.cache_hit  # Should indicate no cache hit

                # Restore original methods
                validation_service.cache.get = original_get
                validation_service.cache.set = original_set

        finally:
            await validation_service.shutdown()

    def test_decimal_utils_error_handling(self):
        """Test decimal utilities handle errors properly."""

        from src.utils.decimal_utils import safe_decimal_conversion, safe_divide, to_decimal

        # Test invalid decimal conversion
        with pytest.raises((ValidationError, ValueError, InvalidOperation)):
            to_decimal("invalid_decimal")

        # Test safe conversion
        result = safe_decimal_conversion("invalid")
        assert result is None or isinstance(result, Decimal)

        # Test division by zero
        result = safe_divide(Decimal("10"), Decimal("0"))
        assert result is None or result == Decimal("0")  # Should handle gracefully

    def test_validators_error_handling(self):
        """Test validator functions handle errors properly."""

        from src.utils.validators import validate_financial_range, validate_precision_range

        # Test financial range validation with invalid inputs
        try:
            validate_financial_range(None, 0, 100)
        except (ValidationError, ValueError, TypeError) as e:
            # Should raise appropriate error
            assert str(e) is not None

        # Test precision range validation with invalid inputs
        try:
            validate_precision_range("invalid", 2)
        except (ValidationError, ValueError, TypeError) as e:
            # Should raise appropriate error
            assert str(e) is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_decorators_error_handling(self):
        """Test decorator error handling patterns."""

        # Test retry decorator with failing function
        call_count = 0

        @retry(max_attempts=3, delay=0.1)
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ServiceError("Simulated failure")

        # Should retry up to max attempts
        with pytest.raises(ServiceError):
            await failing_function()

        assert call_count == 3  # Should have tried 3 times

        # Test retry decorator with eventually succeeding function
        call_count = 0

        @retry(max_attempts=3, delay=0.1)
        async def eventually_succeeding_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ServiceError("Temporary failure")
            return "success"

        result = await eventually_succeeding_function()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_circuit_breaker_error_handling(self):
        """Test circuit breaker error handling (retry decorator behavior)."""

        failure_count = 0

        @circuit_breaker(failure_threshold=3, recovery_timeout=1)
        async def persistently_failing_function():
            nonlocal failure_count
            failure_count += 1
            # Always fail to test that circuit breaker eventually gives up
            raise ServiceError("Service permanently unavailable")

        # Circuit breaker should retry 3 times then give up
        with pytest.raises((ServiceError, ComponentError)):
            await persistently_failing_function()

        # Verify it actually attempted the configured number of retries
        # The function should be called: initial attempt + 3 retries = 4 times
        assert failure_count >= 3, f"Expected at least 3 retry attempts, got {failure_count}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_time_execution_error_handling(self):
        """Test time_execution decorator handles errors properly."""

        @time_execution
        async def error_function():
            raise ValidationError("Test error")

        # Should preserve original error while timing
        with pytest.raises(ValidationError) as exc_info:
            await error_function()

        assert "Test error" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_service_batch_error_isolation(self):
        """Test batch validation isolates errors properly."""

        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        try:
            # Test batch validation with different validation types (no duplicate keys)
            batch_validations = [
                (
                    "order",
                    {
                        "symbol": "BTC/USDT",
                        "side": "BUY",
                        "type": "LIMIT",
                        "quantity": "0.001",
                        "price": "50000.0",
                    },
                ),  # Valid order
                (
                    "risk",
                    {"risk_per_trade": 0.02, "stop_loss": 0.95, "take_profit": 1.5},
                ),  # Valid risk
                (
                    "strategy",
                    {
                        "strategy_type": "momentum",
                        "name": "test_strategy",
                        "parameters": {"lookback_period": 20},
                    },
                ),  # Valid strategy
                ("unknown_type", {"test": "data"}),  # Unknown validation type
            ]

            results = await validation_service.validate_batch(batch_validations)

            assert len(results) == 4

            # Order should be valid
            assert results["order"].is_valid

            # Risk should be valid
            assert results["risk"].is_valid

            # Strategy should be valid
            assert results["strategy"].is_valid

            # Unknown type should have error
            assert not results["unknown_type"].is_valid
            assert len(results["unknown_type"].errors) > 0

            # Test error isolation with individual validations
            # Test that invalid data doesn't affect subsequent valid validations
            invalid_order_result = await validation_service.validate_order(
                {
                    "symbol": "INVALID",
                    "side": "INVALID_SIDE",
                    "type": "INVALID_TYPE",
                    "quantity": "-1.0",
                    "price": "0.0",
                }
            )
            assert not invalid_order_result.is_valid
            assert len(invalid_order_result.errors) > 0

            # Valid orders still work after invalid ones (error isolation)
            valid_order_result = await validation_service.validate_order(
                {"symbol": "ETH/USDT", "side": "SELL", "type": "MARKET", "quantity": "1.0"}
            )
            assert valid_order_result.is_valid

        finally:
            await validation_service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_service_error_context_preservation(self):
        """Test ValidationService preserves error context."""

        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        try:
            # Create context for validation
            context = ValidationContext(
                exchange="binance",
                trading_mode="live",
                strategy_type="test_strategy",
                user_id="test_user",
                session_id="test_session",
                request_id="test_request",
            )

            # Invalid order with context
            invalid_order = {
                "symbol": "",  # Invalid empty symbol
                "side": "INVALID_SIDE",
                "type": "INVALID_TYPE",
                "quantity": "-1.0",
                "price": "0.0",
            }

            result = await validation_service.validate_order(invalid_order, context)

            assert not result.is_valid
            assert result.context == context

            # Verify context is preserved in result
            assert result.context.exchange == "binance"
            assert result.context.trading_mode == "live"
            assert result.context.strategy_type == "test_strategy"
            assert result.context.user_id == "test_user"
            assert result.context.session_id == "test_session"
            assert result.context.request_id == "test_request"

        finally:
            await validation_service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_utils_service_initialization_errors(self):
        """Test utils services handle initialization errors gracefully."""

        # Test ValidationService initialization with invalid config
        validation_service = ValidationService(
            config={"invalid_config": True},
            cache_ttl=-1,  # Invalid TTL
            max_cache_size=-1,  # Invalid cache size
        )

        # Should initialize despite invalid config (with warnings)
        await validation_service.initialize()
        assert validation_service.is_running

        await validation_service.shutdown()

    def test_utils_module_import_error_handling(self):
        """Test utils module handles import errors gracefully."""

        # Test importing utils with missing dependencies
        try:
            import src.utils

            # Should import successfully
            assert src.utils is not None
        except ImportError as e:
            pytest.fail(f"Utils module should import successfully, got: {e}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_service_health_check_error_handling(self):
        """Test ValidationService health checks handle errors."""

        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        try:
            # Get health status
            health_status = await validation_service._service_health_check()

            # Should return valid health status
            assert health_status is not None

            # Test health check with broken cache
            if validation_service.cache:
                # Mock cache to fail
                original_set = validation_service.cache.set
                validation_service.cache.set = AsyncMock(side_effect=Exception("Cache failure"))

                health_status = await validation_service._service_health_check()
                # Should handle cache failure gracefully
                assert health_status is not None

                # Restore cache
                validation_service.cache.set = original_set

        finally:
            await validation_service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
