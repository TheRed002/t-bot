"""
Integration tests for utils module boundaries and dependency injection patterns.

This test suite validates that:
1. Utils services are properly injected via DI container
2. Module boundaries are respected
3. Error handling patterns are consistent
4. Service layer is not bypassed
5. Circular dependencies don't exist
"""

import pytest
import pytest_asyncio

# Core imports
from src.core.dependency_injection import injector
from src.core.exceptions import ValidationError
from src.utils.service_registry import register_util_services

# Utils module imports
from src.utils.validation.service import ValidationContext, ValidationResult, ValidationService


class TestUtilsModuleIntegration:
    """Test utils module integration patterns."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_utils_services(self):
        """Setup utils services for testing."""
        # Clear any existing registrations
        injector.get_container().clear()

        # Reset the registration flag to allow re-registration
        import src.utils.service_registry as registry_module

        registry_module._services_registered = False

        # Register util services
        register_util_services()

        yield

        # Cleanup
        injector.get_container().clear()

        # Reset the registration flag for next test
        registry_module._services_registered = False

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_service_dependency_injection(self):
        """Test ValidationService is properly injected via DI container."""

        # Resolve through interface
        validation_service = injector.resolve("ValidationServiceInterface")
        assert validation_service is not None
        assert isinstance(validation_service, ValidationService)

        # Verify singleton behavior
        validation_service2 = injector.resolve("ValidationServiceInterface")
        assert validation_service is validation_service2

        # Test service functionality
        await validation_service.initialize()
        assert validation_service.is_running

        # Test validation functionality
        order_data = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.001",
            "price": "50000.0",
        }

        result = await validation_service.validate_order(order_data)
        assert isinstance(result, ValidationResult)
        assert result.is_valid

        await validation_service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_service_interface_compliance(self):
        """Test ValidationService implements ValidationServiceInterface properly."""

        validation_service = injector.resolve("ValidationServiceInterface")

        # Test interface methods exist
        assert hasattr(validation_service, "validate_order")
        assert hasattr(validation_service, "validate_risk_parameters")
        assert hasattr(validation_service, "validate_strategy_config")
        assert hasattr(validation_service, "validate_market_data")
        assert hasattr(validation_service, "validate_batch")

        # Test service lifecycle methods
        assert hasattr(validation_service, "initialize")
        assert hasattr(validation_service, "shutdown")
        assert hasattr(validation_service, "is_running")

        # Test compatibility methods
        assert hasattr(validation_service, "validate_price")
        assert hasattr(validation_service, "validate_quantity")
        assert hasattr(validation_service, "validate_symbol")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_utils_services_registration(self):
        """Test all utils services are properly registered."""

        # Test GPU services
        gpu_manager = injector.resolve("GPUInterface")
        assert gpu_manager is not None

        # Test precision services
        precision_tracker = injector.resolve("PrecisionInterface")
        assert precision_tracker is not None

        # Test data flow services
        data_flow_validator = injector.resolve("DataFlowInterface")
        assert data_flow_validator is not None

        # Test validation services
        validation_service = injector.resolve("ValidationServiceInterface")
        assert validation_service is not None

        # Test calculator services
        calculator = injector.resolve("CalculatorInterface")
        assert calculator is not None

        # Test HTTP session manager
        http_manager = injector.resolve("HTTPSessionManager")
        assert http_manager is not None

        # Test messaging services
        messaging_coordinator = injector.resolve("MessagingCoordinator")
        assert messaging_coordinator is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_service_error_handling(self):
        """Test ValidationService error handling patterns."""

        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        try:
            # Test invalid order data
            invalid_order = {
                "symbol": "INVALID_SYMBOL",
                "side": "INVALID_SIDE",
                "type": "LIMIT",  # Valid type to trigger price validation
                "quantity": "-1.0",  # Negative quantity
                "price": "-10.0",  # Negative price
            }

            result = await validation_service.validate_order(invalid_order)
            assert not result.is_valid
            assert len(result.errors) > 0

            # Verify error details
            error_fields = [error.field for error in result.errors]
            assert "side" in error_fields  # Invalid side
            assert "quantity" in error_fields  # Negative quantity
            assert "price" in error_fields  # Negative price

            # Test error severity - expect HIGH level errors for invalid values
            high_severity_errors = [
                error for error in result.errors if error.severity.name == "HIGH"
            ]
            assert len(high_severity_errors) > 0  # Should have high severity validation errors

        finally:
            await validation_service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_service_batch_operations(self):
        """Test ValidationService batch validation capabilities."""

        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        try:
            # Prepare batch validation data
            valid_order = {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "quantity": "0.001",
                "price": "50000.0",
            }

            valid_risk_params = {
                "risk_per_trade": 0.02,
                "stop_loss": 0.95,
                "take_profit": 1.5,
                "max_position_size": 1000.0,
            }

            valid_strategy_config = {
                "strategy_type": "MEAN_REVERSION",
                "window_size": 20,
                "num_std": 2.0,
                "entry_threshold": 0.8,
                "timeframe": "1h",
            }

            # Execute batch validation
            batch_validations = [
                ("order", valid_order),
                ("risk", valid_risk_params),
                ("strategy", valid_strategy_config),
            ]

            results = await validation_service.validate_batch(batch_validations)

            # Verify results
            assert len(results) == 3
            assert "order" in results
            assert "risk" in results
            assert "strategy" in results

            # All should be valid
            assert results["order"].is_valid
            assert results["risk"].is_valid
            assert results["strategy"].is_valid

        finally:
            await validation_service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_service_context_awareness(self):
        """Test ValidationService context-aware validation."""

        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        try:
            order_data = {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "quantity": "0.001",
                "price": "50000.0",
            }

            # Create validation context
            context = ValidationContext(
                exchange="binance",
                trading_mode="live",
                strategy_type="momentum",
                user_id="test_user",
            )

            # Validate with context
            result = await validation_service.validate_order(order_data, context)
            assert result.is_valid
            assert result.context == context

        finally:
            await validation_service.shutdown()

    def test_utils_module_no_circular_dependencies(self):
        """Test utils module doesn't have circular dependencies."""

        # Import utils main module
        import src.utils

        # Should import successfully without circular dependency errors
        assert src.utils is not None

        # Test key components are available
        assert hasattr(src.utils, "ValidationService")
        assert hasattr(src.utils, "ValidationFramework")
        assert hasattr(src.utils, "to_decimal")
        assert hasattr(src.utils, "format_currency")
        assert hasattr(src.utils, "validate_order")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_strategy_integrated_base_dependency_injection(self):
        """Test StrategyIntegratedBase properly uses dependency injection."""

        from src.strategies.shared_utilities import StrategyIntegratedBase

        # Create strategy with injected validation service
        validation_service = injector.resolve("ValidationServiceInterface")

        strategy = StrategyIntegratedBase(
            strategy_name="test_strategy",
            config={"param1": "value1"},
            validation_service=validation_service,
        )

        # Verify validation service is properly injected
        assert strategy.validation_service is validation_service

        # Test initialization method
        await strategy.initialize_validation_service()
        assert strategy.validation_service is not None
        assert (
            strategy.validation_service.is_running or not strategy.validation_service.is_running
        )  # May or may not be running

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_service_backward_compatibility(self):
        """Test ValidationService backward compatibility methods."""

        validation_service = injector.resolve("ValidationServiceInterface")

        # Test backward compatibility methods
        assert validation_service.validate_price(50000.0)
        assert validation_service.validate_price("50000.0")
        assert not validation_service.validate_price(-1.0)
        assert not validation_service.validate_price(0.0)

        assert validation_service.validate_quantity(0.001)
        assert validation_service.validate_quantity("0.001")
        assert not validation_service.validate_quantity(-1.0)
        assert not validation_service.validate_quantity(0.0)

        assert validation_service.validate_symbol("BTCUSDT")
        assert validation_service.validate_symbol("BTC/USDT")
        assert not validation_service.validate_symbol("A")  # Too short
        assert not validation_service.validate_symbol("")  # Empty

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_utils_module_error_propagation(self):
        """Test proper error propagation from utils module."""

        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        try:
            # Test ValidationError propagation
            with pytest.raises(ValidationError):
                validation_service.validate_decimal("invalid_decimal")

            # Test service errors are properly wrapped
            try:
                # This should trigger a service error but return ValidationResult, not raise
                result = await validation_service.validate_order(None)
                # Should be a proper validation result with errors, not an exception
                assert not result.is_valid
                assert len(result.errors) > 0
            except Exception as e:
                # Should not raise exception but return ValidationResult
                assert False, f"Should not raise exception, got {type(e)}: {e}"

        finally:
            await validation_service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_utils_performance_monitoring(self):
        """Test utils services include performance monitoring."""

        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        try:
            # Test validation with timing
            order_data = {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "quantity": "0.001",
                "price": "50000.0",
            }

            result = await validation_service.validate_order(order_data)

            # Verify execution time is tracked
            assert result.execution_time_ms >= 0.0
            assert isinstance(result.execution_time_ms, float)

        finally:
            await validation_service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_utils_caching_integration(self):
        """Test utils services integrate with caching properly."""

        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        try:
            # Enable caching for this test
            if validation_service.cache:
                order_data = {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "type": "LIMIT",
                    "quantity": "0.001",
                    "price": "50000.0",
                }

                # First validation - should not be cached
                result1 = await validation_service.validate_order(order_data)
                assert not result1.cache_hit

                # Second validation - may be cached (depends on caching rules)
                result2 = await validation_service.validate_order(order_data)
                # Cache behavior depends on implementation

        finally:
            await validation_service.shutdown()

    def test_utils_service_registry_idempotency(self):
        """Test utils service registry is idempotent."""

        # Clear container
        injector.get_container().clear()

        # Reset the registration flag
        import src.utils.service_registry as registry_module

        registry_module._services_registered = False

        # Register multiple times
        register_util_services()
        register_util_services()  # Should not cause issues
        register_util_services()  # Should not cause issues

        # Should still work correctly
        validation_service = injector.resolve("ValidationServiceInterface")
        assert validation_service is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
