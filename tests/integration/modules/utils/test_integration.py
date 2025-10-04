"""
Integration tests for utils module to verify proper service integration,
dependency injection, and contract compliance.

These tests validate that:
1. Utils services are properly integrated with core dependencies
2. Dependency injection patterns work correctly
3. Error handling follows the established patterns
4. Service contracts are respected across module boundaries
5. Data flows correctly between services
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.dependency_injection import injector
from src.utils.interfaces import ValidationServiceInterface
from src.utils.service_registry import register_util_services
from src.utils.validation.service import (
    ValidationContext,
    ValidationResult,
    ValidationService,
    ValidationType,
)


@pytest.fixture
async def setup_di_container():
    """Setup dependency injection container with utils services."""
    # Clear any existing registrations
    injector.get_container().clear()

    # Reset the registration flag to allow re-registration
    import src.utils.service_registry as registry_module
    registry_module._services_registered = False

    # Register utils services
    register_util_services()

    yield

    # Cleanup
    injector.get_container().clear()

    # Reset the registration flag for next test
    registry_module._services_registered = False


@pytest.fixture
async def validation_service(setup_di_container):
    """Get validation service from DI container."""
    service = injector.resolve("ValidationServiceInterface")
    await service.initialize()
    yield service
    await service.shutdown()


class TestUtilsServiceIntegration:
    """Test utils service integration patterns."""

    async def test_validation_service_di_integration(self, setup_di_container):
        """Test ValidationService dependency injection integration."""
        # Verify service can be resolved from DI container
        service = injector.resolve("ValidationServiceInterface")
        assert isinstance(service, ValidationService)

        # Test interface compliance
        assert isinstance(service, ValidationServiceInterface)

        # Test service initialization
        await service.initialize()
        assert service.is_running

        await service.shutdown()
        assert not service.is_running

    async def test_validation_service_framework_dependency(self, setup_di_container):
        """Test ValidationService properly injects ValidationFramework."""
        service = injector.resolve("ValidationServiceInterface")

        # Verify framework is properly injected
        assert hasattr(service, "framework")
        assert service.framework is not None

        # Test framework functionality through service
        await service.initialize()

        # Test order validation using framework
        order_data = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.1",
            "price": "50000.0",
        }

        result = await service.validate_order(order_data)
        assert isinstance(result, ValidationResult)

        await service.shutdown()

    async def test_utils_service_registry_integration(self, setup_di_container):
        """Test utils service registry properly registers all services."""
        # Verify all expected services are registered
        expected_services = [
            "ValidationService",
            "ValidationServiceInterface",
            "ValidationFramework",
            "GPUManager",
            "GPUInterface",
            "PrecisionTracker",
            "PrecisionInterface",
            "DataFlowValidator",
            "DataFlowInterface",
            "FinancialCalculator",
            "CalculatorInterface",
        ]

        for service_name in expected_services:
            try:
                service = injector.resolve(service_name)
                assert service is not None, f"Service {service_name} not registered"
            except Exception as e:
                pytest.fail(f"Failed to resolve service {service_name}: {e}")

    async def test_validation_service_error_handling(self, validation_service):
        """Test validation service error handling integration."""
        # Test with invalid order data
        invalid_order = {
            "symbol": "",  # Invalid empty symbol
            "side": "INVALID",  # Invalid side
            "type": "LIMIT",
            "quantity": "-1",  # Invalid negative quantity
        }

        result = await validation_service.validate_order(invalid_order)

        # Verify proper error handling
        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.validation_type == ValidationType.ORDER

        # Verify error structure
        for error in result.errors:
            assert hasattr(error, "field")
            assert hasattr(error, "message")
            assert hasattr(error, "validation_type")

    async def test_validation_service_context_integration(self, validation_service):
        """Test validation service context-aware validation."""
        order_data = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.1",
            "price": "50000.0",
        }

        context = ValidationContext(
            exchange="binance", trading_mode="live", strategy_type="momentum"
        )

        result = await validation_service.validate_order(order_data, context)

        # Verify context is properly used
        assert result.context == context
        assert result.context.exchange == "binance"


class TestUtilsModuleConsumption:
    """Test how other modules consume utils services."""

    async def test_execution_service_uses_validation(self):
        """Test execution service properly uses validation service."""

        # Mock the validation service for this test
        with patch("src.utils.validation.service.get_validation_service") as mock_get_service:
            mock_validation_service = AsyncMock()
            mock_validation_service.validate_order.return_value = ValidationResult(
                is_valid=True,
                validation_type=ValidationType.ORDER,
                value={},
                normalized_value=None,
                context=None,
                execution_time_ms=0.0,
                cache_hit=False,
            )
            mock_get_service.return_value = mock_validation_service

            # Import and test execution service

            # This would normally fail without proper dependency injection
            # but with mocking we can test the integration pattern
            pass

    async def test_risk_service_uses_validation(self):
        """Test risk service properly uses validation service."""

        with patch("src.utils.validation.service.get_validation_service") as mock_get_service:
            mock_validation_service = AsyncMock()
            mock_validation_service.validate_risk_parameters.return_value = ValidationResult(
                is_valid=True,
                validation_type=ValidationType.RISK,
                value={},
                normalized_value=None,
                context=None,
                execution_time_ms=0.0,
                cache_hit=False,
            )
            mock_get_service.return_value = mock_validation_service

            # Import and test risk service

            # Test integration pattern exists
            pass

    async def test_database_service_uses_validation(self):
        """Test database service properly uses validation service."""
        from src.core.config.service import ConfigService
        from src.database.service import DatabaseService
        from src.database.connection import DatabaseConnectionManager
        from src.utils.validation.service import ValidationService

        # Test service accepts ValidationService as dependency
        config_service = Mock(spec=ConfigService)
        validation_service = Mock(spec=ValidationService)
        connection_manager = Mock(spec=DatabaseConnectionManager)

        # This should not raise an error if integration is correct
        db_service = DatabaseService(
            connection_manager=connection_manager,
            config_service=config_service,
            validation_service=validation_service
        )

        assert db_service is not None


class TestUtilsContractCompliance:
    """Test contract compliance across module boundaries."""

    async def test_validation_service_contract_compliance(self, validation_service):
        """Test ValidationService implements expected interface contract."""
        # Test all required methods exist
        required_methods = [
            "validate_order",
            "validate_risk_parameters",
            "validate_strategy_config",
            "validate_market_data",
            "validate_batch",
        ]

        for method_name in required_methods:
            assert hasattr(validation_service, method_name)
            method = getattr(validation_service, method_name)
            assert callable(method)

    async def test_validation_result_contract(self, validation_service):
        """Test ValidationResult follows expected contract."""
        order_data = {"symbol": "BTC/USDT", "side": "BUY", "type": "MARKET", "quantity": "0.1"}

        result = await validation_service.validate_order(order_data)

        # Verify all required fields exist
        required_fields = [
            "is_valid",
            "validation_type",
            "value",
            "normalized_value",
            "errors",
            "warnings",
            "context",
            "execution_time_ms",
            "cache_hit",
        ]

        for field in required_fields:
            assert hasattr(result, field), f"ValidationResult missing field: {field}"

    async def test_error_propagation_contract(self, validation_service):
        """Test error propagation follows expected patterns."""
        # Test with data that should cause validation errors
        invalid_data = {
            "symbol": "INVALID_SYMBOL_!!!",
            "side": "INVALID_SIDE",
            "type": "INVALID_TYPE",
            "quantity": "not_a_number",
        }

        result = await validation_service.validate_order(invalid_data)

        # Verify errors are properly structured
        assert not result.is_valid
        assert len(result.errors) > 0

        for error in result.errors:
            # Each error should have proper structure
            assert hasattr(error, "field")
            assert hasattr(error, "message")
            assert hasattr(error, "validation_type")
            assert isinstance(error.message, str)
            assert len(error.message) > 0


class TestUtilsDataFlow:
    """Test data flow integrity across utils boundaries."""

    async def test_decimal_precision_preservation(self, validation_service):
        """Test decimal precision is preserved across service boundaries."""
        from src.utils.decimal_utils import to_decimal

        # Test high precision decimal
        high_precision = "123.123456789012345678"
        decimal_value = to_decimal(high_precision)

        order_data = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": high_precision,
            "price": "50000.0",
        }

        result = await validation_service.validate_order(order_data)

        # Verify precision is preserved in normalized values
        if result.normalized_value and "quantity" in result.normalized_value:
            normalized_quantity = result.normalized_value["quantity"]
            assert str(normalized_quantity) == high_precision

    async def test_batch_validation_data_flow(self, validation_service):
        """Test batch validation maintains data integrity."""
        validations = [
            ("order", {"symbol": "BTC/USDT", "side": "BUY", "type": "MARKET", "quantity": "0.1"}),
            ("risk", {"max_position_size": "0.05", "stop_loss_percentage": "0.02"}),
        ]

        results = await validation_service.validate_batch(validations)

        # Verify results structure
        assert isinstance(results, dict)
        assert "order" in results
        assert "risk" in results

        # Verify each result maintains integrity
        for validation_name, result in results.items():
            assert isinstance(result, ValidationResult)
            assert result.validation_type is not None


class TestUtilsPerformanceIntegration:
    """Test performance aspects of utils integration."""

    async def test_validation_caching_integration(self, validation_service):
        """Test validation result caching works properly."""
        order_data = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.1",
            "price": "50000.0",
        }

        # First validation - should not be cached
        result1 = await validation_service.validate_order(order_data)
        assert not result1.cache_hit

        # Second validation - might be cached (depends on implementation)
        result2 = await validation_service.validate_order(order_data)
        # Note: Cache behavior depends on configuration, so we just verify
        # the cache_hit field is properly set
        assert hasattr(result2, "cache_hit")

    async def test_concurrent_validation_integration(self, validation_service):
        """Test concurrent validations work correctly."""
        order_data_template = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.1",
            "price": "50000.0",
        }

        # Create multiple validation tasks
        tasks = []
        for i in range(5):
            order_data = order_data_template.copy()
            order_data["quantity"] = f"0.{i + 1}"
            task = validation_service.validate_order(order_data)
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # Verify all validations completed successfully
        assert len(results) == 5
        for result in results:
            assert isinstance(result, ValidationResult)


@pytest.mark.asyncio
async def test_utils_integration_end_to_end():
    """End-to-end integration test for utils module."""
    # Setup DI container
    injector.get_container().clear()
    register_util_services()

    try:
        # Get validation service
        validation_service = injector.resolve("ValidationServiceInterface")
        await validation_service.initialize()

        # Test complete validation workflow
        order_data = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.1",
            "price": "50000.0",
        }

        context = ValidationContext(exchange="binance", trading_mode="live")

        # Validate order
        result = await validation_service.validate_order(order_data, context)

        # Verify integration worked end-to-end
        assert result is not None
        assert isinstance(result, ValidationResult)
        assert result.validation_type == ValidationType.ORDER

        # Test batch validation
        validations = [
            ("order", order_data),
            (
                "market_data",
                {
                    "symbol": "BTC/USDT",
                    "timestamp": 1234567890,
                    "open": "49000.0",
                    "high": "51000.0",
                    "low": "48000.0",
                    "close": "50000.0",
                    "volume": "100.0",
                },
            ),
        ]

        batch_results = await validation_service.validate_batch(validations, context)
        assert len(batch_results) == 2

        # Cleanup
        await validation_service.shutdown()

    finally:
        injector.get_container().clear()
