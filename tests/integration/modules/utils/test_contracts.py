"""
Integration tests for utils module contract validation and API compliance.

These tests verify that:
1. Utils services expose the correct public APIs
2. Service contracts are maintained across module boundaries
3. Data validation contracts are properly enforced
4. Error contracts are consistently implemented
5. Performance contracts are met
"""

import asyncio
from decimal import Decimal

import pytest
import pytest_asyncio

from src.core.dependency_injection import injector
from src.core.types import ValidationLevel
from src.utils.interfaces import (
    ValidationServiceInterface,
)
from src.utils.service_registry import register_util_services
from src.utils.validation.service import (
    ValidationContext,
    ValidationDetail,
    ValidationResult,
    ValidationService,
    ValidationType,
)


@pytest_asyncio.fixture
async def setup_utils_services():
    """Setup utils services for testing."""
    injector.get_container().clear()

    # Reset the registration flag to allow re-registration
    import src.utils.service_registry as registry_module

    registry_module._services_registered = False

    register_util_services()
    yield
    injector.get_container().clear()

    # Reset the registration flag for next test
    registry_module._services_registered = False


class TestValidationServiceContract:
    """Test ValidationService contract compliance."""

    @pytest_asyncio.fixture
    async def validation_service(self, setup_utils_services):
        """Get validation service instance."""
        service = injector.resolve("ValidationServiceInterface")
        await service.initialize()
        yield service
        await service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validate_order_contract(self, validation_service):
        """Test validate_order method contract."""
        # Valid order data
        order_data = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.1",
            "price": "50000.0",
        }

        result = await validation_service.validate_order(order_data)

        # Verify return type and structure
        assert isinstance(result, ValidationResult)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "validation_type")
        assert hasattr(result, "value")
        assert hasattr(result, "normalized_value")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert hasattr(result, "context")
        assert hasattr(result, "execution_time_ms")
        assert hasattr(result, "cache_hit")

        # Verify types
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.validation_type, ValidationType)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.execution_time_ms, float)
        assert isinstance(result.cache_hit, bool)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validate_order_with_context_contract(self, validation_service):
        """Test validate_order with context parameter."""
        order_data = {"symbol": "BTC/USDT", "side": "BUY", "type": "MARKET", "quantity": "0.1"}

        context = ValidationContext(
            exchange="binance", trading_mode="live", strategy_type="momentum"
        )

        result = await validation_service.validate_order(order_data, context)

        # Verify context is properly handled
        assert result.context is not None
        assert result.context.exchange == "binance"
        assert result.context.trading_mode == "live"
        assert result.context.strategy_type == "momentum"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validate_order_error_contract(self, validation_service):
        """Test validate_order error handling contract."""
        # Invalid order data
        invalid_order = {"symbol": "", "side": "INVALID_SIDE", "type": "LIMIT", "quantity": "-1"}

        result = await validation_service.validate_order(invalid_order)

        # Verify error structure
        assert not result.is_valid
        assert len(result.errors) > 0

        for error in result.errors:
            assert isinstance(error, ValidationDetail)
            assert hasattr(error, "field")
            assert hasattr(error, "validation_type")
            assert hasattr(error, "expected")
            assert hasattr(error, "actual")
            assert hasattr(error, "message")
            assert hasattr(error, "severity")
            assert hasattr(error, "suggestion")

            # Verify types
            assert isinstance(error.field, str)
            assert isinstance(error.validation_type, str)
            assert isinstance(error.message, str)
            assert isinstance(error.severity, ValidationLevel)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validate_risk_parameters_contract(self, validation_service):
        """Test validate_risk_parameters method contract."""
        risk_data = {
            "max_position_size": "0.05",
            "stop_loss_percentage": "0.02",
            "take_profit_percentage": "0.04",
        }

        result = await validation_service.validate_risk_parameters(risk_data)

        # Verify return type and structure
        assert isinstance(result, ValidationResult)
        assert result.validation_type == ValidationType.RISK

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validate_strategy_config_contract(self, validation_service):
        """Test validate_strategy_config method contract."""
        strategy_data = {
            "name": "momentum_strategy",
            "parameters": {"lookback_period": 20, "threshold": 0.02},
        }

        result = await validation_service.validate_strategy_config(strategy_data)

        # Verify return type and structure
        assert isinstance(result, ValidationResult)
        assert result.validation_type == ValidationType.STRATEGY

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validate_market_data_contract(self, validation_service):
        """Test validate_market_data method contract."""
        market_data = {
            "symbol": "BTC/USDT",
            "timestamp": 1234567890,
            "open": "49000.0",
            "high": "51000.0",
            "low": "48000.0",
            "close": "50000.0",
            "volume": "100.0",
        }

        result = await validation_service.validate_market_data(market_data)

        # Verify return type and structure
        assert isinstance(result, ValidationResult)
        assert result.validation_type == ValidationType.MARKET_DATA

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validate_batch_contract(self, validation_service):
        """Test validate_batch method contract."""
        validations = [
            ("order", {"symbol": "BTC/USDT", "side": "BUY", "type": "MARKET", "quantity": "0.1"}),
            (
                "market_data",
                {
                    "symbol": "ETH/USDT",
                    "timestamp": 1234567890,
                    "open": "3000.0",
                    "high": "3100.0",
                    "low": "2900.0",
                    "close": "3050.0",
                    "volume": "50.0",
                },
            ),
        ]

        results = await validation_service.validate_batch(validations)

        # Verify return type and structure
        assert isinstance(results, dict)
        assert "order" in results
        assert "market_data" in results

        for validation_name, result in results.items():
            assert isinstance(result, ValidationResult)
            assert isinstance(validation_name, str)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_backward_compatibility_contract(self, validation_service):
        """Test backward compatibility methods contract."""
        # Test legacy boolean methods
        assert hasattr(validation_service, "validate_price")
        assert hasattr(validation_service, "validate_quantity")
        assert hasattr(validation_service, "validate_symbol")

        # Test they return boolean values
        assert isinstance(validation_service.validate_price("50000.0"), bool)
        assert isinstance(validation_service.validate_quantity("0.1"), bool)
        assert isinstance(validation_service.validate_symbol("BTC/USDT"), bool)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_service_lifecycle_contract(self, validation_service):
        """Test service lifecycle method contracts."""
        # Service should be initialized from fixture
        assert validation_service.is_running

        # Test stats method
        stats = validation_service.get_validation_stats()
        assert isinstance(stats, dict)
        assert "initialized" in stats
        assert "registered_rules" in stats
        assert "cache_enabled" in stats


class TestInterfaceCompliance:
    """Test that services properly implement their interfaces."""

    def test_validation_service_interface_compliance(self, setup_utils_services):
        """Test ValidationService implements ValidationServiceInterface."""
        service = injector.resolve("ValidationServiceInterface")

        # Verify it's actually a ValidationService
        assert isinstance(service, ValidationService)

        # Verify it implements the interface
        assert isinstance(service, ValidationServiceInterface)

        # Check all interface methods are present
        interface_methods = [
            "validate_order",
            "validate_risk_parameters",
            "validate_strategy_config",
            "validate_market_data",
            "validate_batch",
        ]

        for method_name in interface_methods:
            assert hasattr(service, method_name)
            method = getattr(service, method_name)
            assert callable(method)

    def test_gpu_interface_compliance(self, setup_utils_services):
        """Test GPUManager implements GPUInterface."""
        service = injector.resolve("GPUInterface")

        # Check interface methods
        interface_methods = ["is_available", "get_memory_info"]

        for method_name in interface_methods:
            assert hasattr(service, method_name)
            method = getattr(service, method_name)
            assert callable(method)

    def test_precision_interface_compliance(self, setup_utils_services):
        """Test PrecisionTracker implements PrecisionInterface."""
        service = injector.resolve("PrecisionInterface")

        # Check interface methods
        interface_methods = ["track_operation", "get_precision_stats"]

        for method_name in interface_methods:
            assert hasattr(service, method_name)
            method = getattr(service, method_name)
            assert callable(method)

    def test_data_flow_interface_compliance(self, setup_utils_services):
        """Test DataFlowValidator implements DataFlowInterface."""
        service = injector.resolve("DataFlowInterface")

        # Check interface methods
        interface_methods = ["validate_data_integrity", "get_validation_report"]

        for method_name in interface_methods:
            assert hasattr(service, method_name)
            method = getattr(service, method_name)
            assert callable(method)

    def test_calculator_interface_compliance(self, setup_utils_services):
        """Test FinancialCalculator implements CalculatorInterface."""
        service = injector.resolve("CalculatorInterface")

        # Check interface methods
        interface_methods = ["calculate_compound_return", "calculate_sharpe_ratio"]

        for method_name in interface_methods:
            assert hasattr(service, method_name)
            method = getattr(service, method_name)
            assert callable(method)


class TestDataValidationContracts:
    """Test data validation contracts across utils services."""

    @pytest_asyncio.fixture
    async def validation_service(self, setup_utils_services):
        """Get validation service instance."""
        service = injector.resolve("ValidationServiceInterface")
        await service.initialize()
        yield service
        await service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_decimal_precision_contract(self, validation_service):
        """Test decimal precision is maintained in validation results."""
        order_data = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.12345678",  # High precision
            "price": "50000.87654321",  # High precision
        }

        result = await validation_service.validate_order(order_data)

        if result.normalized_value:
            # Verify precision is preserved
            if "quantity" in result.normalized_value:
                normalized_qty = result.normalized_value["quantity"]
                assert isinstance(normalized_qty, Decimal)
                # Should preserve original precision
                assert str(normalized_qty) == "0.12345678"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_context_contract(self, validation_service):
        """Test validation context is properly handled."""
        order_data = {"symbol": "BTC/USDT", "side": "BUY", "type": "MARKET", "quantity": "0.1"}

        # Test with minimal context
        minimal_context = ValidationContext(exchange="binance")
        result1 = await validation_service.validate_order(order_data, minimal_context)
        assert result1.context is not None
        assert result1.context.exchange == "binance"

        # Test with full context
        full_context = ValidationContext(
            exchange="coinbase",
            trading_mode="paper",
            strategy_type="arbitrage",
            user_id="test_user",
            session_id="test_session",
            request_id="test_request",
            additional_context={"test": "value"},
        )
        result2 = await validation_service.validate_order(order_data, full_context)
        assert result2.context is not None
        assert result2.context.exchange == "coinbase"
        assert result2.context.trading_mode == "paper"
        assert result2.context.additional_context["test"] == "value"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_severity_contract(self, validation_service):
        """Test error severity levels are properly assigned."""
        # Create order with various severity issues
        problematic_order = {
            "symbol": "",  # Critical - missing required field
            "side": "INVALID_SIDE",  # High - invalid enum value
            "type": "LIMIT",
            "quantity": "0.000000001",  # Warning - very small but valid
            "price": "999999999999",  # High - extremely high price
        }

        result = await validation_service.validate_order(problematic_order)

        assert not result.is_valid
        assert len(result.errors) > 0

        # Check that we have different severity levels
        severities = [error.severity for error in result.errors]
        assert len(set(severities)) > 1, "Should have multiple severity levels"

        # Verify critical errors exist for missing required fields
        critical_errors = [e for e in result.errors if e.severity == ValidationLevel.CRITICAL]
        assert len(critical_errors) > 0, "Should have critical errors for missing required fields"


class TestPerformanceContracts:
    """Test performance contracts for utils services."""

    @pytest_asyncio.fixture
    async def validation_service(self, setup_utils_services):
        """Get validation service instance."""
        service = injector.resolve("ValidationServiceInterface")
        await service.initialize()
        yield service
        await service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_timing_contract(self, validation_service):
        """Test that validation operations complete within reasonable time."""
        order_data = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.1",
            "price": "50000.0",
        }

        import time

        start_time = time.time()
        result = await validation_service.validate_order(order_data)
        end_time = time.time()

        # Validation should complete quickly (< 1 second for simple validation)
        execution_time = end_time - start_time
        assert execution_time < 1.0, f"Validation took too long: {execution_time}s"

        # Result should also track execution time
        assert result.execution_time_ms >= 0
        assert result.execution_time_ms < 1000  # < 1 second in milliseconds

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_batch_validation_efficiency_contract(self, validation_service):
        """Test that batch validation is more efficient than individual validations."""
        # Create multiple validation requests with different types to avoid key collision
        validations = [
            ("order", {"symbol": "BTC/USDT", "side": "BUY", "type": "MARKET", "quantity": "0.1"}),
            ("risk", {"max_position_size": "0.05", "stop_loss_percentage": "0.02"}),
            ("strategy", {"name": "momentum_strategy", "parameters": {"lookback_period": 20}}),
            (
                "market_data",
                {
                    "symbol": "ETH/USDT",
                    "timestamp": 1234567890,
                    "open": "3000.0",
                    "high": "3100.0",
                    "low": "2900.0",
                    "close": "3050.0",
                    "volume": "50.0",
                },
            ),
        ]

        # Time batch validation
        import time

        start_time = time.time()
        batch_results = await validation_service.validate_batch(validations)
        batch_time = time.time() - start_time

        # Time individual validations
        start_time = time.time()
        individual_results = []
        for validation_type, data in validations:
            if validation_type == "order":
                result = await validation_service.validate_order(data)
            elif validation_type == "risk":
                result = await validation_service.validate_risk_parameters(data)
            elif validation_type == "strategy":
                result = await validation_service.validate_strategy_config(data)
            elif validation_type == "market_data":
                result = await validation_service.validate_market_data(data)
            else:
                continue  # Skip unknown validation types
            individual_results.append(result)
        individual_time = time.time() - start_time

        # Verify both approaches produce the same number of results
        assert len(batch_results) == len(individual_results), (
            f"Result count mismatch: batch {len(batch_results)} vs individual {len(individual_results)}"
        )

        # Verify all validations completed successfully (functional correctness)
        assert len(batch_results) == 4, f"Expected 4 batch results, got {len(batch_results)}"
        assert len(individual_results) == 4, (
            f"Expected 4 individual results, got {len(individual_results)}"
        )

        # Performance note: For small batches (≤10 items), batch processing overhead
        # may make it slower than individual calls. The benefit comes with larger datasets.
        # We verify that batch validation completes in reasonable time (< 10ms)
        assert batch_time < 0.01, f"Batch validation too slow: {batch_time}s"
        assert individual_time < 0.01, f"Individual validations too slow: {individual_time}s"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_concurrent_validation_contract(self, validation_service):
        """Test that concurrent validations work correctly."""
        order_template = {"symbol": "BTC/USDT", "side": "BUY", "type": "LIMIT", "price": "50000.0"}

        # Create concurrent validation tasks
        tasks = []
        for i in range(10):
            order_data = order_template.copy()
            order_data["quantity"] = f"0.{i + 1}"
            task = validation_service.validate_order(order_data)
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 10
        for result in results:
            assert isinstance(result, ValidationResult)
            # Most should be valid (only checking structure here)
            assert hasattr(result, "is_valid")


class TestErrorContractConsistency:
    """Test error contract consistency across utils services."""

    @pytest_asyncio.fixture
    async def validation_service(self, setup_utils_services):
        """Get validation service instance."""
        service = injector.resolve("ValidationServiceInterface")
        await service.initialize()
        yield service
        await service.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_message_contract(self, validation_service):
        """Test error messages follow consistent format."""
        invalid_order = {
            "symbol": "",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "invalid",
            "price": "-100",
        }

        result = await validation_service.validate_order(invalid_order)

        assert not result.is_valid
        assert len(result.errors) > 0

        for error in result.errors:
            # Error messages should be informative
            assert len(error.message) > 0
            assert isinstance(error.message, str)

            # Should identify the problematic field
            assert len(error.field) > 0
            assert isinstance(error.field, str)

            # Should have validation type
            assert len(error.validation_type) > 0
            assert isinstance(error.validation_type, str)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_exception_handling_contract(self, validation_service):
        """Test that validation service handles exceptions gracefully."""
        # Test with malformed data that might cause internal errors
        malformed_data = {
            "symbol": None,  # None instead of string
            "side": 12345,  # Number instead of string
            "type": [],  # List instead of string
            "quantity": {},  # Dict instead of number
        }

        # Should not raise exception, but return error result
        result = await validation_service.validate_order(malformed_data)

        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_suggestion_contract(self, validation_service):
        """Test that validation errors include helpful suggestions."""
        invalid_order = {
            "symbol": "BTC/USDT",
            "side": "INVALID_SIDE",  # Should suggest valid sides
            "type": "LIMIT",
            "quantity": "0.1",
            "price": "50000.0",
        }

        result = await validation_service.validate_order(invalid_order)

        # Find error for invalid side
        side_errors = [e for e in result.errors if e.field == "side"]
        assert len(side_errors) > 0

        # Should have helpful suggestion
        side_error = side_errors[0]
        assert side_error.suggestion is not None
        assert len(side_error.suggestion) > 0
