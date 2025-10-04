"""
Integration tests for utils module validation integration.

This test verifies that utils module properly integrates with other modules
and uses ValidationService correctly through dependency injection.
"""

from decimal import Decimal

import pytest

from src.core.dependency_injection import DependencyInjector
from src.core.types import OrderRequest, OrderSide, OrderType
from src.utils.exchange_order_utils import get_order_management_utils
from src.utils.exchange_validation_utils import get_exchange_validation_utils
from src.utils.service_registry import register_util_services


@pytest.fixture(autouse=True)
def setup_utils_services():
    """Setup utils services for each test."""
    from src.core.dependency_injection import get_global_injector

    # Ensure utils services are registered before each test
    global_injector = get_global_injector()

    # Reset the services registration flag to allow re-registration if needed
    import src.utils.service_registry as registry_module
    if hasattr(registry_module, '_services_registered'):
        registry_module._services_registered = False

    # Register utils services (this is idempotent)
    register_util_services()

    yield

    # Don't cleanup - let other tests use the services


@pytest.fixture
def injector():
    """Create a fresh dependency injector for each test."""
    injector = DependencyInjector()
    return injector


@pytest.fixture
def sample_order():
    """Create a sample valid order for testing."""
    return OrderRequest(
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.001"),
        price=Decimal("50000.00"),
    )


@pytest.fixture
def invalid_order():
    """Create an invalid order for testing."""
    return OrderRequest(
        symbol="",  # Invalid empty symbol
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("-1"),  # Invalid negative quantity
        price=Decimal("50000.00"),
    )


class TestUtilsValidationIntegration:
    """Test utils module validation integration with ValidationService."""

    def test_validation_service_is_registered(self, injector):
        """Test that ValidationService is properly registered."""
        from src.core.dependency_injection import get_global_injector

        # Get the global injector where services are actually registered
        global_injector = get_global_injector()

        # ValidationService should be available through the global injector
        validation_service = global_injector.resolve("ValidationService")
        assert validation_service is not None

        # Interface should also be available
        validation_interface = global_injector.resolve("ValidationServiceInterface")
        assert validation_interface is not None

    def test_order_management_utils_uses_validation_service(self, injector):
        """Test that OrderManagementUtils uses ValidationService properly."""
        utils = get_order_management_utils()

        # Should have validation_service injected
        assert utils.validation_service is not None

        # Test valid order validation
        order = OrderRequest(
            symbol="BTCUSDT",  # Use Binance format
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),
        )

        # Should not raise exception for valid order
        utils.validate_order_request(order, "binance")

    def test_order_management_utils_invalid_order(self, injector):
        """Test that OrderManagementUtils properly validates invalid orders."""
        utils = get_order_management_utils()

        # Create a valid order first
        valid_order = OrderRequest(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000.00"),
        )

        # Test validation directly by modifying the order object
        # This bypasses Pydantic validation to test our validation logic
        valid_order.quantity = Decimal("-1")  # Make it invalid after creation

        # Should raise ValidationError
        with pytest.raises(Exception) as exc_info:
            utils.validate_order_request(valid_order, "binance")
        assert "Quantity must be positive" in str(exc_info.value)

    def test_exchange_validation_utils_dependency_injection(self, injector):
        """Test that ExchangeValidationUtils uses ValidationService properly."""
        utils = get_exchange_validation_utils()

        # Should have validation_service (may be None if not available)
        # This tests the factory function works correctly
        assert hasattr(utils, "validation_service")

        # Test valid order validation (use Binance format - no separators)
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,  # Use market order to avoid time_in_force issue
            quantity=Decimal("0.001"),
        )

        # Should not raise exception for valid order
        utils.validate_order_request(order, "binance")

    def test_exchange_validation_utils_invalid_quantity(self, injector):
        """Test that ExchangeValidationUtils properly validates invalid quantities."""
        utils = get_exchange_validation_utils()

        # Create valid order first, then modify to test validation
        valid_order = OrderRequest(
            symbol="BTCUSDT",  # Use Binance format
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),
        )

        # Test validation directly by modifying the order object
        # This bypasses Pydantic validation to test our validation logic
        valid_order.quantity = Decimal("-1")  # Make it invalid after creation

        # Should raise ValidationError
        with pytest.raises(Exception) as exc_info:
            utils.validate_order_request(valid_order, "binance")
        assert "Quantity must be positive" in str(exc_info.value)

    def test_validation_service_fallback(self, injector):
        """Test that utils work correctly when ValidationService is not available."""
        # Create utils with None validation service to test fallback
        from src.utils.exchange_order_utils import OrderManagementUtils

        utils = OrderManagementUtils(validation_service=None)

        # Should still work with fallback validation
        order = OrderRequest(
            symbol="BTC-USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),
        )

        # Should not raise exception for valid order
        utils.validate_order_request(order, "binance")

        # Should still catch invalid orders (create valid order then modify)
        invalid_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),
        )

        # Modify to make invalid after creation
        invalid_order.symbol = ""  # Invalid empty symbol

        with pytest.raises(Exception) as exc_info:
            utils.validate_order_request(invalid_order, "binance")
        assert "Symbol is required" in str(exc_info.value)

    def test_integration_no_direct_validation_framework_usage(self, injector):
        """Test that utils module doesn't directly import ValidationFramework."""
        import sys

        # Test that key utils modules don't have ValidationFramework in their namespace
        modules_to_check = [
            "src.utils.exchange_order_utils",
            "src.utils.exchange_validation_utils",
        ]

        for module_name in modules_to_check:
            if module_name in sys.modules:
                module = sys.modules[module_name]

                # Check that ValidationFramework is not directly used
                module_vars = vars(module)
                assert "ValidationFramework" not in module_vars, (
                    f"{module_name} should not directly use ValidationFramework"
                )

    def test_utils_service_registry_integration(self, injector):
        """Test that utils service registry properly registers services."""
        injector = DependencyInjector()

        # Register utils services
        register_util_services()

        # Check that key services are registered
        services_to_check = [
            "ValidationService",
            "ValidationFramework",
            "GPUManager",
            "PrecisionTracker",
            "DataFlowValidator",
        ]

        for service_name in services_to_check:
            try:
                service = injector.resolve(service_name)
                assert service is not None, f"{service_name} should be registered"
            except Exception as e:
                # Some services might fail to create due to missing dependencies
                # but they should be registered
                assert "not found" not in str(e).lower(), (
                    f"{service_name} should be registered even if creation fails"
                )


if __name__ == "__main__":
    pytest.main([__file__])
