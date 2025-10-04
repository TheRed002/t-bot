"""Quick validation test for utils integration."""

import asyncio

from src.core.dependency_injection import injector
from src.utils.service_registry import register_util_services
from src.utils.validation.service import ValidationResult, ValidationService, ValidationType
import pytest


@pytest.mark.asyncio
async def test_utils_integration_quick():
    """Quick test of utils integration."""
    # Setup
    injector.get_container().clear()
    register_util_services()

    # Get service
    validation_service = injector.resolve("ValidationServiceInterface")
    assert isinstance(validation_service, ValidationService)

    # Initialize
    await validation_service.initialize()
    assert validation_service.is_running

    # Test validation
    order_data = {"symbol": "BTC/USDT", "side": "BUY", "type": "MARKET", "quantity": "0.1"}

    result = await validation_service.validate_order(order_data)
    assert isinstance(result, ValidationResult)
    assert result.validation_type == ValidationType.ORDER

    # Cleanup
    await validation_service.shutdown()
    injector.get_container().clear()

    print("✅ Utils integration test passed!")


if __name__ == "__main__":
    asyncio.run(test_utils_integration_quick())
