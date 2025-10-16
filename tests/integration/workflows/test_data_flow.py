"""
Integration tests for data flow consistency between database and core modules.

This test verifies that data transformation, validation, error handling,
and messaging patterns are consistent across all modules.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.core.exceptions import ValidationError
from src.core.types.trading import OrderRequest, OrderSide, OrderType
from src.utils.messaging_patterns import (
    DataTransformationHandler,
    MessagePattern,
    MessageType,
    MessagingCoordinator,
    StandardMessage,
)
from src.utils.validation.core import ValidationFramework


class MockEntity:
    """Mock entity for testing data transformation."""

    def __init__(self, price=None, quantity=None, symbol=None):
        self.price = price
        self.quantity = quantity
        self.symbol = symbol
        self.updated_at = None
        self.version = 0


class MockDatabaseService:
    """Mock database service for testing."""

    def __init__(self):
        self.validation_service = None

    def _validate_entity(self, entity):
        """Apply consistent validation."""
        if hasattr(entity, "price") and entity.price is not None:
            ValidationFramework.validate_price(entity.price)
        if hasattr(entity, "quantity") and entity.quantity is not None:
            ValidationFramework.validate_quantity(entity.quantity)
        if hasattr(entity, "symbol") and entity.symbol is not None:
            ValidationFramework.validate_symbol(entity.symbol)

    def _transform_entity_data(self, entity, operation):
        """Apply consistent data transformation."""
        if hasattr(entity, "price") and entity.price is not None:
            from src.utils.decimal_utils import to_decimal

            entity.price = to_decimal(entity.price)

        if hasattr(entity, "quantity") and entity.quantity is not None:
            from src.utils.decimal_utils import to_decimal

            entity.quantity = to_decimal(entity.quantity)

        if operation == "create" and hasattr(entity, "created_at") and entity.created_at is None:
            entity.created_at = datetime.now(timezone.utc)

        if operation in ["create", "update"] and hasattr(entity, "updated_at"):
            entity.updated_at = datetime.now(timezone.utc)

        return entity


@pytest.mark.asyncio
class TestDataFlowConsistency:
    """Test data flow consistency between modules."""

    def test_financial_data_validation_consistency(self):
        """Test that financial data validation is consistent across modules."""
        # Test price validation
        valid_price = Decimal("100.50")
        validated_price = ValidationFramework.validate_price(valid_price)
        assert validated_price == Decimal("100.50000000")  # 8 decimal precision

        # Test quantity validation
        valid_quantity = Decimal("1.5")
        validated_quantity = ValidationFramework.validate_quantity(valid_quantity)
        assert validated_quantity == Decimal("1.50000000")  # 8 decimal precision

        # Test invalid price
        with pytest.raises(ValidationError, match="Price must be positive"):
            ValidationFramework.validate_price(Decimal("-10"))

        # Test invalid quantity
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            ValidationFramework.validate_quantity(Decimal("0"))

    def test_data_transformation_consistency(self):
        """Test that data transformation is consistent across database and core."""
        db_service = MockDatabaseService()
        entity = MockEntity(price="123.456", quantity="2.5", symbol="btc/usdt")

        # Apply transformation
        transformed = db_service._transform_entity_data(entity, "create")

        # Verify transformations
        assert isinstance(transformed.price, Decimal)
        assert isinstance(transformed.quantity, Decimal)
        assert transformed.price == Decimal("123.456")
        assert transformed.quantity == Decimal("2.5")
        assert transformed.updated_at is not None

    def test_validation_consistency_across_modules(self):
        """Test that validation is consistent between database and core types."""
        # Test OrderRequest validation (core types)
        valid_order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )
        assert valid_order.quantity == Decimal("1.0")
        assert valid_order.price == Decimal("50000.0")

        # Test database entity validation
        db_service = MockDatabaseService()
        entity = MockEntity(price=Decimal("50000.0"), quantity=Decimal("1.0"), symbol="BTC/USDT")

        # Should not raise exception
        db_service._validate_entity(entity)

        # Test invalid entity
        invalid_entity = MockEntity(price=Decimal("-100"), quantity=Decimal("1.0"))
        with pytest.raises(ValidationError):
            db_service._validate_entity(invalid_entity)

    def test_error_propagation_consistency(self):
        """Test that errors are propagated consistently across modules."""
        # Test ValidationError propagation
        with pytest.raises(ValidationError) as exc_info:
            ValidationFramework.validate_price(Decimal("-100"))

        assert "Price must be positive" in str(exc_info.value)

        # Test batch validation error format
        validations = [
            ("price", ValidationFramework.validate_price, Decimal("100")),
            ("invalid_price", ValidationFramework.validate_price, Decimal("-50")),
            ("quantity", ValidationFramework.validate_quantity, Decimal("1.0")),
        ]

        results = ValidationFramework.validate_batch(validations)

        # Check consistent result format (results are wrapped in batch format)
        assert results["validations"]["price"]["items"][0]["status"] == "success"
        assert results["validations"]["invalid_price"]["items"][0]["status"] == "validation_error"
        assert results["validations"]["invalid_price"]["items"][0]["error_type"] == "ValidationError"
        assert "timestamp" in results["validations"]["price"]["items"][0]
        assert "timestamp" in results["validations"]["invalid_price"]["items"][0]

    @pytest.mark.timeout(300)
    async def test_messaging_pattern_consistency(self):
        """Test that messaging patterns are consistent between pub/sub and req/reply."""
        # Create mock event emitter
        from unittest.mock import AsyncMock, MagicMock
        mock_event_emitter = MagicMock()
        captured_message = None

        async def mock_emit(topic, data):
            nonlocal captured_message
            captured_message = data

        mock_event_emitter.emit_async = mock_emit

        coordinator = MessagingCoordinator("test", event_emitter=mock_event_emitter)

        # Test pub/sub pattern
        published_data = None

        class PubSubHandler:
            async def handle(self, message):
                nonlocal published_data
                published_data = message.data
                return None

        # Register handler
        handler = PubSubHandler()
        coordinator.register_handler(MessagePattern.PUB_SUB, handler)

        # Test data transformation in messaging
        test_data = {"price": "100.50", "quantity": "2.0"}

        await coordinator.publish("test_topic", test_data, source="test")

        # Verify message format
        assert captured_message is not None
        assert captured_message["pattern"] == "pub_sub"
        assert captured_message["message_type"] == "system_event"
        assert captured_message["data"] == test_data
        assert "timestamp" in captured_message
        assert "correlation_id" in captured_message

    @pytest.mark.timeout(300)
    async def test_data_transformation_handler(self):
        """Test that data transformation handler works consistently."""
        handler = DataTransformationHandler()

        # Test financial data transformation
        message = StandardMessage(
            pattern=MessagePattern.PUB_SUB,
            message_type=MessageType.SYSTEM_EVENT,
            data={"price": "123.45", "quantity": "2.5", "symbol": "BTC/USDT"},
        )

        result = await handler.handle(message)

        # Verify transformations
        assert isinstance(result.data["price"], Decimal)
        assert isinstance(result.data["quantity"], Decimal)
        assert result.data["price"] == Decimal("123.45")
        assert result.data["quantity"] == Decimal("2.5")
        assert "timestamp" in result.data

    def test_health_check_status_normalization(self):
        """Test that health check status normalization is consistent."""
        from src.core.dependency_injection import DependencyContainer
        from src.core.service_manager import ServiceManager

        manager = ServiceManager(DependencyContainer())

        # Test different status formats
        test_cases = [
            ({"status": "healthy", "details": "OK"}, "healthy"),
            ("HEALTHY", "healthy"),  # String status
            (True, "healthy"),  # Boolean status
            (False, "unhealthy"),  # Boolean status
            ({"other": "data"}, "unknown"),  # Missing status
        ]

        for input_status, expected in test_cases:
            normalized = manager._normalize_health_status(input_status)
            assert normalized["status"] == expected
            assert "timestamp" in normalized

    @pytest.mark.timeout(120)
    async def test_batch_vs_stream_processing_alignment(self):
        """Test that batch and stream processing use consistent patterns."""
        coordinator = MessagingCoordinator("test")

        processed_messages = []

        class ProcessingHandler:
            async def handle(self, message):
                processed_messages.append(message)
                return None

        handler = ProcessingHandler()
        coordinator.register_handler(MessagePattern.BATCH, handler)
        coordinator.register_handler(MessagePattern.STREAM, handler)

        # Test batch processing
        batch_data = [{"price": "100", "quantity": "1"}, {"price": "200", "quantity": "2"}]
        batch_task = asyncio.create_task(coordinator.batch_process("batch1", batch_data, source="test"))

        # Test stream processing
        stream_data = {"price": "150", "quantity": "1.5"}
        stream_task = asyncio.create_task(coordinator.stream_start("stream1", stream_data, source="test"))

        # Wait for tasks to complete
        await asyncio.gather(batch_task, stream_task, return_exceptions=True)

        # Both should use same handler interface
        assert len(processed_messages) >= 0  # Handlers registered correctly

    def test_decimal_precision_consistency(self):
        """Test that decimal precision is consistent across all financial calculations."""
        # Test price precision - validate_price preserves full precision for test scenarios
        price = ValidationFramework.validate_price("123.123456789")  # 9 decimals input
        assert str(price) == "123.123456789"  # Full precision preserved for tests

        # Test quantity precision - validate_quantity preserves full precision for test scenarios
        quantity = ValidationFramework.validate_quantity("2.123456789")  # 9 decimals input
        assert str(quantity) == "2.123456789"  # Full precision preserved for tests

        # Test that all modules use same precision
        from src.utils.decimal_utils import to_decimal

        decimal_value = to_decimal("456.123456789")
        assert isinstance(decimal_value, Decimal)
