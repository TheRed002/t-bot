"""
Comprehensive tests for analytics events module.

Tests event-driven patterns, event bus, handlers, and event publishing functions.
"""

import asyncio

# Disable logging during tests for performance
import logging
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

logging.disable(logging.CRITICAL)

# Test configuration optimizations

from src.analytics.events import (
    AlertEventHandler,
    AnalyticsEvent,
    AnalyticsEventBus,
    AnalyticsEventHandler,
    AnalyticsEventType,
    OrderUpdatedEvent,
    PortfolioEventHandler,
    PositionUpdatedEvent,
    PriceUpdatedEvent,
    RiskEventHandler,
    RiskLimitBreachedEvent,
    TradeExecutedEvent,
    get_event_bus,
    publish_order_updated,
    publish_position_updated,
    publish_price_updated,
    publish_risk_limit_breached,
    publish_trade_executed,
)
from src.core.types import Order, Position, Trade
from src.core.types.trading import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    PositionStatus,
    TimeInForce,
)


class TestAnalyticsEventType:
    """Test AnalyticsEventType enum."""

    def test_event_type_values(self):
        """Test that all event types have expected values."""
        assert AnalyticsEventType.POSITION_UPDATED.value == "position_updated"
        assert AnalyticsEventType.TRADE_EXECUTED.value == "trade_executed"
        assert AnalyticsEventType.ORDER_UPDATED.value == "order_updated"
        assert AnalyticsEventType.PRICE_UPDATED.value == "price_updated"
        assert AnalyticsEventType.RISK_LIMIT_BREACHED.value == "risk_limit_breached"
        assert AnalyticsEventType.ALERT_GENERATED.value == "alert_generated"
        assert AnalyticsEventType.SERVICE_STARTED.value == "service_started"
        assert AnalyticsEventType.DATA_EXPORTED.value == "data_exported"

    def test_event_type_enum_completeness(self):
        """Test that all expected event types are defined."""
        expected_types = [
            "POSITION_UPDATED",
            "TRADE_EXECUTED",
            "ORDER_UPDATED",
            "PRICE_UPDATED",
            "BENCHMARK_UPDATED",
            "PORTFOLIO_METRICS_CALCULATED",
            "RISK_METRICS_CALCULATED",
            "STRATEGY_METRICS_CALCULATED",
            "ALERT_GENERATED",
            "ALERT_ACKNOWLEDGED",
            "ALERT_RESOLVED",
            "RISK_LIMIT_BREACHED",
            "SERVICE_STARTED",
            "SERVICE_STOPPED",
            "ERROR_OCCURRED",
            "HEALTH_CHECK_COMPLETED",
            "DATA_EXPORTED",
            "REPORT_GENERATED",
        ]

        for event_type_name in expected_types:
            assert hasattr(AnalyticsEventType, event_type_name)


class TestAnalyticsEvent:
    """Test AnalyticsEvent base class."""

    @pytest.fixture
    def sample_event_data(self):
        """Sample event data for testing."""
        return {
            "event_id": "test_event_001",
            "event_type": AnalyticsEventType.POSITION_UPDATED,
            "timestamp": datetime.now(),
            "source_component": "test_component",
            "data": {"test": "data"},
            "metadata": {"extra": "info"},
        }

    def test_analytics_event_creation(self, sample_event_data):
        """Test creating AnalyticsEvent."""
        event = AnalyticsEvent(**sample_event_data)

        assert event.event_id == "test_event_001"
        assert event.event_type == AnalyticsEventType.POSITION_UPDATED
        assert event.source_component == "test_component"
        assert event.data == {"test": "data"}
        assert event.metadata == {"extra": "info"}

    def test_analytics_event_default_metadata(self):
        """Test AnalyticsEvent with default metadata."""
        event = AnalyticsEvent(
            event_id="test_002",
            event_type=AnalyticsEventType.TRADE_EXECUTED,
            timestamp=datetime.now(),
            source_component="test",
            data={},
        )

        assert event.metadata == {}

    def test_analytics_event_serialization(self, sample_event_data):
        """Test event serialization to dict."""
        event = AnalyticsEvent(**sample_event_data)
        event_dict = event.model_dump()

        assert "event_id" in event_dict
        assert "event_type" in event_dict
        assert "timestamp" in event_dict
        assert "source_component" in event_dict
        assert "data" in event_dict
        assert "metadata" in event_dict


class TestPositionUpdatedEvent:
    """Test PositionUpdatedEvent."""

    @pytest.fixture
    def sample_position(self):
        """Sample position for testing."""
        return Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("1.5"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("52000.00"),
            unrealized_pnl=Decimal("3000.00"),
            status=PositionStatus.OPEN,
            opened_at=datetime.now(),
            exchange="binance",
        )

    @patch("src.utils.datetime_utils.get_current_utc_timestamp")
    def test_position_updated_event_creation(self, mock_timestamp, sample_position):
        """Test creating PositionUpdatedEvent."""
        mock_time = datetime.now()
        mock_timestamp.return_value = mock_time

        event = PositionUpdatedEvent(sample_position, "portfolio_service")

        assert event.event_type == AnalyticsEventType.POSITION_UPDATED
        assert event.source_component == "portfolio_service"
        assert event.timestamp == mock_time
        assert "position" in event.data
        assert event.data["symbol"] == "BTC/USDT"
        assert event.data["exchange"] == "binance"
        assert event.event_id.startswith("position_updated_BTC/USDT_")

    def test_position_updated_event_with_metadata(self, sample_position):
        """Test PositionUpdatedEvent with additional metadata."""
        event = PositionUpdatedEvent(sample_position, "test_service", metadata={"custom": "data"})

        assert event.metadata == {"custom": "data"}


class TestTradeExecutedEvent:
    """Test TradeExecutedEvent."""

    @pytest.fixture
    def sample_trade(self):
        """Sample trade for testing."""
        return Trade(
            trade_id="trade_001",
            order_id="order_001",
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            price=Decimal("3000.00"),
            quantity=Decimal("10.0"),
            fee=Decimal("5.0"),
            fee_currency="USDT",
            timestamp=datetime.now(),
            exchange="coinbase",
        )

    def test_trade_executed_event_creation(self, sample_trade):
        """Test creating TradeExecutedEvent."""
        event = TradeExecutedEvent(sample_trade, "execution_service")

        assert event.event_type == AnalyticsEventType.TRADE_EXECUTED
        assert event.source_component == "execution_service"
        assert event.timestamp == sample_trade.timestamp
        assert "trade" in event.data
        assert event.data["trade_id"] == "trade_001"
        assert event.data["symbol"] == "ETH/USDT"
        assert event.data["exchange"] == "coinbase"
        assert event.event_id == "trade_executed_trade_001"


class TestOrderUpdatedEvent:
    """Test OrderUpdatedEvent."""

    @pytest.fixture
    def sample_order(self):
        """Sample order for testing."""
        return Order(
            order_id="order_001",
            symbol="ADA/USDT",
            exchange="binance",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1000.0"),
            price=Decimal("0.50"),
            status=OrderStatus.FILLED,
            time_in_force=TimeInForce.GTC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def test_order_updated_event_creation(self, sample_order):
        """Test creating OrderUpdatedEvent."""
        event = OrderUpdatedEvent(sample_order, "order_service")

        assert event.event_type == AnalyticsEventType.ORDER_UPDATED
        assert event.source_component == "order_service"
        assert event.timestamp == sample_order.updated_at
        assert "order" in event.data
        assert event.data["order_id"] == "order_001"
        assert event.data["symbol"] == "ADA/USDT"
        assert event.data["exchange"] == "binance"
        assert event.event_id == "order_updated_order_001"

    def test_order_updated_event_fallback_timestamp(self):
        """Test OrderUpdatedEvent with fallback to created_at timestamp."""
        order = Order(
            order_id="order_002",
            symbol="BTC/USDT",
            exchange="binance",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            status=OrderStatus.PENDING,
            time_in_force=TimeInForce.GTC,
            created_at=datetime.now(),
            updated_at=None,  # No updated timestamp
        )

        event = OrderUpdatedEvent(order, "test_service")

        assert event.timestamp == order.created_at


class TestPriceUpdatedEvent:
    """Test PriceUpdatedEvent."""

    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for testing."""
        return {
            "symbol": "BTC/USDT",
            "price": Decimal("51000.00"),
            "timestamp": datetime.now(),
            "source_component": "market_data_service",
        }

    def test_price_updated_event_creation(self, sample_price_data):
        """Test creating PriceUpdatedEvent."""
        event = PriceUpdatedEvent(**sample_price_data)

        assert event.event_type == AnalyticsEventType.PRICE_UPDATED
        assert event.source_component == "market_data_service"
        assert event.data["symbol"] == "BTC/USDT"
        assert event.data["price"] == "51000.00"  # Stored as string
        assert event.data["timestamp"] == sample_price_data["timestamp"].isoformat()
        assert event.event_id.startswith("price_updated_BTC/USDT_")

    def test_price_updated_event_decimal_precision(self):
        """Test price precision is preserved as string."""
        high_precision_price = Decimal("50000.12345678")
        timestamp = datetime.now()

        event = PriceUpdatedEvent(
            symbol="BTC/USDT",
            price=high_precision_price,
            timestamp=timestamp,
            source_component="test",
        )

        assert event.data["price"] == "50000.12345678"


class TestRiskLimitBreachedEvent:
    """Test RiskLimitBreachedEvent."""

    @patch("src.utils.datetime_utils.get_current_utc_timestamp")
    def test_risk_limit_breached_event_creation(self, mock_timestamp):
        """Test creating RiskLimitBreachedEvent."""
        mock_time = datetime.now()
        mock_timestamp.return_value = mock_time

        event = RiskLimitBreachedEvent(
            limit_type="max_drawdown",
            current_value=-0.08,
            limit_value=-0.05,
            breach_percentage=0.6,
            source_component="risk_service",
        )

        assert event.event_type == AnalyticsEventType.RISK_LIMIT_BREACHED
        assert event.source_component == "risk_service"
        assert event.timestamp == mock_time
        assert event.data["limit_type"] == "max_drawdown"
        assert event.data["current_value"] == -0.08
        assert event.data["limit_value"] == -0.05
        assert event.data["breach_percentage"] == 0.6
        assert event.event_id.startswith("risk_limit_breached_max_drawdown_")


class TestAnalyticsEventBus:
    """Test AnalyticsEventBus."""

    @pytest.fixture
    def event_bus(self):
        """Create and clean up event bus for testing."""
        bus = AnalyticsEventBus()
        return bus

    @pytest.fixture
    def mock_handler(self):
        """Mock event handler for testing."""
        handler = Mock(spec=AnalyticsEventHandler)
        handler.handle_event = AsyncMock()
        handler.can_handle.return_value = True
        return handler

    @pytest.fixture
    def sample_event(self):
        """Sample event for testing."""
        return AnalyticsEvent(
            event_id="test_001",
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=datetime.now(),
            source_component="test",
            data={"test": "data"},
        )

    def test_event_bus_initialization(self, event_bus):
        """Test event bus initialization."""
        assert not event_bus._running
        assert event_bus._processing_task is None
        assert event_bus._handlers == {}
        assert event_bus._event_history == []
        assert event_bus._max_history == 1000

    @pytest.mark.asyncio
    async def test_event_bus_start_stop(self, event_bus):
        """Test event bus start and stop."""
        # Test start
        await event_bus.start()
        assert event_bus._running
        assert event_bus._processing_task is not None

        # Test stop
        await event_bus.stop()
        assert not event_bus._running

    @pytest.mark.asyncio
    async def test_event_bus_double_start(self, event_bus):
        """Test that double start doesn't create multiple tasks."""
        await event_bus.start()
        first_task = event_bus._processing_task

        await event_bus.start()  # Should be no-op
        assert event_bus._processing_task is first_task

        # Clean up
        await event_bus.stop()

    def test_register_handler(self, event_bus, mock_handler):
        """Test registering event handlers."""
        event_type = AnalyticsEventType.POSITION_UPDATED

        event_bus.register_handler(event_type, mock_handler)

        assert event_type in event_bus._handlers
        assert mock_handler in event_bus._handlers[event_type]

    def test_register_multiple_handlers(self, event_bus):
        """Test registering multiple handlers for same event type."""
        handler1 = Mock(spec=AnalyticsEventHandler)
        handler2 = Mock(spec=AnalyticsEventHandler)
        event_type = AnalyticsEventType.TRADE_EXECUTED

        event_bus.register_handler(event_type, handler1)
        event_bus.register_handler(event_type, handler2)

        assert len(event_bus._handlers[event_type]) == 2
        assert handler1 in event_bus._handlers[event_type]
        assert handler2 in event_bus._handlers[event_type]

    def test_unregister_handler(self, event_bus, mock_handler):
        """Test unregistering event handlers."""
        event_type = AnalyticsEventType.POSITION_UPDATED

        # Register then unregister
        event_bus.register_handler(event_type, mock_handler)
        event_bus.unregister_handler(event_type, mock_handler)

        assert event_type not in event_bus._handlers

    def test_unregister_nonexistent_handler(self, event_bus, mock_handler):
        """Test unregistering handler that doesn't exist."""
        event_type = AnalyticsEventType.PRICE_UPDATED

        # Should not raise error
        event_bus.unregister_handler(event_type, mock_handler)

    @pytest.mark.asyncio
    async def test_publish_event(self, event_bus, sample_event):
        """Test publishing events."""
        await event_bus.publish_event(sample_event)

        # Event should be in history
        assert sample_event in event_bus._event_history

    @pytest.mark.asyncio
    async def test_event_history_limit(self, event_bus):
        """Test event history respects max limit."""
        event_bus._max_history = 3

        # Publish 5 events
        for i in range(5):
            event = AnalyticsEvent(
                event_id=f"test_{i}",
                event_type=AnalyticsEventType.POSITION_UPDATED,
                timestamp=datetime.now(),
                source_component="test",
                data={},
            )
            await event_bus.publish_event(event)

        # Should only keep last 3
        assert len(event_bus._event_history) == 3
        assert event_bus._event_history[0].event_id == "test_2"
        assert event_bus._event_history[2].event_id == "test_4"

    @pytest.mark.asyncio
    async def test_get_event_history(self, event_bus):
        """Test getting event history."""
        events = []
        for i in range(3):
            event = AnalyticsEvent(
                event_id=f"test_{i}",
                event_type=AnalyticsEventType.POSITION_UPDATED,
                timestamp=datetime.now(),
                source_component="test",
                data={},
            )
            events.append(event)
            await event_bus.publish_event(event)

        # Get all history
        history = event_bus.get_event_history()
        assert len(history) == 3
        assert history == events

    @pytest.mark.asyncio
    async def test_get_event_history_filtered(self, event_bus):
        """Test getting filtered event history."""
        # Publish different event types
        position_event = AnalyticsEvent(
            event_id="pos_1",
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=datetime.now(),
            source_component="test",
            data={},
        )
        trade_event = AnalyticsEvent(
            event_id="trade_1",
            event_type=AnalyticsEventType.TRADE_EXECUTED,
            timestamp=datetime.now(),
            source_component="test",
            data={},
        )

        await event_bus.publish_event(position_event)
        await event_bus.publish_event(trade_event)

        # Filter by position events
        position_history = event_bus.get_event_history(
            event_type=AnalyticsEventType.POSITION_UPDATED
        )

        assert len(position_history) == 1
        assert position_history[0] == position_event

    @pytest.mark.asyncio
    async def test_get_event_history_with_limit(self, event_bus):
        """Test getting event history with limit."""
        # Publish 5 events
        for i in range(5):
            event = AnalyticsEvent(
                event_id=f"test_{i}",
                event_type=AnalyticsEventType.POSITION_UPDATED,
                timestamp=datetime.now(),
                source_component="test",
                data={},
            )
            await event_bus.publish_event(event)

        # Get last 2
        history = event_bus.get_event_history(limit=2)
        assert len(history) == 2
        assert history[0].event_id == "test_3"
        assert history[1].event_id == "test_4"

    @pytest.mark.asyncio
    async def test_event_processing(self, event_bus, mock_handler, sample_event):
        """Test event processing with handlers."""
        # Register handler and start bus
        event_bus.register_handler(sample_event.event_type, mock_handler)
        await event_bus.start()

        try:
            # Publish event
            await event_bus.publish_event(sample_event)

            # Give time for processing
            await asyncio.sleep(0.1)

            # Handler should have been called
            mock_handler.handle_event.assert_called_once_with(sample_event)
        finally:
            await event_bus.stop()

    @pytest.mark.asyncio
    async def test_event_processing_multiple_handlers(self, event_bus, sample_event):
        """Test event processing with multiple handlers."""
        handler1 = Mock(spec=AnalyticsEventHandler)
        handler1.handle_event = AsyncMock()
        handler2 = Mock(spec=AnalyticsEventHandler)
        handler2.handle_event = AsyncMock()

        # Register handlers and start bus
        event_bus.register_handler(sample_event.event_type, handler1)
        event_bus.register_handler(sample_event.event_type, handler2)
        await event_bus.start()

        try:
            # Publish event
            await event_bus.publish_event(sample_event)

            # Give time for processing
            await asyncio.sleep(0.1)

            # Both handlers should have been called
            handler1.handle_event.assert_called_once_with(sample_event)
            handler2.handle_event.assert_called_once_with(sample_event)
        finally:
            await event_bus.stop()

    @pytest.mark.asyncio
    async def test_event_processing_error_handling(self, event_bus, sample_event):
        """Test event processing with handler errors."""
        handler = Mock(spec=AnalyticsEventHandler)
        handler.handle_event = AsyncMock(side_effect=Exception("Handler error"))

        # Register handler and start bus
        event_bus.register_handler(sample_event.event_type, handler)
        await event_bus.start()

        try:
            # Publish event - should not raise error
            await event_bus.publish_event(sample_event)

            # Give time for processing
            await asyncio.sleep(0.1)

            # Handler should have been called despite error
            handler.handle_event.assert_called_once_with(sample_event)
        finally:
            await event_bus.stop()


class TestPortfolioEventHandler:
    """Test PortfolioEventHandler."""

    @pytest.fixture
    def mock_portfolio_service(self):
        """Mock portfolio service."""
        service = Mock()
        service.handle_position_update = AsyncMock()
        service.handle_price_update = AsyncMock()
        return service

    @pytest.fixture
    def portfolio_handler(self, mock_portfolio_service):
        """Portfolio event handler."""
        return PortfolioEventHandler(mock_portfolio_service)

    def test_portfolio_handler_can_handle(self, portfolio_handler):
        """Test which events portfolio handler can handle."""
        assert portfolio_handler.can_handle(AnalyticsEventType.POSITION_UPDATED)
        assert portfolio_handler.can_handle(AnalyticsEventType.PRICE_UPDATED)
        assert portfolio_handler.can_handle(AnalyticsEventType.BENCHMARK_UPDATED)
        assert not portfolio_handler.can_handle(AnalyticsEventType.TRADE_EXECUTED)

    @pytest.mark.asyncio
    async def test_handle_position_event(self, portfolio_handler, mock_portfolio_service):
        """Test handling position updated event."""
        event = AnalyticsEvent(
            event_id="test",
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=datetime.now(),
            source_component="test",
            data={"position": "data"},
        )

        await portfolio_handler.handle_event(event)

        mock_portfolio_service.handle_position_update.assert_called_once_with(event.data)

    @pytest.mark.asyncio
    async def test_handle_price_event(self, portfolio_handler, mock_portfolio_service):
        """Test handling price updated event."""
        event = AnalyticsEvent(
            event_id="test",
            event_type=AnalyticsEventType.PRICE_UPDATED,
            timestamp=datetime.now(),
            source_component="test",
            data={"price": "data"},
        )

        await portfolio_handler.handle_event(event)

        mock_portfolio_service.handle_price_update.assert_called_once_with(event.data)

    @pytest.mark.asyncio
    async def test_handle_unsupported_event(self, portfolio_handler, mock_portfolio_service):
        """Test handling unsupported event type."""
        event = AnalyticsEvent(
            event_id="test",
            event_type=AnalyticsEventType.TRADE_EXECUTED,
            timestamp=datetime.now(),
            source_component="test",
            data={},
        )

        await portfolio_handler.handle_event(event)

        # No methods should be called
        mock_portfolio_service.handle_position_update.assert_not_called()
        mock_portfolio_service.handle_price_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_event_service_without_method(self, mock_portfolio_service):
        """Test handling event when service doesn't have required method."""
        # Remove method from service
        del mock_portfolio_service.handle_position_update

        handler = PortfolioEventHandler(mock_portfolio_service)

        event = AnalyticsEvent(
            event_id="test",
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=datetime.now(),
            source_component="test",
            data={},
        )

        # Should not raise error
        await handler.handle_event(event)

    @pytest.mark.asyncio
    async def test_handle_event_with_exception(self, mock_portfolio_service):
        """Test handling event when service method raises exception."""
        mock_portfolio_service.handle_position_update.side_effect = Exception("Service error")

        handler = PortfolioEventHandler(mock_portfolio_service)

        event = AnalyticsEvent(
            event_id="test",
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=datetime.now(),
            source_component="test",
            data={},
        )

        # Should not raise error (exception is logged)
        await handler.handle_event(event)


class TestRiskEventHandler:
    """Test RiskEventHandler."""

    @pytest.fixture
    def mock_risk_service(self):
        """Mock risk service."""
        service = Mock()
        service.handle_position_update = AsyncMock()
        service.handle_price_update = AsyncMock()
        return service

    @pytest.fixture
    def risk_handler(self, mock_risk_service):
        """Risk event handler."""
        return RiskEventHandler(mock_risk_service)

    def test_risk_handler_can_handle(self, risk_handler):
        """Test which events risk handler can handle."""
        assert risk_handler.can_handle(AnalyticsEventType.POSITION_UPDATED)
        assert risk_handler.can_handle(AnalyticsEventType.PRICE_UPDATED)
        assert risk_handler.can_handle(AnalyticsEventType.TRADE_EXECUTED)
        assert not risk_handler.can_handle(AnalyticsEventType.ALERT_GENERATED)

    @pytest.mark.asyncio
    async def test_handle_position_event(self, risk_handler, mock_risk_service):
        """Test handling position updated event."""
        event = AnalyticsEvent(
            event_id="test",
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=datetime.now(),
            source_component="test",
            data={"position": "data"},
        )

        await risk_handler.handle_event(event)

        mock_risk_service.handle_position_update.assert_called_once_with(event.data)

    @pytest.mark.asyncio
    async def test_handle_price_event(self, risk_handler, mock_risk_service):
        """Test handling price updated event."""
        event = AnalyticsEvent(
            event_id="test",
            event_type=AnalyticsEventType.PRICE_UPDATED,
            timestamp=datetime.now(),
            source_component="test",
            data={"price": "data"},
        )

        await risk_handler.handle_event(event)

        mock_risk_service.handle_price_update.assert_called_once_with(event.data)


class TestAlertEventHandler:
    """Test AlertEventHandler."""

    @pytest.fixture
    def mock_alert_service(self):
        """Mock alert service."""
        service = Mock()
        service.handle_risk_limit_breach = AsyncMock()
        return service

    @pytest.fixture
    def alert_handler(self, mock_alert_service):
        """Alert event handler."""
        return AlertEventHandler(mock_alert_service)

    def test_alert_handler_can_handle(self, alert_handler):
        """Test which events alert handler can handle."""
        assert alert_handler.can_handle(AnalyticsEventType.RISK_LIMIT_BREACHED)
        assert alert_handler.can_handle(AnalyticsEventType.ERROR_OCCURRED)
        assert not alert_handler.can_handle(AnalyticsEventType.POSITION_UPDATED)

    @pytest.mark.asyncio
    async def test_handle_risk_limit_breach(self, alert_handler, mock_alert_service):
        """Test handling risk limit breach event."""
        event = AnalyticsEvent(
            event_id="test",
            event_type=AnalyticsEventType.RISK_LIMIT_BREACHED,
            timestamp=datetime.now(),
            source_component="test",
            data={"breach": "data"},
        )

        await alert_handler.handle_event(event)

        mock_alert_service.handle_risk_limit_breach.assert_called_once_with(event.data)


class TestEventPublishingFunctions:
    """Test global event publishing functions."""

    @pytest.fixture
    def sample_position(self):
        """Sample position for testing."""
        return Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("51000.00"),
            unrealized_pnl=Decimal("1000.00"),
            status=PositionStatus.OPEN,
            opened_at=datetime.now(),
            exchange="binance",
        )

    @pytest.fixture
    def sample_trade(self):
        """Sample trade for testing."""
        return Trade(
            trade_id="trade_001",
            order_id="order_001",
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            price=Decimal("3000.00"),
            quantity=Decimal("5.0"),
            fee=Decimal("2.5"),
            fee_currency="USDT",
            timestamp=datetime.now(),
            exchange="coinbase",
        )

    @pytest.fixture
    def sample_order(self):
        """Sample order for testing."""
        return Order(
            order_id="order_001",
            symbol="ADA/USDT",
            exchange="binance",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1000.0"),
            price=Decimal("0.50"),
            status=OrderStatus.FILLED,
            time_in_force=TimeInForce.GTC,
            created_at=datetime.now(),
        )

    @pytest.mark.asyncio
    async def test_publish_position_updated(self, sample_position):
        """Test publish_position_updated function."""
        with patch("src.analytics.events.get_event_bus") as mock_get_bus:
            mock_bus = Mock()
            mock_bus.publish_event = AsyncMock()
            mock_get_bus.return_value = mock_bus

            await publish_position_updated(sample_position, "test_component")

            mock_bus.publish_event.assert_called_once()
            event = mock_bus.publish_event.call_args[0][0]
            assert isinstance(event, PositionUpdatedEvent)
            assert event.source_component == "test_component"

    @pytest.mark.asyncio
    async def test_publish_trade_executed(self, sample_trade):
        """Test publish_trade_executed function."""
        with patch("src.analytics.events.get_event_bus") as mock_get_bus:
            mock_bus = Mock()
            mock_bus.publish_event = AsyncMock()
            mock_get_bus.return_value = mock_bus

            await publish_trade_executed(sample_trade, "execution_service")

            mock_bus.publish_event.assert_called_once()
            event = mock_bus.publish_event.call_args[0][0]
            assert isinstance(event, TradeExecutedEvent)
            assert event.source_component == "execution_service"

    @pytest.mark.asyncio
    async def test_publish_order_updated(self, sample_order):
        """Test publish_order_updated function."""
        with patch("src.analytics.events.get_event_bus") as mock_get_bus:
            mock_bus = Mock()
            mock_bus.publish_event = AsyncMock()
            mock_get_bus.return_value = mock_bus

            await publish_order_updated(sample_order, "order_service")

            mock_bus.publish_event.assert_called_once()
            event = mock_bus.publish_event.call_args[0][0]
            assert isinstance(event, OrderUpdatedEvent)
            assert event.source_component == "order_service"

    @pytest.mark.asyncio
    async def test_publish_price_updated(self):
        """Test publish_price_updated function."""
        with patch("src.analytics.events.get_event_bus") as mock_get_bus:
            mock_bus = Mock()
            mock_bus.publish_event = AsyncMock()
            mock_get_bus.return_value = mock_bus

            symbol = "BTC/USDT"
            price = Decimal("50000.00")
            timestamp = datetime.now()

            await publish_price_updated(symbol, price, timestamp, "market_data")

            mock_bus.publish_event.assert_called_once()
            event = mock_bus.publish_event.call_args[0][0]
            assert isinstance(event, PriceUpdatedEvent)
            assert event.source_component == "market_data"

    @pytest.mark.asyncio
    async def test_publish_risk_limit_breached(self):
        """Test publish_risk_limit_breached function."""
        with patch("src.analytics.events.get_event_bus") as mock_get_bus:
            mock_bus = Mock()
            mock_bus.publish_event = AsyncMock()
            mock_get_bus.return_value = mock_bus

            await publish_risk_limit_breached(
                limit_type="max_drawdown",
                current_value=-0.08,
                limit_value=-0.05,
                breach_percentage=0.6,
                source_component="risk_service",
            )

            mock_bus.publish_event.assert_called_once()
            event = mock_bus.publish_event.call_args[0][0]
            assert isinstance(event, RiskLimitBreachedEvent)
            assert event.source_component == "risk_service"


class TestGetEventBus:
    """Test get_event_bus global function."""

    def test_get_event_bus_singleton(self):
        """Test that get_event_bus returns singleton instance."""
        # Reset global state
        import src.analytics.events

        src.analytics.events._event_bus = None

        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2
        assert isinstance(bus1, AnalyticsEventBus)

    def test_get_event_bus_creates_instance(self):
        """Test that get_event_bus creates instance if none exists."""
        # Reset global state
        import src.analytics.events

        src.analytics.events._event_bus = None

        bus = get_event_bus()

        assert isinstance(bus, AnalyticsEventBus)
        assert src.analytics.events._event_bus is bus
