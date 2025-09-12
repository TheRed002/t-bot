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
        assert AnalyticsEventType.POSITION_UPDATED.value == "position.modified"
        assert AnalyticsEventType.TRADE_EXECUTED.value == "trade.executed"
        assert AnalyticsEventType.ORDER_UPDATED.value == "order.filled"
        assert AnalyticsEventType.PRICE_UPDATED.value == "market.price_update"
        assert AnalyticsEventType.RISK_LIMIT_BREACHED.value == "risk.limit_exceeded"
        assert AnalyticsEventType.ALERT_GENERATED.value == "alert.fired"
        assert AnalyticsEventType.SERVICE_STARTED.value == "system.startup"
        assert AnalyticsEventType.DATA_EXPORTED.value == "metric.exported"

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
            "event_type": AnalyticsEventType.POSITION_UPDATED,
            "timestamp": datetime.now(),
            "source": "test_component",
            "data": {"test": "data"},
        }

    def test_analytics_event_creation(self, sample_event_data):
        """Test creating AnalyticsEvent."""
        event = AnalyticsEvent(**sample_event_data)

        assert event.event_type == AnalyticsEventType.POSITION_UPDATED
        assert event.source == "test_component"
        assert event.data == {"test": "data"}
        assert event.timestamp == sample_event_data["timestamp"]

    def test_analytics_event_simple_creation(self):
        """Test AnalyticsEvent simple creation."""
        timestamp = datetime.now()
        event = AnalyticsEvent(
            event_type=AnalyticsEventType.TRADE_EXECUTED,
            timestamp=timestamp,
            source="test",
            data={},
        )

        assert event.event_type == AnalyticsEventType.TRADE_EXECUTED
        assert event.timestamp == timestamp
        assert event.source == "test"
        assert event.data == {}

    def test_analytics_event_serialization(self, sample_event_data):
        """Test event serialization to dict."""
        event = AnalyticsEvent(**sample_event_data)
        event_dict = event.model_dump()

        assert "event_type" in event_dict
        assert "timestamp" in event_dict
        assert "source" in event_dict
        assert "data" in event_dict


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

    def test_position_updated_event_creation(self, sample_position):
        """Test creating PositionUpdatedEvent."""
        timestamp = datetime.now()

        # PositionUpdatedEvent is just an alias to AnalyticsEvent
        event = PositionUpdatedEvent(
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=timestamp,
            source="portfolio_service",
            data={"position": sample_position.model_dump()}
        )

        assert event.event_type == AnalyticsEventType.POSITION_UPDATED
        assert event.source == "portfolio_service"
        assert event.timestamp == timestamp
        assert "position" in event.data
        assert event.data["position"]["symbol"] == "BTC/USDT"
        assert event.data["position"]["exchange"] == "binance"

    def test_position_updated_event_with_data(self, sample_position):
        """Test PositionUpdatedEvent with position data."""
        timestamp = datetime.now()
        event = PositionUpdatedEvent(
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=timestamp,
            source="test_service",
            data={"position": sample_position.model_dump(), "custom": "data"}
        )

        assert event.data["custom"] == "data"
        assert event.data["position"]["symbol"] == "BTC/USDT"


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
        # TradeExecutedEvent is just an alias to AnalyticsEvent
        event = TradeExecutedEvent(
            event_type=AnalyticsEventType.TRADE_EXECUTED,
            timestamp=sample_trade.timestamp,
            source="execution_service",
            data={"trade": sample_trade.model_dump()}
        )

        assert event.event_type == AnalyticsEventType.TRADE_EXECUTED
        assert event.source == "execution_service"
        assert event.timestamp == sample_trade.timestamp
        assert "trade" in event.data
        assert event.data["trade"]["trade_id"] == "trade_001"
        assert event.data["trade"]["symbol"] == "ETH/USDT"
        assert event.data["trade"]["exchange"] == "coinbase"


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
        # OrderUpdatedEvent is just an alias to AnalyticsEvent
        event = OrderUpdatedEvent(
            event_type=AnalyticsEventType.ORDER_UPDATED,
            timestamp=sample_order.updated_at or sample_order.created_at,
            source="order_service",
            data={"order": sample_order.model_dump()}
        )

        assert event.event_type == AnalyticsEventType.ORDER_UPDATED
        assert event.source == "order_service"
        assert event.timestamp == sample_order.updated_at or sample_order.created_at
        assert "order" in event.data
        assert event.data["order"]["order_id"] == "order_001"
        assert event.data["order"]["symbol"] == "ADA/USDT"
        assert event.data["order"]["exchange"] == "binance"

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

        event = OrderUpdatedEvent(
            event_type=AnalyticsEventType.ORDER_UPDATED,
            timestamp=order.updated_at or order.created_at,
            source="test_service",
            data={"order": order.model_dump()}
        )

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
        # PriceUpdatedEvent is just an alias to AnalyticsEvent
        event = PriceUpdatedEvent(
            event_type=AnalyticsEventType.PRICE_UPDATED,
            timestamp=sample_price_data["timestamp"],
            source=sample_price_data["source_component"],
            data={
                "symbol": sample_price_data["symbol"],
                "price": str(sample_price_data["price"]),
                "timestamp": sample_price_data["timestamp"].isoformat()
            }
        )

        assert event.event_type == AnalyticsEventType.PRICE_UPDATED
        assert event.source == "market_data_service"
        assert event.data["symbol"] == "BTC/USDT"
        assert event.data["price"] == "51000.00"  # Stored as string
        assert event.data["timestamp"] == sample_price_data["timestamp"].isoformat()

    def test_price_updated_event_decimal_precision(self):
        """Test price precision is preserved as string."""
        high_precision_price = Decimal("50000.12345678")
        timestamp = datetime.now()

        event = PriceUpdatedEvent(
            event_type=AnalyticsEventType.PRICE_UPDATED,
            timestamp=timestamp,
            source="test",
            data={
                "symbol": "BTC/USDT",
                "price": str(high_precision_price),
                "timestamp": timestamp.isoformat()
            }
        )

        assert event.data["price"] == "50000.12345678"


class TestRiskLimitBreachedEvent:
    """Test RiskLimitBreachedEvent."""

    def test_risk_limit_breached_event_creation(self):
        """Test creating RiskLimitBreachedEvent."""
        timestamp = datetime.now()

        # RiskLimitBreachedEvent is just an alias to AnalyticsEvent
        event = RiskLimitBreachedEvent(
            event_type=AnalyticsEventType.RISK_LIMIT_BREACHED,
            timestamp=timestamp,
            source="risk_service",
            data={
                "limit_type": "max_drawdown",
                "current_value": -0.08,
                "limit_value": -0.05,
                "breach_percentage": 0.6,
                "severity": "HIGH"
            }
        )

        assert event.event_type == AnalyticsEventType.RISK_LIMIT_BREACHED
        assert event.source == "risk_service"
        assert event.timestamp == timestamp
        assert event.data["limit_type"] == "max_drawdown"
        assert event.data["current_value"] == -0.08
        assert event.data["limit_value"] == -0.05
        assert event.data["breach_percentage"] == 0.6


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
        return handler

    @pytest.fixture
    def sample_event(self):
        """Sample event for testing."""
        return AnalyticsEvent(
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=datetime.now(),
            source="test",
            data={"test": "data"},
        )

    def test_event_bus_initialization(self, event_bus):
        """Test event bus initialization."""
        assert event_bus.handlers == {}
        assert hasattr(event_bus, 'logger')

    @pytest.mark.asyncio
    async def test_event_bus_start_stop(self, event_bus):
        """Test event bus start and stop."""
        # Test start - SimpleEventBus start/stop are no-ops
        await event_bus.start()
        
        # Test stop
        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_event_bus_double_start(self, event_bus):
        """Test that double start doesn't break anything."""
        await event_bus.start()
        await event_bus.start()  # Should be no-op

        # Clean up
        await event_bus.stop()

    def test_register_handler(self, event_bus, mock_handler):
        """Test registering event handlers."""
        event_type = AnalyticsEventType.POSITION_UPDATED

        event_bus.register_handler(event_type, mock_handler)

        assert event_type in event_bus.handlers
        assert mock_handler in event_bus.handlers[event_type]

    def test_register_multiple_handlers(self, event_bus):
        """Test registering multiple handlers for same event type."""
        handler1 = Mock(spec=AnalyticsEventHandler)
        handler2 = Mock(spec=AnalyticsEventHandler)
        event_type = AnalyticsEventType.TRADE_EXECUTED

        event_bus.register_handler(event_type, handler1)
        event_bus.register_handler(event_type, handler2)

        assert len(event_bus.handlers[event_type]) == 2
        assert handler1 in event_bus.handlers[event_type]
        assert handler2 in event_bus.handlers[event_type]

    def test_register_handler_creates_list(self, event_bus, mock_handler):
        """Test registering handler creates handler list."""
        event_type = AnalyticsEventType.POSITION_UPDATED

        # Register handler
        event_bus.register_handler(event_type, mock_handler)
        
        # Should create list with handler
        assert event_type in event_bus.handlers
        assert len(event_bus.handlers[event_type]) == 1
        assert event_bus.handlers[event_type][0] == mock_handler

    @pytest.mark.asyncio
    async def test_publish_event(self, event_bus, sample_event, mock_handler):
        """Test publishing events."""
        # Register handler first
        event_bus.register_handler(sample_event.event_type, mock_handler)
        
        # Publish event
        await event_bus.publish(sample_event)
        
        # Handler should have been called
        mock_handler.handle_event.assert_called_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_publish_event_without_handlers(self, event_bus, sample_event):
        """Test publishing events without handlers doesn't raise error."""
        # Publish event without any registered handlers
        await event_bus.publish(sample_event)
        
        # Should complete without error

    @pytest.mark.asyncio
    async def test_event_processing_multiple_handlers(self, event_bus, sample_event):
        """Test event processing with multiple handlers."""
        handler1 = Mock(spec=AnalyticsEventHandler)
        handler1.handle_event = AsyncMock()
        handler2 = Mock(spec=AnalyticsEventHandler)
        handler2.handle_event = AsyncMock()

        # Register handlers
        event_bus.register_handler(sample_event.event_type, handler1)
        event_bus.register_handler(sample_event.event_type, handler2)

        # Publish event
        await event_bus.publish(sample_event)

        # Both handlers should have been called
        handler1.handle_event.assert_called_once_with(sample_event)
        handler2.handle_event.assert_called_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_event_processing_error_handling(self, event_bus, sample_event):
        """Test event processing with handler errors."""
        handler = Mock(spec=AnalyticsEventHandler)
        handler.handle_event = AsyncMock(side_effect=Exception("Handler error"))

        # Register handler
        event_bus.register_handler(sample_event.event_type, handler)

        # Publish event - should not raise error
        await event_bus.publish(sample_event)

        # Handler should have been called despite error
        handler.handle_event.assert_called_once_with(sample_event)


class TestPortfolioEventHandler:
    """Test PortfolioEventHandler."""

    @pytest.fixture
    def mock_portfolio_service(self):
        """Mock portfolio service."""
        service = Mock()
        service.update_position = Mock()
        return service

    @pytest.fixture
    def portfolio_handler(self, mock_portfolio_service):
        """Portfolio event handler."""
        return PortfolioEventHandler(mock_portfolio_service)

    def test_portfolio_handler_initialization(self, portfolio_handler, mock_portfolio_service):
        """Test portfolio handler initialization."""
        assert portfolio_handler.service == mock_portfolio_service
        assert hasattr(portfolio_handler, 'logger')

    @pytest.mark.asyncio
    async def test_handle_position_event(self, portfolio_handler, mock_portfolio_service):
        """Test handling position updated event."""
        # Create a position data dictionary that matches Position model
        position_data = {
            "symbol": "BTC/USDT",
            "side": "LONG",
            "quantity": "1.5",
            "entry_price": "50000.00",
            "current_price": "52000.00",
            "unrealized_pnl": "3000.00",
            "status": "OPEN",
            "opened_at": datetime.now().isoformat(),
            "exchange": "binance"
        }
        
        event = AnalyticsEvent(
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=datetime.now(),
            source="test",
            data={"position": position_data},
        )

        await portfolio_handler.handle_event(event)

        # Should call update_position with Position object
        mock_portfolio_service.update_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_non_position_event(self, portfolio_handler, mock_portfolio_service):
        """Test handling non-position event."""
        event = AnalyticsEvent(
            event_type=AnalyticsEventType.PRICE_UPDATED,
            timestamp=datetime.now(),
            source="test",
            data={"price": "data"},
        )

        await portfolio_handler.handle_event(event)

        # Should not call update_position for non-position events
        mock_portfolio_service.update_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_trade_event(self, portfolio_handler, mock_portfolio_service):
        """Test handling trade event (should be ignored)."""
        event = AnalyticsEvent(
            event_type=AnalyticsEventType.TRADE_EXECUTED,
            timestamp=datetime.now(),
            source="test",
            data={},
        )

        await portfolio_handler.handle_event(event)

        # Should not call update_position
        mock_portfolio_service.update_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_event_service_without_method(self, mock_portfolio_service):
        """Test handling event when service doesn't have required method."""
        # Remove method from service
        del mock_portfolio_service.update_position

        handler = PortfolioEventHandler(mock_portfolio_service)

        event = AnalyticsEvent(
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=datetime.now(),
            source="test",
            data={"position": {"symbol": "BTC/USDT"}},
        )

        # Should not raise error
        await handler.handle_event(event)

    @pytest.mark.asyncio
    async def test_handle_event_no_service(self):
        """Test handling event when no service provided."""
        handler = PortfolioEventHandler(None)

        event = AnalyticsEvent(
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=datetime.now(),
            source="test",
            data={"position": {"symbol": "BTC/USDT"}},
        )

        # Should not raise error when service is None
        await handler.handle_event(event)


class TestRiskEventHandler:
    """Test RiskEventHandler."""

    @pytest.fixture
    def mock_risk_service(self):
        """Mock risk service."""
        service = Mock()
        service.update_position = Mock()
        return service

    @pytest.fixture
    def risk_handler(self, mock_risk_service):
        """Risk event handler."""
        return RiskEventHandler(mock_risk_service)

    def test_risk_handler_initialization(self, risk_handler, mock_risk_service):
        """Test risk handler initialization."""
        assert risk_handler.service == mock_risk_service
        assert hasattr(risk_handler, 'logger')

    @pytest.mark.asyncio
    async def test_handle_position_event(self, risk_handler, mock_risk_service):
        """Test handling position updated event."""
        position_data = {
            "symbol": "BTC/USDT",
            "side": "LONG",
            "quantity": "1.5",
            "entry_price": "50000.00",
            "current_price": "52000.00",
            "unrealized_pnl": "3000.00",
            "status": "OPEN",
            "opened_at": datetime.now().isoformat(),
            "exchange": "binance"
        }
        
        event = AnalyticsEvent(
            event_type=AnalyticsEventType.POSITION_UPDATED,
            timestamp=datetime.now(),
            source="test",
            data={"position": position_data},
        )

        await risk_handler.handle_event(event)

        mock_risk_service.update_position.assert_called_once()


class TestAlertEventHandler:
    """Test AlertEventHandler."""

    @pytest.fixture
    def mock_alert_service(self):
        """Mock alert service."""
        service = Mock()
        service.handle_alert = Mock()
        return service

    @pytest.fixture
    def alert_handler(self, mock_alert_service):
        """Alert event handler."""
        return AlertEventHandler(mock_alert_service)

    def test_alert_handler_initialization(self, alert_handler, mock_alert_service):
        """Test alert handler initialization."""
        assert alert_handler.service == mock_alert_service
        assert hasattr(alert_handler, 'logger')

    @pytest.mark.asyncio
    async def test_handle_risk_alert(self, alert_handler, mock_alert_service):
        """Test handling risk alert event."""
        event = AnalyticsEvent(
            event_type=AnalyticsEventType.RISK_ALERT,
            timestamp=datetime.now(),
            source="test",
            data={"alert": "data"},
        )

        await alert_handler.handle_event(event)

        mock_alert_service.handle_alert.assert_called_once_with(event.data)


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
        with patch("src.analytics.events._event_bus") as mock_bus:
            mock_bus.publish = AsyncMock()

            await publish_position_updated(sample_position, "test_component")

            mock_bus.publish.assert_called_once()
            event = mock_bus.publish.call_args[0][0]
            assert isinstance(event, AnalyticsEvent)
            assert event.source == "test_component"
            assert event.event_type == AnalyticsEventType.POSITION_UPDATED

    @pytest.mark.asyncio
    async def test_publish_trade_executed(self, sample_trade):
        """Test publish_trade_executed function."""
        with patch("src.analytics.events._event_bus") as mock_bus:
            mock_bus.publish = AsyncMock()

            await publish_trade_executed(sample_trade, "execution_service")

            mock_bus.publish.assert_called_once()
            event = mock_bus.publish.call_args[0][0]
            assert isinstance(event, AnalyticsEvent)
            assert event.source == "execution_service"
            assert event.event_type == AnalyticsEventType.TRADE_EXECUTED

    @pytest.mark.asyncio
    async def test_publish_order_updated(self, sample_order):
        """Test publish_order_updated function."""
        with patch("src.analytics.events._event_bus") as mock_bus:
            mock_bus.publish = AsyncMock()

            await publish_order_updated(sample_order, "order_service")

            mock_bus.publish.assert_called_once()
            event = mock_bus.publish.call_args[0][0]
            assert isinstance(event, AnalyticsEvent)
            assert event.source == "order_service"
            assert event.event_type == AnalyticsEventType.ORDER_UPDATED

    @pytest.mark.asyncio
    async def test_publish_price_updated(self):
        """Test publish_price_updated function."""
        with patch("src.analytics.events._event_bus") as mock_bus:
            mock_bus.publish = AsyncMock()

            symbol = "BTC/USDT"
            price = Decimal("50000.00")
            timestamp = datetime.now()

            await publish_price_updated(symbol, price, timestamp, "market_data")

            mock_bus.publish.assert_called_once()
            event = mock_bus.publish.call_args[0][0]
            assert isinstance(event, AnalyticsEvent)
            assert event.source == "market_data"
            assert event.event_type == AnalyticsEventType.PRICE_UPDATED

    @pytest.mark.asyncio
    async def test_publish_risk_limit_breached(self):
        """Test publish_risk_limit_breached function."""
        with patch("src.analytics.events._event_bus") as mock_bus:
            mock_bus.publish = AsyncMock()

            await publish_risk_limit_breached(
                symbol="BTC/USDT",
                limit_type="max_drawdown",
                current_value=Decimal("-0.08"),
                limit_value=Decimal("-0.05"),
                source="risk_service",
            )

            mock_bus.publish.assert_called_once()
            event = mock_bus.publish.call_args[0][0]
            assert isinstance(event, AnalyticsEvent)
            assert event.source == "risk_service"
            assert event.event_type == AnalyticsEventType.RISK_LIMIT_BREACHED


class TestGetEventBus:
    """Test get_event_bus global function."""

    def test_get_event_bus_returns_instance(self):
        """Test that get_event_bus returns the global instance."""
        bus = get_event_bus()
        assert isinstance(bus, AnalyticsEventBus)
        
    def test_get_event_bus_is_singleton(self):
        """Test that get_event_bus returns same instance."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2
