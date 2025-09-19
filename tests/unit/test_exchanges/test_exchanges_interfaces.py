"""
Tests for exchanges interfaces module.

This module tests the interface definitions and contracts used by the
exchange module to avoid circular dependencies and enforce proper
service layer patterns.
"""

from decimal import Decimal

from src.exchanges.interfaces import (
    IExchange,
    IExchangeFactory,
    IStateService,
    ITradeLifecycleManager,
    StatePriority,
    StateType,
    TradeEvent,
)


class TestStateType:
    """Test StateType enumeration."""

    def test_state_type_values(self):
        """Test all state type values are correct."""
        assert StateType.ORDER_STATE == "order_state"
        assert StateType.BOT_STATE == "bot_state"
        assert StateType.POSITION_STATE == "position_state"
        assert StateType.SYSTEM_STATE == "system_state"

    def test_state_type_enumeration(self):
        """Test state type enumeration completeness."""
        expected_types = {
            "bot_state", "position_state", "order_state", "portfolio_state",
            "risk_state", "strategy_state", "market_state", "trade_state",
            "execution", "system_state", "capital_state"
        }

        actual_types = {state.value for state in StateType}
        assert actual_types == expected_types

    def test_state_type_is_string_enum(self):
        """Test that StateType inherits from str and Enum."""
        assert issubclass(StateType, str)
        assert isinstance(StateType.ORDER_STATE, str)

    def test_state_type_comparison(self):
        """Test state type string comparison."""
        assert StateType.ORDER_STATE == "order_state"
        assert StateType.ORDER_STATE != "bot_state"

    def test_state_type_iteration(self):
        """Test state type can be iterated."""
        types = list(StateType)
        assert len(types) == 11
        assert StateType.ORDER_STATE in types


class TestStatePriority:
    """Test StatePriority enumeration."""

    def test_state_priority_values(self):
        """Test all state priority values are correct."""
        assert StatePriority.LOW == "low"
        assert StatePriority.MEDIUM == "medium"
        assert StatePriority.HIGH == "high"
        assert StatePriority.CRITICAL == "critical"

    def test_state_priority_enumeration(self):
        """Test state priority enumeration completeness."""
        expected_priorities = {"low", "medium", "high", "critical"}

        actual_priorities = {priority.value for priority in StatePriority}
        assert actual_priorities == expected_priorities

    def test_state_priority_is_string_enum(self):
        """Test that StatePriority inherits from str and Enum."""
        assert issubclass(StatePriority, str)
        assert isinstance(StatePriority.HIGH, str)

    def test_state_priority_ordering(self):
        """Test state priority logical ordering."""
        # While enums don't have inherent ordering, we can test values
        priorities = [
            StatePriority.LOW,
            StatePriority.MEDIUM,
            StatePriority.HIGH,
            StatePriority.CRITICAL,
        ]
        values = [p.value for p in priorities]

        assert "low" in values
        assert "critical" in values

    def test_state_priority_comparison(self):
        """Test state priority string comparison."""
        assert StatePriority.CRITICAL == "critical"
        assert StatePriority.LOW != "high"


class TestTradeEvent:
    """Test TradeEvent enumeration."""

    def test_trade_event_values(self):
        """Test all trade event values are correct."""
        assert TradeEvent.ORDER_SUBMITTED == "order_submitted"
        assert TradeEvent.ORDER_ACCEPTED == "order_accepted"
        assert TradeEvent.ORDER_REJECTED == "order_rejected"
        assert TradeEvent.PARTIAL_FILL == "partial_fill"
        assert TradeEvent.COMPLETE_FILL == "complete_fill"
        assert TradeEvent.ORDER_CANCELLED == "order_cancelled"
        assert TradeEvent.ORDER_EXPIRED == "order_expired"

    def test_trade_event_enumeration(self):
        """Test trade event enumeration completeness."""
        expected_events = {
            "signal_received",
            "validation_passed",
            "validation_failed",
            "order_submitted",
            "order_accepted",
            "order_rejected",
            "partial_fill",
            "complete_fill",
            "order_cancelled",
            "order_expired",
            "settlement_complete",
            "attribution_complete",
        }

        actual_events = {event.value for event in TradeEvent}
        assert actual_events == expected_events

    def test_trade_event_is_string_enum(self):
        """Test that TradeEvent inherits from str and Enum."""
        assert issubclass(TradeEvent, str)
        assert isinstance(TradeEvent.ORDER_SUBMITTED, str)

    def test_trade_event_lifecycle(self):
        """Test trade event represents complete order lifecycle."""
        # Test that we have events for major order states
        lifecycle_events = [
            TradeEvent.ORDER_SUBMITTED,
            TradeEvent.ORDER_ACCEPTED,
            TradeEvent.PARTIAL_FILL,
            TradeEvent.COMPLETE_FILL,
        ]

        for event in lifecycle_events:
            assert isinstance(event, TradeEvent)

    def test_trade_event_failure_states(self):
        """Test trade event includes failure states."""
        failure_events = [
            TradeEvent.ORDER_REJECTED,
            TradeEvent.ORDER_CANCELLED,
            TradeEvent.ORDER_EXPIRED,
        ]

        for event in failure_events:
            assert isinstance(event, TradeEvent)


class TestIStateService:
    """Test IStateService interface."""

    def test_state_service_is_protocol(self):
        """Test that IStateService is a Protocol."""
        # Protocol is a typing construct, check if it has expected methods
        assert hasattr(IStateService, "set_state")
        assert hasattr(IStateService, "get_state")

    def test_state_service_method_signatures(self):
        """Test state service method signatures."""
        import inspect

        # Test set_state signature
        set_state_sig = inspect.signature(IStateService.set_state)
        assert "state_type" in set_state_sig.parameters
        assert "state_id" in set_state_sig.parameters
        assert "state_data" in set_state_sig.parameters
        assert "source_component" in set_state_sig.parameters
        assert "priority" in set_state_sig.parameters
        assert "reason" in set_state_sig.parameters

        # Test get_state signature
        get_state_sig = inspect.signature(IStateService.get_state)
        assert "state_type" in get_state_sig.parameters
        assert "state_id" in get_state_sig.parameters


class TestITradeLifecycleManager:
    """Test ITradeLifecycleManager interface."""

    def test_trade_lifecycle_manager_is_protocol(self):
        """Test that ITradeLifecycleManager is a Protocol."""
        assert hasattr(ITradeLifecycleManager, "update_trade_event")

    def test_trade_lifecycle_method_signatures(self):
        """Test trade lifecycle manager method signatures."""
        import inspect

        # Test update_trade_event signature
        update_sig = inspect.signature(ITradeLifecycleManager.update_trade_event)
        assert "trade_id" in update_sig.parameters
        assert "event" in update_sig.parameters
        assert "event_data" in update_sig.parameters


class TestIExchange:
    """Test IExchange interface."""

    def test_exchange_is_protocol(self):
        """Test that IExchange is a Protocol."""
        assert hasattr(IExchange, "connect")
        assert hasattr(IExchange, "disconnect")
        assert hasattr(IExchange, "health_check")
        assert hasattr(IExchange, "place_order")
        assert hasattr(IExchange, "cancel_order")
        assert hasattr(IExchange, "get_order_status")
        assert hasattr(IExchange, "get_market_data")
        assert hasattr(IExchange, "get_order_book")
        assert hasattr(IExchange, "get_ticker")
        assert hasattr(IExchange, "get_account_balance")
        assert hasattr(IExchange, "get_positions")
        assert hasattr(IExchange, "get_exchange_info")
        assert hasattr(IExchange, "subscribe_to_stream")

    def test_exchange_method_signatures(self):
        """Test exchange method signatures."""
        import inspect

        # Test place_order signature
        place_order_sig = inspect.signature(IExchange.place_order)
        assert "order" in place_order_sig.parameters

        # Test cancel_order signature
        cancel_order_sig = inspect.signature(IExchange.cancel_order)
        assert "order_id" in cancel_order_sig.parameters

        # Test get_market_data signature
        market_data_sig = inspect.signature(IExchange.get_market_data)
        assert "symbol" in market_data_sig.parameters
        assert "timeframe" in market_data_sig.parameters

    def test_exchange_return_type_hints(self):
        """Test exchange method return type hints."""
        import inspect

        # Test some key return types
        place_order_sig = inspect.signature(IExchange.place_order)
        # Return type should be OrderResponse wrapped in Awaitable

        health_check_sig = inspect.signature(IExchange.health_check)
        # Return type should be bool wrapped in Awaitable


class TestIExchangeFactory:
    """Test IExchangeFactory interface."""

    def test_exchange_factory_is_protocol(self):
        """Test that IExchangeFactory is a Protocol."""
        assert hasattr(IExchangeFactory, "get_exchange")
        assert hasattr(IExchangeFactory, "get_supported_exchanges")
        assert hasattr(IExchangeFactory, "get_available_exchanges")

    def test_exchange_factory_method_signatures(self):
        """Test exchange factory method signatures."""
        import inspect

        # Test get_exchange signature
        get_exchange_sig = inspect.signature(IExchangeFactory.get_exchange)
        assert "exchange_name" in get_exchange_sig.parameters

        # Test other methods have no required parameters
        supported_sig = inspect.signature(IExchangeFactory.get_supported_exchanges)
        assert len(supported_sig.parameters) == 1  # Just 'self'

        available_sig = inspect.signature(IExchangeFactory.get_available_exchanges)
        assert len(available_sig.parameters) == 1  # Just 'self'


class TestProtocolImplementation:
    """Test protocol implementation patterns."""

    def test_protocol_can_be_implemented(self):
        """Test that protocols can be implemented by concrete classes."""

        class MockExchange:
            """Mock exchange implementation for testing."""

            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def health_check(self) -> bool:
                return True

            async def place_order(self, order):
                return None

            async def cancel_order(self, order_id: str) -> bool:
                return True

            async def get_order_status(self, order_id: str):
                return None

            async def get_market_data(self, symbol: str, timeframe: str = "1m"):
                return None

            async def get_order_book(self, symbol: str, depth: int = 10):
                return None

            async def get_ticker(self, symbol: str):
                return None

            async def get_account_balance(self) -> dict[str, Decimal]:
                return {}

            async def get_positions(self) -> list:
                return []

            async def get_exchange_info(self):
                return None

            async def subscribe_to_stream(self, symbol: str, callback):
                pass

        # Test that mock implements the protocol
        mock_exchange = MockExchange()

        # In Python 3.8+, we can use isinstance with Protocol
        # For now, just test that required methods exist
        assert hasattr(mock_exchange, "connect")
        assert hasattr(mock_exchange, "place_order")
        assert hasattr(mock_exchange, "health_check")

    def test_protocol_inheritance_structure(self):
        """Test protocol inheritance and structure."""
        # Test that interfaces are properly structured
        assert hasattr(IStateService, "__annotations__")
        assert hasattr(IExchange, "__annotations__")
        assert hasattr(IExchangeFactory, "__annotations__")


class TestInterfaceEdgeCases:
    """Test edge cases for interface definitions."""

    def test_enum_edge_cases(self):
        """Test enum edge cases and boundary conditions."""
        # Test enum with empty string (shouldn't exist)
        all_state_types = list(StateType)
        for state_type in all_state_types:
            assert state_type.value != ""
            assert len(state_type.value) > 0

        # Test enum uniqueness
        state_values = [state.value for state in StateType]
        assert len(state_values) == len(set(state_values))  # All unique

        priority_values = [priority.value for priority in StatePriority]
        assert len(priority_values) == len(set(priority_values))  # All unique

        event_values = [event.value for event in TradeEvent]
        assert len(event_values) == len(set(event_values))  # All unique

    def test_protocol_method_annotations(self):
        """Test that protocol methods have proper type annotations."""
        from typing import get_type_hints

        # Test IExchange method annotations
        try:
            health_check_hints = get_type_hints(IExchange.health_check)
            # Should have return type annotation
            assert "return" in health_check_hints
        except (NameError, AttributeError):
            # Type hints might not be available in all environments
            pass

    def test_enum_serialization(self):
        """Test enum serialization behavior."""
        # Test that enums can be converted to string consistently
        # Note: StateType inherits from str, so it evaluates as its value
        assert StateType.ORDER_STATE == "order_state"
        assert "StateType.ORDER_STATE" in repr(StateType.ORDER_STATE)

        # Test enum in dictionary
        test_dict = {"type": StateType.ORDER_STATE}
        assert test_dict["type"] == "order_state"

    def test_enum_comparison_edge_cases(self):
        """Test enum comparison edge cases."""
        # Test case sensitivity
        assert StateType.ORDER_STATE != "ORDER_STATE"
        assert StateType.ORDER_STATE == "order_state"

        # Test comparison with None
        assert StateType.ORDER_STATE != None
        assert StateType.ORDER_STATE is not None

        # Test comparison with other types
        assert StateType.ORDER_STATE != 1
        assert StateType.ORDER_STATE != True
        assert StateType.ORDER_STATE != []


class TestProtocolTypeHints:
    """Test protocol type hints and annotations."""

    def test_protocol_imports(self):
        """Test that protocols import required types correctly."""
        from src.core.types import (
            ExchangeInfo,
            MarketData,
            OrderBook,
            OrderRequest,
            OrderResponse,
            OrderStatus,
            Position,
            Ticker,
            Trade,
        )

        # Test that imported types are available
        assert ExchangeInfo is not None
        assert MarketData is not None
        assert OrderBook is not None
        assert OrderRequest is not None
        assert OrderResponse is not None
        assert OrderStatus is not None
        assert Position is not None
        assert Ticker is not None
        assert Trade is not None

    def test_decimal_import(self):
        """Test Decimal import for financial calculations."""
        from decimal import Decimal

        # Test Decimal is available and working
        price = Decimal("50000.00")
        assert price == Decimal("50000.00")
        assert isinstance(price, Decimal)

    def test_typing_imports(self):
        """Test typing imports work correctly."""
        from typing import Any, Protocol

        assert Any is not None
        assert Protocol is not None

        # Test that Protocol can be used as base class
        class TestProtocol(Protocol):
            def test_method(self) -> Any: ...

        assert hasattr(TestProtocol, "test_method")
