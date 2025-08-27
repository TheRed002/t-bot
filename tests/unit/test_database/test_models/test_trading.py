"""
Unit tests for trading database models.

This module tests all trading-related models including:
- Order
- Position  
- OrderFill
- Trade
"""

import uuid
from decimal import Decimal
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.database.models.base import Base
from src.database.models.trading import Order, OrderFill, Position, Trade


class TestOrderModel:
    """Test Order model functionality."""

    def test_order_model_creation(self):
        """Test Order model instance creation."""
        order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="PENDING",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0"),
        )
        
        assert order.exchange == "binance"
        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
        assert order.type == "LIMIT"
        assert order.status == "PENDING"
        assert order.price == Decimal("50000.00")
        assert order.quantity == Decimal("0.1")
        assert order.filled_quantity == Decimal("0")

    def test_order_id_generation(self):
        """Test Order ID is generated as UUID."""
        order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="MARKET",
            status="PENDING",
            quantity=Decimal("0.1"),
        )
        
        # ID should be generated automatically
        assert order.id is not None
        assert isinstance(order.id, uuid.UUID)

    def test_order_required_fields(self):
        """Test Order model with required fields only."""
        order = Order(
            exchange="coinbase",
            symbol="ETHUSD",
            side="SELL",
            type="MARKET",
            status="FILLED",
            quantity=Decimal("1.5"),
        )
        
        assert order.exchange == "coinbase"
        assert order.symbol == "ETHUSD"
        assert order.side == "SELL"
        assert order.type == "MARKET"
        assert order.status == "FILLED"
        assert order.quantity == Decimal("1.5")
        assert order.price is None  # Optional field

    def test_order_is_filled_property_true(self):
        """Test is_filled property returns True when order is fully filled."""
        order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="FILLED",
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
        )
        
        assert order.is_filled is True

    def test_order_is_filled_property_false_status(self):
        """Test is_filled property returns False when status is not FILLED."""
        order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="PENDING",
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
        )
        
        assert order.is_filled is False

    def test_order_is_filled_property_false_quantity(self):
        """Test is_filled property returns False when not fully filled."""
        order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="FILLED",
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.05"),
        )
        
        assert order.is_filled is False

    def test_order_is_filled_property_none_values(self):
        """Test is_filled property with None values."""
        order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="FILLED",
            quantity=Decimal("0.1"),
            filled_quantity=None,
        )
        
        assert order.is_filled is False

    def test_order_is_active_property_true(self):
        """Test is_active property returns True for active orders."""
        active_statuses = ["PENDING", "OPEN", "PARTIALLY_FILLED"]
        
        for status in active_statuses:
            order = Order(
                exchange="binance",
                symbol="BTCUSDT",
                side="BUY",
                type="LIMIT",
                status=status,
                quantity=Decimal("0.1"),
            )
            assert order.is_active is True, f"Status {status} should be active"

    def test_order_is_active_property_false(self):
        """Test is_active property returns False for inactive orders."""
        inactive_statuses = ["FILLED", "CANCELLED", "REJECTED"]
        
        for status in inactive_statuses:
            order = Order(
                exchange="binance",
                symbol="BTCUSDT",
                side="BUY",
                type="LIMIT",
                status=status,
                quantity=Decimal("0.1"),
            )
            assert order.is_active is False, f"Status {status} should be inactive"

    def test_order_remaining_quantity_property(self):
        """Test remaining_quantity property calculation."""
        order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="PARTIALLY_FILLED",
            quantity=Decimal("1.0"),
            filled_quantity=Decimal("0.3"),
        )
        
        assert order.remaining_quantity == Decimal("0.7")

    def test_order_remaining_quantity_no_fills(self):
        """Test remaining_quantity when no fills."""
        order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="PENDING",
            quantity=Decimal("1.0"),
            filled_quantity=None,
        )
        
        assert order.remaining_quantity == Decimal("1.0")

    def test_order_remaining_quantity_none_quantity(self):
        """Test remaining_quantity when quantity is None."""
        order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="PENDING",
            quantity=None,
        )
        
        assert order.remaining_quantity == Decimal("0")

    def test_order_repr(self):
        """Test Order string representation."""
        order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="PENDING",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
        )
        
        repr_str = repr(order)
        assert "Order" in repr_str
        assert "BUY" in repr_str
        assert "0.1" in repr_str
        assert "BTCUSDT" in repr_str
        assert "50000.00" in repr_str

    def test_order_table_structure(self):
        """Test Order table structure and constraints."""
        # Check table name
        assert Order.__tablename__ == "orders"
        
        # Check required columns exist
        columns = {col.name for col in Order.__table__.columns}
        expected_columns = {
            "id", "exchange", "exchange_order_id", "symbol", "side", "type", "status",
            "price", "quantity", "filled_quantity", "average_fill_price", "bot_id",
            "strategy_id", "position_id", "created_at", "updated_at", "created_by",
            "updated_by", "version", "metadata_json"
        }
        assert expected_columns.issubset(columns)
        
        # Check constraints
        constraint_names = {constraint.name for constraint in Order.__table__.constraints}
        assert "check_quantity_positive" in constraint_names
        assert "check_filled_quantity_non_negative" in constraint_names
        assert "check_filled_quantity_max" in constraint_names


class TestPositionModel:
    """Test Position model functionality."""

    def test_position_model_creation(self):
        """Test Position model instance creation."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="LONG",
            status="OPEN",
            quantity=Decimal("0.5"),
            entry_price=Decimal("45000.00"),
            current_price=Decimal("50000.00"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("2500.00"),
        )
        
        assert position.exchange == "binance"
        assert position.symbol == "BTCUSDT"
        assert position.side == "LONG"
        assert position.status == "OPEN"
        assert position.quantity == Decimal("0.5")
        assert position.entry_price == Decimal("45000.00")
        assert position.current_price == Decimal("50000.00")
        assert position.realized_pnl == Decimal("0")
        assert position.unrealized_pnl == Decimal("2500.00")

    def test_position_id_generation(self):
        """Test Position ID is generated as UUID."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="LONG",
            status="OPEN",
            quantity=Decimal("0.5"),
            entry_price=Decimal("45000.00"),
        )
        
        assert position.id is not None
        assert isinstance(position.id, uuid.UUID)

    def test_position_is_open_property_true(self):
        """Test is_open property returns True when status is OPEN."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="LONG",
            status="OPEN",
            quantity=Decimal("0.5"),
            entry_price=Decimal("45000.00"),
        )
        
        assert position.is_open is True

    def test_position_is_open_property_false(self):
        """Test is_open property returns False when status is not OPEN."""
        closed_statuses = ["CLOSED", "LIQUIDATED"]
        
        for status in closed_statuses:
            position = Position(
                exchange="binance",
                symbol="BTCUSDT",
                side="LONG",
                status=status,
                quantity=Decimal("0.5"),
                entry_price=Decimal("45000.00"),
            )
            assert position.is_open is False

    def test_position_value_property_with_current_price(self):
        """Test value property uses current_price when available."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="LONG",
            status="OPEN",
            quantity=Decimal("0.5"),
            entry_price=Decimal("45000.00"),
            current_price=Decimal("50000.00"),
        )
        
        assert position.value == Decimal("25000.00")  # 0.5 * 50000

    def test_position_value_property_with_entry_price(self):
        """Test value property falls back to entry_price."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="LONG",
            status="OPEN",
            quantity=Decimal("0.5"),
            entry_price=Decimal("45000.00"),
            current_price=None,
        )
        
        assert position.value == Decimal("22500.00")  # 0.5 * 45000

    def test_position_value_property_none_values(self):
        """Test value property with None values."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="LONG",
            status="OPEN",
            quantity=None,
            entry_price=None,
        )
        
        assert position.value == Decimal("0")

    def test_position_calculate_pnl_long_profitable(self):
        """Test P&L calculation for profitable long position."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="LONG",
            status="OPEN",
            quantity=Decimal("1.0"),
            entry_price=Decimal("45000.00"),
        )
        
        pnl = position.calculate_pnl(Decimal("50000.00"))
        assert pnl == Decimal("5000.00")  # (50000 - 45000) * 1

    def test_position_calculate_pnl_long_loss(self):
        """Test P&L calculation for losing long position."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="LONG",
            status="OPEN",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
        )
        
        pnl = position.calculate_pnl(Decimal("45000.00"))
        assert pnl == Decimal("-5000.00")  # (45000 - 50000) * 1

    def test_position_calculate_pnl_short_profitable(self):
        """Test P&L calculation for profitable short position."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="SHORT",
            status="OPEN",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
        )
        
        pnl = position.calculate_pnl(Decimal("45000.00"))
        assert pnl == Decimal("5000.00")  # (50000 - 45000) * 1

    def test_position_calculate_pnl_short_loss(self):
        """Test P&L calculation for losing short position."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="SHORT",
            status="OPEN",
            quantity=Decimal("1.0"),
            entry_price=Decimal("45000.00"),
        )
        
        pnl = position.calculate_pnl(Decimal("50000.00"))
        assert pnl == Decimal("-5000.00")  # (45000 - 50000) * 1

    def test_position_calculate_pnl_current_price(self):
        """Test P&L calculation using current_price from position."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="LONG",
            status="OPEN",
            quantity=Decimal("1.0"),
            entry_price=Decimal("45000.00"),
            current_price=Decimal("48000.00"),
        )
        
        pnl = position.calculate_pnl()  # No price argument
        assert pnl == Decimal("3000.00")  # (48000 - 45000) * 1

    def test_position_calculate_pnl_no_price(self):
        """Test P&L calculation with no price data."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="LONG",
            status="OPEN",
            quantity=Decimal("1.0"),
            entry_price=None,
        )
        
        pnl = position.calculate_pnl()
        assert pnl == Decimal("0")

    def test_position_repr(self):
        """Test Position string representation."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="LONG",
            status="OPEN",
            quantity=Decimal("0.5"),
            entry_price=Decimal("45000.00"),
        )
        
        repr_str = repr(position)
        assert "Position" in repr_str
        assert "LONG" in repr_str
        assert "0.5" in repr_str
        assert "BTCUSDT" in repr_str

    def test_position_table_structure(self):
        """Test Position table structure and constraints."""
        # Check table name
        assert Position.__tablename__ == "positions"
        
        # Check required columns exist
        columns = {col.name for col in Position.__table__.columns}
        expected_columns = {
            "id", "exchange", "symbol", "side", "status", "quantity", "entry_price",
            "exit_price", "current_price", "realized_pnl", "unrealized_pnl", "bot_id",
            "strategy_id", "stop_loss", "take_profit", "max_position_size",
            "created_at", "updated_at", "created_by", "updated_by", "version", "metadata_json"
        }
        assert expected_columns.issubset(columns)
        
        # Check constraints
        constraint_names = {constraint.name for constraint in Position.__table__.constraints}
        assert "check_position_quantity_positive" in constraint_names


class TestOrderFillModel:
    """Test OrderFill model functionality."""

    def test_order_fill_model_creation(self):
        """Test OrderFill model instance creation."""
        fill = OrderFill(
            order_id=uuid.uuid4(),
            exchange_fill_id="fill_123",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            fee=Decimal("5.00"),
            fee_currency="USDT",
        )
        
        assert fill.order_id is not None
        assert fill.exchange_fill_id == "fill_123"
        assert fill.price == Decimal("50000.00")
        assert fill.quantity == Decimal("0.1")
        assert fill.fee == Decimal("5.00")
        assert fill.fee_currency == "USDT"

    def test_order_fill_id_generation(self):
        """Test OrderFill ID is generated as UUID."""
        fill = OrderFill(
            order_id=uuid.uuid4(),
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
        )
        
        assert fill.id is not None
        assert isinstance(fill.id, uuid.UUID)

    def test_order_fill_required_fields_only(self):
        """Test OrderFill with required fields only."""
        order_id = uuid.uuid4()
        fill = OrderFill(
            order_id=order_id,
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
        )
        
        assert fill.order_id == order_id
        assert fill.price == Decimal("50000.00")
        assert fill.quantity == Decimal("0.1")
        assert fill.fee == Decimal("0")  # Default value
        assert fill.fee_currency is None

    def test_order_fill_value_property(self):
        """Test value property calculation."""
        fill = OrderFill(
            order_id=uuid.uuid4(),
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
        )
        
        assert fill.value == Decimal("5000.00")  # 50000 * 0.1

    def test_order_fill_value_property_none_values(self):
        """Test value property with None values."""
        fill = OrderFill(
            order_id=uuid.uuid4(),
            price=None,
            quantity=None,
        )
        
        assert fill.value == Decimal("0")

    def test_order_fill_net_value_property(self):
        """Test net_value property calculation."""
        fill = OrderFill(
            order_id=uuid.uuid4(),
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            fee=Decimal("5.00"),
        )
        
        assert fill.net_value == Decimal("4995.00")  # 5000 - 5

    def test_order_fill_net_value_no_fee(self):
        """Test net_value property with no fee."""
        fill = OrderFill(
            order_id=uuid.uuid4(),
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            fee=None,
        )
        
        assert fill.net_value == Decimal("5000.00")

    def test_order_fill_repr(self):
        """Test OrderFill string representation."""
        fill = OrderFill(
            order_id=uuid.uuid4(),
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
        )
        
        repr_str = repr(fill)
        assert "OrderFill" in repr_str
        assert "0.1" in repr_str
        assert "50000.00" in repr_str

    def test_order_fill_table_structure(self):
        """Test OrderFill table structure and constraints."""
        # Check table name
        assert OrderFill.__tablename__ == "order_fills"
        
        # Check required columns exist
        columns = {col.name for col in OrderFill.__table__.columns}
        expected_columns = {
            "id", "order_id", "exchange_fill_id", "price", "quantity", "fee",
            "fee_currency", "created_at", "updated_at"
        }
        assert expected_columns.issubset(columns)
        
        # Check constraints
        constraint_names = {constraint.name for constraint in OrderFill.__table__.constraints}
        assert "check_fill_quantity_positive" in constraint_names
        assert "check_fill_price_positive" in constraint_names


class TestTradeModel:
    """Test Trade model functionality."""

    def test_trade_model_creation(self):
        """Test Trade model instance creation."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            entry_order_id=uuid.uuid4(),
            exit_order_id=uuid.uuid4(),
            quantity=Decimal("0.1"),
            entry_price=Decimal("45000.00"),
            exit_price=Decimal("50000.00"),
            pnl=Decimal("500.00"),
            pnl_percentage=Decimal("11.1111"),
            fees=Decimal("10.00"),
            net_pnl=Decimal("490.00"),
        )
        
        assert trade.exchange == "binance"
        assert trade.symbol == "BTCUSDT"
        assert trade.side == "BUY"
        assert trade.quantity == Decimal("0.1")
        assert trade.entry_price == Decimal("45000.00")
        assert trade.exit_price == Decimal("50000.00")
        assert trade.pnl == Decimal("500.00")
        assert trade.pnl_percentage == Decimal("11.1111")
        assert trade.fees == Decimal("10.00")
        assert trade.net_pnl == Decimal("490.00")

    def test_trade_id_generation(self):
        """Test Trade ID is generated as UUID."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=Decimal("45000.00"),
            exit_price=Decimal("50000.00"),
            pnl=Decimal("500.00"),
        )
        
        assert trade.id is not None
        assert isinstance(trade.id, uuid.UUID)

    def test_trade_required_fields_only(self):
        """Test Trade with required fields only."""
        trade = Trade(
            exchange="coinbase",
            symbol="ETHUSD",
            side="SELL",
            quantity=Decimal("1.0"),
            entry_price=Decimal("3000.00"),
            exit_price=Decimal("2800.00"),
            pnl=Decimal("-200.00"),
        )
        
        assert trade.exchange == "coinbase"
        assert trade.symbol == "ETHUSD"
        assert trade.side == "SELL"
        assert trade.quantity == Decimal("1.0")
        assert trade.entry_price == Decimal("3000.00")
        assert trade.exit_price == Decimal("2800.00")
        assert trade.pnl == Decimal("-200.00")
        assert trade.fees == Decimal("0")  # Default value

    def test_trade_is_profitable_property_true(self):
        """Test is_profitable property returns True for positive P&L."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=Decimal("45000.00"),
            exit_price=Decimal("50000.00"),
            pnl=Decimal("500.00"),
        )
        
        assert trade.is_profitable is True

    def test_trade_is_profitable_property_false(self):
        """Test is_profitable property returns False for negative P&L."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000.00"),
            exit_price=Decimal("45000.00"),
            pnl=Decimal("-500.00"),
        )
        
        assert trade.is_profitable is False

    def test_trade_is_profitable_property_zero(self):
        """Test is_profitable property with zero P&L."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000.00"),
            exit_price=Decimal("50000.00"),
            pnl=Decimal("0"),
        )
        
        assert trade.is_profitable is False

    def test_trade_is_profitable_property_none(self):
        """Test is_profitable property with None P&L."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000.00"),
            exit_price=Decimal("50000.00"),
            pnl=None,
        )
        
        assert trade.is_profitable is False

    def test_trade_return_percentage_property_positive(self):
        """Test return_percentage property for positive return."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=Decimal("40000.00"),
            exit_price=Decimal("50000.00"),
            pnl=Decimal("1000.00"),
        )
        
        expected = Decimal("25.00")  # (50000 - 40000) / 40000 * 100
        assert trade.return_percentage == expected

    def test_trade_return_percentage_property_negative(self):
        """Test return_percentage property for negative return."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000.00"),
            exit_price=Decimal("40000.00"),
            pnl=Decimal("-1000.00"),
        )
        
        expected = Decimal("-20.00")  # (40000 - 50000) / 50000 * 100
        assert trade.return_percentage == expected

    def test_trade_return_percentage_zero_entry_price(self):
        """Test return_percentage property with zero entry price."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=Decimal("0"),
            exit_price=Decimal("50000.00"),
            pnl=Decimal("5000.00"),
        )
        
        assert trade.return_percentage == Decimal("0")

    def test_trade_return_percentage_none_entry_price(self):
        """Test return_percentage property with None entry price."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=None,
            exit_price=Decimal("50000.00"),
            pnl=Decimal("5000.00"),
        )
        
        assert trade.return_percentage == Decimal("0")

    def test_trade_return_percentage_none_exit_price(self):
        """Test return_percentage property with None exit price."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=Decimal("40000.00"),
            exit_price=None,
            pnl=Decimal("5000.00"),
        )
        
        expected = Decimal("-100.00")  # (0 - 40000) / 40000 * 100
        assert trade.return_percentage == expected

    def test_trade_repr(self):
        """Test Trade string representation."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=Decimal("45000.00"),
            exit_price=Decimal("50000.00"),
            pnl=Decimal("500.00"),
        )
        
        repr_str = repr(trade)
        assert "Trade" in repr_str
        assert "BTCUSDT" in repr_str
        assert "500.00" in repr_str

    def test_trade_table_structure(self):
        """Test Trade table structure and constraints."""
        # Check table name
        assert Trade.__tablename__ == "trades"
        
        # Check required columns exist
        columns = {col.name for col in Trade.__table__.columns}
        expected_columns = {
            "id", "exchange", "symbol", "side", "entry_order_id", "exit_order_id",
            "position_id", "quantity", "entry_price", "exit_price", "pnl",
            "pnl_percentage", "fees", "net_pnl", "bot_id", "strategy_id",
            "created_at", "updated_at", "metadata_json"
        }
        assert expected_columns.issubset(columns)


class TestTradingModelRelationships:
    """Test relationships between trading models."""

    def test_order_fill_relationship(self):
        """Test Order to OrderFill relationship."""
        # Check that Order has fills relationship
        assert hasattr(Order, 'fills')
        
        # Check that OrderFill has order relationship
        assert hasattr(OrderFill, 'order')
        
        # Check relationship back_populates
        order_fills_rel = Order.fills.property
        fill_order_rel = OrderFill.order.property
        
        assert order_fills_rel.back_populates == "order"
        assert fill_order_rel.back_populates == "fills"

    def test_order_position_relationship(self):
        """Test Order to Position relationship."""
        # Check that Order has position relationship
        assert hasattr(Order, 'position')
        
        # Check that Position has orders relationship
        assert hasattr(Position, 'orders')
        
        # Check relationship back_populates
        order_position_rel = Order.position.property
        position_orders_rel = Position.orders.property
        
        assert order_position_rel.back_populates == "orders"
        assert position_orders_rel.back_populates == "position"

    def test_trade_order_relationships(self):
        """Test Trade to Order relationships."""
        # Check that Trade has entry_order and exit_order relationships
        assert hasattr(Trade, 'entry_order')
        assert hasattr(Trade, 'exit_order')
        
        # Check foreign keys
        entry_fk = Trade.entry_order_id.property.columns[0]
        exit_fk = Trade.exit_order_id.property.columns[0]
        
        assert "orders.id" in str(entry_fk.foreign_keys.pop().target_fullname)
        assert "orders.id" in str(exit_fk.foreign_keys.pop().target_fullname)


class TestTradingModelDatabaseConstraints:
    """Test database constraints and validations."""

    def test_order_positive_quantity_constraint(self):
        """Test Order quantity must be positive constraint."""
        # This test would require actual database interaction
        # Here we just verify the constraint exists
        constraint_names = {constraint.name for constraint in Order.__table__.constraints}
        assert "check_quantity_positive" in constraint_names

    def test_order_filled_quantity_constraints(self):
        """Test Order filled quantity constraints."""
        constraint_names = {constraint.name for constraint in Order.__table__.constraints}
        assert "check_filled_quantity_non_negative" in constraint_names
        assert "check_filled_quantity_max" in constraint_names

    def test_position_quantity_constraint(self):
        """Test Position quantity must be positive constraint."""
        constraint_names = {constraint.name for constraint in Position.__table__.constraints}
        assert "check_position_quantity_positive" in constraint_names

    def test_order_fill_constraints(self):
        """Test OrderFill constraints."""
        constraint_names = {constraint.name for constraint in OrderFill.__table__.constraints}
        assert "check_fill_quantity_positive" in constraint_names
        assert "check_fill_price_positive" in constraint_names

    def test_order_unique_exchange_order_id(self):
        """Test Order unique constraint on exchange and exchange_order_id."""
        constraint_names = {constraint.name for constraint in Order.__table__.constraints}
        assert "uq_exchange_order_id" in constraint_names


class TestTradingModelIndexes:
    """Test database indexes for performance."""

    def test_order_indexes(self):
        """Test Order model indexes."""
        index_names = {index.name for index in Order.__table__.indexes}
        expected_indexes = {
            "idx_orders_exchange_symbol",
            "idx_orders_status",
            "idx_orders_bot_id",
            "idx_orders_created_at",
            "idx_orders_status_created_at",
            "idx_orders_bot_id_status",
            "idx_orders_symbol_status",
            "idx_orders_exchange_status_created",
        }
        assert expected_indexes.issubset(index_names)

    def test_position_indexes(self):
        """Test Position model indexes."""
        index_names = {index.name for index in Position.__table__.indexes}
        expected_indexes = {
            "idx_positions_exchange_symbol",
            "idx_positions_status",
            "idx_positions_bot_id",
        }
        assert expected_indexes.issubset(index_names)

    def test_order_fill_indexes(self):
        """Test OrderFill model indexes."""
        index_names = {index.name for index in OrderFill.__table__.indexes}
        expected_indexes = {
            "idx_fills_order_id",
            "idx_fills_created_at",
        }
        assert expected_indexes.issubset(index_names)

    def test_trade_indexes(self):
        """Test Trade model indexes."""
        index_names = {index.name for index in Trade.__table__.indexes}
        expected_indexes = {
            "idx_trades_exchange_symbol",
            "idx_trades_bot_id",
            "idx_trades_created_at",
            "idx_trades_pnl",
            "idx_trades_bot_id_timestamp",
            "idx_trades_symbol_timestamp",
            "idx_trades_strategy_performance",
            "idx_trades_exchange_performance",
        }
        assert expected_indexes.issubset(index_names)


class TestTradingModelEdgeCases:
    """Test edge cases and error scenarios."""

    def test_decimal_precision_handling(self):
        """Test decimal precision in trading models."""
        # Test high precision values
        order = Order(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            status="PENDING",
            price=Decimal("50000.12345678"),
            quantity=Decimal("0.12345678"),
        )
        
        assert order.price == Decimal("50000.12345678")
        assert order.quantity == Decimal("0.12345678")

    def test_large_values_handling(self):
        """Test handling of large decimal values."""
        trade = Trade(
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("999999999999.12345678"),
            entry_price=Decimal("999999999999.12345678"),
            exit_price=Decimal("999999999999.12345678"),
            pnl=Decimal("999999999999.12345678"),
        )
        
        assert trade.quantity == Decimal("999999999999.12345678")
        assert trade.pnl == Decimal("999999999999.12345678")

    def test_zero_values_handling(self):
        """Test handling of zero values."""
        position = Position(
            exchange="binance",
            symbol="BTCUSDT",
            side="LONG",
            status="CLOSED",
            quantity=Decimal("0.00000001"),  # Minimum positive
            entry_price=Decimal("0.00000001"),
        )
        
        pnl = position.calculate_pnl(Decimal("0.00000002"))
        assert pnl == Decimal("0.00000001")