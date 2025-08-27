"""
Unit tests for trading repository implementations.

This module tests all trading-related repositories including OrderRepository,
PositionRepository, TradeRepository, and OrderFillRepository.
"""

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.trading import Order, OrderFill, Position, Trade
from src.database.repository.trading import (
    OrderRepository,
    OrderFillRepository,
    PositionRepository,
    TradeRepository,
)


class TestOrderRepository:
    """Test OrderRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def order_repository(self, mock_session):
        """Create OrderRepository instance for testing."""
        return OrderRepository(mock_session)

    @pytest.fixture
    def sample_order(self):
        """Create sample order entity."""
        return Order(
            id=str(uuid.uuid4()),
            bot_id=str(uuid.uuid4()),
            symbol="BTCUSD",
            side="BUY",
            type="LIMIT",
            quantity=Decimal("1.5"),
            price=Decimal("45000.00"),
            status="OPEN",
            exchange="binance",
            exchange_order_id="12345",
            filled_quantity=Decimal("0.0")
        )

    def test_order_repository_init(self, mock_session):
        """Test OrderRepository initialization."""
        repo = OrderRepository(mock_session)
        
        assert repo.session == mock_session
        assert repo.model == Order
        assert repo.name == "OrderRepository"

    @pytest.mark.asyncio
    async def test_get_active_orders_no_filters(self, order_repository, mock_session):
        """Test get active orders without filters."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(status="OPEN"), Mock(status="PENDING")]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await order_repository.get_active_orders()
        
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_active_orders_with_bot_filter(self, order_repository, mock_session):
        """Test get active orders with bot filter."""
        bot_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(status="OPEN", bot_id=bot_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await order_repository.get_active_orders(bot_id=bot_id)
        
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_active_orders_with_symbol_filter(self, order_repository, mock_session):
        """Test get active orders with symbol filter."""
        symbol = "ETHUSDT"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(status="OPEN", symbol=symbol)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await order_repository.get_active_orders(symbol=symbol)
        
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_active_orders_with_both_filters(self, order_repository, mock_session):
        """Test get active orders with both bot and symbol filters."""
        bot_id = str(uuid.uuid4())
        symbol = "BTCUSD"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(status="OPEN", bot_id=bot_id, symbol=symbol)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await order_repository.get_active_orders(bot_id=bot_id, symbol=symbol)
        
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_by_exchange_id_found(self, order_repository, sample_order):
        """Test get order by exchange ID when found."""
        with patch.object(order_repository, 'get_by', return_value=sample_order):
            result = await order_repository.get_by_exchange_id(
                sample_order.exchange, sample_order.exchange_order_id
            )
            
            assert result == sample_order

    @pytest.mark.asyncio
    async def test_get_by_exchange_id_not_found(self, order_repository):
        """Test get order by exchange ID when not found."""
        with patch.object(order_repository, 'get_by', return_value=None):
            result = await order_repository.get_by_exchange_id("binance", "nonexistent")
            
            assert result is None

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, order_repository, sample_order):
        """Test successful order cancellation."""
        sample_order.is_active = True
        
        with patch.object(order_repository, 'get', return_value=sample_order), \
             patch.object(order_repository, 'update', return_value=sample_order):
            
            result = await order_repository.cancel_order(sample_order.id)
            
            assert result is True
            assert sample_order.status == "CANCELLED"

    @pytest.mark.asyncio
    async def test_cancel_order_inactive(self, order_repository, sample_order):
        """Test cancel order when order is not active."""
        sample_order.is_active = False
        
        with patch.object(order_repository, 'get', return_value=sample_order):
            result = await order_repository.cancel_order(sample_order.id)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, order_repository):
        """Test cancel order when order doesn't exist."""
        with patch.object(order_repository, 'get', return_value=None):
            result = await order_repository.cancel_order("nonexistent_id")
            
            assert result is False

    @pytest.mark.asyncio
    async def test_get_orders_by_position(self, order_repository, mock_session):
        """Test get orders by position."""
        position_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(position_id=position_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await order_repository.get_orders_by_position(position_id)
        
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_recent_orders_no_bot_filter(self, order_repository, mock_session):
        """Test get recent orders without bot filter."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(created_at=datetime.now(timezone.utc))]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await order_repository.get_recent_orders(hours=12)
        
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_recent_orders_with_bot_filter(self, order_repository, mock_session):
        """Test get recent orders with bot filter."""
        bot_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(created_at=datetime.now(timezone.utc), bot_id=bot_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await order_repository.get_recent_orders(hours=6, bot_id=bot_id)
        
        assert len(result) == 1


class TestPositionRepository:
    """Test PositionRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def position_repository(self, mock_session):
        """Create PositionRepository instance for testing."""
        return PositionRepository(mock_session)

    @pytest.fixture
    def sample_position(self):
        """Create sample position entity."""
        return Position(
            id=str(uuid.uuid4()),
            bot_id=str(uuid.uuid4()),
            symbol="BTCUSD",
            side="LONG",
            quantity=Decimal("2.0"),
            entry_price=Decimal("44000.00"),
            current_price=Decimal("45000.00"),
            status="OPEN",
            exchange="binance",
            unrealized_pnl=Decimal("2000.00"),
            realized_pnl=Decimal("0.0"),
            stop_loss=Decimal("42000.00")
        )

    def test_position_repository_init(self, mock_session):
        """Test PositionRepository initialization."""
        repo = PositionRepository(mock_session)
        
        assert repo.session == mock_session
        assert repo.model == Position
        assert repo.name == "PositionRepository"

    @pytest.mark.asyncio
    async def test_get_open_positions_no_filters(self, position_repository, mock_session):
        """Test get open positions without filters."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(status="OPEN"), Mock(status="OPEN")]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await position_repository.get_open_positions()
        
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_open_positions_with_bot_filter(self, position_repository, mock_session):
        """Test get open positions with bot filter."""
        bot_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(status="OPEN", bot_id=bot_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await position_repository.get_open_positions(bot_id=bot_id)
        
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_open_positions_with_symbol_filter(self, position_repository, mock_session):
        """Test get open positions with symbol filter."""
        symbol = "ETHUSDT"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(status="OPEN", symbol=symbol)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await position_repository.get_open_positions(symbol=symbol)
        
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_position_by_symbol_found(self, position_repository, sample_position):
        """Test get position by symbol when found."""
        with patch.object(position_repository, 'get_by', return_value=sample_position):
            result = await position_repository.get_position_by_symbol(
                sample_position.bot_id, sample_position.symbol, sample_position.side
            )
            
            assert result == sample_position

    @pytest.mark.asyncio
    async def test_get_position_by_symbol_not_found(self, position_repository):
        """Test get position by symbol when not found."""
        with patch.object(position_repository, 'get_by', return_value=None):
            result = await position_repository.get_position_by_symbol(
                "bot_id", "BTCUSD", "LONG"
            )
            
            assert result is None

    @pytest.mark.asyncio
    async def test_close_position_success(self, position_repository, sample_position):
        """Test successful position closing."""
        sample_position.is_open = True
        exit_price = Decimal("46000.00")
        
        # Mock the calculate_pnl method
        with patch.object(sample_position, 'calculate_pnl', return_value=Decimal("4000.00")):
            with patch.object(position_repository, 'get', return_value=sample_position), \
                 patch.object(position_repository, 'update', return_value=sample_position):
                
                result = await position_repository.close_position(sample_position.id, exit_price)
                
                assert result is True
                assert sample_position.status == "CLOSED"
                assert sample_position.exit_price == exit_price
                assert sample_position.realized_pnl == Decimal("4000.00")

    @pytest.mark.asyncio
    async def test_close_position_not_open(self, position_repository, sample_position):
        """Test close position when position is not open."""
        sample_position.is_open = False
        
        with patch.object(position_repository, 'get', return_value=sample_position):
            result = await position_repository.close_position(
                sample_position.id, Decimal("46000.00")
            )
            
            assert result is False

    @pytest.mark.asyncio
    async def test_close_position_not_found(self, position_repository):
        """Test close position when position doesn't exist."""
        with patch.object(position_repository, 'get', return_value=None):
            result = await position_repository.close_position(
                "nonexistent_id", Decimal("46000.00")
            )
            
            assert result is False

    @pytest.mark.asyncio
    async def test_update_position_price_success(self, position_repository, sample_position):
        """Test successful position price update."""
        current_price = Decimal("47000.00")
        
        # Mock the calculate_pnl method
        with patch.object(sample_position, 'calculate_pnl', return_value=Decimal("6000.00")):
            with patch.object(position_repository, 'get', return_value=sample_position), \
                 patch.object(position_repository, 'update', return_value=sample_position):
                
                result = await position_repository.update_position_price(
                    sample_position.id, current_price
                )
                
                assert result is True
                assert sample_position.current_price == current_price
                assert sample_position.unrealized_pnl == Decimal("6000.00")

    @pytest.mark.asyncio
    async def test_update_position_price_not_found(self, position_repository):
        """Test update position price when position doesn't exist."""
        with patch.object(position_repository, 'get', return_value=None):
            result = await position_repository.update_position_price(
                "nonexistent_id", Decimal("47000.00")
            )
            
            assert result is False

    @pytest.mark.asyncio
    async def test_get_total_exposure(self, position_repository):
        """Test get total exposure calculation."""
        bot_id = str(uuid.uuid4())
        
        positions = [
            Mock(side="LONG", value=Decimal("10000.00")),
            Mock(side="LONG", value=Decimal("15000.00")),
            Mock(side="SHORT", value=Decimal("8000.00")),
            Mock(side="SHORT", value=Decimal("12000.00"))
        ]
        
        with patch.object(position_repository, 'get_open_positions', return_value=positions):
            result = await position_repository.get_total_exposure(bot_id)
            
            expected = {
                "long": Decimal("25000.00"),   # 10000 + 15000
                "short": Decimal("20000.00"),  # 8000 + 12000
                "net": Decimal("5000.00"),     # 25000 - 20000
                "gross": Decimal("45000.00")   # 25000 + 20000
            }
            
            assert result == expected

    @pytest.mark.asyncio
    async def test_get_total_exposure_no_positions(self, position_repository):
        """Test get total exposure with no positions."""
        bot_id = str(uuid.uuid4())
        
        with patch.object(position_repository, 'get_open_positions', return_value=[]):
            result = await position_repository.get_total_exposure(bot_id)
            
            expected = {
                "long": 0,
                "short": 0,
                "net": 0,
                "gross": 0
            }
            
            assert result == expected


class TestTradeRepository:
    """Test TradeRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def trade_repository(self, mock_session):
        """Create TradeRepository instance for testing."""
        return TradeRepository(mock_session)

    @pytest.fixture
    def sample_trade(self):
        """Create sample trade entity."""
        return Trade(
            id=str(uuid.uuid4()),
            bot_id=str(uuid.uuid4()),
            symbol="BTCUSD",
            side="LONG",
            quantity=Decimal("1.0"),
            entry_price=Decimal("44000.00"),
            exit_price=Decimal("46000.00"),
            pnl=Decimal("2000.00"),
            pnl_percentage=Decimal("4.55"),
            exchange="binance"
        )

    def test_trade_repository_init(self, mock_session):
        """Test TradeRepository initialization."""
        repo = TradeRepository(mock_session)
        
        assert repo.session == mock_session
        assert repo.model == Trade
        assert repo.name == "TradeRepository"

    @pytest.mark.asyncio
    async def test_get_profitable_trades_no_bot_filter(self, trade_repository, mock_session):
        """Test get profitable trades without bot filter."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(pnl=Decimal("100.00")), Mock(pnl=Decimal("250.00"))]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await trade_repository.get_profitable_trades()
        
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_profitable_trades_with_bot_filter(self, trade_repository, mock_session):
        """Test get profitable trades with bot filter."""
        bot_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(pnl=Decimal("100.00"), bot_id=bot_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await trade_repository.get_profitable_trades(bot_id=bot_id)
        
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_trades_by_symbol_no_bot_filter(self, trade_repository, mock_session):
        """Test get trades by symbol without bot filter."""
        symbol = "ETHUSDT"
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(symbol=symbol)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await trade_repository.get_trades_by_symbol(symbol)
        
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_trades_by_symbol_with_bot_filter(self, trade_repository, mock_session):
        """Test get trades by symbol with bot filter."""
        symbol = "BTCUSD"
        bot_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(symbol=symbol, bot_id=bot_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await trade_repository.get_trades_by_symbol(symbol, bot_id=bot_id)
        
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_trade_statistics_with_trades(self, trade_repository, mock_session):
        """Test get trade statistics with trade data."""
        bot_id = str(uuid.uuid4())
        
        # Mock trades with different PnL values
        trades = [
            Mock(pnl=Decimal("100.00")),   # Profitable
            Mock(pnl=Decimal("200.00")),   # Profitable
            Mock(pnl=Decimal("50.00")),    # Profitable
            Mock(pnl=Decimal("-75.00")),   # Losing
            Mock(pnl=Decimal("-25.00")),   # Losing
        ]
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = trades
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await trade_repository.get_trade_statistics(bot_id)
        
        expected = {
            "total_trades": 5,
            "profitable_trades": 3,
            "losing_trades": 2,
            "total_pnl": Decimal("250.00"),    # 100 + 200 + 50 - 75 - 25
            "average_pnl": Decimal("50.00"),   # 250 / 5
            "win_rate": 60.0,                  # 3/5 * 100
            "largest_win": Decimal("200.00"),
            "largest_loss": Decimal("-75.00")
        }
        
        assert result["total_trades"] == expected["total_trades"]
        assert result["profitable_trades"] == expected["profitable_trades"]
        assert result["losing_trades"] == expected["losing_trades"]
        assert result["win_rate"] == expected["win_rate"]

    @pytest.mark.asyncio
    async def test_get_trade_statistics_no_trades(self, trade_repository, mock_session):
        """Test get trade statistics with no trades."""
        bot_id = str(uuid.uuid4())
        
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await trade_repository.get_trade_statistics(bot_id)
        
        expected = {
            "total_trades": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0,
            "average_pnl": 0,
            "win_rate": 0,
            "largest_win": 0,
            "largest_loss": 0
        }
        
        assert result == expected

    @pytest.mark.asyncio
    async def test_create_from_position_success(self, trade_repository):
        """Test successful trade creation from position."""
        position = Mock(
            id=str(uuid.uuid4()),
            exchange="binance",
            symbol="BTCUSD",
            side="LONG",
            quantity=Decimal("1.0"),
            entry_price=Decimal("44000.00"),
            exit_price=Decimal("46000.00"),
            realized_pnl=Decimal("2000.00"),
            bot_id=str(uuid.uuid4()),
            strategy_id=str(uuid.uuid4())
        )
        
        exit_order = Mock(
            id=str(uuid.uuid4()),
            average_fill_price=Decimal("46000.00")
        )
        
        mock_trade = Trade(
            exchange=position.exchange,
            symbol=position.symbol,
            side=position.side,
            position_id=position.id,
            exit_order_id=exit_order.id,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            pnl=position.realized_pnl,
            bot_id=position.bot_id,
            strategy_id=position.strategy_id
        )
        
        with patch.object(trade_repository, 'create', return_value=mock_trade):
            result = await trade_repository.create_from_position(position, exit_order)
            
            assert result == mock_trade
            assert result.exchange == position.exchange
            assert result.symbol == position.symbol
            assert result.side == position.side
            assert result.quantity == position.quantity
            assert result.pnl == position.realized_pnl

    @pytest.mark.asyncio
    async def test_create_from_position_with_percentage_calculation(self, trade_repository):
        """Test trade creation with PnL percentage calculation."""
        position = Mock(
            id=str(uuid.uuid4()),
            exchange="binance",
            symbol="BTCUSD",
            side="LONG",
            quantity=Decimal("2.0"),
            entry_price=Decimal("50000.00"),
            exit_price=Decimal("52000.00"),
            realized_pnl=Decimal("4000.00"),
            bot_id=str(uuid.uuid4()),
            strategy_id=str(uuid.uuid4())
        )
        
        exit_order = Mock(
            id=str(uuid.uuid4()),
            average_fill_price=Decimal("52000.00")
        )
        
        mock_trade = Trade(
            exchange=position.exchange,
            symbol=position.symbol,
            side=position.side,
            position_id=position.id,
            exit_order_id=exit_order.id,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            pnl=position.realized_pnl,
            bot_id=position.bot_id,
            strategy_id=position.strategy_id,
            pnl_percentage=Decimal("4.00")  # 4000 / (50000 * 2) * 100
        )
        
        with patch.object(trade_repository, 'create', return_value=mock_trade):
            result = await trade_repository.create_from_position(position, exit_order)
            
            assert result == mock_trade
            # Percentage should be calculated: PnL / (entry_price * quantity) * 100
            # 4000 / (50000 * 2) * 100 = 4%


class TestOrderFillRepository:
    """Test OrderFillRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def fill_repository(self, mock_session):
        """Create OrderFillRepository instance for testing."""
        return OrderFillRepository(mock_session)

    @pytest.fixture
    def sample_fill(self):
        """Create sample order fill entity."""
        return OrderFill(
            id=str(uuid.uuid4()),
            order_id=str(uuid.uuid4()),
            price=45000.0,
            quantity=0.5,
            fee=22.5,
            fee_currency="USDT",
            exchange_fill_id="fill123"
        )

    def test_fill_repository_init(self, mock_session):
        """Test OrderFillRepository initialization."""
        repo = OrderFillRepository(mock_session)
        
        assert repo.session == mock_session
        assert repo.model == OrderFill
        assert repo.name == "OrderFillRepository"

    @pytest.mark.asyncio
    async def test_get_fills_by_order(self, fill_repository, mock_session):
        """Test get fills by order."""
        order_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(order_id=order_id), Mock(order_id=order_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await fill_repository.get_fills_by_order(order_id)
        
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_total_filled_with_fills(self, fill_repository):
        """Test get total filled with fill data."""
        order_id = str(uuid.uuid4())
        
        fills = [
            Mock(quantity=0.5, price=45000.0, fee=22.5),
            Mock(quantity=0.3, price=45100.0, fee=13.53),
            Mock(quantity=0.2, price=44950.0, fee=8.99)
        ]
        
        with patch.object(fill_repository, 'get_fills_by_order', return_value=fills):
            result = await fill_repository.get_total_filled(order_id)
            
            # Total quantity: 0.5 + 0.3 + 0.2 = 1.0
            # Total value: (0.5 * 45000) + (0.3 * 45100) + (0.2 * 44950) = 22500 + 13530 + 8990 = 45020
            # Average price: 45020 / 1.0 = 45020
            # Total fees: 22.5 + 13.53 + 8.99 = 45.02
            
            expected = {
                "quantity": 1.0,
                "average_price": 45020.0,
                "total_fees": 45.02
            }
            
            assert result["quantity"] == expected["quantity"]
            assert result["average_price"] == expected["average_price"]
            assert result["total_fees"] == expected["total_fees"]

    @pytest.mark.asyncio
    async def test_get_total_filled_no_fills(self, fill_repository):
        """Test get total filled with no fills."""
        order_id = str(uuid.uuid4())
        
        with patch.object(fill_repository, 'get_fills_by_order', return_value=[]):
            result = await fill_repository.get_total_filled(order_id)
            
            expected = {
                "quantity": 0,
                "average_price": 0,
                "total_fees": 0
            }
            
            assert result == expected

    @pytest.mark.asyncio
    async def test_create_fill_with_all_params(self, fill_repository):
        """Test create fill with all parameters."""
        order_id = str(uuid.uuid4())
        price = 45000.0
        quantity = 0.5
        fee = 22.5
        fee_currency = "USDT"
        exchange_fill_id = "fill123"
        
        mock_fill = OrderFill(
            order_id=order_id,
            price=price,
            quantity=quantity,
            fee=fee,
            fee_currency=fee_currency,
            exchange_fill_id=exchange_fill_id
        )
        
        with patch.object(fill_repository, 'create', return_value=mock_fill):
            result = await fill_repository.create_fill(
                order_id, price, quantity, fee, fee_currency, exchange_fill_id
            )
            
            assert result == mock_fill
            assert result.order_id == order_id
            assert result.price == price
            assert result.quantity == quantity
            assert result.fee == fee
            assert result.fee_currency == fee_currency
            assert result.exchange_fill_id == exchange_fill_id

    @pytest.mark.asyncio
    async def test_create_fill_minimal_params(self, fill_repository):
        """Test create fill with minimal parameters."""
        order_id = str(uuid.uuid4())
        price = 45000.0
        quantity = 0.5
        
        mock_fill = OrderFill(
            order_id=order_id,
            price=price,
            quantity=quantity,
            fee=0,
            fee_currency=None,
            exchange_fill_id=None
        )
        
        with patch.object(fill_repository, 'create', return_value=mock_fill):
            result = await fill_repository.create_fill(order_id, price, quantity)
            
            assert result == mock_fill
            assert result.order_id == order_id
            assert result.price == price
            assert result.quantity == quantity
            assert result.fee == 0
            assert result.fee_currency is None
            assert result.exchange_fill_id is None


class TestTradingRepositoryErrorHandling:
    """Test error handling in trading repositories."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def order_repository(self, mock_session):
        """Create OrderRepository instance for testing."""
        return OrderRepository(mock_session)

    @pytest.mark.asyncio
    async def test_database_error_handling(self, order_repository, mock_session):
        """Test database error handling in repository operations."""
        mock_session.execute.side_effect = SQLAlchemyError("Database connection lost")
        
        with pytest.raises(SQLAlchemyError):
            await order_repository.get_active_orders()

    @pytest.mark.asyncio
    async def test_integrity_error_handling(self, order_repository, mock_session):
        """Test integrity error handling during create operations."""
        order = Order(
            id=str(uuid.uuid4()),
            bot_id=str(uuid.uuid4()),
            symbol="BTCUSD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("45000.00"),
            status="PENDING",
            exchange="binance"
        )
        mock_session.flush.side_effect = IntegrityError("Duplicate key", None, None)
        
        with pytest.raises(IntegrityError):
            await order_repository.create(order)
            
        mock_session.rollback.assert_called_once()


class TestTradingRepositoryConcurrency:
    """Test concurrent operations in trading repositories."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def position_repository(self, mock_session):
        """Create PositionRepository instance for testing."""
        return PositionRepository(mock_session)

    @pytest.mark.asyncio
    async def test_concurrent_position_updates(self, position_repository):
        """Test concurrent position price updates."""
        position_id = str(uuid.uuid4())
        position = Position(
            id=position_id,
            bot_id=str(uuid.uuid4()),
            symbol="BTCUSD",
            side="LONG",
            quantity=Decimal("1.0"),
            entry_price=Decimal("45000.00"),
            current_price=Decimal("45000.00"),
            status="OPEN"
        )
        
        with patch.object(position_repository, 'get', return_value=position), \
             patch.object(position_repository, 'update', return_value=position), \
             patch.object(position, 'calculate_pnl', return_value=Decimal("1000.00")):
            
            # Simulate concurrent price updates
            result1 = await position_repository.update_position_price(
                position_id, Decimal("46000.00")
            )
            result2 = await position_repository.update_position_price(
                position_id, Decimal("47000.00")
            )
            
            assert result1 is True
            assert result2 is True

    @pytest.mark.asyncio
    async def test_concurrent_order_operations(self, mock_session):
        """Test concurrent order cancel operations."""
        order_repository = OrderRepository(mock_session)
        order_id = str(uuid.uuid4())
        order = Order(
            id=order_id,
            bot_id=str(uuid.uuid4()),
            symbol="BTCUSD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("45000.00"),
            status="OPEN",
            exchange="binance",
            is_active=True
        )
        
        with patch.object(order_repository, 'get', return_value=order), \
             patch.object(order_repository, 'update', return_value=order):
            
            # Simulate concurrent cancel attempts
            result1 = await order_repository.cancel_order(order_id)
            # Second cancel should fail as order is no longer active
            order.is_active = False  # Simulate first cancel taking effect
            result2 = await order_repository.cancel_order(order_id)
            
            assert result1 is True
            assert result2 is False