"""Tests for database trading service."""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

from src.core.exceptions import ServiceError, ValidationError
from src.database.services.trading_service import TradingService


class TestTradingService:
    """Test the TradingService class."""

    @pytest.fixture
    def mock_order_repo(self):
        """Create a mock order repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_position_repo(self):
        """Create a mock position repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_trade_repo(self):
        """Create a mock trade repository."""
        return AsyncMock()

    @pytest.fixture
    def trading_service(self, mock_order_repo, mock_position_repo, mock_trade_repo):
        """Create a TradingService instance with mocked dependencies."""
        return TradingService(
            order_repo=mock_order_repo,
            position_repo=mock_position_repo,
            trade_repo=mock_trade_repo
        )

    @pytest.fixture
    def mock_order(self):
        """Create a mock order object."""
        order = Mock()
        order.id = "order-123"
        order.status = "PENDING"
        return order

    @pytest.fixture
    def mock_position(self):
        """Create a mock position object."""
        position = Mock()
        position.id = "position-123"
        position.symbol = "BTC-USD"
        position.status = "OPEN"
        position.entry_price = Decimal("100.00")
        position.quantity = Decimal("10.0")
        position.side = "LONG"
        position.current_price = Decimal("105.00")
        position.unrealized_pnl = Decimal("50.00")
        return position

    @pytest.fixture
    def mock_trade(self):
        """Create a mock trade object."""
        trade = Mock()
        trade.id = "trade-123"
        trade.symbol = "BTC-USD"
        trade.side = "BUY"
        trade.quantity = Decimal("10.0")
        trade.entry_price = Decimal("100.00")
        trade.pnl = Decimal("50.00")
        trade.created_at = datetime.now()
        return trade

    async def test_init(self, mock_order_repo, mock_position_repo, mock_trade_repo):
        """Test TradingService initialization."""
        service = TradingService(
            order_repo=mock_order_repo,
            position_repo=mock_position_repo,
            trade_repo=mock_trade_repo
        )
        assert service.order_repo == mock_order_repo
        assert service.position_repo == mock_position_repo
        assert service.trade_repo == mock_trade_repo
        assert service.name == "TradingService"

    async def test_cancel_order_success(self, trading_service, mock_order_repo, mock_order):
        """Test cancelling order successfully."""
        # Arrange
        mock_order_repo.get_by_id.return_value = mock_order
        mock_order_repo.update.return_value = mock_order

        # Act
        result = await trading_service.cancel_order("order-123", "User requested")

        # Assert
        assert result is True
        assert mock_order.status == "CANCELLED"
        mock_order_repo.get_by_id.assert_called_once_with("order-123")
        mock_order_repo.update.assert_called_once_with(mock_order)

    async def test_cancel_order_not_found(self, trading_service, mock_order_repo):
        """Test cancelling order that doesn't exist."""
        # Arrange
        mock_order_repo.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ServiceError, match="Order cancellation failed"):
            await trading_service.cancel_order("order-123")

    async def test_cancel_order_cannot_cancel(self, trading_service, mock_order_repo, mock_order):
        """Test cancelling order that cannot be cancelled."""
        # Arrange
        mock_order.status = "FILLED"
        mock_order_repo.get_by_id.return_value = mock_order

        # Act & Assert
        with pytest.raises(ServiceError, match="Order cancellation failed"):
            await trading_service.cancel_order("order-123")

    async def test_cancel_order_repository_error(self, trading_service, mock_order_repo, mock_order):
        """Test cancelling order with repository error."""
        # Arrange
        mock_order_repo.get_by_id.return_value = mock_order
        mock_order_repo.update.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Order cancellation failed"):
            await trading_service.cancel_order("order-123")

    async def test_close_position_success(self, trading_service, mock_position_repo, mock_position):
        """Test closing position successfully."""
        # Arrange
        close_price = Decimal("110.00")
        mock_position_repo.get_by_id.return_value = mock_position
        mock_position_repo.update.return_value = mock_position

        # Act
        result = await trading_service.close_position("position-123", close_price)

        # Assert
        assert result is True
        assert mock_position.status == "CLOSED"
        assert mock_position.exit_price == close_price
        assert mock_position.realized_pnl == Decimal("100.00")  # (110 - 100) * 10
        mock_position_repo.get_by_id.assert_called_once_with("position-123")
        mock_position_repo.update.assert_called_once_with(mock_position)

    async def test_close_position_not_found(self, trading_service, mock_position_repo):
        """Test closing position that doesn't exist."""
        # Arrange
        mock_position_repo.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ServiceError, match="Position close failed"):
            await trading_service.close_position("position-123", Decimal("110.00"))

    async def test_close_position_not_open(self, trading_service, mock_position_repo, mock_position):
        """Test closing position that is not open."""
        # Arrange
        mock_position.status = "CLOSED"
        mock_position_repo.get_by_id.return_value = mock_position

        # Act & Assert
        with pytest.raises(ServiceError, match="Position close failed"):
            await trading_service.close_position("position-123", Decimal("110.00"))

    async def test_get_trades_by_bot_success(self, trading_service, mock_trade_repo, mock_trade):
        """Test getting trades by bot successfully."""
        # Arrange
        mock_trade_repo.list.return_value = [mock_trade]

        # Act
        result = await trading_service.get_trades_by_bot("bot-123", limit=10, offset=5)

        # Assert
        assert len(result) == 1
        assert result[0] == mock_trade
        mock_trade_repo.list.assert_called_once_with(
            filters={"bot_id": "bot-123"},
            limit=10,
            offset=5,
            order_by="created_at",
            order_desc=True
        )

    async def test_get_trades_by_bot_with_time_filters(self, trading_service, mock_trade_repo, mock_trade):
        """Test getting trades by bot with time filters."""
        # Arrange
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 1, 31)
        mock_trade_repo.list.return_value = [mock_trade]

        # Act
        result = await trading_service.get_trades_by_bot(
            "bot-123", start_time=start_time, end_time=end_time
        )

        # Assert
        assert len(result) == 1
        expected_filters = {
            "bot_id": "bot-123",
            "created_at": {"gte": start_time, "lte": end_time}
        }
        mock_trade_repo.list.assert_called_once_with(
            filters=expected_filters,
            limit=None,
            offset=0,
            order_by="created_at",
            order_desc=True
        )

    async def test_get_positions_by_bot_success(self, trading_service, mock_position_repo, mock_position):
        """Test getting positions by bot successfully."""
        # Arrange
        mock_position_repo.list.return_value = [mock_position]

        # Act
        result = await trading_service.get_positions_by_bot("bot-123")

        # Assert
        assert len(result) == 1
        assert result[0] == mock_position
        mock_position_repo.list.assert_called_once_with(
            filters={"bot_id": "bot-123"},
            order_by="created_at",
            order_desc=True
        )

    async def test_calculate_total_pnl_success(self, trading_service, mock_trade_repo, mock_trade):
        """Test calculating total P&L successfully."""
        # Arrange
        trade1 = Mock()
        trade1.pnl = Decimal("100.00")
        trade2 = Mock()
        trade2.pnl = Decimal("50.00")
        trade3 = Mock()
        trade3.pnl = None  # Should be ignored
        mock_trade_repo.list.return_value = [trade1, trade2, trade3]

        # Act
        result = await trading_service.calculate_total_pnl("bot-123")

        # Assert
        assert result == Decimal("150.00")

    async def test_calculate_total_pnl_no_trades(self, trading_service, mock_trade_repo):
        """Test calculating total P&L with no trades."""
        # Arrange
        mock_trade_repo.list.return_value = []

        # Act
        result = await trading_service.calculate_total_pnl("bot-123")

        # Assert
        assert result == Decimal("0")

    def test_can_cancel_order_pending(self, trading_service, mock_order):
        """Test checking if pending order can be cancelled."""
        # Arrange
        mock_order.status = "PENDING"

        # Act
        result = trading_service._can_cancel_order(mock_order)

        # Assert
        assert result is True

    def test_can_cancel_order_open(self, trading_service, mock_order):
        """Test checking if open order can be cancelled."""
        # Arrange
        mock_order.status = "OPEN"

        # Act
        result = trading_service._can_cancel_order(mock_order)

        # Assert
        assert result is True

    def test_can_cancel_order_partially_filled(self, trading_service, mock_order):
        """Test checking if partially filled order can be cancelled."""
        # Arrange
        mock_order.status = "PARTIALLY_FILLED"

        # Act
        result = trading_service._can_cancel_order(mock_order)

        # Assert
        assert result is True

    def test_can_cancel_order_filled(self, trading_service, mock_order):
        """Test checking if filled order can be cancelled."""
        # Arrange
        mock_order.status = "FILLED"

        # Act
        result = trading_service._can_cancel_order(mock_order)

        # Assert
        assert result is False

    def test_calculate_realized_pnl_long_profit(self, trading_service, mock_position):
        """Test calculating realized P&L for profitable long position."""
        # Arrange
        mock_position.entry_price = Decimal("100.00")
        mock_position.quantity = Decimal("10.0")
        mock_position.side = "LONG"
        close_price = Decimal("110.00")

        # Act
        result = trading_service._calculate_realized_pnl(mock_position, close_price)

        # Assert
        assert result == Decimal("100.00")  # (110 - 100) * 10

    def test_calculate_realized_pnl_long_loss(self, trading_service, mock_position):
        """Test calculating realized P&L for losing long position."""
        # Arrange
        mock_position.entry_price = Decimal("100.00")
        mock_position.quantity = Decimal("10.0")
        mock_position.side = "LONG"
        close_price = Decimal("90.00")

        # Act
        result = trading_service._calculate_realized_pnl(mock_position, close_price)

        # Assert
        assert result == Decimal("-100.00")  # (90 - 100) * 10

    def test_calculate_realized_pnl_short_profit(self, trading_service, mock_position):
        """Test calculating realized P&L for profitable short position."""
        # Arrange
        mock_position.entry_price = Decimal("100.00")
        mock_position.quantity = Decimal("10.0")
        mock_position.side = "SHORT"
        close_price = Decimal("90.00")

        # Act
        result = trading_service._calculate_realized_pnl(mock_position, close_price)

        # Assert
        assert result == Decimal("100.00")  # -(90 - 100) * 10

    def test_calculate_realized_pnl_short_loss(self, trading_service, mock_position):
        """Test calculating realized P&L for losing short position."""
        # Arrange
        mock_position.entry_price = Decimal("100.00")
        mock_position.quantity = Decimal("10.0")
        mock_position.side = "SHORT"
        close_price = Decimal("110.00")

        # Act
        result = trading_service._calculate_realized_pnl(mock_position, close_price)

        # Assert
        assert result == Decimal("-100.00")  # -(110 - 100) * 10

    def test_calculate_realized_pnl_no_entry_price(self, trading_service, mock_position):
        """Test calculating realized P&L with no entry price."""
        # Arrange
        mock_position.entry_price = None
        mock_position.quantity = Decimal("10.0")

        # Act
        result = trading_service._calculate_realized_pnl(mock_position, Decimal("110.00"))

        # Assert
        assert result == Decimal("0")

    def test_calculate_realized_pnl_no_quantity(self, trading_service, mock_position):
        """Test calculating realized P&L with no quantity."""
        # Arrange
        mock_position.entry_price = Decimal("100.00")
        mock_position.quantity = None

        # Act
        result = trading_service._calculate_realized_pnl(mock_position, Decimal("110.00"))

        # Assert
        assert result == Decimal("0")

    def test_calculate_unrealized_pnl_long(self, trading_service, mock_position):
        """Test calculating unrealized P&L for long position."""
        # Arrange
        mock_position.entry_price = Decimal("100.00")
        mock_position.quantity = Decimal("10.0")
        mock_position.side = "LONG"
        current_price = Decimal("105.00")

        # Act
        result = trading_service._calculate_unrealized_pnl(mock_position, current_price)

        # Assert
        assert result == Decimal("50.00")  # (105 - 100) * 10

    def test_calculate_unrealized_pnl_short(self, trading_service, mock_position):
        """Test calculating unrealized P&L for short position."""
        # Arrange
        mock_position.entry_price = Decimal("100.00")
        mock_position.quantity = Decimal("10.0")
        mock_position.side = "SHORT"
        current_price = Decimal("95.00")

        # Act
        result = trading_service._calculate_unrealized_pnl(mock_position, current_price)

        # Assert
        assert result == Decimal("50.00")  # -(95 - 100) * 10

    async def test_update_position_price_success(self, trading_service, mock_position_repo, mock_position):
        """Test updating position price successfully."""
        # Arrange
        current_price = Decimal("105.00")
        mock_position_repo.get_by_id.return_value = mock_position
        mock_position_repo.update.return_value = mock_position

        # Act
        result = await trading_service.update_position_price("position-123", current_price)

        # Assert
        assert result is True
        assert mock_position.current_price == current_price
        assert mock_position.unrealized_pnl == Decimal("50.00")  # (105 - 100) * 10

    async def test_update_position_price_not_found(self, trading_service, mock_position_repo):
        """Test updating price for position that doesn't exist."""
        # Arrange
        mock_position_repo.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ServiceError, match="Position price update failed"):
            await trading_service.update_position_price("position-123", Decimal("105.00"))

    async def test_update_position_price_closed_position(self, trading_service, mock_position_repo, mock_position):
        """Test updating price for closed position."""
        # Arrange
        mock_position.status = "CLOSED"
        mock_position_repo.get_by_id.return_value = mock_position

        # Act
        result = await trading_service.update_position_price("position-123", Decimal("105.00"))

        # Assert
        assert result is False
        mock_position_repo.update.assert_not_called()

    async def test_create_trade_success(self, trading_service, mock_trade_repo, mock_trade):
        """Test creating trade successfully."""
        # Arrange
        trade_data = {
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": Decimal("10.0"),
            "entry_price": Decimal("100.00"),
            "bot_id": "bot-123"
        }
        mock_trade_repo.create.return_value = mock_trade

        # Act
        result = await trading_service.create_trade(trade_data)

        # Assert
        assert result["id"] == "trade-123"
        assert result["symbol"] == "BTC-USD"
        assert result["side"] == "BUY"
        mock_trade_repo.create.assert_called_once()

    async def test_create_trade_missing_symbol(self, trading_service):
        """Test creating trade without symbol."""
        # Arrange
        trade_data = {"side": "BUY", "quantity": Decimal("10.0")}

        # Act & Assert
        with pytest.raises(ValidationError, match="Symbol is required"):
            await trading_service.create_trade(trade_data)

    async def test_create_trade_invalid_side(self, trading_service):
        """Test creating trade with invalid side."""
        # Arrange
        trade_data = {
            "symbol": "BTC-USD",
            "side": "INVALID",
            "quantity": Decimal("10.0")
        }

        # Act & Assert
        with pytest.raises(ValidationError, match="Valid side is required"):
            await trading_service.create_trade(trade_data)

    async def test_create_trade_invalid_quantity(self, trading_service):
        """Test creating trade with invalid quantity."""
        # Arrange
        trade_data = {
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": Decimal("0")
        }

        # Act & Assert
        with pytest.raises(ValidationError, match="Positive quantity is required"):
            await trading_service.create_trade(trade_data)

    async def test_get_positions_success(self, trading_service, mock_position_repo, mock_position):
        """Test getting positions successfully."""
        # Arrange
        mock_position_repo.list.return_value = [mock_position]

        # Act
        result = await trading_service.get_positions(strategy_id="strategy-123", symbol="BTC-USD")

        # Assert
        assert len(result) == 1
        assert result[0]["id"] == "position-123"
        assert result[0]["symbol"] == "BTC-USD"
        mock_position_repo.list.assert_called_once_with(
            filters={"strategy_id": "strategy-123", "symbol": "BTC-USD"},
            order_by="created_at",
            order_desc=True
        )

    async def test_get_trade_statistics_success(self, trading_service, mock_trade_repo):
        """Test getting trade statistics successfully."""
        # Arrange
        trade1 = Mock()
        trade1.pnl = Decimal("100.00")
        trade2 = Mock()
        trade2.pnl = Decimal("-50.00")
        trade3 = Mock()
        trade3.pnl = Decimal("75.00")
        mock_trade_repo.list.return_value = [trade1, trade2, trade3]

        # Act
        result = await trading_service.get_trade_statistics("bot-123")

        # Assert
        assert result["total_trades"] == 3
        assert result["profitable_trades"] == 2
        assert result["losing_trades"] == 1
        assert result["total_pnl"] == Decimal("125.00")
        expected_avg = Decimal("125.00") / Decimal("3")
        assert abs(result["average_pnl"] - expected_avg) < Decimal("0.01")
        expected_win_rate = (Decimal("2") / Decimal("3")) * Decimal("100")
        assert abs(result["win_rate"] - expected_win_rate) < Decimal("0.01")
        assert result["largest_win"] == Decimal("100.00")
        assert result["largest_loss"] == Decimal("-50.00")

    async def test_get_trade_statistics_no_trades(self, trading_service, mock_trade_repo):
        """Test getting trade statistics with no trades."""
        # Arrange
        mock_trade_repo.list.return_value = []

        # Act
        result = await trading_service.get_trade_statistics("bot-123")

        # Assert
        assert result["total_trades"] == 0
        assert result["profitable_trades"] == 0
        assert result["losing_trades"] == 0
        assert result["total_pnl"] == Decimal("0")
        assert result["average_pnl"] == Decimal("0")
        assert result["win_rate"] == Decimal("0")

    async def test_get_total_exposure_success(self, trading_service, mock_position_repo):
        """Test getting total exposure successfully."""
        # Arrange
        long_position = Mock()
        long_position.side = "LONG"
        long_position.quantity = Decimal("10.0")
        long_position.entry_price = Decimal("100.00")
        # Mock that position doesn't have a 'value' attribute
        long_position.value = None
        del long_position.value
        
        short_position = Mock()
        short_position.side = "SHORT"
        short_position.quantity = Decimal("5.0")
        short_position.entry_price = Decimal("200.00")
        # Mock that position doesn't have a 'value' attribute
        short_position.value = None
        del short_position.value
        
        mock_position_repo.list.return_value = [long_position, short_position]

        # Act
        result = await trading_service.get_total_exposure("bot-123")

        # Assert
        assert result["long"] == Decimal("1000.00")  # 10 * 100
        assert result["short"] == Decimal("1000.00")  # 5 * 200
        assert result["net"] == Decimal("0.00")  # 1000 - 1000
        assert result["gross"] == Decimal("2000.00")  # 1000 + 1000


class TestTradingServiceErrorHandling:
    """Test error handling in TradingService."""

    @pytest.fixture
    def trading_service(self):
        """Create a TradingService instance with mock dependencies."""
        return TradingService(
            order_repo=AsyncMock(),
            position_repo=AsyncMock(),
            trade_repo=AsyncMock()
        )

    async def test_get_trades_by_bot_repository_error(self, trading_service):
        """Test getting trades with repository error."""
        # Arrange
        trading_service.trade_repo.list.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Failed to get trades"):
            await trading_service.get_trades_by_bot("bot-123")

    async def test_get_positions_by_bot_repository_error(self, trading_service):
        """Test getting positions with repository error."""
        # Arrange
        trading_service.position_repo.list.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Failed to get positions"):
            await trading_service.get_positions_by_bot("bot-123")

    async def test_calculate_total_pnl_repository_error(self, trading_service):
        """Test calculating P&L with repository error."""
        # Arrange
        trading_service.trade_repo.list.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="P&L calculation failed"):
            await trading_service.calculate_total_pnl("bot-123")

    async def test_update_position_price_repository_error(self, trading_service):
        """Test updating position price with repository error."""
        # Arrange
        trading_service.position_repo.get_by_id.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Position price update failed"):
            await trading_service.update_position_price("position-123", Decimal("105.00"))

    async def test_create_trade_repository_error(self, trading_service):
        """Test creating trade with repository error."""
        # Arrange
        trade_data = {
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": Decimal("10.0")
        }
        trading_service.trade_repo.create.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Trade creation failed"):
            await trading_service.create_trade(trade_data)

    async def test_get_positions_repository_error(self, trading_service):
        """Test getting positions with repository error."""
        # Arrange
        trading_service.position_repo.list.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Get positions failed"):
            await trading_service.get_positions()

    async def test_get_trade_statistics_repository_error(self, trading_service):
        """Test getting trade statistics with repository error."""
        # Arrange
        trading_service.trade_repo.list.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Trade statistics calculation failed"):
            await trading_service.get_trade_statistics("bot-123")

    async def test_get_total_exposure_repository_error(self, trading_service):
        """Test getting total exposure with repository error."""
        # Arrange
        trading_service.position_repo.list.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Exposure calculation failed"):
            await trading_service.get_total_exposure("bot-123")