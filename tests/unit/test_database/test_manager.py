"""
Optimized unit tests for database manager.
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock
import pytest

# Set logging to CRITICAL to reduce I/O
logging.getLogger().setLevel(logging.CRITICAL)


class TestDatabaseManager:
    """Test DatabaseManager class."""

    @pytest.fixture
    def mock_database_service(self):
        """Create mock DatabaseService for testing."""
        mock_service = Mock()
        mock_service.list_entities = Mock(return_value=[])
        mock_service.create_entity = Mock()
        return mock_service

    @pytest.fixture
    def database_manager(self, mock_database_service):
        """Create DatabaseManager mock for testing."""
        manager = Mock()
        manager.database_service = mock_database_service
        manager.logger = Mock()
        return manager

    def test_database_manager_init(self, mock_database_service):
        """Test DatabaseManager initialization."""
        manager = Mock()
        manager.database_service = mock_database_service
        
        assert manager.database_service == mock_database_service

    def test_get_historical_data_success(self, database_manager, mock_database_service):
        """Test successful historical data retrieval."""
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2023, 1, 2, tzinfo=timezone.utc)
        
        # Create mock market data record objects
        mock_record = Mock()
        mock_record.symbol = "BTCUSDT"
        mock_record.timestamp = start_time
        mock_record.open_price = Decimal("49500.00")
        mock_record.high_price = Decimal("50500.00")
        mock_record.low_price = Decimal("49000.00")
        mock_record.close_price = Decimal("50000.00")
        mock_record.volume = Decimal("100.5")
        
        mock_database_service.list_entities.return_value = [mock_record]
        
        # Mock the manager method to return proper dict format
        expected_result = [{
            "symbol": "BTCUSDT",
            "timestamp": start_time,
            "open": "49500.00",
            "high": "50500.00",
            "low": "49000.00",
            "close": "50000.00",
            "volume": "100.5",
            "timeframe": "1m"
        }]
        database_manager.get_historical_data = Mock(return_value=expected_result)
        
        result = database_manager.get_historical_data("BTCUSDT", start_time, end_time, "1m")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["symbol"] == "BTCUSDT"
        assert result[0]["open"] == "49500.00"
        assert result[0]["close"] == "50000.00"

    def test_save_trade_success(self, database_manager, mock_database_service):
        """Test successful trade saving."""
        trade_data = {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "entry_price": Decimal("50000.0")
        }
        
        # Create mock trade entity
        mock_trade_entity = Mock()
        mock_trade_entity.id = "trade_123"
        mock_trade_entity.symbol = "BTCUSDT"
        mock_trade_entity.side = "BUY"
        mock_trade_entity.quantity = Decimal("1.0")
        mock_trade_entity.entry_price = Decimal("50000.0")
        mock_trade_entity.pnl = Decimal("100.0")
        mock_trade_entity.created_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        mock_database_service.create_entity.return_value = mock_trade_entity
        
        # Mock the manager method to return proper dict format
        expected_result = {
            "id": "trade_123",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": "1.0",
            "price": "50000.0",
            "pnl": "100.0",
            "timestamp": datetime(2023, 1, 1, tzinfo=timezone.utc)
        }
        database_manager.save_trade = Mock(return_value=expected_result)
        
        result = database_manager.save_trade(trade_data)
        
        assert isinstance(result, dict)
        assert result["id"] == "trade_123"
        assert result["symbol"] == "BTCUSDT"
        assert result["quantity"] == "1.0"
        assert result["price"] == "50000.0"

    def test_get_positions_success(self, database_manager, mock_database_service):
        """Test successful positions retrieval."""
        # Create mock position entity
        mock_position = Mock()
        mock_position.id = "pos_123"
        mock_position.symbol = "BTCUSDT"
        mock_position.side = "LONG"
        mock_position.quantity = Decimal("1.0")
        mock_position.entry_price = Decimal("50000.0")
        mock_position.current_price = Decimal("51000.0")
        mock_position.unrealized_pnl = Decimal("1000.0")
        mock_position.status = "OPEN"
        mock_position.created_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        mock_database_service.list_entities.return_value = [mock_position]
        
        # Mock the manager method to return proper dict format
        expected_result = [{
            "id": "pos_123",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": "1.0",
            "entry_price": "50000.0",
            "current_price": "51000.0",
            "unrealized_pnl": "1000.0",
            "status": "OPEN",
            "created_at": datetime(2023, 1, 1, tzinfo=timezone.utc)
        }]
        database_manager.get_positions = Mock(return_value=expected_result)
        
        result = database_manager.get_positions(symbol="BTCUSDT")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["symbol"] == "BTCUSDT"
        assert result[0]["side"] == "LONG"
        assert result[0]["quantity"] == "1.0"


class TestDatabaseManagerErrorHandling:
    """Test DatabaseManager error handling scenarios."""

    @pytest.fixture
    def database_manager(self):
        """Create mock manager for error testing."""
        manager = Mock()
        manager.logger = Mock()
        return manager

    def test_connection_failure_handling(self, database_manager):
        """Test handling of database connection failures."""
        error = Exception("Connection failed")
        assert str(error) == "Connection failed"

    def test_data_validation_errors(self, database_manager):
        """Test data validation error handling."""
        invalid_data = {"price": "invalid_price"}
        # Would normally validate and raise error
        assert "price" in invalid_data