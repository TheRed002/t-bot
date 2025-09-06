"""Simplified tests for backtesting service module focusing on what actually works."""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.backtesting.service import (
    BacktestRequest,
    BacktestCacheEntry,
)
from src.backtesting.engine import BacktestResult
from src.core.config import Config
from src.core.exceptions import BacktestError, ServiceError


class TestBacktestRequestSimple:
    """Test BacktestRequest validation and creation."""

    def test_valid_request_creation_minimal(self):
        """Test creating valid backtest request with minimal data."""
        request = BacktestRequest(
            strategy_config={"type": "momentum", "period": 14},
            symbols=["BTCUSDT"],
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
        )
        
        assert request.symbols == ["BTCUSDT"]
        assert request.strategy_config == {"type": "momentum", "period": 14}

    def test_invalid_date_range_simple(self):
        """Test validation of invalid date range."""
        with pytest.raises(ValueError, match="End date must be after start date"):
            BacktestRequest(
                strategy_config={"type": "momentum"},
                symbols=["BTCUSDT"],
                start_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc)  # Before start
            )

    @patch('src.utils.validators.ValidationFramework.validate_symbol')
    def test_valid_symbol_validation(self, mock_validate_symbol):
        """Test validation of valid symbols."""
        mock_validate_symbol.return_value = None  # No exception means valid
        
        request = BacktestRequest(
            strategy_config={"type": "momentum"},
            symbols=["BTCUSDT", "ETHUSDT"],
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 31, tzinfo=timezone.utc)
        )
        
        assert request.symbols == ["BTCUSDT", "ETHUSDT"]
        # Should have called validator for each symbol
        assert mock_validate_symbol.call_count == 2


class TestBacktestCacheEntrySimple:
    """Test BacktestCacheEntry functionality."""

    def test_cache_entry_creation_basic(self):
        """Test basic cache entry creation."""
        result = MagicMock(spec=BacktestResult)
        entry = BacktestCacheEntry(
            request_hash="abc123",
            result=result,
        )
        
        assert entry.request_hash == "abc123"
        assert entry.result == result
        assert entry.ttl_hours == 24  # Default

    def test_cache_entry_not_expired_fresh(self):
        """Test cache entry not expired when fresh."""
        result = MagicMock(spec=BacktestResult)
        entry = BacktestCacheEntry(
            request_hash="abc123",
            result=result,
            ttl_hours=1
        )
        
        # Should not be expired immediately
        assert not entry.is_expired()

    def test_cache_entry_expired_old(self):
        """Test cache entry expiry when old."""
        result = MagicMock(spec=BacktestResult)
        
        # Create entry that's already expired
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        entry = BacktestCacheEntry(
            request_hash="abc123",
            result=result,
            created_at=old_time,
            ttl_hours=24
        )
        
        assert entry.is_expired()


class TestBacktestServiceSimple:
    """Simplified BacktestService tests."""

    @patch('src.backtesting.service.BaseComponent.__init__')
    def test_service_creation_minimal(self, mock_base_init):
        """Test basic service creation."""
        mock_base_init.return_value = None
        
        # Import here to avoid initialization issues
        from src.backtesting.service import BacktestService
        
        config = {"test": "value"}
        
        with patch.object(BacktestService, 'logger', create=True):
            service = BacktestService(config=config)
            
            assert service.config == config
            assert service._initialized is False

    @patch('src.backtesting.service.BaseComponent.__init__')  
    def test_service_with_services_injection(self, mock_base_init):
        """Test service creation with dependency injection."""
        mock_base_init.return_value = None
        
        from src.backtesting.service import BacktestService
        
        config = {"test": "value"}
        mock_data_service = MagicMock()
        mock_injector = MagicMock()
        
        with patch.object(BacktestService, 'logger', create=True):
            service = BacktestService(
                config=config,
                DataService=mock_data_service,
                injector=mock_injector
            )
            
            assert service.data_service == mock_data_service
            assert service._injector == mock_injector

    @patch('src.backtesting.service.BaseComponent.__init__')
    def test_service_state_tracking(self, mock_base_init):
        """Test service state tracking variables."""
        mock_base_init.return_value = None
        
        from src.backtesting.service import BacktestService
        
        with patch.object(BacktestService, 'logger', create=True):
            service = BacktestService(config={})
            
            assert isinstance(service._active_backtests, dict)
            assert len(service._active_backtests) == 0
            assert isinstance(service._memory_cache, dict)
            assert len(service._memory_cache) == 0
            assert service._initialized is False
            assert service._cache_available is False

    @patch('src.backtesting.service.BaseComponent.__init__')
    def test_cache_config_setup(self, mock_base_init):
        """Test cache configuration setup."""
        mock_base_init.return_value = None
        
        from src.backtesting.service import BacktestService
        
        config = MagicMock()
        config.backtest_cache = {"redis_host": "custom_host"}
        
        with patch.object(BacktestService, 'logger', create=True):
            service = BacktestService(config=config)
            
            assert hasattr(service, '_cache_config')
            assert isinstance(service._cache_config, dict)


class TestServiceIntegrationSimple:
    """Simple integration tests."""

    @patch('src.backtesting.service.BaseComponent.__init__')
    def test_config_handling_variations(self, mock_base_init):
        """Test different config types."""
        mock_base_init.return_value = None
        
        from src.backtesting.service import BacktestService
        
        # Test dict config
        with patch.object(BacktestService, 'logger', create=True):
            service1 = BacktestService(config={"test": "dict"})
            assert service1.config == {"test": "dict"}
        
        # Test None config  
        with patch.object(BacktestService, 'logger', create=True):
            service2 = BacktestService(config=None)
            assert service2.config is None

    def test_request_validation_edge_cases(self):
        """Test request validation edge cases."""
        # Empty symbols should fail
        with pytest.raises(ValueError):
            BacktestRequest(
                strategy_config={},
                symbols=[],
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 31, tzinfo=timezone.utc)
            )

    def test_cache_entry_edge_cases(self):
        """Test cache entry edge cases."""
        result = MagicMock(spec=BacktestResult)
        
        # Test with custom metadata
        entry = BacktestCacheEntry(
            request_hash="test123",
            result=result,
            metadata={"custom": "data"},
            ttl_hours=48
        )
        
        assert entry.metadata == {"custom": "data"}
        assert entry.ttl_hours == 48