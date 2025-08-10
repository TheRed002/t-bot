"""
Unit tests for data cleaning component.

This module tests the comprehensive data cleaning system including:
- Missing data imputation strategies
- Outlier handling (remove vs adjust)
- Data smoothing for noisy signals
- Duplicate detection and removal
- Data normalization and standardization

Test Coverage: 90%+
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

# Import the components to test
from src.data.quality.cleaning import (
    DataCleaner, CleaningStrategy, OutlierMethod, CleaningResult
)
from src.core.types import MarketData, Signal, SignalDirection
from src.core.exceptions import DataError, DataValidationError, ValidationError


class TestDataCleaner:
    """Test cases for DataCleaner class"""

    @pytest.fixture
    def cleaner_config(self) -> Dict[str, Any]:
        """Test configuration for cleaner"""
        return {
            'outlier_threshold': 3.0,
            'missing_threshold': 0.1,
            'smoothing_window': 5,
            'duplicate_threshold': 1.0,
            'max_history_size': 100,
            'outlier_strategy': CleaningStrategy.ADJUST
        }

    @pytest.fixture
    def cleaner(self, cleaner_config: Dict[str, Any]) -> DataCleaner:
        """Create cleaner instance for testing"""
        return DataCleaner(cleaner_config)

    @pytest.fixture
    def valid_market_data(self) -> MarketData:
        """Create valid market data for testing"""
        return MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000.00"),
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("49999.00"),
            ask=Decimal("50001.00"),
            open_price=Decimal("49900.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49800.00")
        )

    @pytest.fixture
    def valid_signals(self) -> List[Signal]:
        """Create valid signals for testing"""
        return [
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.75,
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                strategy_name="test_strategy"
            ),
            Signal(
                direction=SignalDirection.SELL,
                confidence=0.85,
                timestamp=datetime.now(timezone.utc) + timedelta(seconds=1),
                symbol="ETHUSDT",
                strategy_name="test_strategy"
            )
        ]

    @pytest.mark.asyncio
    async def test_cleaner_initialization(self, cleaner: DataCleaner):
        """Test cleaner initialization with configuration"""
        assert cleaner.config is not None
        assert cleaner.outlier_threshold == 3.0
        assert cleaner.missing_threshold == 0.1
        assert cleaner.smoothing_window == 5
        assert cleaner.duplicate_threshold == 1.0
        assert cleaner.max_history_size == 100

    @pytest.mark.asyncio
    async def test_clean_market_data_valid(
            self,
            cleaner: DataCleaner,
            valid_market_data: MarketData):
        """Test cleaning of valid market data"""
        cleaned_data, cleaning_result = await cleaner.clean_market_data(valid_market_data)

        assert cleaned_data is not None
        assert cleaning_result.original_data == valid_market_data
        assert cleaning_result.cleaned_data == cleaned_data
        assert cleaning_result.removed_count == 0
        assert cleaning_result.adjusted_count == 0
        assert cleaning_result.imputed_count == 0
        # Should apply normalization
        assert len(cleaning_result.applied_strategies) > 0

    @pytest.mark.asyncio
    async def test_clean_market_data_missing_price(self, cleaner: DataCleaner):
        """Test cleaning with missing price data"""
        # Since Pydantic prevents None values, test with zero price
        data_with_zero_price = MarketData(
            symbol="BTCUSDT",
            price=Decimal("0"),  # Zero price (invalid)
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc)
        )

        # Add some historical data for imputation
        for i in range(10):
            historical_data = MarketData(
                symbol="BTCUSDT",
                price=Decimal(str(50000 + i * 10)),
                volume=Decimal("100.5"),
                timestamp=datetime.now(timezone.utc)
            )
            await cleaner.clean_market_data(historical_data)

        cleaned_data, cleaning_result = await cleaner.clean_market_data(data_with_zero_price)

        assert cleaned_data is not None
        assert cleaned_data.price is not None  # Should be imputed
        assert cleaning_result.imputed_count > 0
        assert "missing_data_imputation" in cleaning_result.applied_strategies

    @pytest.mark.asyncio
    async def test_clean_market_data_missing_ohlc(self, cleaner: DataCleaner):
        """Test cleaning with missing OHLC data"""
        data_with_missing_ohlc = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000.00"),
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc)
            # Missing OHLC data
        )

        cleaned_data, cleaning_result = await cleaner.clean_market_data(data_with_missing_ohlc)

        assert cleaned_data is not None
        assert cleaned_data.open_price is not None  # Should be imputed
        assert cleaned_data.high_price is not None  # Should be imputed
        assert cleaned_data.low_price is not None   # Should be imputed
        assert cleaning_result.imputed_count > 0

    @pytest.mark.asyncio
    async def test_clean_market_data_outlier_detection(
            self, cleaner: DataCleaner):
        """Test outlier detection and handling"""
        # Add historical data to create distribution
        symbol = "BTCUSDT"
        for i in range(20):
            price = 50000 + i * 10  # Normal price progression
            data = MarketData(
                symbol=symbol,
                price=Decimal(str(price)),
                volume=Decimal("100.5"),
                timestamp=datetime.now(timezone.utc)
            )
            await cleaner.clean_market_data(data)

        # Now add an outlier
        outlier_data = MarketData(
            symbol=symbol,
            price=Decimal("60000.00"),  # Significant outlier
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc)
        )

        cleaned_data, cleaning_result = await cleaner.clean_market_data(outlier_data)

        # Should handle outlier (adjust or remove based on strategy)
        assert "outlier_handling" in cleaning_result.applied_strategies
        assert cleaning_result.adjusted_count > 0 or cleaning_result.removed_count > 0

    @pytest.mark.asyncio
    async def test_clean_market_data_smoothing(self, cleaner: DataCleaner):
        """Test data smoothing functionality"""
        # Add historical data for smoothing
        symbol = "BTCUSDT"
        for i in range(10):
            price = 50000 + i * 100  # Some variation
            data = MarketData(
                symbol=symbol,
                price=Decimal(str(price)),
                volume=Decimal("100.5"),
                timestamp=datetime.now(timezone.utc)
            )
            await cleaner.clean_market_data(data)

        # Add current data point
        current_data = MarketData(
            symbol=symbol,
            price=Decimal("51000.00"),
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc)
        )

        cleaned_data, cleaning_result = await cleaner.clean_market_data(current_data)

        # Should apply smoothing
        assert "data_smoothing" in cleaning_result.applied_strategies

    @pytest.mark.asyncio
    async def test_clean_market_data_duplicate_removal(
            self, cleaner: DataCleaner, valid_market_data: MarketData):
        """Test duplicate detection and removal"""
        # Clean the same data twice
        cleaned_data1, _ = await cleaner.clean_market_data(valid_market_data)
        cleaned_data2, cleaning_result = await cleaner.clean_market_data(valid_market_data)

        # Second cleaning should detect duplicate
        assert "duplicate_removal" in cleaning_result.applied_strategies
        assert cleaning_result.removed_count > 0

    @pytest.mark.asyncio
    async def test_clean_market_data_normalization(self, cleaner: DataCleaner):
        """Test data normalization"""
        data_with_issues = MarketData(
            symbol="btcusdt",  # Lowercase
            price=Decimal("50000.123456789"),  # Too many decimal places
            volume=Decimal("100.123456789"),
            timestamp=datetime.now()  # No timezone
        )

        cleaned_data, cleaning_result = await cleaner.clean_market_data(data_with_issues)

        # Should normalize data
        assert "data_normalization" in cleaning_result.applied_strategies
        assert cleaned_data.symbol == "BTCUSDT"  # Should be uppercase
        assert cleaned_data.timestamp.tzinfo is not None  # Should have timezone

    @pytest.mark.asyncio
    async def test_clean_signal_data_valid(
            self,
            cleaner: DataCleaner,
            valid_signals: List[Signal]):
        """Test cleaning of valid signal data"""
        cleaned_signals, cleaning_result = await cleaner.clean_signal_data(valid_signals)

        assert len(cleaned_signals) == len(valid_signals)
        assert cleaning_result.original_data == valid_signals
        assert cleaning_result.cleaned_data == cleaned_signals
        assert cleaning_result.removed_count == 0
        assert cleaning_result.adjusted_count == 0

    @pytest.mark.asyncio
    async def test_clean_signal_data_invalid_signals(
            self, cleaner: DataCleaner):
        """Test cleaning with invalid signals"""
        # Since Pydantic prevents invalid data, test with edge cases
        edge_signals = [
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.0,  # Minimum confidence (edge case)
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                strategy_name="test_strategy"
            ),
            Signal(
                direction=SignalDirection.SELL,
                confidence=1.0,  # Maximum confidence (edge case)
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                strategy_name="test_strategy"
            )
        ]

        cleaned_signals, cleaning_result = await cleaner.clean_signal_data(edge_signals)

        # Should clean edge cases appropriately
        assert len(cleaned_signals) <= len(edge_signals)
        assert cleaning_result.original_data == edge_signals
        assert cleaning_result.cleaned_data == cleaned_signals

    @pytest.mark.asyncio
    async def test_clean_signal_data_duplicate_signals(
            self, cleaner: DataCleaner):
        """Test cleaning with duplicate signals"""
        duplicate_signals = [
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.75,
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                strategy_name="test_strategy"
            ),
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.75,
                # Very close timestamp
                timestamp=datetime.now(timezone.utc) + timedelta(seconds=0.5),
                symbol="BTCUSDT",
                strategy_name="test_strategy"
            )
        ]

        cleaned_signals, cleaning_result = await cleaner.clean_signal_data(duplicate_signals)

        # Should remove duplicates
        assert len(cleaned_signals) < len(duplicate_signals)
        assert cleaning_result.removed_count > 0

    @pytest.mark.asyncio
    async def test_clean_signal_data_confidence_adjustment(
            self, cleaner: DataCleaner):
        """Test signal confidence adjustment"""
        # Test with valid signals that might need confidence adjustment
        signals_with_edge_confidence = [
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.999,  # Very high confidence
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                strategy_name="test_strategy"
            ),
            Signal(
                direction=SignalDirection.SELL,
                confidence=0.001,  # Very low confidence
                timestamp=datetime.now(timezone.utc),
                symbol="ETHUSDT",
                strategy_name="test_strategy"
            )
        ]

        cleaned_signals, cleaning_result = await cleaner.clean_signal_data(signals_with_edge_confidence)

        # Should adjust confidence values appropriately
        for signal in cleaned_signals:
            assert 0.0 <= signal.confidence <= 1.0

        assert cleaning_result.original_data == signals_with_edge_confidence
        assert cleaning_result.cleaned_data == cleaned_signals

    @pytest.mark.asyncio
    async def test_handle_missing_data(self, cleaner: DataCleaner):
        """Test missing data handling"""
        # Test with data that has zero values (which can be imputed)
        data_with_zeros = MarketData(
            symbol="BTCUSDT",
            price=Decimal("0"),  # Zero price
            volume=Decimal("0"),  # Zero volume
            timestamp=datetime.now(timezone.utc)
        )

        # Add historical data for imputation
        for i in range(10):
            historical_data = MarketData(
                symbol="BTCUSDT",
                price=Decimal(str(50000 + i * 10)),
                volume=Decimal(str(100 + i)),
                timestamp=datetime.now(timezone.utc)
            )
            await cleaner.clean_market_data(historical_data)

        cleaned_data, imputed_count = await cleaner._handle_missing_data(data_with_zeros)

        assert cleaned_data is not None
        assert imputed_count > 0  # Should impute some data

    @pytest.mark.asyncio
    async def test_handle_outliers(self, cleaner: DataCleaner):
        """Test outlier handling"""
        # Add historical data
        symbol = "BTCUSDT"
        for i in range(20):
            price = 50000 + i * 10
            data = MarketData(
                symbol=symbol,
                price=Decimal(str(price)),
                volume=Decimal("100.5"),
                timestamp=datetime.now(timezone.utc)
            )
            await cleaner.clean_market_data(data)

        # Add outlier
        outlier_data = MarketData(
            symbol=symbol,
            price=Decimal("60000.00"),
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc)
        )

        cleaned_data, removed_count, adjusted_count = await cleaner._handle_outliers(outlier_data)

        # Should handle outlier
        assert removed_count > 0 or adjusted_count > 0

    @pytest.mark.asyncio
    async def test_smooth_data(self, cleaner: DataCleaner):
        """Test data smoothing"""
        # Add historical data
        symbol = "BTCUSDT"
        for i in range(10):
            price = 50000 + i * 100
            data = MarketData(
                symbol=symbol,
                price=Decimal(str(price)),
                volume=Decimal("100.5"),
                timestamp=datetime.now(timezone.utc)
            )
            await cleaner.clean_market_data(data)

        # Add current data
        current_data = MarketData(
            symbol=symbol,
            price=Decimal("51000.00"),
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc)
        )

        smoothed_data = await cleaner._smooth_data(current_data)

        assert smoothed_data is not None
        assert smoothed_data.price is not None

    @pytest.mark.asyncio
    async def test_remove_duplicates(
            self,
            cleaner: DataCleaner,
            valid_market_data: MarketData):
        """Test duplicate removal"""
        # First data point
        cleaned_data1, removed_count1 = await cleaner._remove_duplicates(valid_market_data)
        assert removed_count1 == 0
        assert cleaned_data1 is not None

        # Same data point (duplicate)
        cleaned_data2, removed_count2 = await cleaner._remove_duplicates(valid_market_data)
        assert removed_count2 > 0
        assert cleaned_data2 is None

    @pytest.mark.asyncio
    async def test_normalize_data(self, cleaner: DataCleaner):
        """Test data normalization"""
        data_to_normalize = MarketData(
            symbol="btcusdt",
            price=Decimal("50000.123456789"),
            volume=Decimal("100.123456789"),
            timestamp=datetime.now()  # No timezone
        )

        normalized_data = await cleaner._normalize_data(data_to_normalize)

        assert normalized_data.symbol == "BTCUSDT"
        assert normalized_data.timestamp.tzinfo is not None
        assert str(normalized_data.price).count('.') == 1
        assert len(str(normalized_data.price).split('.')[1]) <= 8

    @pytest.mark.asyncio
    async def test_is_valid_signal(self, cleaner: DataCleaner):
        """Test signal validation"""
        # Valid signal
        valid_signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.75,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="test_strategy"
        )
        assert await cleaner._is_valid_signal(valid_signal) is True

        # Test with edge case signal
        edge_signal = Signal(
            direction=SignalDirection.SELL,
            confidence=0.0,  # Minimum confidence
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="test_strategy"
        )
        assert await cleaner._is_valid_signal(edge_signal) is True

    @pytest.mark.asyncio
    async def test_clean_confidence(self, cleaner: DataCleaner):
        """Test confidence cleaning"""
        # Test values outside range
        assert await cleaner._clean_confidence(1.5) == 1.0
        assert await cleaner._clean_confidence(-0.1) == 0.0
        assert await cleaner._clean_confidence(0.75) == 0.75

        # Test rounding
        assert await cleaner._clean_confidence(0.123456789) == 0.123

    @pytest.mark.asyncio
    async def test_is_duplicate_signal(self, cleaner: DataCleaner):
        """Test duplicate signal detection"""
        base_signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.75,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="test_strategy"
        )

        existing_signals = [base_signal]

        # Same signal (duplicate)
        duplicate_signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.75,
            timestamp=base_signal.timestamp + timedelta(seconds=0.5),
            symbol="BTCUSDT",
            strategy_name="test_strategy"
        )
        assert await cleaner._is_duplicate_signal(duplicate_signal, existing_signals) is True

        # Different signal (not duplicate)
        different_signal = Signal(
            direction=SignalDirection.SELL,
            confidence=0.75,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="test_strategy"
        )
        assert await cleaner._is_duplicate_signal(different_signal, existing_signals) is False

    @pytest.mark.asyncio
    async def test_create_data_hash(
            self,
            cleaner: DataCleaner,
            valid_market_data: MarketData):
        """Test data hash creation for duplicate detection"""
        hash1 = cleaner._create_data_hash(valid_market_data)
        hash2 = cleaner._create_data_hash(valid_market_data)

        assert hash1 == hash2  # Same data should produce same hash

        # Different data should produce different hash
        different_data = MarketData(
            symbol="ETHUSDT",
            price=Decimal("3000.00"),
            volume=Decimal("50.0"),
            timestamp=datetime.now(timezone.utc)
        )
        hash3 = cleaner._create_data_hash(different_data)
        assert hash1 != hash3

    @pytest.mark.asyncio
    async def test_cleaning_summary(self, cleaner: DataCleaner):
        """Test cleaning summary generation"""
        summary = await cleaner.get_cleaning_summary()

        assert 'cleaning_stats' in summary
        assert 'price_history_size' in summary
        assert 'volume_history_size' in summary
        assert 'cleaning_config' in summary

        # Check config values
        config = summary['cleaning_config']
        assert config['outlier_threshold'] == 3.0
        assert config['smoothing_window'] == 5
        assert config['duplicate_threshold'] == 1.0

    @pytest.mark.asyncio
    async def test_cleaner_error_handling(self, cleaner: DataCleaner):
        """Test cleaner error handling"""
        # Test with None data
        cleaned_data, cleaning_result = await cleaner.clean_market_data(None)
        assert cleaned_data is None
        assert cleaning_result.original_data is None
        assert 'error' in cleaning_result.metadata

        # Test with data that has validation issues but is still valid Pydantic
        malformed_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("-100.00"),  # Negative price (invalid)
            volume=Decimal("100.5"),
            timestamp=datetime.now(timezone.utc)
        )

        cleaned_data, cleaning_result = await cleaner.clean_market_data(malformed_data)
        assert cleaned_data is not None  # Should still clean what it can
        assert cleaning_result.original_data == malformed_data

    @pytest.mark.asyncio
    async def test_cleaner_performance(
            self,
            cleaner: DataCleaner,
            valid_market_data: MarketData):
        """Test cleaner performance"""
        import time

        start_time = time.perf_counter()

        # Perform multiple cleanings
        for _ in range(100):
            await cleaner.clean_market_data(valid_market_data)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Should complete 100 cleanings in reasonable time (< 2 seconds)
        assert total_time < 2.0

        # Average time per cleaning should be < 20ms
        avg_time = total_time / 100
        assert avg_time < 0.02


class TestCleaningStrategy:
    """Test cases for CleaningStrategy enum"""

    def test_cleaning_strategies(self):
        """Test cleaning strategy enum values"""
        assert CleaningStrategy.REMOVE.value == "remove"
        assert CleaningStrategy.ADJUST.value == "adjust"
        assert CleaningStrategy.IMPUTE.value == "impute"
        assert CleaningStrategy.SMOOTH.value == "smooth"


class TestOutlierMethod:
    """Test cases for OutlierMethod enum"""

    def test_outlier_methods(self):
        """Test outlier method enum values"""
        assert OutlierMethod.Z_SCORE.value == "z_score"
        assert OutlierMethod.IQR.value == "iqr"
        assert OutlierMethod.ISOLATION_FOREST.value == "isolation_forest"
        assert OutlierMethod.LOCAL_OUTLIER_FACTOR.value == "lof"


class TestCleaningResult:
    """Test cases for CleaningResult dataclass"""

    def test_cleaning_result_creation(self):
        """Test cleaning result creation"""
        result = CleaningResult(
            original_data="original",
            cleaned_data="cleaned",
            applied_strategies=["normalization"],
            removed_count=1,
            adjusted_count=2,
            imputed_count=3,
            timestamp=datetime.now(timezone.utc),
            metadata={"test": "value"}
        )

        assert result.original_data == "original"
        assert result.cleaned_data == "cleaned"
        assert result.applied_strategies == ["normalization"]
        assert result.removed_count == 1
        assert result.adjusted_count == 2
        assert result.imputed_count == 3
        assert result.metadata["test"] == "value"
