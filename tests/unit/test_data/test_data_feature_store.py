"""
Test suite for feature store.

This module contains comprehensive tests for the FeatureStore
including feature registration, calculation, caching, and batch processing.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from src.core.config import Config
from src.core.types import MarketData
from src.data.features.feature_store import (
    CalculationStatus,
    FeatureCalculationPipeline,
    FeatureMetadata,
    FeatureRequest,
    FeatureStore,
    FeatureType,
    FeatureValue,
)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock(spec=Config)
    config.feature_store = {
        "max_cache_size": 1000,
        "default_ttl": 300,
        "cleanup_interval": 3600,
        "max_concurrent_calculations": 10,
        "calculation_timeout": 30,
        "batch_size": 100,
    }
    return config


@pytest.fixture
def mock_data_service():
    """Mock data service for testing."""
    service = Mock()
    service.get_market_data = AsyncMock()
    return service


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return [
        MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
            open=Decimal(f"{49950 + i * 10}"),
            high=Decimal(f"{50100 + i * 10}"),
            low=Decimal(f"{49900 + i * 10}"),
            close=Decimal(f"{50000 + i * 10}"),
            volume=Decimal("1.5"),
            exchange="binance",
            bid_price=Decimal(f"{49999 + i * 10}"),
            ask_price=Decimal(f"{50001 + i * 10}"),
        )
        for i in range(50)
    ]


@pytest.fixture
def sample_feature_metadata():
    """Sample feature metadata for testing."""
    return FeatureMetadata(
        feature_id="test_feature",
        feature_name="Test Feature",
        feature_type=FeatureType.TECHNICAL_INDICATOR,
        version="1.0.0",
        dependencies=[],
        calculation_cost=1.0,
        cache_ttl=300,
    )


class TestFeatureType:
    """Test FeatureType enum."""

    def test_feature_type_values(self):
        """Test feature type enum values."""
        assert FeatureType.TECHNICAL_INDICATOR.value == "technical_indicator"
        assert FeatureType.STATISTICAL_FEATURE.value == "statistical_feature"
        assert FeatureType.ALTERNATIVE_FEATURE.value == "alternative_feature"
        assert FeatureType.DERIVED_FEATURE.value == "derived_feature"
        assert FeatureType.ML_FEATURE.value == "ml_feature"


class TestCalculationStatus:
    """Test CalculationStatus enum."""

    def test_calculation_status_values(self):
        """Test calculation status enum values."""
        assert CalculationStatus.PENDING.value == "pending"
        assert CalculationStatus.CALCULATING.value == "calculating"
        assert CalculationStatus.COMPLETED.value == "completed"
        assert CalculationStatus.FAILED.value == "failed"
        assert CalculationStatus.CACHED.value == "cached"


class TestFeatureMetadata:
    """Test FeatureMetadata dataclass."""

    def test_metadata_creation_defaults(self):
        """Test metadata creation with defaults."""
        metadata = FeatureMetadata(
            feature_id="test_id",
            feature_name="Test Name",
            feature_type=FeatureType.TECHNICAL_INDICATOR,
        )

        assert metadata.feature_id == "test_id"
        assert metadata.feature_name == "Test Name"
        assert metadata.feature_type == FeatureType.TECHNICAL_INDICATOR
        assert metadata.version == "1.0.0"
        assert metadata.dependencies == []
        assert metadata.calculation_cost == 1.0
        assert metadata.cache_ttl == 300
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.updated_at, datetime)

    def test_metadata_creation_custom_values(self):
        """Test metadata creation with custom values."""
        custom_deps = ["feature1", "feature2"]
        metadata = FeatureMetadata(
            feature_id="custom_id",
            feature_name="Custom Name",
            feature_type=FeatureType.ML_FEATURE,
            version="2.1.0",
            dependencies=custom_deps,
            calculation_cost=2.5,
            cache_ttl=600,
        )

        assert metadata.feature_id == "custom_id"
        assert metadata.feature_name == "Custom Name"
        assert metadata.feature_type == FeatureType.ML_FEATURE
        assert metadata.version == "2.1.0"
        assert metadata.dependencies == custom_deps
        assert metadata.calculation_cost == 2.5
        assert metadata.cache_ttl == 600


class TestFeatureValue:
    """Test FeatureValue dataclass."""

    def test_feature_value_creation(self):
        """Test feature value creation."""
        timestamp = datetime.now(timezone.utc)
        value = FeatureValue(
            feature_id="test_feature",
            symbol="BTCUSDT",
            value=Decimal("0.75"),
            timestamp=timestamp,
            status=CalculationStatus.COMPLETED,
            calculation_time_ms=150.5,
            cache_hit=False,
        )

        assert value.feature_id == "test_feature"
        assert value.symbol == "BTCUSDT"
        assert value.value == Decimal("0.75")
        assert value.timestamp == timestamp
        assert value.status == CalculationStatus.COMPLETED
        assert value.calculation_time_ms == 150.5
        assert value.cache_hit is False
        assert value.metadata == {}

    def test_feature_value_dict_value(self):
        """Test feature value with dictionary value."""
        dict_value = {"macd_line": Decimal("0.5"), "signal_line": Decimal("0.3")}
        value = FeatureValue(
            feature_id="macd",
            symbol="BTCUSDT",
            value=dict_value,
            timestamp=datetime.now(timezone.utc),
            status=CalculationStatus.COMPLETED,
        )

        assert value.value == dict_value
        assert isinstance(value.value["macd_line"], Decimal)

    def test_feature_value_none_value(self):
        """Test feature value with None value."""
        value = FeatureValue(
            feature_id="failed_feature",
            symbol="BTCUSDT",
            value=None,
            timestamp=datetime.now(timezone.utc),
            status=CalculationStatus.FAILED,
        )

        assert value.value is None
        assert value.status == CalculationStatus.FAILED


class TestFeatureRequest:
    """Test FeatureRequest model."""

    def test_feature_request_creation(self):
        """Test feature request creation."""
        request = FeatureRequest(
            symbol="BTCUSDT",
            feature_names=["sma_20", "rsi_14"],
            lookback_period=100,
        )

        assert request.symbol == "BTCUSDT"
        assert request.feature_names == ["sma_20", "rsi_14"]
        assert request.lookback_period == 100
        assert request.parameters == {}
        assert request.use_cache is True
        assert request.force_recalculation is False
        assert request.priority == 5

    def test_feature_request_validation_empty_features(self):
        """Test validation with empty feature names."""
        with pytest.raises(ValidationError):
            FeatureRequest(
                symbol="BTCUSDT",
                feature_names=[],
                lookback_period=100,
            )

    def test_feature_request_validation_symbol_length(self):
        """Test symbol length validation."""
        # Too short
        with pytest.raises(ValueError):
            FeatureRequest(
                symbol="",
                feature_names=["sma_20"],
            )

        # Too long
        with pytest.raises(ValueError):
            FeatureRequest(
                symbol="A" * 25,  # 25 characters
                feature_names=["sma_20"],
            )

    def test_feature_request_validation_lookback_period(self):
        """Test lookback period validation."""
        # Too small
        with pytest.raises(ValueError):
            FeatureRequest(
                symbol="BTCUSDT",
                feature_names=["sma_20"],
                lookback_period=0,
            )

        # Too large
        with pytest.raises(ValueError):
            FeatureRequest(
                symbol="BTCUSDT",
                feature_names=["sma_20"],
                lookback_period=10000,
            )

    def test_feature_request_validation_priority(self):
        """Test priority validation."""
        # Too low
        with pytest.raises(ValueError):
            FeatureRequest(
                symbol="BTCUSDT",
                feature_names=["sma_20"],
                priority=0,
            )

        # Too high
        with pytest.raises(ValueError):
            FeatureRequest(
                symbol="BTCUSDT",
                feature_names=["sma_20"],
                priority=15,
            )

    def test_feature_request_custom_values(self):
        """Test feature request with custom values."""
        custom_params = {"period": 25, "smoothing": 0.1}
        request = FeatureRequest(
            symbol="ETHUSDT",
            feature_names=["custom_ema"],
            lookback_period=200,
            parameters=custom_params,
            use_cache=False,
            force_recalculation=True,
            priority=8,
        )

        assert request.symbol == "ETHUSDT"
        assert request.feature_names == ["custom_ema"]
        assert request.lookback_period == 200
        assert request.parameters == custom_params
        assert request.use_cache is False
        assert request.force_recalculation is True
        assert request.priority == 8


class TestFeatureCalculationPipeline:
    """Test FeatureCalculationPipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        mock_store = Mock()
        mock_store.logger = Mock()

        pipeline = FeatureCalculationPipeline(mock_store)

        assert pipeline.feature_store is mock_store
        assert pipeline.logger is mock_store.logger
        assert isinstance(pipeline.active_calculations, dict)
        assert len(pipeline.active_calculations) == 0

    @pytest.mark.asyncio
    async def test_calculate_batch_empty(self):
        """Test batch calculation with empty requests."""
        mock_store = Mock()
        mock_store.logger = Mock()

        pipeline = FeatureCalculationPipeline(mock_store)

        result = await pipeline.calculate_batch([])

        assert result == {}

    @pytest.mark.asyncio
    async def test_calculate_batch_single_symbol(self):
        """Test batch calculation for single symbol."""
        mock_store = Mock()
        mock_store.logger = Mock()

        pipeline = FeatureCalculationPipeline(mock_store)

        # Mock the symbol batch calculation
        mock_feature_value = FeatureValue(
            feature_id="sma_20",
            symbol="BTCUSDT",
            value=Decimal("50000.00"),
            timestamp=datetime.now(timezone.utc),
            status=CalculationStatus.COMPLETED,
        )

        with patch.object(pipeline, "_calculate_symbol_batch", new_callable=AsyncMock) as mock_calc:
            mock_calc.return_value = [mock_feature_value]

            requests = [
                FeatureRequest(symbol="BTCUSDT", feature_names=["sma_20"]),
            ]

            result = await pipeline.calculate_batch(requests)

            assert "BTCUSDT" in result
            assert len(result["BTCUSDT"]) == 1
            assert result["BTCUSDT"][0] == mock_feature_value
            mock_calc.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculate_batch_multiple_symbols(self):
        """Test batch calculation for multiple symbols."""
        mock_store = Mock()
        mock_store.logger = Mock()

        pipeline = FeatureCalculationPipeline(mock_store)

        # Mock feature values
        btc_value = FeatureValue(
            feature_id="sma_20",
            symbol="BTCUSDT",
            value=Decimal("50000.00"),
            timestamp=datetime.now(timezone.utc),
            status=CalculationStatus.COMPLETED,
        )

        eth_value = FeatureValue(
            feature_id="sma_20",
            symbol="ETHUSDT",
            value=Decimal("3000.00"),
            timestamp=datetime.now(timezone.utc),
            status=CalculationStatus.COMPLETED,
        )

        with patch.object(pipeline, "_calculate_symbol_batch", new_callable=AsyncMock) as mock_calc:
            mock_calc.side_effect = [[btc_value], [eth_value]]

            requests = [
                FeatureRequest(symbol="BTCUSDT", feature_names=["sma_20"]),
                FeatureRequest(symbol="ETHUSDT", feature_names=["sma_20"]),
            ]

            result = await pipeline.calculate_batch(requests)

            assert "BTCUSDT" in result
            assert "ETHUSDT" in result
            assert len(result["BTCUSDT"]) == 1
            assert len(result["ETHUSDT"]) == 1
            assert result["BTCUSDT"][0] == btc_value
            assert result["ETHUSDT"][0] == eth_value
            assert mock_calc.call_count == 2

    @pytest.mark.asyncio
    async def test_calculate_batch_error_handling(self):
        """Test batch calculation error handling."""
        mock_store = Mock()
        mock_store.logger = Mock()

        pipeline = FeatureCalculationPipeline(mock_store)

        with patch.object(pipeline, "_calculate_symbol_batch", new_callable=AsyncMock) as mock_calc:
            mock_calc.side_effect = Exception("Calculation failed")

            requests = [
                FeatureRequest(symbol="BTCUSDT", feature_names=["sma_20"]),
            ]

            result = await pipeline.calculate_batch(requests)

            assert "BTCUSDT" in result
            assert result["BTCUSDT"] == []
            mock_store.logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_calculate_symbol_batch_no_data(self):
        """Test symbol batch calculation with no market data."""
        mock_store = Mock()
        mock_store.logger = Mock()
        mock_store._get_market_data = AsyncMock(return_value=[])

        pipeline = FeatureCalculationPipeline(mock_store)

        requests = [
            FeatureRequest(symbol="BTCUSDT", feature_names=["sma_20"], lookback_period=100),
        ]

        result = await pipeline._calculate_symbol_batch("BTCUSDT", requests)

        assert result == []
        mock_store.logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculate_symbol_batch_success(self, sample_market_data):
        """Test successful symbol batch calculation."""
        mock_store = Mock()
        mock_store.logger = Mock()
        mock_store._get_market_data = AsyncMock(return_value=sample_market_data)

        mock_feature_value = FeatureValue(
            feature_id="sma_20",
            symbol="BTCUSDT",
            value=Decimal("50000.00"),
            timestamp=datetime.now(timezone.utc),
            status=CalculationStatus.COMPLETED,
        )

        mock_store._calculate_single_feature = AsyncMock(return_value=mock_feature_value)

        pipeline = FeatureCalculationPipeline(mock_store)

        requests = [
            FeatureRequest(
                symbol="BTCUSDT", feature_names=["sma_20", "rsi_14"], lookback_period=100
            ),
        ]

        result = await pipeline._calculate_symbol_batch("BTCUSDT", requests)

        # Should have 2 features calculated
        assert len(result) == 2
        mock_store._get_market_data.assert_called_once_with("BTCUSDT", 100)
        assert mock_store._calculate_single_feature.call_count == 2

    @pytest.mark.asyncio
    async def test_calculate_symbol_batch_feature_error(self, sample_market_data):
        """Test symbol batch calculation with feature calculation error."""
        mock_store = Mock()
        mock_store.logger = Mock()
        mock_store._get_market_data = AsyncMock(return_value=sample_market_data)
        mock_store._calculate_single_feature = AsyncMock(
            side_effect=Exception("Feature calc failed")
        )

        pipeline = FeatureCalculationPipeline(mock_store)

        requests = [
            FeatureRequest(symbol="BTCUSDT", feature_names=["sma_20"], lookback_period=100),
        ]

        result = await pipeline._calculate_symbol_batch("BTCUSDT", requests)

        assert result == []
        mock_store.logger.error.assert_called()


class TestFeatureStore:
    """Test FeatureStore class."""

    def test_initialization(self, mock_config, mock_data_service):
        """Test feature store initialization."""
        store = FeatureStore(mock_config, mock_data_service)

        assert store.config is mock_config
        assert store.data_service is mock_data_service
        assert isinstance(store._features, dict)
        assert isinstance(store._calculators, dict)
        assert isinstance(store._feature_cache, dict)
        assert isinstance(store._calculation_locks, dict)
        assert isinstance(store.calculation_pipeline, FeatureCalculationPipeline)
        assert isinstance(store._metrics, dict)
        assert isinstance(store._background_tasks, list)
        assert store._initialized is False

    def test_initialization_without_data_service(self, mock_config):
        """Test initialization without data service."""
        store = FeatureStore(mock_config)

        assert store.data_service is None

    def test_setup_configuration_default(self):
        """Test configuration setup with defaults."""
        config = Mock(spec=Config)
        # No feature_store attribute

        store = FeatureStore(config)

        assert store.cache_config["max_cache_size"] == 10000
        assert store.cache_config["default_ttl"] == 300
        assert store.calculation_config["max_concurrent_calculations"] == 10

    def test_setup_configuration_custom(self, mock_config):
        """Test configuration setup with custom values."""
        store = FeatureStore(mock_config)

        assert store.cache_config["max_cache_size"] == 1000
        assert store.cache_config["default_ttl"] == 300
        assert store.calculation_config["max_concurrent_calculations"] == 10

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config, mock_data_service):
        """Test successful initialization."""
        store = FeatureStore(mock_config, mock_data_service)

        # Mock the initialization methods
        with (
            patch.object(
                store, "_register_builtin_features", new_callable=AsyncMock
            ) as mock_builtin,
            patch.object(
                store, "_initialize_technical_indicators", new_callable=AsyncMock
            ) as mock_tech,
            patch.object(
                store, "_initialize_statistical_features", new_callable=AsyncMock
            ) as mock_stat,
            patch.object(
                store, "_initialize_alternative_features", new_callable=AsyncMock
            ) as mock_alt,
            patch("asyncio.create_task") as mock_create_task,
        ):
            mock_task = AsyncMock()
            mock_create_task.return_value = mock_task

            await store.initialize()

            assert store._initialized is True
            mock_builtin.assert_called_once()
            mock_tech.assert_called_once()
            mock_stat.assert_called_once()
            mock_alt.assert_called_once()
            mock_create_task.assert_called_once()
            assert len(store._background_tasks) == 1

    @pytest.mark.asyncio
    async def test_initialize_twice(self, mock_config, mock_data_service):
        """Test initializing twice (should not reinitialize)."""
        store = FeatureStore(mock_config, mock_data_service)
        store._initialized = True

        await store.initialize()

        # Should return early without doing anything
        assert store._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_error(self, mock_config, mock_data_service):
        """Test initialization error handling."""
        store = FeatureStore(mock_config, mock_data_service)

        with patch.object(
            store, "_register_builtin_features", new_callable=AsyncMock
        ) as mock_builtin:
            mock_builtin.side_effect = Exception("Init failed")

            with pytest.raises(Exception, match="Init failed"):
                await store.initialize()

            assert store._initialized is False

    @pytest.mark.asyncio
    async def test_register_builtin_features(self, mock_config, mock_data_service):
        """Test built-in feature registration."""
        store = FeatureStore(mock_config, mock_data_service)

        # Mock register_feature method
        with patch.object(store, "register_feature", new_callable=AsyncMock) as mock_register:
            await store._register_builtin_features()

            # Should register multiple built-in features
            assert mock_register.call_count >= 4  # sma_20, ema_20, rsi_14, macd

    def test_metrics_initialization(self, mock_config):
        """Test that metrics are properly initialized."""
        store = FeatureStore(mock_config)

        expected_metrics = [
            "total_calculations",
            "cache_hits",
            "cache_misses",
            "failed_calculations",
            "avg_calculation_time",
            "duplicate_calculations_avoided",
        ]

        for metric in expected_metrics:
            assert metric in store._metrics
            assert isinstance(store._metrics[metric], (int, float))

    def test_error_handler_initialization(self, mock_config):
        """Test that error handler is properly initialized."""
        with patch("src.data.features.feature_store.ErrorHandler") as mock_error_handler:
            mock_instance = Mock()
            mock_error_handler.return_value = mock_instance

            store = FeatureStore(mock_config)

            assert store.error_handler is mock_instance
            mock_error_handler.assert_called_once_with(mock_config)


class TestFeatureStoreCalculations:
    """Test feature calculation methods (would need actual implementation details)."""

    def test_calculation_methods_exist(self, mock_config):
        """Test that calculation methods are defined (even if not implemented)."""
        store = FeatureStore(mock_config)

        # These methods should exist in the actual implementation
        expected_methods = [
            "_calculate_sma",
            "_calculate_ema",
            "_calculate_rsi",
            "_calculate_macd",
            "_get_market_data",
            "_calculate_single_feature",
        ]

        for method_name in expected_methods:
            # Check if method exists (may be stubbed in actual implementation)
            assert hasattr(store, method_name), f"Method {method_name} should exist"


class TestIntegrationScenarios:
    """Test integration scenarios between components."""

    @pytest.mark.asyncio
    async def test_pipeline_store_integration(self, mock_config, sample_market_data):
        """Test integration between pipeline and store."""
        store = FeatureStore(mock_config)
        store._get_market_data = AsyncMock(return_value=sample_market_data)

        # Create a simple mock feature calculation
        mock_feature_value = FeatureValue(
            feature_id="test_feature",
            symbol="BTCUSDT",
            value=Decimal("50000.00"),
            timestamp=datetime.now(timezone.utc),
            status=CalculationStatus.COMPLETED,
        )

        store._calculate_single_feature = AsyncMock(return_value=mock_feature_value)

        pipeline = store.calculation_pipeline

        requests = [
            FeatureRequest(symbol="BTCUSDT", feature_names=["test_feature"]),
        ]

        result = await pipeline.calculate_batch(requests)

        assert "BTCUSDT" in result
        assert len(result["BTCUSDT"]) == 1
        assert result["BTCUSDT"][0] == mock_feature_value

    def test_feature_value_decimal_precision(self):
        """Test that feature values maintain decimal precision."""
        high_precision_value = Decimal("50000.12345678")

        feature_value = FeatureValue(
            feature_id="precision_test",
            symbol="BTCUSDT",
            value=high_precision_value,
            timestamp=datetime.now(timezone.utc),
            status=CalculationStatus.COMPLETED,
        )

        assert feature_value.value == high_precision_value
        assert str(feature_value.value) == "50000.12345678"


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_feature_request_edge_cases(self):
        """Test feature request with edge case values."""
        # Minimum valid values
        request = FeatureRequest(
            symbol="A",  # Minimum length
            feature_names=["f"],  # Single feature
            lookback_period=1,  # Minimum lookback
            priority=1,  # Minimum priority
        )

        assert request.symbol == "A"
        assert request.feature_names == ["f"]
        assert request.lookback_period == 1
        assert request.priority == 1

        # Maximum valid values
        request = FeatureRequest(
            symbol="A" * 20,  # Maximum length
            feature_names=["feature"] * 10,  # Multiple features
            lookback_period=5000,  # Maximum lookback
            priority=10,  # Maximum priority
        )

        assert len(request.symbol) == 20
        assert len(request.feature_names) == 10
        assert request.lookback_period == 5000
        assert request.priority == 10

    def test_feature_value_with_complex_dict(self):
        """Test feature value with complex dictionary structure."""
        complex_dict = {
            "bands": {
                "upper": Decimal("51000.00"),
                "middle": Decimal("50000.00"),
                "lower": Decimal("49000.00"),
            },
            "signals": {
                "buy": True,
                "sell": False,
                "strength": Decimal("0.75"),
            },
        }

        feature_value = FeatureValue(
            feature_id="complex_indicator",
            symbol="BTCUSDT",
            value=complex_dict,
            timestamp=datetime.now(timezone.utc),
            status=CalculationStatus.COMPLETED,
        )

        assert feature_value.value == complex_dict
        assert feature_value.value["bands"]["upper"] == Decimal("51000.00")
        assert feature_value.value["signals"]["buy"] is True

    @pytest.mark.asyncio
    async def test_pipeline_with_mixed_success_failure(self, sample_market_data):
        """Test pipeline handling mixed success and failure scenarios."""
        mock_store = Mock()
        mock_store.logger = Mock()
        mock_store._get_market_data = AsyncMock(return_value=sample_market_data)

        # Mock some features to succeed, others to fail
        def mock_calc_side_effect(symbol, feature_name, market_data, params):
            if feature_name == "working_feature":
                return FeatureValue(
                    feature_id=feature_name,
                    symbol=symbol,
                    value=Decimal("100.00"),
                    timestamp=datetime.now(timezone.utc),
                    status=CalculationStatus.COMPLETED,
                )
            else:
                raise Exception(f"Feature {feature_name} failed")

        mock_store._calculate_single_feature = AsyncMock(side_effect=mock_calc_side_effect)

        pipeline = FeatureCalculationPipeline(mock_store)

        requests = [
            FeatureRequest(symbol="BTCUSDT", feature_names=["working_feature", "failing_feature"]),
        ]

        result = await pipeline._calculate_symbol_batch("BTCUSDT", requests)

        # Should have one successful result, one failed (not returned)
        assert len(result) == 1
        assert result[0].feature_id == "working_feature"

        # Should have logged error for failing feature
        mock_store.logger.error.assert_called()
