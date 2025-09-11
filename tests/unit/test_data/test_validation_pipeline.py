"""
Tests for DataValidationPipeline

Comprehensive test coverage for data validation pipeline including:
- Pipeline initialization and configuration
- Validation stage execution
- Data disposition logic
- Quality scoring and metrics
- Error handling and recovery
- Quarantine management
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.config import Config
from src.core.types import MarketData
from src.data.pipeline.validation_pipeline import (
    DataValidationPipeline,
    ValidationAction,
    ValidationDisposition,
    ValidationMetrics,
    ValidationPipelineConfig,
    ValidationPipelineResult,
    ValidationStage,
)
from src.data.validation.data_validator import (
    DataValidator,
    MarketDataValidationResult,
)
from src.utils.validation.validation_types import (
    QualityDimension,
    QualityScore,
    ValidationCategory,
    ValidationIssue,
    ValidationSeverity,
)
from src.utils.pipeline_utilities import PipelineAction


@pytest.fixture
def config():
    """Create test configuration."""
    # Use dict config instead of Config object for testing
    return {
        "validation_pipeline": {
            "min_quality_score": 0.7,
            "quarantine_quality_threshold": 0.5,
            "max_critical_issues": 0,
            "max_error_issues": 3,
            "max_warning_issues": 10,
        }
    }


@pytest.fixture
def mock_validator():
    """Create mock validator."""
    validator = Mock(spec=DataValidator)
    validator.validate_market_data = AsyncMock()
    validator.health_check = AsyncMock(return_value={"status": "healthy"})
    return validator


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return [
        MarketData(
            symbol="BTC/USD",
            price=Decimal("50000.00"),
            volume=Decimal("100.0"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
            open=Decimal("49800.00"),
            high=Decimal("50200.00"),
            low=Decimal("49500.00"),
            close=Decimal("50000.00"),
            bid_price=Decimal("49999.50"),
            ask_price=Decimal("50000.50"),
        ),
        MarketData(
            symbol="ETH/USD",
            price=Decimal("3000.00"),
            volume=Decimal("200.0"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
            open=Decimal("2980.00"),
            high=Decimal("3020.00"),
            low=Decimal("2950.00"),
            close=Decimal("3000.00"),
            bid_price=Decimal("2999.50"),
            ask_price=Decimal("3000.50"),
        ),
    ]


@pytest.fixture
def validation_result_valid():
    """Create valid validation result."""
    return MarketDataValidationResult(
        symbol="BTC/USD",
        is_valid=True,
        quality_score=0.95,
        error_count=0,
        errors=[],
        validation_timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def validation_result_invalid():
    """Create invalid validation result."""
    return MarketDataValidationResult(
        symbol="BTC/USD",
        is_valid=False,
        quality_score=0.3,
        error_count=2,
        errors=[
            "Price is negative",
            "Volume is invalid"
        ],
        validation_timestamp=datetime.now(timezone.utc),
    )


class TestValidationPipelineConfig:
    """Test validation pipeline configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ValidationPipelineConfig()

        assert config.enable_schema_validation is True
        assert config.min_quality_score == 0.7
        assert config.quarantine_quality_threshold == 0.5
        assert config.max_critical_issues == 0
        assert config.batch_size == 100
        assert config.retry_attempts == 3

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        config = ValidationPipelineConfig(
            min_quality_score=0.8,
            quarantine_quality_threshold=0.6,
        )
        assert config.min_quality_score == 0.8
        assert config.quarantine_quality_threshold == 0.6

    def test_config_field_validation(self):
        """Test field validation."""
        # Test invalid quality score
        with pytest.raises(ValueError):
            ValidationPipelineConfig(min_quality_score=1.5)  # > 1.0

        with pytest.raises(ValueError):
            ValidationPipelineConfig(min_quality_score=-0.1)  # < 0.0


class TestValidationDisposition:
    """Test validation disposition model."""

    def test_disposition_creation(self):
        """Test disposition model creation."""
        disposition = ValidationDisposition(
            action=ValidationAction.ACCEPT,
            quality_score=0.95,
            critical_issues=0,
            error_issues=0,
            warning_issues=1,
            reasons=["Minor warning"],
        )

        assert disposition.action == ValidationAction.ACCEPT
        assert disposition.quality_score == 0.95
        assert disposition.warning_issues == 1
        assert len(disposition.reasons) == 1

    def test_disposition_defaults(self):
        """Test disposition defaults."""
        disposition = ValidationDisposition(
            action=ValidationAction.REJECT,
            quality_score=0.2,
            critical_issues=0,
            error_issues=0,
            warning_issues=0,
        )

        assert disposition.critical_issues == 0
        assert disposition.error_issues == 0
        assert disposition.warning_issues == 0
        assert disposition.reasons == []
        assert disposition.metadata == {}


class TestDataValidationPipeline:
    """Test DataValidationPipeline class."""

    def test_initialization(self, config):
        """Test pipeline initialization."""
        pipeline = DataValidationPipeline(config)

        assert pipeline.config == config
        assert isinstance(pipeline.pipeline_config, ValidationPipelineConfig)
        assert pipeline._initialized is False
        assert isinstance(pipeline._active_validations, dict)
        assert isinstance(pipeline._quarantine_store, dict)

    def test_initialization_with_validator(self, config, mock_validator):
        """Test pipeline initialization with injected validator."""
        pipeline = DataValidationPipeline(config, validator=mock_validator)

        assert pipeline.validator == mock_validator

    def test_setup_configuration(self, config):
        """Test configuration setup."""
        pipeline = DataValidationPipeline(config)

        assert pipeline.pipeline_config.min_quality_score == 0.7
        assert pipeline.pipeline_config.max_critical_issues == 0

    def test_setup_configuration_empty(self):
        """Test configuration setup with empty config."""
        config = Config()
        pipeline = DataValidationPipeline(config)

        # Should use defaults
        assert pipeline.pipeline_config.min_quality_score == 0.7

    @pytest.mark.asyncio
    async def test_initialize_success(self, config, mock_validator):
        """Test successful initialization."""
        with patch("src.data.pipeline.validation_pipeline.DataPipeline") as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline

            pipeline = DataValidationPipeline(config, validator=mock_validator)
            await pipeline.initialize()

            assert pipeline._initialized is True
            assert pipeline.data_pipeline == mock_pipeline

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, config, mock_validator):
        """Test initialization when already initialized."""
        pipeline = DataValidationPipeline(config, validator=mock_validator)
        pipeline._initialized = True

        await pipeline.initialize()  # Should not raise

    @pytest.mark.asyncio
    async def test_initialize_with_datapipeline_failure(self, config, mock_validator):
        """Test initialization when DataPipeline creation fails but validation pipeline continues."""
        with patch("src.data.pipeline.validation_pipeline.DataPipeline") as mock_pipeline_class:
            mock_pipeline_class.side_effect = Exception("Init failed")

            pipeline = DataValidationPipeline(config, validator=mock_validator)
            await pipeline.initialize()

            # Should succeed with warning and data_pipeline set to None
            assert pipeline._initialized is True
            assert pipeline.data_pipeline is None

    @pytest.mark.asyncio
    async def test_validate_batch_empty_data(self, config, mock_validator):
        """Test validation with empty data."""
        pipeline = DataValidationPipeline(config, validator=mock_validator)

        result = await pipeline.validate_batch([])

        assert result.total_records == 0
        assert result.dispositions == {}
        assert result.execution_time_ms == 0

    @pytest.mark.asyncio
    async def test_validate_batch_symbol_filter(self, config, mock_validator, sample_market_data):
        """Test validation with symbol filtering."""
        mock_validator.validate_market_data.return_value = []

        pipeline = DataValidationPipeline(config, validator=mock_validator)

        # Filter to only BTC/USD
        result = await pipeline.validate_batch(sample_market_data, symbols=["BTC/USD"])

        # Should only process BTC/USD data
        mock_validator.validate_market_data.assert_called_once()
        call_args = mock_validator.validate_market_data.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].symbol == "BTC/USD"

    @pytest.mark.asyncio
    async def test_validate_batch_success(self, config, mock_validator, sample_market_data, validation_result_valid):
        """Test successful batch validation."""
        mock_validator.validate_market_data.return_value = [validation_result_valid]

        with patch("src.data.pipeline.validation_pipeline.DataPipeline"):
            pipeline = DataValidationPipeline(config, validator=mock_validator)

            result = await pipeline.validate_batch(sample_market_data)

            assert result.total_records == 2
            assert len(result.dispositions) == 2  # Two symbols
            assert result.execution_time_ms >= 0
            assert isinstance(result.pipeline_id, str)

    @pytest.mark.asyncio
    async def test_validate_batch_not_initialized(self, config, mock_validator, sample_market_data):
        """Test validation auto-initializes when not initialized."""
        mock_validator.validate_market_data.return_value = []

        with patch("src.data.pipeline.validation_pipeline.DataPipeline"):
            pipeline = DataValidationPipeline(config, validator=mock_validator)
            assert pipeline._initialized is False

            await pipeline.validate_batch(sample_market_data)

            assert pipeline._initialized is True

    @pytest.mark.asyncio
    async def test_validate_batch_validation_error(self, config, mock_validator, sample_market_data):
        """Test validation with validator error - should handle gracefully and reject records."""
        mock_validator.validate_market_data.side_effect = Exception("Validation failed")

        with patch("src.data.pipeline.validation_pipeline.DataPipeline"):
            pipeline = DataValidationPipeline(config, validator=mock_validator)

            result = await pipeline.validate_batch(sample_market_data)
            
            # Should handle validation errors gracefully and reject records
            assert result.total_records == 2
            assert len(result.dispositions) == 2
            # All records should be rejected due to validation errors
            for disposition in result.dispositions.values():
                assert disposition.action.value == "reject"
                assert disposition.critical_issues == 1  # Error creates critical issue
                assert "Validation error: Validation failed" in disposition.reasons

    def test_group_data_by_symbol(self, config, sample_market_data):
        """Test data grouping by symbol."""
        pipeline = DataValidationPipeline(config)

        groups = pipeline._group_data_by_symbol(sample_market_data)

        assert len(groups) == 2
        assert "BTC/USD" in groups
        assert "ETH/USD" in groups
        assert len(groups["BTC/USD"]) == 1
        assert len(groups["ETH/USD"]) == 1

    def test_group_data_by_symbol_empty(self, config):
        """Test data grouping with empty data."""
        pipeline = DataValidationPipeline(config)

        groups = pipeline._group_data_by_symbol([])

        assert groups == {}

    @pytest.mark.asyncio
    async def test_update_pipeline_stage(self, config):
        """Test pipeline stage update."""
        pipeline = DataValidationPipeline(config)
        pipeline_id = "test-id"
        pipeline._active_validations[pipeline_id] = {"stage": ValidationStage.INTAKE}

        await pipeline._update_pipeline_stage(pipeline_id, ValidationStage.SCHEMA_VALIDATION)

        assert pipeline._active_validations[pipeline_id]["stage"] == ValidationStage.SCHEMA_VALIDATION

    @pytest.mark.asyncio
    async def test_update_pipeline_stage_missing_id(self, config):
        """Test pipeline stage update with missing ID."""
        pipeline = DataValidationPipeline(config)

        # Should not raise error
        await pipeline._update_pipeline_stage("missing-id", ValidationStage.SCHEMA_VALIDATION)

    @pytest.mark.asyncio
    async def test_determine_symbol_disposition_empty_results(self, config):
        """Test symbol disposition with empty results."""
        pipeline = DataValidationPipeline(config)

        disposition = await pipeline._determine_symbol_disposition("BTC/USD", [])

        assert disposition.action == ValidationAction.REJECT
        assert disposition.quality_score == 0.0
        assert disposition.critical_issues == 1
        assert "No validation results" in disposition.reasons

    @pytest.mark.asyncio
    async def test_determine_symbol_disposition_valid(self, config, validation_result_valid):
        """Test symbol disposition with valid results."""
        pipeline = DataValidationPipeline(config)

        disposition = await pipeline._determine_symbol_disposition("BTC/USD", [validation_result_valid])

        assert disposition.action == ValidationAction.ACCEPT
        assert disposition.quality_score == 0.95
        assert disposition.critical_issues == 0

    @pytest.mark.asyncio
    async def test_determine_symbol_disposition_invalid(self, config, validation_result_invalid):
        """Test symbol disposition with invalid results."""
        pipeline = DataValidationPipeline(config)

        disposition = await pipeline._determine_symbol_disposition("BTC/USD", [validation_result_invalid])

        assert disposition.action == ValidationAction.REJECT
        assert disposition.error_issues == 2

    @pytest.mark.asyncio
    async def test_determine_record_disposition_valid(self, config, validation_result_valid):
        """Test record disposition with valid result."""
        pipeline = DataValidationPipeline(config)

        disposition = await pipeline._determine_record_disposition(validation_result_valid)

        assert disposition.action == ValidationAction.ACCEPT
        assert disposition.quality_score == 0.95
        assert disposition.critical_issues == 0

    @pytest.mark.asyncio
    async def test_determine_record_disposition_invalid(self, config, validation_result_invalid):
        """Test record disposition with invalid result."""
        pipeline = DataValidationPipeline(config)

        disposition = await pipeline._determine_record_disposition(validation_result_invalid)

        assert disposition.action == ValidationAction.REJECT
        assert disposition.error_issues == 2
        assert "Failed validation" in disposition.reasons

    def test_determine_action_accept(self, config):
        """Test action determination for accept."""
        pipeline = DataValidationPipeline(config)

        action = pipeline._determine_action(
            quality_score=0.95,
            critical_issues=0,
            error_issues=0,
            warning_issues=0,
            valid_records=1,
            total_records=1,
        )

        assert action == ValidationAction.ACCEPT

    def test_determine_action_accept_with_warning(self, config):
        """Test action determination for accept with warning."""
        pipeline = DataValidationPipeline(config)

        action = pipeline._determine_action(
            quality_score=0.85,
            critical_issues=0,
            error_issues=1,
            warning_issues=2,
            valid_records=1,
            total_records=1,
        )

        assert action == ValidationAction.ACCEPT_WITH_WARNING

    def test_determine_action_quarantine_quality(self, config):
        """Test action determination for quarantine due to quality."""
        pipeline = DataValidationPipeline(config)

        action = pipeline._determine_action(
            quality_score=0.6,  # Below min_quality_score but above quarantine threshold
            critical_issues=0,
            error_issues=0,
            warning_issues=0,
            valid_records=1,
            total_records=1,
        )

        assert action == ValidationAction.QUARANTINE

    def test_determine_action_quarantine_warnings(self, config):
        """Test action determination for quarantine due to warnings."""
        pipeline = DataValidationPipeline(config)

        action = pipeline._determine_action(
            quality_score=0.85,
            critical_issues=0,
            error_issues=0,
            warning_issues=15,  # Above max_warning_issues
            valid_records=1,
            total_records=1,
        )

        assert action == ValidationAction.QUARANTINE

    def test_determine_action_reject_critical(self, config):
        """Test action determination for reject due to critical issues."""
        pipeline = DataValidationPipeline(config)

        action = pipeline._determine_action(
            quality_score=0.95,
            critical_issues=1,  # Above max_critical_issues (0)
            error_issues=0,
            warning_issues=0,
            valid_records=1,
            total_records=1,
        )

        assert action == ValidationAction.REJECT

    def test_determine_action_reject_errors(self, config):
        """Test action determination for reject due to errors."""
        pipeline = DataValidationPipeline(config)

        action = pipeline._determine_action(
            quality_score=0.95,
            critical_issues=0,
            error_issues=5,  # Above max_error_issues (3)
            warning_issues=0,
            valid_records=1,
            total_records=1,
        )

        assert action == ValidationAction.REJECT

    def test_determine_action_reject_quality(self, config):
        """Test action determination for reject due to low quality."""
        pipeline = DataValidationPipeline(config)

        action = pipeline._determine_action(
            quality_score=0.3,  # Below quarantine_quality_threshold
            critical_issues=0,
            error_issues=0,
            warning_issues=0,
            valid_records=1,
            total_records=1,
        )

        assert action == ValidationAction.REJECT

    def test_calculate_pipeline_metrics(self, config, sample_market_data):
        """Test pipeline metrics calculation."""
        pipeline = DataValidationPipeline(config)

        dispositions = {
            "BTC/USD": ValidationDisposition(
                action=ValidationAction.ACCEPT,
                quality_score=0.95,
                critical_issues=0,
                error_issues=0,
                warning_issues=1,
                metadata={"total_records": 1},
            ),
            "ETH/USD": ValidationDisposition(
                action=ValidationAction.QUARANTINE,
                quality_score=0.65,
                critical_issues=0,
                error_issues=0,
                warning_issues=5,
                metadata={"total_records": 1},
            ),
        }

        metrics = pipeline._calculate_pipeline_metrics(dispositions, sample_market_data)

        assert metrics.total_records_processed == 2
        assert metrics.successful_records == 1
        assert metrics.records_quarantined == 1
        assert metrics.records_rejected == 0
        assert metrics.warning_issues == 6
        assert metrics.average_quality_score == 0.8  # (0.95 + 0.65) / 2

    def test_calculate_pipeline_metrics_empty(self, config):
        """Test pipeline metrics calculation with empty dispositions."""
        pipeline = DataValidationPipeline(config)

        metrics = pipeline._calculate_pipeline_metrics({}, [])

        assert metrics.total_records_processed == 0
        assert metrics.successful_records == 0
        assert metrics.average_quality_score == 0.0

    def test_update_session_metrics(self, config):
        """Test session metrics update."""
        pipeline = DataValidationPipeline(config)

        pipeline_metrics = ValidationMetrics()
        pipeline_metrics.total_records_processed = 10
        pipeline_metrics.successful_records = 8
        pipeline_metrics.records_quarantined = 1
        pipeline_metrics.records_rejected = 1

        pipeline._update_session_metrics(pipeline_metrics)

        assert pipeline._session_metrics.total_records_processed == 10
        assert pipeline._session_metrics.successful_records == 8
        assert pipeline._session_metrics.records_quarantined == 1
        assert pipeline._session_metrics.records_rejected == 1

    @pytest.mark.asyncio
    async def test_get_quarantined_data_all(self, config):
        """Test getting all quarantined data."""
        pipeline = DataValidationPipeline(config)
        # Mock some quarantined data
        from src.core.types import MarketData
        from decimal import Decimal
        from datetime import datetime, timezone
        
        mock_data = MarketData(
            symbol="BTC/USD", 
            timestamp=datetime.now(timezone.utc),
            open=Decimal("50000"), high=Decimal("51000"), low=Decimal("49000"), close=Decimal("50500"),
            volume=Decimal("100"), exchange="test"
        )
        
        pipeline._quarantine_store = {
            "BTC/USD": [mock_data],
            "ETH/USD": []
        }

        quarantined = await pipeline.get_quarantined_data()

        assert len(quarantined) == 2
        assert "BTC/USD" in quarantined
        assert "ETH/USD" in quarantined

    @pytest.mark.asyncio
    async def test_get_quarantined_data_symbol(self, config, sample_market_data):
        """Test getting quarantined data for specific symbol."""
        pipeline = DataValidationPipeline(config)
        pipeline._quarantine_store = {
            "BTC/USD": [sample_market_data[0]],
            "ETH/USD": []
        }

        quarantined = await pipeline.get_quarantined_data("BTC/USD")

        assert len(quarantined) == 1
        assert "BTC/USD" in quarantined
        assert len(quarantined["BTC/USD"]) == 1

    @pytest.mark.asyncio
    async def test_get_quarantined_data_missing_symbol(self, config):
        """Test getting quarantined data for missing symbol."""
        pipeline = DataValidationPipeline(config)

        quarantined = await pipeline.get_quarantined_data("MISSING/USD")

        assert quarantined == {"MISSING/USD": []}

    @pytest.mark.asyncio
    async def test_retry_quarantined_data_missing(self, config):
        """Test retrying quarantined data for missing symbol."""
        pipeline = DataValidationPipeline(config)

        result = await pipeline.retry_quarantined_data("MISSING/USD")

        assert result is None

    @pytest.mark.asyncio
    async def test_retry_quarantined_data_empty(self, config):
        """Test retrying quarantined data with empty list."""
        pipeline = DataValidationPipeline(config)
        pipeline._quarantine_store["BTC/USD"] = []

        result = await pipeline.retry_quarantined_data("BTC/USD")

        assert result is None

    @pytest.mark.asyncio
    async def test_retry_quarantined_data_success(self, config, mock_validator, sample_market_data):
        """Test successful retry of quarantined data."""
        mock_validator.validate_market_data.return_value = []

        pipeline = DataValidationPipeline(config, validator=mock_validator)
        pipeline._quarantine_store["BTC/USD"] = [sample_market_data[0]]

        with patch.object(pipeline, "validate_batch") as mock_validate:
            mock_result = ValidationPipelineResult(
                pipeline_id="test-id",
                total_records=1,
                dispositions={
                    "BTC/USD": ValidationDisposition(
                        action=ValidationAction.ACCEPT,
                        quality_score=0.95,
                        critical_issues=0,
                        error_issues=0,
                        warning_issues=0,
                    )
                },
                metrics=ValidationMetrics(),
                execution_time_ms=100,
            )
            mock_validate.return_value = mock_result

            result = await pipeline.retry_quarantined_data("BTC/USD")

            assert result == mock_result
            assert "BTC/USD" not in pipeline._quarantine_store

    @pytest.mark.asyncio
    async def test_retry_quarantined_data_still_quarantined(self, config, mock_validator, sample_market_data):
        """Test retry of quarantined data that's still quarantined."""
        mock_validator.validate_market_data.return_value = []

        pipeline = DataValidationPipeline(config, validator=mock_validator)
        pipeline._quarantine_store["BTC/USD"] = [sample_market_data[0]]

        with patch.object(pipeline, "validate_batch") as mock_validate:
            mock_result = ValidationPipelineResult(
                pipeline_id="test-id",
                total_records=1,
                dispositions={
                    "BTC/USD": ValidationDisposition(
                        action=ValidationAction.QUARANTINE,
                        quality_score=0.65,
                        critical_issues=0,
                        error_issues=0,
                        warning_issues=0,
                    )
                },
                metrics=ValidationMetrics(),
                execution_time_ms=100,
            )
            mock_validate.return_value = mock_result

            result = await pipeline.retry_quarantined_data("BTC/USD")

            assert result == mock_result
            assert "BTC/USD" in pipeline._quarantine_store  # Still quarantined

    @pytest.mark.asyncio
    async def test_get_pipeline_status(self, config):
        """Test getting pipeline status."""
        pipeline = DataValidationPipeline(config)
        pipeline._active_validations = {"id1": {}, "id2": {}}
        pipeline._quarantine_store = {
            "BTC/USD": [Mock(), Mock()],
            "ETH/USD": [Mock()]
        }

        status = await pipeline.get_pipeline_status()

        assert status["active_validations"] == 2
        assert status["quarantined_symbols"] == 2
        assert status["quarantined_records"] == 3
        assert "session_metrics" in status
        assert "configuration" in status

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, config, mock_validator):
        """Test health check when healthy."""
        mock_validator.health_check.return_value = {"status": "healthy"}

        pipeline = DataValidationPipeline(config, validator=mock_validator)
        pipeline._initialized = True
        pipeline.data_pipeline = Mock()

        health = await pipeline.health_check()

        assert health["status"] == "healthy"
        assert health["initialized"] is True
        assert health["validator_available"] is True
        assert health["data_pipeline_available"] is True
        assert health["validator_status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_validator_error(self, config, mock_validator):
        """Test health check with validator error."""
        mock_validator.health_check.side_effect = Exception("Validator error")

        pipeline = DataValidationPipeline(config, validator=mock_validator)
        pipeline._initialized = True

        health = await pipeline.health_check()

        assert health["status"] == "degraded"
        assert "unhealthy: Validator error" in health["validator_status"]

    @pytest.mark.asyncio
    async def test_cleanup_success(self, config, mock_validator):
        """Test successful cleanup."""
        pipeline = DataValidationPipeline(config, validator=mock_validator)
        pipeline._initialized = True
        pipeline._active_validations = {"id1": {}}
        pipeline._quarantine_store = {"BTC/USD": []}

        mock_data_pipeline = Mock()
        mock_data_pipeline.cleanup = AsyncMock()
        pipeline.data_pipeline = mock_data_pipeline

        await pipeline.cleanup()

        assert pipeline._active_validations == {}
        assert pipeline._quarantine_store == {}
        assert pipeline._initialized is False
        mock_data_pipeline.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_validator_with_cleanup(self, config):
        """Test cleanup when validator has cleanup method."""
        mock_validator = Mock()
        mock_validator.cleanup = AsyncMock()
        mock_validator.health_check = AsyncMock(return_value={"status": "healthy"})

        pipeline = DataValidationPipeline(config, validator=mock_validator)
        pipeline._initialized = True

        await pipeline.cleanup()

        mock_validator.cleanup.assert_called_once()
        assert pipeline._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self, config, mock_validator):
        """Test cleanup error handling."""
        pipeline = DataValidationPipeline(config, validator=mock_validator)
        pipeline._initialized = True

        mock_data_pipeline = Mock()
        mock_data_pipeline.cleanup = AsyncMock(side_effect=Exception("Cleanup error"))
        pipeline.data_pipeline = mock_data_pipeline

        # Should not raise exception
        await pipeline.cleanup()

        # Should still set initialized to False
        assert pipeline._initialized is False


class TestValidationPipelineIntegration:
    """Integration tests for validation pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, config, sample_market_data):
        """Test full pipeline execution with real validator."""
        with patch("src.data.pipeline.validation_pipeline.DataPipeline"), \
             patch.object(DataValidator, "validate_market_data") as mock_validate:

            # Setup validation results
            validation_results = [
                MarketDataValidationResult(
                    symbol="BTC/USD",
                    is_valid=True,
                    quality_score=0.95,
                    errors=[],
                    validation_timestamp=datetime.now(timezone.utc),
                ),
                MarketDataValidationResult(
                    symbol="ETH/USD",
                    is_valid=True,
                    quality_score=0.85,
                    errors=[],
                    validation_timestamp=datetime.now(timezone.utc),
                ),
            ]
            mock_validate.return_value = validation_results

            pipeline = DataValidationPipeline(config)
            result = await pipeline.validate_batch(sample_market_data)

            assert result.total_records == 2
            assert len(result.dispositions) == 2
            assert all(d.action == ValidationAction.ACCEPT for d in result.dispositions.values())

    @pytest.mark.asyncio
    async def test_pipeline_with_quarantine_scenario(self, config, sample_market_data):
        """Test pipeline with quarantine scenario."""
        with patch("src.data.pipeline.validation_pipeline.DataPipeline"), \
             patch.object(DataValidator, "validate_market_data") as mock_validate:

            # Setup mixed validation results
            validation_results = [
                MarketDataValidationResult(
                    symbol="BTC/USD",
                    is_valid=True,
                    quality_score=0.65,  # Below min but above quarantine
                    errors=[],
                    validation_timestamp=datetime.now(timezone.utc),
                ),
            ]
            mock_validate.return_value = validation_results

            pipeline = DataValidationPipeline(config)
            result = await pipeline.validate_batch([sample_market_data[0]])

            assert result.total_records == 1
            disposition = result.dispositions["BTC/USD"]
            assert disposition.action == ValidationAction.QUARANTINE

    @pytest.mark.asyncio
    async def test_pipeline_error_recovery(self, config, sample_market_data, mock_validator):
        """Test pipeline error recovery."""
        mock_validator.validate_market_data.side_effect = [
            Exception("Temporary error"),
            []  # Success on retry
        ]

        with patch("src.data.pipeline.validation_pipeline.DataPipeline"):
            pipeline = DataValidationPipeline(config, validator=mock_validator)

            # First call should handle error gracefully
            result = await pipeline.validate_batch(sample_market_data)
            
            # Check that errors were handled properly
            assert result.total_records == 2
            for disposition in result.dispositions.values():
                assert disposition.action == ValidationAction.REJECT

            # Pipeline should handle the error and still be functional
            assert pipeline._initialized is True


class TestValidationEnums:
    """Test validation enumerations."""

    def test_validation_stage_enum(self):
        """Test ValidationStage enum."""
        assert ValidationStage.INTAKE.value == "intake"
        assert ValidationStage.COMPLETED.value == "completed"

    def test_validation_action_alias(self):
        """Test ValidationAction is properly aliased."""
        assert ValidationAction.ACCEPT == PipelineAction.ACCEPT
        assert ValidationAction.REJECT == PipelineAction.REJECT


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_large_batch_processing(self, config, mock_validator):
        """Test processing large batches."""
        # Create large dataset
        large_data = []
        for i in range(1000):
            large_data.append(
                MarketData(
                    symbol=f"COIN{i}/USD",
                    price=Decimal("100.00"),
                    volume=Decimal("10.0"),
                    timestamp=datetime.now(timezone.utc),
                    exchange="binance",
                    open=Decimal("99.50"),
                    high=Decimal("100.50"),
                    low=Decimal("99.00"),
                    close=Decimal("100.00"),
                    bid_price=Decimal("99.99"),
                    ask_price=Decimal("100.01"),
                )
            )

        mock_validator.validate_market_data.return_value = []

        with patch("src.data.pipeline.validation_pipeline.DataPipeline"):
            pipeline = DataValidationPipeline(config, validator=mock_validator)

            result = await pipeline.validate_batch(large_data)

            assert result.total_records == 1000

    @pytest.mark.asyncio
    async def test_zero_quality_score_handling(self, config):
        """Test handling of zero quality scores."""
        pipeline = DataValidationPipeline(config)

        action = pipeline._determine_action(
            quality_score=0.0,
            critical_issues=0,
            error_issues=0,
            warning_issues=0,
            valid_records=1,
            total_records=1,
        )

        assert action == ValidationAction.REJECT

    @pytest.mark.asyncio
    async def test_perfect_quality_score_handling(self, config):
        """Test handling of perfect quality scores."""
        pipeline = DataValidationPipeline(config)

        action = pipeline._determine_action(
            quality_score=1.0,
            critical_issues=0,
            error_issues=0,
            warning_issues=0,
            valid_records=1,
            total_records=1,
        )

        assert action == ValidationAction.ACCEPT

    def test_concurrent_pipeline_tracking(self, config):
        """Test concurrent pipeline execution tracking."""
        pipeline = DataValidationPipeline(config)

        # Simulate multiple concurrent pipelines
        id1 = "pipeline-1"
        id2 = "pipeline-2"

        pipeline._active_validations[id1] = {"stage": ValidationStage.INTAKE}
        pipeline._active_validations[id2] = {"stage": ValidationStage.SCHEMA_VALIDATION}

        assert len(pipeline._active_validations) == 2
        assert pipeline._active_validations[id1]["stage"] == ValidationStage.INTAKE
        assert pipeline._active_validations[id2]["stage"] == ValidationStage.SCHEMA_VALIDATION

    @pytest.mark.asyncio
    async def test_malformed_market_data_handling(self, config, mock_validator):
        """Test handling of malformed market data."""
        # Create market data with None values
        malformed_data = [
            MarketData(
                symbol="BTC/USD",
                price=None,  # Invalid
                volume=Decimal("100.0"),
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
                open=Decimal("50000.0"),
                high=Decimal("51000.0"),
                low=Decimal("49000.0"),
                close=Decimal("50500.0"),
                bid_price=Decimal("50499.0"),
                ask_price=Decimal("50501.0"),
            )
        ]

        mock_validator.validate_market_data.side_effect = ValueError("Invalid price")

        with patch("src.data.pipeline.validation_pipeline.DataPipeline"):
            pipeline = DataValidationPipeline(config, validator=mock_validator)

            # Pipeline should handle validation errors gracefully
            result = await pipeline.validate_batch(malformed_data)
            
            # Should have rejected records due to validation error
            assert result.metrics.records_rejected == 1
            assert result.metrics.successful_records == 0
