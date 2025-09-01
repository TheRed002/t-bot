"""
Test suite for data validation pipeline.

This module contains comprehensive tests for the DataValidationPipeline
including validation stages, disposition logic, metrics calculation,
and error handling scenarios.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import uuid

from src.core.config import Config
from src.core.types import MarketData
from src.data.pipeline.validation_pipeline import (
    DataValidationPipeline,
    ValidationStage,
    ValidationAction,
    ValidationMetrics,
    ValidationPipelineConfig,
    ValidationDisposition,
    ValidationPipelineResult,
)
from src.data.validation.data_validator import (
    DataValidator,
    MarketDataValidationResult,
    QualityScore,
    ValidationIssue,
    ValidationSeverity,
    ValidationCategory,
    QualityDimension,
)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock(spec=Config)
    config.validation_pipeline = {
        "min_quality_score": 0.7,
        "quarantine_quality_threshold": 0.5,
        "max_critical_issues": 0,
        "max_error_issues": 3,
        "max_warning_issues": 10,
        "batch_size": 100,
        "max_workers": 4,
        "timeout_seconds": 60,
        "retry_attempts": 3,
    }
    return config


@pytest.fixture
def mock_validator():
    """Mock data validator for testing."""
    validator = Mock(spec=DataValidator)
    validator.validate_market_data = AsyncMock()
    validator.health_check = AsyncMock(return_value={"status": "healthy"})
    return validator


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return [
        MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("50200.00"),
            high=Decimal("50500.00"),
            low=Decimal("49500.00"),
            close=Decimal("50000.00"),
            volume=Decimal("1.5"),
            exchange="binance",
            bid_price=Decimal("49999.50"),
            ask_price=Decimal("50000.50"),
        ),
        MarketData(
            symbol="ETHUSDT",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("3050.00"),
            high=Decimal("3100.00"),
            low=Decimal("2950.00"),
            close=Decimal("3000.00"),
            volume=Decimal("2.0"),
            exchange="binance",
            bid_price=Decimal("2999.50"),
            ask_price=Decimal("3000.50"),
        ),
    ]


@pytest.fixture
def sample_validation_result():
    """Sample validation result for testing."""
    return MarketDataValidationResult(
        symbol="BTCUSDT",
        is_valid=True,
        quality_score=QualityScore(
            overall=0.85,
            completeness=0.9,
            accuracy=0.8,
            timeliness=0.9,
            consistency=0.8,
        ),
        issues=[],
        validation_timestamp=datetime.now(timezone.utc),
        metadata={"test": "value"},
    )


class TestValidationPipelineConfig:
    """Test validation pipeline configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = ValidationPipelineConfig()
        
        assert config.enable_schema_validation is True
        assert config.enable_business_validation is True
        assert config.min_quality_score == 0.7
        assert config.quarantine_quality_threshold == 0.5
        assert config.max_critical_issues == 0
        assert config.batch_size == 100

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = ValidationPipelineConfig(
            min_quality_score=0.8,
            max_critical_issues=1,
            batch_size=50,
        )
        
        assert config.min_quality_score == 0.8
        assert config.max_critical_issues == 1
        assert config.batch_size == 50

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid quality score
        with pytest.raises(ValueError):
            ValidationPipelineConfig(min_quality_score=1.5)
        
        with pytest.raises(ValueError):
            ValidationPipelineConfig(min_quality_score=-0.1)

    def test_field_constraints(self):
        """Test field constraints."""
        # Test batch size constraints
        with pytest.raises(ValueError):
            ValidationPipelineConfig(batch_size=0)
        
        with pytest.raises(ValueError):
            ValidationPipelineConfig(batch_size=20000)


class TestValidationDisposition:
    """Test validation disposition."""

    def test_disposition_creation(self):
        """Test creating validation disposition."""
        disposition = ValidationDisposition(
            action=ValidationAction.ACCEPT,
            quality_score=0.85,
            critical_issues=0,
            error_issues=0,
            warning_issues=2,
            reasons=["Minor warnings found"],
        )
        
        assert disposition.action == ValidationAction.ACCEPT
        assert disposition.quality_score == 0.85
        assert disposition.critical_issues == 0
        assert disposition.warning_issues == 2
        assert "Minor warnings found" in disposition.reasons

    def test_disposition_validation(self):
        """Test disposition field validation."""
        # Test invalid quality score
        with pytest.raises(ValueError):
            ValidationDisposition(
                action=ValidationAction.ACCEPT,
                quality_score=1.5,
                critical_issues=0,
                error_issues=0,
                warning_issues=0,
            )


class TestValidationMetrics:
    """Test validation metrics."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ValidationMetrics()
        
        assert metrics.total_records_processed == 0
        assert metrics.successful_records == 0
        assert metrics.records_quarantined == 0
        assert metrics.records_rejected == 0
        assert metrics.average_quality_score == 0.0

    def test_metrics_with_values(self):
        """Test metrics with custom values."""
        metrics = ValidationMetrics(
            total_records_processed=100,
            successful_records=85,
            records_quarantined=10,
            records_rejected=5,
            average_quality_score=0.82,
        )
        
        assert metrics.total_records_processed == 100
        assert metrics.successful_records == 85
        assert metrics.records_quarantined == 10
        assert metrics.records_rejected == 5
        assert metrics.average_quality_score == 0.82


class TestDataValidationPipeline:
    """Test DataValidationPipeline class."""

    def test_initialization(self, mock_config, mock_validator):
        """Test pipeline initialization."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        assert pipeline.config == mock_config
        assert pipeline.validator == mock_validator
        assert pipeline.pipeline_config.min_quality_score == 0.7
        assert not pipeline._initialized

    def test_initialization_without_validator(self, mock_config):
        """Test pipeline initialization without injected validator."""
        with patch('src.data.pipeline.validation_pipeline.DataValidator') as mock_validator_cls:
            mock_validator_instance = Mock()
            mock_validator_cls.return_value = mock_validator_instance
            
            pipeline = DataValidationPipeline(mock_config)
            
            assert pipeline.validator == mock_validator_instance
            mock_validator_cls.assert_called_once_with(mock_config)

    @pytest.mark.asyncio
    async def test_initialize(self, mock_config, mock_validator):
        """Test pipeline initialization."""
        with patch('src.data.pipeline.validation_pipeline.DataPipeline') as mock_data_pipeline_cls:
            mock_pipeline_instance = Mock()
            mock_data_pipeline_cls.return_value = mock_pipeline_instance
            
            pipeline = DataValidationPipeline(mock_config, mock_validator)
            
            await pipeline.initialize()
            
            assert pipeline._initialized is True
            assert pipeline.data_pipeline == mock_pipeline_instance
            mock_data_pipeline_cls.assert_called_once_with(mock_config)

    @pytest.mark.asyncio
    async def test_initialize_twice(self, mock_config, mock_validator):
        """Test initializing pipeline twice."""
        with patch('src.data.pipeline.validation_pipeline.DataPipeline'):
            pipeline = DataValidationPipeline(mock_config, mock_validator)
            
            await pipeline.initialize()
            assert pipeline._initialized is True
            
            # Second initialization should not cause issues
            await pipeline.initialize()
            assert pipeline._initialized is True

    @pytest.mark.asyncio
    async def test_validate_batch_empty_data(self, mock_config, mock_validator):
        """Test validating empty data batch."""
        with patch('src.data.pipeline.data_pipeline.MetricsCollector') as mock_metrics_collector:
            mock_metrics_collector.return_value = Mock()
            pipeline = DataValidationPipeline(mock_config, mock_validator)
            
            result = await pipeline.validate_batch([])
            
            assert isinstance(result, ValidationPipelineResult)
            assert result.total_records == 0
            assert len(result.dispositions) == 0
            assert result.metrics.total_records_processed == 0

    @pytest.mark.asyncio
    async def test_validate_batch_success(self, mock_config, mock_validator, sample_market_data, sample_validation_result):
        """Test successful batch validation."""
        # Mock validator to return successful results
        mock_validator.validate_market_data.return_value = [sample_validation_result]
        
        with patch('src.data.pipeline.validation_pipeline.DataPipeline'):
            pipeline = DataValidationPipeline(mock_config, mock_validator)
            
            result = await pipeline.validate_batch(sample_market_data[:1])  # Single symbol
            
            assert isinstance(result, ValidationPipelineResult)
            assert result.total_records == 1
            assert len(result.dispositions) == 1
            assert "BTCUSDT" in result.dispositions
            
            disposition = result.dispositions["BTCUSDT"]
            assert disposition.action == ValidationAction.ACCEPT
            assert disposition.quality_score == 0.85

    @pytest.mark.asyncio
    async def test_validate_batch_with_symbols_filter(self, mock_config, mock_validator, sample_market_data, sample_validation_result):
        """Test batch validation with symbol filtering."""
        mock_validator.validate_market_data.return_value = [sample_validation_result]
        
        with patch('src.data.pipeline.validation_pipeline.DataPipeline'):
            pipeline = DataValidationPipeline(mock_config, mock_validator)
            
            # Filter to only BTCUSDT
            result = await pipeline.validate_batch(sample_market_data, ["BTCUSDT"])
            
            assert result.total_records == 1
            assert "BTCUSDT" in result.dispositions
            assert "ETHUSDT" not in result.dispositions

    @pytest.mark.asyncio
    async def test_validate_batch_validation_error(self, mock_config, mock_validator, sample_market_data):
        """Test batch validation with validation error."""
        # Mock validator to raise exception
        mock_validator.validate_market_data.side_effect = Exception("Validation failed")
        
        with patch('src.data.pipeline.validation_pipeline.DataPipeline'):
            pipeline = DataValidationPipeline(mock_config, mock_validator)
            
            result = await pipeline.validate_batch(sample_market_data[:1])
            
            assert result.total_records == 1
            assert "BTCUSDT" in result.dispositions
            
            disposition = result.dispositions["BTCUSDT"]
            assert disposition.action == ValidationAction.REJECT
            assert disposition.critical_issues == 1
            assert "Validation error: Validation failed" in disposition.reasons

    def test_group_data_by_symbol(self, mock_config, mock_validator, sample_market_data):
        """Test grouping data by symbol."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        groups = pipeline._group_data_by_symbol(sample_market_data)
        
        assert "BTCUSDT" in groups
        assert "ETHUSDT" in groups
        assert len(groups["BTCUSDT"]) == 1
        assert len(groups["ETHUSDT"]) == 1

    @pytest.mark.asyncio
    async def test_determine_symbol_disposition_empty_results(self, mock_config, mock_validator):
        """Test symbol disposition with empty validation results."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        disposition = await pipeline._determine_symbol_disposition("BTCUSDT", [])
        
        assert disposition.action == ValidationAction.REJECT
        assert disposition.quality_score == 0.0
        assert disposition.critical_issues == 1
        assert "No validation results" in disposition.reasons

    @pytest.mark.asyncio
    async def test_determine_symbol_disposition_success(self, mock_config, mock_validator, sample_validation_result):
        """Test successful symbol disposition."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        disposition = await pipeline._determine_symbol_disposition("BTCUSDT", [sample_validation_result])
        
        assert disposition.action == ValidationAction.ACCEPT
        assert disposition.quality_score == 0.85
        assert disposition.critical_issues == 0

    @pytest.mark.asyncio
    async def test_determine_symbol_disposition_with_issues(self, mock_config, mock_validator):
        """Test symbol disposition with validation issues."""
        # Create validation result with issues
        validation_result = MarketDataValidationResult(
            symbol="BTCUSDT",
            is_valid=False,
            quality_score=QualityScore(
                overall=0.4,  # Below quarantine threshold
                completeness=0.5,
                accuracy=0.3,
                timeliness=0.5,
                consistency=0.4,
            ),
            issues=[
                ValidationIssue(
                    category=ValidationCategory.BUSINESS,
                    severity=ValidationSeverity.CRITICAL,
                    dimension=QualityDimension.VALIDITY,
                    message="Price is negative",
                    field_name="price",
                )
            ],
            validation_timestamp=datetime.now(timezone.utc),
        )
        
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        disposition = await pipeline._determine_symbol_disposition("BTCUSDT", [validation_result])
        
        assert disposition.action == ValidationAction.REJECT
        assert disposition.quality_score == 0.4
        assert disposition.critical_issues == 1

    def test_determine_action_accept(self, mock_config, mock_validator):
        """Test action determination for accept case."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        action = pipeline._determine_action(
            quality_score=0.85,
            critical_issues=0,
            error_issues=0,
            warning_issues=0,
            valid_records=1,
            total_records=1,
        )
        
        assert action == ValidationAction.ACCEPT

    def test_determine_action_accept_with_warning(self, mock_config, mock_validator):
        """Test action determination for accept with warning case."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        action = pipeline._determine_action(
            quality_score=0.85,
            critical_issues=0,
            error_issues=1,
            warning_issues=2,
            valid_records=1,
            total_records=1,
        )
        
        assert action == ValidationAction.ACCEPT_WITH_WARNING

    def test_determine_action_quarantine(self, mock_config, mock_validator):
        """Test action determination for quarantine case."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        action = pipeline._determine_action(
            quality_score=0.6,  # Below min but above quarantine threshold
            critical_issues=0,
            error_issues=0,
            warning_issues=0,
            valid_records=1,
            total_records=1,
        )
        
        assert action == ValidationAction.QUARANTINE

    def test_determine_action_reject_critical_issues(self, mock_config, mock_validator):
        """Test action determination for reject case with critical issues."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        action = pipeline._determine_action(
            quality_score=0.85,
            critical_issues=1,  # Above max critical issues (0)
            error_issues=0,
            warning_issues=0,
            valid_records=1,
            total_records=1,
        )
        
        assert action == ValidationAction.REJECT

    def test_determine_action_reject_too_many_errors(self, mock_config, mock_validator):
        """Test action determination for reject case with too many errors."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        action = pipeline._determine_action(
            quality_score=0.85,
            critical_issues=0,
            error_issues=5,  # Above max error issues (3)
            warning_issues=0,
            valid_records=1,
            total_records=1,
        )
        
        assert action == ValidationAction.REJECT

    def test_determine_action_reject_low_quality(self, mock_config, mock_validator):
        """Test action determination for reject case with low quality score."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        action = pipeline._determine_action(
            quality_score=0.3,  # Below quarantine threshold
            critical_issues=0,
            error_issues=0,
            warning_issues=0,
            valid_records=1,
            total_records=1,
        )
        
        assert action == ValidationAction.REJECT

    def test_calculate_pipeline_metrics(self, mock_config, mock_validator):
        """Test pipeline metrics calculation."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        dispositions = {
            "BTCUSDT": ValidationDisposition(
                action=ValidationAction.ACCEPT,
                quality_score=0.85,
                critical_issues=0,
                error_issues=0,
                warning_issues=1,
                metadata={"total_records": 1},
            ),
            "ETHUSDT": ValidationDisposition(
                action=ValidationAction.QUARANTINE,
                quality_score=0.6,
                critical_issues=0,
                error_issues=2,
                warning_issues=0,
                metadata={"total_records": 1},
            ),
        }
        
        sample_data = [Mock()] * 2  # Mock 2 data items
        
        metrics = pipeline._calculate_pipeline_metrics(dispositions, sample_data)
        
        assert metrics.total_records_processed == 2
        assert metrics.successful_records == 1
        assert metrics.records_quarantined == 1
        assert metrics.records_rejected == 0
        assert metrics.error_issues == 2
        assert metrics.warning_issues == 1
        assert metrics.average_quality_score == 0.725  # (0.85 + 0.6) / 2

    def test_update_session_metrics(self, mock_config, mock_validator):
        """Test session metrics update."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        pipeline_metrics = ValidationMetrics(
            total_records_processed=10,
            successful_records=8,
            records_quarantined=1,
            records_rejected=1,
            critical_issues=0,
            error_issues=2,
            warning_issues=5,
        )
        
        pipeline._update_session_metrics(pipeline_metrics)
        
        assert pipeline._session_metrics.total_records_processed == 10
        assert pipeline._session_metrics.successful_records == 8
        assert pipeline._session_metrics.records_quarantined == 1
        assert pipeline._session_metrics.records_rejected == 1
        assert pipeline._session_metrics.error_issues == 2
        assert pipeline._session_metrics.warning_issues == 5

    @pytest.mark.asyncio
    async def test_get_quarantined_data_empty(self, mock_config, mock_validator):
        """Test getting quarantined data when none exists."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        result = await pipeline.get_quarantined_data()
        assert result == {}
        
        result = await pipeline.get_quarantined_data("BTCUSDT")
        assert result == {"BTCUSDT": []}

    @pytest.mark.asyncio
    async def test_get_quarantined_data_with_data(self, mock_config, mock_validator, sample_market_data):
        """Test getting quarantined data when data exists."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        # Manually add quarantined data
        pipeline._quarantine_store["BTCUSDT"] = sample_market_data[:1]
        
        result = await pipeline.get_quarantined_data()
        assert "BTCUSDT" in result
        assert len(result["BTCUSDT"]) == 1
        
        result = await pipeline.get_quarantined_data("BTCUSDT")
        assert result == {"BTCUSDT": sample_market_data[:1]}

    @pytest.mark.asyncio
    async def test_retry_quarantined_data_not_found(self, mock_config, mock_validator):
        """Test retrying quarantined data that doesn't exist."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        result = await pipeline.retry_quarantined_data("BTCUSDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_retry_quarantined_data_empty(self, mock_config, mock_validator):
        """Test retrying quarantined data that is empty."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        pipeline._quarantine_store["BTCUSDT"] = []
        
        result = await pipeline.retry_quarantined_data("BTCUSDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_retry_quarantined_data_success(self, mock_config, mock_validator, sample_market_data, sample_validation_result):
        """Test successful retry of quarantined data."""
        mock_validator.validate_market_data.return_value = [sample_validation_result]
        
        with patch('src.data.pipeline.validation_pipeline.DataPipeline'):
            pipeline = DataValidationPipeline(mock_config, mock_validator)
            pipeline._quarantine_store["BTCUSDT"] = sample_market_data[:1]
            
            result = await pipeline.retry_quarantined_data("BTCUSDT")
            
            assert result is not None
            assert isinstance(result, ValidationPipelineResult)
            assert "BTCUSDT" not in pipeline._quarantine_store  # Should be removed on success

    @pytest.mark.asyncio
    async def test_get_pipeline_status(self, mock_config, mock_validator, sample_market_data):
        """Test getting pipeline status."""
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        # Add some test data
        pipeline._active_validations["test-id"] = {"stage": ValidationStage.INTAKE}
        pipeline._quarantine_store["BTCUSDT"] = sample_market_data[:1]
        
        status = await pipeline.get_pipeline_status()
        
        assert status["active_validations"] == 1
        assert status["quarantined_symbols"] == 1
        assert status["quarantined_records"] == 1
        assert "session_metrics" in status
        assert "configuration" in status

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_config, mock_validator):
        """Test health check when pipeline is healthy."""
        mock_validator.health_check.return_value = {"status": "healthy"}
        
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        health = await pipeline.health_check()
        
        assert health["status"] == "healthy"
        assert health["validator_available"] is True
        assert health["validator_status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_validator_unhealthy(self, mock_config, mock_validator):
        """Test health check when validator is unhealthy."""
        mock_validator.health_check.side_effect = Exception("Validator error")
        
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        health = await pipeline.health_check()
        
        assert health["status"] == "degraded"
        assert "unhealthy: Validator error" in health["validator_status"]

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_config, mock_validator, sample_market_data):
        """Test pipeline cleanup."""
        with patch('src.data.pipeline.validation_pipeline.DataPipeline') as mock_data_pipeline_cls:
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.cleanup = AsyncMock()
            mock_data_pipeline_cls.return_value = mock_pipeline_instance
            
            pipeline = DataValidationPipeline(mock_config, mock_validator)
            await pipeline.initialize()
            
            # Add some test data
            pipeline._active_validations["test-id"] = {"stage": ValidationStage.INTAKE}
            pipeline._quarantine_store["BTCUSDT"] = sample_market_data[:1]
            pipeline._session_metrics.total_records = 10
            
            await pipeline.cleanup()
            
            assert len(pipeline._active_validations) == 0
            assert len(pipeline._quarantine_store) == 0
            assert pipeline._session_metrics.total_records_processed == 0
            assert pipeline._initialized is False
            mock_pipeline_instance.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_validator_cleanup(self, mock_config, mock_validator):
        """Test cleanup when validator has cleanup method."""
        mock_validator.cleanup = AsyncMock()
        
        pipeline = DataValidationPipeline(mock_config, mock_validator)
        
        await pipeline.cleanup()
        
        mock_validator.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self, mock_config, mock_validator):
        """Test cleanup error handling."""
        with patch('src.data.pipeline.validation_pipeline.DataPipeline') as mock_data_pipeline_cls:
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.cleanup = AsyncMock(side_effect=Exception("Cleanup error"))
            mock_data_pipeline_cls.return_value = mock_pipeline_instance
            
            pipeline = DataValidationPipeline(mock_config, mock_validator)
            await pipeline.initialize()
            
            # Cleanup should not raise exception even if data pipeline cleanup fails
            await pipeline.cleanup()
            assert pipeline._initialized is False


class TestEnums:
    """Test enumeration classes."""

    def test_validation_stage_values(self):
        """Test ValidationStage enum values."""
        assert ValidationStage.INTAKE.value == "intake"
        assert ValidationStage.SCHEMA_VALIDATION.value == "schema_validation"
        assert ValidationStage.COMPLETED.value == "completed"

    def test_validation_action_values(self):
        """Test ValidationAction enum values."""
        assert ValidationAction.ACCEPT.value == "accept"
        assert ValidationAction.QUARANTINE.value == "quarantine"
        assert ValidationAction.REJECT.value == "reject"
        assert ValidationAction.RETRY.value == "retry"


class TestValidationPipelineResult:
    """Test ValidationPipelineResult model."""

    def test_result_creation(self):
        """Test creating validation pipeline result."""
        result = ValidationPipelineResult(
            pipeline_id="test-id",
            total_records=10,
            dispositions={
                "BTCUSDT": ValidationDisposition(
                    action=ValidationAction.ACCEPT,
                    quality_score=0.85,
                    critical_issues=0,
                    error_issues=0,
                    warning_issues=0,
                )
            },
            metrics=ValidationMetrics(
                total_records_processed=10,
                successful_records=10,
            ),
            execution_time_ms=1500,
        )
        
        assert result.pipeline_id == "test-id"
        assert result.total_records == 10
        assert "BTCUSDT" in result.dispositions
        assert result.metrics.total_records_processed == 10
        assert result.execution_time_ms == 1500
        assert isinstance(result.timestamp, datetime)

    def test_result_timestamp_default(self):
        """Test that result has default timestamp."""
        result = ValidationPipelineResult(
            pipeline_id="test-id",
            total_records=0,
            dispositions={},
            metrics=ValidationMetrics(),
            execution_time_ms=0,
        )
        
        assert isinstance(result.timestamp, datetime)
        assert result.timestamp.tzinfo is not None  # Should be timezone aware