"""
Tests for Pipeline Utilities.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.utils.pipeline_utilities import (
    PipelineStage,
    ProcessingMode,
    DataQuality,
    PipelineAction,
    PipelineMetrics,
    PipelineRecord,
    PipelineStageProcessor,
    PipelineUtils
)


class TestPipelineEnums:
    """Test pipeline enumeration classes."""

    def test_pipeline_stage_enum(self):
        """Test PipelineStage enum values."""
        assert PipelineStage.INGESTION.value == "ingestion"
        assert PipelineStage.VALIDATION.value == "validation"
        assert PipelineStage.CLEANSING.value == "cleansing"
        assert PipelineStage.TRANSFORMATION.value == "transformation"
        assert PipelineStage.ENRICHMENT.value == "enrichment"
        assert PipelineStage.QUALITY_CHECK.value == "quality_check"
        assert PipelineStage.STORAGE.value == "storage"
        assert PipelineStage.INDEXING.value == "indexing"
        assert PipelineStage.NOTIFICATION.value == "notification"
        assert PipelineStage.COMPLETED.value == "completed"

    def test_processing_mode_enum(self):
        """Test ProcessingMode enum values."""
        assert ProcessingMode.REAL_TIME.value == "real_time"
        assert ProcessingMode.BATCH.value == "batch"
        assert ProcessingMode.STREAM.value == "stream"
        assert ProcessingMode.HYBRID.value == "hybrid"

    def test_data_quality_enum(self):
        """Test DataQuality enum values."""
        assert DataQuality.EXCELLENT.value == "excellent"
        assert DataQuality.GOOD.value == "good"
        assert DataQuality.ACCEPTABLE.value == "acceptable"
        assert DataQuality.POOR.value == "poor"
        assert DataQuality.UNACCEPTABLE.value == "unacceptable"

    def test_pipeline_action_enum(self):
        """Test PipelineAction enum values."""
        assert PipelineAction.ACCEPT.value == "accept"
        assert PipelineAction.ACCEPT_WITH_WARNING.value == "accept_with_warning"
        assert PipelineAction.QUARANTINE.value == "quarantine"
        assert PipelineAction.REJECT.value == "reject"
        assert PipelineAction.RETRY.value == "retry"
        assert PipelineAction.SKIP.value == "skip"


class TestPipelineMetrics:
    """Test PipelineMetrics functionality."""

    def test_pipeline_metrics_initialization(self):
        """Test PipelineMetrics initialization with defaults."""
        metrics = PipelineMetrics()
        
        assert metrics.total_records_processed == 0
        assert metrics.successful_records == 0
        assert metrics.failed_records == 0
        assert metrics.records_rejected == 0
        assert metrics.records_quarantined == 0
        assert metrics.avg_processing_time_ms == 0.0
        assert metrics.throughput_per_second == 0.0
        assert metrics.data_quality_score == 0.0
        assert metrics.pipeline_uptime == 0.0
        assert isinstance(metrics.last_processed_time, datetime)
        assert metrics.processing_stages_completed == 0
        assert metrics.critical_issues == 0
        assert metrics.error_issues == 0
        assert metrics.warning_issues == 0
        assert metrics.average_quality_score == 0.0

    def test_pipeline_metrics_with_values(self):
        """Test PipelineMetrics initialization with custom values."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        metrics = PipelineMetrics(
            total_records_processed=100,
            successful_records=95,
            failed_records=5,
            last_processed_time=custom_time
        )
        
        assert metrics.total_records_processed == 100
        assert metrics.successful_records == 95
        assert metrics.failed_records == 5
        assert metrics.last_processed_time == custom_time

    def test_calculate_success_rate_with_records(self):
        """Test success rate calculation with processed records."""
        metrics = PipelineMetrics(
            total_records_processed=100,
            successful_records=85
        )
        
        success_rate = metrics.calculate_success_rate()
        assert success_rate == 0.85

    def test_calculate_success_rate_no_records(self):
        """Test success rate calculation with no records."""
        metrics = PipelineMetrics()
        
        success_rate = metrics.calculate_success_rate()
        assert success_rate == 0.0

    def test_calculate_failure_rate_with_records(self):
        """Test failure rate calculation with processed records."""
        metrics = PipelineMetrics(
            total_records_processed=100,
            failed_records=15
        )
        
        failure_rate = metrics.calculate_failure_rate()
        assert failure_rate == 0.15

    def test_calculate_failure_rate_no_records(self):
        """Test failure rate calculation with no records."""
        metrics = PipelineMetrics()
        
        failure_rate = metrics.calculate_failure_rate()
        assert failure_rate == 0.0

    def test_update_processing_time_first_record(self):
        """Test updating processing time for first record."""
        metrics = PipelineMetrics()
        
        metrics.update_processing_time(100.0)
        
        assert metrics.avg_processing_time_ms == 100.0

    def test_update_processing_time_multiple_records(self):
        """Test updating processing time with multiple records."""
        metrics = PipelineMetrics(total_records_processed=1)
        metrics.avg_processing_time_ms = 100.0
        
        metrics.update_processing_time(200.0)
        
        # Should be weighted average
        expected = (1 - 1/2) * 100.0 + (1/2) * 200.0
        assert metrics.avg_processing_time_ms == expected

    def test_calculate_throughput_positive_processing_time(self):
        """Test throughput calculation with positive processing time."""
        metrics = PipelineMetrics()
        metrics.avg_processing_time_ms = 50.0  # 50ms per record
        
        throughput = metrics.calculate_throughput()
        
        expected_throughput = (1.0 / 50.0) * 1000  # records per second
        assert throughput == expected_throughput
        assert metrics.throughput_per_second == expected_throughput

    def test_calculate_throughput_zero_processing_time(self):
        """Test throughput calculation with zero processing time."""
        metrics = PipelineMetrics()
        metrics.avg_processing_time_ms = 0.0
        
        throughput = metrics.calculate_throughput()
        
        assert throughput == 0.0

    def test_calculate_throughput_negative_time_window(self):
        """Test throughput calculation with negative time window."""
        metrics = PipelineMetrics()
        metrics.avg_processing_time_ms = 50.0
        
        throughput = metrics.calculate_throughput(time_window_seconds=-1.0)
        
        assert throughput == 0.0

    def test_calculate_throughput_zero_time_window(self):
        """Test throughput calculation with zero time window."""
        metrics = PipelineMetrics()
        metrics.avg_processing_time_ms = 50.0
        
        throughput = metrics.calculate_throughput(time_window_seconds=0.0)
        
        assert throughput == 0.0


class TestPipelineRecord:
    """Test PipelineRecord functionality."""

    def test_pipeline_record_initialization(self):
        """Test PipelineRecord initialization."""
        timestamp = datetime.now(timezone.utc)
        record = PipelineRecord(
            id="test_id",
            data={"key": "value"},
            stage=PipelineStage.INGESTION,
            timestamp=timestamp
        )
        
        assert record.id == "test_id"
        assert record.data == {"key": "value"}
        assert record.stage == PipelineStage.INGESTION
        assert record.timestamp == timestamp
        assert record.processing_time_ms == 0.0
        assert record.metadata == {}
        assert record.errors == []
        assert record.warnings == []
        assert record.quality_score == 0.0

    def test_add_error_without_stage(self):
        """Test adding error without stage."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        
        record.add_error("Test error")
        
        assert len(record.errors) == 1
        assert record.errors[0] == "Test error"

    def test_add_error_with_stage(self):
        """Test adding error with stage."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        
        record.add_error("Test error", PipelineStage.VALIDATION)
        
        assert len(record.errors) == 1
        assert record.errors[0] == "[validation] Test error"

    def test_add_warning_without_stage(self):
        """Test adding warning without stage."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        
        record.add_warning("Test warning")
        
        assert len(record.warnings) == 1
        assert record.warnings[0] == "Test warning"

    def test_add_warning_with_stage(self):
        """Test adding warning with stage."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        
        record.add_warning("Test warning", PipelineStage.VALIDATION)
        
        assert len(record.warnings) == 1
        assert record.warnings[0] == "[validation] Test warning"

    def test_has_errors_true(self):
        """Test has_errors when errors exist."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        record.add_error("Test error")
        
        assert record.has_errors() is True

    def test_has_errors_false(self):
        """Test has_errors when no errors exist."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert record.has_errors() is False

    def test_has_warnings_true(self):
        """Test has_warnings when warnings exist."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        record.add_warning("Test warning")
        
        assert record.has_warnings() is True

    def test_has_warnings_false(self):
        """Test has_warnings when no warnings exist."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert record.has_warnings() is False

    def test_multiple_errors_and_warnings(self):
        """Test adding multiple errors and warnings."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        
        record.add_error("Error 1")
        record.add_error("Error 2", PipelineStage.VALIDATION)
        record.add_warning("Warning 1")
        record.add_warning("Warning 2", PipelineStage.TRANSFORMATION)
        
        assert len(record.errors) == 2
        assert len(record.warnings) == 2
        assert record.has_errors() is True
        assert record.has_warnings() is True


class TestPipelineStageProcessor:
    """Test PipelineStageProcessor protocol."""

    def test_pipeline_stage_processor_protocol(self):
        """Test that PipelineStageProcessor is a protocol."""
        # Create a mock implementation
        class MockProcessor:
            async def process(self, record: PipelineRecord) -> PipelineRecord:
                return record
            
            def get_stage_name(self) -> str:
                return "mock_stage"
        
        processor = MockProcessor()
        
        # Should have the expected methods
        assert hasattr(processor, 'process')
        assert hasattr(processor, 'get_stage_name')
        assert processor.get_stage_name() == "mock_stage"


class TestPipelineUtils:
    """Test PipelineUtils functionality."""

    def test_validate_pipeline_config_valid(self):
        """Test pipeline config validation with valid config."""
        config = {
            "batch_size": 100,
            "max_workers": 4,
            "timeout_seconds": 30.0
        }
        
        errors = PipelineUtils.validate_pipeline_config(config)
        
        assert errors == []

    def test_validate_pipeline_config_missing_fields(self):
        """Test pipeline config validation with missing fields."""
        config = {
            "batch_size": 100
            # Missing max_workers and timeout_seconds
        }
        
        errors = PipelineUtils.validate_pipeline_config(config)
        
        assert len(errors) == 2
        assert "Missing required field: max_workers" in errors
        assert "Missing required field: timeout_seconds" in errors

    def test_validate_pipeline_config_invalid_batch_size(self):
        """Test pipeline config validation with invalid batch size."""
        config = {
            "batch_size": -10,  # Invalid
            "max_workers": 4,
            "timeout_seconds": 30.0
        }
        
        errors = PipelineUtils.validate_pipeline_config(config)
        
        assert "batch_size must be positive integer" in errors

    def test_validate_pipeline_config_non_integer_batch_size(self):
        """Test pipeline config validation with non-integer batch size."""
        config = {
            "batch_size": "100",  # String instead of int
            "max_workers": 4,
            "timeout_seconds": 30.0
        }
        
        errors = PipelineUtils.validate_pipeline_config(config)
        
        assert "batch_size must be positive integer" in errors

    def test_validate_pipeline_config_invalid_max_workers(self):
        """Test pipeline config validation with invalid max workers."""
        config = {
            "batch_size": 100,
            "max_workers": 0,  # Invalid
            "timeout_seconds": 30.0
        }
        
        errors = PipelineUtils.validate_pipeline_config(config)
        
        assert "max_workers must be positive integer" in errors

    def test_validate_pipeline_config_invalid_timeout(self):
        """Test pipeline config validation with invalid timeout."""
        config = {
            "batch_size": 100,
            "max_workers": 4,
            "timeout_seconds": -5.0  # Invalid
        }
        
        errors = PipelineUtils.validate_pipeline_config(config)
        
        assert "timeout_seconds must be positive number" in errors

    def test_validate_pipeline_config_string_timeout(self):
        """Test pipeline config validation with string timeout."""
        config = {
            "batch_size": 100,
            "max_workers": 4,
            "timeout_seconds": "30"  # String instead of number
        }
        
        errors = PipelineUtils.validate_pipeline_config(config)
        
        assert "timeout_seconds must be positive number" in errors

    def test_calculate_data_quality_score_no_issues(self):
        """Test data quality score calculation with no issues."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        
        score = PipelineUtils.calculate_data_quality_score(record)
        
        assert score == 1.0

    def test_calculate_data_quality_score_with_errors(self):
        """Test data quality score calculation with errors."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        record.add_error("Error 1")
        record.add_error("Error 2")
        
        score = PipelineUtils.calculate_data_quality_score(record)
        
        expected_score = 1.0 - (2 * 0.5)  # 2 errors * default error weight
        assert score == expected_score

    def test_calculate_data_quality_score_with_warnings(self):
        """Test data quality score calculation with warnings."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        record.add_warning("Warning 1")
        record.add_warning("Warning 2")
        record.add_warning("Warning 3")
        
        score = PipelineUtils.calculate_data_quality_score(record)
        
        expected_score = 1.0 - (3 * 0.2)  # 3 warnings * default warning weight
        assert score == expected_score

    def test_calculate_data_quality_score_mixed_issues(self):
        """Test data quality score calculation with mixed issues."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        record.add_error("Error 1")
        record.add_warning("Warning 1")
        
        score = PipelineUtils.calculate_data_quality_score(record, error_weight=0.6, warning_weight=0.1)
        
        expected_score = 1.0 - (1 * 0.6) - (1 * 0.1)  # 1 error + 1 warning
        assert score == expected_score

    def test_calculate_data_quality_score_minimum_zero(self):
        """Test data quality score minimum is zero."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        # Add many errors to push score below 0
        for i in range(10):
            record.add_error(f"Error {i}")
        
        score = PipelineUtils.calculate_data_quality_score(record)
        
        assert score == 0.0

    def test_determine_pipeline_action_accept(self):
        """Test pipeline action determination for accept."""
        action = PipelineUtils.determine_pipeline_action(
            quality_score=0.8,
            error_count=0,
            warning_count=2,
            min_quality_threshold=0.7,
            max_warnings=10
        )
        
        assert action == PipelineAction.ACCEPT

    def test_determine_pipeline_action_accept_with_warning(self):
        """Test pipeline action determination for accept with warning."""
        action = PipelineUtils.determine_pipeline_action(
            quality_score=0.6,  # Below threshold
            error_count=0,
            warning_count=5,
            min_quality_threshold=0.7
        )
        
        assert action == PipelineAction.ACCEPT_WITH_WARNING

    def test_determine_pipeline_action_accept_with_warning_too_many_warnings(self):
        """Test pipeline action determination for accept with warning due to too many warnings."""
        action = PipelineUtils.determine_pipeline_action(
            quality_score=0.8,
            error_count=0,
            warning_count=15,  # Too many warnings
            max_warnings=10
        )
        
        assert action == PipelineAction.ACCEPT_WITH_WARNING

    def test_determine_pipeline_action_quarantine(self):
        """Test pipeline action determination for quarantine."""
        action = PipelineUtils.determine_pipeline_action(
            quality_score=0.4,  # Below quarantine threshold
            error_count=0,
            warning_count=5,
            quarantine_threshold=0.5
        )
        
        assert action == PipelineAction.QUARANTINE

    def test_determine_pipeline_action_reject(self):
        """Test pipeline action determination for reject."""
        action = PipelineUtils.determine_pipeline_action(
            quality_score=0.8,
            error_count=2,  # Too many errors
            warning_count=0,
            max_errors=0
        )
        
        assert action == PipelineAction.REJECT

    def test_create_processing_summary(self):
        """Test creating processing summary."""
        metrics = PipelineMetrics(
            total_records_processed=100,
            successful_records=85,
            failed_records=10,
            records_rejected=3,
            records_quarantined=2,
            avg_processing_time_ms=50.0,
            data_quality_score=0.85,
            critical_issues=1,
            error_issues=5,
            warning_issues=12
        )
        
        summary = PipelineUtils.create_processing_summary(metrics, 60.0)
        
        assert summary["total_records"] == 100
        assert summary["successful_records"] == 85
        assert summary["failed_records"] == 10
        assert summary["rejected_records"] == 3
        assert summary["quarantined_records"] == 2
        assert summary["success_rate"] == 0.85
        assert summary["failure_rate"] == 0.1
        assert summary["avg_processing_time_ms"] == 50.0
        assert summary["data_quality_score"] == 0.85
        assert summary["processing_duration_seconds"] == 60.0
        assert summary["critical_issues"] == 1
        assert summary["error_issues"] == 5
        assert summary["warning_issues"] == 12

    def test_log_pipeline_summary_success(self):
        """Test logging pipeline summary with successful processing."""
        mock_logger = MagicMock()
        summary = {
            "total_records": 100,
            "successful_records": 95,
            "failed_records": 5,
            "success_rate": 0.95,
            "avg_processing_time_ms": 25.5,
            "throughput_per_second": 40.0,
            "critical_issues": 0,
            "error_issues": 0,
            "warning_issues": 2
        }
        
        PipelineUtils.log_pipeline_summary("Test", summary, mock_logger)
        
        mock_logger.info.assert_called_once()
        info_call = mock_logger.info.call_args[0][0]
        assert "Test Pipeline Summary" in info_call
        assert "Processed 100 records" in info_call
        assert "95 successful" in info_call

    def test_log_pipeline_summary_with_issues(self):
        """Test logging pipeline summary with issues."""
        mock_logger = MagicMock()
        summary = {
            "total_records": 100,
            "successful_records": 80,
            "failed_records": 20,
            "success_rate": 0.8,
            "avg_processing_time_ms": 35.0,
            "throughput_per_second": 30.0,
            "critical_issues": 2,
            "error_issues": 5,
            "warning_issues": 8
        }
        
        PipelineUtils.log_pipeline_summary("Test", summary, mock_logger)
        
        mock_logger.info.assert_called_once()
        mock_logger.warning.assert_called_once()
        
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Test Pipeline Issues" in warning_call
        assert "2 critical" in warning_call
        assert "5 errors" in warning_call

    def test_log_pipeline_summary_default_logger(self):
        """Test logging pipeline summary with default logger."""
        summary = {
            "total_records": 50,
            "successful_records": 50,
            "failed_records": 0,
            "success_rate": 1.0,
            "avg_processing_time_ms": 10.0,
            "throughput_per_second": 100.0,
            "critical_issues": 0,
            "error_issues": 0,
            "warning_issues": 0
        }
        
        # Should not raise exception when using default logger
        PipelineUtils.log_pipeline_summary("Test", summary)


class TestPipelineUtilsEdgeCases:
    """Test edge cases for pipeline utilities."""

    def test_validate_pipeline_config_empty(self):
        """Test pipeline config validation with empty config."""
        config = {}
        
        errors = PipelineUtils.validate_pipeline_config(config)
        
        assert len(errors) == 3  # All required fields missing

    def test_calculate_data_quality_score_negative_weights(self):
        """Test data quality score calculation with negative weights."""
        record = PipelineRecord(
            id="test",
            data={},
            stage=PipelineStage.INGESTION,
            timestamp=datetime.now(timezone.utc)
        )
        record.add_error("Error")
        
        # Negative weights should still work (though unusual)
        score = PipelineUtils.calculate_data_quality_score(record, error_weight=-0.1)
        
        assert score == 1.1  # Base score + abs(negative weight)

    def test_determine_pipeline_action_edge_thresholds(self):
        """Test pipeline action determination with edge case thresholds."""
        # Quality score exactly at threshold
        action = PipelineUtils.determine_pipeline_action(
            quality_score=0.7,
            error_count=0,
            warning_count=0,
            min_quality_threshold=0.7
        )
        
        assert action == PipelineAction.ACCEPT

    def test_pipeline_metrics_edge_cases(self):
        """Test pipeline metrics with edge case values."""
        metrics = PipelineMetrics(
            total_records_processed=1,
            successful_records=0,
            failed_records=1
        )
        
        assert metrics.calculate_success_rate() == 0.0
        assert metrics.calculate_failure_rate() == 1.0