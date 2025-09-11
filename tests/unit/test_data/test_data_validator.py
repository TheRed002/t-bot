"""Test suite for data validator components."""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.core.config import Config
from src.core.types import MarketData
from src.data.validation.data_validator import (
    DataValidator,
    MarketDataValidationResult,
    _get_utc_now,
)
from src.utils.validation.validation_types import (
    QualityDimension,
    QualityScore,
    ValidationCategory,
    ValidationIssue,
    ValidationSeverity,
)


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_get_utc_now(self):
        """Test getting current UTC datetime."""
        result = _get_utc_now()

        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc


class TestValidationIssue:
    """Test suite for ValidationIssue."""

    def test_initialization_minimal(self):
        """Test minimal initialization."""
        from src.core.types import ValidationLevel
        issue = ValidationIssue(
            field="test_field",
            value="test_value",
            expected="expected_value",
            message="Test message",
            level=ValidationLevel.HIGH,
            category=ValidationCategory.SCHEMA,
        )

        assert issue.category == ValidationCategory.SCHEMA
        assert issue.level == ValidationLevel.HIGH
        assert issue.message == "Test message"
        assert issue.field == "test_field"
        assert issue.value == "test_value"
        assert issue.expected == "expected_value"
        assert isinstance(issue.timestamp, datetime)
        assert issue.metadata == {}
        assert issue.source == "Validator"

    def test_initialization_full(self):
        """Test full initialization."""
        from src.core.types import ValidationLevel
        timestamp = datetime.now(timezone.utc)
        metadata = {"count": 5}

        issue = ValidationIssue(
            field="price",
            value=Decimal("100.50"),
            expected=Decimal("95.00"),
            message="Business rule violation",
            level=ValidationLevel.MEDIUM,
            timestamp=timestamp,
            source="test",
            metadata=metadata,
            category=ValidationCategory.BUSINESS,
        )

        assert issue.category == ValidationCategory.BUSINESS
        assert issue.level == ValidationLevel.MEDIUM
        assert issue.message == "Business rule violation"
        assert issue.field == "price"
        assert issue.value == Decimal("100.50")
        assert issue.expected == Decimal("95.00")
        assert issue.timestamp == timestamp
        assert issue.source == "test"
        assert issue.metadata == metadata


class TestQualityScore:
    """Test suite for QualityScore."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        score = QualityScore()

        assert score.overall == 0.0
        assert score.completeness == 0.0
        assert score.accuracy == 0.0
        assert score.consistency == 0.0
        assert score.timeliness == 0.0
        assert score.validity == 0.0
        assert score.uniqueness == 0.0

    def test_initialization_with_values(self):
        """Test initialization with custom values."""
        score = QualityScore(
            overall=0.85,
            completeness=0.90,
            accuracy=0.95,
            consistency=0.80,
            timeliness=0.75,
            validity=0.88,
            uniqueness=0.92,
        )

        assert score.overall == 0.85
        assert score.completeness == 0.90
        assert score.accuracy == 0.95
        assert score.consistency == 0.80
        assert score.timeliness == 0.75
        assert score.validity == 0.88
        assert score.uniqueness == 0.92

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = QualityScore(
            overall=0.85,
            completeness=0.90,
            accuracy=0.95,
            consistency=0.80,
            timeliness=0.75,
            validity=0.88,
            uniqueness=0.92,
        )

        result = score.to_dict()

        assert isinstance(result, dict)
        assert result["overall"] == 0.85
        assert result["completeness"] == 0.90
        assert result["accuracy"] == 0.95
        assert result["consistency"] == 0.80
        assert result["timeliness"] == 0.75
        assert result["validity"] == 0.88
        assert result["uniqueness"] == 0.92



class TestMarketDataValidationResult:
    """Test suite for MarketDataValidationResult."""

    def test_initialization_minimal(self):
        """Test minimal initialization."""
        result = MarketDataValidationResult(
            symbol="BTCUSDT", is_valid=True
        )

        assert result.symbol == "BTCUSDT"
        assert result.is_valid is True
        assert result.quality_score == 0.0
        assert result.error_count == 0
        assert result.errors == []
        assert result.metadata == {}
        assert isinstance(result.validation_timestamp, datetime)

    def test_initialization_with_errors(self):
        """Test initialization with validation errors."""
        result = MarketDataValidationResult(
            symbol="ETHUSD",
            is_valid=False,
            quality_score=0.75,
            error_count=2,
            errors=["Missing field", "Invalid price"],
            metadata={"source": "test"},
        )

        assert result.symbol == "ETHUSD"
        assert result.is_valid is False
        assert result.quality_score == 0.75
        assert result.error_count == 2
        assert len(result.errors) == 2
        assert "Missing field" in result.errors
        assert "Invalid price" in result.errors
        assert result.metadata["source"] == "test"


class TestDataValidator:
    """Test suite for DataValidator."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock(spec=Config)
        config.data_validator = {
            "strict_validation": True,
            "quality_threshold": 0.8,
            "outlier_detection": True,
            "temporal_validation": True,
        }
        return config

    @pytest.fixture
    def validator(self, mock_config):
        """Create data validator."""
        return DataValidator(config=mock_config)

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="BTCUSDT",
            price=Decimal("45000.12"),
            high_price=Decimal("46000.00"),
            low_price=Decimal("44000.00"),
            open_price=Decimal("45500.00"),
            bid=Decimal("44999.50"),
            ask=Decimal("45000.50"),
            volume=Decimal("1000.5"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

    def test_initialization(self, mock_config):
        """Test validator initialization."""
        validator = DataValidator(config=mock_config)

        assert validator.config is mock_config
        assert hasattr(validator, "logger")

    def test_initialization_with_default_config(self, mock_config):
        """Test validator initialization with default config values."""
        # Remove data_validator config to test defaults
        mock_config.data_validator = None

        validator = DataValidator(config=mock_config)

        assert validator.config is mock_config


class TestEnums:
    """Test suite for validation enums."""

    def test_validation_severity_values(self):
        """Test validation severity enum values."""
        assert ValidationSeverity.LOW.value == "low"
        assert ValidationSeverity.MEDIUM.value == "medium"
        assert ValidationSeverity.HIGH.value == "high"
        assert ValidationSeverity.CRITICAL.value == "critical"

    def test_validation_category_values(self):
        """Test validation category enum values."""
        assert ValidationCategory.SCHEMA.value == "schema"
        assert ValidationCategory.BUSINESS.value == "business"
        assert ValidationCategory.STATISTICAL.value == "statistical"
        assert ValidationCategory.TEMPORAL.value == "temporal"
        assert ValidationCategory.REGULATORY.value == "regulatory"
        assert ValidationCategory.INTEGRITY.value == "integrity"

    def test_quality_dimension_values(self):
        """Test quality dimension enum values."""
        assert QualityDimension.COMPLETENESS.value == "completeness"
        assert QualityDimension.ACCURACY.value == "accuracy"
        assert QualityDimension.CONSISTENCY.value == "consistency"
        assert QualityDimension.TIMELINESS.value == "timeliness"
        assert QualityDimension.VALIDITY.value == "validity"
        assert QualityDimension.UNIQUENESS.value == "uniqueness"
