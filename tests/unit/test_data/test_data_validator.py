"""Test suite for data validator components."""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from src.core.config import Config
from src.core.types import MarketData
from src.data.validation.data_validator import (
    DataValidator,
    MarketDataValidationResult,
    QualityDimension,
    QualityScore,
    ValidationCategory,
    ValidationIssue,
    ValidationRule,
    ValidationSeverity,
    _get_utc_now,
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
        issue = ValidationIssue(
            category=ValidationCategory.SCHEMA,
            severity=ValidationSeverity.ERROR,
            dimension=QualityDimension.COMPLETENESS,
            message="Test message"
        )
        
        assert issue.category == ValidationCategory.SCHEMA
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.dimension == QualityDimension.COMPLETENESS
        assert issue.message == "Test message"
        assert issue.field_name is None
        assert issue.value is None
        assert issue.expected is None
        assert issue.rule_name == ""
        assert isinstance(issue.timestamp, datetime)
        assert issue.metadata == {}

    def test_initialization_full(self):
        """Test full initialization."""
        timestamp = datetime.now(timezone.utc)
        metadata = {"source": "test", "count": 5}
        
        issue = ValidationIssue(
            category=ValidationCategory.BUSINESS,
            severity=ValidationSeverity.WARNING,
            dimension=QualityDimension.ACCURACY,
            message="Business rule violation",
            field_name="price",
            value=Decimal("100.50"),
            expected=Decimal("95.00"),
            rule_name="price_range_check",
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert issue.category == ValidationCategory.BUSINESS
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.dimension == QualityDimension.ACCURACY
        assert issue.message == "Business rule violation"
        assert issue.field_name == "price"
        assert issue.value == Decimal("100.50")
        assert issue.expected == Decimal("95.00")
        assert issue.rule_name == "price_range_check"
        assert issue.timestamp == timestamp
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
            uniqueness=0.92
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
            uniqueness=0.92
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


class TestValidationRule:
    """Test suite for ValidationRule."""

    def test_initialization_minimal(self):
        """Test minimal initialization."""
        rule = ValidationRule(
            name="test_rule",
            category=ValidationCategory.SCHEMA,
            severity=ValidationSeverity.ERROR,
            dimension=QualityDimension.COMPLETENESS,
            description="Test validation rule"
        )
        
        assert rule.name == "test_rule"
        assert rule.category == ValidationCategory.SCHEMA
        assert rule.severity == ValidationSeverity.ERROR
        assert rule.dimension == QualityDimension.COMPLETENESS
        assert rule.description == "Test validation rule"
        assert rule.enabled is True
        assert rule.parameters == {}

    def test_initialization_full(self):
        """Test full initialization."""
        parameters = {"min_value": 0, "max_value": 100}
        
        rule = ValidationRule(
            name="price_range_rule",
            category=ValidationCategory.BUSINESS,
            severity=ValidationSeverity.WARNING,
            dimension=QualityDimension.ACCURACY,
            description="Price must be within range",
            enabled=False,
            parameters=parameters
        )
        
        assert rule.name == "price_range_rule"
        assert rule.category == ValidationCategory.BUSINESS
        assert rule.severity == ValidationSeverity.WARNING
        assert rule.dimension == QualityDimension.ACCURACY
        assert rule.description == "Price must be within range"
        assert rule.enabled is False
        assert rule.parameters == parameters

    def test_validation_name_length(self):
        """Test name validation."""
        # Test minimum length
        with pytest.raises(ValueError):
            ValidationRule(
                name="",  # Too short
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                dimension=QualityDimension.COMPLETENESS,
                description="Test rule"
            )

        # Test maximum length
        with pytest.raises(ValueError):
            ValidationRule(
                name="a" * 101,  # Too long
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                dimension=QualityDimension.COMPLETENESS,
                description="Test rule"
            )

    def test_validation_description_length(self):
        """Test description validation."""
        with pytest.raises(ValueError):
            ValidationRule(
                name="test_rule",
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                dimension=QualityDimension.COMPLETENESS,
                description=""  # Too short
            )


class TestMarketDataValidationResult:
    """Test suite for MarketDataValidationResult."""

    def test_initialization_minimal(self):
        """Test minimal initialization."""
        result = MarketDataValidationResult(
            symbol="BTCUSDT",
            is_valid=True,
            quality_score=QualityScore(overall=0.95)
        )
        
        assert result.symbol == "BTCUSDT"
        assert result.is_valid is True
        assert result.quality_score.overall == 0.95
        assert result.issues == []
        assert result.metadata == {}
        assert isinstance(result.validation_timestamp, datetime)

    def test_initialization_with_issues(self):
        """Test initialization with validation issues."""
        issue = ValidationIssue(
            category=ValidationCategory.SCHEMA,
            severity=ValidationSeverity.ERROR,
            dimension=QualityDimension.COMPLETENESS,
            message="Missing field"
        )
        
        result = MarketDataValidationResult(
            symbol="ETHUSD",
            is_valid=False,
            quality_score=QualityScore(overall=0.75),
            issues=[issue],
            metadata={"source": "test"}
        )
        
        assert result.symbol == "ETHUSD"
        assert result.is_valid is False
        assert result.quality_score.overall == 0.75
        assert len(result.issues) == 1
        assert result.issues[0] is issue
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
            "temporal_validation": True
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
            exchange="binance"
        )

    def test_initialization(self, mock_config):
        """Test validator initialization."""
        validator = DataValidator(config=mock_config)
        
        assert validator.config is mock_config
        assert hasattr(validator, 'logger')

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
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"
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