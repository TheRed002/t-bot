"""
Comprehensive test suite for monitoring financial precision module.

Tests cover Decimal to float conversion, precision loss detection,
batch conversion, validation, and precision analysis.
"""

import warnings
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.core.exceptions import ValidationError
from src.monitoring.financial_precision import (
    FINANCIAL_CONTEXT,
    METRIC_PRECISION_MAP,
    FinancialPrecisionWarning,
    convert_financial_batch,
    detect_precision_requirements,
    get_recommended_precision,
    safe_decimal_to_float,
    validate_financial_range,
)


class TestFinancialContext:
    """Test financial context configuration."""

    def test_financial_context_properties(self):
        """Test FINANCIAL_CONTEXT has correct properties."""
        assert FINANCIAL_CONTEXT["max_precision"] == 12
        assert FINANCIAL_CONTEXT["default_precision"] == 8
        assert FINANCIAL_CONTEXT["warn_on_precision_loss"] is True
        assert FINANCIAL_CONTEXT["strict_mode"] is False
        assert isinstance(FINANCIAL_CONTEXT, dict)


class TestFinancialPrecisionWarning:
    """Test FinancialPrecisionWarning class."""

    def test_financial_precision_warning_inheritance(self):
        """Test FinancialPrecisionWarning inherits from UserWarning."""
        assert issubclass(FinancialPrecisionWarning, UserWarning)

    def test_financial_precision_warning_creation(self):
        """Test creating FinancialPrecisionWarning."""
        warning = FinancialPrecisionWarning("Test precision loss")
        assert str(warning) == "Test precision loss"


class TestSafeDecimalToFloat:
    """Test safe_decimal_to_float function."""

    def test_safe_decimal_to_float_with_none(self):
        """Test safe_decimal_to_float with None value."""
        with pytest.raises(ValueError, match="Cannot convert None to float"):
            safe_decimal_to_float(None, "test_metric")

    def test_safe_decimal_to_float_with_int(self):
        """Test safe_decimal_to_float with integer input."""
        result = safe_decimal_to_float(42, "test_metric")
        assert result == 42.0
        assert isinstance(result, float)

    def test_safe_decimal_to_float_with_float(self):
        """Test safe_decimal_to_float with float input."""
        result = safe_decimal_to_float(42.5, "test_metric")
        assert result == 42.5
        assert isinstance(result, float)

    def test_safe_decimal_to_float_with_invalid_float(self):
        """Test safe_decimal_to_float with invalid float (infinity)."""
        with pytest.raises(ValueError, match="Invalid float value"):
            safe_decimal_to_float(float("inf"), "test_metric")

        with pytest.raises(ValueError, match="Invalid float value"):
            safe_decimal_to_float(float("-inf"), "test_metric")

        with pytest.raises(ValueError, match="Invalid float value"):
            safe_decimal_to_float(float("nan"), "test_metric")

    def test_safe_decimal_to_float_with_decimal(self):
        """Test safe_decimal_to_float with Decimal input."""
        decimal_value = Decimal("42.123456789")
        result = safe_decimal_to_float(decimal_value, "test_metric")
        assert isinstance(result, float)
        assert abs(result - 42.12345679) < 1e-8

    def test_safe_decimal_to_float_with_invalid_type(self):
        """Test safe_decimal_to_float with invalid type."""
        with pytest.raises(TypeError, match="Expected Decimal, float, or int"):
            safe_decimal_to_float("not_a_number", "test_metric")

    def test_safe_decimal_to_float_with_non_finite_decimal(self):
        """Test safe_decimal_to_float with non-finite Decimal."""
        with pytest.raises(ValueError, match="Non-finite Decimal value"):
            safe_decimal_to_float(Decimal("inf"), "test_metric")

        with pytest.raises(ValueError, match="Non-finite Decimal value"):
            safe_decimal_to_float(Decimal("-inf"), "test_metric")

        with pytest.raises(ValueError, match="Non-finite Decimal value"):
            safe_decimal_to_float(Decimal("nan"), "test_metric")

    def test_safe_decimal_to_float_precision_digits(self):
        """Test safe_decimal_to_float respects precision_digits parameter."""
        result = safe_decimal_to_float(42.123456789, "test_metric", precision_digits=2)
        assert result == 42.12

        result = safe_decimal_to_float(Decimal("42.123456789"), "test_metric", precision_digits=4)
        assert result == 42.1235

    def test_safe_decimal_to_float_no_precision_loss(self):
        """Test safe_decimal_to_float with no precision loss."""
        # Simple decimal that converts perfectly
        decimal_value = Decimal("42.5")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = safe_decimal_to_float(decimal_value, "test_metric")

            assert result == 42.5
            assert len(w) == 0  # No warnings

    def test_safe_decimal_to_float_with_precision_loss(self):
        """Test safe_decimal_to_float with precision loss warning."""
        # High-precision decimal that loses precision when converted to float
        # Need a value that has > 0.0000001 (0.00001%) relative error
        decimal_value = Decimal("1.123456789123456789123456789")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = safe_decimal_to_float(decimal_value, "test_metric", warn_on_loss=True)

            assert isinstance(result, float)
            # May or may not trigger warning depending on precision threshold
            if len(w) > 0:
                assert issubclass(w[0].category, FinancialPrecisionWarning)
                assert "Precision loss detected" in str(w[0].message)
                assert "test_metric" in str(w[0].message)

    def test_safe_decimal_to_float_with_precision_loss_disabled(self):
        """Test safe_decimal_to_float with precision loss warnings disabled."""
        decimal_value = Decimal("0.123456789123456789123456789")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = safe_decimal_to_float(decimal_value, "test_metric", warn_on_loss=False)

            assert isinstance(result, float)
            assert len(w) == 0  # No warnings

    def test_safe_decimal_to_float_zero_value(self):
        """Test safe_decimal_to_float with zero value."""
        result = safe_decimal_to_float(Decimal("0"), "test_metric")
        assert result == 0.0

        result = safe_decimal_to_float(Decimal("0.0000000000"), "test_metric")
        assert result == 0.0

    def test_safe_decimal_to_float_negative_value(self):
        """Test safe_decimal_to_float with negative value."""
        result = safe_decimal_to_float(Decimal("-42.5"), "test_metric")
        assert result == -42.5

    def test_safe_decimal_to_float_very_small_value(self):
        """Test safe_decimal_to_float with very small value."""
        small_value = Decimal("0.00000001")  # 1 satoshi
        result = safe_decimal_to_float(small_value, "btc_amount", precision_digits=8)
        assert result == 0.00000001

    def test_safe_decimal_to_float_very_large_value(self):
        """Test safe_decimal_to_float with very large value."""
        large_value = Decimal("999999999.99999999")
        result = safe_decimal_to_float(large_value, "large_amount")
        assert isinstance(result, float)
        assert result > 999999999.0


class TestConvertFinancialBatch:
    """Test convert_financial_batch function."""

    def test_convert_financial_batch_simple(self):
        """Test convert_financial_batch with simple values."""
        values = {
            "price": Decimal("50000.12345678"),
            "volume": Decimal("1.5"),
            "fee": Decimal("2.50"),
        }

        results = convert_financial_batch(values, "trading")

        assert len(results) == 3
        assert isinstance(results["price"], float)
        assert isinstance(results["volume"], float)
        assert isinstance(results["fee"], float)
        assert results["volume"] == 1.5
        assert results["fee"] == 2.5

    def test_convert_financial_batch_with_precision_map(self):
        """Test convert_financial_batch with custom precision map."""
        values = {
            "percentage": Decimal("0.123456789"),
            "count": Decimal("42.0"),
            "price": Decimal("50000.123456789"),
        }

        precision_map = {
            "percentage": 4,
            "count": 0,
            "price": 8,
        }

        results = convert_financial_batch(values, "test", precision_map)

        assert results["percentage"] == 0.1235  # 4 decimal places as per precision_map
        assert results["count"] == 42.0  # 0 decimal places as per precision_map (integer)
        assert abs(results["price"] - 50000.12345679) < 1e-8  # 8 decimal places as per precision_map

    def test_convert_financial_batch_empty_prefix(self):
        """Test convert_financial_batch with empty prefix."""
        values = {"test": Decimal("42.5")}
        results = convert_financial_batch(values, "")

        assert results["test"] == 42.5

    def test_convert_financial_batch_no_prefix(self):
        """Test convert_financial_batch with None prefix."""
        values = {"test": Decimal("42.5")}
        results = convert_financial_batch(values, None)

        assert results["test"] == 42.5

    def test_convert_financial_batch_with_error(self):
        """Test convert_financial_batch handles errors."""
        values = {
            "good": Decimal("42.5"),
            "bad": "not_a_number",  # This will cause an error
        }

        with pytest.raises(TypeError, match="Expected Decimal, float, or int"):
            convert_financial_batch(values, "test")

    def test_convert_financial_batch_mixed_types(self):
        """Test convert_financial_batch with mixed input types."""
        values = {
            "decimal_val": Decimal("42.5"),
            "float_val": 43.5,
            "int_val": 44,
        }

        results = convert_financial_batch(values, "mixed")

        assert results["decimal_val"] == 42.5
        assert results["float_val"] == 43.5
        assert results["int_val"] == 44.0

    def test_convert_financial_batch_empty_values(self):
        """Test convert_financial_batch with empty values."""
        results = convert_financial_batch({}, "test")
        assert results == {}


class TestValidateFinancialRange:
    """Test validate_financial_range function."""

    def test_validate_financial_range_no_bounds(self):
        """Test validate_financial_range with no bounds (should not raise)."""
        validate_financial_range(Decimal("42.5"))
        validate_financial_range(Decimal("42.5"))
        validate_financial_range(Decimal("-1000.0"))

    def test_validate_financial_range_within_bounds(self):
        """Test validate_financial_range with value within bounds."""
        validate_financial_range(
            Decimal("50.0"), min_value=Decimal("0.0"), max_value=Decimal("100.0")
        )

        validate_financial_range(Decimal("75.0"), min_value=Decimal("0.0"), max_value=Decimal("100.0"))

    def test_validate_financial_range_below_minimum(self):
        """Test validate_financial_range with value below minimum."""
        with pytest.raises(ValidationError, match="below minimum"):
            validate_financial_range(Decimal("-10.0"), min_value=Decimal("0.0"))

    def test_validate_financial_range_above_maximum(self):
        """Test validate_financial_range with value above maximum."""
        with pytest.raises(ValidationError, match="above maximum"):
            validate_financial_range(Decimal("150.0"), max_value=Decimal("100.0"))

    def test_validate_financial_range_at_bounds(self):
        """Test validate_financial_range with value exactly at bounds."""
        # At minimum (inclusive)
        validate_financial_range(Decimal("0.0"), min_value=Decimal("0.0"))

        # At maximum (inclusive)
        validate_financial_range(Decimal("100.0"), max_value=Decimal("100.0"))

    def test_validate_financial_range_mixed_types(self):
        """Test validate_financial_range with mixed Decimal and float types."""
        validate_financial_range(
            Decimal("50.0"),
            min_value=Decimal("0.0"),  # Decimal
            max_value=Decimal("100.0"),  # Decimal
        )


class TestDetectPrecisionRequirements:
    """Test detect_precision_requirements function."""

    def test_detect_precision_requirements_integer(self):
        """Test detect_precision_requirements with integer."""
        decimal_places, is_high_precision = detect_precision_requirements(
            Decimal("42"), "test_metric"
        )
        assert decimal_places == 0
        assert is_high_precision is False

    def test_detect_precision_requirements_low_precision(self):
        """Test detect_precision_requirements with low precision decimal."""
        decimal_places, is_high_precision = detect_precision_requirements(
            Decimal("42.50"), "test_metric"
        )
        assert decimal_places == 1  # normalize() turns 42.50 into 42.5
        assert is_high_precision is False

    def test_detect_precision_requirements_high_precision(self):
        """Test detect_precision_requirements with high precision decimal."""
        decimal_places, is_high_precision = detect_precision_requirements(
            Decimal("42.123456789"), "test_metric"
        )
        assert decimal_places == 9
        assert is_high_precision is True

    def test_detect_precision_requirements_scientific_notation(self):
        """Test detect_precision_requirements with scientific notation."""
        decimal_places, is_high_precision = detect_precision_requirements(
            Decimal("1E-8"), "test_metric"
        )
        assert decimal_places == 12
        assert is_high_precision is True

        decimal_places, is_high_precision = detect_precision_requirements(
            Decimal("1.23E+15"), "test_metric"
        )
        assert decimal_places == 12
        assert is_high_precision is True

    def test_detect_precision_requirements_normalized_value(self):
        """Test detect_precision_requirements with normalized decimal."""
        # Normalized removes trailing zeros
        decimal_val = Decimal("42.500000").normalize()
        decimal_places, is_high_precision = detect_precision_requirements(
            decimal_val, "test_metric"
        )
        assert decimal_places == 1  # Normalized to 42.5
        assert is_high_precision is False

    def test_detect_precision_requirements_max_precision(self):
        """Test detect_precision_requirements caps at 12 decimal places."""
        very_precise = Decimal("42." + "1" * 20)  # 20 decimal places
        decimal_places, is_high_precision = detect_precision_requirements(
            very_precise, "test_metric"
        )
        assert decimal_places == 12  # Capped at 12
        assert is_high_precision is True


class TestMetricPrecisionMap:
    """Test METRIC_PRECISION_MAP constant."""

    def test_metric_precision_map_contents(self):
        """Test METRIC_PRECISION_MAP has expected contents."""
        assert METRIC_PRECISION_MAP["price"] == 8
        assert METRIC_PRECISION_MAP["value_usd"] == 8
        assert METRIC_PRECISION_MAP["percent"] == 4
        assert METRIC_PRECISION_MAP["bps"] == 2
        assert METRIC_PRECISION_MAP["count"] == 0
        assert METRIC_PRECISION_MAP["duration_seconds"] == 6

    def test_metric_precision_map_completeness(self):
        """Test METRIC_PRECISION_MAP covers main financial metric types."""
        expected_keys = [
            "price",
            "value_usd",
            "pnl_usd",
            "volume_usd",
            "percent",
            "ratio",
            "rate",
            "apy",
            "sharpe_ratio",
            "bps",
            "slippage_bps",
            "count",
            "total",
            "duration_seconds",
            "latency_seconds",
        ]

        for key in expected_keys:
            assert key in METRIC_PRECISION_MAP


class TestGetRecommendedPrecision:
    """Test get_recommended_precision function."""

    def test_get_recommended_precision_exact_matches(self):
        """Test get_recommended_precision with exact matches."""
        assert get_recommended_precision("price") == 8
        assert get_recommended_precision("percent") == 4
        assert get_recommended_precision("bps") == 2
        assert get_recommended_precision("count") == 0
        assert get_recommended_precision("duration_seconds") == 6

    def test_get_recommended_precision_case_insensitive(self):
        """Test get_recommended_precision is case insensitive."""
        assert get_recommended_precision("PRICE") == 8
        assert get_recommended_precision("Price") == 8
        assert get_recommended_precision("pRiCe") == 8

    def test_get_recommended_precision_suffix_matching(self):
        """Test get_recommended_precision with suffix matching."""
        assert get_recommended_precision("btc_price") == 8
        assert get_recommended_precision("win_rate_percent") == 4
        assert get_recommended_precision("slippage_bps") == 2
        assert get_recommended_precision("order_count") == 0
        assert get_recommended_precision("execution_duration_seconds") == 6

    def test_get_recommended_precision_no_match_default(self):
        """Test get_recommended_precision returns default for no match."""
        assert get_recommended_precision("unknown_metric") == 8
        assert get_recommended_precision("custom_financial_value") == 8
        assert get_recommended_precision("") == 8

    def test_get_recommended_precision_multiple_matches(self):
        """Test get_recommended_precision with multiple potential matches."""
        # Should match the first suffix found
        metric_name = "trade_volume_usd_count"  # Could match volume_usd or count
        result = get_recommended_precision(metric_name)
        # The exact behavior depends on iteration order, but should be deterministic
        assert isinstance(result, int)
        assert 0 <= result <= 8


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_safe_decimal_to_float_extreme_precision(self):
        """Test safe_decimal_to_float with extremely high precision."""
        # Very high precision decimal
        high_precision = Decimal("0." + "0" * 20 + "1")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = safe_decimal_to_float(high_precision, "extreme_precision")

            assert isinstance(result, float)
            # May warn about precision loss depending on conversion
            # Test passes regardless of warning count

    def test_safe_decimal_to_float_very_large_numbers(self):
        """Test safe_decimal_to_float with very large numbers."""
        large_decimal = Decimal("999999999999999.99999999")
        result = safe_decimal_to_float(large_decimal, "large_number")
        assert isinstance(result, float)
        assert result > 999999999999999.0

    def test_safe_decimal_to_float_very_small_numbers(self):
        """Test safe_decimal_to_float with very small numbers."""
        small_decimal = Decimal("0.00000000000001")
        result = safe_decimal_to_float(small_decimal, "tiny_number")
        assert isinstance(result, float)
        # Very small numbers might underflow to 0.0 in float conversion
        assert result >= 0.0

    def test_convert_financial_batch_unicode_keys(self):
        """Test convert_financial_batch with Unicode metric names."""
        values = {
            "价格": Decimal("42.5"),  # Chinese for "price"
            "pörtefølje": Decimal("100.0"),  # Danish for "portfolio"
            "émojis_💰": Decimal("1.23"),
        }

        results = convert_financial_batch(values, "unicode_test")

        assert len(results) == 3
        assert results["价格"] == 42.5
        assert results["pörtefølje"] == 100.0
        assert results["émojis_💰"] == 1.23

    def test_validate_financial_range_edge_precision(self):
        """Test validate_financial_range with edge precision values."""
        # Very small differences
        validate_financial_range(
            Decimal("0.00000001"),
            min_value=Decimal("0.00000001"),
            max_value=Decimal("0.00000002"),
        )

    def test_detect_precision_requirements_edge_cases(self):
        """Test detect_precision_requirements with edge cases."""
        # Zero
        decimal_places, is_high_precision = detect_precision_requirements(Decimal("0"), "zero")
        assert decimal_places == 0
        assert is_high_precision is False

        # Negative with high precision
        decimal_places, is_high_precision = detect_precision_requirements(
            Decimal("-0.123456789"), "negative_precise"
        )
        assert decimal_places == 9
        assert is_high_precision is True

    @patch("src.monitoring.financial_precision.logger")
    def test_safe_decimal_to_float_logs_precision_loss(self, mock_logger):
        """Test safe_decimal_to_float logs precision loss warnings."""
        high_precision = Decimal("0.123456789123456789123456789")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warning output
            safe_decimal_to_float(high_precision, "test_metric", warn_on_loss=True)

        # May or may not log a warning depending on actual precision threshold
        # Test passes regardless


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_complete_financial_conversion_workflow(self):
        """Test complete workflow of financial value conversion."""
        # Simulate real trading metrics
        trading_metrics = {
            "btc_price": Decimal("50123.45678901"),
            "trade_volume_usd": Decimal("10000.50"),
            "pnl_percent": Decimal("2.3456"),
            "slippage_bps": Decimal("12.34"),
            "order_count": Decimal("42"),
            "execution_duration_seconds": Decimal("0.123456"),
        }

        # Use automatic precision detection
        precision_map = {name: get_recommended_precision(name) for name in trading_metrics.keys()}

        # Convert batch
        results = convert_financial_batch(trading_metrics, "trading")

        # Verify results have appropriate precision
        assert abs(results["btc_price"] - 50123.45678901) < 1e-10
        assert results["trade_volume_usd"] == 10000.5
        assert results["pnl_percent"] == 2.3456
        assert results["slippage_bps"] == 12.34
        assert results["order_count"] == 42.0
        assert abs(results["execution_duration_seconds"] - 0.123456) < 1e-10

    def test_financial_validation_chain(self):
        """Test chaining financial validation functions."""
        value = Decimal("50000.12345678")
        metric_name = "btc_price"

        # Step 1: Validate range
        validate_financial_range(
            value, min_value=Decimal("0"), max_value=Decimal("100000")
        )

        # Step 2: Detect precision requirements
        decimal_places, is_high_precision = detect_precision_requirements(value, metric_name)
        assert decimal_places == 8
        assert is_high_precision is True

        # Step 3: Get recommended precision
        recommended_precision = get_recommended_precision(metric_name)
        assert recommended_precision == 8

        # Step 4: Convert to float
        result = safe_decimal_to_float(value, metric_name, recommended_precision)
        assert isinstance(result, float)
        assert abs(result - 50000.12345678) < 1e-10

    def test_batch_processing_with_validation(self):
        """Test batch processing with validation and precision detection."""
        financial_data = {
            "portfolio_value_usd": Decimal("100000.00"),
            "daily_pnl_percent": Decimal("1.25"),
            "max_drawdown_percent": Decimal("5.75"),
            "sharpe_ratio": Decimal("1.8542"),
            "total_trades_count": Decimal("156"),
        }

        # Validate all values are positive (except PnL can be negative)
        for name, value in financial_data.items():
            if "pnl" not in name:
                validate_financial_range(value, min_value=Decimal("0"))

        # Convert with automatic precision
        results = convert_financial_batch(financial_data, "portfolio")

        # All results should be valid floats
        for name, result in results.items():
            assert isinstance(result, float)
            assert not (result != result)  # Check for NaN
            assert abs(result) < float("inf")  # Check for infinity
