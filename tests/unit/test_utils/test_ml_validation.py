"""
Tests for ML Validation utilities.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.core.exceptions import DataValidationError, ValidationError
from src.utils.ml_validation import (
    validate_features,
    validate_targets,
    validate_training_data,
    validate_market_data,
    check_data_quality
)


class TestValidateFeatures:
    """Test feature validation functionality."""

    def test_validate_features_basic(self):
        """Test basic feature validation."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [0.5, 1.5, 2.5, 3.5],
            'feature3': [10, 20, 30, 40]
        })
        
        result = validate_features(X, "TestModel")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert len(result.columns) == 3
        assert not result.isnull().any().any()

    def test_validate_features_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        X = pd.DataFrame()
        
        with pytest.raises(ValidationError, match="Features cannot be empty"):
            validate_features(X, "TestModel")

    def test_validate_features_no_columns(self):
        """Test validation with no columns."""
        X = pd.DataFrame(index=[0, 1, 2])  # No columns
        
        with pytest.raises(ValidationError, match="Features cannot be empty"):
            validate_features(X, "TestModel")

    def test_validate_features_with_missing_values(self):
        """Test validation with missing values."""
        X = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0, 4.0],
            'feature2': [0.5, 1.5, np.nan, 3.5],
            'feature3': [10, 20, 30, np.nan]
        })
        
        result = validate_features(X, "TestModel")
        
        assert not result.isnull().any().any()
        assert len(result) == 4
        assert len(result.columns) == 3

    def test_validate_features_with_infinite_values(self):
        """Test validation with infinite values."""
        X = pd.DataFrame({
            'feature1': [1.0, np.inf, 3.0, 4.0],
            'feature2': [0.5, 1.5, -np.inf, 3.5],
            'feature3': [10, 20, 30, 40]
        })
        
        result = validate_features(X, "TestModel")
        
        assert not np.isinf(result).any().any()
        assert result.loc[1, 'feature1'] == 0
        assert result.loc[2, 'feature2'] == 0

    def test_validate_features_non_numeric_columns(self):
        """Test validation with non-numeric columns that can be converted."""
        X = pd.DataFrame({
            'feature1': ['1.0', '2.0', '3.0', '4.0'],  # String numbers
            'feature2': [0.5, 1.5, 2.5, 3.5],         # Already numeric
            'feature3': [10, 20, 30, 40]               # Integer
        })
        
        result = validate_features(X, "TestModel")
        
        assert pd.api.types.is_numeric_dtype(result['feature1'])
        assert len(result.columns) == 3

    def test_validate_features_non_convertible_columns(self):
        """Test validation with non-convertible columns."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': ['a', 'b', 'c', 'd'],  # Non-convertible strings
            'feature3': [10, 20, 30, 40]
        })
        
        result = validate_features(X, "TestModel")
        
        # Non-convertible column should be converted to numeric (with NaN -> 0)
        assert 'feature2' in result.columns
        assert len(result.columns) == 3
        assert result['feature2'].sum() == 0.0  # All NaN converted to 0

    def test_validate_features_conversion_exception(self):
        """Test validation with conversion causing general exception."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': ['a', 'b', 'c', 'd']  # Non-numeric to trigger to_numeric call
        })
        
        # Mock to_numeric to raise a general exception
        with patch('pandas.to_numeric', side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(DataValidationError, match="Failed to process column"):
                validate_features(X, "TestModel")

    def test_validate_features_error_propagation_failure(self):
        """Test validation when error propagation fails."""
        X = pd.DataFrame({
            'feature1': ['invalid', 'data', 'here', 'fail'],  # Non-numeric data that will cause conversion error
        })
        
        # Mock to_numeric to raise a non-ValueError/TypeError exception (which will trigger error propagation)
        with patch('pandas.to_numeric', side_effect=RuntimeError("Unexpected error")):
            with patch('src.utils.messaging_patterns.ErrorPropagationMixin') as mock_error_prop:
                mock_instance = MagicMock()
                mock_instance.propagate_validation_error.side_effect = Exception("Propagation failed")
                mock_error_prop.return_value = mock_instance
                
                with pytest.raises(DataValidationError):
                    validate_features(X, "TestModel")

    def test_validate_features_all_nan_after_conversion(self):
        """Test validation when conversion results in all NaN."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': ['a', 'b', 'c', 'd']  # Will become NaN after conversion
        })
        
        # Mock to_numeric to return Series with NaN
        with patch('pandas.to_numeric', return_value=pd.Series([np.nan, np.nan, np.nan, np.nan])):
            result = validate_features(X, "TestModel")
            
            # Should fill NaN with 0
            assert not result.isnull().any().any()


class TestValidateTargets:
    """Test target validation functionality."""

    def test_validate_targets_basic(self):
        """Test basic target validation."""
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        
        result = validate_targets(y, "TestModel")
        
        assert isinstance(result, pd.Series)
        assert len(result) == 4
        assert not result.isnull().any()

    def test_validate_targets_empty_series(self):
        """Test validation with empty Series."""
        y = pd.Series([], dtype=float)
        
        with pytest.raises(ValidationError, match="Targets cannot be empty"):
            validate_targets(y, "TestModel")

    def test_validate_targets_with_missing_values(self):
        """Test validation with missing values."""
        y = pd.Series([1.0, np.nan, 3.0, np.nan])
        
        result = validate_targets(y, "TestModel")
        
        assert not result.isnull().any()
        assert len(result) == 4

    def test_validate_targets_with_infinite_values(self):
        """Test validation with infinite values."""
        y = pd.Series([1.0, np.inf, 3.0, -np.inf])
        
        result = validate_targets(y, "TestModel")
        
        assert not np.isinf(result).any()
        assert result.iloc[1] == 0
        assert result.iloc[3] == 0

    def test_validate_targets_non_numeric(self):
        """Test validation with non-numeric targets that can be converted."""
        y = pd.Series(['1.0', '2.0', '3.0', '4.0'])
        
        result = validate_targets(y, "TestModel")
        
        assert pd.api.types.is_numeric_dtype(result)
        assert len(result) == 4

    def test_validate_targets_conversion_value_error(self):
        """Test validation with conversion ValueError."""
        y = pd.Series(['a', 'b', 'c', 'd'])
        
        with pytest.raises(DataValidationError, match="Could not convert targets to numeric"):
            validate_targets(y, "TestModel")

    def test_validate_targets_conversion_type_error(self):
        """Test validation with conversion TypeError."""
        y = pd.Series(['1', '2', '3', '4'])  # String data to trigger conversion
        
        # Mock to_numeric to raise TypeError
        with patch('pandas.to_numeric', side_effect=TypeError("Type error")):
            with pytest.raises(DataValidationError, match="Could not convert targets to numeric"):
                validate_targets(y, "TestModel")

    def test_validate_targets_general_exception(self):
        """Test validation with general exception."""
        y = pd.Series(['1', '2', '3', '4'])  # String data to trigger conversion
        
        # Mock to_numeric to raise general exception
        with patch('pandas.to_numeric', side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(DataValidationError, match="Critical error in target preprocessing"):
                validate_targets(y, "TestModel")


class TestValidateTrainingData:
    """Test training data validation functionality."""

    def test_validate_training_data_basic(self):
        """Test basic training data validation."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [0.5, 1.5, 2.5, 3.5]
        })
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        
        X_clean, y_clean = validate_training_data(X, y, "TestModel")
        
        assert isinstance(X_clean, pd.DataFrame)
        assert isinstance(y_clean, pd.Series)
        assert len(X_clean) == len(y_clean)
        assert len(X_clean) == 4

    def test_validate_training_data_length_mismatch(self):
        """Test validation with mismatched lengths."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],  # 3 rows
            'feature2': [0.5, 1.5, 2.5]
        })
        y = pd.Series([1.0, 2.0, 3.0, 4.0])  # 4 rows
        
        with pytest.raises(ValidationError, match="Feature and target data must have same length"):
            validate_training_data(X, y, "TestModel")

    def test_validate_training_data_alignment_after_cleaning(self):
        """Test data alignment after cleaning causes length mismatch."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [0.5, 1.5, 2.5, 3.5]
        })
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        
        # Mock validate_features to return shorter data
        with patch('src.utils.ml_validation.validate_features') as mock_validate_features:
            mock_validate_features.return_value = X.iloc[:3]  # Return only 3 rows
            
            with patch('src.utils.ml_validation.validate_targets') as mock_validate_targets:
                mock_validate_targets.return_value = y  # Return all 4 rows
                
                X_clean, y_clean = validate_training_data(X, y, "TestModel")
                
                # Should align to minimum length
                assert len(X_clean) == len(y_clean) == 3


class TestValidateMarketData:
    """Test market data validation functionality."""

    def test_validate_market_data_basic(self):
        """Test basic market data validation."""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0],
            'high': [105.0, 106.0, 107.0, 108.0],
            'low': [95.0, 96.0, 97.0, 98.0],
            'close': [104.0, 105.0, 106.0, 107.0],
            'volume': [1000, 1500, 2000, 1800]
        })
        
        result = validate_market_data(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_validate_market_data_empty(self):
        """Test validation with empty DataFrame."""
        data = pd.DataFrame()
        
        with pytest.raises(ValidationError, match="Market data cannot be empty"):
            validate_market_data(data)

    def test_validate_market_data_missing_required_columns(self):
        """Test validation with missing required columns."""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            # Missing 'low', 'close', 'volume'
        })
        
        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_market_data(data)

    def test_validate_market_data_custom_required_columns(self):
        """Test validation with custom required columns."""
        data = pd.DataFrame({
            'price': [100.0, 101.0, 102.0],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        
        result = validate_market_data(data, required_columns=['price', 'timestamp'])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_validate_market_data_non_positive_prices(self):
        """Test validation with non-positive prices."""
        data = pd.DataFrame({
            'open': [100.0, -101.0, 102.0, 103.0],  # Negative price
            'high': [105.0, 106.0, 107.0, 108.0],
            'low': [95.0, 96.0, 97.0, 98.0],
            'close': [104.0, 105.0, 106.0, 107.0],
            'volume': [1000, 1500, 2000, 1800]
        })
        
        with pytest.raises(ValidationError, match="Non-positive values found in open"):
            validate_market_data(data)

    def test_validate_market_data_zero_prices(self):
        """Test validation with zero prices."""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 0.0, 103.0],  # Zero price
            'high': [105.0, 106.0, 107.0, 108.0],
            'low': [95.0, 96.0, 97.0, 98.0],
            'close': [104.0, 105.0, 106.0, 107.0],
            'volume': [1000, 1500, 2000, 1800]
        })
        
        with pytest.raises(ValidationError, match="Non-positive values found in open"):
            validate_market_data(data)

    def test_validate_market_data_invalid_high_prices(self):
        """Test validation with invalid high prices."""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0],
            'high': [95.0, 106.0, 107.0, 108.0],  # High < open
            'low': [95.0, 96.0, 97.0, 98.0],
            'close': [104.0, 105.0, 106.0, 107.0],
            'volume': [1000, 1500, 2000, 1800]
        })
        
        with pytest.raises(ValidationError, match="High price is lower than open/close"):
            validate_market_data(data)

    def test_validate_market_data_invalid_low_prices(self):
        """Test validation with invalid low prices."""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0],
            'high': [105.0, 106.0, 107.0, 108.0],
            'low': [105.0, 96.0, 97.0, 98.0],  # Low > open
            'close': [104.0, 105.0, 106.0, 107.0],
            'volume': [1000, 1500, 2000, 1800]
        })
        
        with pytest.raises(ValidationError, match="Low price is higher than open/close"):
            validate_market_data(data)

    def test_validate_market_data_partial_ohlc_columns(self):
        """Test validation with only some OHLC columns."""
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0],
            'close': [104.0, 105.0, 106.0, 107.0],
            'volume': [1000, 1500, 2000, 1800]
            # Missing 'high' and 'low' - should not check OHLC relationships
        })
        
        result = validate_market_data(data, required_columns=['open', 'close', 'volume'])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4


class TestCheckDataQuality:
    """Test data quality checking functionality."""

    def test_check_data_quality_basic(self):
        """Test basic data quality check."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.5, 1.5, 2.5, 3.5, 4.5],
            'feature3': [10, 20, 30, 40, 50]
        })
        y = pd.Series([1, 0, 1, 0, 1])
        
        quality_report = check_data_quality(X, y)
        
        assert quality_report["total_samples"] == 5
        assert quality_report["total_features"] == 3
        assert quality_report["passed"] is True
        assert len(quality_report["warnings"]) == 0
        assert "missing_data" in quality_report
        assert "constant_features" in quality_report

    def test_check_data_quality_high_missing_data(self):
        """Test quality check with high missing data."""
        X = pd.DataFrame({
            'feature1': [1.0, np.nan, np.nan, np.nan, 5.0],  # 60% missing
            'feature2': [0.5, 1.5, 2.5, 3.5, 4.5],          # No missing
            'feature3': [10, 20, 30, 40, 50]                 # No missing
        })
        
        quality_report = check_data_quality(X, max_missing_pct=0.1)
        
        assert quality_report["passed"] is False
        assert any("high missing data" in warning for warning in quality_report["warnings"])

    def test_check_data_quality_constant_features(self):
        """Test quality check with constant features."""
        X = pd.DataFrame({
            'feature1': [1.0, 1.0, 1.0, 1.0, 1.0],  # Completely constant
            'feature2': [0.5, 1.5, 2.5, 3.5, 4.5],  # Variable
            'feature3': [10, 10, 10, 10, 20]         # Near constant (80%)
        })
        
        quality_report = check_data_quality(X, max_constant_pct=0.7)
        
        assert quality_report["passed"] is False
        assert any("Constant/near-constant" in warning for warning in quality_report["warnings"])
        assert "feature1" in quality_report["constant_features"]
        assert "feature3" in quality_report["constant_features"]

    def test_check_data_quality_single_unique_target(self):
        """Test quality check with single unique target value."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.5, 1.5, 2.5, 3.5, 4.5]
        })
        y = pd.Series([1, 1, 1, 1, 1])  # All same value
        
        quality_report = check_data_quality(X, y)
        
        assert quality_report["passed"] is False
        assert any("Target has only one unique value" in warning for warning in quality_report["warnings"])

    def test_check_data_quality_few_unique_targets(self):
        """Test quality check with very few unique target values."""
        # Create 100 samples where we need < 0.1 * 100 = 10 unique values to trigger warning
        X = pd.DataFrame({
            'feature1': list(range(100)),
            'feature2': [i * 0.5 for i in range(100)]
        })
        # Use only 5 unique values out of 100 samples (5%), which should trigger warning since 5 < 10
        y = pd.Series([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)  # 5 unique values
        
        quality_report = check_data_quality(X, y)
        
        # Should trigger warning since 5 < 10 (0.1 * 100)
        assert any("Target has very few unique values" in warning for warning in quality_report["warnings"])

    def test_check_data_quality_no_targets(self):
        """Test quality check without target data."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.5, 1.5, 2.5, 3.5, 4.5]
        })
        
        quality_report = check_data_quality(X)
        
        # Should not have target-related warnings
        assert not any("Target" in warning for warning in quality_report["warnings"])

    def test_check_data_quality_mixed_issues(self):
        """Test quality check with multiple issues."""
        X = pd.DataFrame({
            'feature1': [1.0, 1.0, 1.0, 1.0, 1.0],          # Constant
            'feature2': [0.5, np.nan, np.nan, np.nan, 4.5],  # High missing
            'feature3': [10, 20, 30, 40, 50]                 # Good feature
        })
        y = pd.Series([1, 1, 1, 1, 1])  # Constant target
        
        quality_report = check_data_quality(X, y, max_missing_pct=0.1)
        
        assert quality_report["passed"] is False
        assert len(quality_report["warnings"]) >= 3  # At least 3 different issues
        assert any("high missing data" in warning for warning in quality_report["warnings"])
        assert any("Constant/near-constant" in warning for warning in quality_report["warnings"])
        assert any("Target has only one unique value" in warning for warning in quality_report["warnings"])

    def test_check_data_quality_empty_dataframe(self):
        """Test quality check with empty DataFrame."""
        X = pd.DataFrame()
        
        quality_report = check_data_quality(X)
        
        assert quality_report["total_samples"] == 0
        assert quality_report["total_features"] == 0

    def test_check_data_quality_single_row(self):
        """Test quality check with single row."""
        X = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [0.5]
        })
        y = pd.Series([1])
        
        quality_report = check_data_quality(X, y)
        
        assert quality_report["total_samples"] == 1
        # Single row means all features are "constant"
        assert quality_report["passed"] is False


class TestMLValidationEdgeCases:
    """Test edge cases and error scenarios."""

    def test_validate_features_with_complex_dataframe(self):
        """Test feature validation with complex DataFrame scenarios."""
        # DataFrame with mixed types and edge cases
        X = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0],
            'string_numeric': ['1.5', '2.5', '3.5'],
            'boolean': [True, False, True],
            'datetime_string': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'mixed': [1, '2.0', 3.5]
        })
        
        result = validate_features(X, "ComplexModel")
        
        # Should handle conversions appropriately
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_validate_targets_with_boolean_series(self):
        """Test target validation with boolean Series."""
        y = pd.Series([True, False, True, False])
        
        result = validate_targets(y, "BoolModel")
        
        assert pd.api.types.is_numeric_dtype(result)
        assert len(result) == 4

    def test_validate_market_data_edge_cases(self):
        """Test market data validation edge cases."""
        # Data where high equals open/close (boundary condition)
        data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [100.0, 101.0, 102.0],  # Equals open
            'low': [100.0, 101.0, 102.0],   # Equals close
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1500, 2000]
        })
        
        # Should not raise error when high/low equal open/close
        result = validate_market_data(data)
        assert isinstance(result, pd.DataFrame)