"""
Unit tests for formatters module.

This module tests the formatting utilities in src.utils.formatters module.
"""

import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import ValidationError

# Import the functions to test
from src.utils.formatters import (
    export_to_file,
    # API response formatting
    format_api_response,
    format_chart_data,
    format_correlation_id,
    # Export formatting
    format_csv_data,
    # Financial formatting
    format_currency,
    format_error_response,
    format_excel_data,
    format_indicator_data,
    format_json_data,
    # Log formatting
    format_log_entry,
    # Chart data formatting
    format_ohlcv_data,
    format_paginated_response,
    format_percentage,
    format_performance_log,
    # Report formatting
    format_performance_report,
    format_pnl,
    format_price,
    format_quantity,
    format_risk_report,
    format_structured_log,
    format_success_response,
    format_trade_report,
)


class TestFinancialFormatting:
    """Test financial formatting functions."""

    def test_format_currency_usd(self):
        """Test currency formatting for USD."""
        amount = Decimal("1234.56")

        result = format_currency(amount, "USD")

        assert "1,234.56" in result and "USD" in result

    def test_format_currency_eur(self):
        """Test currency formatting for EUR."""
        amount = Decimal("1234.56")

        result = format_currency(amount, "EUR")

        assert "1,234.56" in result and "EUR" in result

    def test_format_currency_negative(self):
        """Test currency formatting for negative amount."""
        amount = Decimal("-1234.56")

        result = format_currency(amount, "USD")

        assert "-1,234.56" in result and "USD" in result

    def test_format_currency_zero(self):
        """Test currency formatting for zero."""
        amount = Decimal("0")

        result = format_currency(amount, "USD")

        assert "0" in result and "USD" in result

    def test_format_currency_large_number(self):
        """Test currency formatting for large number."""
        amount = Decimal("1234567.89")

        result = format_currency(amount, "USD")

        assert "1,234,567.89" in result and "USD" in result

    def test_format_currency_invalid_currency(self):
        """Test currency formatting with invalid currency."""
        amount = Decimal("1234.56")

        # Should not raise an error, just use default precision
        result = format_currency(amount, "INVALID")

        assert "INVALID" in result

    def test_format_percentage_positive(self):
        """Test percentage formatting for positive value."""
        value = Decimal("0.1234")

        result = format_percentage(value)

        assert "+12.34%" in result or "12.34%" in result

    def test_format_percentage_negative(self):
        """Test percentage formatting for negative value."""
        value = Decimal("-0.1234")

        result = format_percentage(value)

        assert "-12.34%" in result

    def test_format_percentage_zero(self):
        """Test percentage formatting for zero."""
        value = Decimal("0")

        result = format_percentage(value)

        assert "0.00%" in result

    def test_format_percentage_custom_precision(self):
        """Test percentage formatting with custom precision."""
        value = Decimal("0.123456")

        result = format_percentage(value, precision=3)

        assert "12.346%" in result

    def test_format_percentage_invalid_value(self):
        """Test percentage formatting with invalid value."""
        with pytest.raises(
            ValidationError, match="Value must be Decimal or int for financial precision"
        ):
            format_percentage("invalid")

    def test_format_pnl_positive(self):
        """Test P&L formatting for positive value."""
        pnl = Decimal("1234.56")

        result = format_pnl(pnl)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "1,234.56" in result[0] or "1234.56" in result[0]
        assert result[1] == "green"

    def test_format_pnl_negative(self):
        """Test P&L formatting for negative value."""
        pnl = Decimal("-1234.56")

        result = format_pnl(pnl)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "-1,234.56" in result[0] or "-1234.56" in result[0]
        assert result[1] == "red"

    def test_format_pnl_zero(self):
        """Test P&L formatting for zero."""
        pnl = Decimal("0")

        result = format_pnl(pnl)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "0.00" in result[0] or "0" in result[0]
        assert result[1] == "neutral"

    def test_format_pnl_with_currency(self):
        """Test P&L formatting with custom currency."""
        pnl = Decimal("1234.56")

        result = format_pnl(pnl, "EUR")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "1,234.56" in result[0] or "1234.56" in result[0]
        assert "EUR" in result[0] or result[1] == "green"

    def test_format_quantity_standard(self):
        """Test quantity formatting for standard symbol."""
        quantity = Decimal("1.5")
        symbol = "BTCUSDT"

        result = format_quantity(quantity, symbol)

        assert isinstance(result, str)
        assert "1.5" in result

    def test_format_quantity_small(self):
        """Test quantity formatting for small quantity."""
        quantity = Decimal("0.0001")
        symbol = "BTCUSDT"

        result = format_quantity(quantity, symbol)

        assert isinstance(result, str)
        assert "0.0001" in result

    def test_format_quantity_large(self):
        """Test quantity formatting for large quantity."""
        quantity = Decimal("1000000")
        symbol = "BTCUSDT"

        result = format_quantity(quantity, symbol)

        assert isinstance(result, str)
        assert "1,000,000" in result

    def test_format_quantity_invalid_symbol(self):
        """Test quantity formatting with invalid symbol."""
        quantity = Decimal("1.5")
        symbol = ""  # Empty symbol should not raise error

        result = format_quantity(quantity, symbol)

        assert isinstance(result, str)
        assert "1.5" in result

    def test_format_price_standard(self):
        """Test price formatting for standard symbol."""
        price = Decimal("50000.0")
        symbol = "BTCUSDT"

        result = format_price(price, symbol)

        assert isinstance(result, str)
        assert "50,000" in result

    def test_format_price_crypto(self):
        """Test price formatting for crypto symbol."""
        price = Decimal("0.00012345")
        symbol = "BTCUSDT"

        result = format_price(price, symbol)

        assert isinstance(result, str)
        assert "0.00012345" in result

    def test_format_price_invalid_symbol(self):
        """Test price formatting with invalid symbol."""
        price = Decimal("50000.0")
        symbol = ""  # Empty symbol should not raise error

        result = format_price(price, symbol)

        assert isinstance(result, str)
        assert "50,000" in result

    def test_format_price_negative(self):
        """Test price formatting for negative price."""
        price = Decimal("-50000.0")
        symbol = "BTCUSDT"

        # Negative price should not raise error
        result = format_price(price, symbol)

        assert isinstance(result, str)
        assert "-50,000" in result


class TestAPIResponseFormatting:
    """Test API response formatting functions."""

    def test_format_api_response_success(self):
        """Test API response formatting for success."""
        data = {"key": "value"}

        result = format_api_response(data, success=True)

        assert isinstance(result, dict)
        assert "data" in result
        assert result["data"] == data

    def test_format_api_response_error(self):
        """Test API response formatting for error."""
        data = {"error": "message"}

        result = format_api_response(data, success=False)

        assert isinstance(result, dict)
        assert "data" in result
        assert result["data"] == data

    def test_format_error_response(self):
        """Test error response formatting."""
        error = Exception("Test error")

        result = format_error_response(error)

        assert isinstance(result, dict)
        assert "error" in result
        assert "message" in result["error"]
        assert result["error"]["message"] == "Test error"

    def test_format_error_response_with_code(self):
        """Test error response formatting with error code."""
        error = Exception("Test error")
        error_code = "TEST_ERROR"

        result = format_error_response(error, error_code)

        assert isinstance(result, dict)
        assert "error" in result
        assert "code" in result["error"]
        assert result["error"]["code"] == error_code

    def test_format_success_response(self):
        """Test success response formatting."""
        data = {"key": "value"}

        result = format_success_response(data)

        assert isinstance(result, dict)
        assert "data" in result
        assert result["data"] == data

    def test_format_success_response_with_message(self):
        """Test success response formatting with custom message."""
        data = {"key": "value"}
        message = "Custom success message"

        result = format_success_response(data, message)

        assert isinstance(result, dict)
        assert "data" in result
        assert "message" in result
        assert result["message"] == message

    def test_format_paginated_response(self):
        """Test paginated response formatting."""
        data = [{"id": 1}, {"id": 2}]
        page = 1
        page_size = 10
        total = 100

        result = format_paginated_response(data, page, page_size, total)

        assert isinstance(result, dict)
        assert "data" in result
        assert "pagination" in result
        assert result["data"] == data
        assert result["pagination"]["page"] == page
        assert result["pagination"]["page_size"] == page_size
        assert result["pagination"]["total"] == total

    def test_format_paginated_response_empty(self):
        """Test paginated response formatting with empty data."""
        data = []
        page = 1
        page_size = 10
        total = 0

        result = format_paginated_response(data, page, page_size, total)

        assert isinstance(result, dict)
        assert "data" in result
        assert "pagination" in result
        assert result["data"] == data
        assert result["pagination"]["total"] == total


class TestLogFormatting:
    """Test log formatting functions."""

    def test_format_log_entry_basic(self):
        """Test basic log entry formatting."""
        level = "INFO"
        message = "Test message"

        result = format_log_entry(level, message)

        assert isinstance(result, dict)
        assert result["level"] == level
        assert result["message"] == message

    def test_format_log_entry_with_context(self):
        """Test log entry formatting with additional context."""
        level = "INFO"
        message = "Test message"
        context = {"user_id": 123, "action": "login"}

        result = format_log_entry(level, message, **context)

        assert isinstance(result, dict)
        assert result["level"] == level
        assert result["message"] == message
        assert result["user_id"] == 123
        assert result["action"] == "login"

    def test_format_correlation_id(self):
        """Test correlation ID formatting."""
        correlation_id = "test-correlation-id"

        result = format_correlation_id(correlation_id)

        assert isinstance(result, str)
        assert correlation_id in result

    def test_format_structured_log(self):
        """Test structured log formatting."""
        level = "INFO"
        message = "Test message"
        correlation_id = "test-correlation-id"

        result = format_structured_log(level, message, correlation_id)

        assert isinstance(result, str)
        assert level in result
        assert message in result
        assert correlation_id in result

    def test_format_performance_log(self):
        """Test performance log formatting."""
        function_name = "test_function"
        execution_time_ms = 100.5
        success = True

        result = format_performance_log(function_name, execution_time_ms, success)

        assert isinstance(result, dict)
        assert result["function"] == function_name
        assert result["execution_time_ms"] == execution_time_ms
        assert result["success"] == success

    def test_format_performance_log_slow_operation(self):
        """Test performance log formatting for slow operation."""
        function_name = "slow_function"
        execution_time_ms = 5000.0
        success = True

        result = format_performance_log(function_name, execution_time_ms, success)

        assert isinstance(result, dict)
        assert result["function"] == function_name
        assert result["execution_time_ms"] == execution_time_ms
        assert result["success"] == success


class TestChartDataFormatting:
    """Test chart data formatting functions."""

    def test_format_ohlcv_data(self):
        """Test OHLCV data formatting."""
        ohlcv_data = [
            {
                "timestamp": 1641600000,
                "open": Decimal("50000.0"),
                "high": Decimal("51000.0"),
                "low": Decimal("49000.0"),
                "close": Decimal("50500.0"),
                "volume": Decimal("1000.0"),
            }
        ]

        result = format_ohlcv_data(ohlcv_data)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "timestamp" in result[0]
        assert "open" in result[0]
        assert "high" in result[0]
        assert "low" in result[0]
        assert "close" in result[0]
        assert "volume" in result[0]

    def test_format_ohlcv_data_empty(self):
        """Test OHLCV data formatting with empty data."""
        ohlcv_data = []

        result = format_ohlcv_data(ohlcv_data)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_format_ohlcv_data_invalid(self):
        """Test OHLCV data formatting with invalid data."""
        ohlcv_data = "invalid"

        with pytest.raises(ValidationError, match="OHLCV data must be a list"):
            format_ohlcv_data(ohlcv_data)

    def test_format_indicator_data(self):
        """Test indicator data formatting."""
        indicator_name = "SMA"
        values = [Decimal("50000.0"), Decimal("50100.0"), Decimal("50200.0")]
        timestamps = [
            datetime(2022, 1, 8, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2022, 1, 8, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2022, 1, 8, 2, 0, 0, tzinfo=timezone.utc),
        ]

        result = format_indicator_data(indicator_name, values, timestamps)

        assert isinstance(result, dict)
        assert result["indicator"] == indicator_name
        # Values are converted to strings in the formatter for precision
        assert "timestamps" in result

    def test_format_indicator_data_no_timestamps(self):
        """Test indicator data formatting without timestamps."""
        indicator_name = "SMA"
        values = [Decimal("50000.0"), Decimal("50100.0"), Decimal("50200.0")]

        result = format_indicator_data(indicator_name, values)

        assert isinstance(result, dict)
        assert result["indicator"] == indicator_name
        # Values are converted to strings in the formatter for precision
        assert "values" in result

    def test_format_chart_data(self):
        """Test chart data formatting."""
        symbol = "BTCUSDT"
        ohlcv_data = [
            {
                "timestamp": 1641600000,
                "open": Decimal("50000.0"),
                "high": Decimal("51000.0"),
                "low": Decimal("49000.0"),
                "close": Decimal("50500.0"),
                "volume": Decimal("1000.0"),
            }
        ]
        indicators = {"SMA": [Decimal("50000.0"), Decimal("50100.0"), Decimal("50200.0")]}

        result = format_chart_data(symbol, ohlcv_data, indicators)

        assert isinstance(result, dict)
        assert result["symbol"] == symbol
        assert "ohlcv" in result
        assert "indicators" in result


class TestReportFormatting:
    """Test report formatting functions."""

    def test_format_performance_report(self):
        """Test performance report formatting."""
        performance_data = {
            "total_return": Decimal("0.15"),
            "sharpe_ratio": Decimal("1.2"),
            "max_drawdown": Decimal("-0.05"),
            "volatility": Decimal("0.12"),
        }

        result = format_performance_report(performance_data)

        assert isinstance(result, dict)
        assert result["report_type"] == "performance"
        assert "summary" in result
        assert "details" in result
        assert "timestamp" in result

    def test_format_performance_report_missing_data(self):
        """Test performance report formatting with missing data."""
        performance_data = {}

        result = format_performance_report(performance_data)

        assert isinstance(result, dict)
        assert result["report_type"] == "performance"
        assert "summary" in result
        assert "details" in result
        assert "timestamp" in result

    def test_format_risk_report(self):
        """Test risk report formatting."""
        risk_data = {
            "var_95": Decimal("-0.02"),
            "var_99": Decimal("-0.03"),
            "max_drawdown": Decimal("-0.05"),
            "volatility": Decimal("0.12"),
        }

        result = format_risk_report(risk_data)

        assert isinstance(result, dict)
        assert result["report_type"] == "risk"
        assert "summary" in result
        assert "details" in result
        assert "timestamp" in result

    def test_format_risk_report_missing_data(self):
        """Test risk report formatting with missing data."""
        risk_data = {}

        result = format_risk_report(risk_data)

        assert isinstance(result, dict)
        assert result["report_type"] == "risk"
        assert "summary" in result
        assert "details" in result
        assert "timestamp" in result

    def test_format_trade_report(self):
        """Test trade report formatting."""
        trades = [
            {
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": Decimal("1.0"),
                "price": Decimal("50000.0"),
                "timestamp": datetime.now(),
            }
        ]

        result = format_trade_report(trades)

        assert isinstance(result, dict)
        assert "trades" in result
        assert "summary" in result

    def test_format_trade_report_empty(self):
        """Test trade report formatting with empty trades."""
        trades = []

        result = format_trade_report(trades)

        assert isinstance(result, dict)
        assert "trades" in result
        assert "summary" in result


class TestExportFormatting:
    """Test export formatting functions."""

    def test_format_csv_data(self):
        """Test CSV data formatting."""
        data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
        headers = ["name", "age"]

        result = format_csv_data(data, headers)

        assert isinstance(result, str)
        assert "name,age" in result
        assert "John,30" in result
        assert "Jane,25" in result

    def test_format_csv_data_no_headers(self):
        """Test CSV data formatting without headers."""
        data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]

        result = format_csv_data(data)

        assert isinstance(result, str)
        assert "name,age" in result

    def test_format_excel_data(self):
        """Test Excel data formatting."""
        data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]

        result = format_excel_data(data, "Test Sheet")

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_format_json_data(self):
        """Test JSON data formatting."""
        data = {"key": "value", "number": 123}

        result = format_json_data(data, pretty=True)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == data

    def test_format_json_data_not_pretty(self):
        """Test JSON data formatting without pretty print."""
        data = {"key": "value", "number": 123}

        result = format_json_data(data, pretty=False)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == data

    def test_export_to_file_json(self):
        """Test export to file with JSON format."""
        data = {"key": "value"}

        with patch("builtins.open", MagicMock()) as mock_file:
            export_to_file(data, "test.json", "json")
            mock_file.assert_called_once()

    def test_export_to_file_csv(self):
        """Test export to file with CSV format."""
        data = [{"name": "John", "age": 30}]

        with patch("builtins.open", MagicMock()) as mock_file:
            export_to_file(data, "test.csv", "csv")
            mock_file.assert_called_once()

    def test_export_to_file_invalid_format(self):
        """Test export to file with invalid format."""
        data = {"key": "value"}

        with pytest.raises(ValidationError, match="Unsupported export format"):
            export_to_file(data, "test.txt", "invalid")


class TestFormatterFunctionsIntegration:
    """Test integration between formatter functions."""

    def test_financial_formatting_integration(self):
        """Test integration between financial formatting functions."""
        amount = Decimal("1234.56")
        percentage = Decimal("0.15")

        # Test currency and percentage formatting
        currency_result = format_currency(amount, "USD")
        percentage_result = format_percentage(percentage)

        assert "1,234.56" in currency_result and "USD" in currency_result
        assert "15.00%" in percentage_result

    def test_api_response_formatting_integration(self):
        """Test integration between API response formatting functions."""
        data = {"key": "value"}
        error = Exception("Test error")

        # Test success and error response formatting
        success_response = format_success_response(data)
        error_response = format_error_response(error)

        assert isinstance(success_response, dict)
        assert isinstance(error_response, dict)
        assert "data" in success_response
        assert "error" in error_response

    def test_log_formatting_integration(self):
        """Test integration between log formatting functions."""
        level = "INFO"
        message = "Test message"
        correlation_id = "test-correlation-id"

        # Test log entry and structured log formatting
        log_entry = format_log_entry(level, message)
        structured_log = format_structured_log(level, message, correlation_id)

        assert isinstance(log_entry, dict)
        assert isinstance(structured_log, str)
        assert log_entry["level"] == level
        assert level in structured_log

    def test_chart_data_formatting_integration(self):
        """Test integration between chart data formatting functions."""
        symbol = "BTCUSDT"
        ohlcv_data = [
            {
                "timestamp": 1641600000,
                "open": Decimal("50000.0"),
                "high": Decimal("51000.0"),
                "low": Decimal("49000.0"),
                "close": Decimal("50500.0"),
                "volume": Decimal("1000.0"),
            }
        ]
        indicators = {"SMA": [Decimal("50000.0"), Decimal("50100.0"), Decimal("50200.0")]}

        # Test OHLCV and indicator data formatting
        formatted_ohlcv = format_ohlcv_data(ohlcv_data)
        formatted_indicators = format_indicator_data("SMA", indicators["SMA"])
        chart_data = format_chart_data(symbol, ohlcv_data, indicators)

        assert isinstance(formatted_ohlcv, list)
        assert isinstance(formatted_indicators, dict)
        assert isinstance(chart_data, dict)
        assert chart_data["symbol"] == symbol

    def test_report_formatting_integration(self):
        """Test integration between report formatting functions."""
        performance_data = {"total_return": Decimal("0.15"), "sharpe_ratio": Decimal("1.2")}
        risk_data = {"var_95": Decimal("-0.02"), "max_drawdown": Decimal("-0.05")}
        trades = [
            {
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": Decimal("1.0"),
                "price": Decimal("50000.0"),
            }
        ]

        # Test performance, risk, and trade report formatting
        performance_report = format_performance_report(performance_data)
        risk_report = format_risk_report(risk_data)
        trade_report = format_trade_report(trades)

        assert isinstance(performance_report, dict)
        assert isinstance(risk_report, dict)
        assert isinstance(trade_report, dict)
        assert performance_report["report_type"] == "performance"
        assert risk_report["report_type"] == "risk"
        assert "trades" in trade_report
