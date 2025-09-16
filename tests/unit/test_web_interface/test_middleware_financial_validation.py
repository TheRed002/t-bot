"""
Tests for web_interface.middleware.financial_validation module.
"""

import json
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException, Request, Response

from src.core.exceptions import ValidationError
from src.web_interface.middleware.financial_validation import (
    CurrencyValidationMiddleware,
    DecimalEnforcementMiddleware,
    FinancialValidationMiddleware,
)


@pytest.fixture
def financial_middleware():
    """Create FinancialValidationMiddleware instance."""
    return FinancialValidationMiddleware(app=None)


@pytest.fixture
def decimal_middleware():
    """Create DecimalEnforcementMiddleware instance."""
    return DecimalEnforcementMiddleware(app=None)


@pytest.fixture
def currency_middleware():
    """Create CurrencyValidationMiddleware instance."""
    return CurrencyValidationMiddleware(app=None)


@pytest.fixture
def mock_request():
    """Create a mock request."""
    request = Mock(spec=Request)
    request.url = Mock()
    request.method = "POST"
    return request


@pytest.fixture
def mock_response():
    """Create a mock response."""
    response = Mock(spec=Response)
    response.headers = {}
    return response


class TestFinancialValidationMiddleware:
    """Tests for FinancialValidationMiddleware."""

    def test_is_financial_endpoint_true(self, financial_middleware):
        """Test financial endpoint detection returns True for financial endpoints."""
        assert financial_middleware._is_financial_endpoint("/api/analytics/test")
        assert financial_middleware._is_financial_endpoint("/api/capital/allocate")
        assert financial_middleware._is_financial_endpoint("/api/portfolio/balance")
        assert financial_middleware._is_financial_endpoint("/api/trading/order")
        assert financial_middleware._is_financial_endpoint("/api/risk/var")
        assert financial_middleware._is_financial_endpoint("/api/exchanges/balance")

    def test_is_financial_endpoint_false(self, financial_middleware):
        """Test financial endpoint detection returns False for non-financial endpoints."""
        assert not financial_middleware._is_financial_endpoint("/api/health")
        assert not financial_middleware._is_financial_endpoint("/api/auth/login")
        assert not financial_middleware._is_financial_endpoint("/docs")
        assert not financial_middleware._is_financial_endpoint("/api/bot/status")

    async def test_dispatch_non_financial_endpoint(self, financial_middleware, mock_request):
        """Test dispatch skips validation for non-financial endpoints."""
        mock_request.url.path = "/api/health"
        mock_call_next = AsyncMock()
        mock_response = Mock()
        mock_call_next.return_value = mock_response

        result = await financial_middleware.dispatch(mock_request, mock_call_next)

        assert result == mock_response
        mock_call_next.assert_called_once_with(mock_request)

    async def test_dispatch_get_request(self, financial_middleware, mock_request):
        """Test dispatch skips validation for GET requests."""
        mock_request.url.path = "/api/portfolio/balance"
        mock_request.method = "GET"
        mock_call_next = AsyncMock()
        mock_response = Mock()
        mock_call_next.return_value = mock_response

        result = await financial_middleware.dispatch(mock_request, mock_call_next)

        assert result == mock_response
        mock_call_next.assert_called_once_with(mock_request)

    async def test_dispatch_validation_error(self, financial_middleware, mock_request):
        """Test dispatch handles validation errors."""
        mock_request.url.path = "/api/portfolio/balance"
        mock_request.method = "POST"
        mock_request.body = AsyncMock(return_value=b'{"amount": 123.45}')

        with pytest.raises(HTTPException) as exc_info:
            await financial_middleware.dispatch(mock_request, AsyncMock())

        assert exc_info.value.status_code == 400
        assert "float type" in str(exc_info.value.detail)

    async def test_dispatch_unexpected_error(self, financial_middleware, mock_request):
        """Test dispatch handles unexpected errors."""
        mock_request.url.path = "/api/portfolio/balance"
        mock_request.method = "POST"

        with patch.object(financial_middleware, "_validate_request_body", side_effect=Exception("Unexpected error")):
            mock_call_next = AsyncMock()
            mock_response = Mock()
            mock_call_next.return_value = mock_response

            result = await financial_middleware.dispatch(mock_request, mock_call_next)

            assert result == mock_response
            mock_call_next.assert_called_once_with(mock_request)

    async def test_validate_request_body_empty(self, financial_middleware, mock_request):
        """Test request body validation with empty body."""
        mock_request.body = AsyncMock(return_value=b"")

        # Should not raise any exception
        await financial_middleware._validate_request_body(mock_request)

    async def test_validate_request_body_invalid_json(self, financial_middleware, mock_request):
        """Test request body validation with invalid JSON."""
        mock_request.body = AsyncMock(return_value=b"invalid json")

        # Should not raise any exception for invalid JSON
        await financial_middleware._validate_request_body(mock_request)

    async def test_validate_request_body_valid_data(self, financial_middleware, mock_request):
        """Test request body validation with valid financial data."""
        valid_data = {"amount": "123.45", "currency": "USD"}
        mock_request.body = AsyncMock(return_value=json.dumps(valid_data).encode())

        # Should not raise any exception
        await financial_middleware._validate_request_body(mock_request)

    async def test_validate_response_body_non_json(self, financial_middleware):
        """Test response body validation with non-JSON response."""
        mock_response = Mock()
        mock_response.headers = {"content-type": "text/html"}

        # Should not raise any exception
        await financial_middleware._validate_response_body(mock_response)

    async def test_validate_response_body_json(self, financial_middleware):
        """Test response body validation with JSON response."""
        mock_response = Mock()
        mock_response.headers = {"content-type": "application/json"}

        # Should not raise any exception (simplified validation)
        await financial_middleware._validate_response_body(mock_response)

    def test_validate_financial_data_dict(self, financial_middleware):
        """Test validation of financial data in dictionary format."""
        data = {
            "amount": "123.45",
            "price": "50000.12345678",
            "non_financial": "test",
            "nested": {"balance": "1000.00"}
        }

        # Should not raise exception for valid data
        financial_middleware._validate_financial_data(data, "request")

    def test_validate_financial_data_list(self, financial_middleware):
        """Test validation of financial data in list format."""
        data = [
            {"amount": "123.45"},
            {"price": "50000.12345678"}
        ]

        # Should not raise exception for valid data
        financial_middleware._validate_financial_data(data, "request")

    def test_validate_financial_value_none(self, financial_middleware):
        """Test validation of None financial value."""
        # Should not raise exception for None values
        financial_middleware._validate_financial_value("amount", None, "request")

    def test_validate_financial_value_float_error(self, financial_middleware):
        """Test validation raises error for float values."""
        with pytest.raises(ValidationError) as exc_info:
            financial_middleware._validate_financial_value("amount", 123.45, "request")

        assert "float type" in str(exc_info.value)

    def test_validate_financial_value_invalid_type(self, financial_middleware):
        """Test validation raises error for invalid types."""
        with pytest.raises(ValidationError) as exc_info:
            financial_middleware._validate_financial_value("amount", {"invalid": "type"}, "request")

        assert "must be string, int, or Decimal" in str(exc_info.value)

    def test_validate_financial_value_invalid_decimal(self, financial_middleware):
        """Test validation raises error for invalid decimal values."""
        with pytest.raises(ValidationError) as exc_info:
            financial_middleware._validate_financial_value("amount", "invalid_decimal", "request")

        assert "not a valid decimal" in str(exc_info.value) or "must be finite" in str(exc_info.value)

    def test_validate_financial_value_valid_string(self, financial_middleware):
        """Test validation passes for valid string decimal."""
        # Should not raise exception
        financial_middleware._validate_financial_value("amount", "123.45", "request")

    def test_validate_financial_value_valid_int(self, financial_middleware):
        """Test validation passes for valid integer."""
        # Should not raise exception
        financial_middleware._validate_financial_value("amount", 123, "request")

    def test_validate_decimal_precision_price(self, financial_middleware):
        """Test decimal precision validation for price fields."""
        # Valid price with 8 decimal places
        value = Decimal("50000.12345678")
        financial_middleware._validate_decimal_precision("price", value, "request")

        # Invalid price with too many decimal places
        with pytest.raises(ValidationError) as exc_info:
            value = Decimal("50000.123456789")  # 9 decimal places
            financial_middleware._validate_decimal_precision("price", value, "request")

        assert "too many decimal places" in str(exc_info.value)

    def test_validate_decimal_precision_amount(self, financial_middleware):
        """Test decimal precision validation for amount fields."""
        # Valid amount with 2 decimal places
        value = Decimal("123.45")
        financial_middleware._validate_decimal_precision("amount", value, "request")

        # Invalid amount with too many decimal places
        with pytest.raises(ValidationError) as exc_info:
            value = Decimal("123.456")  # 3 decimal places
            financial_middleware._validate_decimal_precision("amount", value, "request")

        assert "too many decimal places" in str(exc_info.value)

    def test_validate_decimal_precision_integer(self, financial_middleware):
        """Test decimal precision validation for integer values."""
        value = Decimal("123")
        # Should not raise exception for integers
        financial_middleware._validate_decimal_precision("amount", value, "request")

    def test_validate_value_range_infinite(self, financial_middleware):
        """Test value range validation rejects infinite values."""
        with pytest.raises(ValidationError) as exc_info:
            financial_middleware._validate_value_range("amount", Decimal("inf"), "request")

        assert "must be finite" in str(exc_info.value)

    def test_validate_value_range_negative_amount(self, financial_middleware):
        """Test value range validation rejects negative amounts."""
        with pytest.raises(ValidationError) as exc_info:
            financial_middleware._validate_value_range("amount", Decimal("-123.45"), "request")

        assert "should not be negative" in str(exc_info.value)

    def test_validate_value_range_negative_pnl(self, financial_middleware):
        """Test value range validation allows negative PnL values."""
        # Should not raise exception for negative PnL
        financial_middleware._validate_value_range("pnl", Decimal("-123.45"), "request")
        financial_middleware._validate_value_range("unrealized_pnl", Decimal("-100.00"), "request")

    def test_validate_value_range_too_large(self, financial_middleware):
        """Test value range validation rejects values that are too large."""
        with pytest.raises(ValidationError) as exc_info:
            large_value = Decimal("1e13")  # Larger than max allowed
            financial_middleware._validate_value_range("amount", large_value, "request")

        assert "exceeds maximum allowed value" in str(exc_info.value)

    def test_get_field_type_price(self, financial_middleware):
        """Test field type detection for price fields."""
        assert financial_middleware._get_field_type("price") == "price"
        assert financial_middleware._get_field_type("bid_price") == "price"
        assert financial_middleware._get_field_type("ask") == "price"
        assert financial_middleware._get_field_type("last") == "price"

    def test_get_field_type_quantity(self, financial_middleware):
        """Test field type detection for quantity fields."""
        assert financial_middleware._get_field_type("quantity") == "quantity"
        assert financial_middleware._get_field_type("volume") == "quantity"
        assert financial_middleware._get_field_type("total_supply") == "quantity"

    def test_get_field_type_ratio(self, financial_middleware):
        """Test field type detection for ratio fields."""
        assert financial_middleware._get_field_type("ratio") == "ratio"
        assert financial_middleware._get_field_type("exchange_rate") == "ratio"

    def test_get_field_type_percentage(self, financial_middleware):
        """Test field type detection for percentage fields."""
        assert financial_middleware._get_field_type("percentage") == "percentage"
        assert financial_middleware._get_field_type("percent_change") == "percentage"

    def test_get_field_type_default(self, financial_middleware):
        """Test field type detection defaults to amount."""
        assert financial_middleware._get_field_type("unknown_field") == "amount"
        assert financial_middleware._get_field_type("balance") == "amount"


class TestDecimalEnforcementMiddleware:
    """Tests for DecimalEnforcementMiddleware."""

    async def test_dispatch_financial_endpoint_json(self, decimal_middleware, mock_request):
        """Test dispatch adds headers for financial endpoints with JSON response."""
        mock_request.url.path = "/api/portfolio/balance"
        mock_call_next = AsyncMock()
        mock_response = Mock()
        mock_response.headers = {"content-type": "application/json"}
        mock_call_next.return_value = mock_response

        result = await decimal_middleware.dispatch(mock_request, mock_call_next)

        assert result == mock_response
        assert result.headers["X-Financial-Precision"] == "decimal-enforced"
        assert result.headers["X-Float-Usage"] == "prohibited"

    async def test_dispatch_financial_endpoint_non_json(self, decimal_middleware, mock_request):
        """Test dispatch does not add headers for non-JSON responses."""
        mock_request.url.path = "/api/portfolio/balance"
        mock_call_next = AsyncMock()
        mock_response = Mock()
        mock_response.headers = {"content-type": "text/html"}
        mock_call_next.return_value = mock_response

        result = await decimal_middleware.dispatch(mock_request, mock_call_next)

        assert result == mock_response
        assert "X-Financial-Precision" not in result.headers

    async def test_dispatch_non_financial_endpoint(self, decimal_middleware, mock_request):
        """Test dispatch does not add headers for non-financial endpoints."""
        mock_request.url.path = "/api/health"
        mock_call_next = AsyncMock()
        mock_response = Mock()
        mock_response.headers = {"content-type": "application/json"}
        mock_call_next.return_value = mock_response

        result = await decimal_middleware.dispatch(mock_request, mock_call_next)

        assert result == mock_response
        assert "X-Financial-Precision" not in result.headers

    def test_is_financial_endpoint(self, decimal_middleware):
        """Test financial endpoint detection."""
        assert decimal_middleware._is_financial_endpoint("/api/analytics/test")
        assert decimal_middleware._is_financial_endpoint("/api/capital/allocate")
        assert not decimal_middleware._is_financial_endpoint("/api/health")


class TestCurrencyValidationMiddleware:
    """Tests for CurrencyValidationMiddleware."""

    async def test_dispatch_non_financial_endpoint(self, currency_middleware, mock_request):
        """Test dispatch skips validation for non-financial endpoints."""
        mock_request.url.path = "/api/health"
        mock_call_next = AsyncMock()
        mock_response = Mock()
        mock_call_next.return_value = mock_response

        result = await currency_middleware.dispatch(mock_request, mock_call_next)

        assert result == mock_response
        mock_call_next.assert_called_once_with(mock_request)

    async def test_dispatch_get_request(self, currency_middleware, mock_request):
        """Test dispatch skips validation for GET requests."""
        mock_request.url.path = "/api/portfolio/balance"
        mock_request.method = "GET"
        mock_call_next = AsyncMock()
        mock_response = Mock()
        mock_call_next.return_value = mock_response

        result = await currency_middleware.dispatch(mock_request, mock_call_next)

        assert result == mock_response
        mock_call_next.assert_called_once_with(mock_request)

    async def test_dispatch_validation_error(self, currency_middleware, mock_request):
        """Test dispatch handles currency validation errors."""
        mock_request.url.path = "/api/portfolio/balance"
        mock_request.method = "POST"
        mock_request.body = AsyncMock(return_value=b'{"currency": "INVALID"}')

        with pytest.raises(HTTPException) as exc_info:
            await currency_middleware.dispatch(mock_request, AsyncMock())

        assert exc_info.value.status_code == 400
        assert "Invalid currency code" in str(exc_info.value.detail)

    async def test_dispatch_unexpected_error(self, currency_middleware, mock_request):
        """Test dispatch handles unexpected errors."""
        mock_request.url.path = "/api/portfolio/balance"
        mock_request.method = "POST"

        with patch.object(currency_middleware, "_validate_currency_codes", side_effect=Exception("Unexpected error")):
            mock_call_next = AsyncMock()
            mock_response = Mock()
            mock_call_next.return_value = mock_response

            result = await currency_middleware.dispatch(mock_request, mock_call_next)

            assert result == mock_response
            mock_call_next.assert_called_once_with(mock_request)

    async def test_validate_currency_codes_empty_body(self, currency_middleware, mock_request):
        """Test currency validation with empty body."""
        mock_request.body = AsyncMock(return_value=b"")

        # Should not raise any exception
        await currency_middleware._validate_currency_codes(mock_request)

    async def test_validate_currency_codes_invalid_json(self, currency_middleware, mock_request):
        """Test currency validation with invalid JSON."""
        mock_request.body = AsyncMock(return_value=b"invalid json")

        # Should not raise any exception for invalid JSON
        await currency_middleware._validate_currency_codes(mock_request)

    async def test_validate_currency_codes_valid(self, currency_middleware, mock_request):
        """Test currency validation with valid currency codes."""
        valid_data = {"currency": "USD", "base_currency": "BTC", "quote_currency": "ETH"}
        mock_request.body = AsyncMock(return_value=json.dumps(valid_data).encode())

        # Should not raise any exception
        await currency_middleware._validate_currency_codes(mock_request)

    def test_check_currency_fields_dict_valid(self, currency_middleware):
        """Test currency field validation with valid currencies."""
        data = {
            "currency": "USD",
            "base_currency": "BTC",
            "quote_currency": "ETH",
            "asset": "USDT",
            "other_field": "not_currency"
        }

        # Should not raise exception
        currency_middleware._check_currency_fields(data)

    def test_check_currency_fields_dict_invalid(self, currency_middleware):
        """Test currency field validation with invalid currencies."""
        data = {"currency": "INVALID"}

        with pytest.raises(ValidationError) as exc_info:
            currency_middleware._check_currency_fields(data)

        assert "Invalid currency code: INVALID" in str(exc_info.value)

    def test_check_currency_fields_nested_dict(self, currency_middleware):
        """Test currency field validation with nested dictionaries."""
        data = {
            "trade": {
                "base_currency": "BTC",
                "quote_currency": "USD"
            },
            "portfolio": {
                "assets": [
                    {"currency": "ETH"}
                ]
            }
        }

        # Should not raise exception for valid nested currencies
        currency_middleware._check_currency_fields(data)

    def test_check_currency_fields_nested_invalid(self, currency_middleware):
        """Test currency field validation with invalid nested currencies."""
        data = {
            "trade": {
                "base_currency": "INVALID"
            }
        }

        with pytest.raises(ValidationError):
            currency_middleware._check_currency_fields(data)

    def test_check_currency_fields_list(self, currency_middleware):
        """Test currency field validation with list data."""
        data = [
            {"currency": "USD"},
            {"currency": "BTC"},
            {"asset": "ETH"}
        ]

        # Should not raise exception for valid currencies
        currency_middleware._check_currency_fields(data)

    def test_check_currency_fields_list_invalid(self, currency_middleware):
        """Test currency field validation with invalid currency in list."""
        data = [
            {"currency": "USD"},
            {"currency": "INVALID"}
        ]

        with pytest.raises(ValidationError):
            currency_middleware._check_currency_fields(data)

    def test_check_currency_fields_non_string_currency(self, currency_middleware):
        """Test currency field validation with non-string currency value."""
        data = {"currency": 123}

        # Should not raise exception for non-string values (they are ignored)
        currency_middleware._check_currency_fields(data)

    def test_is_financial_endpoint(self, currency_middleware):
        """Test financial endpoint detection."""
        assert currency_middleware._is_financial_endpoint("/api/analytics/test")
        assert currency_middleware._is_financial_endpoint("/api/capital/allocate")
        assert currency_middleware._is_financial_endpoint("/api/portfolio/balance")
        assert currency_middleware._is_financial_endpoint("/api/trading/order")
        assert currency_middleware._is_financial_endpoint("/api/risk/var")
        assert currency_middleware._is_financial_endpoint("/api/exchanges/balance")
        assert not currency_middleware._is_financial_endpoint("/api/health")

    def test_valid_currencies_coverage(self, currency_middleware):
        """Test that valid currencies include major types."""
        # Test major fiat currencies
        assert "USD" in currency_middleware.VALID_CURRENCIES
        assert "EUR" in currency_middleware.VALID_CURRENCIES
        assert "JPY" in currency_middleware.VALID_CURRENCIES

        # Test major cryptocurrencies
        assert "BTC" in currency_middleware.VALID_CURRENCIES
        assert "ETH" in currency_middleware.VALID_CURRENCIES

        # Test stablecoins
        assert "USDT" in currency_middleware.VALID_CURRENCIES
        assert "USDC" in currency_middleware.VALID_CURRENCIES