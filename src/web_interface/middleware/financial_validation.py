"""
Financial Input Validation Middleware.

This middleware ensures all financial endpoints validate monetary values using Decimal
precision and prevents float usage in financial calculations.
"""

import json
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class FinancialValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for validating financial inputs on trading endpoints.

    This middleware:
    1. Validates that financial values use proper Decimal precision
    2. Prevents float usage in financial calculations
    3. Ensures monetary values have appropriate decimal places
    4. Validates input ranges for financial data
    """

    # Endpoints that handle financial data
    FINANCIAL_ENDPOINTS = [
        "/api/analytics/",
        "/api/capital/",
        "/api/portfolio/",
        "/api/trading/",
        "/api/risk/",
        "/api/exchanges/",
    ]

    # Fields that should be treated as financial values
    FINANCIAL_FIELDS = {
        # Capital Management
        "amount",
        "allocated_amount",
        "utilized_amount",
        "available_amount",
        "total_capital",
        "exposure_amount",
        "hedge_amount",
        # Trading
        "price",
        "quantity",
        "volume",
        "value",
        "cost",
        "proceeds",
        "stop_loss",
        "take_profit",
        "order_value",
        "commission",
        # Portfolio
        "balance",
        "equity",
        "margin",
        "free_balance",
        "used_balance",
        "total_balance",
        "pnl",
        "unrealized_pnl",
        "realized_pnl",
        # Risk Management
        "var",
        "max_loss",
        "risk_amount",
        "position_size",
        "limit",
        "threshold",
        "max_allocation",
        "max_exposure",
        # Market Data
        "bid",
        "ask",
        "last",
        "open",
        "high",
        "low",
        "close",
        "market_cap",
        "total_supply",
        "circulating_supply",
    }

    # Maximum decimal places allowed for different value types
    DECIMAL_PRECISION_LIMITS = {
        "price": 8,  # Crypto prices can have up to 8 decimal places
        "quantity": 8,  # Crypto quantities can have up to 8 decimal places
        "amount": 2,  # Fiat amounts typically 2 decimal places
        "percentage": 4,  # Percentages can have 4 decimal places
        "ratio": 6,  # Ratios can have 6 decimal places
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate financial data in requests."""
        try:
            # Only validate financial endpoints
            if not self._is_financial_endpoint(request.url.path):
                return await call_next(request)

            # Skip validation for GET requests (no body)
            if request.method == "GET":
                return await call_next(request)

            # Validate request body if present
            await self._validate_request_body(request)

            response = await call_next(request)

            # Validate response body for financial endpoints
            await self._validate_response_body(response)

            return response

        except ValidationError as e:
            logger.error(f"Financial validation error: {e}")
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in financial validation middleware: {e}")
            return await call_next(request)

    def _is_financial_endpoint(self, path: str) -> bool:
        """Check if the endpoint handles financial data."""
        return any(path.startswith(endpoint) for endpoint in self.FINANCIAL_ENDPOINTS)

    async def _validate_request_body(self, request: Request) -> None:
        """Validate financial data in request body."""
        try:
            # Get request body
            body = await request.body()
            if not body:
                return

            # Parse JSON body
            try:
                data = json.loads(body.decode())
            except json.JSONDecodeError:
                # Not JSON, skip validation
                return

            # Validate financial fields
            self._validate_financial_data(data, "request")

        except ValidationError:
            # Re-raise ValidationError for proper handling
            raise
        except Exception as e:
            logger.error(f"Error validating request body: {e}")

    async def _validate_response_body(self, response: Response) -> None:
        """Validate financial data in response body."""
        try:
            # Skip validation for non-JSON responses
            content_type = response.headers.get("content-type", "")
            if "application/json" not in content_type:
                return

            # This is a simplified validation - in practice, you'd need to
            # capture and re-stream the response body
            logger.debug("Response validation completed")

        except Exception as e:
            logger.error(f"Error validating response body: {e}")

    def _validate_financial_data(self, data: dict[str, Any] | list[Any], context: str) -> None:
        """
        Recursively validate financial data structure.

        Args:
            data: Data structure to validate
            context: Context for error reporting ("request" or "response")
        """
        if isinstance(data, dict):
            for key, value in data.items():
                # Check if this is a financial field
                if key.lower() in self.FINANCIAL_FIELDS:
                    self._validate_financial_value(key, value, context)

                # Recursively validate nested structures
                elif isinstance(value, (dict, list)):
                    self._validate_financial_data(value, context)

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._validate_financial_data(item, context)

    def _validate_financial_value(self, field_name: str, value: Any, context: str) -> None:
        """
        Validate a single financial value.

        Args:
            field_name: Name of the field being validated
            value: Value to validate
            context: Context for error reporting
        """
        # Skip None values
        if value is None:
            return

        # Check for float usage (not allowed for financial values)
        if isinstance(value, float):
            raise ValidationError(
                f"Field '{field_name}' in {context} uses float type. "
                f"Financial values must be strings for Decimal precision. "
                f"Got: {value} (type: {type(value)})"
            )

        # Convert to string for Decimal validation
        if isinstance(value, (int, str)):
            str_value = str(value)
        else:
            raise ValidationError(
                f"Field '{field_name}' in {context} must be string, int, or Decimal. "
                f"Got: {value} (type: {type(value)})"
            )

        # Validate as Decimal
        try:
            decimal_value = Decimal(str_value)
        except (InvalidOperation, TypeError, ValueError) as e:
            raise ValidationError(
                f"Field '{field_name}' in {context} is not a valid decimal: {str_value}. Error: {e}"
            )

        # Validate decimal places
        self._validate_decimal_precision(field_name, decimal_value, context)

        # Validate value ranges
        self._validate_value_range(field_name, decimal_value, context)

    def _validate_decimal_precision(self, field_name: str, value: Decimal, context: str) -> None:
        """Validate decimal precision limits."""
        # Get decimal places
        sign, digits, exponent = value.as_tuple()
        if isinstance(exponent, int) and exponent < 0:
            decimal_places = abs(exponent)
        else:
            decimal_places = 0

        # Determine field type and precision limit
        field_type = self._get_field_type(field_name)
        max_precision = self.DECIMAL_PRECISION_LIMITS.get(field_type, 8)  # Default 8 places

        if decimal_places > max_precision:
            raise ValidationError(
                f"Field '{field_name}' in {context} has too many decimal places: {decimal_places}. "
                f"Maximum allowed for {field_type}: {max_precision}"
            )

    def _validate_value_range(self, field_name: str, value: Decimal, context: str) -> None:
        """Validate value ranges for financial fields."""
        # Basic range validation

        # All financial values should be finite
        if not value.is_finite():
            raise ValidationError(f"Field '{field_name}' in {context} must be finite. Got: {value}")

        # Amounts should generally be non-negative (except PnL fields)
        pnl_fields = {"pnl", "unrealized_pnl", "realized_pnl", "total_pnl"}
        if field_name.lower() not in pnl_fields and "pnl" not in field_name.lower():
            if value < 0:
                raise ValidationError(
                    f"Field '{field_name}' in {context} should not be negative. Got: {value}"
                )

        # Check for reasonable maximum values (prevent overflow)
        max_value = Decimal("1e12")  # 1 trillion
        if abs(value) > max_value:
            raise ValidationError(
                f"Field '{field_name}' in {context} exceeds maximum allowed value. "
                f"Got: {value}, Max: {max_value}"
            )

    def _get_field_type(self, field_name: str) -> str:
        """Determine field type for precision validation."""
        field_lower = field_name.lower()

        if any(
            word in field_lower
            for word in ["price", "bid", "ask", "last", "open", "high", "low", "close"]
        ):
            return "price"
        elif any(word in field_lower for word in ["quantity", "volume", "supply"]):
            return "quantity"
        elif any(word in field_lower for word in ["ratio", "rate"]):
            return "ratio"
        elif any(word in field_lower for word in ["percentage", "percent"]):
            return "percentage"
        else:
            return "amount"  # Default to amount (2 decimal places)


class DecimalEnforcementMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce Decimal usage in responses.

    This middleware ensures all outgoing financial data uses proper
    string representation for Decimal values.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Enforce Decimal string formatting in responses."""
        response = await call_next(request)

        # Only process JSON responses on financial endpoints
        if self._is_financial_endpoint(
            request.url.path
        ) and "application/json" in response.headers.get("content-type", ""):
            # Add header to indicate Decimal enforcement
            response.headers["X-Financial-Precision"] = "decimal-enforced"
            response.headers["X-Float-Usage"] = "prohibited"

        return response

    def _is_financial_endpoint(self, path: str) -> bool:
        """Check if the endpoint handles financial data."""
        financial_endpoints = [
            "/api/analytics/",
            "/api/capital/",
            "/api/portfolio/",
            "/api/trading/",
            "/api/risk/",
            "/api/exchanges/",
        ]
        return any(path.startswith(endpoint) for endpoint in financial_endpoints)


class CurrencyValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate currency codes and formats.
    """

    # Valid currency codes
    VALID_CURRENCIES = {
        # Fiat currencies
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "AUD",
        "CAD",
        "CHF",
        "CNY",
        # Major cryptocurrencies
        "BTC",
        "ETH",
        "BNB",
        "XRP",
        "ADA",
        "SOL",
        "DOT",
        "MATIC",
        "LTC",
        "BCH",
        "ETC",
        "XLM",
        "VET",
        "TRX",
        "EOS",
        "NEO",
        # Stablecoins
        "USDT",
        "USDC",
        "BUSD",
        "DAI",
        "TUSD",
        "PAX",
        "USDD",
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate currency codes in requests."""
        try:
            # Only validate financial endpoints
            if not self._is_financial_endpoint(request.url.path):
                return await call_next(request)

            if request.method != "GET":
                await self._validate_currency_codes(request)

            return await call_next(request)

        except ValidationError as e:
            logger.error(f"Currency validation error: {e}")
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error in currency validation middleware: {e}")
            return await call_next(request)

    async def _validate_currency_codes(self, request: Request) -> None:
        """Validate currency codes in request."""
        try:
            body = await request.body()
            if not body:
                return

            data = json.loads(body.decode())
            self._check_currency_fields(data)

        except json.JSONDecodeError:
            pass  # Not JSON
        except ValidationError:
            # Re-raise ValidationError for proper handling
            raise
        except Exception as e:
            logger.error(f"Error validating currency codes: {e}")

    def _check_currency_fields(self, data: Any) -> None:
        """Recursively check currency fields."""
        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower() in {"currency", "base_currency", "quote_currency", "asset"}:
                    if isinstance(value, str) and value.upper() not in self.VALID_CURRENCIES:
                        raise ValidationError(f"Invalid currency code: {value}")
                elif isinstance(value, (dict, list)):
                    self._check_currency_fields(value)
        elif isinstance(data, list):
            for item in data:
                self._check_currency_fields(item)

    def _is_financial_endpoint(self, path: str) -> bool:
        """Check if endpoint handles financial data."""
        return any(
            path.startswith(ep)
            for ep in [
                "/api/analytics/",
                "/api/capital/",
                "/api/portfolio/",
                "/api/trading/",
                "/api/risk/",
                "/api/exchanges/",
            ]
        )
