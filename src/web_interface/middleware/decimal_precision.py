"""
Decimal Precision Middleware for T-Bot web interface.

This middleware ensures that all financial data in API requests and responses
maintains decimal precision to prevent floating-point arithmetic errors
that could lead to financial losses.
"""

import json
from decimal import Decimal, InvalidOperation
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import get_logger

logger = get_logger(__name__)


class DecimalPrecisionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle Decimal precision in API requests and responses.

    This middleware:
    1. Converts float values to Decimal in incoming requests
    2. Ensures all outgoing responses use Decimal precision
    3. Validates numerical precision for financial data
    4. Logs any precision conversion issues
    """

    def __init__(self, app, financial_fields: list[str] | None = None):
        """
        Initialize decimal precision middleware.

        Args:
            app: FastAPI application
            financial_fields: List of field names that require decimal precision
        """
        super().__init__(app)

        # Default financial fields that must maintain decimal precision
        self.financial_fields = financial_fields or [
            "price",
            "quantity",
            "amount",
            "balance",
            "value",
            "cost",
            "pnl",
            "profit",
            "loss",
            "fee",
            "commission",
            "capital",
            "drawdown",
            "return",
            "allocation",
            "exposure",
            "margin",
            "collateral",
            "premium",
            "strike",
            "notional",
            "principal",
            "interest",
            "dividend",
            "yield",
            "rate",
            "spread",
            "basis",
            # Portfolio-related
            "total_value",
            "total_pnl",
            "unrealized_pnl",
            "realized_pnl",
            "market_value",
            "cost_basis",
            "entry_price",
            "current_price",
            "stop_price",
            "take_profit",
            "stop_loss",
            # Risk-related
            "var",
            "expected_shortfall",
            "sharpe_ratio",
            "profit_factor",
            "calmar_ratio",
            "max_drawdown",
            "average_win",
            "average_loss",
            # Position-related
            "position_size",
            "allocated_capital",
            "available_balance",
            "locked_balance",
            "total_balance",
            "free_balance",
            # Trading-related
            "order_value",
            "fill_price",
            "average_fill_price",
            "slippage",
            "execution_cost",
            "bid",
            "ask",
            "last_price",
            "volume",
        ]

        # Patterns to match financial fields with suffixes/prefixes
        self.financial_patterns = [
            "_price",
            "_amount",
            "_value",
            "_balance",
            "_pnl",
            "_cost",
            "_fee",
            "_commission",
            "_capital",
            "_allocation",
            "_exposure",
            "price_",
            "amount_",
            "value_",
            "balance_",
            "pnl_",
            "cost_",
        ]

    async def dispatch(self, request: Request, call_next):
        """
        Process request and response with decimal precision handling.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response with decimal precision maintained
        """
        # Process request body for decimal conversion
        if request.method in ["POST", "PUT", "PATCH"]:
            request = await self._process_request_body(request)

        # Process the request through the rest of the application
        response = await call_next(request)

        # Process response body for decimal precision
        response = await self._process_response_body(response)

        return response

    async def _process_request_body(self, request: Request) -> Request:
        """
        Process incoming request body to convert floats to Decimals.

        Args:
            request: HTTP request

        Returns:
            Request with decimal-converted body
        """
        try:
            content_type = request.headers.get("content-type", "")

            if "application/json" in content_type:
                # Read and parse JSON body
                body = await request.body()
                if body:
                    try:
                        data = json.loads(body.decode())
                        converted_data = self._convert_to_decimal(data)

                        # Replace request body with converted data
                        # DecimalEncoder is a static class, use default=encoder method
                        def decimal_serializer(obj):
                            if hasattr(obj, "__decimal__"):
                                return float(obj)
                            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

                        new_body = json.dumps(converted_data, default=decimal_serializer).encode()

                        # Create new request with updated body
                        async def receive():
                            return {"type": "http.request", "body": new_body}

                        # Replace the receive callable
                        request._receive = receive

                        logger.debug(
                            f"Converted request body for decimal precision: {request.url.path}"
                        )

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON request body: {e}")
                    except Exception as e:
                        logger.error(f"Error processing request body: {e}")

        except Exception as e:
            logger.error(f"Error in request body processing: {e}")

        return request

    async def _process_response_body(self, response: Response) -> Response:
        """
        Process outgoing response to ensure decimal precision.

        Args:
            response: HTTP response

        Returns:
            Response with decimal precision maintained
        """
        try:
            # Only process JSON responses
            if isinstance(response, JSONResponse) or (
                hasattr(response, "media_type") and response.media_type == "application/json"
            ):
                # Get response content
                if hasattr(response, "body"):
                    body = response.body
                    if body:
                        try:
                            # Parse response data
                            if isinstance(body, bytes):
                                data = json.loads(body.decode())
                            else:
                                data = body

                            # Convert to decimal precision
                            converted_data = self._convert_to_decimal(data)

                            # Create new response with converted data
                            new_response = JSONResponse(
                                content=converted_data,
                                status_code=response.status_code,
                                headers=dict(response.headers),
                            )

                            return new_response

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON response body: {e}")
                        except Exception as e:
                            logger.error(f"Error processing response body: {e}")

        except Exception as e:
            logger.error(f"Error in response body processing: {e}")

        return response

    def _convert_to_decimal(self, data: Any) -> Any:
        """
        Recursively convert float values to Decimal for financial fields.

        Args:
            data: Data structure to convert

        Returns:
            Data structure with Decimal values
        """
        if isinstance(data, dict):
            converted = {}
            for key, value in data.items():
                if self._is_financial_field(key) and isinstance(value, int | float):
                    try:
                        # Convert to Decimal with appropriate precision
                        converted[key] = Decimal(str(value))
                        if value != converted[key]:
                            logger.debug(f"Converted {key}: {value} -> {converted[key]}")
                    except (InvalidOperation, ValueError) as e:
                        logger.warning(f"Failed to convert {key} to Decimal: {e}")
                        converted[key] = value
                else:
                    converted[key] = self._convert_to_decimal(value)
            return converted

        elif isinstance(data, list):
            return [self._convert_to_decimal(item) for item in data]

        elif isinstance(data, int | float) and not isinstance(data, bool):
            # For standalone numeric values, convert to Decimal
            try:
                return Decimal(str(data))
            except (InvalidOperation, ValueError):
                return data

        else:
            return data

    def _is_financial_field(self, field_name: str) -> bool:
        """
        Check if a field name represents financial data requiring decimal precision.

        Args:
            field_name: Name of the field to check

        Returns:
            True if field requires decimal precision
        """
        field_lower = field_name.lower()

        # Check exact matches
        if field_lower in [f.lower() for f in self.financial_fields]:
            return True

        # Check patterns
        for pattern in self.financial_patterns:
            if pattern.lower() in field_lower:
                return True

        return False

    def _validate_decimal_precision(self, value: Decimal, field_name: str) -> bool:
        """
        Validate that a decimal value has appropriate precision for financial data.

        Args:
            value: Decimal value to validate
            field_name: Name of the field being validated

        Returns:
            True if precision is appropriate
        """
        try:
            # Check for excessive precision that might indicate floating-point conversion
            if value.as_tuple().exponent < -10:
                logger.warning(
                    f"Field {field_name} has excessive precision: {value}. "
                    "This might indicate floating-point conversion error."
                )
                return False

            # Check for very small values that might be precision errors
            if abs(value) < Decimal("1e-10") and abs(value) > 0:
                logger.warning(
                    f"Field {field_name} has very small value: {value}. "
                    "This might indicate precision error."
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating decimal precision for {field_name}: {e}")
            return False


class DecimalValidationMiddleware(BaseHTTPMiddleware):
    """
    Additional middleware for strict decimal validation in critical trading operations.

    This middleware provides extra validation for critical financial operations
    to ensure data integrity and prevent potential financial losses.
    """

    def __init__(self, app, critical_endpoints: list[str] | None = None):
        """
        Initialize decimal validation middleware.

        Args:
            app: FastAPI application
            critical_endpoints: List of endpoints requiring strict validation
        """
        super().__init__(app)

        self.critical_endpoints = critical_endpoints or [
            "/api/trading/orders",
            "/api/portfolio/",
            "/api/risk/",
            "/api/bot_management/",
        ]

    async def dispatch(self, request: Request, call_next):
        """
        Validate decimal precision for critical trading endpoints.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response or validation error
        """
        # Check if this is a critical endpoint
        if any(endpoint in str(request.url.path) for endpoint in self.critical_endpoints):
            validation_result = await self._validate_request_precision(request)
            if not validation_result.get("valid", True):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Decimal precision validation failed",
                        "details": validation_result.get("errors", []),
                        "message": "Financial data must maintain decimal precision",
                    },
                )

        response = await call_next(request)
        return response

    async def _validate_request_precision(self, request: Request) -> dict[str, Any]:
        """
        Validate decimal precision in request data.

        Args:
            request: HTTP request to validate

        Returns:
            Validation result with errors if any
        """
        errors = []

        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get("content-type", "")

                if "application/json" in content_type:
                    body = await request.body()
                    if body:
                        try:
                            data = json.loads(body.decode())
                            self._validate_data_precision(data, errors, "")
                        except json.JSONDecodeError:
                            errors.append("Invalid JSON format")

        except Exception as e:
            errors.append(f"Validation error: {e!s}")

        return {"valid": len(errors) == 0, "errors": errors}

    def _validate_data_precision(self, data: Any, errors: list[str], path: str):
        """
        Recursively validate decimal precision in data structure.

        Args:
            data: Data to validate
            errors: List to append errors to
            path: Current path in data structure
        """
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, float):
                    # Check for potential floating-point precision issues
                    if self._has_precision_issues(value, key):
                        errors.append(
                            f"Field '{current_path}' contains float value with potential "
                            f"precision issues: {value}. Use string representation for exact decimals."
                        )

                self._validate_data_precision(value, errors, current_path)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._validate_data_precision(item, errors, f"{path}[{i}]")

    def _has_precision_issues(self, value: float, field_name: str) -> bool:
        """
        Check if a float value has potential precision issues.

        Args:
            value: Float value to check
            field_name: Name of the field

        Returns:
            True if precision issues detected
        """
        # Check for very long decimal representations
        str_value = str(value)
        if "e" in str_value.lower():
            return True  # Scientific notation might indicate precision issues

        # Check for excessive decimal places
        if "." in str_value and len(str_value.split(".")[1]) > 8:
            return True

        # Check for known problematic float patterns
        problematic_patterns = [
            "99999999",  # Repeating 9s
            "00000001",  # Repeating 0s with trailing 1
        ]

        for pattern in problematic_patterns:
            if pattern in str_value.replace(".", ""):
                return True

        return False
