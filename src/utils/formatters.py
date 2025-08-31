"""
Data Formatters for Consistent Output

This module provides comprehensive data formatting and transformation utilities for
financial data, API responses, log formatting, chart data, reports, and exports
to ensure consistent and professional output across all components of the trading bot system.

Key Functions:
- Financial Formatting: currency formatting, percentage display, P&L formatting
- API Response Formatting: JSON standardization, error formatting, success responses
- Log Formatting: structured log formatting, correlation IDs, performance metrics
- Chart Data Formatting: OHLCV formatting, indicator data, chart preparation
- Report Formatting: performance reports, risk reports, trade reports
- Export Formatting: CSV, Excel, JSON export utilities

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
"""

import csv
import io
import json
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path
from typing import Any

import pandas as pd

# Import from P-001 core components
from src.core.exceptions import ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Financial Formatting
# =============================================================================


def format_currency(amount: Decimal | int, currency: str = "USD", precision: int | None = None) -> str:
    """
    Format amount as currency string with proper financial precision.

    Args:
        amount: Amount to format (Decimal or int, no float for precision)
        currency: Currency code (default "USD")
        precision: Number of decimal places (auto-determined if None)

    Returns:
        Formatted currency string

    Raises:
        ValidationError: If amount is invalid
    """
    from src.utils.decimal_utils import DECIMAL_CONTEXT, to_decimal

    if not isinstance(amount, (int, Decimal)):
        raise ValidationError(f"Amount must be Decimal or int for financial precision, got {type(amount).__name__}")

    # Convert to Decimal for precise formatting
    decimal_amount = to_decimal(amount) if not isinstance(amount, Decimal) else amount

    # Determine precision based on currency if not specified
    if precision is None:
        if currency.upper() in ["BTC", "ETH", "LTC", "XRP", "ADA", "DOT", "LINK", "SOL"]:
            precision = 8  # Crypto precision (DECIMAL(20,8))
        elif currency.upper() in ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/GBP"]:
            precision = 4  # Forex precision (DECIMAL(20,4))
        elif currency.upper() in ["USDT", "USDC", "USD", "EUR", "GBP", "CAD", "AUD"]:
            precision = 2  # Fiat precision (DECIMAL(20,2))
        elif currency.upper() in ["JPY", "KRW"]:
            precision = 0  # No decimals for some fiat currencies
        else:
            precision = 4  # Default precision

    # Create quantizer based on precision
    if precision > 0:
        quantizer = Decimal(10) ** -precision
    else:
        quantizer = Decimal("1")

    # Use financial context for quantization
    with localcontext(DECIMAL_CONTEXT):
        formatted_amount = decimal_amount.quantize(quantizer, rounding=ROUND_HALF_UP)

    # Format with thousands separators, ensuring proper precision
    formatted = f"{formatted_amount:,.{precision}f}"

    return f"{formatted} {currency.upper()}"


def format_percentage(value: Decimal | int, precision: int = 2) -> str:
    """
    Format value as percentage with financial precision.

    Args:
        value: Value to format as Decimal (as decimal, e.g., 0.05 for 5%)
        precision: Number of decimal places

    Returns:
        Formatted percentage string

    Raises:
        ValidationError: If value is invalid
    """
    from src.utils.decimal_utils import DECIMAL_CONTEXT, HUNDRED, to_decimal

    if not isinstance(value, (int, Decimal)):
        raise ValidationError(f"Value must be Decimal or int for financial precision, got {type(value).__name__}")

    # Convert to Decimal for precise percentage calculation
    decimal_value = to_decimal(value) if not isinstance(value, Decimal) else value

    # Use financial context for all percentage calculations
    with localcontext(DECIMAL_CONTEXT):
        percentage_decimal = decimal_value * HUNDRED

        # Format with sign and precision using Decimal quantization
        quantizer = Decimal(10) ** -precision
        rounded_percentage = percentage_decimal.quantize(quantizer, rounding=ROUND_HALF_UP)

    # Format with sign using string manipulation (no float conversion)
    percentage_str = str(rounded_percentage)

    if rounded_percentage >= 0:
        return f"+{percentage_str}%"
    else:
        return f"{percentage_str}%"


def format_pnl(pnl: Decimal | int, currency: str = "USD") -> tuple[str, str]:
    """
    Format P&L with appropriate color coding info and financial precision.

    Args:
        pnl: P&L value as Decimal or int
        currency: Currency code

    Returns:
        Tuple of (formatted_pnl, color_indicator)

    Raises:
        ValidationError: If P&L is invalid
    """
    from src.utils.decimal_utils import ZERO, to_decimal

    if not isinstance(pnl, (int, Decimal)):
        raise ValidationError(f"P&L must be Decimal or int for financial precision, got {type(pnl).__name__}")

    # Convert to Decimal for comparison
    pnl_decimal = to_decimal(pnl) if not isinstance(pnl, Decimal) else pnl

    formatted = format_currency(pnl_decimal, currency)

    # Determine color based on P&L value using Decimal comparison
    if pnl_decimal > ZERO:
        color = "green"
        symbol = "+"
    elif pnl_decimal < ZERO:
        color = "red"
        symbol = ""
    else:
        color = "neutral"
        symbol = ""

    return f"{symbol}{formatted}", color


def format_quantity(quantity: Decimal | int, symbol: str) -> str:
    """
    Format trading quantity with appropriate precision based on asset type.

    Args:
        quantity: Quantity to format as Decimal or int
        symbol: Trading symbol for precision context

    Returns:
        Formatted quantity string

    Raises:
        ValidationError: If quantity is invalid
    """
    from src.utils.decimal_utils import DECIMAL_CONTEXT, to_decimal

    if not isinstance(quantity, (int, Decimal)):
        raise ValidationError(f"Quantity must be Decimal or int for financial precision, got {type(quantity).__name__}")

    # Convert to Decimal for precise formatting
    decimal_qty = to_decimal(quantity) if not isinstance(quantity, Decimal) else quantity

    # Determine precision based on symbol type (crypto: 8 decimals, forex: 4, stocks: 2)
    symbol_upper = symbol.upper()
    if any(crypto in symbol_upper for crypto in ["BTC", "ETH", "LTC", "XRP", "ADA", "DOT", "LINK", "SOL", "BNB"]):
        precision = 8  # Crypto precision (DECIMAL(20,8))
    elif any(fiat in symbol_upper for fiat in ["EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD"]):
        precision = 4  # Forex precision (DECIMAL(20,4))
    elif "USDT" in symbol_upper or "USD" in symbol_upper or "USDC" in symbol_upper:
        precision = 2  # Fiat stablecoin precision (DECIMAL(20,2))
    else:
        precision = 4  # Default precision for unknown assets

    # Use financial context for quantization
    with localcontext(DECIMAL_CONTEXT):
        quantizer = Decimal(10) ** -precision
        formatted_qty = decimal_qty.quantize(quantizer, rounding=ROUND_HALF_UP)

    return f"{formatted_qty:,.{precision}f}"


def format_price(price: Decimal | int, symbol: str) -> str:
    """
    Format price with appropriate precision based on asset type.

    Args:
        price: Price to format as Decimal or int
        symbol: Trading symbol for precision context

    Returns:
        Formatted price string

    Raises:
        ValidationError: If price is invalid
    """
    from src.utils.decimal_utils import DECIMAL_CONTEXT, to_decimal

    if not isinstance(price, (int, Decimal)):
        raise ValidationError(f"Price must be Decimal or int for financial precision, got {type(price).__name__}")

    # Convert to Decimal for precise formatting
    decimal_price = to_decimal(price) if not isinstance(price, Decimal) else price

    # Determine precision based on symbol type (crypto: 8 decimals, forex: 4, stocks: 2)
    symbol_upper = symbol.upper()
    if any(crypto in symbol_upper for crypto in ["BTC", "ETH", "LTC", "XRP", "ADA", "DOT", "LINK", "SOL", "BNB"]):
        precision = 8  # Crypto precision (DECIMAL(20,8))
    elif any(fiat in symbol_upper for fiat in ["EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD"]):
        precision = 4  # Forex precision (DECIMAL(20,4))
    elif "USDT" in symbol_upper or "USD" in symbol_upper or "USDC" in symbol_upper:
        precision = 2  # Fiat stablecoin precision (DECIMAL(20,2))
    else:
        precision = 4  # Default precision for unknown assets

    # Use financial context for quantization
    with localcontext(DECIMAL_CONTEXT):
        quantizer = Decimal(10) ** -precision
        formatted_price = decimal_price.quantize(quantizer, rounding=ROUND_HALF_UP)

    return f"{formatted_price:,.{precision}f}"


# =============================================================================
# API Response Formatting
# =============================================================================


def format_api_response(data: Any, success: bool = True, message: str | None = None) -> dict[str, Any]:
    """
    Format standardized API response.

    Args:
        data: Response data
        success: Whether the request was successful
        message: Optional message

    Returns:
        Formatted API response dictionary
    """
    response = {
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    if message:
        response["message"] = message

    return response


def format_error_response(error: Exception, error_code: str | None = None) -> dict[str, Any]:
    """
    Format error response for API.

    Args:
        error: Exception that occurred
        error_code: Optional error code

    Returns:
        Formatted error response dictionary
    """
    error_dict: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
        "details": getattr(error, "details", {}),
    }

    if error_code:
        error_dict["code"] = error_code

    if hasattr(error, "error_code"):
        error_dict["code"] = error.error_code

    response = {
        "success": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": error_dict,
    }

    return response


def format_success_response(data: Any, message: str = "Operation completed successfully") -> dict[str, Any]:
    """
    Format success response for API.

    Args:
        data: Response data
        message: Success message

    Returns:
        Formatted success response dictionary
    """
    return format_api_response(data, success=True, message=message)


def format_paginated_response(data: list[Any], page: int, page_size: int, total: int) -> dict[str, Any]:
    """
    Format paginated response for API.

    Args:
        data: List of data items
        page: Current page number
        page_size: Number of items per page
        total: Total number of items

    Returns:
        Formatted paginated response dictionary
    """
    total_pages = (total + page_size - 1) // page_size

    response = {
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
    }

    return response


# =============================================================================
# Log Formatting
# =============================================================================


def format_log_entry(level: str, message: str, **kwargs: Any) -> dict[str, Any]:
    """
    Format structured log entry.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Log message
        **kwargs: Additional log fields

    Returns:
        Formatted log entry dictionary
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "message": message,
        **kwargs,
    }

    return log_entry


def format_correlation_id(correlation_id: str) -> str:
    """
    Format correlation ID for logging.

    Args:
        correlation_id: Correlation ID string

    Returns:
        Formatted correlation ID
    """
    if not correlation_id:
        return "no-correlation-id"

    # Ensure correlation ID is properly formatted
    formatted = correlation_id.strip()
    if len(formatted) > 50:
        formatted = formatted[:50] + "..."

    return formatted


def format_structured_log(level: str, message: str, correlation_id: str | None = None, **kwargs: Any) -> str:
    """
    Format structured log as JSON string.

    Args:
        level: Log level
        message: Log message
        correlation_id: Optional correlation ID
        **kwargs: Additional log fields

    Returns:
        JSON formatted log string
    """
    log_entry = format_log_entry(level, message, **kwargs)

    if correlation_id:
        log_entry["correlation_id"] = format_correlation_id(correlation_id)

    return json.dumps(log_entry, default=str)


def format_performance_log(
    function_name: str, execution_time_ms: float, success: bool, **kwargs: Any
) -> dict[str, Any]:
    """
    Format performance log entry.

    Args:
        function_name: Name of the function
        execution_time_ms: Execution time in milliseconds
        success: Whether the function executed successfully
        **kwargs: Additional performance metrics

    Returns:
        Formatted performance log dictionary
    """
    return format_log_entry(
        "INFO" if success else "ERROR",
        f"Function {function_name} executed",
        function=function_name,
        execution_time_ms=execution_time_ms,
        success=success,
        **kwargs,
    )


# =============================================================================
# Chart Data Formatting
# =============================================================================


def format_ohlcv_data(ohlcv_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Format OHLCV data for charting using stream processing pattern.

    This function now aligns with core module streaming patterns by processing
    data as a stream of events rather than batch operations.

    Args:
        ohlcv_data: List of OHLCV data dictionaries

    Returns:
        Formatted OHLCV data

    Raises:
        ValidationError: If data format is invalid
    """
    if not isinstance(ohlcv_data, list):
        raise ValidationError("OHLCV data must be a list")

    # Use stream processing pattern consistent with core event system
    formatted_data = []

    for candle in ohlcv_data:
        if not isinstance(candle, dict):
            raise ValidationError("Each OHLCV item must be a dictionary")

        # Validate required fields
        required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
        for field in required_fields:
            if field not in candle:
                raise ValidationError(f"Missing required field: {field}")

        # Format timestamp
        if isinstance(candle["timestamp"], int | float):
            timestamp = datetime.fromtimestamp(candle["timestamp"], tz=timezone.utc)
        else:
            timestamp = candle["timestamp"]

        # Use Decimal for precision then convert to string for JSON serialization
        from src.utils.decimal_utils import to_decimal

        formatted_candle = {
            "timestamp": timestamp.isoformat(),
            "open": str(to_decimal(candle["open"])),
            "high": str(to_decimal(candle["high"])),
            "low": str(to_decimal(candle["low"])),
            "close": str(to_decimal(candle["close"])),
            "volume": str(to_decimal(candle["volume"])),
        }

        formatted_data.append(formatted_candle)

    return formatted_data


def format_indicator_data(
    indicator_name: str, values: list[Decimal], timestamps: list[datetime] | None = None
) -> dict[str, Any]:
    """
    Format indicator data for charting with financial precision.

    Args:
        indicator_name: Name of the indicator
        values: List of Decimal indicator values
        timestamps: Optional list of timestamps

    Returns:
        Formatted indicator data dictionary

    Raises:
        ValidationError: If data format is invalid
    """
    from src.utils.decimal_utils import to_decimal

    if not isinstance(values, list):
        raise ValidationError("Indicator values must be a list")

    if timestamps and len(timestamps) != len(values):
        raise ValidationError("Timestamps and values must have the same length")

    # Convert values to string to preserve precision (no float conversion)
    formatted_values = []
    for v in values:
        if v is not None:
            # Convert to Decimal then to string to preserve precision
            decimal_v = to_decimal(v) if not isinstance(v, Decimal) else v
            formatted_values.append(str(decimal_v))
        else:
            formatted_values.append(None)

    formatted_data = {
        "indicator": indicator_name,
        "values": formatted_values,  # String representation preserves precision
    }

    if timestamps:
        formatted_data["timestamps"] = [t.isoformat() for t in timestamps]

    return formatted_data


def format_chart_data(
    symbol: str, ohlcv_data: list[dict[str, Any]], indicators: dict[str, list[Decimal]] | None = None
) -> dict[str, Any]:
    """
    Format complete chart data with financial precision.

    Args:
        symbol: Trading symbol
        ohlcv_data: OHLCV data
        indicators: Optional indicator data with Decimal values

    Returns:
        Formatted chart data dictionary
    """
    chart_data: dict[str, Any] = {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ohlcv": format_ohlcv_data(ohlcv_data),
    }

    if indicators:
        chart_data["indicators"] = {}
        for indicator_name, values in indicators.items():
            chart_data["indicators"][indicator_name] = format_indicator_data(indicator_name, values)

    return chart_data


# =============================================================================
# Report Formatting
# =============================================================================


def format_performance_report(performance_data: dict[str, Any]) -> dict[str, Any]:
    """
    Format performance report.

    Args:
        performance_data: Performance metrics dictionary

    Returns:
        Formatted performance report
    """
    report = {
        "report_type": "performance",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_trades": performance_data.get("total_trades", 0),
            "winning_trades": performance_data.get("winning_trades", 0),
            "losing_trades": performance_data.get("losing_trades", 0),
            "win_rate": format_percentage(performance_data.get("win_rate", 0)),
            "total_pnl": format_currency(performance_data.get("total_pnl", 0)),
            "sharpe_ratio": f"{performance_data.get('sharpe_ratio', 0):.3f}",
            "max_drawdown": format_percentage(performance_data.get("max_drawdown", 0)),
        },
        "details": performance_data,
    }

    return report


def format_risk_report(risk_data: dict[str, Any]) -> dict[str, Any]:
    """
    Format risk report.

    Args:
        risk_data: Risk metrics dictionary

    Returns:
        Formatted risk report
    """
    report = {
        "report_type": "risk",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "current_drawdown": format_percentage(risk_data.get("current_drawdown", 0)),
            "var_1d": format_currency(risk_data.get("var_1d", 0)),
            "var_5d": format_currency(risk_data.get("var_5d", 0)),
            "portfolio_exposure": format_percentage(risk_data.get("portfolio_exposure", 0)),
            "risk_level": risk_data.get("risk_level", "unknown"),
        },
        "details": risk_data,
    }

    return report


def format_trade_report(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Format trade report.

    Args:
        trades: List of trade dictionaries

    Returns:
        Formatted trade report
    """
    if not trades:
        return {
            "report_type": "trades",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_trades": 0,
                "total_volume": format_currency(0),
                "total_pnl": format_currency(0),
            },
            "trades": [],
        }

    # Calculate summary statistics using Decimal for precision
    from src.utils.decimal_utils import ZERO, to_decimal

    total_volume = ZERO
    total_pnl = ZERO

    for trade in trades:
        quantity = to_decimal(trade.get("quantity", 0))
        price = to_decimal(trade.get("price", 0))
        pnl = to_decimal(trade.get("pnl", 0))

        total_volume += quantity * price
        total_pnl += pnl

    # Format individual trades
    formatted_trades = []
    for trade in trades:
        formatted_trade = {
            "id": trade.get("id", ""),
            "symbol": trade.get("symbol", ""),
            "side": trade.get("side", ""),
            "quantity": format_quantity(trade.get("quantity", 0), trade.get("symbol", "")),
            "price": format_price(trade.get("price", 0), trade.get("symbol", "")),
            "pnl": format_currency(trade.get("pnl", 0)),
            "timestamp": trade.get("timestamp", ""),
            "fee": format_currency(trade.get("fee", 0)),
        }
        formatted_trades.append(formatted_trade)

    report = {
        "report_type": "trades",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_trades": len(trades),
            "total_volume": format_currency(total_volume),
            "total_pnl": format_currency(total_pnl),
        },
        "trades": formatted_trades,
    }

    return report


# =============================================================================
# Export Formatting
# =============================================================================


def format_csv_data(data: list[dict[str, Any]], headers: list[str] | None = None) -> str:
    """
    Format data as CSV string.

    Args:
        data: List of data dictionaries
        headers: Optional list of column headers

    Returns:
        CSV formatted string

    Raises:
        ValidationError: If data format is invalid
    """
    if not isinstance(data, list):
        raise ValidationError("Data must be a list of dictionaries")

    if not data:
        return ""

    # Determine headers if not provided
    if not headers:
        headers = list(data[0].keys())

    # Create CSV string
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)

    # Write header
    writer.writeheader()

    # Write data rows
    for row in data:
        # Ensure all required fields are present
        formatted_row = {header: row.get(header, "") for header in headers}
        writer.writerow(formatted_row)

    return output.getvalue()


def format_excel_data(data: list[dict[str, Any]], sheet_name: str = "Data") -> bytes:
    """
    Format data as Excel file bytes.

    Args:
        data: List of data dictionaries
        sheet_name: Name of the Excel sheet

    Returns:
        Excel file as bytes

    Raises:
        ValidationError: If data format is invalid
    """
    if not isinstance(data, list):
        raise ValidationError("Data must be a list of dictionaries")

    if not data:
        # Create empty DataFrame
        df = pd.DataFrame()
    else:
        # Create DataFrame from data
        df = pd.DataFrame(data)

    # Create Excel writer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    return output.getvalue()


def format_json_data(data: Any, pretty: bool = True) -> str:
    """
    Format data as JSON string.

    Args:
        data: Data to format
        pretty: Whether to format with indentation

    Returns:
        JSON formatted string

    Raises:
        ValidationError: If data cannot be serialized
    """
    try:
        if pretty:
            return json.dumps(data, indent=2, default=str)
        else:
            return json.dumps(data, default=str)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Cannot serialize data to JSON: {e!s}") from e


def export_to_file(data: Any, file_path: str, format_type: str = "json") -> None:
    """
    Export data to file in specified format.

    Args:
        data: Data to export
        file_path: Path to output file
        format_type: Export format ("json", "csv", "excel")

    Raises:
        ValidationError: If export fails
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format_type.lower() == "json":
            content = format_json_data(data)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        elif format_type.lower() == "csv":
            if not isinstance(data, list):
                raise ValidationError("CSV export requires list of dictionaries")
            content = format_csv_data(data)
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(content)

        elif format_type.lower() == "excel":
            if not isinstance(data, list):
                raise ValidationError("Excel export requires list of dictionaries")
            content_bytes = format_excel_data(data)
            with open(path, "wb") as f:
                f.write(content_bytes)

        else:
            raise ValidationError(f"Unsupported export format: {format_type}")

        logger.info(f"Data exported to {file_path} in {format_type} format")

    except Exception as e:
        raise ValidationError(f"Export failed: {e!s}") from e
