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

import json
import csv
import io
from typing import Any, Dict, List, Tuple
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path

# Import from P-001 core components
from src.core.exceptions import ValidationError
from src.core.logging import get_logger


logger = get_logger(__name__)


# =============================================================================
# Financial Formatting
# =============================================================================

def format_currency(
        amount: float,
        currency: str = "USD",
        precision: int = 2) -> str:
    """
    Format amount as currency string.

    Args:
        amount: Amount to format
        currency: Currency code (default "USD")
        precision: Number of decimal places

    Returns:
        Formatted currency string

    Raises:
        ValidationError: If amount is invalid
    """
    if not isinstance(amount, (int, float, Decimal)):
        raise ValidationError(
            f"Amount must be a number, got {type(amount).__name__}")

    # Determine precision based on currency
    if currency.upper() in ["BTC", "ETH"]:
        precision = 8  # Crypto precision
    elif currency.upper() in ["USDT", "USDC", "USD"]:
        precision = 2  # Fiat precision
    elif currency.upper() in ["JPY", "KRW"]:
        precision = 0  # No decimals for some fiat currencies

    # Convert to Decimal for precise formatting
    decimal_amount = Decimal(str(amount))
    formatted_amount = decimal_amount.quantize(
        Decimal(
            f"0.{'0' * (precision - 1)}1") if precision > 0 else Decimal("1"),
        rounding=ROUND_HALF_UP
    )

    # Format with thousands separators
    formatted = f"{formatted_amount:,.{precision}f}"

    return f"{formatted} {currency.upper()}"


def format_percentage(value: float, precision: int = 2) -> str:
    """
    Format value as percentage.

    Args:
        value: Value to format (as decimal, e.g., 0.05 for 5%)
        precision: Number of decimal places

    Returns:
        Formatted percentage string

    Raises:
        ValidationError: If value is invalid
    """
    if not isinstance(value, (int, float, Decimal)):
        raise ValidationError(
            f"Value must be a number, got {type(value).__name__}")

    # Convert to percentage
    percentage = float(value) * 100

    # Format with sign and precision
    if percentage >= 0:
        return f"+{percentage:.{precision}f}%"
    else:
        return f"{percentage:.{precision}f}%"


def format_pnl(pnl: float, currency: str = "USD") -> Tuple[str, str]:
    """
    Format P&L with appropriate color coding info.

    Args:
        pnl: P&L value
        currency: Currency code

    Returns:
        Tuple of (formatted_pnl, color_indicator)

    Raises:
        ValidationError: If P&L is invalid
    """
    if not isinstance(pnl, (int, float, Decimal)):
        raise ValidationError(
            f"P&L must be a number, got {type(pnl).__name__}")

    formatted = format_currency(pnl, currency)

    # Determine color based on P&L value
    if pnl > 0:
        color = "green"
        symbol = "+"
    elif pnl < 0:
        color = "red"
        symbol = ""
    else:
        color = "neutral"
        symbol = ""

    return f"{symbol}{formatted}", color


def format_quantity(quantity: float, symbol: str) -> str:
    """
    Format trading quantity with appropriate precision.

    Args:
        quantity: Quantity to format
        symbol: Trading symbol for precision context

    Returns:
        Formatted quantity string

    Raises:
        ValidationError: If quantity is invalid
    """
    if not isinstance(quantity, (int, float, Decimal)):
        raise ValidationError(
            f"Quantity must be a number, got {type(quantity).__name__}")

    # Determine precision based on symbol
    if "BTC" in symbol.upper():
        precision = 8
    elif "ETH" in symbol.upper():
        precision = 6
    elif "USDT" in symbol.upper() or "USD" in symbol.upper():
        precision = 2
    else:
        precision = 4

    # Convert to Decimal for precise formatting
    decimal_qty = Decimal(str(quantity))
    formatted_qty = decimal_qty.quantize(
        Decimal(f"0.{'0' * (precision - 1)}1"),
        rounding=ROUND_HALF_UP
    )

    return f"{formatted_qty:,.{precision}f}"


def format_price(price: float, symbol: str) -> str:
    """
    Format price with appropriate precision.

    Args:
        price: Price to format
        symbol: Trading symbol for precision context

    Returns:
        Formatted price string

    Raises:
        ValidationError: If price is invalid
    """
    if not isinstance(price, (int, float, Decimal)):
        raise ValidationError(
            f"Price must be a number, got {type(price).__name__}")

    # Determine precision based on symbol
    if "BTC" in symbol.upper():
        precision = 8
    elif "ETH" in symbol.upper():
        precision = 6
    elif "USDT" in symbol.upper() or "USD" in symbol.upper():
        precision = 2
    else:
        precision = 4

    # Convert to Decimal for precise formatting
    decimal_price = Decimal(str(price))
    formatted_price = decimal_price.quantize(
        Decimal(f"0.{'0' * (precision - 1)}1"),
        rounding=ROUND_HALF_UP
    )

    return f"{formatted_price:,.{precision}f}"


# =============================================================================
# API Response Formatting
# =============================================================================

def format_api_response(data: Any, success: bool = True,
                        message: str = None) -> Dict[str, Any]:
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
        "data": data
    }

    if message:
        response["message"] = message

    return response


def format_error_response(
        error: Exception, error_code: str = None) -> Dict[str, Any]:
    """
    Format error response for API.

    Args:
        error: Exception that occurred
        error_code: Optional error code

    Returns:
        Formatted error response dictionary
    """
    response = {
        "success": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "details": getattr(error, 'details', {})
        }
    }

    if error_code:
        response["error"]["code"] = error_code

    if hasattr(error, 'error_code'):
        response["error"]["code"] = error.error_code

    return response


def format_success_response(
        data: Any, message: str = "Operation completed successfully") -> Dict[str, Any]:
    """
    Format success response for API.

    Args:
        data: Response data
        message: Success message

    Returns:
        Formatted success response dictionary
    """
    return format_api_response(data, success=True, message=message)


def format_paginated_response(
        data: List[Any], page: int, page_size: int, total: int) -> Dict[str, Any]:
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
            "has_prev": page > 1
        }
    }

    return response


# =============================================================================
# Log Formatting
# =============================================================================

def format_log_entry(level: str, message: str, **kwargs) -> Dict[str, Any]:
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
        **kwargs
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


def format_structured_log(
        level: str,
        message: str,
        correlation_id: str = None,
        **kwargs) -> str:
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
    function_name: str, execution_time_ms: float,
    success: bool, **kwargs
) -> Dict[str, Any]:
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
        **kwargs
    )


# =============================================================================
# Chart Data Formatting
# =============================================================================

def format_ohlcv_data(
        ohlcv_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format OHLCV data for charting.

    Args:
        ohlcv_data: List of OHLCV data dictionaries

    Returns:
        Formatted OHLCV data

    Raises:
        ValidationError: If data format is invalid
    """
    if not isinstance(ohlcv_data, list):
        raise ValidationError("OHLCV data must be a list")

    formatted_data = []

    for candle in ohlcv_data:
        if not isinstance(candle, dict):
            raise ValidationError("Each OHLCV item must be a dictionary")

        # Validate required fields
        required_fields = ["timestamp", "open",
                           "high", "low", "close", "volume"]
        for field in required_fields:
            if field not in candle:
                raise ValidationError(f"Missing required field: {field}")

        # Format timestamp
        if isinstance(candle["timestamp"], (int, float)):
            timestamp = datetime.fromtimestamp(
                candle["timestamp"], tz=timezone.utc)
        else:
            timestamp = candle["timestamp"]

        formatted_candle = {
            "timestamp": timestamp.isoformat(),
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "volume": float(candle["volume"])
        }

        formatted_data.append(formatted_candle)

    return formatted_data


def format_indicator_data(
    indicator_name: str, values: List[float],
    timestamps: List[datetime] = None
) -> Dict[str, Any]:
    """
    Format indicator data for charting.

    Args:
        indicator_name: Name of the indicator
        values: List of indicator values
        timestamps: Optional list of timestamps

    Returns:
        Formatted indicator data dictionary

    Raises:
        ValidationError: If data format is invalid
    """
    if not isinstance(values, list):
        raise ValidationError("Indicator values must be a list")

    if timestamps and len(timestamps) != len(values):
        raise ValidationError(
            "Timestamps and values must have the same length")

    formatted_data = {
        "indicator": indicator_name,
        "values": [float(v) if v is not None else None for v in values]
    }

    if timestamps:
        formatted_data["timestamps"] = [t.isoformat() for t in timestamps]

    return formatted_data


def format_chart_data(
    symbol: str, ohlcv_data: List[Dict[str, Any]],
    indicators: Dict[str, List[float]] = None
) -> Dict[str, Any]:
    """
    Format complete chart data.

    Args:
        symbol: Trading symbol
        ohlcv_data: OHLCV data
        indicators: Optional indicator data

    Returns:
        Formatted chart data dictionary
    """
    chart_data = {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ohlcv": format_ohlcv_data(ohlcv_data)
    }

    if indicators:
        chart_data["indicators"] = {}
        for indicator_name, values in indicators.items():
            chart_data["indicators"][indicator_name] = format_indicator_data(
                indicator_name, values
            )

    return chart_data


# =============================================================================
# Report Formatting
# =============================================================================

def format_performance_report(
        performance_data: Dict[str, Any]) -> Dict[str, Any]:
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
            "max_drawdown": format_percentage(performance_data.get("max_drawdown", 0))
        },
        "details": performance_data
    }

    return report


def format_risk_report(risk_data: Dict[str, Any]) -> Dict[str, Any]:
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
            "risk_level": risk_data.get("risk_level", "unknown")
        },
        "details": risk_data
    }

    return report


def format_trade_report(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
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
                "total_pnl": format_currency(0)
            },
            "trades": []
        }

    # Calculate summary statistics
    total_volume = sum(
        float(trade.get("quantity", 0)) * float(trade.get("price", 0))
        for trade in trades
    )
    total_pnl = sum(float(trade.get("pnl", 0)) for trade in trades)

    # Format individual trades
    formatted_trades = []
    for trade in trades:
        formatted_trade = {
            "id": trade.get("id", ""),
            "symbol": trade.get("symbol", ""),
            "side": trade.get("side", ""),
            "quantity": format_quantity(float(trade.get("quantity", 0)), trade.get("symbol", "")),
            "price": format_price(float(trade.get("price", 0)), trade.get("symbol", "")),
            "pnl": format_currency(float(trade.get("pnl", 0))),
            "timestamp": trade.get("timestamp", ""),
            "fee": format_currency(float(trade.get("fee", 0)))
        }
        formatted_trades.append(formatted_trade)

    report = {
        "report_type": "trades",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_trades": len(trades),
            "total_volume": format_currency(total_volume),
            "total_pnl": format_currency(total_pnl)
        },
        "trades": formatted_trades
    }

    return report


# =============================================================================
# Export Formatting
# =============================================================================

def format_csv_data(data: List[Dict[str, Any]],
                    headers: List[str] = None) -> str:
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


def format_excel_data(data: List[Dict[str, Any]],
                      sheet_name: str = "Data") -> bytes:
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
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
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
        raise ValidationError(f"Cannot serialize data to JSON: {str(e)}")


def export_to_file(
        data: Any,
        file_path: str,
        format_type: str = "json") -> None:
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
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

        elif format_type.lower() == "csv":
            if not isinstance(data, list):
                raise ValidationError(
                    "CSV export requires list of dictionaries")
            content = format_csv_data(data)
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write(content)

        elif format_type.lower() == "excel":
            if not isinstance(data, list):
                raise ValidationError(
                    "Excel export requires list of dictionaries")
            content = format_excel_data(data)
            with open(path, 'wb') as f:
                f.write(content)

        else:
            raise ValidationError(f"Unsupported export format: {format_type}")

        logger.info(f"Data exported to {file_path} in {format_type} format")

    except Exception as e:
        raise ValidationError(f"Export failed: {str(e)}")
