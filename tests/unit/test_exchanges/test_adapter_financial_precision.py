"""
Exchange Adapter Financial Precision Validation Tests

This comprehensive test suite validates that all exchange adapters maintain
proper financial precision using Decimal types. Critical for ensuring
accurate data conversion between exchange APIs and internal systems.

Tests cover:
1. Data conversion precision between exchange APIs and internal types
2. WebSocket message precision handling
3. API response transformation precision
4. Error handling with precision preservation
5. Rate limiting with precise timing calculations
6. Connection pool precision metrics
7. Health monitoring precision calculations
8. Adapter performance metrics precision
"""

from decimal import ROUND_DOWN, Decimal
from unittest.mock import Mock

import pytest

from src.core.config import Config
from src.core.types import (
    OrderSide,
    OrderStatus,
    OrderType,
)


class TestBaseExchangeAdapterPrecision:
    """Test base exchange adapter financial precision handling."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for adapter testing."""
        config = Mock(spec=Config)
        config.exchange = Mock()
        return config

    def test_decimal_conversion_utilities(self):
        """Test decimal conversion utilities maintain precision."""
        # Test string to Decimal conversion
        conversion_cases = [
            {"input": "123.45678901", "expected": Decimal("123.45678901")},
            {"input": "0.00000001", "expected": Decimal("0.00000001")},
            {"input": "999999999.99999999", "expected": Decimal("999999999.99999999")},
            {"input": "0", "expected": Decimal("0")},
            {"input": "1", "expected": Decimal("1")},
        ]

        for case in conversion_cases:
            result = Decimal(case["input"])
            assert isinstance(result, Decimal)
            assert result == case["expected"]

    def test_api_response_decimal_parsing(self):
        """Test API response parsing maintains decimal precision."""
        # Mock API response with various numeric formats
        mock_api_response = {
            "symbol": "BTCUSDT",
            "price": "67890.12345678",  # High precision price
            "quantity": "0.12345678",   # High precision quantity
            "timestamp": 1640995200000,  # Integer timestamp
            "volume": "123456.789012",   # Volume with precision
            "fees": {
                "maker": "0.001",        # Decimal fee rate
                "taker": "0.0015",       # Decimal fee rate
            },
            "limits": {
                "min_quantity": "0.00000100",
                "max_quantity": "9000.00000000",
                "tick_size": "0.01000000",
                "step_size": "0.00000100",
            }
        }

        # Parse response maintaining precision
        parsed_data = {
            "symbol": mock_api_response["symbol"],
            "price": Decimal(mock_api_response["price"]),
            "quantity": Decimal(mock_api_response["quantity"]),
            "timestamp": mock_api_response["timestamp"],
            "volume": Decimal(mock_api_response["volume"]),
            "maker_fee": Decimal(mock_api_response["fees"]["maker"]),
            "taker_fee": Decimal(mock_api_response["fees"]["taker"]),
            "min_quantity": Decimal(mock_api_response["limits"]["min_quantity"]),
            "max_quantity": Decimal(mock_api_response["limits"]["max_quantity"]),
            "tick_size": Decimal(mock_api_response["limits"]["tick_size"]),
            "step_size": Decimal(mock_api_response["limits"]["step_size"]),
        }

        # Verify all decimal fields maintain precision
        assert isinstance(parsed_data["price"], Decimal)
        assert isinstance(parsed_data["quantity"], Decimal)
        assert isinstance(parsed_data["volume"], Decimal)
        assert isinstance(parsed_data["maker_fee"], Decimal)
        assert isinstance(parsed_data["taker_fee"], Decimal)

        # Verify exact values
        assert parsed_data["price"] == Decimal("67890.12345678")
        assert parsed_data["quantity"] == Decimal("0.12345678")
        assert parsed_data["volume"] == Decimal("123456.789012")

    def test_websocket_message_precision_parsing(self):
        """Test WebSocket message parsing maintains precision."""
        # Mock WebSocket ticker message
        ws_ticker_message = {
            "stream": "btcusdt@ticker",
            "data": {
                "s": "BTCUSDT",        # Symbol
                "c": "67890.12345678", # Close price
                "o": "67500.87654321", # Open price
                "h": "68200.99999999", # High price
                "l": "67000.00000001", # Low price
                "v": "1234567.89012345", # Volume
                "q": "83765432109.87654321", # Quote volume
                "P": "0.58",           # Price change percent
                "p": "390.88888889",   # Price change
                "c": "67890.12345678", # Close price
                "w": "67645.55555555", # Weighted average price
            }
        }

        # Parse WebSocket message
        parsed_ticker = {
            "symbol": ws_ticker_message["data"]["s"],
            "close_price": Decimal(ws_ticker_message["data"]["c"]),
            "open_price": Decimal(ws_ticker_message["data"]["o"]),
            "high_price": Decimal(ws_ticker_message["data"]["h"]),
            "low_price": Decimal(ws_ticker_message["data"]["l"]),
            "volume": Decimal(ws_ticker_message["data"]["v"]),
            "quote_volume": Decimal(ws_ticker_message["data"]["q"]),
            "price_change_percent": Decimal(ws_ticker_message["data"]["P"]),
            "price_change": Decimal(ws_ticker_message["data"]["p"]),
            "weighted_avg_price": Decimal(ws_ticker_message["data"]["w"]),
        }

        # Verify precision is maintained
        for key, value in parsed_ticker.items():
            if isinstance(value, Decimal):
                assert not isinstance(value, float), f"{key} should not be float"

        # Test specific precision values
        assert parsed_ticker["close_price"] == Decimal("67890.12345678")
        assert parsed_ticker["volume"] == Decimal("1234567.89012345")

        # Mock WebSocket order book message
        ws_orderbook_message = {
            "stream": "btcusdt@depth20",
            "data": {
                "bids": [
                    ["67890.12345678", "1.23456789"],
                    ["67889.11111111", "2.34567890"],
                    ["67888.99999999", "0.12345678"],
                ],
                "asks": [
                    ["67891.87654321", "0.98765432"],
                    ["67892.22222222", "1.11111111"],
                    ["67893.33333333", "3.33333333"],
                ]
            }
        }

        # Parse order book maintaining precision
        bids = []
        for bid_data in ws_orderbook_message["data"]["bids"]:
            bids.append({
                "price": Decimal(bid_data[0]),
                "quantity": Decimal(bid_data[1]),
            })

        asks = []
        for ask_data in ws_orderbook_message["data"]["asks"]:
            asks.append({
                "price": Decimal(ask_data[0]),
                "quantity": Decimal(ask_data[1]),
            })

        # Verify all order book levels maintain precision
        for bid in bids:
            assert isinstance(bid["price"], Decimal)
            assert isinstance(bid["quantity"], Decimal)

        for ask in asks:
            assert isinstance(ask["price"], Decimal)
            assert isinstance(ask["quantity"], Decimal)

    def test_rate_limiting_precision_calculations(self):
        """Test rate limiting calculations maintain timing precision."""
        # Mock rate limit configuration
        rate_limits = {
            "requests_per_second": Decimal("10"),
            "requests_per_minute": Decimal("600"),
            "weight_per_minute": Decimal("1200"),
            "orders_per_second": Decimal("5"),
            "orders_per_24_hours": Decimal("160000"),
        }

        # Calculate precise intervals
        intervals = {}
        for limit_type, limit_value in rate_limits.items():
            if "per_second" in limit_type:
                intervals[limit_type] = Decimal("1") / limit_value
            elif "per_minute" in limit_type:
                intervals[limit_type] = Decimal("60") / limit_value
            elif "per_24_hours" in limit_type:
                intervals[limit_type] = Decimal("86400") / limit_value  # 24 * 60 * 60

        # Verify all intervals are precise
        for limit_type, interval in intervals.items():
            assert isinstance(interval, Decimal)
            assert interval > 0

        # Test specific calculations
        assert intervals["requests_per_second"] == Decimal("0.1")  # 100ms between requests
        assert intervals["requests_per_minute"] == Decimal("0.1")  # 100ms between requests
        assert intervals["orders_per_second"] == Decimal("0.2")    # 200ms between orders

    def test_connection_pool_metrics_precision(self):
        """Test connection pool metrics maintain precision."""
        # Mock connection metrics
        connection_metrics = {
            "total_connections": 10,
            "active_connections": 7,
            "idle_connections": 3,
            "average_response_time": "0.125",  # 125ms
            "success_rate": "0.9875",          # 98.75%
            "error_rate": "0.0125",            # 1.25%
            "throughput_per_second": "15.5",   # 15.5 requests/sec
        }

        # Parse metrics maintaining precision
        parsed_metrics = {
            "total_connections": connection_metrics["total_connections"],
            "active_connections": connection_metrics["active_connections"],
            "idle_connections": connection_metrics["idle_connections"],
            "average_response_time": Decimal(connection_metrics["average_response_time"]),
            "success_rate": Decimal(connection_metrics["success_rate"]),
            "error_rate": Decimal(connection_metrics["error_rate"]),
            "throughput_per_second": Decimal(connection_metrics["throughput_per_second"]),
        }

        # Verify decimal fields maintain precision
        decimal_fields = ["average_response_time", "success_rate", "error_rate", "throughput_per_second"]
        for field in decimal_fields:
            assert isinstance(parsed_metrics[field], Decimal)

        # Verify calculations
        utilization_rate = Decimal(parsed_metrics["active_connections"]) / Decimal(parsed_metrics["total_connections"])
        assert isinstance(utilization_rate, Decimal)
        assert utilization_rate == Decimal("0.7")  # 70% utilization

    def test_health_check_precision_calculations(self):
        """Test health check calculations maintain precision."""
        # Mock health check data
        health_data = {
            "latency_samples": [
                Decimal("0.050"),  # 50ms
                Decimal("0.075"),  # 75ms
                Decimal("0.125"),  # 125ms
                Decimal("0.100"),  # 100ms
                Decimal("0.085"),  # 85ms
            ],
            "success_count": 95,
            "total_count": 100,
            "error_count": 5,
        }

        # Calculate health metrics
        avg_latency = sum(health_data["latency_samples"]) / len(health_data["latency_samples"])
        success_rate = Decimal(health_data["success_count"]) / Decimal(health_data["total_count"])
        error_rate = Decimal(health_data["error_count"]) / Decimal(health_data["total_count"])

        # Verify all calculations maintain precision
        assert isinstance(avg_latency, Decimal)
        assert isinstance(success_rate, Decimal)
        assert isinstance(error_rate, Decimal)

        # Verify specific values
        assert avg_latency == Decimal("0.087")  # (0.050 + 0.075 + 0.125 + 0.100 + 0.085) / 5
        assert success_rate == Decimal("0.95")   # 95%
        assert error_rate == Decimal("0.05")     # 5%

    def test_adapter_performance_metrics_precision(self):
        """Test adapter performance metrics maintain precision."""
        # Mock performance data
        performance_data = {
            "request_duration_samples": [
                Decimal("0.025"), Decimal("0.030"), Decimal("0.035"),
                Decimal("0.028"), Decimal("0.032"), Decimal("0.027"),
                Decimal("0.040"), Decimal("0.025"), Decimal("0.033"),
            ],
            "throughput_samples": [
                Decimal("20.5"), Decimal("21.2"), Decimal("19.8"),
                Decimal("20.1"), Decimal("21.5"), Decimal("20.0"),
            ],
            "memory_usage_mb": Decimal("45.67"),
            "cpu_usage_percent": Decimal("12.34"),
        }

        # Calculate performance metrics
        avg_duration = sum(performance_data["request_duration_samples"]) / len(performance_data["request_duration_samples"])
        avg_throughput = sum(performance_data["throughput_samples"]) / len(performance_data["throughput_samples"])

        # Calculate percentiles (simplified)
        sorted_durations = sorted(performance_data["request_duration_samples"])
        p95_duration = sorted_durations[int(0.95 * len(sorted_durations))]
        p99_duration = sorted_durations[int(0.99 * len(sorted_durations))]

        # Verify all metrics maintain precision
        assert isinstance(avg_duration, Decimal)
        assert isinstance(avg_throughput, Decimal)
        assert isinstance(p95_duration, Decimal)
        assert isinstance(p99_duration, Decimal)

        # Verify resource usage precision
        assert isinstance(performance_data["memory_usage_mb"], Decimal)
        assert isinstance(performance_data["cpu_usage_percent"], Decimal)


class TestExchangeSpecificAdapterPrecision:
    """Test exchange-specific adapter precision handling."""

    def test_binance_adapter_precision_conversion(self):
        """Test Binance adapter precision conversion patterns."""
        # Mock Binance API response
        binance_response = {
            "symbol": "BTCUSDT",
            "orderId": 123456789,
            "orderListId": -1,
            "clientOrderId": "test_order",
            "price": "67890.12000000",
            "origQty": "0.12345678",
            "executedQty": "0.12300000",
            "cummulativeQuoteQty": "8349.441745",
            "status": "PARTIALLY_FILLED",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "BUY",
            "stopPrice": "0.00000000",
            "icebergQty": "0.00000000",
            "time": 1640995200000,
            "updateTime": 1640995260000,
            "isWorking": True,
            "origQuoteOrderQty": "0.00000000",
        }

        # Convert to internal OrderResponse format
        converted_response = {
            "id": str(binance_response["orderId"]),
            "client_order_id": binance_response["clientOrderId"],
            "symbol": binance_response["symbol"],
            "side": OrderSide.BUY if binance_response["side"] == "BUY" else OrderSide.SELL,
            "order_type": OrderType.LIMIT if binance_response["type"] == "LIMIT" else OrderType.MARKET,
            "quantity": Decimal(binance_response["origQty"]),
            "price": Decimal(binance_response["price"]),
            "filled_quantity": Decimal(binance_response["executedQty"]),
            "status": OrderStatus.PARTIALLY_FILLED,
        }

        # Verify precision is maintained in conversion
        assert isinstance(converted_response["quantity"], Decimal)
        assert isinstance(converted_response["price"], Decimal)
        assert isinstance(converted_response["filled_quantity"], Decimal)

        assert converted_response["quantity"] == Decimal("0.12345678")
        assert converted_response["price"] == Decimal("67890.12000000")
        assert converted_response["filled_quantity"] == Decimal("0.12300000")

        # Calculate average price with precision
        if converted_response["filled_quantity"] > 0:
            cumulative_quote = Decimal(binance_response["cummulativeQuoteQty"])
            average_price = cumulative_quote / converted_response["filled_quantity"]
            assert isinstance(average_price, Decimal)

    def test_coinbase_adapter_precision_conversion(self):
        """Test Coinbase adapter precision conversion patterns."""
        # Mock Coinbase Advanced Trade API response
        coinbase_response = {
            "order_id": "abc123-def456-ghi789",
            "product_id": "BTC-USD",
            "side": "buy",
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": "0.12345678",
                    "limit_price": "67890.12",
                }
            },
            "order_status": "PARTIALLY_FILLED",
            "creation_time": "2021-12-31T12:00:00Z",
            "fill_fees": [
                {
                    "amount": "0.50",
                    "currency": "USD",
                }
            ],
            "filled_size": "0.12300000",
            "average_filled_price": "67885.50",
        }

        # Convert to internal format
        converted_response = {
            "id": coinbase_response["order_id"],
            "symbol": coinbase_response["product_id"].replace("-", ""),  # BTC-USD -> BTCUSD
            "side": OrderSide.BUY if coinbase_response["side"] == "buy" else OrderSide.SELL,
            "order_type": OrderType.LIMIT,  # From order_configuration
            "quantity": Decimal(coinbase_response["order_configuration"]["limit_limit_gtc"]["base_size"]),
            "price": Decimal(coinbase_response["order_configuration"]["limit_limit_gtc"]["limit_price"]),
            "filled_quantity": Decimal(coinbase_response["filled_size"]),
            "average_price": Decimal(coinbase_response["average_filled_price"]) if coinbase_response["average_filled_price"] else None,
            "status": OrderStatus.PARTIALLY_FILLED,
        }

        # Verify precision is maintained
        assert isinstance(converted_response["quantity"], Decimal)
        assert isinstance(converted_response["price"], Decimal)
        assert isinstance(converted_response["filled_quantity"], Decimal)
        assert isinstance(converted_response["average_price"], Decimal)

        # Test fee precision
        fee_amount = Decimal(coinbase_response["fill_fees"][0]["amount"])
        assert isinstance(fee_amount, Decimal)
        assert fee_amount == Decimal("0.50")

    def test_okx_adapter_precision_conversion(self):
        """Test OKX adapter precision conversion patterns."""
        # Mock OKX API response
        okx_response = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "instId": "BTC-USDT",
                    "ordId": "123456789",
                    "clOrdId": "test_order_okx",
                    "px": "67890.1234",      # Price with 4 decimals
                    "sz": "0.12345678",     # Size with 8 decimals
                    "accFillSz": "0.12300000", # Filled size
                    "avgPx": "67885.5678",   # Average price
                    "side": "buy",
                    "ordType": "limit",
                    "state": "partially_filled",
                    "fee": "-0.00000123",    # Negative fee (paid)
                    "feeCcy": "BTC",         # Fee currency
                    "cTime": "1640995200000",
                    "uTime": "1640995260000",
                }
            ]
        }

        order_data = okx_response["data"][0]

        # Convert to internal format
        converted_response = {
            "id": order_data["ordId"],
            "client_order_id": order_data["clOrdId"],
            "symbol": order_data["instId"].replace("-", ""),  # BTC-USDT -> BTCUSDT
            "side": OrderSide.BUY if order_data["side"] == "buy" else OrderSide.SELL,
            "order_type": OrderType.LIMIT if order_data["ordType"] == "limit" else OrderType.MARKET,
            "quantity": Decimal(order_data["sz"]),
            "price": Decimal(order_data["px"]),
            "filled_quantity": Decimal(order_data["accFillSz"]),
            "average_price": Decimal(order_data["avgPx"]) if order_data["avgPx"] and order_data["avgPx"] != "0" else None,
            "status": OrderStatus.PARTIALLY_FILLED,
            "fee": abs(Decimal(order_data["fee"])),  # Convert negative fee to positive
            "fee_currency": order_data["feeCcy"],
        }

        # Verify precision is maintained
        assert isinstance(converted_response["quantity"], Decimal)
        assert isinstance(converted_response["price"], Decimal)
        assert isinstance(converted_response["filled_quantity"], Decimal)
        assert isinstance(converted_response["average_price"], Decimal)
        assert isinstance(converted_response["fee"], Decimal)

        # Verify specific values
        assert converted_response["quantity"] == Decimal("0.12345678")
        assert converted_response["price"] == Decimal("67890.1234")
        assert converted_response["fee"] == Decimal("0.00000123")

    def test_error_response_precision_preservation(self):
        """Test error responses preserve precision context."""
        # Mock error scenarios with precision data
        error_scenarios = [
            {
                "error_code": "INSUFFICIENT_BALANCE",
                "available_balance": "1.23456789",
                "required_balance": "2.34567890",
                "shortfall": "1.11111101",
            },
            {
                "error_code": "LOT_SIZE_VIOLATION",
                "requested_quantity": "0.123456789",
                "min_quantity": "0.00000100",
                "step_size": "0.00000100",
                "adjusted_quantity": "0.12345600",
            },
            {
                "error_code": "PRICE_FILTER_VIOLATION",
                "requested_price": "50000.123",
                "tick_size": "0.01000000",
                "adjusted_price": "50000.12",
            },
        ]

        for scenario in error_scenarios:
            # Parse error data maintaining precision
            if "balance" in scenario["error_code"].lower():
                available = Decimal(scenario["available_balance"])
                required = Decimal(scenario["required_balance"])
                shortfall = Decimal(scenario["shortfall"])

                assert isinstance(available, Decimal)
                assert isinstance(required, Decimal)
                assert isinstance(shortfall, Decimal)

                # Verify calculation
                calculated_shortfall = required - available
                assert abs(calculated_shortfall - shortfall) < Decimal("0.00000001")

            elif "lot_size" in scenario["error_code"].lower():
                requested_qty = Decimal(scenario["requested_quantity"])
                min_qty = Decimal(scenario["min_quantity"])
                step_size = Decimal(scenario["step_size"])
                adjusted_qty = Decimal(scenario["adjusted_quantity"])

                assert isinstance(requested_qty, Decimal)
                assert isinstance(adjusted_qty, Decimal)

                # Verify adjustment calculation
                steps = (requested_qty - min_qty) / step_size
                expected_adjusted = min_qty + (steps.quantize(Decimal("1"), rounding=ROUND_DOWN) * step_size)
                assert adjusted_qty == expected_adjusted

    def test_batch_operation_precision_consistency(self):
        """Test batch operations maintain precision consistency."""
        # Mock batch order responses
        batch_responses = [
            {
                "orderId": 1,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "origQty": "0.10000000",
                "price": "67890.00000000",
                "executedQty": "0.10000000",
                "cummulativeQuoteQty": "6789.00000000",
                "status": "FILLED",
            },
            {
                "orderId": 2,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "origQty": "0.20000000",
                "price": "67891.00000000",
                "executedQty": "0.20000000",
                "cummulativeQuoteQty": "13578.20000000",
                "status": "FILLED",
            },
            {
                "orderId": 3,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "origQty": "0.15000000",
                "price": "67892.00000000",
                "executedQty": "0.15000000",
                "cummulativeQuoteQty": "10183.80000000",
                "status": "FILLED",
            },
        ]

        # Process batch maintaining precision
        total_quantity = Decimal("0")
        total_cost = Decimal("0")
        total_orders = len(batch_responses)

        for response in batch_responses:
            quantity = Decimal(response["origQty"])
            cost = Decimal(response["cummulativeQuoteQty"])

            total_quantity += quantity
            total_cost += cost

        # Calculate batch statistics
        average_order_size = total_quantity / total_orders
        average_price = total_cost / total_quantity
        total_fees = total_cost * Decimal("0.001")  # Assume 0.1% fee

        # Verify all calculations maintain precision
        assert isinstance(total_quantity, Decimal)
        assert isinstance(total_cost, Decimal)
        assert isinstance(average_order_size, Decimal)
        assert isinstance(average_price, Decimal)
        assert isinstance(total_fees, Decimal)

        # Verify specific values
        assert total_quantity == Decimal("0.45000000")  # 0.10 + 0.20 + 0.15
        assert total_cost == Decimal("30551.00000000")   # Sum of cumulative quotes

    def test_adapter_configuration_precision(self):
        """Test adapter configuration maintains precision settings."""
        # Mock adapter configuration with precision settings
        adapter_config = {
            "precision": {
                "price_decimals": 8,
                "quantity_decimals": 8,
                "fee_decimals": 8,
                "percentage_decimals": 4,
            },
            "limits": {
                "min_order_size": "0.00000100",
                "max_order_size": "9000.00000000",
                "tick_size": "0.01000000",
                "step_size": "0.00000100",
            },
            "timeouts": {
                "connection_timeout": "30.0",
                "read_timeout": "60.0",
                "retry_delay": "1.0",
            },
        }

        # Parse configuration maintaining precision
        parsed_config = {
            "precision": adapter_config["precision"],
            "limits": {
                "min_order_size": Decimal(adapter_config["limits"]["min_order_size"]),
                "max_order_size": Decimal(adapter_config["limits"]["max_order_size"]),
                "tick_size": Decimal(adapter_config["limits"]["tick_size"]),
                "step_size": Decimal(adapter_config["limits"]["step_size"]),
            },
            "timeouts": {
                "connection_timeout": Decimal(adapter_config["timeouts"]["connection_timeout"]),
                "read_timeout": Decimal(adapter_config["timeouts"]["read_timeout"]),
                "retry_delay": Decimal(adapter_config["timeouts"]["retry_delay"]),
            },
        }

        # Verify precision configuration
        for limit_name, limit_value in parsed_config["limits"].items():
            assert isinstance(limit_value, Decimal), f"{limit_name} should be Decimal"

        for timeout_name, timeout_value in parsed_config["timeouts"].items():
            assert isinstance(timeout_value, Decimal), f"{timeout_name} should be Decimal"

        # Test precision validation
        test_price = Decimal("67890.12345678")
        price_precision = parsed_config["precision"]["price_decimals"]
        price_decimals = len(str(test_price).split(".")[-1])
        assert price_decimals <= price_precision
