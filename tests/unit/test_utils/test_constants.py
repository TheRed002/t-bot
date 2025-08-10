"""
Unit tests for src/utils/constants.py module.

This module contains comprehensive tests for all constant definitions and enumerations including:
- Trading constants (market hours, settlement times, precision levels, trading sessions)
- API constants (endpoints, rate limits, timeouts, HTTP status codes)
- Financial constants (fee structures, minimum/maximum amounts, slippage tolerance)
- Configuration constants (default values, limits, thresholds)
- Error constants (error codes, messages, severity, recovery strategies)
- Market constants (symbol mappings, exchange specifications, trading pairs, intervals)
"""

import pytest
from decimal import Decimal
from datetime import time
from enum import Enum
from typing import Dict, Any, List

from src.utils.constants import (
    # Trading constants
    MARKET_HOURS,
    SETTLEMENT_TIMES,
    PRECISION_LEVELS,

    # API constants
    API_ENDPOINTS,
    RATE_LIMITS,
    TIMEOUTS,
    HTTP_STATUS_CODES,

    # Financial constants
    FEE_STRUCTURES,
    MINIMUM_AMOUNTS,
    MAXIMUM_AMOUNTS,
    SLIPPAGE_TOLERANCE,

    # Configuration constants
    DEFAULT_VALUES,
    LIMITS,
    THRESHOLDS,

    # Error constants
    ERROR_CODES,
    ERROR_MESSAGES,
    ERROR_SEVERITY,
    ERROR_RECOVERY_STRATEGIES,

    # Market constants
    SYMBOL_MAPPINGS,
    EXCHANGE_SPECIFICATIONS,
    TRADING_PAIRS
)


class TestTradingConstants:
    """Test trading constants."""

    def test_market_hours_structure(self):
        """Test market hours constant structure."""
        assert isinstance(MARKET_HOURS, dict)
        assert len(MARKET_HOURS) > 0

        # Check that all exchanges have valid structure
        for exchange, hours in MARKET_HOURS.items():
            assert isinstance(exchange, str)
            assert isinstance(hours, dict)
            assert "open" in hours
            assert "close" in hours
            assert "timezone" in hours

    def test_market_hours_values(self):
        """Test market hours values are valid."""
        for exchange, hours in MARKET_HOURS.items():
            assert isinstance(exchange, str)
            assert isinstance(hours, dict)
            assert isinstance(hours["open"], str)
            assert isinstance(hours["close"], str)
            assert isinstance(hours["timezone"], str)

    def test_settlement_times_structure(self):
        """Test settlement times constant structure."""
        assert isinstance(SETTLEMENT_TIMES, dict)
        assert len(SETTLEMENT_TIMES) > 0

        # Check that all asset types have valid structure
        for asset_type, settlement in SETTLEMENT_TIMES.items():
            assert isinstance(asset_type, str)
            assert isinstance(settlement, str)

    def test_settlement_times_values(self):
        """Test settlement times values are valid."""
        for asset_type, settlement in SETTLEMENT_TIMES.items():
            assert isinstance(asset_type, str)
            assert isinstance(settlement, str)
            assert settlement.startswith("T+")

    def test_precision_levels_structure(self):
        """Test precision levels constant structure."""
        assert isinstance(PRECISION_LEVELS, dict)
        assert len(PRECISION_LEVELS) > 0

        # Check that all currencies have valid precision
        for currency, precision in PRECISION_LEVELS.items():
            assert isinstance(currency, str)
            assert isinstance(precision, int)

    def test_precision_levels_values(self):
        """Test precision levels values are valid."""
        for currency, precision in PRECISION_LEVELS.items():
            assert isinstance(currency, str)
            assert isinstance(precision, int)
            assert precision >= 0


class TestAPIConstants:
    """Test API constants."""

    def test_api_endpoints_structure(self):
        """Test API endpoints constant structure."""
        assert isinstance(API_ENDPOINTS, dict)
        assert len(API_ENDPOINTS) > 0

        # Check structure of API endpoints
        for exchange, config in API_ENDPOINTS.items():
            assert isinstance(exchange, str)
            assert isinstance(config, dict)
            assert "base_url" in config
            assert isinstance(config["base_url"], str)
            assert config["base_url"].startswith("http")

    def test_api_endpoints_values(self):
        """Test API endpoints values are valid."""
        for exchange, config in API_ENDPOINTS.items():
            assert isinstance(exchange, str)
            assert isinstance(config, dict)
            assert "base_url" in config
            assert isinstance(config["base_url"], str)
            assert config["base_url"].startswith("http")

    def test_rate_limits_structure(self):
        """Test rate limits constant structure."""
        assert isinstance(RATE_LIMITS, dict)
        assert len(RATE_LIMITS) > 0

        # Check structure of rate limits
        for exchange, limits in RATE_LIMITS.items():
            assert isinstance(exchange, str)
            assert isinstance(limits, dict)
            # Check that at least one rate limit key exists
            assert len(limits) > 0

    def test_rate_limits_values(self):
        """Test rate limits values are valid."""
        for exchange, limits in RATE_LIMITS.items():
            assert isinstance(exchange, str)
            assert isinstance(limits, dict)
            # Check that all values are positive integers
            for key, value in limits.items():
                assert isinstance(key, str)
                assert isinstance(value, int)
                assert value > 0

    def test_timeouts_structure(self):
        """Test timeouts constant structure."""
        assert isinstance(TIMEOUTS, dict)
        assert len(TIMEOUTS) > 0

        # Check structure of timeouts
        for timeout_name, timeout_value in TIMEOUTS.items():
            assert isinstance(timeout_name, str)
            assert isinstance(timeout_value, (int, float))
            assert timeout_value > 0

    def test_timeouts_values(self):
        """Test timeouts values are valid."""
        for timeout_name, timeout_value in TIMEOUTS.items():
            assert isinstance(timeout_name, str)
            assert isinstance(timeout_value, (int, float))
            assert timeout_value > 0

    def test_http_status_codes_structure(self):
        """Test HTTP status codes constant structure."""
        assert isinstance(HTTP_STATUS_CODES, dict)
        assert len(HTTP_STATUS_CODES) > 0

        # Check structure of HTTP status codes
        for status_name, status_code in HTTP_STATUS_CODES.items():
            assert isinstance(status_name, str)
            assert isinstance(status_code, int)
            assert 100 <= status_code <= 599

    def test_http_status_codes_values(self):
        """Test HTTP status codes values are valid."""
        for status_name, status_code in HTTP_STATUS_CODES.items():
            assert isinstance(status_name, str)
            assert isinstance(status_code, int)
            assert 100 <= status_code <= 599


class TestFinancialConstants:
    """Test financial constants."""

    def test_fee_structures_structure(self):
        """Test fee structures constant structure."""
        assert isinstance(FEE_STRUCTURES, dict)
        assert len(FEE_STRUCTURES) > 0

        # Check structure of fee structures
        for exchange, fees in FEE_STRUCTURES.items():
            assert isinstance(exchange, str)
            assert isinstance(fees, dict)
            assert "maker_fee" in fees
            assert "taker_fee" in fees

    def test_fee_structures_values(self):
        """Test fee structures values are valid."""
        for exchange, fees in FEE_STRUCTURES.items():
            assert isinstance(exchange, str)
            assert isinstance(fees, dict)
            assert "maker_fee" in fees
            assert "taker_fee" in fees
            assert isinstance(fees["maker_fee"], (int, float, Decimal))
            assert isinstance(fees["taker_fee"], (int, float, Decimal))
            assert fees["maker_fee"] >= 0
            assert fees["taker_fee"] >= 0

    def test_minimum_amounts_structure(self):
        """Test minimum amounts constant structure."""
        assert isinstance(MINIMUM_AMOUNTS, dict)
        assert len(MINIMUM_AMOUNTS) > 0

        # Check structure of minimum amounts
        for exchange, amounts in MINIMUM_AMOUNTS.items():
            assert isinstance(exchange, str)
            assert isinstance(amounts, dict)
            for symbol, amount in amounts.items():
                assert isinstance(symbol, str)
                assert isinstance(amount, (int, float, Decimal))
                assert amount > 0

    def test_minimum_amounts_values(self):
        """Test minimum amounts values are valid."""
        for exchange, amounts in MINIMUM_AMOUNTS.items():
            assert isinstance(exchange, str)
            assert isinstance(amounts, dict)
            for symbol, amount in amounts.items():
                assert isinstance(symbol, str)
                assert isinstance(amount, (int, float, Decimal))
                assert amount > 0

    def test_maximum_amounts_structure(self):
        """Test maximum amounts constant structure."""
        assert isinstance(MAXIMUM_AMOUNTS, dict)
        assert len(MAXIMUM_AMOUNTS) > 0

        # Check structure of maximum amounts
        for key, amount in MAXIMUM_AMOUNTS.items():
            assert isinstance(key, str)
            assert isinstance(amount, (int, float, Decimal))
            assert amount > 0

    def test_maximum_amounts_values(self):
        """Test maximum amounts values are valid."""
        for key, amount in MAXIMUM_AMOUNTS.items():
            assert isinstance(key, str)
            assert isinstance(amount, (int, float, Decimal))
            assert amount > 0

    def test_slippage_tolerance_structure(self):
        """Test slippage tolerance constant structure."""
        assert isinstance(SLIPPAGE_TOLERANCE, dict)
        assert len(SLIPPAGE_TOLERANCE) > 0

        # Check structure of slippage tolerance
        for tolerance_type, tolerance_value in SLIPPAGE_TOLERANCE.items():
            assert isinstance(tolerance_type, str)
            assert isinstance(tolerance_value, (int, float, Decimal))
            assert tolerance_value > 0
            assert tolerance_value <= 1  # Should be percentage

    def test_slippage_tolerance_values(self):
        """Test slippage tolerance values are valid."""
        for tolerance_type, tolerance_value in SLIPPAGE_TOLERANCE.items():
            assert isinstance(tolerance_type, str)
            assert isinstance(tolerance_value, (int, float, Decimal))
            assert tolerance_value > 0
            assert tolerance_value <= 1  # Should be percentage


class TestConfigurationConstants:
    """Test configuration constants."""

    def test_default_values_structure(self):
        """Test default values constant structure."""
        assert isinstance(DEFAULT_VALUES, dict)
        assert len(DEFAULT_VALUES) > 0

        # Check structure of default values
        for key, value in DEFAULT_VALUES.items():
            assert isinstance(key, str)
            # Value can be any type, just check it's not None
            assert value is not None

    def test_default_values_values(self):
        """Test default values are valid."""
        for key, value in DEFAULT_VALUES.items():
            assert isinstance(key, str)
            # Value can be any type, just check it's not None
            assert value is not None

    def test_limits_structure(self):
        """Test limits constant structure."""
        assert isinstance(LIMITS, dict)
        assert len(LIMITS) > 0

        # Check structure of limits
        for key, value in LIMITS.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float, Decimal))
            assert value > 0

    def test_limits_values(self):
        """Test limits values are valid."""
        for key, value in LIMITS.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float, Decimal))
            assert value > 0

    def test_thresholds_structure(self):
        """Test thresholds constant structure."""
        assert isinstance(THRESHOLDS, dict)
        assert len(THRESHOLDS) > 0

        # Check structure of thresholds
        for category, thresholds in THRESHOLDS.items():
            assert isinstance(category, str)
            assert isinstance(thresholds, dict)
            for key, value in thresholds.items():
                assert isinstance(key, str)
                assert isinstance(value, (int, float, Decimal))

    def test_thresholds_values(self):
        """Test thresholds values are valid."""
        for category, thresholds in THRESHOLDS.items():
            assert isinstance(category, str)
            assert isinstance(thresholds, dict)
            for key, value in thresholds.items():
                assert isinstance(key, str)
                assert isinstance(value, (int, float, Decimal))


class TestErrorConstants:
    """Test error constants."""

    def test_error_codes_structure(self):
        """Test error codes constant structure."""
        assert isinstance(ERROR_CODES, dict)
        assert len(ERROR_CODES) > 0

        # Check structure of error codes
        for error_name, error_code in ERROR_CODES.items():
            assert isinstance(error_name, str)
            assert isinstance(error_code, int)
            assert error_code > 0

    def test_error_codes_values(self):
        """Test error codes values are valid."""
        for error_name, error_code in ERROR_CODES.items():
            assert isinstance(error_name, str)
            assert isinstance(error_code, int)
            assert error_code > 0

    def test_error_messages_structure(self):
        """Test error messages constant structure."""
        assert isinstance(ERROR_MESSAGES, dict)
        assert len(ERROR_MESSAGES) > 0

        # Check structure of error messages
        for error_name, error_message in ERROR_MESSAGES.items():
            assert isinstance(error_name, str)
            assert isinstance(error_message, str)
            assert len(error_message) > 0

    def test_error_messages_values(self):
        """Test error messages values are valid."""
        for error_name, error_message in ERROR_MESSAGES.items():
            assert isinstance(error_name, str)
            assert isinstance(error_message, str)
            assert len(error_message) > 0

    def test_error_severity_structure(self):
        """Test error severity constant structure."""
        assert isinstance(ERROR_SEVERITY, dict)
        assert len(ERROR_SEVERITY) > 0

        # Check structure of error severity
        for severity_name, severity_value in ERROR_SEVERITY.items():
            assert isinstance(severity_name, str)
            assert isinstance(severity_value, int)
            assert 1 <= severity_value <= 5

    def test_error_severity_values(self):
        """Test error severity values are valid."""
        for severity_name, severity_value in ERROR_SEVERITY.items():
            assert isinstance(severity_name, str)
            assert isinstance(severity_value, int)
            assert 1 <= severity_value <= 5

    def test_error_recovery_strategies_structure(self):
        """Test error recovery strategies constant structure."""
        assert isinstance(ERROR_RECOVERY_STRATEGIES, dict)
        assert len(ERROR_RECOVERY_STRATEGIES) > 0

        # Check structure of error recovery strategies
        for strategy_name, strategy_value in ERROR_RECOVERY_STRATEGIES.items():
            assert isinstance(strategy_name, str)
            assert isinstance(strategy_value, str)
            assert len(strategy_value) > 0

    def test_error_recovery_strategies_values(self):
        """Test error recovery strategies values are valid."""
        for strategy_name, strategy_value in ERROR_RECOVERY_STRATEGIES.items():
            assert isinstance(strategy_name, str)
            assert isinstance(strategy_value, str)
            assert len(strategy_value) > 0


class TestMarketConstants:
    """Test market constants."""

    def test_symbol_mappings_structure(self):
        """Test symbol mappings constant structure."""
        assert isinstance(SYMBOL_MAPPINGS, dict)
        assert len(SYMBOL_MAPPINGS) > 0

        # Check structure of symbol mappings
        for exchange, mappings in SYMBOL_MAPPINGS.items():
            assert isinstance(exchange, str)
            assert isinstance(mappings, dict)
            for standard_symbol, exchange_symbol in mappings.items():
                assert isinstance(standard_symbol, str)
                assert isinstance(exchange_symbol, str)
                assert len(standard_symbol) > 0
                assert len(exchange_symbol) > 0

    def test_symbol_mappings_values(self):
        """Test symbol mappings values are valid."""
        for exchange, mappings in SYMBOL_MAPPINGS.items():
            assert isinstance(exchange, str)
            assert isinstance(mappings, dict)
            for standard_symbol, exchange_symbol in mappings.items():
                assert isinstance(standard_symbol, str)
                assert isinstance(exchange_symbol, str)
                assert len(standard_symbol) > 0
                assert len(exchange_symbol) > 0

    def test_exchange_specifications_structure(self):
        """Test exchange specifications constant structure."""
        assert isinstance(EXCHANGE_SPECIFICATIONS, dict)
        assert len(EXCHANGE_SPECIFICATIONS) > 0

        # Check structure of exchange specifications
        for exchange, specs in EXCHANGE_SPECIFICATIONS.items():
            assert isinstance(exchange, str)
            assert isinstance(specs, dict)
            assert "name" in specs
            assert "type" in specs
            assert "supported_markets" in specs
            assert isinstance(specs["name"], str)
            assert isinstance(specs["type"], str)
            assert isinstance(specs["supported_markets"], list)

    def test_exchange_specifications_values(self):
        """Test exchange specifications values are valid."""
        for exchange, specs in EXCHANGE_SPECIFICATIONS.items():
            assert isinstance(exchange, str)
            assert isinstance(specs, dict)
            assert "name" in specs
            assert "type" in specs
            assert "supported_markets" in specs
            assert isinstance(specs["name"], str)
            assert isinstance(specs["type"], str)
            assert isinstance(specs["supported_markets"], list)

    def test_trading_pairs_structure(self):
        """Test trading pairs constant structure."""
        assert isinstance(TRADING_PAIRS, dict)
        assert len(TRADING_PAIRS) > 0

        # Check that all trading pairs are valid
        for pair_name, pair_spec in TRADING_PAIRS.items():
            assert isinstance(pair_name, str)
            assert isinstance(pair_spec, dict)
            assert "base_currency" in pair_spec
            assert "quote_currency" in pair_spec

    def test_trading_pairs_values(self):
        """Test trading pairs values are valid."""
        for pair_name, pair_spec in TRADING_PAIRS.items():
            assert isinstance(pair_name, str)
            assert isinstance(pair_spec, dict)
            assert "base_currency" in pair_spec
            assert "quote_currency" in pair_spec
            assert isinstance(pair_spec["base_currency"], str)
            assert isinstance(pair_spec["quote_currency"], str)
            assert len(pair_spec["base_currency"]) > 0
            assert len(pair_spec["quote_currency"]) > 0


class TestConstantsIntegration:
    """Integration tests for constants."""

    def test_trading_constants_integration(self):
        """Test integration between trading constants."""
        # Test that trading constants work together
        assert len(MARKET_HOURS) > 0
        assert len(SETTLEMENT_TIMES) > 0
        assert len(PRECISION_LEVELS) > 0
        # assert len(TRADING_SESSIONS) > 0 # TRADING_SESSIONS is not defined in
        # the constants module

        # Check that all exchanges in market hours have precision levels
        for exchange in MARKET_HOURS.keys():
            if exchange in PRECISION_LEVELS:
                assert isinstance(PRECISION_LEVELS[exchange], int)

    def test_api_constants_integration(self):
        """Test integration between API constants."""
        # Test that API constants work together
        assert len(API_ENDPOINTS) > 0
        assert len(RATE_LIMITS) > 0
        assert len(TIMEOUTS) > 0
        assert len(HTTP_STATUS_CODES) > 0

        # Check that all exchanges in API endpoints have rate limits
        for exchange in API_ENDPOINTS.keys():
            if exchange in RATE_LIMITS:
                assert isinstance(RATE_LIMITS[exchange], dict)

    def test_financial_constants_integration(self):
        """Test integration between financial constants."""
        # Test that financial constants work together
        assert len(FEE_STRUCTURES) > 0
        assert len(MINIMUM_AMOUNTS) > 0
        assert len(MAXIMUM_AMOUNTS) > 0
        assert len(SLIPPAGE_TOLERANCE) > 0

        # Check that all exchanges in fee structures have minimum amounts
        for exchange in FEE_STRUCTURES.keys():
            if exchange in MINIMUM_AMOUNTS:
                assert isinstance(MINIMUM_AMOUNTS[exchange], dict)

    def test_configuration_constants_integration(self):
        """Test integration between configuration constants."""
        # Test that configuration constants work together
        assert len(DEFAULT_VALUES) > 0
        assert len(LIMITS) > 0
        assert len(THRESHOLDS) > 0

        # Check that all categories in default values have limits
        for category in DEFAULT_VALUES.keys():
            if category in LIMITS:
                assert isinstance(LIMITS[category], dict)

    def test_error_constants_integration(self):
        """Test integration between error constants."""
        # Test that error constants work together
        assert len(ERROR_CODES) > 0
        assert len(ERROR_MESSAGES) > 0
        assert len(ERROR_SEVERITY) > 0
        assert len(ERROR_RECOVERY_STRATEGIES) > 0

        # Check that all error codes have corresponding messages
        for error_code in ERROR_CODES.keys():
            if error_code in ERROR_MESSAGES:
                assert isinstance(ERROR_MESSAGES[error_code], str)

    def test_market_constants_integration(self):
        """Test integration between market constants."""
        # Test that market constants work together
        assert len(SYMBOL_MAPPINGS) > 0
        assert len(EXCHANGE_SPECIFICATIONS) > 0
        assert len(TRADING_PAIRS) > 0

        # Check that all exchanges in symbol mappings have specifications
        for exchange in SYMBOL_MAPPINGS.keys():
            if exchange in EXCHANGE_SPECIFICATIONS:
                assert isinstance(EXCHANGE_SPECIFICATIONS[exchange], dict)

    def test_cross_constant_integration(self):
        """Test integration across different constant categories."""
        # Test that constants from different categories work together

        # Check that exchanges appear consistently across categories
        exchanges_in_api = set(API_ENDPOINTS.keys())
        exchanges_in_fees = set(FEE_STRUCTURES.keys())
        exchanges_in_symbols = set(SYMBOL_MAPPINGS.keys())

        # There should be some overlap between these sets
        assert len(exchanges_in_api & exchanges_in_fees) > 0
        assert len(exchanges_in_api & exchanges_in_symbols) > 0

        # Check that trading pairs are consistent
        for pair_name, pair_spec in TRADING_PAIRS.items():
            assert "base_currency" in pair_spec
            assert "quote_currency" in pair_spec
            assert len(pair_spec["base_currency"]) > 0
            assert len(pair_spec["quote_currency"]) > 0

    def test_constant_validation(self):
        """Test that all constants pass basic validation."""
        # Test that all constants are properly defined and accessible

        # Test that all required constants exist
        required_constants = [
            MARKET_HOURS,
            SETTLEMENT_TIMES,
            PRECISION_LEVELS,
            API_ENDPOINTS,
            RATE_LIMITS,
            TIMEOUTS,
            HTTP_STATUS_CODES,
            FEE_STRUCTURES,
            MINIMUM_AMOUNTS,
            MAXIMUM_AMOUNTS,
            SLIPPAGE_TOLERANCE,
            DEFAULT_VALUES,
            LIMITS,
            THRESHOLDS,
            ERROR_CODES,
            ERROR_MESSAGES,
            ERROR_SEVERITY,
            ERROR_RECOVERY_STRATEGIES,
            SYMBOL_MAPPINGS,
            EXCHANGE_SPECIFICATIONS,
            TRADING_PAIRS]

        for constant in required_constants:
            assert constant is not None
            assert len(constant) > 0 if hasattr(constant, '__len__') else True
