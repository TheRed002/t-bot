"""Tests for string utilities module."""

import pytest

from src.core.exceptions import ValidationError
from src.utils.string_utils import (
    camel_to_snake,
    extract_numbers,
    generate_hash,
    normalize_symbol,
    parse_trading_pair,
    snake_to_camel,
    truncate,
    validate_email,
)


class TestNormalizeSymbol:
    """Test normalize_symbol function."""

    def test_normalize_symbol_valid_input(self):
        """Test normalize_symbol with valid inputs."""
        assert normalize_symbol("btcusdt") == "BTCUSDT"
        assert normalize_symbol("ETH/USDT") == "ETH/USDT"
        assert normalize_symbol("BTC-USDC") == "BTC-USDC"
        assert normalize_symbol("eth_btc") == "ETH_BTC"
        assert normalize_symbol("  BTC123  ") == "BTC123"

    def test_normalize_symbol_empty_input(self):
        """Test normalize_symbol with empty input."""
        with pytest.raises(ValidationError, match="Symbol cannot be empty"):
            normalize_symbol("")

    def test_normalize_symbol_none_input(self):
        """Test normalize_symbol with None input."""
        with pytest.raises(ValidationError, match="Symbol cannot be empty"):
            normalize_symbol(None)

    def test_normalize_symbol_invalid_characters(self):
        """Test normalize_symbol with invalid characters."""
        with pytest.raises(ValidationError, match="Symbol contains no valid characters"):
            normalize_symbol("@#$%^&*()")

    def test_normalize_symbol_whitespace_only(self):
        """Test normalize_symbol with whitespace only."""
        with pytest.raises(ValidationError, match="Symbol contains no valid characters"):
            normalize_symbol("   ")

    def test_normalize_symbol_mixed_valid_invalid(self):
        """Test normalize_symbol with mixed valid/invalid characters."""
        assert normalize_symbol("BTC@USDT#") == "BTCUSDT"
        assert normalize_symbol("ETH$123") == "ETH123"


class TestParseTradingPair:
    """Test parse_trading_pair function."""

    def test_parse_trading_pair_common_quotes(self):
        """Test parse_trading_pair with common quote currencies."""
        assert parse_trading_pair("BTCUSDT") == ("BTC", "USDT")
        assert parse_trading_pair("ETHUSDC") == ("ETH", "USDC")
        assert parse_trading_pair("ADABTC") == ("ADA", "BTC")
        assert parse_trading_pair("DOTETH") == ("DOT", "ETH")
        assert parse_trading_pair("LINKBNB") == ("LINK", "BNB")

    def test_parse_trading_pair_with_separators(self):
        """Test parse_trading_pair with separators."""
        assert parse_trading_pair("BTC/USDT") == ("BTC", "USDT")
        assert parse_trading_pair("ETH-USDC") == ("ETH", "USDC")
        assert parse_trading_pair("ADA_BTC") == ("ADA", "BTC")

    def test_parse_trading_pair_lowercase(self):
        """Test parse_trading_pair with lowercase input."""
        assert parse_trading_pair("btcusdt") == ("BTC", "USDT")
        assert parse_trading_pair("eth/usdc") == ("ETH", "USDC")

    def test_parse_trading_pair_fallback_split(self):
        """Test parse_trading_pair with fallback split logic."""
        # Should split at position 3 for 6+ character pairs
        assert parse_trading_pair("ABCDEF") == ("ABC", "DEF")
        assert parse_trading_pair("XYZABC") == ("XYZ", "ABC")

    def test_parse_trading_pair_empty_input(self):
        """Test parse_trading_pair with empty input."""
        with pytest.raises(ValidationError, match="Trading pair cannot be empty"):
            parse_trading_pair("")

    def test_parse_trading_pair_invalid_format(self):
        """Test parse_trading_pair with invalid format."""
        with pytest.raises(ValidationError, match="Cannot parse trading pair"):
            parse_trading_pair("ABCDE")  # Too short for fallback split

        with pytest.raises(ValidationError, match="Cannot parse trading pair"):
            parse_trading_pair("AB")

    def test_parse_trading_pair_no_base_currency(self):
        """Test parse_trading_pair where base currency would be empty."""
        with pytest.raises(ValidationError, match="Cannot parse trading pair"):
            parse_trading_pair("USDT")  # Only quote currency


class TestGenerateHash:
    """Test generate_hash function."""

    def test_generate_hash_consistent_output(self):
        """Test generate_hash produces consistent output."""
        data = "test_data"
        hash1 = generate_hash(data)
        hash2 = generate_hash(data)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64-character hex string

    def test_generate_hash_different_inputs(self):
        """Test generate_hash produces different outputs for different inputs."""
        hash1 = generate_hash("data1")
        hash2 = generate_hash("data2")
        assert hash1 != hash2

    def test_generate_hash_empty_string(self):
        """Test generate_hash with empty string."""
        result = generate_hash("")
        assert len(result) == 64
        assert result.isalnum()  # Should be valid hex

    def test_generate_hash_special_characters(self):
        """Test generate_hash with special characters."""
        result = generate_hash("!@#$%^&*()_+")
        assert len(result) == 64
        assert result.isalnum()


class TestValidateEmail:
    """Test validate_email function."""

    def test_validate_email_valid_emails(self):
        """Test validate_email with valid email addresses."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org",
            "user123@example123.com",
            "user_name@example-domain.com",
        ]
        for email in valid_emails:
            assert validate_email(email) is True

    def test_validate_email_invalid_emails(self):
        """Test validate_email with invalid email addresses."""
        invalid_emails = [
            "invalid",
            "@example.com",
            "user@",
            "user@.com", 
            "user@example",
            "user @example.com",  # Space not allowed
            "",
        ]
        for email in invalid_emails:
            assert validate_email(email) is False

    def test_validate_email_edge_cases(self):
        """Test validate_email with edge cases."""
        assert validate_email("a@b.co") is True  # Minimum valid format
        assert validate_email("very.long.email.address@very.long.domain.name.com") is True


class TestExtractNumbers:
    """Test extract_numbers function."""

    def test_extract_numbers_integers(self):
        """Test extract_numbers with integers."""
        assert extract_numbers("123 456 789") == [123.0, 456.0, 789.0]
        assert extract_numbers("Price: 100") == [100.0]

    def test_extract_numbers_floats(self):
        """Test extract_numbers with floating point numbers."""
        assert extract_numbers("Price: 123.45") == [123.45]
        assert extract_numbers("Values: 1.1 2.2 3.3") == [1.1, 2.2, 3.3]

    def test_extract_numbers_negative_numbers(self):
        """Test extract_numbers with negative numbers."""
        assert extract_numbers("Loss: -123.45") == [-123.45]
        assert extract_numbers("-1 -2.5 -3") == [-1.0, -2.5, -3.0]

    def test_extract_numbers_mixed_text(self):
        """Test extract_numbers with mixed text and numbers."""
        text = "Buy 100.5 shares at $25.75 each, total: $2609.875"
        expected = [100.5, 25.75, 2609.875]
        assert extract_numbers(text) == expected

    def test_extract_numbers_no_numbers(self):
        """Test extract_numbers with text containing no numbers."""
        assert extract_numbers("No numbers here") == []
        assert extract_numbers("") == []

    def test_extract_numbers_decimal_only(self):
        """Test extract_numbers with decimal-only numbers."""
        assert extract_numbers(".5 .75 .125") == [0.5, 0.75, 0.125]


class TestCamelToSnake:
    """Test camel_to_snake function."""

    def test_camel_to_snake_simple_cases(self):
        """Test camel_to_snake with simple cases."""
        assert camel_to_snake("camelCase") == "camel_case"
        assert camel_to_snake("PascalCase") == "pascal_case"
        assert camel_to_snake("simpleWord") == "simple_word"

    def test_camel_to_snake_complex_cases(self):
        """Test camel_to_snake with complex cases."""
        assert camel_to_snake("HTTPResponseCode") == "http_response_code"
        assert camel_to_snake("XMLParser") == "xml_parser"
        assert camel_to_snake("getUserID") == "get_user_id"

    def test_camel_to_snake_already_snake_case(self):
        """Test camel_to_snake with already snake_case input."""
        assert camel_to_snake("already_snake_case") == "already_snake_case"
        assert camel_to_snake("simple") == "simple"

    def test_camel_to_snake_with_numbers(self):
        """Test camel_to_snake with numbers."""
        assert camel_to_snake("version2API") == "version2_api"
        assert camel_to_snake("user123Name") == "user123_name"

    def test_camel_to_snake_empty_string(self):
        """Test camel_to_snake with empty string."""
        assert camel_to_snake("") == ""


class TestSnakeToCamel:
    """Test snake_to_camel function."""

    def test_snake_to_camel_simple_cases(self):
        """Test snake_to_camel with simple cases."""
        assert snake_to_camel("snake_case") == "snakeCase"
        assert snake_to_camel("simple_word") == "simpleWord"
        assert snake_to_camel("user_name") == "userName"

    def test_snake_to_camel_multiple_underscores(self):
        """Test snake_to_camel with multiple underscores."""
        assert snake_to_camel("first_middle_last_name") == "firstMiddleLastName"
        assert snake_to_camel("http_response_code") == "httpResponseCode"

    def test_snake_to_camel_already_camel_case(self):
        """Test snake_to_camel with already camelCase input."""
        assert snake_to_camel("alreadyCamelCase") == "alreadyCamelCase"
        assert snake_to_camel("simple") == "simple"

    def test_snake_to_camel_with_numbers(self):
        """Test snake_to_camel with numbers."""
        assert snake_to_camel("user_123_name") == "user123Name"
        assert snake_to_camel("version_2_api") == "version2Api"

    def test_snake_to_camel_empty_string(self):
        """Test snake_to_camel with empty string."""
        assert snake_to_camel("") == ""

    def test_snake_to_camel_single_underscore(self):
        """Test snake_to_camel with single underscore components."""
        assert snake_to_camel("a_b_c") == "aBC"


class TestTruncate:
    """Test truncate function."""

    def test_truncate_no_truncation_needed(self):
        """Test truncate when text is shorter than max_length."""
        text = "Short text"
        assert truncate(text, 20) == "Short text"
        assert truncate(text, len(text)) == "Short text"

    def test_truncate_with_default_suffix(self):
        """Test truncate with default suffix."""
        text = "This is a long text that needs truncation"
        result = truncate(text, 20)
        assert len(result) == 20
        assert result.endswith("...")
        assert result == "This is a long te..."

    def test_truncate_with_custom_suffix(self):
        """Test truncate with custom suffix."""
        text = "This is a long text"
        result = truncate(text, 10, " [more]")
        assert len(result) == 10
        assert result.endswith(" [more]")

    def test_truncate_suffix_longer_than_max_length(self):
        """Test truncate when suffix is longer than max_length."""
        text = "Long text"
        result = truncate(text, 2, "...")
        assert result == ".."
        assert len(result) == 2

    def test_truncate_max_length_equal_to_suffix(self):
        """Test truncate when max_length equals suffix length."""
        text = "Long text"
        result = truncate(text, 3, "...")
        assert result == "..."
        assert len(result) == 3

    def test_truncate_empty_text(self):
        """Test truncate with empty text."""
        assert truncate("", 10) == ""

    def test_truncate_zero_max_length(self):
        """Test truncate with zero max_length."""
        text = "Some text"
        result = truncate(text, 0, "...")
        assert result == ""

    def test_truncate_exact_length_boundary(self):
        """Test truncate at exact boundary conditions."""
        text = "12345"
        assert truncate(text, 5) == "12345"  # Exactly max_length
        assert truncate(text, 4, "...") == "1..."  # Needs truncation