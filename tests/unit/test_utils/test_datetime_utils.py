"""Tests for datetime utilities module."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import pytz

from src.core.exceptions import ValidationError
from src.utils.datetime_utils import (
    convert_timezone,
    get_current_utc_timestamp,
    get_redis_key_ttl,
    get_trading_session,
    is_market_open,
    parse_datetime,
    parse_timeframe,
    to_timestamp,
)


class TestGetCurrentUtcTimestamp:
    """Test get_current_utc_timestamp function."""

    def test_get_current_utc_timestamp_returns_utc(self):
        """Test get_current_utc_timestamp returns UTC timezone."""
        result = get_current_utc_timestamp()
        assert result.tzinfo == timezone.utc

    def test_get_current_utc_timestamp_recent(self):
        """Test get_current_utc_timestamp returns recent time."""
        before = datetime.now(timezone.utc)
        result = get_current_utc_timestamp()
        after = datetime.now(timezone.utc)
        
        assert before <= result <= after


class TestToTimestamp:
    """Test to_timestamp function."""

    def test_to_timestamp_with_timezone(self):
        """Test to_timestamp with timezone-aware datetime."""
        dt = datetime(2023, 12, 25, 15, 30, 45, tzinfo=timezone.utc)
        result = to_timestamp(dt)
        assert result == "2023-12-25 15:30:45"

    def test_to_timestamp_without_timezone(self):
        """Test to_timestamp with naive datetime (adds UTC)."""
        dt = datetime(2023, 12, 25, 15, 30, 45)
        result = to_timestamp(dt)
        assert result == "2023-12-25 15:30:45"

    def test_to_timestamp_custom_format(self):
        """Test to_timestamp with custom format."""
        dt = datetime(2023, 12, 25, 15, 30, 45, tzinfo=timezone.utc)
        result = to_timestamp(dt, "%Y/%m/%d %H:%M")
        assert result == "2023/12/25 15:30"

    def test_to_timestamp_different_timezone(self):
        """Test to_timestamp with different timezone."""
        eastern = pytz.timezone("US/Eastern")
        dt = datetime(2023, 12, 25, 15, 30, 45, tzinfo=eastern)
        result = to_timestamp(dt)
        # Should format the time in the original timezone
        assert "2023-12-25 15:30:45" == result


class TestParseTimeframe:
    """Test parse_timeframe function."""

    def test_parse_timeframe_seconds(self):
        """Test parse_timeframe with seconds."""
        assert parse_timeframe("30s") == 30
        assert parse_timeframe("1s") == 1

    def test_parse_timeframe_minutes(self):
        """Test parse_timeframe with minutes."""
        assert parse_timeframe("5m") == 300  # 5 * 60
        assert parse_timeframe("15m") == 900  # 15 * 60

    def test_parse_timeframe_hours(self):
        """Test parse_timeframe with hours."""
        assert parse_timeframe("1h") == 3600  # 1 * 3600
        assert parse_timeframe("4h") == 14400  # 4 * 3600

    def test_parse_timeframe_days(self):
        """Test parse_timeframe with days."""
        assert parse_timeframe("1d") == 86400  # 1 * 86400
        assert parse_timeframe("7d") == 604800  # 7 * 86400

    def test_parse_timeframe_case_insensitive(self):
        """Test parse_timeframe is case insensitive."""
        assert parse_timeframe("1H") == 3600
        assert parse_timeframe("5M") == 300
        assert parse_timeframe("1D") == 86400

    def test_parse_timeframe_empty_input(self):
        """Test parse_timeframe with empty input."""
        with pytest.raises(ValidationError, match="Timeframe cannot be empty"):
            parse_timeframe("")

    def test_parse_timeframe_invalid_format(self):
        """Test parse_timeframe with invalid format."""
        with pytest.raises(ValidationError, match="Invalid timeframe format"):
            parse_timeframe("invalid")

        with pytest.raises(ValidationError, match="Invalid timeframe format"):
            parse_timeframe("1x")  # Invalid unit

        with pytest.raises(ValidationError, match="Invalid timeframe format"):
            parse_timeframe("m5")  # Wrong order

    def test_parse_timeframe_no_number(self):
        """Test parse_timeframe with no number."""
        with pytest.raises(ValidationError, match="Invalid timeframe format"):
            parse_timeframe("m")

    def test_parse_timeframe_large_numbers(self):
        """Test parse_timeframe with large numbers."""
        assert parse_timeframe("100s") == 100
        assert parse_timeframe("999m") == 59940  # 999 * 60


class TestGetTradingSession:
    """Test get_trading_session function."""

    def test_get_trading_session_crypto_low_activity(self):
        """Test get_trading_session for crypto during low activity hours."""
        dt = datetime(2023, 12, 25, 3, 0, 0, tzinfo=timezone.utc)  # 3 AM UTC
        assert get_trading_session(dt, "binance") == "low_activity"
        assert get_trading_session(dt, "okx") == "low_activity"
        assert get_trading_session(dt, "coinbase") == "low_activity"

    def test_get_trading_session_crypto_asian_session(self):
        """Test get_trading_session for crypto during Asian session."""
        dt = datetime(2023, 12, 25, 10, 0, 0, tzinfo=timezone.utc)  # 10 AM UTC
        assert get_trading_session(dt, "binance") == "asian_session"

    def test_get_trading_session_crypto_european_session(self):
        """Test get_trading_session for crypto during European session."""
        dt = datetime(2023, 12, 25, 16, 0, 0, tzinfo=timezone.utc)  # 4 PM UTC
        assert get_trading_session(dt, "binance") == "european_session"

    def test_get_trading_session_crypto_american_session(self):
        """Test get_trading_session for crypto during American session."""
        dt = datetime(2023, 12, 25, 23, 0, 0, tzinfo=timezone.utc)  # 11 PM UTC
        assert get_trading_session(dt, "binance") == "american_session"

    def test_get_trading_session_traditional_regular(self):
        """Test get_trading_session for traditional market regular hours."""
        dt = datetime(2023, 12, 25, 12, 0, 0, tzinfo=timezone.utc)  # 12 PM UTC
        assert get_trading_session(dt, "nasdaq") == "regular"

    def test_get_trading_session_traditional_pre_market(self):
        """Test get_trading_session for traditional market pre-market."""
        dt = datetime(2023, 12, 25, 7, 0, 0, tzinfo=timezone.utc)  # 7 AM UTC
        assert get_trading_session(dt, "nasdaq") == "pre_market"

    def test_get_trading_session_traditional_post_market(self):
        """Test get_trading_session for traditional market post-market."""
        dt = datetime(2023, 12, 25, 18, 0, 0, tzinfo=timezone.utc)  # 6 PM UTC
        assert get_trading_session(dt, "nasdaq") == "post_market"

    def test_get_trading_session_traditional_closed(self):
        """Test get_trading_session for traditional market closed."""
        dt = datetime(2023, 12, 25, 2, 0, 0, tzinfo=timezone.utc)  # 2 AM UTC
        assert get_trading_session(dt, "nasdaq") == "closed"

    def test_get_trading_session_naive_datetime(self):
        """Test get_trading_session with naive datetime (adds UTC)."""
        dt = datetime(2023, 12, 25, 10, 0, 0)  # No timezone
        result = get_trading_session(dt, "binance")
        assert result == "asian_session"


class TestIsMarketOpen:
    """Test is_market_open function."""

    def test_is_market_open_crypto_always_open(self):
        """Test is_market_open for crypto exchanges (always open)."""
        dt_weekday = datetime(2023, 12, 25, 15, 0, 0, tzinfo=timezone.utc)  # Monday
        dt_weekend = datetime(2023, 12, 30, 15, 0, 0, tzinfo=timezone.utc)  # Saturday
        
        assert is_market_open(dt_weekday, "binance") is True
        assert is_market_open(dt_weekend, "binance") is True
        assert is_market_open(dt_weekday, "okx") is True
        assert is_market_open(dt_weekend, "coinbase") is True

    def test_is_market_open_traditional_weekday_hours(self):
        """Test is_market_open for traditional market during weekday hours."""
        dt = datetime(2023, 12, 25, 12, 0, 0, tzinfo=timezone.utc)  # Monday 12 PM
        assert is_market_open(dt, "nasdaq") is True

    def test_is_market_open_traditional_weekend(self):
        """Test is_market_open for traditional market during weekend."""
        dt_saturday = datetime(2023, 12, 30, 12, 0, 0, tzinfo=timezone.utc)  # Saturday
        dt_sunday = datetime(2023, 12, 31, 12, 0, 0, tzinfo=timezone.utc)  # Sunday
        
        assert is_market_open(dt_saturday, "nasdaq") is False
        assert is_market_open(dt_sunday, "nasdaq") is False

    def test_is_market_open_traditional_after_hours(self):
        """Test is_market_open for traditional market after hours."""
        dt_early = datetime(2023, 12, 25, 7, 0, 0, tzinfo=timezone.utc)  # Monday 7 AM
        dt_late = datetime(2023, 12, 25, 18, 0, 0, tzinfo=timezone.utc)  # Monday 6 PM
        
        assert is_market_open(dt_early, "nasdaq") is False
        assert is_market_open(dt_late, "nasdaq") is False


class TestConvertTimezone:
    """Test convert_timezone function."""

    def test_convert_timezone_utc_to_eastern(self):
        """Test convert_timezone from UTC to Eastern."""
        dt = datetime(2023, 12, 25, 15, 30, 0, tzinfo=timezone.utc)
        result = convert_timezone(dt, "US/Eastern")
        
        assert result.tzinfo.zone == "US/Eastern"
        # During winter, Eastern is UTC-5
        assert result.hour == 10  # 15 - 5 = 10

    def test_convert_timezone_naive_datetime(self):
        """Test convert_timezone with naive datetime (assumes UTC)."""
        dt = datetime(2023, 12, 25, 15, 30, 0)  # No timezone
        result = convert_timezone(dt, "US/Pacific")
        
        assert result.tzinfo.zone == "US/Pacific"
        # Should assume input was UTC and convert

    def test_convert_timezone_to_utc(self):
        """Test convert_timezone to UTC."""
        eastern = pytz.timezone("US/Eastern")
        dt = datetime(2023, 12, 25, 10, 30, 0, tzinfo=eastern)
        result = convert_timezone(dt, "UTC")
        
        assert result.tzinfo.zone == "UTC"
        assert result.hour == 15  # 10 + 5 = 15

    def test_convert_timezone_invalid_timezone(self):
        """Test convert_timezone with invalid timezone."""
        dt = datetime(2023, 12, 25, 15, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValidationError, match="Invalid timezone"):
            convert_timezone(dt, "Invalid/Timezone")

    def test_convert_timezone_empty_timezone(self):
        """Test convert_timezone with empty timezone string."""
        dt = datetime(2023, 12, 25, 15, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValidationError, match="Timezone must be a non-empty string"):
            convert_timezone(dt, "")

        with pytest.raises(ValidationError, match="Timezone must be a non-empty string"):
            convert_timezone(dt, "   ")

    def test_convert_timezone_non_string_timezone(self):
        """Test convert_timezone with non-string timezone."""
        dt = datetime(2023, 12, 25, 15, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValidationError, match="Timezone must be a non-empty string"):
            convert_timezone(dt, None)


class TestParseDatetime:
    """Test parse_datetime function."""

    def test_parse_datetime_with_format(self):
        """Test parse_datetime with explicit format."""
        result = parse_datetime("2023-12-25 15:30:45", "%Y-%m-%d %H:%M:%S")
        assert result == datetime(2023, 12, 25, 15, 30, 45)

    def test_parse_datetime_common_formats(self):
        """Test parse_datetime with common formats (no explicit format)."""
        # ISO format
        result1 = parse_datetime("2023-12-25T15:30:45")
        assert result1 == datetime(2023, 12, 25, 15, 30, 45)

        # ISO with Z
        result2 = parse_datetime("2023-12-25T15:30:45Z")
        assert result2 == datetime(2023, 12, 25, 15, 30, 45)

        # Date only
        result3 = parse_datetime("2023-12-25")
        assert result3 == datetime(2023, 12, 25, 0, 0, 0)

        # Time only
        result4 = parse_datetime("15:30:45")
        assert result4.time() == datetime(1900, 1, 1, 15, 30, 45).time()

    def test_parse_datetime_invalid_with_format(self):
        """Test parse_datetime with invalid datetime for given format."""
        with pytest.raises(ValidationError, match="Cannot parse datetime"):
            parse_datetime("2023-12-25", "%H:%M:%S")

    def test_parse_datetime_invalid_no_format(self):
        """Test parse_datetime with invalid datetime and no format."""
        with pytest.raises(ValidationError, match="Cannot parse datetime string"):
            parse_datetime("invalid-date-string")

    def test_parse_datetime_empty_string(self):
        """Test parse_datetime with empty string."""
        with pytest.raises(ValidationError, match="Cannot parse datetime string"):
            parse_datetime("")


class TestGetRedisKeyTtl:
    """Test get_redis_key_ttl function."""

    def test_get_redis_key_ttl_metrics(self):
        """Test get_redis_key_ttl for metrics keys."""
        assert get_redis_key_ttl("system_metrics") == 300
        assert get_redis_key_ttl("trading_METRICS_data") == 300

    def test_get_redis_key_ttl_cache(self):
        """Test get_redis_key_ttl for cache keys."""
        assert get_redis_key_ttl("user_cache") == 1800
        assert get_redis_key_ttl("data_CACHE_store") == 1800

    def test_get_redis_key_ttl_session(self):
        """Test get_redis_key_ttl for session keys."""
        assert get_redis_key_ttl("user_session") == 86400
        assert get_redis_key_ttl("auth_SESSION_token") == 86400

    def test_get_redis_key_ttl_temporary(self):
        """Test get_redis_key_ttl for temporary keys."""
        assert get_redis_key_ttl("temp_data") == 60
        assert get_redis_key_ttl("tmp_file") == 60

    def test_get_redis_key_ttl_state(self):
        """Test get_redis_key_ttl for state keys."""
        assert get_redis_key_ttl("bot_state") == 600
        assert get_redis_key_ttl("trading_STATE_info") == 600

    def test_get_redis_key_ttl_orderbook(self):
        """Test get_redis_key_ttl for orderbook keys."""
        assert get_redis_key_ttl("orderbook_data") == 5
        assert get_redis_key_ttl("order_book_snapshot") == 5

    def test_get_redis_key_ttl_price_ticker(self):
        """Test get_redis_key_ttl for price/ticker keys."""
        assert get_redis_key_ttl("price_data") == 10
        assert get_redis_key_ttl("ticker_info") == 10

    def test_get_redis_key_ttl_position(self):
        """Test get_redis_key_ttl for position keys."""
        assert get_redis_key_ttl("position_data") == 300

    def test_get_redis_key_ttl_balance(self):
        """Test get_redis_key_ttl for balance keys."""
        assert get_redis_key_ttl("balance_info") == 300

    def test_get_redis_key_ttl_trade(self):
        """Test get_redis_key_ttl for trade keys."""
        assert get_redis_key_ttl("trade_history") == 3600

    def test_get_redis_key_ttl_candle_kline(self):
        """Test get_redis_key_ttl for candle/kline keys."""
        assert get_redis_key_ttl("candle_data") == 60
        assert get_redis_key_ttl("kline_info") == 60

    def test_get_redis_key_ttl_lock(self):
        """Test get_redis_key_ttl for lock keys."""
        assert get_redis_key_ttl("distributed_lock") == 30

    def test_get_redis_key_ttl_rate_limit(self):
        """Test get_redis_key_ttl for rate limit keys."""
        assert get_redis_key_ttl("rate_limit_counter") == 60

    def test_get_redis_key_ttl_default(self):
        """Test get_redis_key_ttl with default TTL."""
        assert get_redis_key_ttl("unknown_key") == 3600  # Default
        assert get_redis_key_ttl("random_data") == 3600

    def test_get_redis_key_ttl_custom_default(self):
        """Test get_redis_key_ttl with custom default TTL."""
        assert get_redis_key_ttl("unknown_key", 7200) == 7200

    def test_get_redis_key_ttl_case_insensitive(self):
        """Test get_redis_key_ttl is case insensitive."""
        assert get_redis_key_ttl("METRICS_data") == 300
        assert get_redis_key_ttl("Cache_Store") == 1800
        assert get_redis_key_ttl("SESSION_info") == 86400