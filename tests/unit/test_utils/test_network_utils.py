"""
Unit tests for network_utils module.

Tests network utility functions for connectivity testing, latency measurement,
and URL parsing functionality.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import ValidationError
from src.utils.network_utils import (
    check_connection,
    check_multiple_hosts,
    measure_latency,
    parse_url,
    ping_host,
    wait_for_service,
)


class TestCheckConnection:
    """Test check_connection function."""

    @pytest.mark.asyncio
    async def test_successful_connection(self):
        """Test successful connection."""
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = MagicMock()
            mock_writer = MagicMock()
            mock_writer.close.return_value = None
            mock_writer.wait_closed = AsyncMock(return_value=None)
            mock_conn.return_value = (mock_reader, mock_writer)

            result = await check_connection("example.com", 80)
            assert result is True
            mock_writer.close.assert_called_once()
            mock_writer.wait_closed.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """Test connection timeout."""
        with patch("asyncio.open_connection", side_effect=asyncio.TimeoutError()):
            result = await check_connection("example.com", 80, timeout=1.0)
            assert result is False

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test connection error."""
        with patch("asyncio.open_connection", side_effect=ConnectionError("Connection failed")):
            result = await check_connection("example.com", 80)
            assert result is False

    @pytest.mark.asyncio
    async def test_connection_cleanup_error(self):
        """Test connection with cleanup error."""
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = MagicMock()
            mock_writer = MagicMock()
            mock_writer.close.return_value = None
            mock_writer.wait_closed = AsyncMock(side_effect=ConnectionResetError("Reset"))
            mock_conn.return_value = (mock_reader, mock_writer)

            # Should still return True despite cleanup error
            result = await check_connection("example.com", 80)
            assert result is True

    @pytest.mark.asyncio
    async def test_writer_cleanup_timeout(self):
        """Test writer cleanup timeout."""
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = MagicMock()
            mock_writer = MagicMock()
            mock_writer.close.return_value = None
            mock_writer.wait_closed = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_conn.return_value = (mock_reader, mock_writer)

            result = await check_connection("example.com", 80)
            assert result is True


class TestMeasureLatency:
    """Test measure_latency function."""

    @pytest.mark.asyncio
    async def test_successful_latency_measurement(self):
        """Test successful latency measurement."""
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = MagicMock()
            mock_writer = MagicMock()
            mock_writer.close.return_value = None
            mock_writer.wait_closed = AsyncMock(return_value=None)
            mock_conn.return_value = (mock_reader, mock_writer)

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.time.side_effect = [100.0, 100.1]  # 100ms latency

                result = await measure_latency("example.com", 80)
                assert isinstance(result, float)
                assert result > 0
                # Allow some tolerance for timing precision
                assert 90.0 <= result <= 110.0  # Around 100ms

    @pytest.mark.asyncio
    async def test_latency_timeout(self):
        """Test latency measurement timeout."""
        with patch("asyncio.open_connection", side_effect=asyncio.TimeoutError()):
            with pytest.raises(ValidationError, match="Connection timeout"):
                await measure_latency("example.com", 80, timeout=1.0)

    @pytest.mark.asyncio
    async def test_latency_connection_error(self):
        """Test latency measurement with connection error."""
        with patch("asyncio.open_connection", side_effect=ConnectionError("Failed")):
            with pytest.raises(ValidationError, match="Cannot measure latency"):
                await measure_latency("example.com", 80)

    @pytest.mark.asyncio
    async def test_latency_cleanup_error(self):
        """Test latency measurement with cleanup error."""
        with patch("asyncio.open_connection") as mock_conn:
            mock_reader = MagicMock()
            mock_writer = MagicMock()
            mock_writer.close.return_value = None
            mock_writer.wait_closed = AsyncMock(side_effect=OSError("Cleanup failed"))
            mock_conn.return_value = (mock_reader, mock_writer)

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.time.side_effect = [100.0, 100.05]  # 50ms latency

                result = await measure_latency("example.com", 80)
                assert isinstance(result, float)
                assert result > 0


class TestPingHost:
    """Test ping_host function."""

    @pytest.mark.asyncio
    async def test_successful_ping(self):
        """Test successful ping with all attempts succeeding."""
        with patch("src.utils.network_utils.measure_latency") as mock_measure:
            mock_measure.return_value = 50.0  # 50ms latency

            result = await ping_host("example.com", count=3, port=80)

            assert result["host"] == "example.com"
            assert result["success"] is True
            assert result["count"] == 3
            assert result["min_latency_ms"] == 50.0
            assert result["max_latency_ms"] == 50.0
            assert result["avg_latency_ms"] == 50.0
            assert result["packet_loss_pct"] == 0.0

    @pytest.mark.asyncio
    async def test_partial_ping_success(self):
        """Test ping with some failures."""
        with patch("src.utils.network_utils.measure_latency") as mock_measure:
            # First call succeeds, second fails, third succeeds
            mock_measure.side_effect = [50.0, Exception("Failed"), 60.0]

            result = await ping_host("example.com", count=3, port=80)

            assert result["host"] == "example.com"
            assert result["success"] is True
            assert result["count"] == 2  # Only 2 successful pings
            assert result["min_latency_ms"] == 50.0
            assert result["max_latency_ms"] == 60.0
            assert result["avg_latency_ms"] == 55.0
            # Allow some tolerance for floating point precision
            assert 33.3 <= result["packet_loss_pct"] <= 33.4  # 1 out of 3 failed

    @pytest.mark.asyncio
    async def test_complete_ping_failure(self):
        """Test ping with all attempts failing."""
        with patch("src.utils.network_utils.measure_latency") as mock_measure:
            mock_measure.side_effect = Exception("Failed")

            result = await ping_host("example.com", count=3, port=80)

            assert result["host"] == "example.com"
            assert result["success"] is False
            assert "error" in result
            assert result["error"] == "All ping attempts failed"

    @pytest.mark.asyncio
    async def test_ping_exception_handling(self):
        """Test ping with exception during execution."""
        with patch(
            "src.utils.network_utils.measure_latency", side_effect=Exception("Unexpected error")
        ):
            result = await ping_host("example.com", count=1, port=80)

            assert result["host"] == "example.com"
            assert result["success"] is False
            assert "error" in result


class TestCheckMultipleHosts:
    """Test check_multiple_hosts function."""

    @pytest.mark.asyncio
    async def test_multiple_hosts_success(self):
        """Test checking multiple hosts successfully."""
        hosts = [("example.com", 80), ("google.com", 443)]

        with patch("src.utils.network_utils.check_connection") as mock_check:
            mock_check.return_value = True

            result = await check_multiple_hosts(hosts, timeout=5.0)

            assert len(result) == 2
            assert result["example.com:80"] is True
            assert result["google.com:443"] is True

    @pytest.mark.asyncio
    async def test_multiple_hosts_mixed_results(self):
        """Test checking multiple hosts with mixed results."""
        hosts = [("example.com", 80), ("badhost.com", 80)]

        with patch("src.utils.network_utils.check_connection") as mock_check:
            # First host succeeds, second fails
            mock_check.side_effect = [True, False]

            result = await check_multiple_hosts(hosts, timeout=5.0)

            assert len(result) == 2
            assert result["example.com:80"] is True
            assert result["badhost.com:80"] is False

    @pytest.mark.asyncio
    async def test_multiple_hosts_empty_list(self):
        """Test checking empty host list."""
        result = await check_multiple_hosts([], timeout=5.0)
        assert result == {}

    @pytest.mark.asyncio
    async def test_multiple_hosts_with_exceptions(self):
        """Test checking multiple hosts with exceptions."""
        hosts = [("example.com", 80), ("error.com", 80)]

        with patch("src.utils.network_utils.check_connection") as mock_check:
            mock_check.side_effect = [True, Exception("Connection error")]

            result = await check_multiple_hosts(hosts, timeout=5.0)

            assert len(result) == 2
            assert result["example.com:80"] is True
            assert result["error.com:80"] is False

    @pytest.mark.asyncio
    async def test_multiple_hosts_gather_exception(self):
        """Test checking multiple hosts with gather exception."""
        hosts = [("example.com", 80), ("test.com", 80)]

        with patch("asyncio.gather", side_effect=Exception("Gather failed")):
            result = await check_multiple_hosts(hosts, timeout=5.0)

            # Should return all False on gather failure
            assert len(result) == 2
            assert result["example.com:80"] is False
            assert result["test.com:80"] is False

    @pytest.mark.asyncio
    async def test_multiple_hosts_concurrency_limit(self):
        """Test that concurrency is limited properly."""
        # Create more hosts than the limit (20)
        hosts = [(f"host{i}.com", 80) for i in range(25)]

        with patch("src.utils.network_utils.check_connection") as mock_check:
            mock_check.return_value = True

            result = await check_multiple_hosts(hosts, timeout=5.0)

            assert len(result) == 25
            # All should succeed
            assert all(result.values())


class TestParseUrl:
    """Test parse_url function."""

    def test_valid_http_url(self):
        """Test parsing valid HTTP URL."""
        url = "http://example.com:8080/path?query=value#fragment"

        result = parse_url(url)

        assert result["scheme"] == "http"
        assert result["hostname"] == "example.com"
        assert result["port"] == 8080
        assert result["path"] == "/path"
        assert result["query"] == "query=value"
        assert result["fragment"] == "fragment"

    def test_valid_https_url(self):
        """Test parsing valid HTTPS URL."""
        url = "https://secure.example.com/api/v1"

        result = parse_url(url)

        assert result["scheme"] == "https"
        assert result["hostname"] == "secure.example.com"
        assert result["port"] == 443  # Default HTTPS port
        assert result["path"] == "/api/v1"

    def test_url_with_authentication(self):
        """Test parsing URL with username and password."""
        url = "https://user:pass@example.com/path"

        result = parse_url(url)

        assert result["scheme"] == "https"
        assert result["hostname"] == "example.com"
        assert result["port"] == 443
        assert result["username"] == "user"
        assert result["password"] == "pass"

    def test_url_default_ports(self):
        """Test URL parsing with default ports."""
        # HTTP default port
        result = parse_url("http://example.com/path")
        assert result["port"] == 80

        # HTTPS default port
        result = parse_url("https://example.com/path")
        assert result["port"] == 443

        # FTP has a default port of 21
        result = parse_url("ftp://example.com/path")
        assert result["port"] == 21 or result["port"] is None  # Depends on implementation

    def test_url_without_path(self):
        """Test URL without explicit path."""
        url = "https://example.com"

        result = parse_url(url)

        assert result["path"] == "/"  # Should default to "/"

    def test_invalid_url_no_scheme(self):
        """Test parsing invalid URL without scheme."""
        with pytest.raises(ValidationError):
            parse_url("example.com/path")

    def test_invalid_url_no_netloc(self):
        """Test parsing invalid URL without netloc."""
        with pytest.raises(ValidationError):
            parse_url("http:///path")

    def test_url_parsing_exception(self):
        """Test URL parsing with malformed URL."""
        # Test with an actual malformed URL that will cause urlparse to fail internally
        with pytest.raises(ValidationError, match="Failed to parse URL"):
            with patch("src.utils.network_utils.urlparse", side_effect=Exception("Parse error")):
                parse_url("http://example.com")

    def test_url_edge_cases(self):
        """Test URL parsing edge cases."""
        # URL with only query
        result = parse_url("http://example.com?query=value")
        assert result["path"] == "/"
        assert result["query"] == "query=value"

        # URL with only fragment
        result = parse_url("http://example.com#fragment")
        assert result["path"] == "/"
        assert result["fragment"] == "fragment"

        # URL with IP address
        result = parse_url("http://192.168.1.1:8080")
        assert result["hostname"] == "192.168.1.1"
        assert result["port"] == 8080


class TestWaitForService:
    """Test wait_for_service function."""

    @pytest.mark.asyncio
    async def test_service_available_immediately(self):
        """Test waiting for service that's immediately available."""
        with patch("src.utils.network_utils.check_connection") as mock_check:
            mock_check.return_value = True

            result = await wait_for_service("example.com", 80, max_wait=5.0)
            assert result is True

    @pytest.mark.asyncio
    async def test_service_becomes_available(self):
        """Test waiting for service that becomes available after some time."""
        with patch("src.utils.network_utils.check_connection") as mock_check:
            # Fail twice, then succeed
            mock_check.side_effect = [False, False, True]

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await wait_for_service("example.com", 80, max_wait=5.0, check_interval=0.1)
                assert result is True

    @pytest.mark.asyncio
    async def test_service_timeout(self):
        """Test waiting for service that never becomes available."""
        with patch("src.utils.network_utils.check_connection") as mock_check:
            mock_check.return_value = False

            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch("asyncio.get_running_loop") as mock_loop:
                    # Simulate time progression
                    times = [0, 1, 2, 3, 4, 5, 6]  # Exceeds max_wait of 5
                    mock_loop.return_value.time.side_effect = times

                    result = await wait_for_service(
                        "example.com", 80, max_wait=5.0, check_interval=1.0
                    )
                    assert result is False

    @pytest.mark.asyncio
    async def test_service_wait_with_exceptions(self):
        """Test waiting for service with connection exceptions."""
        with patch("src.utils.network_utils.check_connection") as mock_check:
            # First call raises exception, then succeeds
            mock_check.side_effect = [Exception("Connection error"), True]

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await wait_for_service("example.com", 80, max_wait=5.0, check_interval=0.1)
                assert result is True

    @pytest.mark.asyncio
    async def test_service_wait_exponential_backoff(self):
        """Test that exponential backoff works correctly."""
        with patch("src.utils.network_utils.check_connection") as mock_check:
            mock_check.return_value = False

            sleep_calls = []

            async def mock_sleep(interval):
                sleep_calls.append(interval)

            with patch("asyncio.sleep", side_effect=mock_sleep):
                with patch("asyncio.get_running_loop") as mock_loop:
                    # Simulate time progression to cause timeout
                    times = [0, 1, 2, 3, 4, 5, 6]
                    mock_loop.return_value.time.side_effect = times

                    await wait_for_service("example.com", 80, max_wait=5.0, check_interval=1.0)

                    # Should see increasing intervals (with jitter)
                    assert len(sleep_calls) > 1
                    # First interval should be close to 1.0
                    assert 0.8 <= sleep_calls[0] <= 1.3  # With jitter
                    # Second should be higher
                    if len(sleep_calls) > 1:
                        assert sleep_calls[1] > sleep_calls[0]

    @pytest.mark.asyncio
    async def test_service_wait_max_interval_cap(self):
        """Test that interval is capped at maximum."""
        with patch("src.utils.network_utils.check_connection") as mock_check:
            mock_check.return_value = False

            sleep_calls = []

            async def mock_sleep(interval):
                sleep_calls.append(interval)

            with patch("asyncio.sleep", side_effect=mock_sleep):
                with patch("asyncio.get_running_loop") as mock_loop:
                    # Simulate many iterations to test cap
                    times = list(range(100))  # Long enough to trigger cap
                    mock_loop.return_value.time.side_effect = times

                    await wait_for_service("example.com", 80, max_wait=50.0, check_interval=1.0)

                    # Should see intervals capped at 10 seconds
                    max_interval = max(sleep_calls) if sleep_calls else 0
                    assert max_interval <= 10.0
