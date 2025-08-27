"""Network utilities for the T-Bot trading system."""

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

from src.core.exceptions import ValidationError

# Module level logger for static methods
logger = logging.getLogger(__name__)


async def test_connection(host: str, port: int, timeout: float = 5.0) -> bool:
    """
    Test network connection to a host and port.

    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Connection timeout in seconds

    Returns:
        True if connection successful, False otherwise
    """
    reader = None
    writer = None
    try:
        # Create connection task
        conn_task = asyncio.create_task(asyncio.open_connection(host, port))
        try:
            reader, writer = await asyncio.wait_for(conn_task, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            # Cancel the connection task if it times out
            conn_task.cancel()
            try:
                await conn_task
            except asyncio.CancelledError:
                pass
            logger.debug(f"Connection test timed out for {host}:{port}")
            return False
    except Exception as e:
        logger.debug(f"Connection test failed for {host}:{port}: {e!s}")
        return False
    finally:
        if writer:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                # Ignore errors during cleanup
                pass


async def measure_latency(host: str, port: int, timeout: float = 5.0) -> float:
    """
    Measure network latency to a host and port.

    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Connection timeout in seconds

    Returns:
        Latency in milliseconds

    Raises:
        ValidationError: If connection fails
    """
    reader = None
    writer = None
    try:
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        # Create connection task
        conn_task = asyncio.create_task(asyncio.open_connection(host, port))
        try:
            reader, writer = await asyncio.wait_for(conn_task, timeout=timeout)
        except asyncio.TimeoutError:
            # Cancel the connection task if it times out
            conn_task.cancel()
            try:
                await conn_task
            except asyncio.CancelledError:
                pass
            raise ValidationError(f"Connection timeout to {host}:{port}") from None

        end_time = loop.time()
        latency_ms = (end_time - start_time) * 1000

        return latency_ms

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Failed to measure latency for {host}:{port}: {e!s}")
        raise ValidationError(f"Cannot measure latency to {host}:{port}: {e!s}") from e
    finally:
        if writer:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                # Ignore errors during cleanup
                pass


async def ping_host(host: str, count: int = 3, port: int = 80) -> dict[str, Any]:
    """
    Ping a host and return statistics.

    Args:
        host: Hostname or IP address
        count: Number of ping attempts
        port: Port to test (default 80 for HTTP)

    Returns:
        Dictionary with ping statistics
    """
    try:
        latencies = []

        for i in range(count):
            try:
                latency = await measure_latency(host, port)
                latencies.append(latency)
                await asyncio.sleep(0.1)  # Small delay between pings
            except Exception as e:
                logger.warning(f"Ping attempt {i + 1} failed for {host}: {e!s}")

        if not latencies:
            return {"host": host, "success": False, "error": "All ping attempts failed"}

        return {
            "host": host,
            "success": True,
            "count": len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "packet_loss_pct": ((count - len(latencies)) / count) * 100,
        }

    except Exception as e:
        logger.error(f"Ping failed for {host}: {e!s}")
        return {"host": host, "success": False, "error": str(e)}


async def check_multiple_hosts(
    hosts: list[tuple[str, int]], timeout: float = 5.0
) -> dict[str, bool]:
    """
    Check connectivity to multiple hosts in parallel.

    Args:
        hosts: List of (host, port) tuples
        timeout: Connection timeout

    Returns:
        Dictionary mapping host:port to connectivity status
    """
    tasks = []
    for host, port in hosts:
        task = test_connection(host, port, timeout)
        tasks.append((f"{host}:{port}", task))

    results = {}
    for host_port, task in tasks:
        try:
            results[host_port] = await task
        except Exception as e:
            logger.warning(f"Failed to check {host_port}: {e}")
            results[host_port] = False

    return results


def parse_url(url: str) -> dict[str, Any]:
    """
    Parse URL into components.

    Args:
        url: URL to parse

    Returns:
        Dictionary with URL components

    Raises:
        ValidationError: If URL is invalid
    """
    try:
        parsed = urlparse(url)

        if not parsed.scheme or not parsed.netloc:
            raise ValidationError("Invalid URL format")

        # Extract port or use default
        if parsed.port:
            port = parsed.port
        elif parsed.scheme == "https":
            port = 443
        elif parsed.scheme == "http":
            port = 80
        else:
            port = None

        # Extract hostname
        hostname = parsed.hostname or parsed.netloc

        return {
            "scheme": parsed.scheme,
            "hostname": hostname,
            "port": port,
            "path": parsed.path or "/",
            "query": parsed.query,
            "fragment": parsed.fragment,
            "username": parsed.username,
            "password": parsed.password,
        }

    except Exception as e:
        raise ValidationError(f"Failed to parse URL '{url}': {e!s}") from e


async def wait_for_service(
    host: str, port: int, max_wait: float = 30.0, check_interval: float = 1.0
) -> bool:
    """
    Wait for a service to become available.

    Args:
        host: Service hostname
        port: Service port
        max_wait: Maximum time to wait in seconds
        check_interval: Interval between checks

    Returns:
        True if service became available, False if timeout
    """
    loop = asyncio.get_running_loop()
    start_time = loop.time()

    while loop.time() - start_time < max_wait:
        if await test_connection(host, port, timeout=2.0):
            logger.info(f"Service {host}:{port} is now available")
            return True

        await asyncio.sleep(check_interval)

    logger.warning(f"Service {host}:{port} did not become available within {max_wait}s")
    return False
