"""Network utilities for the T-Bot trading system."""

import asyncio
from typing import Any
from urllib.parse import urlparse

from src.core.exceptions import ValidationError
from src.core.logging import get_logger

# Module level logger for static methods
logger = get_logger(__name__)


async def check_connection(host: str, port: int, timeout: float = 5.0) -> bool:
    """
    Test network connection to a host and port.

    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Connection timeout in seconds

    Returns:
        True if connection successful, False otherwise
    """
    writer = None
    try:
        # Create connection with timeout using async context management
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.debug(f"Connection test timed out for {host}:{port}")
            return False
    except Exception as e:
        logger.debug(f"Connection test failed for {host}:{port}: {e!s}")
        return False
    finally:
        if writer:
            try:
                writer.close()
                # Add timeout to wait_closed to prevent hanging
                await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
            except (OSError, asyncio.TimeoutError, ConnectionResetError) as e:
                # Ignore expected errors during cleanup
                logger.debug(f"Error during writer cleanup in test_connection: {e}")


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
    writer = None
    try:
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        # Create connection with timeout
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
        except asyncio.TimeoutError:
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
            try:
                writer.close()
                # Add timeout to wait_closed to prevent hanging
                await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
            except (OSError, asyncio.TimeoutError, ConnectionResetError) as e:
                # Ignore expected errors during cleanup
                logger.debug(f"Error during writer cleanup in measure_latency: {e}")


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


async def check_multiple_hosts(hosts: list[tuple[str, int]], timeout: float = 5.0) -> dict[str, bool]:
    """
    Check connectivity to multiple hosts in parallel.

    Args:
        hosts: List of (host, port) tuples
        timeout: Connection timeout

    Returns:
        Dictionary mapping host:port to connectivity status
    """
    if not hosts:
        return {}

    # Create tasks for parallel execution with connection limit
    max_concurrent = min(len(hosts), 20)  # Limit concurrent connections
    semaphore = asyncio.Semaphore(max_concurrent)

    async def test_with_semaphore(host: str, port: int) -> bool:
        """Test connection with semaphore for backpressure."""
        async with semaphore:
            return await check_connection(host, port, timeout)

    tasks = []
    host_port_mapping = []

    for host, port in hosts:
        host_port = f"{host}:{port}"
        task = asyncio.create_task(test_with_semaphore(host, port))
        tasks.append(task)
        host_port_mapping.append(host_port)

    # Execute all tasks concurrently using asyncio.gather with proper error handling
    try:
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Unexpected error in check_multiple_hosts: {e}")
        # Cancel remaining tasks on failure
        for task in tasks:
            if not task.done():
                task.cancel()
        # Wait for cancelled tasks to complete
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.debug(f"Error during task cancellation: {e}")
            pass  # Ignore cancellation errors
        # Return all False if gather fails unexpectedly
        return {host_port: False for host_port in host_port_mapping}

    # Process results
    results = {}
    for i, result in enumerate(results_list):
        host_port = host_port_mapping[i]
        if isinstance(result, Exception):
            logger.warning(f"Failed to check {host_port}: {result}")
            results[host_port] = False
        else:
            # result is guaranteed to be bool here due to isinstance check
            results[host_port] = bool(result)

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


async def wait_for_service(host: str, port: int, max_wait: float = 30.0, check_interval: float = 1.0) -> bool:
    """
    Wait for a service to become available with exponential backoff.

    Args:
        host: Service hostname
        port: Service port
        max_wait: Maximum time to wait in seconds
        check_interval: Initial interval between checks

    Returns:
        True if service became available, False if timeout
    """
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    current_interval = check_interval
    max_interval = min(check_interval * 8, 10.0)  # Cap at 10 seconds

    while loop.time() - start_time < max_wait:
        try:
            if await check_connection(host, port, timeout=2.0):
                logger.info(f"Service {host}:{port} is now available")
                return True
        except Exception as e:
            logger.debug(f"Connection test failed for {host}:{port}: {e}")

        # Wait with current interval
        await asyncio.sleep(current_interval)

        # Exponential backoff with jitter
        import random

        jitter = random.uniform(0.1, 0.3) * current_interval
        current_interval = min(current_interval * 1.5 + jitter, max_interval)

    logger.warning(f"Service {host}:{port} did not become available within {max_wait}s")
    return False
