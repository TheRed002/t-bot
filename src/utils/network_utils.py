"""Network utilities for the T-Bot trading system."""

import asyncio
from typing import Any, Dict, Optional

from src.core.logging import get_logger
from src.core.exceptions import ValidationError

logger = get_logger(__name__)


class NetworkUtils:
    """All network operations."""
    
    @staticmethod
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
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception as e:
            logger.debug(f"Connection test failed for {host}:{port}: {str(e)}")
            return False
    
    @staticmethod
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
        try:
            start_time = asyncio.get_event_loop().time()
            
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            
            writer.close()
            await writer.wait_closed()
            
            return latency_ms
            
        except Exception as e:
            logger.error(f"Failed to measure latency for {host}:{port}: {str(e)}")
            raise ValidationError(f"Cannot measure latency to {host}:{port}: {str(e)}")
    
    @staticmethod
    async def ping_host(host: str, count: int = 3, port: int = 80) -> Dict[str, Any]:
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
                    latency = await NetworkUtils.measure_latency(host, port)
                    latencies.append(latency)
                    await asyncio.sleep(0.1)  # Small delay between pings
                except Exception as e:
                    logger.warning(f"Ping attempt {i + 1} failed for {host}: {str(e)}")
            
            if not latencies:
                return {
                    "host": host,
                    "success": False,
                    "error": "All ping attempts failed"
                }
            
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
            logger.error(f"Ping failed for {host}: {str(e)}")
            return {"host": host, "success": False, "error": str(e)}
    
    @staticmethod
    async def check_multiple_hosts(hosts: list, timeout: float = 5.0) -> Dict[str, bool]:
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
            task = NetworkUtils.test_connection(host, port, timeout)
            tasks.append((f"{host}:{port}", task))
        
        results = {}
        for host_port, task in tasks:
            try:
                results[host_port] = await task
            except Exception as e:
                logger.warning(f"Failed to check {host_port}: {e}")
                results[host_port] = False
        
        return results
    
    @staticmethod
    def parse_url(url: str) -> Dict[str, Any]:
        """
        Parse URL into components.
        
        Args:
            url: URL to parse
            
        Returns:
            Dictionary with URL components
            
        Raises:
            ValidationError: If URL is invalid
        """
        from urllib.parse import urlparse
        
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
            raise ValidationError(f"Failed to parse URL '{url}': {str(e)}")
    
    @staticmethod
    async def wait_for_service(
        host: str,
        port: int,
        max_wait: float = 30.0,
        check_interval: float = 1.0
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
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < max_wait:
            if await NetworkUtils.test_connection(host, port, timeout=2.0):
                logger.info(f"Service {host}:{port} is now available")
                return True
            
            await asyncio.sleep(check_interval)
        
        logger.warning(f"Service {host}:{port} did not become available within {max_wait}s")
        return False