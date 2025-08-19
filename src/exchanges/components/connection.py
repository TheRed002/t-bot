"""Connection management component for exchanges."""

import aiohttp
import asyncio
from typing import Any, Dict, Optional
from datetime import datetime

from src.core.logging import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """
    Handles all connection logic for exchanges.
    
    This component manages HTTP sessions, connection pooling,
    and request execution.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        timeout: int = 30,
        max_connections: int = 10
    ):
        """
        Initialize connection manager.
        
        Args:
            base_url: Base URL for the exchange API
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use testnet
            timeout: Request timeout in seconds
            max_connections: Maximum concurrent connections
        """
        self.base_url = base_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.timeout = timeout
        self.max_connections = max_connections
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self._connection_time: Optional[datetime] = None
        self._logger = logger
    
    async def connect(self) -> None:
        """Establish connection to exchange."""
        if self._connected:
            self._logger.warning("Already connected")
            return
        
        # Create connector with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            force_close=True
        )
        
        # Create session
        timeout_config = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout_config
        )
        
        self._connected = True
        self._connection_time = datetime.utcnow()
        self._logger.info(f"Connected to {self.base_url}")
    
    async def disconnect(self) -> None:
        """Close connection to exchange."""
        if not self._connected:
            self._logger.warning("Not connected")
            return
        
        if self.session:
            await self.session.close()
            # Wait a bit for the session to fully close
            await asyncio.sleep(0.25)
            self.session = None
        
        self._connected = False
        self._connection_time = None
        self._logger.info(f"Disconnected from {self.base_url}")
    
    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """
        Execute HTTP request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            headers: Additional headers
            signed: Whether request needs signing
            
        Returns:
            Response data
            
        Raises:
            ConnectionError: If not connected
            Exception: On request failure
        """
        if not self._connected or not self.session:
            raise ConnectionError("Not connected to exchange")
        
        url = f"{self.base_url}{endpoint}"
        
        # Prepare headers
        request_headers = headers or {}
        if self.api_key and signed:
            request_headers['X-API-KEY'] = self.api_key
            # Add signature if needed (exchange-specific)
            if self.api_secret:
                signature = self._sign_request(method, endpoint, params, data)
                request_headers['X-SIGNATURE'] = signature
        
        # Execute request
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=request_headers
            ) as response:
                response_data = await response.json()
                
                # Check for errors
                if response.status >= 400:
                    self._logger.error(
                        f"Request failed: {method} {url} - "
                        f"Status: {response.status} - Response: {response_data}"
                    )
                    raise Exception(f"Request failed with status {response.status}: {response_data}")
                
                return response_data
                
        except asyncio.TimeoutError:
            self._logger.error(f"Request timeout: {method} {url}")
            raise
        except Exception as e:
            self._logger.error(f"Request error: {method} {url} - {e}")
            raise
    
    def _sign_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict],
        data: Optional[Dict]
    ) -> str:
        """
        Sign request for authentication.
        
        This is a placeholder - actual implementation depends on exchange.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            
        Returns:
            Signature string
        """
        # This would be implemented based on exchange requirements
        # For example, HMAC-SHA256 of request data
        import hmac
        import hashlib
        import json
        
        if not self.api_secret:
            return ""
        
        # Create signature payload
        payload = f"{method}{endpoint}"
        if params:
            payload += json.dumps(params, sort_keys=True)
        if data:
            payload += json.dumps(data, sort_keys=True)
        
        # Create HMAC signature
        signature = hmac.new(
            self.api_secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self.session is not None
    
    def get_connection_time(self) -> Optional[datetime]:
        """Get connection establishment time."""
        return self._connection_time
    
    async def test_connection(self) -> bool:
        """
        Test connection with a simple request.
        
        Returns:
            True if connection is working
        """
        try:
            # Try a simple endpoint (exchange-specific)
            await self.request('GET', '/api/v1/ping', signed=False)
            return True
        except Exception as e:
            self._logger.error(f"Connection test failed: {e}")
            return False