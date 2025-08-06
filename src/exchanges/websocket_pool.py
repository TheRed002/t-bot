"""
WebSocket connection pool for managing connection pools across exchanges.

This module implements connection pooling for WebSocket connections with
health monitoring and automatic connection management.

CRITICAL: This integrates with P-001 (core types, exceptions, config),
P-002A (error handling), and P-003+ (exchange interfaces).
"""

import asyncio
import time
from typing import Dict, Optional, Any, List, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

# MANDATORY: Import from P-001
from src.core.types import (
    MarketData, OrderRequest, OrderResponse, Position,
    ConnectionType
)
from src.core.exceptions import (
    ExchangeRateLimitError, ExchangeConnectionError, ExchangeError, ValidationError
)
from src.core.config import Config
from src.core.logging import get_logger

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import RecoveryScenario

# MANDATORY: Import from P-007A (placeholder until P-007A is implemented)
# from src.utils.decorators import time_execution

logger = get_logger(__name__)


@dataclass
class PooledConnection:
    """Information about a pooled connection."""
    connection_id: str
    exchange: str
    connection_type: ConnectionType
    connection: Any
    created_at: datetime
    last_used: datetime
    is_healthy: bool = True
    message_count: int = 0
    subscription_count: int = 0


class WebSocketConnectionPool:
    """
    WebSocket connection pool for managing connection pools.
    
    This class manages individual WebSocket connection pools with health
    monitoring and automatic connection management.
    """
    
    def __init__(self, exchange: str, max_connections: int = 10, 
                 max_messages_per_second: int = 100, max_subscriptions: int = 50):
        """
        Initialize WebSocket connection pool.
        
        Args:
            exchange: Exchange name
            max_connections: Maximum number of connections in pool
            max_messages_per_second: Maximum messages per second per connection
            max_subscriptions: Maximum subscriptions per connection
        """
        self.exchange = exchange
        self.max_connections = max_connections
        self.max_messages_per_second = max_messages_per_second
        self.max_subscriptions = max_subscriptions
        
        # Connection pools by type
        self.connection_pools: Dict[ConnectionType, List[PooledConnection]] = defaultdict(list)
        
        # Active connections
        self.active_connections: Dict[str, PooledConnection] = {}
        
        # Connection usage tracking
        self.message_counters: Dict[str, List[datetime]] = defaultdict(list)
        self.subscription_counters: Dict[str, int] = defaultdict(int)
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.connection_timeout = 300  # seconds (5 minutes)
        
        # TODO: Remove in production
        logger.debug("WebSocketConnectionPool initialized", 
                    exchange=exchange, max_connections=max_connections)
    
    
    async def get_connection(self, connection_type: ConnectionType) -> Optional[PooledConnection]:
        """
        Get a connection from the pool.
        
        Args:
            connection_type: Type of connection needed
            
        Returns:
            PooledConnection if available, None otherwise
            
        Raises:
            ValidationError: If parameters are invalid
            ExchangeConnectionError: If connection retrieval fails
        """
        try:
            # Validate parameters
            if not connection_type:
                raise ValidationError("Connection type is required")
            
            # Check if we have available connections
            available_connections = [
                conn for conn in self.connection_pools[connection_type]
                if self._is_connection_healthy(conn) and self._can_use_connection(conn)
            ]
            
            if available_connections:
                # Return the least recently used connection
                connection = min(available_connections, key=lambda c: c.last_used)
                connection.last_used = datetime.now()
                
                logger.debug("Reused connection from pool", 
                           connection_id=connection.connection_id,
                           connection_type=connection_type.value)
                return connection
            
            # Create new connection if pool not full
            if len(self.active_connections) < self.max_connections:
                connection = await self._create_connection(connection_type)
                if connection:
                    self.connection_pools[connection_type].append(connection)
                    self.active_connections[connection.connection_id] = connection
                    
                    logger.info("Created new connection", 
                              connection_id=connection.connection_id,
                              connection_type=connection_type.value)
                    return connection
            
            logger.warning("No available connections in pool", 
                         connection_type=connection_type.value,
                         active_connections=len(self.active_connections),
                         max_connections=self.max_connections)
            return None
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error("Failed to get connection from pool", 
                        connection_type=connection_type.value, error=str(e))
            raise ExchangeConnectionError(f"Connection retrieval failed: {str(e)}")
    
    
    async def release_connection(self, connection: PooledConnection) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            connection: Connection to release
            
        Raises:
            ValidationError: If connection is invalid
            ExchangeConnectionError: If release fails
        """
        try:
            # Validate connection
            if not connection or not connection.connection_id:
                raise ValidationError("Valid connection is required")
            
            # Update connection stats
            connection.last_used = datetime.now()
            
            # Check if connection is still healthy
            if not self._is_connection_healthy(connection):
                logger.warning("Removing unhealthy connection from pool", 
                             connection_id=connection.connection_id)
                await self._remove_connection(connection)
            else:
                logger.debug("Released connection back to pool", 
                           connection_id=connection.connection_id)
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error("Failed to release connection", 
                        connection_id=connection.connection_id if connection else None,
                        error=str(e))
            raise ExchangeConnectionError(f"Connection release failed: {str(e)}")
    
    async def _create_connection(self, connection_type: ConnectionType) -> Optional[PooledConnection]:
        """
        Create a new WebSocket connection.
        
        Args:
            connection_type: Type of connection to create
            
        Returns:
            PooledConnection if successful, None otherwise
        """
        try:
            # Generate unique connection ID
            connection_id = f"{self.exchange}_{connection_type.value}_{int(time.time() * 1000)}"
            
            # Create connection object (placeholder implementation)
            # In a real implementation, this would create an actual WebSocket connection
            connection_obj = self._create_websocket_connection(connection_type)
            
            if not connection_obj:
                logger.error("Failed to create WebSocket connection", 
                           connection_type=connection_type.value)
                return None
            
            # Create pooled connection
            pooled_connection = PooledConnection(
                connection_id=connection_id,
                exchange=self.exchange,
                connection_type=connection_type,
                connection=connection_obj,
                created_at=datetime.now(),
                last_used=datetime.now()
            )
            
            logger.info("Created new WebSocket connection", 
                       connection_id=connection_id,
                       connection_type=connection_type.value)
            
            return pooled_connection
            
        except Exception as e:
            logger.error("Failed to create connection", 
                        connection_type=connection_type.value, error=str(e))
            return None
    
    def _create_websocket_connection(self, connection_type: ConnectionType) -> Any:
        """
        Create actual WebSocket connection (placeholder implementation).
        
        Args:
            connection_type: Type of connection to create
            
        Returns:
            WebSocket connection object
        """
        # This is a placeholder implementation
        # In a real implementation, this would create an actual WebSocket connection
        # based on the exchange and connection type
        
        class MockWebSocketConnection:
            def __init__(self, conn_id, conn_type):
                self.id = conn_id
                self.type = conn_type
                self.is_connected = True
                self.exchange = self.exchange
                self.stream_type = conn_type.value
        
        return MockWebSocketConnection(
            f"{self.exchange}_{connection_type.value}_{int(time.time() * 1000)}",
            connection_type
        )
    
    def _is_connection_healthy(self, connection: PooledConnection) -> bool:
        """
        Check if a connection is healthy.
        
        Args:
            connection: Connection to check
            
        Returns:
            bool: True if connection is healthy
        """
        try:
            # Check if connection is too old
            if datetime.now() - connection.created_at > timedelta(seconds=self.connection_timeout):
                logger.debug("Connection too old", connection_id=connection.connection_id)
                return False
            
            # Check if connection has been inactive for too long
            if datetime.now() - connection.last_used > timedelta(seconds=self.connection_timeout):
                logger.debug("Connection inactive for too long", connection_id=connection.connection_id)
                return False
            
            # Check message rate limits
            if not self._check_message_rate_limit(connection):
                logger.debug("Connection exceeded message rate limit", connection_id=connection.connection_id)
                return False
            
            # Check subscription limits
            if not self._check_subscription_limit(connection):
                logger.debug("Connection exceeded subscription limit", connection_id=connection.connection_id)
                return False
            
            # Check if connection object is still valid
            if hasattr(connection.connection, 'is_connected'):
                if not connection.connection.is_connected:
                    logger.debug("Connection not connected", connection_id=connection.connection_id)
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Health check failed", connection_id=connection.connection_id, error=str(e))
            return False
    
    def _can_use_connection(self, connection: PooledConnection) -> bool:
        """
        Check if a connection can be used.
        
        Args:
            connection: Connection to check
            
        Returns:
            bool: True if connection can be used
        """
        try:
            # Check message rate limit
            if not self._check_message_rate_limit(connection):
                return False
            
            # Check subscription limit
            if not self._check_subscription_limit(connection):
                return False
            
            return True
            
        except Exception as e:
            logger.error("Connection usability check failed", 
                        connection_id=connection.connection_id, error=str(e))
            return False
    
    def _check_message_rate_limit(self, connection: PooledConnection) -> bool:
        """
        Check if connection is within message rate limits.
        
        Args:
            connection: Connection to check
            
        Returns:
            bool: True if within limits
        """
        try:
            now = datetime.now()
            second_ago = now - timedelta(seconds=1)
            
            # Clean old message timestamps
            self.message_counters[connection.connection_id] = [
                t for t in self.message_counters[connection.connection_id]
                if t > second_ago
            ]
            
            # Check if adding a message would exceed limit
            current_messages = len(self.message_counters[connection.connection_id])
            return current_messages < self.max_messages_per_second
            
        except Exception as e:
            logger.error("Message rate limit check failed", 
                        connection_id=connection.connection_id, error=str(e))
            return False
    
    def _check_subscription_limit(self, connection: PooledConnection) -> bool:
        """
        Check if connection is within subscription limits.
        
        Args:
            connection: Connection to check
            
        Returns:
            bool: True if within limits
        """
        try:
            current_subscriptions = self.subscription_counters.get(connection.connection_id, 0)
            return current_subscriptions < self.max_subscriptions
            
        except Exception as e:
            logger.error("Subscription limit check failed", 
                        connection_id=connection.connection_id, error=str(e))
            return False
    
    async def _remove_connection(self, connection: PooledConnection) -> None:
        """
        Remove a connection from the pool.
        
        Args:
            connection: Connection to remove
        """
        try:
            # Remove from active connections
            if connection.connection_id in self.active_connections:
                del self.active_connections[connection.connection_id]
            
            # Remove from connection pools
            if connection in self.connection_pools[connection.connection_type]:
                self.connection_pools[connection.connection_type].remove(connection)
            
            # Clean up counters
            if connection.connection_id in self.message_counters:
                del self.message_counters[connection.connection_id]
            
            if connection.connection_id in self.subscription_counters:
                del self.subscription_counters[connection.connection_id]
            
            # Close connection if it has a close method
            if hasattr(connection.connection, 'close'):
                try:
                    await connection.connection.close()
                except Exception as e:
                    logger.warning("Failed to close connection", 
                                 connection_id=connection.connection_id, error=str(e))
            
            logger.info("Removed connection from pool", connection_id=connection.connection_id)
            
        except Exception as e:
            logger.error("Failed to remove connection", 
                        connection_id=connection.connection_id, error=str(e))
    
    def record_message(self, connection_id: str) -> None:
        """
        Record a message sent on a connection.
        
        Args:
            connection_id: Connection ID
        """
        try:
            now = datetime.now()
            self.message_counters[connection_id].append(now)
            
            # Clean old entries (keep last 1000)
            if len(self.message_counters[connection_id]) > 1000:
                self.message_counters[connection_id] = self.message_counters[connection_id][-1000:]
            
        except Exception as e:
            logger.error("Failed to record message", connection_id=connection_id, error=str(e))
    
    def record_subscription(self, connection_id: str) -> bool:
        """
        Record a subscription on a connection.
        
        Args:
            connection_id: Connection ID
            
        Returns:
            bool: True if subscription was recorded successfully
        """
        try:
            current_count = self.subscription_counters.get(connection_id, 0)
            
            if current_count >= self.max_subscriptions:
                logger.warning("Subscription limit exceeded", 
                             connection_id=connection_id,
                             current_count=current_count,
                             max_subscriptions=self.max_subscriptions)
                return False
            
            self.subscription_counters[connection_id] = current_count + 1
            return True
            
        except Exception as e:
            logger.error("Failed to record subscription", connection_id=connection_id, error=str(e))
            return False
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Dict containing pool statistics
        """
        try:
            total_connections = len(self.active_connections)
            connections_by_type = {}
            
            for conn_type, connections in self.connection_pools.items():
                connections_by_type[conn_type.value] = len(connections)
            
            return {
                "exchange": self.exchange,
                "total_connections": total_connections,
                "max_connections": self.max_connections,
                "connections_by_type": connections_by_type,
                "max_messages_per_second": self.max_messages_per_second,
                "max_subscriptions": self.max_subscriptions,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get pool stats", error=str(e))
            return {}
    
    def get_connection_stats(self, connection_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific connection.
        
        Args:
            connection_id: Connection ID
            
        Returns:
            Dict containing connection statistics
        """
        try:
            if connection_id not in self.active_connections:
                return {}
            
            connection = self.active_connections[connection_id]
            message_count = len(self.message_counters.get(connection_id, []))
            subscription_count = self.subscription_counters.get(connection_id, 0)
            
            return {
                "connection_id": connection_id,
                "exchange": connection.exchange,
                "connection_type": connection.connection_type.value,
                "created_at": connection.created_at.isoformat(),
                "last_used": connection.last_used.isoformat(),
                "is_healthy": connection.is_healthy,
                "message_count": message_count,
                "subscription_count": subscription_count,
                "max_messages_per_second": self.max_messages_per_second,
                "max_subscriptions": self.max_subscriptions
            }
            
        except Exception as e:
            logger.error("Failed to get connection stats", connection_id=connection_id, error=str(e))
            return {}
    
    async def cleanup_old_connections(self) -> int:
        """
        Clean up old and unused connections.
        
        Returns:
            int: Number of connections removed
        """
        try:
            removed_count = 0
            now = datetime.now()
            
            for connection_type, connections in self.connection_pools.items():
                for connection in connections[:]:  # Copy list to avoid modification during iteration
                    # Check if connection is too old
                    if now - connection.created_at > timedelta(seconds=self.connection_timeout):
                        logger.info("Removing old connection", connection_id=connection.connection_id)
                        await self._remove_connection(connection)
                        removed_count += 1
                        continue
                    
                    # Check if connection has been inactive for too long
                    if now - connection.last_used > timedelta(seconds=self.connection_timeout):
                        logger.info("Removing inactive connection", connection_id=connection.connection_id)
                        await self._remove_connection(connection)
                        removed_count += 1
                        continue
                    
                    # Check if connection is unhealthy
                    if not self._is_connection_healthy(connection):
                        logger.info("Removing unhealthy connection", connection_id=connection.connection_id)
                        await self._remove_connection(connection)
                        removed_count += 1
            
            if removed_count > 0:
                logger.info("Cleaned up connections", removed_count=removed_count)
            
            return removed_count
            
        except Exception as e:
            logger.error("Failed to cleanup connections", error=str(e))
            return 0
    
    async def close_all_connections(self) -> None:
        """Close all connections in the pool."""
        try:
            for connection_type, connections in self.connection_pools.items():
                for connection in connections[:]:
                    await self._remove_connection(connection)
            
            logger.info("Closed all connections in pool", exchange=self.exchange)
            
        except Exception as e:
            logger.error("Failed to close all connections", error=str(e))
    
    def update_limits(self, max_connections: Optional[int] = None,
                     max_messages_per_second: Optional[int] = None,
                     max_subscriptions: Optional[int] = None) -> None:
        """
        Update pool limits.
        
        Args:
            max_connections: New maximum connections
            max_messages_per_second: New maximum messages per second
            max_subscriptions: New maximum subscriptions
        """
        try:
            if max_connections is not None:
                if max_connections <= 0:
                    raise ValidationError("Max connections must be positive")
                self.max_connections = max_connections
                logger.info("Updated max connections", new_limit=max_connections)
            
            if max_messages_per_second is not None:
                if max_messages_per_second <= 0:
                    raise ValidationError("Max messages per second must be positive")
                self.max_messages_per_second = max_messages_per_second
                logger.info("Updated max messages per second", new_limit=max_messages_per_second)
            
            if max_subscriptions is not None:
                if max_subscriptions <= 0:
                    raise ValidationError("Max subscriptions must be positive")
                self.max_subscriptions = max_subscriptions
                logger.info("Updated max subscriptions", new_limit=max_subscriptions)
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error("Failed to update pool limits", error=str(e)) 