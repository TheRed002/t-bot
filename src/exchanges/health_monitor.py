"""
Connection health monitor for WebSocket connection monitoring and recovery.

This module implements connection health monitoring with automatic recovery
capabilities for WebSocket connections across all exchanges.

CRITICAL: This integrates with P-001 (core types, exceptions, config),
P-002A (error handling), and P-003+ (exchange interfaces).
"""

import asyncio
import time
from typing import Dict, Optional, Any, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

# MANDATORY: Import from P-001
from src.core.types import MarketData, OrderRequest, OrderResponse, Position
from src.core.exceptions import (
    ExchangeRateLimitError, ExchangeConnectionError, ExchangeError, ValidationError
)
from src.core.config import Config
from src.core.logging import get_logger

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import RecoveryScenario

# MANDATORY: Import from P-007A (placeholder until P-007A is implemented)
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class ConnectionStatus(Enum):
    """Connection status enumeration."""
    HEALTHY = "healthy"
    FAILED = "failed"
    RECOVERED = "recovered"
    UNKNOWN = "unknown"


@dataclass
class ConnectionInfo:
    """Connection information for health monitoring."""
    connection_id: str
    exchange: str
    stream_type: str
    status: ConnectionStatus
    last_ping: Optional[datetime] = None
    last_pong: Optional[datetime] = None
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    recovery_attempts: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ConnectionHealthMonitor:
    """
    Connection health monitor for WebSocket connections.
    
    This class monitors WebSocket connection health and triggers automatic
    recovery when connections fail or become unresponsive.
    """
    
    def __init__(self, config: Config):
        """
        Initialize connection health monitor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Connection tracking
        self.connections: Dict[str, ConnectionInfo] = {}
        self.health_check_interval = 30  # seconds
        self.max_recovery_attempts = 5
        self.recovery_backoff_base = 2  # exponential backoff base
        
        # Health check callbacks
        self.health_check_callbacks: Dict[str, Callable] = {}
        self.recovery_callbacks: Dict[str, Callable] = {}
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # TODO: Remove in production
        logger.debug("ConnectionHealthMonitor initialized", 
                    health_check_interval=self.health_check_interval)
    
    
    @time_execution
    async def monitor_connection(self, connection: Any) -> None:
        """
        Start monitoring a connection.
        
        Args:
            connection: WebSocket connection object
            
        Raises:
            ValidationError: If connection parameters are invalid
            ExchangeConnectionError: If monitoring fails
        """
        try:
            # Validate connection
            if not connection:
                raise ValidationError("Connection is required")
            
            # Extract connection info
            connection_id = getattr(connection, 'id', str(id(connection)))
            exchange = getattr(connection, 'exchange', 'unknown')
            stream_type = getattr(connection, 'stream_type', 'unknown')
            
            # Create connection info
            conn_info = ConnectionInfo(
                connection_id=connection_id,
                exchange=exchange,
                stream_type=stream_type,
                status=ConnectionStatus.HEALTHY
            )
            
            # Register connection
            self.connections[connection_id] = conn_info
            
            # Start monitoring if not already started
            if not self.is_monitoring:
                await self._start_monitoring()
            
            logger.info("Connection monitoring started", 
                       connection_id=connection_id, exchange=exchange, stream_type=stream_type)
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error("Failed to start connection monitoring", 
                        connection=str(connection), error=str(e))
            raise ExchangeConnectionError(f"Connection monitoring failed: {str(e)}")
    
    
    @time_execution
    async def mark_failed(self, connection: Any) -> None:
        """
        Mark a connection as failed.
        
        Args:
            connection: WebSocket connection object
            
        Raises:
            ValidationError: If connection parameters are invalid
            ExchangeConnectionError: If marking failed fails
        """
        try:
            # Validate connection
            if not connection:
                raise ValidationError("Connection is required")
            
            # Get connection ID
            connection_id = getattr(connection, 'id', str(id(connection)))
            
            if connection_id not in self.connections:
                logger.warning("Connection not found in monitor", connection_id=connection_id)
                return
            
            # Update connection info
            conn_info = self.connections[connection_id]
            conn_info.status = ConnectionStatus.FAILED
            conn_info.failure_count += 1
            conn_info.last_failure = datetime.now()
            
            logger.warning("Connection marked as failed", 
                          connection_id=connection_id, 
                          failure_count=conn_info.failure_count)
            
            # Trigger recovery if within limits
            if conn_info.recovery_attempts < self.max_recovery_attempts:
                await self._trigger_recovery(conn_info)
            else:
                logger.error("Max recovery attempts exceeded", 
                           connection_id=connection_id, 
                           recovery_attempts=conn_info.recovery_attempts)
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error("Failed to mark connection as failed", 
                        connection=str(connection), error=str(e))
            raise ExchangeConnectionError(f"Failed to mark connection as failed: {str(e)}")
    
    
    async def _trigger_recovery(self, conn_info: ConnectionInfo) -> None:
        """
        Trigger connection recovery.
        
        Args:
            conn_info: Connection information
        """
        try:
            # Calculate backoff delay
            delay = self.recovery_backoff_base ** conn_info.recovery_attempts
            
            logger.info("Triggering connection recovery", 
                       connection_id=conn_info.connection_id,
                       recovery_attempt=conn_info.recovery_attempts + 1,
                       delay=delay)
            
            # Wait for backoff delay
            await asyncio.sleep(delay)
            
            # Increment recovery attempts
            conn_info.recovery_attempts += 1
            
            # Call recovery callback if registered
            if conn_info.connection_id in self.recovery_callbacks:
                try:
                    await self.recovery_callbacks[conn_info.connection_id](conn_info)
                    conn_info.status = ConnectionStatus.RECOVERED
                    logger.info("Connection recovery successful", 
                               connection_id=conn_info.connection_id)
                except Exception as e:
                    logger.error("Connection recovery failed", 
                               connection_id=conn_info.connection_id, error=str(e))
                    conn_info.status = ConnectionStatus.FAILED
            else:
                logger.warning("No recovery callback registered", 
                             connection_id=conn_info.connection_id)
            
        except Exception as e:
            logger.error("Failed to trigger connection recovery", 
                        connection_id=conn_info.connection_id, error=str(e))
    
    async def _start_monitoring(self) -> None:
        """Start the monitoring task."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Connection health monitoring started")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self.is_monitoring:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
        except asyncio.CancelledError:
            logger.info("Connection monitoring cancelled")
        except Exception as e:
            logger.error("Connection monitoring loop failed", error=str(e))
        finally:
            self.is_monitoring = False
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all monitored connections."""
        for connection_id, conn_info in list(self.connections.items()):
            try:
                # Skip failed connections that are being recovered
                if conn_info.status == ConnectionStatus.FAILED:
                    continue
                
                # Perform health check
                is_healthy = await self._check_connection_health(conn_info)
                
                if not is_healthy:
                    logger.warning("Connection health check failed", 
                                 connection_id=connection_id)
                    await self.mark_failed(self._get_connection_by_id(connection_id))
                else:
                    # Update last ping time
                    conn_info.last_ping = datetime.now()
                    
            except Exception as e:
                logger.error("Health check failed for connection", 
                           connection_id=connection_id, error=str(e))
    
    async def _check_connection_health(self, conn_info: ConnectionInfo) -> bool:
        """
        Check if a connection is healthy.
        
        Args:
            conn_info: Connection information
            
        Returns:
            bool: True if connection is healthy
        """
        try:
            # Check if connection has been inactive for too long
            if conn_info.last_ping:
                time_since_ping = datetime.now() - conn_info.last_ping
                if time_since_ping.total_seconds() > self.health_check_interval * 2:
                    logger.warning("Connection inactive for too long", 
                                 connection_id=conn_info.connection_id,
                                 time_since_ping=time_since_ping.total_seconds())
                    return False
            
            # Call health check callback if registered
            if conn_info.connection_id in self.health_check_callbacks:
                try:
                    is_healthy = await self.health_check_callbacks[conn_info.connection_id](conn_info)
                    return is_healthy
                except Exception as e:
                    logger.error("Health check callback failed", 
                               connection_id=conn_info.connection_id, error=str(e))
                    return False
            
            # Default health check (ping-based)
            return await self._default_health_check(conn_info)
            
        except Exception as e:
            logger.error("Connection health check failed", 
                        connection_id=conn_info.connection_id, error=str(e))
            return False
    
    async def _default_health_check(self, conn_info: ConnectionInfo) -> bool:
        """
        Default ping-based health check.
        
        Args:
            conn_info: Connection information
            
        Returns:
            bool: True if connection responds to ping
        """
        try:
            # This is a placeholder for actual ping implementation
            # In a real implementation, this would send a ping to the WebSocket
            # and wait for a pong response
            
            # For now, assume connection is healthy if it has recent activity
            if conn_info.last_ping:
                time_since_ping = datetime.now() - conn_info.last_ping
                return time_since_ping.total_seconds() < self.health_check_interval
            
            return True
            
        except Exception as e:
            logger.error("Default health check failed", 
                        connection_id=conn_info.connection_id, error=str(e))
            return False
    
    def _get_connection_by_id(self, connection_id: str) -> Any:
        """Get connection object by ID (placeholder implementation)."""
        # This would return the actual connection object
        # For now, return a mock object
        class MockConnection:
            def __init__(self, conn_id):
                self.id = conn_id
        
        return MockConnection(connection_id)
    
    def register_health_check_callback(self, connection_id: str, callback: Callable) -> None:
        """
        Register a health check callback for a connection.
        
        Args:
            connection_id: Connection ID
            callback: Health check callback function
        """
        self.health_check_callbacks[connection_id] = callback
        logger.debug("Health check callback registered", connection_id=connection_id)
    
    def register_recovery_callback(self, connection_id: str, callback: Callable) -> None:
        """
        Register a recovery callback for a connection.
        
        Args:
            connection_id: Connection ID
            callback: Recovery callback function
        """
        self.recovery_callbacks[connection_id] = callback
        logger.debug("Recovery callback registered", connection_id=connection_id)
    
    def unregister_connection(self, connection_id: str) -> None:
        """
        Unregister a connection from monitoring.
        
        Args:
            connection_id: Connection ID
        """
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        if connection_id in self.health_check_callbacks:
            del self.health_check_callbacks[connection_id]
        
        if connection_id in self.recovery_callbacks:
            del self.recovery_callbacks[connection_id]
        
        logger.info("Connection unregistered from monitoring", connection_id=connection_id)
    
    def get_connection_status(self, connection_id: str) -> Optional[ConnectionInfo]:
        """
        Get connection status.
        
        Args:
            connection_id: Connection ID
            
        Returns:
            ConnectionInfo if found, None otherwise
        """
        return self.connections.get(connection_id)
    
    def get_all_connection_status(self) -> Dict[str, ConnectionInfo]:
        """
        Get status of all monitored connections.
        
        Returns:
            Dict of connection ID to ConnectionInfo
        """
        return self.connections.copy()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get health monitoring summary.
        
        Returns:
            Dict containing health summary
        """
        total_connections = len(self.connections)
        healthy_connections = sum(1 for c in self.connections.values() 
                                if c.status == ConnectionStatus.HEALTHY)
        failed_connections = sum(1 for c in self.connections.values() 
                               if c.status == ConnectionStatus.FAILED)
        recovered_connections = sum(1 for c in self.connections.values() 
                                  if c.status == ConnectionStatus.RECOVERED)
        
        return {
            "total_connections": total_connections,
            "healthy_connections": healthy_connections,
            "failed_connections": failed_connections,
            "recovered_connections": recovered_connections,
            "health_percentage": (healthy_connections / total_connections * 100) if total_connections > 0 else 0,
            "is_monitoring": self.is_monitoring,
            "timestamp": datetime.now().isoformat()
        }
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring task."""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Connection health monitoring stopped")
    
    def update_health_check_interval(self, interval: int) -> None:
        """
        Update health check interval.
        
        Args:
            interval: New interval in seconds
        """
        if interval <= 0:
            raise ValidationError("Health check interval must be positive")
        
        self.health_check_interval = interval
        logger.info("Health check interval updated", new_interval=interval)
    
    def update_max_recovery_attempts(self, max_attempts: int) -> None:
        """
        Update maximum recovery attempts.
        
        Args:
            max_attempts: New maximum attempts
        """
        if max_attempts <= 0:
            raise ValidationError("Max recovery attempts must be positive")
        
        self.max_recovery_attempts = max_attempts
        logger.info("Max recovery attempts updated", new_max_attempts=max_attempts)
    
    def reset_connection_stats(self, connection_id: str) -> None:
        """
        Reset statistics for a connection.
        
        Args:
            connection_id: Connection ID
        """
        if connection_id in self.connections:
            conn_info = self.connections[connection_id]
            conn_info.failure_count = 0
            conn_info.recovery_attempts = 0
            conn_info.last_failure = None
            conn_info.status = ConnectionStatus.HEALTHY
            
            logger.info("Connection statistics reset", connection_id=connection_id)
    
    def get_failed_connections(self) -> List[ConnectionInfo]:
        """
        Get list of failed connections.
        
        Returns:
            List of failed connection info
        """
        return [conn_info for conn_info in self.connections.values() 
                if conn_info.status == ConnectionStatus.FAILED]
    
    def get_connection_failure_rate(self) -> float:
        """
        Calculate connection failure rate.
        
        Returns:
            float: Failure rate as percentage
        """
        if not self.connections:
            return 0.0
        
        failed_count = len(self.get_failed_connections())
        return (failed_count / len(self.connections)) * 100 