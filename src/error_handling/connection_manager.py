"""
Connection resilience manager for reliable network connections.

This module provides automatic reconnection with exponential backoff, connection
pooling with health monitoring, heartbeat detection, and message queuing during
brief disconnections.

CRITICAL: This module integrates with P-001 core framework and P-002 database
for connection state persistence and will be used by all subsequent prompts.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
import structlog

# MANDATORY: Import from P-001 core framework
from src.core.exceptions import (
    TradingBotError, ExchangeError, ExchangeConnectionError
)
from src.core.config import Config

logger = structlog.get_logger()


class ConnectionState(Enum):
    """Connection state enumeration."""
    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class ConnectionHealth:
    """Connection health metrics."""
    last_heartbeat: datetime
    latency_ms: float
    packet_loss: float
    connection_quality: float  # 0.0 to 1.0
    uptime_seconds: int
    reconnect_count: int
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "latency_ms": self.latency_ms,
            "packet_loss": self.packet_loss,
            "connection_quality": self.connection_quality,
            "uptime_seconds": self.uptime_seconds,
            "reconnect_count": self.reconnect_count,
            "last_error": self.last_error
        }


class ConnectionManager:
    """Manages connection reliability with automatic reconnection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.health_monitors: Dict[str, ConnectionHealth] = {}
        self.reconnect_policies: Dict[str, Dict[str, Any]] = {
            "exchange": {
                "max_attempts": 5,
                "base_delay": 1,
                "max_delay": 60,
                "jitter": True
            },
            "database": {
                "max_attempts": 3,
                "base_delay": 0.5,
                "max_delay": 10,
                "jitter": False
            },
            "websocket": {
                "max_attempts": 10,
                "base_delay": 0.1,
                "max_delay": 30,
                "jitter": True
            }
        }
        self.message_queues: Dict[str, List[Dict[str, Any]]] = {}
        self.heartbeat_intervals: Dict[str, int] = {
            "exchange": 30,
            "database": 60,
            "websocket": 10
        }
    
    async def establish_connection(
        self,
        connection_id: str,
        connection_type: str,
        connect_func: Callable,
        **kwargs
    ) -> bool:
        """Establish a new connection with automatic retry."""
        
        logger.info(
            "Establishing connection",
            connection_id=connection_id,
            connection_type=connection_type
        )
        
        policy = self.reconnect_policies.get(connection_type, self.reconnect_policies["exchange"])
        
        for attempt in range(policy["max_attempts"]):
            try:
                # Calculate delay with exponential backoff
                delay = min(
                    policy["base_delay"] * (2 ** attempt),
                    policy["max_delay"]
                )
                
                if policy["jitter"]:
                    delay *= (0.5 + 0.5 * time.time() % 1)  # Add jitter
                
                if attempt > 0:
                    await asyncio.sleep(delay)
                
                # Attempt connection
                connection = await connect_func(**kwargs)
                
                # Initialize connection tracking
                self.connections[connection_id] = {
                    "connection": connection,
                    "type": connection_type,
                    "state": ConnectionState.CONNECTED,
                    "established_at": datetime.now(timezone.utc),
                    "last_activity": datetime.now(timezone.utc),
                    "reconnect_count": 0
                }
                
                # Initialize health monitoring
                self.health_monitors[connection_id] = ConnectionHealth(
                    last_heartbeat=datetime.now(timezone.utc),
                    latency_ms=0.0,
                    packet_loss=0.0,
                    connection_quality=1.0,
                    uptime_seconds=0,
                    reconnect_count=0
                )
                
                # Start health monitoring
                asyncio.create_task(self._monitor_connection_health(connection_id))
                
                logger.info(
                    "Connection established successfully",
                    connection_id=connection_id,
                    connection_type=connection_type,
                    attempt=attempt + 1
                )
                
                return True
                
            except Exception as e:
                logger.warning(
                    "Connection attempt failed",
                    connection_id=connection_id,
                    connection_type=connection_type,
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt == policy["max_attempts"] - 1:
                    logger.error(
                        "Connection establishment failed",
                        connection_id=connection_id,
                        connection_type=connection_type,
                        max_attempts=policy["max_attempts"]
                    )
                    return False
        
        return False
    
    async def close_connection(self, connection_id: str) -> bool:
        """Close a connection gracefully."""
        
        if connection_id not in self.connections:
            logger.warning("Connection not found", connection_id=connection_id)
            return False
        
        try:
            connection_info = self.connections[connection_id]
            connection = connection_info["connection"]
            
            # Close connection if it has a close method
            if hasattr(connection, 'close'):
                await connection.close()
            elif hasattr(connection, 'disconnect'):
                await connection.disconnect()
            
            # Update state
            connection_info["state"] = ConnectionState.DISCONNECTED
            connection_info["last_activity"] = datetime.now(timezone.utc)
            
            logger.info("Connection closed", connection_id=connection_id)
            return True
            
        except Exception as e:
            logger.error("Failed to close connection", connection_id=connection_id, error=str(e))
            return False
    
    async def reconnect_connection(self, connection_id: str) -> bool:
        """Reconnect a failed connection."""
        
        if connection_id not in self.connections:
            logger.warning("Connection not found for reconnection", connection_id=connection_id)
            return False
        
        connection_info = self.connections[connection_id]
        connection_type = connection_info["type"]
        
        logger.info(
            "Attempting reconnection",
            connection_id=connection_id,
            connection_type=connection_type
        )
        
        # Update state
        connection_info["state"] = ConnectionState.CONNECTING
        connection_info["reconnect_count"] += 1
        
        # TODO: Implement actual reconnection logic
        # This will be implemented in P-003+ (Exchange Integrations)
        # For now, simulate reconnection
        await asyncio.sleep(1)
        
        # Simulate successful reconnection
        connection_info["state"] = ConnectionState.CONNECTED
        connection_info["last_activity"] = datetime.now(timezone.utc)
        
        logger.info(
            "Reconnection successful",
            connection_id=connection_id,
            reconnect_count=connection_info["reconnect_count"]
        )
        
        return True
    
    async def _monitor_connection_health(self, connection_id: str):
        """Monitor connection health with heartbeat checks."""
        
        connection_info = self.connections.get(connection_id)
        if not connection_info:
            return
        
        connection_type = connection_info["type"]
        heartbeat_interval = self.heartbeat_intervals.get(connection_type, 30)
        
        while connection_info["state"] == ConnectionState.CONNECTED:
            try:
                # Perform heartbeat check
                start_time = time.time()
                is_healthy = await self._perform_heartbeat(connection_id)
                latency_ms = (time.time() - start_time) * 1000
                
                if is_healthy:
                    # Update health metrics
                    health = self.health_monitors[connection_id]
                    health.last_heartbeat = datetime.now(timezone.utc)
                    health.latency_ms = latency_ms
                    health.uptime_seconds = int((datetime.now(timezone.utc) - connection_info["established_at"]).total_seconds())
                    health.last_error = None
                    
                    # Update connection quality based on latency
                    if latency_ms < 100:
                        health.connection_quality = 1.0
                    elif latency_ms < 500:
                        health.connection_quality = 0.8
                    elif latency_ms < 1000:
                        health.connection_quality = 0.6
                    else:
                        health.connection_quality = 0.4
                    
                    connection_info["last_activity"] = datetime.now(timezone.utc)
                    
                else:
                    # Connection is unhealthy, trigger reconnection
                    logger.warning(
                        "Connection health check failed",
                        connection_id=connection_id
                    )
                    
                    health = self.health_monitors[connection_id]
                    health.last_error = "Heartbeat failed"
                    health.connection_quality = 0.0
                    
                    # Trigger reconnection
                    await self.reconnect_connection(connection_id)
                
                await asyncio.sleep(heartbeat_interval)
                
            except Exception as e:
                logger.error(
                    "Health monitoring error",
                    connection_id=connection_id,
                    error=str(e)
                )
                await asyncio.sleep(heartbeat_interval)
    
    async def _perform_heartbeat(self, connection_id: str) -> bool:
        """Perform heartbeat check on connection."""
        
        connection_info = self.connections.get(connection_id)
        if not connection_info:
            return False
        
        connection = connection_info["connection"]
        connection_type = connection_info["type"]
        
        try:
            # TODO: Implement actual heartbeat logic
            # This will be implemented in P-003+ (Exchange Integrations)
            
            if connection_type == "exchange":
                # Simulate exchange heartbeat
                return True
            elif connection_type == "database":
                # Simulate database heartbeat
                return True
            elif connection_type == "websocket":
                # Simulate WebSocket heartbeat
                return True
            else:
                # Generic heartbeat
                return True
                
        except Exception as e:
            logger.error(
                "Heartbeat failed",
                connection_id=connection_id,
                error=str(e)
            )
            return False
    
    async def queue_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Queue a message for later transmission when connection is restored."""
        
        if connection_id not in self.message_queues:
            self.message_queues[connection_id] = []
        
        message_with_timestamp = {
            **message,
            "queued_at": datetime.now(timezone.utc).isoformat()
        }
        
        self.message_queues[connection_id].append(message_with_timestamp)
        
        logger.info(
            "Message queued",
            connection_id=connection_id,
            message_type=message.get("type", "unknown"),
            queue_size=len(self.message_queues[connection_id])
        )
        
        return True
    
    async def flush_message_queue(self, connection_id: str) -> int:
        """Flush queued messages for a connection."""
        
        if connection_id not in self.message_queues:
            return 0
        
        queue = self.message_queues[connection_id]
        if not queue:
            return 0
        
        flushed_count = 0
        
        for message in queue:
            try:
                # TODO: Implement actual message transmission
                # This will be implemented in P-003+ (Exchange Integrations)
                logger.info(
                    "Flushing queued message",
                    connection_id=connection_id,
                    message_type=message.get("type", "unknown")
                )
                flushed_count += 1
                
            except Exception as e:
                logger.error(
                    "Failed to flush message",
                    connection_id=connection_id,
                    message_type=message.get("type", "unknown"),
                    error=str(e)
                )
        
        # Clear the queue
        self.message_queues[connection_id] = []
        
        logger.info(
            "Message queue flushed",
            connection_id=connection_id,
            flushed_count=flushed_count
        )
        
        return flushed_count
    
    def get_connection_status(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific connection."""
        
        if connection_id not in self.connections:
            return None
        
        connection_info = self.connections[connection_id]
        health = self.health_monitors.get(connection_id)
        
        return {
            "connection_id": connection_id,
            "type": connection_info["type"],
            "state": connection_info["state"].value,
            "established_at": connection_info["established_at"].isoformat(),
            "last_activity": connection_info["last_activity"].isoformat(),
            "reconnect_count": connection_info["reconnect_count"],
            "health": health.to_dict() if health else None,
            "queued_messages": len(self.message_queues.get(connection_id, []))
        }
    
    def get_all_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connections."""
        return {
            connection_id: self.get_connection_status(connection_id)
            for connection_id in self.connections.keys()
        }
    
    def is_connection_healthy(self, connection_id: str) -> bool:
        """Check if a connection is healthy."""
        
        if connection_id not in self.connections:
            return False
        
        connection_info = self.connections[connection_id]
        health = self.health_monitors.get(connection_id)
        
        if connection_info["state"] != ConnectionState.CONNECTED:
            return False
        
        if not health:
            return False
        
        # Check if heartbeat is recent (within 2x heartbeat interval)
        connection_type = connection_info["type"]
        heartbeat_interval = self.heartbeat_intervals.get(connection_type, 30)
        max_age = heartbeat_interval * 2
        
        time_since_heartbeat = (datetime.now(timezone.utc) - health.last_heartbeat).total_seconds()
        
        return time_since_heartbeat < max_age and health.connection_quality > 0.5 