"""
Centralized Resource Manager for T-Bot Trading System.

This module provides comprehensive resource lifecycle management,
connection pool monitoring, and automatic cleanup to prevent
resource leaks in trading systems.
"""

import asyncio
import gc
import logging
import threading
import time
import weakref
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import psutil


class ResourceType(Enum):
    """Types of resources being managed."""

    DATABASE_CONNECTION = "database_connection"
    REDIS_CONNECTION = "redis_connection"
    WEBSOCKET_CONNECTION = "websocket_connection"
    HTTP_SESSION = "http_session"
    FILE_HANDLE = "file_handle"
    THREAD_POOL = "thread_pool"
    ASYNCIO_TASK = "asyncio_task"
    SEMAPHORE = "semaphore"
    LOCK = "lock"
    CACHE_ENTRY = "cache_entry"


class ResourceState(Enum):
    """Resource lifecycle states."""

    CREATED = "created"
    ACTIVE = "active"
    IDLE = "idle"
    CLEANUP_PENDING = "cleanup_pending"
    DESTROYED = "destroyed"
    LEAKED = "leaked"


@dataclass
class ResourceInfo:
    """Information about a managed resource."""

    resource_id: str
    resource_type: ResourceType
    state: ResourceState
    created_at: datetime
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cleanup_callback: Callable[[], None] | None = None
    async_cleanup_callback: Callable[[], Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    memory_usage_bytes: int = 0

    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


class ResourceMonitor:
    """Monitors resource usage and detects leaks."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._start_time = time.time()
        self._last_gc_check = time.time()

    def get_memory_usage(self) -> dict[str, Any]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / (1024 * 1024),
            "uptime_hours": (time.time() - self._start_time) / 3600,
        }

    def get_connection_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        try:
            process = psutil.Process()
            connections = process.net_connections()

            stats = {
                "total_connections": len(connections),
                "established": 0,
                "listen": 0,
                "time_wait": 0,
                "close_wait": 0,
                "file_descriptors": process.num_fds() if hasattr(process, "num_fds") else 0,
                "threads": process.num_threads(),
            }

            for conn in connections:
                if conn.status:
                    stats[conn.status.lower()] = stats.get(conn.status.lower(), 0) + 1

            return stats

        except Exception as e:
            self.logger.warning(f"Failed to get connection stats: {e}")
            return {}

    def get_gc_stats(self) -> dict[str, Any]:
        """Get garbage collection statistics."""
        current_time = time.time()

        # Force GC if it's been a while
        if current_time - self._last_gc_check > 300:  # 5 minutes
            collected = gc.collect()
            self._last_gc_check = current_time

            return {
                "objects_collected": collected,
                "gc_counts": gc.get_count(),
                "gc_thresholds": gc.get_threshold(),
                "gc_stats": gc.get_stats(),
                "forced_collection": True,
            }

        return {
            "gc_counts": gc.get_count(),
            "gc_thresholds": gc.get_threshold(),
            "forced_collection": False,
        }


class ResourceManager:
    """
    Centralized resource lifecycle manager.

    Features:
    - Resource registration and tracking
    - Automatic cleanup on idle timeout
    - Connection pool monitoring
    - Memory leak detection
    - Resource usage reporting
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Resource tracking
        self._resources: dict[str, ResourceInfo] = {}
        self._resource_refs: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()
        self._resource_lock = threading.RLock()

        # Resource pools by type
        self._pools: dict[ResourceType, dict[str, Any]] = defaultdict(dict)

        # Configuration
        self._cleanup_interval = 60  # seconds
        self._idle_timeout = 300  # seconds
        self._leak_detection_enabled = True

        # Background tasks
        self._cleanup_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Monitoring
        self._monitor = ResourceMonitor()
        self._stats = {
            "resources_created": 0,
            "resources_destroyed": 0,
            "cleanup_runs": 0,
            "leaks_detected": 0,
        }

    async def start(self):
        """Start the resource manager."""
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()

        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("Resource manager started")

    async def stop(self):
        """Stop the resource manager and cleanup all resources."""
        self.logger.info("Stopping resource manager...")

        # Signal shutdown
        self._running = False
        self._shutdown_event.set()

        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await asyncio.wait_for(self._monitor_task, timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Cleanup all resources
        await self.cleanup_all_resources()

        self.logger.info("Resource manager stopped")

    def register_resource(
        self,
        resource: Any,
        resource_type: ResourceType,
        resource_id: str | None = None,
        cleanup_callback: Callable[[], None] | None = None,
        async_cleanup_callback: Callable[[], Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a resource for management."""
        if resource_id is None:
            resource_id = f"{resource_type.value}_{id(resource)}_{time.time()}"

        with self._resource_lock:
            resource_info = ResourceInfo(
                resource_id=resource_id,
                resource_type=resource_type,
                state=ResourceState.CREATED,
                created_at=datetime.now(timezone.utc),
                cleanup_callback=cleanup_callback,
                async_cleanup_callback=async_cleanup_callback,
                metadata=metadata or {},
            )

            self._resources[resource_id] = resource_info

            # Store weak reference to the actual resource
            try:
                self._resource_refs[resource_id] = resource
            except TypeError:
                # Some objects can't be weakly referenced
                pass

            self._stats["resources_created"] += 1

        self.logger.debug(f"Registered resource: {resource_id} ({resource_type.value})")
        return resource_id

    async def unregister_resource(self, resource_id: str):
        """Unregister and cleanup a resource."""
        resource_info = None
        with self._resource_lock:
            resource_info = self._resources.get(resource_id)
            if not resource_info:
                return

            # Mark as cleanup pending
            resource_info.state = ResourceState.CLEANUP_PENDING

        # Execute cleanup callbacks outside the lock to prevent deadlocks
        if resource_info.async_cleanup_callback:
            try:
                await resource_info.async_cleanup_callback()
            except Exception as e:
                self.logger.error(f"Error in async cleanup callback for {resource_id}: {e}")
        elif resource_info.cleanup_callback:
            try:
                resource_info.cleanup_callback()
            except Exception as e:
                self.logger.error(f"Error in cleanup callback for {resource_id}: {e}")

        # Remove from tracking
        with self._resource_lock:
            if resource_id in self._resources:
                resource_info = self._resources[resource_id]
                resource_info.state = ResourceState.DESTROYED
                del self._resources[resource_id]
                self._resource_refs.pop(resource_id, None)
                self._stats["resources_destroyed"] += 1

        self.logger.debug(f"Unregistered resource: {resource_id}")

    def touch_resource(self, resource_id: str):
        """Mark resource as recently accessed."""
        with self._resource_lock:
            resource_info = self._resources.get(resource_id)
            if resource_info:
                resource_info.touch()
                if resource_info.state == ResourceState.IDLE:
                    resource_info.state = ResourceState.ACTIVE

    async def cleanup_all_resources(self):
        """Cleanup all registered resources."""
        with self._resource_lock:
            resource_ids = list(self._resources.keys())

        # Use asyncio.gather for concurrent cleanup of WebSocket connections
        cleanup_tasks = []
        for resource_id in resource_ids:
            try:
                cleanup_tasks.append(self.unregister_resource(resource_id))
            except Exception as e:
                self.logger.error(f"Error setting up cleanup for resource {resource_id}: {e}")

        # Execute all cleanup tasks concurrently with timeout
        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True), timeout=30.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Resource cleanup timed out after 30 seconds")

        self.logger.info(f"Cleaned up {len(resource_ids)} resources")

    async def _cleanup_loop(self):
        """Background cleanup loop."""
        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    await self._perform_cleanup()
                    self._stats["cleanup_runs"] += 1

                    # Sleep with cancellation check
                    try:
                        await asyncio.wait_for(
                            asyncio.sleep(self._cleanup_interval),
                            timeout=self._cleanup_interval + 1.0,
                        )
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in cleanup loop: {e}")
                    await asyncio.sleep(self._cleanup_interval)

        except asyncio.CancelledError:
            pass
        finally:
            self.logger.debug("Cleanup loop stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    self._log_resource_stats()

                    # Check for memory leaks
                    if self._leak_detection_enabled:
                        self._detect_leaks()

                    # Sleep with cancellation check
                    try:
                        await asyncio.wait_for(
                            asyncio.sleep(300),  # Monitor every 5 minutes
                            timeout=301.0,
                        )
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(300)

        except asyncio.CancelledError:
            pass
        finally:
            self.logger.debug("Monitoring loop stopped")

    async def _perform_cleanup(self):
        """Perform resource cleanup."""
        current_time = datetime.now(timezone.utc)
        idle_resources = []

        with self._resource_lock:
            for resource_id, resource_info in self._resources.items():
                # Check if resource has been idle too long
                idle_time = (current_time - resource_info.last_accessed).total_seconds()

                if idle_time > self._idle_timeout:
                    resource_info.state = ResourceState.IDLE
                    idle_resources.append(resource_id)

        # Cleanup idle resources with concurrent processing
        cleanup_tasks = []
        for resource_id in idle_resources:
            cleanup_tasks.append(self._cleanup_idle_resource(resource_id))

        if cleanup_tasks:
            # Process cleanup tasks concurrently with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True), timeout=60.0
                )
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Idle resource cleanup timed out for {len(cleanup_tasks)} resources"
                )

    async def _cleanup_idle_resource(self, resource_id: str):
        """Cleanup a single idle resource."""
        try:
            await self.unregister_resource(resource_id)
            self.logger.debug(f"Cleaned up idle resource: {resource_id}")
        except Exception as e:
            self.logger.error(f"Error cleaning up idle resource {resource_id}: {e}")

    def _detect_leaks(self):
        """Detect potential resource leaks."""
        current_time = datetime.now(timezone.utc)
        potential_leaks = []

        with self._resource_lock:
            for resource_id, resource_info in self._resources.items():
                # Check if resource is very old and never accessed
                age = (current_time - resource_info.created_at).total_seconds()

                if (
                    age > 3600  # Older than 1 hour
                    and resource_info.access_count == 0  # Never accessed
                    and resource_info.state
                    not in [ResourceState.CLEANUP_PENDING, ResourceState.DESTROYED]
                ):
                    potential_leaks.append(resource_id)
                    resource_info.state = ResourceState.LEAKED

        if potential_leaks:
            self.logger.warning(
                f"Detected {len(potential_leaks)} potential resource leaks: "
                f"{potential_leaks[:5]}..."
            )
            self._stats["leaks_detected"] += len(potential_leaks)

    def _log_resource_stats(self) -> None:
        """Log resource statistics."""
        memory_stats = self._monitor.get_memory_usage()
        connection_stats = self._monitor.get_connection_stats()
        gc_stats = self._monitor.get_gc_stats()

        with self._resource_lock:
            resource_counts: dict[str, int] = defaultdict(int)
            for resource_info in self._resources.values():
                resource_counts[resource_info.resource_type.value] += 1

        self.logger.info(
            f"Resource Stats - Memory: {memory_stats['rss_mb']:.1f}MB, "
            f"Connections: {connection_stats.get('total_connections', 0)}, "
            f"Tracked Resources: {len(self._resources)}, "
            f"By Type: {dict(resource_counts)}"
        )

        if gc_stats.get("forced_collection"):
            self.logger.debug(f"GC collected {gc_stats.get('objects_collected', 0)} objects")

    def get_resource_stats(self) -> dict[str, Any]:
        """Get comprehensive resource statistics."""
        with self._resource_lock:
            resource_counts: dict[str, int] = defaultdict(int)
            state_counts: dict[str, int] = defaultdict(int)

            for resource_info in self._resources.values():
                resource_counts[resource_info.resource_type.value] += 1
                state_counts[resource_info.state.value] += 1

        return {
            "tracked_resources": len(self._resources),
            "by_type": dict(resource_counts),
            "by_state": dict(state_counts),
            "stats": self._stats.copy(),
            "memory": self._monitor.get_memory_usage(),
            "connections": self._monitor.get_connection_stats(),
            "gc": self._monitor.get_gc_stats(),
        }


# Global resource manager instance
_resource_manager: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    """Get or create the global resource manager."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


async def initialize_resource_manager():
    """Initialize the global resource manager."""
    manager = get_resource_manager()
    await manager.start()
    return manager


async def shutdown_resource_manager():
    """Shutdown the global resource manager."""
    global _resource_manager
    if _resource_manager:
        await _resource_manager.stop()
        _resource_manager = None
