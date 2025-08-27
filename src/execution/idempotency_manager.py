"""
Order Idempotency Manager for preventing duplicate orders.

This module provides a centralized idempotency system that prevents duplicate orders
across all exchanges, handles retry logic with consistent client_order_ids, and
maintains proper expiration for idempotency keys.

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError

# MANDATORY: Import from P-001
from src.core.types import OrderRequest, OrderResponse

# MANDATORY: Import from P-007A
from src.utils import log_calls, time_execution


class IdempotencyKey:
    """Represents an idempotency key with metadata."""

    def __init__(
        self,
        key: str,
        client_order_id: str,
        order_hash: str,
        created_at: datetime,
        expires_at: datetime,
        retry_count: int = 0,
        status: str = "active",
        metadata: dict[str, Any] | None = None,
    ):
        self.key = key
        self.client_order_id = client_order_id
        self.order_hash = order_hash
        self.created_at = created_at
        self.expires_at = expires_at
        self.retry_count = retry_count
        self.status = status  # active, completed, expired, failed
        self.metadata = metadata or {}
        self.last_accessed = created_at

        # Thread safety
        self._lock = RLock()

    def is_expired(self) -> bool:
        """Check if the idempotency key is expired."""
        return datetime.now(timezone.utc) > self.expires_at

    def increment_retry(self) -> None:
        """Increment retry count with thread safety."""
        with self._lock:
            self.retry_count += 1
            self.last_accessed = datetime.now(timezone.utc)

    def mark_completed(self, order_response: OrderResponse | None = None) -> None:
        """Mark the key as completed with thread safety."""
        with self._lock:
            self.status = "completed"
            self.last_accessed = datetime.now(timezone.utc)
            if order_response:
                self.metadata["order_response"] = {
                    "id": order_response.id,
                    "status": order_response.status,
                    "filled_quantity": float(order_response.filled_quantity),
                    "timestamp": order_response.timestamp.isoformat(),
                }

    def mark_failed(self, error_message: str) -> None:
        """Mark the key as failed with thread safety."""
        with self._lock:
            self.status = "failed"
            self.last_accessed = datetime.now(timezone.utc)
            self.metadata["error"] = error_message

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "key": self.key,
            "client_order_id": self.client_order_id,
            "order_hash": self.order_hash,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "retry_count": self.retry_count,
            "status": self.status,
            "metadata": self.metadata,
            "last_accessed": self.last_accessed.isoformat(),
        }


class OrderIdempotencyManager(BaseComponent):
    """
    Centralized idempotency manager for preventing duplicate orders.

    This manager provides:
    - Unique client_order_id generation for each order
    - Duplicate order detection using content-based hashing
    - Redis-backed or in-memory caching with proper expiration
    - Retry logic that reuses the same client_order_id
    - Thread-safe and async-safe operations
    - Comprehensive audit trail for compliance
    """

    def __init__(self, config: Config, redis_client=None):
        """
        Initialize idempotency manager.

        Args:
            config: Application configuration
            redis_client: Optional Redis client for persistent storage
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.redis_client = redis_client

        # In-memory cache with thread safety
        self._cache_lock = RLock()
        self._in_memory_cache: dict[str, IdempotencyKey] = {}
        self._order_hash_to_key: dict[str, str] = {}  # order_hash -> key mapping
        self._client_order_id_to_key: dict[
            str, str
        ] = {}  # client_order_id -> key mapping for O(1) lookup

        # Configuration
        self.default_expiration_hours = 24
        self.max_retries = 3
        self.cleanup_interval_minutes = 60  # Clean up expired keys every hour
        self.use_redis = redis_client is not None

        # Statistics
        self.stats = {
            "total_keys_created": 0,
            "duplicate_orders_prevented": 0,
            "expired_keys_cleaned": 0,
            "failed_operations": 0,
            "redis_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Background cleanup task tracking
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_started = False
        self._is_running = False
        self._background_tasks: set[asyncio.Task] = set()

        self.logger.info("Order idempotency manager initialized", use_redis=self.use_redis)

        # Register cleanup for proper resource management
        import weakref

        weakref.finalize(self, self._cleanup_on_del)

    async def start(self) -> None:
        """Start the idempotency manager and background tasks."""
        if not self._cleanup_started:
            self._is_running = True
            self._start_cleanup_task()
            self._cleanup_started = True
            self.logger.info("Idempotency manager started")

    def _start_cleanup_task(self) -> None:
        """Start the periodic cleanup task."""

        async def cleanup_periodically():
            while self._is_running:
                try:
                    await asyncio.sleep(self.cleanup_interval_minutes * 60)
                    if self._is_running:  # Check again after sleep
                        await self._cleanup_expired_keys()
                except asyncio.CancelledError:
                    self.logger.debug("Cleanup task cancelled")
                    break
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
                    # Continue running unless explicitly stopped
                    if not self._is_running:
                        break

        self._cleanup_task = asyncio.create_task(cleanup_periodically())
        self._background_tasks.add(self._cleanup_task)

        # Clean up completed tasks
        self._cleanup_task.add_done_callback(self._background_tasks.discard)

    def _generate_order_hash(self, order: OrderRequest) -> str:
        """
        Generate a unique hash for an order based on its content.

        This hash is used to detect duplicate orders even if they have
        different client_order_ids.

        Args:
            order: Order request to hash

        Returns:
            str: SHA-256 hash of the order content
        """
        # Create order fingerprint using critical fields
        order_data = {
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": str(order.quantity),
            "price": str(order.price) if order.price else None,
            "stop_price": str(order.stop_price) if order.stop_price else None,
            "time_in_force": order.time_in_force,
        }

        # Sort keys for consistent hashing
        order_json = json.dumps(order_data, sort_keys=True)
        return hashlib.sha256(order_json.encode()).hexdigest()

    def _generate_client_order_id(self, order: OrderRequest) -> str:
        """
        Generate a unique client_order_id for an order.

        Args:
            order: Order request

        Returns:
            str: Unique client_order_id
        """
        # Create a unique ID with timestamp and random component
        timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        unique_id = uuid4().hex[:8]

        # Include symbol and side for easy identification
        symbol_prefix = order.symbol[:6].upper()  # First 6 chars of symbol
        side_prefix = order.side.value[:1].upper()  # B or S

        return f"T-{symbol_prefix}-{side_prefix}-{timestamp}-{unique_id}"

    def _generate_idempotency_key(self, order_hash: str) -> str:
        """
        Generate an idempotency key based on order hash.

        Args:
            order_hash: Hash of the order content

        Returns:
            str: Idempotency key
        """
        return f"idempotency:order:{order_hash}"

    @time_execution
    @log_calls
    async def get_or_create_idempotency_key(
        self,
        order: OrderRequest,
        expiration_hours: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """
        Get existing or create new idempotency key for an order.

        Args:
            order: Order request
            expiration_hours: Custom expiration time (default: 24 hours)
            metadata: Additional metadata to store

        Returns:
            tuple[str, bool]: (client_order_id, is_duplicate)

        Raises:
            ValidationError: If order validation fails
            ExecutionError: If operation fails
        """
        try:
            # Validate order
            if not order.symbol or not order.quantity:
                raise ValidationError("Order must have symbol and quantity")

            if order.quantity <= 0:
                raise ValidationError("Order quantity must be positive")

            # Generate order hash for duplicate detection
            order_hash = self._generate_order_hash(order)
            idempotency_key = self._generate_idempotency_key(order_hash)

            # Check for existing key
            existing_key = await self._get_idempotency_key(idempotency_key)

            if existing_key:
                # Check if key is expired
                if existing_key.is_expired():
                    await self._delete_idempotency_key(idempotency_key)
                    self.logger.debug(f"Expired idempotency key removed: {idempotency_key}")
                else:
                    # Duplicate order detected
                    self.stats["duplicate_orders_prevented"] += 1
                    self.stats["cache_hits"] += 1

                    self.logger.warning(
                        "Duplicate order detected",
                        order_hash=order_hash,
                        existing_client_order_id=existing_key.client_order_id,
                        retry_count=existing_key.retry_count,
                    )

                    return existing_key.client_order_id, True

            # Create new idempotency key
            client_order_id = order.client_order_id or self._generate_client_order_id(order)
            expiration = expiration_hours or self.default_expiration_hours

            created_at = datetime.now(timezone.utc)
            expires_at = created_at + timedelta(hours=expiration)

            new_key = IdempotencyKey(
                key=idempotency_key,
                client_order_id=client_order_id,
                order_hash=order_hash,
                created_at=created_at,
                expires_at=expires_at,
                retry_count=0,
                status="active",
                metadata=metadata or {},
            )

            # Store the key
            await self._store_idempotency_key(new_key)

            # Update statistics
            self.stats["total_keys_created"] += 1
            self.stats["cache_misses"] += 1

            self.logger.info(
                "New idempotency key created",
                client_order_id=client_order_id,
                order_hash=order_hash,
                expires_at=expires_at.isoformat(),
            )

            return client_order_id, False

        except Exception as e:
            self.stats["failed_operations"] += 1
            self.logger.error(f"Failed to get/create idempotency key: {e}")
            raise ExecutionError(f"Idempotency key operation failed: {e}") from e

    @log_calls
    async def mark_order_completed(
        self, client_order_id: str, order_response: OrderResponse
    ) -> bool:
        """
        Mark an order as completed using its client_order_id.

        Args:
            client_order_id: Client order ID
            order_response: Order response from exchange

        Returns:
            bool: True if successfully marked as completed
        """
        try:
            # Find the idempotency key by client_order_id
            idempotency_key = await self._find_key_by_client_order_id(client_order_id)

            if not idempotency_key:
                self.logger.warning(
                    f"No idempotency key found for client_order_id: {client_order_id}"
                )
                return False

            # Mark as completed
            idempotency_key.mark_completed(order_response)
            await self._store_idempotency_key(idempotency_key)

            self.logger.info(
                "Order marked as completed",
                client_order_id=client_order_id,
                order_id=order_response.id,
                status=order_response.status,
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to mark order as completed: {e}")
            return False

    @log_calls
    async def mark_order_failed(self, client_order_id: str, error_message: str) -> bool:
        """
        Mark an order as failed using its client_order_id.

        Args:
            client_order_id: Client order ID
            error_message: Error message

        Returns:
            bool: True if successfully marked as failed
        """
        try:
            # Find the idempotency key by client_order_id
            idempotency_key = await self._find_key_by_client_order_id(client_order_id)

            if not idempotency_key:
                self.logger.warning(
                    f"No idempotency key found for client_order_id: {client_order_id}"
                )
                return False

            # Mark as failed
            idempotency_key.mark_failed(error_message)
            await self._store_idempotency_key(idempotency_key)

            self.logger.info(
                "Order marked as failed", client_order_id=client_order_id, error=error_message
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to mark order as failed: {e}")
            return False

    @log_calls
    async def can_retry_order(self, client_order_id: str) -> tuple[bool, int]:
        """
        Check if an order can be retried and increment retry count.

        Args:
            client_order_id: Client order ID

        Returns:
            tuple[bool, int]: (can_retry, current_retry_count)
        """
        try:
            # Find the idempotency key
            idempotency_key = await self._find_key_by_client_order_id(client_order_id)

            if not idempotency_key:
                self.logger.warning(f"No idempotency key found for retry check: {client_order_id}")
                return False, 0

            # Check if expired
            if idempotency_key.is_expired():
                self.logger.warning(f"Cannot retry expired order: {client_order_id}")
                return False, idempotency_key.retry_count

            # Check retry limit
            if idempotency_key.retry_count >= self.max_retries:
                self.logger.warning(
                    f"Maximum retries exceeded for order: {client_order_id}",
                    retry_count=idempotency_key.retry_count,
                    max_retries=self.max_retries,
                )
                return False, idempotency_key.retry_count

            # Increment retry count
            idempotency_key.increment_retry()
            await self._store_idempotency_key(idempotency_key)

            self.logger.info(
                "Order retry allowed",
                client_order_id=client_order_id,
                retry_count=idempotency_key.retry_count,
            )

            return True, idempotency_key.retry_count

        except Exception as e:
            self.logger.error(f"Failed to check retry eligibility: {e}")
            return False, 0

    async def _get_idempotency_key(self, key: str) -> IdempotencyKey | None:
        """Get idempotency key from cache or Redis."""
        try:
            # Check in-memory cache first
            with self._cache_lock:
                if key in self._in_memory_cache:
                    return self._in_memory_cache[key]

            # Check Redis if available
            if self.use_redis and self.redis_client:
                try:
                    data = await self.redis_client.get(key)
                    if data:
                        self.stats["redis_operations"] += 1
                        key_data = json.loads(data)
                        idempotency_key = IdempotencyKey(
                            key=key_data["key"],
                            client_order_id=key_data["client_order_id"],
                            order_hash=key_data["order_hash"],
                            created_at=datetime.fromisoformat(key_data["created_at"]),
                            expires_at=datetime.fromisoformat(key_data["expires_at"]),
                            retry_count=key_data.get("retry_count", 0),
                            status=key_data.get("status", "active"),
                            metadata=key_data.get("metadata", {}),
                        )

                        # Cache in memory for faster access
                        with self._cache_lock:
                            self._in_memory_cache[key] = idempotency_key
                            self._order_hash_to_key[idempotency_key.order_hash] = key
                            self._client_order_id_to_key[idempotency_key.client_order_id] = key

                        return idempotency_key
                except Exception as e:
                    self.logger.warning(f"Redis get operation failed: {e}")

            return None

        except Exception as e:
            self.logger.error(f"Failed to get idempotency key: {e}")
            return None

    async def _store_idempotency_key(self, idempotency_key: IdempotencyKey) -> bool:
        """Store idempotency key in cache and Redis."""
        try:
            key = idempotency_key.key

            # Store in memory cache
            with self._cache_lock:
                self._in_memory_cache[key] = idempotency_key
                self._order_hash_to_key[idempotency_key.order_hash] = key
                self._client_order_id_to_key[idempotency_key.client_order_id] = key

            # Store in Redis if available
            if self.use_redis and self.redis_client:
                try:
                    data = json.dumps(idempotency_key.to_dict())
                    ttl_seconds = int(
                        (idempotency_key.expires_at - datetime.now(timezone.utc)).total_seconds()
                    )

                    if ttl_seconds > 0:
                        await self.redis_client.set(key, data, ttl=ttl_seconds)
                        self.stats["redis_operations"] += 1
                except Exception as e:
                    self.logger.warning(f"Redis store operation failed: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to store idempotency key: {e}")
            return False

    async def _delete_idempotency_key(self, key: str) -> bool:
        """Delete idempotency key from cache and Redis."""
        try:
            # Remove from memory cache
            with self._cache_lock:
                idempotency_key = self._in_memory_cache.pop(key, None)
                if idempotency_key:
                    self._order_hash_to_key.pop(idempotency_key.order_hash, None)
                    self._client_order_id_to_key.pop(idempotency_key.client_order_id, None)

            # Remove from Redis if available
            if self.use_redis and self.redis_client:
                try:
                    await self.redis_client.delete(key)
                    self.stats["redis_operations"] += 1
                except Exception as e:
                    self.logger.warning(f"Redis delete operation failed: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to delete idempotency key: {e}")
            return False

    async def _find_key_by_client_order_id(self, client_order_id: str) -> IdempotencyKey | None:
        """Find idempotency key by client_order_id."""
        try:
            # O(1) lookup using client_order_id index
            with self._cache_lock:
                key = self._client_order_id_to_key.get(client_order_id)
                if key:
                    return self._in_memory_cache.get(key)

            # If Redis is available, we'd need a reverse index
            # For now, this is a limitation - we rely on in-memory cache
            # In production, consider using Redis hash maps or secondary indices

            return None

        except Exception as e:
            self.logger.error(f"Failed to find key by client_order_id: {e}")
            return None

    async def _cleanup_expired_keys(self) -> int:
        """Clean up expired idempotency keys."""
        try:
            expired_count = 0

            # Clean up memory cache
            with self._cache_lock:
                expired_keys = []
                for key, idempotency_key in self._in_memory_cache.items():
                    if idempotency_key.is_expired():
                        expired_keys.append(key)

                for key in expired_keys:
                    idempotency_key = self._in_memory_cache.pop(key, None)
                    if idempotency_key:
                        self._order_hash_to_key.pop(idempotency_key.order_hash, None)
                        self._client_order_id_to_key.pop(idempotency_key.client_order_id, None)
                        expired_count += 1

            # Redis keys will expire automatically with TTL
            self.stats["expired_keys_cleaned"] += expired_count

            if expired_count > 0:
                self.logger.info(f"Cleaned up {expired_count} expired idempotency keys")

            return expired_count

        except Exception as e:
            self.logger.error(f"Cleanup operation failed: {e}")
            return 0

    def get_statistics(self) -> dict[str, Any]:
        """Get idempotency manager statistics."""
        with self._cache_lock:
            cache_size = len(self._in_memory_cache)

        return {
            "total_keys_created": self.stats["total_keys_created"],
            "duplicate_orders_prevented": self.stats["duplicate_orders_prevented"],
            "expired_keys_cleaned": self.stats["expired_keys_cleaned"],
            "failed_operations": self.stats["failed_operations"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "redis_operations": self.stats["redis_operations"],
            "cache_size": cache_size,
            "use_redis": self.use_redis,
            "configuration": {
                "default_expiration_hours": self.default_expiration_hours,
                "max_retries": self.max_retries,
                "cleanup_interval_minutes": self.cleanup_interval_minutes,
            },
        }

    async def get_active_keys(self, include_metadata: bool = False) -> list[dict[str, Any]]:
        """Get list of active idempotency keys."""
        try:
            active_keys = []

            with self._cache_lock:
                for idempotency_key in self._in_memory_cache.values():
                    if not idempotency_key.is_expired() and idempotency_key.status == "active":
                        key_info = {
                            "client_order_id": idempotency_key.client_order_id,
                            "order_hash": idempotency_key.order_hash[:16] + "...",  # Truncate
                            "created_at": idempotency_key.created_at.isoformat(),
                            "expires_at": idempotency_key.expires_at.isoformat(),
                            "retry_count": idempotency_key.retry_count,
                            "status": idempotency_key.status,
                        }

                        if include_metadata:
                            key_info["metadata"] = idempotency_key.metadata

                        active_keys.append(key_info)

            return active_keys

        except Exception as e:
            self.logger.error(f"Failed to get active keys: {e}")
            return []

    async def force_expire_key(self, client_order_id: str) -> bool:
        """Force expire an idempotency key."""
        try:
            idempotency_key = await self._find_key_by_client_order_id(client_order_id)

            if not idempotency_key:
                self.logger.warning(f"No idempotency key found to expire: {client_order_id}")
                return False

            # Delete the key immediately
            await self._delete_idempotency_key(idempotency_key.key)

            self.logger.info(f"Forced expiration of idempotency key: {client_order_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to force expire key: {e}")
            return False

    async def stop(self) -> None:
        """Stop the idempotency manager and cancel all background tasks."""
        try:
            self.logger.info("Stopping idempotency manager...")

            # Signal shutdown to background tasks
            self._is_running = False
            self._cleanup_started = False

            # Cancel and wait for cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.warning(f"Error during cleanup task cancellation: {e}")

            # Cancel all background tasks
            if self._background_tasks:
                self.logger.debug(f"Cancelling {len(self._background_tasks)} background tasks")
                for task in list(self._background_tasks):
                    if not task.done():
                        task.cancel()

                # Wait for all tasks to complete with timeout
                if self._background_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*self._background_tasks, return_exceptions=True),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        self.logger.warning("Background tasks did not complete within timeout")

                self._background_tasks.clear()

            # Final cleanup of expired keys
            try:
                await self._cleanup_expired_keys()
            except Exception as e:
                self.logger.warning(f"Final cleanup failed: {e}")

            # Clear all caches
            with self._cache_lock:
                self._in_memory_cache.clear()
                self._order_hash_to_key.clear()
                self._client_order_id_to_key.clear()

            self.logger.info("Idempotency manager stopped successfully")

        except Exception as e:
            self.logger.error(f"Idempotency manager stop failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the idempotency manager (alias for stop)."""
        await self.stop()

    def _cleanup_on_del(self) -> None:
        """Emergency cleanup when object is deleted."""
        try:
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
        except Exception:
            pass  # Ignore errors during emergency cleanup
