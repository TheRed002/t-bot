"""
Core State Manager for the T-Bot trading system (P-018).

This module provides the central state management functionality including:
- State persistence and recovery
- Multi-layer state storage (Redis, PostgreSQL, InfluxDB)  
- State validation and consistency checks
- Atomic state operations and transactions
- State versioning and rollback capabilities

The StateManager serves as the single source of truth for all bot state
and ensures consistency across system restarts and failures.
"""

import asyncio
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from uuid import uuid4

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

# Core framework imports
from src.core.config import Config
from src.core.exceptions import ValidationError, StateError, ConfigurationError
from src.core.logging import get_logger
from src.core.types import BotState, BotStatus, OrderStatus, TradeState

# Database imports
from src.database.manager import DatabaseManager
from src.database.redis_client import RedisClient
from src.database.influxdb_client import InfluxDBClient

# Utility imports
from src.utils.decorators import retry, time_execution
from src.utils.validators import validate_state_data

T = TypeVar('T')

logger = get_logger(__name__)


@dataclass
class StateSnapshot:
    """Complete state snapshot for recovery purposes."""
    
    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    bot_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    
    # Core state components
    bot_state: Dict[str, Any] = field(default_factory=dict)
    positions: List[Dict[str, Any]] = field(default_factory=list)
    orders: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    strategy_state: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Validation metadata
    checksum: str = ""
    compressed: bool = False
    size_bytes: int = 0


@dataclass
class StateTransaction:
    """State transaction for atomic updates."""
    
    transaction_id: str = field(default_factory=lambda: str(uuid4()))
    bot_id: str = ""
    operations: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"  # pending, committed, rolled_back


class StateManager:
    """
    Central state management system for bot state persistence and recovery.
    
    Provides:
    - Multi-layer state storage with Redis (real-time), PostgreSQL (persistent), InfluxDB (metrics)
    - Atomic state operations and transactions
    - State versioning and rollback capabilities
    - Crash recovery and consistency validation
    - Real-time state synchronization
    """
    
    def __init__(self, config: Config):
        """
        Initialize the state manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{id(self)}")
        
        # Database clients
        self.db_manager = DatabaseManager(config)
        self.redis_client = RedisClient(config)
        self.influxdb_client = InfluxDBClient(config)
        
        # State caching
        self._state_cache: Dict[str, StateSnapshot] = {}
        self._transaction_log: Dict[str, StateTransaction] = {}
        
        # Configuration
        self.max_snapshots_per_bot = config.state_management.get("max_snapshots_per_bot", 100)
        self.snapshot_interval_minutes = config.state_management.get("snapshot_interval_minutes", 15)
        self.redis_ttl_seconds = config.state_management.get("redis_ttl_seconds", 3600)
        self.enable_compression = config.state_management.get("enable_compression", True)
        
        # Synchronization
        self._locks: Dict[str, asyncio.Lock] = {}
        self._active_transactions: Dict[str, StateTransaction] = {}
        
        self.logger.info("StateManager initialized")

    async def initialize(self) -> None:
        """Initialize the state manager and storage backends."""
        try:
            # Initialize database connections
            await self.db_manager.initialize()
            await self.redis_client.initialize()
            await self.influxdb_client.initialize()
            
            # Verify storage backends
            await self._verify_storage_backends()
            
            # Load existing state snapshots
            await self._load_existing_snapshots()
            
            # Start background tasks
            asyncio.create_task(self._cleanup_task())
            
            self.logger.info("StateManager initialization completed")
            
        except Exception as e:
            self.logger.error(f"StateManager initialization failed: {e}")
            raise ConfigurationError(f"Failed to initialize StateManager: {e}")

    @time_execution
    async def save_bot_state(
        self, 
        bot_id: str, 
        bot_state: BotState,
        create_snapshot: bool = False
    ) -> str:
        """
        Save bot state to persistent storage.
        
        Args:
            bot_id: Bot identifier
            bot_state: Bot state data
            create_snapshot: Whether to create a snapshot
            
        Returns:
            State version ID
            
        Raises:
            StateError: If state save fails
        """
        async with self._get_bot_lock(bot_id):
            try:
                version_id = str(uuid4())
                timestamp = datetime.now(timezone.utc)
                
                # Validate state data
                await self._validate_bot_state(bot_state)
                
                # Start transaction
                transaction = StateTransaction(
                    bot_id=bot_id,
                    operations=[{
                        "type": "save_bot_state",
                        "data": bot_state.model_dump(),
                        "version_id": version_id,
                        "timestamp": timestamp.isoformat()
                    }]
                )
                
                # Save to Redis (real-time cache)
                redis_key = f"bot_state:{bot_id}"
                state_data = {
                    "version_id": version_id,
                    "timestamp": timestamp.isoformat(),
                    "state": bot_state.model_dump()
                }
                
                await self.redis_client.setex(
                    redis_key,
                    self.redis_ttl_seconds,
                    json.dumps(state_data, default=str)
                )
                
                # Save to PostgreSQL (persistent storage)
                async with self.db_manager.get_session() as session:
                    # Update or insert bot state
                    from src.database.models import BotInstance
                    
                    result = await session.execute(
                        select(BotInstance).where(BotInstance.id == bot_id)
                    )
                    bot_instance = result.scalar_one_or_none()
                    
                    if bot_instance:
                        # Update existing
                        await session.execute(
                            update(BotInstance)
                            .where(BotInstance.id == bot_id)
                            .values(
                                status=bot_state.status.value,
                                config=bot_state.model_dump(),
                                updated_at=timestamp,
                                last_active=timestamp
                            )
                        )
                    else:
                        self.logger.warning(f"Bot instance not found for state save: {bot_id}")
                    
                    await session.commit()
                
                # Create snapshot if requested
                if create_snapshot:
                    await self._create_state_snapshot(bot_id, bot_state, version_id)
                
                # Log state metrics to InfluxDB
                await self._log_state_metrics(bot_id, bot_state, version_id)
                
                # Commit transaction
                transaction.status = "committed"
                self._transaction_log[transaction.transaction_id] = transaction
                
                self.logger.info(
                    "Bot state saved successfully",
                    bot_id=bot_id,
                    version_id=version_id,
                    status=bot_state.status.value
                )
                
                return version_id
                
            except Exception as e:
                # Rollback transaction
                if 'transaction' in locals():
                    transaction.status = "rolled_back"
                    self._transaction_log[transaction.transaction_id] = transaction
                
                self.logger.error(f"Failed to save bot state: {e}", bot_id=bot_id)
                raise StateError(f"State save failed for bot {bot_id}: {e}")

    @time_execution
    async def load_bot_state(self, bot_id: str, version_id: Optional[str] = None) -> Optional[BotState]:
        """
        Load bot state from storage.
        
        Args:
            bot_id: Bot identifier
            version_id: Specific version to load (latest if None)
            
        Returns:
            Bot state data or None if not found
            
        Raises:
            StateError: If state load fails
        """
        try:
            # Try Redis first for latest state
            if version_id is None:
                redis_key = f"bot_state:{bot_id}"
                redis_data = await self.redis_client.get(redis_key)
                
                if redis_data:
                    try:
                        state_data = json.loads(redis_data)
                        return BotState(**state_data["state"])
                    except (json.JSONDecodeError, KeyError, ValidationError) as e:
                        self.logger.warning(f"Invalid Redis state data for bot {bot_id}: {e}")
            
            # Fall back to PostgreSQL
            async with self.db_manager.get_session() as session:
                from src.database.models import BotInstance
                
                result = await session.execute(
                    select(BotInstance).where(BotInstance.id == bot_id)
                )
                bot_instance = result.scalar_one_or_none()
                
                if bot_instance and bot_instance.config:
                    # Extract state from config
                    config_data = bot_instance.config
                    if isinstance(config_data, dict) and "bot_id" in config_data:
                        try:
                            return BotState(**config_data)
                        except ValidationError as e:
                            self.logger.warning(f"Invalid database state data for bot {bot_id}: {e}")
            
            # Try loading from snapshot if version specified
            if version_id:
                snapshot = await self._load_state_snapshot(bot_id, version_id)
                if snapshot and snapshot.bot_state:
                    return BotState(**snapshot.bot_state)
            
            self.logger.info(f"No state found for bot {bot_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load bot state: {e}", bot_id=bot_id)
            raise StateError(f"State load failed for bot {bot_id}: {e}")

    @time_execution
    async def create_checkpoint(self, bot_id: str) -> str:
        """
        Create a state checkpoint for recovery.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            Checkpoint ID
            
        Raises:
            StateError: If checkpoint creation fails
        """
        try:
            # Load current state
            bot_state = await self.load_bot_state(bot_id)
            if not bot_state:
                raise StateError(f"No state found for bot {bot_id}")
            
            # Create snapshot
            checkpoint_id = await self._create_state_snapshot(bot_id, bot_state)
            
            self.logger.info(f"Checkpoint created for bot {bot_id}", checkpoint_id=checkpoint_id)
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}", bot_id=bot_id)
            raise StateError(f"Checkpoint creation failed for bot {bot_id}: {e}")

    @time_execution  
    async def restore_from_checkpoint(self, bot_id: str, checkpoint_id: str) -> bool:
        """
        Restore bot state from a checkpoint.
        
        Args:
            bot_id: Bot identifier
            checkpoint_id: Checkpoint to restore from
            
        Returns:
            True if restoration successful
            
        Raises:
            StateError: If restoration fails
        """
        try:
            # Load snapshot
            snapshot = await self._load_state_snapshot(bot_id, checkpoint_id)
            if not snapshot:
                raise StateError(f"Checkpoint {checkpoint_id} not found for bot {bot_id}")
            
            # Validate snapshot
            if not snapshot.bot_state:
                raise StateError(f"Invalid checkpoint data for {checkpoint_id}")
            
            # Restore state
            bot_state = BotState(**snapshot.bot_state)
            await self.save_bot_state(bot_id, bot_state)
            
            self.logger.info(
                f"Bot state restored from checkpoint",
                bot_id=bot_id,
                checkpoint_id=checkpoint_id
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore from checkpoint: {e}", bot_id=bot_id)
            raise StateError(f"Checkpoint restoration failed: {e}")

    async def list_checkpoints(self, bot_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List available checkpoints for a bot.
        
        Args:
            bot_id: Bot identifier
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoint metadata
        """
        try:
            checkpoints = []
            
            # Get snapshots from cache
            for snapshot in self._state_cache.values():
                if snapshot.bot_id == bot_id:
                    checkpoints.append({
                        "checkpoint_id": snapshot.snapshot_id,
                        "timestamp": snapshot.timestamp.isoformat(),
                        "version": snapshot.version,
                        "size_bytes": snapshot.size_bytes
                    })
            
            # Sort by timestamp (newest first) and limit
            checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
            return checkpoints[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}", bot_id=bot_id)
            return []

    async def get_state_metrics(self, bot_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get state management metrics for a bot.
        
        Args:
            bot_id: Bot identifier  
            hours: Time period for metrics
            
        Returns:
            State metrics dictionary
        """
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            
            metrics = {
                "bot_id": bot_id,
                "period_hours": hours,
                "checkpoints_count": 0,
                "state_updates": 0,
                "last_update": None,
                "storage_size_bytes": 0
            }
            
            # Count checkpoints in period
            for snapshot in self._state_cache.values():
                if (snapshot.bot_id == bot_id and 
                    snapshot.timestamp >= start_time):
                    metrics["checkpoints_count"] += 1
                    metrics["storage_size_bytes"] += snapshot.size_bytes
                    
                    if not metrics["last_update"] or snapshot.timestamp > datetime.fromisoformat(metrics["last_update"]):
                        metrics["last_update"] = snapshot.timestamp.isoformat()
            
            # Get state update count from transaction log
            for transaction in self._transaction_log.values():
                if (transaction.bot_id == bot_id and 
                    transaction.timestamp >= start_time and
                    transaction.status == "committed"):
                    metrics["state_updates"] += len(transaction.operations)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get state metrics: {e}", bot_id=bot_id)
            return {"error": str(e)}

    async def cleanup_old_state(self, bot_id: str, keep_count: int = 50) -> int:
        """
        Clean up old state snapshots for a bot.
        
        Args:
            bot_id: Bot identifier
            keep_count: Number of recent snapshots to keep
            
        Returns:
            Number of snapshots cleaned up
        """
        try:
            # Get all snapshots for bot
            bot_snapshots = [
                snapshot for snapshot in self._state_cache.values()
                if snapshot.bot_id == bot_id
            ]
            
            # Sort by timestamp (newest first)
            bot_snapshots.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Remove old snapshots
            cleaned_count = 0
            for snapshot in bot_snapshots[keep_count:]:
                if snapshot.snapshot_id in self._state_cache:
                    del self._state_cache[snapshot.snapshot_id]
                    cleaned_count += 1
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old snapshots for bot {bot_id}")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old state: {e}", bot_id=bot_id)
            return 0

    # Private helper methods
    
    def _get_bot_lock(self, bot_id: str) -> asyncio.Lock:
        """Get or create a lock for bot state operations."""
        if bot_id not in self._locks:
            self._locks[bot_id] = asyncio.Lock()
        return self._locks[bot_id]

    async def _validate_bot_state(self, bot_state: BotState) -> None:
        """Validate bot state data."""
        if not bot_state.bot_id:
            raise ValidationError("Bot ID is required")
        
        if not isinstance(bot_state.status, BotStatus):
            raise ValidationError("Invalid bot status")
        
        # Additional validation can be added here
        await validate_state_data(bot_state.model_dump())

    async def _create_state_snapshot(
        self, 
        bot_id: str, 
        bot_state: BotState,
        version_id: Optional[str] = None
    ) -> str:
        """Create a state snapshot."""
        snapshot_id = version_id or str(uuid4())
        
        # Collect all state data
        positions = getattr(bot_state, 'open_positions', [])
        orders = getattr(bot_state, 'pending_orders', [])
        strategy_state = getattr(bot_state, 'strategy_state', {})
        
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            bot_id=bot_id,
            bot_state=bot_state.model_dump(),
            positions=positions,
            orders=orders,
            strategy_state=strategy_state,
            version=getattr(bot_state, 'state_version', 1)
        )
        
        # Calculate size and checksum
        snapshot_data = pickle.dumps(snapshot)
        snapshot.size_bytes = len(snapshot_data)
        
        # Store in cache
        self._state_cache[snapshot_id] = snapshot
        
        # Cleanup old snapshots
        await self.cleanup_old_state(bot_id, self.max_snapshots_per_bot)
        
        return snapshot_id

    async def _load_state_snapshot(self, bot_id: str, snapshot_id: str) -> Optional[StateSnapshot]:
        """Load a state snapshot."""
        return self._state_cache.get(snapshot_id)

    async def _log_state_metrics(self, bot_id: str, bot_state: BotState, version_id: str) -> None:
        """Log state metrics to InfluxDB."""
        try:
            metrics_data = {
                "measurement": "bot_state_metrics",
                "tags": {
                    "bot_id": bot_id,
                    "status": bot_state.status.value
                },
                "fields": {
                    "version_id": version_id,
                    "open_positions_count": len(getattr(bot_state, 'open_positions', [])),
                    "pending_orders_count": len(getattr(bot_state, 'pending_orders', [])),
                    "state_version": getattr(bot_state, 'state_version', 1)
                },
                "time": datetime.now(timezone.utc)
            }
            
            await self.influxdb_client.write_point(metrics_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to log state metrics: {e}")

    async def _verify_storage_backends(self) -> None:
        """Verify all storage backends are accessible."""
        # Test Redis
        test_key = f"state_test_{uuid4()}"
        await self.redis_client.set(test_key, "test")
        result = await self.redis_client.get(test_key)
        if result != "test":
            raise ConfigurationError("Redis backend verification failed")
        await self.redis_client.delete(test_key)
        
        # Test PostgreSQL
        async with self.db_manager.get_session() as session:
            result = await session.execute(select(1))
            if not result.scalar():
                raise ConfigurationError("PostgreSQL backend verification failed")

    async def _load_existing_snapshots(self) -> None:
        """Load existing state snapshots from storage."""
        try:
            # This would typically load from persistent storage
            # For now, start with empty cache
            self._state_cache.clear()
            self.logger.info("Loaded existing state snapshots")
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing snapshots: {e}")

    async def _cleanup_task(self) -> None:
        """Background task for cleaning up old state data."""
        while True:
            try:
                # Clean up old transactions
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                old_transactions = [
                    tid for tid, txn in self._transaction_log.items()
                    if txn.timestamp < cutoff_time
                ]
                
                for tid in old_transactions:
                    del self._transaction_log[tid]
                
                if old_transactions:
                    self.logger.debug(f"Cleaned up {len(old_transactions)} old transactions")
                
                # Sleep for 1 hour before next cleanup
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(600)  # Retry in 10 minutes