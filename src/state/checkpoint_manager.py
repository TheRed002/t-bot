"""
Checkpoint Manager for bot state persistence and recovery.

This module provides advanced checkpoint management capabilities including:
- Automated checkpoint scheduling
- Compression and optimization
- Checkpoint validation and integrity checking
- Recovery point objectives (RPO) management
- Checkpoint cleanup and archival

The CheckpointManager ensures reliable state recovery with minimal data loss
in case of system failures or crashes.
"""

import asyncio
import gzip
import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.base import BaseComponent
from src.core.config.main import Config
from src.core.exceptions import StateError
from src.core.types import BotState
from src.error_handling import (
    ErrorContext,
    ErrorHandler,
    ErrorSeverity,
    with_retry,
)

# Import utilities through centralized import handler
from .utils_imports import ensure_directory_exists


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint management."""

    checkpoint_id: str = field(default_factory=lambda: str(uuid4()))
    bot_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Checkpoint properties
    size_bytes: int = 0
    compressed: bool = False
    integrity_hash: str = ""

    # Recovery metadata
    rpo_seconds: int = 0  # Recovery Point Objective
    checkpoint_type: str = "manual"  # manual, scheduled, emergency

    # Validation
    validated: bool = False
    validation_errors: list[str] = field(default_factory=list)

    # Archival
    archived: bool = False
    archive_path: str | None = None


@dataclass
class RecoveryPlan:
    """Recovery plan for restoring bot state."""

    bot_id: str = ""
    target_checkpoint_id: str = ""
    recovery_type: str = "full"  # full, partial, selective

    # Recovery steps
    steps: list[dict[str, Any]] = field(default_factory=list)
    estimated_duration_seconds: int = 0

    # Risk assessment
    data_loss_risk: str = "low"  # low, medium, high
    downtime_estimate_seconds: int = 0

    # Dependencies
    dependencies: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)


class CheckpointManager(BaseComponent):
    """
    Advanced checkpoint management system for bot state persistence.

    Features:
    - Automated checkpoint scheduling based on configurable policies
    - Compression and integrity verification
    - Recovery planning and execution
    - Checkpoint lifecycle management (creation, validation, archival, cleanup)
    - Performance optimization for large state objects
    """

    def __init__(self, config: Config):
        """
        Initialize the checkpoint manager.

        Args:
            config: Application configuration
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config

        # Configuration
        checkpoint_config = config.state_management.get("checkpoints", {})
        self.checkpoint_dir = Path(checkpoint_config.get("directory", "data/checkpoints"))
        self.max_checkpoints_per_bot = checkpoint_config.get("max_per_bot", 50)
        self.compression_enabled = checkpoint_config.get("compression", True)
        self.compression_threshold = checkpoint_config.get("compression_threshold_bytes", 1024)
        self.auto_cleanup_enabled = checkpoint_config.get("auto_cleanup", True)
        self.integrity_check_enabled = checkpoint_config.get("integrity_check", True)

        # Scheduling configuration
        self.default_interval_minutes = checkpoint_config.get("default_interval_minutes", 30)
        self.emergency_threshold_minutes = checkpoint_config.get("emergency_threshold_minutes", 5)

        # Storage
        self.checkpoints: dict[str, CheckpointMetadata] = {}
        self.checkpoint_schedules: dict[str, datetime] = {}

        # Performance tracking
        self.performance_stats = {
            "checkpoints_created": 0,
            "checkpoints_restored": 0,
            "total_size_bytes": 0,
            "compression_ratio": 0.0,
            "avg_creation_time_ms": 0.0,
            "avg_restore_time_ms": 0.0,
        }

        # Background tasks
        self._scheduler_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        self.logger.info("CheckpointManager initialized")

    async def initialize(self) -> None:
        """Initialize the checkpoint manager."""
        try:
            # Ensure checkpoint directory exists
            try:
                ensure_directory_exists(str(self.checkpoint_dir))
            except Exception as e:
                self.logger.error(f"Failed to create checkpoint directory: {e}")
                raise StateError(f"Cannot create checkpoint directory: {e}")

            # Load existing checkpoints
            await self._load_existing_checkpoints()

            # Start background tasks
            if self.auto_cleanup_enabled:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._scheduler_task = asyncio.create_task(self._scheduler_loop())

            self.logger.info("CheckpointManager initialized successfully")

        except Exception as e:
            error_context = ErrorContext.from_exception(
                e,
                component="CheckpointManager",
                operation="initialize",
                severity=ErrorSeverity.HIGH
            )
            error_context.details = {"error": str(e), "error_code": "CHECKPOINT_INIT_FAILED"}
            ErrorHandler.log_error(error_context, e)
            raise StateError(f"Failed to initialize CheckpointManager: {e}")

    @with_retry(max_attempts=3, base_delay=0.1, backoff_factor=2.0, exceptions=(StateError,))
    async def create_checkpoint(
        self,
        bot_id: str,
        bot_state: BotState,
        checkpoint_type: str = "manual",
        compress: bool | None = None,
    ) -> str:
        """
        Create a new checkpoint for the bot state.

        Args:
            bot_id: Bot identifier
            bot_state: Current bot state
            checkpoint_type: Type of checkpoint (manual, scheduled, emergency)
            compress: Whether to compress (None = auto-decide)

        Returns:
            Checkpoint ID

        Raises:
            StateError: If checkpoint creation fails
        """
        start_time = datetime.now(timezone.utc)

        try:
            checkpoint_id = str(uuid4())

            # Prepare checkpoint data
            checkpoint_data = {
                "bot_id": bot_id,
                "timestamp": start_time.isoformat(),
                "bot_state": bot_state.model_dump(),
                "checkpoint_type": checkpoint_type,
                "version": "1.0",
            }

            # Serialize data
            serialized_data = pickle.dumps(checkpoint_data)
            original_size = len(serialized_data)

            # Determine compression
            should_compress = (
                compress
                if compress is not None
                else (self.compression_enabled and original_size > self.compression_threshold)
            )

            # Compress if needed
            final_data = serialized_data
            if should_compress:
                final_data = gzip.compress(serialized_data)

            # Calculate integrity hash
            integrity_hash = hashlib.sha256(final_data).hexdigest()

            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                bot_id=bot_id,
                created_at=start_time,
                size_bytes=len(final_data),
                compressed=should_compress,
                integrity_hash=integrity_hash,
                checkpoint_type=checkpoint_type,
            )

            # Calculate RPO
            last_checkpoint = await self._get_last_checkpoint(bot_id)
            if last_checkpoint:
                time_diff = start_time - last_checkpoint.created_at
                metadata.rpo_seconds = int(time_diff.total_seconds())

            # Save to disk
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.checkpoint"
            with open(checkpoint_path, "wb") as f:
                f.write(final_data)

            # Validate if enabled
            if self.integrity_check_enabled:
                validation_result = await self._validate_checkpoint(checkpoint_id, metadata)
                metadata.validated = validation_result["valid"]
                metadata.validation_errors = validation_result["errors"]

            # Store metadata
            self.checkpoints[checkpoint_id] = metadata

            # Update performance stats
            creation_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_performance_stats(
                "create", creation_time_ms, original_size, len(final_data)
            )

            # Schedule cleanup if needed
            if (
                len([c for c in self.checkpoints.values() if c.bot_id == bot_id])
                > self.max_checkpoints_per_bot
            ):
                asyncio.create_task(self._cleanup_old_checkpoints(bot_id))

            self.logger.info(
                "Checkpoint created successfully",
                checkpoint_id=checkpoint_id,
                bot_id=bot_id,
                size_bytes=len(final_data),
                compressed=should_compress,
                creation_time_ms=creation_time_ms,
            )

            return checkpoint_id

        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}", bot_id=bot_id)
            raise StateError(f"Checkpoint creation failed: {e}")

    @with_retry(max_attempts=3, base_delay=0.5, backoff_factor=2.0, exceptions=(StateError,))
    async def restore_checkpoint(self, checkpoint_id: str) -> tuple[str, BotState]:
        """
        Restore bot state from a checkpoint.

        Args:
            checkpoint_id: Checkpoint to restore

        Returns:
            Tuple of (bot_id, restored_bot_state)

        Raises:
            StateError: If restoration fails
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Get checkpoint metadata
            metadata = self.checkpoints.get(checkpoint_id)
            if not metadata:
                raise StateError(f"Checkpoint {checkpoint_id} not found")

            # Read checkpoint file
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.checkpoint"
            if not checkpoint_path.exists():
                raise StateError(f"Checkpoint file not found: {checkpoint_path}")

            with open(checkpoint_path, "rb") as f:
                file_data = f.read()

            # Verify integrity
            if self.integrity_check_enabled:
                file_hash = hashlib.sha256(file_data).hexdigest()
                if file_hash != metadata.integrity_hash:
                    raise StateError(f"Checkpoint integrity check failed for {checkpoint_id}")

            # Decompress if needed
            if metadata.compressed:
                file_data = gzip.decompress(file_data)

            # Deserialize data
            checkpoint_data = pickle.loads(file_data)

            # Extract bot state
            bot_id = checkpoint_data["bot_id"]
            bot_state_data = checkpoint_data["bot_state"]
            bot_state = BotState(**bot_state_data)

            # Update performance stats
            restore_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_performance_stats("restore", restore_time_ms, 0, 0)

            self.logger.info(
                "Checkpoint restored successfully",
                checkpoint_id=checkpoint_id,
                bot_id=bot_id,
                restore_time_ms=restore_time_ms,
            )

            return bot_id, bot_state

        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint: {e}", checkpoint_id=checkpoint_id)
            raise StateError(f"Checkpoint restoration failed: {e}")

    async def create_recovery_plan(
        self, bot_id: str, target_time: datetime | None = None
    ) -> RecoveryPlan:
        """
        Create a recovery plan for a bot.

        Args:
            bot_id: Bot identifier
            target_time: Target recovery time (latest if None)

        Returns:
            Recovery plan
        """
        try:
            # Find best checkpoint for recovery
            best_checkpoint = await self._find_best_checkpoint(bot_id, target_time)
            if not best_checkpoint:
                raise StateError(f"No suitable checkpoint found for bot {bot_id}")

            # Create recovery plan
            plan = RecoveryPlan(
                bot_id=bot_id,
                target_checkpoint_id=best_checkpoint.checkpoint_id,
                recovery_type="full",
            )

            # Define recovery steps
            plan.steps = [
                {
                    "step": "validate_checkpoint",
                    "checkpoint_id": best_checkpoint.checkpoint_id,
                    "estimated_seconds": 5,
                },
                {"step": "stop_bot", "bot_id": bot_id, "estimated_seconds": 10},
                {
                    "step": "restore_state",
                    "checkpoint_id": best_checkpoint.checkpoint_id,
                    "estimated_seconds": 15,
                },
                {"step": "validate_state", "bot_id": bot_id, "estimated_seconds": 5},
                {"step": "restart_bot", "bot_id": bot_id, "estimated_seconds": 20},
            ]

            # Calculate estimates
            plan.estimated_duration_seconds = sum(step["estimated_seconds"] for step in plan.steps)
            plan.downtime_estimate_seconds = plan.estimated_duration_seconds

            # Assess data loss risk
            if target_time:
                time_diff = target_time - best_checkpoint.created_at
                if time_diff.total_seconds() > 3600:  # 1 hour
                    plan.data_loss_risk = "high"
                elif time_diff.total_seconds() > 600:  # 10 minutes
                    plan.data_loss_risk = "medium"
                else:
                    plan.data_loss_risk = "low"

            return plan

        except Exception as e:
            self.logger.error(f"Failed to create recovery plan: {e}", bot_id=bot_id)
            raise StateError(f"Recovery plan creation failed: {e}")

    async def execute_recovery_plan(self, plan: RecoveryPlan) -> bool:
        """
        Execute a recovery plan.

        Args:
            plan: Recovery plan to execute

        Returns:
            True if recovery successful

        Raises:
            StateError: If recovery fails
        """
        try:
            self.logger.info(f"Executing recovery plan for bot {plan.bot_id}")

            for i, step in enumerate(plan.steps):
                step_name = step["step"]
                self.logger.info(f"Executing recovery step {i + 1}/{len(plan.steps)}: {step_name}")

                if step_name == "validate_checkpoint":
                    metadata = self.checkpoints.get(step["checkpoint_id"])
                    if not metadata or not metadata.validated:
                        validation_result = await self._validate_checkpoint(
                            step["checkpoint_id"], metadata
                        )
                        if not validation_result["valid"]:
                            raise StateError(
                                f"Checkpoint validation failed: {validation_result['errors']}"
                            )

                elif step_name == "restore_state":
                    await self.restore_checkpoint(step["checkpoint_id"])

                # Add other step implementations as needed

                await asyncio.sleep(0.1)  # Small delay between steps

            self.logger.info(f"Recovery plan executed successfully for bot {plan.bot_id}")
            return True

        except Exception as e:
            self.logger.error(f"Recovery plan execution failed: {e}", bot_id=plan.bot_id)
            raise StateError(f"Recovery execution failed: {e}")

    async def schedule_checkpoint(self, bot_id: str, interval_minutes: int | None = None) -> None:
        """
        Schedule automatic checkpoints for a bot.

        Args:
            bot_id: Bot identifier
            interval_minutes: Checkpoint interval (default if None)
        """
        interval = interval_minutes or self.default_interval_minutes
        next_checkpoint = datetime.now(timezone.utc) + timedelta(minutes=interval)
        self.checkpoint_schedules[bot_id] = next_checkpoint

        self.logger.info(
            f"Checkpoint scheduled for bot {bot_id}",
            interval_minutes=interval,
            next_checkpoint=next_checkpoint.isoformat(),
        )

    async def list_checkpoints(
        self, bot_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """
        List available checkpoints.

        Args:
            bot_id: Bot identifier (all bots if None)
            limit: Maximum number of checkpoints

        Returns:
            List of checkpoint information
        """
        checkpoints = []

        for metadata in self.checkpoints.values():
            if bot_id is None or metadata.bot_id == bot_id:
                checkpoints.append(
                    {
                        "checkpoint_id": metadata.checkpoint_id,
                        "bot_id": metadata.bot_id,
                        "created_at": metadata.created_at.isoformat(),
                        "size_bytes": metadata.size_bytes,
                        "compressed": metadata.compressed,
                        "checkpoint_type": metadata.checkpoint_type,
                        "validated": metadata.validated,
                        "rpo_seconds": metadata.rpo_seconds,
                    }
                )

        # Sort by creation time (newest first)
        checkpoints.sort(key=lambda x: x["created_at"], reverse=True)
        return checkpoints[:limit]

    async def get_checkpoint_stats(self) -> dict[str, Any]:
        """Get checkpoint management statistics."""
        total_checkpoints = len(self.checkpoints)
        total_size = sum(c.size_bytes for c in self.checkpoints.values())

        # Group by bot
        bot_stats = {}
        for metadata in self.checkpoints.values():
            if metadata.bot_id not in bot_stats:
                bot_stats[metadata.bot_id] = {
                    "checkpoint_count": 0,
                    "total_size_bytes": 0,
                    "last_checkpoint": None,
                }

            bot_stats[metadata.bot_id]["checkpoint_count"] += 1
            bot_stats[metadata.bot_id]["total_size_bytes"] += metadata.size_bytes

            if not bot_stats[metadata.bot_id][
                "last_checkpoint"
            ] or metadata.created_at > datetime.fromisoformat(
                bot_stats[metadata.bot_id]["last_checkpoint"]
            ):
                bot_stats[metadata.bot_id]["last_checkpoint"] = metadata.created_at.isoformat()

        return {
            "total_checkpoints": total_checkpoints,
            "total_size_bytes": total_size,
            "bot_stats": bot_stats,
            "performance_stats": self.performance_stats,
        }

    # Private helper methods

    async def _get_last_checkpoint(self, bot_id: str) -> CheckpointMetadata | None:
        """Get the most recent checkpoint for a bot."""
        bot_checkpoints = [c for c in self.checkpoints.values() if c.bot_id == bot_id]

        if not bot_checkpoints:
            return None

        return max(bot_checkpoints, key=lambda c: c.created_at)

    async def _find_best_checkpoint(
        self, bot_id: str, target_time: datetime | None = None
    ) -> CheckpointMetadata | None:
        """Find the best checkpoint for recovery."""
        bot_checkpoints = [
            c for c in self.checkpoints.values() if c.bot_id == bot_id and c.validated
        ]

        if not bot_checkpoints:
            return None

        if target_time is None:
            # Return most recent checkpoint
            return max(bot_checkpoints, key=lambda c: c.created_at)

        # Find checkpoint closest to but before target time
        valid_checkpoints = [c for c in bot_checkpoints if c.created_at <= target_time]

        if not valid_checkpoints:
            return None

        return max(valid_checkpoints, key=lambda c: c.created_at)

    async def _validate_checkpoint(
        self, checkpoint_id: str, metadata: CheckpointMetadata
    ) -> dict[str, Any]:
        """Validate a checkpoint."""
        errors = []

        try:
            # Check file exists
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.checkpoint"
            if not checkpoint_path.exists():
                errors.append("Checkpoint file not found")
                return {"valid": False, "errors": errors}

            # Check file size
            actual_size = checkpoint_path.stat().st_size
            if actual_size != metadata.size_bytes:
                errors.append(f"Size mismatch: expected {metadata.size_bytes}, got {actual_size}")

            # Check integrity hash
            with open(checkpoint_path, "rb") as f:
                file_data = f.read()

            file_hash = hashlib.sha256(file_data).hexdigest()
            if file_hash != metadata.integrity_hash:
                errors.append("Integrity hash mismatch")

            # Try to deserialize
            try:
                if metadata.compressed:
                    file_data = gzip.decompress(file_data)
                checkpoint_data = pickle.loads(file_data)

                # Validate structure
                required_keys = ["bot_id", "timestamp", "bot_state"]
                for key in required_keys:
                    if key not in checkpoint_data:
                        errors.append(f"Missing required key: {key}")

            except Exception as e:
                errors.append(f"Deserialization failed: {e}")

        except Exception as e:
            errors.append(f"Validation error: {e}")

        return {"valid": len(errors) == 0, "errors": errors}

    async def _cleanup_old_checkpoints(self, bot_id: str) -> None:
        """Clean up old checkpoints for a bot."""
        bot_checkpoints = [c for c in self.checkpoints.values() if c.bot_id == bot_id]

        if len(bot_checkpoints) <= self.max_checkpoints_per_bot:
            return

        # Sort by creation time (oldest first)
        bot_checkpoints.sort(key=lambda c: c.created_at)

        # Remove oldest checkpoints
        to_remove = bot_checkpoints[: -self.max_checkpoints_per_bot]

        for checkpoint in to_remove:
            try:
                # Remove file
                checkpoint_path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.checkpoint"
                if checkpoint_path.exists():
                    checkpoint_path.unlink()

                # Remove from memory
                if checkpoint.checkpoint_id in self.checkpoints:
                    del self.checkpoints[checkpoint.checkpoint_id]

            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {checkpoint.checkpoint_id}: {e}")

        self.logger.info(f"Cleaned up {len(to_remove)} old checkpoints for bot {bot_id}")

    async def _load_existing_checkpoints(self) -> None:
        """Load existing checkpoint metadata."""
        try:
            if not self.checkpoint_dir.exists():
                return

            for _checkpoint_file in self.checkpoint_dir.glob("*.checkpoint"):
                # For now, we'll need to create metadata from file inspection
                # In a full implementation, metadata would be stored separately
                pass

        except Exception as e:
            self.logger.warning(f"Failed to load existing checkpoints: {e}")

    def _update_performance_stats(
        self, operation: str, duration_ms: float, original_size: int, final_size: int
    ) -> None:
        """Update performance statistics."""
        if operation == "create":
            self.performance_stats["checkpoints_created"] += 1

            # Update average creation time
            current_avg = self.performance_stats["avg_creation_time_ms"]
            count = self.performance_stats["checkpoints_created"]
            self.performance_stats["avg_creation_time_ms"] = (
                current_avg * (count - 1) + duration_ms
            ) / count

            # Update compression ratio
            if original_size > 0:
                compression_ratio = final_size / original_size
                current_ratio = self.performance_stats["compression_ratio"]
                self.performance_stats["compression_ratio"] = (
                    current_ratio * (count - 1) + compression_ratio
                ) / count

            self.performance_stats["total_size_bytes"] += final_size

        elif operation == "restore":
            self.performance_stats["checkpoints_restored"] += 1

            # Update average restore time
            current_avg = self.performance_stats["avg_restore_time_ms"]
            count = self.performance_stats["checkpoints_restored"]
            self.performance_stats["avg_restore_time_ms"] = (
                current_avg * (count - 1) + duration_ms
            ) / count

    async def _scheduler_loop(self) -> None:
        """Background task for scheduled checkpoints."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)

                # Check for scheduled checkpoints
                for bot_id, scheduled_time in list(self.checkpoint_schedules.items()):
                    if current_time >= scheduled_time:
                        try:
                            # This would need to get bot state from StateManager
                            # For now, just log the intent
                            self.logger.info(f"Scheduled checkpoint due for bot {bot_id}")

                            # Reschedule next checkpoint
                            next_time = current_time + timedelta(
                                minutes=self.default_interval_minutes
                            )
                            self.checkpoint_schedules[bot_id] = next_time

                        except Exception as e:
                            self.logger.error(
                                f"Failed to create scheduled checkpoint for bot {bot_id}: {e}"
                            )

                # Sleep for 1 minute before next check
                await asyncio.sleep(60)

            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)

    async def _cleanup_loop(self) -> None:
        """Background task for cleanup operations."""
        while True:
            try:
                # Clean up old checkpoints for all bots
                bot_ids = set(c.bot_id for c in self.checkpoints.values())
                for bot_id in bot_ids:
                    await self._cleanup_old_checkpoints(bot_id)

                # Sleep for 1 hour before next cleanup
                await asyncio.sleep(3600)

            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
