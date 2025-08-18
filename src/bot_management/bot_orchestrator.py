"""
Central orchestrator for managing multiple bot instances.

This module implements the BotOrchestrator class that serves as the central
controller for all bot instances, handling bot creation, lifecycle management,
resource coordination, and performance aggregation.

CRITICAL: This integrates with P-016 (execution engine), P-011 (strategies),
P-003+ (exchanges), and P-010A (capital management) components.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError
from src.core.logging import get_logger
from src.core.types import BotConfiguration, BotPriority, BotStatus

# MANDATORY: Import from P-002A (error handling)
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007A (utils)
from src.utils.decorators import log_calls, time_execution

from .bot_coordinator import BotCoordinator

# Import other bot management components
from .bot_instance import BotInstance
from .bot_monitor import BotMonitor
from .resource_manager import ResourceManager


class BotOrchestrator:
    """
    Central orchestrator for managing multiple bot instances.

    This class provides:
    - Bot creation and deletion operations
    - Bot lifecycle management (start/stop/pause/resume)
    - Global resource coordination
    - Performance aggregation across all bots
    - Emergency shutdown capabilities
    - Health monitoring and status reporting
    """

    def __init__(self, config: Config):
        """
        Initialize bot orchestrator.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.BotOrchestrator")
        self.error_handler = ErrorHandler(config.error_handling)

        # Core management components
        self.resource_manager = ResourceManager(config)
        self.bot_coordinator = BotCoordinator(config)
        self.bot_monitor = BotMonitor(config)

        # Bot instance tracking
        self.bot_instances: dict[str, BotInstance] = {}
        self.bot_configurations: dict[str, BotConfiguration] = {}

        # Orchestrator state
        self.is_running = False
        self.orchestrator_task = None
        self.emergency_shutdown = False

        # Bot management
        self.bots: dict[str, Any] = {}  # Bot instances keyed by bot_id
        self.orchestration_statistics = {
            "total_bots_created": 0,
            "total_bots_started": 0,
            "total_bots_stopped": 0,
            "total_errors": 0,
        }

        # Performance aggregation
        self.global_metrics = {
            "total_bots": 0,
            "running_bots": 0,
            "paused_bots": 0,
            "error_bots": 0,
            "total_trades": 0,
            "total_pnl": Decimal("0"),
            "total_allocated_capital": Decimal("0"),
            "average_win_rate": 0.0,
            "last_updated": datetime.now(timezone.utc),
        }

        # Configuration limits
        self.max_bots = config.bot_management.get("max_bots", 50)
        self.max_concurrent_startups = config.bot_management.get("max_concurrent_startups", 5)
        self.health_check_interval = config.bot_management.get("health_check_interval", 60)

        self.logger.info("Bot orchestrator initialized")

    @log_calls
    async def start(self) -> None:
        """
        Start the bot orchestrator system.

        Raises:
            ExecutionError: If startup fails
        """
        try:
            if self.is_running:
                self.logger.warning("Bot orchestrator is already running")
                return

            self.logger.info("Starting bot orchestrator")

            # Start resource manager
            await self.resource_manager.start()

            # Start bot coordinator
            await self.bot_coordinator.start()

            # Start bot monitor
            await self.bot_monitor.start()

            # Start orchestrator monitoring task
            self.orchestrator_task = asyncio.create_task(self._orchestrator_loop())

            self.is_running = True
            self.logger.info("Bot orchestrator started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start bot orchestrator: {e}")
            raise ExecutionError(f"Orchestrator startup failed: {e}")

    @log_calls
    async def stop(self) -> None:
        """
        Stop the bot orchestrator and all managed bots.

        Raises:
            ExecutionError: If shutdown fails
        """
        try:
            if not self.is_running:
                self.logger.warning("Bot orchestrator is not running")
                return

            self.logger.info("Stopping bot orchestrator")
            self.is_running = False

            # Stop orchestrator monitoring
            if self.orchestrator_task:
                self.orchestrator_task.cancel()

            # Stop all bot instances
            await self._stop_all_bots()

            # Stop management components
            await self.bot_monitor.stop()
            await self.bot_coordinator.stop()
            await self.resource_manager.stop()

            self.logger.info("Bot orchestrator stopped successfully")

        except Exception as e:
            self.logger.error(f"Failed to stop bot orchestrator: {e}")
            raise ExecutionError(f"Orchestrator shutdown failed: {e}")

    @time_execution
    @log_calls
    async def create_bot(self, bot_config: BotConfiguration) -> str:
        """
        Create a new bot instance.

        Args:
            bot_config: Bot configuration

        Returns:
            str: Bot ID of created bot

        Raises:
            ValidationError: If configuration is invalid
            ExecutionError: If bot creation fails
        """
        try:
            # Validate bot configuration
            await self._validate_bot_configuration(bot_config)

            # Check bot limits
            if len(self.bot_instances) >= self.max_bots:
                raise ExecutionError(f"Maximum bot limit reached: {self.max_bots}")

            # Check if bot ID already exists
            if bot_config.bot_id in self.bot_instances:
                raise ValidationError(f"Bot ID already exists: {bot_config.bot_id}")

            # Request resource allocation
            allocation_successful = await self.resource_manager.request_resources(
                bot_config.bot_id, bot_config.allocated_capital, bot_config.priority
            )

            if not allocation_successful:
                raise ExecutionError("Failed to allocate required resources")

            # Create bot instance
            bot_instance = BotInstance(self.config, bot_config)

            # Register with coordinator
            await self.bot_coordinator.register_bot(bot_config.bot_id, bot_config)

            # Register with monitor
            await self.bot_monitor.register_bot(bot_config.bot_id)

            # Store bot instance and configuration
            self.bot_instances[bot_config.bot_id] = bot_instance
            self.bot_configurations[bot_config.bot_id] = bot_config

            # Auto-start if configured
            if bot_config.auto_start:
                await self.start_bot(bot_config.bot_id)

            # Update global metrics
            await self._update_global_metrics()

            self.logger.info(
                "Bot created successfully",
                bot_id=bot_config.bot_id,
                bot_type=bot_config.bot_type.value,
                strategy=bot_config.strategy_name,
                auto_start=bot_config.auto_start,
            )

            return bot_config.bot_id

        except Exception as e:
            # Cleanup on failure
            await self._cleanup_failed_bot_creation(bot_config.bot_id)
            self.logger.error(f"Failed to create bot: {e}", bot_id=bot_config.bot_id)
            raise

    @log_calls
    async def delete_bot(self, bot_id: str, force: bool = False) -> bool:
        """
        Delete a bot instance.

        Args:
            bot_id: Bot identifier
            force: Force deletion even if bot is running

        Returns:
            bool: True if deletion successful

        Raises:
            ValidationError: If bot not found or cannot be deleted
            ExecutionError: If deletion fails
        """
        try:
            if bot_id not in self.bot_instances:
                raise ValidationError(f"Bot not found: {bot_id}")

            bot_instance = self.bot_instances[bot_id]
            bot_state = bot_instance.get_bot_state()

            # Check if bot can be safely deleted
            if not force and bot_state.status in [BotStatus.RUNNING, BotStatus.STARTING]:
                raise ValidationError("Cannot delete running bot (use force=True to override)")

            # Stop bot if running
            if bot_state.status in [BotStatus.RUNNING, BotStatus.PAUSED, BotStatus.STARTING]:
                await self.stop_bot(bot_id)

            # Unregister from components
            await self.bot_monitor.unregister_bot(bot_id)
            await self.bot_coordinator.unregister_bot(bot_id)

            # Release resources
            await self.resource_manager.release_resources(bot_id)

            # Remove from tracking
            del self.bot_instances[bot_id]
            del self.bot_configurations[bot_id]

            # Update global metrics
            await self._update_global_metrics()

            self.logger.info("Bot deleted successfully", bot_id=bot_id)
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete bot: {e}", bot_id=bot_id)
            raise ExecutionError(f"Bot deletion failed: {e}")

    @log_calls
    async def start_bot(self, bot_id: str) -> bool:
        """
        Start a specific bot instance.

        Args:
            bot_id: Bot identifier

        Returns:
            bool: True if start successful

        Raises:
            ValidationError: If bot not found
            ExecutionError: If start fails
        """
        try:
            if bot_id not in self.bot_instances:
                raise ValidationError(f"Bot not found: {bot_id}")

            bot_instance = self.bot_instances[bot_id]
            bot_state = bot_instance.get_bot_state()

            if bot_state.status == BotStatus.RUNNING:
                self.logger.warning("Bot is already running", bot_id=bot_id)
                return True

            if bot_state.status == BotStatus.STARTING:
                self.logger.warning("Bot is already starting", bot_id=bot_id)
                return True

            # Verify resource availability
            resources_available = await self.resource_manager.verify_resources(bot_id)
            if not resources_available:
                raise ExecutionError("Required resources not available")

            # Start bot instance
            await bot_instance.start()

            # Update global metrics
            await self._update_global_metrics()

            self.logger.info("Bot started successfully", bot_id=bot_id)
            return True

        except Exception as e:
            self.logger.error(f"Failed to start bot: {e}", bot_id=bot_id)
            raise ExecutionError(f"Bot start failed: {e}")

    @log_calls
    async def stop_bot(self, bot_id: str) -> bool:
        """
        Stop a specific bot instance.

        Args:
            bot_id: Bot identifier

        Returns:
            bool: True if stop successful

        Raises:
            ValidationError: If bot not found
            ExecutionError: If stop fails
        """
        try:
            if bot_id not in self.bot_instances:
                raise ValidationError(f"Bot not found: {bot_id}")

            bot_instance = self.bot_instances[bot_id]
            bot_state = bot_instance.get_bot_state()

            if bot_state.status in [BotStatus.STOPPED, BotStatus.STOPPING]:
                self.logger.warning("Bot is already stopped or stopping", bot_id=bot_id)
                return True

            # Stop bot instance
            await bot_instance.stop()

            # Update global metrics
            await self._update_global_metrics()

            self.logger.info("Bot stopped successfully", bot_id=bot_id)
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop bot: {e}", bot_id=bot_id)
            raise ExecutionError(f"Bot stop failed: {e}")

    @log_calls
    async def pause_bot(self, bot_id: str) -> bool:
        """
        Pause a specific bot instance.

        Args:
            bot_id: Bot identifier

        Returns:
            bool: True if pause successful
        """
        try:
            if bot_id not in self.bot_instances:
                raise ValidationError(f"Bot not found: {bot_id}")

            bot_instance = self.bot_instances[bot_id]
            await bot_instance.pause()

            await self._update_global_metrics()
            self.logger.info("Bot paused successfully", bot_id=bot_id)
            return True

        except Exception as e:
            self.logger.error(f"Failed to pause bot: {e}", bot_id=bot_id)
            raise ExecutionError(f"Bot pause failed: {e}")

    @log_calls
    async def resume_bot(self, bot_id: str) -> bool:
        """
        Resume a paused bot instance.

        Args:
            bot_id: Bot identifier

        Returns:
            bool: True if resume successful
        """
        try:
            if bot_id not in self.bot_instances:
                raise ValidationError(f"Bot not found: {bot_id}")

            bot_instance = self.bot_instances[bot_id]
            await bot_instance.resume()

            await self._update_global_metrics()
            self.logger.info("Bot resumed successfully", bot_id=bot_id)
            return True

        except Exception as e:
            self.logger.error(f"Failed to resume bot: {e}", bot_id=bot_id)
            raise ExecutionError(f"Bot resume failed: {e}")

    @log_calls
    async def start_all_bots(self, priority_filter: BotPriority | None = None) -> dict[str, bool]:
        """
        Start all bots or bots matching priority filter.

        Args:
            priority_filter: Optional priority filter

        Returns:
            dict: Bot ID to success status mapping
        """
        results = {}

        # Filter bots by priority if specified
        bots_to_start = []
        for bot_id, config in self.bot_configurations.items():
            if priority_filter is None or config.priority == priority_filter:
                bot_instance = self.bot_instances[bot_id]
                bot_state = bot_instance.get_bot_state()
                if bot_state.status in [BotStatus.CREATED, BotStatus.STOPPED]:
                    bots_to_start.append(bot_id)

        # Start bots in batches to respect concurrent startup limits
        batch_size = self.max_concurrent_startups
        for i in range(0, len(bots_to_start), batch_size):
            batch = bots_to_start[i : i + batch_size]

            # Start batch concurrently
            tasks = [self.start_bot(bot_id) for bot_id in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Record results
            for bot_id, result in zip(batch, batch_results, strict=False):
                results[bot_id] = not isinstance(result, Exception)
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to start bot in batch: {result}", bot_id=bot_id)

            # Small delay between batches
            if i + batch_size < len(bots_to_start):
                await asyncio.sleep(2)

        self.logger.info(
            "Batch bot start completed",
            total_bots=len(bots_to_start),
            successful=sum(results.values()),
            failed=len(results) - sum(results.values()),
        )

        return results

    @log_calls
    async def emergency_shutdown(self, reason: str) -> None:
        """
        Emergency shutdown of all bots and orchestrator.

        Args:
            reason: Reason for emergency shutdown
        """
        self.logger.critical(f"EMERGENCY SHUTDOWN INITIATED: {reason}")
        self.emergency_shutdown = True

        try:
            # Stop all bots immediately
            stop_tasks = []
            for bot_id in list(self.bot_instances.keys()):
                task = asyncio.create_task(self.stop_bot(bot_id))
                stop_tasks.append(task)

            # Wait for all stops with timeout
            await asyncio.wait_for(
                asyncio.gather(*stop_tasks, return_exceptions=True), timeout=30.0
            )

            # Stop orchestrator
            await self.stop()

            self.logger.critical("Emergency shutdown completed")

        except Exception as e:
            self.logger.critical(f"Emergency shutdown failed: {e}")

    async def get_orchestrator_status(self) -> dict[str, Any]:
        """Get comprehensive orchestrator status."""
        bot_summaries = {}
        for bot_id, bot_instance in self.bot_instances.items():
            bot_summaries[bot_id] = await bot_instance.get_bot_summary()

        return {
            "orchestrator": {
                "is_running": self.is_running,
                "emergency_shutdown": self.emergency_shutdown,
                "total_bots": len(self.bot_instances),
                "last_health_check": datetime.now(timezone.utc).isoformat(),
            },
            "global_metrics": self.global_metrics,
            "bots": bot_summaries,
            "resource_status": await self.resource_manager.get_resource_summary(),
            "coordination_status": await self.bot_coordinator.get_coordination_summary(),
        }

    async def get_bot_list(self, status_filter: BotStatus | None = None) -> list[dict[str, Any]]:
        """
        Get list of all bots with optional status filter.

        Args:
            status_filter: Optional status filter

        Returns:
            list: List of bot summaries
        """
        bot_list = []

        for _bot_id, bot_instance in self.bot_instances.items():
            bot_state = bot_instance.get_bot_state()

            if status_filter is None or bot_state.status == status_filter:
                bot_summary = await bot_instance.get_bot_summary()
                bot_list.append(bot_summary)

        return bot_list

    async def _orchestrator_loop(self) -> None:
        """Main orchestrator monitoring loop."""
        try:
            while self.is_running and not self.emergency_shutdown:
                try:
                    # Update global metrics
                    await self._update_global_metrics()

                    # Check bot health
                    await self._check_bot_health()

                    # Check resource constraints
                    await self._check_resource_constraints()

                    # Check for error conditions requiring intervention
                    await self._check_error_conditions()

                    # Wait for next cycle
                    await asyncio.sleep(self.health_check_interval)

                except Exception as e:
                    self.logger.error(f"Orchestrator loop error: {e}")
                    await asyncio.sleep(10)

        except asyncio.CancelledError:
            self.logger.info("Orchestrator monitoring cancelled")

    async def _validate_bot_configuration(self, bot_config: BotConfiguration) -> None:
        """Validate bot configuration."""
        # Check required fields
        if not bot_config.bot_id:
            raise ValidationError("Bot ID is required")

        if not bot_config.bot_name:
            raise ValidationError("Bot name is required")

        if not bot_config.strategy_name:
            raise ValidationError("Strategy name is required")

        if not bot_config.exchanges:
            raise ValidationError("At least one exchange is required")

        if not bot_config.symbols:
            raise ValidationError("At least one symbol is required")

        if bot_config.allocated_capital <= 0:
            raise ValidationError("Allocated capital must be positive")

        if bot_config.risk_percentage <= 0 or bot_config.risk_percentage > 1:
            raise ValidationError("Risk percentage must be between 0 and 1")

    async def _update_global_metrics(self) -> None:
        """Update global performance metrics."""
        total_bots = len(self.bot_instances)
        running_bots = 0
        paused_bots = 0
        error_bots = 0
        total_trades = 0
        total_pnl = Decimal("0")
        total_allocated_capital = Decimal("0")
        win_rates = []

        for bot_instance in self.bot_instances.values():
            bot_state = bot_instance.get_bot_state()
            bot_metrics = bot_instance.get_bot_metrics()
            bot_config = bot_instance.get_bot_config()

            # Count by status
            if bot_state.status == BotStatus.RUNNING:
                running_bots += 1
            elif bot_state.status == BotStatus.PAUSED:
                paused_bots += 1
            elif bot_state.status == BotStatus.ERROR:
                error_bots += 1

            # Aggregate metrics
            total_trades += bot_metrics.total_trades
            total_pnl += bot_metrics.total_pnl
            total_allocated_capital += bot_config.allocated_capital

            if bot_metrics.win_rate > 0:
                win_rates.append(bot_metrics.win_rate)

        # Update global metrics
        self.global_metrics.update(
            {
                "total_bots": total_bots,
                "running_bots": running_bots,
                "paused_bots": paused_bots,
                "error_bots": error_bots,
                "total_trades": total_trades,
                "total_pnl": total_pnl,
                "total_allocated_capital": total_allocated_capital,
                "average_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0.0,
                "last_updated": datetime.now(timezone.utc),
            }
        )

    async def _check_bot_health(self) -> None:
        """Check health of all bot instances."""
        unhealthy_bots = []

        for bot_id, bot_instance in self.bot_instances.items():
            bot_metrics = bot_instance.get_bot_metrics()

            # Check heartbeat freshness
            if bot_metrics.last_heartbeat:
                heartbeat_age = datetime.now(timezone.utc) - bot_metrics.last_heartbeat
                if heartbeat_age.total_seconds() > self.config.bot_management.get(
                    "heartbeat_timeout", 120
                ):
                    unhealthy_bots.append(bot_id)

        if unhealthy_bots:
            self.logger.warning(
                "Unhealthy bots detected", unhealthy_bots=unhealthy_bots, count=len(unhealthy_bots)
            )

    async def _check_resource_constraints(self) -> None:
        """Check for resource constraint violations."""
        resource_status = await self.resource_manager.get_resource_summary()

        # Check if any resources are over-allocated
        for resource_type, status in resource_status.items():
            if isinstance(status, dict) and status.get("usage_percentage", 0) > 0.9:
                self.logger.warning(
                    "High resource usage detected",
                    resource_type=resource_type,
                    usage=status.get("usage_percentage", 0),
                )

    async def _check_error_conditions(self) -> None:
        """Check for conditions requiring emergency intervention."""
        error_bot_count = self.global_metrics["error_bots"]
        total_bot_count = self.global_metrics["total_bots"]

        # Check if too many bots are in error state
        if total_bot_count > 0 and error_bot_count / total_bot_count > 0.5:
            self.logger.critical(
                "High error rate detected",
                error_bots=error_bot_count,
                total_bots=total_bot_count,
                error_rate=error_bot_count / total_bot_count,
            )

    async def _stop_all_bots(self) -> None:
        """Stop all bot instances."""
        if not self.bot_instances:
            return

        self.logger.info("Stopping all bots", bot_count=len(self.bot_instances))

        # Stop all bots concurrently
        stop_tasks = []
        for bot_id in list(self.bot_instances.keys()):
            task = asyncio.create_task(self.stop_bot(bot_id))
            stop_tasks.append(task)

        # Wait for all stops
        await asyncio.gather(*stop_tasks, return_exceptions=True)

    async def _cleanup_failed_bot_creation(self, bot_id: str) -> None:
        """Cleanup resources after failed bot creation."""
        try:
            # Remove from tracking if added
            if bot_id in self.bot_instances:
                del self.bot_instances[bot_id]
            if bot_id in self.bot_configurations:
                del self.bot_configurations[bot_id]

            # Release resources
            await self.resource_manager.release_resources(bot_id)

            # Unregister from components
            await self.bot_coordinator.unregister_bot(bot_id)
            await self.bot_monitor.unregister_bot(bot_id)

        except Exception as e:
            self.logger.warning(f"Cleanup after failed bot creation failed: {e}", bot_id=bot_id)
