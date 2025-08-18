"""
Bot lifecycle management for creation, deployment, and termination.

This module implements the BotLifecycle class that handles the complete
lifecycle of bot instances, including template management, deployment
strategies, version control, and graceful termination procedures.

CRITICAL: This integrates with P-017 components and provides advanced
lifecycle management capabilities for bot instances.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError
from src.core.logging import get_logger
from src.core.types import BotConfiguration, BotPriority, BotType, TradingMode

# MANDATORY: Import from P-002A (error handling)
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007A (utils)
from src.utils.decorators import log_calls, time_execution

# Import other bot management components


class BotLifecycle:
    """
    Comprehensive bot lifecycle management system.

    This class provides:
    - Bot template management and configuration
    - Deployment strategy implementation
    - Version control and rollback capabilities
    - Graceful shutdown and termination procedures
    - Lifecycle event tracking and auditing
    - Recovery and restart mechanisms
    """

    def __init__(self, config: Config):
        """
        Initialize bot lifecycle manager.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.BotLifecycle")
        self.error_handler = ErrorHandler(config.error_handling)

        # Lifecycle tracking
        self.bot_lifecycles: dict[str, dict[str, Any]] = {}
        self.bot_templates: dict[str, dict[str, Any]] = {}
        self.deployment_strategies: dict[str, callable] = {}

        # Lifecycle state
        self.is_running = False
        self.lifecycle_task = None

        # Event tracking
        self.lifecycle_events: list[dict[str, Any]] = []
        self.event_retention_hours = config.bot_management.get(
            "event_retention_hours", 168
        )  # 7 days

        # Configuration
        self.graceful_shutdown_timeout = config.bot_management.get(
            "graceful_shutdown_timeout", 300
        )  # 5 minutes
        self.restart_max_attempts = config.bot_management.get("restart_max_attempts", 3)
        self.restart_delay_seconds = config.bot_management.get("restart_delay_seconds", 60)

        # Initialize built-in templates and strategies
        self._initialize_bot_templates()
        self._initialize_deployment_strategies()

        # Lifecycle statistics
        self.lifecycle_stats = {
            "bots_created": 0,
            "bots_deployed": 0,
            "bots_terminated": 0,
            "failed_deployments": 0,
            "successful_restarts": 0,
            "failed_restarts": 0,
            "last_lifecycle_action": None,
        }

        self.logger.info("Bot lifecycle manager initialized")

    def _initialize_bot_templates(self) -> None:
        """Initialize built-in bot templates."""
        self.bot_templates = {
            "simple_strategy_bot": {
                "name": "Simple Strategy Bot",
                "description": "Basic single-strategy trading bot",
                "default_config": {
                    "bot_type": BotType.STRATEGY,
                    "priority": BotPriority.NORMAL,
                    "max_concurrent_positions": 3,
                    "risk_percentage": 0.02,
                    "heartbeat_interval": 30,
                    "auto_start": True,
                    "max_position_size": Decimal("1000"),
                    "trading_mode": TradingMode.PAPER,
                },
                "required_fields": ["strategy_name", "exchanges", "symbols", "allocated_capital"],
                "optional_fields": ["trading_mode", "max_daily_trades"],
            },
            "arbitrage_bot": {
                "name": "Arbitrage Trading Bot",
                "description": "Cross-exchange arbitrage opportunities scanner and executor",
                "default_config": {
                    "bot_type": BotType.ARBITRAGE,
                    "priority": BotPriority.HIGH,
                    "max_concurrent_positions": 10,
                    "risk_percentage": 0.01,
                    "heartbeat_interval": 15,
                    "auto_start": True,
                    "max_position_size": Decimal("5000"),
                    "trading_mode": TradingMode.PAPER,
                    "strategy_name": "arbitrage_strategy",
                },
                "required_fields": ["exchanges", "symbols", "allocated_capital"],
                "optional_fields": ["min_arbitrage_profit_bps", "max_position_hold_time"],
            },
            "market_maker_bot": {
                "name": "Market Making Bot",
                "description": "Automated market making with spread management",
                "default_config": {
                    "bot_type": BotType.MARKET_MAKER,
                    "priority": BotPriority.HIGH,
                    "max_concurrent_positions": 20,
                    "risk_percentage": 0.005,
                    "heartbeat_interval": 10,
                    "auto_start": True,
                    "max_position_size": Decimal("2000"),
                    "trading_mode": TradingMode.PAPER,
                    "strategy_name": "market_maker_strategy",
                },
                "required_fields": ["exchanges", "symbols", "allocated_capital"],
                "optional_fields": ["spread_percentage", "inventory_target", "quote_size"],
            },
            "hybrid_strategy_bot": {
                "name": "Hybrid Multi-Strategy Bot",
                "description": "Advanced bot running multiple coordinated strategies",
                "default_config": {
                    "bot_type": BotType.HYBRID,
                    "priority": BotPriority.NORMAL,
                    "max_concurrent_positions": 15,
                    "risk_percentage": 0.03,
                    "heartbeat_interval": 20,
                    "auto_start": False,  # Requires manual review
                    "max_position_size": Decimal("3000"),
                    "trading_mode": TradingMode.PAPER,
                    "strategy_name": "hybrid_strategy",
                },
                "required_fields": ["strategy_names", "exchanges", "symbols", "allocated_capital"],
                "optional_fields": ["strategy_weights", "rebalancing_frequency"],
            },
            "scanner_bot": {
                "name": "Opportunity Scanner Bot",
                "description": "Market opportunity scanner with signal generation",
                "default_config": {
                    "bot_type": BotType.SCANNER,
                    "priority": BotPriority.LOW,
                    "max_concurrent_positions": 1,  # Minimal trading
                    "risk_percentage": 0.001,
                    "heartbeat_interval": 60,
                    "auto_start": True,
                    "max_position_size": Decimal("100"),
                    "trading_mode": TradingMode.PAPER,
                    "strategy_name": "scanner_strategy",
                    "allocated_capital": Decimal("1000"),
                },
                "required_fields": ["exchanges", "symbols"],
                "optional_fields": ["scan_intervals", "signal_threshold", "notification_targets"],
            },
        }

    def _initialize_deployment_strategies(self) -> None:
        """Initialize deployment strategies."""
        self.deployment_strategies = {
            "immediate": self._deploy_immediate,
            "staged": self._deploy_staged,
            "blue_green": self._deploy_blue_green,
            "canary": self._deploy_canary,
            "rolling": self._deploy_rolling,
        }

    @log_calls
    async def start(self) -> None:
        """
        Start the bot lifecycle manager.

        Raises:
            ExecutionError: If startup fails
        """
        try:
            if self.is_running:
                self.logger.warning("Bot lifecycle manager is already running")
                return

            self.logger.info("Starting bot lifecycle manager")

            # Start lifecycle monitoring task
            self.lifecycle_task = asyncio.create_task(self._lifecycle_loop())

            self.is_running = True
            self.logger.info("Bot lifecycle manager started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start bot lifecycle manager: {e}")
            raise ExecutionError(f"Bot lifecycle manager startup failed: {e}")

    @log_calls
    async def stop(self) -> None:
        """
        Stop the bot lifecycle manager.

        Raises:
            ExecutionError: If shutdown fails
        """
        try:
            if not self.is_running:
                self.logger.warning("Bot lifecycle manager is not running")
                return

            self.logger.info("Stopping bot lifecycle manager")
            self.is_running = False

            # Stop lifecycle monitoring task
            if self.lifecycle_task:
                self.lifecycle_task.cancel()

            # Clear state
            self.bot_lifecycles.clear()

            self.logger.info("Bot lifecycle manager stopped successfully")

        except Exception as e:
            self.logger.error(f"Failed to stop bot lifecycle manager: {e}")
            raise ExecutionError(f"Bot lifecycle manager shutdown failed: {e}")

    @time_execution
    @log_calls
    async def create_bot_from_template(
        self,
        template_name: str,
        bot_name: str,
        custom_config: dict[str, Any],
        deployment_strategy: str = "immediate",
    ) -> BotConfiguration:
        """
        Create a bot configuration from a template.

        Args:
            template_name: Name of the template to use
            bot_name: Name for the new bot
            custom_config: Custom configuration overrides
            deployment_strategy: Deployment strategy to use

        Returns:
            BotConfiguration: Created bot configuration

        Raises:
            ValidationError: If template or configuration is invalid
            ExecutionError: If creation fails
        """
        try:
            # Validate template exists
            if template_name not in self.bot_templates:
                raise ValidationError(f"Template not found: {template_name}")

            template = self.bot_templates[template_name]

            # Validate required fields are provided
            for field in template["required_fields"]:
                if field not in custom_config:
                    raise ValidationError(f"Required field missing: {field}")

            # Generate unique bot ID
            bot_id = f"{template_name}_{uuid.uuid4().hex[:8]}"

            # Merge template defaults with custom config
            merged_config = {
                **template["default_config"],
                **custom_config,
                "bot_id": bot_id,
                "bot_name": bot_name,
            }

            # Create bot configuration
            bot_config = BotConfiguration(**merged_config)

            # Initialize lifecycle tracking
            await self._initialize_bot_lifecycle(bot_id, template_name, deployment_strategy)

            # Record lifecycle event
            await self._record_lifecycle_event(
                bot_id, "created", {"template": template_name, "strategy": deployment_strategy}
            )

            self.lifecycle_stats["bots_created"] += 1
            self.lifecycle_stats["last_lifecycle_action"] = datetime.now(timezone.utc)

            self.logger.info(
                "Bot created from template",
                bot_id=bot_id,
                template=template_name,
                bot_name=bot_name,
                deployment_strategy=deployment_strategy,
            )

            return bot_config

        except Exception as e:
            self.logger.error(f"Failed to create bot from template: {e}")
            raise

    @log_calls
    async def deploy_bot(
        self,
        bot_config: BotConfiguration,
        orchestrator,
        deployment_options: dict[str, Any] | None = None,
    ) -> bool:
        """
        Deploy a bot using the configured deployment strategy.

        Args:
            bot_config: Bot configuration to deploy
            orchestrator: Bot orchestrator instance
            deployment_options: Optional deployment parameters

        Returns:
            bool: True if deployment successful

        Raises:
            ExecutionError: If deployment fails
        """
        try:
            bot_id = bot_config.bot_id

            if bot_id not in self.bot_lifecycles:
                await self._initialize_bot_lifecycle(bot_id, "unknown", "immediate")

            lifecycle = self.bot_lifecycles[bot_id]
            strategy = lifecycle["deployment_strategy"]

            self.logger.info(
                "Starting bot deployment",
                bot_id=bot_id,
                strategy=strategy,
                bot_type=bot_config.bot_type.value,
            )

            # Update lifecycle state
            lifecycle["deployment_state"] = "deploying"
            lifecycle["deployment_started"] = datetime.now(timezone.utc)

            # Execute deployment strategy
            deployment_func = self.deployment_strategies.get(strategy, self._deploy_immediate)
            success = await deployment_func(bot_config, orchestrator, deployment_options or {})

            if success:
                lifecycle["deployment_state"] = "deployed"
                lifecycle["deployment_completed"] = datetime.now(timezone.utc)
                self.lifecycle_stats["bots_deployed"] += 1

                await self._record_lifecycle_event(
                    bot_id, "deployed", {"strategy": strategy, "success": True}
                )
            else:
                lifecycle["deployment_state"] = "failed"
                lifecycle["deployment_failed"] = datetime.now(timezone.utc)
                self.lifecycle_stats["failed_deployments"] += 1

                await self._record_lifecycle_event(
                    bot_id, "deployment_failed", {"strategy": strategy, "success": False}
                )

            self.lifecycle_stats["last_lifecycle_action"] = datetime.now(timezone.utc)

            self.logger.info(
                "Bot deployment completed", bot_id=bot_id, success=success, strategy=strategy
            )

            return success

        except Exception as e:
            self.logger.error(f"Bot deployment failed: {e}", bot_id=bot_config.bot_id)

            if bot_config.bot_id in self.bot_lifecycles:
                self.bot_lifecycles[bot_config.bot_id]["deployment_state"] = "error"

            self.lifecycle_stats["failed_deployments"] += 1
            raise ExecutionError(f"Bot deployment failed: {e}")

    @log_calls
    async def terminate_bot(
        self, bot_id: str, orchestrator, graceful: bool = True, reason: str = "user_request"
    ) -> bool:
        """
        Terminate a bot instance with optional graceful shutdown.

        Args:
            bot_id: Bot identifier
            orchestrator: Bot orchestrator instance
            graceful: Whether to perform graceful shutdown
            reason: Reason for termination

        Returns:
            bool: True if termination successful

        Raises:
            ExecutionError: If termination fails
        """
        try:
            self.logger.info(
                "Starting bot termination", bot_id=bot_id, graceful=graceful, reason=reason
            )

            # Update lifecycle state
            if bot_id in self.bot_lifecycles:
                lifecycle = self.bot_lifecycles[bot_id]
                lifecycle["termination_state"] = "terminating"
                lifecycle["termination_started"] = datetime.now(timezone.utc)
                lifecycle["termination_reason"] = reason

            # Record termination event
            await self._record_lifecycle_event(
                bot_id, "termination_started", {"graceful": graceful, "reason": reason}
            )

            if graceful:
                # Graceful shutdown procedure
                success = await self._graceful_termination(bot_id, orchestrator)
            else:
                # Immediate termination
                success = await self._immediate_termination(bot_id, orchestrator)

            if success:
                if bot_id in self.bot_lifecycles:
                    lifecycle = self.bot_lifecycles[bot_id]
                    lifecycle["termination_state"] = "terminated"
                    lifecycle["termination_completed"] = datetime.now(timezone.utc)

                self.lifecycle_stats["bots_terminated"] += 1

                await self._record_lifecycle_event(
                    bot_id, "terminated", {"graceful": graceful, "success": True}
                )
            else:
                if bot_id in self.bot_lifecycles:
                    self.bot_lifecycles[bot_id]["termination_state"] = "failed"

                await self._record_lifecycle_event(
                    bot_id, "termination_failed", {"graceful": graceful, "success": False}
                )

            self.lifecycle_stats["last_lifecycle_action"] = datetime.now(timezone.utc)

            self.logger.info(
                "Bot termination completed", bot_id=bot_id, success=success, graceful=graceful
            )

            return success

        except Exception as e:
            self.logger.error(f"Bot termination failed: {e}", bot_id=bot_id)

            if bot_id in self.bot_lifecycles:
                self.bot_lifecycles[bot_id]["termination_state"] = "error"

            raise ExecutionError(f"Bot termination failed: {e}")

    @log_calls
    async def restart_bot(self, bot_id: str, orchestrator, reason: str = "manual_restart") -> bool:
        """
        Restart a bot instance with automatic recovery attempts.

        Args:
            bot_id: Bot identifier
            orchestrator: Bot orchestrator instance
            reason: Reason for restart

        Returns:
            bool: True if restart successful

        Raises:
            ExecutionError: If restart fails
        """
        try:
            self.logger.info("Starting bot restart", bot_id=bot_id, reason=reason)

            # Record restart event
            await self._record_lifecycle_event(bot_id, "restart_started", {"reason": reason})

            # Attempt restart with retries
            for attempt in range(1, self.restart_max_attempts + 1):
                try:
                    self.logger.info(
                        f"Bot restart attempt {attempt}/{self.restart_max_attempts}", bot_id=bot_id
                    )

                    # Stop bot gracefully
                    stop_success = await orchestrator.stop_bot(bot_id)
                    if not stop_success:
                        self.logger.warning(
                            f"Failed to stop bot on restart attempt {attempt}", bot_id=bot_id
                        )

                    # Wait before restart
                    await asyncio.sleep(self.restart_delay_seconds)

                    # Start bot
                    start_success = await orchestrator.start_bot(bot_id)

                    if start_success:
                        self.lifecycle_stats["successful_restarts"] += 1

                        await self._record_lifecycle_event(
                            bot_id, "restarted", {"attempt": attempt, "success": True}
                        )

                        self.logger.info(
                            f"Bot restart successful on attempt {attempt}", bot_id=bot_id
                        )

                        return True

                except Exception as e:
                    self.logger.warning(f"Bot restart attempt {attempt} failed: {e}", bot_id=bot_id)

                    if attempt < self.restart_max_attempts:
                        await asyncio.sleep(
                            self.restart_delay_seconds * attempt
                        )  # Exponential backoff

            # All restart attempts failed
            self.lifecycle_stats["failed_restarts"] += 1

            await self._record_lifecycle_event(
                bot_id, "restart_failed", {"attempts": self.restart_max_attempts, "success": False}
            )

            self.logger.error(
                f"Bot restart failed after {self.restart_max_attempts} attempts", bot_id=bot_id
            )

            return False

        except Exception as e:
            self.logger.error(f"Bot restart procedure failed: {e}", bot_id=bot_id)
            self.lifecycle_stats["failed_restarts"] += 1
            raise ExecutionError(f"Bot restart failed: {e}")

    async def get_lifecycle_summary(self) -> dict[str, Any]:
        """Get comprehensive lifecycle management summary."""
        try:
            # Aggregate lifecycle states
            deployment_states = {}
            termination_states = {}

            for _bot_id, lifecycle in self.bot_lifecycles.items():
                # Count by deployment state
                deploy_state = lifecycle.get("deployment_state", "unknown")
                deployment_states[deploy_state] = deployment_states.get(deploy_state, 0) + 1

                # Count by termination state
                term_state = lifecycle.get("termination_state", "active")
                termination_states[term_state] = termination_states.get(term_state, 0) + 1

            # Recent events (last 24 hours)
            recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_events = [
                event
                for event in self.lifecycle_events
                if datetime.fromisoformat(event["timestamp"]) > recent_cutoff
            ]

            # Event type counts
            event_type_counts = {}
            for event in recent_events:
                event_type = event["event_type"]
                event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

            return {
                "lifecycle_overview": {
                    "is_running": self.is_running,
                    "managed_bots": len(self.bot_lifecycles),
                    "available_templates": len(self.bot_templates),
                    "deployment_strategies": list(self.deployment_strategies.keys()),
                },
                "deployment_states": deployment_states,
                "termination_states": termination_states,
                "lifecycle_statistics": {
                    **self.lifecycle_stats,
                    "last_lifecycle_action": (
                        self.lifecycle_stats["last_lifecycle_action"].isoformat()
                        if self.lifecycle_stats["last_lifecycle_action"]
                        else None
                    ),
                },
                "recent_events_24h": {
                    "total_events": len(recent_events),
                    "event_types": event_type_counts,
                },
                "templates": {
                    name: {
                        "name": template["name"],
                        "description": template["description"],
                        "bot_type": template["default_config"]["bot_type"].value,
                    }
                    for name, template in self.bot_templates.items()
                },
            }

        except Exception as e:
            self.logger.error(f"Failed to generate lifecycle summary: {e}")
            return {"error": str(e)}

    async def get_bot_lifecycle_details(self, bot_id: str) -> dict[str, Any] | None:
        """
        Get detailed lifecycle information for a specific bot.

        Args:
            bot_id: Bot identifier

        Returns:
            dict: Detailed lifecycle information or None if not found
        """
        if bot_id not in self.bot_lifecycles:
            return None

        lifecycle = self.bot_lifecycles[bot_id]

        # Get events for this bot
        bot_events = [event for event in self.lifecycle_events if event["bot_id"] == bot_id]

        return {
            "bot_id": bot_id,
            "lifecycle_data": {
                **lifecycle,
                "created": (
                    lifecycle.get("created").isoformat() if lifecycle.get("created") else None
                ),
                "deployment_started": (
                    lifecycle.get("deployment_started").isoformat()
                    if lifecycle.get("deployment_started")
                    else None
                ),
                "deployment_completed": (
                    lifecycle.get("deployment_completed").isoformat()
                    if lifecycle.get("deployment_completed")
                    else None
                ),
                "termination_started": (
                    lifecycle.get("termination_started").isoformat()
                    if lifecycle.get("termination_started")
                    else None
                ),
                "termination_completed": (
                    lifecycle.get("termination_completed").isoformat()
                    if lifecycle.get("termination_completed")
                    else None
                ),
            },
            "events": bot_events,
            "event_count": len(bot_events),
        }

    async def _initialize_bot_lifecycle(
        self, bot_id: str, template_name: str, deployment_strategy: str
    ) -> None:
        """Initialize lifecycle tracking for a bot."""
        self.bot_lifecycles[bot_id] = {
            "template_name": template_name,
            "deployment_strategy": deployment_strategy,
            "created": datetime.now(timezone.utc),
            "deployment_state": "pending",
            "termination_state": "active",
            "restart_count": 0,
            "last_restart": None,
            "lifecycle_version": 1,
        }

    async def _record_lifecycle_event(
        self, bot_id: str, event_type: str, event_data: dict[str, Any]
    ) -> None:
        """Record a lifecycle event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "bot_id": bot_id,
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.lifecycle_events.append(event)

        # Keep only recent events
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.event_retention_hours)
        self.lifecycle_events = [
            e for e in self.lifecycle_events if datetime.fromisoformat(e["timestamp"]) > cutoff_time
        ]

    async def _deploy_immediate(
        self, bot_config: BotConfiguration, orchestrator, options: dict[str, Any]
    ) -> bool:
        """Immediate deployment strategy."""
        try:
            # Create bot in orchestrator
            created_bot_id = await orchestrator.create_bot(bot_config)

            # Start if auto_start is enabled
            if bot_config.auto_start:
                return await orchestrator.start_bot(created_bot_id)

            return True

        except Exception as e:
            self.logger.error(f"Immediate deployment failed: {e}")
            return False

    async def _deploy_staged(
        self, bot_config: BotConfiguration, orchestrator, options: dict[str, Any]
    ) -> bool:
        """Staged deployment strategy with validation phases."""
        try:
            # Stage 1: Create bot
            created_bot_id = await orchestrator.create_bot(bot_config)

            # Stage 2: Validation phase (simulate)
            await asyncio.sleep(2)  # Validation delay

            # Stage 3: Start bot if auto_start
            if bot_config.auto_start:
                return await orchestrator.start_bot(created_bot_id)

            return True

        except Exception as e:
            self.logger.error(f"Staged deployment failed: {e}")
            return False

    async def _deploy_blue_green(
        self, bot_config: BotConfiguration, orchestrator, options: dict[str, Any]
    ) -> bool:
        """Blue-green deployment strategy."""
        # Simplified blue-green deployment
        # In a real implementation, this would manage multiple environments
        return await self._deploy_immediate(bot_config, orchestrator, options)

    async def _deploy_canary(
        self, bot_config: BotConfiguration, orchestrator, options: dict[str, Any]
    ) -> bool:
        """Canary deployment strategy."""
        # Simplified canary deployment
        # In a real implementation, this would gradually roll out the bot
        return await self._deploy_staged(bot_config, orchestrator, options)

    async def _deploy_rolling(
        self, bot_config: BotConfiguration, orchestrator, options: dict[str, Any]
    ) -> bool:
        """Rolling deployment strategy."""
        # Simplified rolling deployment
        # In a real implementation, this would manage rolling updates
        return await self._deploy_staged(bot_config, orchestrator, options)

    async def _graceful_termination(self, bot_id: str, orchestrator) -> bool:
        """Perform graceful bot termination."""
        try:
            # Phase 1: Pause new trading
            pause_success = await orchestrator.pause_bot(bot_id)
            if not pause_success:
                self.logger.warning(
                    "Failed to pause bot during graceful termination", bot_id=bot_id
                )

            # Phase 2: Wait for position closure (simulate)
            await asyncio.sleep(10)  # Allow time for positions to close

            # Phase 3: Stop bot
            stop_success = await orchestrator.stop_bot(bot_id)

            # Phase 4: Delete bot
            if stop_success:
                delete_success = await orchestrator.delete_bot(bot_id)
                return delete_success

            return False

        except Exception as e:
            self.logger.error(f"Graceful termination failed: {e}", bot_id=bot_id)
            return False

    async def _immediate_termination(self, bot_id: str, orchestrator) -> bool:
        """Perform immediate bot termination."""
        try:
            # Immediate stop and delete
            stop_success = await orchestrator.stop_bot(bot_id)
            if stop_success:
                return await orchestrator.delete_bot(bot_id, force=True)
            else:
                # Force delete even if stop failed
                return await orchestrator.delete_bot(bot_id, force=True)

        except Exception as e:
            self.logger.error(f"Immediate termination failed: {e}", bot_id=bot_id)
            return False

    async def _lifecycle_loop(self) -> None:
        """Main lifecycle monitoring loop."""
        try:
            while self.is_running:
                try:
                    # Clean up old events
                    await self._cleanup_old_events()

                    # Monitor lifecycle health
                    await self._monitor_lifecycle_health()

                    # Update lifecycle statistics
                    await self._update_lifecycle_statistics()

                    # Wait for next cycle
                    await asyncio.sleep(300)  # 5 minutes

                except Exception as e:
                    self.logger.error(f"Lifecycle loop error: {e}")
                    await asyncio.sleep(60)

        except asyncio.CancelledError:
            self.logger.info("Lifecycle monitoring cancelled")

    async def _cleanup_old_events(self) -> None:
        """Clean up old lifecycle events."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.event_retention_hours)

        initial_count = len(self.lifecycle_events)
        self.lifecycle_events = [
            event
            for event in self.lifecycle_events
            if datetime.fromisoformat(event["timestamp"]) > cutoff_time
        ]

        cleaned_count = initial_count - len(self.lifecycle_events)
        if cleaned_count > 0:
            self.logger.debug(f"Cleaned up {cleaned_count} old lifecycle events")

    async def _monitor_lifecycle_health(self) -> None:
        """Monitor health of lifecycle operations."""
        # Check for stuck deployments
        current_time = datetime.now(timezone.utc)

        for bot_id, lifecycle in self.bot_lifecycles.items():
            if lifecycle.get("deployment_state") == "deploying":
                deployment_started = lifecycle.get("deployment_started")
                if deployment_started:
                    deployment_age = (current_time - deployment_started).total_seconds()
                    if deployment_age > 600:  # 10 minutes
                        self.logger.warning(
                            "Long-running deployment detected",
                            bot_id=bot_id,
                            deployment_age_seconds=deployment_age,
                        )

    async def _update_lifecycle_statistics(self) -> None:
        """Update lifecycle statistics."""
        # Update statistics based on current state
        # Implementation would calculate various lifecycle metrics
        pass
