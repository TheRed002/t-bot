"""
Bot Management Service for comprehensive bot lifecycle management.

This service implements the complete bot management business logic using the service
layer pattern, orchestrating database, risk management, state management, and
monitoring services to provide a clean application layer interface.

The BotService is the primary orchestrator for all bot-related operations, using
dependency injection to integrate with:
- StateService: Bot state management and persistence
- RiskService: Risk assessment and monitoring
- ExecutionService: Order execution and trade management
- StrategyService: Trading strategy management and configuration
- DatabaseService: Data persistence (via other services)
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from src.core.base.service import BaseService

# TODO: Caching will be implemented in a future module
# from src.core.caching import (
#     cache_bot_status,
#     cached,
#     get_cache_manager,
# )
from src.core.exceptions import ServiceError, ValidationError
from src.core.types import (
    BotConfiguration,
    BotMetrics,
    BotPriority,
    BotState,
    BotStatus,
)

# Import monitoring components with error handling
try:
    from src.monitoring import TradingMetrics, get_tracer
except ImportError as e:
    # Fallback if monitoring module is not available
    import logging

    logging.getLogger(__name__).warning(f"Failed to import monitoring components: {e}")
    TradingMetrics = None

    def get_tracer(x):
        return None


# Import state types for StateService integration
from src.state import StatePriority, StateType

# Type hints for services (resolved via dependency injection)
if TYPE_CHECKING:
    pass


class BotService(BaseService):
    """
    Comprehensive bot management service.

    This service provides the business logic layer for bot management,
    orchestrating multiple services to handle:
    - Bot lifecycle management (create, start, stop, delete)
    - Resource allocation and monitoring
    - Health checks and monitoring
    - Performance tracking
    - Risk management integration

    Dependencies:
    - StateService: Bot state persistence and synchronization
    - RiskService: Risk assessment and limits enforcement
    - ExecutionService: Order execution and trade management
    - StrategyService: Trading strategy configuration and validation
    - CapitalService: Capital allocation and management
    - DatabaseService: Metrics storage and retrieval (via other services)
    """

    def __init__(self):
        """
        Initialize bot service.

        Note: Dependencies are resolved during startup via dependency injection.
        """
        super().__init__(name="BotService")

        # Declare required service dependencies for DI resolution
        self.add_dependency("ConfigService")
        self.add_dependency("StateService")
        self.add_dependency("RiskService")
        self.add_dependency("ExecutionService")
        self.add_dependency("StrategyService")
        self.add_dependency("CapitalService")
        self.add_dependency("DatabaseService")
        self.add_dependency("MetricsCollector")  # Add monitoring dependency

        # Service instances (resolved during startup via DI)
        self._config_service = None
        self._state_service = None
        self._risk_service = None
        self._execution_service = None
        self._strategy_service = None
        self._capital_service = None
        self._database_service = None
        self._metrics_collector = None  # Monitoring metrics

        # Initialize OpenTelemetry tracer with error handling
        try:
            self._tracer = get_tracer(__name__) if get_tracer else None
        except Exception as e:
            self._logger.warning(f"Failed to initialize tracer: {e}")
            self._tracer = None

        # Bot tracking (local cache - state persisted via StateService)
        self._active_bots: dict[str, dict[str, Any]] = {}
        self._bot_configurations: dict[str, BotConfiguration] = {}
        self._bot_metrics: dict[str, BotMetrics] = {}

        # Initialize trading metrics with error handling
        try:
            self._trading_metrics = TradingMetrics() if TradingMetrics else None
        except Exception as e:
            self._logger.warning(f"Failed to initialize trading metrics: {e}")
            self._trading_metrics = None

        # TODO: Initialize cache manager when caching module is available
        # self.cache_manager = get_cache_manager()

        # Configuration limits from config service (resolved later via DI)
        # Note: Config will be loaded during _do_start() when dependencies are resolved
        self._max_concurrent_bots = 50  # Default value, will be overridden
        self._max_capital_allocation = Decimal("1000000")  # Default value, will be overridden
        self._heartbeat_timeout_seconds = 300  # Default value, will be overridden
        self._health_check_interval = 60  # Default value, will be overridden
        self._bot_startup_timeout_seconds = 120  # Default value, will be overridden
        self._bot_shutdown_timeout_seconds = 60  # Default value, will be overridden

        self._logger.info("BotService initialized with dependency injection")

    async def _do_start(self) -> None:
        """Start the bot service and resolve all dependencies."""
        try:
            # Resolve all service dependencies through DI container
            self._config_service = self.resolve_dependency("ConfigService")
            self._state_service = self.resolve_dependency("StateService")
            self._risk_service = self.resolve_dependency("RiskService")
            self._execution_service = self.resolve_dependency("ExecutionService")
            self._strategy_service = self.resolve_dependency("StrategyService")
            self._capital_service = self.resolve_dependency("CapitalService")
            self._database_service = self.resolve_dependency("DatabaseService")
            self._metrics_collector = self.resolve_dependency("MetricsCollector")

            # Verify all critical dependencies are available
            if not all(
                [
                    self._config_service,
                    self._state_service,
                    self._risk_service,
                    self._execution_service,
                    self._strategy_service,
                    self._capital_service,
                ]
            ):
                raise ServiceError("Failed to resolve all required service dependencies")

            # Validate MetricsCollector is available
            if not self._metrics_collector:
                self._logger.warning(
                    "MetricsCollector not available - metrics recording will be limited"
                )

            # Load configuration values now that ConfigService is available
            bot_config = self._config_service.get_config().get("bot_management_service", {})
            self._max_concurrent_bots = bot_config.get("max_concurrent_bots", 50)
            self._max_capital_allocation = Decimal(
                str(bot_config.get("max_capital_allocation", 1000000))
            )
            self._heartbeat_timeout_seconds = bot_config.get("heartbeat_timeout_seconds", 300)
            self._health_check_interval = bot_config.get("health_check_interval_seconds", 60)
            self._bot_startup_timeout_seconds = bot_config.get("bot_startup_timeout_seconds", 120)
            self._bot_shutdown_timeout_seconds = bot_config.get("bot_shutdown_timeout_seconds", 60)

            # Load existing bot states from StateService
            await self._load_existing_bot_states()

            self._logger.info(
                "BotService started successfully with all dependencies resolved",
                dependencies=[
                    "ConfigService",
                    "StateService",
                    "RiskService",
                    "ExecutionService",
                    "StrategyService",
                    "CapitalService",
                    "DatabaseService",
                ],
                config_values={
                    "max_concurrent_bots": self._max_concurrent_bots,
                    "max_capital_allocation": str(self._max_capital_allocation),
                    "heartbeat_timeout_seconds": self._heartbeat_timeout_seconds,
                    "health_check_interval_seconds": self._health_check_interval,
                },
            )

        except Exception as e:
            self._logger.error(f"Failed to start BotService: {e}")
            raise ServiceError(f"BotService startup failed: {e}") from e

    async def execute_with_monitoring(
        self, operation_name: str, operation_func: Any, *args, **kwargs
    ) -> Any:
        """
        Override parent method to add OpenTelemetry tracing and Prometheus metrics.

        Args:
            operation_name: Name of the operation for tracking
            operation_func: Function to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Operation result
        """
        # Create OpenTelemetry span
        with self._tracer.start_as_current_span(f"bot_service.{operation_name}") as span:
            span.set_attribute("bot.operation", operation_name)
            span.set_attribute("bot.service", self._name)

            try:
                # Execute with parent monitoring
                result = await super().execute_with_monitoring(
                    operation_name, operation_func, *args, **kwargs
                )

                # Record success metric with error handling
                if self._metrics_collector:
                    try:
                        self._metrics_collector.increment(
                            "bot_operations_total",
                            labels={"operation": operation_name, "status": "success"},
                        )
                    except Exception as metrics_error:
                        self._logger.debug(f"Failed to record success metric: {metrics_error}")

                return result

            except Exception as e:
                # Record error metric with error handling
                if self._metrics_collector:
                    try:
                        self._metrics_collector.increment(
                            "bot_operations_total",
                            labels={"operation": operation_name, "status": "error"},
                        )
                        self._metrics_collector.increment(
                            "bot_operations_errors_total",
                            labels={"operation": operation_name, "error_type": type(e).__name__},
                        )
                    except Exception as metrics_error:
                        self._logger.debug(f"Failed to record error metric: {metrics_error}")

                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    async def _do_stop(self) -> None:
        """Stop the bot service and clean up."""
        # Stop all active bots
        await self._stop_all_active_bots()

        # Clear tracking
        self._active_bots.clear()
        self._bot_configurations.clear()
        self._bot_metrics.clear()

        self._logger.info("BotService stopped and cleaned up")

    # Core Bot Management Operations

    async def create_bot(self, bot_config: BotConfiguration) -> str:
        """
        Create a new bot instance.

        Args:
            bot_config: Bot configuration

        Returns:
            str: Bot ID

        Raises:
            ServiceError: If bot creation fails
            ValidationError: If configuration is invalid
        """
        return await self.execute_with_monitoring("create_bot", self._create_bot_impl, bot_config)

    async def _create_bot_impl(self, bot_config: BotConfiguration) -> str:
        """Implementation of bot creation using service layer."""
        # Validate configuration structure
        await self._validate_bot_configuration(bot_config)

        # Check resource limits
        if len(self._active_bots) >= self._max_concurrent_bots:
            raise ServiceError(f"Maximum bot limit reached: {self._max_concurrent_bots}")

        if bot_config.bot_id in self._active_bots:
            raise ValidationError(f"Bot ID already exists: {bot_config.bot_id}")

        # Skip risk service validation for now - methods don't exist
        # TODO: Implement risk validation when RiskService provides appropriate methods
        self._logger.warning(
            "Risk validation skipped - RiskService methods not available", bot_id=bot_config.bot_id
        )

        # Allocate capital through CapitalService
        capital_allocated = await self._capital_service.allocate_capital(
            bot_config.bot_id,
            bot_config.allocated_capital,
            priority=bot_config.priority.value if hasattr(bot_config, "priority") else "medium",
        )

        if not capital_allocated:
            raise ServiceError(f"Failed to allocate capital for bot {bot_config.bot_id}")

        # Initialize bot state with proper state type
        bot_state_data = {
            "bot_id": bot_config.bot_id,
            "status": BotStatus.CREATED.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "configuration": bot_config.dict(),
            "allocated_capital": float(bot_config.allocated_capital),
            "current_positions": {},
            "active_orders": {},
            "last_heartbeat": None,
            "error_count": 0,
            "restart_count": 0,
        }

        # Store state through StateService with proper state type
        state_persisted = False
        try:
            await self._state_service.set_state(
                StateType.BOT_STATE,
                bot_config.bot_id,
                bot_state_data,
                source_component="BotService",
                priority=StatePriority.HIGH,
                reason="Bot creation",
            )
            state_persisted = True
        except Exception as e:
            self._logger.error(f"Failed to persist bot state: {e}", bot_id=bot_config.bot_id)
            # Continue with bot creation - state can be synced later
            # Record metric for state persistence failure with error handling
            if self._metrics_collector:
                try:
                    self._metrics_collector.increment(
                        "bot_state_persistence_errors_total",
                        labels={"operation": "create_bot", "bot_id": bot_config.bot_id},
                    )
                except Exception as metrics_error:
                    self._logger.debug(
                        f"Failed to record persistence error metric: {metrics_error}"
                    )

            # Try to store minimal state in memory cache as fallback
            if not state_persisted:
                self._active_bots[bot_config.bot_id] = {
                    "state_persisted": False,
                    "state_data": bot_state_data,
                    "retry_count": 0,
                }

        # Initialize metrics through DatabaseService via StateService
        bot_metrics_data = {
            "bot_id": bot_config.bot_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_trades": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "uptime_percentage": 0.0,
            "error_count": 0,
            "last_heartbeat": None,
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "api_calls_count": 0,
        }

        # Store metrics state separately with proper state type
        # Using BOT_STATE with metrics prefix for consistency
        try:
            await self._state_service.set_state(
                StateType.BOT_STATE,  # Keep metrics with bot state
                f"metrics_{bot_config.bot_id}",  # Clear naming convention
                bot_metrics_data,
                source_component="BotService",
                priority=StatePriority.MEDIUM,
                reason="Bot metrics initialization",
            )
        except Exception as e:
            self._logger.warning(f"Failed to persist bot metrics: {e}", bot_id=bot_config.bot_id)
            # Non-critical - metrics can be recreated

        # Create local tracking objects for performance
        bot_state = BotState(
            bot_id=bot_config.bot_id,
            status=BotStatus.CREATED,
            created_at=datetime.now(timezone.utc),
            configuration=bot_config,
        )

        bot_metrics = BotMetrics(bot_id=bot_config.bot_id, created_at=datetime.now(timezone.utc))

        # Track in local service cache
        self._active_bots[bot_config.bot_id] = {
            "config": bot_config,
            "state": bot_state,
            "metrics": bot_metrics,
            "created_at": datetime.now(timezone.utc),
        }

        self._bot_configurations[bot_config.bot_id] = bot_config
        self._bot_metrics[bot_config.bot_id] = bot_metrics

        self._logger.info(
            "Bot created successfully via service layer",
            bot_id=bot_config.bot_id,
            bot_type=getattr(bot_config, "bot_type", "unknown"),
            strategy=getattr(bot_config, "strategy_name", "unknown"),
            allocated_capital=float(bot_config.allocated_capital),
        )

        # Auto-start if configured
        if getattr(bot_config, "auto_start", False):
            await self.start_bot(bot_config.bot_id)

        return bot_config.bot_id

    async def start_bot(self, bot_id: str) -> bool:
        """
        Start a bot instance.

        Args:
            bot_id: Bot identifier

        Returns:
            bool: True if started successfully

        Raises:
            ServiceError: If bot cannot be started
        """
        return await self.execute_with_monitoring("start_bot", self._start_bot_impl, bot_id)

    async def _start_bot_impl(self, bot_id: str) -> bool:
        """Implementation of bot startup using service layer."""
        if bot_id not in self._active_bots:
            raise ServiceError(f"Bot not found: {bot_id}")

        bot_data = self._active_bots[bot_id]
        bot_config = bot_data["config"]

        # Check current state from StateService
        try:
            current_state_data = await self._state_service.get_state(StateType.BOT_STATE, bot_id)
            if current_state_data and current_state_data.get("status") == BotStatus.RUNNING.value:
                self._logger.warning("Bot is already running", bot_id=bot_id)
                return True
        except Exception as e:
            self._logger.warning(
                f"Failed to get bot state, proceeding with startup: {e}", bot_id=bot_id
            )
            current_state_data = None

        # Skip pre-start risk check - method doesn't exist
        # TODO: Implement when RiskService provides pre_start_risk_check method
        self._logger.warning(
            "Pre-start risk check skipped - RiskService method not available", bot_id=bot_id
        )

        # Validate strategy configuration through StrategyService
        strategy_params = getattr(bot_config, "strategy_parameters", {})
        strategy_validation = await self._strategy_service.validate_strategy(
            getattr(bot_config, "strategy_name", "default"), strategy_params
        )
        if not strategy_validation.get("valid", False):
            raise ServiceError(
                f"Strategy validation failed for bot {bot_id}: "
                f"{strategy_validation.get('error', 'Unknown strategy issue')}"
            )

        # Update state to starting through StateService
        starting_state = {
            "status": BotStatus.STARTING.value,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "last_state_change": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Merge with existing state to avoid data loss
            merged_state = (
                {**current_state_data, **starting_state} if current_state_data else starting_state
            )
            await self._state_service.set_state(
                StateType.BOT_STATE,
                bot_id,
                merged_state,
                source_component="BotService",
                priority=StatePriority.HIGH,
                reason="Bot startup initiated",
            )
        except Exception as e:
            self._logger.error(f"Failed to update bot state to starting: {e}", bot_id=bot_id)
            # Continue with startup - state will be eventually consistent
            # Mark for retry
            if bot_id in self._active_bots:
                self._active_bots[bot_id]["pending_state_update"] = merged_state

        try:
            # Initialize strategy through StrategyService
            strategy_init = await self._strategy_service.initialize_strategy(
                bot_id, getattr(bot_config, "strategy_name", "default"), strategy_params
            )
            if not strategy_init.get("success", False):
                raise ServiceError(
                    f"Strategy initialization failed for bot {bot_id}: "
                    f"{strategy_init.get('error', 'Unknown initialization error')}"
                )

            # Start execution engine through ExecutionService
            execution_started = await self._execution_service.start_bot_execution(
                bot_id, bot_config.dict()
            )
            if not execution_started:
                raise ServiceError(f"Failed to start execution engine for bot: {bot_id}")

            # Update state to running
            running_state = {
                "status": BotStatus.RUNNING.value,
                "running_since": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            }

            current_state_data = await self._state_service.get_state(StateType.BOT_STATE, bot_id)
            await self._state_service.set_state(
                StateType.BOT_STATE,
                bot_id,
                {**current_state_data, **running_state} if current_state_data else running_state,
                source_component="BotService",
                priority=StatePriority.HIGH,
                reason="Bot startup completed successfully",
            )

            # Update local tracking for performance
            bot_data["state"].status = BotStatus.RUNNING
            bot_data["started_at"] = datetime.now(timezone.utc)

            self._logger.info(
                "Bot started successfully via service layer",
                bot_id=bot_id,
                strategy=getattr(bot_config, "strategy_name", "default"),
            )
            return True

        except Exception as e:
            # Update state to error if startup fails
            error_state = {
                "status": BotStatus.ERROR.value,
                "error_message": str(e),
                "error_timestamp": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            current_state_data = await self._state_service.get_state(StateType.BOT_STATE, bot_id)
            await self._state_service.set_state(
                StateType.BOT_STATE,
                bot_id,
                {**current_state_data, **error_state} if current_state_data else error_state,
                source_component="BotService",
                priority=StatePriority.CRITICAL,
                reason=f"Bot startup failed: {e!s}",
            )

            raise ServiceError(f"Bot startup failed for {bot_id}: {e}") from e

    async def stop_bot(self, bot_id: str) -> bool:
        """
        Stop a bot instance.

        Args:
            bot_id: Bot identifier

        Returns:
            bool: True if stopped successfully
        """
        return await self.execute_with_monitoring("stop_bot", self._stop_bot_impl, bot_id)

    async def _stop_bot_impl(self, bot_id: str) -> bool:
        """Implementation of bot shutdown using service layer."""
        if bot_id not in self._active_bots:
            raise ServiceError(f"Bot not found: {bot_id}")

        bot_data = self._active_bots[bot_id]

        # Get current state
        current_state_data = await self._state_service.get_state(StateType.BOT_STATE, bot_id)
        if not current_state_data:
            self._logger.warning("Bot state not found in StateService", bot_id=bot_id)
            current_state_data = {}

        # Update state to stopping
        stopping_state = {
            "status": BotStatus.STOPPING.value,
            "stopping_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        await self._state_service.set_state(
            StateType.BOT_STATE,
            bot_id,
            {**current_state_data, **stopping_state},
            source_component="BotService",
            priority=StatePriority.HIGH,
            reason="Bot shutdown initiated",
        )

        try:
            # Stop execution engine through ExecutionService
            execution_stopped = await self._execution_service.stop_bot_execution(bot_id)
            if not execution_stopped:
                self._logger.warning("Execution engine did not stop cleanly", bot_id=bot_id)

            # Cleanup strategy through StrategyService
            await self._strategy_service.cleanup_strategy(bot_id)

            # Update state to stopped
            stopped_state = {
                "status": BotStatus.STOPPED.value,
                "stopped_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "last_heartbeat": None,
            }

            current_state_data = await self._state_service.get_state(StateType.BOT_STATE, bot_id)
            await self._state_service.set_state(
                StateType.BOT_STATE,
                bot_id,
                {**current_state_data, **stopped_state} if current_state_data else stopped_state,
                source_component="BotService",
                priority=StatePriority.HIGH,
                reason="Bot shutdown completed successfully",
            )

            # Update local tracking for performance
            bot_data["state"].status = BotStatus.STOPPED
            bot_data["stopped_at"] = datetime.now(timezone.utc)

            self._logger.info("Bot stopped successfully via service layer", bot_id=bot_id)
            return True

        except Exception as e:
            # Update state to error if shutdown fails
            error_state = {
                "status": BotStatus.ERROR.value,
                "error_message": f"Shutdown error: {e!s}",
                "error_timestamp": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            current_state_data = await self._state_service.get_state(StateType.BOT_STATE, bot_id)
            await self._state_service.set_state(
                StateType.BOT_STATE,
                bot_id,
                {**current_state_data, **error_state} if current_state_data else error_state,
                source_component="BotService",
                priority=StatePriority.CRITICAL,
                reason=f"Bot shutdown failed: {e!s}",
            )

            raise ServiceError(f"Bot shutdown failed for {bot_id}: {e}") from e

    async def delete_bot(self, bot_id: str, force: bool = False) -> bool:
        """
        Delete a bot instance.

        Args:
            bot_id: Bot identifier
            force: Force deletion even if running

        Returns:
            bool: True if deleted successfully
        """
        return await self.execute_with_monitoring(
            "delete_bot", self._delete_bot_impl, bot_id, force
        )

    async def _delete_bot_impl(self, bot_id: str, force: bool = False) -> bool:
        """Implementation of bot deletion."""
        if bot_id not in self._active_bots:
            raise ServiceError(f"Bot not found: {bot_id}")

        bot_data = self._active_bots[bot_id]
        current_status = bot_data["state"].status

        # Check if bot can be deleted
        if not force and current_status in [BotStatus.RUNNING, BotStatus.STARTING]:
            raise ServiceError(
                f"Cannot delete running bot: {bot_id}. Use force=True or stop first."
            )

        # Stop if running
        if current_status in [BotStatus.RUNNING, BotStatus.STARTING, BotStatus.PAUSED]:
            await self._stop_bot_impl(bot_id)

        # Skip risk check for deletion - method doesn't exist
        # TODO: Implement when RiskService provides approve_bot_deletion method
        self._logger.warning(
            "Bot deletion risk check skipped - RiskService method not available", bot_id=bot_id
        )

        # Archive bot data in database
        await self._database_service.archive_bot_record(bot_id)

        # Remove from state service
        await self._state_service.delete_state(
            StateType.BOT_STATE,
            bot_id,
            source_component="BotService",
            reason="Bot deletion requested",
        )

        # Cleanup from strategy service
        await self._strategy_service.cleanup_strategy(bot_id)

        # Remove from local tracking
        del self._active_bots[bot_id]
        del self._bot_configurations[bot_id]
        if bot_id in self._bot_metrics:
            del self._bot_metrics[bot_id]

        self._logger.info("Bot deleted successfully", bot_id=bot_id)
        return True

    # Bot Monitoring and Health

    # TODO: Add caching decorator when available
    # @cache_bot_status(bot_id_arg_name="bot_id", ttl=30)  # Cache for 30 seconds
    async def get_bot_status(self, bot_id: str) -> dict[str, Any]:
        """
        Get comprehensive bot status.

        Args:
            bot_id: Bot identifier

        Returns:
            dict: Bot status and metrics
        """
        return await self.execute_with_monitoring(
            "get_bot_status", self._get_bot_status_impl, bot_id
        )

    async def _get_bot_status_impl(self, bot_id: str) -> dict[str, Any]:
        """Implementation of get bot status."""
        if bot_id not in self._active_bots:
            raise ServiceError(f"Bot not found: {bot_id}")

        # Get current state from state service
        current_state = await self._state_service.get_state(StateType.BOT_STATE, bot_id)

        # Get latest metrics
        latest_metrics = await self._database_service.get_bot_metrics(bot_id, limit=1)

        # Skip risk assessment - method doesn't exist
        # TODO: Implement when RiskService provides get_bot_risk_status method
        risk_status = {"status": "unknown", "reason": "RiskService method not available"}

        # Get execution status
        execution_status = await self._execution_service.get_bot_execution_status(bot_id)

        return {
            "bot_id": bot_id,
            "state": current_state,
            "metrics": latest_metrics[0] if latest_metrics else None,
            "risk_status": risk_status,
            "execution_status": execution_status,
            "service_status": "healthy",
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    # TODO: Add caching decorator when available
    # @cached(
    #     ttl=30, namespace="bot", data_type="default", key_generator=lambda self: "all_bots_status"
    # )
    async def get_all_bots_status(self) -> dict[str, Any]:
        """
        Get status of all active bots.

        Returns:
            dict: Summary of all bot statuses
        """
        return await self.execute_with_monitoring(
            "get_all_bots_status", self._get_all_bots_status_impl
        )

    async def _get_all_bots_status_impl(self) -> dict[str, Any]:
        """Implementation of get all bots status."""
        bot_statuses = {}
        summary = {
            "total_bots": len(self._active_bots),
            "running": 0,
            "stopped": 0,
            "error": 0,
            "paused": 0,
        }

        for bot_id in self._active_bots.keys():
            try:
                bot_status = await self._get_bot_status_impl(bot_id)
                bot_statuses[bot_id] = bot_status

                # Update summary
                state_status = bot_status["state"].get("status", "unknown")
                if state_status == BotStatus.RUNNING.value:
                    summary["running"] += 1
                elif state_status == BotStatus.STOPPED.value:
                    summary["stopped"] += 1
                elif state_status == BotStatus.ERROR.value:
                    summary["error"] += 1
                elif state_status == BotStatus.PAUSED.value:
                    summary["paused"] += 1

            except Exception as e:
                self._logger.warning(f"Failed to get status for bot {bot_id}: {e}")
                bot_statuses[bot_id] = {"error": str(e)}
                summary["error"] += 1

        return {
            "summary": summary,
            "bots": bot_statuses,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def update_bot_metrics(self, bot_id: str, metrics: dict[str, Any]) -> bool:
        """
        Update metrics for a bot.

        Args:
            bot_id: Bot identifier
            metrics: Metrics data

        Returns:
            bool: True if updated successfully
        """
        return await self.execute_with_monitoring(
            "update_bot_metrics", self._update_bot_metrics_impl, bot_id, metrics
        )

    async def _update_bot_metrics_impl(self, bot_id: str, metrics: dict[str, Any]) -> bool:
        """Implementation of update bot metrics."""
        if bot_id not in self._active_bots:
            raise ServiceError(f"Bot not found: {bot_id}")

        # Store metrics in database
        metrics_record = {"bot_id": bot_id, "timestamp": datetime.now(timezone.utc), **metrics}

        await self._database_service.store_bot_metrics(metrics_record)

        # Update local tracking
        if bot_id in self._bot_metrics:
            self._bot_metrics[bot_id] = BotMetrics(**metrics_record)

        # Update trading metrics in monitoring system with error handling
        if self._metrics_collector and self._trading_metrics:
            try:
                # Record trading performance metrics
                if "pnl" in metrics:
                    self._trading_metrics.record_pnl(
                        strategy=self._active_bots[bot_id].get("strategy", "unknown"),
                        pnl=metrics["pnl"],
                        bot_id=bot_id,
                    )

                if "trades_count" in metrics:
                    self._trading_metrics.increment_trades(
                        count=metrics["trades_count"], bot_id=bot_id
                    )
            except Exception as e:
                self._logger.debug(f"Failed to update trading metrics: {e}")

            if "win_rate" in metrics:
                try:
                    self._metrics_collector.gauge(
                        "bot_win_rate", metrics["win_rate"], labels={"bot_id": bot_id}
                    )
                except Exception as e:
                    self._logger.debug(f"Failed to record win rate metric: {e}")

        # Skip risk check on metrics - method doesn't exist
        # TODO: Implement when RiskService provides evaluate_bot_metrics method
        self._logger.debug(
            "Bot metrics risk evaluation skipped - RiskService method not available", bot_id=bot_id
        )

        return True

    # Batch Operations

    async def start_all_bots(self, priority_filter: BotPriority | None = None) -> dict[str, bool]:
        """
        Start all bots or filtered by priority.

        Args:
            priority_filter: Optional priority filter

        Returns:
            dict: Bot ID to success status mapping
        """
        return await self.execute_with_monitoring(
            "start_all_bots", self._start_all_bots_impl, priority_filter
        )

    async def _start_all_bots_impl(
        self, priority_filter: BotPriority | None = None
    ) -> dict[str, bool]:
        """Implementation of start all bots."""
        results = {}
        bots_to_start = []

        # Filter bots
        for bot_id, bot_data in self._active_bots.items():
            bot_config = bot_data["config"]
            if priority_filter is None or bot_config.priority == priority_filter:
                if bot_data["state"].status in [BotStatus.CREATED, BotStatus.STOPPED]:
                    bots_to_start.append(bot_id)

        # Start bots in controlled batches
        batch_size = 5  # Prevent system overload
        for i in range(0, len(bots_to_start), batch_size):
            batch = bots_to_start[i : i + batch_size]

            # Start batch concurrently
            tasks = [self._start_bot_impl(bot_id) for bot_id in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Record results
            for bot_id, result in zip(batch, batch_results, strict=False):
                results[bot_id] = not isinstance(result, Exception)
                if isinstance(result, Exception):
                    self._logger.error(f"Failed to start bot: {result}", bot_id=bot_id)

            # Small delay between batches
            if i + batch_size < len(bots_to_start):
                await asyncio.sleep(2)

        return results

    async def stop_all_bots(self) -> dict[str, bool]:
        """
        Stop all active bots.

        Returns:
            dict: Bot ID to success status mapping
        """
        return await self.execute_with_monitoring("stop_all_bots", self._stop_all_bots_impl)

    async def _stop_all_bots_impl(self) -> dict[str, bool]:
        """Implementation of stop all bots."""
        results = {}
        running_bots = [
            bot_id
            for bot_id, bot_data in self._active_bots.items()
            if bot_data["state"].status in [BotStatus.RUNNING, BotStatus.PAUSED]
        ]

        # Stop all bots concurrently
        tasks = [self._stop_bot_impl(bot_id) for bot_id in running_bots]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for bot_id, result in zip(running_bots, batch_results, strict=False):
            results[bot_id] = not isinstance(result, Exception)
            if isinstance(result, Exception):
                self._logger.error(f"Failed to stop bot: {result}", bot_id=bot_id)

        return results

    # Health and Monitoring

    async def perform_health_check(self, bot_id: str) -> dict[str, Any]:
        """
        Perform comprehensive health check on a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            dict: Health check results
        """
        return await self.execute_with_monitoring(
            "perform_health_check", self._perform_health_check_impl, bot_id
        )

    async def _perform_health_check_impl(self, bot_id: str) -> dict[str, Any]:
        """Implementation of health check."""
        if bot_id not in self._active_bots:
            return {"bot_id": bot_id, "status": "not_found", "healthy": False, "checks": {}}

        health_results = {
            "bot_id": bot_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "healthy": True,
            "checks": {},
        }

        # State service health
        try:
            state_health = await self._state_service.health_check()
            health_results["checks"]["state_service"] = {
                "healthy": state_health.get("status") == "healthy",
                "details": state_health,
            }
        except Exception as e:
            health_results["checks"]["state_service"] = {"healthy": False, "error": str(e)}
            health_results["healthy"] = False

        # Risk service health - use service's own health check if available
        try:
            if hasattr(self._risk_service, "health_check"):
                risk_health = await self._risk_service.health_check()
                health_results["checks"]["risk_service"] = {
                    "healthy": risk_health.get("status") == "healthy",
                    "details": risk_health,
                }
            else:
                # Basic health check - just verify service exists
                health_results["checks"]["risk_service"] = {
                    "healthy": self._risk_service is not None,
                    "details": {
                        "status": (
                            "service available" if self._risk_service else "service unavailable"
                        )
                    },
                }
        except Exception as e:
            health_results["checks"]["risk_service"] = {"healthy": False, "error": str(e)}
            health_results["healthy"] = False

        # Execution service health
        try:
            execution_health = await self._execution_service.health_check()
            health_results["checks"]["execution_service"] = {
                "healthy": execution_health.get("status") == "healthy",
                "details": execution_health,
            }
        except Exception as e:
            health_results["checks"]["execution_service"] = {"healthy": False, "error": str(e)}
            health_results["healthy"] = False

        # Bot-specific checks
        bot_data = self._active_bots[bot_id]

        # Check if bot has been updated recently
        last_update = bot_data.get("last_metrics_update")
        if last_update:
            time_since_update = (datetime.now(timezone.utc) - last_update).total_seconds()
            if time_since_update > 300:  # 5 minutes
                health_results["checks"]["metrics_freshness"] = {
                    "healthy": False,
                    "details": f"No metrics update for {time_since_update} seconds",
                }
                health_results["healthy"] = False
            else:
                health_results["checks"]["metrics_freshness"] = {
                    "healthy": True,
                    "details": f"Last update {time_since_update} seconds ago",
                }

        return health_results

    # Utility and Validation Methods

    async def _validate_bot_configuration(self, bot_config: BotConfiguration) -> None:
        """Validate bot configuration."""
        # Basic validation
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

        if bot_config.allocated_capital > self._max_capital_allocation:
            raise ValidationError(
                f"Capital allocation exceeds limit: {self._max_capital_allocation}"
            )

        if bot_config.risk_percentage <= 0 or bot_config.risk_percentage > 1:
            raise ValidationError("Risk percentage must be between 0 and 1")

        # Validate strategy exists
        strategy_valid = await self._strategy_service.validate_strategy(
            bot_config.strategy_name, bot_config.strategy_parameters
        )
        if not strategy_valid["valid"]:
            raise ValidationError(f"Invalid strategy: {strategy_valid['error']}")

    async def _stop_all_active_bots(self) -> None:
        """Stop all active bots during service shutdown."""
        if not self._active_bots:
            return

        self._logger.info("Stopping all active bots for service shutdown")

        # Use the regular stop all implementation
        try:
            await self._stop_all_bots_impl()
        except Exception as e:
            self._logger.error(f"Error stopping bots during shutdown: {e}")

    # Helper Methods for Service Layer Integration

    async def _load_existing_bot_states(self) -> None:
        """Load existing bot states from StateService on startup."""
        try:
            # Query StateService for all bot states
            # Check if method exists to maintain compatibility
            if hasattr(self._state_service, "get_states_by_type"):
                bot_states = await self._state_service.get_states_by_type(
                    StateType.BOT_STATE,
                    limit=1000,  # Reasonable limit for bot states
                    include_metadata=True,
                )
            else:
                # Fallback: try alternative method or return empty list
                self._logger.warning(
                    "StateService.get_states_by_type not available, using fallback"
                )
                bot_states = []

            loaded_count = 0
            for state_data in bot_states:
                try:
                    if isinstance(state_data, dict) and "data" in state_data:
                        bot_data = state_data["data"]
                        bot_id = bot_data.get("bot_id")

                        if not bot_id:
                            continue

                        # Skip metrics entries (they have bot_metrics_ prefix now)
                        if bot_id.startswith("bot_metrics_"):
                            continue

                        # Reconstruct bot configuration from state
                        config_data = bot_data.get("configuration", {})
                        if config_data:
                            try:
                                bot_config = BotConfiguration(**config_data)

                                # Create local bot state object
                                bot_state = BotState(
                                    bot_id=bot_id,
                                    status=BotStatus(
                                        bot_data.get("status", BotStatus.STOPPED.value)
                                    ),
                                    created_at=datetime.fromisoformat(
                                        bot_data.get(
                                            "created_at", datetime.now(timezone.utc).isoformat()
                                        )
                                    ),
                                    configuration=bot_config,
                                )

                                # Create placeholder metrics
                                bot_metrics = BotMetrics(
                                    bot_id=bot_id, created_at=datetime.now(timezone.utc)
                                )

                                # Add to local tracking
                                self._active_bots[bot_id] = {
                                    "config": bot_config,
                                    "state": bot_state,
                                    "metrics": bot_metrics,
                                    "created_at": datetime.fromisoformat(
                                        bot_data.get(
                                            "created_at", datetime.now(timezone.utc).isoformat()
                                        )
                                    ),
                                }

                                self._bot_configurations[bot_id] = bot_config
                                self._bot_metrics[bot_id] = bot_metrics

                                loaded_count += 1

                            except Exception as e:
                                self._logger.warning(
                                    f"Failed to reconstruct bot configuration for {bot_id}: {e}"
                                )

                except Exception as e:
                    self._logger.warning(f"Failed to process bot state data: {e}")

            self._logger.info(f"Loaded {loaded_count} existing bot states from StateService")

        except Exception as e:
            self._logger.error(f"Failed to load existing bot states: {e}")
            # Don't raise - service should still start even if loading fails

    # Service-specific health check
    async def _service_health_check(self) -> Any:
        """Service-specific health check."""
        from src.core.base.interfaces import HealthStatus

        try:
            # Check if all required services are available
            required_services = [
                "DatabaseService",
                "StateService",
                "RiskService",
                "ExecutionService",
                "StrategyService",
            ]

            for service_name in required_services:
                try:
                    service = getattr(self, f"_{service_name.lower()}", None)
                    if not service:
                        return HealthStatus.UNHEALTHY

                    # Try to call health check if available
                    if hasattr(service, "health_check"):
                        health_result = await service.health_check()
                        if health_result.get("status") != "healthy":
                            return HealthStatus.DEGRADED

                except Exception:
                    return HealthStatus.DEGRADED

            # Check active bot count
            if len(self._active_bots) > self._max_concurrent_bots * 0.9:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error(f"Service health check failed: {e}")
            return HealthStatus.UNHEALTHY
