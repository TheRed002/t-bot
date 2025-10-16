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

# Import event system for coordination
from src.core.events import BotEvent, BotEventType, get_event_publisher
from src.core.exceptions import ServiceError, StateConsistencyError, ValidationError

# Import state types for StateService integration
from src.core.types import (
    BotConfiguration,
    BotMetrics,
    BotPriority,
    BotState,
    BotStatus,
    StatePriority,
    StateType,
)

# Import common utilities
from src.utils.bot_service_helpers import (
    create_bot_metrics_data,
    create_bot_state_data,
    safe_close_connection,
    safe_import_monitoring,
)

# Caching will be implemented in a future module
from .data_transformer import BotManagementDataTransformer

# Get monitoring components with fallback
_monitoring = safe_import_monitoring()
TradingMetrics = _monitoring["TradingMetrics"]
get_tracer = _monitoring["get_tracer"]
get_metrics_collector = _monitoring["get_metrics_collector"]

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

    def __init__(
        self,
        exchange_service,  # IExchangeService interface
        capital_service,  # CapitalService
        execution_service=None,  # ExecutionServiceInterface
        risk_service=None,  # RiskServiceInterface
        state_service=None,  # StateService (concrete class)
        strategy_service=None,  # StrategyServiceInterface
        metrics_collector=None,  # MetricsCollector
        config_service=None,  # Config service for bot configuration
        analytics_service=None,  # Analytics service for metrics and monitoring
    ):
        """
        Initialize bot service with essential dependencies.

        Args:
            exchange_service: Exchange management service interface (REQUIRED)
            capital_service: Capital management service (REQUIRED)
            execution_service: ExecutionServiceInterface
            risk_service: RiskServiceInterface
            state_service: StateService (concrete class)
            strategy_service: StrategyServiceInterface
            metrics_collector: MetricsCollector instance
            config_service: Configuration service for bot settings
            analytics_service: Analytics service for metrics and monitoring
        """
        super().__init__(name="BotService")

        # Validate critical dependencies
        if not exchange_service:
            raise ServiceError("ExchangeService is required for bot management")
        if not capital_service:
            raise ServiceError("CapitalService is required for bot management")

        # Ensure analytics service is available for metrics operations
        if not analytics_service:
            self._logger.warning("Analytics service not provided - some metrics operations may not work")

        # MetricsCollector is optional - will use None if not available

        # Store essential dependencies
        self._exchange_service = exchange_service
        self._capital_service = capital_service
        self._execution_service = execution_service
        self._risk_service = risk_service
        self._state_service = state_service
        self._strategy_service = strategy_service
        self._metrics_collector = metrics_collector
        self._config_service = config_service
        self._analytics_service = analytics_service

        # Repository layer deprecated - no longer storing repository references

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
            # Use singleton metrics collector to avoid duplicate metric registrations
            metrics_collector = get_metrics_collector()
            self._trading_metrics = metrics_collector.trading_metrics if TradingMetrics else None
        except Exception as e:
            self._logger.warning(f"Failed to initialize trading metrics: {e}")
            self._trading_metrics = None

        # Cache manager will be initialized when caching module is available

        # Configuration limits from config service (resolved later via DI)
        # Note: Config will be loaded during _do_start() when dependencies are resolved
        self._max_concurrent_bots = 50  # Default value, will be overridden
        self._max_capital_allocation = Decimal("1000000")  # Default value, will be overridden
        self._high_capital_threshold = Decimal("100000")  # Default value, will be overridden
        self._heartbeat_timeout_seconds = 300  # Default value, will be overridden
        self._health_check_interval = 60  # Default value, will be overridden
        self._bot_startup_timeout_seconds = 120  # Default value, will be overridden
        self._bot_shutdown_timeout_seconds = 60  # Default value, will be overridden

        # Event system for coordination and monitoring
        self._event_publisher = get_event_publisher()

        # Setup event handlers for analytics and risk monitoring
        self._setup_event_handlers()

        self._logger.info("BotService initialized with dependency injection")

    async def _do_start(self) -> None:
        """Start the bot service."""
        try:
            # Verify critical dependencies are available (config_service is optional)
            required_services = [
                self._state_service,
                self._risk_service,
                self._execution_service,
                self._strategy_service,
                self._capital_service,
            ]
            if not all(required_services):
                missing = [
                    name
                    for name, service in zip(
                        [
                            "state_service",
                            "risk_service",
                            "execution_service",
                            "strategy_service",
                            "capital_service",
                        ],
                        required_services,
                        strict=False,
                    )
                    if not service
                ]
                raise ServiceError(f"Required service dependencies not injected: {missing}")

            # Validate MetricsCollector is available
            if not self._metrics_collector:
                self._logger.debug(
                    "MetricsCollector not available - metrics recording will be limited"
                )

            # Load configuration values now that ConfigService is available
            bot_config = {}
            if self._config_service and hasattr(self._config_service, "get_config"):
                bot_config = self._config_service.get_config().get("bot_management_service", {})
            self._max_concurrent_bots = bot_config.get("max_concurrent_bots", 50)
            self._max_capital_allocation = Decimal(
                str(bot_config.get("max_capital_allocation", 1000000))
            )
            self._high_capital_threshold = Decimal(
                str(bot_config.get("high_capital_threshold", 100000))
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
                        await self._metrics_collector.increment(
                            "bot_operations_total",
                            labels={"operation": operation_name, "status": "success"},
                        )
                    except Exception as metrics_error:
                        self._logger.debug(f"Failed to record success metric: {metrics_error}")

                return result

            except Exception as e:
                # Transform error using consistent data transformation for cross-module communication
                error_data = BotManagementDataTransformer.transform_error_to_event_data(
                    e,
                    context={
                        "operation": operation_name,
                        "component": "BotService",
                        "module": "bot_management"
                    }
                )

                # Record error metric with error handling
                if self._metrics_collector:
                    try:
                        await self._metrics_collector.increment(
                            "bot_operations_total",
                            labels={"operation": operation_name, "status": "error"},
                        )
                        await self._metrics_collector.increment(
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

        # Validate exchange connectivity and configuration
        await self._validate_exchange_configuration(bot_config)

        # Check resource limits
        if len(self._active_bots) >= self._max_concurrent_bots:
            raise ServiceError(f"Maximum bot limit reached: {self._max_concurrent_bots}")

        if bot_config.bot_id in self._active_bots:
            raise ValidationError(f"Bot ID already exists: {bot_config.bot_id}")

        # Enhanced risk validation with analytics support
        if self._risk_service and hasattr(self._risk_service, "validate_bot_configuration"):
            try:
                # Get current portfolio for risk context
                current_portfolio = {}
                if self._analytics_service and hasattr(
                    self._analytics_service, "get_portfolio_summary"
                ):
                    current_portfolio = await self._analytics_service.get_portfolio_summary()

                # Apply boundary validation for cross-module communication
                risk_data = BotManagementDataTransformer.apply_cross_module_validation(
                    {
                        "bot_config": getattr(bot_config, "__dict__", {}),
                        "current_portfolio": current_portfolio,
                        "operation": "validate_bot_configuration"
                    },
                    source_module="bot_management",
                    target_module="risk_management"
                )

                risk_validation = await self._risk_service.validate_bot_configuration(
                    bot_config=bot_config, current_portfolio=current_portfolio
                )
                if not risk_validation.get("approved", True):
                    raise ValidationError(
                        f"Risk validation failed: "
                        f"{risk_validation.get('reason', 'Unknown risk issue')}"
                    )

            except AttributeError:
                self._logger.warning(
                    "Advanced risk validation not available - using basic checks",
                    bot_id=bot_config.bot_id,
                )
                # Basic risk checks
                if bot_config.allocated_capital > self._max_capital_allocation:
                    raise ValidationError(
                        f"Capital exceeds maximum allocation: {self._max_capital_allocation}"
                    ) from None
        else:
            self._logger.warning(
                "Risk validation service not available - using basic validation only",
                bot_id=bot_config.bot_id,
            )

        # Allocate capital through CapitalService
        # Use the first exchange from bot config if available
        exchange = bot_config.exchanges[0] if bot_config.exchanges else "default"
        capital_allocated = await self._capital_service.allocate_capital(
            strategy_id=bot_config.strategy_name,
            exchange=exchange,
            amount=bot_config.allocated_capital,
            authorized_by=bot_config.bot_id,
        )

        if not capital_allocated:
            raise ServiceError(f"Failed to allocate capital for bot {bot_config.bot_id}")

        # Initialize bot state using utility helper
        bot_state_data = create_bot_state_data(
            bot_config.bot_id,
            BotStatus.READY.value,
            bot_config.dict(),
            str(bot_config.allocated_capital),
        )

        # Store state through StateService with proper state type - apply boundary validation
        state_data = BotManagementDataTransformer.apply_cross_module_validation(
            {
                "state_type": StateType.BOT_STATE.value,
                "state_id": bot_config.bot_id,
                "state_data": bot_state_data,
                "operation": "set_state"
            },
            source_module="bot_management",
            target_module="state"
        )

        state_persisted = False
        state_connection = None
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
                    await self._metrics_collector.increment(
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
        finally:
            await safe_close_connection(state_connection, "state connection")

        # Initialize metrics using utility helper
        bot_metrics_data = create_bot_metrics_data(bot_config.bot_id)

        # Store metrics state separately with proper state type
        # Using BOT_STATE with metrics prefix for consistency
        metrics_connection = None
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
        finally:
            await safe_close_connection(metrics_connection, "metrics connection")

        # Create local tracking objects for performance
        bot_state = BotState(
            bot_id=bot_config.bot_id,
            status=BotStatus.READY,
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

        # Record bot creation in analytics
        if self._analytics_service:
            try:
                await self._analytics_service.record_bot_creation(
                    bot_id=bot_config.bot_id,
                    strategy=bot_config.strategy_name,
                    capital=bot_config.allocated_capital,
                    exchanges=bot_config.exchanges,
                    symbols=bot_config.symbols,
                )
            except Exception as e:
                self._logger.warning(f"Failed to record bot creation in analytics: {e}")

        # Publish bot creation event for coordination with standardized data transformation
        try:
            # Apply consistent data transformation for cross-module communication
            raw_event_data = {
                "bot_id": bot_config.bot_id,
                "strategy": bot_config.strategy_name,
                "capital": str(bot_config.allocated_capital),
                "exchanges": bot_config.exchanges,
                "symbols": bot_config.symbols,
                "auto_start": getattr(bot_config, "auto_start", False),
            }

            event_data = BotManagementDataTransformer.transform_for_pub_sub(
                "BOT_CREATED",
                raw_event_data,
                metadata={
                    "target_modules": ["coordination", "monitoring", "analytics"],
                    "processing_priority": "high"
                }
            )

            await self._event_publisher.publish(
                BotEvent(
                    event_type=BotEventType.BOT_CREATED,
                    bot_id=bot_config.bot_id,
                    data=event_data,
                    source="bot_management",
                    priority="high",
                )
            )
        except Exception as e:
            self._logger.warning(f"Failed to publish bot creation event: {e}")

        self._logger.info(
            "Bot created successfully via service layer",
            bot_id=bot_config.bot_id,
            bot_type=getattr(bot_config, "bot_type", "unknown"),
            strategy=getattr(bot_config, "strategy_name", "unknown"),
            allocated_capital=str(bot_config.allocated_capital),
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

        # Enhanced pre-start risk assessment with analytics
        if self._risk_service and hasattr(self._risk_service, "pre_start_risk_assessment"):
            try:
                current_positions = []
                if self._execution_service and hasattr(
                    self._execution_service, "get_bot_positions"
                ):
                    current_positions = await self._execution_service.get_bot_positions(bot_id)

                risk_check = await self._risk_service.pre_start_risk_assessment(
                    bot_id=bot_id, current_positions=current_positions
                )

                # Check risk level with proper enum comparison
                risk_level = risk_check.get("risk_level", "LOW")
                if isinstance(risk_level, str):
                    from src.core.types.risk import RiskLevel

                    try:
                        risk_level_enum = RiskLevel(risk_level)
                        if risk_level_enum.value in ["HIGH", "CRITICAL"]:
                            raise ServiceError(
                                f"Pre-start risk check failed: "
                                f"{risk_check.get('details', 'High risk detected')}"
                            )
                    except ValueError:
                        # Fallback for string comparison
                        if risk_level in ["HIGH", "CRITICAL"]:
                            raise ServiceError(
                                f"Pre-start risk check failed: "
                                f"{risk_check.get('details', 'High risk detected')}"
                            ) from None

            except AttributeError:
                self._logger.debug(
                    "Advanced pre-start risk assessment not available - using basic checks",
                    bot_id=bot_id,
                )
                # Basic risk validation if advanced methods not available
                if bot_config.allocated_capital > self._high_capital_threshold:
                    self._logger.warning(
                        f"High capital allocation detected for bot {bot_id}: "
                        f"{bot_config.allocated_capital}"
                    )
                risk_threshold = self.config.bot_management.risk_threshold
                if bot_config.risk_percentage > risk_threshold:  # Configurable risk threshold
                    self._logger.warning(
                        f"High risk percentage for bot {bot_id}: {bot_config.risk_percentage}"
                    )
        else:
            self._logger.debug(
                "Risk service not available - using basic pre-start validation", bot_id=bot_id
            )
            # Basic validation without risk service
            if bot_config.allocated_capital <= 0:
                raise ValidationError(f"Invalid capital allocation for bot {bot_id}")
            if bot_config.risk_percentage <= 0 or bot_config.risk_percentage > 1:
                raise ValidationError(f"Invalid risk percentage for bot {bot_id}")
            if not bot_config.exchanges:
                raise ValidationError(f"No exchanges configured for bot {bot_id}")
            if not bot_config.symbols:
                raise ValidationError(f"No trading symbols configured for bot {bot_id}")

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

        # Publish bot starting event with standardized data transformation
        try:
            # Apply consistent data transformation for cross-module communication
            raw_event_data = {
                "strategy": getattr(bot_config, "strategy_name", "unknown"),
                "bot_id": bot_id,
                "status": "starting"
            }

            event_data = BotManagementDataTransformer.transform_for_pub_sub(
                "BOT_STARTING",
                raw_event_data,
                metadata={
                    "target_modules": ["monitoring", "execution", "risk_management"],
                    "processing_priority": "high"
                }
            )

            await self._event_publisher.publish(
                BotEvent(
                    event_type=BotEventType.BOT_STARTING,
                    bot_id=bot_id,
                    data=event_data,
                    source="bot_management",
                )
            )
        except Exception as e:
            self._logger.warning(f"Failed to publish bot starting event: {e}")

        # Update state to starting through StateService
        starting_state = {
            "status": BotStatus.INITIALIZING.value,
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

            # Publish bot started event
            try:
                await self._event_publisher.publish(
                    BotEvent(
                        event_type=BotEventType.BOT_STARTED,
                        bot_id=bot_id,
                        data={
                            "strategy": getattr(bot_config, "strategy_name", "default"),
                            "execution_started": True,
                        },
                        source="BotService",
                        priority="high",
                    )
                )
            except Exception as e:
                self._logger.warning(f"Failed to publish bot started event: {e}")

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

        # Publish bot stopping event
        try:
            await self._event_publisher.publish(
                BotEvent(
                    event_type=BotEventType.BOT_STOPPING,
                    bot_id=bot_id,
                    data={"reason": "user_requested"},
                    source="BotService",
                )
            )
        except Exception as e:
            self._logger.warning(f"Failed to publish bot stopping event: {e}")

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

            # Publish bot stopped event
            try:
                await self._event_publisher.publish(
                    BotEvent(
                        event_type=BotEventType.BOT_STOPPED,
                        bot_id=bot_id,
                        data={"clean_shutdown": True},
                        source="BotService",
                    )
                )
            except Exception as e:
                self._logger.warning(f"Failed to publish bot stopped event: {e}")

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
        if not force and current_status in [BotStatus.RUNNING, BotStatus.INITIALIZING]:
            raise ServiceError(
                f"Cannot delete running bot: {bot_id}. Use force=True or stop first."
            )

        # Stop if running
        if current_status in [BotStatus.RUNNING, BotStatus.INITIALIZING, BotStatus.PAUSED]:
            await self._stop_bot_impl(bot_id)

        # Enhanced risk assessment for bot deletion
        if self._risk_service and self._analytics_service:
            try:
                # Get current bot positions and evaluate deletion risk
                current_portfolio = (
                    await self._analytics_service.get_portfolio_summary()
                    if hasattr(self._analytics_service, "get_portfolio_summary")
                    else {}
                )
                bot_positions = current_portfolio.get("bot_positions", {}).get(bot_id, {})

                if bot_positions:
                    deletion_risk = (
                        await self._risk_service.assess_deletion_risk(
                            bot_id=bot_id,
                            current_positions=bot_positions,
                            portfolio_context=current_portfolio,
                        )
                        if hasattr(self._risk_service, "assess_deletion_risk")
                        else {"risk_level": "medium", "can_delete": True}
                    )

                    if not deletion_risk.get("can_delete", True):
                        raise ServiceError(
                            f"Cannot delete bot {bot_id}: "
                            f"{deletion_risk.get('reason', 'High deletion risk')}"
                        )

                    self._logger.info(
                        "Bot deletion risk assessment passed",
                        bot_id=bot_id,
                        risk_level=deletion_risk.get("risk_level"),
                    )
                else:
                    self._logger.debug(
                        f"No active positions found for bot {bot_id}, deletion proceeding"
                    )

            except AttributeError:
                self._logger.warning(
                    "Advanced deletion risk check not available - using basic validation",
                    bot_id=bot_id,
                )
                # Basic validation: ensure bot is stopped (unless forced)
                if not force and current_status in [BotStatus.RUNNING, BotStatus.INITIALIZING]:
                    raise ServiceError(
                        f"Cannot delete running bot {bot_id}. Stop the bot first."
                    ) from None
        else:
            self._logger.debug(
                "Deletion risk assessment not available - ensuring bot is stopped", bot_id=bot_id
            )

        # Archive bot data using service layer pattern
        try:
            # Archive metrics data through monitoring service
            # Handle metrics archival through monitoring service
            if self._analytics_service:
                try:
                    await self._analytics_service.archive_bot_metrics(bot_id)
                except Exception as e:
                    self._logger.warning(f"Failed to archive metrics for bot {bot_id}: {e}")

            # Update bot status through state service
            try:
                await self._state_service.set_state(
                    StateType.BOT_STATE,
                    bot_id,
                    {"status": BotStatus.STOPPED.value, "archived": True},
                    source_component="BotService"
                )
            except StateConsistencyError as e:
                self._logger.error(f"State consistency error updating bot status for {bot_id}: {e}")
                raise ServiceError(f"Failed to update bot state: {e}") from e
            except Exception as e:
                self._logger.warning(f"Failed to update bot status for {bot_id}: {e}")

            self._logger.info(f"Bot {bot_id} archived successfully via service layer")

        except Exception as e:
            # Service operations failed - this is a critical error
            self._logger.error(f"Failed to archive bot {bot_id} via service layer: {e}")
            raise ServiceError(f"Bot archival failed - service layer error: {e}") from e

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

    # Caching decorator will be added when available
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

        # Get latest metrics through analytics service
        try:
            if self._analytics_service:
                latest_metrics = await self._analytics_service.get_bot_metrics(bot_id)
                latest_metrics = [latest_metrics] if latest_metrics else []
            else:
                latest_metrics = []
        except Exception as e:
            self._logger.error(f"Failed to get metrics via analytics service: {e}")
            raise ServiceError(f"Failed to retrieve bot metrics: {e}") from e

        # Enhanced risk assessment with analytics
        risk_status = {"status": "unknown", "reason": "RiskService method not available"}
        if self._risk_service and self._analytics_service:
            try:
                if hasattr(self._risk_service, "assess_bot_risk"):
                    risk_assessment = await self._risk_service.assess_bot_risk(bot_id)
                    risk_status = {
                        "status": risk_assessment.get("status", "unknown"),
                        "risk_score": risk_assessment.get("risk_score", 0.0),
                        "risk_factors": risk_assessment.get("risk_factors", []),
                        "last_assessment": risk_assessment.get(
                            "timestamp", datetime.now(timezone.utc).isoformat()
                        ),
                    }
                elif hasattr(self._analytics_service, "get_bot_risk_status"):
                    risk_status = await self._analytics_service.get_bot_risk_status(bot_id)
            except Exception as e:
                self._logger.debug(f"Failed to get risk status: {e}")

        # Get execution status
        execution_status = await self._execution_service.get_bot_execution_status(bot_id)

        # Get analytics performance data
        analytics_data = None
        if self._analytics_service and hasattr(self._analytics_service, "get_bot_performance"):
            try:
                analytics_data = await self._analytics_service.get_bot_performance(bot_id)
            except Exception as e:
                self._logger.debug(f"Failed to get analytics data: {e}")

        return {
            "bot_id": bot_id,
            "state": current_state,
            "metrics": latest_metrics[0] if latest_metrics else None,
            "risk_status": risk_status,
            "execution_status": execution_status,
            "analytics": analytics_data,
            "service_status": "healthy",
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    # Caching decorator will be added when available
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
                # Continue processing other bots - this is expected behavior

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

        # Store metrics using repository layer
        try:
            bot_metrics = BotMetrics(
                bot_id=bot_id,
                created_at=datetime.now(timezone.utc),
                total_trades=metrics.get("trades_count", 0),
                profitable_trades=metrics.get("profitable_trades", 0),
                losing_trades=metrics.get("losing_trades", 0),
            )
            # Save metrics through analytics service
            if self._analytics_service:
                await self._analytics_service.store_bot_metrics(bot_metrics)
            else:
                self._logger.warning("Analytics service not available - metrics not persisted")

            self._logger.debug(f"Metrics stored successfully for bot {bot_id}")

        except Exception as e:
            self._logger.error(f"Failed to store metrics via service layer: {e}")
            raise ServiceError(f"Metrics storage failed: {e}") from e

        # Update local tracking
        if bot_id in self._bot_metrics:
            self._bot_metrics[bot_id] = BotMetrics(**metrics_record)

        # Publish metrics update event with standardized data transformation
        try:
            # Apply consistent data transformation for analytics and monitoring modules
            event_data = BotManagementDataTransformer.apply_cross_module_validation(
                BotManagementDataTransformer.transform_for_pub_sub(
                    "BOT_METRICS_UPDATE",
                    metrics,
                    metadata={
                        "target_modules": ["analytics", "monitoring", "risk_management"],
                        "processing_priority": "normal",
                        "data_type": "financial_metrics"
                    }
                ),
                target_module="analytics",
                source_module="bot_management"
            )

            await self._event_publisher.publish(
                BotEvent(
                    event_type=BotEventType.BOT_METRICS_UPDATE,
                    bot_id=bot_id,
                    data=event_data,
                    source="bot_management",
                )
            )
        except Exception as e:
            self._logger.warning(f"Failed to publish metrics update event: {e}")

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

        # Enhanced risk evaluation with analytics integration
        if self._analytics_service and self._risk_service:
            try:
                # Calculate comprehensive bot risk metrics
                if hasattr(self._risk_service, "calculate_bot_risk_metrics"):
                    current_positions = []
                    if hasattr(self._execution_service, "get_bot_positions"):
                        current_positions = await self._execution_service.get_bot_positions(bot_id)

                    risk_metrics = await self._risk_service.calculate_bot_risk_metrics(
                        bot_id=bot_id, performance_data=metrics, current_positions=current_positions
                    )

                    # Handle high-risk situations
                    if risk_metrics and risk_metrics.get("requires_action", False):
                        await self._handle_risk_action(bot_id, risk_metrics)

                    # Record risk metrics in analytics
                    if hasattr(self._analytics_service, "record_risk_metrics"):
                        await self._analytics_service.record_risk_metrics(risk_metrics)

                elif hasattr(self._analytics_service, "calculate_bot_risk"):
                    # Alternative: Use analytics service for risk calculation
                    risk_metrics = await self._analytics_service.calculate_bot_risk(bot_id)
                    risk_score_threshold = self.config.bot_management.risk_score_threshold
                    if (
                        risk_metrics
                        and getattr(risk_metrics, "risk_score", 0) > risk_score_threshold
                    ):
                        await self._handle_high_risk_bot(bot_id, risk_metrics)

            except Exception as e:
                self._logger.warning(f"Risk evaluation failed: {e}", bot_id=bot_id)
        else:
            self._logger.debug("Advanced risk evaluation not available", bot_id=bot_id)

        return True

    async def _handle_risk_action(self, bot_id: str, risk_metrics: dict[str, Any]) -> None:
        """Handle required risk actions for a bot."""
        try:
            risk_level = risk_metrics.get("risk_level", "LOW")
            actions = risk_metrics.get("recommended_actions", [])

            self._logger.warning(
                f"Risk action required for bot {bot_id}", risk_level=risk_level, actions=actions
            )

            # Take appropriate actions based on risk level
            if "PAUSE_BOT" in actions:
                # Pause the bot temporarily
                bot_data = self._active_bots.get(bot_id)
                if bot_data and bot_data["state"].status == BotStatus.RUNNING:
                    self._logger.warning(f"Pausing bot {bot_id} due to risk concerns")
                    # In a full implementation, this would pause the bot

            elif "REDUCE_POSITION" in actions:
                # Request position reduction
                if self._execution_service and hasattr(
                    self._execution_service, "reduce_bot_positions"
                ):
                    await self._execution_service.reduce_bot_positions(
                        bot_id, reduction_factor=risk_metrics.get("reduction_factor", 0.5)
                    )

            elif "ALERT_OPERATOR" in actions:
                # Send alert to monitoring system
                if self._metrics_collector:
                    try:
                        await self._metrics_collector.increment(
                            "bot_risk_alerts_total",
                            labels={
                                "bot_id": bot_id,
                                "risk_level": risk_level,
                                "action": "operator_alert",
                            },
                        )
                    except Exception as e:
                        self._logger.debug(f"Failed to record risk alert metric: {e}")

        except Exception as e:
            self._logger.error(f"Failed to handle risk action for bot {bot_id}: {e}")

    async def _handle_high_risk_bot(self, bot_id: str, risk_metrics: Any) -> None:
        """Handle high-risk bot situations."""
        try:
            risk_score = getattr(risk_metrics, "risk_score", 0.0)
            self._logger.warning(f"High risk detected for bot {bot_id}", risk_score=risk_score)

            # Record high risk event
            if self._analytics_service and hasattr(self._analytics_service, "record_risk_alert"):
                await self._analytics_service.record_risk_alert(
                    {
                        "bot_id": bot_id,
                        "risk_score": risk_score,
                        "timestamp": datetime.now(timezone.utc),
                        "alert_type": "high_risk_bot",
                    }
                )

            # Publish risk alert event
            try:
                await self._event_publisher.publish(
                    BotEvent(
                        event_type=BotEventType.BOT_RISK_ALERT,
                        bot_id=bot_id,
                        data={
                            "risk_score": risk_score,
                            "alert_type": "high_risk_bot",
                            "severity": "high",
                        },
                        source="BotService",
                        priority="critical",
                    )
                )
            except Exception as e:
                self._logger.warning(f"Failed to publish risk alert event: {e}")

            # Take immediate protective actions
            bot_data = self._active_bots.get(bot_id)
            if bot_data and bot_data["state"].status == BotStatus.RUNNING:
                # Could implement automatic pause or position reduction here
                self._logger.warning(
                    f"Bot {bot_id} requires immediate attention due to high risk score"
                )

        except Exception as e:
            self._logger.error(f"Failed to handle high-risk bot {bot_id}: {e}")

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for analytics and risk monitoring integration."""
        try:
            # Setup event handlers with available services
            from src.core.events import setup_bot_management_events

            # Configure event publisher with analytics and risk services if available
            event_publisher = setup_bot_management_events(
                analytics_service=getattr(self, "_analytics_service", None),
                risk_service=getattr(self, "_risk_service", None)
            )

            # Store reference to event publisher for potential usage in bot operations
            self._event_publisher = event_publisher

            self._logger.info("Event handlers configured for bot management")

        except Exception as e:
            self._logger.warning(f"Failed to setup event handlers: {e}")
            # Continue without event handlers - graceful degradation
            self._event_publisher = None

    def _calculate_bot_rate_requirements(self, bot_config: BotConfiguration) -> int:
        """
        Calculate estimated API rate requirements for a bot.

        Args:
            bot_config: Bot configuration

        Returns:
            int: Estimated API calls per minute
        """
        try:
            base_rate = 10  # Base rate per minute for heartbeat, status checks

            # Trading frequency factor
            strategy_name = bot_config.strategy_name.lower()
            if "scalping" in strategy_name or "hft" in strategy_name:
                trading_factor = 100  # High frequency trading
            elif "arbitrage" in strategy_name:
                trading_factor = 50  # Moderate frequency for arbitrage
            elif "market_making" in strategy_name:
                trading_factor = 40  # Market making requires frequent updates
            else:
                trading_factor = 20  # Default trading frequency

            # Symbol multiplier
            symbol_multiplier = len(bot_config.symbols)

            # Exchange multiplier
            exchange_multiplier = len(bot_config.exchanges)

            # Calculate total estimated rate
            estimated_rate = base_rate + (trading_factor * symbol_multiplier * exchange_multiplier)

            return min(estimated_rate, 1000)  # Cap at reasonable maximum

        except Exception as e:
            self._logger.debug(f"Failed to calculate bot rate requirements: {e}")
            return 50  # Safe default

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
                if bot_data["state"].status in [BotStatus.READY, BotStatus.STOPPED]:
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
                "healthy": state_health.healthy,
                "details": state_health.to_dict(),
            }
        except Exception as e:
            health_results["checks"]["state_service"] = {"healthy": False, "error": str(e)}
            health_results["healthy"] = False

        # Risk service health - use service's own health check if available
        try:
            if hasattr(self._risk_service, "health_check"):
                risk_health = await self._risk_service.health_check()
                health_results["checks"]["risk_service"] = {
                    "healthy": risk_health.healthy,
                    "details": risk_health.to_dict(),
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
                "healthy": execution_health.healthy,
                "details": execution_health.to_dict(),
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

    async def _validate_exchange_configuration(self, bot_config: BotConfiguration) -> None:
        """
        Validate exchange configuration and connectivity.

        Args:
            bot_config: Bot configuration to validate

        Raises:
            ValidationError: If exchange configuration is invalid
            ServiceError: If exchange connectivity fails
        """
        if not bot_config.exchanges:
            raise ValidationError("At least one exchange is required")

        for exchange_name in bot_config.exchanges:
            try:
                # Validate exchange exists and is available
                exchange_available = await self._exchange_service.is_exchange_available(
                    exchange_name
                )
                if not exchange_available:
                    raise ValidationError(f"Exchange {exchange_name} is not available")

                # Validate exchange supports required symbols
                for symbol in bot_config.symbols:
                    symbol_supported = await self._exchange_service.is_symbol_supported(
                        exchange_name, symbol
                    )
                    if not symbol_supported:
                        raise ValidationError(
                            f"Symbol {symbol} is not supported on exchange {exchange_name}"
                        )

                # Validate exchange connection health
                exchange_health = await self._exchange_service.get_exchange_health(exchange_name)
                if exchange_health.get("status") != "healthy":
                    raise ValidationError(
                        f"Exchange {exchange_name} is not healthy: {exchange_health.get('reason')}"
                    )

                # Enhanced validations for production trading

                # Check trading permissions
                if hasattr(self._exchange_service, "get_trading_permissions"):
                    try:
                        trading_permissions = await self._exchange_service.get_trading_permissions(
                            exchange_name
                        )
                        if not trading_permissions.get("spot_trading", True) and "spot" in getattr(
                            bot_config, "trading_types", ["spot"]
                        ):
                            raise ValidationError(f"Spot trading not permitted on {exchange_name}")
                        if not trading_permissions.get(
                            "margin_trading", False
                        ) and "margin" in getattr(bot_config, "trading_types", []):
                            raise ValidationError(
                                f"Margin trading not permitted on {exchange_name}"
                            )
                    except AttributeError:
                        self._logger.debug(
                            f"Trading permissions check not available for {exchange_name}"
                        )

                # Validate minimum order sizes for each symbol
                for symbol in bot_config.symbols:
                    try:
                        if hasattr(self._exchange_service, "get_min_order_size"):
                            min_order_size = await self._exchange_service.get_min_order_size(
                                exchange_name, symbol
                            )
                            min_order_value = getattr(bot_config, "min_order_value", None)
                            if min_order_value and min_order_value < min_order_size:
                                raise ValidationError(
                                    f"Bot min order value {min_order_value} below "
                                    f"exchange minimum {min_order_size} for {symbol} "
                                    f"on {exchange_name}"
                                )
                    except AttributeError:
                        self._logger.debug(
                            f"Min order size check not available for {exchange_name}"
                        )

                # Check rate limit compatibility
                try:
                    if hasattr(self._exchange_service, "get_rate_limits"):
                        bot_rate_requirements = self._calculate_bot_rate_requirements(bot_config)
                        exchange_limits = await self._exchange_service.get_rate_limits(
                            exchange_name
                        )
                        rate_limit_per_minute = exchange_limits.get("requests_per_minute", 1000)

                        rate_limit_threshold = self.config.bot_management.rate_limit_threshold
                        if bot_rate_requirements > rate_limit_per_minute * rate_limit_threshold:
                            self._logger.warning(
                                f"Bot may exceed 80% of rate limits on {exchange_name}",
                                bot_requirements=bot_rate_requirements,
                                exchange_limit=rate_limit_per_minute,
                                bot_id=bot_config.bot_id,
                            )
                            # Not a hard error, but log warning
                except AttributeError:
                    self._logger.debug(f"Rate limit check not available for {exchange_name}")

                # Validate account balance if required
                try:
                    if (
                        hasattr(self._exchange_service, "get_account_balance")
                        and bot_config.allocated_capital > 0
                    ):
                        account_balance = await self._exchange_service.get_account_balance(
                            exchange_name
                        )
                        available_balance = account_balance.get("available", 0)

                        if Decimal(str(available_balance)) < bot_config.allocated_capital:
                            self._logger.warning(
                                f"Insufficient balance on {exchange_name}",
                                available=available_balance,
                                required=str(bot_config.allocated_capital),
                                bot_id=bot_config.bot_id,
                            )
                            # Not a hard error - capital might be allocated from overall portfolio

                except AttributeError:
                    self._logger.debug(f"Account balance check not available for {exchange_name}")

                self._logger.info(
                    "Exchange validation passed",
                    exchange=exchange_name,
                    symbols=bot_config.symbols,
                    bot_id=bot_config.bot_id,
                )

            except Exception as e:
                self._logger.error(
                    f"Exchange validation failed for {exchange_name}: {e}", bot_id=bot_config.bot_id
                )
                raise ValidationError(f"Exchange validation failed for {exchange_name}: {e}") from e

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
                                # Continue processing other configurations

                except Exception as e:
                    self._logger.warning(f"Failed to process bot state data: {e}")
                    # Continue processing other bot states

            self._logger.info(f"Loaded {loaded_count} existing bot states from StateService")

        except Exception as e:
            self._logger.error(f"Failed to load existing bot states: {e}")
            # Don't raise - service should still start even if loading fails

    # Service-specific health check
    async def _service_health_check(self) -> Any:
        """Service-specific health check."""
        from src.core.base import HealthStatus

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
                        # Handle both HealthCheckResult objects and dict responses
                        is_healthy = (
                            health_result.healthy if hasattr(health_result, "healthy")
                            else health_result.get("status") == "healthy" if isinstance(health_result, dict)
                            else False
                        )
                        if not is_healthy:
                            return HealthStatus.DEGRADED

                except Exception as e:
                    self._logger.debug(f"Service health check failed: {e}")
                    return HealthStatus.DEGRADED

            # Check active bot count
            concurrent_bots_threshold = self.config.bot_management.concurrent_bots_threshold
            if len(self._active_bots) > self._max_concurrent_bots * concurrent_bots_threshold:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error(f"Service health check failed: {e}")
            return HealthStatus.UNHEALTHY
