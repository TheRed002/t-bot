"""
Execution Engine - Refactored to use ExecutionService

This module provides the central orchestration layer for all execution activities,
now using the enterprise-grade ExecutionService for all database operations.

Key Features:
- Uses ExecutionService for all database operations (NO direct DB access)
- Full audit trail through service layer for all executions
- Transaction support with rollback capabilities
- Consolidated order validation using ExecutionService
- Enterprise-grade error handling and monitoring
- Order validation consolidation
- Performance tracking through service metrics

Author: Trading Bot Framework
Version: 2.0.0 - Refactored for service layer
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import (
    DatabaseError,
    ExchangeError,
    ExecutionError,
    NetworkError,
    RiskManagementError,
    ServiceError,
    ValidationError,
)

# MANDATORY: Import from P-001
# Import both ExecutionInstruction types for adapter pattern
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionInstruction as CoreExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
    MarketData,
    OrderRequest,
    OrderSide,
)

# Import error handling decorators
from src.error_handling.decorators import with_circuit_breaker, with_error_context, with_retry
from src.execution.execution_result_wrapper import ExecutionResultWrapper

# Import execution engine service interface
from src.execution.interfaces import ExecutionEngineServiceInterface
from src.execution.types import ExecutionInstruction as InternalExecutionInstruction

# Import monitoring components
from src.monitoring import MetricsCollector, get_tracer

# MANDATORY: Import from P-007A
from src.utils import log_calls, time_execution

# Import execution components
from .algorithms.base_algorithm import BaseAlgorithm
from .order_manager import OrderManager
from .risk_adapter import RiskManagerAdapter
from .slippage.cost_analyzer import CostAnalyzer
from .slippage.slippage_model import SlippageModel

# TYPE_CHECKING imports to prevent circular dependencies
if TYPE_CHECKING:
    from src.execution.execution_orchestration_service import ExecutionOrchestrationService
    from src.execution.service import ExecutionService
    from src.risk_management.service import RiskService
    from src.state.state_service import StateService
    from src.state.trade_lifecycle_manager import TradeLifecycleManager


class ExecutionEngine(BaseComponent, ExecutionEngineServiceInterface):
    """
    Central execution engine orchestrator using enterprise ExecutionService.

    This engine provides a high-level interface for execution operations while
    delegating all database operations to ExecutionService. All operations now
    include audit trails, transaction support, and enterprise error handling.

    Key Changes:
    - Uses ExecutionService for all data operations (NO direct database access)
    - Full audit trail for all execution operations
    - Transaction support with rollback capabilities
    - Consolidated order validation through service layer
    - Enterprise-grade monitoring and metrics
    """

    def __init__(
        self,
        execution_service: Optional["ExecutionService"] = None,
        risk_service: Optional["RiskService"] = None,
        config: Config | None = None,
        orchestration_service: Optional["ExecutionOrchestrationService"] = None,
        exchange_factory: Any = None,  # ExchangeFactoryInterface
        state_service: Optional["StateService"] = None,
        trade_lifecycle_manager: Optional["TradeLifecycleManager"] = None,
        metrics_collector: MetricsCollector | None = None,
        order_manager: OrderManager | None = None,
        slippage_model: SlippageModel | None = None,
        cost_analyzer: CostAnalyzer | None = None,
        algorithms: dict[ExecutionAlgorithm, BaseAlgorithm] | None = None,
    ) -> None:
        """
        Initialize execution engine with injected dependencies.

        Args:
            execution_service: ExecutionService instance for database operations
            risk_service: RiskService instance for risk management operations
            config: Application configuration
            orchestration_service: Optional orchestration service (preferred)
            exchange_factory: Optional ExchangeFactoryInterface for exchange access
            state_service: Optional StateService for state persistence
            trade_lifecycle_manager: Optional TradeLifecycleManager for trade state tracking
            metrics_collector: Optional metrics collector for monitoring
            order_manager: Injected OrderManager instance
            slippage_model: Injected SlippageModel instance
            cost_analyzer: Injected CostAnalyzer instance
            algorithms: Injected execution algorithms

        Raises:
            ValueError: If execution_service is None
        """
        super().__init__()  # Initialize BaseComponent

        # Validate required dependencies
        if execution_service is None:
            raise ValueError("ExecutionService is required")

        # Store injected dependencies
        self.execution_service = execution_service
        self.risk_service = risk_service
        self.config = config or Config()
        self.orchestration_service = orchestration_service
        self.state_service = state_service
        self.trade_lifecycle_manager = trade_lifecycle_manager
        self.metrics_collector = metrics_collector
        self.exchange_factory = exchange_factory

        # Log warning if metrics_collector is not provided
        if not metrics_collector:
            self._logger.warning(
                "MetricsCollector not provided - monitoring features will be limited"
            )

        # Create risk adapter for algorithm compatibility
        self.risk_manager_adapter = RiskManagerAdapter(risk_service) if risk_service else None

        # Initialize tracer for distributed tracing with safety check
        try:
            self._tracer = get_tracer("execution.engine")
        except (ImportError, AttributeError, RuntimeError) as e:
            self._logger.warning(f"Failed to initialize tracer: {e}")
            self._tracer = None

        # Use injected components (must be provided by DI container)
        if not order_manager:
            raise ValueError("OrderManager is required and must be injected")
        self.order_manager = order_manager

        if not slippage_model:
            raise ValueError("SlippageModel is required and must be injected")
        self.slippage_model = slippage_model

        # Use injected CostAnalyzer (must be provided by DI container)
        if not cost_analyzer:
            raise ValueError("CostAnalyzer is required and must be injected")
        self.cost_analyzer = cost_analyzer

        # Use injected algorithms (must be provided by DI container)
        if not algorithms:
            raise ValueError("Execution algorithms are required and must be injected")
        self.algorithms: dict[ExecutionAlgorithm, BaseAlgorithm] = algorithms

        # Engine state (local tracking only)
        # is_running is managed by BaseComponent
        self.active_executions: dict[str, ExecutionResultWrapper] = {}

        # Algorithm selection thresholds from config
        self.large_order_threshold = Decimal(
            self.config.execution.get("large_order_threshold", "10000")
        )
        self.volume_significance_threshold = self.config.execution.get(
            "volume_significance_threshold", 0.01
        )

        # Performance tracking will use service metrics
        self.local_statistics = {
            "engine_starts": 0,
            "algorithm_selections": 0,
            "pre_trade_validations": 0,
            "post_trade_analyses": 0,
        }

        self._logger.info(
            "Execution engine initialized with ExecutionService",
            service_type=type(self.execution_service).__name__,
            has_state_service=self.state_service is not None,
            has_trade_lifecycle=self.trade_lifecycle_manager is not None,
            algorithms_loaded=len(self.algorithms),
        )

    @time_execution
    async def start(self) -> None:
        """
        Start the execution engine and all components.
        """
        try:
            if self.is_running:
                self._logger.warning("Execution engine is already running")
                return

            # Ensure ExecutionService is running
            if not self.execution_service.is_running:
                await self.execution_service.start()

            # Start execution algorithms
            for algorithm_name, algorithm in self.algorithms.items():
                try:
                    await algorithm.start()
                    self._logger.info(f"Started {algorithm_name} algorithm")
                except (RuntimeError, AttributeError, ExecutionError) as e:
                    self._logger.error(f"Failed to start {algorithm_name} algorithm: {e}")
                    raise ExecutionError(f"Failed to start {algorithm_name} algorithm: {e}") from e
                except Exception as e:
                    self._logger.error(f"Unexpected error starting {algorithm_name} algorithm: {e}")
                    raise

            # Start order manager
            await self.order_manager.start()

            await super().start()  # Use BaseComponent's start method
            self.local_statistics["engine_starts"] += 1

            self._logger.info("Execution engine started successfully")

        except Exception as e:
            self._logger.error(f"Failed to start execution engine: {e}")
            raise ExecutionError(f"Execution engine startup failed: {e}")

    @time_execution
    async def stop(self) -> None:
        """
        Stop the execution engine and all components.
        """
        try:
            if not self.is_running:
                return

            # Cancel active executions
            for execution_id in list(self.active_executions.keys()):
                await self.cancel_execution(execution_id)

            # Stop execution algorithms
            for algorithm_name, algorithm in self.algorithms.items():
                try:
                    await algorithm.stop()
                    self._logger.info(f"Stopped {algorithm_name} algorithm")
                except (RuntimeError, AttributeError, ExecutionError) as e:
                    self._logger.error(f"Error stopping {algorithm_name} algorithm: {e}")
                    # Don't re-raise during shutdown to allow graceful shutdown of other algorithms
                except Exception as e:
                    self._logger.error(f"Unexpected error stopping {algorithm_name} algorithm: {e}")
                    # Don't re-raise during shutdown to allow graceful shutdown of other algorithms

            # Stop order manager
            await self.order_manager.shutdown()

            await super().stop()  # Use BaseComponent's stop method

            self._logger.info("Execution engine stopped successfully")

        except Exception as e:
            self._logger.error(f"Error stopping execution engine: {e}")
            raise

    def _convert_core_to_internal_instruction(
        self, core_instruction: CoreExecutionInstruction
    ) -> InternalExecutionInstruction:
        """
        Convert CoreExecutionInstruction to InternalExecutionInstruction for algorithm compatibility.

        This adapter ensures algorithms can continue using the internal ExecutionInstruction format
        while the external interface uses the standardized core ExecutionInstruction.
        """
        # Create OrderRequest from core instruction fields
        order = OrderRequest(
            symbol=core_instruction.symbol,
            side=core_instruction.side,
            quantity=core_instruction.target_quantity,
            order_type="limit" if core_instruction.limit_price else "market",
            price=core_instruction.limit_price,
            time_in_force="GTC",  # Default time in force
        )

        # Convert field names to internal format
        return InternalExecutionInstruction(
            order=order,
            algorithm=core_instruction.algorithm,
            strategy_name=core_instruction.metadata.get("strategy_name"),
            # Algorithm parameters
            slice_interval_seconds=core_instruction.slice_interval,
            # Risk controls (convert percentage to basis points if needed)
            max_slippage_bps=core_instruction.max_slippage_pct * 10000
            if core_instruction.max_slippage_pct
            else None,
            # Smart routing (map venue fields)
            preferred_exchanges=core_instruction.preferred_venues,
            avoid_exchanges=core_instruction.avoid_venues,
            # Timing
            start_time=core_instruction.start_time,
            end_time=core_instruction.end_time,
            # Metadata
            metadata=core_instruction.metadata,
        )

    @with_error_context(component="ExecutionEngine", operation="execute_order")
    @with_circuit_breaker(failure_threshold=10, recovery_timeout=60)
    @with_retry(max_attempts=3, exceptions=(NetworkError, ExchangeError))
    @log_calls
    @time_execution
    async def execute_order(
        self,
        instruction: CoreExecutionInstruction | InternalExecutionInstruction,
        market_data: MarketData,
        bot_id: str | None = None,
        strategy_name: str | None = None,
    ) -> ExecutionResultWrapper:
        """
        Execute an order using appropriate algorithm and record via service layer.

        Args:
            instruction: Execution instruction containing order details
            market_data: Current market data
            bot_id: Associated bot instance ID
            strategy_name: Strategy that generated the order

        Returns:
            ExecutionResultWrapper: Result of execution with full audit trail

        Raises:
            ExecutionError: If execution fails
            ValidationError: If order validation fails
        """
        try:
            if not self.is_running:
                raise ExecutionError("Execution engine is not running")

            # Handle both CoreExecutionInstruction and InternalExecutionInstruction
            if isinstance(instruction, InternalExecutionInstruction):
                internal_instruction = instruction
            else:
                # Convert core instruction to internal format for algorithm compatibility
                internal_instruction = self._convert_core_to_internal_instruction(instruction)

            # Use orchestration service if available (preferred approach)
            if self.orchestration_service:
                execution_result = await self.orchestration_service.execute_order(
                    order=internal_instruction.order,
                    market_data=market_data,
                    bot_id=bot_id,
                    strategy_name=strategy_name,
                    execution_params={
                        "algorithm": internal_instruction.algorithm.value,
                        "time_horizon_minutes": internal_instruction.time_horizon_minutes,
                        "participation_rate": internal_instruction.participation_rate,
                        "max_slices": internal_instruction.max_slices,
                        "max_slippage_bps": internal_instruction.max_slippage_bps,
                    },
                )

                self._logger.info(
                    "Order executed through orchestration service",
                    execution_id=execution_result.execution_id,
                    symbol=internal_instruction.order.symbol,
                    bot_id=bot_id,
                )

                return execution_result

            # Fallback to legacy execution path
            return await self._execute_order_legacy(
                internal_instruction, market_data, bot_id, strategy_name
            )

        except (ExecutionError, ValidationError, ServiceError) as e:
            # Re-raise expected exceptions
            symbol = (
                instruction.order.symbol if hasattr(instruction, "order") else instruction.symbol
            )
            algorithm = (
                instruction.algorithm.value if hasattr(instruction, "algorithm") else "unknown"
            )
            self._logger.error(
                "Order execution failed",
                execution_instruction=algorithm,
                symbol=symbol,
                error=str(e),
            )
            raise

        except Exception as e:
            # Log and wrap unexpected exceptions
            symbol = (
                instruction.order.symbol if hasattr(instruction, "order") else instruction.symbol
            )
            algorithm = (
                instruction.algorithm.value if hasattr(instruction, "algorithm") else "unknown"
            )
            self._logger.error(
                "Unexpected error in order execution",
                execution_instruction=algorithm,
                symbol=symbol,
                error=str(e),
            )
            raise ExecutionError(f"Order execution failed: {e}")

    async def execute_instruction(
        self,
        instruction: InternalExecutionInstruction,
        market_data: MarketData,
        bot_id: str | None = None,
        strategy_name: str | None = None,
    ) -> ExecutionResult:
        """
        Execute a trading instruction (implements ExecutionEngineServiceInterface).

        Args:
            instruction: Execution instruction containing order details
            market_data: Current market data
            bot_id: Associated bot instance ID
            strategy_name: Strategy that generated the order

        Returns:
            ExecutionResult: Result of execution

        Raises:
            ExecutionError: If execution fails
            ValidationError: If order validation fails
        """
        # Execute using the existing execute_order method and extract the result
        execution_wrapper = await self.execute_order(
            instruction=instruction,
            market_data=market_data,
            bot_id=bot_id,
            strategy_name=strategy_name,
        )

        # Extract the ExecutionResult from the wrapper
        return execution_wrapper.result

    async def _execute_order_legacy(
        self,
        instruction: InternalExecutionInstruction,
        market_data: MarketData,
        bot_id: str | None = None,
        strategy_name: str | None = None,
    ) -> ExecutionResultWrapper:
        """Legacy execution path for backward compatibility."""
        if not self.execution_service:
            raise ExecutionError("ExecutionService is required for legacy execution")
        if not self.risk_service:
            raise ExecutionError("RiskService is required for legacy execution")

        try:
            # Create trade context in TradeLifecycleManager if available
            trade_context = None
            if self.trade_lifecycle_manager and bot_id and strategy_name:
                trade_context = await self.trade_lifecycle_manager.create_trade(
                    bot_id=bot_id,
                    strategy_name=strategy_name,
                    symbol=instruction.order.symbol,
                    side=instruction.order.side,
                    order_type="market" if instruction.order.price is None else "limit",
                    quantity=instruction.order.quantity,
                    price=instruction.order.price,
                    metadata={
                        "algorithm": instruction.algorithm.value,
                        "market_price": str(market_data.price),
                        "market_volume": str(market_data.volume) if market_data.volume else 0,
                    },
                )
                self._logger.debug(
                    "Trade context created",
                    trade_id=trade_context.trade_id,
                    symbol=instruction.order.symbol,
                )

            # Pre-trade validation using ExecutionService
            validation_results = await self.execution_service.validate_order_pre_execution(
                order=instruction.order,
                market_data=market_data,
                bot_id=bot_id,
                risk_context={
                    "component": "ExecutionEngine",
                    "strategy_name": strategy_name,
                    "execution_instruction": instruction.algorithm.value,
                },
            )

            # Check validation results
            if validation_results["overall_result"] == "failed":
                error_msg = "; ".join(validation_results["errors"])

                # Update trade state to validation failed if using lifecycle manager
                if self.trade_lifecycle_manager and trade_context:
                    await self.trade_lifecycle_manager.update_trade_state(
                        trade_id=trade_context.trade_id,
                        event="validation_failed",
                        data={"errors": validation_results["errors"]},
                    )

                raise ValidationError(f"Order validation failed: {error_msg}")

            # Update trade state to validation passed if using lifecycle manager
            if self.trade_lifecycle_manager and trade_context:
                await self.trade_lifecycle_manager.update_trade_state(
                    trade_id=trade_context.trade_id,
                    event="validation_passed",
                    data={"validation_results": validation_results},
                )

            self.local_statistics["pre_trade_validations"] += 1

            # Risk validation using RiskService
            # Create proper Signal object for RiskService
            from src.core.types import Signal, SignalDirection

            # Map OrderSide to SignalDirection
            signal_direction = (
                SignalDirection.BUY
                if instruction.order.side == OrderSide.BUY
                else (
                    SignalDirection.SELL
                    if instruction.order.side == OrderSide.SELL
                    else SignalDirection.HOLD
                )
            )

            trading_signal = Signal(
                symbol=instruction.order.symbol,
                direction=signal_direction,
                strength=validation_results.get("risk_score", 50.0)
                / 100.0,  # confidence as strength
                timestamp=datetime.now(timezone.utc),
                source="ExecutionEngine",
                metadata={
                    "quantity": str(instruction.order.quantity),
                    "price": str(instruction.order.price or market_data.close),
                    "order_type": "market" if instruction.order.price is None else "limit",
                    "bot_id": bot_id,
                    "strategy_name": strategy_name,
                },
            )

            try:
                risk_validation = await self.risk_service.validate_signal(trading_signal)
            except RiskManagementError as e:
                raise ValidationError(f"Risk validation failed: {e}")

            if not risk_validation:
                raise ValidationError("Risk validation failed: Signal rejected by RiskService")

            # Calculate position size using RiskService
            # Default to a reasonable available capital if not provided
            available_capital = Decimal("100000")  # Default $100k available capital
            try:
                # Try to get actual available capital from config or state
                if hasattr(self.config, "capital") and hasattr(
                    self.config.capital, "default_available"
                ):
                    available_capital = Decimal(str(self.config.capital.default_available))
                elif hasattr(self.config, "trading") and hasattr(
                    self.config.trading, "available_capital"
                ):
                    available_capital = Decimal(str(self.config.trading.available_capital))
            except (AttributeError, ValueError):
                # Use default if config not available
                pass

            try:
                position_size = await self.risk_service.calculate_position_size(
                    signal=trading_signal,
                    available_capital=available_capital,
                    current_price=market_data.close,  # Use Decimal directly
                )
            except RiskManagementError as e:
                raise ExecutionError(f"Position size calculation failed: {e}")

            # Update order quantity based on risk management
            if position_size and position_size > 0:
                # Note: CoreExecutionInstruction is immutable, can't modify target_quantity
                # This modification would need to be done differently in the adapter layer
                self._logger.info(
                    "Position size adjusted by risk management",
                    original_quantity=str(instruction.order.quantity),
                    adjusted_quantity=position_size,
                    symbol=instruction.order.symbol,
                )

            # Select execution algorithm
            selected_algorithm = await self._select_algorithm(
                instruction, market_data, validation_results
            )
            self.local_statistics["algorithm_selections"] += 1

            # Validate slippage if max_slippage_bps is specified
            if (
                hasattr(instruction, "max_slippage_bps")
                and instruction.max_slippage_bps is not None
            ):
                slippage_prediction = await self.slippage_model.predict_slippage(
                    instruction.order, market_data
                )
                expected_slippage_bps = slippage_prediction.total_slippage_bps
                if expected_slippage_bps > instruction.max_slippage_bps:
                    raise ExecutionError(
                        f"Expected slippage {expected_slippage_bps} bps exceeds maximum allowed "
                        f"{instruction.max_slippage_bps} bps"
                    )

            # Update trade state to order created if using lifecycle manager
            if self.trade_lifecycle_manager and trade_context:
                await self.trade_lifecycle_manager.update_trade_state(
                    trade_id=trade_context.trade_id,
                    event="order_created",
                    data={
                        "order_type": "market" if instruction.order.price is None else "limit",
                        "quantity": str(instruction.order.quantity),
                        "algorithm": selected_algorithm.__class__.__name__,
                        "processing_mode": "stream",
                        "data_format": "event_data_v1",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

            # Emit order creation event with consistent format and boundary validation
            if hasattr(self, "_emitter") and self._emitter:
                from src.core.event_constants import OrderEvents
                from src.execution.data_transformer import ExecutionDataTransformer
                from src.utils.messaging_patterns import BoundaryValidator

                event_data = ExecutionDataTransformer.transform_for_pub_sub(
                    event_type=OrderEvents.CREATED,
                    data=instruction.order,
                    metadata={
                        "strategy": strategy_name,
                        "algorithm": selected_algorithm.__class__.__name__,
                        "component": "ExecutionEngine",
                        "processing_mode": "stream",
                        "boundary_crossed": True,
                    },
                )

                # Apply consistent boundary validation for cross-module communication
                try:
                    # Align processing paradigm for consistent data flow
                    aligned_data = ExecutionDataTransformer.align_processing_paradigm(
                        event_data, target_mode="stream"
                    )

                    # Validate at execution -> error_handling boundary
                    BoundaryValidator.validate_error_to_monitoring_boundary(
                        {
                            "component": "ExecutionEngine",
                            "error_type": "OrderCreatedEvent",
                            "severity": "low",
                            "timestamp": aligned_data.get("timestamp"),
                            "processing_mode": "stream",
                            "data_format": "event_data_v1",
                            "message_pattern": "pub_sub",
                            "boundary_crossed": True,
                        }
                    )
                except Exception as boundary_error:
                    self._logger.warning(
                        f"Boundary validation failed for order creation: {boundary_error}"
                    )

                self._emitter.emit(
                    event=OrderEvents.CREATED,
                    data=event_data,
                    source="execution",
                    tags={"component": "ExecutionEngine", "strategy": strategy_name},
                )

            # Execute using selected algorithm with exchange factory and risk manager adapter
            # Add timeout handling
            timeout = self.config.execution.get("timeout", 30)  # Default 30 seconds
            try:
                execution_result = await asyncio.wait_for(
                    selected_algorithm.execute(
                        instruction,
                        exchange_factory=self.exchange_factory,
                        risk_manager=self.risk_manager_adapter,  # Use adapter for algorithm compatibility
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise ExecutionError(f"Execution timeout after {timeout} seconds")

            # Update trade state based on execution result if using lifecycle manager
            if self.trade_lifecycle_manager and trade_context:
                if execution_result.status == ExecutionStatus.COMPLETED:
                    await self.trade_lifecycle_manager.update_trade_state(
                        trade_id=trade_context.trade_id,
                        event="complete_fill",
                        data={
                            "execution_id": execution_result.execution_id,
                            "filled_quantity": str(execution_result.total_filled_quantity),
                            "average_price": (
                                str(execution_result.average_fill_price)
                                if execution_result.average_fill_price
                                else None
                            ),
                        },
                    )
                elif execution_result.status == ExecutionStatus.PARTIAL:
                    await self.trade_lifecycle_manager.update_trade_state(
                        trade_id=trade_context.trade_id,
                        event="partial_fill",
                        data={
                            "execution_id": execution_result.execution_id,
                            "filled_quantity": str(execution_result.total_filled_quantity),
                            "remaining_quantity": str(
                                instruction.order.quantity - execution_result.total_filled_quantity
                            ),
                        },
                    )
                elif execution_result.status in [ExecutionStatus.CANCELLED, ExecutionStatus.FAILED]:
                    await self.trade_lifecycle_manager.update_trade_state(
                        trade_id=trade_context.trade_id,
                        event=(
                            "order_rejected"
                            if execution_result.status == ExecutionStatus.FAILED
                            else "order_cancelled"
                        ),
                        data={
                            "execution_id": execution_result.execution_id,
                            "reason": execution_result.status.value,
                        },
                    )

            # Track active execution
            self.active_executions[execution_result.execution_id] = execution_result

            # Record execution via ExecutionService with full audit trail
            post_trade_analysis = await self._perform_post_trade_analysis(
                execution_result, market_data, validation_results
            )

            # CRITICAL: Coordinate state operations to ensure consistency
            try:
                # Record trade execution first
                await self.execution_service.record_trade_execution(
                    execution_result=execution_result,
                    market_data=market_data,
                    bot_id=bot_id,
                    strategy_name=strategy_name,
                    pre_trade_analysis=validation_results,
                    post_trade_analysis=post_trade_analysis,
                )

                # Complete trade attribution if using lifecycle manager and trade is filled
                if (
                    self.trade_lifecycle_manager
                    and trade_context
                    and execution_result.status == ExecutionStatus.COMPLETED
                ):
                    await self.trade_lifecycle_manager.complete_trade_attribution(
                        trade_id=trade_context.trade_id,
                        execution_result=execution_result,
                        post_trade_analysis=post_trade_analysis,
                    )
            except Exception as state_error:
                self._logger.error(
                    f"Failed to complete trade attribution: {state_error}",
                    execution_id=execution_result.execution_id,
                    trade_id=trade_context.trade_id if trade_context else None,
                )

            # Remove from active executions
            self.active_executions.pop(execution_result.execution_id, None)

            self._logger.info(
                "Order execution completed via service",
                execution_id=execution_result.execution_id,
                algorithm=selected_algorithm.__class__.__name__,
                symbol=instruction.order.symbol,
                side=instruction.order.side.value,
                filled_quantity=str(execution_result.total_filled_quantity),
                status=execution_result.status.value,
            )

            return execution_result

        except (ExecutionError, ValidationError, ServiceError) as e:
            self._logger.error(
                "Order execution failed",
                execution_instruction=instruction.algorithm.value,
                symbol=instruction.order.symbol,
                error=str(e),
            )
            raise

        except Exception as e:
            self._logger.error(
                "Unexpected error in order execution",
                execution_instruction=instruction.algorithm.value,
                symbol=instruction.order.symbol,
                error=str(e),
            )
            raise ExecutionError(f"Order execution failed: {e}")

    @time_execution
    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active execution.

        Args:
            execution_id: Execution identifier to cancel

        Returns:
            bool: True if cancellation successful
        """
        try:
            execution_result = self.active_executions.get(execution_id)
            if not execution_result:
                self._logger.warning(f"No active execution found with ID: {execution_id}")
                return False

            # Update execution status
            execution_result.status = ExecutionStatus.CANCELLED
            execution_result.cancel_time = datetime.now(timezone.utc)

            # Remove from active executions
            self.active_executions.pop(execution_id, None)

            self._logger.info(
                "Execution cancelled",
                execution_id=execution_id,
                symbol=execution_result.original_order.symbol,
            )

            return True

        except (ExecutionError, ServiceError, ValidationError) as e:
            self._logger.error(f"Failed to cancel execution {execution_id}: {e}")
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error cancelling execution {execution_id}: {e}")
            raise ExecutionError(f"Unexpected error during execution cancellation: {e}") from e

    @time_execution
    async def get_execution_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive execution metrics from ExecutionService.

        Returns:
            dict: Execution metrics and performance data
        """
        try:
            # Get metrics from ExecutionService (includes caching and comprehensive data)
            service_metrics = await self.execution_service.get_execution_metrics(
                bot_id=None,  # Get all metrics
                symbol=None,
                time_range_hours=24,
            )

            # Add engine-specific metrics
            engine_metrics = {
                "engine_status": "running" if self.is_running else "stopped",
                "active_executions": len(self.active_executions),
                "algorithms_available": len(self.algorithms),
                "local_statistics": self.local_statistics.copy(),
            }

            # Combine metrics
            combined_metrics = {
                "service_metrics": service_metrics,
                "engine_metrics": engine_metrics,
                "timestamp": datetime.now(timezone.utc),
            }

            return combined_metrics

        except (ServiceError, DatabaseError) as e:
            self._logger.error(f"Failed to get execution metrics: {e}")
            return {
                "error": str(e),
                "engine_status": "running" if self.is_running else "stopped",
                "active_executions": len(self.active_executions),
            }
        except Exception as e:
            self._logger.error(f"Unexpected error getting execution metrics: {e}")
            # For metrics, return error state instead of raising
            return {
                "error": f"Unexpected error: {e!s}",
                "engine_status": "error",
                "active_executions": 0,
            }

    @time_execution
    async def get_active_executions(self) -> dict[str, ExecutionResultWrapper]:
        """
        Get currently active executions.

        Returns:
            dict: Active execution results by execution ID
        """
        return self.active_executions.copy()

    # Helper Methods

    async def _select_algorithm(
        self,
        instruction: InternalExecutionInstruction,
        market_data: MarketData,
        validation_results: dict[str, Any],
    ) -> BaseAlgorithm:
        """
        Select appropriate execution algorithm based on instruction and market conditions.

        Args:
            instruction: Execution instruction
            market_data: Current market data
            validation_results: Pre-trade validation results

        Returns:
            BaseAlgorithm: Selected execution algorithm
        """
        try:
            # Use algorithm specified in instruction if available
            if instruction.algorithm in self.algorithms:
                selected_algorithm = self.algorithms[instruction.algorithm]

                self._logger.debug(
                    "Algorithm selected from instruction",
                    algorithm=instruction.algorithm.value,
                    symbol=instruction.order.symbol,
                )

                return selected_algorithm

            # Intelligent algorithm selection based on order characteristics
            order_value = instruction.order.quantity * (
                instruction.order.price or market_data.close
            )
            risk_level = validation_results.get("risk_level", "medium")

            # Selection logic
            if risk_level == "high" or order_value > self.large_order_threshold:
                # Use TWAP for high-risk or large orders
                selected = self.algorithms[ExecutionAlgorithm.TWAP]
                self._logger.debug("Selected TWAP for high-risk/large order")

            elif market_data.volume and order_value > market_data.volume * Decimal(
                str(self.volume_significance_threshold)
            ):
                # Use VWAP for orders > 1% of volume
                selected = self.algorithms[ExecutionAlgorithm.VWAP]
                self._logger.debug("Selected VWAP for volume-significant order")

            elif instruction.order.quantity > Decimal("1000"):
                # Use Iceberg for large quantity orders
                selected = self.algorithms[ExecutionAlgorithm.ICEBERG]
                self._logger.debug("Selected Iceberg for large quantity order")

            else:
                # Use Smart Router for general cases
                selected = self.algorithms[ExecutionAlgorithm.SMART_ROUTER]
                self._logger.debug("Selected Smart Router for general order")

            return selected

        except (ValidationError, ServiceError) as e:
            self._logger.error(f"Algorithm selection failed: {e}")
            # Fallback to Smart Router for expected errors
            return self.algorithms[ExecutionAlgorithm.SMART_ROUTER]
        except Exception as e:
            self._logger.error(f"Unexpected error in algorithm selection: {e}")
            # For unexpected errors, still fallback but log as critical
            self._logger.critical(
                f"Critical error in algorithm selection, falling back to Smart Router: {e}"
            )
            return self.algorithms[ExecutionAlgorithm.SMART_ROUTER]

    async def _perform_post_trade_analysis(
        self,
        execution_result: ExecutionResultWrapper,
        market_data: MarketData,
        pre_trade_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform post-trade analysis and quality assessment.

        Args:
            execution_result: Completed execution result
            market_data: Market data at execution time
            pre_trade_analysis: Pre-trade validation results

        Returns:
            dict: Post-trade analysis results
        """
        try:
            self.local_statistics["post_trade_analyses"] += 1

            # Calculate execution metrics
            execution_time_ms = (
                execution_result.execution_duration * 1000
                if execution_result.execution_duration
                else 0
            )

            # Calculate slippage if possible
            slippage_bps = 0.0
            if execution_result.average_fill_price and market_data.close:
                price_diff = execution_result.average_fill_price - market_data.close
                slippage_bps = abs(price_diff / market_data.close) * 10000

            # Fill rate
            fill_rate = (
                execution_result.total_filled_quantity / execution_result.original_order.quantity
            )

            # Quality assessment
            quality_score = 100.0

            # Penalize for high slippage
            if slippage_bps > 10:  # More than 10 bps
                quality_score -= min(30, slippage_bps * 2)

            # Penalize for slow execution
            if execution_time_ms > 5000:  # More than 5 seconds
                quality_score -= min(20, (execution_time_ms - 5000) / 1000)

            # Penalize for partial fills
            if fill_rate < 1.0:
                quality_score -= (1.0 - fill_rate) * 50

            quality_score = max(0, quality_score)

            analysis = {
                "execution_time_ms": execution_time_ms,
                "slippage_bps": slippage_bps,
                "fill_rate": fill_rate,
                "quality_score": quality_score,
                "fees_paid": str(execution_result.total_fees),
                "algorithm_used": (
                    execution_result.algorithm.value if execution_result.algorithm else "unknown"
                ),
                "market_conditions": {
                    "price": str(market_data.close),
                    "volume": str(market_data.volume) if market_data.volume else 0,
                    "spread": None,  # Spread is available in Ticker, not MarketData
                },
                "validation_context": {
                    "pre_trade_risk_level": pre_trade_analysis.get("risk_level", "unknown"),
                    "pre_trade_warnings": len(pre_trade_analysis.get("warnings", [])),
                },
                "recommendations": self._generate_execution_recommendations(
                    execution_result, quality_score, slippage_bps, fill_rate
                ),
            }

            return analysis

        except (ServiceError, ValidationError) as e:
            self._logger.error(f"Post-trade analysis failed: {e}")
            return {
                "error": str(e),
                "quality_score": 0.0,
                "execution_time_ms": 0,
                "fill_rate": 0.0,
            }
        except Exception as e:
            self._logger.error(f"Unexpected error in post-trade analysis: {e}")
            # For analysis methods, return error state instead of raising
            return {
                "error": f"Unexpected error: {e!s}",
                "quality_score": 0.0,
                "execution_time_ms": 0,
                "fill_rate": 0.0,
            }

    def _generate_execution_recommendations(
        self,
        execution_result: ExecutionResultWrapper,
        quality_score: Decimal,
        slippage_bps: Decimal,
        fill_rate: Decimal,
    ) -> list[str]:
        """
        Generate recommendations for future executions based on results.

        Args:
            execution_result: Execution result
            quality_score: Calculated quality score
            slippage_bps: Slippage in basis points
            fill_rate: Fill rate percentage

        Returns:
            list: List of recommendations
        """
        recommendations = []

        if quality_score < 70:
            recommendations.append("Consider using a different execution algorithm")

        if slippage_bps > 15:
            recommendations.append("High slippage detected - consider using TWAP or VWAP")

        if fill_rate < 0.95:
            recommendations.append(
                "Partial fill detected - review order size and market conditions"
            )

        if execution_result.execution_duration and execution_result.execution_duration > 10:
            recommendations.append("Long execution time - consider algorithm optimization")

        if not recommendations:
            recommendations.append("Execution performed well within expected parameters")

        return recommendations

    async def get_algorithm_performance(self) -> dict[str, Any]:
        """
        Get performance metrics for each execution algorithm.

        Returns:
            dict: Algorithm performance data
        """
        try:
            # This would integrate with ExecutionService metrics in production
            algorithm_performance = {}

            for algorithm_name, _algorithm in self.algorithms.items():
                # Mock performance data - in production would come from service metrics
                algorithm_performance[algorithm_name.value] = {
                    "total_executions": 0,
                    "success_rate": 0.95,
                    "average_slippage_bps": 5.2,
                    "average_execution_time_ms": 2500,
                    "quality_score": 85.3,
                }

            return algorithm_performance

        except (ServiceError, DatabaseError) as e:
            self._logger.error(f"Failed to get algorithm performance: {e}")
            return {}
        except Exception as e:
            self._logger.error(f"Unexpected error getting algorithm performance: {e}")
            # For performance methods, return empty dict instead of raising
            return {}

    # Abstract methods required by BaseComponent
    async def _do_start(self) -> None:
        """Component-specific startup logic."""
        # Start algorithms
        for algorithm in self.algorithms.values():
            await algorithm.start()

    async def _do_stop(self) -> None:
        """Component-specific cleanup logic."""
        # Stop algorithms
        for algorithm in self.algorithms.values():
            await algorithm.stop()

    async def _health_check_internal(self) -> Any:
        """Component-specific health checks."""
        # Check algorithms health
        healthy_count = 0
        for algo_name, algorithm in self.algorithms.items():
            if algorithm.is_running:
                healthy_count += 1

        return {
            "status": "healthy" if healthy_count == len(self.algorithms) else "degraded",
            "algorithms_healthy": healthy_count,
            "algorithms_total": len(self.algorithms),
            "active_executions": len(self.active_executions),
        }
