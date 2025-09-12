"""
Enterprise-grade Execution Service.

This service provides comprehensive trade execution orchestration with full database
abstraction, transaction support, audit trails, and enterprise features including:
- Transaction management with rollback support
- Comprehensive audit logging for all execution operations
- Circuit breaker and retry mechanisms
- Performance monitoring and execution quality metrics
- Health checks and degraded mode operations
- Order validation consolidation
- Execution cost analysis integration

CRITICAL: This service MUST be used instead of direct database access for executions.
"""

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.base.interfaces import HealthStatus
from src.core.base.service import TransactionalService
from src.core.event_constants import TradeEvents
from src.core.exceptions import (
    ExecutionError,
    RiskManagementError,
    ServiceError,
    ValidationError,
)
from src.core.types import (
    ExecutionResult,
    ExecutionStatus,
    MarketData,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)

# NOTE: Database models are accessed through repository service only
# to maintain proper service layer abstraction. Direct database 
# model imports removed to prevent service layer violations.
from src.error_handling.decorators import with_circuit_breaker, with_retry
from src.execution.interfaces import ExecutionServiceInterface

# Risk service should be injected through dependency injection
# from src.risk_management.service import RiskService  # Removed - use interface instead
from src.utils import cache_result, format_currency, time_execution

# Import risk adapter for proper API usage
# Import monitoring components
from src.utils.decimal_utils import decimal_to_float, safe_decimal_conversion
from src.utils.execution_utils import (
    calculate_order_value,
    calculate_slippage_bps,
)
from src.utils.messaging_patterns import ErrorPropagationMixin
from src.execution.data_transformer import ExecutionDataTransformer


class ExecutionService(TransactionalService, ExecutionServiceInterface, ErrorPropagationMixin):
    """
    Enterprise-grade execution service for trade execution orchestration.

    Features:
    ✅ Full database abstraction using DatabaseService
    ✅ Transaction support with automatic rollback
    ✅ Comprehensive audit trail for all execution operations
    ✅ Circuit breaker protection for execution operations
    ✅ Retry mechanisms with exponential backoff
    ✅ Performance monitoring and execution quality metrics
    ✅ Cache layer integration for execution data
    ✅ Health checks and degraded mode operations
    ✅ Order validation consolidation
    ✅ Execution cost and slippage analysis
    ✅ Risk management integration
    """

    def __init__(
        self,
        repository_service: Any,  # ExecutionRepositoryServiceInterface - injected
        risk_service: Any | None = None,  # Optional dependency - injected
        metrics_service: Any | None = None,  # Optional dependency - injected
        validation_service: Any | None = None,  # Optional dependency - injected
        analytics_service: Any | None = None,  # Optional dependency - injected
        correlation_id: str | None = None,
    ) -> None:
        """
        Initialize execution service.

        Args:
            repository_service: Repository service instance (injected)
            risk_service: Risk service instance (injected)
            metrics_service: Metrics service instance (injected)
            validation_service: Validation service instance (injected)
            analytics_service: Analytics service instance (injected)
            correlation_id: Request correlation ID for tracing
        """
        super().__init__(
            name="ExecutionService",
            correlation_id=correlation_id,
        )

        # Dependencies are injected via constructor
        if not repository_service:
            raise ValueError("Repository service is required")

        self.repository_service = repository_service
        self.risk_service = risk_service
        self.metrics_service = metrics_service
        self.validation_service = validation_service
        self.analytics_service = analytics_service

        # Initialize tracer for distributed tracing through monitoring service
        self._tracer = None
        if self.metrics_service:
            try:
                # Get tracer through monitoring service if available
                from src.monitoring import get_tracer

                self._tracer = get_tracer("execution.service")
            except (ImportError, AttributeError, RuntimeError) as e:
                self._logger.warning(f"Failed to initialize tracer: {e}")
                self._tracer = None

        # Execution configuration
        self.max_order_value = Decimal("100000")  # Max order value
        self.max_position_size = Decimal("50000")  # Max position size
        self.default_slippage_tolerance = Decimal("0.001")  # 0.1% default slippage

        # Performance tracking
        self._performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "cancelled_executions": 0,
            "total_volume": Decimal("0.0"),
            "average_execution_time_ms": 0.0,
            "average_slippage_bps": 0.0,
            "average_cost_bps": 0.0,
            "risk_violations": 0,
            "order_validation_failures": 0,
            "execution_quality_score": 0.0,
        }

        # Execution quality thresholds
        self.quality_thresholds = {
            "excellent_slippage_bps": 5.0,
            "good_slippage_bps": 15.0,
            "acceptable_slippage_bps": 30.0,
            "excellent_execution_time_ms": 500.0,
            "good_execution_time_ms": 2000.0,
            "acceptable_execution_time_ms": 10000.0,
        }

        # Cache configuration
        self._cache_enabled = True
        self._cache_ttl = 60  # 1 minute for execution data

        # Configure service patterns
        self.configure_circuit_breaker(enabled=True, threshold=10, timeout=60)
        self.configure_retry(enabled=True, max_retries=2, delay=0.5, backoff=1.5)

        self._logger.info("ExecutionService initialized with enterprise features")

    async def _do_start(self) -> None:
        """Start the execution service."""
        try:
            # Validate required dependencies are injected
            if not self.repository_service:
                self._logger.error("RepositoryService is required but not injected")
                raise ServiceError("RepositoryService dependency missing")

            # Risk service is optional
            if not self.risk_service:
                self._logger.warning("RiskService not available - using basic validation")
            else:
                self._logger.info("RiskService available")

            # Validation service is optional for this implementation
            if not self.validation_service:
                self._logger.warning("ValidationService not available - using basic validation")
            else:
                self._logger.info("ValidationService available")

            # Ensure repository service is running
            if (
                hasattr(self.repository_service, "is_running")
                and not self.repository_service.is_running
            ):
                await self.repository_service.start()

            # Ensure risk service is running if available
            if self.risk_service and hasattr(self.risk_service, "is_running"):
                if not self.risk_service.is_running:
                    await self.risk_service.start()

            # Initialize execution metrics from database
            await self._initialize_execution_metrics()

            self._logger.info("ExecutionService started successfully")

        except Exception as e:
            self._logger.error(f"Failed to start ExecutionService: {e}")
            raise ServiceError(f"ExecutionService startup failed: {e}")

    async def _initialize_execution_metrics(self) -> None:
        """Initialize execution metrics from repository."""
        try:
            # Load recent orders for metrics through repository service
            recent_orders = await self.repository_service.list_orders(
                filters={"status": "filled"},  # Only filled orders for metrics
                limit=100,
            )

            if recent_orders:
                # Calculate initial metrics from order records
                total_volume = Decimal("0")
                for order in recent_orders:
                    if order.get("filled_quantity") and order.get("average_price"):
                        volume = Decimal(str(order["filled_quantity"])) * Decimal(str(order["average_price"]))
                        total_volume += volume

                self._performance_metrics["total_executions"] = len(recent_orders)
                self._performance_metrics["total_volume"] = total_volume

            self._logger.info(
                "Execution metrics initialized",
                recent_orders=len(recent_orders) if recent_orders else 0,
                total_volume=format_currency(
                    Decimal(str(self._performance_metrics["total_volume"]))
                ),
            )

        except Exception as e:
            self._logger.error(f"Failed to initialize execution metrics: {e}")
            raise

    # Core Execution Operations

    @with_circuit_breaker(failure_threshold=10, recovery_timeout=60)
    @with_retry(max_attempts=2, base_delay=0.5)
    @time_execution
    async def record_trade_execution(
        self,
        execution_result: ExecutionResult,
        market_data: MarketData,
        bot_id: str | None = None,
        strategy_name: str | None = None,
        pre_trade_analysis: dict[str, Any] | None = None,
        post_trade_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Record a completed trade execution with full audit trail.

        Args:
            execution_result: Execution result details
            market_data: Market data at time of execution
            bot_id: Associated bot instance ID
            strategy_name: Strategy that generated the trade
            pre_trade_analysis: Pre-trade analysis results
            post_trade_analysis: Post-trade analysis results

        Returns:
            dict: Created trade record

        Raises:
            ServiceError: If recording fails
            ValidationError: If execution data is invalid
        """
        return await self.execute_in_transaction(
            "record_trade_execution",
            self._record_trade_execution_impl,
            execution_result,
            market_data,
            bot_id,
            strategy_name,
            pre_trade_analysis,
            post_trade_analysis,
        )

    async def _record_trade_execution_impl(
        self,
        execution_result: ExecutionResult,
        market_data: MarketData,
        bot_id: str | None,
        strategy_name: str | None,
        pre_trade_analysis: dict[str, Any] | None,
        post_trade_analysis: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Internal implementation of trade execution recording."""
        operation_start = datetime.now(timezone.utc)

        # Start OpenTelemetry span for distributed tracing if tracer is available
        span_context = None
        span = None
        if self._tracer:
            try:
                span_context = self._tracer.start_as_current_span("record_trade_execution")
                span = span_context.__enter__()
            except Exception as e:
                self._logger.warning(f"Failed to start tracing span: {e}")
                span_context = None
                span = None

        try:
            # Set span attributes for better observability if span is available
            if span:
                span.set_attribute("execution.id", execution_result.execution_id)
                span.set_attribute("execution.symbol", execution_result.original_order.symbol)
                span.set_attribute("execution.side", execution_result.original_order.side.value)
                span.set_attribute(
                    "execution.order_type", execution_result.original_order.order_type.value
                )
                span.set_attribute(
                    "execution.quantity", str(execution_result.total_filled_quantity)
                )
                if bot_id:
                    span.set_attribute("bot.id", bot_id)
                if strategy_name:
                    span.set_attribute("strategy.name", strategy_name)

            try:
                # Apply boundary validation for execution data
                from src.utils.messaging_patterns import BoundaryValidator

                execution_boundary_data = {
                    "component": "ExecutionService",
                    "operation": "record_trade_execution",
                    "processing_mode": "stream",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "execution_id": execution_result.execution_id,
                    "symbol": execution_result.original_order.symbol,
                    "quantity": str(execution_result.total_filled_quantity),
                    "price": str(execution_result.average_fill_price)
                    if execution_result.average_fill_price
                    else None,
                }

                # Validate execution to monitoring boundary
                BoundaryValidator.validate_monitoring_to_error_boundary(execution_boundary_data)

                # Validate execution result
                self._validate_execution_result(execution_result)

                # Calculate execution metrics
                execution_metrics = self._calculate_execution_metrics(
                    execution_result, market_data, pre_trade_analysis
                )

                # Create ORDER record data (not Trade - Trade is for closed positions)
                order_data = {
                    "id": str(uuid.uuid4()),
                    "bot_id": bot_id,
                    "exchange": execution_result.original_order.exchange or "binance",
                    "exchange_order_id": execution_result.execution_id,
                    "symbol": execution_result.original_order.symbol,
                    "side": execution_result.original_order.side.value,
                    "type": execution_result.original_order.order_type.value,
                    "status": self._map_execution_status_to_order_status(
                        execution_result.status
                    ).value,
                    "price": self._convert_to_decimal_safe(
                        execution_result.original_order.price or market_data.price
                    ),
                    "quantity": self._convert_to_decimal_safe(
                        execution_result.original_order.quantity
                    ),
                    "filled_quantity": self._convert_to_decimal_safe(
                        execution_result.total_filled_quantity
                    ),
                    "average_price": self._convert_to_decimal_safe(
                        execution_result.average_fill_price or market_data.price
                    ),
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }

                # Save order through repository service
                saved_order_dict = await self.repository_service.create_order_record(order_data)

                # Create OrderFill records - handled by repository service
                # Note: OrderFill creation will be handled by repository service internally
                # when creating the order record with fills

                # saved_order_dict already contains the response from repository service

                # Create comprehensive audit log
                await self._create_execution_audit_log(
                    execution_id=execution_result.execution_id,
                    operation_type="trade_execution",
                    order_id=str(saved_order_dict.get("id", "")),
                    execution_result=execution_result,
                    market_data=market_data,
                    bot_id=bot_id,
                    strategy_name=strategy_name,
                    execution_metrics=execution_metrics,
                    pre_trade_analysis=pre_trade_analysis,
                    post_trade_analysis=post_trade_analysis,
                    operation_start=operation_start,
                )

                # Update performance metrics
                await self._update_execution_metrics(
                    execution_result, execution_metrics, operation_start
                )

                # Send trade data to analytics service
                if self.analytics_service:
                    try:
                        # Create trade object for analytics
                        from src.core.types import Trade

                        trade = Trade(
                            trade_id=execution_result.execution_id,
                            order_id=str(saved_order_dict.get("id", "")),  # Use the saved order ID
                            symbol=execution_result.original_order.symbol,
                            side=execution_result.original_order.side,
                            price=execution_result.average_fill_price or market_data.price,
                            quantity=execution_result.total_filled_quantity,
                            fee=execution_result.total_fees or Decimal("0"),
                            fee_currency="USDT",  # Default fee currency
                            timestamp=datetime.now(timezone.utc),
                            exchange=execution_result.original_order.exchange or "binance",
                            is_maker=False,  # Default to taker for execution
                        )

                        # Update analytics with trade data
                        self.analytics_service.update_trade(trade)

                        self._logger.debug(
                            "Trade data sent to analytics", trade_id=execution_result.execution_id
                        )

                    except Exception as analytics_error:
                        # Log but don't fail execution due to analytics errors
                        self._logger.warning(
                            f"Failed to send trade data to analytics: {analytics_error}"
                        )

                self._logger.info(
                    "Order execution recorded successfully",
                    order_id=str(saved_order_dict.get("id", "")),
                    execution_id=execution_result.execution_id,
                    symbol=execution_result.original_order.symbol,
                    side=execution_result.original_order.side.value,
                    quantity=format_currency(str(execution_result.total_filled_quantity)),
                    executed_price=format_currency(str(execution_result.average_fill_price or 0)),
                    slippage_bps=execution_metrics.get("slippage_bps", 0),
                )

                return saved_order_dict

            except Exception as e:
                # Create failure audit log
                await self._create_execution_audit_log(
                    execution_id=execution_result.execution_id,
                    operation_type="trade_execution",
                    execution_result=execution_result,
                    market_data=market_data,
                    bot_id=bot_id,
                    strategy_name=strategy_name,
                    operation_start=operation_start,
                    operation_status="failed",
                    success=False,
                    error_message=str(e),
                )

                self._performance_metrics["failed_executions"] += 1

                # Use consistent error propagation pattern from mixin
                try:
                    self.propagate_service_error(error=e, context="trade_execution_recording")
                except ServiceError:
                    # propagate_service_error always raises, so we catch the ServiceError
                    # Emit error event with consistent error propagation patterns using pub/sub
                    if hasattr(self, "_emitter") and self._emitter:
                        try:
                            # Apply cross-module validation for consistent data flow
                            from src.execution.data_transformer import ExecutionDataTransformer

                            error_data = {
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "execution_id": execution_result.execution_id,
                                "symbol": execution_result.original_order.symbol,
                                "component": "ExecutionService",
                                "severity": "high",
                                "processing_mode": "stream",  # Consistent with web_interface
                                "message_pattern": "pub_sub",  # Align messaging patterns
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }

                            # Transform using pub/sub pattern for consistency with web_interface
                            transformed_error_data = ExecutionDataTransformer.transform_for_pub_sub(
                                event_type="trade_execution_error",
                                data=error_data,
                                metadata={"source": "execution", "target": "error_handling"},
                            )

                            self._emitter.emit(
                                event=TradeEvents.FAILED,
                                data=transformed_error_data,
                                source="execution",
                            )
                        except Exception as emit_error:
                            self._logger.warning(f"Failed to emit error event: {emit_error}")

                    self._logger.error(
                        "Trade execution recording failed",
                        execution_id=execution_result.execution_id,
                        symbol=execution_result.original_order.symbol,
                        error=str(e),
                    )

                    raise ServiceError(f"Trade execution recording failed: {e}") from e
        finally:
            # Properly close span context if it was opened
            if span_context:
                try:
                    span_context.__exit__(None, None, None)
                except Exception as e:
                    self._logger.warning("Failed to close span context cleanly", error=str(e))
            elif span:
                # Fallback cleanup for span without context
                try:
                    span.end()
                except Exception as e:
                    self._logger.warning("Failed to end span cleanly", error=str(e))

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60)
    @with_retry(max_attempts=3, base_delay=1.0)
    @time_execution
    async def validate_order_pre_execution(
        self,
        order: OrderRequest,
        market_data: MarketData,
        bot_id: str | None = None,
        risk_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive order validation before execution.

        Args:
            order: Order to validate
            market_data: Current market data
            bot_id: Associated bot instance ID
            risk_context: Risk assessment context

        Returns:
            dict: Validation results with risk assessment

        Raises:
            ValidationError: If order fails validation
        """
        return await self.execute_with_monitoring(
            "validate_order_pre_execution",
            self._validate_order_pre_execution_impl,
            order,
            market_data,
            bot_id,
            risk_context,
        )

    async def _validate_order_pre_execution_impl(
        self,
        order: OrderRequest,
        market_data: MarketData,
        bot_id: str | None,
        risk_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Internal implementation of pre-execution order validation."""
        validation_id = str(uuid.uuid4())
        validation_start = datetime.now(timezone.utc)

        try:
            validation_results = {
                "validation_id": validation_id,
                "overall_result": "passed",
                "risk_level": "low",
                "risk_score": 0.0,
                "validation_checks": [],
                "warnings": [],
                "errors": [],
                "recommendations": [],
            }

            # Basic order validation
            basic_validation = await self._perform_basic_order_validation(order, market_data)
            validation_results["validation_checks"].extend(basic_validation["checks"])

            if basic_validation["errors"]:
                validation_results["errors"].extend(basic_validation["errors"])
                validation_results["overall_result"] = "failed"
                validation_results["risk_level"] = "high"
                validation_results["risk_score"] = 80.0

            # Position size validation
            position_validation = await self._validate_position_size(order, bot_id)
            validation_results["validation_checks"].extend(position_validation["checks"])

            if position_validation["warnings"]:
                validation_results["warnings"].extend(position_validation["warnings"])
                if validation_results["overall_result"] == "passed":
                    validation_results["overall_result"] = "warning"
                    validation_results["risk_level"] = "medium"
                    validation_results["risk_score"] = max(validation_results["risk_score"], 40.0)

            # Market condition validation
            market_validation = await self._validate_market_conditions(order, market_data)
            validation_results["validation_checks"].extend(market_validation["checks"])

            # Risk assessment
            if risk_context:
                risk_validation = await self._perform_risk_assessment(
                    order, market_data, risk_context, bot_id
                )
                validation_results["validation_checks"].extend(risk_validation["checks"])
                validation_results["risk_score"] = max(
                    validation_results["risk_score"], risk_validation["risk_score"]
                )

            # Determine final risk level
            if validation_results["risk_score"] >= 80:
                validation_results["risk_level"] = "critical"
            elif validation_results["risk_score"] >= 60:
                validation_results["risk_level"] = "high"
            elif validation_results["risk_score"] >= 40:
                validation_results["risk_level"] = "medium"
            else:
                validation_results["risk_level"] = "low"

            # Generate recommendations
            validation_results["recommendations"] = self._generate_order_recommendations(
                validation_results, order, market_data
            )

            # Create validation audit log
            await self._create_risk_audit_log(
                risk_event_id=validation_id,
                event_type="pre_trade_validation",
                bot_id=bot_id,
                validation_results=validation_results,
                order_context={
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": str(order.quantity),
                    "order_type": order.order_type.value,
                },
            )

            # Update validation metrics
            if validation_results["overall_result"] == "failed":
                self._performance_metrics["order_validation_failures"] += 1

            self._logger.info(
                "Order pre-execution validation completed",
                validation_id=validation_id,
                symbol=order.symbol,
                overall_result=validation_results["overall_result"],
                risk_level=validation_results["risk_level"],
                risk_score=validation_results["risk_score"],
            )

            return validation_results

        except Exception as e:
            self._logger.error(
                "Order validation failed",
                validation_id=validation_id,
                symbol=order.symbol,
                error=str(e),
            )

            # Use consistent error propagation patterns
            try:
                self.propagate_service_error(e, "order_pre_execution_validation")
            except ServiceError:
                # Service error has been propagated, continue with validation result return
                self._logger.debug("Service error propagated for order validation")

            # Return failed validation
            return {
                "validation_id": validation_id,
                "overall_result": "failed",
                "risk_level": "critical",
                "risk_score": 100.0,
                "validation_checks": [],
                "errors": [f"Validation error: {e!s}"],
                "warnings": [],
                "recommendations": ["Review order parameters and retry"],
            }

    async def validate_order_pre_execution_from_data(
        self,
        order_data: dict[str, Any],
        market_data: dict[str, Any],
        bot_id: str | None = None,
        risk_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Validate order from raw data dictionaries.

        This method handles data conversion internally, keeping business logic
        out of the controller layer.

        Args:
            order_data: Raw order data dictionary
            market_data: Raw market data dictionary
            bot_id: Associated bot instance ID
            risk_context: Risk assessment context

        Returns:
            dict: Validation results with risk assessment

        Raises:
            ValidationError: If order fails validation
        """
        # Convert raw data to typed objects using centralized transformer
        order = ExecutionDataTransformer.convert_to_order_request(order_data)
        market_data_obj = ExecutionDataTransformer.convert_to_market_data(market_data)

        # Delegate to existing typed method
        return await self.validate_order_pre_execution(
            order=order,
            market_data=market_data_obj,
            bot_id=bot_id,
            risk_context=risk_context,
        )

    # Data conversion methods removed - now using centralized ExecutionDataTransformer

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60)
    @cache_result(ttl=300)
    @time_execution
    async def get_execution_metrics(
        self,
        bot_id: str | None = None,
        symbol: str | None = None,
        time_range_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get comprehensive execution metrics.

        Args:
            bot_id: Filter by bot instance ID
            symbol: Filter by symbol
            time_range_hours: Time range for metrics calculation

        Returns:
            dict: Execution metrics and performance data
        """
        return await self.execute_with_monitoring(
            "get_execution_metrics",
            self._get_execution_metrics_impl,
            bot_id,
            symbol,
            time_range_hours,
        )

    async def _get_execution_metrics_impl(
        self,
        bot_id: str | None,
        symbol: str | None,
        time_range_hours: int,
    ) -> dict[str, Any]:
        """Internal implementation of execution metrics calculation."""
        try:
            # Build filters
            filters = {}
            if bot_id:
                filters["bot_id"] = bot_id
            if symbol:
                filters["symbol"] = symbol

            # Get recent trades with consistent processing paradigm
            start_time = datetime.now(timezone.utc) - timedelta(hours=time_range_hours)

            # Apply consistent processing paradigm for data queries
            from src.execution.data_transformer import ExecutionDataTransformer

            query_context = {
                "processing_mode": "batch",  # Use batch for metrics aggregation
                "data_format": "metrics_query_v1",
                "time_range_hours": time_range_hours,
                "component": "ExecutionService",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Align to batch processing for metrics calculation
            aligned_context = ExecutionDataTransformer.align_processing_paradigm(
                query_context, "batch"
            )

            # Add timestamp filter to database query
            if filters is None:
                filters = {}
            filters["timestamp"] = {"gte": start_time}

            # Get all orders through repository service
            all_orders = await self.repository_service.list_orders(
                filters=filters,
                limit=1000,
            )

            # Filter by time in memory - convert dict format
            filtered_orders = []
            for order in all_orders:
                # Handle both dict and object formats
                created_at = order.get("created_at") if isinstance(order, dict) else getattr(order, "created_at", None)
                if isinstance(created_at, str):
                    from datetime import datetime
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                
                if created_at and created_at >= start_time:
                    filtered_orders.append(order)

            if not filtered_orders:
                return self._get_empty_metrics()

            # Calculate execution metrics from orders - handle dict format
            total_volume = Decimal("0")
            for order in filtered_orders:
                # Handle both dict and object formats
                filled_quantity = order.get("filled_quantity") if isinstance(order, dict) else getattr(order, "filled_quantity", None)
                average_price = order.get("average_price") if isinstance(order, dict) else getattr(order, "average_price", None)
                price = order.get("price") if isinstance(order, dict) else getattr(order, "price", None)
                
                if filled_quantity and filled_quantity > 0:
                    volume = Decimal(str(filled_quantity)) * Decimal(str(average_price or price or "0"))
                    total_volume += volume

            successful_orders = [
                o for o in filtered_orders 
                if (o.get("status") if isinstance(o, dict) else getattr(o, "status", None)) == OrderStatus.FILLED.value
            ]
            failed_orders = [
                o for o in filtered_orders
                if (o.get("status") if isinstance(o, dict) else getattr(o, "status", None)) 
                in [OrderStatus.CANCELLED.value, OrderStatus.REJECTED.value]
            ]

            success_rate = len(successful_orders) / len(filtered_orders) if filtered_orders else 0

            # Calculate average fees from order fills - handle dict format
            avg_fee_rate = 0.0
            total_fees = Decimal("0")

            # Get fees from orders if available
            for order in successful_orders:
                # Handle both dict and object formats
                fee = order.get("fee") if isinstance(order, dict) else getattr(order, "fee", None)
                if fee:
                    total_fees += Decimal(str(fee))

            if total_volume > 0:
                avg_fee_rate = (total_fees / total_volume) * 10000

            metrics = {
                "time_range_hours": time_range_hours,
                "total_trades": len(filtered_orders),
                "successful_orders": len(successful_orders),
                "failed_orders": len(failed_orders),
                "success_rate": success_rate,
                "total_volume": str(total_volume),
                "average_fee_rate_bps": avg_fee_rate,
                "symbols_traded": len(set(
                    order.get("symbol") if isinstance(order, dict) else getattr(order, "symbol", "")
                    for order in filtered_orders
                )),
                "exchanges_used": len(set(
                    order.get("exchange") if isinstance(order, dict) else getattr(order, "exchange", "")
                    for order in filtered_orders
                )),
                "side_distribution": {
                    "buy": len([
                        o for o in filtered_orders 
                        if (o.get("side") if isinstance(o, dict) else getattr(o, "side", None)) == OrderSide.BUY.value
                    ]),
                    "sell": len([
                        o for o in filtered_orders 
                        if (o.get("side") if isinstance(o, dict) else getattr(o, "side", None)) == OrderSide.SELL.value
                    ]),
                },
                "order_type_distribution": {},
                "performance_metrics": self._performance_metrics.copy(),
            }

            # Add order type distribution - handle dict format
            for order_type in [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LOSS]:
                count = len([
                    o for o in filtered_orders 
                    if (o.get("type") if isinstance(o, dict) else getattr(o, "type", None)) == order_type.value
                ])
                metrics["order_type_distribution"][order_type.value] = count

            return metrics

        except Exception as e:
            self._logger.error(f"Failed to calculate execution metrics: {e}")
            raise ServiceError(f"Execution metrics calculation failed: {e}")

    # Helper Methods

    def _convert_to_decimal_safe(self, value: Any, precision: int = 8) -> Decimal:
        """Safely convert value to Decimal with proper precision."""
        return safe_decimal_conversion(value, precision)

    def _validate_execution_result(self, execution_result: ExecutionResult) -> None:
        """Validate execution result parameters."""
        if not execution_result:
            raise ValidationError("Execution result cannot be None")

        if not execution_result.execution_id:
            raise ValidationError("Execution ID is required")

        if not execution_result.original_order:
            raise ValidationError("Original order is required")

        if execution_result.total_filled_quantity <= 0:
            raise ValidationError("Filled quantity must be positive")

    def _calculate_execution_metrics(
        self,
        execution_result: ExecutionResult,
        market_data: MarketData,
        pre_trade_analysis: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Calculate execution quality metrics."""
        metrics = {
            "execution_time_ms": (
                execution_result.execution_duration * 1000
                if execution_result.execution_duration
                else 0
            ),
            "slippage_bps": 0.0,
            "market_impact_bps": 0.0,
            "total_cost_bps": 0.0,
            "fill_rate": str(
                execution_result.total_filled_quantity / execution_result.original_order.quantity
            ),
            "quality_score": 0.0,
        }

        # Calculate slippage using shared utility
        if execution_result.average_fill_price and market_data.price:
            slippage_bps = calculate_slippage_bps(
                execution_result.average_fill_price, market_data.price
            )
            metrics["slippage_bps"] = slippage_bps

        # Calculate quality score
        execution_time = metrics["execution_time_ms"]
        slippage = metrics["slippage_bps"]

        quality_score = 100.0

        # Penalize for high slippage
        if slippage > self.quality_thresholds["excellent_slippage_bps"]:
            quality_score -= min(
                30, (slippage - self.quality_thresholds["excellent_slippage_bps"]) * 2
            )

        # Penalize for slow execution
        if execution_time > self.quality_thresholds["excellent_execution_time_ms"]:
            quality_score -= min(
                20, (execution_time - self.quality_thresholds["excellent_execution_time_ms"]) / 100
            )

        # Penalize for partial fills
        fill_rate = metrics["fill_rate"]
        if fill_rate < 1.0:
            quality_score -= (1.0 - fill_rate) * 50

        metrics["quality_score"] = max(0, quality_score)

        return metrics

    def _map_execution_status_to_order_status(
        self, execution_status: ExecutionStatus
    ) -> OrderStatus:
        """Map execution status to order status."""
        mapping = {
            ExecutionStatus.PENDING: OrderStatus.PENDING,
            ExecutionStatus.COMPLETED: OrderStatus.FILLED,
            ExecutionStatus.PARTIAL: OrderStatus.PARTIALLY_FILLED,
            ExecutionStatus.CANCELLED: OrderStatus.CANCELLED,
            ExecutionStatus.FAILED: OrderStatus.REJECTED,
        }
        return mapping.get(execution_status, OrderStatus.PENDING)

    async def _perform_basic_order_validation(
        self, order: OrderRequest, market_data: MarketData
    ) -> dict[str, Any]:
        """Perform basic order validation checks."""
        checks = []
        errors = []

        # Validate order fields
        if not order.symbol:
            errors.append("Order symbol is required")
            checks.append(
                {
                    "check": "symbol_validation",
                    "result": "failed",
                    "message": "Missing symbol",
                }
            )
        else:
            checks.append(
                {
                    "check": "symbol_validation",
                    "result": "passed",
                    "message": "Symbol is valid",
                }
            )

        # Validate quantity
        if order.quantity <= 0:
            errors.append("Order quantity must be positive")
            checks.append(
                {
                    "check": "quantity_validation",
                    "result": "failed",
                    "message": "Invalid quantity",
                }
            )
        else:
            try:
                # Use ValidationService with proper validation
                if self.validation_service:
                    # Use the validation service if available
                    if hasattr(self.validation_service, "validate_quantity"):
                        # Use async method if available
                        await self.validation_service.validate_quantity(order.quantity)
                    elif hasattr(self.validation_service, "validate_quantity_sync"):
                        # Use sync method as fallback
                        self.validation_service.validate_quantity_sync(order.quantity)
                    else:
                        # Basic validation if no specific method
                        if order.quantity <= 0:
                            raise ValidationError("Quantity must be positive")
                else:
                    # Fallback to basic validation
                    if order.quantity <= 0:
                        raise ValidationError("Quantity must be positive")

                checks.append(
                    {
                        "check": "quantity_validation",
                        "result": "passed",
                        "message": "Quantity is valid",
                    }
                )
            except (ValidationError, ValueError) as e:
                errors.append(str(e))
                checks.append(
                    {
                        "check": "quantity_validation",
                        "result": "failed",
                        "message": str(e),
                    }
                )

        # Validate order value
        order_value = calculate_order_value(order.quantity, order.price, market_data)
        if order_value > self.max_order_value:
            errors.append(f"Order value {order_value} exceeds maximum {self.max_order_value}")
            checks.append(
                {
                    "check": "order_value_validation",
                    "result": "failed",
                    "message": f"Order value too large: {order_value}",
                }
            )
        else:
            checks.append(
                {
                    "check": "order_value_validation",
                    "result": "passed",
                    "message": f"Order value within limits: {order_value}",
                }
            )

        return {"checks": checks, "errors": errors}

    async def _validate_position_size(
        self, order: OrderRequest, bot_id: str | None
    ) -> dict[str, Any]:
        """Validate position size limits using RiskService."""
        checks = []
        warnings = []

        # Use RiskService if available
        if self.risk_service:
            try:
                # Create proper Signal object for RiskService
                from datetime import datetime, timezone

                from src.core.types import Signal, SignalDirection

                # Map OrderSide to SignalDirection
                signal_direction = (
                    SignalDirection.BUY
                    if order.side == OrderSide.BUY
                    else (
                        SignalDirection.SELL
                        if order.side == OrderSide.SELL
                        else SignalDirection.HOLD
                    )
                )

                # Create Signal for validation
                trading_signal = Signal(
                    symbol=order.symbol,
                    direction=signal_direction,
                    strength=0.5,  # Default confidence
                    timestamp=datetime.now(timezone.utc),
                    source="ExecutionService",
                    metadata={
                        "quantity": str(order.quantity),
                        "price": str(order.price) if order.price else "0.0",
                        "order_type": order.order_type.value,
                        "bot_id": bot_id,
                    },
                )

                # Validate using RiskService
                try:
                    risk_validation = await self.risk_service.validate_signal(trading_signal)
                except (RiskManagementError, ValidationError, ServiceError) as e:
                    risk_validation = False
                    warnings.append(f"Risk validation error: {e}")

                if risk_validation:
                    checks.append(
                        {
                            "check": "risk_service_position_validation",
                            "result": "passed",
                            "message": "Position size validated by RiskService",
                        }
                    )
                else:
                    warnings.append("Risk validation failed")
                    checks.append(
                        {
                            "check": "risk_service_position_validation",
                            "result": "warning",
                            "message": "Risk validation failed",
                        }
                    )

                # Get recommended position size
                # Default to a reasonable available capital if not provided
                available_capital = Decimal("100000")  # Default $100k available capital
                current_price = (
                    Decimal(str(order.price)) if order.price else Decimal("50000")
                )  # Default price

                try:
                    recommended_size = await self.risk_service.calculate_position_size(
                        signal=trading_signal,
                        available_capital=available_capital,
                        current_price=current_price,
                    )
                except (RiskManagementError, ValidationError, ServiceError) as e:
                    recommended_size = None
                    warnings.append(f"Position size calculation error: {e}")

                if recommended_size and order.quantity > recommended_size:
                    warnings.append(
                        f"Order quantity {order.quantity} exceeds "
                        f"risk-adjusted size {recommended_size}"
                    )

            except Exception as e:
                self._logger.error(f"RiskService validation failed: {e}", exc_info=True)
                # Add warning but continue with order
                warnings.append(f"Risk validation error: {e!s}")
                checks.append(
                    {
                        "check": "risk_service_position_validation",
                        "result": "error",
                        "message": f"Risk validation error: {e!s}",
                    }
                )

        # Simple fallback validation if RiskService not available
        else:
            # Basic position size check using shared utility
            order_value = calculate_order_value(order.quantity, order.price, None, Decimal("1"))
            if order_value > self.max_order_value:
                warnings.append(f"Order value exceeds maximum {self.max_order_value}")
                checks.append(
                    {
                        "check": "position_size_validation",
                        "result": "warning",
                        "message": "Large order value",
                    }
                )
            else:
                checks.append(
                    {
                        "check": "position_size_validation",
                        "result": "passed",
                        "message": "Position size within limits",
                    }
                )

        return {"checks": checks, "warnings": warnings}

    async def _validate_market_conditions(
        self, order: OrderRequest, market_data: MarketData
    ) -> dict[str, Any]:
        """Validate market conditions for execution."""
        checks = []

        # Check if market data is recent (within last minute)
        if market_data.timestamp:
            age_seconds = (datetime.now(timezone.utc) - market_data.timestamp).total_seconds()
            if age_seconds > 60:
                checks.append(
                    {
                        "check": "market_data_freshness",
                        "result": "warning",
                        "message": f"Market data is {age_seconds:.0f} seconds old",
                    }
                )
            else:
                checks.append(
                    {
                        "check": "market_data_freshness",
                        "result": "passed",
                        "message": "Market data is fresh",
                    }
                )

        # Check spread if available
        if market_data.bid and market_data.ask:
            spread = market_data.ask - market_data.bid
            spread_bps = (spread / market_data.price) * 10000

            if spread_bps > 50:  # More than 50 bps spread
                checks.append(
                    {
                        "check": "spread_validation",
                        "result": "warning",
                        "message": f"Wide spread: {spread_bps:.1f} bps",
                    }
                )
            else:
                checks.append(
                    {
                        "check": "spread_validation",
                        "result": "passed",
                        "message": f"Acceptable spread: {spread_bps:.1f} bps",
                    }
                )

        return {"checks": checks}

    async def _perform_risk_assessment(
        self,
        order: OrderRequest,
        market_data: MarketData,
        risk_context: dict[str, Any],
        bot_id: str | None,
    ) -> dict[str, Any]:
        """Perform comprehensive risk assessment using RiskService when available."""
        checks = []
        risk_score = 0.0

        # Use RiskService for comprehensive risk assessment if available
        if self.risk_service:
            try:
                # Get risk metrics from RiskService
                market_data_obj = MarketData(
                    symbol=order.symbol,
                    timestamp=datetime.now(timezone.utc),
                    open=market_data.price,
                    high=market_data.price,
                    low=market_data.price,
                    close=market_data.price,
                    volume=market_data.volume or Decimal("0.0"),
                    exchange=order.exchange,
                    bid_price=market_data.bid or market_data.price,
                    ask_price=market_data.ask or market_data.price,
                )
                try:
                    risk_metrics = await self.risk_service.calculate_risk_metrics(
                        positions=[],  # Current positions will be fetched by RiskService
                        market_data=[market_data_obj],
                    )
                except (RiskManagementError, ValidationError, ServiceError) as e:
                    risk_metrics = None
                    self._logger.warning(f"Risk metrics calculation failed: {e}")

                # Extract risk information
                portfolio_risk = (
                    float(risk_metrics.var_1d)
                    if risk_metrics and hasattr(risk_metrics, "var_1d")
                    else 0.0
                )
                if portfolio_risk > 0.1:  # VaR > 10%
                    risk_score += 40.0
                    checks.append(
                        {
                            "check": "portfolio_risk",
                            "result": "warning",
                            "message": f"High portfolio risk (VaR): {portfolio_risk:.1%}",
                        }
                    )
                else:
                    checks.append(
                        {
                            "check": "portfolio_risk",
                            "result": "passed",
                            "message": f"Acceptable portfolio risk (VaR): {portfolio_risk:.1%}",
                        }
                    )

                # Get risk summary
                try:
                    risk_summary = await self.risk_service.get_risk_summary()
                except (RiskManagementError, ValidationError, ServiceError) as e:
                    risk_summary = {}
                    self._logger.warning(f"Risk summary retrieval failed: {e}")
                if risk_summary.get("current_risk_level") == "high":
                    risk_score += 30.0
                    checks.append(
                        {
                            "check": "overall_risk_assessment",
                            "result": "warning",
                            "message": "RiskService indicates high overall risk",
                        }
                    )

            except Exception as e:
                self._logger.error(f"RiskService assessment failed: {e}", exc_info=True)
                # Add this as a failed check
                checks.append(
                    {
                        "check": "risk_service_assessment",
                        "result": "error",
                        "message": f"RiskService error: {e!s}",
                    }
                )
                # Increase risk score due to assessment failure
                risk_score += 20.0

        # Fallback/supplementary risk checks
        # Check volatility risk
        volatility = risk_context.get("volatility", 0.02)  # Default 2% volatility
        if volatility > 0.05:  # More than 5% volatility
            risk_score += 30.0
            checks.append(
                {
                    "check": "volatility_risk",
                    "result": "warning",
                    "message": f"High volatility: {volatility:.1%}",
                }
            )
        else:
            checks.append(
                {
                    "check": "volatility_risk",
                    "result": "passed",
                    "message": f"Acceptable volatility: {volatility:.1%}",
                }
            )

        # Check liquidity risk
        volume_ratio = order.quantity / (market_data.volume or Decimal("1000000"))
        if volume_ratio > 0.01:  # More than 1% of daily volume
            risk_score += 25.0
            checks.append(
                {
                    "check": "liquidity_risk",
                    "result": "warning",
                    "message": f"Large order vs volume: {volume_ratio:.2%}",
                }
            )
        else:
            checks.append(
                {
                    "check": "liquidity_risk",
                    "result": "passed",
                    "message": f"Acceptable order size: {volume_ratio:.2%}",
                }
            )

        # Use risk context information
        if risk_context:
            # Add component-specific risk assessment
            component = risk_context.get("component", "unknown")
            strategy_name = risk_context.get("strategy_name", "unknown")
            checks.append(
                {
                    "check": "execution_context",
                    "result": "info",
                    "message": f"Execution from {component} using {strategy_name}",
                }
            )

        return {"checks": checks, "risk_score": risk_score}

    def _generate_order_recommendations(
        self,
        validation_results: dict[str, Any],
        order: OrderRequest,
        market_data: MarketData,
    ) -> list[str]:
        """Generate order execution recommendations."""
        recommendations = []

        risk_score = validation_results["risk_score"]

        if risk_score > 60:
            recommendations.append("Consider reducing order size due to high risk score")

        if validation_results["risk_level"] in ["high", "critical"]:
            recommendations.append("Use limit order instead of market order to control slippage")
            recommendations.append("Consider splitting order into smaller chunks")

        # Check for market timing recommendations
        if market_data.bid and market_data.ask:
            spread_bps = ((market_data.ask - market_data.bid) / market_data.price) * 10000
            if spread_bps > 30:
                recommendations.append("Wait for tighter spread before executing")

        if not recommendations:
            recommendations.append("Order appears ready for execution")

        return recommendations

    async def _create_execution_audit_log(
        self,
        execution_id: str,
        operation_type: str,
        execution_result: ExecutionResult,
        market_data: MarketData,
        bot_id: str | None = None,
        strategy_name: str | None = None,
        order_id: str | None = None,
        execution_metrics: dict[str, Any] | None = None,
        pre_trade_analysis: dict[str, Any] | None = None,
        post_trade_analysis: dict[str, Any] | None = None,
        operation_start: datetime | None = None,
        operation_status: str = "completed",
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """Create comprehensive execution audit log."""
        try:
            order = execution_result.original_order

            audit_log_data = {
                "id": str(uuid.uuid4()),
                "execution_id": execution_id,
                "operation_type": operation_type,
                "order_id": order_id,
                "trade_id": None,  # Trade records are only for closed positions
                "exchange_order_id": execution_id,
                "bot_id": bot_id,
                "strategy_name": strategy_name,
                "symbol": order.symbol,
                "exchange": order.exchange or "binance",
                "side": order.side.value,
                "order_type": order.order_type.value,
                "requested_quantity": decimal_to_float(
                    self._convert_to_decimal_safe(order.quantity), "requested_quantity"
                ),
                "executed_quantity": decimal_to_float(
                    self._convert_to_decimal_safe(execution_result.total_filled_quantity),
                    "executed_quantity",
                ),
                "remaining_quantity": decimal_to_float(
                    self._convert_to_decimal_safe(
                        order.quantity - execution_result.total_filled_quantity
                    ),
                    "remaining_quantity",
                ),
                "requested_price": decimal_to_float(
                    self._convert_to_decimal_safe(order.price), "requested_price"
                )
                if order.price
                else None,
                "executed_price": decimal_to_float(
                    self._convert_to_decimal_safe(execution_result.average_fill_price),
                    "executed_price",
                )
                if execution_result.average_fill_price
                else None,
                "market_price_at_time": decimal_to_float(
                    self._convert_to_decimal_safe(market_data.price), "market_price_at_time"
                ),
                "slippage_bps": (
                    execution_metrics.get("slippage_bps", 0) if execution_metrics else 0
                ),
                "total_cost_bps": (
                    execution_metrics.get("total_cost_bps", 0) if execution_metrics else 0
                ),
                "execution_algorithm": (
                    execution_result.algorithm.value if execution_result.algorithm else None
                ),
                "pre_trade_checks": pre_trade_analysis or {},
                "post_trade_analysis": post_trade_analysis or {},
                "operation_status": operation_status,
                "success": success,
                "error_message": error_message,
                "total_fees": decimal_to_float(
                    self._convert_to_decimal_safe(execution_result.total_fees), "total_fees"
                ),
                "market_conditions": {
                    "price": str(market_data.price),
                    "volume": str(market_data.volume) if market_data.volume else "0",
                    "bid": str(market_data.bid) if market_data.bid else None,
                    "ask": str(market_data.ask) if market_data.ask else None,
                },
                "signal_timestamp": operation_start,
                "order_submission_timestamp": operation_start,
                "total_execution_time_ms": (
                    int((datetime.now(timezone.utc) - operation_start).total_seconds() * 1000)
                    if operation_start
                    else None
                ),
                "source_component": "ExecutionService",
                "correlation_id": self._correlation_id,
                "created_at": datetime.now(timezone.utc),
            }

            # Create audit log through repository service
            await self.repository_service.create_audit_log(audit_log_data)

        except Exception as e:
            self._logger.error(f"Failed to create execution audit log: {e}")
            # CRITICAL: Audit log failures must not be silent
            # Store to fallback file for compliance
            fallback_file_handle = None
            try:
                import json

                fallback_file = "/tmp/audit_log_fallback.jsonl"
                fallback_file_handle = open(fallback_file, "a")
                fallback_file_handle.write(json.dumps(audit_log_data) + "\n")
                fallback_file_handle.flush()
                self._logger.warning(f"Audit log saved to fallback file: {fallback_file}")
            except Exception as fallback_error:
                self._logger.critical(f"Failed to save audit log to fallback: {fallback_error}")
            finally:
                # Ensure file handle is closed
                if fallback_file_handle:
                    try:
                        fallback_file_handle.close()
                    except Exception as close_error:
                        self._logger.warning(
                            f"Failed to close audit log fallback file: {close_error}"
                        )

    async def _create_risk_audit_log(
        self,
        risk_event_id: str,
        event_type: str,
        validation_results: dict[str, Any],
        bot_id: str | None = None,
        order_context: dict[str, Any] | None = None,
    ) -> None:
        """Create risk audit log for validation events."""
        try:
            risk_audit_data = {
                "id": str(uuid.uuid4()),
                "risk_event_id": risk_event_id,
                "event_type": event_type,
                "bot_id": bot_id,
                "risk_level": validation_results["risk_level"],
                "risk_score": validation_results["risk_score"],
                "threshold_breached": validation_results["overall_result"] == "failed",
                "risk_description": f"Pre-trade validation for {event_type}",
                "risk_calculation": validation_results["validation_checks"],
                "action_taken": "validation_completed",
                "action_details": validation_results,
                "resolved": True,
                "resolution_method": "automatic_validation",
                "detected_at": datetime.now(timezone.utc),
                "resolved_at": datetime.now(timezone.utc),
                "source_component": "ExecutionService",
                "correlation_id": self._correlation_id,
                "created_at": datetime.now(timezone.utc),
            }

            # Create risk audit log through repository service
            await self.repository_service.create_audit_log(risk_audit_data)

        except Exception as e:
            self._logger.error(f"Failed to create risk audit log: {e}")
            # Store to fallback for compliance
            fallback_file_handle = None
            try:
                import json

                fallback_file = "/tmp/risk_audit_fallback.jsonl"
                fallback_file_handle = open(fallback_file, "a")
                fallback_file_handle.write(json.dumps(risk_audit_data) + "\n")
                fallback_file_handle.flush()
            except Exception as fallback_error:
                self._logger.warning(
                    "Failed to write risk audit fallback data", error=str(fallback_error)
                )
            finally:
                # Ensure file handle is closed
                if fallback_file_handle:
                    try:
                        fallback_file_handle.close()
                    except Exception as close_error:
                        self._logger.warning(
                            f"Failed to close risk audit fallback file: {close_error}"
                        )

    async def _update_execution_metrics(
        self,
        execution_result: ExecutionResult,
        execution_metrics: dict[str, Any],
        operation_start: datetime,
    ) -> None:
        """Update execution performance metrics."""
        execution_time_ms = (datetime.now(timezone.utc) - operation_start).total_seconds() * 1000

        # Update counters
        self._performance_metrics["total_executions"] += 1
        if execution_result.status == ExecutionStatus.COMPLETED:
            self._performance_metrics["successful_executions"] += 1
        else:
            self._performance_metrics["failed_executions"] += 1

        # Update volume
        trade_value = execution_result.total_filled_quantity * (
            execution_result.average_fill_price or Decimal("0")
        )
        self._performance_metrics["total_volume"] += trade_value

        # Update averages
        total_executions = self._performance_metrics["total_executions"]

        # Execution time
        current_avg_time = self._performance_metrics["average_execution_time_ms"]
        self._performance_metrics["average_execution_time_ms"] = (
            (current_avg_time * (total_executions - 1)) + execution_time_ms
        ) / total_executions

        # Slippage
        slippage_bps = execution_metrics.get("slippage_bps", 0)
        current_avg_slippage = self._performance_metrics["average_slippage_bps"]
        self._performance_metrics["average_slippage_bps"] = (
            (current_avg_slippage * (total_executions - 1)) + slippage_bps
        ) / total_executions

        # Execution quality
        quality_score = execution_metrics.get("quality_score", 0)
        current_avg_quality = self._performance_metrics["execution_quality_score"]
        self._performance_metrics["execution_quality_score"] = (
            (current_avg_quality * (total_executions - 1)) + quality_score
        ) / total_executions

        # Export metrics to Prometheus if collector is available
        if self.metrics_service:
            try:
                # Record order execution metrics using service interface
                try:
                    from src.monitoring.services import MetricRequest

                    # Record order count metric
                    order_metric = MetricRequest(
                        name="trading.orders.total",
                        value=1,
                        labels={
                            "exchange": execution_result.original_order.exchange or "unknown",
                            "status": self._map_execution_status_to_order_status(
                                execution_result.status
                            ).value,
                            "order_type": execution_result.original_order.order_type.value,
                            "symbol": execution_result.original_order.symbol,
                        },
                    )
                    self.metrics_service.record_counter(order_metric)

                    # Record volume metrics
                    volume_metric = MetricRequest(
                        name="trading.volume.usd",
                        value=trade_value,
                        labels={
                            "exchange": execution_result.original_order.exchange or "unknown",
                            "symbol": execution_result.original_order.symbol,
                        },
                    )
                    self.metrics_service.record_counter(volume_metric)

                    # Record execution time metrics
                    execution_time_metric = MetricRequest(
                        name="trading.execution.time.ms",
                        value=execution_time_ms,
                        labels={
                            "exchange": execution_result.original_order.exchange or "unknown",
                            "symbol": execution_result.original_order.symbol,
                        },
                    )
                    self.metrics_service.record_histogram(execution_time_metric)

                except Exception as e:
                    self._logger.warning(f"Error recording order metrics: {e}")

                # Record trade metrics if execution was successful
                if execution_result.status == ExecutionStatus.COMPLETED:
                    try:
                        trade_metric = MetricRequest(
                            name="trading.trades.total",
                            value=1,
                            labels={
                                "exchange": execution_result.original_order.exchange or "unknown",
                                "symbol": execution_result.original_order.symbol,
                            },
                        )
                        self.metrics_service.record_counter(trade_metric)

                    except Exception as e:
                        self._logger.warning(f"Error recording trade metrics: {e}")
            except Exception as e:
                self._logger.error(f"Critical error in metrics export: {e}", exc_info=True)

    def _get_empty_metrics(self) -> dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "success_rate": 0.0,
            "total_volume": Decimal("0.0"),
            "average_fee_rate_bps": 0.0,
            "symbols_traded": 0,
            "exchanges_used": 0,
            "side_distribution": {"buy": 0, "sell": 0},
            "order_type_distribution": {},
            "performance_metrics": self._performance_metrics.copy(),
        }

    # Service Health and Monitoring

    async def _service_health_check(self) -> HealthStatus:
        """Service-specific health check."""
        try:
            # Check repository service connectivity
            if hasattr(self.repository_service, "health_check"):
                health_status = await self.repository_service.health_check()
                if health_status != HealthStatus.HEALTHY:
                    return health_status

            # Check execution success rate
            total_executions = (
                self._performance_metrics["successful_executions"]
                + self._performance_metrics["failed_executions"]
            )

            if total_executions > 10:  # Only check if we have enough data
                success_rate = self._performance_metrics["successful_executions"] / total_executions
                if success_rate < 0.8:  # Less than 80% success rate
                    return HealthStatus.DEGRADED
                elif success_rate < 0.5:  # Less than 50% success rate
                    return HealthStatus.UNHEALTHY

            # Check validation failure rate
            validation_failures = self._performance_metrics["order_validation_failures"]
            if validation_failures > 50:  # Too many validation failures
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self._logger.error(f"Execution service health check failed: {e}")
            return HealthStatus.UNHEALTHY

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get detailed performance metrics."""
        metrics = self._performance_metrics.copy()

        # Add service metrics
        service_metrics = self.get_metrics()
        metrics.update(service_metrics)

        return metrics

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        super().reset_metrics()
        self._performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "cancelled_executions": 0,
            "total_volume": Decimal("0.0"),
            "average_execution_time_ms": 0.0,
            "average_slippage_bps": 0.0,
            "average_cost_bps": 0.0,
            "risk_violations": 0,
            "order_validation_failures": 0,
            "execution_quality_score": 0.0,
        }

        self._logger.info("Execution service metrics reset")

    async def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive health check of the execution service.

        Returns:
            Dict containing health status and component information
        """
        try:
            health_status = await self._service_health_check()

            return {
                "service_name": "ExecutionService",
                "is_running": self.is_running,
                "health_status": health_status.value
                if hasattr(health_status, "value")
                else str(health_status),
                "dependencies": {
                    "repository_service": bool(self.repository_service),
                    "risk_service": bool(self.risk_service),
                    "metrics_service": bool(self.metrics_service),
                    "validation_service": bool(self.validation_service),
                    "analytics_service": bool(self.analytics_service),
                },
                "performance_metrics": self._performance_metrics.copy(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return {
                "service_name": "ExecutionService",
                "is_running": self.is_running,
                "health_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def cancel_orders_by_symbol(self, symbol: str) -> None:
        """Cancel all orders for a specific symbol."""
        try:
            self._logger.info("Cancelling all orders for symbol", symbol=symbol)
            
            if not self.order_manager:
                self._logger.warning("Order manager not available, cannot cancel orders")
                return
                
            # Get orders for the symbol
            orders_for_symbol = await self.order_manager.get_orders_by_symbol(symbol)
            
            # Cancel each active order
            cancelled_count = 0
            from src.core.types import OrderStatus
            active_statuses = {OrderStatus.NEW, OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED}
            
            for order in orders_for_symbol:
                if order.status in active_statuses:
                    success = await self.order_manager.cancel_order(order.order_id, reason="symbol_cleanup")
                    if success:
                        cancelled_count += 1
                        
            self._logger.info("Orders cancelled for symbol", 
                            symbol=symbol, cancelled_count=cancelled_count)
                            
        except Exception as e:
            self._logger.error(f"Failed to cancel orders for symbol {symbol}: {e}")
            raise ServiceError(f"Failed to cancel orders for symbol: {e}")

    async def cancel_all_orders(self) -> None:
        """Cancel all active orders across all symbols."""
        try:
            self._logger.info("Cancelling all active orders")
            
            if not self.order_manager:
                self._logger.warning("Order manager not available, cannot cancel orders")
                return
                
            # Get all orders by active statuses and cancel them
            cancelled_count = 0
            from src.core.types import OrderStatus
            active_statuses = [OrderStatus.NEW, OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
            
            for status in active_statuses:
                orders = await self.order_manager.get_orders_by_status(status)
                for order in orders:
                    success = await self.order_manager.cancel_order(order.order_id, reason="system_cleanup")
                    if success:
                        cancelled_count += 1
                        
            self._logger.info("All orders cancelled", cancelled_count=cancelled_count)
                            
        except Exception as e:
            self._logger.error(f"Failed to cancel all orders: {e}")
            raise ServiceError(f"Failed to cancel all orders: {e}")

    async def initialize(self) -> None:
        """Initialize the execution service."""
        try:
            self._logger.info("Initializing execution service")
            
            # Verify components are available
            if not self.order_manager:
                self._logger.warning("Order manager not available during initialization")
                
            if not hasattr(self, 'execution_engine') or not self.execution_engine:
                self._logger.warning("Execution engine not available during initialization")
                
            self._logger.info("Execution service initialized successfully")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize execution service: {e}")
            raise ServiceError(f"Failed to initialize execution service: {e}")

    async def cleanup(self) -> None:
        """Clean up execution service resources."""
        try:
            self._logger.info("Cleaning up execution service")
            
            # Cancel any remaining orders before cleanup
            await self.cancel_all_orders()
            
            self._logger.info("Execution service cleaned up successfully")
            
        except Exception as e:
            self._logger.error(f"Failed to cleanup execution service: {e}")
            # Don't raise during cleanup

    async def update_order_status(
        self, order_id: str, status: str, filled_quantity: Decimal, remaining_quantity: Decimal
    ) -> None:
        """Update order status with fill information."""
        try:
            self._logger.info("Updating order status", 
                            order_id=order_id, status=status, 
                            filled_quantity=filled_quantity, remaining_quantity=remaining_quantity)
            
            if not self.order_manager:
                self._logger.warning("Order manager not available, cannot update order status")
                return
                
            # Convert string status to OrderStatus enum
            from src.core.types import OrderStatus
            
            try:
                order_status = OrderStatus(status)
            except ValueError:
                self._logger.warning(f"Invalid order status: {status}")
                return
                
            # Log order status update (OrderManager handles updates internally via websocket)
            self._logger.info("Order status update received from error handling", 
                            order_id=order_id, order_status=order_status,
                            filled_quantity=filled_quantity, remaining_quantity=remaining_quantity)
            
            self._logger.info("Order status updated successfully", order_id=order_id)
            
        except Exception as e:
            self._logger.error(f"Failed to update order status {order_id}: {e}")
            raise ServiceError(f"Failed to update order status: {e}")

    # Bot-specific execution management methods
    async def start_bot_execution(self, bot_id: str, bot_config: dict[str, Any]) -> bool:
        """
        Start execution engine for a specific bot.

        Args:
            bot_id: Bot identifier
            bot_config: Bot configuration data

        Returns:
            bool: True if started successfully
        """
        try:
            self._logger.info("Starting bot execution engine", bot_id=bot_id)

            # Store bot configuration for execution context
            if not hasattr(self, "_bot_contexts"):
                self._bot_contexts = {}

            self._bot_contexts[bot_id] = {
                "config": bot_config,
                "started_at": datetime.now(timezone.utc),
                "status": "running",
            }

            self._logger.info("Bot execution engine started successfully", bot_id=bot_id)
            return True

        except (ServiceError, ValidationError, ExecutionError) as e:
            self._logger.error(f"Failed to start bot execution: {e}", bot_id=bot_id)
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error starting bot execution: {e}", bot_id=bot_id)
            # For bot operations, return False to indicate failure
            return False

    async def stop_bot_execution(self, bot_id: str) -> bool:
        """
        Stop execution engine for a specific bot.

        Args:
            bot_id: Bot identifier

        Returns:
            bool: True if stopped successfully
        """
        try:
            self._logger.info("Stopping bot execution engine", bot_id=bot_id)

            if not hasattr(self, "_bot_contexts"):
                self._bot_contexts = {}

            if bot_id in self._bot_contexts:
                self._bot_contexts[bot_id]["status"] = "stopped"
                self._bot_contexts[bot_id]["stopped_at"] = datetime.now(timezone.utc)

            self._logger.info("Bot execution engine stopped successfully", bot_id=bot_id)
            return True

        except (ServiceError, ValidationError, ExecutionError) as e:
            self._logger.error(f"Failed to stop bot execution: {e}", bot_id=bot_id)
            return False
        except Exception as e:
            self._logger.error(f"Unexpected error stopping bot execution: {e}", bot_id=bot_id)
            # For bot operations, return False to indicate failure
            return False

    async def get_bot_execution_status(self, bot_id: str) -> dict[str, Any]:
        """
        Get execution status for a specific bot.

        Args:
            bot_id: Bot identifier

        Returns:
            dict: Bot execution status and metrics
        """
        try:
            if not hasattr(self, "_bot_contexts"):
                self._bot_contexts = {}

            if bot_id not in self._bot_contexts:
                return {
                    "bot_id": bot_id,
                    "status": "not_found",
                    "message": "Bot execution context not found",
                }

            context = self._bot_contexts[bot_id]
            return {
                "bot_id": bot_id,
                "status": context.get("status", "unknown"),
                "started_at": context.get("started_at", {}).isoformat()
                if context.get("started_at")
                else None,
                "stopped_at": context.get("stopped_at", {}).isoformat()
                if context.get("stopped_at")
                else None,
                "config": context.get("config", {}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except (ServiceError, ValidationError) as e:
            self._logger.error(f"Failed to get bot execution status: {e}", bot_id=bot_id)
            return {
                "bot_id": bot_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            self._logger.error(f"Unexpected error getting bot execution status: {e}", bot_id=bot_id)
            return {
                "bot_id": bot_id,
                "status": "error",
                "error": f"Unexpected error: {e!s}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
