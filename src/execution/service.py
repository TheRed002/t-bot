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
from src.core.exceptions import (
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

# Import database models for type checking and instantiation
from src.database.models import (
    ExecutionAuditLog,
    Order,
    OrderFill,
    RiskAuditLog,
    Trade,
)

# NOTE: Database models should be accessed through DatabaseService
# not imported directly to maintain proper abstraction
# DatabaseService will be injected
from src.error_handling.decorators import with_circuit_breaker, with_retry

# Import risk adapter for proper API usage
# Import monitoring components
from src.monitoring import MetricsCollector, get_tracer

# Import risk management for integration
from src.risk_management.service import RiskService
from src.utils import ValidationFramework, cache_result, format_currency, time_execution
from src.monitoring.financial_precision import safe_decimal_to_float


class ExecutionService(TransactionalService):
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
        database_service=None,
        risk_service: RiskService | None = None,
        metrics_collector: MetricsCollector | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize execution service.

        Args:
            database_service: Database service instance (injected)
            risk_service: Risk service instance (injected)
            metrics_collector: Metrics collector instance (injected)
            correlation_id: Request correlation ID for tracing
        """
        super().__init__(
            name="ExecutionService",
            correlation_id=correlation_id,
        )

        self.database_service = database_service
        self.risk_service = risk_service
        self.metrics_collector = metrics_collector

        # Initialize tracer for distributed tracing with safety check
        try:
            self._tracer = get_tracer("execution.service")
        except Exception as e:
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
            "total_volume": 0.0,
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
            # Resolve database service if not injected
            if not self.database_service:
                self.database_service = self.resolve_dependency("DatabaseService")

            # Resolve risk service if not injected
            if not self.risk_service:
                try:
                    self.risk_service = self.resolve_dependency("RiskService")
                    self._logger.info("RiskService resolved successfully")
                except Exception as e:
                    self._logger.warning(f"RiskService not available: {e}")
                    # Continue without RiskService - fallback to basic validation

            # Ensure database service is running
            if (
                hasattr(self.database_service, "is_running")
                and not self.database_service.is_running
            ):
                await self.database_service.start()

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
        """Initialize execution metrics from database."""
        try:
            # Load recent executions for metrics
            recent_trades = await self.database_service.list_entities(
                model_class=Trade,  # Use actual model class
                limit=100,
                order_by="timestamp",
                order_desc=True,
            )

            if recent_trades:
                # Calculate initial metrics
                total_volume = sum(
                    trade.quantity * trade.executed_price for trade in recent_trades
                )

                self._performance_metrics["total_executions"] = len(recent_trades)
                self._performance_metrics["total_volume"] = total_volume

            self._logger.info(
                "Execution metrics initialized",
                recent_trades=len(recent_trades),
                total_volume=format_currency(self._performance_metrics["total_volume"]),
            )

        except Exception as e:
            self._logger.error(f"Failed to initialize execution metrics: {e}")
            raise

    # Core Execution Operations

    @with_circuit_breaker(failure_threshold=10, recovery_timeout=60.0)
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
        if self._tracer:
            span_context = self._tracer.start_as_current_span("record_trade_execution")
            span = span_context.__enter__()
        else:
            span = None
            span_context = None

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
                    "status": self._map_execution_status_to_order_status(execution_result.status).value,
                    "price": self._convert_to_decimal_safe(execution_result.original_order.price or market_data.price),
                    "quantity": self._convert_to_decimal_safe(execution_result.original_order.quantity),
                    "filled_quantity": self._convert_to_decimal_safe(execution_result.total_filled_quantity),
                    "average_fill_price": self._convert_to_decimal_safe(execution_result.average_fill_price or market_data.price),
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }

                # Create Order instance
                order = Order(**order_data)

                # Save order to database
                saved_order = await self.database_service.create_entity(order)

                # Create OrderFill records
                if execution_result.total_filled_quantity > 0:
                    fill_data = {
                        "id": str(uuid.uuid4()),
                        "order_id": saved_order.id,
                        "exchange_fill_id": execution_result.execution_id,
                        "price": self._convert_to_decimal_safe(execution_result.average_fill_price),
                        "quantity": self._convert_to_decimal_safe(execution_result.total_filled_quantity),
                        "fee": self._convert_to_decimal_safe(execution_result.total_fees),
                        "fee_currency": "USDT",
                        "created_at": datetime.now(timezone.utc),
                    }
                    fill = OrderFill(**fill_data)
                    await self.database_service.create_entity(fill)

                # Convert to dict for response
                saved_order_dict = {
                    "id": str(saved_order.id),
                    "bot_id": saved_order.bot_id,
                    "exchange_order_id": saved_order.exchange_order_id,
                    "symbol": saved_order.symbol,
                    "side": saved_order.side,
                    "type": saved_order.type,
                    "status": saved_order.status,
                    "quantity": float(saved_order.quantity),
                    "filled_quantity": float(saved_order.filled_quantity),
                    "average_fill_price": float(saved_order.average_fill_price) if saved_order.average_fill_price else None,
                }

                # Create comprehensive audit log
                await self._create_execution_audit_log(
                    execution_id=execution_result.execution_id,
                    operation_type="trade_execution",
                    order_id=str(saved_order.id),
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

                self._logger.info(
                    "Order execution recorded successfully",
                    order_id=str(saved_order.id),
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

                self._logger.error(
                    "Trade execution recording failed",
                    execution_id=execution_result.execution_id,
                    symbol=execution_result.original_order.symbol,
                    error=str(e),
                )

                raise ServiceError(f"Trade execution recording failed: {e}")
        finally:
            # Properly close span context if it was opened
            if span_context:
                try:
                    span_context.__exit__(None, None, None)
                except Exception:
                    pass

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
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

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
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

            # Get recent trades
            start_time = datetime.now(timezone.utc) - timedelta(hours=time_range_hours)

            # Add timestamp filter to database query
            if filters is None:
                filters = {}
            filters["timestamp"] = {"gte": start_time}

            # For now, get all orders and filter in memory
            # TODO: Enhance DatabaseService to support complex filters
            all_orders = await self.database_service.list_entities(
                model_class=Order,  # Query Orders, not Trades
                filters=filters,
                order_by="created_at",  # Use created_at for Order model
                order_desc=True,
                limit=1000,
            )

            # Filter by time in memory
            filtered_orders = [
                order for order in all_orders
                if order.created_at >= start_time
            ]

            if not filtered_orders:
                return self._get_empty_metrics()

            # Calculate execution metrics from orders
            total_volume = sum(
                order.filled_quantity * (order.average_fill_price or order.price or Decimal("0"))
                for order in filtered_orders
                if order.filled_quantity and order.filled_quantity > 0
            )

            successful_orders = [
                o
                for o in filtered_orders
                if o.status == OrderStatus.FILLED.value
            ]
            failed_orders = [
                t
                for t in filtered_orders
                if hasattr(t, "status")
                and t.status in [OrderStatus.CANCELLED.value, OrderStatus.REJECTED.value]
            ]

            success_rate = len(successful_orders) / len(filtered_orders) if filtered_orders else 0

            # Calculate average fees from order fills
            avg_fee_rate = 0.0
            total_fees = Decimal("0")
            
            # Get fees from OrderFills if available
            for order in successful_orders:
                if hasattr(order, 'fills') and order.fills:
                    order_fees = sum(fill.fee for fill in order.fills if fill.fee)
                    total_fees += order_fees
            
            if total_volume > 0:
                avg_fee_rate = float((total_fees / total_volume) * 10000)

            metrics = {
                "time_range_hours": time_range_hours,
                "total_trades": len(filtered_orders),
                "successful_orders": len(successful_orders),
                "failed_orders": len(failed_orders),
                "success_rate": success_rate,
                "total_volume": float(total_volume),
                "average_fee_rate_bps": avg_fee_rate,
                "symbols_traded": len(set(order.symbol for order in filtered_orders)),
                "exchanges_used": len(set(order.exchange for order in filtered_orders)),
                "side_distribution": {
                    "buy": len(
                        [
                            o
                            for o in filtered_orders
                            if o.side == OrderSide.BUY.value
                        ]
                    ),
                    "sell": len(
                        [
                            o
                            for o in filtered_orders
                            if o.side == OrderSide.SELL.value
                        ]
                    ),
                },
                "order_type_distribution": {},
                "performance_metrics": self._performance_metrics.copy(),
            }

            # Add order type distribution
            for order_type in [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LOSS]:
                count = len(
                    [
                        o
                        for o in filtered_orders
                        if o.type == order_type.value
                    ]
                )
                metrics["order_type_distribution"][order_type.value] = count

            return metrics

        except Exception as e:
            self._logger.error(f"Failed to calculate execution metrics: {e}")
            raise ServiceError(f"Execution metrics calculation failed: {e}")

    # Helper Methods

    def _convert_to_decimal_safe(self, value: Any, precision: int = 8) -> Decimal:
        """Safely convert value to Decimal with proper precision."""
        if value is None:
            return Decimal("0")
        
        if isinstance(value, Decimal):
            # Quantize to required precision
            return value.quantize(Decimal(f"0.{'0' * precision}"))
        
        # Convert to string first to avoid float precision issues
        decimal_value = Decimal(str(value))
        return decimal_value.quantize(Decimal(f"0.{'0' * precision}"))

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

        # Calculate slippage
        if execution_result.average_fill_price and market_data.price:
            price_diff = execution_result.average_fill_price - market_data.price
            slippage_bps = abs(price_diff / market_data.price) * 10000
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
                ValidationFramework.validate_quantity(str(order.quantity))
                checks.append(
                    {
                        "check": "quantity_validation",
                        "result": "passed",
                        "message": "Quantity is valid",
                    }
                )
            except ValidationError as e:
                errors.append(str(e))
                checks.append(
                    {
                        "check": "quantity_validation",
                        "result": "failed",
                        "message": str(e),
                    }
                )

        # Validate order value
        order_value = order.quantity * (order.price or market_data.price)
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

    async def _validate_position_size(self, order: OrderRequest, bot_id: str | None) -> dict[str, Any]:
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
                risk_validation = await self.risk_service.validate_signal(trading_signal)

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
                recommended_size = await self.risk_service.calculate_position_size(
                    signal=trading_signal,
                    available_capital=None,
                    current_price=Decimal(str(order.price)) if order.price else None,
                )

                if recommended_size and order.quantity > recommended_size:
                    warnings.append(
                        f"Order quantity {order.quantity} exceeds risk-adjusted size {recommended_size}"
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
            # Basic position size check
            if (order.quantity * (order.price or Decimal("1"))) > self.max_order_value:
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
                risk_metrics = await self.risk_service.calculate_risk_metrics(
                    positions=[],  # Current positions will be fetched by RiskService
                    market_data={
                        order.symbol: {
                            "price": str(market_data.price),
                            "volume": str(market_data.volume) if market_data.volume else "0.0",
                            "bid": (
                                str(market_data.bid)
                                if market_data.bid
                                else str(market_data.price)
                            ),
                            "ask": (
                                str(market_data.ask)
                                if market_data.ask
                                else str(market_data.price)
                            ),
                        }
                    },
                )

                # Extract risk information
                portfolio_risk = risk_metrics.get("portfolio_var", 0.0)
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
                risk_summary = await self.risk_service.get_risk_summary()
                if risk_summary.get("total_risk_level") == "high":
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
                "requested_quantity": safe_decimal_to_float(self._convert_to_decimal_safe(order.quantity), "requested_quantity"),
                "executed_quantity": safe_decimal_to_float(self._convert_to_decimal_safe(execution_result.total_filled_quantity), "executed_quantity"),
                "remaining_quantity": safe_decimal_to_float(self._convert_to_decimal_safe(order.quantity - execution_result.total_filled_quantity), "remaining_quantity"),
                "requested_price": safe_decimal_to_float(self._convert_to_decimal_safe(order.price), "requested_price") if order.price else None,
                "executed_price": safe_decimal_to_float(self._convert_to_decimal_safe(execution_result.average_fill_price), "executed_price") if execution_result.average_fill_price else None,
                "market_price_at_time": safe_decimal_to_float(self._convert_to_decimal_safe(market_data.price), "market_price_at_time"),
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
                "total_fees": safe_decimal_to_float(self._convert_to_decimal_safe(execution_result.total_fees), "total_fees"),
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

            # Create ExecutionAuditLog instance
            audit_log = ExecutionAuditLog(**audit_log_data)
            await self.database_service.create_entity(audit_log)

        except Exception as e:
            self._logger.error(f"Failed to create execution audit log: {e}")
            # CRITICAL: Audit log failures must not be silent
            # Store to fallback file for compliance
            try:
                import json
                fallback_file = "/tmp/audit_log_fallback.jsonl"
                with open(fallback_file, "a") as f:
                    f.write(json.dumps(audit_log_data) + "\n")
                self._logger.warning(f"Audit log saved to fallback file: {fallback_file}")
            except Exception as fallback_error:
                self._logger.critical(f"Failed to save audit log to fallback: {fallback_error}")

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
                "risk_score": float(validation_results["risk_score"]),
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

            # Create RiskAuditLog instance
            risk_audit_log = RiskAuditLog(**risk_audit_data)
            await self.database_service.create_entity(risk_audit_log)

        except Exception as e:
            self._logger.error(f"Failed to create risk audit log: {e}")
            # Store to fallback for compliance
            try:
                import json
                fallback_file = "/tmp/risk_audit_fallback.jsonl"
                with open(fallback_file, "a") as f:
                    f.write(json.dumps(risk_audit_data) + "\n")
            except Exception:
                pass

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
        trade_value = (
            execution_result.total_filled_quantity * (execution_result.average_fill_price or Decimal("0"))
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
        if self.metrics_collector:
            try:
                # Validate metrics collector is properly initialized
                if not hasattr(self.metrics_collector, "trading_metrics"):
                    self._logger.warning("MetricsCollector missing trading_metrics attribute")
                    return

                # Record order metrics with validation
                try:
                    self.metrics_collector.trading_metrics.record_order(
                        exchange=execution_result.original_order.exchange or "unknown",
                        status=self._map_execution_status_to_order_status(execution_result.status),
                        order_type=execution_result.original_order.order_type,
                        symbol=execution_result.original_order.symbol,
                        execution_time=execution_time_ms / 1000.0,  # Convert to seconds
                        slippage_bps=slippage_bps,
                    )
                except AttributeError as e:
                    self._logger.warning(f"MetricsCollector API error recording order: {e}")
                except ValueError as e:
                    self._logger.warning(f"Invalid metric value for order recording: {e}")
                except Exception as e:
                    self._logger.warning(f"Unexpected error recording order metrics: {e}")

                # Record trade metrics if execution was successful
                if execution_result.status == ExecutionStatus.COMPLETED:
                    try:
                        # TODO: Implement actual P&L calculation based on position tracking
                        # Currently hardcoded to 0.0 as it requires position history and entry prices
                        pnl_usd = 0.0
                        self.metrics_collector.trading_metrics.record_trade(
                            exchange=execution_result.original_order.exchange or "unknown",
                            strategy="unknown",  # Would need to be passed in
                            symbol=execution_result.original_order.symbol,
                            pnl_usd=pnl_usd,
                            volume_usd=trade_value,
                        )
                    except AttributeError as e:
                        self._logger.warning(f"MetricsCollector API error recording trade: {e}")
                    except ValueError as e:
                        self._logger.warning(f"Invalid metric value for trade recording: {e}")
                    except Exception as e:
                        self._logger.warning(f"Unexpected error recording trade metrics: {e}")
            except Exception as e:
                self._logger.error(f"Critical error in metrics export: {e}", exc_info=True)

    def _get_empty_metrics(self) -> dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "success_rate": 0.0,
            "total_volume": 0.0,
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
            # Check database connectivity
            if hasattr(self.database_service, "health_check"):
                health_status = await self.database_service.health_check()
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
            "total_volume": 0.0,
            "average_execution_time_ms": 0.0,
            "average_slippage_bps": 0.0,
            "average_cost_bps": 0.0,
            "risk_violations": 0,
            "order_validation_failures": 0,
            "execution_quality_score": 0.0,
        }

        self._logger.info("Execution service metrics reset")
