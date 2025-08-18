"""
Smart Order Router for optimal multi-exchange execution.

This module implements an intelligent order routing algorithm that analyzes
multiple exchanges to find the best execution venue based on liquidity,
fees, slippage, and other market conditions.

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import (
    ExecutionAlgorithm,
    ExecutionInstruction,
    ExecutionResult,
    ExecutionStatus,
    OrderRequest,
)

# MANDATORY: Import from P-002A
# MANDATORY: Import from P-007A
from src.utils.decorators import log_calls, time_execution

from .base_algorithm import BaseAlgorithm

logger = get_logger(__name__)


class SmartOrderRouter(BaseAlgorithm):
    """
    Smart Order Router for optimal multi-exchange execution.

    This algorithm analyzes multiple exchanges to determine the optimal
    routing strategy based on various factors including:
    - Liquidity depth and market impact
    - Trading fees and costs
    - Exchange reliability and latency
    - Order book conditions
    - Historical execution quality

    The router can split orders across multiple exchanges or route to
    the single best venue depending on market conditions.
    """

    def __init__(self, config: Config):
        """
        Initialize Smart Order Router.

        Args:
            config: Application configuration
        """
        super().__init__(config)

        # Router configuration
        self.supported_exchanges = ["binance", "okx", "coinbase"]
        self.max_parallel_executions = 3
        self.min_split_size_pct = 0.1  # Minimum 10% for splits

        # Routing weights for decision making
        self.fee_weight = 0.3
        self.liquidity_weight = 0.4
        self.reliability_weight = 0.2
        self.latency_weight = 0.1

        # Exchange scoring cache
        self.exchange_scores = {}
        self.score_cache_ttl_seconds = 300  # 5 minutes cache
        self.last_score_update = {}

        # Market data cache for routing decisions
        self.market_data_cache = {}
        self.market_data_ttl_seconds = 30  # 30 seconds cache

        # Execution thresholds
        self.min_order_value_for_splitting = Decimal("1000")  # Minimum value to consider splitting
        self.max_slippage_tolerance_bps = 50  # 0.5% maximum slippage

        self.logger.info("Smart Order Router initialized with multi-exchange support")

    def get_algorithm_type(self) -> ExecutionAlgorithm:
        """Get the algorithm type enum."""
        return ExecutionAlgorithm.SMART_ROUTER

    async def _validate_algorithm_parameters(self, instruction: ExecutionInstruction) -> None:
        """
        Validate Smart Router-specific parameters.

        Args:
            instruction: Execution instruction to validate

        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate preferred exchanges
        if instruction.preferred_exchanges:
            for exchange in instruction.preferred_exchanges:
                if exchange not in self.supported_exchanges:
                    raise ValidationError(
                        f"Unsupported exchange: {exchange}. Supported: {self.supported_exchanges}"
                    )

        # Validate exchange restrictions
        if instruction.avoid_exchanges:
            remaining_exchanges = [
                ex for ex in self.supported_exchanges if ex not in instruction.avoid_exchanges
            ]
            if not remaining_exchanges:
                raise ValidationError("Cannot avoid all supported exchanges")

        # Validate slippage tolerance
        if instruction.max_slippage_bps is not None:
            if instruction.max_slippage_bps <= 0:
                raise ValidationError("Max slippage must be positive")
            if instruction.max_slippage_bps > 1000:  # 10%
                raise ValidationError("Max slippage too high (maximum 10%)")

    @time_execution
    @log_calls
    async def execute(
        self, instruction: ExecutionInstruction, exchange_factory=None, risk_manager=None
    ) -> ExecutionResult:
        """
        Execute an order using Smart Order Router.

        Args:
            instruction: Execution instruction with routing preferences
            exchange_factory: Factory for creating exchange instances
            risk_manager: Risk manager for order validation

        Returns:
            ExecutionResult: Result of the smart routing execution

        Raises:
            ExecutionError: If execution fails
            ValidationError: If instruction is invalid
        """
        try:
            # Validate instruction
            await self.validate_instruction(instruction)

            # Create execution result for tracking
            execution_result = await self._create_execution_result(instruction)
            execution_id = execution_result.execution_id

            # Register execution as running
            self.current_executions[execution_id] = execution_result
            self.is_running = True
            execution_result.status = ExecutionStatus.RUNNING

            self.logger.info(
                "Starting Smart Router execution",
                execution_id=execution_id,
                symbol=instruction.order.symbol,
                quantity=float(instruction.order.quantity),
            )

            # Get exchange factory
            if not exchange_factory:
                raise ExecutionError("Exchange factory is required for Smart Router")

            # Analyze routing options
            routing_plan = await self._create_routing_plan(instruction, exchange_factory)

            # Execute routing plan
            await self._execute_routing_plan(
                routing_plan, execution_result, exchange_factory, risk_manager
            )

            # Finalize execution result
            await self._finalize_execution(execution_result)

            # Update statistics
            if execution_result.status == ExecutionStatus.COMPLETED:
                self.successful_executions += 1
            else:
                self.failed_executions += 1
            self.total_executions += 1

            self.logger.info(
                "Smart Router execution completed",
                execution_id=execution_id,
                status=execution_result.status.value,
                filled_quantity=float(execution_result.total_filled_quantity),
                exchanges_used=len(routing_plan["routes"]),
            )

            return execution_result

        except Exception as e:
            # Handle execution failure
            if "execution_id" in locals() and execution_id in self.current_executions:
                await self._update_execution_result(
                    self.current_executions[execution_id],
                    status=ExecutionStatus.FAILED,
                    error_message=str(e),
                )
                self.failed_executions += 1
                self.total_executions += 1

            self.logger.error(
                "Smart Router execution failed",
                execution_id=execution_id if "execution_id" in locals() else "unknown",
                error=str(e),
            )
            raise ExecutionError(f"Smart Router execution failed: {e}")

        finally:
            self.is_running = False

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an ongoing Smart Router execution.

        Args:
            execution_id: ID of the execution to cancel

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        try:
            if execution_id not in self.current_executions:
                self.logger.warning(f"Execution not found for cancellation: {execution_id}")
                return False

            execution_result = self.current_executions[execution_id]

            if execution_result.status not in [ExecutionStatus.RUNNING, ExecutionStatus.PENDING]:
                self.logger.warning(
                    f"Cannot cancel execution in status: {execution_result.status.value}"
                )
                return False

            # Update status to cancelled
            await self._update_execution_result(execution_result, status=ExecutionStatus.CANCELLED)

            self.logger.info(f"Smart Router execution cancelled: {execution_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel Smart Router execution: {e}")
            return False

    async def _create_routing_plan(
        self, instruction: ExecutionInstruction, exchange_factory
    ) -> dict[str, Any]:
        """
        Create optimal routing plan by analyzing all available exchanges.

        Args:
            instruction: Execution instruction
            exchange_factory: Factory for exchange access

        Returns:
            dict: Routing plan with exchange allocations
        """
        try:
            # Determine candidate exchanges
            candidate_exchanges = await self._get_candidate_exchanges(instruction)

            # Score each exchange
            exchange_scores = await self._score_exchanges(
                candidate_exchanges, instruction, exchange_factory
            )

            # Determine routing strategy
            order_value = instruction.order.quantity * (instruction.order.price or Decimal("50000"))

            if (
                order_value >= self.min_order_value_for_splitting
                and len(exchange_scores) > 1
                and not instruction.is_urgent
            ):
                # Use multi-exchange routing for large orders
                routes = await self._create_split_routing(instruction, exchange_scores)
                strategy = "split_routing"
            else:
                # Use single best exchange
                best_exchange = max(exchange_scores, key=exchange_scores.get)
                routes = [
                    {
                        "exchange": best_exchange,
                        "quantity": instruction.order.quantity,
                        "allocation_pct": 1.0,
                        "priority": 1,
                    }
                ]
                strategy = "single_exchange"

            routing_plan = {
                "strategy": strategy,
                "total_quantity": instruction.order.quantity,
                "routes": routes,
                "exchange_scores": exchange_scores,
                "candidate_exchanges": candidate_exchanges,
            }

            self.logger.debug(
                "Routing plan created",
                strategy=strategy,
                num_routes=len(routes),
                best_exchange=routes[0]["exchange"],
            )

            return routing_plan

        except Exception as e:
            self.logger.error(f"Failed to create routing plan: {e}")
            # Fallback to simple single-exchange routing
            return {
                "strategy": "fallback",
                "total_quantity": instruction.order.quantity,
                "routes": [
                    {
                        "exchange": "binance",  # Default fallback
                        "quantity": instruction.order.quantity,
                        "allocation_pct": 1.0,
                        "priority": 1,
                    }
                ],
                "exchange_scores": {},
                "candidate_exchanges": ["binance"],
            }

    async def _get_candidate_exchanges(self, instruction: ExecutionInstruction) -> list[str]:
        """
        Get list of candidate exchanges for routing.

        Args:
            instruction: Execution instruction with preferences

        Returns:
            list[str]: List of candidate exchange names
        """
        # Start with preferred exchanges if specified
        if instruction.preferred_exchanges:
            candidates = [
                ex for ex in instruction.preferred_exchanges if ex in self.supported_exchanges
            ]
        else:
            candidates = self.supported_exchanges.copy()

        # Remove avoided exchanges
        if instruction.avoid_exchanges:
            candidates = [ex for ex in candidates if ex not in instruction.avoid_exchanges]

        # Ensure we have at least one candidate
        if not candidates:
            candidates = ["binance"]  # Fallback

        return candidates

    async def _score_exchanges(
        self, exchanges: list[str], instruction: ExecutionInstruction, exchange_factory
    ) -> dict[str, float]:
        """
        Score exchanges based on multiple factors.

        Args:
            exchanges: List of exchanges to score
            instruction: Execution instruction
            exchange_factory: Factory for exchange access

        Returns:
            dict: Exchange scores (higher is better)
        """
        scores = {}

        for exchange_name in exchanges:
            try:
                # Check cache first
                cache_key = f"{exchange_name}_{instruction.order.symbol}"
                current_time = datetime.now(timezone.utc)

                if (
                    cache_key in self.last_score_update
                    and (current_time - self.last_score_update[cache_key]).total_seconds()
                    < self.score_cache_ttl_seconds
                ):
                    scores[exchange_name] = self.exchange_scores.get(cache_key, 0.5)
                    continue

                # Calculate fresh score
                exchange = await exchange_factory.get_exchange(exchange_name)
                if not exchange:
                    scores[exchange_name] = 0.0
                    continue

                # Factor 1: Fee score (lower fees = higher score)
                fee_score = await self._calculate_fee_score(exchange, instruction)

                # Factor 2: Liquidity score
                liquidity_score = await self._calculate_liquidity_score(
                    exchange, instruction.order.symbol
                )

                # Factor 3: Reliability score
                reliability_score = await self._calculate_reliability_score(exchange_name)

                # Factor 4: Latency score
                latency_score = await self._calculate_latency_score(exchange)

                # Weighted composite score
                composite_score = (
                    fee_score * self.fee_weight
                    + liquidity_score * self.liquidity_weight
                    + reliability_score * self.reliability_weight
                    + latency_score * self.latency_weight
                )

                scores[exchange_name] = composite_score

                # Update cache
                self.exchange_scores[cache_key] = composite_score
                self.last_score_update[cache_key] = current_time

                self.logger.debug(
                    "Exchange scored",
                    exchange=exchange_name,
                    composite_score=composite_score,
                    fee_score=fee_score,
                    liquidity_score=liquidity_score,
                    reliability_score=reliability_score,
                    latency_score=latency_score,
                )

            except Exception as e:
                self.logger.warning(f"Failed to score exchange {exchange_name}: {e}")
                scores[exchange_name] = 0.1  # Low but non-zero score

        return scores

    async def _calculate_fee_score(self, exchange, instruction: ExecutionInstruction) -> float:
        """Calculate fee-based score for an exchange."""
        try:
            # Simplified fee calculation (would use actual fee schedules)
            # Default fees by exchange (as percentages)
            default_fees = {"binance": 0.1, "okx": 0.08, "coinbase": 0.15}  # 0.1%  # 0.08%  # 0.15%

            exchange_name = exchange.exchange_name
            fee_pct = default_fees.get(exchange_name, 0.2)

            # Convert to score (lower fees = higher score)
            # Normalize between 0 and 1
            max_fee = 0.3  # 0.3% maximum expected fee
            fee_score = max(0.0, (max_fee - fee_pct) / max_fee)

            return fee_score

        except Exception as e:
            self.logger.warning(f"Fee score calculation failed: {e}")
            return 0.5

    async def _calculate_liquidity_score(self, exchange, symbol: str) -> float:
        """Calculate liquidity-based score for an exchange."""
        try:
            # Get market data to assess liquidity
            market_data = await exchange.get_market_data(symbol)

            if not market_data or not market_data.bid or not market_data.ask:
                return 0.3  # Low score for missing data

            # Calculate spread as liquidity indicator
            spread = market_data.ask - market_data.bid
            spread_bps = (spread / market_data.price) * 10000

            # Calculate volume score
            volume_score = min(1.0, float(market_data.volume) / 1000000)  # Normalize by 1M

            # Calculate spread score (tighter spreads = higher score)
            max_spread_bps = 100  # 1% maximum expected spread
            spread_score = max(0.0, (max_spread_bps - float(spread_bps)) / max_spread_bps)

            # Combine volume and spread scores
            liquidity_score = (volume_score * 0.6) + (spread_score * 0.4)

            return liquidity_score

        except Exception as e:
            self.logger.warning(f"Liquidity score calculation failed: {e}")
            return 0.5

    async def _calculate_reliability_score(self, exchange_name: str) -> float:
        """Calculate reliability score based on historical performance."""
        # Simplified reliability scoring (would use actual uptime data)
        reliability_scores = {
            "binance": 0.95,  # 95% reliability
            "okx": 0.90,  # 90% reliability
            "coinbase": 0.92,  # 92% reliability
        }

        return reliability_scores.get(exchange_name, 0.8)

    async def _calculate_latency_score(self, exchange) -> float:
        """Calculate latency score for an exchange."""
        try:
            # Simple health check to measure responsiveness
            start_time = datetime.now(timezone.utc)
            is_healthy = await exchange.health_check()
            end_time = datetime.now(timezone.utc)

            if not is_healthy:
                return 0.0

            # Calculate latency score
            latency_ms = (end_time - start_time).total_seconds() * 1000
            max_latency = 1000  # 1 second maximum expected latency
            latency_score = max(0.0, (max_latency - latency_ms) / max_latency)

            return latency_score

        except Exception as e:
            self.logger.warning(f"Latency score calculation failed: {e}")
            return 0.5

    async def _create_split_routing(
        self, instruction: ExecutionInstruction, exchange_scores: dict[str, float]
    ) -> list[dict[str, Any]]:
        """
        Create multi-exchange split routing plan.

        Args:
            instruction: Execution instruction
            exchange_scores: Scored exchanges

        Returns:
            list: Route allocations
        """
        # Sort exchanges by score
        sorted_exchanges = sorted(exchange_scores.items(), key=lambda x: x[1], reverse=True)

        # Use top exchanges for splitting
        num_splits = min(3, len(sorted_exchanges))  # Maximum 3-way split
        top_exchanges = sorted_exchanges[:num_splits]

        # Calculate allocations based on scores
        total_score = sum(score for _, score in top_exchanges)
        routes = []
        remaining_quantity = instruction.order.quantity

        for i, (exchange_name, score) in enumerate(top_exchanges):
            if i == len(top_exchanges) - 1:
                # Last exchange gets remainder
                allocation_quantity = remaining_quantity
            else:
                # Allocate based on score proportion
                allocation_pct = score / total_score
                allocation_quantity = instruction.order.quantity * Decimal(str(allocation_pct))

                # Ensure minimum allocation
                min_allocation = instruction.order.quantity * Decimal(str(self.min_split_size_pct))
                allocation_quantity = max(allocation_quantity, min_allocation)

            if allocation_quantity > remaining_quantity:
                allocation_quantity = remaining_quantity

            routes.append(
                {
                    "exchange": exchange_name,
                    "quantity": allocation_quantity,
                    "allocation_pct": float(allocation_quantity / instruction.order.quantity),
                    "priority": i + 1,
                    "score": score,
                }
            )

            remaining_quantity -= allocation_quantity

            if remaining_quantity <= 0:
                break

        return routes

    async def _execute_routing_plan(
        self,
        routing_plan: dict[str, Any],
        execution_result: ExecutionResult,
        exchange_factory,
        risk_manager,
    ) -> None:
        """
        Execute the routing plan across multiple exchanges.

        Args:
            routing_plan: Routing plan with allocations
            execution_result: Execution result to update
            exchange_factory: Factory for exchange access
            risk_manager: Risk manager for validation
        """
        try:
            if routing_plan["strategy"] == "single_exchange":
                # Execute on single best exchange
                await self._execute_single_exchange_route(
                    routing_plan["routes"][0], execution_result, exchange_factory, risk_manager
                )
            else:
                # Execute split routing
                await self._execute_split_routing(
                    routing_plan["routes"], execution_result, exchange_factory, risk_manager
                )

        except Exception as e:
            self.logger.error(f"Routing plan execution failed: {e}")
            raise ExecutionError(f"Routing execution failed: {e}")

    async def _execute_single_exchange_route(
        self,
        route: dict[str, Any],
        execution_result: ExecutionResult,
        exchange_factory,
        risk_manager,
    ) -> None:
        """Execute order on a single exchange."""
        try:
            exchange = await exchange_factory.get_exchange(route["exchange"])
            if not exchange:
                raise ExecutionError(f"Failed to get exchange: {route['exchange']}")

            # Create order
            order = OrderRequest(
                symbol=execution_result.original_order.symbol,
                side=execution_result.original_order.side,
                order_type=execution_result.original_order.order_type,
                quantity=route["quantity"],
                price=execution_result.original_order.price,
                client_order_id=f"{execution_result.execution_id}_sr_{route['exchange']}",
            )

            # Validate with risk manager
            if risk_manager:
                portfolio_value = Decimal("100000")  # Default portfolio value
                is_valid = await risk_manager.validate_order(order, portfolio_value)
                if not is_valid:
                    raise ExecutionError("Risk manager rejected smart router order")

            # Place order
            order_response = await exchange.place_order(order)

            # Update execution result
            await self._update_execution_result(execution_result, child_order=order_response)

            self.logger.info(
                "Smart router single exchange execution completed",
                exchange=route["exchange"],
                quantity=float(route["quantity"]),
                order_id=order_response.id,
            )

        except Exception as e:
            self.logger.error(f"Single exchange route execution failed: {e}")
            raise

    async def _execute_split_routing(
        self,
        routes: list[dict[str, Any]],
        execution_result: ExecutionResult,
        exchange_factory,
        risk_manager,
    ) -> None:
        """Execute split routing across multiple exchanges."""
        try:
            # Create execution tasks for parallel execution
            tasks = []

            for route in routes:
                task = self._execute_route_async(
                    route, execution_result, exchange_factory, risk_manager
                )
                tasks.append(task)

            # Execute routes in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            successful_routes = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(
                        f"Route execution failed for {routes[i]['exchange']}: {result}"
                    )
                else:
                    successful_routes += 1

            if successful_routes == 0:
                raise ExecutionError("All routes failed during split execution")

            self.logger.info(
                f"Split routing completed: {successful_routes}/{len(routes)} routes successful"
            )

        except Exception as e:
            self.logger.error(f"Split routing execution failed: {e}")
            raise

    async def _execute_route_async(
        self,
        route: dict[str, Any],
        execution_result: ExecutionResult,
        exchange_factory,
        risk_manager,
    ) -> None:
        """Execute a single route asynchronously."""
        try:
            exchange = await exchange_factory.get_exchange(route["exchange"])
            if not exchange:
                raise ExecutionError(f"Failed to get exchange: {route['exchange']}")

            # Create order for this route
            order = OrderRequest(
                symbol=execution_result.original_order.symbol,
                side=execution_result.original_order.side,
                order_type=execution_result.original_order.order_type,
                quantity=route["quantity"],
                price=execution_result.original_order.price,
                client_order_id=f"{execution_result.execution_id}_sr_{route['exchange']}",
            )

            # Validate with risk manager
            if risk_manager:
                portfolio_value = Decimal("100000")  # Default portfolio value
                is_valid = await risk_manager.validate_order(order, portfolio_value)
                if not is_valid:
                    raise ExecutionError(f"Risk manager rejected order for {route['exchange']}")

            # Place order
            order_response = await exchange.place_order(order)

            # Update execution result (thread-safe update)
            await self._update_execution_result(execution_result, child_order=order_response)

            self.logger.info(
                "Route executed successfully",
                exchange=route["exchange"],
                quantity=float(route["quantity"]),
                allocation_pct=route["allocation_pct"],
                order_id=order_response.id,
            )

        except Exception as e:
            self.logger.error(f"Route execution failed for {route['exchange']}: {e}")
            raise

    async def _finalize_execution(self, execution_result: ExecutionResult) -> None:
        """
        Finalize the Smart Router execution result.

        Args:
            execution_result: Execution result to finalize
        """
        try:
            # Determine final status
            if execution_result.status == ExecutionStatus.CANCELLED:
                # Already set to cancelled
                pass
            elif execution_result.total_filled_quantity >= execution_result.original_order.quantity:
                execution_result.status = ExecutionStatus.COMPLETED
            elif execution_result.total_filled_quantity > 0:
                execution_result.status = ExecutionStatus.PARTIALLY_FILLED
            else:
                execution_result.status = ExecutionStatus.FAILED
                execution_result.error_message = "No fills received"

            # Calculate Smart Router-specific metrics
            if execution_result.total_filled_quantity > 0:
                # Calculate routing effectiveness
                fill_rate = (
                    execution_result.total_filled_quantity
                    / execution_result.original_order.quantity
                )
                execution_result.metadata["smart_routing_fill_rate"] = float(fill_rate)
                execution_result.metadata["multi_exchange_execution"] = (
                    len(execution_result.child_orders) > 1
                )
                execution_result.metadata["exchanges_used"] = len(
                    set(
                        order.client_order_id.split("_sr_")[-1]
                        for order in execution_result.child_orders
                        if "_sr_" in order.client_order_id
                    )
                )

                # Calculate slippage metrics
                await self._calculate_slippage_metrics(
                    execution_result, expected_price=execution_result.original_order.price
                )

            self.logger.debug(
                "Smart Router execution finalized",
                execution_id=execution_result.execution_id,
                final_status=execution_result.status.value,
                routing_effectiveness=execution_result.metadata.get("smart_routing_fill_rate", 0.0),
            )

        except Exception as e:
            self.logger.error(f"Failed to finalize Smart Router execution: {e}")
            execution_result.error_message = f"Finalization failed: {e}"
