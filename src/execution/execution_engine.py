"""
Main execution engine orchestrator for T-Bot trading system.

This module provides the central orchestration layer for all execution activities,
coordinating between algorithms, order management, slippage analysis, and cost
tracking to deliver optimal execution performance.

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
    MarketData,
    SlippageMetrics,
)

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007A
from src.utils.decorators import log_calls, time_execution

# Import execution components
from .algorithms.base_algorithm import BaseAlgorithm
from .algorithms.iceberg import IcebergAlgorithm
from .algorithms.smart_router import SmartOrderRouter
from .algorithms.twap import TWAPAlgorithm
from .algorithms.vwap import VWAPAlgorithm
from .order_manager import OrderManager
from .slippage.cost_analyzer import CostAnalyzer
from .slippage.slippage_model import SlippageModel

logger = get_logger(__name__)


class ExecutionEngine:
    """
    Central execution engine orchestrator for advanced order execution.

    This engine provides:
    - Unified interface for all execution algorithms
    - Intelligent algorithm selection based on order characteristics
    - Pre-trade slippage prediction and cost estimation
    - Post-trade cost analysis and performance measurement
    - Order lifecycle management and monitoring
    - Risk-aware execution with validation integration
    - Performance tracking and optimization recommendations

    The engine serves as the primary entry point for all execution requests
    and coordinates between various execution components to deliver optimal
    trading performance.
    """

    def __init__(self, config: Config):
        """
        Initialize execution engine with all components.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.error_handler = ErrorHandler(config.error_handling)

        # Initialize core components
        self.order_manager = OrderManager(config)
        self.slippage_model = SlippageModel(config)
        self.cost_analyzer = CostAnalyzer(config)

        # Initialize execution algorithms
        self.algorithms: dict[ExecutionAlgorithm, BaseAlgorithm] = {
            ExecutionAlgorithm.TWAP: TWAPAlgorithm(config),
            ExecutionAlgorithm.VWAP: VWAPAlgorithm(config),
            ExecutionAlgorithm.ICEBERG: IcebergAlgorithm(config),
            ExecutionAlgorithm.SMART_ROUTER: SmartOrderRouter(config),
        }

        # Engine state
        self.is_running = False
        self.active_executions: dict[str, ExecutionResult] = {}

        # Performance tracking
        self.execution_statistics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_volume": Decimal("0"),
            "total_cost_bps": Decimal("0"),
            "average_execution_time_seconds": 0.0,
            "algorithm_usage": {alg.value: 0 for alg in ExecutionAlgorithm},
        }

        # Algorithm selection parameters
        self.algorithm_selection_rules = {
            "large_order_threshold": Decimal("10000"),  # > $10k for advanced algorithms
            "urgency_threshold_minutes": 30,  # < 30min for aggressive execution
            "stealth_volume_ratio": 0.05,  # > 5% of volume for stealth
            "multi_exchange_threshold": Decimal("50000"),  # > $50k for smart routing
        }

        self.logger.info("Execution engine initialized with all components")

    async def start(self) -> None:
        """Start the execution engine."""
        if self.is_running:
            self.logger.warning("Execution engine is already running")
            return

        # Start order manager
        await self.order_manager.start()

        self.is_running = True
        self.logger.info("Execution engine started")

    async def stop(self) -> None:
        """Stop the execution engine and cleanup resources."""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel active executions
        cancellation_tasks = []
        for execution_id in list(self.active_executions.keys()):
            task = asyncio.create_task(self.cancel_execution(execution_id))
            cancellation_tasks.append(task)

        if cancellation_tasks:
            await asyncio.gather(*cancellation_tasks, return_exceptions=True)

        # Shutdown order manager
        await self.order_manager.shutdown()

        self.logger.info("Execution engine stopped")

    @time_execution
    @log_calls
    async def execute_order(
        self,
        instruction: ExecutionInstruction,
        exchange_factory,
        risk_manager=None,
        market_data: MarketData | None = None,
    ) -> ExecutionResult:
        """
        Execute an order using optimal algorithm selection and comprehensive monitoring.

        Args:
            instruction: Execution instruction with order and parameters
            exchange_factory: Factory for exchange access
            risk_manager: Optional risk manager for validation
            market_data: Optional current market data

        Returns:
            ExecutionResult: Comprehensive execution result with metrics

        Raises:
            ExecutionError: If execution fails
            ValidationError: If instruction is invalid
        """
        try:
            if not self.is_running:
                raise ExecutionError("Execution engine is not running")

            # Validate instruction
            await self._validate_execution_instruction(instruction)

            # Get market data if not provided
            if not market_data:
                market_data = await self._get_market_data(instruction, exchange_factory)

            self.logger.info(
                "Starting order execution",
                symbol=instruction.order.symbol,
                quantity=float(instruction.order.quantity),
                algorithm=instruction.algorithm.value,
            )

            # Pre-trade analysis
            pre_trade_analysis = await self._perform_pre_trade_analysis(instruction, market_data)

            # Select optimal algorithm if not specified or validate specified algorithm
            optimal_algorithm = await self._select_optimal_algorithm(
                instruction, market_data, pre_trade_analysis
            )

            # Execute with selected algorithm
            execution_result = await self._execute_with_algorithm(
                instruction, optimal_algorithm, exchange_factory, risk_manager
            )

            # Post-trade analysis
            await self._perform_post_trade_analysis(
                execution_result, market_data, pre_trade_analysis
            )

            # Update statistics
            await self._update_execution_statistics(execution_result)

            self.logger.info(
                "Order execution completed",
                execution_id=execution_result.execution_id,
                status=execution_result.status.value,
                algorithm=execution_result.algorithm.value,
                filled_quantity=float(execution_result.total_filled_quantity),
            )

            return execution_result

        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")

            # Update failure statistics
            self.execution_statistics["failed_executions"] += 1
            self.execution_statistics["total_executions"] += 1

            raise ExecutionError(f"Order execution failed: {e}")

    async def _validate_execution_instruction(self, instruction: ExecutionInstruction) -> None:
        """Validate execution instruction."""
        if not instruction or not instruction.order:
            raise ValidationError("Invalid execution instruction")

        if not instruction.order.symbol:
            raise ValidationError("Order symbol is required")

        if instruction.order.quantity <= 0:
            raise ValidationError("Order quantity must be positive")

        # Validate algorithm is supported
        if instruction.algorithm not in self.algorithms:
            raise ValidationError(f"Unsupported algorithm: {instruction.algorithm.value}")

    async def _get_market_data(
        self, instruction: ExecutionInstruction, exchange_factory
    ) -> MarketData:
        """Get current market data for the order."""
        try:
            # Get exchange (prefer first preferred exchange or default)
            exchange_name = "binance"
            if instruction.preferred_exchanges:
                exchange_name = instruction.preferred_exchanges[0]

            exchange = await exchange_factory.get_exchange(exchange_name)
            if not exchange:
                raise ExecutionError(f"Failed to get exchange: {exchange_name}")

            # Get market data
            market_data = await exchange.get_market_data(instruction.order.symbol)
            if not market_data:
                raise ExecutionError(f"Failed to get market data for {instruction.order.symbol}")

            return market_data

        except Exception as e:
            self.logger.error(f"Market data retrieval failed: {e}")
            raise ExecutionError(f"Market data retrieval failed: {e}")

    async def _perform_pre_trade_analysis(
        self, instruction: ExecutionInstruction, market_data: MarketData
    ) -> dict[str, Any]:
        """Perform pre-trade analysis including slippage prediction."""
        try:
            # Predict slippage
            slippage_prediction = await self.slippage_model.predict_slippage(
                instruction.order,
                market_data,
                instruction.participation_rate,
                instruction.time_horizon_minutes,
            )

            # Calculate order value
            order_value = instruction.order.quantity * market_data.price

            # Assess market conditions
            market_conditions = {
                "volatility": await self._estimate_volatility(market_data),
                "liquidity": await self._estimate_liquidity(market_data),
                "spread_bps": await self._calculate_spread_bps(market_data),
                "volume_ratio": float(instruction.order.quantity / max(market_data.volume, 1)),
            }

            return {
                "slippage_prediction": slippage_prediction,
                "order_value": float(order_value),
                "market_conditions": market_conditions,
                "analysis_timestamp": datetime.now(timezone.utc),
            }

        except Exception as e:
            self.logger.warning(f"Pre-trade analysis failed: {e}")
            # Return minimal analysis on failure
            return {"error": str(e), "analysis_timestamp": datetime.now(timezone.utc)}

    async def _select_optimal_algorithm(
        self,
        instruction: ExecutionInstruction,
        market_data: MarketData,
        pre_trade_analysis: dict[str, Any],
    ) -> BaseAlgorithm:
        """Select optimal execution algorithm based on order characteristics."""
        try:
            # If algorithm is explicitly specified, validate and use it
            if instruction.algorithm != ExecutionAlgorithm.SMART_ROUTER:
                algorithm = self.algorithms[instruction.algorithm]

                # Validate the specified algorithm is appropriate
                if await algorithm.validate_instruction(instruction):
                    self.logger.info(f"Using specified algorithm: {instruction.algorithm.value}")
                    return algorithm
                else:
                    self.logger.warning(
                        f"Specified algorithm {instruction.algorithm.value} validation failed, "
                        "falling back to algorithm selection"
                    )

            # Algorithm selection logic
            order_value = Decimal(str(pre_trade_analysis.get("order_value", 0)))
            market_conditions = pre_trade_analysis.get("market_conditions", {})

            # Urgency-based selection
            if instruction.is_urgent:
                self.logger.info("Using SMART_ROUTER for urgent execution")
                return self.algorithms[ExecutionAlgorithm.SMART_ROUTER]

            # Large order stealth execution
            volume_ratio = market_conditions.get("volume_ratio", 0)
            if volume_ratio > self.algorithm_selection_rules["stealth_volume_ratio"]:
                self.logger.info("Using ICEBERG for stealth execution (large volume ratio)")
                return self.algorithms[ExecutionAlgorithm.ICEBERG]

            # Multi-exchange routing for very large orders
            if order_value >= self.algorithm_selection_rules["multi_exchange_threshold"]:
                self.logger.info("Using SMART_ROUTER for multi-exchange execution")
                return self.algorithms[ExecutionAlgorithm.SMART_ROUTER]

            # Time-sensitive execution
            if (
                instruction.time_horizon_minutes
                and instruction.time_horizon_minutes
                <= self.algorithm_selection_rules["urgency_threshold_minutes"]
            ):
                self.logger.info("Using TWAP for time-sensitive execution")
                return self.algorithms[ExecutionAlgorithm.TWAP]

            # Volume-based execution for normal orders
            if order_value >= self.algorithm_selection_rules["large_order_threshold"]:
                self.logger.info("Using VWAP for volume-based execution")
                return self.algorithms[ExecutionAlgorithm.VWAP]

            # Default to TWAP for smaller orders
            self.logger.info("Using TWAP as default algorithm")
            return self.algorithms[ExecutionAlgorithm.TWAP]

        except Exception as e:
            self.logger.error(f"Algorithm selection failed: {e}")
            # Fallback to TWAP
            return self.algorithms[ExecutionAlgorithm.TWAP]

    async def _execute_with_algorithm(
        self,
        instruction: ExecutionInstruction,
        algorithm: BaseAlgorithm,
        exchange_factory,
        risk_manager,
    ) -> ExecutionResult:
        """Execute order with selected algorithm."""
        try:
            # Update instruction with selected algorithm
            instruction.algorithm = algorithm.get_algorithm_type()

            # Track active execution
            execution_result = await algorithm.execute(instruction, exchange_factory, risk_manager)

            # Register active execution
            self.active_executions[execution_result.execution_id] = execution_result

            # Update algorithm usage statistics
            self.execution_statistics["algorithm_usage"][instruction.algorithm.value] += 1

            return execution_result

        except Exception as e:
            self.logger.error(f"Algorithm execution failed: {e}")
            raise
        finally:
            # Clean up active execution tracking
            if "execution_result" in locals():
                execution_id = execution_result.execution_id
                if execution_id in self.active_executions:
                    del self.active_executions[execution_id]

    async def _perform_post_trade_analysis(
        self,
        execution_result: ExecutionResult,
        market_data: MarketData,
        pre_trade_analysis: dict[str, Any],
    ) -> None:
        """Perform post-trade cost analysis."""
        try:
            # Get completion market data (simplified - would get actual end-of-execution data)
            completion_market_data = market_data  # Placeholder

            # Perform TCA analysis
            tca_analysis = await self.cost_analyzer.analyze_execution(
                execution_result, market_data, completion_market_data
            )

            # Update slippage model with actual results
            if "slippage_prediction" in pre_trade_analysis:
                pre_trade_analysis["slippage_prediction"]

                # Create actual slippage metrics for model learning
                actual_slippage = SlippageMetrics(
                    symbol=execution_result.original_order.symbol,
                    order_size=execution_result.total_filled_quantity,
                    market_price=market_data.price,
                    execution_price=execution_result.average_fill_price or market_data.price,
                    price_slippage_bps=(
                        abs(execution_result.price_slippage) * Decimal("10000") / market_data.price
                        if execution_result.price_slippage
                        else Decimal("0")
                    ),
                    market_impact_bps=(
                        abs(execution_result.market_impact) * Decimal("10000") / market_data.price
                        if execution_result.market_impact
                        else Decimal("0")
                    ),
                    timing_cost_bps=Decimal("0"),  # Simplified
                    total_cost_bps=tca_analysis.get("tca_metrics", {}).get("total_cost_bps", 0),
                    spread_bps=await self._calculate_spread_bps(market_data),
                    volume_ratio=float(
                        execution_result.total_filled_quantity / max(market_data.volume, 1)
                    ),
                    volatility=await self._estimate_volatility(market_data),
                    timestamp=datetime.now(timezone.utc),
                )

                # Update slippage model
                await self.slippage_model.update_historical_data(
                    execution_result.original_order.symbol,
                    actual_slippage,
                    pre_trade_analysis.get("market_conditions", {}),
                )

            # Store analysis in execution result metadata
            execution_result.metadata["post_trade_analysis"] = tca_analysis
            execution_result.metadata["pre_trade_analysis"] = pre_trade_analysis

        except Exception as e:
            self.logger.warning(f"Post-trade analysis failed: {e}")

    async def _update_execution_statistics(self, execution_result: ExecutionResult) -> None:
        """Update execution engine statistics."""
        try:
            self.execution_statistics["total_executions"] += 1

            if execution_result.status == ExecutionStatus.COMPLETED:
                self.execution_statistics["successful_executions"] += 1
            else:
                self.execution_statistics["failed_executions"] += 1

            # Update volume statistics
            self.execution_statistics["total_volume"] += execution_result.total_filled_quantity

            # Update execution time statistics
            if execution_result.execution_duration:
                total_executions = self.execution_statistics["total_executions"]
                current_avg = self.execution_statistics["average_execution_time_seconds"]
                new_avg = (
                    (current_avg * (total_executions - 1)) + execution_result.execution_duration
                ) / total_executions
                self.execution_statistics["average_execution_time_seconds"] = new_avg

            # Update cost statistics if available
            post_trade = execution_result.metadata.get("post_trade_analysis", {})
            if "tca_metrics" in post_trade:
                total_cost_bps = post_trade["tca_metrics"].get("total_cost_bps", 0)

                # Running average of costs
                total_executions = self.execution_statistics["total_executions"]
                current_avg_cost = self.execution_statistics["total_cost_bps"]
                new_avg_cost = (
                    (current_avg_cost * (total_executions - 1)) + Decimal(str(total_cost_bps))
                ) / total_executions
                self.execution_statistics["total_cost_bps"] = new_avg_cost

        except Exception as e:
            self.logger.warning(f"Statistics update failed: {e}")

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        try:
            if execution_id not in self.active_executions:
                self.logger.warning(f"Execution not found: {execution_id}")
                return False

            execution_result = self.active_executions[execution_id]
            algorithm = self.algorithms.get(execution_result.algorithm)

            if algorithm:
                success = await algorithm.cancel_execution(execution_id)
                self.logger.info(
                    f"Execution cancellation {'successful' if success else 'failed'}: {execution_id}"
                )
                return success
            else:
                self.logger.error(f"Algorithm not found for execution: {execution_id}")
                return False

        except Exception as e:
            self.logger.error(f"Execution cancellation failed: {e}")
            return False

    async def get_execution_status(self, execution_id: str) -> ExecutionStatus | None:
        """Get status of an execution."""
        if execution_id in self.active_executions:
            return self.active_executions[execution_id].status

        # Check with algorithms
        for algorithm in self.algorithms.values():
            status = await algorithm.get_execution_status(execution_id)
            if status:
                return status

        return None

    async def _estimate_volatility(self, market_data: MarketData) -> float:
        """Estimate market volatility from market data."""
        try:
            if market_data.high_price and market_data.low_price and market_data.price:
                daily_range = float(market_data.high_price - market_data.low_price)
                volatility = daily_range / float(market_data.price)
                return volatility
            return 0.02  # 2% default
        except Exception:
            return 0.02

    async def _estimate_liquidity(self, market_data: MarketData) -> float:
        """Estimate market liquidity from market data."""
        try:
            # Simple liquidity estimate based on volume
            volume_score = min(1.0, float(market_data.volume) / 1000000)  # Normalize by 1M
            return volume_score
        except Exception:
            return 0.5  # Default medium liquidity

    async def _calculate_spread_bps(self, market_data: MarketData) -> Decimal:
        """Calculate bid-ask spread in basis points."""
        try:
            if market_data.bid and market_data.ask and market_data.price:
                spread = market_data.ask - market_data.bid
                spread_bps = (spread / market_data.price) * Decimal("10000")
                return spread_bps
            return Decimal("20")  # Default 20 bps
        except Exception:
            return Decimal("20")

    @log_calls
    async def get_engine_summary(self) -> dict[str, Any]:
        """Get comprehensive execution engine summary."""
        try:
            # Get component summaries
            order_manager_summary = await self.order_manager.get_order_manager_summary()
            slippage_model_summary = await self.slippage_model.get_model_summary()
            cost_analyzer_summary = await self.cost_analyzer.get_performance_report()

            # Get algorithm summaries
            algorithm_summaries = {}
            for alg_type, algorithm in self.algorithms.items():
                algorithm_summaries[alg_type.value] = await algorithm.get_algorithm_summary()

            # Calculate performance metrics
            total_executions = self.execution_statistics["total_executions"]
            success_rate = 0.0
            if total_executions > 0:
                success_rate = self.execution_statistics["successful_executions"] / total_executions

            return {
                "engine_status": {
                    "is_running": self.is_running,
                    "active_executions": len(self.active_executions),
                    "supported_algorithms": list(self.algorithms.keys()),
                },
                "performance_statistics": {
                    "total_executions": total_executions,
                    "success_rate": success_rate,
                    "total_volume": float(self.execution_statistics["total_volume"]),
                    "average_cost_bps": float(self.execution_statistics["total_cost_bps"]),
                    "average_execution_time_seconds": self.execution_statistics[
                        "average_execution_time_seconds"
                    ],
                    "algorithm_usage": self.execution_statistics["algorithm_usage"],
                },
                "component_summaries": {
                    "order_manager": order_manager_summary,
                    "slippage_model": slippage_model_summary,
                    "cost_analyzer": cost_analyzer_summary,
                    "algorithms": algorithm_summaries,
                },
                "configuration": {"algorithm_selection_rules": self.algorithm_selection_rules},
            }

        except Exception as e:
            self.logger.error(f"Engine summary generation failed: {e}")
            return {"error": str(e)}

    async def optimize_execution_parameters(
        self, symbol: str, lookback_days: int = 30
    ) -> dict[str, Any]:
        """
        Analyze historical execution data to provide optimization recommendations.

        Args:
            symbol: Trading symbol to analyze
            lookback_days: Days of historical data to analyze

        Returns:
            dict: Optimization recommendations
        """
        try:
            # Get historical performance data
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=lookback_days)

            performance_report = await self.cost_analyzer.get_performance_report(
                symbol, start_date, end_date
            )

            recommendations = {
                "symbol": symbol,
                "analysis_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": lookback_days,
                },
                "current_performance": performance_report,
                "recommendations": [],
            }

            # Analyze cost patterns and generate recommendations
            if "cost_statistics" in performance_report:
                cost_stats = performance_report["cost_statistics"]
                avg_cost = cost_stats.get("average_cost_bps", 0)

                if avg_cost > 30:  # High cost threshold
                    recommendations["recommendations"].append(
                        {
                            "area": "cost_reduction",
                            "priority": "high",
                            "suggestion": "Consider using more passive execution strategies",
                            "expected_improvement": "20-30% cost reduction",
                        }
                    )

                if avg_cost > 50:  # Very high cost
                    recommendations["recommendations"].append(
                        {
                            "area": "algorithm_selection",
                            "priority": "critical",
                            "suggestion": "Review algorithm selection criteria for large orders",
                            "expected_improvement": "30-50% cost reduction",
                        }
                    )

            # Add algorithm-specific recommendations
            algorithm_usage = self.execution_statistics["algorithm_usage"]
            most_used_algorithm = max(algorithm_usage, key=algorithm_usage.get)

            recommendations["recommendations"].append(
                {
                    "area": "algorithm_diversification",
                    "priority": "medium",
                    "suggestion": f"Currently using {most_used_algorithm} most frequently. Consider diversifying algorithm usage based on market conditions.",
                    "expected_improvement": "5-15% performance improvement",
                }
            )

            return recommendations

        except Exception as e:
            self.logger.error(f"Execution optimization analysis failed: {e}")
            return {"error": str(e)}
