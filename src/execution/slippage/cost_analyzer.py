"""
Transaction Cost Analysis (TCA) for execution performance measurement.

This module provides comprehensive transaction cost analysis capabilities
to measure and analyze execution quality, identifying sources of cost
and opportunities for improvement in trading execution.

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import numpy as np

from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import (
    ExecutionResult,
    MarketData,
    OrderResponse,
    OrderSide,
    SlippageMetrics,
    SlippageType,
)

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007A
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class CostAnalyzer:
    """
    Advanced Transaction Cost Analysis (TCA) engine.
    
    This analyzer provides comprehensive measurement and analysis of
    execution costs, including:
    - Implementation shortfall analysis
    - Benchmark comparison (VWAP, TWAP, arrival price)
    - Cost component breakdown
    - Execution quality scoring
    - Performance attribution analysis
    - Statistical analysis and reporting
    
    The analyzer helps traders understand execution performance and
    identify areas for improvement in their execution strategies.
    """

    def __init__(self, config: Config):
        """
        Initialize transaction cost analyzer.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.error_handler = ErrorHandler(config.error_handling)
        
        # TCA configuration
        self.benchmark_lookback_hours = 24  # Look back 24 hours for benchmarks
        self.cost_component_weights = {
            "market_impact": 0.4,
            "timing_cost": 0.3,
            "spread_cost": 0.2,
            "opportunity_cost": 0.1
        }
        
        # Performance thresholds
        self.excellent_threshold_bps = 5    # < 5 bps excellent
        self.good_threshold_bps = 15        # < 15 bps good
        self.acceptable_threshold_bps = 30  # < 30 bps acceptable
        self.poor_threshold_above_bps = 30  # > 30 bps poor
        
        # Analysis storage
        self.execution_analyses = {}  # Store historical analyses
        self.benchmark_cache = {}     # Cache benchmark calculations
        self.performance_statistics = {}  # Aggregate performance stats
        
        # Statistical parameters
        self.confidence_levels = [0.9, 0.95, 0.99]
        self.min_samples_for_stats = 10
        self.outlier_threshold_std = 3.0  # 3 standard deviations
        
        self.logger.info("Transaction cost analyzer initialized")

    @time_execution
    async def analyze_execution(
        self,
        execution_result: ExecutionResult,
        market_data_at_decision: MarketData,
        market_data_at_completion: MarketData | None = None,
        benchmarks: dict[str, Decimal] | None = None
    ) -> dict[str, Any]:
        """
        Perform comprehensive transaction cost analysis on an execution.
        
        Args:
            execution_result: Completed execution result
            market_data_at_decision: Market data when decision was made
            market_data_at_completion: Market data at execution completion
            benchmarks: Optional benchmark prices for comparison
            
        Returns:
            dict: Comprehensive TCA analysis results
            
        Raises:
            ValidationError: If inputs are invalid
            ExecutionError: If analysis fails
        """
        try:
            # Validate inputs
            if not execution_result or not market_data_at_decision:
                raise ValidationError("Execution result and decision market data required")
                
            if execution_result.total_filled_quantity <= 0:
                raise ValidationError("No filled quantity to analyze")
                
            if not execution_result.average_fill_price:
                raise ValidationError("Average fill price required for analysis")
                
            self.logger.info(
                "Starting TCA analysis",
                execution_id=execution_result.execution_id,
                symbol=execution_result.original_order.symbol,
                filled_quantity=float(execution_result.total_filled_quantity)
            )
            
            # Calculate core TCA metrics
            tca_metrics = await self._calculate_core_tca_metrics(
                execution_result, market_data_at_decision, market_data_at_completion
            )
            
            # Calculate benchmark comparisons
            benchmark_analysis = await self._calculate_benchmark_comparisons(
                execution_result, market_data_at_decision, benchmarks
            )
            
            # Analyze cost components
            cost_breakdown = await self._analyze_cost_components(
                execution_result, market_data_at_decision, tca_metrics
            )
            
            # Calculate execution quality score
            quality_score = await self._calculate_execution_quality_score(
                tca_metrics, cost_breakdown
            )
            
            # Generate performance attribution
            attribution = await self._generate_performance_attribution(
                execution_result, tca_metrics, cost_breakdown
            )
            
            # Compile comprehensive analysis
            analysis = {
                "execution_id": execution_result.execution_id,
                "symbol": execution_result.original_order.symbol,
                "analysis_timestamp": datetime.now(timezone.utc),
                
                # Core metrics
                "tca_metrics": tca_metrics,
                "benchmark_analysis": benchmark_analysis,
                "cost_breakdown": cost_breakdown,
                "quality_score": quality_score,
                "attribution": attribution,
                
                # Execution details
                "execution_summary": {
                    "algorithm": execution_result.algorithm.value,
                    "total_quantity": float(execution_result.original_order.quantity),
                    "filled_quantity": float(execution_result.total_filled_quantity),
                    "fill_rate": float(execution_result.total_filled_quantity / execution_result.original_order.quantity),
                    "number_of_trades": execution_result.number_of_trades,
                    "execution_duration_seconds": execution_result.execution_duration,
                    "average_fill_price": float(execution_result.average_fill_price)
                },
                
                # Market conditions
                "market_conditions": {
                    "decision_price": float(market_data_at_decision.price),
                    "decision_spread_bps": await self._calculate_spread_bps(market_data_at_decision),
                    "volatility_estimate": await self._estimate_volatility(market_data_at_decision),
                    "volume": float(market_data_at_decision.volume)
                }
            }
            
            # Store analysis for historical tracking
            symbol = execution_result.original_order.symbol
            if symbol not in self.execution_analyses:
                self.execution_analyses[symbol] = []
            self.execution_analyses[symbol].append(analysis)
            
            # Update performance statistics
            await self._update_performance_statistics(symbol, analysis)
            
            self.logger.info(
                "TCA analysis completed",
                execution_id=execution_result.execution_id,
                implementation_shortfall_bps=tca_metrics["implementation_shortfall_bps"],
                quality_score=quality_score["overall_score"],
                quality_rating=quality_score["rating"]
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"TCA analysis failed: {e}")
            raise ExecutionError(f"Transaction cost analysis failed: {e}")

    async def _calculate_core_tca_metrics(
        self,
        execution_result: ExecutionResult,
        market_data_at_decision: MarketData,
        market_data_at_completion: MarketData | None
    ) -> dict[str, Any]:
        """
        Calculate core transaction cost analysis metrics.
        
        Args:
            execution_result: Execution result
            market_data_at_decision: Market data at decision time
            market_data_at_completion: Market data at completion
            
        Returns:
            dict: Core TCA metrics
        """
        try:
            # Decision price (price when trade decision was made)
            decision_price = market_data_at_decision.price
            
            # Arrival price (price when execution started)
            arrival_price = decision_price  # Simplified assumption
            
            # Average execution price
            execution_price = execution_result.average_fill_price
            
            # Completion price (price when execution finished)
            if market_data_at_completion:
                completion_price = market_data_at_completion.price
            else:
                completion_price = execution_price  # Fallback
                
            # Calculate implementation shortfall
            if execution_result.original_order.side == OrderSide.BUY:
                # For buy orders: positive shortfall means paid more than decision price
                implementation_shortfall = execution_price - decision_price
                slippage_direction = 1  # Positive slippage is adverse for buys
            else:
                # For sell orders: positive shortfall means received less than decision price
                implementation_shortfall = decision_price - execution_price
                slippage_direction = 1  # Positive slippage is adverse for sells
                
            # Convert to basis points
            implementation_shortfall_bps = (implementation_shortfall / decision_price) * Decimal("10000")
            
            # Calculate arrival price slippage
            if execution_result.original_order.side == OrderSide.BUY:
                arrival_slippage = execution_price - arrival_price
            else:
                arrival_slippage = arrival_price - execution_price
                
            arrival_slippage_bps = (arrival_slippage / arrival_price) * Decimal("10000")
            
            # Calculate market impact (pre-trade vs post-trade price movement)
            market_impact = completion_price - arrival_price
            if execution_result.original_order.side == OrderSide.SELL:
                market_impact = -market_impact  # Reverse for sell orders
                
            market_impact_bps = (market_impact / arrival_price) * Decimal("10000")
            
            # Calculate timing cost (decision delay cost)
            timing_cost = arrival_price - decision_price
            if execution_result.original_order.side == OrderSide.SELL:
                timing_cost = -timing_cost
                
            timing_cost_bps = (timing_cost / decision_price) * Decimal("10000")
            
            # Calculate opportunity cost (for unfilled portion)
            unfilled_quantity = execution_result.original_order.quantity - execution_result.total_filled_quantity
            if unfilled_quantity > 0:
                opportunity_cost = (completion_price - decision_price) * unfilled_quantity
                if execution_result.original_order.side == OrderSide.SELL:
                    opportunity_cost = -opportunity_cost
            else:
                opportunity_cost = Decimal("0")
                
            opportunity_cost_bps = Decimal("0")
            if execution_result.original_order.quantity > 0:
                opportunity_cost_bps = (opportunity_cost / (decision_price * execution_result.original_order.quantity)) * Decimal("10000")
            
            return {
                "decision_price": float(decision_price),
                "arrival_price": float(arrival_price),
                "execution_price": float(execution_price),
                "completion_price": float(completion_price),
                "implementation_shortfall": float(implementation_shortfall),
                "implementation_shortfall_bps": float(implementation_shortfall_bps),
                "arrival_slippage_bps": float(arrival_slippage_bps),
                "market_impact_bps": float(market_impact_bps),
                "timing_cost_bps": float(timing_cost_bps),
                "opportunity_cost_bps": float(opportunity_cost_bps),
                "total_cost_bps": float(
                    arrival_slippage_bps + timing_cost_bps + opportunity_cost_bps
                )
            }
            
        except Exception as e:
            self.logger.error(f"Core TCA metrics calculation failed: {e}")
            return {}

    async def _calculate_benchmark_comparisons(
        self,
        execution_result: ExecutionResult,
        market_data_at_decision: MarketData,
        benchmarks: dict[str, Decimal] | None
    ) -> dict[str, Any]:
        """
        Calculate performance relative to various benchmarks.
        
        Args:
            execution_result: Execution result
            market_data_at_decision: Market data at decision
            benchmarks: Optional benchmark prices
            
        Returns:
            dict: Benchmark comparison results
        """
        try:
            execution_price = execution_result.average_fill_price
            side = execution_result.original_order.side
            
            comparisons = {}
            
            # Arrival price benchmark
            arrival_price = market_data_at_decision.price
            if side == OrderSide.BUY:
                arrival_performance_bps = ((execution_price - arrival_price) / arrival_price) * Decimal("10000")
            else:
                arrival_performance_bps = ((arrival_price - execution_price) / arrival_price) * Decimal("10000")
                
            comparisons["arrival_price"] = {
                "benchmark_price": float(arrival_price),
                "performance_bps": float(arrival_performance_bps),
                "description": "Performance vs arrival price"
            }
            
            # Mid-price benchmark (if bid/ask available)
            if market_data_at_decision.bid and market_data_at_decision.ask:
                mid_price = (market_data_at_decision.bid + market_data_at_decision.ask) / Decimal("2")
                if side == OrderSide.BUY:
                    mid_performance_bps = ((execution_price - mid_price) / mid_price) * Decimal("10000")
                else:
                    mid_performance_bps = ((mid_price - execution_price) / mid_price) * Decimal("10000")
                    
                comparisons["mid_price"] = {
                    "benchmark_price": float(mid_price),
                    "performance_bps": float(mid_performance_bps),
                    "description": "Performance vs mid price"
                }
            
            # Add custom benchmarks if provided
            if benchmarks:
                for bench_name, bench_price in benchmarks.items():
                    if side == OrderSide.BUY:
                        bench_performance_bps = ((execution_price - bench_price) / bench_price) * Decimal("10000")
                    else:
                        bench_performance_bps = ((bench_price - execution_price) / bench_price) * Decimal("10000")
                        
                    comparisons[bench_name] = {
                        "benchmark_price": float(bench_price),
                        "performance_bps": float(bench_performance_bps),
                        "description": f"Performance vs {bench_name}"
                    }
            
            return comparisons
            
        except Exception as e:
            self.logger.error(f"Benchmark comparison calculation failed: {e}")
            return {}

    async def _analyze_cost_components(
        self,
        execution_result: ExecutionResult,
        market_data_at_decision: MarketData,
        tca_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Analyze breakdown of execution costs by component.
        
        Args:
            execution_result: Execution result
            market_data_at_decision: Market data at decision
            tca_metrics: Core TCA metrics
            
        Returns:
            dict: Cost component breakdown
        """
        try:
            # Extract cost components from TCA metrics
            timing_cost_bps = Decimal(str(abs(tca_metrics.get("timing_cost_bps", 0))))
            market_impact_bps = Decimal(str(abs(tca_metrics.get("market_impact_bps", 0))))
            opportunity_cost_bps = Decimal(str(abs(tca_metrics.get("opportunity_cost_bps", 0))))
            
            # Calculate spread cost
            spread_cost_bps = await self._calculate_spread_cost_component(
                execution_result, market_data_at_decision
            )
            
            # Calculate total explicit costs (fees, commissions)
            explicit_cost_bps = await self._calculate_explicit_costs(execution_result)
            
            # Calculate total costs
            total_implicit_cost_bps = (
                timing_cost_bps + market_impact_bps + 
                spread_cost_bps + opportunity_cost_bps
            )
            total_cost_bps = total_implicit_cost_bps + explicit_cost_bps
            
            # Calculate component percentages
            component_breakdown = {}
            if total_cost_bps > 0:
                component_breakdown = {
                    "timing_cost_pct": float((timing_cost_bps / total_cost_bps) * 100),
                    "market_impact_pct": float((market_impact_bps / total_cost_bps) * 100),
                    "spread_cost_pct": float((spread_cost_bps / total_cost_bps) * 100),
                    "opportunity_cost_pct": float((opportunity_cost_bps / total_cost_bps) * 100),
                    "explicit_cost_pct": float((explicit_cost_bps / total_cost_bps) * 100)
                }
            
            return {
                "cost_components_bps": {
                    "timing_cost": float(timing_cost_bps),
                    "market_impact": float(market_impact_bps),
                    "spread_cost": float(spread_cost_bps),
                    "opportunity_cost": float(opportunity_cost_bps),
                    "explicit_cost": float(explicit_cost_bps)
                },
                "total_implicit_cost_bps": float(total_implicit_cost_bps),
                "total_explicit_cost_bps": float(explicit_cost_bps),
                "total_cost_bps": float(total_cost_bps),
                "component_percentages": component_breakdown,
                "dominant_cost_component": max(
                    component_breakdown.items(), 
                    key=lambda x: x[1], 
                    default=("unknown", 0)
                )[0].replace("_pct", "") if component_breakdown else "unknown"
            }
            
        except Exception as e:
            self.logger.error(f"Cost component analysis failed: {e}")
            return {}

    async def _calculate_spread_cost_component(
        self,
        execution_result: ExecutionResult,
        market_data: MarketData
    ) -> Decimal:
        """Calculate the spread cost component."""
        try:
            if not market_data.bid or not market_data.ask:
                return Decimal("10")  # Default 10 bps
                
            spread = market_data.ask - market_data.bid
            spread_bps = (spread / market_data.price) * Decimal("10000")
            
            # Typically pay about half the spread
            spread_cost_bps = spread_bps * Decimal("0.5")
            
            return spread_cost_bps
            
        except Exception:
            return Decimal("10")  # Default

    async def _calculate_explicit_costs(self, execution_result: ExecutionResult) -> Decimal:
        """Calculate explicit costs (fees, commissions)."""
        try:
            # Use the total fees from execution result if available
            if execution_result.total_fees > 0:
                # Convert fees to basis points of trade value
                trade_value = execution_result.total_filled_quantity * execution_result.average_fill_price
                if trade_value > 0:
                    explicit_cost_bps = (execution_result.total_fees / trade_value) * Decimal("10000")
                    return explicit_cost_bps
                    
            # Default fee estimate
            return Decimal("8")  # 8 bps default fee estimate
            
        except Exception:
            return Decimal("8")  # Default

    async def _calculate_execution_quality_score(
        self,
        tca_metrics: dict[str, Any],
        cost_breakdown: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate overall execution quality score.
        
        Args:
            tca_metrics: Core TCA metrics
            cost_breakdown: Cost component breakdown
            
        Returns:
            dict: Quality score and rating
        """
        try:
            total_cost_bps = cost_breakdown.get("total_cost_bps", 0)
            
            # Determine quality rating based on total cost
            if total_cost_bps <= self.excellent_threshold_bps:
                rating = "excellent"
                score = 90 + (10 * (self.excellent_threshold_bps - total_cost_bps) / self.excellent_threshold_bps)
            elif total_cost_bps <= self.good_threshold_bps:
                rating = "good"
                score = 70 + (20 * (self.good_threshold_bps - total_cost_bps) / (self.good_threshold_bps - self.excellent_threshold_bps))
            elif total_cost_bps <= self.acceptable_threshold_bps:
                rating = "acceptable"
                score = 50 + (20 * (self.acceptable_threshold_bps - total_cost_bps) / (self.acceptable_threshold_bps - self.good_threshold_bps))
            else:
                rating = "poor"
                score = max(0, 50 - (total_cost_bps - self.acceptable_threshold_bps))
                
            # Adjust score based on cost component balance
            component_pcts = cost_breakdown.get("component_percentages", {})
            if component_pcts:
                # Penalize if one component dominates (indicates imbalanced execution)
                max_component_pct = max(component_pcts.values())
                if max_component_pct > 70:  # If one component > 70%
                    score *= 0.9  # 10% penalty
                    
            score = max(0, min(100, score))  # Clamp between 0-100
            
            return {
                "overall_score": float(score),
                "rating": rating,
                "total_cost_bps": total_cost_bps,
                "cost_efficiency": "high" if total_cost_bps <= self.good_threshold_bps else "medium" if total_cost_bps <= self.acceptable_threshold_bps else "low"
            }
            
        except Exception as e:
            self.logger.error(f"Quality score calculation failed: {e}")
            return {
                "overall_score": 50.0,
                "rating": "unknown",
                "total_cost_bps": 0,
                "cost_efficiency": "unknown"
            }

    async def _generate_performance_attribution(
        self,
        execution_result: ExecutionResult,
        tca_metrics: dict[str, Any],
        cost_breakdown: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate performance attribution analysis.
        
        Args:
            execution_result: Execution result
            tca_metrics: TCA metrics
            cost_breakdown: Cost breakdown
            
        Returns:
            dict: Performance attribution analysis
        """
        try:
            attribution = {
                "algorithm_performance": {},
                "market_condition_impact": {},
                "execution_efficiency": {},
                "improvement_opportunities": []
            }
            
            # Algorithm performance attribution
            algorithm = execution_result.algorithm.value
            total_cost_bps = cost_breakdown.get("total_cost_bps", 0)
            
            attribution["algorithm_performance"] = {
                "algorithm_used": algorithm,
                "cost_effectiveness": "good" if total_cost_bps <= 20 else "average" if total_cost_bps <= 40 else "poor",
                "execution_speed": "fast" if execution_result.execution_duration and execution_result.execution_duration < 300 else "medium" if execution_result.execution_duration and execution_result.execution_duration < 1800 else "slow",
                "fill_rate": float(execution_result.total_filled_quantity / execution_result.original_order.quantity)
            }
            
            # Market condition impact
            market_impact_bps = abs(tca_metrics.get("market_impact_bps", 0))
            timing_cost_bps = abs(tca_metrics.get("timing_cost_bps", 0))
            
            attribution["market_condition_impact"] = {
                "market_impact_severity": "low" if market_impact_bps <= 10 else "medium" if market_impact_bps <= 25 else "high",
                "timing_impact": "favorable" if timing_cost_bps <= 5 else "neutral" if timing_cost_bps <= 15 else "adverse",
                "market_volatility_effect": "low"  # Simplified
            }
            
            # Execution efficiency
            number_of_trades = execution_result.number_of_trades
            quantity_per_trade = float(execution_result.total_filled_quantity) / max(number_of_trades, 1)
            
            attribution["execution_efficiency"] = {
                "trade_frequency": "high" if number_of_trades > 10 else "medium" if number_of_trades > 3 else "low",
                "average_trade_size": quantity_per_trade,
                "execution_consistency": "good"  # Simplified
            }
            
            # Improvement opportunities
            opportunities = []
            
            # Check dominant cost components for improvement suggestions
            dominant_component = cost_breakdown.get("dominant_cost_component", "")
            
            if dominant_component == "timing_cost" and timing_cost_bps > 15:
                opportunities.append({
                    "area": "timing",
                    "suggestion": "Consider faster execution to reduce timing costs",
                    "potential_savings_bps": timing_cost_bps * 0.3
                })
                
            if dominant_component == "market_impact" and market_impact_bps > 20:
                opportunities.append({
                    "area": "market_impact",
                    "suggestion": "Use more passive execution strategy to reduce market impact",
                    "potential_savings_bps": market_impact_bps * 0.4
                })
                
            if dominant_component == "spread_cost":
                opportunities.append({
                    "area": "spread_cost",
                    "suggestion": "Use limit orders or improve timing to reduce spread costs",
                    "potential_savings_bps": cost_breakdown.get("cost_components_bps", {}).get("spread_cost", 0) * 0.5
                })
                
            attribution["improvement_opportunities"] = opportunities
            
            return attribution
            
        except Exception as e:
            self.logger.error(f"Performance attribution generation failed: {e}")
            return {}

    async def _calculate_spread_bps(self, market_data: MarketData) -> float:
        """Calculate bid-ask spread in basis points."""
        try:
            if market_data.bid and market_data.ask and market_data.price:
                spread = market_data.ask - market_data.bid
                spread_bps = (spread / market_data.price) * 10000
                return float(spread_bps)
            return 20.0  # Default 20 bps
        except Exception:
            return 20.0

    async def _estimate_volatility(self, market_data: MarketData) -> float:
        """Estimate volatility from market data."""
        try:
            if market_data.high_price and market_data.low_price and market_data.price:
                daily_range = float(market_data.high_price - market_data.low_price)
                volatility = daily_range / float(market_data.price)
                return volatility
            return 0.02  # Default 2% volatility
        except Exception:
            return 0.02

    async def _update_performance_statistics(
        self,
        symbol: str,
        analysis: dict[str, Any]
    ) -> None:
        """Update aggregated performance statistics."""
        try:
            if symbol not in self.performance_statistics:
                self.performance_statistics[symbol] = {
                    "total_analyses": 0,
                    "average_cost_bps": 0.0,
                    "average_quality_score": 0.0,
                    "cost_history": [],
                    "last_updated": datetime.now(timezone.utc)
                }
                
            stats = self.performance_statistics[symbol]
            
            # Update counters
            stats["total_analyses"] += 1
            
            # Update running averages
            cost_bps = analysis.get("cost_breakdown", {}).get("total_cost_bps", 0)
            quality_score = analysis.get("quality_score", {}).get("overall_score", 0)
            
            # Simple running average
            n = stats["total_analyses"]
            stats["average_cost_bps"] = ((stats["average_cost_bps"] * (n - 1)) + cost_bps) / n
            stats["average_quality_score"] = ((stats["average_quality_score"] * (n - 1)) + quality_score) / n
            
            # Add to cost history (keep last 100)
            stats["cost_history"].append({
                "timestamp": analysis["analysis_timestamp"],
                "cost_bps": cost_bps,
                "quality_score": quality_score
            })
            
            if len(stats["cost_history"]) > 100:
                stats["cost_history"] = stats["cost_history"][-100:]
                
            stats["last_updated"] = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Performance statistics update failed: {e}")

    @log_calls
    async def get_performance_report(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            dict: Performance report
        """
        try:
            if symbol:
                symbols = [symbol] if symbol in self.execution_analyses else []
            else:
                symbols = list(self.execution_analyses.keys())
                
            if not symbols:
                return {"error": "No execution data available"}
                
            # Filter analyses by date if specified
            filtered_analyses = []
            for sym in symbols:
                for analysis in self.execution_analyses[sym]:
                    analysis_time = analysis["analysis_timestamp"]
                    if start_date and analysis_time < start_date:
                        continue
                    if end_date and analysis_time > end_date:
                        continue
                    filtered_analyses.append(analysis)
                    
            if not filtered_analyses:
                return {"error": "No analyses in specified date range"}
                
            # Calculate aggregate statistics
            costs = [a["cost_breakdown"]["total_cost_bps"] for a in filtered_analyses]
            quality_scores = [a["quality_score"]["overall_score"] for a in filtered_analyses]
            
            report = {
                "summary": {
                    "total_executions": len(filtered_analyses),
                    "symbols_analyzed": len(symbols),
                    "date_range": {
                        "start": start_date.isoformat() if start_date else "all",
                        "end": end_date.isoformat() if end_date else "all"
                    }
                },
                "cost_statistics": {
                    "average_cost_bps": float(np.mean(costs)),
                    "median_cost_bps": float(np.median(costs)),
                    "std_cost_bps": float(np.std(costs)),
                    "min_cost_bps": float(np.min(costs)),
                    "max_cost_bps": float(np.max(costs)),
                    "percentiles": {
                        "25th": float(np.percentile(costs, 25)),
                        "75th": float(np.percentile(costs, 75)),
                        "90th": float(np.percentile(costs, 90)),
                        "95th": float(np.percentile(costs, 95))
                    }
                },
                "quality_statistics": {
                    "average_quality_score": float(np.mean(quality_scores)),
                    "median_quality_score": float(np.median(quality_scores)),
                    "excellent_executions": len([s for s in quality_scores if s >= 90]),
                    "good_executions": len([s for s in quality_scores if 70 <= s < 90]),
                    "poor_executions": len([s for s in quality_scores if s < 50])
                },
                "performance_by_symbol": {}
            }
            
            # Add per-symbol breakdown
            for sym in symbols:
                if sym in self.performance_statistics:
                    report["performance_by_symbol"][sym] = self.performance_statistics[sym].copy()
                    
            return report
            
        except Exception as e:
            self.logger.error(f"Performance report generation failed: {e}")
            return {"error": str(e)}