"""
Quality Controller for trade validation and quality assessment (P-024).

This module provides comprehensive quality controls for trading operations including:
- Pre-trade validation and risk checks
- Real-time execution quality monitoring
- Post-trade analysis and quality scoring
- Performance attribution and benchmarking
- Quality trend analysis and alerting

The QualityController ensures high-quality trade execution and provides
detailed insights for continuous improvement of trading strategies.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from scipy import stats

# Core framework imports
from src.core.config import Config
from src.core.exceptions import ValidationError, StateError
from src.core.logging import get_logger
from src.core.types import OrderRequest, ExecutionResult, MarketData

# Database imports
from src.database.manager import DatabaseManager
from src.database.influxdb_client import InfluxDBClient

# Utility imports
from src.utils.decorators import retry, time_execution
from src.utils.validators import validate_order_data

logger = get_logger(__name__)


class QualityLevel(Enum):
    """Quality assessment levels."""
    
    EXCELLENT = "excellent"    # 90-100
    GOOD = "good"             # 70-89
    FAIR = "fair"             # 50-69
    POOR = "poor"             # 30-49
    CRITICAL = "critical"     # 0-29


class ValidationResult(Enum):
    """Validation result enumeration."""
    
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationCheck:
    """Individual validation check result."""
    
    check_name: str = ""
    result: ValidationResult = ValidationResult.PASSED
    score: float = 100.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "low"  # low, medium, high, critical


@dataclass
class PreTradeValidation:
    """Pre-trade validation results."""
    
    validation_id: str = field(default_factory=lambda: str(uuid4()))
    order_request: Optional[OrderRequest] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Overall results
    overall_result: ValidationResult = ValidationResult.PASSED
    overall_score: float = 100.0
    
    # Individual checks
    checks: List[ValidationCheck] = field(default_factory=list)
    
    # Risk assessment
    risk_level: str = "low"  # low, medium, high, critical
    risk_score: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Processing time
    validation_time_ms: float = 0.0


@dataclass
class PostTradeAnalysis:
    """Post-trade analysis results."""
    
    analysis_id: str = field(default_factory=lambda: str(uuid4()))
    trade_id: str = ""
    execution_result: Optional[ExecutionResult] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Quality scores
    execution_quality_score: float = 100.0
    timing_quality_score: float = 100.0
    price_quality_score: float = 100.0
    overall_quality_score: float = 100.0
    
    # Performance metrics
    slippage_bps: float = 0.0
    execution_time_seconds: float = 0.0
    fill_rate: float = 100.0
    
    # Market impact analysis
    market_impact_bps: float = 0.0
    temporary_impact_bps: float = 0.0
    permanent_impact_bps: float = 0.0
    
    # Benchmark comparison
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    
    # Issues and recommendations
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityTrend:
    """Quality trend analysis."""
    
    metric_name: str = ""
    time_period: str = "1d"  # 1h, 1d, 1w, 1m
    
    # Trend statistics
    current_value: float = 0.0
    previous_value: float = 0.0
    change_percentage: float = 0.0
    trend_direction: str = "stable"  # improving, declining, stable
    
    # Statistical measures
    mean: float = 0.0
    std_dev: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    percentile_95: float = 0.0
    percentile_5: float = 0.0
    
    # Alerts
    alert_triggered: bool = False
    alert_level: str = "none"  # none, warning, critical


class QualityController:
    """
    Comprehensive quality control system for trading operations.
    
    Features:
    - Multi-layer pre-trade validation (risk, market, technical)
    - Real-time execution quality monitoring
    - Advanced post-trade analysis with benchmarking
    - Quality trend analysis and alerting
    - Performance attribution and improvement recommendations
    """
    
    def __init__(self, config: Config):
        """
        Initialize the quality controller.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{id(self)}")
        
        # Database clients
        self.db_manager = DatabaseManager(config)
        self.influxdb_client = InfluxDBClient(config)
        
        # Quality configuration
        quality_config = config.quality_controls
        self.min_quality_score = quality_config.get("min_quality_score", 70.0)
        self.slippage_threshold_bps = quality_config.get("slippage_threshold_bps", 20.0)
        self.execution_time_threshold_seconds = quality_config.get("execution_time_threshold_seconds", 30.0)
        self.market_impact_threshold_bps = quality_config.get("market_impact_threshold_bps", 10.0)
        
        # Validation rules
        self.validation_rules = {
            "position_size": {"max_percentage": 10.0, "weight": 25.0},
            "market_hours": {"enforce": True, "weight": 15.0},
            "liquidity": {"min_volume_usd": 100000, "weight": 20.0},
            "volatility": {"max_daily_range": 0.05, "weight": 15.0},
            "correlation": {"max_correlation": 0.8, "weight": 10.0},
            "risk_limits": {"max_var": 0.02, "weight": 15.0}
        }
        
        # Benchmarks
        self.benchmarks = {
            "slippage_p50": 5.0,    # 50th percentile slippage in bps
            "slippage_p90": 15.0,   # 90th percentile slippage in bps
            "execution_time_p50": 10.0,  # 50th percentile execution time in seconds
            "execution_time_p90": 30.0,  # 90th percentile execution time in seconds
            "fill_rate_p50": 95.0,  # 50th percentile fill rate
            "market_impact_p50": 3.0,  # 50th percentile market impact in bps
        }
        
        # Quality history
        self.validation_history: List[PreTradeValidation] = []
        self.analysis_history: List[PostTradeAnalysis] = []
        
        # Performance tracking
        self.quality_metrics = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "average_quality_score": 0.0,
            "total_analyses": 0,
            "average_execution_quality": 0.0,
            "slippage_incidents": 0,
            "execution_time_violations": 0
        }
        
        self.logger.info("QualityController initialized")

    async def initialize(self) -> None:
        """Initialize the quality controller."""
        try:
            # Initialize database connections
            await self.db_manager.initialize()
            await self.influxdb_client.initialize()
            
            # Load historical benchmarks
            await self._load_benchmarks()
            
            # Start monitoring tasks
            asyncio.create_task(self._quality_monitoring_loop())
            asyncio.create_task(self._trend_analysis_loop())
            
            self.logger.info("QualityController initialization completed")
            
        except Exception as e:
            self.logger.error(f"QualityController initialization failed: {e}")
            raise StateError(f"Failed to initialize QualityController: {e}")

    @time_execution
    async def validate_pre_trade(
        self,
        order_request: OrderRequest,
        market_data: Optional[MarketData] = None,
        portfolio_context: Optional[Dict[str, Any]] = None
    ) -> PreTradeValidation:
        """
        Perform comprehensive pre-trade validation.
        
        Args:
            order_request: Order to validate
            market_data: Current market data
            portfolio_context: Portfolio context for risk assessment
            
        Returns:
            Validation results
            
        Raises:
            ValidationError: If validation fails critically
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            validation = PreTradeValidation(order_request=order_request)
            
            # Basic order validation
            await self._validate_order_structure(order_request, validation)
            
            # Market conditions validation
            if market_data:
                await self._validate_market_conditions(order_request, market_data, validation)
            
            # Portfolio risk validation
            if portfolio_context:
                await self._validate_portfolio_risk(order_request, portfolio_context, validation)
            
            # Liquidity validation
            await self._validate_liquidity(order_request, validation)
            
            # Timing validation
            await self._validate_timing(order_request, validation)
            
            # Correlation validation
            await self._validate_correlation(order_request, portfolio_context, validation)
            
            # Calculate overall results
            validation.overall_score = self._calculate_overall_score(validation.checks)
            validation.overall_result = self._determine_overall_result(validation.checks)
            validation.risk_level = self._assess_risk_level(validation.checks)
            validation.risk_score = self._calculate_risk_score(validation.checks)
            
            # Generate recommendations
            validation.recommendations = self._generate_recommendations(validation.checks)
            
            # Record processing time
            validation.validation_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            
            # Store validation
            self.validation_history.append(validation)
            await self._log_validation_metrics(validation)
            
            # Update quality metrics
            self._update_quality_metrics("validation", validation)
            
            self.logger.info(
                "Pre-trade validation completed",
                validation_id=validation.validation_id,
                overall_score=validation.overall_score,
                result=validation.overall_result.value,
                risk_level=validation.risk_level
            )
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Pre-trade validation failed: {e}")
            raise ValidationError(f"Pre-trade validation error: {e}")

    @time_execution
    async def analyze_post_trade(
        self,
        trade_id: str,
        execution_result: ExecutionResult,
        market_data_before: Optional[MarketData] = None,
        market_data_after: Optional[MarketData] = None
    ) -> PostTradeAnalysis:
        """
        Perform comprehensive post-trade analysis.
        
        Args:
            trade_id: Trade identifier
            execution_result: Execution results
            market_data_before: Market data before trade
            market_data_after: Market data after trade
            
        Returns:
            Post-trade analysis results
        """
        try:
            analysis = PostTradeAnalysis(
                trade_id=trade_id,
                execution_result=execution_result
            )
            
            # Execution quality analysis
            analysis.execution_quality_score = await self._analyze_execution_quality(execution_result)
            
            # Timing analysis
            analysis.timing_quality_score = await self._analyze_timing_quality(execution_result)
            
            # Price quality analysis
            analysis.price_quality_score = await self._analyze_price_quality(
                execution_result, market_data_before
            )
            
            # Calculate slippage
            analysis.slippage_bps = await self._calculate_slippage(
                execution_result, market_data_before
            )
            
            # Market impact analysis
            if market_data_before and market_data_after:
                impact_analysis = await self._analyze_market_impact(
                    execution_result, market_data_before, market_data_after
                )
                analysis.market_impact_bps = impact_analysis["total_impact"]
                analysis.temporary_impact_bps = impact_analysis["temporary_impact"]
                analysis.permanent_impact_bps = impact_analysis["permanent_impact"]
            
            # Calculate execution metrics
            analysis.execution_time_seconds = getattr(
                execution_result, 'execution_duration_seconds', 0.0
            )
            analysis.fill_rate = (
                execution_result.total_filled_quantity / 
                getattr(execution_result, 'original_quantity', execution_result.total_filled_quantity)
            ) * 100
            
            # Benchmark comparison
            analysis.benchmark_scores = await self._compare_to_benchmarks(analysis)
            
            # Overall quality score
            analysis.overall_quality_score = (
                analysis.execution_quality_score * 0.4 +
                analysis.timing_quality_score * 0.3 +
                analysis.price_quality_score * 0.3
            )
            
            # Identify issues and recommendations
            analysis.issues = await self._identify_issues(analysis)
            analysis.recommendations = await self._generate_trade_recommendations(analysis)
            
            # Store analysis
            self.analysis_history.append(analysis)
            await self._log_analysis_metrics(analysis)
            
            # Update quality metrics
            self._update_quality_metrics("analysis", analysis)
            
            self.logger.info(
                "Post-trade analysis completed",
                trade_id=trade_id,
                overall_quality_score=analysis.overall_quality_score,
                slippage_bps=analysis.slippage_bps,
                execution_time=analysis.execution_time_seconds
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Post-trade analysis failed: {e}", trade_id=trade_id)
            raise StateError(f"Post-trade analysis error: {e}")

    async def get_quality_summary(
        self,
        bot_id: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get quality summary for recent period.
        
        Args:
            bot_id: Filter by bot ID
            hours: Time period in hours
            
        Returns:
            Quality summary
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            # Filter validations and analyses by time
            recent_validations = [
                v for v in self.validation_history
                if v.timestamp >= cutoff_time
            ]
            
            recent_analyses = [
                a for a in self.analysis_history
                if a.timestamp >= cutoff_time
            ]
            
            # Calculate summary metrics
            summary = {
                "period_hours": hours,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "validation_summary": self._summarize_validations(recent_validations),
                "analysis_summary": self._summarize_analyses(recent_analyses),
                "quality_trends": await self._get_quality_trends(hours),
                "alerts": await self._get_quality_alerts(),
                "recommendations": await self._get_improvement_recommendations()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get quality summary: {e}")
            return {"error": str(e)}

    async def get_quality_trend_analysis(
        self,
        metric: str,
        days: int = 7
    ) -> QualityTrend:
        """
        Get quality trend analysis for a specific metric.
        
        Args:
            metric: Metric name to analyze
            days: Number of days to analyze
            
        Returns:
            Quality trend analysis
        """
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)
            
            # Extract metric values from history
            values = []
            timestamps = []
            
            if metric == "overall_quality_score":
                for analysis in self.analysis_history:
                    if analysis.timestamp >= start_time:
                        values.append(analysis.overall_quality_score)
                        timestamps.append(analysis.timestamp)
            elif metric == "slippage_bps":
                for analysis in self.analysis_history:
                    if analysis.timestamp >= start_time:
                        values.append(analysis.slippage_bps)
                        timestamps.append(analysis.timestamp)
            elif metric == "execution_time_seconds":
                for analysis in self.analysis_history:
                    if analysis.timestamp >= start_time:
                        values.append(analysis.execution_time_seconds)
                        timestamps.append(analysis.timestamp)
            
            if not values:
                return QualityTrend(metric_name=metric)
            
            # Calculate trend statistics
            values_array = np.array(values)
            
            trend = QualityTrend(
                metric_name=metric,
                time_period=f"{days}d",
                current_value=values[-1] if values else 0.0,
                previous_value=values[0] if values else 0.0,
                mean=float(np.mean(values_array)),
                std_dev=float(np.std(values_array)),
                min_value=float(np.min(values_array)),
                max_value=float(np.max(values_array)),
                percentile_95=float(np.percentile(values_array, 95)),
                percentile_5=float(np.percentile(values_array, 5))
            )
            
            # Calculate change percentage
            if trend.previous_value != 0:
                trend.change_percentage = (
                    (trend.current_value - trend.previous_value) / trend.previous_value
                ) * 100
            
            # Determine trend direction
            if len(values) >= 2:
                recent_slope, _, _, p_value, _ = stats.linregress(
                    range(len(values)), values
                )
                
                if p_value < 0.05:  # Statistically significant trend
                    if recent_slope > 0:
                        trend.trend_direction = "improving" if metric == "overall_quality_score" else "declining"
                    else:
                        trend.trend_direction = "declining" if metric == "overall_quality_score" else "improving"
                else:
                    trend.trend_direction = "stable"
            
            # Check for alerts
            trend.alert_triggered, trend.alert_level = self._check_trend_alerts(metric, trend)
            
            return trend
            
        except Exception as e:
            self.logger.error(f"Failed to get quality trend analysis: {e}")
            return QualityTrend(metric_name=metric)

    # Private helper methods
    
    async def _validate_order_structure(
        self, 
        order_request: OrderRequest, 
        validation: PreTradeValidation
    ) -> None:
        """Validate basic order structure."""
        check = ValidationCheck(check_name="order_structure")
        
        try:
            # Validate order data
            await validate_order_data(order_request.model_dump())
            
            # Check quantity is positive
            if order_request.quantity <= 0:
                check.result = ValidationResult.FAILED
                check.score = 0.0
                check.message = "Order quantity must be positive"
                check.severity = "critical"
            
            # Check symbol format
            elif len(order_request.symbol) < 3:
                check.result = ValidationResult.FAILED
                check.score = 0.0
                check.message = "Invalid symbol format"
                check.severity = "high"
            
            # Check price for limit orders
            elif (order_request.order_type.value in ["limit", "stop_loss"] and 
                  (not order_request.price or order_request.price <= 0)):
                check.result = ValidationResult.FAILED
                check.score = 0.0
                check.message = "Limit orders require positive price"
                check.severity = "high"
            
            else:
                check.result = ValidationResult.PASSED
                check.score = 100.0
                check.message = "Order structure is valid"
            
        except Exception as e:
            check.result = ValidationResult.FAILED
            check.score = 0.0
            check.message = f"Order validation error: {e}"
            check.severity = "critical"
        
        validation.checks.append(check)

    async def _validate_market_conditions(
        self,
        order_request: OrderRequest,
        market_data: MarketData,
        validation: PreTradeValidation
    ) -> None:
        """Validate current market conditions."""
        check = ValidationCheck(check_name="market_conditions")
        
        try:
            score = 100.0
            issues = []
            
            # Check bid-ask spread
            if market_data.bid and market_data.ask:
                spread = market_data.ask - market_data.bid
                spread_pct = (spread / market_data.price) * 100
                
                if spread_pct > 1.0:  # 1% spread threshold
                    score -= 30.0
                    issues.append(f"Wide bid-ask spread: {spread_pct:.2f}%")
                elif spread_pct > 0.5:
                    score -= 15.0
                    issues.append(f"Moderate bid-ask spread: {spread_pct:.2f}%")
            
            # Check volume
            if market_data.volume:
                # This would compare against historical average volume
                # For now, just check if volume is very low
                if market_data.volume < 1000:  # Threshold would be symbol-specific
                    score -= 20.0
                    issues.append("Low trading volume")
            
            # Set results
            if score >= 70:
                check.result = ValidationResult.PASSED
            elif score >= 50:
                check.result = ValidationResult.WARNING
            else:
                check.result = ValidationResult.FAILED
            
            check.score = score
            check.message = "; ".join(issues) if issues else "Market conditions acceptable"
            check.details = {
                "spread_pct": spread_pct if market_data.bid and market_data.ask else None,
                "volume": float(market_data.volume) if market_data.volume else None
            }
            
        except Exception as e:
            check.result = ValidationResult.FAILED
            check.score = 0.0
            check.message = f"Market conditions validation error: {e}"
            check.severity = "medium"
        
        validation.checks.append(check)

    async def _validate_portfolio_risk(
        self,
        order_request: OrderRequest,
        portfolio_context: Dict[str, Any],
        validation: PreTradeValidation
    ) -> None:
        """Validate portfolio risk implications."""
        check = ValidationCheck(check_name="portfolio_risk")
        
        try:
            score = 100.0
            issues = []
            
            # Check position size limits
            portfolio_value = portfolio_context.get("total_value", 0)
            order_value = float(order_request.quantity * (order_request.price or 0))
            
            if portfolio_value > 0:
                position_pct = (order_value / portfolio_value) * 100
                max_position_pct = self.validation_rules["position_size"]["max_percentage"]
                
                if position_pct > max_position_pct:
                    score -= 50.0
                    issues.append(f"Position size {position_pct:.1f}% exceeds limit {max_position_pct}%")
                elif position_pct > max_position_pct * 0.8:
                    score -= 20.0
                    issues.append(f"Position size {position_pct:.1f}% near limit")
            
            # Check concentration risk
            symbol_exposure = portfolio_context.get("symbol_exposure", {})
            current_exposure = symbol_exposure.get(order_request.symbol, 0)
            new_exposure_pct = ((current_exposure + order_value) / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            if new_exposure_pct > 20.0:  # 20% concentration limit
                score -= 30.0
                issues.append(f"Symbol concentration {new_exposure_pct:.1f}% too high")
            
            # Set results
            if score >= 70:
                check.result = ValidationResult.PASSED
                check.severity = "low"
            elif score >= 50:
                check.result = ValidationResult.WARNING
                check.severity = "medium"
            else:
                check.result = ValidationResult.FAILED
                check.severity = "high"
            
            check.score = score
            check.message = "; ".join(issues) if issues else "Portfolio risk acceptable"
            check.details = {
                "position_pct": position_pct if portfolio_value > 0 else None,
                "symbol_exposure_pct": new_exposure_pct
            }
            
        except Exception as e:
            check.result = ValidationResult.FAILED
            check.score = 0.0
            check.message = f"Portfolio risk validation error: {e}"
            check.severity = "medium"
        
        validation.checks.append(check)

    async def _validate_liquidity(
        self,
        order_request: OrderRequest,
        validation: PreTradeValidation
    ) -> None:
        """Validate market liquidity for the order."""
        check = ValidationCheck(check_name="liquidity")
        
        try:
            # This would typically check order book depth, recent volume, etc.
            # For now, implement basic checks
            
            score = 100.0
            issues = []
            
            # Placeholder liquidity check - would need real market data
            # This is simplified for demonstration
            estimated_volume = 1000000  # Would get from market data
            order_value = float(order_request.quantity * (order_request.price or 100))
            
            if order_value > estimated_volume * 0.1:  # Order > 10% of recent volume
                score -= 40.0
                issues.append("Order size large relative to market volume")
            elif order_value > estimated_volume * 0.05:  # Order > 5% of recent volume
                score -= 20.0
                issues.append("Order size moderate relative to market volume")
            
            # Set results
            check.result = ValidationResult.PASSED if score >= 70 else ValidationResult.WARNING
            check.score = score
            check.message = "; ".join(issues) if issues else "Liquidity adequate"
            check.details = {"estimated_market_impact_pct": (order_value / estimated_volume) * 100}
            
        except Exception as e:
            check.result = ValidationResult.FAILED
            check.score = 0.0
            check.message = f"Liquidity validation error: {e}"
            check.severity = "medium"
        
        validation.checks.append(check)

    async def _validate_timing(
        self,
        order_request: OrderRequest,
        validation: PreTradeValidation
    ) -> None:
        """Validate order timing."""
        check = ValidationCheck(check_name="timing")
        
        try:
            score = 100.0
            issues = []
            
            current_time = datetime.now(timezone.utc)
            
            # Check market hours (simplified - would need proper market calendar)
            hour = current_time.hour
            if hour < 9 or hour > 16:  # Outside typical US market hours
                if self.validation_rules["market_hours"]["enforce"]:
                    score -= 30.0
                    issues.append("Outside market hours")
                else:
                    score -= 10.0
                    issues.append("Outside typical market hours")
            
            # Check for major events (would integrate with calendar)
            # This is a placeholder
            
            check.result = ValidationResult.PASSED if score >= 70 else ValidationResult.WARNING
            check.score = score
            check.message = "; ".join(issues) if issues else "Timing acceptable"
            check.details = {"market_hour": hour}
            
        except Exception as e:
            check.result = ValidationResult.FAILED
            check.score = 0.0
            check.message = f"Timing validation error: {e}"
            check.severity = "low"
        
        validation.checks.append(check)

    async def _validate_correlation(
        self,
        order_request: OrderRequest,
        portfolio_context: Optional[Dict[str, Any]],
        validation: PreTradeValidation
    ) -> None:
        """Validate correlation with existing positions."""
        check = ValidationCheck(check_name="correlation")
        
        try:
            score = 100.0
            issues = []
            
            if portfolio_context and "correlations" in portfolio_context:
                correlations = portfolio_context["correlations"]
                symbol_correlations = correlations.get(order_request.symbol, {})
                
                for existing_symbol, correlation in symbol_correlations.items():
                    if abs(correlation) > self.validation_rules["correlation"]["max_correlation"]:
                        score -= 20.0
                        issues.append(f"High correlation with {existing_symbol}: {correlation:.2f}")
            
            check.result = ValidationResult.PASSED if score >= 70 else ValidationResult.WARNING
            check.score = score
            check.message = "; ".join(issues) if issues else "Correlation risk acceptable"
            
        except Exception as e:
            check.result = ValidationResult.WARNING
            check.score = 90.0
            check.message = f"Correlation validation warning: {e}"
            check.severity = "low"
        
        validation.checks.append(check)

    def _calculate_overall_score(self, checks: List[ValidationCheck]) -> float:
        """Calculate overall validation score."""
        if not checks:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for check in checks:
            weight = self.validation_rules.get(check.check_name, {}).get("weight", 10.0)
            total_weighted_score += check.score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _determine_overall_result(self, checks: List[ValidationCheck]) -> ValidationResult:
        """Determine overall validation result."""
        if any(check.result == ValidationResult.FAILED for check in checks):
            return ValidationResult.FAILED
        elif any(check.result == ValidationResult.WARNING for check in checks):
            return ValidationResult.WARNING
        else:
            return ValidationResult.PASSED

    def _assess_risk_level(self, checks: List[ValidationCheck]) -> str:
        """Assess overall risk level."""
        critical_issues = sum(1 for check in checks if check.severity == "critical")
        high_issues = sum(1 for check in checks if check.severity == "high")
        
        if critical_issues > 0:
            return "critical"
        elif high_issues > 1:
            return "high"
        elif high_issues > 0:
            return "medium"
        else:
            return "low"

    def _calculate_risk_score(self, checks: List[ValidationCheck]) -> float:
        """Calculate numerical risk score."""
        risk_score = 0.0
        
        for check in checks:
            if check.severity == "critical":
                risk_score += 40.0
            elif check.severity == "high":
                risk_score += 20.0
            elif check.severity == "medium":
                risk_score += 10.0
            elif check.severity == "low":
                risk_score += 5.0
        
        return min(risk_score, 100.0)

    def _generate_recommendations(self, checks: List[ValidationCheck]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for check in checks:
            if check.result == ValidationResult.FAILED:
                if check.check_name == "position_size":
                    recommendations.append("Consider reducing position size to stay within limits")
                elif check.check_name == "market_conditions":
                    recommendations.append("Wait for better market conditions or use limit orders")
                elif check.check_name == "liquidity":
                    recommendations.append("Split large orders or use TWAP execution")
                elif check.check_name == "timing":
                    recommendations.append("Consider waiting for market hours")
            elif check.result == ValidationResult.WARNING:
                if check.check_name == "correlation":
                    recommendations.append("Monitor correlation risk with existing positions")
                elif check.check_name == "portfolio_risk":
                    recommendations.append("Review portfolio concentration and diversification")
        
        return recommendations

    async def _analyze_execution_quality(self, execution_result: ExecutionResult) -> float:
        """Analyze execution quality."""
        try:
            score = 100.0
            
            # Fill rate analysis
            fill_rate = (
                execution_result.total_filled_quantity / 
                getattr(execution_result, 'original_quantity', execution_result.total_filled_quantity)
            )
            
            if fill_rate < 0.9:  # Less than 90% filled
                score -= 30.0
            elif fill_rate < 0.95:  # Less than 95% filled
                score -= 15.0
            
            # Execution efficiency (placeholder - would analyze order book impact)
            # This would involve complex market microstructure analysis
            
            return max(score, 0.0)
            
        except Exception as e:
            self.logger.warning(f"Execution quality analysis error: {e}")
            return 50.0

    async def _analyze_timing_quality(self, execution_result: ExecutionResult) -> float:
        """Analyze execution timing quality."""
        try:
            score = 100.0
            
            execution_time = getattr(execution_result, 'execution_duration_seconds', 0.0)
            
            # Penalize slow executions
            if execution_time > self.execution_time_threshold_seconds:
                score -= 40.0
            elif execution_time > self.execution_time_threshold_seconds * 0.5:
                score -= 20.0
            
            return max(score, 0.0)
            
        except Exception as e:
            self.logger.warning(f"Timing quality analysis error: {e}")
            return 50.0

    async def _analyze_price_quality(
        self,
        execution_result: ExecutionResult,
        market_data_before: Optional[MarketData]
    ) -> float:
        """Analyze price quality relative to market."""
        try:
            score = 100.0
            
            if not market_data_before:
                return 50.0  # Can't analyze without reference price
            
            # Calculate price improvement/slippage
            reference_price = market_data_before.price
            execution_price = execution_result.average_fill_price
            
            if reference_price and execution_price:
                price_diff_pct = abs(execution_price - reference_price) / reference_price * 100
                
                if price_diff_pct > 0.5:  # More than 0.5% slippage
                    score -= 40.0
                elif price_diff_pct > 0.2:  # More than 0.2% slippage
                    score -= 20.0
            
            return max(score, 0.0)
            
        except Exception as e:
            self.logger.warning(f"Price quality analysis error: {e}")
            return 50.0

    async def _calculate_slippage(
        self,
        execution_result: ExecutionResult,
        market_data_before: Optional[MarketData]
    ) -> float:
        """Calculate slippage in basis points."""
        try:
            if not market_data_before or not market_data_before.price:
                return 0.0
            
            reference_price = market_data_before.price
            execution_price = execution_result.average_fill_price
            
            if reference_price and execution_price:
                slippage = (execution_price - reference_price) / reference_price
                return float(slippage * 10000)  # Convert to basis points
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Slippage calculation error: {e}")
            return 0.0

    async def _analyze_market_impact(
        self,
        execution_result: ExecutionResult,
        market_data_before: MarketData,
        market_data_after: MarketData
    ) -> Dict[str, float]:
        """Analyze market impact of the trade."""
        try:
            # This is a simplified market impact analysis
            # Real implementation would require tick-by-tick data and sophisticated models
            
            price_before = market_data_before.price
            price_after = market_data_after.price
            
            if not price_before or not price_after:
                return {"total_impact": 0.0, "temporary_impact": 0.0, "permanent_impact": 0.0}
            
            total_impact = ((price_after - price_before) / price_before) * 10000  # basis points
            
            # For simplification, assume half is temporary, half is permanent
            temporary_impact = total_impact * 0.5
            permanent_impact = total_impact * 0.5
            
            return {
                "total_impact": float(abs(total_impact)),
                "temporary_impact": float(abs(temporary_impact)),
                "permanent_impact": float(abs(permanent_impact))
            }
            
        except Exception as e:
            self.logger.warning(f"Market impact analysis error: {e}")
            return {"total_impact": 0.0, "temporary_impact": 0.0, "permanent_impact": 0.0}

    async def _compare_to_benchmarks(self, analysis: PostTradeAnalysis) -> Dict[str, float]:
        """Compare analysis results to benchmarks."""
        try:
            benchmark_scores = {}
            
            # Slippage benchmark
            if analysis.slippage_bps <= self.benchmarks["slippage_p50"]:
                benchmark_scores["slippage"] = 100.0
            elif analysis.slippage_bps <= self.benchmarks["slippage_p90"]:
                benchmark_scores["slippage"] = 70.0
            else:
                benchmark_scores["slippage"] = 30.0
            
            # Execution time benchmark
            if analysis.execution_time_seconds <= self.benchmarks["execution_time_p50"]:
                benchmark_scores["execution_time"] = 100.0
            elif analysis.execution_time_seconds <= self.benchmarks["execution_time_p90"]:
                benchmark_scores["execution_time"] = 70.0
            else:
                benchmark_scores["execution_time"] = 30.0
            
            # Fill rate benchmark
            if analysis.fill_rate >= self.benchmarks["fill_rate_p50"]:
                benchmark_scores["fill_rate"] = 100.0
            else:
                benchmark_scores["fill_rate"] = 50.0
            
            # Market impact benchmark
            if analysis.market_impact_bps <= self.benchmarks["market_impact_p50"]:
                benchmark_scores["market_impact"] = 100.0
            else:
                benchmark_scores["market_impact"] = 50.0
            
            return benchmark_scores
            
        except Exception as e:
            self.logger.warning(f"Benchmark comparison error: {e}")
            return {}

    async def _identify_issues(self, analysis: PostTradeAnalysis) -> List[str]:
        """Identify issues from post-trade analysis."""
        issues = []
        
        if analysis.slippage_bps > self.slippage_threshold_bps:
            issues.append(f"High slippage: {analysis.slippage_bps:.1f} bps")
        
        if analysis.execution_time_seconds > self.execution_time_threshold_seconds:
            issues.append(f"Slow execution: {analysis.execution_time_seconds:.1f} seconds")
        
        if analysis.fill_rate < 95.0:
            issues.append(f"Low fill rate: {analysis.fill_rate:.1f}%")
        
        if analysis.market_impact_bps > self.market_impact_threshold_bps:
            issues.append(f"High market impact: {analysis.market_impact_bps:.1f} bps")
        
        if analysis.overall_quality_score < self.min_quality_score:
            issues.append(f"Low overall quality: {analysis.overall_quality_score:.1f}")
        
        return issues

    async def _generate_trade_recommendations(self, analysis: PostTradeAnalysis) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if analysis.slippage_bps > self.slippage_threshold_bps:
            recommendations.append("Consider using limit orders or TWAP execution for better pricing")
        
        if analysis.execution_time_seconds > self.execution_time_threshold_seconds:
            recommendations.append("Use more aggressive execution algorithms for time-sensitive trades")
        
        if analysis.market_impact_bps > self.market_impact_threshold_bps:
            recommendations.append("Split large orders to reduce market impact")
        
        if analysis.fill_rate < 95.0:
            recommendations.append("Review order sizing and market conditions before trading")
        
        return recommendations

    def _summarize_validations(self, validations: List[PreTradeValidation]) -> Dict[str, Any]:
        """Summarize validation results."""
        if not validations:
            return {"count": 0}
        
        passed = sum(1 for v in validations if v.overall_result == ValidationResult.PASSED)
        failed = sum(1 for v in validations if v.overall_result == ValidationResult.FAILED)
        warning = sum(1 for v in validations if v.overall_result == ValidationResult.WARNING)
        
        avg_score = sum(v.overall_score for v in validations) / len(validations)
        avg_time = sum(v.validation_time_ms for v in validations) / len(validations)
        
        return {
            "count": len(validations),
            "passed": passed,
            "failed": failed,
            "warning": warning,
            "pass_rate": (passed / len(validations)) * 100,
            "average_score": avg_score,
            "average_time_ms": avg_time
        }

    def _summarize_analyses(self, analyses: List[PostTradeAnalysis]) -> Dict[str, Any]:
        """Summarize analysis results."""
        if not analyses:
            return {"count": 0}
        
        avg_quality = sum(a.overall_quality_score for a in analyses) / len(analyses)
        avg_slippage = sum(a.slippage_bps for a in analyses) / len(analyses)
        avg_execution_time = sum(a.execution_time_seconds for a in analyses) / len(analyses)
        avg_fill_rate = sum(a.fill_rate for a in analyses) / len(analyses)
        
        high_quality = sum(1 for a in analyses if a.overall_quality_score >= 80)
        issues_count = sum(len(a.issues) for a in analyses)
        
        return {
            "count": len(analyses),
            "average_quality_score": avg_quality,
            "average_slippage_bps": avg_slippage,
            "average_execution_time_seconds": avg_execution_time,
            "average_fill_rate": avg_fill_rate,
            "high_quality_trades": high_quality,
            "high_quality_rate": (high_quality / len(analyses)) * 100,
            "total_issues": issues_count
        }

    async def _get_quality_trends(self, hours: int) -> List[Dict[str, Any]]:
        """Get quality trends for the period."""
        trends = []
        
        # Get trends for key metrics
        for metric in ["overall_quality_score", "slippage_bps", "execution_time_seconds"]:
            trend = await self.get_quality_trend_analysis(metric, days=hours//24 or 1)
            trends.append({
                "metric": metric,
                "current_value": trend.current_value,
                "trend_direction": trend.trend_direction,
                "change_percentage": trend.change_percentage
            })
        
        return trends

    async def _get_quality_alerts(self) -> List[Dict[str, Any]]:
        """Get active quality alerts."""
        alerts = []
        
        # Check recent performance against thresholds
        recent_analyses = self.analysis_history[-10:] if self.analysis_history else []
        
        if recent_analyses:
            avg_quality = sum(a.overall_quality_score for a in recent_analyses) / len(recent_analyses)
            if avg_quality < self.min_quality_score:
                alerts.append({
                    "type": "quality_degradation",
                    "severity": "warning",
                    "message": f"Average quality score {avg_quality:.1f} below threshold {self.min_quality_score}"
                })
            
            avg_slippage = sum(a.slippage_bps for a in recent_analyses) / len(recent_analyses)
            if avg_slippage > self.slippage_threshold_bps:
                alerts.append({
                    "type": "high_slippage",
                    "severity": "warning",
                    "message": f"Average slippage {avg_slippage:.1f} bps above threshold {self.slippage_threshold_bps}"
                })
        
        return alerts

    async def _get_improvement_recommendations(self) -> List[str]:
        """Get improvement recommendations based on recent performance."""
        recommendations = []
        
        # Analyze recent performance patterns
        recent_analyses = self.analysis_history[-20:] if self.analysis_history else []
        
        if recent_analyses:
            # Check for consistent issues
            high_slippage_count = sum(1 for a in recent_analyses if a.slippage_bps > self.slippage_threshold_bps)
            if high_slippage_count > len(recent_analyses) * 0.3:  # More than 30% of trades
                recommendations.append("Consider implementing more sophisticated execution algorithms")
            
            slow_execution_count = sum(1 for a in recent_analyses if a.execution_time_seconds > self.execution_time_threshold_seconds)
            if slow_execution_count > len(recent_analyses) * 0.2:  # More than 20% of trades
                recommendations.append("Review execution infrastructure and consider co-location")
            
            low_fill_count = sum(1 for a in recent_analyses if a.fill_rate < 95.0)
            if low_fill_count > len(recent_analyses) * 0.1:  # More than 10% of trades
                recommendations.append("Improve order sizing and market timing strategies")
        
        return recommendations

    def _update_quality_metrics(self, operation: str, data: Any) -> None:
        """Update quality metrics."""
        if operation == "validation":
            self.quality_metrics["total_validations"] += 1
            if data.overall_result == ValidationResult.PASSED:
                self.quality_metrics["passed_validations"] += 1
            else:
                self.quality_metrics["failed_validations"] += 1
            
            # Update average quality score
            total = self.quality_metrics["total_validations"]
            current_avg = self.quality_metrics["average_quality_score"]
            new_avg = (current_avg * (total - 1) + data.overall_score) / total
            self.quality_metrics["average_quality_score"] = new_avg
            
        elif operation == "analysis":
            self.quality_metrics["total_analyses"] += 1
            
            # Update average execution quality
            total = self.quality_metrics["total_analyses"]
            current_avg = self.quality_metrics["average_execution_quality"]
            new_avg = (current_avg * (total - 1) + data.overall_quality_score) / total
            self.quality_metrics["average_execution_quality"] = new_avg
            
            # Update incident counters
            if data.slippage_bps > self.slippage_threshold_bps:
                self.quality_metrics["slippage_incidents"] += 1
            
            if data.execution_time_seconds > self.execution_time_threshold_seconds:
                self.quality_metrics["execution_time_violations"] += 1

    def _check_trend_alerts(self, metric: str, trend: QualityTrend) -> Tuple[bool, str]:
        """Check if trend triggers alerts."""
        alert_triggered = False
        alert_level = "none"
        
        # Check for significant degradation
        if metric == "overall_quality_score":
            if trend.current_value < 50 and trend.trend_direction == "declining":
                alert_triggered = True
                alert_level = "critical"
            elif trend.current_value < 70 and abs(trend.change_percentage) > 20:
                alert_triggered = True
                alert_level = "warning"
        
        elif metric == "slippage_bps":
            if trend.current_value > self.slippage_threshold_bps * 2:
                alert_triggered = True
                alert_level = "critical"
            elif trend.current_value > self.slippage_threshold_bps and trend.trend_direction == "declining":
                alert_triggered = True
                alert_level = "warning"
        
        return alert_triggered, alert_level

    async def _load_benchmarks(self) -> None:
        """Load historical benchmarks."""
        try:
            # In a full implementation, this would load benchmarks from database
            # For now, use default values
            self.logger.info("Loaded quality benchmarks")
            
        except Exception as e:
            self.logger.warning(f"Failed to load benchmarks: {e}")

    async def _log_validation_metrics(self, validation: PreTradeValidation) -> None:
        """Log validation metrics to InfluxDB."""
        try:
            metrics_data = {
                "measurement": "trade_validation_metrics",
                "tags": {
                    "validation_id": validation.validation_id,
                    "result": validation.overall_result.value,
                    "risk_level": validation.risk_level
                },
                "fields": {
                    "overall_score": validation.overall_score,
                    "risk_score": validation.risk_score,
                    "validation_time_ms": validation.validation_time_ms,
                    "checks_count": len(validation.checks)
                },
                "time": validation.timestamp
            }
            
            await self.influxdb_client.write_point(metrics_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to log validation metrics: {e}")

    async def _log_analysis_metrics(self, analysis: PostTradeAnalysis) -> None:
        """Log analysis metrics to InfluxDB."""
        try:
            metrics_data = {
                "measurement": "trade_analysis_metrics",
                "tags": {
                    "analysis_id": analysis.analysis_id,
                    "trade_id": analysis.trade_id
                },
                "fields": {
                    "overall_quality_score": analysis.overall_quality_score,
                    "execution_quality_score": analysis.execution_quality_score,
                    "timing_quality_score": analysis.timing_quality_score,
                    "price_quality_score": analysis.price_quality_score,
                    "slippage_bps": analysis.slippage_bps,
                    "execution_time_seconds": analysis.execution_time_seconds,
                    "fill_rate": analysis.fill_rate,
                    "market_impact_bps": analysis.market_impact_bps
                },
                "time": analysis.timestamp
            }
            
            await self.influxdb_client.write_point(metrics_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to log analysis metrics: {e}")

    async def _quality_monitoring_loop(self) -> None:
        """Background quality monitoring loop."""
        while True:
            try:
                # Monitor quality metrics and trigger alerts
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Quality monitoring loop error: {e}")
                await asyncio.sleep(300)

    async def _trend_analysis_loop(self) -> None:
        """Background trend analysis loop."""
        while True:
            try:
                # Analyze quality trends and update benchmarks
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                self.logger.error(f"Trend analysis loop error: {e}")
                await asyncio.sleep(3600)