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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import stats

from src.core.base.component import BaseComponent
from src.core.config.main import Config
from src.core.exceptions import StateError, ValidationError
from src.core.types import ExecutionResult, MarketData, OrderRequest
from src.database.service import DatabaseService

# Backward compatibility imports for tests
try:
    from src.database.manager import DatabaseManager
except ImportError:
    DatabaseManager = None  # type: ignore

try:
    from src.database.redis_client import RedisClient
except ImportError:
    RedisClient = None  # type: ignore

try:
    from src.database.influxdb_client import InfluxDBClient
except ImportError:
    InfluxDBClient = None  # type: ignore

from .utils_imports import time_execution

# Removed unused validate_order_data import


class QualityLevel(Enum):
    """Quality assessment levels."""

    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"  # 70-89
    FAIR = "fair"  # 50-69
    POOR = "poor"  # 30-49
    CRITICAL = "critical"  # 0-29


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
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "low"  # low, medium, high, critical


@dataclass
class PreTradeValidation:
    """Pre-trade validation results."""

    validation_id: str = field(default_factory=lambda: str(uuid4()))
    order_request: OrderRequest | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Overall results
    overall_result: ValidationResult = ValidationResult.PASSED
    overall_score: float = 100.0

    # Individual checks
    checks: list[ValidationCheck] = field(default_factory=list)

    # Risk assessment
    risk_level: str = "low"  # low, medium, high, critical
    risk_score: float = 0.0

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # Processing time
    validation_time_ms: float = 0.0


@dataclass
class PostTradeAnalysis:
    """Post-trade analysis results."""

    analysis_id: str = field(default_factory=lambda: str(uuid4()))
    trade_id: str = ""
    execution_result: ExecutionResult | None = None
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
    benchmark_scores: dict[str, float] = field(default_factory=dict)

    # Issues and recommendations
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


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


class MetricsStorage(ABC):
    """
    Abstract interface for metrics storage operations.

    This interface abstracts metrics persistence operations,
    allowing different storage backends (InfluxDB, TimescaleDB, etc.)
    without tight coupling.
    """

    @abstractmethod
    async def store_validation_metrics(self, validation_data: dict[str, Any]) -> bool:
        """Store validation metrics."""
        raise NotImplementedError

    @abstractmethod
    async def store_analysis_metrics(self, analysis_data: dict[str, Any]) -> bool:
        """Store analysis metrics."""
        raise NotImplementedError

    @abstractmethod
    async def get_historical_metrics(
        self, metric_type: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """Retrieve historical metrics."""
        raise NotImplementedError


class InfluxDBMetricsStorage(MetricsStorage):
    """
    InfluxDB implementation of MetricsStorage interface.

    This implementation provides the actual InfluxDB integration
    while maintaining abstraction for the rest of the system.
    """

    def __init__(self, config: Config | None = None):
        """Initialize InfluxDB metrics storage."""
        self.config = config
        self._influx_client = None
        self._available = False

        if config:
            try:
                # Import InfluxDB client only when actually needed
                from src.database.influxdb_client import InfluxDBClient

                self._influx_client = InfluxDBClient(config)
                self._available = True
            except ImportError:
                # InfluxDB client not available - gracefully degrade
                self._available = False

    async def close(self) -> None:
        """Close InfluxDB client connection."""
        if self._influx_client and self._available:
            try:
                if hasattr(self._influx_client, 'close'):
                    # Use asyncio.wait_for with timeout for safe cleanup
                    await asyncio.wait_for(
                        self._influx_client.close(),
                        timeout=5.0
                    )
                elif hasattr(self._influx_client, 'disconnect'):
                    # Use asyncio.wait_for with timeout for safe cleanup
                    await asyncio.wait_for(
                        self._influx_client.disconnect(),
                        timeout=5.0
                    )
                self._available = False
            except asyncio.TimeoutError:
                # Log timeout but don't raise to avoid interfering with cleanup
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("InfluxDB client close timeout")
            except Exception as e:
                # Log error but don't raise to avoid interfering with cleanup
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error closing InfluxDB client: {e}")
            finally:
                self._influx_client = None

    async def store_validation_metrics(self, validation_data: dict[str, Any]) -> bool:
        """Store validation metrics to InfluxDB."""
        if not self._available or not self._influx_client:
            return False

        try:
            from influxdb_client import Point

            point = Point("trade_validation_metrics")

            # Add tags
            for key in ["validation_id", "result", "risk_level"]:
                if key in validation_data:
                    point.tag(key, str(validation_data[key]))

            # Add fields
            for key in ["overall_score", "risk_score", "validation_time_ms", "checks_count"]:
                if key in validation_data:
                    point.field(key, validation_data[key] if isinstance(validation_data[key], (int, float)) else float(validation_data[key]))

            # Set timestamp
            if "timestamp" in validation_data:
                point.time(validation_data["timestamp"])

            return await self._influx_client.write_point(point)

        except Exception as e:
            self.logger.error(f"Failed to store validation metrics: {e}")
            return False

    async def store_analysis_metrics(self, analysis_data: dict[str, Any]) -> bool:
        """Store analysis metrics to InfluxDB."""
        if not self._available or not self._influx_client:
            return False

        try:
            from influxdb_client import Point

            point = Point("trade_analysis_metrics")

            # Add tags
            for key in ["analysis_id", "trade_id"]:
                if key in analysis_data:
                    point.tag(key, str(analysis_data[key]))

            # Add fields
            metric_fields = [
                "overall_quality_score",
                "execution_quality_score",
                "timing_quality_score",
                "price_quality_score",
                "slippage_bps",
                "execution_time_seconds",
                "fill_rate",
                "market_impact_bps",
            ]

            for key in metric_fields:
                if key in analysis_data:
                    point.field(key, analysis_data[key] if isinstance(analysis_data[key], (int, float)) else float(analysis_data[key]))

            # Set timestamp
            if "timestamp" in analysis_data:
                point.time(analysis_data["timestamp"])

            return await self._influx_client.write_point(point)

        except Exception as e:
            self.logger.error(f"Failed to store analysis metrics: {e}")
            return False

    async def get_historical_metrics(
        self, metric_type: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """Retrieve historical metrics from InfluxDB."""
        if not self._available or not self._influx_client:
            return []

        try:
            # This would implement actual InfluxDB querying
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            self.logger.error(f"Failed to get historical metrics: {e}")
            return []


class NullMetricsStorage(MetricsStorage):
    """
    Null implementation of MetricsStorage for testing or when metrics storage is disabled.
    """

    async def store_validation_metrics(self, validation_data: dict[str, Any]) -> bool:
        """No-op metrics storage."""
        return True

    async def store_analysis_metrics(self, analysis_data: dict[str, Any]) -> bool:
        """No-op metrics storage."""
        return True

    async def get_historical_metrics(
        self, metric_type: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """Return empty metrics."""
        return []


class QualityController(BaseComponent):
    """
    Comprehensive quality control system for trading operations.

    Features:
    - Multi-layer pre-trade validation (risk, market, technical)
    - Real-time execution quality monitoring
    - Advanced post-trade analysis with benchmarking
    - Quality trend analysis and alerting
    - Performance attribution and improvement recommendations

    The controller now uses service abstractions for database operations
    and metrics storage, eliminating direct database dependencies.
    """

    def __init__(
        self,
        config: Config,
        metrics_storage: MetricsStorage | None = None,
    ):
        """
        Initialize the quality controller.

        Args:
            config: Application configuration
            metrics_storage: Optional metrics storage service for logging metrics
        """
        super().__init__(
            name="QualityController", config=config.__dict__ if hasattr(config, "__dict__") else {}
        )
        self.config = config
        # Logger is already provided by BaseComponent

        # Injected dependencies - use service abstractions
        self.metrics_storage = metrics_storage or NullMetricsStorage()

        # Ensure we have a metrics storage implementation
        if metrics_storage is None and config:
            # Try to create InfluxDB storage if config is available
            try:
                self.metrics_storage = InfluxDBMetricsStorage(config)
            except Exception as e:
                self.logger.warning(f"Failed to create InfluxDB metrics storage: {e}")
                # Fall back to null storage if InfluxDB is not available
                self.metrics_storage = NullMetricsStorage()

        # Quality configuration - use safe defaults if config sections not available
        self.min_quality_score = 70.0
        self.slippage_threshold_bps = 20.0
        self.execution_time_threshold_seconds = 30.0
        self.market_impact_threshold_bps = 10.0

        # Try to load quality config from various possible config locations
        risk_config = getattr(config, "risk", {})
        if risk_config and hasattr(risk_config, "quality"):
            quality_config = getattr(risk_config, "quality", {})
            self.min_quality_score = getattr(quality_config, "min_quality_score", 70.0)
            self.slippage_threshold_bps = getattr(quality_config, "slippage_threshold_bps", 20.0)
            self.execution_time_threshold_seconds = getattr(
                quality_config, "execution_time_threshold_seconds", 30.0
            )
            self.market_impact_threshold_bps = getattr(
                quality_config, "market_impact_threshold_bps", 10.0
            )

        # Validation rules
        self.validation_rules = {
            "position_size": {"max_percentage": 10.0, "weight": 25.0},
            "market_hours": {"enforce": True, "weight": 15.0},
            "liquidity": {"min_volume_usd": 100000, "weight": 20.0},
            "volatility": {"max_daily_range": 0.05, "weight": 15.0},
            "correlation": {"max_correlation": 0.8, "weight": 10.0},
            "risk_limits": {"max_var": 0.02, "weight": 15.0},
        }

        # Consistency rules for test compatibility
        self.consistency_rules: list = []

        # Benchmarks
        self.benchmarks = {
            "slippage_p50": 5.0,  # 50th percentile slippage in bps
            "slippage_p90": 15.0,  # 90th percentile slippage in bps
            "execution_time_p50": 10.0,  # 50th percentile execution time in seconds
            "execution_time_p90": 30.0,  # 90th percentile execution time in seconds
            "fill_rate_p50": 95.0,  # 50th percentile fill rate
            "market_impact_p50": 3.0,  # 50th percentile market impact in bps
        }

        # Quality history
        self.validation_history: list[PreTradeValidation] = []
        self.analysis_history: list[PostTradeAnalysis] = []

        # Performance tracking
        self.quality_metrics = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "average_quality_score": 0.0,
            "total_analyses": 0,
            "average_execution_quality": 0.0,
            "slippage_incidents": 0,
            "execution_time_violations": 0,
        }

        self.logger.info("QualityController initialized")

    async def initialize(self) -> None:
        """Initialize the quality controller."""
        try:
            # Load historical benchmarks (using default values)
            await self._load_benchmarks()

            # Start monitoring tasks
            self._quality_task = asyncio.create_task(self._quality_monitoring_loop())
            self._trend_task = asyncio.create_task(self._trend_analysis_loop())

            self.logger.info("QualityController initialization completed")

            # Log service availability
            metrics_status = type(self.metrics_storage).__name__
            self.logger.info(
                f"Quality controller services - Metrics: {metrics_status}"
            )

        except Exception as e:
            self.logger.error(f"QualityController initialization failed: {e}")
            raise StateError(f"Failed to initialize QualityController: {e}") from e

    @time_execution
    async def validate_pre_trade(
        self,
        order_request: OrderRequest,
        market_data: MarketData | None = None,
        portfolio_context: dict[str, Any] | None = None,
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
                risk_level=validation.risk_level,
            )

            return validation

        except Exception as e:
            self.logger.error(f"Pre-trade validation failed: {e}")
            raise ValidationError(f"Pre-trade validation error: {e}") from e

    @time_execution
    async def analyze_post_trade(
        self,
        trade_id: str,
        execution_result: ExecutionResult,
        market_data_before: MarketData | None = None,
        market_data_after: MarketData | None = None,
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
            analysis = PostTradeAnalysis(trade_id=trade_id, execution_result=execution_result)

            # Execution quality analysis
            analysis.execution_quality_score = await self._analyze_execution_quality(
                execution_result
            )

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
                execution_result, "execution_duration_seconds", 0.0
            )
            analysis.fill_rate = (
                execution_result.filled_quantity
                / getattr(execution_result, "target_quantity", execution_result.filled_quantity)
            ) * 100

            # Benchmark comparison
            analysis.benchmark_scores = await self._compare_to_benchmarks(analysis)

            # Overall quality score
            analysis.overall_quality_score = (
                analysis.execution_quality_score * 0.4
                + analysis.timing_quality_score * 0.3
                + analysis.price_quality_score * 0.3
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
                execution_time=analysis.execution_time_seconds,
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Post-trade analysis failed: {e}", trade_id=trade_id)
            raise StateError(f"Post-trade analysis error: {e}") from e

    async def get_quality_summary(
        self, bot_id: str | None = None, hours: int = 24
    ) -> dict[str, Any]:
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
            recent_validations = [v for v in self.validation_history if v.timestamp >= cutoff_time]

            recent_analyses = [a for a in self.analysis_history if a.timestamp >= cutoff_time]

            # Calculate summary metrics
            summary = {
                "period_hours": hours,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "validation_summary": self._summarize_validations(recent_validations),
                "analysis_summary": self._summarize_analyses(recent_analyses),
                "quality_trends": await self._get_quality_trends(hours),
                "alerts": await self._get_quality_alerts(),
                "recommendations": await self._get_improvement_recommendations(),
            }

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get quality summary: {e}")
            return {"error": str(e)}

    async def get_quality_trend_analysis(self, metric: str, days: int = 7) -> QualityTrend:
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
                mean=np.mean(values_array),
                std_dev=np.std(values_array),
                min_value=np.min(values_array),
                max_value=np.max(values_array),
                percentile_95=np.percentile(values_array, 95),
                percentile_5=np.percentile(values_array, 5),
            )

            # Calculate change percentage
            if trend.previous_value != 0:
                trend.change_percentage = (
                    (trend.current_value - trend.previous_value) / trend.previous_value
                ) * 100

            # Determine trend direction
            if len(values) >= 2:
                recent_slope, _, _, p_value, _ = stats.linregress(range(len(values)), values)

                if p_value < 0.05:  # Statistically significant trend
                    if recent_slope > 0:
                        trend.trend_direction = (
                            "improving" if metric == "overall_quality_score" else "declining"
                        )
                    else:
                        trend.trend_direction = (
                            "declining" if metric == "overall_quality_score" else "improving"
                        )
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
        self, order_request: OrderRequest, validation: PreTradeValidation
    ) -> None:
        """Validate basic order structure."""
        check = ValidationCheck(check_name="order_structure")

        try:
            # Validate order data
            # Note: Manual validation below replaces validate_order_data call

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
            elif order_request.order_type.value in ["limit", "stop_loss"] and (
                not order_request.price or order_request.price <= 0
            ):
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
        self, order_request: OrderRequest, market_data: MarketData, validation: PreTradeValidation
    ) -> None:
        """Validate current market conditions."""
        check = ValidationCheck(check_name="market_conditions")

        try:
            score = 100.0
            issues = []

            # Check bid-ask spread (use high-low as proxy if bid/ask not available)
            if hasattr(market_data, "bid") and hasattr(market_data, "ask"):
                spread = market_data.ask - market_data.bid
                spread_pct = (spread / market_data.close) * 100
            else:
                # Use high-low spread as proxy
                spread = market_data.high - market_data.low
                spread_pct = (spread / market_data.close) * 100

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
                "spread_pct": spread_pct if "spread_pct" in locals() else None,
                "volume": str(market_data.volume) if market_data.volume else None,
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
        portfolio_context: dict[str, Any],
        validation: PreTradeValidation,
    ) -> None:
        """Validate portfolio risk implications."""
        check = ValidationCheck(check_name="portfolio_risk")

        try:
            score = 100.0
            issues = []

            # Check position size limits
            portfolio_value = Decimal(str(portfolio_context.get("total_value", 0)))
            order_value = order_request.quantity * (order_request.price or Decimal("0"))

            if portfolio_value > 0:
                position_pct = (order_value / portfolio_value) * Decimal("100")
                max_position_pct = Decimal(str(self.validation_rules["position_size"]["max_percentage"]))

                if position_pct > max_position_pct:
                    score -= 50.0
                    issues.append(
                        f"Position size {position_pct:.1f}% exceeds limit {max_position_pct}%"
                    )
                elif position_pct > max_position_pct * Decimal("0.8"):
                    score -= 20.0
                    issues.append(f"Position size {position_pct:.1f}% near limit")

            # Check concentration risk
            symbol_exposure = portfolio_context.get("symbol_exposure", {})
            current_exposure = Decimal(str(symbol_exposure.get(order_request.symbol, 0)))
            new_exposure_pct = (
                ((current_exposure + order_value) / portfolio_value) * Decimal("100")
                if portfolio_value > 0
                else Decimal("0")
            )

            if new_exposure_pct > Decimal("20.0"):  # 20% concentration limit
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
                "symbol_exposure_pct": new_exposure_pct,
            }

        except Exception as e:
            check.result = ValidationResult.FAILED
            check.score = 0.0
            check.message = f"Portfolio risk validation error: {e}"
            check.severity = "medium"

        validation.checks.append(check)

    async def _validate_liquidity(
        self, order_request: OrderRequest, validation: PreTradeValidation
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
            estimated_volume = Decimal("1000000")  # Would get from market data
            order_value = order_request.quantity * (order_request.price or Decimal("100"))

            if order_value > estimated_volume * Decimal("0.1"):  # Order > 10% of recent volume
                score -= 40.0
                issues.append("Order size large relative to market volume")
            elif order_value > estimated_volume * Decimal("0.05"):  # Order > 5% of recent volume
                score -= 20.0
                issues.append("Order size moderate relative to market volume")

            # Set results
            check.result = ValidationResult.PASSED if score >= 70 else ValidationResult.WARNING
            check.score = score
            check.message = "; ".join(issues) if issues else "Liquidity adequate"
            check.details = {"estimated_market_impact_pct": str((order_value / estimated_volume) * Decimal("100"))}

        except Exception as e:
            check.result = ValidationResult.FAILED
            check.score = 0.0
            check.message = f"Liquidity validation error: {e}"
            check.severity = "medium"

        validation.checks.append(check)

    async def _validate_timing(
        self, order_request: OrderRequest, validation: PreTradeValidation
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
        portfolio_context: dict[str, Any] | None,
        validation: PreTradeValidation,
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

    def _calculate_overall_score(self, checks: list[ValidationCheck]) -> float:
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

    def _determine_overall_result(self, checks: list[ValidationCheck]) -> ValidationResult:
        """Determine overall validation result."""
        if any(check.result == ValidationResult.FAILED for check in checks):
            return ValidationResult.FAILED
        elif any(check.result == ValidationResult.WARNING for check in checks):
            return ValidationResult.WARNING
        else:
            return ValidationResult.PASSED

    def _assess_risk_level(self, checks: list[ValidationCheck]) -> str:
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

    def _calculate_risk_score(self, checks: list[ValidationCheck]) -> float:
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

    def _generate_recommendations(self, checks: list[ValidationCheck]) -> list[str]:
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
            fill_rate = execution_result.filled_quantity / getattr(
                execution_result, "target_quantity", execution_result.filled_quantity
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

            execution_time = getattr(execution_result, "execution_duration_seconds", 0.0)

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
        self, execution_result: ExecutionResult, market_data_before: MarketData | None
    ) -> float:
        """Analyze price quality relative to market."""
        try:
            score = 100.0

            if not market_data_before:
                return 50.0  # Can't analyze without reference price

            # Calculate price improvement/slippage
            reference_price = market_data_before.close
            execution_price = execution_result.average_price

            if reference_price and execution_price:
                price_diff_pct = abs(execution_price - reference_price) / reference_price * Decimal("100")

                if price_diff_pct > Decimal("0.5"):  # More than 0.5% slippage
                    score -= 40.0
                elif price_diff_pct > Decimal("0.2"):  # More than 0.2% slippage
                    score -= 20.0

            return max(score, 0.0)

        except Exception as e:
            self.logger.warning(f"Price quality analysis error: {e}")
            return 50.0

    async def _calculate_slippage(
        self, execution_result: ExecutionResult, market_data_before: MarketData | None
    ) -> float:
        """Calculate slippage in basis points."""
        try:
            if not market_data_before or not market_data_before.close:
                return 0.0

            reference_price = market_data_before.close
            execution_price = execution_result.average_price

            if reference_price and execution_price:
                slippage = (execution_price - reference_price) / reference_price
                return slippage * 10000  # Convert to basis points

            return 0.0

        except Exception as e:
            self.logger.warning(f"Slippage calculation error: {e}")
            return 0.0

    async def _analyze_market_impact(
        self,
        execution_result: ExecutionResult,
        market_data_before: MarketData,
        market_data_after: MarketData,
    ) -> dict[str, Decimal]:
        """Analyze market impact of the trade."""
        try:
            # This is a simplified market impact analysis
            # Real implementation would require tick-by-tick data and sophisticated models

            price_before = Decimal(str(market_data_before.close)) if market_data_before.close else None
            price_after = Decimal(str(market_data_after.close)) if market_data_after.close else None

            if not price_before or not price_after:
                return {"total_impact": Decimal("0"), "temporary_impact": Decimal("0"), "permanent_impact": Decimal("0")}

            total_impact = ((price_after - price_before) / price_before) * Decimal("10000")  # basis points

            # For simplification, assume half is temporary, half is permanent
            temporary_impact = total_impact * Decimal("0.5")
            permanent_impact = total_impact * Decimal("0.5")

            return {
                "total_impact": abs(total_impact),
                "temporary_impact": abs(temporary_impact),
                "permanent_impact": abs(permanent_impact),
            }

        except Exception as e:
            self.logger.warning(f"Market impact analysis error: {e}")
            return {"total_impact": Decimal("0"), "temporary_impact": Decimal("0"), "permanent_impact": Decimal("0")}

    async def _compare_to_benchmarks(self, analysis: PostTradeAnalysis) -> dict[str, float]:
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

    async def _identify_issues(self, analysis: PostTradeAnalysis) -> list[str]:
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

    async def _generate_trade_recommendations(self, analysis: PostTradeAnalysis) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if analysis.slippage_bps > self.slippage_threshold_bps:
            recommendations.append(
                "Consider using limit orders or TWAP execution for better pricing"
            )

        if analysis.execution_time_seconds > self.execution_time_threshold_seconds:
            recommendations.append(
                "Use more aggressive execution algorithms for time-sensitive trades"
            )

        if analysis.market_impact_bps > self.market_impact_threshold_bps:
            recommendations.append("Split large orders to reduce market impact")

        if analysis.fill_rate < 95.0:
            recommendations.append("Review order sizing and market conditions before trading")

        return recommendations

    def _summarize_validations(self, validations: list[PreTradeValidation]) -> dict[str, Any]:
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
            "average_time_ms": avg_time,
        }

    def _summarize_analyses(self, analyses: list[PostTradeAnalysis]) -> dict[str, Any]:
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
            "total_issues": issues_count,
        }

    async def _get_quality_trends(self, hours: int) -> list[dict[str, Any]]:
        """Get quality trends for the period."""
        trends = []

        # Get trends for key metrics
        for metric in ["overall_quality_score", "slippage_bps", "execution_time_seconds"]:
            trend = await self.get_quality_trend_analysis(metric, days=hours // 24 or 1)
            trends.append(
                {
                    "metric": metric,
                    "current_value": trend.current_value,
                    "trend_direction": trend.trend_direction,
                    "change_percentage": trend.change_percentage,
                }
            )

        return trends

    async def _get_quality_alerts(self) -> list[dict[str, Any]]:
        """Get active quality alerts."""
        alerts = []

        # Check recent performance against thresholds
        recent_analyses = self.analysis_history[-10:] if self.analysis_history else []

        if recent_analyses:
            avg_quality = sum(a.overall_quality_score for a in recent_analyses) / len(
                recent_analyses
            )
            if avg_quality < self.min_quality_score:
                alerts.append(
                    {
                        "type": "quality_degradation",
                        "severity": "warning",
                        "message": f"Average quality score {avg_quality:.1f} below "
                        f"threshold {self.min_quality_score}",
                    }
                )

            avg_slippage = sum(a.slippage_bps for a in recent_analyses) / len(recent_analyses)
            if avg_slippage > self.slippage_threshold_bps:
                alerts.append(
                    {
                        "type": "high_slippage",
                        "severity": "warning",
                        "message": f"Average slippage {avg_slippage:.1f} bps above "
                        f"threshold {self.slippage_threshold_bps}",
                    }
                )

        return alerts

    async def _get_improvement_recommendations(self) -> list[str]:
        """Get improvement recommendations based on recent performance."""
        recommendations = []

        # Analyze recent performance patterns
        recent_analyses = self.analysis_history[-20:] if self.analysis_history else []

        if recent_analyses:
            # Check for consistent issues
            high_slippage_count = sum(
                1 for a in recent_analyses if a.slippage_bps > self.slippage_threshold_bps
            )
            if high_slippage_count > len(recent_analyses) * 0.3:  # More than 30% of trades
                recommendations.append(
                    "Consider implementing more sophisticated execution algorithms"
                )

            slow_execution_count = sum(
                1
                for a in recent_analyses
                if a.execution_time_seconds > self.execution_time_threshold_seconds
            )
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

    def _check_trend_alerts(self, metric: str, trend: QualityTrend) -> tuple[bool, str]:
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
            elif (
                trend.current_value > self.slippage_threshold_bps
                and trend.trend_direction == "declining"
            ):
                alert_triggered = True
                alert_level = "warning"

        return alert_triggered, alert_level

    async def _load_benchmarks(self) -> None:
        """Load historical benchmarks (using default values)."""
        try:
            self.logger.info("Loading default quality benchmarks")

            # Use default values configured in __init__
            self.logger.info("Quality benchmarks loaded successfully")

        except Exception as e:
            self.logger.warning(f"Failed to load benchmarks: {e}")
            # Continue with default benchmarks on failure

    async def _log_validation_metrics(self, validation: PreTradeValidation) -> None:
        """Log validation metrics using metrics storage abstraction."""
        try:
            # Prepare metrics data
            metrics_data = {
                "validation_id": validation.validation_id,
                "result": validation.overall_result.value,
                "risk_level": validation.risk_level,
                "overall_score": validation.overall_score,
                "risk_score": validation.risk_score,
                "validation_time_ms": validation.validation_time_ms,
                "checks_count": len(validation.checks),
                "timestamp": validation.timestamp,
            }

            # Store using metrics storage interface
            success = await self.metrics_storage.store_validation_metrics(metrics_data)

            if not success:
                self.logger.debug("Metrics storage not available or failed")

        except Exception as e:
            self.logger.warning(f"Failed to log validation metrics: {e}")

    async def _log_analysis_metrics(self, analysis: PostTradeAnalysis) -> None:
        """Log analysis metrics using metrics storage abstraction."""
        try:
            # Prepare metrics data
            metrics_data = {
                "analysis_id": analysis.analysis_id,
                "trade_id": analysis.trade_id,
                "overall_quality_score": analysis.overall_quality_score,
                "execution_quality_score": analysis.execution_quality_score,
                "timing_quality_score": analysis.timing_quality_score,
                "price_quality_score": analysis.price_quality_score,
                "slippage_bps": analysis.slippage_bps,
                "execution_time_seconds": analysis.execution_time_seconds,
                "fill_rate": analysis.fill_rate,
                "market_impact_bps": analysis.market_impact_bps,
                "timestamp": analysis.timestamp,
            }

            # Store using metrics storage interface
            success = await self.metrics_storage.store_analysis_metrics(metrics_data)

            if not success:
                self.logger.debug("Metrics storage not available or failed")

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

    def get_quality_metrics(self) -> dict[str, Any]:
        """
        Get current quality metrics.

        Returns:
            Dictionary containing quality metrics with proper types
        """
        # Return a copy with proper typing for metrics fields
        return {
            "total_validations": self.quality_metrics.get("total_validations", 0),
            "passed_validations": self.quality_metrics.get("passed_validations", 0),
            "failed_validations": self.quality_metrics.get("failed_validations", 0),
            "average_quality_score": self.quality_metrics.get("average_quality_score", 0.0),
            "total_analyses": self.quality_metrics.get("total_analyses", 0),
            "average_execution_quality": self.quality_metrics.get("average_execution_quality", 0.0),
            "slippage_incidents": self.quality_metrics.get("slippage_incidents", 0),
            "execution_time_violations": self.quality_metrics.get("execution_time_violations", 0),
            "avg_validation_time_ms": self._calculate_avg_validation_time(),
            "avg_analysis_time_ms": self._calculate_avg_analysis_time(),
        }

    async def get_summary_statistics(
        self, hours: int = 24, bot_id: str | None = None
    ) -> dict[str, Any]:
        """
        Get summary statistics for quality control.

        Args:
            hours: Time period in hours to analyze
            bot_id: Optional bot ID filter

        Returns:
            Summary statistics dictionary
        """
        try:
            # Calculate time boundaries
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)

            # Filter validation history
            recent_validations = [v for v in self.validation_history if v.timestamp >= start_time]

            # Filter analysis history
            recent_analyses = [a for a in self.analysis_history if a.timestamp >= start_time]

            # Calculate statistics
            total_validations = len(recent_validations)
            passed_validations = sum(
                1 for v in recent_validations if v.overall_result == ValidationResult.PASSED
            )

            total_analyses = len(recent_analyses)
            avg_quality_score = sum(a.overall_quality_score for a in recent_analyses) / max(
                total_analyses, 1
            )

            return {
                "total_validations": total_validations,
                "passed_validations": passed_validations,
                "validation_pass_rate": (passed_validations / max(total_validations, 1)) * 100,
                "total_analyses": total_analyses,
                "average_quality_score": avg_quality_score,
                "period_hours": hours,
                "bot_id": bot_id,
            }

        except Exception as e:
            self.logger.error(f"Failed to get summary statistics: {e}")
            return {
                "total_validations": 0,
                "passed_validations": 0,
                "validation_pass_rate": 0.0,
                "total_analyses": 0,
                "average_quality_score": 0.0,
                "period_hours": hours,
                "bot_id": bot_id,
            }

    def _calculate_avg_validation_time(self) -> float:
        """Calculate average validation time in milliseconds."""
        if not self.validation_history:
            return 0.0

        recent_validations = self.validation_history[-100:]  # Last 100 validations
        total_time = sum(v.validation_time_ms for v in recent_validations)
        return total_time / len(recent_validations)

    def _calculate_avg_analysis_time(self) -> float:
        """Calculate average analysis time in milliseconds."""
        if not self.analysis_history:
            return 0.0

        # Post-trade analysis doesn't track time, so return a default
        return 50.0  # Default 50ms for analysis

    async def validate_state_consistency(self, state: Any) -> bool:
        """
        Validate state consistency.

        Args:
            state: State to validate

        Returns:
            True if state is consistent
        """
        try:
            if not state:
                return False

            # Check if it's a portfolio state with required fields
            if hasattr(state, "total_value") and hasattr(state, "available_cash"):
                # Basic consistency checks for portfolio state
                total_value = getattr(state, "total_value", 0)
                available_cash = getattr(state, "available_cash", 0)
                total_positions_value = getattr(state, "total_positions_value", 0)

                # Total value should be sum of cash and positions
                expected_total = available_cash + total_positions_value
                tolerance = abs(expected_total * Decimal("0.01"))  # 1% tolerance

                if abs(total_value - expected_total) > tolerance:
                    return False

            return True

        except Exception as e:
            self.logger.warning(f"State consistency validation error: {e}")
            return False

    async def validate_portfolio_balance(self, portfolio_state: Any) -> bool:
        """
        Validate portfolio balance.

        Args:
            portfolio_state: Portfolio state to validate

        Returns:
            True if portfolio balance is valid
        """
        try:
            if not portfolio_state:
                return False

            # Check for negative cash
            available_cash = getattr(portfolio_state, "available_cash", 0)
            if available_cash < 0:
                return False

            # Check for reasonable total value
            total_value = getattr(portfolio_state, "total_value", 0)
            if total_value < 0:
                return False

            # Check positions are reasonable
            positions = getattr(portfolio_state, "positions", {})
            if positions:
                for position in positions.values():
                    if hasattr(position, "quantity") and position.quantity <= 0:
                        return False

            return True

        except Exception as e:
            self.logger.warning(f"Portfolio balance validation error: {e}")
            return False

    async def validate_position_consistency(self, position: Any, related_orders: list) -> bool:
        """
        Validate position consistency with related orders.

        Args:
            position: Position to validate
            related_orders: Related orders

        Returns:
            True if position is consistent
        """
        try:
            if not position or not related_orders:
                return True  # No validation needed if no data

            # Check if filled quantity matches position quantity
            total_filled = sum(
                order.filled_quantity
                for order in related_orders
                if hasattr(order, "filled_quantity") and order.filled_quantity > 0
            )

            position_quantity = getattr(position, "quantity", Decimal("0"))
            tolerance = abs(position_quantity * Decimal("0.01"))  # 1% tolerance

            return abs(total_filled - position_quantity) <= tolerance

        except Exception as e:
            self.logger.warning(f"Position consistency validation error: {e}")
            return False

    async def run_integrity_checks(self, state: Any) -> dict[str, Any]:
        """
        Run comprehensive integrity checks.

        Args:
            state: State to check

        Returns:
            Integrity check results
        """
        try:
            results = {"passed_checks": 0, "failed_checks": 0, "warnings": []}

            # Run state consistency check
            if await self.validate_state_consistency(state):
                results["passed_checks"] += 1
            else:
                results["failed_checks"] += 1
                results["warnings"].append("State consistency check failed")

            # Run portfolio balance check if applicable
            if hasattr(state, "available_cash"):
                if await self.validate_portfolio_balance(state):
                    results["passed_checks"] += 1
                else:
                    results["failed_checks"] += 1
                    results["warnings"].append("Portfolio balance check failed")

            return results

        except Exception as e:
            return {
                "passed_checks": 0,
                "failed_checks": 1,
                "warnings": [f"Integrity check error: {e}"],
            }

    async def suggest_corrections(self, state: Any) -> list[dict[str, Any]]:
        """
        Suggest corrections for problematic state.

        Args:
            state: State to analyze

        Returns:
            List of correction suggestions
        """
        try:
            corrections = []

            # Check for negative cash
            if hasattr(state, "available_cash") and state.available_cash < 0:
                corrections.append(
                    {
                        "field": "available_cash",
                        "issue": "Negative cash balance",
                        "description": "Available cash should not be negative",
                        "suggested_action": "Review recent transactions and adjust balance",
                    }
                )

            # Check for inconsistent total value
            if (
                hasattr(state, "total_value")
                and hasattr(state, "available_cash")
                and hasattr(state, "total_positions_value")
            ):
                expected_total = state.available_cash + state.total_positions_value
                if abs(state.total_value - expected_total) > expected_total * Decimal("0.01"):
                    corrections.append(
                        {
                            "field": "total_value",
                            "issue": "Inconsistent total value calculation",
                            "description": "Total value does not match sum of cash and positions",
                            "suggested_action": "Recalculate total value from components",
                        }
                    )

            return corrections

        except Exception as e:
            return [
                {
                    "field": "unknown",
                    "issue": "Analysis error",
                    "description": f"Error analyzing state: {e}",
                    "suggested_action": "Check state data integrity",
                }
            ]

    async def cleanup(self) -> None:
        """Clean up resources used by the quality controller."""
        try:
            # Close metrics storage connection with timeout
            if hasattr(self.metrics_storage, 'close'):
                await asyncio.wait_for(
                    self.metrics_storage.close(),
                    timeout=5.0
                )
            
            self.logger.info("QualityController cleanup completed")
        except asyncio.TimeoutError:
            self.logger.warning("QualityController cleanup timeout")
        except Exception as e:
            self.logger.error(f"Error during QualityController cleanup: {e}")

    def add_validation_rule(self, name: str, rule: callable) -> None:
        """
        Add a custom validation rule.

        Args:
            name: Rule name
            rule: Validation function
        """
        try:
            self.consistency_rules.append({"name": name, "rule": rule, "weight": 10.0})
            self.logger.info(f"Added validation rule: {name}")

        except Exception as e:
            self.logger.error(f"Failed to add validation rule {name}: {e}")
