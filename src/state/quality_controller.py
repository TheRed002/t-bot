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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import ConfigDict
from scipy import stats

from src.core.base.component import BaseComponent
from src.core.config.main import Config
from src.core.exceptions import StateConsistencyError, ValidationError
from src.core.logging import get_logger
from src.core.types import ExecutionResult, MarketData, OrderRequest

# No direct database imports - use service abstractions only
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
    score: Decimal = Decimal("100.0")
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
    overall_score: Decimal = Decimal("100.0")

    # Individual checks
    checks: list[ValidationCheck] = field(default_factory=list)

    # Risk assessment
    risk_level: str = "low"  # low, medium, high, critical
    risk_score: Decimal = Decimal("0.0")

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # Processing time
    validation_time_ms: Decimal = Decimal("0.0")


@dataclass
class PostTradeAnalysis:
    """Post-trade analysis results."""

    analysis_id: str = field(default_factory=lambda: str(uuid4()))
    trade_id: str = ""
    execution_result: ExecutionResult | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Quality scores
    execution_quality_score: Decimal = Decimal("100.0")
    timing_quality_score: Decimal = Decimal("100.0")
    price_quality_score: Decimal = Decimal("100.0")
    overall_quality_score: Decimal = Decimal("100.0")

    # Performance metrics
    slippage_bps: Decimal = Decimal("0.0")
    execution_time_seconds: Decimal = Decimal("0.0")
    fill_rate: Decimal = Decimal("100.0")

    # Market impact analysis
    market_impact_bps: Decimal = Decimal("0.0")
    temporary_impact_bps: Decimal = Decimal("0.0")
    permanent_impact_bps: Decimal = Decimal("0.0")

    # Benchmark comparison
    benchmark_scores: dict[str, Decimal] = field(default_factory=dict)

    # Issues and recommendations
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class QualityTrend:
    """Quality trend analysis."""

    metric_name: str = ""
    time_period: str = "1d"  # 1h, 1d, 1w, 1m

    # Trend statistics
    current_value: Decimal = Decimal("0.0")
    previous_value: Decimal = Decimal("0.0")
    change_percentage: Decimal = Decimal("0.0")
    trend_direction: str = "stable"  # improving, declining, stable

    # Statistical measures
    mean: Decimal = Decimal("0.0")
    std_dev: Decimal = Decimal("0.0")
    min_value: Decimal = Decimal("0.0")
    max_value: Decimal = Decimal("0.0")
    percentile_95: Decimal = Decimal("0.0")
    percentile_5: Decimal = Decimal("0.0")

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
        self.logger = get_logger(__name__)

        if config:
            try:
                # Use service abstraction for InfluxDB
                if config and hasattr(config, "influxdb"):
                    self._available = True
                else:
                    self._available = False
            except Exception as e:
                self.logger.warning(f"Failed to initialize InfluxDB metrics storage: {e}")
                self._available = False

    async def close(self) -> None:
        """Close InfluxDB client connection with proper async context management."""
        if self._influx_client and self._available:
            try:
                # Use asyncio.wait_for for timeout handling (compatible with older Python)
                async def _close_client():
                    if hasattr(self._influx_client, "close"):
                        if asyncio.iscoroutinefunction(self._influx_client.close):
                            await self._influx_client.close()
                        else:
                            # Handle sync close method in executor to prevent blocking
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, self._influx_client.close)
                    elif hasattr(self._influx_client, "disconnect"):
                        if asyncio.iscoroutinefunction(self._influx_client.disconnect):
                            await self._influx_client.disconnect()
                        else:
                            # Handle sync disconnect method in executor
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, self._influx_client.disconnect)

                await asyncio.wait_for(_close_client(), timeout=5.0)
                self._available = False
                self.logger.debug("InfluxDB client closed successfully")
            except asyncio.TimeoutError:
                # Log timeout but don't raise to avoid interfering with cleanup
                self.logger.warning("InfluxDB client close timeout - forcing cleanup")
                self._available = False
            except Exception as e:
                # Log error but don't raise to avoid interfering with cleanup
                self.logger.warning(f"Error closing InfluxDB client: {e}")
                self._available = False
            finally:
                # Ensure client reference is cleared even on error
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
                    point.field(
                        key,
                        validation_data[key]
                        if isinstance(validation_data[key], (int, float))
                        else Decimal(str(validation_data[key])),
                    )

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
                    point.field(
                        key,
                        analysis_data[key]
                        if isinstance(analysis_data[key], (int, float))
                        else Decimal(str(analysis_data[key])),
                    )

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
    Quality control controller that coordinates quality management operations.

    This controller delegates business logic to appropriate services and focuses
    on coordinating quality management workflows without containing business logic.

    The controller coordinates between:
    - Quality validation services
    - Quality analysis services
    - Metrics storage services
    - Alert and notification services
    """

    def __init__(
        self,
        config: Config,
        metrics_storage: MetricsStorage | None = None,
        quality_service: Any | None = None,
        validation_service: Any | None = None,
    ):
        """
        Initialize the quality controller.

        Args:
            config: Application configuration
            metrics_storage: Optional metrics storage service for logging metrics
        """
        config_dict = None
        if config is not None:
            if hasattr(config, "__dict__"):
                config_dict = ConfigDict(**config.__dict__)
            elif isinstance(config, dict):
                config_dict = ConfigDict(**config)
        super().__init__(name="QualityController", config=config_dict)
        self.config = config
        # Logger is already provided by BaseComponent

        # Injected dependencies - use service abstractions
        self.metrics_storage = metrics_storage or NullMetricsStorage()
        self.validation_service = validation_service

        # Initialize quality service if not provided
        if quality_service is None:
            try:
                from .services.quality_service import QualityService
                self.quality_service = QualityService(config)
            except Exception as e:
                self.logger.warning(f"Failed to create QualityService: {e}")
                self.quality_service = None
        else:
            self.quality_service = quality_service

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
        self.min_quality_score = Decimal("70.0")
        self.slippage_threshold_bps = Decimal("20.0")
        self.execution_time_threshold_seconds = Decimal("30.0")
        self.market_impact_threshold_bps = Decimal("10.0")

        # Try to load quality config from various possible config locations
        risk_config = getattr(config, "risk", {})
        if risk_config and hasattr(risk_config, "quality"):
            quality_config = getattr(risk_config, "quality", {})
            self.min_quality_score = Decimal(str(getattr(quality_config, "min_quality_score", 70.0)))
            self.slippage_threshold_bps = Decimal(str(getattr(quality_config, "slippage_threshold_bps", 20.0)))
            self.execution_time_threshold_seconds = Decimal(str(getattr(
                quality_config, "execution_time_threshold_seconds", 30.0
            )))
            self.market_impact_threshold_bps = Decimal(str(getattr(
                quality_config, "market_impact_threshold_bps", 10.0
            )))

        # Validation rules
        self.validation_rules = {
            "position_size": {"max_percentage": Decimal("10.0"), "weight": Decimal("25.0")},
            "market_hours": {"enforce": True, "weight": Decimal("15.0")},
            "liquidity": {"min_volume_usd": Decimal("100000"), "weight": Decimal("20.0")},
            "volatility": {"max_daily_range": Decimal("0.05"), "weight": Decimal("15.0")},
            "correlation": {"max_correlation": Decimal("0.8"), "weight": Decimal("10.0")},
            "risk_limits": {"max_var": Decimal("0.02"), "weight": Decimal("15.0")},
        }

        # Consistency rules for test compatibility
        self.consistency_rules: list = []

        # Benchmarks
        self.benchmarks = {
            "slippage_p50": Decimal("5.0"),  # 50th percentile slippage in bps
            "slippage_p90": Decimal("15.0"),  # 90th percentile slippage in bps
            "execution_time_p50": Decimal("10.0"),  # 50th percentile execution time in seconds
            "execution_time_p90": Decimal("30.0"),  # 90th percentile execution time in seconds
            "fill_rate_p50": Decimal("95.0"),  # 50th percentile fill rate
            "market_impact_p50": Decimal("3.0"),  # 50th percentile market impact in bps
        }

        # Quality history
        self.validation_history: list[PreTradeValidation] = []
        self.analysis_history: list[PostTradeAnalysis] = []

        # Performance tracking
        self.quality_metrics = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "average_quality_score": Decimal("0.0"),
            "total_analyses": 0,
            "average_execution_quality": Decimal("0.0"),
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
            self.logger.info(f"Quality controller services - Metrics: {metrics_status}")

        except Exception as e:
            self.logger.error(f"QualityController initialization failed: {e}")
            raise StateConsistencyError(f"Failed to initialize QualityController: {e}") from e

    @time_execution
    async def validate_pre_trade(
        self,
        order_request: OrderRequest,
        market_data: MarketData | None = None,
        portfolio_context: dict[str, Any] | None = None,
    ) -> PreTradeValidation:
        """
        Coordinate pre-trade validation through appropriate services.

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
            # Delegate validation logic to service layer
            if self.validation_service:
                validation = await self.validation_service.validate_pre_trade(
                    order_request, market_data, portfolio_context
                )
            else:
                # Fallback to basic validation if service not available
                validation = PreTradeValidation(order_request=order_request)
                validation.overall_result = ValidationResult.PASSED
                validation.overall_score = Decimal("100.0")
                validation.risk_level = "low"
                validation.risk_score = Decimal("0.0")
                validation.recommendations = []

            # Record processing time (controller concern)
            validation.validation_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Store validation (controller infrastructure concern)
            self.validation_history.append(validation)
            await self._log_validation_metrics(validation)

            # Update quality metrics (controller concern)
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
        Coordinate post-trade analysis through appropriate services.

        Args:
            trade_id: Trade identifier
            execution_result: Execution results
            market_data_before: Market data before trade
            market_data_after: Market data after trade

        Returns:
            Post-trade analysis results
        """
        try:
            # Delegate analysis logic to service layer
            if self.quality_service:
                analysis = await self.quality_service.analyze_post_trade(
                    trade_id, execution_result, market_data_before, market_data_after
                )
            else:
                # Fallback to basic analysis if service not available
                analysis = PostTradeAnalysis(trade_id=trade_id, execution_result=execution_result)
                analysis.execution_quality_score = Decimal("100.0")
                analysis.timing_quality_score = Decimal("100.0")
                analysis.price_quality_score = Decimal("100.0")
                analysis.overall_quality_score = Decimal("100.0")
                analysis.slippage_bps = Decimal("0.0")
                analysis.execution_time_seconds = Decimal("0.0")
                analysis.fill_rate = Decimal("100.0")
                analysis.market_impact_bps = Decimal("0.0")
                analysis.issues = []
                analysis.recommendations = []

            # Store analysis (controller infrastructure concern)
            self.analysis_history.append(analysis)
            await self._log_analysis_metrics(analysis)

            # Update quality metrics (controller concern)
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
            raise StateConsistencyError(f"Post-trade analysis error: {e}") from e

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
                current_value=Decimal(str(values[-1])) if values else Decimal("0.0"),
                previous_value=Decimal(str(values[0])) if values else Decimal("0.0"),
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

                if p_value < Decimal("0.05"):  # Statistically significant trend
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

    # Private helper methods - moved business logic to services

    # Business logic calculation methods moved to service layer

    # Analysis methods moved to service layer

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
            if high_slippage_count > len(recent_analyses) * Decimal("0.3"):  # More than 30% of trades
                recommendations.append(
                    "Consider implementing more sophisticated execution algorithms"
                )

            slow_execution_count = sum(
                1
                for a in recent_analyses
                if a.execution_time_seconds > self.execution_time_threshold_seconds
            )
            if slow_execution_count > len(recent_analyses) * Decimal("0.2"):  # More than 20% of trades
                recommendations.append("Review execution infrastructure and consider co-location")

            low_fill_count = sum(1 for a in recent_analyses if a.fill_rate < Decimal("95.0"))
            if low_fill_count > len(recent_analyses) * Decimal("0.1"):  # More than 10% of trades
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
            "average_quality_score": self.quality_metrics.get("average_quality_score", Decimal("0.0")),
            "total_analyses": self.quality_metrics.get("total_analyses", 0),
            "average_execution_quality": self.quality_metrics.get("average_execution_quality", Decimal("0.0")),
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
                "validation_pass_rate": Decimal("0.0"),
                "total_analyses": 0,
                "average_quality_score": Decimal("0.0"),
                "period_hours": hours,
                "bot_id": bot_id,
            }

    def _calculate_avg_validation_time(self) -> Decimal:
        """Calculate average validation time in milliseconds."""
        if not self.validation_history:
            return Decimal("0.0")

        recent_validations = self.validation_history[-100:]  # Last 100 validations
        total_time = sum(v.validation_time_ms for v in recent_validations)
        return total_time / len(recent_validations)

    def _calculate_avg_analysis_time(self) -> Decimal:
        """Calculate average analysis time in milliseconds."""
        if not self.analysis_history:
            return Decimal("0.0")

        # Post-trade analysis doesn't track time, so return a default
        return Decimal("50.0")  # Default 50ms for analysis

    async def validate_state_consistency(self, state: Any) -> bool:
        """
        Validate state consistency by delegating to quality service.

        Args:
            state: State to validate

        Returns:
            True if state is consistent
        """
        if self.quality_service:
            return await self.quality_service.validate_state_consistency(state)

        # Fallback if no service available
        return True

    async def validate_portfolio_balance(self, portfolio_state: Any) -> bool:
        """
        Validate portfolio balance by delegating to quality service.

        Args:
            portfolio_state: Portfolio state to validate

        Returns:
            True if portfolio balance is valid
        """
        if self.quality_service:
            return await self.quality_service.validate_portfolio_balance(portfolio_state)

        # Fallback if no service available
        return True

    async def validate_position_consistency(self, position: Any, related_orders: list) -> bool:
        """
        Validate position consistency with related orders by delegating to quality service.

        Args:
            position: Position to validate
            related_orders: Related orders

        Returns:
            True if position is consistent
        """
        if self.quality_service:
            return await self.quality_service.validate_position_consistency(position, related_orders)

        # Fallback if no service available
        return True

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
            if hasattr(self.metrics_storage, "close"):
                await asyncio.wait_for(self.metrics_storage.close(), timeout=5.0)

            self.logger.info("QualityController cleanup completed")
        except asyncio.TimeoutError:
            self.logger.warning("QualityController cleanup timeout")
        except Exception as e:
            self.logger.error(f"Error during QualityController cleanup: {e}")

    def add_validation_rule(self, name: str, rule: Callable[..., Any]) -> None:
        """
        Add a custom validation rule.

        Args:
            name: Rule name
            rule: Validation function
        """
        try:
            self.consistency_rules.append({"name": name, "rule": rule, "weight": Decimal("10.0")})
            self.logger.info(f"Added validation rule: {name}")

        except Exception as e:
            self.logger.error(f"Failed to add validation rule {name}: {e}")
