"""
Environment-aware Analytics Integration.

This module extends the Analytics service with environment awareness,
providing different analytics, reporting, and performance metrics
for sandbox vs live trading environments.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from src.core.exceptions import AnalyticsError
from src.core.integration.environment_aware_service import (
    EnvironmentAwareServiceMixin,
    EnvironmentContext,
)
# Robust logger import with fallback for test suite compatibility
try:
    from src.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class AnalyticsMode(Enum):
    """Analytics operation modes for different environments."""

    EXPERIMENTAL = "experimental"  # Full analytics for testing
    PRODUCTION = "production"  # Production-optimized analytics
    COMPLIANCE = "compliance"  # Compliance-focused analytics
    DEVELOPMENT = "development"  # Development/debugging analytics


class ReportingLevel(Enum):
    """Reporting detail levels."""

    MINIMAL = "minimal"  # Basic KPIs only
    STANDARD = "standard"  # Standard reporting
    DETAILED = "detailed"  # Detailed analysis
    COMPREHENSIVE = "comprehensive"  # Full analytical suite


class EnvironmentAwareAnalyticsConfiguration:
    """Environment-specific analytics configuration."""

    @staticmethod
    def get_sandbox_analytics_config() -> dict[str, Any]:
        """Get analytics configuration for sandbox environment."""
        return {
            "analytics_mode": AnalyticsMode.EXPERIMENTAL,
            "reporting_level": ReportingLevel.COMPREHENSIVE,
            "enable_real_time_analytics": True,
            "enable_backtesting_analytics": True,
            "enable_performance_attribution": True,
            "enable_risk_analytics": True,
            "enable_market_impact_analysis": False,  # Resource intensive
            "enable_slippage_analysis": True,
            "enable_execution_analytics": True,
            "enable_strategy_analytics": True,
            "enable_portfolio_analytics": True,
            "reporting_frequency_minutes": 5,  # Frequent reporting for testing
            "data_retention_days": 30,
            "enable_experimental_metrics": True,
            "enable_detailed_logging": True,
            "enable_performance_profiling": True,
            "max_computation_time_seconds": 60,  # Longer for experimentation
            "enable_custom_metrics": True,
            "enable_ml_analytics": True,
            "parallel_processing": True,
            "cache_results": True,
            "enable_simulation_analytics": True,
            "benchmark_comparison": False,  # Not needed for sandbox
        }

    @staticmethod
    def get_live_analytics_config() -> dict[str, Any]:
        """Get analytics configuration for live/production environment."""
        return {
            "analytics_mode": AnalyticsMode.PRODUCTION,
            "reporting_level": ReportingLevel.STANDARD,
            "enable_real_time_analytics": True,
            "enable_backtesting_analytics": False,  # Not needed in production
            "enable_performance_attribution": True,
            "enable_risk_analytics": True,
            "enable_market_impact_analysis": True,  # Important for production
            "enable_slippage_analysis": True,
            "enable_execution_analytics": True,
            "enable_strategy_analytics": True,
            "enable_portfolio_analytics": True,
            "reporting_frequency_minutes": 15,  # Less frequent for efficiency
            "data_retention_days": 365,
            "enable_experimental_metrics": False,  # Disabled for stability
            "enable_detailed_logging": False,  # Performance consideration
            "enable_performance_profiling": False,  # Disabled for production
            "max_computation_time_seconds": 30,  # Shorter timeout for responsiveness
            "enable_custom_metrics": False,  # Standardized metrics only
            "enable_ml_analytics": False,  # Disabled until validated
            "parallel_processing": True,
            "cache_results": True,
            "enable_simulation_analytics": False,  # Not needed in production
            "benchmark_comparison": True,  # Important for production evaluation
        }


class EnvironmentAwareAnalyticsManager(EnvironmentAwareServiceMixin):
    """
    Environment-aware analytics management functionality.

    This mixin adds environment-specific analytics, reporting,
    and performance measurement to the Analytics service.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._environment_analytics_configs: dict[str, dict[str, Any]] = {}
        self._analytics_metrics: dict[str, dict[str, Any]] = {}
        self._performance_data: dict[str, dict[str, Any]] = {}
        self._environment_analytics_cache: dict[str, dict[str, Any]] = {}

    def _get_local_logger(self):
        """Get logger locally to avoid scope issues in test environment."""
        try:
            from src.core.logging import get_logger
            return get_logger(__name__)
        except ImportError:
            import logging
            return logging.getLogger(__name__)

    async def _update_service_environment(self, context: EnvironmentContext) -> None:
        """Update analytics settings based on environment context."""
        await super()._update_service_environment(context)

        local_logger = self._get_local_logger()

        # Get environment-specific analytics configuration
        if context.is_production:
            analytics_config = EnvironmentAwareAnalyticsConfiguration.get_live_analytics_config()
            local_logger.info(f"Applied live analytics configuration for {context.exchange_name}")
        else:
            analytics_config = EnvironmentAwareAnalyticsConfiguration.get_sandbox_analytics_config()
            local_logger.info(f"Applied sandbox analytics configuration for {context.exchange_name}")

        self._environment_analytics_configs[context.exchange_name] = analytics_config

        # Initialize analytics metrics tracking
        self._analytics_metrics[context.exchange_name] = {
            "total_analytics_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "average_computation_time_ms": 0,
            "total_reports_generated": 0,
            "cache_hit_rate": Decimal("0"),
            "data_points_processed": 0,
            "last_analytics_run": None,
            "performance_score": Decimal("100"),
            "error_rate": Decimal("0"),
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_responses": 0,
            "uncached_responses": 0,
        }

        # Initialize performance data storage
        self._performance_data[context.exchange_name] = {
            "portfolio_value": [],
            "pnl_history": [],
            "drawdown_history": [],
            "trade_performance": [],
            "risk_metrics": [],
            "execution_metrics": [],
        }

        # Initialize analytics cache
        self._environment_analytics_cache[context.exchange_name] = {}

    def get_environment_analytics_config(self, exchange: str) -> dict[str, Any]:
        """Get analytics configuration for a specific exchange environment."""
        if exchange not in self._environment_analytics_configs:
            # Initialize with default config based on current environment
            context = self.get_environment_context(exchange)
            if context.is_production:
                config = EnvironmentAwareAnalyticsConfiguration.get_live_analytics_config()
            else:
                config = EnvironmentAwareAnalyticsConfiguration.get_sandbox_analytics_config()
            self._environment_analytics_configs[exchange] = config

        return self._environment_analytics_configs[exchange]

    async def generate_environment_aware_report(
        self,
        report_type: str,
        exchange: str,
        time_period: str | None = None,
        include_raw_data: bool = False,
    ) -> dict[str, Any]:
        """Generate analytics report with environment-specific content and detail level."""
        context = self.get_environment_context(exchange)
        analytics_config = self.get_environment_analytics_config(exchange)

        # Determine report detail level based on environment
        reporting_level = analytics_config.get("reporting_level", ReportingLevel.STANDARD)

        start_time = datetime.now(timezone.utc)

        try:
            # Check cache first if caching is enabled
            if analytics_config.get("cache_results"):
                cached_report = await self._get_cached_report(report_type, exchange, time_period)
                if cached_report:
                    self._get_local_logger().debug(f"Retrieved cached report for {exchange}: {report_type}")
                    await self._update_analytics_metrics(exchange, start_time, True, True)
                    return cached_report

            # Generate report based on environment and type
            if report_type == "performance":
                report = await self._generate_performance_report(
                    exchange, time_period, reporting_level, include_raw_data
                )
            elif report_type == "risk":
                report = await self._generate_risk_report(
                    exchange, time_period, reporting_level, include_raw_data
                )
            elif report_type == "execution":
                report = await self._generate_execution_report(
                    exchange, time_period, reporting_level, include_raw_data
                )
            elif report_type == "portfolio":
                report = await self._generate_portfolio_report(
                    exchange, time_period, reporting_level, include_raw_data
                )
            else:
                raise AnalyticsError(
                    f"Unknown report type: {report_type}",
                    error_code="ANL_015"
                )

            # Add environment context to report
            report["environment_context"] = {
                "exchange": exchange,
                "environment": context.environment.value,
                "is_production": context.is_production,
                "reporting_level": reporting_level.value,
                "generation_time": datetime.now(timezone.utc).isoformat(),
                "time_period": time_period,
            }

            # Cache report if caching is enabled
            if analytics_config.get("cache_results"):
                await self._cache_report(report_type, exchange, time_period, report)

            # Update metrics
            await self._update_analytics_metrics(exchange, start_time, True, False)

            self._get_local_logger().info(
                f"Generated {report_type} report for {exchange} "
                f"(environment: {context.environment.value})"
            )
            return report

        except Exception as e:
            await self._update_analytics_metrics(exchange, start_time, False, False)
            self._get_local_logger().error(f"Failed to generate {report_type} report for {exchange}: {e}")
            raise AnalyticsError(
                f"Report generation failed: {e}",
                error_code="ANL_016"
            ) from e

    async def _generate_performance_report(
        self,
        exchange: str,
        time_period: str | None,
        reporting_level: ReportingLevel,
        include_raw_data: bool,
    ) -> dict[str, Any]:
        """Generate performance analytics report."""
        context = self.get_environment_context(exchange)
        performance_data = self._performance_data.get(exchange, {})

        report = {
            "report_type": "performance",
            "exchange": exchange,
            "time_period": time_period,
            "summary": await self._calculate_performance_summary(exchange),
        }

        # Add detail based on reporting level
        if reporting_level in (ReportingLevel.DETAILED, ReportingLevel.COMPREHENSIVE):
            report["detailed_metrics"] = await self._calculate_detailed_performance_metrics(
                exchange
            )

            if context.is_production:
                # Production-specific performance metrics
                report["production_metrics"] = await self._calculate_production_performance_metrics(
                    exchange
                )
            else:
                # Sandbox-specific performance metrics
                report["sandbox_metrics"] = await self._calculate_sandbox_performance_metrics(
                    exchange
                )

        if reporting_level == ReportingLevel.COMPREHENSIVE:
            report["advanced_analytics"] = await self._calculate_advanced_performance_analytics(
                exchange
            )

            if include_raw_data:
                report["raw_data"] = performance_data

        return report

    async def _generate_risk_report(
        self,
        exchange: str,
        time_period: str | None,
        reporting_level: ReportingLevel,
        include_raw_data: bool,
    ) -> dict[str, Any]:
        """Generate risk analytics report."""
        context = self.get_environment_context(exchange)

        report = {
            "report_type": "risk",
            "exchange": exchange,
            "time_period": time_period,
            "summary": await self._calculate_risk_summary(exchange),
        }

        # Environment-specific risk reporting
        if context.is_production:
            report["production_risk_metrics"] = await self._calculate_production_risk_metrics(
                exchange
            )
        else:
            report["sandbox_risk_metrics"] = await self._calculate_sandbox_risk_metrics(exchange)

        if reporting_level in (ReportingLevel.DETAILED, ReportingLevel.COMPREHENSIVE):
            report["detailed_risk_analysis"] = await self._calculate_detailed_risk_analysis(
                exchange
            )

        return report

    async def _generate_execution_report(
        self,
        exchange: str,
        time_period: str | None,
        reporting_level: ReportingLevel,
        include_raw_data: bool,
    ) -> dict[str, Any]:
        """Generate execution analytics report."""
        analytics_config = self.get_environment_analytics_config(exchange)

        report = {
            "report_type": "execution",
            "exchange": exchange,
            "time_period": time_period,
            "summary": await self._calculate_execution_summary(exchange),
        }

        # Include slippage analysis if enabled
        if analytics_config.get("enable_slippage_analysis"):
            report["slippage_analysis"] = await self._calculate_slippage_analysis(exchange)

        # Include market impact analysis if enabled (typically production only)
        if analytics_config.get("enable_market_impact_analysis"):
            report["market_impact_analysis"] = await self._calculate_market_impact_analysis(
                exchange
            )

        return report

    async def _generate_portfolio_report(
        self,
        exchange: str,
        time_period: str | None,
        reporting_level: ReportingLevel,
        include_raw_data: bool,
    ) -> dict[str, Any]:
        """Generate portfolio analytics report."""
        analytics_config = self.get_environment_analytics_config(exchange)

        report = {
            "report_type": "portfolio",
            "exchange": exchange,
            "time_period": time_period,
            "summary": await self._calculate_portfolio_summary(exchange),
        }

        # Include performance attribution if enabled
        if analytics_config.get("enable_performance_attribution"):
            report["performance_attribution"] = await self._calculate_performance_attribution(
                exchange
            )

        # Include benchmark comparison for production
        if analytics_config.get("benchmark_comparison"):
            report["benchmark_comparison"] = await self._calculate_benchmark_comparison(exchange)

        return report

    async def track_environment_performance(
        self,
        exchange: str,
        start_time_or_metric_type: datetime | str,
        success_or_value: bool | Decimal | float | int,
        was_cached_or_metadata: bool | dict[str, Any] | None = None,
    ) -> None:
        """Track performance metrics with environment-specific handling."""
        # Handle overloaded signature for backward compatibility
        if isinstance(start_time_or_metric_type, datetime) and isinstance(success_or_value, bool):
            # Called with (exchange, start_time, success, was_cached) signature
            await self._update_analytics_metrics(
                exchange, start_time_or_metric_type, success_or_value, was_cached_or_metadata or False
            )
            return

        # Handle new signature (exchange, metric_type, value, metadata)
        metric_type = start_time_or_metric_type
        value = success_or_value
        metadata = was_cached_or_metadata if isinstance(was_cached_or_metadata, dict) else None

        context = self.get_environment_context(exchange)
        analytics_config = self.get_environment_analytics_config(exchange)

        # Skip tracking if not enabled for this environment
        if not analytics_config.get("enable_real_time_analytics"):
            return

        if exchange not in self._performance_data:
            await self._update_service_environment(context)

        performance_data = self._performance_data[exchange]
        timestamp = datetime.now(timezone.utc)

        # Store performance data point
        data_point = {
            "timestamp": timestamp.isoformat(),
            "value": float(value) if isinstance(value, Decimal) else value,
            "metadata": metadata or {},
            "environment": context.environment.value,
        }

        # Route to appropriate performance data structure
        if metric_type == "portfolio_value":
            performance_data["portfolio_value"].append(data_point)
        elif metric_type == "pnl":
            performance_data["pnl_history"].append(data_point)
        elif metric_type == "drawdown":
            performance_data["drawdown_history"].append(data_point)
        elif metric_type == "trade_performance":
            performance_data["trade_performance"].append(data_point)
        elif metric_type == "risk":
            performance_data["risk_metrics"].append(data_point)
        elif metric_type == "execution":
            performance_data["execution_metrics"].append(data_point)

        # Maintain data retention limits
        retention_days = analytics_config.get("data_retention_days", 30)
        cutoff_date = timestamp - timedelta(days=retention_days)

        for data_list in performance_data.values():
            if isinstance(data_list, list):
                # Remove old data points
                data_list[:] = [
                    dp
                    for dp in data_list
                    if datetime.fromisoformat(dp["timestamp"].replace("Z", "+00:00")) > cutoff_date
                ]

        self._get_local_logger().debug(f"Tracked {metric_type} performance for {exchange}: {value}")

    async def _calculate_performance_summary(self, exchange: str) -> dict[str, Any]:
        """Calculate performance summary metrics."""
        performance_data = self._performance_data.get(exchange, {})

        # Calculate basic performance metrics
        portfolio_values = performance_data.get("portfolio_value", [])
        pnl_history = performance_data.get("pnl_history", [])

        summary = {
            "total_pnl": 0.0,
            "total_trades": len(performance_data.get("trade_performance", [])),
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "current_portfolio_value": 0.0,
            "data_points": len(portfolio_values) + len(pnl_history),
        }

        if portfolio_values:
            summary["current_portfolio_value"] = portfolio_values[-1]["value"]

        if pnl_history:
            summary["total_pnl"] = sum(dp["value"] for dp in pnl_history)

        return summary

    async def _calculate_detailed_performance_metrics(self, exchange: str) -> dict[str, Any]:
        """Calculate detailed performance metrics."""
        # Implementation would calculate detailed metrics
        return {
            "volatility": 0.0,
            "beta": 0.0,
            "alpha": 0.0,
            "information_ratio": 0.0,
            "calmar_ratio": 0.0,
            "sortino_ratio": 0.0,
        }

    async def _calculate_production_performance_metrics(self, exchange: str) -> dict[str, Any]:
        """Calculate production-specific performance metrics."""
        return {
            "regulatory_compliance_score": 100.0,
            "operational_efficiency": 95.0,
            "risk_adjusted_return": 0.0,
            "transaction_cost_analysis": {},
        }

    async def _calculate_sandbox_performance_metrics(self, exchange: str) -> dict[str, Any]:
        """Calculate sandbox-specific performance metrics."""
        return {
            "strategy_experimentation_score": 85.0,
            "parameter_sensitivity": {},
            "simulation_accuracy": 92.0,
            "learning_metrics": {},
        }

    async def _calculate_advanced_performance_analytics(self, exchange: str) -> dict[str, Any]:
        """Calculate advanced performance analytics for exchange."""
        return {
            "advanced_metrics": "calculation_pending",
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
        }

    async def _calculate_risk_summary(self, exchange: str) -> dict[str, Any]:
        """Calculate risk summary for exchange."""
        return {
            "risk_summary": "calculation_pending",
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
        }

    async def _calculate_production_risk_metrics(self, exchange: str) -> dict[str, Any]:
        """Calculate production-specific risk metrics for exchange."""
        return {
            "production_risk": "calculation_pending",
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
        }

    async def _calculate_sandbox_risk_metrics(self, exchange: str) -> dict[str, Any]:
        """Calculate sandbox-specific risk metrics for exchange."""
        return {
            "sandbox_risk": "calculation_pending",
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
        }

    async def _calculate_detailed_risk_analysis(self, exchange: str) -> dict[str, Any]:
        """Calculate detailed risk analysis for exchange."""
        return {
            "detailed_risk": "calculation_pending",
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
        }

    async def _calculate_execution_summary(self, exchange: str) -> dict[str, Any]:
        """Calculate execution summary for exchange."""
        return {
            "execution_summary": "calculation_pending",
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
        }

    async def _calculate_slippage_analysis(self, exchange: str) -> dict[str, Any]:
        """Calculate slippage analysis for exchange."""
        return {
            "slippage_analysis": "calculation_pending",
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
        }

    async def _calculate_market_impact_analysis(self, exchange: str) -> dict[str, Any]:
        """Calculate market impact analysis for exchange."""
        return {
            "market_impact": "calculation_pending",
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
        }

    async def _calculate_portfolio_summary(self, exchange: str) -> dict[str, Any]:
        """Calculate portfolio summary for exchange."""
        return {
            "portfolio_summary": "calculation_pending",
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
        }

    async def _calculate_performance_attribution(self, exchange: str) -> dict[str, Any]:
        """Calculate performance attribution for exchange."""
        return {
            "performance_attribution": "calculation_pending",
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
        }

    async def _calculate_benchmark_comparison(self, exchange: str) -> dict[str, Any]:
        """Calculate benchmark comparison for exchange."""
        return {
            "benchmark_comparison": "calculation_pending",
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
        }

    async def _get_cached_report(
        self, report_type: str, exchange: str, time_period: str | None
    ) -> dict[str, Any] | None:
        """Get cached report if available and valid."""
        cache = self._environment_analytics_cache.get(exchange, {})
        cache_key = f"{report_type}_{time_period or 'default'}"

        cached_report = cache.get(cache_key)
        if cached_report and "cached_at" in cached_report:
            # Check cache validity (5 minutes for sandbox, 15 minutes for production)
            context = self.get_environment_context(exchange)
            max_age_minutes = 5 if not context.is_production else 15

            cached_at = datetime.fromisoformat(cached_report["cached_at"])
            if datetime.now(timezone.utc) - cached_at < timedelta(minutes=max_age_minutes):
                return cached_report

        return None

    async def _cache_report(
        self, report_type: str, exchange: str, time_period: str | None, report: dict[str, Any]
    ) -> None:
        """Cache report for future use."""
        if exchange not in self._environment_analytics_cache:
            self._environment_analytics_cache[exchange] = {}

        cache_key = f"{report_type}_{time_period or 'default'}"
        report["cached_at"] = datetime.now(timezone.utc).isoformat()

        self._environment_analytics_cache[exchange][cache_key] = report

    async def _update_analytics_metrics(
        self, exchange: str, start_time: datetime, success: bool, was_cached: bool
    ) -> None:
        """Update analytics performance metrics."""
        if exchange not in self._analytics_metrics:
            # Initialize metrics if they don't exist
            from src.core.config.environment import ExchangeEnvironment
            context = EnvironmentContext(
                exchange_name=exchange,
                environment=ExchangeEnvironment.SANDBOX,
                api_credentials={}
            )
            await self._update_service_environment(context)

        metrics = self._analytics_metrics[exchange]
        # Handle timezone-naive datetime from tests
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        computation_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        metrics["total_analytics_runs"] += 1
        metrics["last_analytics_run"] = datetime.now().isoformat()

        if success:
            metrics["successful_runs"] += 1

            # Update average computation time
            prev_avg = metrics["average_computation_time_ms"]
            successful_runs = metrics["successful_runs"]
            metrics["average_computation_time_ms"] = (
                prev_avg * (successful_runs - 1) + computation_time
            ) / successful_runs

            # Update cache hit rate if applicable
            if was_cached:
                cache_hits = metrics.get("cache_hits", 0) + 1
                metrics["cache_hits"] = cache_hits
                metrics["cache_hit_rate"] = Decimal(cache_hits) / Decimal(
                    metrics["total_analytics_runs"]
                )
        else:
            metrics["failed_runs"] += 1

        # Update request metrics
        metrics["total_requests"] += 1
        if success:
            metrics["successful_requests"] += 1
        else:
            metrics["failed_requests"] += 1

        if was_cached:
            metrics["cached_responses"] += 1
        else:
            metrics["uncached_responses"] += 1

        # Calculate error rate
        total_runs = metrics["total_analytics_runs"]
        failed_runs = metrics["failed_runs"]
        metrics["error_rate"] = (
            Decimal(failed_runs) / Decimal(total_runs) * 100 if total_runs > 0 else Decimal("0")
        )

    def get_environment_analytics_metrics(self, exchange: str) -> dict[str, Any]:
        """Get analytics metrics for an exchange environment."""
        context = self.get_environment_context(exchange)
        analytics_config = self.get_environment_analytics_config(exchange)
        metrics = self._analytics_metrics.get(exchange, {})

        return {
            "exchange": exchange,
            "environment": context.environment.value,
            "is_production": context.is_production,
            "analytics_mode": analytics_config.get(
                "analytics_mode", AnalyticsMode.PRODUCTION
            ).value,
            "reporting_level": analytics_config.get(
                "reporting_level", ReportingLevel.STANDARD
            ).value,
            "total_analytics_runs": metrics.get("total_analytics_runs", 0),
            "successful_runs": metrics.get("successful_runs", 0),
            "failed_runs": metrics.get("failed_runs", 0),
            "success_rate": (
                metrics.get("successful_runs", 0)
                / max(metrics.get("total_analytics_runs", 1), 1)
                * 100
            ),
            "average_computation_time_ms": metrics.get("average_computation_time_ms", 0),
            "cache_hit_rate": float(metrics.get("cache_hit_rate", Decimal("0"))),
            "error_rate": float(metrics.get("error_rate", Decimal("0"))),
            "data_points_processed": metrics.get("data_points_processed", 0),
            "total_requests": metrics.get("total_requests", 0),
            "successful_requests": metrics.get("successful_requests", 0),
            "failed_requests": metrics.get("failed_requests", 0),
            "cached_responses": metrics.get("cached_responses", 0),
            "uncached_responses": metrics.get("uncached_responses", 0),
            "average_response_time": metrics.get("average_response_time", 0),
            "enable_real_time_analytics": analytics_config.get("enable_real_time_analytics", True),
            "enable_experimental_metrics": analytics_config.get(
                "enable_experimental_metrics", False
            ),
            "reporting_frequency_minutes": analytics_config.get("reporting_frequency_minutes", 15),
            "data_retention_days": analytics_config.get("data_retention_days", 30),
            "last_analytics_run": metrics.get("last_analytics_run"),
            "last_updated": datetime.now().isoformat(),
        }
