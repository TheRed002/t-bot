"""
Reporting Service.

This service provides a proper service layer implementation for performance reporting,
following service layer patterns and using dependency injection.
"""

from datetime import datetime
from typing import Any

from src.analytics.interfaces import ReportingServiceProtocol
from src.analytics.reporting.performance_reporter import PerformanceReporter
from src.analytics.types import AnalyticsConfiguration, AnalyticsReport, ReportType
from src.core.base.service import BaseService
from src.core.exceptions import ComponentError, ValidationError


class ReportingService(BaseService, ReportingServiceProtocol):
    """
    Service layer implementation for performance reporting.

    This service acts as a facade over the PerformanceReporter,
    providing proper service layer abstraction and dependency injection.
    """

    def __init__(
        self,
        config: AnalyticsConfiguration,
        performance_reporter: PerformanceReporter | None = None,
    ):
        """
        Initialize the reporting service.

        Args:
            config: Analytics configuration
            performance_reporter: Injected performance reporter engine (optional)
        """
        super().__init__()
        self.config = config

        # Use dependency injection - performance_reporter must be injected
        if performance_reporter is None:
            raise ComponentError(
                "performance_reporter must be injected via dependency injection",
                component="ReportingService",
                operation="__init__",
                context={"missing_dependency": "performance_reporter"},
            )

        self._reporter = performance_reporter

        self.logger.info("ReportingService initialized")

    async def start(self) -> None:
        """Start the reporting service."""
        try:
            if hasattr(self._reporter, "start"):
                await self._reporter.start()
            self.logger.info("Reporting service started")
        except Exception as e:
            raise ComponentError(
                f"Failed to start reporting service: {e}",
                component="ReportingService",
                operation="start",
            ) from e

    async def stop(self) -> None:
        """Stop the reporting service."""
        try:
            if hasattr(self._reporter, "stop"):
                await self._reporter.stop()
            self.logger.info("Reporting service stopped")
        except Exception as e:
            self.logger.error(f"Error stopping reporting service: {e}")

    async def generate_performance_report(
        self,
        report_type: ReportType,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> AnalyticsReport:
        """
        Generate performance report.

        Args:
            report_type: Type of report to generate
            start_date: Report start date
            end_date: Report end date

        Returns:
            Performance report

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If report generation fails
        """
        if not isinstance(report_type, ReportType):
            raise ValidationError(
                "Invalid report_type parameter",
                field_name="report_type",
                field_value=type(report_type),
                expected_type="ReportType",
            )

        if start_date and end_date and start_date >= end_date:
            raise ValidationError(
                "Invalid date range",
                field_name="date_range",
                field_value=f"{start_date} to {end_date}",
                validation_rule="start_date must be before end_date",
            )

        try:
            return await self._reporter.generate_performance_report(
                report_type, start_date, end_date
            )
        except Exception as e:
            raise ComponentError(
                f"Failed to generate performance report: {e}",
                component="ReportingService",
                operation="generate_performance_report",
                context={
                    "report_type": report_type.value,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            ) from e

    async def add_transaction_cost(
        self,
        timestamp: datetime,
        symbol: str,
        cost_type: str,
        cost_amount: float,
        trade_value: float,
    ) -> None:
        """
        Add transaction cost data.

        Args:
            timestamp: Transaction timestamp
            symbol: Trading symbol
            cost_type: Type of cost
            cost_amount: Cost amount
            trade_value: Trade value

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If addition fails
        """
        if not isinstance(symbol, str) or not symbol:
            raise ValidationError(
                "Invalid symbol parameter",
                field_name="symbol",
                field_value=symbol,
                expected_type="non-empty str",
            )

        if cost_amount < 0:
            raise ValidationError(
                "Invalid cost_amount parameter",
                field_name="cost_amount",
                field_value=cost_amount,
                validation_rule="must be non-negative",
            )

        try:
            await self._reporter.add_transaction_cost(
                timestamp, symbol, cost_type, cost_amount, trade_value
            )
        except Exception as e:
            raise ComponentError(
                f"Failed to add transaction cost: {e}",
                component="ReportingService",
                operation="add_transaction_cost",
                context={
                    "symbol": symbol,
                    "cost_type": cost_type,
                    "cost_amount": cost_amount,
                },
            ) from e

    async def generate_comprehensive_institutional_report(
        self, period: str = "monthly", include_regulatory: bool = True, include_esg: bool = False
    ) -> dict[str, Any]:
        """
        Generate comprehensive institutional report.

        Args:
            period: Report period
            include_regulatory: Include regulatory reporting
            include_esg: Include ESG analysis

        Returns:
            Comprehensive institutional report

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If report generation fails
        """
        valid_periods = ["daily", "weekly", "monthly", "quarterly", "annual"]
        if period not in valid_periods:
            raise ValidationError(
                "Invalid period parameter",
                field_name="period",
                field_value=period,
                validation_rule=f"must be one of {valid_periods}",
            )

        try:
            return await self._reporter.generate_comprehensive_institutional_report(
                period=period, include_regulatory=include_regulatory, include_esg=include_esg
            )
        except Exception as e:
            raise ComponentError(
                f"Failed to generate comprehensive institutional report: {e}",
                component="ReportingService",
                operation="generate_comprehensive_institutional_report",
                context={
                    "period": period,
                    "include_regulatory": include_regulatory,
                    "include_esg": include_esg,
                },
            ) from e
