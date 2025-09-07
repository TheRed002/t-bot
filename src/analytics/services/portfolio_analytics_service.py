"""
Portfolio Analytics Service.

This service provides a proper service layer implementation for portfolio analytics,
following service layer patterns and using dependency injection.
"""

from typing import Any

from src.analytics.interfaces import PortfolioServiceProtocol
from src.analytics.portfolio.portfolio_analytics import PortfolioAnalyticsEngine
from src.analytics.types import AnalyticsConfiguration, BenchmarkData
from src.core.base.service import BaseService
from src.core.exceptions import ComponentError, ValidationError
from src.core.types import Position


class PortfolioAnalyticsService(BaseService, PortfolioServiceProtocol):
    """
    Service layer implementation for portfolio analytics.

    This service acts as a facade over the PortfolioAnalyticsEngine,
    providing proper service layer abstraction and dependency injection.
    """

    def __init__(
        self,
        config: AnalyticsConfiguration,
        analytics_engine: PortfolioAnalyticsEngine | None = None,
    ):
        """
        Initialize the portfolio analytics service.

        Args:
            config: Analytics configuration
            analytics_engine: Injected analytics engine (optional)
        """
        super().__init__()
        self.config = config

        # Use dependency injection - engine must be injected
        if analytics_engine is None:
            raise ComponentError(
                "analytics_engine must be injected via dependency injection",
                component="PortfolioAnalyticsService",
                operation="__init__",
                context={"missing_dependency": "analytics_engine"},
            )

        self._engine = analytics_engine

        self.logger.info("PortfolioAnalyticsService initialized")

    async def start(self) -> None:
        """Start the portfolio analytics service."""
        try:
            if hasattr(self._engine, "start"):
                await self._engine.start()
            self.logger.info("Portfolio analytics service started")
        except Exception as e:
            raise ComponentError(
                f"Failed to start portfolio analytics service: {e}",
                component="PortfolioAnalyticsService",
                operation="start",
            ) from e

    async def stop(self) -> None:
        """Stop the portfolio analytics service."""
        try:
            if hasattr(self._engine, "stop"):
                await self._engine.stop()
            self.logger.info("Portfolio analytics service stopped")
        except Exception as e:
            self.logger.error(f"Error stopping portfolio analytics service: {e}")

    def update_position(self, position: Position) -> None:
        """
        Update position data.

        Args:
            position: Position to update

        Raises:
            ValidationError: If position data is invalid
            ComponentError: If update fails
        """
        if not isinstance(position, Position):
            raise ValidationError(
                "Invalid position parameter",
                field_name="position",
                field_value=type(position),
                expected_type="Position",
            )

        try:
            self._engine.update_position(position)
            self.logger.debug(f"Position updated: {position.symbol}")
        except Exception as e:
            raise ComponentError(
                f"Failed to update position: {e}",
                component="PortfolioAnalyticsService",
                operation="update_position",
                context={"symbol": position.symbol},
            ) from e

    def update_benchmark_data(self, benchmark_name: str, data: BenchmarkData) -> None:
        """
        Update benchmark data.

        Args:
            benchmark_name: Benchmark name
            data: Benchmark data

        Raises:
            ValidationError: If data is invalid
            ComponentError: If update fails
        """
        if not isinstance(benchmark_name, str) or not benchmark_name:
            raise ValidationError(
                "Invalid benchmark_name parameter",
                field_name="benchmark_name",
                field_value=benchmark_name,
                expected_type="non-empty str",
            )

        try:
            self._engine.update_benchmark_data(benchmark_name, data)
            self.logger.debug(f"Benchmark updated: {benchmark_name}")
        except Exception as e:
            raise ComponentError(
                f"Failed to update benchmark data: {e}",
                component="PortfolioAnalyticsService",
                operation="update_benchmark_data",
                context={"benchmark": benchmark_name},
            ) from e

    async def get_portfolio_composition(self) -> dict[str, Any]:
        """
        Get portfolio composition.

        Returns:
            Portfolio composition data

        Raises:
            ComponentError: If retrieval fails
        """
        try:
            return await self._engine.get_portfolio_composition()
        except Exception as e:
            raise ComponentError(
                f"Failed to get portfolio composition: {e}",
                component="PortfolioAnalyticsService",
                operation="get_portfolio_composition",
            ) from e

    async def calculate_correlation_matrix(self) -> Any:
        """
        Calculate correlation matrix.

        Returns:
            Correlation matrix

        Raises:
            ComponentError: If calculation fails
        """
        try:
            return await self._engine.calculate_correlation_matrix()
        except Exception as e:
            raise ComponentError(
                f"Failed to calculate correlation matrix: {e}",
                component="PortfolioAnalyticsService",
                operation="calculate_correlation_matrix",
            ) from e

    async def optimize_portfolio_mvo(self) -> dict[str, Any]:
        """
        Optimize portfolio using Mean-Variance Optimization.

        Returns:
            Optimization results

        Raises:
            ComponentError: If optimization fails
        """
        try:
            return await self._engine.optimize_portfolio_mvo()
        except Exception as e:
            raise ComponentError(
                f"Failed to optimize portfolio (MVO): {e}",
                component="PortfolioAnalyticsService",
                operation="optimize_portfolio_mvo",
            ) from e

    async def optimize_black_litterman(self) -> dict[str, Any]:
        """
        Optimize portfolio using Black-Litterman model.

        Returns:
            Optimization results

        Raises:
            ComponentError: If optimization fails
        """
        try:
            return await self._engine.optimize_black_litterman()
        except Exception as e:
            raise ComponentError(
                f"Failed to optimize portfolio (Black-Litterman): {e}",
                component="PortfolioAnalyticsService",
                operation="optimize_black_litterman",
            ) from e

    async def optimize_risk_parity(self) -> dict[str, Any]:
        """
        Optimize portfolio using Risk Parity.

        Returns:
            Optimization results

        Raises:
            ComponentError: If optimization fails
        """
        try:
            return await self._engine.optimize_risk_parity()
        except Exception as e:
            raise ComponentError(
                f"Failed to optimize portfolio (Risk Parity): {e}",
                component="PortfolioAnalyticsService",
                operation="optimize_risk_parity",
            ) from e

    async def generate_institutional_analytics_report(self) -> dict[str, Any]:
        """
        Generate institutional analytics report.

        Returns:
            Analytics report

        Raises:
            ComponentError: If report generation fails
        """
        try:
            return await self._engine.generate_institutional_analytics_report()
        except Exception as e:
            raise ComponentError(
                f"Failed to generate institutional analytics report: {e}",
                component="PortfolioAnalyticsService",
                operation="generate_institutional_analytics_report",
            ) from e
