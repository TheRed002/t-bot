"""Analytics Repository Implementation.

This module provides the concrete implementation of the AnalyticsDataRepository
that integrates with the database module following proper architectural patterns.
"""

from datetime import datetime
from decimal import Decimal

from src.analytics.interfaces import AnalyticsDataRepository
from src.analytics.types import PortfolioMetrics, PositionMetrics, RiskMetrics
from src.core.base.component import BaseComponent
from src.core.exceptions import DataError, ValidationError
from src.database.models.analytics import (
    AnalyticsPortfolioMetrics,
    AnalyticsPositionMetrics,
    AnalyticsRiskMetrics,
)
from src.database.uow import UnitOfWork


class AnalyticsRepository(BaseComponent, AnalyticsDataRepository):
    """
    Concrete implementation of analytics data repository.

    Integrates analytics data storage with the database module using
    proper repository patterns and unit of work.
    """

    def __init__(self, uow: UnitOfWork):
        """
        Initialize analytics repository.

        Args:
            uow: Unit of work for database transactions
        """
        super().__init__()
        self.uow = uow

    async def store_portfolio_metrics(self, metrics: PortfolioMetrics) -> None:
        """
        Store portfolio metrics in database.

        Args:
            metrics: Portfolio metrics to store

        Raises:
            ValidationError: If metrics are invalid
            DataError: If storage fails
        """
        try:
            async with self.uow:
                db_metrics = AnalyticsPortfolioMetrics(
                    timestamp=metrics.timestamp,
                    total_value=metrics.total_value,
                    unrealized_pnl=metrics.unrealized_pnl,
                    realized_pnl=metrics.realized_pnl,
                    daily_pnl=metrics.daily_return,
                    number_of_positions=metrics.positions_count,
                    leverage_ratio=metrics.leverage,
                    margin_usage=metrics.margin_used,
                    cash_balance=metrics.cash,
                )

                await self.uow.analytics_repository.create(db_metrics)
                await self.uow.commit()

                self.logger.debug(f"Stored portfolio metrics for timestamp {metrics.timestamp}")

        except Exception as e:
            await self.uow.rollback()
            raise DataError(
                f"Failed to store portfolio metrics: {e}", context={"timestamp": metrics.timestamp}
            ) from e

    async def store_position_metrics(self, metrics: list[PositionMetrics]) -> None:
        """
        Store position metrics in database.

        Args:
            metrics: List of position metrics to store

        Raises:
            ValidationError: If metrics are invalid
            DataError: If storage fails
        """
        if not metrics:
            return

        try:
            async with self.uow:
                db_metrics_list = []
                for metric in metrics:
                    db_metric = AnalyticsPositionMetrics(
                        timestamp=metric.timestamp,
                        symbol=metric.symbol,
                        exchange=metric.exchange,
                        quantity=metric.quantity,
                        market_value=metric.market_value,
                        unrealized_pnl=metric.unrealized_pnl,
                        realized_pnl=metric.realized_pnl,
                        average_price=metric.entry_price,
                        current_price=metric.current_price,
                        position_side=metric.side,
                    )
                    db_metrics_list.append(db_metric)

                for db_metric in db_metrics_list:
                    await self.uow.analytics_repository.create(db_metric)

                await self.uow.commit()

                self.logger.debug(f"Stored {len(metrics)} position metrics")

        except Exception as e:
            await self.uow.rollback()
            raise DataError(
                f"Failed to store position metrics: {e}", context={"count": len(metrics)}
            ) from e

    async def store_risk_metrics(self, metrics: RiskMetrics) -> None:
        """
        Store risk metrics in database.

        Args:
            metrics: Risk metrics to store

        Raises:
            ValidationError: If metrics are invalid
            DataError: If storage fails
        """
        try:
            async with self.uow:
                db_metrics = AnalyticsRiskMetrics(
                    timestamp=metrics.timestamp,
                    portfolio_var_95=metrics.portfolio_var_95,
                    portfolio_var_99=metrics.portfolio_var_99,
                    expected_shortfall_95=metrics.expected_shortfall,
                    maximum_drawdown=metrics.max_drawdown,
                    volatility=metrics.volatility,
                    sharpe_ratio=None,  # Not available in RiskMetrics
                    sortino_ratio=None,  # Not available in RiskMetrics
                    correlation_risk=metrics.correlation_risk,
                    concentration_risk=metrics.concentration_risk,
                )

                await self.uow.analytics_repository.create(db_metrics)
                await self.uow.commit()

                self.logger.debug(f"Stored risk metrics for timestamp {metrics.timestamp}")

        except Exception as e:
            await self.uow.rollback()
            raise DataError(
                f"Failed to store risk metrics: {e}", context={"timestamp": metrics.timestamp}
            ) from e

    async def get_historical_portfolio_metrics(
        self, start_date: datetime, end_date: datetime
    ) -> list[PortfolioMetrics]:
        """
        Retrieve historical portfolio metrics.

        Args:
            start_date: Start date for retrieval
            end_date: End date for retrieval

        Returns:
            List of historical portfolio metrics

        Raises:
            ValidationError: If date range is invalid
            DataError: If retrieval fails
        """
        if start_date >= end_date:
            raise ValidationError(
                "Start date must be before end date",
                field_name="date_range",
                field_value=f"{start_date} to {end_date}",
            )

        try:
            async with self.uow:
                db_metrics = await self.uow.analytics_repository.find_by_date_range(
                    model_class=AnalyticsPortfolioMetrics, start_date=start_date, end_date=end_date
                )

                result = []
                for db_metric in db_metrics:
                    metric = PortfolioMetrics(
                        timestamp=db_metric.timestamp,
                        total_value=db_metric.total_value,
                        cash=db_metric.cash_balance or Decimal("0"),
                        invested_capital=db_metric.total_value
                        - (db_metric.cash_balance or Decimal("0")),
                        unrealized_pnl=db_metric.unrealized_pnl,
                        realized_pnl=db_metric.realized_pnl,
                        total_pnl=db_metric.unrealized_pnl + db_metric.realized_pnl,
                        daily_return=db_metric.daily_pnl,
                        leverage=db_metric.leverage_ratio,
                        margin_used=db_metric.margin_usage,
                    )
                    result.append(metric)

                self.logger.debug(f"Retrieved {len(result)} historical portfolio metrics")
                return result

        except Exception as e:
            raise DataError(
                f"Failed to retrieve historical portfolio metrics: {e}",
                context={"start_date": start_date, "end_date": end_date},
            ) from e

    async def get_latest_portfolio_metrics(self) -> PortfolioMetrics | None:
        """
        Get the latest portfolio metrics.

        Returns:
            Latest portfolio metrics or None if none exist
        """
        try:
            async with self.uow:
                db_metric = await self.uow.analytics_repository.find_latest(
                    AnalyticsPortfolioMetrics
                )

                if not db_metric:
                    return None

                return PortfolioMetrics(
                    timestamp=db_metric.timestamp,
                    total_value=db_metric.total_value,
                    cash=db_metric.cash_balance or Decimal("0"),
                    invested_capital=db_metric.total_value
                    - (db_metric.cash_balance or Decimal("0")),
                    unrealized_pnl=db_metric.unrealized_pnl,
                    realized_pnl=db_metric.realized_pnl,
                    total_pnl=db_metric.unrealized_pnl + db_metric.realized_pnl,
                    daily_return=db_metric.daily_pnl,
                    leverage=db_metric.leverage_ratio,
                    margin_used=db_metric.margin_usage,
                )

        except Exception as e:
            raise DataError(f"Failed to retrieve latest portfolio metrics: {e}") from e

    async def get_latest_risk_metrics(self) -> RiskMetrics | None:
        """
        Get the latest risk metrics.

        Returns:
            Latest risk metrics or None if none exist
        """
        try:
            async with self.uow:
                db_metric = await self.uow.analytics_repository.find_latest(AnalyticsRiskMetrics)

                if not db_metric:
                    return None

                return RiskMetrics(
                    timestamp=db_metric.timestamp,
                    portfolio_var_95=db_metric.portfolio_var_95,
                    portfolio_var_99=db_metric.portfolio_var_99,
                    max_drawdown=db_metric.maximum_drawdown,
                    volatility=db_metric.volatility,
                )

        except Exception as e:
            raise DataError(f"Failed to retrieve latest risk metrics: {e}") from e
