"""Analytics Repository Implementation.

This module provides the concrete implementation of the AnalyticsDataRepository
that integrates with the database module following proper architectural patterns.
"""

from datetime import datetime

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.analytics.interfaces import AnalyticsDataRepository, DataTransformationServiceProtocol
from src.analytics.types import PortfolioMetrics, PositionMetrics, RiskMetrics
from src.core.base.component import BaseComponent
from src.core.exceptions import DataError, ValidationError
from src.database.models.analytics import (
    AnalyticsPortfolioMetrics,
    AnalyticsPositionMetrics,
    AnalyticsRiskMetrics,
)
from src.database.repository.base import DatabaseRepository


class AnalyticsRepository(BaseComponent, AnalyticsDataRepository):
    """
    Concrete implementation of analytics data repository.

    Integrates analytics data storage with the database module using
    proper repository patterns.
    """

    def __init__(
        self,
        session: AsyncSession | None,
        transformation_service: DataTransformationServiceProtocol = None,
    ):
        """
        Initialize analytics repository.

        Args:
            session: Database session for operations (can be None for testing)
            transformation_service: Service for data transformation business logic
        """
        super().__init__()
        self.session = session
        self.transformation_service = transformation_service

        # Initialize specific repositories for different metrics only if session is available
        if session is not None:
            self.portfolio_repo = DatabaseRepository(
                session=session,
                model=AnalyticsPortfolioMetrics,
                entity_type=AnalyticsPortfolioMetrics,
                name="AnalyticsPortfolioRepository",
            )

            self.position_repo = DatabaseRepository(
                session=session,
                model=AnalyticsPositionMetrics,
                entity_type=AnalyticsPositionMetrics,
                name="AnalyticsPositionRepository",
            )

            self.risk_repo = DatabaseRepository(
                session=session,
                model=AnalyticsRiskMetrics,
                entity_type=AnalyticsRiskMetrics,
                name="AnalyticsRiskRepository",
            )
        else:
            self.portfolio_repo = None
            self.position_repo = None
            self.risk_repo = None

    async def store_portfolio_metrics(self, metrics: PortfolioMetrics) -> None:
        """
        Store portfolio metrics in database using transformation service.

        Args:
            metrics: Portfolio metrics to store

        Raises:
            ValidationError: If metrics are invalid
            DataError: If storage fails
        """
        try:
            if self.portfolio_repo is None:
                raise DataError(
                    "Database session not available for portfolio metrics storage",
                    error_code="ANL_001",
                )

            # Extract bot_id and validate using consistent patterns
            bot_id = getattr(metrics, "bot_id", None)
            if not bot_id:
                raise ValidationError(
                    "bot_id is required for analytics portfolio metrics storage",
                    error_code="ANL_002",
                    field_name="bot_id",
                    field_value=bot_id,
                    expected_type="str",
                )

            # Use transformation service for business logic
            if self.transformation_service:
                transform_data = self.transformation_service.transform_portfolio_to_dict(
                    metrics, bot_id
                )
                db_metrics = AnalyticsPortfolioMetrics(**transform_data)
            else:
                # Fallback to simple mapping without business logic
                db_metrics = AnalyticsPortfolioMetrics(
                    timestamp=metrics.timestamp,
                    bot_id=bot_id,
                    total_value=metrics.total_value,
                    unrealized_pnl=metrics.unrealized_pnl,
                    realized_pnl=metrics.realized_pnl,
                    daily_pnl=metrics.daily_return,
                    number_of_positions=metrics.positions_count,
                    leverage_ratio=metrics.leverage,
                    margin_usage=getattr(metrics, "margin_used", None),
                    cash_balance=metrics.cash,
                )

            await self.portfolio_repo.create(db_metrics)
            self.logger.debug(f"Stored portfolio metrics for timestamp {metrics.timestamp}")

        except ValidationError:
            raise
        except Exception as e:
            raise DataError(
                f"Failed to store portfolio metrics: {e}",
                error_code="ANL_003",
                context={"timestamp": metrics.timestamp},
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
            if self.position_repo is None:
                raise DataError(
                    "Database session not available for position metrics storage",
                    error_code="ANL_004",
                )

            for metric in metrics:
                # Extract and validate bot_id
                bot_id = getattr(metric, "bot_id", None)
                if not bot_id:
                    raise ValidationError(
                        "bot_id is required for analytics position metrics storage",
                        error_code="ANL_005",
                        field_name="bot_id",
                        field_value=bot_id,
                        expected_type="str",
                    )

                # Use transformation service for business logic
                if self.transformation_service:
                    transform_data = self.transformation_service.transform_position_to_dict(
                        metric, bot_id
                    )
                    db_metric = AnalyticsPositionMetrics(**transform_data)
                else:
                    # Fallback to simple mapping without business logic
                    db_metric = AnalyticsPositionMetrics(
                        timestamp=metric.timestamp,
                        bot_id=bot_id,
                        position_id=getattr(metric, "position_id", None),
                        symbol=metric.symbol,
                        exchange=getattr(metric, "exchange", None),
                        quantity=getattr(metric, "quantity", 0),
                        market_value=metric.market_value,
                        unrealized_pnl=metric.unrealized_pnl,
                        realized_pnl=metric.realized_pnl,
                        average_price=metric.entry_price,
                        current_price=metric.current_price,
                        position_side=metric.side,
                    )

                await self.position_repo.create(db_metric)

            self.logger.debug(f"Stored {len(metrics)} position metrics")

        except ValidationError:
            raise
        except Exception as e:
            raise DataError(
                f"Failed to store position metrics: {e}",
                error_code="ANL_006",
                context={"count": len(metrics)},
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
            if self.risk_repo is None:
                raise DataError(
                    "Database session not available for risk metrics storage", error_code="ANL_007"
                )

            # Extract and validate bot_id
            bot_id = getattr(metrics, "bot_id", None)
            if not bot_id:
                raise ValidationError(
                    "bot_id is required for analytics risk metrics storage",
                    error_code="ANL_008",
                    field_name="bot_id",
                    field_value=bot_id,
                    expected_type="str",
                )

            # Use transformation service for business logic
            if self.transformation_service:
                transform_data = self.transformation_service.transform_risk_to_dict(metrics, bot_id)
                db_metrics = AnalyticsRiskMetrics(**transform_data)
            else:
                # Fallback to simple mapping without business logic
                db_metrics = AnalyticsRiskMetrics(
                    timestamp=metrics.timestamp,
                    bot_id=bot_id,
                    portfolio_var_95=getattr(metrics, "portfolio_var_95", None),
                    portfolio_var_99=getattr(metrics, "portfolio_var_99", None),
                    expected_shortfall_95=metrics.expected_shortfall,
                    maximum_drawdown=metrics.max_drawdown,
                    volatility=metrics.volatility,
                    sharpe_ratio=getattr(metrics, "sharpe_ratio", None),
                    sortino_ratio=getattr(metrics, "sortino_ratio", None),
                    correlation_risk=getattr(metrics, "correlation_risk", None),
                    concentration_risk=getattr(metrics, "concentration_risk", None),
                )

            await self.risk_repo.create(db_metrics)
            self.logger.debug(f"Stored risk metrics for timestamp {metrics.timestamp}")

        except ValidationError:
            raise
        except Exception as e:
            raise DataError(
                f"Failed to store risk metrics: {e}",
                error_code="ANL_009",
                context={"timestamp": metrics.timestamp},
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
                error_code="ANL_010",
                field_name="date_range",
                field_value=f"{start_date} to {end_date}",
            )

        try:
            if self.session is None:
                raise DataError(
                    "Database session not available for historical metrics retrieval",
                    error_code="ANL_011",
                )

            # Import select locally to avoid module loading issues during tests
            try:
                from sqlalchemy import select
            except ImportError:
                # Fallback - this should not happen but ensures robustness
                from sqlalchemy.sql import select

            # Use SQLAlchemy directly for date range query
            stmt = (
                select(AnalyticsPortfolioMetrics)
                .where(
                    AnalyticsPortfolioMetrics.timestamp >= start_date,
                    AnalyticsPortfolioMetrics.timestamp <= end_date,
                )
                .order_by(AnalyticsPortfolioMetrics.timestamp)
            )

            result_proxy = await self.session.execute(stmt)
            db_metrics = result_proxy.scalars().all()

            # Transform database models to domain objects using service
            result = []
            for db_metric in db_metrics:
                if self.transformation_service:
                    # Convert db model to dict for service transformation
                    db_data = {
                        "timestamp": db_metric.timestamp,
                        "total_value": db_metric.total_value,
                        "cash_balance": db_metric.cash_balance,
                        "unrealized_pnl": db_metric.unrealized_pnl,
                        "realized_pnl": db_metric.realized_pnl,
                        "daily_pnl": db_metric.daily_pnl,
                        "number_of_positions": db_metric.number_of_positions,
                        "leverage_ratio": db_metric.leverage_ratio,
                        "margin_usage": db_metric.margin_usage,
                    }
                    portfolio_metric = self.transformation_service.transform_dict_to_portfolio(
                        db_data
                    )
                    portfolio_metric.bot_id = str(db_metric.bot_id) if db_metric.bot_id else None
                    result.append(portfolio_metric)
                else:
                    # Fallback to simple transformation
                    result.append(
                        PortfolioMetrics(
                            timestamp=db_metric.timestamp,
                            bot_id=str(db_metric.bot_id) if db_metric.bot_id else None,
                            total_value=db_metric.total_value,
                            cash=db_metric.cash_balance or 0,
                            invested_capital=db_metric.total_value - (db_metric.cash_balance or 0),
                            unrealized_pnl=db_metric.unrealized_pnl,
                            realized_pnl=db_metric.realized_pnl,
                            total_pnl=db_metric.unrealized_pnl + db_metric.realized_pnl,
                            daily_return=db_metric.daily_pnl,
                            positions_count=db_metric.number_of_positions or 0,
                            leverage=db_metric.leverage_ratio,
                        )
                    )

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
            if self.session is None:
                return None

            # Import select locally to avoid module loading issues during tests
            try:
                from sqlalchemy import select, desc
            except ImportError:
                # Fallback - this should not happen but ensures robustness
                from sqlalchemy.sql import select, desc

            # Use SQLAlchemy directly for latest query
            stmt = (
                select(AnalyticsPortfolioMetrics)
                .order_by(desc(AnalyticsPortfolioMetrics.timestamp))
                .limit(1)
            )

            result_proxy = await self.session.execute(stmt)
            db_metric = result_proxy.scalar_one_or_none()

            if not db_metric:
                return None

            # Transform database model to domain object
            return PortfolioMetrics(
                timestamp=db_metric.timestamp,
                bot_id=str(db_metric.bot_id) if db_metric.bot_id else None,
                total_value=db_metric.total_value,
                cash=db_metric.cash_balance or 0,
                invested_capital=db_metric.total_value - (db_metric.cash_balance or 0),
                unrealized_pnl=db_metric.unrealized_pnl,
                realized_pnl=db_metric.realized_pnl,
                total_pnl=db_metric.unrealized_pnl + db_metric.realized_pnl,
                daily_return=db_metric.daily_pnl,
                positions_count=db_metric.number_of_positions or 0,
                leverage=db_metric.leverage_ratio,
            )

        except Exception as e:
            raise DataError(
                f"Failed to retrieve latest portfolio metrics: {e}", error_code="ANL_012"
            ) from e

    async def get_latest_risk_metrics(self) -> RiskMetrics | None:
        """
        Get the latest risk metrics.

        Returns:
            Latest risk metrics or None if none exist
        """
        try:
            if self.session is None:
                return None

            # Import select locally to avoid module loading issues during tests
            try:
                from sqlalchemy import select, desc
            except ImportError:
                # Fallback - this should not happen but ensures robustness
                from sqlalchemy.sql import select, desc

            # Use SQLAlchemy directly for latest query
            stmt = (
                select(AnalyticsRiskMetrics).order_by(desc(AnalyticsRiskMetrics.timestamp)).limit(1)
            )

            result_proxy = await self.session.execute(stmt)
            db_metric = result_proxy.scalar_one_or_none()

            if not db_metric:
                return None

            # Transform database model to domain object
            return RiskMetrics(
                timestamp=db_metric.timestamp,
                bot_id=str(db_metric.bot_id) if db_metric.bot_id else None,
                portfolio_var_95=db_metric.portfolio_var_95,
                portfolio_var_99=db_metric.portfolio_var_99,
                expected_shortfall=db_metric.expected_shortfall_95,
                max_drawdown=db_metric.maximum_drawdown,
                volatility=db_metric.volatility,
                sharpe_ratio=db_metric.sharpe_ratio,
                sortino_ratio=db_metric.sortino_ratio,
                correlation_risk=db_metric.correlation_risk,
                concentration_risk=db_metric.concentration_risk,
            )

        except Exception as e:
            raise DataError(
                f"Failed to retrieve latest risk metrics: {e}", error_code="ANL_013"
            ) from e
