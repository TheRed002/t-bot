"""Analytics Repository Implementation.

This module provides the concrete implementation of the AnalyticsDataRepository
that integrates with the database module following proper architectural patterns.
"""

from datetime import datetime

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.analytics.interfaces import AnalyticsDataRepository
from src.analytics.services.data_transformation_service import DataTransformationService
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
        transformation_service: DataTransformationService | None = None,
    ):
        """
        Initialize analytics repository.

        Args:
            session: Database session for operations (can be None for testing)
            transformation_service: Data transformation service (injected dependency)
        """
        super().__init__()
        self.session = session

        # Use dependency injection for transformation service
        if transformation_service is None:
            from src.core.exceptions import ComponentError

            raise ComponentError(
                "transformation_service must be injected via dependency injection",
                component="AnalyticsRepository",
                operation="__init__",
                context={"missing_dependency": "transformation_service"},
            )
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
                raise DataError("Database session not available for portfolio metrics storage")

            # Extract bot_id and validate using consistent patterns
            bot_id = getattr(metrics, "bot_id", None)
            if not bot_id:
                raise ValidationError(
                    "bot_id is required for analytics portfolio metrics storage",
                    field_name="bot_id",
                    field_value=bot_id,
                    expected_type="UUID",
                )

            # Use transformation service for data conversion
            db_metrics = self.transformation_service.transform_portfolio_metrics_to_db(
                metrics, bot_id
            )

            await self.portfolio_repo.create(db_metrics)
            self.logger.debug(f"Stored portfolio metrics for timestamp {metrics.timestamp}")

        except ValidationError:
            raise
        except Exception as e:
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
            if self.position_repo is None:
                raise DataError("Database session not available for position metrics storage")

            for metric in metrics:
                # Extract and validate bot_id
                bot_id = getattr(metric, "bot_id", None)
                if not bot_id:
                    raise ValidationError(
                        "bot_id is required for analytics position metrics storage",
                        field_name="bot_id",
                        field_value=bot_id,
                        expected_type="UUID",
                    )

                # Use transformation service for data conversion
                db_metric = self.transformation_service.transform_position_metrics_to_db(
                    metric, bot_id
                )

                await self.position_repo.create(db_metric)

            self.logger.debug(f"Stored {len(metrics)} position metrics")

        except ValidationError:
            raise
        except Exception as e:
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
            if self.risk_repo is None:
                raise DataError("Database session not available for risk metrics storage")

            # Extract and validate bot_id
            bot_id = getattr(metrics, "bot_id", None)
            if not bot_id:
                raise ValidationError(
                    "bot_id is required for analytics risk metrics storage",
                    field_name="bot_id",
                    field_value=bot_id,
                    expected_type="UUID",
                )

            # Use transformation service for data conversion
            db_metrics = self.transformation_service.transform_risk_metrics_to_db(metrics, bot_id)

            await self.risk_repo.create(db_metrics)
            self.logger.debug(f"Stored risk metrics for timestamp {metrics.timestamp}")

        except ValidationError:
            raise
        except Exception as e:
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
            if self.session is None:
                raise DataError("Database session not available for historical metrics retrieval")

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

            # Use transformation service for data conversion
            result = [
                self.transformation_service.transform_db_to_portfolio_metrics(db_metric)
                for db_metric in db_metrics
            ]

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

            # Use transformation service for data conversion
            return self.transformation_service.transform_db_to_portfolio_metrics(db_metric)

        except Exception as e:
            raise DataError(f"Failed to retrieve latest portfolio metrics: {e}") from e

    async def get_latest_risk_metrics(self) -> RiskMetrics | None:
        """
        Get the latest risk metrics.

        Returns:
            Latest risk metrics or None if none exist
        """
        try:
            if self.session is None:
                return None

            # Use SQLAlchemy directly for latest query
            stmt = (
                select(AnalyticsRiskMetrics).order_by(desc(AnalyticsRiskMetrics.timestamp)).limit(1)
            )

            result_proxy = await self.session.execute(stmt)
            db_metric = result_proxy.scalar_one_or_none()

            if not db_metric:
                return None

            # Use transformation service for data conversion
            return self.transformation_service.transform_db_to_risk_metrics(db_metric)

        except Exception as e:
            raise DataError(f"Failed to retrieve latest risk metrics: {e}") from e
