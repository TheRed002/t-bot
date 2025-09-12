"""
Data Transformation Service for Analytics Module.

This service handles all data transformation logic that was previously
embedded in the repository layer, following proper service layer patterns.
"""

from decimal import Decimal

from src.analytics.types import PortfolioMetrics, PositionMetrics, RiskMetrics
from src.core.base.service import BaseService
from src.core.exceptions import ValidationError
from src.database.models.analytics import (
    AnalyticsPortfolioMetrics,
    AnalyticsPositionMetrics,
    AnalyticsRiskMetrics,
)


class DataTransformationService(BaseService):
    """
    Service for transforming analytics data between domain and persistence layers.

    This service ensures proper separation of concerns by handling all
    data transformations outside the repository layer.
    """

    def transform_portfolio_metrics_to_db(
        self, metrics: PortfolioMetrics, bot_id: str
    ) -> AnalyticsPortfolioMetrics:
        """
        Transform domain portfolio metrics to database model.

        Args:
            metrics: Domain portfolio metrics
            bot_id: Bot identifier

        Returns:
            Database portfolio metrics model

        Raises:
            ValidationError: If transformation fails
        """
        if not bot_id:
            raise ValidationError(
                "bot_id is required for analytics portfolio metrics transformation",
                field_name="bot_id",
                field_value=bot_id,
                expected_type="non-empty str",
            )

        return AnalyticsPortfolioMetrics(
            timestamp=metrics.timestamp,
            total_value=metrics.total_value,
            unrealized_pnl=metrics.unrealized_pnl,
            realized_pnl=metrics.realized_pnl,
            daily_pnl=getattr(metrics, "daily_return", Decimal("0")),
            number_of_positions=getattr(metrics, "positions_count", 0),
            leverage_ratio=getattr(metrics, "leverage", None),
            margin_usage=getattr(metrics, "margin_used", None),
            cash_balance=getattr(metrics, "cash", None),
            bot_id=bot_id,
        )

    def transform_position_metrics_to_db(
        self, metric: PositionMetrics, bot_id: str
    ) -> AnalyticsPositionMetrics:
        """
        Transform domain position metrics to database model.

        Args:
            metric: Domain position metrics
            bot_id: Bot identifier

        Returns:
            Database position metrics model

        Raises:
            ValidationError: If transformation fails
        """
        if not bot_id:
            raise ValidationError(
                "bot_id is required for analytics position metrics transformation",
                field_name="bot_id",
                field_value=bot_id,
                expected_type="non-empty str",
            )

        return AnalyticsPositionMetrics(
            timestamp=metric.timestamp,
            symbol=metric.symbol,
            exchange=metric.exchange,
            quantity=metric.quantity,
            market_value=metric.market_value,
            unrealized_pnl=metric.unrealized_pnl,
            realized_pnl=getattr(metric, "realized_pnl", Decimal("0")),
            average_price=getattr(metric, "entry_price", metric.current_price),
            current_price=metric.current_price,
            position_side=getattr(metric, "side", "LONG"),
            bot_id=bot_id,
            position_id=getattr(metric, "position_id", None),
        )

    def transform_risk_metrics_to_db(
        self, metrics: RiskMetrics, bot_id: str
    ) -> AnalyticsRiskMetrics:
        """
        Transform domain risk metrics to database model.

        Args:
            metrics: Domain risk metrics
            bot_id: Bot identifier

        Returns:
            Database risk metrics model

        Raises:
            ValidationError: If transformation fails
        """
        if not bot_id:
            raise ValidationError(
                "bot_id is required for analytics risk metrics transformation",
                field_name="bot_id",
                field_value=bot_id,
                expected_type="non-empty str",
            )

        return AnalyticsRiskMetrics(
            timestamp=metrics.timestamp,
            portfolio_var_95=getattr(metrics, "portfolio_var_95", None),
            portfolio_var_99=getattr(metrics, "portfolio_var_99", None),
            expected_shortfall_95=getattr(metrics, "expected_shortfall", None),
            maximum_drawdown=getattr(metrics, "max_drawdown", None),
            volatility=getattr(metrics, "volatility", None),
            sharpe_ratio=getattr(metrics, "sharpe_ratio", None),
            sortino_ratio=getattr(metrics, "sortino_ratio", None),
            correlation_risk=getattr(metrics, "correlation_risk", None),
            concentration_risk=getattr(metrics, "concentration_risk", None),
            bot_id=bot_id,
        )

    def transform_db_to_portfolio_metrics(
        self, db_metric: AnalyticsPortfolioMetrics
    ) -> PortfolioMetrics:
        """
        Transform database portfolio metrics to domain model.

        Args:
            db_metric: Database portfolio metrics

        Returns:
            Domain portfolio metrics
        """
        return PortfolioMetrics(
            timestamp=db_metric.timestamp,  # type: ignore[arg-type]
            total_value=db_metric.total_value,
            cash=db_metric.cash_balance or Decimal("0"),
            invested_capital=db_metric.total_value - (db_metric.cash_balance or Decimal("0")),
            unrealized_pnl=db_metric.unrealized_pnl,
            realized_pnl=db_metric.realized_pnl,
            total_pnl=db_metric.unrealized_pnl + db_metric.realized_pnl,
        )

    def transform_db_to_risk_metrics(self, db_metric: AnalyticsRiskMetrics) -> RiskMetrics:
        """
        Transform database risk metrics to domain model.

        Args:
            db_metric: Database risk metrics

        Returns:
            Domain risk metrics
        """
        return RiskMetrics(
            timestamp=db_metric.timestamp,  # type: ignore[arg-type]
            portfolio_var_95=db_metric.portfolio_var_95,
            portfolio_var_99=db_metric.portfolio_var_99,
            max_drawdown=db_metric.maximum_drawdown,
            volatility=db_metric.volatility,
        )
