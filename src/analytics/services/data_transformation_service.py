"""
Data Transformation Service for Analytics Module.

Provides consistent data transformation patterns between domain and persistence layers,
aligned with database module patterns and messaging standards.
"""

from decimal import Decimal
from typing import Any

from src.analytics.interfaces import DataTransformationServiceProtocol
from src.analytics.types import PortfolioMetrics, PositionMetrics, RiskMetrics
from src.core.base.service import BaseService
from src.core.exceptions import ValidationError
from src.database.models.analytics import (
    AnalyticsPortfolioMetrics,
    AnalyticsPositionMetrics,
    AnalyticsRiskMetrics,
)
from src.utils.decimal_utils import to_decimal


class DataTransformationService(BaseService, DataTransformationServiceProtocol):
    """
    Service for transforming analytics data between domain and persistence layers.

    This service only handles business logic for data transformation,
    without direct coupling to infrastructure models.
    """

    def transform_portfolio_to_dict(self, metrics: PortfolioMetrics, bot_id: str) -> dict[str, Any]:
        """
        Transform PortfolioMetrics to dictionary for persistence.

        Args:
            metrics: Domain portfolio metrics object
            bot_id: Bot identifier for database association

        Returns:
            Dict[str, Any]: Data dictionary ready for persistence

        Raises:
            ValidationError: If transformation validation fails
        """
        try:
            # Validate required fields
            if not bot_id:
                raise ValidationError("bot_id is required for portfolio metrics")

            # Apply business rules for data transformation
            daily_pnl = to_decimal(0)  # Default business rule
            if hasattr(metrics, "daily_return") and metrics.daily_return is not None:
                daily_pnl = to_decimal(metrics.daily_return)

            return {
                "bot_id": bot_id,
                "timestamp": metrics.timestamp,
                "total_value": to_decimal(metrics.total_value),
                "unrealized_pnl": to_decimal(metrics.unrealized_pnl),
                "realized_pnl": to_decimal(metrics.realized_pnl),
                "daily_pnl": daily_pnl,
                "number_of_positions": getattr(metrics, "positions_count", 0),
                "leverage_ratio": to_decimal(metrics.leverage)
                if hasattr(metrics, "leverage") and metrics.leverage is not None
                else None,
                "margin_usage": to_decimal(metrics.margin_used)
                if hasattr(metrics, "margin_used") and metrics.margin_used is not None
                else None,
                "cash_balance": to_decimal(metrics.cash)
                if hasattr(metrics, "cash") and metrics.cash is not None
                else None,
            }
        except Exception as e:
            raise ValidationError(
                f"Failed to transform portfolio metrics: {e}",
                context={
                    "bot_id": bot_id,
                    "transformation_type": "portfolio_metrics_to_dict",
                    "processing_mode": "stream",
                },
            ) from e

    def transform_position_to_dict(self, metric: PositionMetrics, bot_id: str) -> dict[str, Any]:
        """
        Transform PositionMetrics to dictionary for persistence.

        Args:
            metric: Domain position metrics object
            bot_id: Bot identifier for database association

        Returns:
            Dict[str, Any]: Data dictionary ready for persistence

        Raises:
            ValidationError: If transformation validation fails
        """
        try:
            # Validate required fields
            if not bot_id:
                raise ValidationError("bot_id is required for position metrics")

            # Apply business rules for side normalization
            position_side = metric.side.upper() if metric.side else "UNKNOWN"

            return {
                "bot_id": bot_id,
                "timestamp": metric.timestamp,
                "position_id": getattr(metric, "position_id", None),
                "symbol": metric.symbol,
                "exchange": getattr(metric, "exchange", None),
                "quantity": to_decimal(getattr(metric, "size", 0)),
                "market_value": to_decimal(metric.market_value),
                "unrealized_pnl": to_decimal(metric.unrealized_pnl),
                "realized_pnl": to_decimal(metric.realized_pnl)
                if metric.realized_pnl is not None
                else None,
                "average_price": to_decimal(metric.entry_price),
                "current_price": to_decimal(metric.current_price),
                "position_side": position_side,
                "processing_mode": "stream",
                "message_pattern": "pub_sub",
            }
        except Exception as e:
            context_symbol = getattr(metric, "symbol", "unknown")
            raise ValidationError(
                f"Failed to transform position metrics: {e}",
                context={
                    "bot_id": bot_id,
                    "symbol": context_symbol,
                    "transformation_type": "position_metrics_to_dict",
                    "processing_mode": "stream",
                },
            ) from e

    def transform_risk_to_dict(self, metrics: RiskMetrics, bot_id: str) -> dict[str, Any]:
        """
        Transform RiskMetrics to dictionary for persistence.

        Args:
            metrics: Domain risk metrics object
            bot_id: Bot identifier for database association

        Returns:
            Dict[str, Any]: Data dictionary ready for persistence

        Raises:
            ValidationError: If transformation validation fails
        """
        try:
            # Validate required fields
            if not bot_id:
                raise ValidationError("bot_id is required for risk metrics")

            return {
                "bot_id": bot_id,
                "timestamp": metrics.timestamp,
                "portfolio_var_95": to_decimal(getattr(metrics, "portfolio_var_95", None))
                if getattr(metrics, "portfolio_var_95", None) is not None
                else None,
                "portfolio_var_99": to_decimal(getattr(metrics, "portfolio_var_99", None))
                if getattr(metrics, "portfolio_var_99", None) is not None
                else None,
                "expected_shortfall_95": to_decimal(metrics.expected_shortfall)
                if metrics.expected_shortfall is not None
                else None,
                "maximum_drawdown": to_decimal(metrics.max_drawdown),
                "volatility": to_decimal(metrics.volatility)
                if metrics.volatility is not None
                else None,
                "sharpe_ratio": to_decimal(getattr(metrics, "sharpe_ratio", None))
                if getattr(metrics, "sharpe_ratio", None) is not None
                else None,
                "sortino_ratio": to_decimal(getattr(metrics, "sortino_ratio", None))
                if getattr(metrics, "sortino_ratio", None) is not None
                else None,
                "correlation_risk": to_decimal(getattr(metrics, "correlation_risk", None))
                if getattr(metrics, "correlation_risk", None) is not None
                else None,
                "concentration_risk": to_decimal(getattr(metrics, "concentration_risk", None))
                if getattr(metrics, "concentration_risk", None) is not None
                else None,
                "processing_mode": "stream",
                "message_pattern": "pub_sub",
            }
        except Exception as e:
            raise ValidationError(
                f"Failed to transform risk metrics: {e}",
                context={
                    "bot_id": bot_id,
                    "transformation_type": "risk_metrics_to_dict",
                    "processing_mode": "stream",
                },
            ) from e

    def transform_dict_to_portfolio(self, data: dict[str, Any]) -> PortfolioMetrics:
        """
        Transform data dictionary to PortfolioMetrics domain object.

        Args:
            data: Dictionary containing portfolio metrics data

        Returns:
            PortfolioMetrics: Domain metrics object
        """
        # Apply business rules for domain object creation
        cash_balance = data.get("cash_balance") or Decimal("0")
        total_value = data.get("total_value") or Decimal("0")

        return PortfolioMetrics(
            timestamp=data["timestamp"],
            total_value=total_value,
            cash=cash_balance,
            invested_capital=total_value - cash_balance,
            unrealized_pnl=data.get("unrealized_pnl") or Decimal("0"),
            realized_pnl=data.get("realized_pnl") or Decimal("0"),
            total_pnl=(data.get("unrealized_pnl") or Decimal("0"))
            + (data.get("realized_pnl") or Decimal("0")),
            daily_return=data.get("daily_pnl"),
            positions_count=data.get("number_of_positions", 0),
            leverage=data.get("leverage_ratio"),
            margin_used=data.get("margin_usage"),
        )

    def transform_dict_to_risk(self, data: dict[str, Any]) -> RiskMetrics:
        """
        Transform data dictionary to RiskMetrics domain object.

        Args:
            data: Dictionary containing risk metrics data

        Returns:
            RiskMetrics: Domain risk metrics object
        """
        return RiskMetrics(
            timestamp=data["timestamp"],
            portfolio_var_95=data.get("portfolio_var_95"),
            portfolio_var_99=data.get("portfolio_var_99"),
            expected_shortfall=data.get("expected_shortfall_95"),
            max_drawdown=data.get("maximum_drawdown") or Decimal("0"),
            volatility=data.get("volatility"),
            sharpe_ratio=data.get("sharpe_ratio"),
            sortino_ratio=data.get("sortino_ratio"),
            correlation_risk=data.get("correlation_risk"),
            concentration_risk=data.get("concentration_risk"),
        )

    def transform_portfolio_metrics_to_db(self, metrics: PortfolioMetrics, bot_id: str) -> AnalyticsPortfolioMetrics:
        """
        Transform PortfolioMetrics to database model.

        Args:
            metrics: Domain portfolio metrics object
            bot_id: Bot identifier for database association

        Returns:
            AnalyticsPortfolioMetrics: Database model instance

        Raises:
            ValidationError: If transformation validation fails
        """
        try:
            if not bot_id:
                raise ValidationError("bot_id is required for portfolio metrics")

            # Map domain fields to database fields
            return AnalyticsPortfolioMetrics(
                bot_id=bot_id,
                timestamp=metrics.timestamp,
                total_value=to_decimal(metrics.total_value),
                unrealized_pnl=to_decimal(metrics.unrealized_pnl) if metrics.unrealized_pnl is not None else Decimal("0"),
                realized_pnl=to_decimal(metrics.realized_pnl) if metrics.realized_pnl is not None else Decimal("0"),
                daily_pnl=to_decimal(getattr(metrics, "daily_return", 0)) if hasattr(metrics, "daily_return") and metrics.daily_return is not None else Decimal("0"),
                number_of_positions=getattr(metrics, "positions_count", 0),
                leverage_ratio=to_decimal(metrics.leverage) if hasattr(metrics, "leverage") and metrics.leverage is not None else None,
                margin_usage=to_decimal(metrics.margin_used) if hasattr(metrics, "margin_used") and metrics.margin_used is not None else None,
                cash_balance=to_decimal(metrics.cash) if hasattr(metrics, "cash") and metrics.cash is not None else None,
            )
        except Exception as e:
            raise ValidationError(
                f"Failed to transform portfolio metrics to database model: {e}",
                context={
                    "bot_id": bot_id,
                    "transformation_type": "portfolio_metrics_to_db",
                },
            ) from e

    def transform_position_metrics_to_db(self, metrics: PositionMetrics, bot_id: str) -> AnalyticsPositionMetrics:
        """
        Transform PositionMetrics to database model.

        Args:
            metrics: Domain position metrics object
            bot_id: Bot identifier for database association

        Returns:
            AnalyticsPositionMetrics: Database model instance

        Raises:
            ValidationError: If transformation validation fails
        """
        try:
            if not bot_id:
                raise ValidationError("bot_id is required for position metrics")

            # Map domain fields to database fields
            return AnalyticsPositionMetrics(
                bot_id=bot_id,
                timestamp=metrics.timestamp,
                position_id=getattr(metrics, "position_id", None),
                symbol=metrics.symbol,
                exchange=getattr(metrics, "exchange", "binance"),  # Default exchange if not provided
                quantity=to_decimal(metrics.quantity),
                market_value=to_decimal(metrics.market_value),
                unrealized_pnl=to_decimal(metrics.unrealized_pnl) if metrics.unrealized_pnl is not None else Decimal("0"),
                realized_pnl=to_decimal(metrics.realized_pnl) if metrics.realized_pnl is not None else Decimal("0"),
                average_price=to_decimal(metrics.entry_price),
                current_price=to_decimal(metrics.current_price),
                position_side=getattr(metrics, "side", "LONG").upper(),
            )
        except Exception as e:
            raise ValidationError(
                f"Failed to transform position metrics to database model: {e}",
                context={
                    "bot_id": bot_id,
                    "symbol": getattr(metrics, "symbol", "unknown"),
                    "transformation_type": "position_metrics_to_db",
                },
            ) from e

    def transform_risk_metrics_to_db(self, metrics: RiskMetrics, bot_id: str) -> AnalyticsRiskMetrics:
        """
        Transform RiskMetrics to database model.

        Args:
            metrics: Domain risk metrics object
            bot_id: Bot identifier for database association

        Returns:
            AnalyticsRiskMetrics: Database model instance

        Raises:
            ValidationError: If transformation validation fails
        """
        try:
            if not bot_id:
                raise ValidationError("bot_id is required for risk metrics")

            # Map domain fields to database fields
            return AnalyticsRiskMetrics(
                bot_id=bot_id,
                timestamp=metrics.timestamp,
                portfolio_var_95=to_decimal(metrics.portfolio_var_95) if metrics.portfolio_var_95 is not None else None,
                portfolio_var_99=to_decimal(metrics.portfolio_var_99) if metrics.portfolio_var_99 is not None else None,
                expected_shortfall_95=to_decimal(getattr(metrics, "expected_shortfall", None)) if getattr(metrics, "expected_shortfall", None) is not None else None,
                maximum_drawdown=to_decimal(metrics.max_drawdown) if metrics.max_drawdown is not None else None,
                volatility=to_decimal(metrics.volatility) if metrics.volatility is not None else None,
                sharpe_ratio=to_decimal(getattr(metrics, "sharpe_ratio", None)) if getattr(metrics, "sharpe_ratio", None) is not None else None,
                sortino_ratio=to_decimal(getattr(metrics, "sortino_ratio", None)) if getattr(metrics, "sortino_ratio", None) is not None else None,
                correlation_risk=to_decimal(getattr(metrics, "correlation_risk", None)) if getattr(metrics, "correlation_risk", None) is not None else None,
                concentration_risk=to_decimal(getattr(metrics, "concentration_risk", None)) if getattr(metrics, "concentration_risk", None) is not None else None,
            )
        except Exception as e:
            raise ValidationError(
                f"Failed to transform risk metrics to database model: {e}",
                context={
                    "bot_id": bot_id,
                    "transformation_type": "risk_metrics_to_db",
                },
            ) from e

    def transform_db_to_portfolio_metrics(self, db_metrics: AnalyticsPortfolioMetrics) -> PortfolioMetrics:
        """
        Transform database model to PortfolioMetrics domain object.

        Args:
            db_metrics: Database portfolio metrics model

        Returns:
            PortfolioMetrics: Domain metrics object

        Raises:
            ValidationError: If transformation validation fails
        """
        try:
            if not db_metrics:
                raise ValidationError("db_metrics is required for transformation")

            # Map database fields to domain fields
            cash_balance = db_metrics.cash_balance or Decimal("0")
            total_value = db_metrics.total_value

            return PortfolioMetrics(
                timestamp=db_metrics.timestamp,
                total_value=total_value,
                cash=cash_balance,
                invested_capital=total_value - cash_balance,
                unrealized_pnl=db_metrics.unrealized_pnl,
                realized_pnl=db_metrics.realized_pnl,
                total_pnl=db_metrics.unrealized_pnl + db_metrics.realized_pnl,
                daily_return=db_metrics.daily_pnl,
                positions_count=db_metrics.number_of_positions or 0,
                leverage=db_metrics.leverage_ratio,
                margin_used=db_metrics.margin_usage,
            )
        except Exception as e:
            raise ValidationError(
                f"Failed to transform database model to portfolio metrics: {e}",
                context={
                    "transformation_type": "db_to_portfolio_metrics",
                },
            ) from e

    def transform_db_to_risk_metrics(self, db_metrics: AnalyticsRiskMetrics) -> RiskMetrics:
        """
        Transform database model to RiskMetrics domain object.

        Args:
            db_metrics: Database risk metrics model

        Returns:
            RiskMetrics: Domain risk metrics object

        Raises:
            ValidationError: If transformation validation fails
        """
        try:
            if not db_metrics:
                raise ValidationError("db_metrics is required for transformation")

            # Map database fields to domain fields
            return RiskMetrics(
                timestamp=db_metrics.timestamp,
                portfolio_var_95=db_metrics.portfolio_var_95,
                portfolio_var_99=db_metrics.portfolio_var_99,
                expected_shortfall=db_metrics.expected_shortfall_95,
                max_drawdown=db_metrics.maximum_drawdown,
                volatility=db_metrics.volatility,
                correlation_risk=db_metrics.correlation_risk,
                concentration_risk=db_metrics.concentration_risk,
            )
        except Exception as e:
            raise ValidationError(
                f"Failed to transform database model to risk metrics: {e}",
                context={
                    "transformation_type": "db_to_risk_metrics",
                },
            ) from e
