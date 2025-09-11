"""
Risk Metrics Service Implementation.

This service handles all risk metrics calculations through dependency injection,
following proper service layer patterns.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

from src.core.base.service import BaseService
from src.core.exceptions import RiskManagementError
from src.core.types import MarketData, Position, RiskLevel, RiskMetrics
from src.utils.decimal_utils import (
    ONE,
    ZERO,
    decimal_to_float,
    format_decimal,
    safe_divide,
    to_decimal,
)
from src.utils.messaging_patterns import BoundaryValidator, ErrorPropagationMixin
from src.utils.risk_calculations import (
    calculate_current_drawdown,
    calculate_expected_shortfall,
    calculate_max_drawdown,
    calculate_portfolio_value,
    calculate_sharpe_ratio,
    calculate_var,
    determine_risk_level,
    validate_risk_inputs,
)

if TYPE_CHECKING:
    from src.database.service import DatabaseService
    from src.state import StateService


class RiskMetricsService(BaseService):
    """Service for calculating comprehensive risk metrics."""

    def __init__(
        self,
        database_service: "DatabaseService",
        state_service: "StateService",
        config=None,
        correlation_id: str | None = None,
    ):
        """
        Initialize risk metrics service.

        Args:
            database_service: Database service for data access
            state_service: State service for state management
            config: Application configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="RiskMetricsService",
            config=config.__dict__ if config else {},
            correlation_id=correlation_id,
        )

        self.database_service = database_service
        self.state_service = state_service
        self.config = config

    async def calculate_metrics(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics with consistent boundary validation.

        Args:
            positions: Current portfolio positions
            market_data: Current market data

        Returns:
            Calculated risk metrics

        Raises:
            RiskManagementError: If calculation fails
        """
        try:
            # Create error handler with consistent propagation
            error_handler = ErrorPropagationMixin()

            # Validate inputs at module boundary with batch processing alignment
            try:
                # Convert positions to batch format for consistent validation
                position_batch = []
                for position in positions:
                    position_dict = {
                        "symbol": position.symbol,
                        "quantity": format_decimal(position.quantity)
                        if hasattr(position, "quantity")
                        else "0",
                        "current_price": format_decimal(position.current_price)
                        if hasattr(position, "current_price") and position.current_price
                        else "0",
                        "timestamp": position.timestamp.isoformat()
                        if hasattr(position, "timestamp")
                        else None,
                    }
                    position_batch.append(position_dict)

                # Validate using batch processing paradigm for consistency with monitoring
                from src.utils.messaging_patterns import ProcessingParadigmAligner

                batch_data = ProcessingParadigmAligner.create_batch_from_stream(position_batch)
                for position_dict in batch_data["items"]:
                    BoundaryValidator.validate_database_entity(position_dict, "validate")

                # Convert market data to batch format for consistent processing
                market_batch = []
                for data in market_data:
                    market_dict = {
                        "symbol": data.symbol,
                        "price": format_decimal(data.close) if hasattr(data, "close") else "0",
                        "volume": format_decimal(data.volume)
                        if hasattr(data, "volume") and data.volume
                        else 0,
                        "timestamp": data.timestamp.isoformat()
                        if hasattr(data, "timestamp")
                        else None,
                    }
                    market_batch.append(market_dict)

                # Validate using batch processing paradigm
                market_batch_data = ProcessingParadigmAligner.create_batch_from_stream(market_batch)
                for market_dict in market_batch_data["items"]:
                    BoundaryValidator.validate_database_entity(market_dict, "validate")

            except Exception as e:
                # Check if it's a validation error and propagate accordingly
                if hasattr(e, "__class__") and (
                    "ValidationError" in e.__class__.__name__
                    or "DataValidationError" in e.__class__.__name__
                ):
                    error_handler.propagate_validation_error(e, "risk_metrics_boundary_validation")
                else:
                    error_handler.propagate_service_error(e, "risk_metrics_boundary_validation")
                return await self._create_empty_metrics()

            if not positions:
                return await self._create_empty_metrics()

            # Calculate portfolio value using centralized utility with validation
            portfolio_value = calculate_portfolio_value(positions, market_data)

            # Validate risk calculation inputs using centralized utility
            if not validate_risk_inputs(portfolio_value, positions, market_data):
                self.logger.warning("Risk input validation failed, using default metrics")
                return await self._create_empty_metrics()

            # Update portfolio history
            await self._update_portfolio_history(portfolio_value)

            # Get historical data for calculations
            portfolio_history = await self._get_portfolio_history()

            # Calculate individual risk components using centralized utilities
            returns = await self._calculate_returns_from_history(portfolio_history)

            var_1d = calculate_var(returns, to_decimal("0.95"), 1)
            var_5d = calculate_var(returns, to_decimal("0.95"), 5)
            expected_shortfall = calculate_expected_shortfall(returns)
            max_dd, _, _ = calculate_max_drawdown(portfolio_history)
            current_drawdown = calculate_current_drawdown(portfolio_value, portfolio_history)
            sharpe_ratio = calculate_sharpe_ratio(returns)

            # Calculate additional metrics
            total_exposure = await self._calculate_total_exposure(positions, market_data)
            correlation_risk = await self._calculate_correlation_risk(positions)
            beta = await self._calculate_portfolio_beta(positions)

            # Determine risk level using centralized utility
            risk_level = determine_risk_level(
                var_1d, current_drawdown, sharpe_ratio, portfolio_value
            )

            # Create risk metrics with consistent Decimal types
            # Apply consistent data transformation for financial fields
            from src.utils.messaging_patterns import DataTransformationHandler

            # Create transformation handler for financial consistency
            _ = DataTransformationHandler()

            # Prepare metrics data with financial transformations
            metrics_data = {
                "portfolio_value": portfolio_value,
                "total_exposure": total_exposure,
                "var_1d": var_1d,
                "var_95": var_1d,  # Same as var_1d for 95% confidence
                "var_99": var_5d,  # Approximation for 99% confidence
                "expected_shortfall": expected_shortfall,
                "max_drawdown": max_dd,  # Use proper variable
                "current_drawdown": current_drawdown,
                "correlation_risk": correlation_risk,
                "leverage": safe_divide(total_exposure, portfolio_value, ONE),
            }

            # Apply consistent data transformation patterns matching monitoring module
            from src.utils.messaging_patterns import MessagingCoordinator, ProcessingParadigmAligner

            coordinator = MessagingCoordinator("RiskMetricsTransform")
            transformed_data = coordinator._apply_data_transformation(metrics_data)

            # Align processing modes for consistency with monitoring module
            transformed_data = ProcessingParadigmAligner.align_processing_modes(
                source_mode="batch", target_mode="batch", data=transformed_data
            )

            # Create risk metrics with consistent Decimal precision
            metrics = RiskMetrics(
                timestamp=datetime.now(timezone.utc),
                portfolio_value=transformed_data["portfolio_value"],
                total_exposure=transformed_data["total_exposure"],
                var_1d=transformed_data["var_1d"],
                var_95=transformed_data["var_95"],
                var_99=transformed_data["var_99"],
                expected_shortfall=transformed_data["expected_shortfall"],
                max_drawdown=transformed_data["max_drawdown"],
                current_drawdown=transformed_data["current_drawdown"],
                sharpe_ratio=sharpe_ratio,  # Keep as Decimal for precision
                beta=beta,  # Keep as Decimal for precision
                correlation_risk=transformed_data["correlation_risk"],
                risk_level=risk_level,
                position_count=len(positions),
                leverage=transformed_data["leverage"],  # Keep as Decimal for precision
            )

            # Store metrics in state
            await self._store_metrics(metrics)

            self.logger.info(
                "Risk metrics calculated",
                portfolio_value=format_decimal(portfolio_value),
                var_1d=format_decimal(var_1d),
                risk_level=risk_level.value,
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            raise RiskManagementError(f"Risk metrics calculation failed: {e}") from e

    async def get_portfolio_value(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> Decimal:
        """
        Calculate current portfolio value using centralized utility.

        Args:
            positions: Current positions
            market_data: Market data for positions

        Returns:
            Current portfolio value
        """
        # Use centralized utility for portfolio value calculation
        return calculate_portfolio_value(positions, market_data)

    async def _create_empty_metrics(self) -> RiskMetrics:
        """Create empty risk metrics for portfolios with no positions."""
        return RiskMetrics(
            timestamp=datetime.now(timezone.utc),
            portfolio_value=ZERO,
            total_exposure=ZERO,
            var_1d=ZERO,
            var_95=ZERO,
            var_99=ZERO,
            expected_shortfall=ZERO,
            max_drawdown=ZERO,
            current_drawdown=ZERO,
            sharpe_ratio=None,
            beta=None,
            correlation_risk=ZERO,
            risk_level=RiskLevel.LOW,
            position_count=0,
            leverage=ONE,
        )

    async def _update_portfolio_history(self, portfolio_value: Decimal) -> None:
        """Update portfolio value history."""
        try:
            # Store portfolio value in state service
            from src.core.types import StateType

            history_data = (
                await self.state_service.get_state(StateType.RISK_STATE, "portfolio_history") or []
            )
            history_data.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "value": str(portfolio_value),
                }
            )

            if len(history_data) > 252:
                history_data = history_data[-252:]

            from src.core.types import StateType

            await self.state_service.set_state(
                state_type=StateType.RISK_STATE,
                state_id="portfolio_history",
                state_data=history_data,
                source_component="RiskMetricsService",
                reason="Portfolio history update",
            )

        except Exception as e:
            self.logger.error(f"Error updating portfolio history: {e}")

    async def _get_portfolio_history(self) -> list[Decimal]:
        """Get portfolio value history."""
        try:
            from src.core.types import StateType

            history_data = (
                await self.state_service.get_state(StateType.RISK_STATE, "portfolio_history") or []
            )
            return [to_decimal(item["value"]) for item in history_data if "value" in item]
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {e}")
            return []

    async def _calculate_returns_from_history(self, history: list[Decimal]) -> list[Decimal]:
        """
        Calculate returns from portfolio value history.

        Args:
            history: Portfolio value history

        Returns:
            List of portfolio returns
        """
        if len(history) < 2:
            return []

        returns = []
        for i in range(1, len(history)):
            if history[i - 1] > ZERO:
                return_val = safe_divide(history[i] - history[i - 1], history[i - 1], ZERO)
                returns.append(return_val)

        return returns

    async def _calculate_var(
        self, days: int, portfolio_value: Decimal, history: list[Decimal]
    ) -> Decimal:
        """Calculate Value at Risk."""
        if len(history) < 30:
            # Conservative estimate for insufficient data
            base_var_pct = to_decimal("0.02")  # 2% base VaR
            scaled_var_pct = base_var_pct * to_decimal(str(np.sqrt(days)))
            return portfolio_value * scaled_var_pct

        # Calculate returns using Decimal precision
        returns = []
        for i in range(1, len(history)):
            if history[i - 1] > ZERO:
                return_val = safe_divide(history[i] - history[i - 1], history[i - 1], ZERO)
                returns.append(
                    decimal_to_float(return_val)
                )  # Convert to float only for numpy operations

        if len(returns) < 10:
            return portfolio_value * to_decimal("0.02")

        # Calculate VaR using percentile method with proper precision
        returns_array = np.array(returns)
        daily_var = np.percentile(returns_array, 5)  # 95% confidence level

        # Scale for time horizon using Decimal for precision
        scaling_factor = Decimal(str(np.sqrt(days)))
        scaled_var = to_decimal(str(daily_var)) * scaling_factor

        return portfolio_value * abs(scaled_var)

    async def _calculate_expected_shortfall(
        self, portfolio_value: Decimal, history: list[Decimal]
    ) -> Decimal:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(history) < 30:
            return portfolio_value * to_decimal("0.025")

        # Calculate returns using Decimal precision
        returns = []
        for i in range(1, len(history)):
            if history[i - 1] > ZERO:
                return_val = (history[i] - history[i - 1]) / history[i - 1]
                returns.append(
                    decimal_to_float(return_val)
                )  # Convert to float only for numpy operations

        if not returns:
            return portfolio_value * to_decimal("0.025")

        # Find worst 5% of returns
        returns_array = np.array(returns)
        threshold = np.percentile(returns_array, 5)
        worst_returns = returns_array[returns_array <= threshold]

        if len(worst_returns) == 0:
            return portfolio_value * to_decimal("0.02")

        # Use Decimal for final calculation
        mean_worst = to_decimal(str(abs(np.mean(worst_returns))))
        expected_shortfall = portfolio_value * mean_worst
        return expected_shortfall

    async def _calculate_max_drawdown(self, history: list[Decimal]) -> Decimal:
        """Calculate maximum historical drawdown."""
        if len(history) < 2:
            return ZERO

        values = [decimal_to_float(val) for i, val in enumerate(history)]
        running_max = np.maximum.accumulate(values)
        drawdowns = (running_max - values) / running_max

        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        return to_decimal(str(max_drawdown))

    async def _calculate_current_drawdown(
        self, portfolio_value: Decimal, history: list[Decimal]
    ) -> Decimal:
        """Calculate current drawdown from peak."""
        if len(history) < 2:
            return ZERO

        peak_value = max(history)
        if peak_value <= ZERO:
            return ZERO

        current_drawdown = safe_divide(peak_value - portfolio_value, peak_value, ZERO)
        return max(ZERO, current_drawdown)

    async def _calculate_sharpe_ratio(self, history: list[Decimal]) -> Decimal | None:
        """Calculate Sharpe ratio."""
        if len(history) < 30:
            return None

        # Calculate returns using Decimal precision
        returns = []
        for i in range(1, len(history)):
            if history[i - 1] > ZERO:
                return_val = (history[i] - history[i - 1]) / history[i - 1]
                returns.append(
                    decimal_to_float(return_val)
                )  # Convert to float only for numpy operations

        if len(returns) < 30:
            return None

        returns_array = np.array(returns)

        # Annualized metrics with Decimal precision
        mean_return = to_decimal(str(np.mean(returns_array) * 252))
        volatility = to_decimal(str(np.std(returns_array) * np.sqrt(252)))

        if volatility == ZERO:
            return None

        sharpe_ratio = safe_divide(mean_return, volatility, ZERO)
        return sharpe_ratio

    async def _calculate_total_exposure(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> Decimal:
        """Calculate total portfolio exposure."""
        total_exposure = ZERO
        price_lookup = {data.symbol: data.close for data in market_data}

        for position in positions:
            current_price = price_lookup.get(position.symbol, position.current_price)
            if current_price and current_price > ZERO:
                position_exposure = abs(position.quantity) * current_price
                total_exposure += position_exposure

        return total_exposure

    async def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal:
        """Calculate correlation risk between positions."""
        if len(positions) < 2:
            return ZERO

        # Simplified correlation risk calculation using Decimal precision
        # In production, this would use actual correlation matrices
        unique_symbols = len(set(pos.symbol for pos in positions))
        if unique_symbols == 0:
            return ZERO

        concentration = safe_divide(ONE, to_decimal(str(unique_symbols)), ZERO)
        return ONE - concentration

    async def _calculate_portfolio_beta(self, positions: list[Position]) -> Decimal | None:
        """Calculate portfolio beta (placeholder for future implementation)."""
        # This would require market index data
        return None

    async def _determine_risk_level(
        self,
        var_1d: Decimal,
        current_drawdown: Decimal,
        sharpe_ratio: Decimal | None,
        portfolio_value: Decimal,
    ) -> RiskLevel:
        """Determine risk level based on metrics."""
        # Calculate VaR as percentage of portfolio
        var_1d_pct = safe_divide(var_1d, portfolio_value, ZERO) if portfolio_value > ZERO else ZERO

        # Risk thresholds
        var_critical = to_decimal("0.10")  # 10% VaR
        var_high = to_decimal("0.05")  # 5% VaR
        var_medium = to_decimal("0.02")  # 2% VaR

        drawdown_critical = to_decimal("0.20")  # 20% drawdown
        drawdown_high = to_decimal("0.10")  # 10% drawdown
        drawdown_medium = to_decimal("0.05")  # 5% drawdown

        # Check for critical risk
        if var_1d_pct > var_critical or current_drawdown > drawdown_critical:
            return RiskLevel.CRITICAL

        # Check for high risk
        if (
            var_1d_pct > var_high
            or current_drawdown > drawdown_high
            or (sharpe_ratio is not None and sharpe_ratio < to_decimal("-1.0"))
        ):
            return RiskLevel.HIGH

        # Check for medium risk
        if (
            var_1d_pct > var_medium
            or current_drawdown > drawdown_medium
            or (sharpe_ratio is not None and sharpe_ratio < to_decimal("0.5"))
        ):
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    async def _store_metrics(self, metrics: RiskMetrics) -> None:
        """Store calculated metrics in state service."""
        try:
            metrics_data = {
                "timestamp": metrics.timestamp.isoformat(),
                "portfolio_value": str(metrics.portfolio_value),
                "total_exposure": str(metrics.total_exposure),
                "var_1d": str(metrics.var_1d),
                "current_drawdown": str(metrics.current_drawdown),
                "risk_level": metrics.risk_level.value,
                "position_count": metrics.position_count,
            }

            from src.core.types import StateType

            await self.state_service.set_state(
                state_type=StateType.RISK_STATE,
                state_id="latest_metrics",
                state_data=metrics_data,
                source_component="RiskMetricsService",
                reason="Latest risk metrics update",
            )

        except Exception as e:
            self.logger.error(f"Error storing risk metrics: {e}")
