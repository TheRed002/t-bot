"""
Slippage prediction model for execution cost estimation.

This module implements sophisticated models to predict execution slippage
based on market conditions, order characteristics, and historical patterns,
enabling better execution planning and cost estimation.

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np

from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import (
    MarketData,
    OrderRequest,
    OrderSide,
    SlippageMetrics,
)

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007A
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class SlippageModel:
    """
    Advanced slippage prediction model for execution cost estimation.

    This model uses multiple approaches to predict execution slippage:
    - Market impact models based on order size and liquidity
    - Historical slippage patterns and regression models
    - Real-time market condition adjustments
    - Volatility and spread-based predictions

    The model provides both point estimates and confidence intervals
    for slippage predictions to support risk-aware execution planning.
    """

    def __init__(self, config: Config):
        """
        Initialize slippage prediction model.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.error_handler = ErrorHandler(config.error_handling)

        # Model parameters
        self.market_impact_coefficient = 0.5  # Square root market impact law coefficient
        self.volatility_adjustment_factor = 1.2  # Volatility impact multiplier
        self.spread_cost_factor = 0.5  # Bid-ask spread cost factor
        self.participation_rate_threshold = 0.1  # 10% participation rate threshold

        # Historical data storage
        self.historical_slippage = {}  # Symbol -> slippage history
        self.market_conditions_history = {}  # Symbol -> market conditions
        self.model_parameters = {}  # Symbol -> fitted model parameters

        # Model configuration
        self.min_historical_samples = 10  # Minimum samples for modeling
        self.confidence_levels = [0.68, 0.95]  # 1σ and 2σ confidence intervals
        self.model_update_frequency_hours = 6  # Update models every 6 hours

        # Market regime indicators
        self.volatility_regimes = {
            "low": {"threshold": 0.01, "multiplier": 0.8},  # < 1% daily vol
            "normal": {"threshold": 0.03, "multiplier": 1.0},  # 1-3% daily vol
            "high": {"threshold": 0.05, "multiplier": 1.5},  # 3-5% daily vol
            "extreme": {"threshold": float("inf"), "multiplier": 2.0},  # > 5% daily vol
        }

        self.logger.info("Slippage prediction model initialized")

    @time_execution
    async def predict_slippage(
        self,
        order: OrderRequest,
        market_data: MarketData,
        participation_rate: float | None = None,
        time_horizon_minutes: int | None = None,
    ) -> SlippageMetrics:
        """
        Predict execution slippage for an order.

        Args:
            order: Order request to analyze
            market_data: Current market data
            participation_rate: Expected participation rate (optional)
            time_horizon_minutes: Execution time horizon (optional)

        Returns:
            SlippageMetrics: Predicted slippage metrics

        Raises:
            ValidationError: If inputs are invalid
            ExecutionError: If prediction fails
        """
        try:
            # Validate inputs
            if not order or not market_data:
                raise ValidationError("Order and market data are required")

            if order.quantity <= 0:
                raise ValidationError("Order quantity must be positive")

            if not market_data.price or market_data.price <= 0:
                raise ValidationError("Valid market price required")

            self.logger.debug(
                "Predicting slippage",
                symbol=order.symbol,
                quantity=float(order.quantity),
                side=order.side.value,
                market_price=float(market_data.price),
            )

            # Calculate market impact slippage
            market_impact_bps = await self._calculate_market_impact_slippage(
                order, market_data, participation_rate, time_horizon_minutes
            )

            # Calculate timing cost slippage
            timing_cost_bps = await self._calculate_timing_cost_slippage(
                order, market_data, time_horizon_minutes
            )

            # Calculate spread cost
            spread_cost_bps = await self._calculate_spread_cost(order, market_data)

            # Calculate volatility adjustment
            volatility_adjustment = await self._calculate_volatility_adjustment(
                order.symbol, market_data
            )

            # Apply volatility adjustment to all components
            adjusted_market_impact = market_impact_bps * volatility_adjustment
            adjusted_timing_cost = timing_cost_bps * volatility_adjustment
            adjusted_spread_cost = spread_cost_bps * volatility_adjustment

            # Calculate total slippage
            total_slippage_bps = (
                adjusted_market_impact + adjusted_timing_cost + adjusted_spread_cost
            )

            # Determine order size to volume ratio
            daily_volume = float(market_data.volume) * 24  # Estimate daily volume
            volume_ratio = float(order.quantity) / max(daily_volume, 1.0)

            # Calculate execution price
            execution_price = await self._calculate_expected_execution_price(
                order, market_data, total_slippage_bps
            )

            # Create slippage metrics
            slippage_metrics = SlippageMetrics(
                symbol=order.symbol,
                order_size=order.quantity,
                market_price=market_data.price,
                execution_price=execution_price,
                price_slippage_bps=total_slippage_bps,
                market_impact_bps=adjusted_market_impact,
                timing_cost_bps=adjusted_timing_cost,
                total_cost_bps=total_slippage_bps + adjusted_spread_cost,
                spread_bps=adjusted_spread_cost,
                volume_ratio=volume_ratio,
                volatility=volatility_adjustment,
                timestamp=datetime.now(timezone.utc),
            )

            self.logger.info(
                "Slippage prediction completed",
                symbol=order.symbol,
                total_slippage_bps=float(total_slippage_bps),
                market_impact_bps=float(adjusted_market_impact),
                timing_cost_bps=float(adjusted_timing_cost),
                volume_ratio=volume_ratio,
            )

            return slippage_metrics

        except Exception as e:
            self.logger.error(f"Slippage prediction failed: {e}")
            raise ExecutionError(f"Slippage prediction failed: {e}")

    async def _calculate_market_impact_slippage(
        self,
        order: OrderRequest,
        market_data: MarketData,
        participation_rate: float | None,
        time_horizon_minutes: int | None,
    ) -> Decimal:
        """
        Calculate market impact slippage using square root law.

        Args:
            order: Order request
            market_data: Market data
            participation_rate: Participation rate
            time_horizon_minutes: Time horizon

        Returns:
            Decimal: Market impact slippage in basis points
        """
        try:
            # Estimate daily volume if not provided
            daily_volume = float(market_data.volume) * 24  # Rough daily volume estimate

            # Calculate order size as fraction of daily volume
            order_size_fraction = float(order.quantity) / max(daily_volume, 1.0)

            # Apply square root market impact law
            # Market Impact = coefficient * sqrt(order_size / daily_volume)
            base_impact = self.market_impact_coefficient * math.sqrt(order_size_fraction)

            # Adjust for participation rate if provided
            if participation_rate and participation_rate > 0:
                # Higher participation rate increases impact
                participation_adjustment = 1.0 + (participation_rate - 0.1) * 2.0
                base_impact *= max(participation_adjustment, 0.5)

            # Adjust for time horizon
            if time_horizon_minutes:
                # Longer time horizon reduces impact (more time to spread execution)
                time_adjustment = 1.0 / math.sqrt(max(time_horizon_minutes / 60.0, 0.1))
                base_impact *= time_adjustment

            # Convert to basis points
            impact_bps = Decimal(str(base_impact * 10000))  # Convert to bps

            # Cap maximum impact
            max_impact_bps = Decimal("500")  # 5% maximum impact
            impact_bps = min(impact_bps, max_impact_bps)

            return impact_bps

        except Exception as e:
            self.logger.warning(f"Market impact calculation failed: {e}")
            return Decimal("10")  # Default 10 bps

    async def _calculate_timing_cost_slippage(
        self, order: OrderRequest, market_data: MarketData, time_horizon_minutes: int | None
    ) -> Decimal:
        """
        Calculate timing cost slippage based on market volatility.

        Args:
            order: Order request
            market_data: Market data
            time_horizon_minutes: Execution time horizon

        Returns:
            Decimal: Timing cost slippage in basis points
        """
        try:
            # Estimate volatility from recent price data (simplified)
            # In practice, this would use historical returns
            if market_data.high_price and market_data.low_price and market_data.price:
                # Use high-low range as volatility proxy
                daily_range = float(market_data.high_price - market_data.low_price)
                daily_volatility = daily_range / float(market_data.price)
            else:
                # Default volatility assumption
                daily_volatility = 0.02  # 2% daily volatility

            # Calculate timing cost based on execution time
            if time_horizon_minutes and time_horizon_minutes > 0:
                # Timing cost increases with square root of time
                time_factor = math.sqrt(time_horizon_minutes / (24 * 60))  # Fraction of day
                timing_cost = daily_volatility * time_factor
            else:
                # Immediate execution - use intraday volatility
                timing_cost = daily_volatility * 0.1  # 10% of daily vol for immediate execution

            # Convert to basis points
            timing_cost_bps = Decimal(str(timing_cost * 10000))

            # Cap timing cost
            max_timing_cost_bps = Decimal("200")  # 2% maximum timing cost
            timing_cost_bps = min(timing_cost_bps, max_timing_cost_bps)

            return timing_cost_bps

        except Exception as e:
            self.logger.warning(f"Timing cost calculation failed: {e}")
            return Decimal("5")  # Default 5 bps

    async def _calculate_spread_cost(self, order: OrderRequest, market_data: MarketData) -> Decimal:
        """
        Calculate bid-ask spread cost.

        Args:
            order: Order request
            market_data: Market data

        Returns:
            Decimal: Spread cost in basis points
        """
        try:
            if not market_data.bid or not market_data.ask or not market_data.price:
                # Use default spread if bid/ask not available
                default_spread_bps = Decimal("20")  # 20 bps default
                return default_spread_bps

            # Calculate bid-ask spread
            spread = market_data.ask - market_data.bid
            spread_bps = (spread / market_data.price) * Decimal("10000")

            # Apply spread cost factor (typically pay half the spread)
            spread_cost_bps = spread_bps * Decimal(str(self.spread_cost_factor))

            # Adjust based on order side and urgency
            if order.side == OrderSide.BUY:
                # Buy orders typically pay closer to ask
                spread_cost_bps *= Decimal("1.2")
            else:
                # Sell orders typically receive closer to bid
                spread_cost_bps *= Decimal("0.8")

            return spread_cost_bps

        except Exception as e:
            self.logger.warning(f"Spread cost calculation failed: {e}")
            return Decimal("15")  # Default 15 bps

    async def _calculate_volatility_adjustment(self, symbol: str, market_data: MarketData) -> float:
        """
        Calculate volatility-based adjustment factor.

        Args:
            symbol: Trading symbol
            market_data: Market data

        Returns:
            float: Volatility adjustment multiplier
        """
        try:
            # Estimate current volatility
            if market_data.high_price and market_data.low_price and market_data.price:
                daily_range = float(market_data.high_price - market_data.low_price)
                current_volatility = daily_range / float(market_data.price)
            else:
                current_volatility = 0.02  # Default 2% volatility

            # Determine volatility regime
            for regime_name, regime_config in self.volatility_regimes.items():
                if current_volatility <= regime_config["threshold"]:
                    adjustment_factor = regime_config["multiplier"]

                    self.logger.debug(
                        "Volatility regime determined",
                        symbol=symbol,
                        volatility=current_volatility,
                        regime=regime_name,
                        adjustment_factor=adjustment_factor,
                    )

                    return adjustment_factor

            # Fallback to extreme regime
            return self.volatility_regimes["extreme"]["multiplier"]

        except Exception as e:
            self.logger.warning(f"Volatility adjustment calculation failed: {e}")
            return 1.0  # No adjustment

    async def _calculate_expected_execution_price(
        self, order: OrderRequest, market_data: MarketData, slippage_bps: Decimal
    ) -> Decimal:
        """
        Calculate expected execution price including slippage.

        Args:
            order: Order request
            market_data: Market data
            slippage_bps: Total slippage in basis points

        Returns:
            Decimal: Expected execution price
        """
        try:
            base_price = market_data.price
            slippage_fraction = slippage_bps / Decimal("10000")

            if order.side == OrderSide.BUY:
                # For buy orders, slippage increases the price
                execution_price = base_price * (Decimal("1") + slippage_fraction)
            else:
                # For sell orders, slippage decreases the price
                execution_price = base_price * (Decimal("1") - slippage_fraction)

            return execution_price

        except Exception as e:
            self.logger.warning(f"Execution price calculation failed: {e}")
            return market_data.price

    @log_calls
    async def update_historical_data(
        self, symbol: str, actual_slippage: SlippageMetrics, market_conditions: dict[str, Any]
    ) -> None:
        """
        Update historical slippage data for model improvement.

        Args:
            symbol: Trading symbol
            actual_slippage: Actual observed slippage
            market_conditions: Market conditions during execution
        """
        try:
            # Initialize history for symbol if needed
            if symbol not in self.historical_slippage:
                self.historical_slippage[symbol] = []
                self.market_conditions_history[symbol] = []

            # Add new data point
            self.historical_slippage[symbol].append(
                {
                    "timestamp": actual_slippage.timestamp,
                    "slippage_bps": float(actual_slippage.total_cost_bps),
                    "market_impact_bps": float(actual_slippage.market_impact_bps),
                    "timing_cost_bps": float(actual_slippage.timing_cost_bps),
                    "order_size": float(actual_slippage.order_size),
                    "volume_ratio": actual_slippage.volume_ratio,
                    "volatility": actual_slippage.volatility,
                }
            )

            self.market_conditions_history[symbol].append(
                {"timestamp": actual_slippage.timestamp, "conditions": market_conditions}
            )

            # Limit history size
            max_history_size = 1000
            if len(self.historical_slippage[symbol]) > max_history_size:
                self.historical_slippage[symbol] = self.historical_slippage[symbol][
                    -max_history_size:
                ]
                self.market_conditions_history[symbol] = self.market_conditions_history[symbol][
                    -max_history_size:
                ]

            self.logger.debug(
                "Historical slippage data updated",
                symbol=symbol,
                data_points=len(self.historical_slippage[symbol]),
            )

            # Trigger model update if enough data
            if len(self.historical_slippage[symbol]) >= self.min_historical_samples:
                await self._update_model_parameters(symbol)

        except Exception as e:
            self.logger.error(f"Failed to update historical data: {e}")

    async def _update_model_parameters(self, symbol: str) -> None:
        """
        Update model parameters based on historical data.

        Args:
            symbol: Trading symbol
        """
        try:
            if symbol not in self.historical_slippage:
                return

            data = self.historical_slippage[symbol]
            if len(data) < self.min_historical_samples:
                return

            # Extract features and targets
            features = []
            targets = []

            for record in data:
                # Features: volume_ratio, volatility, order_size (log)
                feature_vector = [
                    record["volume_ratio"],
                    record["volatility"],
                    math.log(max(record["order_size"], 1.0)),
                ]
                features.append(feature_vector)
                targets.append(record["slippage_bps"])

            # Simple linear regression using numpy
            X = np.array(features)
            y = np.array(targets)

            # Add intercept term
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

            # Solve normal equations: β = (X'X)^(-1)X'y
            try:
                beta = np.linalg.solve(
                    X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y
                )

                # Store updated parameters
                self.model_parameters[symbol] = {
                    "intercept": float(beta[0]),
                    "volume_ratio_coef": float(beta[1]),
                    "volatility_coef": float(beta[2]),
                    "order_size_coef": float(beta[3]),
                    "updated_at": datetime.now(timezone.utc),
                    "sample_size": len(data),
                }

                self.logger.info(
                    "Model parameters updated",
                    symbol=symbol,
                    sample_size=len(data),
                    intercept=float(beta[0]),
                )

            except np.linalg.LinAlgError:
                self.logger.warning(
                    f"Failed to update model parameters for {symbol}: singular matrix"
                )

        except Exception as e:
            self.logger.error(f"Model parameter update failed for {symbol}: {e}")

    async def get_slippage_confidence_interval(
        self, predicted_slippage: SlippageMetrics, confidence_level: float = 0.95
    ) -> tuple[Decimal, Decimal]:
        """
        Get confidence interval for slippage prediction.

        Args:
            predicted_slippage: Predicted slippage metrics
            confidence_level: Confidence level (0.68, 0.95, etc.)

        Returns:
            tuple: (lower_bound, upper_bound) in basis points
        """
        try:
            # Base prediction
            base_slippage = predicted_slippage.total_cost_bps

            # Use historical data to estimate prediction uncertainty if available
            symbol = predicted_slippage.symbol

            if symbol in self.historical_slippage and len(self.historical_slippage[symbol]) > 5:
                # Calculate prediction error statistics
                errors = []
                for record in self.historical_slippage[symbol]:
                    # Simple error calculation (would be more sophisticated in practice)
                    predicted = Decimal(str(record["slippage_bps"]))  # Placeholder
                    actual = Decimal(str(record["slippage_bps"]))
                    error = abs(predicted - actual)
                    errors.append(float(error))

                # Calculate standard error
                std_error = Decimal(str(np.std(errors))) if errors else Decimal("10")
            else:
                # Default uncertainty estimate
                std_error = base_slippage * Decimal("0.2")  # 20% uncertainty

            # Calculate confidence interval based on normal distribution
            if confidence_level == 0.68:
                z_score = 1.0  # 1 standard deviation
            elif confidence_level == 0.95:
                z_score = 1.96  # 1.96 standard deviations
            else:
                z_score = 2.58  # 2.58 standard deviations for 99%

            margin_of_error = std_error * Decimal(str(z_score))

            lower_bound = max(Decimal("0"), base_slippage - margin_of_error)
            upper_bound = base_slippage + margin_of_error

            return lower_bound, upper_bound

        except Exception as e:
            self.logger.warning(f"Confidence interval calculation failed: {e}")
            # Return wide interval as fallback
            base = predicted_slippage.total_cost_bps
            return base * Decimal("0.5"), base * Decimal("1.5")

    async def get_model_summary(self, symbol: str | None = None) -> dict[str, Any]:
        """
        Get summary of slippage model performance.

        Args:
            symbol: Optional symbol filter

        Returns:
            dict: Model performance summary
        """
        try:
            if symbol:
                symbols = [symbol] if symbol in self.historical_slippage else []
            else:
                symbols = list(self.historical_slippage.keys())

            summary = {
                "symbols_tracked": len(symbols),
                "total_predictions": sum(len(self.historical_slippage[s]) for s in symbols),
                "models_fitted": len(self.model_parameters),
                "symbol_details": {},
            }

            for sym in symbols:
                data_points = len(self.historical_slippage[sym])
                has_model = sym in self.model_parameters

                summary["symbol_details"][sym] = {
                    "data_points": data_points,
                    "has_fitted_model": has_model,
                    "model_parameters": self.model_parameters.get(sym, {}),
                    "last_update": (
                        self.model_parameters[sym]["updated_at"].isoformat() if has_model else None
                    ),
                }

            return summary

        except Exception as e:
            self.logger.error(f"Model summary generation failed: {e}")
            return {"error": str(e)}
