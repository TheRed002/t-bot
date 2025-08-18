"""
Adaptive Risk Parameters Module

This module implements dynamic risk parameter adjustment based on market regimes.
It provides volatility-adjusted position sizing, correlation-based portfolio limits,
momentum-based stop loss adjustment, and market stress testing.

CRITICAL: This module integrates with existing risk management framework from P-008/P-009.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.exceptions import RiskManagementError, ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import MarketRegime, Position, Signal

# MANDATORY: Import from P-008/P-009
# MANDATORY: Import from P-010 (regime detection)
from src.risk_management.regime_detection import MarketRegimeDetector

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class AdaptiveRiskManager:
    """
    Adaptive risk management system that adjusts parameters based on market regimes.

    This class implements dynamic risk parameter adjustment using market regime detection
    to optimize risk management for different market conditions.
    """

    def __init__(self, config: dict[str, Any], regime_detector: MarketRegimeDetector):
        """Initialize adaptive risk manager with configuration and regime detector."""
        self.config = config
        self.regime_detector = regime_detector

        # Base risk parameters
        self.base_position_size_pct = config.get("base_position_size_pct", 0.02)  # 2%
        self.base_stop_loss_pct = config.get("base_stop_loss_pct", 0.02)  # 2%
        self.base_take_profit_pct = config.get("base_take_profit_pct", 0.04)  # 4%

        # Regime-specific adjustments
        self.regime_adjustments = {
            MarketRegime.LOW_VOLATILITY: {
                "position_size_multiplier": 1.2,  # Increase position size
                "stop_loss_multiplier": 0.8,  # Tighter stops
                "take_profit_multiplier": 1.1,  # Slightly higher targets
                "max_positions_multiplier": 1.1,  # More positions allowed
            },
            MarketRegime.MEDIUM_VOLATILITY: {
                "position_size_multiplier": 1.0,  # Standard sizing
                "stop_loss_multiplier": 1.0,  # Standard stops
                "take_profit_multiplier": 1.0,  # Standard targets
                "max_positions_multiplier": 1.0,  # Standard position count
            },
            MarketRegime.HIGH_VOLATILITY: {
                "position_size_multiplier": 0.7,  # Reduce position size
                "stop_loss_multiplier": 1.3,  # Wider stops
                "take_profit_multiplier": 1.2,  # Higher targets
                "max_positions_multiplier": 0.8,  # Fewer positions
            },
            MarketRegime.CRISIS: {
                "position_size_multiplier": 0.5,  # Minimal position size
                "stop_loss_multiplier": 1.5,  # Very wide stops
                "take_profit_multiplier": 1.5,  # Higher targets
                "max_positions_multiplier": 0.5,  # Minimal positions
            },
            MarketRegime.TRENDING_UP: {
                "position_size_multiplier": 1.1,  # Slightly larger positions
                "stop_loss_multiplier": 0.9,  # Tighter stops
                "take_profit_multiplier": 1.2,  # Higher targets
                "max_positions_multiplier": 1.05,  # More positions
            },
            MarketRegime.TRENDING_DOWN: {
                "position_size_multiplier": 0.8,  # Smaller positions
                "stop_loss_multiplier": 1.2,  # Wider stops
                "take_profit_multiplier": 0.9,  # Lower targets
                "max_positions_multiplier": 0.9,  # Fewer positions
            },
            MarketRegime.RANGING: {
                "position_size_multiplier": 0.9,  # Smaller positions
                "stop_loss_multiplier": 1.1,  # Wider stops
                "take_profit_multiplier": 0.8,  # Lower targets
                "max_positions_multiplier": 0.95,  # Fewer positions
            },
        }

        # Correlation-based adjustments
        self.correlation_adjustments = {
            MarketRegime.HIGH_CORRELATION: {
                "position_size_multiplier": 0.8,  # Reduce size due to correlation
                "max_positions_multiplier": 0.7,  # Fewer positions
            },
            MarketRegime.LOW_CORRELATION: {
                "position_size_multiplier": 1.1,  # Increase size due to diversification
                "max_positions_multiplier": 1.1,  # More positions
            },
        }

        # Momentum-based adjustments
        self.momentum_window = config.get("momentum_window", 20)
        self.momentum_threshold = config.get("momentum_threshold", 0.1)

        # Stress testing parameters
        self.stress_test_scenarios = {
            "market_crash": {"price_shock": -0.20, "volatility_multiplier": 3.0},
            "flash_crash": {"price_shock": -0.10, "volatility_multiplier": 5.0},
            "volatility_spike": {"price_shock": -0.05, "volatility_multiplier": 2.5},
            "correlation_breakdown": {"price_shock": -0.15, "volatility_multiplier": 2.0},
        }

        logger.info(
            "Adaptive risk manager initialized",
            base_position_size=self.base_position_size_pct,
            base_stop_loss=self.base_stop_loss_pct,
        )

    @time_execution
    async def calculate_adaptive_position_size(
        self, signal: Signal, current_regime: MarketRegime, portfolio_value: Decimal
    ) -> Decimal:
        """
        Calculate adaptive position size based on market regime.

        Args:
            signal: Trading signal
            current_regime: Current market regime
            portfolio_value: Current portfolio value

        Returns:
            Decimal: Adaptive position size
        """
        try:
            # Validate signal
            if not signal.symbol or signal.symbol.strip() == "":
                raise RiskManagementError("Invalid signal: symbol cannot be empty")

            # Get base position size
            base_size = portfolio_value * Decimal(str(self.base_position_size_pct))

            # Apply regime-specific adjustments
            regime_adj = self.regime_adjustments.get(
                current_regime, self.regime_adjustments[MarketRegime.MEDIUM_VOLATILITY]
            )

            position_multiplier = regime_adj["position_size_multiplier"]

            # Apply correlation adjustments if available
            correlation_regime = await self._get_correlation_regime()
            if correlation_regime:
                corr_adj = self.correlation_adjustments.get(correlation_regime, {})
                position_multiplier *= corr_adj.get("position_size_multiplier", 1.0)

            # Apply momentum adjustments
            momentum_multiplier = await self._calculate_momentum_adjustment(signal.symbol)
            position_multiplier *= momentum_multiplier

            # Calculate final position size
            adaptive_size = base_size * Decimal(str(position_multiplier))

            # Validate position size
            min_size = portfolio_value * Decimal("0.001")  # 0.1% minimum
            max_size = portfolio_value * Decimal("0.1")  # 10% maximum

            adaptive_size = max(min_size, min(adaptive_size, max_size))

            logger.debug(
                "Adaptive position size calculated",
                symbol=signal.symbol,
                base_size=float(base_size),
                adaptive_size=float(adaptive_size),
                regime=current_regime.value,
                multiplier=position_multiplier,
            )

            return adaptive_size

        except Exception as e:
            logger.error(
                "Error calculating adaptive position size", symbol=signal.symbol, error=str(e)
            )
            raise RiskManagementError(f"Adaptive position sizing failed: {e!s}")

    @time_execution
    async def calculate_adaptive_stop_loss(
        self, signal: Signal, current_regime: MarketRegime, entry_price: Decimal
    ) -> Decimal:
        """
        Calculate adaptive stop loss based on market regime.

        Args:
            signal: Trading signal
            current_regime: Current market regime
            entry_price: Position entry price

        Returns:
            Decimal: Adaptive stop loss price
        """
        try:
            # Validate signal
            if not signal.symbol or signal.symbol.strip() == "":
                raise RiskManagementError("Invalid signal: symbol cannot be empty")

            # Validate entry price
            if entry_price <= 0:
                raise RiskManagementError(
                    f"Invalid entry price: {entry_price}. Price must be positive"
                )

            # Get base stop loss percentage
            base_stop_loss_pct = self.base_stop_loss_pct

            # Apply regime-specific adjustments
            regime_adj = self.regime_adjustments.get(
                current_regime, self.regime_adjustments[MarketRegime.MEDIUM_VOLATILITY]
            )

            stop_loss_multiplier = regime_adj["stop_loss_multiplier"]
            adaptive_stop_loss_pct = Decimal(str(base_stop_loss_pct)) * Decimal(
                str(stop_loss_multiplier)
            )

            # Calculate stop loss price
            if signal.direction.value == "buy":
                stop_loss_price = entry_price * (1 - adaptive_stop_loss_pct)
            else:
                stop_loss_price = entry_price * (1 + adaptive_stop_loss_pct)

            logger.debug(
                "Adaptive stop loss calculated",
                symbol=signal.symbol,
                entry_price=float(entry_price),
                stop_loss_price=float(stop_loss_price),
                regime=current_regime.value,
                multiplier=stop_loss_multiplier,
            )

            return stop_loss_price

        except Exception as e:
            logger.error("Error calculating adaptive stop loss", symbol=signal.symbol, error=str(e))
            raise RiskManagementError(f"Adaptive stop loss calculation failed: {e!s}")

    @time_execution
    async def calculate_adaptive_take_profit(
        self, signal: Signal, current_regime: MarketRegime, entry_price: Decimal
    ) -> Decimal:
        """
        Calculate adaptive take profit based on market regime.

        Args:
            signal: Trading signal
            current_regime: Current market regime
            entry_price: Position entry price

        Returns:
            Decimal: Adaptive take profit price
        """
        try:
            # Validate signal
            if not signal.symbol or signal.symbol.strip() == "":
                raise RiskManagementError("Invalid signal: symbol cannot be empty")

            # Get base take profit percentage
            base_take_profit_pct = self.base_take_profit_pct

            # Apply regime-specific adjustments
            regime_adj = self.regime_adjustments.get(
                current_regime, self.regime_adjustments[MarketRegime.MEDIUM_VOLATILITY]
            )

            take_profit_multiplier = regime_adj["take_profit_multiplier"]
            adaptive_take_profit_pct = Decimal(str(base_take_profit_pct)) * Decimal(
                str(take_profit_multiplier)
            )

            # Calculate take profit price
            if signal.direction.value == "buy":
                take_profit_price = entry_price * (1 + adaptive_take_profit_pct)
            else:
                take_profit_price = entry_price * (1 - adaptive_take_profit_pct)

            logger.debug(
                "Adaptive take profit calculated",
                symbol=signal.symbol,
                entry_price=float(entry_price),
                take_profit_price=float(take_profit_price),
                regime=current_regime.value,
                multiplier=take_profit_multiplier,
            )

            return take_profit_price

        except Exception as e:
            logger.error(
                "Error calculating adaptive take profit", symbol=signal.symbol, error=str(e)
            )
            raise RiskManagementError(f"Adaptive take profit calculation failed: {e!s}")

    @time_execution
    async def calculate_adaptive_portfolio_limits(
        self, current_regime: MarketRegime, base_limits: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate adaptive portfolio limits based on market regime.

        Args:
            current_regime: Current market regime
            base_limits: Base portfolio limits

        Returns:
            Dict[str, Any]: Adaptive portfolio limits
        """
        try:
            # Validate base limits
            if base_limits is None:
                raise RiskManagementError("Base limits cannot be None")

            # Get regime-specific adjustments
            regime_adj = self.regime_adjustments.get(
                current_regime, self.regime_adjustments[MarketRegime.MEDIUM_VOLATILITY]
            )

            # Apply adjustments to portfolio limits
            adaptive_limits = base_limits.copy()

            # Adjust maximum positions
            if "max_positions" in adaptive_limits:
                max_positions_multiplier = regime_adj["max_positions_multiplier"]
                adaptive_limits["max_positions"] = int(
                    adaptive_limits["max_positions"] * max_positions_multiplier
                )

            # Adjust maximum portfolio exposure
            if "max_portfolio_exposure" in adaptive_limits:
                if current_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS]:
                    # Reduce exposure
                    adaptive_limits["max_portfolio_exposure"] *= 0.8
                elif current_regime == MarketRegime.LOW_VOLATILITY:
                    # Increase exposure
                    adaptive_limits["max_portfolio_exposure"] *= 1.1

            # Adjust correlation limits
            correlation_regime = await self._get_correlation_regime()
            if correlation_regime == MarketRegime.HIGH_CORRELATION:
                if "max_correlation_exposure" in adaptive_limits:
                    # Reduce correlation exposure
                    adaptive_limits["max_correlation_exposure"] *= 0.7

            logger.debug(
                "Adaptive portfolio limits calculated",
                regime=current_regime.value,
                adaptive_limits=adaptive_limits,
            )

            return adaptive_limits

        except Exception as e:
            logger.error("Error calculating adaptive portfolio limits", error=str(e))
            raise RiskManagementError(f"Adaptive portfolio limits calculation failed: {e!s}")

    @time_execution
    async def run_stress_test(
        self, portfolio_positions: list[Position], scenario_name: str = "market_crash"
    ) -> dict[str, Any]:
        """
        Run stress test on current portfolio.

        Args:
            portfolio_positions: Current portfolio positions
            scenario_name: Stress test scenario name

        Returns:
            Dict[str, Any]: Stress test results
        """
        try:
            # Validate portfolio positions
            if portfolio_positions is None:
                raise RiskManagementError("Portfolio positions cannot be None")

            if scenario_name not in self.stress_test_scenarios:
                raise ValidationError(f"Unknown stress test scenario: {scenario_name}")

            scenario = self.stress_test_scenarios[scenario_name]

            # Calculate portfolio value before stress
            initial_value = sum(
                float(pos.quantity * pos.current_price) for pos in portfolio_positions
            )

            # Apply stress scenario
            stressed_positions = []
            for position in portfolio_positions:
                # Apply price shock
                price_shock = scenario["price_shock"]
                stressed_price = position.current_price * (1 + Decimal(str(price_shock)))

                # Create stressed position
                stressed_position = Position(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    current_price=stressed_price,
                    unrealized_pnl=position.quantity * (stressed_price - position.entry_price),
                    side=position.side,
                    timestamp=position.timestamp,
                )
                stressed_positions.append(stressed_position)

            # Calculate portfolio value after stress
            stressed_value = sum(
                float(pos.quantity * pos.current_price) for pos in stressed_positions
            )

            # Calculate stress metrics
            value_change = stressed_value - initial_value
            value_change_pct = (value_change / initial_value) if initial_value > 0 else 0

            # Calculate maximum drawdown
            max_drawdown = min(0, value_change_pct)

            stress_results = {
                "scenario": scenario_name,
                "initial_value": initial_value,
                "stressed_value": stressed_value,
                "value_change": value_change,
                "value_change_pct": value_change_pct,
                "max_drawdown": max_drawdown,
                "positions_affected": len(portfolio_positions),
                "timestamp": datetime.now(),
            }

            logger.warning(
                "Stress test completed",
                scenario=scenario_name,
                value_change_pct=value_change_pct,
                max_drawdown=max_drawdown,
            )

            return stress_results

        except ValidationError:
            # Re-raise ValidationError as-is
            raise
        except Exception as e:
            logger.error("Error running stress test", scenario=scenario_name, error=str(e))
            raise RiskManagementError(f"Stress test failed: {e!s}")

    async def _get_correlation_regime(self) -> MarketRegime | None:
        """Get current correlation regime from detector."""
        try:
            # This would typically get correlation regime from the regime detector
            # For now, return None to use default behavior
            return None
        except Exception as e:
            logger.error("Error getting correlation regime", error=str(e))
            return None

    async def _calculate_momentum_adjustment(self, symbol: str) -> float:
        """
        Calculate momentum-based position size adjustment.

        Args:
            symbol: Trading symbol

        Returns:
            float: Momentum adjustment multiplier
        """
        try:
            # TODO: Remove in production - This is a placeholder implementation
            # In production, this would calculate actual momentum from price
            # data
            momentum_multiplier = 1.0  # Default no adjustment

            # Simple momentum calculation (placeholder)
            # This would use actual price data to calculate momentum
            if symbol.endswith("USDT"):
                momentum_multiplier = 1.05  # Slight positive momentum for USDT pairs
            elif symbol.endswith("BTC"):
                momentum_multiplier = 0.95  # Slight negative momentum for BTC pairs

            return momentum_multiplier

        except Exception as e:
            logger.error("Error calculating momentum adjustment", symbol=symbol, error=str(e))
            return 1.0  # Return neutral adjustment on error

    def get_adaptive_parameters(self, regime: MarketRegime) -> dict[str, Any]:
        """
        Get adaptive parameters for a specific regime.

        Args:
            regime: Market regime

        Returns:
            Dict[str, Any]: Adaptive parameters for the regime
        """
        regime_adj = self.regime_adjustments.get(
            regime, self.regime_adjustments[MarketRegime.MEDIUM_VOLATILITY]
        )

        return {
            "position_size_multiplier": regime_adj["position_size_multiplier"],
            "stop_loss_multiplier": regime_adj["stop_loss_multiplier"],
            "take_profit_multiplier": regime_adj["take_profit_multiplier"],
            "max_positions_multiplier": regime_adj["max_positions_multiplier"],
            "regime": regime.value,
        }

    def get_stress_test_scenarios(self) -> list[str]:
        """
        Get available stress test scenarios.

        Returns:
            List[str]: Available scenario names
        """
        return list(self.stress_test_scenarios.keys())

    def update_regime_detector(self, new_detector: MarketRegimeDetector) -> None:
        """
        Update the regime detector reference.

        Args:
            new_detector: New regime detector instance
        """
        self.regime_detector = new_detector
        logger.info("Regime detector updated")
