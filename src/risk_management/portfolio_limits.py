"""
Portfolio Limits Module for P-008 Risk Management Framework.

This module enforces portfolio-level risk controls including:
- Maximum positions per strategy/symbol
- Correlation limits between positions
- Sector/asset class exposure limits
- Leverage limits for margin trading
- Concentration risk monitoring

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

from decimal import Decimal
from typing import Any

import numpy as np

from src.core.base.component import BaseComponent
from src.core.config.main import Config
from src.core.exceptions import PositionLimitError, ValidationError

# MANDATORY: Import from P-001
from src.core.types import Position, PositionLimits

# Note: Removed ErrorHandler import to avoid dependency issues
# Complex error handling should be done at service layer
from src.utils.decimal_utils import (
    ZERO,
    format_decimal,
    safe_divide,
    to_decimal,
)

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution


class PortfolioLimits(BaseComponent):
    """
    Portfolio limits enforcer for risk management.

    This class enforces various portfolio-level risk controls to prevent
    excessive concentration and maintain diversification.
    """

    def __init__(self, config: Config):
        """
        Initialize portfolio limits with configuration.

        Args:
            config: Application configuration containing risk settings
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.risk_config = config.risk
        # Note: Removed ErrorHandler to avoid dependency issues

        # Portfolio state tracking
        self.positions: list[Position] = []
        self.total_portfolio_value = Decimal("0")
        self.position_limits: PositionLimits | None = None

        # Correlation tracking using Decimal for financial precision
        self.correlation_matrix: dict[str, dict[str, Decimal]] = {}
        self.return_history: dict[str, list[Decimal]] = {}
        self.price_history: dict[str, list[Decimal]] = {}

        # Sector/asset classification
        self.sector_mapping = {
            "BTC": "cryptocurrency",
            "ETH": "cryptocurrency",
            "BNB": "cryptocurrency",
            "ADA": "cryptocurrency",
            "SOL": "cryptocurrency",
            "DOT": "cryptocurrency",
            "LINK": "cryptocurrency",
            "UNI": "cryptocurrency",
            "MATIC": "cryptocurrency",
            "AVAX": "cryptocurrency",
            "USDT": "stablecoin",
            "USDC": "stablecoin",
            "BUSD": "stablecoin",
            "DAI": "stablecoin",
        }

        self.logger.info("Portfolio limits initialized")

    @time_execution
    async def check_portfolio_limits(self, new_position: Position) -> bool:
        """
        Check if adding a new position would violate portfolio limits.

        Args:
            new_position: Position to be added

        Returns:
            bool: True if position addition is allowed

        Raises:
            PositionLimitError: If portfolio limits would be violated
        """
        try:
            # Validate input
            if not new_position or not new_position.symbol:
                raise ValidationError("Invalid position for portfolio limit check")

            # Check total position count limit
            if not self._check_total_positions_limit(new_position):
                return False

            # Check positions per symbol limit
            if not self._check_positions_per_symbol_limit(new_position):
                return False

            # Check portfolio exposure limit
            if not self._check_portfolio_exposure_limit(new_position):
                return False

            # Check sector exposure limit
            if not self._check_sector_exposure_limit(new_position):
                return False

            # Check correlation exposure limit
            if not self._check_correlation_exposure_limit(new_position):
                return False

            # Check leverage limit
            if not self._check_leverage_limit(new_position):
                return False

            self.logger.info(
                "Portfolio limits check passed",
                symbol=new_position.symbol,
                quantity=format_decimal(new_position.quantity),
            )

            return True

        except Exception as e:
            self.logger.error(
                "Portfolio limits check failed",
                error=str(e),
                symbol=new_position.symbol if new_position else None,
            )
            raise PositionLimitError(f"Portfolio limits check failed: {e}") from e

    @time_execution
    def _check_total_positions_limit(self, new_position: Position) -> bool:
        """
        Check if adding position would exceed total positions limit.

        Args:
            new_position: Position to be added

        Returns:
            bool: True if within total positions limit
        """
        current_positions = len(self.positions)
        max_positions = self.risk_config.max_total_positions

        if current_positions >= max_positions:
            self._log_risk_violation_sync(
                "total_positions_limit",
                {
                    "current_positions": current_positions,
                    "max_positions": max_positions,
                    "new_symbol": new_position.symbol,
                },
            )
            return False

        return True

    @time_execution
    def _check_positions_per_symbol_limit(self, new_position: Position) -> bool:
        """
        Check if adding position would exceed positions per symbol limit.

        Args:
            new_position: Position to be added

        Returns:
            bool: True if within positions per symbol limit
        """
        symbol = new_position.symbol
        current_symbol_positions = sum(1 for pos in self.positions if pos.symbol == symbol)
        max_positions_per_symbol = self.risk_config.max_positions_per_symbol

        if current_symbol_positions >= max_positions_per_symbol:
            self._log_risk_violation_sync(
                "positions_per_symbol_limit",
                {
                    "symbol": symbol,
                    "current_positions": current_symbol_positions,
                    "max_positions_per_symbol": max_positions_per_symbol,
                },
            )
            return False

        return True

    @time_execution
    def _check_portfolio_exposure_limit(self, new_position: Position) -> bool:
        """
        Check if adding position would exceed portfolio exposure limit.

        Args:
            new_position: Position to be added

        Returns:
            bool: True if within portfolio exposure limit
        """
        if not self.total_portfolio_value or self.total_portfolio_value <= ZERO:
            self.logger.warning("No valid portfolio value to check exposure against")
            return True  # No portfolio value to check against

        # Validate new position values
        if (new_position.quantity and new_position.quantity <= ZERO) or (
            new_position.current_price and new_position.current_price <= ZERO
        ):
            self.logger.warning(
                "Invalid new position values",
                quantity=format_decimal(new_position.quantity) if new_position.quantity else "0",
                price=format_decimal(new_position.current_price)
                if new_position.current_price
                else "0",
            )
            return False

        # Calculate current portfolio exposure with validation
        current_exposure = ZERO
        for pos in self.positions:
            if (pos.quantity and pos.quantity > ZERO) and (
                pos.current_price and pos.current_price > ZERO
            ):
                position_value = (
                    pos.quantity * pos.current_price if pos.quantity and pos.current_price else ZERO
                )
                position_exposure = abs(position_value)
                current_exposure += position_exposure

        # Add new position exposure using Decimal precision
        new_value = (
            new_position.quantity * new_position.current_price
            if new_position.quantity and new_position.current_price
            else ZERO
        )
        new_exposure = abs(new_value)
        total_exposure = current_exposure + new_exposure

        # Calculate exposure percentage with safe division
        exposure_percentage = safe_divide(total_exposure, self.total_portfolio_value, ZERO)
        max_exposure = Decimal(str(self.risk_config.max_portfolio_exposure))

        if exposure_percentage > max_exposure:
            self._log_risk_violation_sync(
                "portfolio_exposure_limit",
                {
                    "current_exposure": format_decimal(current_exposure),
                    "new_exposure": format_decimal(new_exposure),
                    "total_exposure": format_decimal(total_exposure),
                    "exposure_percentage": format_decimal(exposure_percentage),
                    "max_exposure": format_decimal(max_exposure),
                    "portfolio_value": format_decimal(self.total_portfolio_value),
                },
            )
            return False

        return True

    @time_execution
    def _check_sector_exposure_limit(self, new_position: Position) -> bool:
        """
        Check if adding position would exceed sector exposure limit.

        Args:
            new_position: Position to be added

        Returns:
            bool: True if within sector exposure limit
        """
        if not self.total_portfolio_value or self.total_portfolio_value == 0:
            return True

        # Determine sector for new position
        symbol_base = new_position.symbol.replace("USDT", "").replace("BTC", "").replace("ETH", "")
        sector = self.sector_mapping.get(symbol_base, "other")

        # Calculate current sector exposure
        sector_exposure = Decimal("0")
        for pos in self.positions:
            pos_symbol_base = pos.symbol.replace("USDT", "").replace("BTC", "").replace("ETH", "")
            pos_sector = self.sector_mapping.get(pos_symbol_base, "other")

            if pos_sector == sector:
                # Calculate sector exposure with Decimal precision
                pos_value = (
                    pos.quantity * pos.current_price if pos.quantity and pos.current_price else ZERO
                )
                sector_exposure += abs(pos_value)

        # Add new position to sector exposure with Decimal precision
        new_pos_value = (
            new_position.quantity * new_position.current_price
            if new_position.quantity and new_position.current_price
            else ZERO
        )
        new_exposure = abs(new_pos_value)
        total_sector_exposure = sector_exposure + new_exposure

        # Calculate sector exposure percentage
        sector_exposure_percentage = total_sector_exposure / self.total_portfolio_value
        max_sector_exposure = Decimal(str(self.risk_config.max_sector_exposure))

        if sector_exposure_percentage > max_sector_exposure:
            self._log_risk_violation_sync(
                "sector_exposure_limit",
                {
                    "sector": sector,
                    "current_sector_exposure": format_decimal(sector_exposure),
                    "new_exposure": format_decimal(new_exposure),
                    "total_sector_exposure": format_decimal(total_sector_exposure),
                    "sector_exposure_percentage": format_decimal(sector_exposure_percentage),
                    "max_sector_exposure": format_decimal(max_sector_exposure),
                },
            )
            return False

        return True

    @time_execution
    def _check_correlation_exposure_limit(self, new_position: Position) -> bool:
        """
        Check if adding position would exceed correlation exposure limit.

        Args:
            new_position: Position to be added

        Returns:
            bool: True if within correlation exposure limit
        """
        if len(self.positions) < 2:
            return True  # Need at least 2 positions for correlation

        # Calculate correlations with existing positions
        symbol = new_position.symbol
        high_correlation_exposure = Decimal("0")

        for pos in self.positions:
            correlation = self._get_correlation(symbol, pos.symbol)

            if correlation > to_decimal("0.7"):
                # Calculate position exposure with Decimal precision
                pos_value = (
                    pos.quantity * pos.current_price if pos.quantity and pos.current_price else ZERO
                )
                position_exposure = abs(pos_value)
                high_correlation_exposure += position_exposure

        # Add new position exposure if it has high correlations
        new_pos_value = (
            new_position.quantity * new_position.current_price
            if new_position.quantity and new_position.current_price
            else ZERO
        )
        new_exposure = abs(new_pos_value)
        total_correlated_exposure = high_correlation_exposure + new_exposure

        # Calculate correlated exposure percentage
        if self.total_portfolio_value > 0:
            correlated_exposure_percentage = total_correlated_exposure / self.total_portfolio_value
            max_correlation_exposure = Decimal(str(self.risk_config.max_correlation_exposure))

            if correlated_exposure_percentage > max_correlation_exposure:
                self._log_risk_violation_sync(
                    "correlation_exposure_limit",
                    {
                        "current_correlated_exposure": format_decimal(high_correlation_exposure),
                        "new_exposure": format_decimal(new_exposure),
                        "total_correlated_exposure": format_decimal(total_correlated_exposure),
                        "correlated_exposure_percentage": format_decimal(
                            correlated_exposure_percentage
                        ),
                        "max_correlation_exposure": format_decimal(max_correlation_exposure),
                    },
                )
                return False

        return True

    @time_execution
    def _check_leverage_limit(self, new_position: Position) -> bool:
        """
        Check if adding position would exceed leverage limit.

        Args:
            new_position: Position to be added

        Returns:
            bool: True if within leverage limit
        """
        # This can be extended for margin trading
        max_leverage = Decimal(str(self.risk_config.max_leverage))

        if max_leverage > Decimal("1.0"):
            self.logger.warning("Leverage trading not yet implemented")
            return False

        return True

    @time_execution
    def _get_correlation(self, symbol1: str, symbol2: str) -> Decimal:
        """
        Get correlation between two symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol

        Returns:
            float: Correlation coefficient (-1 to 1)
        """
        # Get return histories
        returns1 = self.return_history.get(symbol1, [])
        returns2 = self.return_history.get(symbol2, [])

        if len(returns1) < 10 or len(returns2) < 10:
            return ZERO  # Insufficient data

        # Align return series to same length
        min_length = min(len(returns1), len(returns2))
        returns1_aligned = returns1[-min_length:]
        returns2_aligned = returns2[-min_length:]

        if min_length < 10:
            return ZERO  # Need at least 10 data points

        # Convert to float arrays for numpy correlation calculation
        try:
            returns1_float = [float(ret) for ret in returns1_aligned]
            returns2_float = [float(ret) for ret in returns2_aligned]
            correlation = np.corrcoef(returns1_float, returns2_float)[0, 1]
            return to_decimal(str(correlation)) if not np.isnan(correlation) else ZERO
        except Exception as e:
            self.logger.warning(f"Correlation calculation failed: {e}")
            return ZERO

    @time_execution
    async def update_portfolio_state(
        self, positions: list[Position], portfolio_value: Decimal
    ) -> None:
        """
        Update portfolio state for limit calculations.

        Args:
            positions: Current portfolio positions
            portfolio_value: Current total portfolio value
        """
        self.positions = positions
        self.total_portfolio_value = portfolio_value

        self.logger.info(
            "Portfolio state updated",
            position_count=len(positions),
            portfolio_value=format_decimal(portfolio_value),
        )

    @time_execution
    async def update_return_history(self, symbol: str, price: Decimal) -> None:
        """
        Update return history for correlation calculations.

        Args:
            symbol: Trading symbol
            price: Current price as Decimal for financial precision
        """
        # Convert price to Decimal if needed
        price_decimal = to_decimal(price) if not isinstance(price, Decimal) else price

        # Validate price
        if price_decimal <= ZERO:
            self.logger.warning(f"Invalid price for {symbol}: {price_decimal}")
            return
        if symbol not in self.return_history:
            self.return_history[symbol] = []
            self.price_history[symbol] = []

        # Store the current price as Decimal for financial precision
        self.price_history[symbol].append(price_decimal)

        # Calculate return if we have previous price using Decimal precision
        if len(self.price_history[symbol]) > 1:
            prev_price_decimal = self.price_history[symbol][-2]
            if prev_price_decimal > ZERO:
                return_rate_decimal = safe_divide(
                    price_decimal - prev_price_decimal, prev_price_decimal, ZERO
                )
                self.return_history[symbol].append(return_rate_decimal)

        # Keep only recent history
        max_history = 252  # One year of trading days
        if len(self.return_history[symbol]) > max_history:
            self.return_history[symbol] = self.return_history[symbol][-max_history:]
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]

    @time_execution
    async def get_portfolio_summary(self) -> dict[str, Any]:
        """
        Get comprehensive portfolio limits summary.

        Returns:
            Dict containing current portfolio state and limits
        """
        # Calculate current exposures
        total_exposure = sum(
            (abs(pos.quantity * pos.current_price) if pos.quantity and pos.current_price else ZERO
            for pos in self.positions), ZERO
        )

        exposure_percentage = (
            total_exposure / self.total_portfolio_value if self.total_portfolio_value > 0 else ZERO
        )

        # Calculate sector exposures
        sector_exposures = {}
        for pos in self.positions:
            symbol_base = pos.symbol.replace("USDT", "").replace("BTC", "").replace("ETH", "")
            sector = self.sector_mapping.get(symbol_base, "other")

            if sector not in sector_exposures:
                sector_exposures[sector] = Decimal("0")

            sector_exposures[sector] += (
                abs(pos.quantity * pos.current_price)
                if pos.quantity and pos.current_price
                else ZERO
            )

        # Convert to percentages using Decimal precision
        sector_exposure_percentages = {}
        for sector, exposure in sector_exposures.items():
            if self.total_portfolio_value > 0:
                sector_exposure_percentages[sector] = format_decimal(
                    exposure / self.total_portfolio_value
                )
            else:
                sector_exposure_percentages[sector] = "0.0"

        summary = {
            "total_positions": len(self.positions),
            "portfolio_value": format_decimal(self.total_portfolio_value),
            "total_exposure": format_decimal(total_exposure),
            "exposure_percentage": format_decimal(exposure_percentage),
            "max_exposure_percentage": self.risk_config.max_portfolio_exposure,
            "sector_exposures": sector_exposure_percentages,
            "max_sector_exposure": self.risk_config.max_sector_exposure,
            "max_positions": self.risk_config.max_total_positions,
            "max_positions_per_symbol": self.risk_config.max_positions_per_symbol,
            "max_leverage": format_decimal(to_decimal(self.risk_config.max_leverage)),
        }

        return summary

    @time_execution
    async def _log_risk_violation(self, violation_type: str, details: dict[str, Any]) -> None:
        """
        Log risk violation for monitoring and alerting.

        Args:
            violation_type: Type of risk violation
            details: Additional violation details
        """
        self.logger.warning(
            "Portfolio limit violation detected", violation_type=violation_type, details=details
        )

    @time_execution
    def _log_risk_violation_sync(self, violation_type: str, details: dict[str, Any]) -> None:
        """
        Log risk violation for monitoring and alerting (synchronous version).

        Args:
            violation_type: Type of risk violation
            details: Additional violation details
        """
        self.logger.warning(
            "Portfolio limit violation detected", violation_type=violation_type, details=details
        )
