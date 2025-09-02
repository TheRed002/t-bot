"""
Base Risk Manager for P-008 Risk Management Framework.

This module provides the abstract base class for all risk management implementations.
It defines the core interface that all risk managers must implement.

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.base.component import BaseComponent
from src.core.config.main import Config
from src.core.exceptions import (
    RiskManagementError,
    ValidationError,
)

# MANDATORY: Import from P-001
from src.core.types import (
    MarketData,
    OrderRequest,
    Position,
    PositionLimits,
    RiskLevel,
    RiskMetrics,
    Signal,
)

# MANDATORY: Import from P-002A
from src.error_handling import ErrorHandler

# Monitoring integration
from src.monitoring.metrics import MetricsCollector
from src.utils.decimal_utils import ZERO, format_decimal

# MANDATORY: Import from P-003+
# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution


class BaseRiskManager(BaseComponent, ABC):
    """
    Abstract base class for risk management implementations.

    This class defines the core interface that all risk managers must implement.
    It provides position sizing, portfolio monitoring, and risk limit enforcement.

    CRITICAL: All implementations must follow the exact interface defined here.
    """

    def __init__(self, config: Config):
        """
        Initialize the risk manager with configuration.

        Args:
            config: Application configuration containing risk settings
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.risk_config = config.risk
        self.error_handler = ErrorHandler(config)

        # Risk state tracking
        self.current_risk_level = RiskLevel.LOW
        self.last_risk_calculation = datetime.now(timezone.utc)
        self.risk_metrics: RiskMetrics | None = None
        self.position_limits: PositionLimits | None = None

        # Portfolio state
        self.positions: list[Position] = []
        self.total_portfolio_value = Decimal("0")
        self.current_drawdown = Decimal("0")
        self.max_drawdown = Decimal("0")

        # Initialize monitoring integration
        try:
            self.metrics_collector: MetricsCollector | None = MetricsCollector()
        except Exception as e:
            self.logger.warning(f"Failed to initialize metrics collector: {e}")
            self.metrics_collector = None

        # Mark as initialized
        self.logger.info(
            "Risk manager initialized",
            risk_config=dict(self.risk_config),
            monitoring_enabled=self.metrics_collector is not None,
        )

    @abstractmethod
    @time_execution
    async def calculate_position_size(self, signal: Signal, portfolio_value: Decimal) -> Decimal:
        """
        Calculate optimal position size for a trading signal.

        Args:
            signal: Trading signal with direction and confidence
            portfolio_value: Current total portfolio value

        Returns:
            Decimal: Calculated position size in base currency

        Raises:
            RiskManagementError: If position size calculation fails
            PositionLimitError: If calculated size exceeds limits
        """
        pass

    @abstractmethod
    @time_execution
    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate a trading signal against risk limits.

        Args:
            signal: Trading signal to validate

        Returns:
            bool: True if signal passes risk validation

        Raises:
            ValidationError: If signal validation fails
        """
        pass

    @abstractmethod
    @time_execution
    async def validate_order(self, order: OrderRequest, portfolio_value: Decimal) -> bool:
        """
        Validate an order request against risk limits.

        Args:
            order: Order request to validate
            portfolio_value: Current total portfolio value

        Returns:
            bool: True if order passes risk validation

        Raises:
            ValidationError: If order validation fails
            PositionLimitError: If order exceeds position limits
        """
        pass

    @abstractmethod
    @time_execution
    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the portfolio.

        Args:
            positions: Current portfolio positions
            market_data: Current market data for all positions

        Returns:
            RiskMetrics: Calculated risk metrics

        Raises:
            RiskManagementError: If risk calculation fails
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    @time_execution
    async def should_exit_position(self, position: Position, market_data: MarketData) -> bool:
        """
        Determine if a position should be closed based on risk criteria.

        Args:
            position: Position to evaluate
            market_data: Current market data for the position

        Returns:
            bool: True if position should be closed
        """
        pass

    # Standard methods that can be overridden
    @time_execution
    async def update_portfolio_state(self, positions: list[Position], portfolio_value: Decimal) -> None:
        """
        Update internal portfolio state for risk calculations.

        Args:
            positions: Current portfolio positions
            portfolio_value: Current total portfolio value
        """
        self.positions = positions
        self.total_portfolio_value = portfolio_value

        # Calculate current drawdown
        if self.risk_metrics:
            self.current_drawdown = self.risk_metrics.current_drawdown or ZERO
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown

        self.logger.info(
            "Portfolio state updated",
            position_count=len(positions),
            portfolio_value=format_decimal(portfolio_value),
        )

    @time_execution
    async def get_risk_summary(self) -> dict[str, Any]:
        """
        Get comprehensive risk summary for monitoring and reporting.

        Returns:
            Dict containing current risk state and metrics
        """
        summary = {
            "risk_level": self.current_risk_level.value,
            "total_positions": len(self.positions),
            "portfolio_value": format_decimal(self.total_portfolio_value),
            "current_drawdown": format_decimal(self.current_drawdown),
            "max_drawdown": format_decimal(self.max_drawdown),
            "last_calculation": self.last_risk_calculation.isoformat(),
            "position_limits": self.position_limits.model_dump() if self.position_limits else None,
            "risk_metrics": self.risk_metrics.model_dump() if self.risk_metrics else None,
        }

        return summary

    @time_execution
    async def emergency_stop(self, reason: str) -> None:
        """
        Trigger emergency stop due to risk violation.

        Args:
            reason: Reason for emergency stop
        """
        self.current_risk_level = RiskLevel.CRITICAL
        self.logger.critical("Emergency stop triggered", reason=reason)

        # Create error context and handle error
        error = RiskManagementError(f"Emergency stop: {reason}")
        error_context = self.error_handler.create_error_context(
            error=error, component="risk_manager", operation="emergency_stop"
        )
        await self.error_handler.handle_error(error, error_context)

    @time_execution
    async def validate_risk_parameters(self) -> bool:
        """
        Validate all risk parameters are within acceptable ranges.

        Returns:
            bool: True if all parameters are valid
        """
        try:
            # Validate position size parameters
            if not (0 < self.risk_config.risk_per_trade <= 1):
                raise ValidationError("Invalid risk per trade percentage")

            if not (self.risk_config.max_position_size > 0):
                raise ValidationError("Invalid max position size")

            # Validate portfolio limits
            if not (0 < self.risk_config.max_portfolio_concentration <= 1):
                raise ValidationError("Invalid max portfolio concentration")

            if not (0 < self.risk_config.max_drawdown <= 1):
                raise ValidationError("Invalid max drawdown percentage")

            # Validate Kelly Criterion parameters
            if not (0 < self.risk_config.kelly_fraction <= 1):
                raise ValidationError("Invalid Kelly fraction")

            # Validate daily loss limit
            if not (self.risk_config.max_daily_loss > 0):
                raise ValidationError("Invalid max daily loss")

            self.logger.info("Risk parameters validated successfully")
            return True

        except Exception as e:
            self.logger.error("Risk parameter validation failed", error=str(e))
            raise ValidationError(f"Risk parameter validation failed: {e}") from e

    def _calculate_portfolio_exposure(self, positions: list[Position]) -> Decimal:
        """
        Calculate total portfolio exposure as percentage of portfolio value.

        Args:
            positions: List of current positions

        Returns:
            Decimal: Portfolio exposure as decimal (0.0 to 1.0)
        """
        try:
            if not self.total_portfolio_value or self.total_portfolio_value == 0:
                return Decimal("0")

            total_exposure = sum(
                abs(pos.quantity * pos.current_price) if pos.quantity and pos.current_price else ZERO
                for pos in positions
            )

            return total_exposure / self.total_portfolio_value
        except Exception as e:
            self.logger.error(f"Portfolio exposure calculation failed: {e}")
            raise

    def _check_drawdown_limit(self, current_drawdown: Decimal) -> bool:
        """
        Check if current drawdown exceeds maximum allowed.

        Args:
            current_drawdown: Current drawdown as decimal

        Returns:
            bool: True if drawdown is within limits
        """
        max_drawdown = Decimal(str(self.risk_config.max_drawdown))
        return current_drawdown <= max_drawdown

    def _check_daily_loss_limit(self, daily_pnl: Decimal) -> bool:
        """
        Check if daily loss exceeds maximum allowed.

        Args:
            daily_pnl: Daily P&L as decimal (negative for losses)

        Returns:
            bool: True if daily loss is within limits
        """
        if daily_pnl >= 0:
            return True  # No loss to check

        # Use the absolute max_daily_loss from config (not a percentage)
        max_daily_loss = self.risk_config.max_daily_loss
        return abs(daily_pnl) <= max_daily_loss

    async def _log_risk_violation(self, violation_type: str, details: dict[str, Any]) -> None:
        """
        Log risk violation for monitoring and alerting.

        Args:
            violation_type: Type of risk violation
            details: Additional violation details
        """
        self.logger.warning(
            "Risk violation detected",
            violation_type=violation_type,
            details=details,
            risk_level=self.current_risk_level.value,
        )

        # Update monitoring metrics
        if self.metrics_collector:
            try:
                severity = self._determine_violation_severity(violation_type, details)
                self.metrics_collector.increment_counter(
                    "risk_limit_violations_total",
                    labels={"limit_type": violation_type, "severity": severity},
                )
            except Exception as e:
                self.logger.warning(f"Failed to update violation metric: {e}")

        self.logger.info("Risk violation details", violation_type=violation_type, details=details)

    def _determine_violation_severity(self, violation_type: str, details: dict[str, Any]) -> str:
        """
        Determine severity of risk violation for monitoring.

        Args:
            violation_type: Type of violation
            details: Violation details

        Returns:
            Severity level: "critical", "high", "medium", or "low"
        """
        critical_violations = {"daily_loss_limit", "drawdown_limit", "emergency_stop"}
        high_violations = {"position_limit", "exposure_limit", "leverage_limit"}
        medium_violations = {"concentration_limit", "sector_exposure", "correlation_limit"}

        if violation_type in critical_violations:
            return "critical"
        elif violation_type in high_violations:
            return "high"
        elif violation_type in medium_violations:
            return "medium"
        else:
            return "low"

    async def cleanup(self) -> None:
        """Cleanup risk manager resources."""
        try:
            # Clear portfolio state with proper resource management
            if hasattr(self, "positions") and self.positions:
                self.positions.clear()

            # Clear metrics and limits references
            self.risk_metrics = None
            self.position_limits = None

            # Clear metrics collector reference to prevent memory leaks
            if hasattr(self, "metrics_collector") and self.metrics_collector:
                try:
                    if hasattr(self.metrics_collector, "close"):
                        await self.metrics_collector.close()
                    elif hasattr(self.metrics_collector, "cleanup"):
                        await self.metrics_collector.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up metrics collector: {e}")
                finally:
                    self.metrics_collector = None

            self.logger.info("Risk manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during risk manager cleanup: {e}")
        finally:
            # Call parent cleanup if it exists
            if hasattr(super(), "cleanup"):
                try:
                    await super().cleanup()
                except Exception as e:
                    self.logger.warning(f"Error in parent cleanup: {e}")
