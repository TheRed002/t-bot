"""
Environment-aware Risk Management Integration.

This module extends the Risk Management service with environment awareness,
providing different risk controls and parameters for sandbox vs live trading.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.exceptions import ConfigurationError, RiskManagementError
from src.core.integration.environment_aware_service import (
    EnvironmentAwareServiceMixin,
    EnvironmentContext,
)
from src.core.logging import get_logger
from src.core.types import OrderRequest, Position
from src.utils.decimal_utils import format_decimal

logger = get_logger(__name__)


class EnvironmentAwareRiskConfiguration:
    """Environment-specific risk configuration."""

    @staticmethod
    def get_sandbox_risk_config() -> dict[str, Any]:
        """Get risk configuration for sandbox environment."""
        return {
            "max_position_size_pct": Decimal("0.20"),  # Higher risk tolerance for testing
            "risk_per_trade": Decimal("0.05"),  # 5% risk per trade
            "max_total_exposure_pct": Decimal("1.00"),  # Allow 100% exposure
            "max_daily_loss_pct": Decimal("0.10"),  # 10% daily loss limit
            "position_timeout_minutes": 60,  # 1 hour timeout
            "enable_stop_loss": True,
            "enable_take_profit": True,
            "slippage_tolerance_pct": Decimal("0.05"),  # 5% slippage tolerance
            "max_order_value_usd": Decimal("10000"),  # $10k max order
            "leverage_multiplier": Decimal("2.0"),  # Allow 2x leverage
            "correlation_limit": Decimal("0.8"),  # Allow high correlation
            "volatility_adjustment": True,
            "kelly_adjustment_factor": Decimal("0.8"),  # More aggressive Kelly
            "circuit_breaker_threshold": Decimal("0.15"),  # 15% loss triggers breaker
        }

    @staticmethod
    def get_live_risk_config() -> dict[str, Any]:
        """Get risk configuration for live/production environment."""
        return {
            "max_position_size_pct": Decimal("0.05"),  # Conservative position sizing
            "risk_per_trade": Decimal("0.02"),  # 2% risk per trade
            "max_total_exposure_pct": Decimal("0.50"),  # Max 50% exposure
            "max_daily_loss_pct": Decimal("0.05"),  # 5% daily loss limit
            "position_timeout_minutes": 30,  # 30 minute timeout
            "enable_stop_loss": True,
            "enable_take_profit": True,
            "slippage_tolerance_pct": Decimal("0.02"),  # 2% slippage tolerance
            "max_order_value_usd": Decimal("50000"),  # $50k max order
            "leverage_multiplier": Decimal("1.0"),  # No leverage for safety
            "correlation_limit": Decimal("0.5"),  # Lower correlation limit
            "volatility_adjustment": True,
            "kelly_adjustment_factor": Decimal("0.25"),  # Conservative Kelly
            "circuit_breaker_threshold": Decimal("0.08"),  # 8% loss triggers breaker
        }


class EnvironmentAwareRiskManager(EnvironmentAwareServiceMixin):
    """
    Environment-aware risk management functionality.

    This mixin adds environment-specific risk controls to the Risk Management service.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._environment_risk_configs: dict[str, dict[str, Any]] = {}
        self._environment_risk_states: dict[str, dict[str, Any]] = {}

    async def _update_service_environment(self, context: EnvironmentContext) -> None:
        """Update risk management settings based on environment context."""
        await super()._update_service_environment(context)

        # Get environment-specific risk configuration
        if context.is_production:
            risk_config = EnvironmentAwareRiskConfiguration.get_live_risk_config()
            logger.info(f"Applied live risk configuration for {context.exchange_name}")
        else:
            risk_config = EnvironmentAwareRiskConfiguration.get_sandbox_risk_config()
            logger.info(f"Applied sandbox risk configuration for {context.exchange_name}")

        self._environment_risk_configs[context.exchange_name] = risk_config

        # Initialize risk state tracking for this environment
        self._environment_risk_states[context.exchange_name] = {
            "daily_pnl": Decimal("0"),
            "total_exposure": Decimal("0"),
            "active_positions": 0,
            "circuit_breaker_triggered": False,
            "last_reset": None,
        }

    def get_environment_risk_config(self, exchange: str) -> dict[str, Any]:
        """Get risk configuration for a specific exchange environment."""
        if exchange not in self._environment_risk_configs:
            # Initialize with default config based on current environment
            context = self.get_environment_context(exchange)
            if context.is_production:
                config = EnvironmentAwareRiskConfiguration.get_live_risk_config()
            else:
                config = EnvironmentAwareRiskConfiguration.get_sandbox_risk_config()
            self._environment_risk_configs[exchange] = config

        return self._environment_risk_configs[exchange]

    def calculate_environment_aware_position_size(
        self,
        order_request: OrderRequest,
        exchange: str,
        account_balance: Decimal,
        market_data: Any | None = None,
    ) -> Decimal:
        """Calculate position size considering environment-specific risk parameters."""
        try:
            context = self.get_environment_context(exchange)
            risk_config = self.get_environment_risk_config(exchange)

            # Base position size calculation with safe key access
            if "risk_per_trade" not in risk_config:
                raise ConfigurationError(f"Missing 'risk_per_trade' in risk config for {exchange}")
            if "max_position_size_pct" not in risk_config:
                raise ConfigurationError(
                    f"Missing 'max_position_size_pct' in risk config for {exchange}"
                )

            risk_per_trade = risk_config["risk_per_trade"]
            max_position_size_pct = risk_config["max_position_size_pct"]
        except KeyError as e:
            logger.error(f"Missing configuration key for {exchange}: {e}")
            raise ConfigurationError(f"Risk configuration incomplete for {exchange}: {e}") from e
        except Exception as e:
            logger.error(f"Error accessing risk configuration for {exchange}: {e}")
            raise RiskManagementError(
                f"Failed to calculate position size for {exchange}: {e}"
            ) from e

        # Calculate base position size
        base_position_size = account_balance * risk_per_trade
        max_position_size = account_balance * max_position_size_pct

        # Apply environment-specific adjustments
        if context.is_production:
            # Additional safety measures for production
            position_size = min(base_position_size, max_position_size)

            # Check for correlation limits and portfolio exposure
            if hasattr(self, "_check_portfolio_correlation"):
                correlation_adjustment = self._check_portfolio_correlation(
                    order_request.symbol, exchange
                )
                position_size *= correlation_adjustment

        else:
            # More flexible sizing for sandbox
            position_size = min(base_position_size * Decimal("1.5"), max_position_size)

        # Apply volatility adjustment if enabled
        if risk_config.get("volatility_adjustment") and market_data:
            volatility_factor = self._calculate_volatility_adjustment(market_data, risk_config)
            position_size *= volatility_factor

        # Ensure minimum position size
        min_position_size = account_balance * Decimal("0.001")  # 0.1% minimum
        position_size = max(position_size, min_position_size)

        # Apply maximum order value limit if specified
        if "max_order_value_usd" in risk_config and order_request.price:
            max_order_qty = risk_config["max_order_value_usd"] / order_request.price
            if position_size > max_order_qty:
                position_size = max_order_qty
                logger.warning(f"Position size capped by max order value for {exchange}")

        logger.info(
            f"Calculated environment-aware position size: {position_size} for {exchange} "
            f"(environment: {context.environment.value})"
        )

        return position_size

    async def validate_environment_order(
        self,
        order_request: OrderRequest,
        exchange: str,
        current_positions: list[Position] | None = None,
    ) -> bool:
        """Validate order against environment-specific risk rules."""
        context = self.get_environment_context(exchange)
        risk_config = self.get_environment_risk_config(exchange)

        # Check if circuit breaker is triggered
        risk_state = self._environment_risk_states.get(exchange, {})
        if risk_state.get("circuit_breaker_triggered"):
            raise RiskManagementError(
                f"Circuit breaker triggered for {exchange} - trading disabled",
                context={"exchange": exchange, "environment": context.environment.value},
            )

        # Environment-specific validations
        if context.is_production:
            # Strict validation for production
            if not await self._validate_production_order(order_request, exchange, risk_config):
                return False
        else:
            # More lenient validation for sandbox
            if not await self._validate_sandbox_order(order_request, exchange, risk_config):
                return False

        # Common validations
        return await self._validate_common_risk_rules(
            order_request, exchange, risk_config, current_positions
        )

    async def _validate_production_order(
        self, order_request: OrderRequest, exchange: str, risk_config: dict[str, Any]
    ) -> bool:
        """Production-specific order validations."""

        # Check leverage limits (should be 1.0 for production)
        if hasattr(order_request, "leverage") and order_request.leverage:
            max_leverage = risk_config.get("leverage_multiplier", Decimal("1.0"))
            if order_request.leverage > max_leverage:
                logger.error(
                    f"Leverage {order_request.leverage} exceeds limit {max_leverage} for production"
                )
                return False

        # Check slippage tolerance
        if hasattr(order_request, "expected_slippage"):
            max_slippage = risk_config.get("slippage_tolerance_pct", Decimal("0.02"))
            if order_request.expected_slippage > max_slippage:
                logger.error(
                    f"Expected slippage {order_request.expected_slippage} "
                    f"exceeds tolerance {max_slippage}"
                )
                return False

        # Additional production safeguards
        # Check for suspicious order patterns
        if await self._detect_suspicious_order_pattern(order_request, exchange):
            logger.error(f"Suspicious order pattern detected for {exchange}")
            return False

        return True

    async def _validate_sandbox_order(
        self, order_request: OrderRequest, exchange: str, risk_config: dict[str, Any]
    ) -> bool:
        """Sandbox-specific order validations (more lenient)."""

        # Check basic order value limits
        if order_request.price and order_request.quantity:
            order_value = order_request.price * order_request.quantity
            max_order_value = risk_config.get("max_order_value_usd", Decimal("10000"))

            if order_value > max_order_value:
                logger.warning(f"Order value {order_value} exceeds sandbox limit {max_order_value}")
                # In sandbox, we might just warn but allow the order with reduced size
                return True

        return True

    async def _validate_common_risk_rules(
        self,
        order_request: OrderRequest,
        exchange: str,
        risk_config: dict[str, Any],
        current_positions: list[Position] | None = None,
    ) -> bool:
        """Common risk validations for all environments."""

        # Check position limits
        current_positions = current_positions or []
        max_positions = risk_config.get("max_total_positions", 10)

        if len(current_positions) >= max_positions:
            logger.error(f"Maximum positions ({max_positions}) reached for {exchange}")
            return False

        # Check daily loss limits
        daily_loss_limit = risk_config.get("max_daily_loss_pct", Decimal("0.10"))
        risk_state = self._environment_risk_states.get(exchange, {})
        current_daily_pnl = risk_state.get("daily_pnl", Decimal("0"))

        if current_daily_pnl < -daily_loss_limit:
            logger.error(f"Daily loss limit ({daily_loss_limit}) exceeded for {exchange}")
            return False

        return True

    def _calculate_volatility_adjustment(
        self, market_data: Any, risk_config: dict[str, Any]
    ) -> Decimal:
        """Calculate position size adjustment based on volatility."""
        try:
            # This would use actual market data to calculate volatility
            # For now, return a basic adjustment factor
            target_volatility = risk_config.get("volatility_target", Decimal("0.02"))

            # Extract volatility from market data if available, otherwise use placeholder
            if isinstance(market_data, dict) and "volatility" in market_data:
                estimated_volatility = Decimal(str(market_data["volatility"]))
            else:
                estimated_volatility = Decimal("0.025")  # Placeholder

            if estimated_volatility > target_volatility:
                # Reduce position size for high volatility
                return target_volatility / estimated_volatility
            else:
                # Allow slightly larger position for low volatility
                return min(Decimal("1.2"), target_volatility / estimated_volatility)

        except Exception as e:
            logger.warning(f"Failed to calculate volatility adjustment: {e}")
            return Decimal("1.0")

    async def _detect_suspicious_order_pattern(
        self, order_request: OrderRequest, exchange: str
    ) -> bool:
        """Detect suspicious order patterns that might indicate issues."""
        # Placeholder for sophisticated pattern detection
        # In real implementation, this would check for:
        # - Unusual order sizes
        # - High frequency patterns
        # - Market manipulation patterns
        # - Coordinated trading patterns

        return False  # No suspicious patterns detected

    async def update_environment_risk_state(
        self, exchange: str, pnl_change: Decimal, exposure_change: Decimal, position_change: int = 0
    ) -> None:
        """Update risk state tracking for an environment."""
        if exchange not in self._environment_risk_states:
            self._environment_risk_states[exchange] = {
                "daily_pnl": Decimal("0"),
                "total_exposure": Decimal("0"),
                "active_positions": 0,
                "circuit_breaker_triggered": False,
                "last_reset": None,
            }

        risk_state = self._environment_risk_states[exchange]
        risk_config = self.get_environment_risk_config(exchange)

        # Update state
        risk_state["daily_pnl"] += pnl_change
        risk_state["total_exposure"] += exposure_change
        risk_state["active_positions"] += position_change

        # Check circuit breaker condition
        circuit_breaker_threshold = risk_config.get("circuit_breaker_threshold", Decimal("0.10"))
        if risk_state["daily_pnl"] < -circuit_breaker_threshold:
            risk_state["circuit_breaker_triggered"] = True
            logger.critical(
                f"Circuit breaker triggered for {exchange} - daily loss: {risk_state['daily_pnl']}"
            )

            # Notify external systems
            await self._notify_circuit_breaker_triggered(exchange, risk_state["daily_pnl"])

    async def _notify_circuit_breaker_triggered(self, exchange: str, loss_amount: Decimal) -> None:
        """Notify external systems that circuit breaker was triggered."""
        # This would integrate with alerting systems, notifications, etc.
        logger.critical(
            f"CIRCUIT BREAKER TRIGGERED for {exchange}",
            extra={
                "exchange": exchange,
                "loss_amount": format_decimal(loss_amount),
                "timestamp": str(datetime.now()),
                "severity": "CRITICAL",
            },
        )

    async def reset_environment_risk_state(self, exchange: str) -> None:
        """Reset risk state for an exchange (typically daily reset)."""
        if exchange in self._environment_risk_states:
            risk_state = self._environment_risk_states[exchange]
            risk_state.update(
                {
                    "daily_pnl": Decimal("0"),
                    "total_exposure": Decimal("0"),
                    "circuit_breaker_triggered": False,
                    "last_reset": datetime.now(),
                }
            )

            logger.info(f"Reset risk state for {exchange}")

    def get_environment_risk_metrics(self, exchange: str) -> dict[str, Any]:
        """Get current risk metrics for an exchange environment."""
        context = self.get_environment_context(exchange)
        risk_config = self.get_environment_risk_config(exchange)
        risk_state = self._environment_risk_states.get(exchange, {})

        return {
            "exchange": exchange,
            "environment": context.environment.value,
            "is_production": context.is_production,
            "risk_level": context.risk_level,
            "daily_pnl": format_decimal(risk_state.get("daily_pnl", Decimal("0"))),
            "total_exposure": format_decimal(risk_state.get("total_exposure", Decimal("0"))),
            "active_positions": risk_state.get("active_positions", 0),
            "circuit_breaker_active": risk_state.get("circuit_breaker_triggered", False),
            "max_position_size_pct": format_decimal(
                risk_config.get("max_position_size_pct", Decimal("0.05"))
            ),
            "risk_per_trade": format_decimal(risk_config.get("risk_per_trade", Decimal("0.02"))),
            "max_daily_loss_pct": format_decimal(
                risk_config.get("max_daily_loss_pct", Decimal("0.05"))
            ),
            "circuit_breaker_threshold": format_decimal(
                risk_config.get("circuit_breaker_threshold", Decimal("0.10"))
            ),
            "last_updated": datetime.now().isoformat(),
        }
