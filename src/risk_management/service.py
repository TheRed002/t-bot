"""
Enterprise-grade Risk Management Service.

This service eliminates all direct database access and provides a comprehensive
risk management framework with:
- Centralized risk calculations and metrics
- Position sizing with Kelly Criterion and volatility adjustment
- Real-time risk monitoring with caching
- Portfolio risk aggregation
- Stop-loss management
- Risk state management via StateService
- Circuit breaker patterns for risk controls

CRITICAL: This is the ONLY risk management service - replaces all direct
database access from risk_manager.py, position_sizing.py, and risk_metrics.py
"""

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel, Field

from src.core.base.interfaces import HealthStatus
from src.core.base.service import BaseService

# Caching imports - using abstraction to avoid tight coupling
from src.core.caching.cache_decorators import cache_risk_metrics, cached
from src.core.exceptions import RiskManagementError, ServiceError, StateError, ValidationError
from src.core.types import (
    AlertSeverity,
    MarketData,
    OrderRequest,
    Position,
    PositionSizeMethod,
    RiskLevel,
    RiskMetrics,
    Signal,
    StateType,
)
from src.core.types.base import ConfigDict

# DatabaseService will be injected
from src.error_handling.decorators import (
    with_circuit_breaker,
    with_error_context,
    with_retry,
)

# Monitoring integration
from src.monitoring.interfaces import MetricsServiceInterface

# Repository interfaces for proper separation of concerns
# StateService will be injected
from src.utils.constants import POSITION_SIZING_LIMITS
from src.utils.decimal_utils import (
    ONE,
    ZERO,
    clamp_decimal,
    decimal_to_float,
    format_decimal,
    safe_divide,
    to_decimal,
)
from src.utils.decorators import cache_result, time_execution, timeout

# Type variable for entities
T = TypeVar("T")


class RiskConfiguration(BaseModel):
    """Risk management configuration model."""

    # Position sizing configuration
    position_sizing_method: PositionSizeMethod = PositionSizeMethod.KELLY_CRITERION
    max_position_size_pct: Decimal = Field(
        default=Decimal("0.10"), ge=Decimal("0.01"), le=Decimal("0.25")
    )
    min_position_size_pct: Decimal = Field(
        default=Decimal("0.01"), ge=Decimal("0.001"), le=Decimal("0.05")
    )
    default_position_size_pct: Decimal = Field(
        default=Decimal("0.05"), ge=Decimal("0.01"), le=Decimal("0.15")
    )

    # Kelly Criterion configuration
    kelly_lookback_days: int = Field(default=30, ge=10, le=252)
    kelly_half_factor: Decimal = Field(default=Decimal("0.5"), ge=Decimal("0.1"), le=Decimal("1.0"))

    # Volatility adjustment
    volatility_window: int = Field(default=20, ge=5, le=100)
    volatility_target: Decimal = Field(
        default=Decimal("0.02"), ge=Decimal("0.005"), le=Decimal("0.10")
    )

    # Portfolio limits
    max_total_positions: int = Field(default=10, ge=1, le=50)
    max_positions_per_symbol: int = Field(default=3, ge=1, le=10)
    max_portfolio_exposure: Decimal = Field(
        default=Decimal("0.80"), ge=Decimal("0.10"), le=Decimal("1.00")
    )
    max_sector_exposure: Decimal = Field(
        default=Decimal("0.30"), ge=Decimal("0.05"), le=Decimal("0.50")
    )
    max_correlation_exposure: Decimal = Field(
        default=Decimal("0.20"), ge=Decimal("0.05"), le=Decimal("0.40")
    )

    # Risk thresholds
    var_strength_level: Decimal = Field(
        default=Decimal("0.95"), ge=Decimal("0.90"), le=Decimal("0.99")
    )
    max_drawdown_threshold: Decimal = Field(
        default=Decimal("0.20"), ge=Decimal("0.05"), le=Decimal("0.50")
    )
    min_sharpe_ratio: Decimal = Field(default=Decimal("0.50"), ge=Decimal("0.0"), le=Decimal("3.0"))

    # Stop-loss configuration
    default_stop_loss_pct: Decimal = Field(
        default=Decimal("0.05"), ge=Decimal("0.01"), le=Decimal("0.20")
    )
    trailing_stop_enabled: bool = Field(default=True)
    trailing_stop_distance: Decimal = Field(
        default=Decimal("0.03"), ge=Decimal("0.01"), le=Decimal("0.10")
    )

    # Risk monitoring
    risk_check_interval: int = Field(default=60, ge=10, le=300)  # seconds
    emergency_stop_threshold: Decimal = Field(
        default=Decimal("0.30"), ge=Decimal("0.10"), le=Decimal("0.50")
    )

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)


class PortfolioMetrics(BaseModel):
    """Portfolio metrics model for caching."""

    total_value: Decimal = ZERO
    total_exposure: Decimal = ZERO
    total_pnl: Decimal = ZERO
    unrealized_pnl: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    position_count: int = 0
    leverage: Decimal = ONE
    correlation_risk: Decimal = ZERO
    concentration_risk: Decimal = ZERO
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "var_95": "0.05",
                "var_99": "0.10",
                "cvar_95": "0.07",
            }
        }
    )


class RiskAlert(BaseModel):
    """Risk alert model."""

    alert_id: str
    alert_type: str
    severity: RiskLevel
    message: str
    details: dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "alert_id": "risk_001",
                "alert_type": "position_limit",
                "severity": "high",
            }
        }
    )


class RiskService(BaseService):
    """
    Enterprise Risk Management Service.

    Provides comprehensive risk management with:
    - Position sizing (Fixed %, Kelly Criterion, Volatility-Adjusted, Confidence-Weighted)
    - Portfolio risk metrics (VaR, Expected Shortfall, Drawdown, Sharpe)
    - Real-time risk monitoring with circuit breakers
    - State management integration
    - Database service integration (NO direct DB access)
    - Caching layer for performance
    - Risk alerts and emergency controls
    """

    def __init__(
        self,
        risk_metrics_repository=None,
        portfolio_repository=None,
        state_service=None,
        analytics_service=None,
        config=None,
        correlation_id: str | None = None,
        metrics_service: MetricsServiceInterface | None = None,
        cache_service=None,  # Use interface instead of concrete implementation
        alert_service=None,  # AlertServiceInterface for monitoring integration
    ):
        """
        Initialize Risk Service.

        Args:
            risk_metrics_repository: Repository for risk metrics data access
            portfolio_repository: Repository for portfolio data access
            state_service: State service for state management (injected)
            analytics_service: Analytics service for risk metrics (injected)
            config: Application configuration
            correlation_id: Request correlation ID
            metrics_service: Metrics service for monitoring (injected)
            alert_service: Alert service for monitoring integration (injected)
        """
        super().__init__(
            name="RiskService",
            config=(
                dict(
                    config.model_dump()
                    if config and hasattr(config, "model_dump")
                    else (config.__dict__ if config and hasattr(config, "__dict__") else {})
                )
                if config
                else {}
            ),
            correlation_id=correlation_id,
        )

        # Service dependencies
        self.risk_metrics_repository = risk_metrics_repository
        self.portfolio_repository = portfolio_repository
        self.state_service = state_service
        self.alert_service = alert_service
        self.analytics_service = analytics_service
        self.app_config = config

        # Risk configuration with defaults if config not provided
        if config and hasattr(config, "risk"):
            self.risk_config = RiskConfiguration(
                position_sizing_method=PositionSizeMethod(config.risk.position_sizing_method),
                max_position_size_pct=to_decimal(config.risk.max_position_size_pct),
                min_position_size_pct=to_decimal(config.risk.get("min_position_size_pct", "0.01")),
                default_position_size_pct=to_decimal(config.risk.default_position_size_pct),
                kelly_lookback_days=config.risk.get("kelly_lookback_days", 30),
                volatility_window=config.risk.get("volatility_window", 20),
                volatility_target=to_decimal(config.risk.get("volatility_target", "0.02")),
                max_total_positions=config.risk.max_total_positions,
                max_positions_per_symbol=config.risk.max_positions_per_symbol,
                max_portfolio_exposure=to_decimal(config.risk.max_portfolio_exposure),
                var_strength_level=to_decimal(config.risk.get("var_strength_level", "0.95")),
            )
        else:
            # Use default configuration
            self.risk_config = RiskConfiguration()

        # Internal state
        self._current_risk_level = RiskLevel.LOW
        self._portfolio_metrics: PortfolioMetrics | None = None
        self._risk_alerts: list[RiskAlert] = []
        self._emergency_stop_triggered = False

        self._price_history: dict[str, list[Decimal]] = defaultdict(list)
        self._return_history: dict[str, list[float]] = defaultdict(list)
        self._portfolio_value_history: list[Decimal] = []

        # Locks for thread-safe access to shared state
        self._state_lock = asyncio.Lock()
        self._history_lock = asyncio.Lock()
        self._emergency_lock = asyncio.Lock()

        # Resource cleanup tracking
        self._cleanup_tasks: set[asyncio.Task] = set()
        self._resource_locks: dict[str, asyncio.Lock] = {
            "state": self._state_lock,
            "history": self._history_lock,
            "emergency": self._emergency_lock,
        }

        # Performance tracking
        self._risk_calculations_count = 0
        self._cache_hit_rate = 0.0
        self._last_risk_check = datetime.now(timezone.utc)

        # Configure circuit breaker for risk-critical operations
        self.configure_circuit_breaker(
            enabled=True,
            threshold=3,  # Lower threshold for risk operations
            timeout=30,  # Shorter timeout for risk recovery
        )

        # Configure retry with shorter delays for risk operations
        self.configure_retry(
            enabled=True,
            max_retries=2,  # Fewer retries for risk operations
            delay=0.5,
            backoff=1.5,
        )

        # Initialize cache service through dependency injection
        self.cache_service = cache_service  # Injected cache service interface

        # Initialize monitoring integration
        self.metrics_service = metrics_service
        self.risk_metrics = None
        if metrics_service:
            # Initialize RiskMetrics if we have a metrics collector
            from src.monitoring.metrics import get_metrics_collector

            collector = get_metrics_collector()
            if collector:
                self.risk_metrics = RiskMetrics(collector)

        self.logger.info(
            "RiskService initialized",
            config=self.risk_config.model_dump(),
            dependencies=["DatabaseService", "StateService"],
            monitoring_enabled=self.metrics_service is not None,
        )

    async def _do_start(self) -> None:
        """Start the risk service."""
        try:
            # Verify dependencies are available
            if not await self._verify_dependencies():
                raise ServiceError("Risk service dependencies not available")

            # Load initial risk state
            await self._load_risk_state()

            # Start background risk monitoring
            self._risk_monitor_task = asyncio.create_task(self._risk_monitoring_loop())

            self.logger.info("RiskService started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start RiskService: {e}")
            raise ServiceError(f"RiskService startup failed: {e}") from e

    async def _do_stop(self) -> None:
        """Stop the risk service."""
        task = None
        try:
            # Cancel background tasks with proper cleanup
            if hasattr(self, "_risk_monitor_task") and self._risk_monitor_task:
                task = self._risk_monitor_task
                try:
                    task.cancel()
                    # Wait for task to be cancelled with timeout
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    self.logger.info("Risk monitoring task cancelled")
                except Exception as e:
                    self.logger.warning(f"Error cancelling risk monitoring task: {e}")
                finally:
                    self._risk_monitor_task = None  # type: ignore
                    task = None

            # Clean up all resources
            await self._cleanup_resources()

            # Clean up any connection resources
            await self._cleanup_connection_resources()

            # Save final risk state
            try:
                await self._save_risk_state()
            except Exception as e:
                self.logger.error(f"Failed to save final risk state: {e}")
                raise

            self.logger.info("RiskService stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping RiskService: {e}")
            # Don't re-raise to prevent shutdown failures
        finally:
            # Ensure cleanup happens even if errors occur
            try:
                await self._cleanup_resources()
                await self._cleanup_connection_resources()
            except Exception as cleanup_error:
                self.logger.error(f"Cleanup error during shutdown: {cleanup_error}")
            finally:
                # Ensure task reference is cleared
                task = None

    # Position Sizing Operations

    @with_error_context(component="risk_management", operation="calculate_position_size")
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=30)
    @with_retry(max_attempts=2, base_delay=0.5)
    @timeout(10.0)
    @time_execution
    @cached(
        ttl=60,
        namespace="risk",
        data_type="risk_metrics",
        key_generator=lambda self, signal, available_capital, current_price, method=None: (
            f"position_size:{signal.symbol}:{signal.direction.value}:{signal.strength}:{method}"
        ),
    )
    async def calculate_position_size(
        self,
        signal: Signal,
        available_capital: Decimal,
        current_price: Decimal,
        method: PositionSizeMethod | None = None,
    ) -> Decimal:
        """
        Calculate optimal position size using specified method.

        Args:
            signal: Trading signal with direction and strength
            available_capital: Available capital for position
            current_price: Current market price
            method: Position sizing method (defaults to config)

        Returns:
            Position size in base currency

        Raises:
            RiskManagementError: If calculation fails
            ValidationError: If inputs are invalid
        """
        return await self.execute_with_monitoring(
            "calculate_position_size",
            self._calculate_position_size_impl,
            signal,
            available_capital,
            current_price,
            method,
        )

    async def _calculate_position_size_impl(
        self,
        signal: Signal,
        available_capital: Decimal,
        current_price: Decimal,
        method: PositionSizeMethod | None,
    ) -> Decimal:
        """Internal implementation of position size calculation."""
        # Validate inputs
        self._validate_position_sizing_inputs(signal, available_capital, current_price)

        # Use configured method if not specified
        if method is None:
            method = self.risk_config.position_sizing_method

        self.logger.info(
            "Calculating position size",
            signal_symbol=signal.symbol,
            signal_strength=signal.strength,
            signal_direction=signal.direction.value,
            available_capital=format_decimal(available_capital),
            current_price=format_decimal(current_price),
            method=method.value,
        )

        # Calculate position size based on method
        if method == PositionSizeMethod.FIXED_PERCENTAGE:
            position_size = await self._fixed_percentage_sizing(signal, available_capital)
        elif method == PositionSizeMethod.KELLY_CRITERION:
            position_size = await self._kelly_criterion_sizing(signal, available_capital)
        elif method == PositionSizeMethod.VOLATILITY_ADJUSTED:
            position_size = await self._volatility_adjusted_sizing(signal, available_capital)
        elif method == PositionSizeMethod.CONFIDENCE_WEIGHTED:
            position_size = await self._strength_weighted_sizing(signal, available_capital)
        else:
            raise ValidationError(f"Unsupported position sizing method: {method}")

        # Apply hard limits
        position_size = self._apply_position_size_limits(position_size, available_capital)

        # Validate against portfolio constraints
        position_size = await self._apply_portfolio_constraints(
            position_size, signal.symbol, available_capital
        )

        self.logger.info(
            "Position size calculated",
            symbol=signal.symbol,
            method=method.value,
            raw_size=format_decimal(position_size),
            final_size=format_decimal(position_size),
            capital_utilization=format_decimal(safe_divide(position_size, available_capital, ZERO)),
        )

        return position_size

    async def _fixed_percentage_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal:
        """Calculate position size using fixed percentage method."""
        base_size = available_capital * self.risk_config.default_position_size_pct
        strength_adjusted_size = base_size * to_decimal(signal.strength)

        self.logger.info(
            "Fixed percentage sizing",
            base_size=format_decimal(base_size),
            strength=signal.strength,
            adjusted_size=format_decimal(strength_adjusted_size),
        )

        return strength_adjusted_size

    def _validate_kelly_data(
        self, winning_returns: list, losing_returns: list, returns_decimal: list
    ) -> bool:
        """Validate data for Kelly Criterion calculation."""
        if not winning_returns or not losing_returns:
            self.logger.warning(
                "Insufficient win/loss data for Kelly Criterion",
                winning_trades=len(winning_returns),
                losing_trades=len(losing_returns),
            )
            return False

        total_trades = to_decimal(len(returns_decimal))
        if total_trades <= ZERO:
            self.logger.warning("No trades data available for Kelly calculation")
            return False

        win_count = to_decimal(len(winning_returns))
        loss_count = to_decimal(len(losing_returns))

        if win_count <= ZERO or loss_count <= ZERO:
            self.logger.warning("Invalid win/loss counts for Kelly calculation")
            return False

        return True

    def _calculate_kelly_statistics(
        self, winning_returns: list, losing_returns: list, returns_decimal: list
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Calculate Kelly statistics: win probability, win/loss ratio, and Kelly fraction."""
        total_trades = to_decimal(len(returns_decimal))
        win_count = to_decimal(len(winning_returns))
        loss_count = to_decimal(len(losing_returns))

        win_probability = win_count / total_trades

        avg_win = sum(winning_returns, ZERO) / win_count
        avg_loss = abs(sum(losing_returns, ZERO) / loss_count)

        # Check for a reasonable min_avg_loss_threshold or use a default
        min_threshold = POSITION_SIZING_LIMITS.get("min_avg_loss_threshold", 0.001)
        if avg_loss <= to_decimal(str(min_threshold)) or avg_loss <= ZERO:
            raise ValidationError(f"Invalid average loss: {avg_loss}")

        win_loss_ratio = safe_divide(avg_win, avg_loss, ZERO)
        if win_loss_ratio <= ZERO:
            raise ValidationError(f"Invalid win/loss ratio: {win_loss_ratio}")

        if win_probability <= ZERO or (ONE - win_probability) <= ZERO:
            raise ValidationError(f"Invalid probabilities: {win_probability}")

        kelly_fraction = safe_divide(
            win_probability * win_loss_ratio - (ONE - win_probability), win_loss_ratio, ZERO
        )

        return win_probability, win_loss_ratio, kelly_fraction

    @cache_result(ttl=60)  # Cache Kelly calculations for 1 minute
    async def _kelly_criterion_sizing(self, signal: Signal, available_capital: Decimal) -> Decimal:
        """Calculate position size using Kelly Criterion with Half-Kelly safety."""
        symbol = signal.symbol
        returns = self._return_history.get(symbol, [])

        if len(returns) < self.risk_config.kelly_lookback_days:
            self.logger.warning(
                "Insufficient data for Kelly Criterion, falling back to fixed percentage",
                symbol=symbol,
                available_data=len(returns),
                required_data=self.risk_config.kelly_lookback_days,
            )
            return await self._fixed_percentage_sizing(signal, available_capital)

        try:
            # Get recent returns and prepare data
            recent_returns = returns[-self.risk_config.kelly_lookback_days :]
            returns_decimal = [to_decimal(r) for r in recent_returns]
            winning_returns = [r for r in returns_decimal if r > ZERO]
            losing_returns = [r for r in returns_decimal if r < ZERO]

            # Validate data
            if not self._validate_kelly_data(winning_returns, losing_returns, returns_decimal):
                return await self._fixed_percentage_sizing(signal, available_capital)

            # Calculate statistics
            win_probability, win_loss_ratio, kelly_fraction = self._calculate_kelly_statistics(
                winning_returns, losing_returns, returns_decimal
            )

            if kelly_fraction <= ZERO:
                self.logger.warning(
                    "Negative Kelly fraction (negative edge)",
                    kelly_fraction=format_decimal(kelly_fraction),
                    win_probability=format_decimal(win_probability),
                    win_loss_ratio=format_decimal(win_loss_ratio),
                )
                return available_capital * self.risk_config.min_position_size_pct

            # Apply Half-Kelly and strength adjustment
            half_kelly = kelly_fraction * self.risk_config.kelly_half_factor
            strength_adjusted = half_kelly * to_decimal(signal.strength)

            # Apply bounds
            bounded_fraction = clamp_decimal(
                strength_adjusted,
                self.risk_config.min_position_size_pct,
                self.risk_config.max_position_size_pct,
            )

            position_size = available_capital * bounded_fraction

            self.logger.info(
                "Kelly Criterion calculation",
                win_probability=format_decimal(win_probability),
                win_loss_ratio=format_decimal(win_loss_ratio),
                full_kelly=format_decimal(kelly_fraction),
                half_kelly=format_decimal(half_kelly),
                strength_adjusted=format_decimal(strength_adjusted),
                bounded_fraction=format_decimal(bounded_fraction),
                position_size=format_decimal(position_size),
            )

            return position_size

        except (ValueError, ZeroDivisionError) as e:
            self.logger.warning(f"Kelly Criterion validation failed: {e}")
            return await self._fixed_percentage_sizing(signal, available_capital)
        except Exception as e:
            self.logger.error(f"Kelly Criterion calculation failed: {e}")
            return await self._fixed_percentage_sizing(signal, available_capital)

    async def _volatility_adjusted_sizing(
        self, signal: Signal, available_capital: Decimal
    ) -> Decimal:
        """Calculate position size using volatility adjustment."""
        symbol = signal.symbol
        prices = self._price_history.get(symbol, [])

        if len(prices) < self.risk_config.volatility_window:
            self.logger.warning(
                "Insufficient data for volatility adjustment",
                symbol=symbol,
                available_data=len(prices),
                required_data=self.risk_config.volatility_window,
            )
            return await self._fixed_percentage_sizing(signal, available_capital)

        try:
            # Calculate volatility
            prices_array = np.array(
                [
                    decimal_to_float(p)
                    for i, p in enumerate(prices[-self.risk_config.volatility_window :])
                ]
            )
            returns = np.diff(prices_array) / prices_array[:-1]
            volatility = to_decimal(str(np.std(returns)))

            # Calculate adjustment factor
            target_volatility = self.risk_config.volatility_target
            min_volatility = to_decimal(str(POSITION_SIZING_LIMITS["min_volatility_threshold"]))
            volatility_adjustment = safe_divide(
                target_volatility, max(volatility, min_volatility), ONE
            )

            # Cap adjustment to reasonable bounds - configurable limits
            min_volatility_adjustment = to_decimal(
                str(POSITION_SIZING_LIMITS["min_volatility_adjustment"])
            )
            max_volatility_adjustment = to_decimal(
                str(POSITION_SIZING_LIMITS["max_volatility_adjustment"])
            )
            volatility_adjustment = max(
                min_volatility_adjustment, min(volatility_adjustment, max_volatility_adjustment)
            )

            # Calculate position size
            base_size = available_capital * self.risk_config.default_position_size_pct
            adjusted_size = (
                base_size * to_decimal(volatility_adjustment) * to_decimal(signal.strength)
            )

            self.logger.info(
                "Volatility-adjusted sizing",
                volatility=volatility,
                target_volatility=target_volatility,
                adjustment_factor=volatility_adjustment,
                base_size=format_decimal(base_size),
                adjusted_size=format_decimal(adjusted_size),
            )

            return adjusted_size

        except Exception as e:
            self.logger.error(f"Volatility adjustment failed: {e}")
            return await self._fixed_percentage_sizing(signal, available_capital)

    async def _strength_weighted_sizing(
        self, signal: Signal, available_capital: Decimal
    ) -> Decimal:
        """Calculate position size using strength weighting for ML strategies."""
        base_size = available_capital * self.risk_config.default_position_size_pct

        # Non-linear strength scaling
        strength = to_decimal(signal.strength)
        strength_weight = strength**2  # Square for non-linear scaling

        position_size = base_size * strength_weight

        self.logger.info(
            "Confidence-weighted sizing",
            strength=format_decimal(strength),
            strength_weight=format_decimal(strength_weight),
            base_size=format_decimal(base_size),
            position_size=format_decimal(position_size),
        )

        return position_size

    def _validate_position_sizing_inputs(
        self, signal: Signal, available_capital: Decimal, current_price: Decimal
    ) -> None:
        """Validate inputs for position sizing with comprehensive security checks."""
        # Validate signal object
        if not signal:
            error = ValidationError("Signal cannot be None")
            self._emit_validation_error(error, {"component": "position_sizing", "input": "signal"})
            raise error

        if not hasattr(signal, "symbol") or not signal.symbol:
            error = ValidationError("Signal must have a valid symbol")
            self._emit_validation_error(
                error, {"component": "position_sizing", "signal": str(signal)}
            )
            raise error

        # Sanitize and validate symbol
        if not isinstance(signal.symbol, str):
            error = ValidationError(f"Signal symbol must be string, got {type(signal.symbol)}")
            self._emit_validation_error(
                error, {"component": "position_sizing", "symbol_type": str(type(signal.symbol))}
            )
            raise error

        symbol = signal.symbol.strip().upper()
        if not symbol or len(symbol) > 20:  # Reasonable symbol length limit
            error = ValidationError(f"Invalid symbol format: {signal.symbol}")
            self._emit_validation_error(
                error, {"component": "position_sizing", "symbol": signal.symbol}
            )
            raise error

        # Validate signal strength with bounds checking
        if not hasattr(signal, "strength") or signal.strength is None:
            raise ValidationError("Signal must have a strength value")

        if not isinstance(signal.strength, int | float | Decimal):
            raise ValidationError(f"Signal strength must be numeric, got {type(signal.strength)}")

        if not (0 < signal.strength <= 1):
            raise ValidationError(
                f"Signal strength must be between 0 and 1 (exclusive): {signal.strength}"
            )

        # Validate signal direction
        if not hasattr(signal, "direction") or not signal.direction:
            raise ValidationError("Signal must have a valid direction")

        # Validate available capital with bounds
        if not isinstance(available_capital, Decimal):
            raise ValidationError(
                f"Available capital must be Decimal, got {type(available_capital)}"
            )

        if available_capital <= ZERO:
            raise ValidationError(f"Available capital must be positive: {available_capital}")

        # Sanity check for reasonable capital amounts
        max_reasonable_capital = Decimal("1000000000")  # 1 billion limit
        if available_capital > max_reasonable_capital:
            raise ValidationError(
                f"Available capital exceeds reasonable limit: {available_capital}"
            )

        # Validate current price with bounds
        if not isinstance(current_price, Decimal):
            raise ValidationError(f"Current price must be Decimal, got {type(current_price)}")

        if current_price <= ZERO:
            raise ValidationError(f"Current price must be positive: {current_price}")

        # Sanity check for reasonable price ranges
        min_reasonable_price = Decimal("0.000001")
        max_reasonable_price = Decimal("10000000")  # 10 million per unit
        if not (min_reasonable_price <= current_price <= max_reasonable_price):
            raise ValidationError(f"Current price outside reasonable range: {current_price}")

    def _apply_position_size_limits(
        self, position_size: Decimal, available_capital: Decimal
    ) -> Decimal:
        """Apply hard position size limits."""
        # Maximum limit
        max_size = min(
            available_capital * self.risk_config.max_position_size_pct,
            available_capital * to_decimal(str(POSITION_SIZING_LIMITS["max_position_size_pct"])),
        )

        # Minimum limit
        min_size = available_capital * self.risk_config.min_position_size_pct

        if position_size > max_size:
            self.logger.warning(
                "Position size exceeds maximum, capping",
                calculated_size=format_decimal(position_size),
                max_size=format_decimal(max_size),
            )
            return max_size

        if position_size < min_size:
            self.logger.warning(
                "Position size below minimum, setting to zero",
                calculated_size=format_decimal(position_size),
                min_size=format_decimal(min_size),
            )
            return ZERO

        return position_size

    async def _apply_portfolio_constraints(
        self, position_size: Decimal, symbol: str, available_capital: Decimal
    ) -> Decimal:
        """Apply portfolio-level constraints to position size."""
        try:
            # Get current portfolio metrics
            portfolio_metrics = await self.get_portfolio_metrics()

            # Check total portfolio exposure
            potential_exposure = portfolio_metrics.total_exposure + position_size
            max_exposure = available_capital * self.risk_config.max_portfolio_exposure

            if potential_exposure > max_exposure:
                # Reduce position size to stay within exposure limits
                available_exposure = max_exposure - portfolio_metrics.total_exposure
                if available_exposure <= ZERO:
                    self.logger.warning(
                        "Portfolio exposure limit reached",
                        current_exposure=format_decimal(portfolio_metrics.total_exposure),
                        max_exposure=format_decimal(max_exposure),
                    )
                    return ZERO

                position_size = min(position_size, available_exposure)
                self.logger.warning(
                    "Position size reduced due to portfolio exposure limits",
                    original_size=format_decimal(position_size),
                    adjusted_size=format_decimal(position_size),
                )

            return position_size

        except Exception as e:
            self.logger.error(f"Portfolio constraint application failed: {e}")
            return position_size

    # Risk Metrics and Monitoring

    @with_error_context(component="risk_management", operation="calculate_risk_metrics")
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60)
    @with_retry(max_attempts=3, base_delay=1.0)
    @cache_result(ttl=30)  # Cache risk metrics for 30 seconds
    @time_execution
    @cache_risk_metrics(ttl=60)  # Use specialized risk metrics caching decorator
    async def calculate_risk_metrics(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for current portfolio.

        Args:
            positions: Current portfolio positions
            market_data: Current market data

        Returns:
            Calculated risk metrics

        Raises:
            RiskManagementError: If calculation fails
        """
        return await self.execute_with_monitoring(
            "calculate_risk_metrics", self._calculate_risk_metrics_impl, positions, market_data
        )

    async def _calculate_risk_metrics_impl(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> RiskMetrics:
        """Internal implementation of risk metrics calculation."""
        self._risk_calculations_count += 1

        if not positions:
            return self._create_empty_risk_metrics()

        try:
            # Calculate portfolio value and update history
            portfolio_value = await self._calculate_portfolio_value(positions, market_data)
            await self._update_portfolio_history(portfolio_value)

            # Calculate individual risk components
            var_1d = await self._calculate_var(1, portfolio_value)
            var_5d = await self._calculate_var(5, portfolio_value)
            expected_shortfall = await self._calculate_expected_shortfall(portfolio_value)
            max_drawdown = await self._calculate_max_drawdown()
            current_drawdown = await self._calculate_current_drawdown(portfolio_value)
            sharpe_ratio = await self._calculate_sharpe_ratio()

            # Calculate additional metrics
            total_exposure = sum(
                pos.quantity * pos.current_price if pos.quantity and pos.current_price else ZERO
                for pos in positions
            )
            beta = await self._calculate_portfolio_beta(positions)
            correlation_risk = await self._calculate_correlation_risk(positions)

            # Determine risk level
            risk_level = self._determine_risk_level(
                var_1d=var_1d,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                portfolio_value=portfolio_value,
            )

            # Update current risk level
            self._current_risk_level = risk_level

            # Create risk metrics
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(timezone.utc),
                total_exposure=total_exposure if isinstance(total_exposure, Decimal) else ZERO,
                var_1d=var_1d,
                var_95=var_1d,
                var_99=var_5d,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,  # Keep as Decimal for financial precision
                beta=beta,  # Keep as Decimal for financial precision
                correlation_risk=correlation_risk,
                risk_level=risk_level,
                portfolio_value=portfolio_value,
                position_count=len(positions),
                leverage=safe_divide(
                    total_exposure if isinstance(total_exposure, Decimal) else ZERO,
                    portfolio_value,
                    ONE,
                ),  # Keep as Decimal for financial precision
            )

            # Update portfolio metrics cache
            self._portfolio_metrics = PortfolioMetrics(
                total_value=portfolio_value,
                total_exposure=total_exposure if isinstance(total_exposure, Decimal) else ZERO,
                position_count=len(positions),
                leverage=safe_divide(
                    total_exposure if isinstance(total_exposure, Decimal) else ZERO,
                    portfolio_value,
                    ONE,
                ),
                correlation_risk=correlation_risk,
                timestamp=datetime.now(timezone.utc),
            )

            # Save risk state
            await self._save_risk_metrics(risk_metrics)

            # Send risk metrics to analytics service
            if self.analytics_service:
                try:
                    await self.analytics_service.record_risk_metrics(risk_metrics)
                    self.logger.info("Risk metrics sent to analytics service")
                except Exception as e:
                    self.logger.warning(f"Failed to send risk metrics to analytics: {e}")

            # Update monitoring metrics using RiskMetrics
            if self.risk_metrics:
                try:
                    # Update VaR metrics using RiskMetrics methods
                    if var_1d is not None:
                        var_1d_float = decimal_to_float(var_1d, "risk_var_1d", precision_digits=2)
                        self.risk_metrics.record_var(0.95, "1d", var_1d_float)

                    if var_5d is not None:
                        var_5d_float = decimal_to_float(var_5d, "risk_var_5d", precision_digits=2)
                        self.risk_metrics.record_var(0.95, "5d", var_5d_float)

                    # Update drawdown metrics using RiskMetrics
                    if max_drawdown is not None:
                        drawdown_float = decimal_to_float(
                            max_drawdown, "risk_max_drawdown", precision_digits=4
                        )
                        self.risk_metrics.record_drawdown("30d", drawdown_float)

                    # Update Sharpe ratio using RiskMetrics
                    if sharpe_ratio is not None:
                        sharpe_float = decimal_to_float(
                            sharpe_ratio, "risk_sharpe_ratio", precision_digits=4
                        )
                        self.risk_metrics.record_sharpe_ratio("30d", sharpe_float)
                except Exception as e:
                    self.logger.warning(f"Failed to update monitoring metrics: {e}")

            self.logger.info(
                "Risk metrics calculated",
                portfolio_value=format_decimal(portfolio_value),
                total_exposure=format_decimal(
                    total_exposure if isinstance(total_exposure, Decimal) else ZERO
                ),
                var_1d=format_decimal(var_1d),
                current_drawdown=format_decimal(current_drawdown),
                risk_level=risk_level.value,
                calculation_count=self._risk_calculations_count,
            )

            return risk_metrics

        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            raise RiskManagementError(f"Risk metrics calculation failed: {e}") from e

    def _create_empty_risk_metrics(self) -> RiskMetrics:
        """Create empty risk metrics for portfolios with no positions."""
        return RiskMetrics(
            timestamp=datetime.now(timezone.utc),
            total_exposure=ZERO,
            var_95=ZERO,
            var_99=ZERO,
            expected_shortfall=ZERO,
            max_drawdown=ZERO,
            current_drawdown=ZERO,
            sharpe_ratio=None,
            beta=None,
            correlation_risk=ZERO,
            risk_level=RiskLevel.LOW,
            portfolio_value=ZERO,
            position_count=0,
            leverage=ONE,
        )

    async def _calculate_portfolio_value(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> Decimal:
        """Calculate current portfolio value."""
        portfolio_value = ZERO

        # Create price lookup for efficiency
        price_lookup = {data.symbol: data.price for data in market_data}

        for position in positions:
            current_price = price_lookup.get(position.symbol, position.current_price)
            if current_price > ZERO:
                # Update position with current price
                position.current_price = current_price
                position.unrealized_pnl = position.quantity * (current_price - position.entry_price)

                # Add to portfolio value
                position_value = position.quantity * current_price
                portfolio_value += position_value

        return portfolio_value

    async def _update_portfolio_history(self, portfolio_value: Decimal) -> None:
        """Update portfolio value history."""
        # Store as Decimal for precision
        self._portfolio_value_history.append(portfolio_value)

        # Calculate returns if we have previous values
        if len(self._portfolio_value_history) > 1:
            prev_value = self._portfolio_value_history[-2]
            if prev_value > ZERO:
                portfolio_return = (portfolio_value - prev_value) / prev_value
                self.logger.info(
                    "Portfolio return calculated", portfolio_return=format_decimal(portfolio_return)
                )

        # Limit history size for memory management
        max_portfolio_history = POSITION_SIZING_LIMITS.get("max_portfolio_history", 252)
        if len(self._portfolio_value_history) > max_portfolio_history:
            self._portfolio_value_history = self._portfolio_value_history[-max_portfolio_history:]

    async def _calculate_var(self, days: int, portfolio_value: Decimal) -> Decimal:
        """Calculate Value at Risk for specified time horizon."""
        if len(self._portfolio_value_history) < 30:
            # Conservative estimate for insufficient data
            base_var_pct = to_decimal(POSITION_SIZING_LIMITS["var_base_conservative"])
            scaled_var_pct = base_var_pct * Decimal(str(np.sqrt(days)))
            return portfolio_value * scaled_var_pct

        # Calculate portfolio returns with safe division
        values = self._portfolio_value_history
        returns = []
        for i in range(1, len(values)):
            prev_val = values[i - 1]
            curr_val = values[i]
            if prev_val > ZERO:
                return_val = safe_divide(
                    curr_val - prev_val, prev_val, ZERO
                )  # Keep as Decimal for precision
                returns.append(return_val)

        if not returns or len(returns) < 10:
            # Need at least minimum returns for reliable VaR - use conservative fallback
            min_returns_required = POSITION_SIZING_LIMITS["min_returns_for_var"]
            conservative_var_pct = to_decimal(POSITION_SIZING_LIMITS["var_base_conservative"])
            self.logger.warning(
                "Insufficient returns data for VaR calculation",
                available_returns=len(returns),
                required_minimum=min_returns_required,
            )
            return portfolio_value * conservative_var_pct

        # Calculate daily volatility with validation
        daily_volatility = np.std(returns)
        if daily_volatility == 0 or np.isnan(daily_volatility):
            conservative_var_fallback = to_decimal(
                POSITION_SIZING_LIMITS["var_fallback_conservative"]
            )
            self.logger.warning("Zero or invalid volatility, using conservative VaR estimate")
            return portfolio_value * conservative_var_fallback

        # Z-score for strength level with validation
        strength_level = decimal_to_float(
            to_decimal(self.risk_config.var_strength_level), "var_strength_level"
        )  # Only convert to float for numpy operations
        if not (0.5 <= strength_level <= 0.999):
            self.logger.warning(
                "Invalid VaR confidence level, using default",
                provided_level=strength_level,
                default_level=0.95,
            )
            strength_level = 0.95

        if strength_level == 0.95:
            z_score = 1.645
        elif strength_level == 0.99:
            z_score = 2.326
        else:
            try:
                from scipy.stats import norm

                z_score = norm.ppf(strength_level)
                if np.isnan(z_score) or np.isinf(z_score):
                    z_score = 1.645  # Default to 95% strength
            except ImportError:
                z_score = 1.645  # Default to 95% strength if scipy not available

        # VaR calculation with bounds checking
        if days <= 0:
            self.logger.warning(
                "Invalid VaR time horizon, using default", provided_days=days, default_days=1
            )
            days = 1

        var_percentage = daily_volatility * np.sqrt(days) * z_score

        # Cap VaR at reasonable maximum
        max_var_percentage = POSITION_SIZING_LIMITS["max_var_percentage"]
        var_percentage = min(var_percentage, max_var_percentage)

        return portfolio_value * Decimal(str(var_percentage))

    async def _calculate_expected_shortfall(self, portfolio_value: Decimal) -> Decimal:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(self._portfolio_value_history) < POSITION_SIZING_LIMITS["min_history_for_shortfall"]:
            return portfolio_value * to_decimal(
                POSITION_SIZING_LIMITS["shortfall_base_conservative"]
            )

        # Calculate returns with safe division
        values = self._portfolio_value_history
        returns = []
        for i in range(1, len(values)):
            if values[i - 1] > ZERO:
                return_rate = (values[i] - values[i - 1]) / values[i - 1]
                returns.append(
                    decimal_to_float(return_rate)
                )  # Convert to float only for numpy operations

        if not returns:
            return portfolio_value * Decimal("0.025")

        # Find worst returns below VaR threshold
        strength_level = decimal_to_float(
            to_decimal(self.risk_config.var_strength_level), "var_strength_level"
        )  # Only convert to float for numpy operations
        threshold = np.percentile(returns, (1 - strength_level) * 100)
        worst_returns = [r for r in returns if r <= threshold]

        if not worst_returns:
            return portfolio_value * to_decimal(
                POSITION_SIZING_LIMITS["shortfall_default_conservative"]
            )

        expected_shortfall = portfolio_value * Decimal(str(abs(np.mean(worst_returns))))
        return expected_shortfall

    async def _calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum historical drawdown."""
        if len(self._portfolio_value_history) < 2:
            return ZERO

        # Calculate running maximum and drawdowns with safe division
        values = self._portfolio_value_history
        if not values or all(v <= ZERO for v in values):
            self.logger.warning("All portfolio values are zero or negative")
            return ZERO

        running_max = [values[0]]
        for val in values[1:]:
            running_max.append(max(running_max[-1], val))

        drawdowns = []
        for _i, (curr_val, peak_val) in enumerate(zip(values, running_max, strict=False)):
            if peak_val > ZERO:
                dd = decimal_to_float(
                    safe_divide(peak_val - curr_val, peak_val, ZERO), f"drawdown_{_i}"
                )
                drawdowns.append(dd)

        if not drawdowns:
            return ZERO

        max_drawdown = max(drawdowns)
        return Decimal(str(max_drawdown))

    async def _calculate_current_drawdown(self, portfolio_value: Decimal) -> Decimal:
        """Calculate current drawdown from peak."""
        if len(self._portfolio_value_history) < 2:
            return ZERO

        # Find peak value from Decimal history
        peak_value = max(self._portfolio_value_history)
        if peak_value <= ZERO:
            return ZERO

        # Use Decimal arithmetic with safe division to preserve precision
        current_drawdown_decimal = safe_divide(peak_value - portfolio_value, peak_value, ZERO)
        current_drawdown = max(ZERO, current_drawdown_decimal)
        return current_drawdown

    async def _calculate_sharpe_ratio(self) -> Decimal | None:
        """Calculate Sharpe ratio."""
        if len(self._portfolio_value_history) < 30:
            return None

        # Calculate returns with safe division
        values = self._portfolio_value_history
        returns = []
        for i in range(1, len(values)):
            if values[i - 1] > ZERO:
                return_rate = (values[i] - values[i - 1]) / values[i - 1]
                returns.append(
                    decimal_to_float(return_rate)
                )  # Convert to float only for numpy operations

        if not returns:
            return None

        # Annualized metrics
        mean_return = np.mean(returns) * 252  # Annualized
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        if volatility == 0:
            return None

        sharpe_ratio = mean_return / volatility
        return Decimal(str(sharpe_ratio))

    async def _calculate_portfolio_beta(self, positions: list[Position]) -> Decimal | None:
        """Calculate portfolio beta (placeholder for future implementation)."""
        # This would require market index data
        # For now, return None indicating beta calculation is not available
        return None

    async def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal:
        """Calculate correlation risk between positions."""
        if len(positions) < 2:
            return ZERO

        # Simplified correlation risk calculation
        # In production, this would use historical correlation matrices
        position_count = len(positions)
        max_correlation_risk = Decimal(str(position_count - 1)) / Decimal(str(position_count))

        # Apply a conservative estimate
        return max_correlation_risk * Decimal("0.5")  # 50% of theoretical maximum

    def _determine_risk_level(
        self,
        var_1d: Decimal,
        current_drawdown: Decimal,
        sharpe_ratio: Decimal | None,
        portfolio_value: Decimal,
    ) -> RiskLevel:
        """Determine risk level based on current metrics."""
        # Calculate VaR as percentage of portfolio
        var_1d_pct = safe_divide(var_1d, portfolio_value, ZERO) if portfolio_value > ZERO else ZERO

        # Risk thresholds
        var_critical = to_decimal(POSITION_SIZING_LIMITS["var_critical_threshold"])
        var_high = to_decimal(POSITION_SIZING_LIMITS["var_high_threshold"])
        var_medium = to_decimal(POSITION_SIZING_LIMITS["var_medium_threshold"])

        drawdown_critical = to_decimal(POSITION_SIZING_LIMITS["drawdown_critical_threshold"])
        drawdown_high = to_decimal(POSITION_SIZING_LIMITS["drawdown_high_threshold"])
        drawdown_medium = to_decimal(POSITION_SIZING_LIMITS["drawdown_medium_threshold"])

        # Check for critical risk
        if var_1d_pct > var_critical or current_drawdown > drawdown_critical:
            return RiskLevel.CRITICAL

        # Check for high risk
        if (
            var_1d_pct > var_high
            or current_drawdown > drawdown_high
            or (sharpe_ratio is not None and sharpe_ratio < Decimal("-1.0"))
        ):
            return RiskLevel.HIGH

        # Check for medium risk
        if (
            var_1d_pct > var_medium
            or current_drawdown > drawdown_medium
            or (sharpe_ratio is not None and sharpe_ratio < Decimal("0.5"))
        ):
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    # Portfolio and Position Management

    @cache_result(ttl=10)  # Cache for 10 seconds
    async def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get current portfolio metrics."""
        if self._portfolio_metrics is None:
            # Initialize with empty metrics if not available
            self._portfolio_metrics = PortfolioMetrics()

        return self._portfolio_metrics

    @with_error_context(component="risk_management", operation="validate_signal")
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @with_retry(max_attempts=3, base_delay=1.0)
    @time_execution
    @cached(
        ttl=30,
        namespace="risk",
        data_type="risk_metrics",
        key_generator=lambda self, signal: (
            f"signal_validation:{signal.symbol}:{signal.direction.value}:{signal.strength}"
        ),
    )
    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate trading signal against risk constraints.

        Args:
            signal: Trading signal to validate

        Returns:
            True if signal passes risk validation
        """
        return await self.execute_with_monitoring(
            "validate_signal", self._validate_signal_impl, signal
        )

    async def _validate_signal_impl(self, signal: Signal) -> bool:
        """Internal implementation of signal validation."""
        try:
            # Check signal strength threshold
            min_signal_strength = to_decimal(POSITION_SIZING_LIMITS["min_signal_strength"])
            if to_decimal(signal.strength) < min_signal_strength:
                self.logger.warning(
                    "Signal strength too low",
                    symbol=signal.symbol,
                    strength=signal.strength,
                    min_required=format_decimal(min_signal_strength),
                )
                return False

            # Check current risk level
            if self._current_risk_level == RiskLevel.CRITICAL:
                self.logger.warning(
                    "Risk level critical - rejecting signal",
                    symbol=signal.symbol,
                    risk_level=self._current_risk_level.value,
                )
                return False

            # Check emergency stop
            if self._emergency_stop_triggered:
                self.logger.warning(
                    "Emergency stop active - rejecting signal",
                    symbol=signal.symbol,
                )
                return False

            # Get current positions for symbol
            positions = await self._get_positions_for_symbol(signal.symbol)
            if len(positions) >= self.risk_config.max_positions_per_symbol:
                self.logger.warning(
                    "Max positions per symbol reached",
                    symbol=signal.symbol,
                    current_positions=len(positions),
                    max_allowed=self.risk_config.max_positions_per_symbol,
                )
                return False

            self.logger.info(
                "Signal validated successfully",
                symbol=signal.symbol,
                direction=signal.direction.value,
                strength=signal.strength,
            )

            return True

        except Exception as e:
            self.logger.error(f"Signal validation failed: {e}")
            return False

    @with_error_context(component="risk_management", operation="validate_order")
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @with_retry(max_attempts=3, base_delay=1.0)
    @time_execution
    @cached(
        ttl=15,
        namespace="risk",
        data_type="risk_metrics",
        key_generator=lambda self, order: (
            f"order_validation:{order.symbol}:{order.quantity}:{getattr(order, 'price', 'market')}"
        ),
    )
    async def validate_order(self, order: OrderRequest) -> bool:
        """
        Validate order against risk constraints.

        Args:
            order: Order to validate

        Returns:
            True if order passes risk validation
        """
        return await self.execute_with_monitoring(
            "validate_order", self._validate_order_impl, order
        )

    async def _validate_order_impl(self, order: OrderRequest) -> bool:
        """Internal implementation of order validation."""
        try:
            # Check emergency stop
            if self._emergency_stop_triggered:
                self.logger.warning("Emergency stop active - rejecting order", symbol=order.symbol)
                return False

            # Check order size limits
            portfolio_metrics = await self.get_portfolio_metrics()
            max_order_size = portfolio_metrics.total_value * self.risk_config.max_position_size_pct

            order_value = order.quantity * (
                order.price if hasattr(order, "price") else Decimal("1")
            )

            if order_value > max_order_size:
                self.logger.warning(
                    "Order size exceeds limit",
                    symbol=order.symbol,
                    order_value=format_decimal(order_value),
                    max_size=format_decimal(max_order_size),
                )
                return False

            # Check portfolio exposure
            potential_exposure = portfolio_metrics.total_exposure + order_value
            max_exposure = portfolio_metrics.total_value * self.risk_config.max_portfolio_exposure

            if potential_exposure > max_exposure:
                self.logger.warning(
                    "Order would exceed portfolio exposure limit",
                    current_exposure=format_decimal(portfolio_metrics.total_exposure),
                    additional_exposure=format_decimal(order_value),
                    max_exposure=format_decimal(max_exposure),
                )
                return False

            self.logger.info(
                "Order validated successfully",
                symbol=order.symbol,
                side=order.side.value,
                quantity=format_decimal(order.quantity),
            )

            return True

        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            return False

    # Risk Monitoring and Alerts

    async def _risk_monitoring_loop(self) -> None:
        """Background task for continuous risk monitoring."""
        self.logger.info("Starting risk monitoring loop")

        while True:
            try:
                await self._perform_risk_check()
                await asyncio.sleep(self.risk_config.risk_check_interval)

            except asyncio.CancelledError:
                self.logger.info("Risk monitoring loop cancelled")
                break

            except Exception as e:
                self.logger.error(f"Risk monitoring loop error: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    async def _perform_risk_check(self) -> None:
        """Perform comprehensive risk check."""
        try:
            self._last_risk_check = datetime.now(timezone.utc)

            # Get current positions and market data
            positions = await self._get_all_positions()
            market_data = await self._get_current_market_data()

            if not positions:
                return

            # Calculate current risk metrics
            risk_metrics = await self.calculate_risk_metrics(positions, market_data)

            # Check for risk alerts
            await self._check_risk_alerts(risk_metrics)

            # Check for emergency stop conditions
            await self._check_emergency_stop_conditions(risk_metrics)

            self.logger.info(
                "Risk check completed",
                risk_level=risk_metrics.risk_level.value,
                portfolio_value=format_decimal(risk_metrics.portfolio_value),
                current_drawdown=format_decimal(risk_metrics.current_drawdown),
            )

        except Exception as e:
            self.logger.error(f"Risk check failed: {e}")

    async def _check_risk_alerts(self, risk_metrics: RiskMetrics) -> None:
        """Check for risk alert conditions."""
        alerts = []

        # VaR threshold alerts
        if risk_metrics.var_95 and risk_metrics.portfolio_value > ZERO:
            var_pct = risk_metrics.var_95 / risk_metrics.portfolio_value
            if var_pct > to_decimal(POSITION_SIZING_LIMITS["var_high_threshold"]):
                alerts.append(
                    RiskAlert(
                        alert_id=f"var_high_{datetime.now(timezone.utc).isoformat()}",
                        alert_type="HIGH_VAR",
                        severity=RiskLevel.HIGH,
                        message=f"VaR exceeds 5% threshold: {format_decimal(var_pct * 100)}%",
                        details={
                            "var_95": str(risk_metrics.var_95),
                            "var_percentage": str(var_pct),
                        },
                    )
                )

        # Drawdown alerts
        drawdown_threshold = to_decimal(POSITION_SIZING_LIMITS["drawdown_high_threshold"])
        if risk_metrics.current_drawdown > drawdown_threshold:
            alerts.append(
                RiskAlert(
                    alert_id=f"drawdown_high_{datetime.now(timezone.utc).isoformat()}",
                    alert_type="HIGH_DRAWDOWN",
                    severity=RiskLevel.HIGH,
                    message=(
                        f"Current drawdown exceeds 10%: "
                        f"{format_decimal(risk_metrics.current_drawdown * 100)}%"
                    ),
                    details={"current_drawdown": str(risk_metrics.current_drawdown)},
                )
            )

        # Correlation risk alerts
        # 50% correlation risk (keeping as reasonable threshold)
        correlation_threshold = to_decimal("0.50")
        if risk_metrics.correlation_risk > correlation_threshold:
            alerts.append(
                RiskAlert(
                    alert_id=f"correlation_high_{datetime.now(timezone.utc).isoformat()}",
                    alert_type="HIGH_CORRELATION",
                    severity=RiskLevel.MEDIUM,
                    message=(
                        f"High correlation risk detected: "
                        f"{format_decimal(risk_metrics.correlation_risk * 100)}%"
                    ),
                    details={"correlation_risk": str(risk_metrics.correlation_risk)},
                )
            )

        # Add alerts to the list
        self._risk_alerts.extend(alerts)

        # Send alerts to analytics service
        if self.analytics_service and alerts:
            try:
                for alert in alerts:
                    await self.analytics_service.record_risk_alert(alert)
                self.logger.info(f"Sent {len(alerts)} risk alerts to analytics service")
            except Exception as e:
                self.logger.warning(f"Failed to send risk alerts to analytics: {e}")

        # Send alerts to monitoring system
        if self.alert_service and alerts:
            try:
                from src.monitoring.services import AlertRequest

                for alert in alerts:
                    # Convert RiskLevel to AlertSeverity
                    severity_mapping = {
                        RiskLevel.LOW: AlertSeverity.LOW,
                        RiskLevel.MEDIUM: AlertSeverity.MEDIUM,
                        RiskLevel.HIGH: AlertSeverity.HIGH,
                        RiskLevel.CRITICAL: AlertSeverity.CRITICAL,
                    }

                    alert_request = AlertRequest(
                        rule_name=f"risk_management_{alert.alert_type.lower()}",
                        severity=severity_mapping.get(alert.severity, AlertSeverity.MEDIUM),
                        message=alert.message,
                        labels={
                            "source": "risk_management",
                            "alert_type": alert.alert_type,
                            "alert_id": alert.alert_id,
                            "component": "risk_service",
                        },
                        annotations={
                            "timestamp": alert.timestamp.isoformat(),
                            "details": str(alert.details),
                            "risk_level": alert.severity.value,
                        },
                    )

                    fingerprint = await self.alert_service.create_alert(alert_request)
                    self.logger.info(f"Risk alert published to monitoring system: {fingerprint}")

                    # Publish standardized risk event
                    event_type_map = {
                        "position_limit": "limit_exceeded",
                        "portfolio_limit": "limit_exceeded",
                        "drawdown_limit": "limit_exceeded",
                        "risk_level_high": "exposure_warning",
                        "risk_level_critical": "margin_call",
                    }

                    event_type = event_type_map.get(alert.alert_type.lower(), "limit_exceeded")
                    await self._emit_state_event(
                        f"risk_{event_type}",
                        {
                            "alert_id": alert.alert_id,
                            "severity": alert.severity.value,
                            "message": alert.message,
                            "details": alert.details,
                            "timestamp": alert.timestamp.isoformat(),
                        },
                    )

            except Exception as e:
                self.logger.warning(f"Failed to send risk alerts to monitoring system: {e}")

        # Log alerts
        for alert in alerts:
            self.logger.warning(
                "Risk alert generated",
                alert_id=alert.alert_id,
                alert_type=alert.alert_type,
                severity=alert.severity.value,
                message=alert.message,
            )

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        self._risk_alerts = [alert for alert in self._risk_alerts if alert.timestamp > cutoff_time]

    async def _check_emergency_stop_conditions(self, risk_metrics: RiskMetrics) -> None:
        """Check for emergency stop conditions."""
        emergency_conditions = []

        # Check extreme drawdown
        if risk_metrics.current_drawdown > self.risk_config.emergency_stop_threshold:
            emergency_conditions.append(
                f"Extreme drawdown: {format_decimal(risk_metrics.current_drawdown * 100)}%"
            )

        # Check extreme VaR
        if risk_metrics.var_95 and risk_metrics.portfolio_value > ZERO:
            var_pct = risk_metrics.var_95 / risk_metrics.portfolio_value
            var_critical_threshold = to_decimal(
                POSITION_SIZING_LIMITS["var_critical_threshold"]
            ) * to_decimal("1.5")
            if var_pct > var_critical_threshold:
                emergency_conditions.append(f"Extreme VaR: {format_decimal(var_pct * 100)}%")

        # Trigger emergency stop if conditions are met
        if emergency_conditions and not self._emergency_stop_triggered:
            await self.trigger_emergency_stop(
                reason=f"Emergency conditions detected: {'; '.join(emergency_conditions)}"
            )

    async def trigger_emergency_stop(self, reason: str) -> None:
        """
        Trigger emergency stop of all trading operations.

        Args:
            reason: Reason for emergency stop
        """
        lock = None
        try:
            lock = self._emergency_lock
            async with lock:
                try:
                    self.logger.critical(
                        "EMERGENCY STOP TRIGGERED",
                        reason=reason,
                        risk_level=self._current_risk_level.value,
                        portfolio_metrics=(
                            self._portfolio_metrics.model_dump()
                            if self._portfolio_metrics
                            else None
                        ),
                    )

                    # Set emergency stop flag
                    self._emergency_stop_triggered = True
                    self._current_risk_level = RiskLevel.CRITICAL

                    # Create critical alert
                    emergency_alert = RiskAlert(
                        alert_id=f"emergency_stop_{datetime.now(timezone.utc).isoformat()}",
                        alert_type="EMERGENCY_STOP",
                        severity=RiskLevel.CRITICAL,
                        message=f"Emergency stop triggered: {reason}",
                        details={
                            "reason": reason,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    self._risk_alerts.append(emergency_alert)

                    # Save emergency state
                    await self._save_emergency_state(reason)

                    # Notify other services via state service
                    await self.state_service.set_state(
                        state_type=StateType.RISK_STATE,
                        state_id="emergency_stop",
                        state_data={
                            "emergency_stop": True,
                            "reason": reason,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "risk_level": self._current_risk_level.value,
                        },
                        source_component="RiskService",
                        reason=f"Emergency stop triggered: {reason}",
                    )

                    self.logger.info("Emergency stop completed successfully")

                except StateError as e:
                    self.logger.error(f"State service error during emergency stop: {e}")
                    # Still set the flag locally even if state service fails
                    self._emergency_stop_triggered = True
                    raise RiskManagementError(
                        f"Emergency stop partially failed (state service error): {e}"
                    ) from e
                except Exception as e:
                    self.logger.error(f"Emergency stop execution failed: {e}")
                    raise RiskManagementError(f"Emergency stop failed: {e}") from e
                finally:
                    # Ensure internal resources are properly handled
                    pass
        finally:
            # Ensure lock reference is cleared
            lock = None

    async def reset_emergency_stop(self, reason: str) -> None:
        """
        Reset emergency stop (admin function).

        Args:
            reason: Reason for resetting emergency stop
        """
        lock = None
        try:
            lock = self._emergency_lock
            async with lock:
                try:
                    self.logger.warning(
                        "Emergency stop reset requested",
                        reason=reason,
                        current_state=self._emergency_stop_triggered,
                    )

                    # Reset emergency stop flag
                    self._emergency_stop_triggered = False
                    self._current_risk_level = RiskLevel.LOW

                    # Clear emergency state
                    await self.state_service.delete_state(
                        state_type=StateType.RISK_STATE,
                        state_id="emergency_stop",
                        source_component="RiskService",
                        reason=f"Emergency stop reset: {reason}",
                    )

                    self.logger.info("Emergency stop reset successfully", reason=reason)

                except StateError as e:
                    self.logger.error(f"State service error during emergency stop reset: {e}")
                    # Do NOT reset locally if state service fails to ensure consistency
                    # Re-raise to notify caller of failure
                    raise RiskManagementError(
                        f"Failed to reset emergency stop in state service: {e}"
                    ) from e
                except Exception as e:
                    self.logger.error(f"Emergency stop reset failed: {e}")
                    raise RiskManagementError(f"Emergency stop reset failed: {e}") from e
                finally:
                    # Ensure internal resources are properly handled
                    pass
        finally:
            # Ensure lock reference is cleared
            lock = None

    # Market Data Integration

    @time_execution
    async def update_price_history(self, symbol: str, price: Decimal) -> None:
        """
        Update price history for risk calculations.

        Args:
            symbol: Trading symbol
            price: Current price
        """
        lock = None
        try:
            lock = self._history_lock
            async with lock:
                try:
                    # Update price history
                    self._price_history[symbol].append(price)

                    # Calculate and store returns
                    if len(self._price_history[symbol]) > 1:
                        prev_price = self._price_history[symbol][-2]
                        if prev_price > ZERO:
                            # Keep calculation in Decimal for precision,
                            # convert to float only for numpy operations when needed
                            return_rate_decimal = safe_divide(price - prev_price, prev_price, ZERO)
                            return_rate = decimal_to_float(
                                return_rate_decimal, f"return_rate_{symbol}"
                            )
                            self._return_history[symbol].append(return_rate)

                    # Limit history size for memory management with bounds checking
                    max_history = max(
                        self.risk_config.volatility_window * 2,
                        POSITION_SIZING_LIMITS.get("min_price_history", 100),
                    )
                    # Add absolute maximum to prevent excessive memory usage
                    max_history = min(
                        max_history, POSITION_SIZING_LIMITS.get("max_price_history", 1000)
                    )

                    if len(self._price_history[symbol]) > max_history:
                        self._price_history[symbol] = self._price_history[symbol][-max_history:]

                    if len(self._return_history[symbol]) > max_history:
                        self._return_history[symbol] = self._return_history[symbol][-max_history:]

                    # Periodic cleanup of old symbols with no recent data
                    if len(self._price_history) > POSITION_SIZING_LIMITS.get(
                        "max_symbols_tracked", 100
                    ):
                        await self._cleanup_stale_symbols()

                    self.logger.info(
                        "Price history updated",
                        symbol=symbol,
                        price=format_decimal(price),
                        history_length=len(self._price_history[symbol]),
                        returns_length=len(self._return_history[symbol]),
                        total_symbols=len(self._price_history),
                    )

                except Exception as e:
                    self.logger.error(f"Price history update failed for {symbol}: {e}")
                    raise
                finally:
                    # Ensure internal resources are cleaned up
                    pass
        except Exception as e:
            self.logger.error(f"Failed to acquire history lock for {symbol}: {e}")
            # Don't re-raise to prevent blocking other operations
        finally:
            # Ensure lock reference is cleared
            lock = None

    async def _get_all_positions(self) -> list[Position]:
        """Get all current positions via StateService."""
        try:
            # Get positions from state service instead of direct database access
            positions_state = await self.state_service.get_state(
                StateType.PORTFOLIO_STATE, "positions"
            )
            if not positions_state:
                return []

            # Convert state data to Position objects if needed
            positions = []
            for pos_data in positions_state.get("open_positions", []):
                if isinstance(pos_data, Position):
                    positions.append(pos_data)

            return positions
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            # Propagate state service errors to trigger appropriate risk controls
            raise RiskManagementError(
                f"Unable to retrieve positions for risk assessment: {e!s}",
                error_code="STATE_002",
                details={"operation": "get_all_positions", "original_error": str(e)},
            ) from e

    async def _get_positions_for_symbol(self, symbol: str) -> list[Position]:
        """Get positions for specific symbol via StateService."""
        try:
            positions = await self._get_all_positions()
            symbol_positions = [pos for pos in positions if pos.symbol == symbol]
            return symbol_positions
        except Exception as e:
            self.logger.error(f"Failed to get positions for {symbol}: {e}")
            # Propagate state service errors to trigger appropriate risk controls
            raise RiskManagementError(
                f"Unable to retrieve positions for {symbol}: {e!s}",
                error_code="STATE_002",
                details={
                    "operation": "get_positions_for_symbol",
                    "symbol": symbol,
                    "original_error": str(e),
                },
            ) from e

    async def _get_current_market_data(self) -> list[MarketData]:
        """Get current market data (placeholder - would integrate with market data service)."""
        # This would typically get data from a MarketDataService
        # For now, return empty list
        return []

    # State Management Integration

    async def _load_risk_state(self) -> None:
        """Load risk state from StateService."""
        try:
            # Load emergency stop state
            emergency_state = await self.state_service.get_state(
                state_type=StateType.RISK_STATE, state_id="emergency_stop"
            )

            if emergency_state:
                # Validate state data structure
                if not isinstance(emergency_state, dict):
                    self.logger.error(f"Invalid emergency state format: {type(emergency_state)}")
                    emergency_state = {}

                if emergency_state.get("emergency_stop"):
                    self._emergency_stop_triggered = True
                    self._current_risk_level = RiskLevel.CRITICAL
                    self.logger.warning(
                        "Loaded emergency stop state",
                        reason=emergency_state.get("reason"),
                        timestamp=emergency_state.get("timestamp"),
                    )

            # Load risk configuration overrides
            risk_config_state = await self.state_service.get_state(
                state_type=StateType.RISK_STATE, state_id="risk_configuration"
            )

            if risk_config_state:
                # Apply any configuration overrides
                self.logger.info("Loaded risk configuration state")

        except StateError as e:
            self.logger.error(f"State service error while loading risk state: {e}")
            # Re-raise to let caller handle critical state errors
            raise
        except Exception as e:
            self.logger.error(f"Failed to load risk state: {e}")

    async def _save_risk_state(self) -> None:
        """Save current risk state to StateService."""
        try:
            # Serialize portfolio metrics with proper Decimal handling
            portfolio_metrics_data = None
            if self._portfolio_metrics:
                if hasattr(self._portfolio_metrics, "model_dump"):
                    # Use mode='json' to properly serialize Decimals
                    portfolio_metrics_data = self._portfolio_metrics.model_dump(mode="json")
                else:
                    # Fallback for objects without model_dump - convert to dict
                    portfolio_metrics_data = (
                        self._portfolio_metrics.__dict__
                        if hasattr(self._portfolio_metrics, "__dict__")
                        else {}
                    )

            risk_state = {
                "current_risk_level": self._current_risk_level.value,
                "emergency_stop_triggered": self._emergency_stop_triggered,
                "last_risk_check": self._last_risk_check.isoformat(),
                "risk_calculations_count": self._risk_calculations_count,
                "active_alerts_count": len(self._risk_alerts),
                "portfolio_metrics": portfolio_metrics_data,
            }

            await self.state_service.set_state(
                state_type=StateType.RISK_STATE,
                state_id="current_risk_state",
                state_data=risk_state,
                source_component="RiskService",
                reason="Periodic risk state save",
            )

        except StateError as e:
            self.logger.error(f"State service error while saving risk state: {e}")
            # Monitor state persistence failures
            if self.metrics_service:
                from src.monitoring.services import MetricRequest

                self.metrics_service.record_counter(
                    MetricRequest(
                        name="risk_state_persistence_failures_total",
                        value=1,
                        labels={
                            "component": "RiskService",
                            "operation": "save_risk_state",
                            "error_type": "StateError",
                        },
                    )
                )
            # Re-raise for critical state updates
            raise RiskManagementError(f"Failed to persist risk state: {e}") from e
        except Exception as e:
            self.logger.error(f"Failed to save risk state: {e}")
            if self.metrics_service:
                from src.monitoring.services import MetricRequest

                self.metrics_service.record_counter(
                    MetricRequest(
                        name="risk_state_persistence_failures_total",
                        value=1,
                        labels={
                            "component": "RiskService",
                            "operation": "save_risk_state",
                            "error_type": "GeneralError",
                        },
                    )
                )
            raise RiskManagementError(f"Failed to save risk state: {e}") from e

    async def _save_risk_metrics(self, risk_metrics: RiskMetrics) -> None:
        """Save risk metrics to StateService and potentially database."""
        try:
            # Properly serialize risk metrics with Decimal preservation
            if hasattr(risk_metrics, "model_dump"):
                # Use mode='json' to properly handle Decimal serialization
                metrics_data = risk_metrics.model_dump(mode="json")
            else:
                # Fallback for objects without model_dump - convert to dict
                metrics_data = risk_metrics.__dict__ if hasattr(risk_metrics, "__dict__") else {}

            # Save to state service for real-time access
            await self.state_service.set_state(
                state_type=StateType.RISK_STATE,
                state_id="latest_risk_metrics",
                state_data=metrics_data,
                source_component="RiskService",
                reason="Risk metrics update",
            )

            # Track successful saves
            if self.metrics_service:
                from src.monitoring.services import MetricRequest

                self.metrics_service.record_counter(
                    MetricRequest(
                        name="risk_metrics_saved_total",
                        value=1,
                        labels={"component": "RiskService", "operation": "save_risk_metrics"},
                    )
                )

        except StateError as e:
            self.logger.error(f"State service error while saving risk metrics: {e}")
            if self.metrics_service:
                from src.monitoring.services import MetricRequest

                self.metrics_service.record_counter(
                    MetricRequest(
                        name="risk_metrics_save_failures_total",
                        value=1,
                        labels={
                            "component": "RiskService",
                            "operation": "save_risk_metrics",
                            "error_type": "StateError",
                        },
                    )
                )
            # Re-raise as risk metrics are important for system operation
            raise RiskManagementError(f"Failed to persist risk metrics: {e}") from e
        except Exception as e:
            self.logger.error(f"Failed to save risk metrics: {e}")
            if self.metrics_service:
                from src.monitoring.services import MetricRequest

                self.metrics_service.record_counter(
                    MetricRequest(
                        name="risk_metrics_save_failures_total",
                        value=1,
                        labels={
                            "component": "RiskService",
                            "operation": "save_risk_metrics",
                            "error_type": "GeneralError",
                        },
                    )
                )
            raise RiskManagementError(f"Failed to save risk metrics: {e}") from e

    async def _save_emergency_state(self, reason: str) -> None:
        """Save emergency stop state."""
        try:
            emergency_state = {
                "emergency_stop": True,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_level": self._current_risk_level.value,
                "portfolio_metrics": (
                    self._portfolio_metrics.model_dump() if self._portfolio_metrics else None
                ),
            }

            await self.state_service.set_state(
                state_type=StateType.RISK_STATE,
                state_id="emergency_stop",
                state_data=emergency_state,
                source_component="RiskService",
                reason=f"Emergency state saved: {reason}",
            )

        except StateError as e:
            self.logger.error(f"State service error while saving emergency state: {e}")
            # Re-raise as emergency state is critical
            raise RiskManagementError(f"Critical: Failed to save emergency state - {e}") from e
        except Exception as e:
            self.logger.error(f"Failed to save emergency state: {e}")
            raise RiskManagementError(f"Critical: Failed to save emergency state - {e}") from e

    # Utility Methods

    async def _verify_dependencies(self) -> bool:
        """Verify that required dependencies are available."""
        try:
            # Test repository services
            if not self.risk_metrics_repository:
                self.logger.error("RiskMetricsRepository not available")
                return False

            if not self.portfolio_repository:
                self.logger.error("PortfolioRepository not available")
                return False

            # Test state service
            if not self.state_service:
                self.logger.error("StateService not available")
                return False

            # Test state service connectivity
            try:
                health_status = await self.state_service.get_health_status()
                if health_status.get("overall_status") == "healthy":
                    self.logger.info("StateService health check passed")
                else:
                    self.logger.warning(
                        f"StateService health check shows degraded status: {health_status.get('overall_status')}"
                    )
            except Exception as e:
                self.logger.warning(f"StateService health check failed: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Dependency verification failed: {e}")
            return False

    async def _service_health_check(self) -> Any:
        """Service-specific health check."""
        try:
            # Check if risk calculations are current
            time_since_check = (datetime.now(timezone.utc) - self._last_risk_check).total_seconds()
            if time_since_check > self.risk_config.risk_check_interval * 2:
                return HealthStatus.DEGRADED

            # Check emergency stop status
            if self._emergency_stop_triggered:
                return HealthStatus.DEGRADED

            # Check state service dependency health
            try:
                health_status = await self.state_service.get_health_status()
                if health_status.get("overall_status") != "healthy":
                    self.logger.warning(
                        f"StateService health check shows degraded status: {health_status.get('overall_status')}"
                    )
                    return HealthStatus.DEGRADED
            except Exception as e:
                self.logger.warning(f"StateService health check failed: {e}")
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self.logger.error(f"Risk service health check failed: {e}")
            return HealthStatus.UNHEALTHY

    async def _emit_state_event(self, event_type: str, event_data: dict) -> None:
        """Emit state event using consistent pattern with state module."""
        try:
            from src.state.consistency import emit_state_event

            await emit_state_event(event_type, event_data)
        except Exception as e:
            self.logger.warning(f"Failed to emit state event {event_type}: {e}")

    def _emit_validation_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Emit validation error event with consistent format matching execution module."""
        try:
            # Emit error event with consistent format
            if hasattr(self, "_emitter") and self._emitter:
                from src.core.event_constants import RiskEvents

                error_data = {
                    "event_type": RiskEvents.VALIDATION_ERROR,
                    "error": str(error),
                    "component": "RiskService",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **context,
                }

                self._emitter.emit(
                    event=RiskEvents.VALIDATION_ERROR,
                    data=error_data,
                    source="risk_management",
                )
        except Exception as emit_error:
            self.logger.warning(f"Failed to emit validation error event: {emit_error}")

    # Public API Methods

    async def get_risk_alerts(self, limit: int | None = None) -> list[RiskAlert]:
        """
        Get current risk alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of risk alerts
        """
        alerts = sorted(self._risk_alerts, key=lambda x: x.timestamp, reverse=True)
        if limit:
            alerts = alerts[:limit]
        return alerts

    async def acknowledge_risk_alert(self, alert_id: str) -> bool:
        """
        Acknowledge a risk alert.

        Args:
            alert_id: ID of alert to acknowledge

        Returns:
            True if alert was found and acknowledged
        """
        for alert in self._risk_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"Risk alert acknowledged: {alert_id}")
                return True

        self.logger.warning(f"Risk alert not found for acknowledgment: {alert_id}")
        return False

    def get_current_risk_level(self) -> RiskLevel:
        """Get current risk level."""
        return self._current_risk_level

    def is_emergency_stop_active(self) -> bool:
        """Check if emergency stop is currently active."""
        return self._emergency_stop_triggered

    async def get_risk_summary(self) -> dict[str, Any]:
        """
        Get comprehensive risk summary.

        Returns:
            Dictionary with risk summary information
        """
        portfolio_metrics = await self.get_portfolio_metrics()

        return {
            "current_risk_level": self._current_risk_level.value,
            "emergency_stop_active": self._emergency_stop_triggered,
            "portfolio_metrics": portfolio_metrics.model_dump(),
            "active_alerts": len([alert for alert in self._risk_alerts if not alert.acknowledged]),
            "total_alerts": len(self._risk_alerts),
            "last_risk_check": self._last_risk_check.isoformat(),
            "risk_calculations_count": self._risk_calculations_count,
            "cache_hit_rate": self._cache_hit_rate,
            "service_metrics": self.get_metrics(),
        }

    async def _cleanup_resources(self) -> None:
        """Clean up all resources used by the risk service."""
        history_lock = None
        state_lock = None
        try:
            # Clean up memory structures with size limits
            max_cleanup_items = POSITION_SIZING_LIMITS.get("max_cleanup_items", 1000)

            try:
                history_lock = self._history_lock
                async with history_lock:
                    # Clear history data with logging for monitoring
                    price_symbols = len(self._price_history)
                    return_symbols = len(self._return_history)
                    portfolio_history_len = len(self._portfolio_value_history)

                    # Clear collections
                    self._price_history.clear()
                    self._return_history.clear()
                    self._portfolio_value_history.clear()

                    self.logger.info(
                        "Memory structures cleaned up",
                        price_symbols=price_symbols,
                        return_symbols=return_symbols,
                        portfolio_history_len=portfolio_history_len,
                    )
            finally:
                history_lock = None

            # Clean up alerts with limit to prevent excessive memory usage
            try:
                state_lock = self._state_lock
                async with state_lock:
                    alerts_count = len(self._risk_alerts)
                    alerts_to_keep = POSITION_SIZING_LIMITS.get("max_alerts_to_keep", 100)
                    if alerts_count > max_cleanup_items:
                        # Keep only the most recent alerts
                        self._risk_alerts = self._risk_alerts[-alerts_to_keep:]
                        self.logger.info(f"Cleaned up {alerts_count - alerts_to_keep} old alerts")
                    else:
                        self._risk_alerts.clear()
            finally:
                state_lock = None

            # Cancel any remaining cleanup tasks with timeout
            cleanup_tasks = list(self._cleanup_tasks)
            if cleanup_tasks:
                for task in cleanup_tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=5.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                        except Exception as e:
                            self.logger.warning(f"Error cancelling cleanup task: {e}")
                        finally:
                            # Ensure task reference is cleared
                            task = None

                self._cleanup_tasks.clear()
                self.logger.info(f"Cancelled {len(cleanup_tasks)} cleanup tasks")

            # Close cache service connections if available
            if self.cache_service:
                try:
                    await self.cache_service.close()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up cache service: {e}")

        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")
            # Don't re-raise to prevent shutdown failures
        finally:
            # Ensure all lock references are cleared
            history_lock = None
            state_lock = None

    async def _cleanup_stale_symbols(self) -> None:
        """Clean up symbols with no recent data to prevent memory leaks."""
        try:
            stale_symbols = []
            for symbol in list(self._price_history.keys()):
                # If no data or very little data, consider it stale
                min_data_threshold = POSITION_SIZING_LIMITS.get("min_data_for_symbol", 10)
                if (
                    len(self._price_history[symbol]) == 0
                    or len(self._price_history[symbol]) < min_data_threshold
                ):
                    stale_symbols.append(symbol)

            # Remove stale symbols
            for symbol in stale_symbols:
                if symbol in self._price_history:
                    del self._price_history[symbol]
                if symbol in self._return_history:
                    del self._return_history[symbol]

            if stale_symbols:
                self.logger.info(f"Cleaned up {len(stale_symbols)} stale symbols")

        except Exception as e:
            self.logger.error(f"Error cleaning up stale symbols: {e}")

    def reset_metrics(self) -> None:
        """Reset all risk metrics and counters."""
        super().reset_metrics()

        self._risk_calculations_count = 0
        self._cache_hit_rate = 0.0
        self._risk_alerts.clear()
        self._portfolio_value_history.clear()
        self._price_history.clear()
        self._return_history.clear()

        self.logger.info("Risk service metrics reset")

    @with_error_context(component="risk_management", operation="should_exit_position")
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @with_retry(max_attempts=3, base_delay=1.0)
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
        return await self.execute_with_monitoring(
            "should_exit_position", self._should_exit_position_impl, position, market_data
        )

    async def _should_exit_position_impl(self, position: Position, market_data: MarketData) -> bool:
        """Internal implementation of position exit evaluation."""
        try:
            # Check emergency stop first
            if self._emergency_stop_triggered:
                self.logger.warning(
                    "Emergency stop active - position should be closed",
                    symbol=position.symbol,
                    position_id=position.id,
                )
                return True

            # Check stop-loss conditions
            if hasattr(position, "stop_loss") and position.stop_loss:
                if position.side.value == "BUY" and market_data.close <= position.stop_loss:
                    self.logger.info(
                        "Stop-loss triggered for long position",
                        symbol=position.symbol,
                        current_price=format_decimal(market_data.close),
                        stop_loss=format_decimal(position.stop_loss),
                    )
                    return True
                elif position.side.value == "SELL" and market_data.close >= position.stop_loss:
                    self.logger.info(
                        "Stop-loss triggered for short position",
                        symbol=position.symbol,
                        current_price=format_decimal(market_data.close),
                        stop_loss=format_decimal(position.stop_loss),
                    )
                    return True

            # Check if position loss exceeds risk limits
            if position.unrealized_pnl:
                position_loss_pct = safe_divide(
                    abs(position.unrealized_pnl), position.quantity * position.entry_price
                )

                # Use configured max position loss or default to 10%
                max_position_loss = to_decimal(POSITION_SIZING_LIMITS["drawdown_high_threshold"])
                if (
                    self.app_config
                    and hasattr(self.app_config, "risk")
                    and hasattr(self.app_config.risk, "max_position_loss_pct")
                ):
                    max_position_loss = to_decimal(self.app_config.risk.max_position_loss_pct)

                if position.unrealized_pnl < ZERO and position_loss_pct > max_position_loss:
                    self.logger.info(
                        "Position loss limit exceeded",
                        symbol=position.symbol,
                        loss_pct=format_decimal(position_loss_pct),
                        max_loss=format_decimal(max_position_loss),
                    )
                    return True

            # Check trailing stop conditions
            trailing_stop_price = getattr(
                position, "trailing_stop_price", None
            ) or position.metadata.get("trailing_stop_price")
            if self.risk_config.trailing_stop_enabled and trailing_stop_price is not None:
                trailing_stop_decimal = to_decimal(trailing_stop_price)
                if position.side.value == "BUY" and market_data.close <= trailing_stop_decimal:
                    self.logger.info(
                        "Trailing stop triggered for long position",
                        symbol=position.symbol,
                        current_price=format_decimal(market_data.close),
                        trailing_stop=format_decimal(trailing_stop_decimal),
                    )
                    return True
                elif position.side.value == "SELL" and market_data.close >= trailing_stop_decimal:
                    self.logger.info(
                        "Trailing stop triggered for short position",
                        symbol=position.symbol,
                        current_price=format_decimal(market_data.close),
                        trailing_stop=format_decimal(trailing_stop_decimal),
                    )
                    return True

            # Check portfolio-level risk criteria
            if self._current_risk_level == RiskLevel.CRITICAL:
                self.logger.warning(
                    "Critical risk level - position should be closed",
                    symbol=position.symbol,
                    risk_level=self._current_risk_level.value,
                )
                return True

            # Check if drawdown exceeds emergency threshold
            if self._portfolio_metrics and self._portfolio_metrics.total_value > ZERO:
                current_drawdown = await self._calculate_current_drawdown(
                    self._portfolio_metrics.total_value
                )
                if current_drawdown > self.risk_config.emergency_stop_threshold:
                    self.logger.warning(
                        "Emergency drawdown threshold exceeded - position should be closed",
                        symbol=position.symbol,
                        current_drawdown=format_decimal(current_drawdown),
                        threshold=format_decimal(self.risk_config.emergency_stop_threshold),
                    )
                    return True

            return False

        except Exception as e:
            self.logger.error(
                "Position exit evaluation failed",
                symbol=position.symbol,
                error=str(e),
            )
            # Conservative approach: don't exit on evaluation errors
            return False

    @asynccontextmanager
    async def risk_monitoring_context(self, operation: str) -> Any:
        """
        Async context manager for risk monitoring operations.

        Provides automatic error handling, metrics collection, and cleanup.

        Args:
            operation: Name of the operation for monitoring

        Yields:
            Self for chained operations
        """
        start_time = datetime.now(timezone.utc)
        operation_id = f"{operation}_{id(self)}_{start_time.timestamp()}"

        try:
            self.logger.info(
                "Starting risk operation", operation=operation, operation_id=operation_id
            )

            # Initialize monitoring context
            if self._metrics_collector and hasattr(self._metrics_collector, "start_operation"):
                await self._metrics_collector.start_operation(operation, operation_id)

            yield self

            # Success metrics
            if self._metrics_collector:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                if hasattr(self._metrics_collector, "record_operation_success"):
                    await self._metrics_collector.record_operation_success(operation, duration)
                else:
                    # Use standard metrics if operation-specific methods not available
                    self._metrics_collector.increment_counter(
                        "risk_operations_total",
                        labels={"operation": operation, "status": "success"},
                    )
                    self._metrics_collector.observe_histogram(
                        "risk_operation_duration_seconds", duration, labels={"operation": operation}
                    )

            self.logger.info(
                "Risk operation completed successfully",
                operation=operation,
                operation_id=operation_id,
                duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            )

        except Exception as e:
            # Error metrics and logging
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            if self._metrics_collector:
                if hasattr(self._metrics_collector, "record_operation_error"):
                    await self._metrics_collector.record_operation_error(
                        operation, str(e), error_type=type(e).__name__
                    )
                else:
                    # Use standard metrics if operation-specific methods not available
                    self._metrics_collector.increment_counter(
                        "risk_operations_total",
                        labels={
                            "operation": operation,
                            "status": "error",
                            "error_type": type(e).__name__,
                        },
                    )

            self.logger.error(
                "Risk operation failed",
                operation=operation,
                operation_id=operation_id,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=duration * 1000,
            )

            # Re-raise the exception after logging
            raise

        finally:
            # Cleanup operations
            try:
                if self._metrics_collector and hasattr(self._metrics_collector, "end_operation"):
                    await self._metrics_collector.end_operation(operation_id)
            except Exception as cleanup_error:
                self.logger.warning(
                    "Error during risk operation cleanup",
                    operation=operation,
                    cleanup_error=str(cleanup_error),
                )

    async def _cleanup_connection_resources(self) -> None:
        """Clean up connection resources including WebSocket connections."""
        try:
            # Clean up any active WebSocket connections
            # This is a placeholder - in real implementation would close actual connections

            # Clean up circuit breaker manager connections if initialized
            if hasattr(self, "_circuit_breaker_manager") and self._circuit_breaker_manager:
                try:
                    await self._circuit_breaker_manager.cleanup_resources()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up circuit breaker manager: {e}")

            # Clean up any monitoring service connections
            if hasattr(self, "_monitoring_service_connections"):
                connection_count = 0
                for connection in getattr(self, "_monitoring_service_connections", []):
                    try:
                        if hasattr(connection, "close"):
                            await connection.close()
                        connection_count += 1
                    except Exception as e:
                        self.logger.warning(f"Error closing monitoring connection: {e}")

                if connection_count > 0:
                    self.logger.info(f"Cleaned up {connection_count} monitoring connections")

            self.logger.info("Connection resources cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Error cleaning up connection resources: {e}")
            # Don't re-raise to prevent shutdown failures
