"""
Environment-aware Strategy Integration.

This module extends the Strategy service with environment awareness,
providing different strategy configurations and behaviors for sandbox vs live trading.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from src.core.exceptions import StrategyError
from src.core.integration.environment_aware_service import (
    EnvironmentAwareServiceMixin,
    EnvironmentContext,
)
from src.core.logging import get_logger
from src.core.types import (
    MarketData,
    Signal,
    StrategyConfig,
)

logger = get_logger(__name__)


class StrategyMode(Enum):
    """Strategy operation modes for different environments."""

    EXPERIMENTAL = "experimental"  # High-risk strategies for testing
    CONSERVATIVE = "conservative"  # Low-risk strategies for production
    BALANCED = "balanced"  # Moderate risk strategies
    PAPER_ONLY = "paper_only"  # Simulation mode only
    VALIDATION = "validation"  # Strategy validation mode


class EnvironmentAwareStrategyConfiguration:
    """Environment-specific strategy configuration."""

    @staticmethod
    def get_sandbox_strategy_config() -> dict[str, Any]:
        """Get strategy configuration for sandbox environment."""
        return {
            "strategy_mode": StrategyMode.EXPERIMENTAL,
            "max_position_size_pct": Decimal("0.15"),  # 15% max position
            "enable_experimental_strategies": True,
            "enable_high_frequency_strategies": True,
            "enable_arbitrage_strategies": True,
            "enable_ml_strategies": True,
            "min_signal_confidence": Decimal("0.4"),  # Lower confidence threshold
            "max_drawdown_pct": Decimal("0.20"),  # 20% max drawdown
            "rebalancing_frequency_minutes": 15,  # Frequent rebalancing
            "enable_dynamic_parameters": True,
            "enable_genetic_optimization": True,
            "backtest_required": False,  # Not required for sandbox
            "paper_trading_mode": False,  # Real execution in sandbox
            "strategy_timeout_minutes": 30,
            "max_concurrent_strategies": 10,
            "enable_cross_strategy_coordination": True,
            "risk_override_allowed": True,  # Allow risk parameter overrides
            "enable_strategy_hot_reload": True,  # Hot reload for testing
        }

    @staticmethod
    def get_live_strategy_config() -> dict[str, Any]:
        """Get strategy configuration for live/production environment."""
        return {
            "strategy_mode": StrategyMode.CONSERVATIVE,
            "max_position_size_pct": Decimal("0.05"),  # 5% max position
            "enable_experimental_strategies": False,
            "enable_high_frequency_strategies": False,  # Disabled for stability
            "enable_arbitrage_strategies": True,  # Safe arbitrage allowed
            "enable_ml_strategies": False,  # Disabled until thoroughly tested
            "min_signal_confidence": Decimal("0.7"),  # Higher confidence threshold
            "max_drawdown_pct": Decimal("0.08"),  # 8% max drawdown
            "rebalancing_frequency_minutes": 60,  # Less frequent rebalancing
            "enable_dynamic_parameters": False,  # Static parameters for stability
            "enable_genetic_optimization": False,  # Disabled for production
            "backtest_required": True,  # Mandatory backtesting
            "paper_trading_mode": False,  # Real execution
            "strategy_timeout_minutes": 60,
            "max_concurrent_strategies": 5,
            "enable_cross_strategy_coordination": False,  # Simpler coordination
            "risk_override_allowed": False,  # No risk overrides in production
            "enable_strategy_hot_reload": False,  # Disabled for stability
        }


class EnvironmentAwareStrategyManager(EnvironmentAwareServiceMixin):
    """
    Environment-aware strategy management functionality.

    This mixin adds environment-specific strategy behaviors and safeguards
    to the Strategy service.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._environment_strategy_configs: dict[str, dict[str, Any]] = {}
        self._strategy_performance_tracking: dict[str, dict[str, Any]] = {}
        self._active_strategies_by_env: dict[str, list[str]] = {}
        self._strategy_backtests: dict[str, dict[str, Any]] = {}

    async def _update_service_environment(self, context: EnvironmentContext) -> None:
        """Update strategy settings based on environment context."""
        await super()._update_service_environment(context)

        # Get environment-specific strategy configuration
        if context.is_production:
            strategy_config = EnvironmentAwareStrategyConfiguration.get_live_strategy_config()
            logger.info(f"Applied live strategy configuration for {context.exchange_name}")
        else:
            strategy_config = EnvironmentAwareStrategyConfiguration.get_sandbox_strategy_config()
            logger.info(f"Applied sandbox strategy configuration for {context.exchange_name}")

        self._environment_strategy_configs[context.exchange_name] = strategy_config

        # Initialize performance tracking for this environment
        self._strategy_performance_tracking[context.exchange_name] = {
            "total_signals": 0,
            "profitable_signals": 0,
            "total_pnl": Decimal("0"),
            "max_drawdown": Decimal("0"),
            "current_drawdown": Decimal("0"),
            "win_rate": Decimal("0"),
            "sharpe_ratio": Decimal("0"),
            "active_strategies": [],
            "last_rebalance": None,
            "strategy_violations": 0,
        }

        # Initialize active strategies list
        self._active_strategies_by_env[context.exchange_name] = []

        # Disable incompatible strategies in production
        if context.is_production:
            await self._disable_experimental_strategies(context.exchange_name)

    def get_environment_strategy_config(self, exchange: str) -> dict[str, Any]:
        """Get strategy configuration for a specific exchange environment."""
        if exchange not in self._environment_strategy_configs:
            # Initialize with default config based on current environment
            context = self.get_environment_context(exchange)
            if context.is_production:
                config = EnvironmentAwareStrategyConfiguration.get_live_strategy_config()
            else:
                config = EnvironmentAwareStrategyConfiguration.get_sandbox_strategy_config()
            self._environment_strategy_configs[exchange] = config

        return self._environment_strategy_configs[exchange]

    async def deploy_environment_aware_strategy(
        self, strategy_config: StrategyConfig, exchange: str, force_deploy: bool = False
    ) -> bool:
        """Deploy strategy with environment-specific validation and configuration."""
        context = self.get_environment_context(exchange)
        env_config = self.get_environment_strategy_config(exchange)

        # Environment-specific validation
        if not await self.validate_strategy_for_environment(strategy_config, exchange):
            raise StrategyError(
                f"Strategy validation failed for {exchange}",
                context={
                    "strategy": strategy_config.name,
                    "environment": context.environment.value,
                },
            )

        # Check if backtesting is required (production environments)
        if env_config.get("backtest_required") and context.is_production:
            if not await self._verify_strategy_backtest(strategy_config, exchange):
                if not force_deploy:
                    raise StrategyError(
                        "Strategy deployment requires valid backtest results in production",
                        context={"strategy": strategy_config.name},
                    )
                else:
                    logger.warning(
                        f"Force deploying strategy without backtest: {strategy_config.name}"
                    )

        # Check concurrent strategy limits
        max_concurrent = env_config.get("max_concurrent_strategies", 5)
        active_strategies = self._active_strategies_by_env.get(exchange, [])

        if len(active_strategies) >= max_concurrent:
            raise StrategyError(
                f"Maximum concurrent strategies ({max_concurrent}) reached for {exchange}",
                context={"active_count": len(active_strategies)},
            )

        # Apply environment-specific configuration overrides
        adjusted_config = await self._apply_environment_strategy_adjustments(
            strategy_config, exchange
        )

        try:
            # Deploy the strategy
            success = await self._deploy_strategy_with_config(adjusted_config, exchange)

            if success:
                # Track deployment
                self._active_strategies_by_env[exchange].append(strategy_config.name)

                # Initialize performance tracking for this strategy
                await self._initialize_strategy_tracking(strategy_config.name, exchange)

                logger.info(
                    f"Successfully deployed strategy {strategy_config.name} on {exchange}",
                    extra={"environment": context.environment.value},
                )

            return success

        except Exception as e:
            logger.error(f"Strategy deployment failed: {e}")
            raise StrategyError(f"Failed to deploy strategy: {e}")

    async def validate_strategy_for_environment(
        self, strategy_config: StrategyConfig, exchange: str
    ) -> bool:
        """Validate strategy configuration against environment constraints."""
        context = self.get_environment_context(exchange)
        env_config = self.get_environment_strategy_config(exchange)

        # Check if strategy type is allowed in this environment
        if context.is_production:
            # Production validations
            if not await self._validate_production_strategy(strategy_config, env_config):
                return False
        else:
            # Sandbox validations (more lenient)
            if not await self._validate_sandbox_strategy(strategy_config, env_config):
                return False

        # Common validations
        return await self._validate_common_strategy_rules(strategy_config, exchange, env_config)

    async def _validate_production_strategy(
        self, strategy_config: StrategyConfig, env_config: dict[str, Any]
    ) -> bool:
        """Production-specific strategy validations."""

        # Check if experimental strategies are disabled
        if not env_config.get("enable_experimental_strategies"):
            if hasattr(strategy_config, "is_experimental") and strategy_config.is_experimental:
                logger.error(
                    f"Experimental strategy {strategy_config.name} not allowed in production"
                )
                return False

        # Check if ML strategies are allowed
        if not env_config.get("enable_ml_strategies"):
            if hasattr(strategy_config, "uses_ml") and strategy_config.uses_ml:
                logger.error(f"ML strategy {strategy_config.name} not allowed in production")
                return False

        # Check if high-frequency strategies are allowed
        if not env_config.get("enable_high_frequency_strategies"):
            if hasattr(strategy_config, "is_high_frequency") and strategy_config.is_high_frequency:
                logger.error(
                    f"High-frequency strategy {strategy_config.name} not allowed in production"
                )
                return False

        # Validate risk parameters are within production limits
        max_position_size = env_config.get("max_position_size_pct", Decimal("0.05"))
        if strategy_config.max_position_size > max_position_size:
            logger.error(
                f"Strategy position size {strategy_config.max_position_size} exceeds "
                f"production limit {max_position_size}"
            )
            return False

        return True

    async def _validate_sandbox_strategy(
        self, strategy_config: StrategyConfig, env_config: dict[str, Any]
    ) -> bool:
        """Sandbox-specific strategy validations (more lenient)."""

        # Basic sanity checks for sandbox
        if not strategy_config.name:
            logger.error("Strategy must have a name")
            return False

        # Check position size limits (more generous for sandbox)
        max_position_size = env_config.get("max_position_size_pct", Decimal("0.15"))
        if strategy_config.max_position_size > max_position_size:
            logger.warning(
                f"Strategy position size {strategy_config.max_position_size} exceeds "
                f"sandbox recommendation {max_position_size} but allowing for testing"
            )
            # Allow but log warning

        return True

    async def _validate_common_strategy_rules(
        self, strategy_config: StrategyConfig, exchange: str, env_config: dict[str, Any]
    ) -> bool:
        """Common strategy validations for all environments."""

        # Check signal confidence requirements
        min_confidence = env_config.get("min_signal_confidence", Decimal("0.5"))
        if hasattr(strategy_config, "min_signal_confidence"):
            if strategy_config.min_signal_confidence < min_confidence:
                logger.error(
                    f"Strategy confidence {strategy_config.min_signal_confidence} "
                    f"below required minimum {min_confidence}"
                )
                return False

        # Check drawdown limits
        max_drawdown = env_config.get("max_drawdown_pct", Decimal("0.15"))
        if hasattr(strategy_config, "max_drawdown"):
            if strategy_config.max_drawdown > max_drawdown:
                logger.error(
                    f"Strategy max drawdown {strategy_config.max_drawdown} "
                    f"exceeds limit {max_drawdown}"
                )
                return False

        return True

    async def _apply_environment_strategy_adjustments(
        self, strategy_config: StrategyConfig, exchange: str
    ) -> StrategyConfig:
        """Apply environment-specific adjustments to strategy configuration."""
        context = self.get_environment_context(exchange)
        env_config = self.get_environment_strategy_config(exchange)

        # Create adjusted config
        adjusted_config = (
            strategy_config.model_copy()
            if hasattr(strategy_config, "model_copy")
            else strategy_config
        )

        # Apply environment-specific overrides
        if context.is_production:
            # Conservative adjustments for production
            if hasattr(adjusted_config, "max_position_size"):
                prod_max = env_config.get("max_position_size_pct", Decimal("0.05"))
                adjusted_config.max_position_size = min(adjusted_config.max_position_size, prod_max)

            if hasattr(adjusted_config, "rebalancing_frequency"):
                prod_freq = env_config.get("rebalancing_frequency_minutes", 60)
                adjusted_config.rebalancing_frequency = max(
                    adjusted_config.rebalancing_frequency, prod_freq
                )

        else:
            # More aggressive settings for sandbox testing
            if hasattr(adjusted_config, "enable_experimental_features"):
                adjusted_config.enable_experimental_features = env_config.get(
                    "enable_experimental_strategies", True
                )

        return adjusted_config

    async def generate_environment_aware_signal(
        self, strategy_name: str, market_data: MarketData, exchange: str
    ) -> "Signal | None":
        """Generate trading signal with environment-specific filtering and validation."""
        context = self.get_environment_context(exchange)
        env_config = self.get_environment_strategy_config(exchange)

        # Generate base signal (this would call the actual strategy)
        base_signal = await self._generate_base_signal(strategy_name, market_data, exchange)

        if not base_signal:
            return None

        # Apply environment-specific signal filtering
        filtered_signal = await self._apply_environment_signal_filters(
            base_signal, exchange, env_config
        )

        if not filtered_signal:
            logger.debug(f"Signal filtered out for {exchange} environment")
            return None

        # Validate signal against environment rules
        if not await self._validate_signal_for_environment(filtered_signal, exchange):
            logger.debug(f"Signal validation failed for {exchange}")
            return None

        # Update performance tracking
        await self._update_signal_tracking(strategy_name, exchange, filtered_signal)

        return filtered_signal

    async def _apply_environment_signal_filters(
        self, signal: Signal, exchange: str, env_config: dict[str, Any]
    ) -> "Signal | None":
        """Apply environment-specific filters to trading signals."""
        context = self.get_environment_context(exchange)

        # Check minimum confidence threshold (stored in metadata)
        min_confidence = env_config.get("min_signal_confidence", Decimal("0.5"))
        signal_confidence = signal.metadata.get("confidence", signal.strength)
        if signal_confidence < min_confidence:
            logger.debug(f"Signal confidence {signal_confidence} below threshold {min_confidence}")
            return None

        # Apply position sizing limits (stored in metadata)
        max_position_pct = env_config.get("max_position_size_pct", Decimal("0.05"))
        position_size_pct = signal.metadata.get("position_size_pct", Decimal("0.02"))
        if position_size_pct > max_position_pct:
            # Adjust position size instead of rejecting signal
            signal.metadata["position_size_pct"] = max_position_pct
            logger.info(f"Adjusted signal position size to {max_position_pct} for {exchange}")

        # Production-specific filters
        if context.is_production:
            # More conservative signal filtering for production
            risk_score = signal.metadata.get("risk_score")
            if risk_score and risk_score > Decimal("0.7"):
                logger.debug("High-risk signal filtered out in production")
                return None

        return signal

    async def _validate_signal_for_environment(self, signal: Signal, exchange: str) -> bool:
        """Validate signal against environment-specific rules."""
        context = self.get_environment_context(exchange)
        tracking = self._strategy_performance_tracking.get(exchange, {})

        # Check current drawdown limits
        current_drawdown = tracking.get("current_drawdown", Decimal("0"))
        max_drawdown = self.get_environment_strategy_config(exchange).get(
            "max_drawdown_pct", Decimal("0.15")
        )

        if current_drawdown > max_drawdown:
            logger.warning(f"Strategy drawdown {current_drawdown} exceeds limit {max_drawdown}")
            return False

        # Production-specific validations
        if context.is_production:
            # Check if we're in a high-volatility period
            if await self._is_high_volatility_period(signal.symbol, exchange):
                logger.info("Rejecting signal during high volatility period in production")
                return False

        return True

    async def _deploy_strategy_with_config(
        self, strategy_config: StrategyConfig, exchange: str
    ) -> bool:
        """Deploy strategy with the given configuration."""
        # This would interface with the actual strategy deployment system
        logger.info(f"Deploying strategy {strategy_config.name} on {exchange}")
        return True  # Mock success

    async def _generate_base_signal(
        self, strategy_name: str, market_data: MarketData, exchange: str
    ) -> "Signal | None":
        """Generate base signal from strategy (mock implementation)."""
        # This would call the actual strategy implementation
        from src.core.types import SignalDirection
        
        return Signal(
            symbol=market_data.symbol,
            direction=SignalDirection.BUY,
            strength=Decimal("0.75"),
            timestamp=datetime.now(timezone.utc),
            source=strategy_name,
            metadata={
                "strategy_name": strategy_name,
                "action": "BUY", 
                "confidence": Decimal("0.75"),
                "position_size_pct": Decimal("0.05"),
                "price": market_data.price,
            }
        )

    async def _verify_strategy_backtest(
        self, strategy_config: StrategyConfig, exchange: str
    ) -> bool:
        """Verify that strategy has valid backtest results."""
        backtest_key = f"{strategy_config.name}_{exchange}"

        if backtest_key not in self._strategy_backtests:
            logger.warning(f"No backtest results found for {strategy_config.name}")
            return False

        backtest = self._strategy_backtests[backtest_key]

        # Check backtest quality criteria
        min_sharpe = Decimal("1.0")  # Minimum Sharpe ratio
        max_drawdown = Decimal("0.15")  # Maximum drawdown

        if backtest.get("sharpe_ratio", Decimal("0")) < min_sharpe:
            logger.error(
                f"Strategy Sharpe ratio {backtest.get('sharpe_ratio')} below minimum {min_sharpe}"
            )
            return False

        if backtest.get("max_drawdown", Decimal("1")) > max_drawdown:
            logger.error(
                f"Strategy max drawdown {backtest.get('max_drawdown')} exceeds limit {max_drawdown}"
            )
            return False

        return True

    async def _initialize_strategy_tracking(self, strategy_name: str, exchange: str) -> None:
        """Initialize performance tracking for a strategy."""
        if exchange not in self._strategy_performance_tracking:
            return

        tracking = self._strategy_performance_tracking[exchange]
        if strategy_name not in tracking.get("active_strategies", []):
            tracking["active_strategies"].append(strategy_name)

    async def _update_signal_tracking(
        self, strategy_name: str, exchange: str, signal: Signal
    ) -> None:
        """Update signal generation tracking."""
        if exchange not in self._strategy_performance_tracking:
            return

        tracking = self._strategy_performance_tracking[exchange]
        tracking["total_signals"] = tracking.get("total_signals", 0) + 1
        tracking["last_signal"] = datetime.now().isoformat()

    async def _disable_experimental_strategies(self, exchange: str) -> None:
        """Disable experimental strategies in production environment."""
        active_strategies = self._active_strategies_by_env.get(exchange, [])

        for strategy_name in active_strategies[
            :
        ]:  # Copy list to avoid modification during iteration
            # This would check if strategy is experimental and disable it
            # For now, just log
            logger.info(f"Checking strategy {strategy_name} for production compatibility")

    async def _is_high_volatility_period(self, symbol: str, exchange: str) -> bool:
        """Check if current period has high volatility."""
        # This would implement actual volatility analysis
        return False  # Mock: not high volatility

    async def update_strategy_performance(
        self, strategy_name: str, exchange: str, pnl_change: Decimal, is_profitable: bool
    ) -> None:
        """Update strategy performance metrics."""
        if exchange not in self._strategy_performance_tracking:
            return

        tracking = self._strategy_performance_tracking[exchange]

        # Update PnL and performance metrics
        tracking["total_pnl"] = tracking.get("total_pnl", Decimal("0")) + pnl_change

        if is_profitable:
            tracking["profitable_signals"] = tracking.get("profitable_signals", 0) + 1

        # Update drawdown tracking
        if pnl_change < 0:
            current_dd = tracking.get("current_drawdown", Decimal("0"))
            tracking["current_drawdown"] = current_dd + abs(pnl_change)
            tracking["max_drawdown"] = max(
                tracking.get("max_drawdown", Decimal("0")), tracking["current_drawdown"]
            )
        else:
            # Reduce current drawdown on profitable trades
            current_dd = tracking.get("current_drawdown", Decimal("0"))
            tracking["current_drawdown"] = max(Decimal("0"), current_dd - pnl_change)

        # Calculate win rate
        total_signals = tracking.get("total_signals", 0)
        if total_signals > 0:
            tracking["win_rate"] = tracking.get("profitable_signals", 0) / total_signals

    def get_environment_strategy_metrics(self, exchange: str) -> dict[str, Any]:
        """Get strategy metrics for an exchange environment."""
        context = self.get_environment_context(exchange)
        strategy_config = self.get_environment_strategy_config(exchange)
        tracking = self._strategy_performance_tracking.get(exchange, {})

        return {
            "exchange": exchange,
            "environment": context.environment.value,
            "is_production": context.is_production,
            "strategy_mode": strategy_config.get("strategy_mode", StrategyMode.BALANCED).value,
            "active_strategies": len(self._active_strategies_by_env.get(exchange, [])),
            "max_concurrent_strategies": strategy_config.get("max_concurrent_strategies", 5),
            "total_signals": tracking.get("total_signals", 0),
            "profitable_signals": tracking.get("profitable_signals", 0),
            "win_rate": tracking.get("win_rate", Decimal("0")),
            "total_pnl": tracking.get("total_pnl", Decimal("0")),
            "current_drawdown": tracking.get("current_drawdown", Decimal("0")),
            "max_drawdown": tracking.get("max_drawdown", Decimal("0")),
            "max_position_size_pct": strategy_config.get("max_position_size_pct", Decimal("0.05")),
            "min_signal_confidence": strategy_config.get("min_signal_confidence", Decimal("0.6")),
            "enable_experimental_strategies": strategy_config.get(
                "enable_experimental_strategies", False
            ),
            "backtest_required": strategy_config.get("backtest_required", False),
            "strategy_violations": tracking.get("strategy_violations", 0),
            "last_rebalance": tracking.get("last_rebalance"),
            "last_updated": datetime.now().isoformat(),
        }

    async def rebalance_strategies_for_environment(self, exchange: str) -> dict[str, Any]:
        """Rebalance strategies based on environment configuration."""
        context = self.get_environment_context(exchange)
        env_config = self.get_environment_strategy_config(exchange)

        rebalance_freq = env_config.get("rebalancing_frequency_minutes", 60)
        tracking = self._strategy_performance_tracking.get(exchange, {})

        last_rebalance = tracking.get("last_rebalance")
        if last_rebalance:
            last_time = datetime.fromisoformat(last_rebalance)
            if datetime.now() - last_time < timedelta(minutes=rebalance_freq):
                return {
                    "status": "skipped",
                    "reason": "too_soon",
                    "next_rebalance_minutes": rebalance_freq,
                }

        # Perform rebalancing logic here
        logger.info(
            f"Rebalancing strategies for {exchange} (environment: {context.environment.value})"
        )

        # Update tracking
        tracking["last_rebalance"] = datetime.now().isoformat()

        return {
            "status": "completed",
            "exchange": exchange,
            "environment": context.environment.value,
            "active_strategies": len(self._active_strategies_by_env.get(exchange, [])),
            "rebalance_time": tracking["last_rebalance"],
        }
