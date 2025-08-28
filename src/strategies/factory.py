"""
Strategy Factory - Dependency injection and strategy instantiation.

This module provides:
- Strategy creation with dependency injection
- Configuration validation
- Service integration
- Strategy registry management
"""

from typing import Any

from src.core.exceptions import StrategyError, ValidationError
from src.core.types import StrategyConfig, StrategyType
from src.error_handling import (
    ErrorHandler,
    ErrorSeverity,
    get_global_error_handler,
    with_error_context,
    with_retry,
)
from src.strategies.interfaces import BaseStrategyInterface, StrategyFactoryInterface
from src.strategies.service import StrategyService
from src.strategies.validation import ValidationFramework
from src.utils.decorators import time_execution


class StrategyFactory(StrategyFactoryInterface):
    """
    Factory for creating strategies with proper dependency injection.

    Features:
    - Dependency injection for all strategy dependencies
    - Configuration validation before creation
    - Service integration
    - Strategy type validation
    """

    def __init__(
        self,
        strategy_service: StrategyService,
        validation_framework: ValidationFramework | None = None,
    ):
        """
        Initialize strategy factory.

        Args:
            strategy_service: Strategy service for dependency resolution
            validation_framework: Optional validation framework
        """
        self._strategy_service = strategy_service
        self._validation_framework = validation_framework
        self._error_handler: ErrorHandler = get_global_error_handler()

        # Registry of strategy types to their implementation classes
        self._strategy_registry: dict[StrategyType, type] = {}

        # Initialize built-in strategies
        self._register_builtin_strategies()

    def _register_builtin_strategies(self) -> None:
        """Register built-in strategy types."""
        try:
            # Import dynamic strategies
            from src.strategies.dynamic.adaptive_momentum import AdaptiveMomentumStrategy
            from src.strategies.dynamic.volatility_breakout import VolatilityBreakoutStrategy

            # Register dynamic strategy types
            self._strategy_registry[StrategyType.MOMENTUM] = AdaptiveMomentumStrategy
            self._strategy_registry[StrategyType.VOLATILITY_BREAKOUT] = VolatilityBreakoutStrategy

            # Import and register static strategies
            try:
                from src.strategies.static.arbitrage_scanner import ArbitrageOpportunity

                self._strategy_registry[StrategyType.ARBITRAGE] = ArbitrageOpportunity
            except ImportError:
                pass

            try:
                from src.strategies.static.mean_reversion import MeanReversionStrategy

                self._strategy_registry[StrategyType.MEAN_REVERSION] = MeanReversionStrategy
            except ImportError:
                pass

            try:
                from src.strategies.static.trend_following import TrendFollowingStrategy

                self._strategy_registry[StrategyType.TREND_FOLLOWING] = TrendFollowingStrategy
            except ImportError:
                pass

            try:
                from src.strategies.static.market_making import MarketMakingStrategy

                self._strategy_registry[StrategyType.MARKET_MAKING] = MarketMakingStrategy
            except ImportError:
                pass

            try:
                from src.strategies.static.breakout import BreakoutStrategy

                self._strategy_registry[StrategyType.BREAKOUT] = BreakoutStrategy
            except ImportError:
                pass

            try:
                from src.strategies.static.cross_exchange_arbitrage import (
                    CrossExchangeArbitrageStrategy,
                )

                self._strategy_registry[StrategyType.CROSS_EXCHANGE_ARBITRAGE] = (
                    CrossExchangeArbitrageStrategy
                )
            except ImportError:
                pass

            try:
                from src.strategies.static.triangular_arbitrage import TriangularArbitrageStrategy

                self._strategy_registry[StrategyType.TRIANGULAR_ARBITRAGE] = (
                    TriangularArbitrageStrategy
                )
            except ImportError:
                pass

            # Import and register hybrid strategies
            try:
                from src.strategies.hybrid.ensemble import EnsembleStrategy

                self._strategy_registry[StrategyType.ENSEMBLE] = EnsembleStrategy
            except ImportError:
                pass

            try:
                from src.strategies.hybrid.fallback import FallbackStrategy

                self._strategy_registry[StrategyType.FALLBACK] = FallbackStrategy
            except ImportError:
                pass

            try:
                from src.strategies.hybrid.rule_based_ai import RuleBasedAIStrategy

                self._strategy_registry[StrategyType.RULE_BASED_AI] = RuleBasedAIStrategy
            except ImportError:
                pass

        except ImportError:
            # Log warning but don't fail - strategies will be registered manually
            pass

    def register_strategy_type(self, strategy_type: StrategyType, strategy_class: type) -> None:
        """
        Register a strategy type with its implementation class.

        Args:
            strategy_type: Strategy type enum
            strategy_class: Strategy implementation class

        Raises:
            StrategyError: If strategy type is already registered
        """
        if strategy_type in self._strategy_registry:
            raise StrategyError(f"Strategy type {strategy_type} already registered")

        # Validate that class implements BaseStrategyInterface
        if not issubclass(strategy_class, BaseStrategyInterface):
            raise StrategyError(
                f"Strategy class {strategy_class.__name__} must implement BaseStrategyInterface"
            )

        self._strategy_registry[strategy_type] = strategy_class

    @time_execution
    @with_error_context(operation="create_strategy")
    @with_retry(max_attempts=3, base_delay=1.0)
    async def create_strategy(
        self, strategy_type: StrategyType, config: StrategyConfig
    ) -> BaseStrategyInterface:
        """
        Create a strategy instance with full dependency injection.

        Args:
            strategy_type: Type of strategy to create
            config: Strategy configuration

        Returns:
            Fully configured strategy instance

        Raises:
            StrategyError: If strategy creation fails
            ValidationError: If configuration is invalid
        """
        # Validate strategy type is supported
        if strategy_type not in self._strategy_registry:
            raise StrategyError(
                f"Unsupported strategy type: {strategy_type}. "
                f"Supported types: {list(self._strategy_registry.keys())}"
            )

        # Validate configuration
        if not self.validate_strategy_requirements(strategy_type, config):
            raise ValidationError(f"Invalid configuration for strategy type {strategy_type}")

        # Get strategy class
        strategy_class = self._strategy_registry[strategy_type]

        try:
            # Create strategy instance
            strategy = strategy_class(config.model_dump())

            # Inject dependencies
            await self._inject_dependencies(strategy, config)

            # Set validation framework if available
            if self._validation_framework:
                strategy.set_validation_framework(self._validation_framework)

            # Initialize strategy
            await strategy.initialize(config)

            return strategy

        except Exception as e:
            await self._error_handler.handle_error(
                error=e,
                context={
                    "strategy_type": strategy_type.value,
                    "config": config.model_dump(),
                    "operation": "create_strategy",
                },
                severity=ErrorSeverity.CRITICAL,
            )
            raise StrategyError(f"Failed to create strategy {strategy_type}: {e!s}")

    async def _inject_dependencies(
        self, strategy: BaseStrategyInterface, config: StrategyConfig
    ) -> None:
        """
        Inject all required dependencies into strategy.

        Args:
            strategy: Strategy instance to inject into
            config: Strategy configuration
        """
        try:
            # Inject risk manager if required
            if config.requires_risk_manager:
                risk_manager = self._strategy_service.resolve_dependency("RiskManager")
                strategy.set_risk_manager(risk_manager)

            # Inject exchange if required
            if config.requires_exchange:
                exchange_factory = self._strategy_service.resolve_dependency("ExchangeFactory")
                exchange = await exchange_factory.get_exchange(config.exchange_type)
                strategy.set_exchange(exchange)

            # Inject data service
            try:
                data_service = self._strategy_service.resolve_dependency("DataService")
                strategy.set_data_service(data_service)
            except (KeyError, AttributeError, ImportError) as e:
                # Data service is optional for some strategies
                pass
            except Exception as e:
                # Unexpected error resolving data service - log but continue
                pass

        except Exception as e:
            await self._error_handler.handle_error(
                error=e,
                context={
                    "strategy": strategy.__class__.__name__,
                    "config": config.model_dump(),
                    "operation": "inject_dependencies",
                },
                severity=ErrorSeverity.HIGH,
            )
            raise StrategyError(f"Failed to inject dependencies: {e!s}")

    def get_supported_strategies(self) -> list[StrategyType]:
        """
        Get list of supported strategy types.

        Returns:
            List of supported strategy types
        """
        return list(self._strategy_registry.keys())

    def validate_strategy_requirements(
        self, strategy_type: StrategyType, config: StrategyConfig
    ) -> bool:
        """
        Validate strategy requirements and configuration.

        Args:
            strategy_type: Strategy type to validate
            config: Strategy configuration

        Returns:
            True if requirements are met
        """
        try:
            # Check if strategy type is supported
            if strategy_type not in self._strategy_registry:
                return False

            # Validate basic configuration
            if not config.name or not config.strategy_id:
                return False

            # Validate strategy-specific requirements
            return self._validate_strategy_specific_requirements(strategy_type, config)

        except (ValueError, TypeError, AttributeError) as e:
            # Configuration validation errors
            return False
        except Exception as e:
            # Unexpected validation errors
            return False

    def _validate_strategy_specific_requirements(
        self, strategy_type: StrategyType, config: StrategyConfig
    ) -> bool:
        """
        Validate strategy-specific requirements.

        Args:
            strategy_type: Strategy type
            config: Strategy configuration

        Returns:
            True if strategy-specific requirements are met
        """
        # Get required parameters for each strategy type
        required_params = self._get_required_parameters(strategy_type)

        # Check if all required parameters are present
        for param in required_params:
            if param not in config.parameters:
                return False

        # Strategy-specific validation logic
        if strategy_type == StrategyType.MOMENTUM:
            return self._validate_momentum_strategy_config(config)
        elif strategy_type == StrategyType.MEAN_REVERSION:
            return self._validate_mean_reversion_strategy_config(config)
        elif strategy_type == StrategyType.ARBITRAGE:
            return self._validate_arbitrage_strategy_config(config)
        elif strategy_type == StrategyType.VOLATILITY_BREAKOUT:
            return self._validate_volatility_strategy_config(config)

        return True

    def _get_required_parameters(self, strategy_type: StrategyType) -> list[str]:
        """
        Get required parameters for a strategy type.

        Args:
            strategy_type: Strategy type

        Returns:
            List of required parameter names
        """
        parameter_requirements = {
            StrategyType.MOMENTUM: ["lookback_period", "momentum_threshold", "signal_strength"],
            StrategyType.MEAN_REVERSION: [
                "mean_period",
                "deviation_threshold",
                "reversion_strength",
            ],
            StrategyType.ARBITRAGE: ["exchanges", "min_profit_threshold", "max_exposure"],
            StrategyType.VOLATILITY_BREAKOUT: [
                "volatility_period",
                "breakout_threshold",
                "volume_confirmation",
            ],
        }

        return parameter_requirements.get(strategy_type, [])

    def _validate_momentum_strategy_config(self, config: StrategyConfig) -> bool:
        """Validate momentum strategy specific configuration."""
        params = config.parameters

        # Validate lookback period
        lookback = params.get("lookback_period", 0)
        if not isinstance(lookback, int) or lookback < 5 or lookback > 200:
            return False

        # Validate momentum threshold
        threshold = params.get("momentum_threshold", 0)
        if not isinstance(threshold, int | float) or threshold < 0 or threshold > 1:
            return False

        return True

    def _validate_mean_reversion_strategy_config(self, config: StrategyConfig) -> bool:
        """Validate mean reversion strategy specific configuration."""
        params = config.parameters

        # Validate mean period
        mean_period = params.get("mean_period", 0)
        if not isinstance(mean_period, int) or mean_period < 10 or mean_period > 200:
            return False

        # Validate deviation threshold
        deviation = params.get("deviation_threshold", 0)
        if not isinstance(deviation, int | float) or deviation < 0.5 or deviation > 5.0:
            return False

        return True

    def _validate_arbitrage_strategy_config(self, config: StrategyConfig) -> bool:
        """Validate arbitrage strategy specific configuration."""
        params = config.parameters

        # Validate exchanges list
        exchanges = params.get("exchanges", [])
        if not isinstance(exchanges, list) or len(exchanges) < 2:
            return False

        # Validate profit threshold
        profit_threshold = params.get("min_profit_threshold", 0)
        if not isinstance(profit_threshold, int | float) or profit_threshold <= 0:
            return False

        return True

    def _validate_volatility_strategy_config(self, config: StrategyConfig) -> bool:
        """Validate volatility breakout strategy specific configuration."""
        params = config.parameters

        # Validate volatility period
        vol_period = params.get("volatility_period", 0)
        if not isinstance(vol_period, int) or vol_period < 5 or vol_period > 100:
            return False

        # Validate breakout threshold
        breakout_threshold = params.get("breakout_threshold", 0)
        if not isinstance(breakout_threshold, int | float) or breakout_threshold < 1.0:
            return False

        return True

    async def create_strategy_with_validation(
        self,
        strategy_type: StrategyType,
        config: StrategyConfig,
        validate_dependencies: bool = True,
    ) -> BaseStrategyInterface:
        """
        Create strategy with comprehensive validation.

        Args:
            strategy_type: Strategy type to create
            config: Strategy configuration
            validate_dependencies: Whether to validate dependency availability

        Returns:
            Validated strategy instance

        Raises:
            StrategyError: If creation or validation fails
        """
        # Validate configuration using validation framework
        if self._validation_framework:
            validation_result = await self._validation_framework.validate_strategy_config(config)
            if not validation_result.is_valid:
                raise ValidationError(
                    f"Strategy configuration validation failed: {validation_result.errors}"
                )

        # Validate dependency availability if requested
        if validate_dependencies:
            if not await self._validate_dependency_availability(config):
                raise StrategyError("Required dependencies are not available")

        # Create strategy
        strategy = await self.create_strategy(strategy_type, config)

        # Final validation of created strategy
        if not await self._validate_created_strategy(strategy):
            raise StrategyError("Created strategy failed final validation")

        return strategy

    async def _validate_dependency_availability(self, config: StrategyConfig) -> bool:
        """
        Validate that all required dependencies are available.

        Args:
            config: Strategy configuration

        Returns:
            True if all dependencies are available
        """
        try:
            if config.requires_risk_manager:
                self._strategy_service.resolve_dependency("RiskManager")

            if config.requires_exchange:
                exchange_factory = self._strategy_service.resolve_dependency("ExchangeFactory")
                if not exchange_factory.is_exchange_supported(config.exchange_type):
                    return False

            return True

        except (AttributeError, TypeError, ValueError) as e:
            # Strategy-specific requirement validation errors
            return False
        except Exception as e:
            # Unexpected requirement validation errors
            return False

    async def _validate_created_strategy(self, strategy: BaseStrategyInterface) -> bool:
        """
        Validate a created strategy instance.

        Args:
            strategy: Strategy instance to validate

        Returns:
            True if strategy is valid
        """
        try:
            # Check that strategy has required properties
            if not hasattr(strategy, "name") or not strategy.name:
                return False

            if not hasattr(strategy, "strategy_type"):
                return False

            if not hasattr(strategy, "status"):
                return False

            # Check that required methods are callable
            required_methods = [
                "generate_signals",
                "validate_signal",
                "get_position_size",
                "should_exit",
                "start",
                "stop",
                "initialize",
            ]

            for method_name in required_methods:
                if not hasattr(strategy, method_name):
                    return False
                if not callable(getattr(strategy, method_name)):
                    return False

            return True

        except (AttributeError, TypeError) as e:
            # Strategy instance validation errors
            return False
        except Exception as e:
            # Unexpected strategy validation errors
            return False

    def get_strategy_info(self, strategy_type: StrategyType) -> dict[str, Any]:
        """
        Get information about a strategy type.

        Args:
            strategy_type: Strategy type to get info for

        Returns:
            Strategy information dictionary
        """
        if strategy_type not in self._strategy_registry:
            return {}

        strategy_class = self._strategy_registry[strategy_type]
        required_params = self._get_required_parameters(strategy_type)

        return {
            "strategy_type": strategy_type.value,
            "class_name": strategy_class.__name__,
            "module": strategy_class.__module__,
            "required_parameters": required_params,
            "supports_backtesting": True,  # All strategies support backtesting
            "requires_risk_manager": True,
            "requires_exchange": True,
        }

    def list_available_strategies(self) -> dict[str, Any]:
        """
        List all available strategies with their information.

        Returns:
            Dictionary mapping strategy types to their information
        """
        return {
            strategy_type.value: self.get_strategy_info(strategy_type)
            for strategy_type in self.get_supported_strategies()
        }
