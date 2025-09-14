"""
Strategy Factory - Dependency injection and strategy instantiation.

This module provides:
- Strategy creation with dependency injection
- Configuration validation
- Service integration
- Strategy registry management
"""

from decimal import Decimal
from typing import Any

from src.core.exceptions import StrategyError, ValidationError
from src.core.logging import get_logger
from src.core.types import StrategyConfig, StrategyType
from src.error_handling import (
    ErrorHandler,
    ErrorSeverity,
    get_global_error_handler,
    with_error_context,
    with_retry,
)

# Enhanced imports for comprehensive integration
from src.core.logging import PerformanceMonitor
from src.monitoring import AlertManager, MetricsCollector
from src.strategies.dependencies import StrategyServiceContainer, create_strategy_service_container
from src.strategies.interfaces import BaseStrategyInterface, StrategyFactoryInterface
from src.strategies.validation import ValidationFramework
from src.utils.validators import validate_financial_range

from src.utils.decorators import time_execution

# Constants for production configuration
MIN_LOOKBACK_PERIOD = 5
MAX_LOOKBACK_PERIOD = 200
MIN_MOMENTUM_THRESHOLD = 0
MAX_MOMENTUM_THRESHOLD = 1
MIN_MEAN_PERIOD = 10
MAX_MEAN_PERIOD = 200
MIN_DEVIATION_THRESHOLD = 0.5
MAX_DEVIATION_THRESHOLD = 5.0
MIN_EXCHANGES_REQUIRED = 2
MIN_VOLATILITY_PERIOD = 5
MAX_VOLATILITY_PERIOD = 100
MIN_BREAKOUT_THRESHOLD = 1.0
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0


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
        service_container: StrategyServiceContainer | None = None,
        validation_framework: ValidationFramework | None = None,
        repository=None,
        risk_manager=None,
        exchange_factory=None,
        data_service=None,
        service_manager=None,
    ):
        """
        Initialize strategy factory with comprehensive dependency injection.

        Args:
            validation_framework: Optional validation framework
            repository: Strategy repository (injected)
            risk_manager: Risk manager service (injected)
            exchange_factory: Exchange factory (injected)
            data_service: Data service (injected)
            service_manager: Service manager for comprehensive service resolution
        """
        self.logger = get_logger(__name__)
        self._validation_framework = validation_framework
        self._error_handler: ErrorHandler = get_global_error_handler()

        # Use service container for dependency injection
        self._service_container = service_container
        
        # Fallback to direct injection for backward compatibility
        self._repository = repository
        self._risk_manager = risk_manager
        self._exchange_factory = exchange_factory
        self._data_service = data_service
        self._service_manager = service_manager

        # Registry of strategy types to their implementation classes
        self._strategy_registry: dict[StrategyType, type] = {}

        # Comprehensive monitoring services - will be injected if available
        self._metrics_collector: MetricsCollector | None = None
        self._alert_manager: AlertManager | None = None
        self._performance_monitor: PerformanceMonitor | None = None
        self._telemetry_collector = None

        # Try to get monitoring services from service manager
        if service_manager:
            try:
                self._metrics_collector = service_manager.get_service("metrics_collector")
            except Exception as e:
                self.logger.debug(f"Metrics collector not available: {e}")

            try:
                self._alert_manager = service_manager.get_service("alert_manager")
            except Exception as e:
                self.logger.debug(f"Alert manager not available: {e}")

            try:
                self._performance_monitor = service_manager.get_service("performance_monitor")
            except Exception as e:
                self.logger.debug(f"Performance monitor not available: {e}")

            try:
                self._telemetry_collector = service_manager.get_service("telemetry_collector")
            except Exception as e:
                self.logger.debug(f"Telemetry collector not available: {e}")

        # Initialize built-in strategies
        self._register_builtin_strategies()

        self.logger.info(
            "StrategyFactory initialized with comprehensive integration",
            monitoring_services={
                "metrics_collector": self._metrics_collector is not None,
                "alert_manager": self._alert_manager is not None,
                "performance_monitor": self._performance_monitor is not None,
                "telemetry_collector": self._telemetry_collector is not None,
            }
        )

    def _register_builtin_strategies(self) -> None:
        """Register built-in strategy types."""
        # Initialize empty strategy registry - defer imports to avoid circular dependencies
        # Strategies will be imported and registered on-demand when create_strategy is called
        pass

    def _lazy_load_strategy_class(self, strategy_type: StrategyType) -> type | None:
        """Lazy load strategy class to avoid circular imports."""
        try:
            # Check if already loaded
            if strategy_type in self._strategy_registry:
                return self._strategy_registry[strategy_type]

            # Import strategies on-demand
            if strategy_type == StrategyType.MOMENTUM:
                try:
                    from src.strategies.dynamic.adaptive_momentum import AdaptiveMomentumStrategy

                    self._strategy_registry[StrategyType.MOMENTUM] = AdaptiveMomentumStrategy
                    return AdaptiveMomentumStrategy
                except ImportError:
                    pass

                try:
                    from src.strategies.static.breakout import BreakoutStrategy

                    # Only override if not already set
                    if StrategyType.MOMENTUM not in self._strategy_registry:
                        self._strategy_registry[StrategyType.MOMENTUM] = BreakoutStrategy
                    return BreakoutStrategy
                except ImportError:
                    pass

            elif strategy_type == StrategyType.MEAN_REVERSION:
                try:
                    from src.strategies.static.mean_reversion import MeanReversionStrategy

                    self._strategy_registry[StrategyType.MEAN_REVERSION] = MeanReversionStrategy
                    return MeanReversionStrategy
                except ImportError:
                    pass

            elif strategy_type == StrategyType.ARBITRAGE:
                try:
                    from src.strategies.static.arbitrage_scanner import ArbitrageOpportunity

                    self._strategy_registry[StrategyType.ARBITRAGE] = ArbitrageOpportunity
                    return ArbitrageOpportunity
                except ImportError:
                    pass

            elif strategy_type == StrategyType.MARKET_MAKING:
                try:
                    from src.strategies.static.market_making import MarketMakingStrategy

                    self._strategy_registry[StrategyType.MARKET_MAKING] = MarketMakingStrategy
                    return MarketMakingStrategy
                except ImportError:
                    pass

            elif strategy_type == StrategyType.TREND_FOLLOWING:
                try:
                    from src.strategies.static.trend_following import TrendFollowingStrategy

                    self._strategy_registry[StrategyType.TREND_FOLLOWING] = TrendFollowingStrategy
                    return TrendFollowingStrategy
                except ImportError:
                    pass

            elif strategy_type == StrategyType.CUSTOM:
                # Try hybrid strategies
                try:
                    from src.strategies.hybrid.ensemble import EnsembleStrategy

                    self._strategy_registry[StrategyType.CUSTOM] = EnsembleStrategy
                    return EnsembleStrategy
                except ImportError:
                    pass

            return None

        except Exception as e:
            self.logger.error(f"Failed to lazy load strategy class for {strategy_type}: {e}")
            return None

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
    @with_retry(max_attempts=DEFAULT_RETRY_ATTEMPTS, base_delay=DEFAULT_RETRY_DELAY)
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
        # Get strategy class using lazy loading
        strategy_class = self._lazy_load_strategy_class(strategy_type)
        if strategy_class is None:
            raise StrategyError(
                f"Unsupported strategy type: {strategy_type}. "
                f"No implementation found for this strategy type."
            )

        # Validate configuration
        if not self.validate_strategy_requirements(strategy_type, config):
            raise ValidationError(f"Invalid configuration for strategy type {strategy_type}")

        try:
            # Create service container for strategy
            if not self._service_container:
                self._service_container = await self._create_comprehensive_service_container(config)

            # Create strategy instance with service container
            strategy = strategy_class(config.model_dump())

            # Inject dependencies
            await self._inject_dependencies(strategy, config)

            # Set validation framework if available
            if self._validation_framework and hasattr(strategy, "set_validation_framework"):
                strategy.set_validation_framework(self._validation_framework)

            # Initialize strategy
            await strategy.initialize(config)

            return strategy

        except Exception as e:
            if self._error_handler:
                await self._error_handler.handle_error(
                    error=e,
                    context={
                        "strategy_type": strategy_type.value,
                        "config": config.model_dump(),
                        "operation": "create_strategy",
                    },
                    severity=ErrorSeverity.CRITICAL,
                )
            else:
                self.logger.error(f"Failed to create strategy {strategy_type}: {e!s}")
            raise StrategyError(f"Failed to create strategy {strategy_type}: {e!s}")

    async def _create_comprehensive_service_container(self, config: StrategyConfig) -> StrategyServiceContainer:
        """Create comprehensive service container with all available services."""
        try:
            # Get services from service manager if available
            risk_service = None
            execution_service = None
            monitoring_service = None
            state_service = None
            capital_service = None
            analytics_service = None
            ml_service = None

            if self._service_manager:
                try:
                    risk_service = await self._service_manager.get_service("risk_management")
                except Exception as e:
                    self.logger.debug(f"Failed to get risk_management service: {e}")
                    risk_service = self._risk_manager  # Fallback to injected dependency

                try:
                    execution_service = await self._service_manager.get_service("execution")
                except Exception as e:
                    self.logger.debug(f"Failed to get execution service: {e}")
                    pass  # No fallback available

                try:
                    monitoring_service = await self._service_manager.get_service("monitoring")
                except Exception as e:
                    self.logger.debug(f"Failed to get monitoring service: {e}")
                    pass  # No fallback available

                try:
                    state_service = await self._service_manager.get_service("state")
                except Exception as e:
                    self.logger.debug(f"Failed to get state service: {e}")
                    pass  # No fallback available

                try:
                    capital_service = await self._service_manager.get_service("capital_management")
                except Exception as e:
                    self.logger.debug(f"Failed to get capital_management service: {e}")
                    pass  # No fallback available

                try:
                    analytics_service = await self._service_manager.get_service("analytics")
                except Exception as e:
                    self.logger.debug(f"Failed to get analytics service: {e}")
                    pass  # No fallback available

                try:
                    ml_service = await self._service_manager.get_service("ml")
                except Exception as e:
                    self.logger.debug(f"Failed to get ml service: {e}")
                    pass  # No fallback available

            # Create the comprehensive service container
            container = create_strategy_service_container(
                risk_service=risk_service,
                data_service=self._data_service,
                execution_service=execution_service,
                monitoring_service=monitoring_service,
                state_service=state_service,
                capital_service=capital_service,
                analytics_service=analytics_service,
                ml_service=ml_service,
            )

            self.logger.info(
                "Comprehensive service container created",
                strategy_name=config.name,
                services_available=container.get_service_status()
            )

            return container

        except Exception as e:
            self.logger.error(f"Failed to create service container: {e}")
            # Return basic container as fallback
            return create_strategy_service_container(
                risk_service=self._risk_manager,
                data_service=self._data_service
            )

    async def _enhance_strategy_with_integrations(self, strategy, config: StrategyConfig) -> None:
        """Enhance strategy with comprehensive integrations."""
        try:
            # Enhance with shared utilities if strategy supports them
            if hasattr(strategy, "_apply_shared_utilities"):
                strategy._apply_shared_utilities()

            # Set advanced configuration options
            if hasattr(strategy, "configure_advanced_features"):
                advanced_config = {
                    "enable_caching": True,
                    "enable_metrics": True,
                    "enable_alerting": True,
                    "enable_tracing": True
                }
                strategy.configure_advanced_features(advanced_config)

            # Initialize strategy-specific monitoring
            if self._metrics_collector and hasattr(strategy, "setup_custom_metrics"):
                strategy.setup_custom_metrics(self._metrics_collector)

        except Exception as e:
            self.logger.warning(f"Failed to enhance strategy with integrations: {e}")

    def _validate_configuration_parameters(self, config: StrategyConfig) -> bool:
        """Validate configuration parameters using utils validators."""
        try:
            # Validate position size if present
            if hasattr(config, "position_size_pct"):
                try:
                    validate_financial_range(
                        Decimal(str(config.position_size_pct)),
                        min_value=Decimal("0.001"),
                        max_value=Decimal("0.5"),
                        field_name="position_size_pct"
                    )
                except (ValueError, ValidationError) as e:
                    self.logger.error(f"Position size validation failed: {e}")
                    return False

            # Validate risk parameters
            if hasattr(config, "risk_per_trade"):
                try:
                    validate_financial_range(
                        Decimal(str(config.risk_per_trade)),
                        min_value=Decimal("0.001"),
                        max_value=Decimal("0.1"),
                        field_name="risk_per_trade"
                    )
                except (ValueError, ValidationError) as e:
                    self.logger.error(f"Risk per trade validation failed: {e}")
                    return False

            # Validate parameter ranges based on strategy type
            parameters = getattr(config, "parameters", {})

            if "lookback_period" in parameters:
                lookback = parameters["lookback_period"]
                if not isinstance(lookback, int) or lookback < MIN_LOOKBACK_PERIOD or lookback > MAX_LOOKBACK_PERIOD:
                    return False

            if "threshold" in parameters:
                threshold = parameters["threshold"]
                if not isinstance(threshold, (int, float)) or threshold < 0:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def _get_integration_status(self, strategy) -> dict[str, bool]:
        """Get integration status for a strategy."""
        return {
            "has_service_container": hasattr(strategy, "services"),
            "has_monitoring": hasattr(strategy, "_metrics_collector") and strategy._metrics_collector is not None,
            "has_caching": hasattr(strategy, "cache_manager"),
            "has_validation": hasattr(strategy, "market_data_validator"),
            "has_error_handling": hasattr(strategy, "_error_handler") and strategy._error_handler is not None,
            "has_shared_utilities": hasattr(strategy, "_apply_shared_utilities"),
        }

    async def _inject_dependencies(
        self, strategy: BaseStrategyInterface, config: StrategyConfig
    ) -> None:
        """
        Inject all required dependencies into strategy using service container.

        Args:
            strategy: Strategy instance to inject into
            config: Strategy configuration
        """
        try:
            # Use service container if available
            if self._service_container and hasattr(strategy, "set_services"):
                strategy.set_services(self._service_container)
                return

            # Fallback to individual service injection
            # Inject repository if available
            if self._repository and hasattr(strategy, "set_repository"):
                strategy.set_repository(self._repository)

            # Inject risk manager if required and available
            risk_service = self._service_container.risk_service if self._service_container else self._risk_manager
            if config.requires_risk_manager and risk_service:
                if hasattr(strategy, "set_risk_manager"):
                    strategy.set_risk_manager(risk_service)

            # Inject exchange if required and available
            if config.requires_exchange and self._exchange_factory:
                try:
                    exchange = await self._exchange_factory.get_exchange(config.exchange_type)
                    if hasattr(strategy, "set_exchange"):
                        strategy.set_exchange(exchange)
                except Exception as e:
                    self.logger.warning(f"Failed to get exchange {config.exchange_type}: {e}")

            # Inject data service if available
            data_service = self._service_container.data_service if self._service_container else self._data_service
            if data_service and hasattr(strategy, "set_data_service"):
                strategy.set_data_service(data_service)

        except Exception as e:
            if self._error_handler:
                await self._error_handler.handle_error(
                    error=e,
                    context={
                        "strategy": strategy.__class__.__name__,
                        "config": config.model_dump(),
                        "operation": "inject_dependencies",
                    },
                    severity=ErrorSeverity.HIGH,
                )
            else:
                self.logger.error(f"Failed to inject dependencies: {e!s}")
            raise StrategyError(f"Failed to inject dependencies: {e!s}")

    def get_supported_strategies(self) -> list[StrategyType]:
        """
        Get list of supported strategy types.

        Returns:
            List of supported strategy types
        """
        # Try to lazy load all strategy types to populate the registry
        all_strategy_types = [
            StrategyType.MOMENTUM,
            StrategyType.MEAN_REVERSION,
            StrategyType.ARBITRAGE,
            StrategyType.MARKET_MAKING,
            StrategyType.TREND_FOLLOWING,
            StrategyType.CUSTOM,
        ]

        supported = []
        for strategy_type in all_strategy_types:
            if self._lazy_load_strategy_class(strategy_type) is not None:
                supported.append(strategy_type)

        return supported

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
            # Check if strategy type is supported using lazy loading
            strategy_class = self._lazy_load_strategy_class(strategy_type)
            if strategy_class is None:
                return False

            # Validate basic configuration
            if not config.name or not config.strategy_id:
                return False

            # Validate strategy-specific requirements
            return self._validate_strategy_specific_requirements(strategy_type, config)

        except (ValueError, TypeError, AttributeError):
            # Configuration validation errors
            return False
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
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
        elif strategy_type == StrategyType.CUSTOM:
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
            StrategyType.CUSTOM: [
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
        if (
            not isinstance(lookback, int)
            or lookback < MIN_LOOKBACK_PERIOD
            or lookback > MAX_LOOKBACK_PERIOD
        ):
            return False

        # Validate momentum threshold
        threshold = params.get("momentum_threshold", 0)
        if (
            not isinstance(threshold, (int, float))
            or threshold < MIN_MOMENTUM_THRESHOLD
            or threshold > MAX_MOMENTUM_THRESHOLD
        ):
            return False

        return True

    def _validate_mean_reversion_strategy_config(self, config: StrategyConfig) -> bool:
        """Validate mean reversion strategy specific configuration."""
        params = config.parameters

        # Validate mean period
        mean_period = params.get("mean_period", 0)
        if (
            not isinstance(mean_period, int)
            or mean_period < MIN_MEAN_PERIOD
            or mean_period > MAX_MEAN_PERIOD
        ):
            return False

        # Validate deviation threshold
        deviation = params.get("deviation_threshold", 0)
        if (
            not isinstance(deviation, (int, float))
            or deviation < MIN_DEVIATION_THRESHOLD
            or deviation > MAX_DEVIATION_THRESHOLD
        ):
            return False

        return True

    def _validate_arbitrage_strategy_config(self, config: StrategyConfig) -> bool:
        """Validate arbitrage strategy specific configuration."""
        params = config.parameters

        # Validate exchanges list
        exchanges = params.get("exchanges", [])
        if not isinstance(exchanges, list) or len(exchanges) < MIN_EXCHANGES_REQUIRED:
            return False

        # Validate profit threshold
        profit_threshold = params.get("min_profit_threshold", 0)
        if not isinstance(profit_threshold, (int, float)) or profit_threshold <= 0:
            return False

        return True

    def _validate_volatility_strategy_config(self, config: StrategyConfig) -> bool:
        """Validate volatility breakout strategy specific configuration."""
        params = config.parameters

        # Validate volatility period
        vol_period = params.get("volatility_period", 0)
        if (
            not isinstance(vol_period, int)
            or vol_period < MIN_VOLATILITY_PERIOD
            or vol_period > MAX_VOLATILITY_PERIOD
        ):
            return False

        # Validate breakout threshold
        breakout_threshold = params.get("breakout_threshold", 0)
        if (
            not isinstance(breakout_threshold, (int, float))
            or breakout_threshold < MIN_BREAKOUT_THRESHOLD
        ):
            return False

        return True

    @time_execution
    @with_error_context(operation="create_strategy_with_validation")
    @with_retry(max_attempts=DEFAULT_RETRY_ATTEMPTS, base_delay=DEFAULT_RETRY_DELAY)
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
            if not self._validate_dependency_availability_sync(config):
                raise StrategyError("Required dependencies are not available")

        # Create strategy
        strategy = await self.create_strategy(strategy_type, config)

        # Final validation of created strategy
        if not await self._validate_created_strategy(strategy):
            raise StrategyError("Created strategy failed final validation")

        return strategy

    def _validate_dependency_availability_sync(self, config: StrategyConfig) -> bool:
        """
        Validate that all required dependencies are available.

        Args:
            config: Strategy configuration

        Returns:
            True if all dependencies are available
        """
        try:
            if config.requires_risk_manager and not self._risk_manager:
                return False

            if config.requires_exchange:
                if not self._exchange_factory:
                    return False
                # Note: is_exchange_supported check removed since we don't have that method consistently
                # Individual exchange creation will handle validation

            return True

        except (AttributeError, TypeError, ValueError):
            # Strategy-specific requirement validation errors
            return False
        except Exception as e:
            self.logger.error(f"Strategy specific requirement validation failed: {e}")
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

        except (AttributeError, TypeError):
            # Strategy instance validation errors
            return False
        except Exception as e:
            self.logger.error(f"Strategy validation failed: {e}")
            return False

    def get_strategy_info(self, strategy_type: StrategyType) -> dict[str, Any]:
        """
        Get information about a strategy type.

        Args:
            strategy_type: Strategy type to get info for

        Returns:
            Strategy information dictionary
        """
        strategy_class = self._lazy_load_strategy_class(strategy_type)
        if strategy_class is None:
            return {}
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
