"""
Enhanced Strategy Factory for Dynamic Strategies - Day 13

This factory creates strategy instances with proper service layer integration,
dependency injection, and enhanced architecture support.

Features:
- Automatic service dependency injection
- Enhanced strategy configuration validation
- Support for both legacy and refactored strategies
- Comprehensive initialization and validation
- Proper error handling and logging
"""

from typing import Any

from src.base import BaseComponent
from src.core.types import StrategyType
from src.data.features.technical_indicators import TechnicalIndicators
from src.risk_management.adaptive_risk import AdaptiveRiskManager
from src.risk_management.regime_detection import MarketRegimeDetector
from src.strategies.base import BaseStrategy
from src.strategies.service import StrategyService

# Import strategies
from .adaptive_momentum import AdaptiveMomentumStrategy
from .volatility_breakout import VolatilityBreakoutStrategy


class EnhancedDynamicStrategyFactory(BaseComponent):
    """
    Enhanced factory for creating dynamic strategies with service layer integration.

    This factory handles:
    - Strategy creation with proper dependency injection
    - Service layer integration setup
    - Configuration validation and enhancement
    - Legacy and refactored strategy support
    """

    def __init__(self):
        """Initialize the enhanced strategy factory."""
        super().__init__()

        # Service dependencies (will be injected)
        self._technical_indicators: TechnicalIndicators | None = None
        self._strategy_service: StrategyService | None = None
        self._regime_detector: MarketRegimeDetector | None = None
        self._adaptive_risk_manager: AdaptiveRiskManager | None = None

        # Strategy type mappings
        self._strategy_mappings = {
            # Legacy strategies
            "adaptive_momentum": AdaptiveMomentumStrategy,
            "volatility_breakout": VolatilityBreakoutStrategy,
            # Enhanced strategies (default)
            "adaptive_momentum_enhanced": AdaptiveMomentumStrategy,
            "volatility_breakout_enhanced": VolatilityBreakoutStrategy,
        }

        # Default to enhanced versions
        self._default_mappings = {
            StrategyType.DYNAMIC: {
                "adaptive_momentum": AdaptiveMomentumStrategy,
                "volatility_breakout": VolatilityBreakoutStrategy,
            }
        }

        self.logger.info(
            "Enhanced dynamic strategy factory initialized",
            available_strategies=list(self._strategy_mappings.keys()),
        )

    def set_technical_indicators(self, technical_indicators: TechnicalIndicators) -> None:
        """Set technical indicators service for strategy dependency injection."""
        self._technical_indicators = technical_indicators
        self.logger.info("Technical indicators service set in factory")

    def set_strategy_service(self, strategy_service: StrategyService) -> None:
        """Set strategy service for lifecycle management."""
        self._strategy_service = strategy_service
        self.logger.info("Strategy service set in factory")

    def set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None:
        """Set regime detector for adaptive strategies."""
        self._regime_detector = regime_detector
        self.logger.info("Regime detector set in factory")

    def set_adaptive_risk_manager(self, adaptive_risk_manager: AdaptiveRiskManager) -> None:
        """Set adaptive risk manager for dynamic risk adjustment."""
        self._adaptive_risk_manager = adaptive_risk_manager
        self.logger.info("Adaptive risk manager set in factory")

    async def create_strategy(
        self, strategy_name: str, config: dict[str, Any], use_enhanced: bool = True
    ) -> BaseStrategy | None:
        """
        Create a strategy instance with proper service integration.

        Args:
            strategy_name: Name of the strategy to create
            config: Strategy configuration dictionary
            use_enhanced: Whether to use enhanced versions (default: True)

        Returns:
            Configured strategy instance or None if creation fails
        """
        try:
            # Determine strategy class to use
            strategy_class = self._resolve_strategy_class(strategy_name, use_enhanced)
            if not strategy_class:
                self.logger.error(
                    "Unknown strategy type",
                    strategy_name=strategy_name,
                    available_strategies=list(self._strategy_mappings.keys()),
                )
                return None

            # Enhance configuration with defaults
            enhanced_config = await self._enhance_configuration(strategy_name, config)

            # Create strategy instance
            self.logger.info(
                "Creating strategy instance",
                strategy_name=strategy_name,
                strategy_class=strategy_class.__name__,
                enhanced_config=use_enhanced,
            )

            strategy = strategy_class(enhanced_config)

            # Inject service dependencies
            await self._inject_dependencies(strategy, strategy_name)

            # Validate strategy setup
            if not await self._validate_strategy_setup(strategy):
                self.logger.error(
                    "Strategy setup validation failed",
                    strategy_name=strategy_name,
                )
                return None

            self.logger.info(
                "Strategy created successfully",
                strategy_name=strategy_name,
                strategy_class=strategy_class.__name__,
                strategy_id=getattr(strategy, "config", {}).get("strategy_id", "unknown"),
            )

            return strategy

        except Exception as e:
            self.logger.error(
                "Strategy creation failed",
                strategy_name=strategy_name,
                error=str(e),
                exc_info=True,
            )
            return None

    def _resolve_strategy_class(self, strategy_name: str, use_enhanced: bool):
        """Resolve strategy class based on name and enhancement preference."""
        try:
            # Direct mapping lookup
            if strategy_name in self._strategy_mappings:
                return self._strategy_mappings[strategy_name]

            # Enhanced version preference
            if use_enhanced:
                enhanced_name = f"{strategy_name}_enhanced"
                if enhanced_name in self._strategy_mappings:
                    return self._strategy_mappings[enhanced_name]

            # Default mapping lookup
            for _strategy_type, mappings in self._default_mappings.items():
                if strategy_name in mappings:
                    return mappings[strategy_name]

            return None

        except Exception as e:
            self.logger.error(
                "Strategy class resolution failed",
                strategy_name=strategy_name,
                error=str(e),
            )
            return None

    async def _enhance_configuration(
        self, strategy_name: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Enhance strategy configuration with defaults and validations."""
        try:
            enhanced_config = config.copy()

            # Ensure required fields
            if "name" not in enhanced_config:
                enhanced_config["name"] = f"{strategy_name}_instance"

            if "strategy_id" not in enhanced_config:
                from uuid import uuid4

                enhanced_config["strategy_id"] = str(uuid4())

            # Add strategy-specific defaults
            if "parameters" not in enhanced_config:
                enhanced_config["parameters"] = {}

            # Add common defaults
            parameters = enhanced_config["parameters"]

            if strategy_name in ["adaptive_momentum", "adaptive_momentum_enhanced"]:
                parameters.setdefault("fast_ma_period", 20)
                parameters.setdefault("slow_ma_period", 50)
                parameters.setdefault("rsi_period", 14)
                parameters.setdefault("momentum_lookback", 10)
                parameters.setdefault("volume_threshold", 1.5)
                parameters.setdefault("min_data_points", 50)

            elif strategy_name in ["volatility_breakout", "volatility_breakout_enhanced"]:
                parameters.setdefault("atr_period", 14)
                parameters.setdefault("breakout_multiplier", 2.0)
                parameters.setdefault("consolidation_period", 20)
                parameters.setdefault("time_decay_factor", 0.95)
                parameters.setdefault("min_data_points", 50)
                parameters.setdefault("breakout_cooldown_minutes", 30)

            # Add common strategy defaults
            enhanced_config.setdefault("min_confidence", 0.3)
            enhanced_config.setdefault("position_size_pct", 0.02)
            enhanced_config.setdefault("max_positions", 5)
            enhanced_config.setdefault("enabled", True)

            self.logger.debug(
                "Configuration enhanced",
                strategy_name=strategy_name,
                original_config=config,
                enhanced_config=enhanced_config,
            )

            return enhanced_config

        except Exception as e:
            self.logger.error(
                "Configuration enhancement failed",
                strategy_name=strategy_name,
                error=str(e),
            )
            return config

    async def _inject_dependencies(self, strategy: BaseStrategy, strategy_name: str) -> None:
        """Inject service dependencies into strategy instance."""
        try:
            # Inject technical indicators service
            if self._technical_indicators and hasattr(strategy, "set_technical_indicators"):
                strategy.set_technical_indicators(self._technical_indicators)

            # Inject strategy service
            if self._strategy_service and hasattr(strategy, "set_strategy_service"):
                strategy.set_strategy_service(self._strategy_service)

            # Inject regime detector
            if self._regime_detector and hasattr(strategy, "set_regime_detector"):
                strategy.set_regime_detector(self._regime_detector)

            # Inject adaptive risk manager
            if self._adaptive_risk_manager and hasattr(strategy, "set_adaptive_risk_manager"):
                strategy.set_adaptive_risk_manager(self._adaptive_risk_manager)

            # Use BaseStrategy's built-in dependency injection methods
            if hasattr(strategy, "set_data_service") and self._strategy_service:
                # Get data service from strategy service if available
                data_service = getattr(self._strategy_service, "_data_service", None)
                if data_service:
                    strategy.set_data_service(data_service)

            self.logger.debug(
                "Dependencies injected",
                strategy_name=strategy_name,
                technical_indicators=self._technical_indicators is not None,
                strategy_service=self._strategy_service is not None,
                regime_detector=self._regime_detector is not None,
                adaptive_risk_manager=self._adaptive_risk_manager is not None,
            )

        except Exception as e:
            self.logger.error(
                "Dependency injection failed",
                strategy_name=strategy_name,
                error=str(e),
            )

    async def _validate_strategy_setup(self, strategy: BaseStrategy) -> bool:
        """Validate that strategy is properly set up with required dependencies."""
        try:
            # Check basic strategy properties
            if not hasattr(strategy, "name") or not strategy.name:
                self.logger.error("Strategy missing name")
                return False

            if not hasattr(strategy, "strategy_type"):
                self.logger.error("Strategy missing strategy_type property")
                return False

            # Check required methods
            required_methods = [
                "_generate_signals_impl",
                "validate_signal",
                "get_position_size",
                "should_exit",
            ]

            for method_name in required_methods:
                if not hasattr(strategy, method_name):
                    self.logger.error(
                        "Strategy missing required method",
                        method_name=method_name,
                    )
                    return False

            # Validate configuration
            if hasattr(strategy, "config"):
                config = strategy.config
                if hasattr(config, "min_confidence") and config.min_confidence <= 0:
                    self.logger.error("Invalid min_confidence in strategy config")
                    return False

                if hasattr(config, "position_size_pct") and config.position_size_pct <= 0:
                    self.logger.error("Invalid position_size_pct in strategy config")
                    return False

            self.logger.debug(
                "Strategy setup validation passed",
                strategy_name=strategy.name,
            )

            return True

        except Exception as e:
            self.logger.error(
                "Strategy setup validation failed",
                error=str(e),
            )
            return False

    def get_available_strategies(self) -> dict[str, str]:
        """Get list of available strategies with descriptions."""
        return {
            "adaptive_momentum": "Legacy adaptive momentum strategy",
            "adaptive_momentum_enhanced": "Enhanced momentum strategy with service layer",
            "volatility_breakout": "Legacy volatility breakout strategy",
            "volatility_breakout_enhanced": "Enhanced breakout strategy with service layer",
        }

    def get_strategy_requirements(self, strategy_name: str) -> dict[str, Any]:
        """Get requirements and recommendations for a strategy."""
        requirements = {
            "required_services": [],
            "recommended_services": [],
            "required_parameters": [],
            "optional_parameters": [],
        }

        if strategy_name in ["adaptive_momentum", "adaptive_momentum_enhanced"]:
            requirements.update(
                {
                    "required_services": ["TechnicalIndicators", "DataService"],
                    "recommended_services": ["RegimeDetector", "AdaptiveRiskManager"],
                    "required_parameters": ["fast_ma_period", "slow_ma_period", "rsi_period"],
                    "optional_parameters": [
                        "momentum_lookback",
                        "volume_threshold",
                        "min_data_points",
                    ],
                }
            )
        elif strategy_name in ["volatility_breakout", "volatility_breakout_enhanced"]:
            requirements.update(
                {
                    "required_services": ["TechnicalIndicators", "DataService"],
                    "recommended_services": ["RegimeDetector", "AdaptiveRiskManager"],
                    "required_parameters": ["atr_period", "breakout_multiplier"],
                    "optional_parameters": [
                        "consolidation_period",
                        "time_decay_factor",
                        "breakout_cooldown_minutes",
                    ],
                }
            )

        return requirements

    async def create_multiple_strategies(
        self, strategy_configs: dict[str, dict[str, Any]], use_enhanced: bool = True
    ) -> dict[str, BaseStrategy | None]:
        """
        Create multiple strategy instances.

        Args:
            strategy_configs: Dictionary mapping strategy names to their configs
            use_enhanced: Whether to use enhanced versions

        Returns:
            Dictionary mapping strategy names to created instances (or None if failed)
        """
        strategies = {}

        for strategy_name, config in strategy_configs.items():
            try:
                strategy = await self.create_strategy(strategy_name, config, use_enhanced)
                strategies[strategy_name] = strategy

                if strategy:
                    self.logger.info(
                        "Successfully created strategy in batch",
                        strategy_name=strategy_name,
                    )
                else:
                    self.logger.error(
                        "Failed to create strategy in batch",
                        strategy_name=strategy_name,
                    )
            except Exception as e:
                self.logger.error(
                    "Exception during batch strategy creation",
                    strategy_name=strategy_name,
                    error=str(e),
                )
                strategies[strategy_name] = None

        success_count = sum(1 for s in strategies.values() if s is not None)
        self.logger.info(
            "Batch strategy creation completed",
            total_requested=len(strategy_configs),
            successful=success_count,
            failed=len(strategy_configs) - success_count,
        )

        return strategies
