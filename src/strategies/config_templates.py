"""
Strategy Configuration Templates - Production-ready strategy configurations.

This module provides comprehensive configuration templates for all strategy types,
including parameter ranges, risk settings, and deployment configurations for
institutional trading environments.

Key Features:
- Production-tested parameter combinations
- Risk-adjusted configurations for different market regimes
- Multi-timeframe strategy variants
- Exchange-specific optimizations
- Backtesting-validated parameter sets
- Performance monitoring configurations
"""

from typing import Any

from src.core.types import StrategyType


class StrategyConfigTemplates:
    """
    Comprehensive strategy configuration templates for production deployment.

    Each template includes:
    - Core strategy parameters
    - Risk management settings
    - Performance monitoring configuration
    - Exchange-specific optimizations
    - Backtesting validation metrics
    """

    @staticmethod
    def get_arbitrage_scanner_config(
        risk_level: str = "medium",
        exchanges: list[str] = None,
        symbols: list[str] = None,
    ) -> dict[str, Any]:
        """
        Get arbitrage scanner configuration.

        Args:
            risk_level: Risk level (conservative, medium, aggressive)
            exchanges: List of exchanges to monitor
            symbols: List of symbols to scan

        Returns:
            Complete arbitrage scanner configuration
        """
        if exchanges is None:
            exchanges = ["binance", "okx", "coinbase"]

        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"]

        base_config = {
            "name": "arbitrage_scanner_v1",
            "strategy_id": "arb_scanner_001",
            "strategy_type": StrategyType.ARBITRAGE.value,
            "exchange_type": "multi_exchange",
            "symbols": symbols,
            "requires_risk_manager": True,
            "requires_exchange": True,
            "min_confidence": 0.7,
            "position_size_pct": 0.02,
            "parameters": {
                # Core arbitrage parameters
                "scan_interval": 100,  # milliseconds
                "max_execution_time": 500,  # milliseconds
                "exchanges": exchanges,
                "symbols": symbols,
                "max_opportunities": 10,
                "triangular_paths": [
                    ["BTCUSDT", "ETHBTC", "ETHUSDT"],
                    ["BTCUSDT", "BNBBTC", "BNBUSDT"],
                    ["ETHUSDT", "ADAETH", "ADAUSDT"],
                ],
                # Risk parameters
                "total_capital": 10000,
                "risk_per_trade": 0.02,
                "max_position_size": 0.1,
                "max_concurrent_trades": 3,
                # Performance parameters
                "slippage_tolerance": 0.001,
                "latency_threshold": 200,  # milliseconds
                "profit_taking_threshold": 0.5,
            },
            # Backtesting configuration
            "backtesting": {
                "enabled": True,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000,
                "commission": 0.001,
                "slippage": 0.0005,
            },
            # Monitoring configuration
            "monitoring": {
                "enabled": True,
                "metrics_interval": 60,  # seconds
                "alert_thresholds": {
                    "max_drawdown": 0.05,
                    "min_win_rate": 0.7,
                    "max_latency": 500,
                },
            },
        }

        # Risk level adjustments
        risk_adjustments = {
            "conservative": {
                "min_profit_threshold": 0.002,  # 0.2%
                "max_position_size_pct": 0.05,
                "max_concurrent_trades": 2,
                "risk_per_trade": 0.01,
            },
            "medium": {
                "min_profit_threshold": 0.0015,  # 0.15%
                "max_position_size_pct": 0.1,
                "max_concurrent_trades": 3,
                "risk_per_trade": 0.02,
            },
            "aggressive": {
                "min_profit_threshold": 0.001,  # 0.1%
                "max_position_size_pct": 0.15,
                "max_concurrent_trades": 5,
                "risk_per_trade": 0.03,
            },
        }

        # Apply risk level adjustments
        if risk_level in risk_adjustments:
            base_config["parameters"].update(risk_adjustments[risk_level])

        return base_config

    @staticmethod
    def get_mean_reversion_config(
        timeframe: str = "1h",
        risk_level: str = "medium",
    ) -> dict[str, Any]:
        """
        Get mean reversion strategy configuration.

        Args:
            timeframe: Trading timeframe (5m, 15m, 1h, 4h, 1d)
            risk_level: Risk level (conservative, medium, aggressive)

        Returns:
            Complete mean reversion configuration
        """
        timeframe_params = {
            "5m": {
                "lookback_period": 48,  # 4 hours of 5m candles
                "entry_threshold": 2.5,
                "exit_threshold": 0.5,
                "atr_period": 20,
                "min_volume_ratio": 2.0,
            },
            "15m": {
                "lookback_period": 32,  # 8 hours of 15m candles
                "entry_threshold": 2.2,
                "exit_threshold": 0.6,
                "atr_period": 16,
                "min_volume_ratio": 1.8,
            },
            "1h": {
                "lookback_period": 24,  # 24 hours
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "atr_period": 14,
                "min_volume_ratio": 1.5,
            },
            "4h": {
                "lookback_period": 20,  # ~3 days
                "entry_threshold": 1.8,
                "exit_threshold": 0.4,
                "atr_period": 12,
                "min_volume_ratio": 1.3,
            },
            "1d": {
                "lookback_period": 20,  # 20 days
                "entry_threshold": 1.5,
                "exit_threshold": 0.3,
                "atr_period": 10,
                "min_volume_ratio": 1.2,
            },
        }

        base_config = {
            "name": f"mean_reversion_{timeframe}_v1",
            "strategy_id": f"mean_rev_{timeframe}_001",
            "strategy_type": StrategyType.MEAN_REVERSION.value,
            "exchange_type": "single",
            "symbols": ["BTCUSDT"],
            "requires_risk_manager": True,
            "requires_exchange": True,
            "min_confidence": 0.6,
            "position_size_pct": 0.03,
            "parameters": {
                **timeframe_params.get(timeframe, timeframe_params["1h"]),
                "atr_multiplier": 2.0,
                "volume_filter": True,
                "confirmation_timeframe": timeframe,
                # Risk management
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "max_holding_time": 48,  # hours
                "max_position_size_pct": 0.1,
                # Signal filtering
                "min_price_change": 0.005,  # 0.5%
                "volatility_filter": True,
                "trend_filter": False,  # Disable for pure mean reversion
            },
            "backtesting": {
                "enabled": True,
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000,
                "commission": 0.001,
                "slippage": 0.0005,
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 300,
                "alert_thresholds": {
                    "max_drawdown": 0.08,
                    "min_sharpe_ratio": 0.8,
                    "min_win_rate": 0.55,
                },
            },
        }

        # Risk level adjustments
        risk_adjustments = {
            "conservative": {
                "position_size_pct": 0.02,
                "entry_threshold": base_config["parameters"]["entry_threshold"] * 1.2,
                "stop_loss_pct": 0.015,
                "max_holding_time": 24,
            },
            "medium": {
                "position_size_pct": 0.03,
                "stop_loss_pct": 0.02,
            },
            "aggressive": {
                "position_size_pct": 0.05,
                "entry_threshold": base_config["parameters"]["entry_threshold"] * 0.8,
                "stop_loss_pct": 0.025,
                "max_holding_time": 72,
            },
        }

        if risk_level in risk_adjustments:
            base_config["parameters"].update(risk_adjustments[risk_level])
            if "position_size_pct" in risk_adjustments[risk_level]:
                base_config["position_size_pct"] = risk_adjustments[risk_level]["position_size_pct"]

        return base_config

    @staticmethod
    def get_trend_following_config(
        timeframe: str = "1h",
        trend_strength: str = "medium",
    ) -> dict[str, Any]:
        """
        Get trend following strategy configuration.

        Args:
            timeframe: Trading timeframe
            trend_strength: Trend strength requirement (weak, medium, strong)

        Returns:
            Complete trend following configuration
        """
        timeframe_params = {
            "5m": {
                "fast_ma": 12,
                "slow_ma": 26,
                "rsi_period": 14,
                "min_volume_ratio": 1.8,
            },
            "15m": {
                "fast_ma": 20,
                "slow_ma": 50,
                "rsi_period": 14,
                "min_volume_ratio": 1.5,
            },
            "1h": {
                "fast_ma": 20,
                "slow_ma": 50,
                "rsi_period": 14,
                "min_volume_ratio": 1.3,
            },
            "4h": {
                "fast_ma": 12,
                "slow_ma": 30,
                "rsi_period": 14,
                "min_volume_ratio": 1.2,
            },
            "1d": {
                "fast_ma": 10,
                "slow_ma": 20,
                "rsi_period": 14,
                "min_volume_ratio": 1.1,
            },
        }

        trend_params = {
            "weak": {
                "rsi_overbought": 75,
                "rsi_oversold": 25,
                "min_trend_strength": 0.5,
                "confirmation_candles": 1,
            },
            "medium": {
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "min_trend_strength": 0.7,
                "confirmation_candles": 2,
            },
            "strong": {
                "rsi_overbought": 65,
                "rsi_oversold": 35,
                "min_trend_strength": 0.8,
                "confirmation_candles": 3,
            },
        }

        base_config = {
            "name": f"trend_following_{timeframe}_v1",
            "strategy_id": f"trend_{timeframe}_001",
            "strategy_type": StrategyType.TREND_FOLLOWING.value,
            "exchange_type": "single",
            "symbols": ["BTCUSDT"],
            "requires_risk_manager": True,
            "requires_exchange": True,
            "min_confidence": 0.65,
            "position_size_pct": 0.04,
            "parameters": {
                **timeframe_params.get(timeframe, timeframe_params["1h"]),
                **trend_params.get(trend_strength, trend_params["medium"]),
                "volume_confirmation": True,
                "max_pyramid_levels": 3,
                "trailing_stop_pct": 0.02,
                "time_exit_hours": 48,
                # Risk management
                "stop_loss_pct": 0.025,
                "take_profit_ratio": 2.5,  # Risk-reward ratio
                "max_position_size_pct": 0.15,
                # Signal filtering
                "momentum_confirmation": True,
                "volatility_threshold": 0.02,
                "liquidity_threshold": 100000,  # Min volume in quote currency
            },
            "backtesting": {
                "enabled": True,
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000,
                "commission": 0.001,
                "slippage": 0.001,
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 300,
                "alert_thresholds": {
                    "max_drawdown": 0.10,
                    "min_sharpe_ratio": 0.6,
                    "min_win_rate": 0.50,
                    "max_consecutive_losses": 5,
                },
            },
        }

        return base_config

    @staticmethod
    def get_market_making_config(
        symbol: str = "BTCUSDT",
        spread_type: str = "dynamic",
        inventory_management: str = "aggressive",
    ) -> dict[str, Any]:
        """
        Get market making strategy configuration.

        Args:
            symbol: Trading symbol
            spread_type: Spread management (fixed, dynamic, adaptive)
            inventory_management: Inventory control (conservative, balanced, aggressive)

        Returns:
            Complete market making configuration
        """
        spread_params = {
            "fixed": {
                "base_spread": 0.001,
                "adaptive_spreads": False,
                "volatility_multiplier": 1.0,
            },
            "dynamic": {
                "base_spread": 0.0008,
                "adaptive_spreads": True,
                "volatility_multiplier": 2.0,
                "competition_monitoring": True,
            },
            "adaptive": {
                "base_spread": 0.0006,
                "adaptive_spreads": True,
                "volatility_multiplier": 3.0,
                "competition_monitoring": True,
                "spread_optimization": True,
            },
        }

        inventory_params = {
            "conservative": {
                "target_inventory": 0.3,
                "max_inventory": 0.8,
                "inventory_skew": True,
                "rebalance_threshold": 0.1,
                "inventory_risk_aversion": 0.2,
            },
            "balanced": {
                "target_inventory": 0.5,
                "max_inventory": 1.0,
                "inventory_skew": True,
                "rebalance_threshold": 0.15,
                "inventory_risk_aversion": 0.1,
            },
            "aggressive": {
                "target_inventory": 0.7,
                "max_inventory": 1.5,
                "inventory_skew": True,
                "rebalance_threshold": 0.2,
                "inventory_risk_aversion": 0.05,
            },
        }

        base_config = {
            "name": f"market_making_{symbol.lower()}_v1",
            "strategy_id": f"mm_{symbol.lower()}_001",
            "strategy_type": StrategyType.MARKET_MAKING.value,
            "exchange_type": "single",
            "symbols": [symbol],
            "requires_risk_manager": True,
            "requires_exchange": True,
            "min_confidence": 0.8,
            "position_size_pct": 0.05,
            "parameters": {
                **spread_params.get(spread_type, spread_params["dynamic"]),
                **inventory_params.get(inventory_management, inventory_params["balanced"]),
                "order_levels": 5,
                "base_order_size": 0.01,
                "size_multiplier": 1.5,
                "order_size_distribution": "exponential",
                # Risk management
                "max_position_value": 10000,
                "stop_loss_inventory": 2.0,
                "daily_loss_limit": 100,
                "min_profit_per_trade": 0.00001,
                # Order management
                "order_refresh_time": 30,  # seconds
                "competitive_quotes_enabled": True,
                "order_book_levels": 10,
                "tick_size_optimization": True,
                # Performance optimization
                "latency_optimization": True,
                "order_batching": True,
                "smart_routing": True,
            },
            "backtesting": {
                "enabled": True,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 50000,
                "commission": 0.0005,  # Maker fee
                "slippage": 0.0001,  # Minimal for market making
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 30,  # High frequency monitoring
                "alert_thresholds": {
                    "max_inventory_deviation": 0.3,
                    "min_spread_capture": 0.0002,
                    "max_adverse_selection": 0.1,
                    "min_fill_rate": 0.6,
                },
            },
        }

        return base_config

    @staticmethod
    def get_volatility_breakout_config(
        volatility_regime: str = "medium",
        breakout_type: str = "range",
    ) -> dict[str, Any]:
        """
        Get volatility breakout strategy configuration.

        Args:
            volatility_regime: Expected volatility (low, medium, high)
            breakout_type: Breakout detection (range, bollinger, atr)

        Returns:
            Complete volatility breakout configuration
        """
        volatility_params = {
            "low": {
                "volatility_period": 20,
                "breakout_threshold": 1.5,
                "min_volatility": 0.01,
                "volume_multiplier": 2.0,
            },
            "medium": {
                "volatility_period": 14,
                "breakout_threshold": 2.0,
                "min_volatility": 0.015,
                "volume_multiplier": 1.8,
            },
            "high": {
                "volatility_period": 10,
                "breakout_threshold": 2.5,
                "min_volatility": 0.02,
                "volume_multiplier": 1.5,
            },
        }

        breakout_params = {
            "range": {
                "consolidation_period": 20,
                "range_threshold": 0.02,
                "false_breakout_filter": True,
            },
            "bollinger": {
                "bb_period": 20,
                "bb_std": 2.0,
                "squeeze_detection": True,
            },
            "atr": {
                "atr_period": 14,
                "atr_multiplier": 1.5,
                "trailing_atr": True,
            },
        }

        base_config = {
            "name": f"volatility_breakout_{volatility_regime}_v1",
            "strategy_id": f"vol_break_{volatility_regime}_001",
            "strategy_type": StrategyType.VOLATILITY_BREAKOUT.value,
            "exchange_type": "single",
            "symbols": ["BTCUSDT"],
            "requires_risk_manager": True,
            "requires_exchange": True,
            "min_confidence": 0.7,
            "position_size_pct": 0.04,
            "parameters": {
                **volatility_params.get(volatility_regime, volatility_params["medium"]),
                **breakout_params.get(breakout_type, breakout_params["range"]),
                "volume_confirmation": True,
                "trend_filter": True,
                "regime_filter": True,
                # Position management
                "pyramiding_enabled": False,
                "scale_in_levels": 2,
                "profit_target_ratio": 2.0,
                # Risk management
                "stop_loss_pct": 0.03,
                "trailing_stop": True,
                "time_stop_hours": 24,
                "max_holding_time": 72,
                # Signal validation
                "confirmation_period": 3,  # candles
                "momentum_threshold": 0.02,
                "liquidity_check": True,
            },
            "backtesting": {
                "enabled": True,
                "start_date": "2023-06-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000,
                "commission": 0.001,
                "slippage": 0.001,
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 300,
                "alert_thresholds": {
                    "max_drawdown": 0.12,
                    "min_profit_factor": 1.3,
                    "min_win_rate": 0.45,
                    "max_avg_loss": 0.04,
                },
            },
        }

        return base_config

    @staticmethod
    def get_ensemble_config(
        strategy_types: list[str] = None,
        voting_method: str = "weighted",
        correlation_limit: float = 0.7,
    ) -> dict[str, Any]:
        """
        Get ensemble strategy configuration.

        Args:
            strategy_types: List of strategy types to include
            voting_method: Voting mechanism (majority, weighted, confidence)
            correlation_limit: Maximum correlation between strategies

        Returns:
            Complete ensemble configuration
        """
        if strategy_types is None:
            strategy_types = [
                "mean_reversion",
                "trend_following",
                "momentum",
                "volatility_breakout",
            ]

        base_config = {
            "name": f"ensemble_{voting_method}_v1",
            "strategy_id": f"ensemble_{voting_method}_001",
            "strategy_type": StrategyType.ENSEMBLE.value,
            "exchange_type": "single",
            "symbols": ["BTCUSDT"],
            "requires_risk_manager": True,
            "requires_exchange": True,
            "min_confidence": 0.6,
            "position_size_pct": 0.06,
            "parameters": {
                "strategy_types": strategy_types,
                "voting_method": voting_method,
                "correlation_limit": correlation_limit,
                "min_strategies": 2,
                "max_strategies": len(strategy_types),
                # Weighting parameters
                "performance_weight": 0.4,
                "confidence_weight": 0.3,
                "diversity_weight": 0.2,
                "recency_weight": 0.1,
                # Dynamic adjustment
                "weight_adaptation": True,
                "adaptation_period": 100,  # trades
                "min_weight": 0.1,
                "max_weight": 0.5,
                # Signal aggregation
                "signal_threshold": 0.6,
                "consensus_required": 0.6,  # 60% of strategies must agree
                "veto_power": True,  # Any strategy can veto if confidence < 0.3
                # Risk management
                "portfolio_heat": 0.02,  # Max risk per ensemble signal
                "correlation_monitoring": True,
                "regime_awareness": True,
                "strategy_rotation": True,
            },
            "backtesting": {
                "enabled": True,
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 50000,
                "commission": 0.001,
                "slippage": 0.0008,
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 300,
                "alert_thresholds": {
                    "max_strategy_correlation": correlation_limit,
                    "min_active_strategies": 2,
                    "max_weight_concentration": 0.6,
                    "min_ensemble_confidence": 0.5,
                },
            },
        }

        return base_config

    @classmethod
    def get_all_templates(cls) -> dict[str, dict[str, Any]]:
        """
        Get all available strategy configuration templates.

        Returns:
            Dictionary of all strategy templates
        """
        return {
            # Arbitrage strategies
            "arbitrage_conservative": cls.get_arbitrage_scanner_config("conservative"),
            "arbitrage_medium": cls.get_arbitrage_scanner_config("medium"),
            "arbitrage_aggressive": cls.get_arbitrage_scanner_config("aggressive"),
            # Mean reversion strategies
            "mean_reversion_5m": cls.get_mean_reversion_config("5m", "medium"),
            "mean_reversion_1h": cls.get_mean_reversion_config("1h", "medium"),
            "mean_reversion_4h": cls.get_mean_reversion_config("4h", "conservative"),
            "mean_reversion_1d": cls.get_mean_reversion_config("1d", "conservative"),
            # Trend following strategies
            "trend_following_5m": cls.get_trend_following_config("5m", "strong"),
            "trend_following_1h": cls.get_trend_following_config("1h", "medium"),
            "trend_following_4h": cls.get_trend_following_config("4h", "medium"),
            "trend_following_1d": cls.get_trend_following_config("1d", "weak"),
            # Market making strategies
            "market_making_btc": cls.get_market_making_config("BTCUSDT", "dynamic", "balanced"),
            "market_making_eth": cls.get_market_making_config("ETHUSDT", "adaptive", "aggressive"),
            "market_making_stable": cls.get_market_making_config(
                "BTCUSDT", "fixed", "conservative"
            ),
            # Volatility breakout strategies
            "volatility_breakout_low": cls.get_volatility_breakout_config("low", "range"),
            "volatility_breakout_medium": cls.get_volatility_breakout_config("medium", "bollinger"),
            "volatility_breakout_high": cls.get_volatility_breakout_config("high", "atr"),
            # Ensemble strategies
            "ensemble_conservative": cls.get_ensemble_config(
                ["mean_reversion", "market_making"], "weighted", 0.5
            ),
            "ensemble_balanced": cls.get_ensemble_config(
                ["mean_reversion", "trend_following", "volatility_breakout"], "confidence", 0.7
            ),
            "ensemble_aggressive": cls.get_ensemble_config(
                ["trend_following", "momentum", "volatility_breakout", "breakout"], "majority", 0.8
            ),
        }

    @classmethod
    def get_template_by_name(cls, template_name: str) -> dict[str, Any]:
        """
        Get specific template by name.

        Args:
            template_name: Name of template to retrieve

        Returns:
            Strategy configuration template

        Raises:
            KeyError: If template name not found
        """
        all_templates = cls.get_all_templates()
        if template_name not in all_templates:
            available = ", ".join(all_templates.keys())
            raise KeyError(f"Template '{template_name}' not found. Available: {available}")

        return all_templates[template_name]

    @classmethod
    def list_available_templates(cls) -> list[str]:
        """
        List all available template names.

        Returns:
            List of template names
        """
        return list(cls.get_all_templates().keys())

    @classmethod
    def get_templates_by_strategy_type(cls, strategy_type: str) -> dict[str, dict[str, Any]]:
        """
        Get all templates for a specific strategy type.

        Args:
            strategy_type: Strategy type to filter by

        Returns:
            Dictionary of templates for the specified type
        """
        all_templates = cls.get_all_templates()
        return {
            name: config
            for name, config in all_templates.items()
            if strategy_type.lower() in name.lower()
        }

    @classmethod
    def validate_template(cls, template: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate a strategy template configuration.

        Args:
            template: Template configuration to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        required_fields = [
            "name",
            "strategy_id",
            "strategy_type",
            "exchange_type",
            "symbols",
            "min_confidence",
            "position_size_pct",
            "parameters",
        ]

        for field in required_fields:
            if field not in template:
                errors.append(f"Missing required field: {field}")

        # Validate parameters
        if "parameters" in template:
            params = template["parameters"]

            # Check numeric parameters are in reasonable ranges
            numeric_checks = {
                "position_size_pct": (0.001, 0.5, "Position size must be between 0.1% and 50%"),
                "min_confidence": (0.1, 1.0, "Min confidence must be between 0.1 and 1.0"),
                "stop_loss_pct": (0.005, 0.2, "Stop loss must be between 0.5% and 20%"),
            }

            for param, (min_val, max_val, message) in numeric_checks.items():
                if param in params:
                    value = params[param]
                    if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                        errors.append(f"{param}: {message}")

        # Validate backtesting configuration
        if "backtesting" in template:
            backtest = template["backtesting"]
            if backtest.get("enabled"):
                required_backtest_fields = ["start_date", "end_date", "initial_capital"]
                for field in required_backtest_fields:
                    if field not in backtest:
                        errors.append(f"Backtesting missing required field: {field}")

        return len(errors) == 0, errors
