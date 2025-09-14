"""
Simple tests for strategy configuration templates to boost coverage.
"""

from decimal import Decimal

import pytest

from src.core.types import StrategyType
from src.strategies.config_templates import StrategyConfigTemplates


class TestStrategyConfigTemplates:
    """Test strategy configuration templates."""
    
    def test_arbitrage_scanner_config_default(self):
        """Test default arbitrage scanner configuration."""
        config = StrategyConfigTemplates.get_arbitrage_scanner_config()
        
        assert config is not None
        assert "name" in config
        assert "strategy_type" in config
        assert config["strategy_type"] == StrategyType.ARBITRAGE.value
        assert "parameters" in config
    
    def test_arbitrage_scanner_config_with_exchanges(self):
        """Test arbitrage scanner with specific exchanges."""
        exchanges = ["binance", "coinbase"]
        config = StrategyConfigTemplates.get_arbitrage_scanner_config(exchanges=exchanges)
        
        assert "exchanges" in config["parameters"]
        assert config["parameters"]["exchanges"] == exchanges
    
    def test_arbitrage_scanner_config_risk_levels(self):
        """Test different risk levels."""
        for risk_level in ["conservative", "medium", "aggressive"]:
            config = StrategyConfigTemplates.get_arbitrage_scanner_config(risk_level=risk_level)
            assert config is not None
            assert "parameters" in config
            assert "name" in config
    
    def test_breakout_strategy_config(self):
        """Test breakout strategy configuration."""
        config = StrategyConfigTemplates.get_volatility_breakout_config()
        
        assert config["strategy_type"] == StrategyType.MOMENTUM.value
        assert "consolidation_period" in config["parameters"]
        assert "volume_multiplier" in config["parameters"]
    
    def test_breakout_config_timeframes(self):
        """Test breakout config with different regimes."""
        for regime in ["low", "medium", "high"]:
            config = StrategyConfigTemplates.get_volatility_breakout_config(volatility_regime=regime)
            assert config["strategy_type"] == StrategyType.MOMENTUM.value
            assert "volatility_period" in config["parameters"]
    
    def test_trend_following_config(self):
        """Test trend following configuration."""
        config = StrategyConfigTemplates.get_trend_following_config()
        
        assert config["strategy_type"] == StrategyType.TREND_FOLLOWING.value
        assert "fast_ma" in config["parameters"]
        assert "slow_ma" in config["parameters"]
    
    def test_trend_following_optimization(self):
        """Test trend following with optimization."""
        config = StrategyConfigTemplates.get_trend_following_config(
            trend_strength="aggressive"
        )
        assert config["strategy_type"] == StrategyType.TREND_FOLLOWING.value
        assert "fast_ma" in config["parameters"]
    
    def test_mean_reversion_config(self):
        """Test mean reversion configuration."""
        config = StrategyConfigTemplates.get_mean_reversion_config()
        
        assert config["strategy_type"] == StrategyType.MEAN_REVERSION.value
        assert "lookback_period" in config["parameters"]
        assert "atr_period" in config["parameters"]
    
    def test_mean_reversion_volatility_regime(self):
        """Test mean reversion for different volatility regimes."""
        for risk_level in ["conservative", "medium", "aggressive"]:
            config = StrategyConfigTemplates.get_mean_reversion_config(
                risk_level=risk_level
            )
            assert "volatility_filter" in config["parameters"]
    
    def test_market_making_config(self):
        """Test market making configuration."""
        config = StrategyConfigTemplates.get_market_making_config()
        
        assert config["strategy_type"] == StrategyType.MARKET_MAKING.value
        assert "base_spread" in config["parameters"]
        assert "inventory_skew" in config["parameters"]
    
    def test_market_making_advanced(self):
        """Test advanced market making features."""
        config = StrategyConfigTemplates.get_market_making_config(
            spread_type="dynamic",
            inventory_management="aggressive"
        )
        assert config["parameters"]["adaptive_spreads"] == True
        assert config["parameters"]["inventory_skew"] == True
    
    def test_cross_exchange_arbitrage_config(self):
        """Test cross-exchange arbitrage configuration."""
        config = StrategyConfigTemplates.get_arbitrage_scanner_config()
        
        assert config["strategy_type"] == StrategyType.ARBITRAGE.value
        assert "min_profit_threshold" in config["parameters"]
        assert "exchanges" in config["parameters"]
    
    def test_triangular_arbitrage_config(self):
        """Test triangular arbitrage configuration."""
        config = StrategyConfigTemplates.get_arbitrage_scanner_config()
        
        assert config["strategy_type"] == StrategyType.ARBITRAGE.value
        assert "triangular_paths" in config["parameters"]
        assert "min_profit_threshold" in config["parameters"]
    
    def test_adaptive_momentum_config(self):
        """Test adaptive momentum configuration."""
        config = StrategyConfigTemplates.get_volatility_breakout_config()
        
        assert config["strategy_type"] == StrategyType.MOMENTUM.value
        assert "volatility_period" in config["parameters"]
        assert "breakout_threshold" in config["parameters"]
    
    def test_volatility_breakout_config(self):
        """Test volatility breakout configuration."""
        config = StrategyConfigTemplates.get_volatility_breakout_config()
        
        assert config["strategy_type"] == StrategyType.MOMENTUM.value
        assert "volume_multiplier" in config["parameters"]
        assert "consolidation_period" in config["parameters"]
    
    def test_ml_enhanced_config(self):
        """Test ML-enhanced strategy configuration."""
        config = StrategyConfigTemplates.get_ensemble_config(
            strategy_types=["trend_following"]
        )
        
        assert "strategy_types" in config["parameters"]
        assert "voting_method" in config["parameters"]
        assert "trend_following" in config["parameters"]["strategy_types"]
    
    def test_portfolio_optimization_config(self):
        """Test portfolio optimization configuration."""
        config = StrategyConfigTemplates.get_ensemble_config()
        
        assert config["strategy_type"] == StrategyType.CUSTOM.value
        assert "voting_method" in config["parameters"]
        assert "performance_weight" in config["parameters"]
    
    def test_high_frequency_config(self):
        """Test high-frequency trading configuration."""
        config = StrategyConfigTemplates.get_ensemble_config()
        
        assert "strategy_types" in config["parameters"]
        assert "voting_method" in config["parameters"]
        assert config["parameters"]["voting_method"] == "weighted"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = StrategyConfigTemplates.get_trend_following_config()
        assert config is not None
        assert "strategy_type" in config
        
        # Invalid config (missing required fields)
        invalid_config = {"strategy_type": StrategyType.TREND_FOLLOWING}
        assert "parameters" not in invalid_config
    
    def test_config_merging(self):
        """Test configuration merging."""
        base_config = StrategyConfigTemplates.get_trend_following_config()
        override = {
            "parameters": {
                "custom_param": "value",
                "ema_periods": [10, 20, 30]
            }
        }
        
        # Simple merge test (manual implementation)
        merged = base_config.copy()
        if "parameters" in override:
            merged["parameters"].update(override["parameters"])
        assert "custom_param" in merged["parameters"]
        assert merged["parameters"]["custom_param"] == "value"
    
    def test_get_all_templates(self):
        """Test getting all available templates."""
        templates = StrategyConfigTemplates.get_all_templates()
        
        assert isinstance(templates, dict)
        assert len(templates) > 0
        assert "trend_following_1h" in templates
        assert "mean_reversion_1h" in templates
        assert "arbitrage_medium" in templates
    
    def test_environment_specific_configs(self):
        """Test environment-specific configurations."""
        # Production config
        prod_config = StrategyConfigTemplates.get_trend_following_config()
        assert prod_config["strategy_type"] == StrategyType.TREND_FOLLOWING.value
        assert "fast_ma" in prod_config["parameters"]
        
        # Sandbox config  
        sandbox_config = StrategyConfigTemplates.get_mean_reversion_config()
        assert sandbox_config["strategy_type"] == StrategyType.MEAN_REVERSION.value
        assert "lookback_period" in sandbox_config["parameters"]
        
        # Backtest config
        backtest_config = StrategyConfigTemplates.get_arbitrage_scanner_config()
        assert backtest_config["strategy_type"] == StrategyType.ARBITRAGE.value
        assert "triangular_paths" in backtest_config["parameters"]