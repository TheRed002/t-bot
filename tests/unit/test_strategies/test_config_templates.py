"""Unit tests for strategies config templates."""

import pytest
from src.strategies.config_templates import StrategyConfigTemplates
from src.core.types import StrategyType


def test_get_arbitrage_scanner_config_defaults():
    """Test arbitrage scanner config with defaults."""
    config = StrategyConfigTemplates.get_arbitrage_scanner_config()
    
    assert config["name"] == "arbitrage_scanner_v1"
    assert config["strategy_id"] == "arb_scanner_001"
    assert config["strategy_type"] == StrategyType.ARBITRAGE.value
    assert config["exchange_type"] == "multi_exchange"
    assert config["symbols"] == ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"]
    assert config["requires_risk_manager"] is True
    assert config["requires_exchange"] is True
    assert config["min_confidence"] == 0.7
    assert config["position_size_pct"] == 0.02
    
    # Check parameters
    params = config["parameters"]
    assert params["scan_interval"] == 100
    assert params["max_execution_time"] == 500
    assert params["exchanges"] == ["binance", "okx", "coinbase"]
    assert params["symbols"] == ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"]
    assert params["max_opportunities"] == 10
    assert len(params["triangular_paths"]) == 3
    
    # Check backtesting config
    assert config["backtesting"]["enabled"] is True
    assert config["backtesting"]["initial_capital"] == 10000
    
    # Check monitoring config
    assert config["monitoring"]["enabled"] is True
    assert config["monitoring"]["metrics_interval"] == 60


def test_get_arbitrage_scanner_config_conservative():
    """Test arbitrage scanner config with conservative risk level."""
    config = StrategyConfigTemplates.get_arbitrage_scanner_config("conservative")
    
    params = config["parameters"]
    assert params["min_profit_threshold"] == 0.002
    assert params["max_position_size_pct"] == 0.05
    assert params["max_concurrent_trades"] == 2
    assert params["risk_per_trade"] == 0.01


def test_get_arbitrage_scanner_config_aggressive():
    """Test arbitrage scanner config with aggressive risk level."""
    config = StrategyConfigTemplates.get_arbitrage_scanner_config("aggressive")
    
    params = config["parameters"]
    assert params["min_profit_threshold"] == 0.001
    assert params["max_position_size_pct"] == 0.15
    assert params["max_concurrent_trades"] == 5
    assert params["risk_per_trade"] == 0.03


def test_get_arbitrage_scanner_config_custom_exchanges():
    """Test arbitrage scanner config with custom exchanges."""
    custom_exchanges = ["binance", "kucoin"]
    config = StrategyConfigTemplates.get_arbitrage_scanner_config(
        exchanges=custom_exchanges
    )
    
    assert config["parameters"]["exchanges"] == custom_exchanges


def test_get_arbitrage_scanner_config_custom_symbols():
    """Test arbitrage scanner config with custom symbols."""
    custom_symbols = ["BTCUSDT", "ETHUSDT"]
    config = StrategyConfigTemplates.get_arbitrage_scanner_config(
        symbols=custom_symbols
    )
    
    assert config["symbols"] == custom_symbols
    assert config["parameters"]["symbols"] == custom_symbols


def test_get_mean_reversion_config_defaults():
    """Test mean reversion config with defaults."""
    config = StrategyConfigTemplates.get_mean_reversion_config()
    
    assert config["name"] == "mean_reversion_1h_v1"
    assert config["strategy_id"] == "mean_rev_1h_001"
    assert config["strategy_type"] == StrategyType.MEAN_REVERSION.value
    assert config["exchange_type"] == "single"
    assert config["symbol"] == "BTCUSDT"
    
    # Check 1h timeframe parameters
    params = config["parameters"]
    assert params["lookback_period"] == 24
    assert params["entry_threshold"] == 2.0
    assert params["exit_threshold"] == 0.5
    assert params["atr_period"] == 14


def test_get_mean_reversion_config_different_timeframes():
    """Test mean reversion config with different timeframes."""
    # Test 5m
    config_5m = StrategyConfigTemplates.get_mean_reversion_config("5m")
    assert config_5m["parameters"]["lookback_period"] == 48
    assert config_5m["parameters"]["entry_threshold"] == 2.5
    
    # Test 4h
    config_4h = StrategyConfigTemplates.get_mean_reversion_config("4h")
    assert config_4h["parameters"]["lookback_period"] == 20
    assert config_4h["parameters"]["entry_threshold"] == 1.8
    
    # Test 1d
    config_1d = StrategyConfigTemplates.get_mean_reversion_config("1d")
    assert config_1d["parameters"]["lookback_period"] == 20
    assert config_1d["parameters"]["entry_threshold"] == 1.5


def test_get_mean_reversion_config_risk_levels():
    """Test mean reversion config with different risk levels."""
    # Conservative
    config_conservative = StrategyConfigTemplates.get_mean_reversion_config(
        risk_level="conservative"
    )
    assert config_conservative["position_size_pct"] == 0.02
    assert config_conservative["parameters"]["stop_loss_pct"] == 0.015
    
    # Aggressive
    config_aggressive = StrategyConfigTemplates.get_mean_reversion_config(
        risk_level="aggressive"
    )
    assert config_aggressive["position_size_pct"] == 0.05
    assert config_aggressive["parameters"]["stop_loss_pct"] == 0.025


def test_get_trend_following_config_defaults():
    """Test trend following config with defaults."""
    config = StrategyConfigTemplates.get_trend_following_config()
    
    assert config["name"] == "trend_following_1h_v1"
    assert config["strategy_id"] == "trend_1h_001"
    assert config["strategy_type"] == StrategyType.TREND_FOLLOWING.value
    
    # Check medium trend strength parameters
    params = config["parameters"]
    assert params["rsi_overbought"] == 70
    assert params["rsi_oversold"] == 30
    assert params["min_trend_strength"] == 0.7
    assert params["confirmation_candles"] == 2


def test_get_trend_following_config_trend_strengths():
    """Test trend following config with different trend strengths."""
    # Weak
    config_weak = StrategyConfigTemplates.get_trend_following_config(
        trend_strength="weak"
    )
    params_weak = config_weak["parameters"]
    assert params_weak["rsi_overbought"] == 75
    assert params_weak["min_trend_strength"] == 0.5
    
    # Strong
    config_strong = StrategyConfigTemplates.get_trend_following_config(
        trend_strength="strong"
    )
    params_strong = config_strong["parameters"]
    assert params_strong["rsi_overbought"] == 65
    assert params_strong["min_trend_strength"] == 0.8


def test_get_market_making_config_defaults():
    """Test market making config with defaults."""
    config = StrategyConfigTemplates.get_market_making_config()
    
    assert config["name"] == "market_making_btcusdt_v1"
    assert config["strategy_id"] == "mm_btcusdt_001"
    assert config["strategy_type"] == StrategyType.MARKET_MAKING.value
    assert config["symbols"] == ["BTCUSDT"]
    
    # Check dynamic spread parameters
    params = config["parameters"]
    assert params["base_spread"] == 0.0008
    assert params["adaptive_spreads"] is True
    assert params["volatility_multiplier"] == 2.0
    
    # Check aggressive inventory parameters (default)
    assert params["target_inventory"] == 0.7
    assert params["max_inventory"] == 1.5


def test_get_market_making_config_spread_types():
    """Test market making config with different spread types."""
    # Fixed
    config_fixed = StrategyConfigTemplates.get_market_making_config(
        spread_type="fixed"
    )
    params_fixed = config_fixed["parameters"]
    assert params_fixed["base_spread"] == 0.001
    assert params_fixed["adaptive_spreads"] is False
    
    # Adaptive
    config_adaptive = StrategyConfigTemplates.get_market_making_config(
        spread_type="adaptive"
    )
    params_adaptive = config_adaptive["parameters"]
    assert params_adaptive["base_spread"] == 0.0006
    assert params_adaptive["spread_optimization"] is True


def test_get_market_making_config_inventory_management():
    """Test market making config with different inventory management."""
    # Conservative
    config_conservative = StrategyConfigTemplates.get_market_making_config(
        inventory_management="conservative"
    )
    params_conservative = config_conservative["parameters"]
    assert params_conservative["target_inventory"] == 0.3
    assert params_conservative["max_inventory"] == 0.8
    
    # Aggressive
    config_aggressive = StrategyConfigTemplates.get_market_making_config(
        inventory_management="aggressive"
    )
    params_aggressive = config_aggressive["parameters"]
    assert params_aggressive["target_inventory"] == 0.7
    assert params_aggressive["max_inventory"] == 1.5


def test_get_volatility_breakout_config_defaults():
    """Test volatility breakout config with defaults."""
    config = StrategyConfigTemplates.get_volatility_breakout_config()
    
    assert config["name"] == "volatility_breakout_medium_v1"
    assert config["strategy_id"] == "vol_break_medium_001"
    assert config["strategy_type"] == StrategyType.MOMENTUM.value
    
    # Check medium volatility parameters
    params = config["parameters"]
    assert params["volatility_period"] == 14
    assert params["breakout_threshold"] == 2.0
    assert params["min_volatility"] == 0.015
    
    # Check range breakout parameters (default)
    assert params["consolidation_period"] == 20
    assert params["range_threshold"] == 0.02


def test_get_volatility_breakout_config_volatility_regimes():
    """Test volatility breakout config with different regimes."""
    # Low volatility
    config_low = StrategyConfigTemplates.get_volatility_breakout_config("low")
    params_low = config_low["parameters"]
    assert params_low["volatility_period"] == 20
    assert params_low["breakout_threshold"] == 1.5
    assert params_low["min_volatility"] == 0.01
    
    # High volatility
    config_high = StrategyConfigTemplates.get_volatility_breakout_config("high")
    params_high = config_high["parameters"]
    assert params_high["volatility_period"] == 10
    assert params_high["breakout_threshold"] == 2.5
    assert params_high["min_volatility"] == 0.02


def test_get_volatility_breakout_config_breakout_types():
    """Test volatility breakout config with different breakout types."""
    # Bollinger
    config_bb = StrategyConfigTemplates.get_volatility_breakout_config(
        breakout_type="bollinger"
    )
    params_bb = config_bb["parameters"]
    assert params_bb["bb_period"] == 20
    assert params_bb["bb_std"] == 2.0
    assert params_bb["squeeze_detection"] is True
    
    # ATR
    config_atr = StrategyConfigTemplates.get_volatility_breakout_config(
        breakout_type="atr"
    )
    params_atr = config_atr["parameters"]
    assert params_atr["atr_period"] == 14
    assert params_atr["atr_multiplier"] == 1.5
    assert params_atr["trailing_atr"] is True


def test_get_ensemble_config_defaults():
    """Test ensemble config with defaults."""
    config = StrategyConfigTemplates.get_ensemble_config()
    
    assert config["name"] == "ensemble_weighted_v1"
    assert config["strategy_id"] == "ensemble_weighted_001"
    assert config["strategy_type"] == StrategyType.CUSTOM.value
    
    params = config["parameters"]
    assert params["strategy_types"] == [
        "mean_reversion",
        "trend_following", 
        "momentum",
        "volatility_breakout"
    ]
    assert params["voting_method"] == "weighted"
    assert params["correlation_limit"] == 0.7


def test_get_ensemble_config_custom_strategies():
    """Test ensemble config with custom strategy types."""
    custom_strategies = ["mean_reversion", "trend_following"]
    config = StrategyConfigTemplates.get_ensemble_config(
        strategy_types=custom_strategies
    )
    
    params = config["parameters"]
    assert params["strategy_types"] == custom_strategies
    assert params["max_strategies"] == len(custom_strategies)


def test_get_ensemble_config_voting_methods():
    """Test ensemble config with different voting methods."""
    config_majority = StrategyConfigTemplates.get_ensemble_config(
        voting_method="majority"
    )
    assert config_majority["name"] == "ensemble_majority_v1"
    assert config_majority["parameters"]["voting_method"] == "majority"
    
    config_confidence = StrategyConfigTemplates.get_ensemble_config(
        voting_method="confidence"
    )
    assert config_confidence["name"] == "ensemble_confidence_v1"
    assert config_confidence["parameters"]["voting_method"] == "confidence"


def test_get_all_templates():
    """Test getting all available templates."""
    templates = StrategyConfigTemplates.get_all_templates()
    
    # Should have all template types
    assert "arbitrage_conservative" in templates
    assert "arbitrage_medium" in templates
    assert "arbitrage_aggressive" in templates
    assert "mean_reversion_5m" in templates
    assert "mean_reversion_1h" in templates
    assert "trend_following_5m" in templates
    assert "market_making_btc" in templates
    assert "volatility_breakout_low" in templates
    assert "ensemble_conservative" in templates
    
    # Check that each template is a valid config
    for name, config in templates.items():
        assert isinstance(config, dict)
        assert "name" in config
        assert "strategy_id" in config
        assert "strategy_type" in config


def test_get_template_by_name():
    """Test getting specific template by name."""
    template = StrategyConfigTemplates.get_template_by_name("arbitrage_medium")
    
    assert template["name"] == "arbitrage_scanner_v1"
    assert template["strategy_id"] == "arb_scanner_001"
    assert template["parameters"]["risk_per_trade"] == 0.02


def test_get_template_by_name_not_found():
    """Test getting non-existent template."""
    with pytest.raises(KeyError) as exc_info:
        StrategyConfigTemplates.get_template_by_name("non_existent")
    
    assert "Template 'non_existent' not found" in str(exc_info.value)
    assert "Available:" in str(exc_info.value)


def test_list_available_templates():
    """Test listing available template names."""
    template_names = StrategyConfigTemplates.list_available_templates()
    
    assert isinstance(template_names, list)
    assert len(template_names) > 0
    assert "arbitrage_medium" in template_names
    assert "mean_reversion_1h" in template_names
    assert "trend_following_4h" in template_names


def test_get_templates_by_strategy_type():
    """Test filtering templates by strategy type."""
    # Arbitrage templates
    arbitrage_templates = StrategyConfigTemplates.get_templates_by_strategy_type("arbitrage")
    assert len(arbitrage_templates) == 3
    assert "arbitrage_conservative" in arbitrage_templates
    assert "arbitrage_medium" in arbitrage_templates
    assert "arbitrage_aggressive" in arbitrage_templates
    
    # Mean reversion templates
    mean_rev_templates = StrategyConfigTemplates.get_templates_by_strategy_type("mean_reversion")
    assert len(mean_rev_templates) == 4
    assert all("mean_reversion" in name for name in mean_rev_templates.keys())
    
    # Market making templates
    mm_templates = StrategyConfigTemplates.get_templates_by_strategy_type("market_making")
    assert len(mm_templates) == 3
    assert all("market_making" in name for name in mm_templates.keys())


def test_validate_template_valid():
    """Test validating a valid template."""
    template = StrategyConfigTemplates.get_arbitrage_scanner_config()
    is_valid, errors = StrategyConfigTemplates.validate_template(template)
    
    assert is_valid is True
    assert len(errors) == 0


def test_validate_template_missing_fields():
    """Test validating template with missing required fields."""
    template = {"name": "test"}  # Missing many required fields
    
    is_valid, errors = StrategyConfigTemplates.validate_template(template)
    
    assert is_valid is False
    assert len(errors) > 0
    assert any("Missing required field: strategy_id" in error for error in errors)
    assert any("Missing required field: parameters" in error for error in errors)


def test_validate_template_invalid_numeric_ranges():
    """Test validating template with invalid numeric values."""
    template = StrategyConfigTemplates.get_arbitrage_scanner_config()
    template["position_size_pct"] = 0.8  # Too high (max is 0.5)
    template["min_confidence"] = 1.5  # Too high (max is 1.0)
    template["parameters"]["stop_loss_pct"] = 0.0001  # Too low (min is 0.005)
    
    is_valid, errors = StrategyConfigTemplates.validate_template(template)
    
    assert is_valid is False
    assert len(errors) >= 1
    assert any("stop_loss_pct" in error for error in errors)


def test_validate_template_invalid_backtesting():
    """Test validating template with invalid backtesting config."""
    template = StrategyConfigTemplates.get_arbitrage_scanner_config()
    del template["backtesting"]["start_date"]  # Remove required field
    
    is_valid, errors = StrategyConfigTemplates.validate_template(template)
    
    assert is_valid is False
    assert any("Backtesting missing required field: start_date" in error for error in errors)


def test_validate_template_disabled_backtesting():
    """Test validating template with disabled backtesting."""
    template = StrategyConfigTemplates.get_arbitrage_scanner_config()
    template["backtesting"]["enabled"] = False
    del template["backtesting"]["start_date"]  # Should not cause error when disabled
    
    is_valid, errors = StrategyConfigTemplates.validate_template(template)
    
    assert is_valid is True
    assert len(errors) == 0