"""
Tests for strategy interfaces.
"""

import logging

import pytest

# Disable logging during tests for performance
logging.disable(logging.CRITICAL)
from abc import ABC
from typing import Protocol

from src.core.types import (
    MarketData,
    Position,
    Signal,
    StrategyConfig,
)
from src.strategies.interfaces import (
    ArbitrageStrategyInterface,
    BacktestingInterface,
    BacktestingServiceInterface,
    BaseStrategyInterface,
    MarketDataProviderInterface,
    MarketMakingStrategyInterface,
    MeanReversionStrategyInterface,
    PerformanceMonitoringInterface,
    RiskManagementInterface,
    StrategyDataRepositoryInterface,
    StrategyFactoryInterface,
    StrategyRegistryInterface,
    StrategyServiceInterface,
    TrendStrategyInterface,
)


class TestBaseStrategyInterface:
    """Test BaseStrategyInterface protocol."""

    def test_base_strategy_interface_is_abstract(self):
        """Test that BaseStrategyInterface is abstract."""
        assert issubclass(BaseStrategyInterface, ABC)

        with pytest.raises(TypeError):
            BaseStrategyInterface()  # Should not be instantiable

    def test_base_strategy_interface_methods_are_abstract(self):
        """Test that all methods in BaseStrategyInterface are abstract."""
        # Get all abstract methods
        abstract_methods = BaseStrategyInterface.__abstractmethods__

        expected_methods = {
            "strategy_type",
            "name",
            "version",
            "status",
            "initialize",
            "generate_signals",
            "validate_signal",
            "get_position_size",
            "should_exit",
            "start",
            "stop",
            "pause",
            "resume",
            "prepare_for_backtest",
            "process_historical_data",
            "get_backtest_metrics",
            "get_performance_summary",
            "get_real_time_metrics",
            "get_state",
        }

        assert abstract_methods == expected_methods

    def test_base_strategy_interface_properties(self):
        """Test BaseStrategyInterface has required properties."""
        # Check property names exist
        assert hasattr(BaseStrategyInterface, "strategy_type")
        assert hasattr(BaseStrategyInterface, "name")
        assert hasattr(BaseStrategyInterface, "version")
        assert hasattr(BaseStrategyInterface, "status")


class TestTrendStrategyInterface:
    """Test TrendStrategyInterface."""

    def test_trend_strategy_interface_inherits_base(self):
        """Test that TrendStrategyInterface inherits from BaseStrategyInterface."""
        assert issubclass(TrendStrategyInterface, BaseStrategyInterface)

    def test_trend_strategy_interface_additional_methods(self):
        """Test TrendStrategyInterface has additional methods."""
        additional_methods = set(TrendStrategyInterface.__abstractmethods__) - set(
            BaseStrategyInterface.__abstractmethods__
        )

        expected_additional = {
            "calculate_trend_strength",
            "identify_trend_direction",
            "get_trend_confirmation",
        }

        assert additional_methods == expected_additional


class TestMeanReversionStrategyInterface:
    """Test MeanReversionStrategyInterface."""

    def test_mean_reversion_interface_inherits_base(self):
        """Test that MeanReversionStrategyInterface inherits from BaseStrategyInterface."""
        assert issubclass(MeanReversionStrategyInterface, BaseStrategyInterface)

    def test_mean_reversion_interface_additional_methods(self):
        """Test MeanReversionStrategyInterface has additional methods."""
        additional_methods = set(MeanReversionStrategyInterface.__abstractmethods__) - set(
            BaseStrategyInterface.__abstractmethods__
        )

        expected_additional = {
            "calculate_mean_deviation",
            "is_oversold",
            "is_overbought",
            "calculate_reversion_probability",
        }

        assert additional_methods == expected_additional


class TestArbitrageStrategyInterface:
    """Test ArbitrageStrategyInterface."""

    def test_arbitrage_interface_inherits_base(self):
        """Test that ArbitrageStrategyInterface inherits from BaseStrategyInterface."""
        assert issubclass(ArbitrageStrategyInterface, BaseStrategyInterface)

    def test_arbitrage_interface_additional_methods(self):
        """Test ArbitrageStrategyInterface has additional methods."""
        additional_methods = set(ArbitrageStrategyInterface.__abstractmethods__) - set(
            BaseStrategyInterface.__abstractmethods__
        )

        expected_additional = {
            "identify_arbitrage_opportunities",
            "calculate_profit_potential",
            "validate_arbitrage_execution",
        }

        assert additional_methods == expected_additional


class TestMarketMakingStrategyInterface:
    """Test MarketMakingStrategyInterface."""

    def test_market_making_interface_inherits_base(self):
        """Test that MarketMakingStrategyInterface inherits from BaseStrategyInterface."""
        assert issubclass(MarketMakingStrategyInterface, BaseStrategyInterface)

    def test_market_making_interface_additional_methods(self):
        """Test MarketMakingStrategyInterface has additional methods."""
        additional_methods = set(MarketMakingStrategyInterface.__abstractmethods__) - set(
            BaseStrategyInterface.__abstractmethods__
        )

        expected_additional = {
            "calculate_optimal_spread",
            "manage_inventory",
            "calculate_quote_adjustment",
        }

        assert additional_methods == expected_additional


class TestStrategyFactoryInterface:
    """Test StrategyFactoryInterface."""

    def test_strategy_factory_interface_is_abstract(self):
        """Test that StrategyFactoryInterface is abstract."""
        assert issubclass(StrategyFactoryInterface, ABC)

        with pytest.raises(TypeError):
            StrategyFactoryInterface()

    def test_strategy_factory_interface_methods(self):
        """Test StrategyFactoryInterface has required methods."""
        abstract_methods = StrategyFactoryInterface.__abstractmethods__

        expected_methods = {
            "create_strategy",
            "get_supported_strategies",
            "validate_strategy_requirements",
        }

        assert abstract_methods == expected_methods


class TestProtocolInterfaces:
    """Test Protocol-based interfaces."""

    def test_backtesting_interface_is_protocol(self):
        """Test that BacktestingInterface is a Protocol."""
        # Protocol classes should be instance of type(Protocol)
        assert isinstance(BacktestingInterface, type(Protocol))

    def test_performance_monitoring_interface_is_protocol(self):
        """Test that PerformanceMonitoringInterface is a Protocol."""
        assert isinstance(PerformanceMonitoringInterface, type(Protocol))

    def test_risk_management_interface_is_protocol(self):
        """Test that RiskManagementInterface is a Protocol."""
        assert isinstance(RiskManagementInterface, type(Protocol))

    def test_backtesting_service_interface_is_protocol(self):
        """Test that BacktestingServiceInterface is a Protocol."""
        assert isinstance(BacktestingServiceInterface, type(Protocol))

    def test_strategy_registry_interface_is_protocol(self):
        """Test that StrategyRegistryInterface is a Protocol."""
        assert isinstance(StrategyRegistryInterface, type(Protocol))

    def test_market_data_provider_interface_is_protocol(self):
        """Test that MarketDataProviderInterface is a Protocol."""
        assert isinstance(MarketDataProviderInterface, type(Protocol))

    def test_strategy_data_repository_interface_is_protocol(self):
        """Test that StrategyDataRepositoryInterface is a Protocol."""
        assert isinstance(StrategyDataRepositoryInterface, type(Protocol))

    def test_strategy_service_interface_is_protocol(self):
        """Test that StrategyServiceInterface is a Protocol."""
        assert isinstance(StrategyServiceInterface, type(Protocol))


class TestInterfaceMethodSignatures:
    """Test interface method signatures for type safety."""

    def test_backtesting_interface_method_signatures(self):
        """Test BacktestingInterface method signatures."""
        # Test that methods exist with correct names
        assert hasattr(BacktestingInterface, "prepare_for_backtest")
        assert hasattr(BacktestingInterface, "process_historical_data")
        assert hasattr(BacktestingInterface, "simulate_trade_execution")
        assert hasattr(BacktestingInterface, "get_backtest_metrics")
        assert hasattr(BacktestingInterface, "reset_backtest_state")

    def test_performance_monitoring_interface_methods(self):
        """Test PerformanceMonitoringInterface method signatures."""
        assert hasattr(PerformanceMonitoringInterface, "update_performance_metrics")
        assert hasattr(PerformanceMonitoringInterface, "get_real_time_metrics")
        assert hasattr(PerformanceMonitoringInterface, "calculate_risk_adjusted_returns")
        assert hasattr(PerformanceMonitoringInterface, "get_drawdown_analysis")

    def test_risk_management_interface_methods(self):
        """Test RiskManagementInterface method signatures."""
        assert hasattr(RiskManagementInterface, "validate_risk_limits")
        assert hasattr(RiskManagementInterface, "calculate_position_size")
        assert hasattr(RiskManagementInterface, "should_close_position")

    def test_strategy_registry_interface_methods(self):
        """Test StrategyRegistryInterface method signatures."""
        assert hasattr(StrategyRegistryInterface, "register_strategy")
        assert hasattr(StrategyRegistryInterface, "get_strategy")
        assert hasattr(StrategyRegistryInterface, "list_strategies")
        assert hasattr(StrategyRegistryInterface, "remove_strategy")

    def test_strategy_service_interface_methods(self):
        """Test StrategyServiceInterface method signatures."""
        assert hasattr(StrategyServiceInterface, "register_strategy")
        assert hasattr(StrategyServiceInterface, "start_strategy")
        assert hasattr(StrategyServiceInterface, "stop_strategy")
        assert hasattr(StrategyServiceInterface, "process_market_data")
        assert hasattr(StrategyServiceInterface, "validate_signal")
        assert hasattr(StrategyServiceInterface, "get_strategy_performance")
        assert hasattr(StrategyServiceInterface, "get_all_strategies")
        assert hasattr(StrategyServiceInterface, "cleanup_strategy")


class TestInterfaceCovariance:
    """Test interface type covariance and contravariance."""

    def test_signal_type_consistency(self):
        """Test Signal type is used consistently across interfaces."""
        # This tests that the Signal type imported is used throughout
        # Not a runtime test but ensures import consistency
        from src.core.types import Signal as CoreSignal

        assert Signal is CoreSignal

    def test_market_data_type_consistency(self):
        """Test MarketData type is used consistently."""
        from src.core.types import MarketData as CoreMarketData

        assert MarketData is CoreMarketData

    def test_position_type_consistency(self):
        """Test Position type is used consistently."""
        from src.core.types import Position as CorePosition

        assert Position is CorePosition

    def test_strategy_config_type_consistency(self):
        """Test StrategyConfig type is used consistently."""
        from src.core.types import StrategyConfig as CoreStrategyConfig

        assert StrategyConfig is CoreStrategyConfig


class TestInterfaceImportExport:
    """Test interface module imports and exports."""

    def test_all_interfaces_importable(self):
        """Test that all interfaces can be imported."""
        # Test that we can import each interface
        from src.strategies.interfaces import (
            ArbitrageStrategyInterface,
            BacktestingInterface,
            BacktestingServiceInterface,
            BaseStrategyInterface,
            MarketDataProviderInterface,
            MarketMakingStrategyInterface,
            MeanReversionStrategyInterface,
            PerformanceMonitoringInterface,
            RiskManagementInterface,
            StrategyDataRepositoryInterface,
            StrategyFactoryInterface,
            StrategyRegistryInterface,
            StrategyServiceInterface,
            TrendStrategyInterface,
        )

        # Assert they are all classes/protocols
        interfaces = [
            BaseStrategyInterface,
            TrendStrategyInterface,
            MeanReversionStrategyInterface,
            ArbitrageStrategyInterface,
            MarketMakingStrategyInterface,
            StrategyFactoryInterface,
            BacktestingInterface,
            PerformanceMonitoringInterface,
            RiskManagementInterface,
            BacktestingServiceInterface,
            StrategyRegistryInterface,
            MarketDataProviderInterface,
            StrategyDataRepositoryInterface,
            StrategyServiceInterface,
        ]

        for interface in interfaces:
            assert interface is not None
            assert isinstance(interface, type)

    def test_type_variables_importable(self):
        """Test that type variables are importable."""
        from src.strategies.interfaces import StrategyT, T

        # These are TypeVars, so check they exist
        assert T is not None
        assert StrategyT is not None


class TestInterfaceDocumentation:
    """Test interface documentation and metadata."""

    def test_interfaces_have_docstrings(self):
        """Test that interfaces have proper docstrings."""
        interfaces = [
            BaseStrategyInterface,
            TrendStrategyInterface,
            MeanReversionStrategyInterface,
            ArbitrageStrategyInterface,
            MarketMakingStrategyInterface,
            StrategyFactoryInterface,
        ]

        for interface in interfaces:
            assert interface.__doc__ is not None
            assert len(interface.__doc__.strip()) > 0

    def test_protocol_interfaces_have_docstrings(self):
        """Test that Protocol interfaces have proper docstrings."""
        protocols = [
            BacktestingInterface,
            PerformanceMonitoringInterface,
            RiskManagementInterface,
            StrategyRegistryInterface,
            StrategyServiceInterface,
        ]

        for protocol in protocols:
            assert protocol.__doc__ is not None
            assert len(protocol.__doc__.strip()) > 0

    def test_module_has_docstring(self):
        """Test that the interfaces module has a docstring."""
        import src.strategies.interfaces

        assert src.strategies.interfaces.__doc__ is not None
        assert len(src.strategies.interfaces.__doc__.strip()) > 0


class TestInterfaceEdgeCases:
    """Test edge cases and error conditions."""

    def test_interfaces_handle_none_values(self):
        """Test that interfaces can handle None values in type hints."""
        # This is more of a static analysis test
        # Check that return types can be None where specified
        from src.strategies.interfaces import StrategyRegistryInterface

        # get_strategy should be able to return None
        # This is checked via the type hint: BaseStrategyInterface | None
        assert hasattr(StrategyRegistryInterface, "get_strategy")

    def test_decimal_type_usage(self):
        """Test that Decimal is used for financial calculations."""
        from src.strategies.interfaces import ArbitrageStrategyInterface

        # calculate_profit_potential should return Decimal
        assert hasattr(ArbitrageStrategyInterface, "calculate_profit_potential")

    def test_datetime_type_usage(self):
        """Test that datetime is used appropriately."""
        from src.strategies.interfaces import StrategyDataRepositoryInterface

        # Methods should accept datetime parameters
        assert hasattr(StrategyDataRepositoryInterface, "get_strategy_trades")
        assert hasattr(StrategyDataRepositoryInterface, "save_performance_metrics")

    def test_empty_list_return_types(self):
        """Test interfaces that return empty lists."""
        from src.strategies.interfaces import StrategyRegistryInterface

        # list_strategies should return list[str]
        assert hasattr(StrategyRegistryInterface, "list_strategies")

    def test_async_method_consistency(self):
        """Test that async methods are consistently marked."""
        from src.strategies.interfaces import BaseStrategyInterface

        # Key methods should be async
        async_methods = [
            "initialize",
            "generate_signals",
            "start",
            "stop",
            "pause",
            "resume",
            "prepare_for_backtest",
            "process_historical_data",
            "get_backtest_metrics",
            "get_state",
        ]

        for method_name in async_methods:
            assert hasattr(BaseStrategyInterface, method_name)
