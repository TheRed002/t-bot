"""Tests for backtesting dependency injection registration."""

import pytest
from unittest.mock import MagicMock, patch, call

from src.backtesting.di_registration import (
    register_backtesting_services,
    configure_backtesting_dependencies,
    get_backtest_service
)
from src.core.dependency_injection import DependencyInjector


class TestBacktestingDIRegistration:
    """Test backtesting dependency injection registration."""

    def test_register_backtesting_services_basic(self):
        """Test basic service registration."""
        injector = MagicMock(spec=DependencyInjector)
        
        register_backtesting_services(injector)
        
        # Check that all expected services were registered
        assert injector.register_factory.call_count == 8
        
        # Verify service names were registered
        service_names = [call[0][0] for call in injector.register_factory.call_args_list]
        expected_names = [
            "MetricsCalculator", "MonteCarloAnalyzer", "WalkForwardAnalyzer",
            "PerformanceAttributor", "DataReplayManager", "TradeSimulator",
            "BacktestEngineFactory", "BacktestService"
        ]
        
        for name in expected_names:
            assert name in service_names

    def test_register_backtesting_services_interface_registration(self):
        """Test interface service registration."""
        injector = MagicMock(spec=DependencyInjector)
        
        register_backtesting_services(injector)
        
        # Verify register_service was called for interfaces
        assert injector.register_service.call_count == 2
        
        # Check interface names were registered
        interface_names = [call[0][0] for call in injector.register_service.call_args_list]
        expected_interfaces = ["BacktestServiceInterface", "BacktestEngineFactoryInterface"]
        
        for name in expected_interfaces:
            assert name in interface_names

    @patch('src.backtesting.metrics.MetricsCalculator')
    def test_metrics_calculator_factory(self, mock_metrics_calculator):
        """Test MetricsCalculator factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_instance = MagicMock()
        mock_metrics_calculator.return_value = mock_instance
        
        register_backtesting_services(injector)
        
        # Get the factory function
        factory_call = injector.register_factory.call_args_list[0]
        factory_func = factory_call[0][1]
        
        # Call the factory
        result = factory_func()
        
        assert result == mock_instance
        mock_metrics_calculator.assert_called_once()

    @patch('src.backtesting.analysis.MonteCarloAnalyzer')
    def test_monte_carlo_analyzer_factory(self, mock_analyzer):
        """Test MonteCarloAnalyzer factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_config = MagicMock()
        mock_engine_factory = MagicMock()
        mock_instance = MagicMock()
        
        injector.resolve.side_effect = lambda x: {
            "Config": mock_config,
            "BacktestEngineFactory": mock_engine_factory
        }[x]
        mock_analyzer.return_value = mock_instance
        
        register_backtesting_services(injector)
        
        # Get the Monte Carlo analyzer factory
        factory_call = [call for call in injector.register_factory.call_args_list 
                       if call[0][0] == "MonteCarloAnalyzer"][0]
        factory_func = factory_call[0][1]
        
        # Call the factory
        result = factory_func()
        
        assert result == mock_instance
        mock_analyzer.assert_called_once_with(config=mock_config, engine_factory=mock_engine_factory)

    @patch('src.backtesting.analysis.WalkForwardAnalyzer')
    def test_walk_forward_analyzer_factory(self, mock_analyzer):
        """Test WalkForwardAnalyzer factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_config = MagicMock()
        mock_engine_factory = MagicMock()
        mock_instance = MagicMock()
        
        injector.resolve.side_effect = lambda x: {
            "Config": mock_config,
            "BacktestEngineFactory": mock_engine_factory
        }[x]
        mock_analyzer.return_value = mock_instance
        
        register_backtesting_services(injector)
        
        # Get the Walk Forward analyzer factory
        factory_call = [call for call in injector.register_factory.call_args_list 
                       if call[0][0] == "WalkForwardAnalyzer"][0]
        factory_func = factory_call[0][1]
        
        # Call the factory
        result = factory_func()
        
        assert result == mock_instance
        mock_analyzer.assert_called_once_with(config=mock_config, engine_factory=mock_engine_factory)

    @patch('src.backtesting.attribution.PerformanceAttributor')
    def test_performance_attributor_factory(self, mock_attributor):
        """Test PerformanceAttributor factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_config = MagicMock()
        mock_instance = MagicMock()
        
        injector.resolve.return_value = mock_config
        mock_attributor.return_value = mock_instance
        
        register_backtesting_services(injector)
        
        # Get the Performance Attributor factory
        factory_call = [call for call in injector.register_factory.call_args_list 
                       if call[0][0] == "PerformanceAttributor"][0]
        factory_func = factory_call[0][1]
        
        # Call the factory
        result = factory_func()
        
        assert result == mock_instance
        mock_attributor.assert_called_once_with(config=mock_config)

    @patch('src.backtesting.data_replay.DataReplayManager')
    def test_data_replay_manager_factory(self, mock_manager):
        """Test DataReplayManager factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_config = MagicMock()
        mock_instance = MagicMock()
        
        injector.resolve.return_value = mock_config
        mock_manager.return_value = mock_instance
        
        register_backtesting_services(injector)
        
        # Get the Data Replay Manager factory
        factory_call = [call for call in injector.register_factory.call_args_list 
                       if call[0][0] == "DataReplayManager"][0]
        factory_func = factory_call[0][1]
        
        # Call the factory
        result = factory_func()
        
        assert result == mock_instance
        mock_manager.assert_called_once_with(config=mock_config)

    @patch('src.backtesting.simulator.TradeSimulator')
    @patch('src.backtesting.simulator.SimulationConfig')
    def test_trade_simulator_factory(self, mock_sim_config, mock_simulator):
        """Test TradeSimulator factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_config_instance = MagicMock()
        mock_simulator_instance = MagicMock()
        
        mock_sim_config.return_value = mock_config_instance
        mock_simulator.return_value = mock_simulator_instance
        
        register_backtesting_services(injector)
        
        # Get the Trade Simulator factory
        factory_call = [call for call in injector.register_factory.call_args_list 
                       if call[0][0] == "TradeSimulator"][0]
        factory_func = factory_call[0][1]
        
        # Call the factory
        result = factory_func()
        
        assert result == mock_simulator_instance
        mock_sim_config.assert_called_once()
        mock_simulator.assert_called_once_with(mock_config_instance)

    @patch('src.backtesting.engine.BacktestEngine')
    def test_backtest_engine_factory(self, mock_engine):
        """Test BacktestEngine factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_metrics_calculator = MagicMock()
        mock_data_service = MagicMock()
        mock_execution_service = MagicMock()
        mock_engine_instance = MagicMock()
        
        injector.resolve.side_effect = lambda x: {
            "MetricsCalculator": mock_metrics_calculator,
            "DataService": mock_data_service,
            "ExecutionService": mock_execution_service
        }.get(x, None)
        mock_engine.return_value = mock_engine_instance
        
        register_backtesting_services(injector)
        
        # Get the Backtest Engine factory
        factory_call = [call for call in injector.register_factory.call_args_list 
                       if call[0][0] == "BacktestEngineFactory"][0]
        factory_func = factory_call[0][1]
        
        # Call the outer factory to get the engine creator
        engine_creator = factory_func()
        
        # Call the engine creator
        config = MagicMock()
        strategy = MagicMock()
        result = engine_creator(config, strategy)
        
        assert result == mock_engine_instance
        mock_engine.assert_called_once()

    @patch('src.backtesting.service.BacktestService')
    def test_backtest_service_factory(self, mock_service):
        """Test BacktestService factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_config = MagicMock()
        mock_service_instance = MagicMock()
        
        # Mock resolve to return config and optional services
        def mock_resolve(service_name):
            if service_name == "Config":
                return mock_config
            elif service_name in ["DataService", "ExecutionService", "RiskService", 
                                "StrategyService", "CapitalService", "MLService"]:
                return MagicMock()
            else:
                raise Exception(f"Service {service_name} not found")
        
        injector.resolve.side_effect = mock_resolve
        mock_service.return_value = mock_service_instance
        
        register_backtesting_services(injector)
        
        # Get the Backtest Service factory
        factory_call = [call for call in injector.register_factory.call_args_list 
                       if call[0][0] == "BacktestService"][0]
        factory_func = factory_call[0][1]
        
        # Call the factory
        result = factory_func()
        
        assert result == mock_service_instance
        mock_service.assert_called_once()

    def test_backtest_service_factory_without_injector(self):
        """Test BacktestService factory validation logic."""
        injector = MagicMock(spec=DependencyInjector)
        
        register_backtesting_services(injector)
        
        # Get the Backtest Service factory
        factory_call = [call for call in injector.register_factory.call_args_list 
                       if call[0][0] == "BacktestService"][0]
        factory_func = factory_call[0][1]
        
        # The factory function exists and was registered
        assert factory_func is not None
        assert callable(factory_func)

    def test_configure_backtesting_dependencies_with_injector(self):
        """Test configure dependencies with existing injector."""
        existing_injector = MagicMock(spec=DependencyInjector)
        
        result = configure_backtesting_dependencies(existing_injector)
        
        assert result == existing_injector
        # Should have called register methods
        assert existing_injector.register_factory.called

    @patch('src.backtesting.di_registration.DependencyInjector')
    def test_configure_backtesting_dependencies_without_injector(self, mock_injector_class):
        """Test configure dependencies without existing injector."""
        mock_injector = MagicMock(spec=DependencyInjector)
        mock_injector_class.return_value = mock_injector
        
        result = configure_backtesting_dependencies()
        
        assert result == mock_injector
        mock_injector_class.assert_called_once()

    def test_get_backtest_service(self):
        """Test get_backtest_service service locator."""
        injector = MagicMock(spec=DependencyInjector)
        mock_service = MagicMock()
        injector.resolve.return_value = mock_service
        
        result = get_backtest_service(injector)
        
        assert result == mock_service
        injector.resolve.assert_called_once_with("BacktestService")

    def test_backtest_engine_factory_with_service_resolution_errors(self):
        """Test BacktestEngine factory handles service resolution errors."""
        injector = MagicMock(spec=DependencyInjector)
        mock_metrics_calculator = MagicMock()
        
        # Mock resolve to raise exception for optional services
        def mock_resolve(service_name):
            if service_name == "MetricsCalculator":
                return mock_metrics_calculator
            else:
                raise Exception(f"Service {service_name} not found")
        
        injector.resolve.side_effect = mock_resolve
        
        with patch('src.backtesting.engine.BacktestEngine') as mock_engine:
            mock_engine_instance = MagicMock()
            mock_engine.return_value = mock_engine_instance
            
            register_backtesting_services(injector)
            
            # Get the factory and create engine
            factory_call = [call for call in injector.register_factory.call_args_list 
                           if call[0][0] == "BacktestEngineFactory"][0]
            factory_func = factory_call[0][1]
            engine_creator = factory_func()
            
            # Should handle missing services gracefully
            config = MagicMock()
            strategy = MagicMock()
            result = engine_creator(config, strategy, data_service="custom", execution_engine_service="custom")
            
            assert result == mock_engine_instance
            mock_engine.assert_called_once()

    def test_backtest_service_factory_with_missing_optional_services(self):
        """Test BacktestService factory handles missing optional services."""
        injector = MagicMock(spec=DependencyInjector)
        mock_config = MagicMock()
        
        # Mock resolve to return config but raise for optional services
        def mock_resolve(service_name):
            if service_name == "Config":
                return mock_config
            else:
                raise Exception(f"Service {service_name} not found")
        
        injector.resolve.side_effect = mock_resolve
        
        with patch('src.backtesting.service.BacktestService') as mock_service:
            mock_service_instance = MagicMock()
            mock_service.return_value = mock_service_instance
            
            register_backtesting_services(injector)
            
            # Get and call the factory
            factory_call = [call for call in injector.register_factory.call_args_list 
                           if call[0][0] == "BacktestService"][0]
            factory_func = factory_call[0][1]
            
            result = factory_func()
            
            assert result == mock_service_instance
            # Should have been called with None for missing services
            call_kwargs = mock_service.call_args[1]
            assert call_kwargs["config"] == mock_config
            assert call_kwargs["injector"] == injector
            # Optional services should be None
            for service in ["DataService", "ExecutionService", "RiskService", 
                           "StrategyService", "CapitalService", "MLService"]:
                assert call_kwargs[service] is None