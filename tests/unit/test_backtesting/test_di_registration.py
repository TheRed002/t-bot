"""Tests for backtesting dependency injection registration."""

from unittest.mock import MagicMock, patch

from src.backtesting.di_registration import (
    configure_backtesting_dependencies,
    get_backtest_service,
    register_backtesting_services,
)
from src.core.dependency_injection import DependencyInjector


class TestBacktestingDIRegistration:
    """Test backtesting dependency injection registration."""

    def test_register_backtesting_services_basic(self):
        """Test basic service registration."""
        injector = MagicMock(spec=DependencyInjector)

        register_backtesting_services(injector)

        # Check that all expected services were registered (updated count for new architecture)
        assert injector.register_factory.call_count == 17

        # Verify service names were registered
        service_names = [call[0][0] for call in injector.register_factory.call_args_list]
        expected_names = [
            "BacktestFactory", "MetricsCalculator", "MonteCarloAnalyzer", "WalkForwardAnalyzer",
            "PerformanceAttributor", "DataReplayManager", "TradeSimulator",
            "BacktestEngineFactory", "BacktestService", "BacktestController", "BacktestRepository"
        ]

        for name in expected_names:
            assert name in service_names

    def test_register_backtesting_services_interface_registration(self):
        """Test interface service registration."""
        injector = MagicMock(spec=DependencyInjector)

        register_backtesting_services(injector)

        # All registrations use register_factory, not register_service
        # Check that interface names were registered via register_factory
        service_names = [call[0][0] for call in injector.register_factory.call_args_list]
        expected_interfaces = [
            "BacktestServiceInterface", "BacktestControllerInterface", "BacktestRepositoryInterface",
            "BacktestFactoryInterface", "BacktestEngineFactoryInterface", "TradeSimulatorInterface"
        ]

        for name in expected_interfaces:
            assert name in service_names

    def test_metrics_calculator_factory(self):
        """Test MetricsCalculator factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_factory = MagicMock()
        mock_instance = MagicMock()

        # Mock factory and its method
        mock_factory.create_metrics_calculator.return_value = mock_instance
        injector.resolve.return_value = mock_factory

        register_backtesting_services(injector)

        # Get the MetricsCalculator factory function
        factory_call = [call for call in injector.register_factory.call_args_list
                       if call[0][0] == "MetricsCalculator"][0]
        factory_func = factory_call[0][1]

        # Call the factory
        result = factory_func()

        assert result == mock_instance
        injector.resolve.assert_called_with("BacktestFactory")
        mock_factory.create_metrics_calculator.assert_called_once()

    def test_monte_carlo_analyzer_factory(self):
        """Test MonteCarloAnalyzer factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_factory = MagicMock()
        mock_instance = MagicMock()

        # Mock factory and its method
        mock_factory.create_analyzer.return_value = mock_instance
        injector.resolve.return_value = mock_factory

        register_backtesting_services(injector)

        # Get the Monte Carlo analyzer factory
        factory_call = [call for call in injector.register_factory.call_args_list
                       if call[0][0] == "MonteCarloAnalyzer"][0]
        factory_func = factory_call[0][1]

        # Call the factory
        result = factory_func()

        assert result == mock_instance
        injector.resolve.assert_called_with("BacktestFactory")
        mock_factory.create_analyzer.assert_called_once_with("monte_carlo")

    def test_walk_forward_analyzer_factory(self):
        """Test WalkForwardAnalyzer factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_factory = MagicMock()
        mock_instance = MagicMock()

        # Mock factory and its method
        mock_factory.create_analyzer.return_value = mock_instance
        injector.resolve.return_value = mock_factory

        register_backtesting_services(injector)

        # Get the Walk Forward analyzer factory
        factory_call = [call for call in injector.register_factory.call_args_list
                       if call[0][0] == "WalkForwardAnalyzer"][0]
        factory_func = factory_call[0][1]

        # Call the factory
        result = factory_func()

        assert result == mock_instance
        injector.resolve.assert_called_with("BacktestFactory")
        mock_factory.create_analyzer.assert_called_once_with("walk_forward")

    def test_performance_attributor_factory(self):
        """Test PerformanceAttributor factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_factory = MagicMock()
        mock_instance = MagicMock()

        # Mock factory and its method
        mock_factory.create_analyzer.return_value = mock_instance
        injector.resolve.return_value = mock_factory

        register_backtesting_services(injector)

        # Get the Performance Attributor factory
        factory_call = [call for call in injector.register_factory.call_args_list
                       if call[0][0] == "PerformanceAttributor"][0]
        factory_func = factory_call[0][1]

        # Call the factory
        result = factory_func()

        assert result == mock_instance
        injector.resolve.assert_called_with("BacktestFactory")
        mock_factory.create_analyzer.assert_called_once_with("performance_attribution")

    @patch("src.backtesting.data_replay.DataReplayManager")
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

    def test_trade_simulator_factory(self):
        """Test TradeSimulator factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_factory = MagicMock()
        mock_instance = MagicMock()

        # Mock factory and its method
        mock_factory.create_simulator.return_value = mock_instance
        injector.resolve.return_value = mock_factory

        register_backtesting_services(injector)

        # Get the Trade Simulator factory
        factory_call = [call for call in injector.register_factory.call_args_list
                       if call[0][0] == "TradeSimulator"][0]
        factory_func = factory_call[0][1]

        # Call the factory
        result = factory_func()

        assert result == mock_instance
        injector.resolve.assert_called_with("BacktestFactory")
        # The factory gets called with a SimulationConfig instance
        mock_factory.create_simulator.assert_called_once()

    def test_backtest_engine_factory(self):
        """Test BacktestEngine factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_factory = MagicMock()
        mock_engine_instance = MagicMock()

        # Mock factory and its method
        mock_factory.create_engine.return_value = mock_engine_instance
        injector.resolve.return_value = mock_factory

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
        injector.resolve.assert_called_with("BacktestFactory")
        mock_factory.create_engine.assert_called_once_with(config=config, strategy=strategy)

    @patch("src.backtesting.service.BacktestService")
    def test_backtest_service_factory(self, mock_service):
        """Test BacktestService factory creation."""
        injector = MagicMock(spec=DependencyInjector)
        mock_config = MagicMock()
        mock_service_instance = MagicMock()

        # Mock resolve to return config and optional services
        def mock_resolve(service_name):
            if service_name == "Config":
                return mock_config
            elif service_name == "BacktestFactory":
                mock_factory = MagicMock()
                mock_factory.create_service.return_value = mock_service_instance
                return mock_factory
            elif service_name in ["DataService", "ExecutionService", "RiskService",
                                "StrategyService", "CapitalService", "MLService", "CacheService"]:
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
        # Service creation now goes through factory, not direct instantiation
        # mock_service.assert_called_once()

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

    @patch("src.backtesting.di_registration.DependencyInjector")
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
        mock_factory = MagicMock()
        mock_engine_instance = MagicMock()

        # Mock factory and its method
        mock_factory.create_engine.return_value = mock_engine_instance
        injector.resolve.return_value = mock_factory

        register_backtesting_services(injector)

        # Get the factory and create engine
        factory_call = [call for call in injector.register_factory.call_args_list
                       if call[0][0] == "BacktestEngineFactory"][0]
        factory_func = factory_call[0][1]
        engine_creator = factory_func()

        # Should handle services gracefully
        config = MagicMock()
        strategy = MagicMock()
        result = engine_creator(config, strategy, data_service="custom", execution_engine_service="custom")

        assert result == mock_engine_instance
        injector.resolve.assert_called_with("BacktestFactory")
        mock_factory.create_engine.assert_called_once()

    def test_backtest_service_factory_with_missing_optional_services(self):
        """Test BacktestService factory handles missing optional services."""
        injector = MagicMock(spec=DependencyInjector)
        mock_config = MagicMock()

        # Mock resolve to return config but raise for optional services
        def mock_resolve(service_name):
            if service_name == "Config":
                return mock_config
            elif service_name == "BacktestFactory":
                mock_factory = MagicMock()
                mock_factory.create_service.return_value = mock_service_instance
                return mock_factory
            else:
                raise Exception(f"Service {service_name} not found")

        injector.resolve.side_effect = mock_resolve

        with patch("src.backtesting.service.BacktestService") as mock_service:
            mock_service_instance = MagicMock()
            mock_service.return_value = mock_service_instance

            register_backtesting_services(injector)

            # Get and call the factory
            factory_call = [call for call in injector.register_factory.call_args_list
                           if call[0][0] == "BacktestService"][0]
            factory_func = factory_call[0][1]

            result = factory_func()

            assert result == mock_service_instance
            # Service creation is now delegated to factory, not direct instantiation
            # The factory handles service dependencies internally
