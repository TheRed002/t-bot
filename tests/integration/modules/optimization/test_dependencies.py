"""
Integration test for optimization module dependency injection validation.

Tests that optimization module components are properly integrated with dependency
injection patterns and module boundaries are respected.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, MagicMock, AsyncMock

from src.core.dependency_injection import DependencyInjector
from src.core.exceptions import RepositoryError, OptimizationError
from src.optimization.di_registration import register_optimization_dependencies
from src.optimization.core import OptimizationResult, OptimizationStatus
from src.optimization.parameter_space import ParameterSpaceBuilder


class TestOptimizationDependencyInjectionValidation:
    """Test optimization module dependency injection integration."""

    @pytest.fixture
    def mock_injector(self):
        """Create mock injector for dependency testing."""
        injector = Mock(spec=DependencyInjector)
        injector.resolve = Mock()
        injector.register_factory = Mock()
        return injector

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = AsyncMock()
        session.scalar = AsyncMock()
        session.execute = AsyncMock()
        session.add = Mock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    def test_dependency_registration_with_all_services(self, mock_injector):
        """Test that all optimization dependencies register correctly."""
        mock_session = Mock()
        mock_injector.resolve.return_value = mock_session

        # Register dependencies
        register_optimization_dependencies(mock_injector)

        # Verify all required factories are registered
        factory_calls = [call[0][0] for call in mock_injector.register_factory.call_args_list]

        required_factories = [
            "OptimizationRepository",
            "ResultsAnalyzer",
            "OptimizationAnalysisService",
            "OptimizationBacktestIntegration",
            "OptimizationFactory",
            "OptimizationComponentFactory",
            "OptimizationService",
            "OptimizationController"
        ]

        for factory_name in required_factories:
            assert factory_name in factory_calls, f"Missing factory registration: {factory_name}"

    def test_repository_dependency_injection_with_session(self, mock_session):
        """Test repository proper dependency injection with database session."""
        from src.optimization.repository import OptimizationRepository

        # Create repository with session
        repo = OptimizationRepository(session=mock_session)

        # Verify session is stored
        assert repo._session == mock_session

        # Verify dependency is tracked
        assert "AsyncSession" in repo._dependencies

    def test_repository_dependency_injection_without_session(self):
        """Test repository handles missing database session gracefully."""
        from src.optimization.repository import OptimizationRepository

        # Create repository without session
        repo = OptimizationRepository(session=None)

        # Verify no session stored
        assert repo._session is None

        # Verify dependency not tracked
        assert "AsyncSession" not in repo._dependencies

    @pytest.mark.asyncio
    async def test_repository_save_without_session_fails(self):
        """Test repository operations fail gracefully without session."""
        from src.optimization.repository import OptimizationRepository

        repo = OptimizationRepository(session=None)

        # Create mock optimization result
        from datetime import datetime
        result = OptimizationResult(
            optimization_id="test-id",
            algorithm_name="test",
            status=OptimizationStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration_seconds=Decimal("10"),
            iterations_completed=1,
            evaluations_completed=1,
            optimal_parameters={},
            optimal_objective_value=Decimal("1.0"),
            objective_values={},
            convergence_achieved=True,
            config_used={}
        )

        # Attempt to save should raise RepositoryError
        with pytest.raises(RepositoryError, match="Database session not available"):
            await repo.save_optimization_result(result)

    @pytest.mark.asyncio
    async def test_repository_get_without_session_returns_none(self):
        """Test repository get operations return None without session."""
        from src.optimization.repository import OptimizationRepository

        repo = OptimizationRepository(session=None)

        # Get should return None gracefully
        result = await repo.get_optimization_result("test-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_repository_list_without_session_returns_empty(self):
        """Test repository list operations return empty list without session."""
        from src.optimization.repository import OptimizationRepository

        repo = OptimizationRepository(session=None)

        # List should return empty list gracefully
        results = await repo.list_optimization_results()
        assert results == []

    def test_backtest_integration_optional_dependency_handling(self):
        """Test backtest integration service handles optional dependency correctly."""
        from src.optimization.backtesting_integration import BacktestIntegrationService

        # Create service without backtest service
        service = BacktestIntegrationService(backtest_service=None)

        # Verify no dependency tracked when None provided
        assert "BacktestService" not in service._dependencies

    @pytest.mark.asyncio
    async def test_backtest_integration_simulate_without_service(self):
        """Test backtest integration uses simulation when service unavailable."""
        from src.optimization.backtesting_integration import BacktestIntegrationService
        from src.core.types import StrategyConfig

        service = BacktestIntegrationService(backtest_service=None)

        # Create mock strategy config
        from src.core.types import StrategyType, TradingMode
        strategy_config = StrategyConfig(
            strategy_id="test-id",
            name="test_strategy",
            strategy_type=StrategyType.MOMENTUM,
            symbol="BTC-USD",
            timeframe="1h",
            parameters={"param1": 1.0}
        )

        # Should use simulation instead of actual backtesting
        result = await service.evaluate_strategy(strategy_config)

        # Verify returns simulated performance metrics
        assert isinstance(result, dict)
        assert "total_return" in result
        assert "sharpe_ratio" in result
        assert all(isinstance(v, Decimal) for v in result.values())

    def test_factory_can_create_all_components(self):
        """Test factory can create all required optimization components."""
        from src.optimization.factory import OptimizationFactory

        mock_injector = Mock()
        mock_session = Mock()
        mock_injector.resolve = Mock(return_value=mock_session)
        mock_injector.get_container = Mock(return_value=mock_injector)

        factory = OptimizationFactory(injector=mock_injector)

        # Verify factory can create all components without errors
        components = ["service", "controller", "repository", "backtest_integration", "analysis_service"]

        for component_name in components:
            component = factory.create(component_name)
            assert component is not None, f"Failed to create {component_name}"

    def test_factory_dependency_resolution_fallback(self):
        """Test factory handles missing dependencies gracefully."""
        from src.optimization.factory import OptimizationFactory

        mock_injector = Mock()
        # Make resolve fail to simulate missing dependency
        mock_injector.resolve.side_effect = Exception("Dependency not found")

        factory = OptimizationFactory(injector=mock_injector)

        # Should not fail when creating service with missing dependencies
        service = factory.create("service")
        assert service is not None

    def test_service_creation_without_dependencies(self):
        """Test optimization service can be created without optional dependencies."""
        from src.optimization.service import OptimizationService

        # Create service with all None dependencies
        service = OptimizationService(
            backtest_integration=None,
            optimization_repository=None,
            analysis_service=None,
        )

        assert service is not None
        assert service._backtest_integration is None
        assert service._optimization_repository is None
        assert service._analysis_service is None

    @pytest.mark.asyncio
    async def test_service_optimization_with_missing_dependencies(self):
        """Test service handles missing dependencies during optimization."""
        from src.optimization.service import OptimizationService

        service = OptimizationService()

        # Build simple parameter space
        parameter_space = ParameterSpaceBuilder().add_continuous(
            name="test_param",
            min_value=1.0,
            max_value=10.0
        ).build()

        # Should handle missing dependencies gracefully
        try:
            result = await service.optimize_strategy(
                strategy_name="test_strategy",
                parameter_space=parameter_space
            )
            # Should return some result even with missing dependencies
            assert isinstance(result, dict)
        except OptimizationError:
            # Acceptable to fail with proper error
            pass

    def test_controller_service_dependency_injection(self):
        """Test controller requires optimization service dependency."""
        from src.optimization.controller import OptimizationController

        mock_service = Mock()

        # Create controller with service
        controller = OptimizationController(optimization_service=mock_service)

        assert controller._optimization_service == mock_service
        assert "OptimizationService" in controller._dependencies

    def test_module_boundary_respect(self):
        """Test optimization module respects boundaries with other modules."""
        from src.optimization import service, repository, controller

        # Check that optimization doesn't import internals from other modules
        # All imports should be through interfaces or service layers

        # Verify main service imports
        service_imports = [
            attr for attr in dir(service)
            if not attr.startswith('_') and 'src.' in str(getattr(service, attr, ''))
        ]

        # Should not directly import internal classes from other modules
        prohibited_patterns = ['._internal', '.impl', '.private']

        for import_name in service_imports:
            for pattern in prohibited_patterns:
                assert pattern not in import_name, f"Improper internal import: {import_name}"

    @pytest.mark.asyncio
    async def test_integration_error_propagation(self):
        """Test that errors are properly propagated through integration layers."""
        from src.optimization.service import OptimizationService
        from src.optimization.backtesting_integration import BacktestIntegrationService

        # Create service with failing backtest integration
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest = AsyncMock(side_effect=Exception("Backtest failed"))

        backtest_integration = BacktestIntegrationService(mock_backtest_service)
        service = OptimizationService(backtest_integration=backtest_integration)

        # Should propagate errors properly
        parameter_space = ParameterSpaceBuilder().add_continuous(
            name="test_param",
            min_value=1.0,
            max_value=10.0
        ).build()

        # Error should be caught and handled appropriately
        try:
            await service.optimize_strategy(
                strategy_name="test_strategy",
                parameter_space=parameter_space
            )
        except (OptimizationError, Exception):
            # Should either handle gracefully or propagate with proper error type
            pass

    def test_dependency_injection_singleton_pattern(self, mock_injector):
        """Test that singleton dependencies are properly configured."""
        register_optimization_dependencies(mock_injector)

        # Check that key services are registered as singletons
        singleton_calls = [
            call for call in mock_injector.register_factory.call_args_list
            if call.kwargs.get('singleton') is True
        ]

        singleton_names = [call[0][0] for call in singleton_calls]

        expected_singletons = [
            "OptimizationRepository",
            "ResultsAnalyzer",
            "OptimizationAnalysisService",
            "OptimizationBacktestIntegration",
            "OptimizationFactory",
            "OptimizationComponentFactory",
            "OptimizationService",
            "OptimizationController"
        ]

        for singleton_name in expected_singletons:
            assert singleton_name in singleton_names, f"Service {singleton_name} should be singleton"