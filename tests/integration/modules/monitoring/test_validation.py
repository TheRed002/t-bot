"""
Integration tests for monitoring module boundaries and proper dependency injection.

Tests verify that monitoring services integrate correctly with other modules
and that dependency injection patterns are properly implemented.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.dependency_injection import DependencyInjector
from src.monitoring.di_registration import register_monitoring_services
from src.monitoring.services import (
    AlertRequest,
    DefaultAlertService,
    DefaultMetricsService, 
    DefaultPerformanceService,
    MetricRequest,
    MonitoringService,
)


class TestMonitoringDependencyInjection:
    """Test monitoring dependency injection integration."""

    @pytest.fixture
    def injector(self):
        """Create dependency injector for testing."""
        return DependencyInjector()

    @pytest.fixture
    def mock_alert_manager(self):
        """Mock alert manager."""
        mock = MagicMock()
        mock.fire_alert = AsyncMock()
        mock.resolve_alert = AsyncMock()
        mock.acknowledge_alert = AsyncMock(return_value=True)
        mock.get_active_alerts = MagicMock(return_value=[])
        mock.get_alert_stats = MagicMock(return_value={})
        mock.add_rule = MagicMock()
        mock.add_escalation_policy = MagicMock()
        return mock

    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock metrics collector."""
        mock = MagicMock()
        mock.record_counter = MagicMock()
        mock.record_gauge = MagicMock()
        mock.record_histogram = MagicMock()
        return mock

    @pytest.fixture
    def mock_performance_profiler(self):
        """Mock performance profiler."""
        mock = MagicMock()
        mock.get_performance_summary = MagicMock(return_value={})
        mock.record_execution_time = MagicMock()
        mock.record_order_execution = MagicMock()
        mock.record_market_data_processing = MagicMock()
        mock.get_latency_stats = MagicMock(return_value={})
        mock.get_system_resource_stats = MagicMock(return_value={})
        return mock

    @pytest.mark.asyncio
    async def test_monitoring_services_registration(self, injector, mock_alert_manager, mock_metrics_collector, mock_performance_profiler):
        """Test that monitoring services register correctly with dependency injection."""
        # Mock the components
        with patch('src.monitoring.di_registration.AlertManager', return_value=mock_alert_manager), \
             patch('src.monitoring.di_registration.MetricsCollector', return_value=mock_metrics_collector), \
             patch('src.monitoring.di_registration.PerformanceProfiler', return_value=mock_performance_profiler):
            
            # Register services
            register_monitoring_services(injector)
            
            # Verify all core components are registered
            assert injector.resolve("MetricsCollector") is not None
            assert injector.resolve("AlertManager") is not None
            assert injector.resolve("PerformanceProfiler") is not None
            
            # Verify service implementations are registered
            assert injector.resolve("DefaultMetricsService") is not None
            assert injector.resolve("DefaultAlertService") is not None
            assert injector.resolve("DefaultPerformanceService") is not None
            
            # Verify service interfaces are bound
            assert injector.resolve("MetricsServiceInterface") is not None
            assert injector.resolve("AlertServiceInterface") is not None
            assert injector.resolve("PerformanceServiceInterface") is not None
            
            # Verify composite service is registered
            assert injector.resolve("MonitoringService") is not None
            assert injector.resolve("MonitoringServiceInterface") is not None

    @pytest.mark.asyncio
    async def test_default_alert_service_dependency_injection(self, mock_alert_manager):
        """Test DefaultAlertService constructor accepts correct dependency injection parameters."""
        # Test with valid parameters only
        service = DefaultAlertService(alert_manager=mock_alert_manager)

        assert service._alert_manager is mock_alert_manager
        assert hasattr(service, '_data_transformer')

        # Test with minimal parameters
        service_minimal = DefaultAlertService(alert_manager=mock_alert_manager)
        assert service_minimal._alert_manager is mock_alert_manager

    @pytest.mark.asyncio
    async def test_default_metrics_service_dependency_injection(self, mock_metrics_collector):
        """Test DefaultMetricsService constructor accepts correct dependency injection parameters."""
        # Test with valid parameters only
        service = DefaultMetricsService(metrics_collector=mock_metrics_collector)

        assert service._metrics_collector is mock_metrics_collector

        # Test with minimal parameters
        service_minimal = DefaultMetricsService(metrics_collector=mock_metrics_collector)
        assert service_minimal._metrics_collector is mock_metrics_collector

    @pytest.mark.asyncio
    async def test_default_performance_service_dependency_injection(self, mock_performance_profiler):
        """Test DefaultPerformanceService constructor accepts correct dependency injection parameters."""
        # Test with valid parameters only
        service = DefaultPerformanceService(performance_profiler=mock_performance_profiler)

        assert service._performance_profiler is mock_performance_profiler

        # Test with minimal parameters
        service_minimal = DefaultPerformanceService(performance_profiler=mock_performance_profiler)
        assert service_minimal._performance_profiler is mock_performance_profiler

    @pytest.mark.asyncio
    async def test_monitoring_service_composition(self):
        """Test that MonitoringService properly composes service interfaces."""
        mock_alert_service = MagicMock()
        mock_alert_service.create_alert = AsyncMock()
        
        mock_metrics_service = MagicMock()
        mock_metrics_service.record_counter = MagicMock()
        
        mock_performance_service = MagicMock()
        mock_performance_service.get_performance_summary = MagicMock()
        
        # Test successful composition
        monitoring_service = MonitoringService(
            alert_service=mock_alert_service,
            metrics_service=mock_metrics_service,
            performance_service=mock_performance_service,
        )
        
        assert monitoring_service.alerts is mock_alert_service
        assert monitoring_service.metrics is mock_metrics_service
        assert monitoring_service.performance is mock_performance_service

    @pytest.mark.asyncio
    async def test_monitoring_service_validation(self):
        """Test that MonitoringService validates service interfaces."""
        mock_metrics_service = MagicMock()
        mock_metrics_service.record_counter = MagicMock()
        
        mock_performance_service = MagicMock()
        mock_performance_service.get_performance_summary = MagicMock()
        
        # Test with invalid alert service (missing required methods)
        invalid_alert_service = MagicMock()
        del invalid_alert_service.create_alert  # Remove required method
        
        with pytest.raises(Exception):  # Should validate interface
            MonitoringService(
                alert_service=invalid_alert_service,
                metrics_service=mock_metrics_service,
                performance_service=mock_performance_service,
            )


class TestMonitoringModuleBoundaries:
    """Test monitoring module boundaries and API usage."""

    @pytest.fixture 
    def mock_monitoring_service(self):
        """Mock monitoring service."""
        service = MagicMock()
        service.start = AsyncMock()
        service.stop = AsyncMock()
        service.health_check = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_main_app_monitoring_integration(self, mock_monitoring_service):
        """Test that main application correctly integrates monitoring service."""
        from src.main import Application
        
        app = Application()
        app.components["injector"] = MagicMock()
        
        # Mock the dependency registration and resolution
        with patch('src.monitoring.di_registration.register_monitoring_services') as mock_register, \
             patch.object(app.components["injector"], 'resolve', return_value=mock_monitoring_service):
            
            await app._initialize_monitoring()
            
            # Verify registration was called
            mock_register.assert_called_once_with(app.components["injector"])
            
            # Verify service was resolved and started
            app.components["injector"].resolve.assert_called_with("MonitoringServiceInterface")
            mock_monitoring_service.start.assert_called_once()
            
            # Verify service was stored
            assert app.components["monitoring_service"] is mock_monitoring_service
            assert app.health_status["components"]["monitoring"] == "initialized"

    @pytest.mark.asyncio
    async def test_execution_service_monitoring_integration(self):
        """Test execution service properly uses monitoring interfaces."""
        from src.monitoring.interfaces import MetricsServiceInterface
        
        # Mock the monitoring service interface
        mock_metrics_service = MagicMock(spec=MetricsServiceInterface)
        
        # Import and test execution service usage
        from src.execution.service import ExecutionService
        
        mock_repository_service = MagicMock()

        execution_service = ExecutionService(
            repository_service=mock_repository_service,
            metrics_service=mock_metrics_service,
        )

        # Verify monitoring service is properly injected
        assert execution_service.metrics_service is mock_metrics_service

    @pytest.mark.asyncio
    async def test_strategy_monitoring_integration(self):
        """Test strategy base properly integrates with monitoring."""
        from src.strategies.base import BaseStrategy
        from src.strategies.dependencies import StrategyServiceContainer
        
        # Create mock monitoring service
        mock_monitoring_service = MagicMock()
        
        # Create service container with monitoring
        services = StrategyServiceContainer()
        services.monitoring_service = mock_monitoring_service
        
        # Create a concrete strategy implementation for testing
        class TestStrategy(BaseStrategy):
            @property
            def strategy_type(self):
                from src.core.types import StrategyType
                return StrategyType.MOMENTUM
                
            async def _generate_signals_impl(self, data):
                return []
                
            def should_exit(self, position, current_price, market_data):
                return False
                
            def validate_signal(self, signal):
                return True
        
        strategy = TestStrategy(
            config={
                "name": "test",
                "strategy_id": "test-001",
                "strategy_type": "mean_reversion",
                "symbol": "BTCUSDT",
                "timeframe": "1h"
            },
            services=services
        )
        
        # Verify monitoring service is available
        assert strategy.services.monitoring_service is mock_monitoring_service


class TestMonitoringErrorIntegration:
    """Test monitoring error handling integration."""

    @pytest.mark.asyncio
    async def test_monitoring_service_error_propagation(self):
        """Test that monitoring services properly propagate errors."""
        from src.monitoring.services import DefaultAlertService
        from src.core.exceptions import ValidationError
        
        mock_alert_manager = MagicMock()
        service = DefaultAlertService(alert_manager=mock_alert_manager)
        
        # Test invalid request validation
        with pytest.raises(ValidationError):
            await service.create_alert("invalid_request")  # Should be AlertRequest object

    @pytest.mark.asyncio
    async def test_monitoring_circuit_breaker_integration(self):
        """Test monitoring integrates with circuit breaker patterns."""
        # This would test that monitoring services use circuit breakers
        # when calling external dependencies like Grafana, Prometheus, etc.
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])