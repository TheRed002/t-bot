"""Tests for analytics environment integration module."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from src.analytics.environment_integration import (
    AnalyticsMode,
    ReportingLevel,
    EnvironmentAwareAnalyticsConfiguration,
    EnvironmentAwareAnalyticsManager,
)
from src.core.integration.environment_aware_service import EnvironmentContext


class TestAnalyticsMode:
    """Test AnalyticsMode enum."""

    def test_analytics_mode_values(self):
        """Test that all analytics modes have correct values."""
        assert AnalyticsMode.EXPERIMENTAL.value == "experimental"
        assert AnalyticsMode.PRODUCTION.value == "production"
        assert AnalyticsMode.COMPLIANCE.value == "compliance"
        assert AnalyticsMode.DEVELOPMENT.value == "development"

    def test_analytics_mode_count(self):
        """Test that we have the expected number of modes."""
        assert len(AnalyticsMode) == 4


class TestReportingLevel:
    """Test ReportingLevel enum."""

    def test_reporting_level_values(self):
        """Test that all reporting levels have correct values."""
        assert ReportingLevel.MINIMAL.value == "minimal"
        assert ReportingLevel.STANDARD.value == "standard"
        assert ReportingLevel.DETAILED.value == "detailed"
        assert ReportingLevel.COMPREHENSIVE.value == "comprehensive"

    def test_reporting_level_count(self):
        """Test that we have the expected number of levels."""
        assert len(ReportingLevel) == 4


class TestEnvironmentAwareAnalyticsConfiguration:
    """Test EnvironmentAwareAnalyticsConfiguration."""

    def test_get_sandbox_analytics_config(self):
        """Test sandbox analytics configuration."""
        config = EnvironmentAwareAnalyticsConfiguration.get_sandbox_analytics_config()

        # Verify structure
        assert isinstance(config, dict)

        # Verify key sandbox-specific settings
        assert config["analytics_mode"] == AnalyticsMode.EXPERIMENTAL
        assert config["reporting_level"] == ReportingLevel.COMPREHENSIVE
        assert config["enable_real_time_analytics"] is True
        assert config["enable_backtesting_analytics"] is True
        assert config["enable_experimental_metrics"] is True
        assert config["enable_detailed_logging"] is True
        assert config["enable_performance_profiling"] is True

        # Verify resource-intensive features are disabled
        assert config["enable_market_impact_analysis"] is False
        assert config["benchmark_comparison"] is False

        # Verify timing settings for sandbox
        assert config["reporting_frequency_minutes"] == 5  # More frequent
        assert config["max_computation_time_seconds"] == 60  # Longer timeout
        assert config["data_retention_days"] == 30

    def test_get_live_analytics_config(self):
        """Test live/production analytics configuration."""
        config = EnvironmentAwareAnalyticsConfiguration.get_live_analytics_config()

        # Verify structure
        assert isinstance(config, dict)

        # Verify key production-specific settings
        assert config["analytics_mode"] == AnalyticsMode.PRODUCTION
        assert config["reporting_level"] == ReportingLevel.STANDARD
        assert config["enable_real_time_analytics"] is True
        assert config["enable_risk_analytics"] is True
        assert config["enable_market_impact_analysis"] is True  # Important for production
        assert config["benchmark_comparison"] is True

        # Verify experimental features are disabled
        assert config["enable_backtesting_analytics"] is False
        assert config["enable_experimental_metrics"] is False
        assert config["enable_detailed_logging"] is False
        assert config["enable_performance_profiling"] is False
        assert config["enable_custom_metrics"] is False
        assert config["enable_ml_analytics"] is False
        assert config["enable_simulation_analytics"] is False

        # Verify timing settings for production
        assert config["reporting_frequency_minutes"] == 15  # Less frequent
        assert config["max_computation_time_seconds"] == 30  # Shorter timeout
        assert config["data_retention_days"] == 365

    def test_sandbox_vs_live_config_differences(self):
        """Test that sandbox and live configs have appropriate differences."""
        sandbox_config = EnvironmentAwareAnalyticsConfiguration.get_sandbox_analytics_config()
        live_config = EnvironmentAwareAnalyticsConfiguration.get_live_analytics_config()

        # Sandbox should be more permissive/experimental
        assert sandbox_config["enable_experimental_metrics"] != live_config["enable_experimental_metrics"]
        assert sandbox_config["enable_detailed_logging"] != live_config["enable_detailed_logging"]
        assert sandbox_config["reporting_frequency_minutes"] < live_config["reporting_frequency_minutes"]
        assert sandbox_config["max_computation_time_seconds"] > live_config["max_computation_time_seconds"]

        # Both should have same core functionality
        assert sandbox_config["enable_real_time_analytics"] == live_config["enable_real_time_analytics"]
        assert sandbox_config["enable_risk_analytics"] == live_config["enable_risk_analytics"]


class TestEnvironmentAwareAnalyticsManager:
    """Test EnvironmentAwareAnalyticsManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = EnvironmentAwareAnalyticsManager()
        self.mock_logger = Mock()
        self.manager.logger = self.mock_logger

    def test_initialization(self):
        """Test manager initialization."""
        manager = EnvironmentAwareAnalyticsManager()
        assert hasattr(manager, '_environment_analytics_cache')
        assert isinstance(manager._environment_analytics_cache, dict)
        assert hasattr(manager, '_analytics_metrics')
        assert isinstance(manager._analytics_metrics, dict)

    def test_get_environment_analytics_config_sandbox(self):
        """Test getting analytics config for sandbox environment."""
        config = self.manager.get_environment_analytics_config('binance')

        # Should return sandbox config by default (or based on environment context)
        assert isinstance(config, dict)
        assert 'analytics_mode' in config
        assert 'reporting_level' in config

    def test_get_environment_analytics_config_caching(self):
        """Test that analytics config is cached properly."""
        exchange = 'binance'

        # First call
        config1 = self.manager.get_environment_analytics_config(exchange)

        # Second call should return cached result
        config2 = self.manager.get_environment_analytics_config(exchange)

        # Should be the same object due to caching
        assert config1 is config2

    @pytest.mark.asyncio
    async def test_generate_environment_aware_report(self):
        """Test generating environment-aware reports."""
        report = await self.manager.generate_environment_aware_report(
            'performance', 'binance', '1h'
        )

        assert isinstance(report, dict)
        assert 'report_type' in report
        assert 'exchange' in report
        assert 'environment_context' in report
        assert report['report_type'] == 'performance'
        assert report['exchange'] == 'binance'
        assert report['environment_context']['environment'] == 'sandbox'

    @pytest.mark.asyncio
    async def test_generate_environment_aware_report_with_caching(self):
        """Test that reports are cached when appropriate."""
        # Mock the cache to simulate cached result
        cache_key = "performance_binance_1h"
        cached_report = {
            'cached': True,
            'report_type': 'performance',
            'exchange': 'binance'
        }

        with patch.object(self.manager, '_get_cached_report', return_value=cached_report):
            report = await self.manager.generate_environment_aware_report(
                'binance', 'performance', '1h'
            )

            assert report['cached'] is True

    @pytest.mark.asyncio
    async def test_track_environment_performance(self):
        """Test tracking environment performance."""
        start_time = datetime.now()
        success = True
        was_cached = False

        await self.manager.track_environment_performance(
            'binance', start_time, success, was_cached
        )

        # Should update analytics metrics
        assert 'binance' in self.manager._analytics_metrics
        metrics = self.manager._analytics_metrics['binance']
        assert 'total_requests' in metrics
        assert 'successful_requests' in metrics
        assert 'failed_requests' in metrics
        assert 'cached_responses' in metrics
        assert 'uncached_responses' in metrics

    def test_get_environment_analytics_metrics(self):
        """Test getting environment analytics metrics."""
        # First add some metrics
        exchange = 'binance'
        self.manager._analytics_metrics[exchange] = {
            'total_requests': 100,
            'successful_requests': 95,
            'failed_requests': 5,
            'cached_responses': 50,
            'uncached_responses': 50,
            'average_response_time': 0.5,
            'cache_hit_rate': 0.5
        }

        metrics = self.manager.get_environment_analytics_metrics(exchange)

        assert isinstance(metrics, dict)
        assert metrics['total_requests'] == 100
        assert metrics['successful_requests'] == 95
        assert metrics['failed_requests'] == 5
        assert metrics['cache_hit_rate'] == 0.5

    def test_get_environment_analytics_metrics_no_data(self):
        """Test getting metrics when no data exists."""
        metrics = self.manager.get_environment_analytics_metrics('nonexistent')

        assert isinstance(metrics, dict)
        assert metrics['total_requests'] == 0
        assert metrics['successful_requests'] == 0
        assert metrics['failed_requests'] == 0
        assert metrics['cache_hit_rate'] == 0.0

    @pytest.mark.asyncio
    async def test_update_service_environment(self):
        """Test updating service environment."""
        # Mock the environment context
        context = Mock()
        context.exchange = 'binance'
        context.environment = 'sandbox'

        await self.manager._update_service_environment(context)

        # Should complete without error
        assert True

    @pytest.mark.asyncio
    async def test_generate_performance_report(self):
        """Test generating performance report."""
        report = await self.manager._generate_performance_report(
            'binance', '1h', ReportingLevel.DETAILED, True
        )

        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'detailed_metrics' in report
        # Environment defaults to sandbox, so we get sandbox_metrics not production_metrics
        assert 'sandbox_metrics' in report

    @pytest.mark.asyncio
    async def test_generate_risk_report(self):
        """Test generating risk report."""
        report = await self.manager._generate_risk_report(
            'binance', '1h', ReportingLevel.STANDARD, False
        )

        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'report_type' in report
        assert report['report_type'] == 'risk'

    @pytest.mark.asyncio
    async def test_generate_execution_report(self):
        """Test generating execution report."""
        report = await self.manager._generate_execution_report(
            'binance', '1h', ReportingLevel.STANDARD, False
        )

        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'report_type' in report
        assert report['report_type'] == 'execution'

    @pytest.mark.asyncio
    async def test_generate_portfolio_report(self):
        """Test generating portfolio report."""
        report = await self.manager._generate_portfolio_report(
            'binance', '1h', ReportingLevel.STANDARD, False
        )

        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'report_type' in report
        assert report['report_type'] == 'portfolio'


class TestEnvironmentIntegrationEdgeCases:
    """Test edge cases and error handling."""

    def test_analytics_mode_enum_membership(self):
        """Test checking membership in AnalyticsMode enum."""
        assert AnalyticsMode.PRODUCTION in AnalyticsMode
        assert "invalid_mode" not in [mode.value for mode in AnalyticsMode]

    def test_reporting_level_enum_membership(self):
        """Test checking membership in ReportingLevel enum."""
        assert ReportingLevel.STANDARD in ReportingLevel
        assert "invalid_level" not in [level.value for level in ReportingLevel]

    @pytest.mark.asyncio
    async def test_manager_error_handling(self):
        """Test error handling in manager methods."""
        manager = EnvironmentAwareAnalyticsManager()

        # Test with invalid exchange
        try:
            await manager.generate_environment_aware_report(None, 'performance', '1h')
        except Exception:
            # Should handle gracefully or raise appropriate error
            pass

        # Test with invalid report type
        try:
            await manager.generate_environment_aware_report('binance', None, '1h')
        except Exception:
            # Should handle gracefully or raise appropriate error
            pass


class TestConfigurationConsistency:
    """Test configuration consistency and validation."""

    def test_all_config_keys_present(self):
        """Test that both configs have the same keys."""
        sandbox_config = EnvironmentAwareAnalyticsConfiguration.get_sandbox_analytics_config()
        live_config = EnvironmentAwareAnalyticsConfiguration.get_live_analytics_config()

        sandbox_keys = set(sandbox_config.keys())
        live_keys = set(live_config.keys())

        # Both configs should have the same keys
        assert sandbox_keys == live_keys

    def test_config_value_types(self):
        """Test that config values have appropriate types."""
        configs = [
            EnvironmentAwareAnalyticsConfiguration.get_sandbox_analytics_config(),
            EnvironmentAwareAnalyticsConfiguration.get_live_analytics_config(),
        ]

        for config in configs:
            # Enum values
            assert isinstance(config['analytics_mode'], AnalyticsMode)
            assert isinstance(config['reporting_level'], ReportingLevel)

            # Boolean values
            bool_keys = [
                'enable_real_time_analytics',
                'enable_backtesting_analytics',
                'enable_performance_attribution',
                'enable_risk_analytics',
                'parallel_processing',
                'cache_results',
            ]

            for key in bool_keys:
                assert isinstance(config[key], bool), f"{key} should be boolean"

            # Integer values
            int_keys = [
                'reporting_frequency_minutes',
                'data_retention_days',
                'max_computation_time_seconds',
            ]

            for key in int_keys:
                assert isinstance(config[key], int), f"{key} should be integer"
                assert config[key] > 0, f"{key} should be positive"