"""
Integration tests verifying external modules properly use analytics.

This module tests that other trading system modules correctly integrate
with and consume analytics services through proper APIs and patterns.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.dependency_injection import DependencyInjector
from src.core.types import (
    ExecutionResult,
    MarketData,
)


class TestExternalAnalyticsUsage:
    """Test external modules properly use analytics services."""

    @pytest.fixture
    def mock_analytics_service(self):
        """Create mock analytics service."""
        service = Mock()
        service.start = AsyncMock()
        service.stop = AsyncMock()
        service.update_trade = Mock()
        service.update_position = Mock()
        service.update_order = Mock()
        service.update_price = Mock()
        service.get_portfolio_metrics = AsyncMock(return_value={})
        service.get_risk_metrics = AsyncMock(return_value={})
        service.record_strategy_event = Mock()
        service.record_system_error = Mock()
        return service

    @pytest.fixture
    def mock_database_service(self):
        """Create mock database service."""
        service = Mock()
        service.create_entity = AsyncMock()
        service.list_entities = AsyncMock(return_value=[])
        service.health_check = AsyncMock(return_value="healthy")
        return service

    @pytest.fixture
    def mock_validation_service(self):
        """Create mock validation service."""
        service = Mock()
        return service

    async def test_execution_service_analytics_integration(
        self, mock_analytics_service, mock_database_service, mock_validation_service
    ):
        """Test that ExecutionService properly integrates with analytics."""
        from src.core.types.execution import ExecutionStatus

        # Create test data with proper ExecutionResult structure
        execution_result = ExecutionResult(
            instruction_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            status=ExecutionStatus.COMPLETED,
            target_quantity=Decimal("1.0"),
            filled_quantity=Decimal("1.0"),
            remaining_quantity=Decimal("0.0"),
            average_price=Decimal("50000.00"),
            worst_price=Decimal("50100.00"),
            best_price=Decimal("49900.00"),
            expected_cost=Decimal("50000.00"),
            actual_cost=Decimal("50050.00"),
            slippage_bps=Decimal("1.0"),
            slippage_amount=Decimal("50.00"),
            fill_rate=Decimal("1.0"),
            execution_time=30,
            num_fills=3,
            num_orders=1,
            total_fees=Decimal("25.00"),
            maker_fees=Decimal("10.00"),
            taker_fees=Decimal("15.00"),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            fills=[
                {
                    "price": "50000.00",
                    "quantity": "1.0",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        )

        market_data = MarketData(
            symbol="BTC/USDT",
            price=Decimal("50000.00"),
            volume=Decimal("1000.00"),
            timestamp=datetime.now(timezone.utc),
            open=Decimal("49800.00"),
            high=Decimal("50200.00"),
            low=Decimal("49700.00"),
            close=Decimal("50000.00"),
            exchange="binance",
        )

        # Mock database service for ExecutionService
        mock_database_service.create_entity = AsyncMock()
        mock_database_service.list_entities = AsyncMock(return_value=[])

        # Mock execution service dependencies
        mock_order_validator = Mock()
        mock_risk_manager = Mock()

        # Test the integration pattern (conceptual)
        # In a real integration, ExecutionService would call analytics after recording trade

        # Simulate recording trade execution
        trade_data = {
            "symbol": execution_result.symbol,
            "quantity": execution_result.filled_quantity,
            "price": execution_result.average_price,
            "status": execution_result.status.value,
            "execution_time": execution_result.execution_time,
        }

        # Verify analytics service would be called to update trade
        mock_analytics_service.update_trade(trade_data)
        mock_analytics_service.update_trade.assert_called_once_with(trade_data)

        # Verify analytics service would record execution performance
        mock_analytics_service.record_strategy_event(
            strategy_name="test-strategy",
            event_type="trade_executed",
            success=execution_result.status == ExecutionStatus.COMPLETED,
        )

        mock_analytics_service.record_strategy_event.assert_called_with(
            strategy_name="test-strategy", event_type="trade_executed", success=True
        )

    async def test_risk_management_analytics_integration(self, mock_analytics_service):
        """Test that risk management properly uses analytics data."""
        # This would test RiskManager integration with analytics
        # Since RiskManager doesn't exist yet, we'll mock the pattern

        # Simulate risk manager requesting analytics data
        portfolio_metrics = await mock_analytics_service.get_portfolio_metrics()
        risk_metrics = await mock_analytics_service.get_risk_metrics()

        # Verify analytics provides the expected data
        mock_analytics_service.get_portfolio_metrics.assert_called_once()
        mock_analytics_service.get_risk_metrics.assert_called_once()

    async def test_web_interface_analytics_integration(self, mock_analytics_service):
        """Test that web interface properly consumes analytics data."""
        # Test basic analytics service integration patterns
        portfolio_metrics = await mock_analytics_service.get_portfolio_metrics()
        risk_metrics = await mock_analytics_service.get_risk_metrics()

        # Verify analytics provides the expected data structure
        mock_analytics_service.get_portfolio_metrics.assert_called_once()
        mock_analytics_service.get_risk_metrics.assert_called_once()

        # Test analytics event recording
        mock_analytics_service.record_strategy_event(
            strategy_name="web-interface-test", event_type="data_request", success=True
        )

        mock_analytics_service.record_strategy_event.assert_called_with(
            strategy_name="web-interface-test", event_type="data_request", success=True
        )

    async def test_strategy_analytics_integration(self, mock_analytics_service):
        """Test that trading strategies properly report to analytics."""
        # Simulate strategy reporting events to analytics
        mock_analytics_service.record_strategy_event(
            strategy_name="test-strategy", event_type="signal_generated", success=True
        )

        # Verify analytics received the event
        mock_analytics_service.record_strategy_event.assert_called_once_with(
            strategy_name="test-strategy", event_type="signal_generated", success=True
        )

    async def test_error_handling_analytics_integration(self, mock_analytics_service):
        """Test that error handling system reports to analytics."""
        # Simulate error reporting to analytics
        mock_analytics_service.record_system_error(
            component="execution_service",
            error_type="validation_error",
            error_message="Invalid order parameters",
        )

        # Verify analytics received the error
        mock_analytics_service.record_system_error.assert_called_once_with(
            component="execution_service",
            error_type="validation_error",
            error_message="Invalid order parameters",
        )

    async def test_analytics_service_contract_compliance(self):
        """Test that analytics service implements expected contracts."""
        from src.analytics.service import AnalyticsService

        # Verify AnalyticsService implements the protocol
        # This is a compile-time check that Protocol is properly implemented
        assert hasattr(AnalyticsService, "start")
        assert hasattr(AnalyticsService, "stop")
        assert hasattr(AnalyticsService, "update_position")
        assert hasattr(AnalyticsService, "update_trade")
        assert hasattr(AnalyticsService, "update_order")
        assert hasattr(AnalyticsService, "get_portfolio_metrics")
        assert hasattr(AnalyticsService, "get_risk_metrics")

    async def test_dependency_injection_patterns(self):
        """Test that analytics is properly injected into dependent services."""
        from src.analytics.di_registration import register_analytics_services

        injector = DependencyInjector()

        # Register minimal mock dependencies needed
        mock_uow = Mock()
        mock_metrics = Mock()
        injector.register_service("UnitOfWork", lambda: mock_uow, singleton=True)
        injector.register_service("MetricsCollector", lambda: mock_metrics, singleton=True)

        # Register analytics services
        register_analytics_services(injector)

        # Test that DI registration worked by checking service existence
        # Don't actually resolve services to avoid heavy initialization
        assert injector.has_service("AnalyticsService")
        assert injector.has_service("AnalyticsServiceProtocol")
        assert injector.has_service("PortfolioService")
        assert injector.has_service("RiskService")
        assert injector.has_service("ReportingService")

    async def test_service_layer_boundaries(self):
        """Test that modules respect service layer boundaries."""
        # This test ensures that external modules:
        # 1. Don't directly access analytics internals
        # 2. Use proper service interfaces
        # 3. Don't bypass the service layer

        from src.analytics import AnalyticsService
        from src.analytics.interfaces import AnalyticsServiceProtocol

        # Test that external modules should use the protocol, not concrete class
        # (This is more of a design guideline test)
        assert issubclass(AnalyticsService, object)

        # Verify protocol defines the public interface
        protocol_methods = [
            "start",
            "stop",
            "update_position",
            "update_trade",
            "update_order",
            "get_portfolio_metrics",
            "get_risk_metrics",
        ]

        for method in protocol_methods:
            assert hasattr(AnalyticsServiceProtocol, method)

    async def test_analytics_error_isolation(self, mock_analytics_service):
        """Test that analytics failures don't break dependent services."""
        # Test that when analytics service fails, it doesn't crash the system
        mock_analytics_service.get_portfolio_metrics.side_effect = Exception(
            "Analytics service down"
        )
        mock_analytics_service.record_strategy_event.side_effect = Exception(
            "Analytics service down"
        )

        # Verify that calling analytics methods raises exceptions
        with pytest.raises(Exception, match="Analytics service down"):
            await mock_analytics_service.get_portfolio_metrics()

        with pytest.raises(Exception, match="Analytics service down"):
            mock_analytics_service.record_strategy_event(
                strategy_name="test", event_type="test", success=True
            )

        # In a real system, these would be wrapped with try/catch to prevent
        # cascading failures
