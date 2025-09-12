"""
Comprehensive tests for analytics interfaces module.

Tests the protocol definitions and abstract base classes
used for dependency injection and service abstraction.
"""

# Disable logging during tests for performance
import logging

import pytest

logging.disable(logging.CRITICAL)

# Set pytest markers for optimization
pytestmark = pytest.mark.unit
import inspect
from abc import ABC
from decimal import Decimal
from typing import Protocol, get_type_hints
from unittest.mock import Mock

from src.analytics.interfaces import (
    AlertServiceProtocol,
    AnalyticsDataRepository,
    AnalyticsServiceProtocol,
    ExportServiceProtocol,
    MetricsCalculationService,
    OperationalServiceProtocol,
    PortfolioServiceProtocol,
    RealtimeAnalyticsServiceProtocol,
    ReportingServiceProtocol,
    RiskCalculationService,
    RiskServiceProtocol,
)
from src.analytics.types import (
    AnalyticsAlert,
    AnalyticsReport,
    OperationalMetrics,
    PortfolioMetrics,
    ReportType,
    RiskMetrics,
)
from src.core.types import Order, Position, Trade


class TestProtocolDefinitions:
    """Test that all protocols are properly defined."""

    def test_analytics_service_protocol_is_protocol(self):
        """Test AnalyticsServiceProtocol is a Protocol."""
        assert issubclass(AnalyticsServiceProtocol, Protocol)

    def test_alert_service_protocol_is_protocol(self):
        """Test AlertServiceProtocol is a Protocol."""
        assert issubclass(AlertServiceProtocol, Protocol)

    def test_risk_service_protocol_is_protocol(self):
        """Test RiskServiceProtocol is a Protocol."""
        assert issubclass(RiskServiceProtocol, Protocol)

    def test_portfolio_service_protocol_is_protocol(self):
        """Test PortfolioServiceProtocol is a Protocol."""
        assert issubclass(PortfolioServiceProtocol, Protocol)

    def test_reporting_service_protocol_is_protocol(self):
        """Test ReportingServiceProtocol is a Protocol."""
        assert issubclass(ReportingServiceProtocol, Protocol)

    def test_export_service_protocol_is_protocol(self):
        """Test ExportServiceProtocol is a Protocol."""
        assert issubclass(ExportServiceProtocol, Protocol)

    def test_operational_service_protocol_is_protocol(self):
        """Test OperationalServiceProtocol is a Protocol."""
        assert issubclass(OperationalServiceProtocol, Protocol)

    def test_realtime_analytics_service_protocol_is_protocol(self):
        """Test RealtimeAnalyticsServiceProtocol is a Protocol."""
        assert issubclass(RealtimeAnalyticsServiceProtocol, Protocol)


class TestAbstractBaseClasses:
    """Test that abstract base classes are properly defined."""

    def test_analytics_data_repository_is_abc(self):
        """Test AnalyticsDataRepository is an ABC."""
        assert issubclass(AnalyticsDataRepository, ABC)

    def test_metrics_calculation_service_is_abc(self):
        """Test MetricsCalculationService is an ABC."""
        assert issubclass(MetricsCalculationService, ABC)

    def test_risk_calculation_service_is_abc(self):
        """Test RiskCalculationService is an ABC."""
        assert issubclass(RiskCalculationService, ABC)

    def test_cannot_instantiate_abstract_repository(self):
        """Test that abstract repository cannot be instantiated."""
        with pytest.raises(TypeError):
            AnalyticsDataRepository()

    def test_cannot_instantiate_abstract_metrics_service(self):
        """Test that abstract metrics service cannot be instantiated."""
        with pytest.raises(TypeError):
            MetricsCalculationService()

    def test_cannot_instantiate_abstract_risk_service(self):
        """Test that abstract risk service cannot be instantiated."""
        with pytest.raises(TypeError):
            RiskCalculationService()


class TestAnalyticsServiceProtocol:
    """Test AnalyticsServiceProtocol methods and signatures."""

    def test_protocol_methods_exist(self):
        """Test that all expected methods exist."""
        expected_methods = [
            "start",
            "stop",
            "update_position",
            "update_trade",
            "update_order",
            "get_portfolio_metrics",
            "get_risk_metrics",
        ]

        for method_name in expected_methods:
            assert hasattr(AnalyticsServiceProtocol, method_name)

    def test_start_method_signature(self):
        """Test start method signature."""
        method = AnalyticsServiceProtocol.start
        sig = inspect.signature(method)

        assert len(sig.parameters) == 1  # Only 'self'
        assert "self" in sig.parameters

        hints = get_type_hints(method)
        assert hints.get("return") is type(None)

    def test_stop_method_signature(self):
        """Test stop method signature."""
        method = AnalyticsServiceProtocol.stop
        sig = inspect.signature(method)

        assert len(sig.parameters) == 1  # Only 'self'
        hints = get_type_hints(method)
        assert hints.get("return") is type(None)

    def test_update_position_method_signature(self):
        """Test update_position method signature."""
        method = AnalyticsServiceProtocol.update_position
        sig = inspect.signature(method)

        assert "position" in sig.parameters
        hints = get_type_hints(method)
        assert hints.get("position") == Position

    def test_update_trade_method_signature(self):
        """Test update_trade method signature."""
        method = AnalyticsServiceProtocol.update_trade
        sig = inspect.signature(method)

        assert "trade" in sig.parameters
        hints = get_type_hints(method)
        assert hints.get("trade") == Trade

    def test_update_order_method_signature(self):
        """Test update_order method signature."""
        method = AnalyticsServiceProtocol.update_order
        sig = inspect.signature(method)

        assert "order" in sig.parameters
        hints = get_type_hints(method)
        assert hints.get("order") == Order

    def test_get_portfolio_metrics_return_type(self):
        """Test get_portfolio_metrics return type."""
        method = AnalyticsServiceProtocol.get_portfolio_metrics
        hints = get_type_hints(method)

        # Should return PortfolioMetrics | None
        return_type = hints.get("return")
        assert return_type is not None

    def test_get_risk_metrics_return_type(self):
        """Test get_risk_metrics return type."""
        method = AnalyticsServiceProtocol.get_risk_metrics
        hints = get_type_hints(method)

        assert hints.get("return") == RiskMetrics


class TestAlertServiceProtocol:
    """Test AlertServiceProtocol methods and signatures."""

    def test_generate_alert_method_signature(self):
        """Test generate_alert method signature."""
        method = AlertServiceProtocol.generate_alert
        sig = inspect.signature(method)

        expected_params = ["rule_name", "severity", "message", "labels", "annotations"]
        for param in expected_params:
            assert param in sig.parameters

        assert "kwargs" in sig.parameters

        hints = get_type_hints(method)
        assert hints.get("return") == AnalyticsAlert

    def test_get_active_alerts_return_type(self):
        """Test get_active_alerts return type."""
        method = AlertServiceProtocol.get_active_alerts
        hints = get_type_hints(method)

        return_type = hints.get("return")
        # Should be list[AnalyticsAlert]
        assert return_type is not None

    def test_acknowledge_alert_method_signature(self):
        """Test acknowledge_alert method signature."""
        method = AlertServiceProtocol.acknowledge_alert
        sig = inspect.signature(method)

        assert "fingerprint" in sig.parameters
        assert "acknowledged_by" in sig.parameters

        hints = get_type_hints(method)
        assert hints.get("return") == bool


class TestRiskServiceProtocol:
    """Test RiskServiceProtocol methods and signatures."""

    def test_calculate_var_method_signature(self):
        """Test calculate_var method signature."""
        method = RiskServiceProtocol.calculate_var
        sig = inspect.signature(method)

        expected_params = ["confidence_level", "time_horizon", "method"]
        for param in expected_params:
            assert param in sig.parameters

    def test_run_stress_test_method_signature(self):
        """Test run_stress_test method signature."""
        method = RiskServiceProtocol.run_stress_test
        sig = inspect.signature(method)

        assert "scenario_name" in sig.parameters
        assert "scenario_params" in sig.parameters

    def test_get_risk_metrics_return_type(self):
        """Test get_risk_metrics return type."""
        method = RiskServiceProtocol.get_risk_metrics
        hints = get_type_hints(method)

        assert hints.get("return") == RiskMetrics


class TestPortfolioServiceProtocol:
    """Test PortfolioServiceProtocol methods and signatures."""

    def test_calculate_portfolio_metrics_return_type(self):
        """Test calculate_portfolio_metrics return type."""
        method = PortfolioServiceProtocol.calculate_portfolio_metrics
        hints = get_type_hints(method)

        assert hints.get("return") == PortfolioMetrics

    def test_get_portfolio_composition_method_exists(self):
        """Test get_portfolio_composition method exists."""
        assert hasattr(PortfolioServiceProtocol, "get_portfolio_composition")

    def test_calculate_correlation_matrix_method_exists(self):
        """Test calculate_correlation_matrix method exists."""
        assert hasattr(PortfolioServiceProtocol, "calculate_correlation_matrix")


class TestReportingServiceProtocol:
    """Test ReportingServiceProtocol methods and signatures."""

    def test_generate_performance_report_signature(self):
        """Test generate_performance_report method signature."""
        method = ReportingServiceProtocol.generate_performance_report
        sig = inspect.signature(method)

        assert "report_type" in sig.parameters
        assert "start_date" in sig.parameters
        assert "end_date" in sig.parameters

        hints = get_type_hints(method)
        assert hints.get("report_type") == ReportType
        assert hints.get("return") == AnalyticsReport


class TestExportServiceProtocol:
    """Test ExportServiceProtocol methods and signatures."""

    def test_export_portfolio_data_signature(self):
        """Test export_portfolio_data method signature."""
        method = ExportServiceProtocol.export_portfolio_data
        sig = inspect.signature(method)

        assert "format" in sig.parameters
        assert "include_metadata" in sig.parameters

        # Check default values
        assert sig.parameters["format"].default == "json"
        assert sig.parameters["include_metadata"].default is True

        hints = get_type_hints(method)
        assert hints.get("return") == str

    def test_export_risk_data_signature(self):
        """Test export_risk_data method signature."""
        method = ExportServiceProtocol.export_risk_data
        sig = inspect.signature(method)

        assert "format" in sig.parameters
        assert "include_metadata" in sig.parameters

        hints = get_type_hints(method)
        assert hints.get("return") == str


class TestOperationalServiceProtocol:
    """Test OperationalServiceProtocol methods and signatures."""

    def test_get_operational_metrics_return_type(self):
        """Test get_operational_metrics return type."""
        method = OperationalServiceProtocol.get_operational_metrics
        hints = get_type_hints(method)

        assert hints.get("return") == OperationalMetrics

    def test_record_strategy_event_signature(self):
        """Test record_strategy_event method signature."""
        method = OperationalServiceProtocol.record_strategy_event
        sig = inspect.signature(method)

        assert "strategy_name" in sig.parameters
        assert "event_type" in sig.parameters
        assert "success" in sig.parameters
        assert "kwargs" in sig.parameters

        # Check default value
        assert sig.parameters["success"].default is True

    def test_record_system_error_signature(self):
        """Test record_system_error method signature."""
        method = OperationalServiceProtocol.record_system_error
        sig = inspect.signature(method)

        expected_params = ["component", "error_type", "error_message"]
        for param in expected_params:
            assert param in sig.parameters

        assert "kwargs" in sig.parameters


class TestRealtimeAnalyticsServiceProtocol:
    """Test RealtimeAnalyticsServiceProtocol methods and signatures."""

    def test_protocol_methods_exist(self):
        """Test that all expected methods exist."""
        expected_methods = [
            "start",
            "stop",
            "update_position",
            "update_trade",
            "update_order",
            "update_price",
            "get_portfolio_metrics",
            "get_position_metrics",
            "get_strategy_metrics",
        ]

        for method_name in expected_methods:
            assert hasattr(RealtimeAnalyticsServiceProtocol, method_name)

    def test_update_price_method_signature(self):
        """Test update_price method signature."""
        method = RealtimeAnalyticsServiceProtocol.update_price
        sig = inspect.signature(method)

        assert "symbol" in sig.parameters
        assert "price" in sig.parameters

        hints = get_type_hints(method)
        assert hints.get("price") == Decimal

    def test_get_position_metrics_signature(self):
        """Test get_position_metrics method signature."""
        method = RealtimeAnalyticsServiceProtocol.get_position_metrics
        sig = inspect.signature(method)

        assert "symbol" in sig.parameters

        # Check default value is None
        assert sig.parameters["symbol"].default is None

    def test_get_strategy_metrics_signature(self):
        """Test get_strategy_metrics method signature."""
        method = RealtimeAnalyticsServiceProtocol.get_strategy_metrics
        sig = inspect.signature(method)

        assert "strategy" in sig.parameters

        # Check default value is None
        assert sig.parameters["strategy"].default is None


class TestAbstractRepositoryMethods:
    """Test AnalyticsDataRepository abstract methods."""

    def test_store_portfolio_metrics_is_abstract(self):
        """Test store_portfolio_metrics is abstract."""
        method = AnalyticsDataRepository.store_portfolio_metrics
        assert getattr(method, "__isabstractmethod__", False)

    def test_store_position_metrics_is_abstract(self):
        """Test store_position_metrics is abstract."""
        method = AnalyticsDataRepository.store_position_metrics
        assert getattr(method, "__isabstractmethod__", False)

    def test_store_risk_metrics_is_abstract(self):
        """Test store_risk_metrics is abstract."""
        method = AnalyticsDataRepository.store_risk_metrics
        assert getattr(method, "__isabstractmethod__", False)

    def test_get_historical_portfolio_metrics_is_abstract(self):
        """Test get_historical_portfolio_metrics is abstract."""
        method = AnalyticsDataRepository.get_historical_portfolio_metrics
        assert getattr(method, "__isabstractmethod__", False)

    def test_repository_method_signatures(self):
        """Test repository method signatures."""
        # Test store_portfolio_metrics
        method = AnalyticsDataRepository.store_portfolio_metrics
        hints = get_type_hints(method)
        assert hints.get("metrics") == PortfolioMetrics

        # Test get_historical_portfolio_metrics
        method = AnalyticsDataRepository.get_historical_portfolio_metrics
        sig = inspect.signature(method)
        assert "start_date" in sig.parameters
        assert "end_date" in sig.parameters


class TestAbstractCalculationServices:
    """Test abstract calculation service methods."""

    def test_metrics_calculation_service_abstract_methods(self):
        """Test MetricsCalculationService abstract methods."""
        abstract_methods = [
            "calculate_portfolio_metrics",
            "calculate_position_metrics",
            "calculate_strategy_metrics",
        ]

        for method_name in abstract_methods:
            method = getattr(MetricsCalculationService, method_name)
            assert getattr(method, "__isabstractmethod__", False)

    def test_risk_calculation_service_abstract_methods(self):
        """Test RiskCalculationService abstract methods."""
        abstract_methods = ["calculate_portfolio_var", "calculate_risk_metrics"]

        for method_name in abstract_methods:
            method = getattr(RiskCalculationService, method_name)
            assert getattr(method, "__isabstractmethod__", False)

    def test_calculate_portfolio_var_signature(self):
        """Test calculate_portfolio_var method signature."""
        method = RiskCalculationService.calculate_portfolio_var
        sig = inspect.signature(method)

        expected_params = ["positions", "confidence_level", "time_horizon", "method"]
        for param in expected_params:
            assert param in sig.parameters

        # Check default value for method parameter
        assert sig.parameters["method"].default == "historical"


class TestProtocolCompatibility:
    """Test protocol compatibility and duck typing."""

    def test_mock_implements_analytics_service_protocol(self):
        """Test that a mock can implement AnalyticsServiceProtocol."""

        # Create a mock that implements the protocol
        mock_service = Mock(spec=AnalyticsServiceProtocol)

        # Verify it has the expected methods
        assert hasattr(mock_service, "start")
        assert hasattr(mock_service, "stop")
        assert hasattr(mock_service, "update_position")
        assert hasattr(mock_service, "get_portfolio_metrics")

    def test_mock_implements_alert_service_protocol(self):
        """Test that a mock can implement AlertServiceProtocol."""
        mock_service = Mock(spec=AlertServiceProtocol)

        assert hasattr(mock_service, "generate_alert")
        assert hasattr(mock_service, "get_active_alerts")
        assert hasattr(mock_service, "acknowledge_alert")

    def test_protocol_inheritance_structure(self):
        """Test that protocols have proper inheritance structure."""
        # Protocols should inherit from Protocol
        protocols = [
            AnalyticsServiceProtocol,
            AlertServiceProtocol,
            RiskServiceProtocol,
            PortfolioServiceProtocol,
            ReportingServiceProtocol,
            ExportServiceProtocol,
            OperationalServiceProtocol,
            RealtimeAnalyticsServiceProtocol,
        ]

        for protocol_class in protocols:
            assert issubclass(protocol_class, Protocol)


class TestTypeHintConsistency:
    """Test type hint consistency across interfaces."""

    def test_position_parameter_consistency(self):
        """Test Position type consistency across protocols."""
        protocols_with_position = [
            AnalyticsServiceProtocol,
            RealtimeAnalyticsServiceProtocol,
        ]

        for protocol in protocols_with_position:
            if hasattr(protocol, "update_position"):
                method = protocol.update_position
                hints = get_type_hints(method)
                assert hints.get("position") == Position

    def test_trade_parameter_consistency(self):
        """Test Trade type consistency across protocols."""
        protocols_with_trade = [
            AnalyticsServiceProtocol,
            RealtimeAnalyticsServiceProtocol,
        ]

        for protocol in protocols_with_trade:
            if hasattr(protocol, "update_trade"):
                method = protocol.update_trade
                hints = get_type_hints(method)
                assert hints.get("trade") == Trade

    def test_datetime_parameter_consistency(self):
        """Test datetime type consistency across interfaces."""
        # Check ReportingServiceProtocol
        method = ReportingServiceProtocol.generate_performance_report
        hints = get_type_hints(method)

        # start_date and end_date should allow None
        start_date_type = hints.get("start_date")
        end_date_type = hints.get("end_date")

        # These should be Optional[datetime] or datetime | None
        assert start_date_type is not None
        assert end_date_type is not None


class TestInterfaceDocumentation:
    """Test that interfaces have proper documentation."""

    def test_protocols_have_docstrings(self):
        """Test that all protocols have docstrings."""
        protocols = [
            AnalyticsServiceProtocol,
            AlertServiceProtocol,
            RiskServiceProtocol,
            PortfolioServiceProtocol,
            ReportingServiceProtocol,
            ExportServiceProtocol,
            OperationalServiceProtocol,
            RealtimeAnalyticsServiceProtocol,
        ]

        for protocol in protocols:
            assert protocol.__doc__ is not None
            assert len(protocol.__doc__.strip()) > 0

    def test_abstract_classes_have_docstrings(self):
        """Test that all abstract classes have docstrings."""
        abstract_classes = [
            AnalyticsDataRepository,
            MetricsCalculationService,
            RiskCalculationService,
        ]

        for cls in abstract_classes:
            assert cls.__doc__ is not None
            assert len(cls.__doc__.strip()) > 0
