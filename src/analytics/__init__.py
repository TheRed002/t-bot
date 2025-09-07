"""
Analytics Module for T-Bot Trading System.

This module provides comprehensive real-time analytics and reporting capabilities
for institutional-grade trading operations, including:

- Real-time trading analytics and P&L tracking
- Portfolio analytics with risk decomposition
- Performance attribution and benchmarking
- Risk reporting and monitoring
- Operational analytics and system monitoring

The analytics system is designed to meet the exacting standards of top-tier
hedge funds and asset managers, providing the deep insights required for
professional trading operations.
"""

from src.analytics import services

# Dependency injection registration
from src.analytics.di_registration import (
    configure_analytics_dependencies,
    get_analytics_factory,
    get_analytics_service,
    register_analytics_services,
)
from src.analytics.factory import AnalyticsServiceFactory, create_default_analytics_service
from src.analytics.interfaces import (
    AlertServiceProtocol,
    AnalyticsServiceProtocol,
    ExportServiceProtocol,
    OperationalServiceProtocol,
    PortfolioServiceProtocol,
    RealtimeAnalyticsServiceProtocol,
    ReportingServiceProtocol,
    RiskServiceProtocol,
)
from src.analytics.service import AnalyticsService

__all__ = [
    "AlertServiceProtocol",
    "AnalyticsService",
    "AnalyticsServiceFactory",
    "AnalyticsServiceProtocol",
    "ExportServiceProtocol",
    "OperationalServiceProtocol",
    "PortfolioServiceProtocol",
    "RealtimeAnalyticsServiceProtocol",
    "ReportingServiceProtocol",
    "RiskServiceProtocol",
    "configure_analytics_dependencies",
    "create_default_analytics_service",
    "get_analytics_factory",
    "get_analytics_service",
    "register_analytics_services",
    "services",
]
