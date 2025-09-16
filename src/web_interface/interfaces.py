"""
Service interfaces for web interface module.

This module defines the service interfaces specific to the web interface layer,
ensuring proper separation of concerns and dependency inversion.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Protocol


class WebPortfolioServiceInterface(Protocol):
    """Interface for web portfolio service operations."""

    async def get_portfolio_summary_data(self) -> dict[str, Any]:
        """Get processed portfolio summary data with business logic."""
        ...

    async def calculate_pnl_periods(
        self, total_pnl: Decimal, total_trades: int, win_rate: float
    ) -> dict[str, dict[str, Any]]:
        """Calculate P&L data for different periods with business logic."""
        ...

    async def get_processed_positions(
        self, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Get positions with business logic processing and filtering."""
        ...

    async def calculate_pnl_metrics(self, period: str) -> dict[str, Any]:
        """Calculate P&L metrics for a specific period with business logic."""
        ...

    def generate_mock_balances(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Generate mock balance data (business logic for development/testing)."""
        ...

    def calculate_asset_allocation(self) -> list[dict[str, Any]]:
        """Calculate asset allocation with business logic."""
        ...

    def generate_performance_chart_data(self, period: str, resolution: str) -> dict[str, Any]:
        """Generate performance chart data with business logic."""
        ...


class WebTradingServiceInterface(Protocol):
    """Interface for web trading service operations."""

    async def validate_order_request(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> dict[str, Any]:
        """Validate order request with web-specific business logic."""
        ...

    async def format_order_response(
        self, order_result: dict[str, Any], request_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Format order response with web-specific formatting."""
        ...

    async def get_formatted_orders(
        self, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Get orders with web-specific formatting and business logic."""
        ...

    async def get_formatted_trades(
        self, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Get trades with web-specific formatting and business logic."""
        ...

    async def get_market_data_with_context(
        self, symbol: str, exchange: str = "binance"
    ) -> dict[str, Any]:
        """Get market data with web-specific context and formatting."""
        ...

    async def generate_order_book_data(
        self, symbol: str, exchange: str, depth: int
    ) -> dict[str, Any]:
        """Generate order book data with web-specific business logic."""
        ...


class WebBotServiceInterface(Protocol):
    """Interface for web bot service operations."""

    async def validate_bot_configuration(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Validate bot configuration with web-specific business logic."""
        ...

    async def format_bot_response(self, bot_data: dict[str, Any]) -> dict[str, Any]:
        """Format bot data for web response."""
        ...

    async def get_formatted_bot_list(
        self, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Get bot list with web-specific formatting and filtering."""
        ...

    async def calculate_bot_metrics(self, bot_id: str) -> dict[str, Any]:
        """Calculate bot metrics with web-specific business logic."""
        ...

    async def validate_bot_operation(self, bot_id: str, operation: str) -> dict[str, Any]:
        """Validate bot operation with web-specific checks."""
        ...

    async def create_bot_configuration(self, request_data: dict[str, Any], user_id: str) -> Any:
        """Create bot configuration object with business logic validation."""
        ...


class WebMonitoringServiceInterface(Protocol):
    """Interface for web monitoring service operations."""

    async def get_system_health_summary(self) -> dict[str, Any]:
        """Get system health summary with web-specific formatting."""
        ...

    async def get_performance_metrics(self, component: str | None = None) -> dict[str, Any]:
        """Get performance metrics with web-specific processing."""
        ...

    async def get_error_summary(self, time_range: str = "24h") -> dict[str, Any]:
        """Get error summary with web-specific analysis."""
        ...

    async def get_alert_dashboard_data(self) -> dict[str, Any]:
        """Get alert dashboard data with web-specific formatting."""
        ...


class WebRiskServiceInterface(Protocol):
    """Interface for web risk service operations."""

    async def get_risk_dashboard_data(self) -> dict[str, Any]:
        """Get risk dashboard data with web-specific formatting."""
        ...

    async def validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Validate risk parameters with web-specific business logic."""
        ...

    async def calculate_position_risk(
        self, symbol: str, quantity: Decimal, price: Decimal
    ) -> dict[str, Any]:
        """Calculate position risk with web-specific metrics."""
        ...

    async def get_portfolio_risk_breakdown(self) -> dict[str, Any]:
        """Get portfolio risk breakdown with web-specific analysis."""
        ...


class WebStrategyServiceInterface(Protocol):
    """Interface for web strategy service operations."""

    async def get_formatted_strategies(self) -> list[dict[str, Any]]:
        """Get strategies with web-specific formatting."""
        ...

    async def validate_strategy_parameters(
        self, strategy_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate strategy parameters with web-specific business logic."""
        ...

    async def get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]:
        """Get strategy performance data with web-specific metrics."""
        ...

    async def format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]:
        """Format backtest results for web display."""
        ...


class WebDataServiceInterface(Protocol):
    """Interface for web data service operations."""

    async def get_market_overview(self, exchange: str = "binance") -> dict[str, Any]:
        """Get market overview with web-specific formatting."""
        ...

    async def get_symbol_analytics(self, symbol: str) -> dict[str, Any]:
        """Get symbol analytics with web-specific metrics."""
        ...

    async def get_historical_chart_data(
        self, symbol: str, timeframe: str, period: str
    ) -> dict[str, Any]:
        """Get historical chart data with web-specific formatting."""
        ...

    async def get_real_time_feed_status(self) -> dict[str, Any]:
        """Get real-time feed status with web-specific diagnostics."""
        ...


# Base interface for all web services
class WebServiceInterface(ABC):
    """Base interface for all web services."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the web service."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup web service resources."""
        pass

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status."""
        pass

    @abstractmethod
    def get_service_info(self) -> dict[str, Any]:
        """Get service information and capabilities."""
        pass


class WebStrategyServiceExtendedInterface(Protocol):
    """Extended interface for web strategy service operations."""

    async def get_formatted_strategies(self) -> list[dict[str, Any]]:
        """Get strategies with web-specific formatting."""
        ...

    async def validate_strategy_parameters(
        self, strategy_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate strategy parameters with web-specific business logic."""
        ...

    async def get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]:
        """Get strategy performance data with web-specific metrics."""
        ...

    async def format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]:
        """Format backtest results for web display."""
        ...

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status."""
        ...

    def get_service_info(self) -> dict[str, Any]:
        """Get service information and capabilities."""
        ...
