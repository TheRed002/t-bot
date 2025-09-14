"""Backtesting service interfaces for dependency injection."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.core.base.interfaces import HealthCheckResult

if TYPE_CHECKING:
    from src.backtesting.service import BacktestRequest, BacktestResult
    from src.core.types import MarketData
    from src.data.types import DataRequest
    from src.database.models import MarketDataRecord


class DataServiceInterface(ABC):
    """Data service interface for backtesting dependencies."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the data service."""
        pass

    @abstractmethod
    async def store_market_data(
        self,
        data: "MarketData | list[MarketData]",
        exchange: str,
        validate: bool = True,
    ) -> bool:
        """Store market data."""
        pass

    @abstractmethod
    async def get_market_data(self, request: "DataRequest") -> "list[MarketDataRecord]":
        """Get market data."""
        pass

    @abstractmethod
    async def get_recent_data(
        self, symbol: str, limit: int = 100, exchange: str = "binance"
    ) -> "list[MarketData]":
        """Get recent market data."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        pass


@runtime_checkable
class BacktestServiceInterface(Protocol):
    """Interface for BacktestService."""

    async def initialize(self) -> None:
        """Initialize the service."""
        ...

    async def run_backtest(self, request: "BacktestRequest") -> "BacktestResult":
        """Run a backtest."""
        ...

    async def run_backtest_from_dict(self, request_data: dict[str, Any]) -> "BacktestResult":
        """Run backtest from dictionary request data."""
        ...

    async def serialize_result(self, result: "BacktestResult") -> dict[str, Any]:
        """Serialize BacktestResult for API response."""
        ...

    async def get_active_backtests(self) -> dict[str, dict[str, Any]]:
        """Get active backtests."""
        ...

    async def cancel_backtest(self, backtest_id: str) -> bool:
        """Cancel a backtest."""
        ...

    async def clear_cache(self, pattern: str = "*") -> int:
        """Clear backtest cache."""
        ...

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        ...

    async def health_check(self) -> HealthCheckResult:
        """Health check."""
        ...

    async def get_backtest_result(self, result_id: str) -> dict[str, Any] | None:
        """Get a specific backtest result by ID."""
        ...

    async def list_backtest_results(
        self, limit: int = 50, offset: int = 0, strategy_type: str | None = None
    ) -> list[dict[str, Any]]:
        """List backtest results with filtering."""
        ...

    async def delete_backtest_result(self, result_id: str) -> bool:
        """Delete a specific backtest result by ID."""
        ...

    async def cleanup(self) -> None:
        """Cleanup resources."""
        ...


class MetricsCalculatorInterface(Protocol):
    """Interface for MetricsCalculator."""

    def calculate_all(
        self,
        equity_curve: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        daily_returns: list[float],
        initial_capital: float,
    ) -> dict[str, Any]:
        """Calculate all metrics."""
        ...


class BacktestAnalyzerInterface(ABC):
    """Base interface for backtest analyzers."""

    @abstractmethod
    async def run_analysis(self, **kwargs) -> dict[str, Any]:
        """Run analysis."""
        pass


class BacktestEngineFactoryInterface(Protocol):
    """Interface for BacktestEngine factory."""

    def __call__(self, config: Any, strategy: Any, **kwargs) -> Any:
        """Create BacktestEngine instance."""
        ...


class ComponentFactoryInterface(Protocol):
    """Interface for component factories."""

    def __call__(self) -> Any:
        """Create component instance."""
        ...


@runtime_checkable
class BacktestControllerInterface(Protocol):
    """Interface for BacktestController."""

    async def run_backtest(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Handle backtest request via API."""
        ...

    async def get_active_backtests(self) -> dict[str, Any]:
        """Get status of active backtests."""
        ...

    async def cancel_backtest(self, backtest_id: str) -> dict[str, Any]:
        """Cancel a specific backtest."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        ...

    async def get_backtest_result(self, result_id: str) -> dict[str, Any]:
        """Get a specific backtest result by ID."""
        ...

    async def list_backtest_results(
        self, limit: int = 50, offset: int = 0, strategy_type: str | None = None
    ) -> dict[str, Any]:
        """List backtest results with filtering."""
        ...

    async def delete_backtest_result(self, result_id: str) -> dict[str, Any]:
        """Delete a specific backtest result by ID."""
        ...


@runtime_checkable
class BacktestRepositoryInterface(Protocol):
    """Interface for BacktestRepository."""

    async def save_backtest_result(
        self, result_data: dict[str, Any], request_data: dict[str, Any]
    ) -> str:
        """Save backtest result to database."""
        ...

    async def get_backtest_result(self, result_id: str) -> dict[str, Any] | None:
        """Retrieve backtest result by ID."""
        ...

    async def list_backtest_results(
        self, limit: int = 50, offset: int = 0, strategy_type: str | None = None
    ) -> list[dict[str, Any]]:
        """List backtest results with pagination."""
        ...

    async def delete_backtest_result(self, result_id: str) -> bool:
        """Delete backtest result by ID."""
        ...


@runtime_checkable
class BacktestFactoryInterface(Protocol):
    """Interface for BacktestFactory."""

    def create_controller(self) -> Any:
        """Create BacktestController."""
        ...

    def create_service(self, config: Any) -> Any:
        """Create BacktestService."""
        ...

    def create_repository(self) -> Any:
        """Create BacktestRepository."""
        ...

    def create_engine(self, config: Any, strategy: Any, **kwargs) -> Any:
        """Create BacktestEngine."""
        ...


@runtime_checkable
class TradeSimulatorInterface(Protocol):
    """Interface for TradeSimulator."""

    async def execute_order(self, order_request: Any, market_data: Any, **kwargs) -> dict[str, Any]:
        """Execute order simulation."""
        ...

    async def get_simulation_results(self) -> dict[str, Any]:
        """Get simulation results."""
        ...


@runtime_checkable
class CacheServiceInterface(Protocol):
    """Interface for caching service to decouple from Redis."""

    async def initialize(self) -> None:
        """Initialize cache service."""
        ...

    async def get(self, key: str) -> Any:
        """Get value from cache."""
        ...

    async def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache with TTL."""
        ...

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        ...

    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        ...

    async def cleanup(self) -> None:
        """Cleanup cache resources."""
        ...
