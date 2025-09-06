"""Backtesting service interfaces for dependency injection."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.core.base.interfaces import HealthCheckResult

if TYPE_CHECKING:
    from src.backtesting.service import BacktestRequest, BacktestResult


@runtime_checkable
class BacktestServiceInterface(Protocol):
    """Interface for BacktestService."""

    async def initialize(self) -> None:
        """Initialize the service."""
        ...

    async def run_backtest(self, request: "BacktestRequest") -> "BacktestResult":
        """Run a backtest."""
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
