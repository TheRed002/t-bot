"""
Strategy Interfaces - Abstract interfaces for different strategy types and backtesting.

This module provides:
- Backtesting interface integration
- Strategy type-specific interfaces
- Performance monitoring interfaces
- Risk management interfaces
"""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, TypeVar

from src.core.types import (
    MarketData,
    MarketRegime,
    Position,
    Signal,
    StrategyConfig,
    StrategyStatus,
    StrategyType,
)

# Type variables for generic interfaces
T = TypeVar("T")
StrategyT = TypeVar("StrategyT", bound="BaseStrategyInterface")


class BacktestingInterface(Protocol):
    """Protocol for backtesting integration with strategies."""

    async def prepare_for_backtest(self, config: dict[str, Any]) -> None:
        """
        Prepare strategy for backtesting mode.

        Args:
            config: Backtesting configuration
        """
        ...

    async def process_historical_data(self, data: MarketData) -> list[Signal]:
        """
        Process historical market data during backtesting.

        Args:
            data: Historical market data point

        Returns:
            List of generated signals
        """
        ...

    async def simulate_trade_execution(
        self, signal: Signal, market_data: MarketData
    ) -> dict[str, Any]:
        """
        Simulate trade execution for backtesting.

        Args:
            signal: Signal to execute
            market_data: Current market data

        Returns:
            Simulation result dictionary
        """
        ...

    async def get_backtest_metrics(self) -> dict[str, Any]:
        """
        Get strategy-specific metrics for backtesting.

        Returns:
            Dictionary with backtest metrics
        """
        ...

    async def reset_backtest_state(self) -> None:
        """Reset strategy state for new backtest run."""
        ...


class PerformanceMonitoringInterface(Protocol):
    """Protocol for strategy performance monitoring."""

    def update_performance_metrics(self, trade_result: dict[str, Any]) -> None:
        """
        Update performance metrics with trade result.

        Args:
            trade_result: Result of executed trade
        """
        ...

    def get_real_time_metrics(self) -> dict[str, Any]:
        """
        Get real-time performance metrics.

        Returns:
            Current performance metrics
        """
        ...

    def calculate_risk_adjusted_returns(self) -> dict[str, Decimal]:
        """
        Calculate risk-adjusted return metrics.

        Returns:
            Risk-adjusted metrics (Sharpe, Sortino, etc.)
        """
        ...

    def get_drawdown_analysis(self) -> dict[str, Any]:
        """
        Get detailed drawdown analysis.

        Returns:
            Drawdown analysis data
        """
        ...


class RiskManagementInterface(Protocol):
    """Protocol for strategy risk management integration."""

    async def validate_risk_limits(self, signal: Signal) -> bool:
        """
        Validate signal against risk limits.

        Args:
            signal: Signal to validate

        Returns:
            True if signal passes risk checks
        """
        ...

    def calculate_position_size(self, signal: Signal, account_balance: Decimal) -> Decimal:
        """
        Calculate appropriate position size.

        Args:
            signal: Trading signal
            account_balance: Current account balance

        Returns:
            Position size
        """
        ...

    def should_close_position(self, position: Position, current_data: MarketData) -> bool:
        """
        Determine if position should be closed for risk management.

        Args:
            position: Current position
            current_data: Current market data

        Returns:
            True if position should be closed
        """
        ...


class BaseStrategyInterface(ABC):
    """
    Base interface that all strategies must implement.

    This interface defines the core contract for all strategy implementations,
    including backtesting capabilities and performance monitoring.
    """

    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the strategy name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Get the strategy version."""
        pass

    @property
    @abstractmethod
    def status(self) -> StrategyStatus:
        """Get the current strategy status."""
        pass

    # Core strategy methods
    @abstractmethod
    async def initialize(self, config: StrategyConfig) -> None:
        """
        Initialize the strategy with configuration.

        Args:
            config: Strategy configuration
        """
        pass

    @abstractmethod
    async def generate_signals(self, data: MarketData) -> list[Signal]:
        """
        Generate trading signals from market data.

        Args:
            data: Market data for signal generation

        Returns:
            List of trading signals
        """
        pass

    @abstractmethod
    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate a generated signal.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid
        """
        pass

    @abstractmethod
    def get_position_size(self, signal: Signal) -> Decimal:
        """
        Calculate position size for a signal.

        Args:
            signal: Signal for position sizing

        Returns:
            Position size
        """
        pass

    @abstractmethod
    def should_exit(self, position: Position, data: MarketData) -> bool:
        """
        Determine if a position should be exited.

        Args:
            position: Current position
            data: Current market data

        Returns:
            True if position should be exited
        """
        pass

    # Lifecycle methods
    @abstractmethod
    async def start(self) -> None:
        """Start the strategy."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the strategy."""
        pass

    @abstractmethod
    async def pause(self) -> None:
        """Pause the strategy."""
        pass

    @abstractmethod
    async def resume(self) -> None:
        """Resume the strategy."""
        pass

    # Backtesting interface
    @abstractmethod
    async def prepare_for_backtest(self, config: dict[str, Any]) -> None:
        """Prepare strategy for backtesting mode."""
        pass

    @abstractmethod
    async def process_historical_data(self, data: MarketData) -> list[Signal]:
        """Process historical data during backtesting."""
        pass

    @abstractmethod
    async def get_backtest_metrics(self) -> dict[str, Any]:
        """Get strategy-specific backtest metrics."""
        pass

    # Performance monitoring
    @abstractmethod
    def get_performance_summary(self) -> dict[str, Any]:
        """Get strategy performance summary."""
        pass

    @abstractmethod
    def get_real_time_metrics(self) -> dict[str, Any]:
        """Get real-time performance metrics."""
        pass

    @abstractmethod
    async def get_state(self) -> dict[str, Any]:
        """
        Get current strategy state.
        
        Returns:
            Dictionary containing strategy state information
        """
        pass


class TrendStrategyInterface(BaseStrategyInterface):
    """Interface for trend-following strategies."""

    @abstractmethod
    def calculate_trend_strength(self, data: MarketData) -> float:
        """
        Calculate trend strength indicator.

        Args:
            data: Market data

        Returns:
            Trend strength (0.0 to 1.0)
        """
        pass

    @abstractmethod
    def identify_trend_direction(self, data: MarketData) -> int:
        """
        Identify trend direction.

        Args:
            data: Market data

        Returns:
            1 for uptrend, -1 for downtrend, 0 for sideways
        """
        pass

    @abstractmethod
    def get_trend_confirmation(self, data: MarketData) -> bool:
        """
        Check for trend confirmation signals.

        Args:
            data: Market data

        Returns:
            True if trend is confirmed
        """
        pass


class MeanReversionStrategyInterface(BaseStrategyInterface):
    """Interface for mean reversion strategies."""

    @abstractmethod
    def calculate_mean_deviation(self, data: MarketData) -> float:
        """
        Calculate deviation from mean.

        Args:
            data: Market data

        Returns:
            Deviation from mean
        """
        pass

    @abstractmethod
    def is_oversold(self, data: MarketData) -> bool:
        """
        Check if asset is oversold.

        Args:
            data: Market data

        Returns:
            True if oversold
        """
        pass

    @abstractmethod
    def is_overbought(self, data: MarketData) -> bool:
        """
        Check if asset is overbought.

        Args:
            data: Market data

        Returns:
            True if overbought
        """
        pass

    @abstractmethod
    def calculate_reversion_probability(self, data: MarketData) -> float:
        """
        Calculate probability of mean reversion.

        Args:
            data: Market data

        Returns:
            Reversion probability (0.0 to 1.0)
        """
        pass


class ArbitrageStrategyInterface(BaseStrategyInterface):
    """Interface for arbitrage strategies."""

    @abstractmethod
    async def identify_arbitrage_opportunities(
        self, market_data_sources: list[MarketData]
    ) -> list[dict[str, Any]]:
        """
        Identify arbitrage opportunities across markets.

        Args:
            market_data_sources: List of market data from different sources

        Returns:
            List of arbitrage opportunities
        """
        pass

    @abstractmethod
    def calculate_profit_potential(self, opportunity: dict[str, Any]) -> Decimal:
        """
        Calculate profit potential for an arbitrage opportunity.

        Args:
            opportunity: Arbitrage opportunity data

        Returns:
            Expected profit
        """
        pass

    @abstractmethod
    def validate_arbitrage_execution(self, opportunity: dict[str, Any]) -> bool:
        """
        Validate if arbitrage can be executed.

        Args:
            opportunity: Arbitrage opportunity

        Returns:
            True if execution is viable
        """
        pass


class MarketMakingStrategyInterface(BaseStrategyInterface):
    """Interface for market making strategies."""

    @abstractmethod
    def calculate_optimal_spread(self, data: MarketData) -> tuple[Decimal, Decimal]:
        """
        Calculate optimal bid-ask spread.

        Args:
            data: Market data

        Returns:
            Tuple of (bid_price, ask_price)
        """
        pass

    @abstractmethod
    def manage_inventory(self, current_position: Position) -> dict[str, Any]:
        """
        Manage inventory position.

        Args:
            current_position: Current market making position

        Returns:
            Inventory management actions
        """
        pass

    @abstractmethod
    def calculate_quote_adjustment(self, market_impact: float) -> float:
        """
        Calculate quote adjustment based on market impact.

        Args:
            market_impact: Measured market impact

        Returns:
            Quote adjustment factor
        """
        pass


class StrategyFactoryInterface(ABC):
    """Interface for strategy factories."""

    @abstractmethod
    async def create_strategy(
        self, strategy_type: StrategyType, config: StrategyConfig
    ) -> BaseStrategyInterface:
        """
        Create a strategy instance.

        Args:
            strategy_type: Type of strategy to create
            config: Strategy configuration

        Returns:
            Strategy instance
        """
        pass

    @abstractmethod
    def get_supported_strategies(self) -> list[StrategyType]:
        """
        Get list of supported strategy types.

        Returns:
            List of supported strategy types
        """
        pass

    @abstractmethod
    def validate_strategy_requirements(
        self, strategy_type: StrategyType, config: StrategyConfig
    ) -> bool:
        """
        Validate strategy requirements.

        Args:
            strategy_type: Strategy type
            config: Strategy configuration

        Returns:
            True if requirements are met
        """
        pass


class BacktestingServiceInterface(Protocol):
    """Interface for backtesting service integration."""

    async def run_backtest(
        self, strategy: BaseStrategyInterface, config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Run backtest for a strategy.

        Args:
            strategy: Strategy to backtest
            config: Backtest configuration

        Returns:
            Backtest results
        """
        ...

    async def validate_backtest_config(self, config: dict[str, Any]) -> bool:
        """
        Validate backtest configuration.

        Args:
            config: Backtest configuration

        Returns:
            True if configuration is valid
        """
        ...

    async def get_historical_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> list[MarketData]:
        """
        Get historical data for backtesting.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date

        Returns:
            List of historical market data
        """
        ...


class StrategyRegistryInterface(Protocol):
    """Interface for strategy registry."""

    def register_strategy(self, strategy_id: str, strategy: BaseStrategyInterface) -> None:
        """
        Register a strategy.

        Args:
            strategy_id: Unique strategy identifier
            strategy: Strategy instance
        """
        ...

    def get_strategy(self, strategy_id: str) -> BaseStrategyInterface | None:
        """
        Get registered strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Strategy instance if found
        """
        ...

    def list_strategies(self) -> list[str]:
        """
        List all registered strategy IDs.

        Returns:
            List of strategy IDs
        """
        ...

    def remove_strategy(self, strategy_id: str) -> bool:
        """
        Remove strategy from registry.

        Args:
            strategy_id: Strategy identifier

        Returns:
            True if strategy was removed
        """
        ...


class MarketDataProviderInterface(Protocol):
    """Interface for market data access."""

    async def get_current_price(self, symbol: str) -> Decimal | None:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if not available
        """
        ...

    async def get_market_regime(self, symbol: str) -> MarketRegime:
        """
        Get current market regime.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current market regime
        """
        ...


class StrategyDataRepositoryInterface(Protocol):
    """
    Interface for strategy data persistence.
    
    This protocol defines the contract for accessing strategy-related
    data from the database without exposing database implementation details.
    """

    async def load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None:
        """
        Load persisted strategy state.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Strategy state or None if not found
        """
        ...

    async def save_strategy_state(self, strategy_id: str, state: dict[str, Any]) -> None:
        """
        Save strategy state.
        
        Args:
            strategy_id: Strategy identifier
            state: Strategy state to persist
        """
        ...

    async def get_strategy_trades(
        self,
        strategy_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> list[dict[str, Any]]:
        """
        Get trades for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of trades as dictionaries
        """
        ...

    async def save_trade(self, strategy_id: str, trade: dict[str, Any]) -> None:
        """
        Save a trade.
        
        Args:
            strategy_id: Strategy identifier
            trade: Trade data to save
        """
        ...

    async def get_strategy_positions(self, strategy_id: str) -> list[dict[str, Any]]:
        """
        Get current positions for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            List of positions as dictionaries
        """
        ...

    async def save_performance_metrics(
        self,
        strategy_id: str,
        metrics: dict[str, Any],
        timestamp: datetime | None = None
    ) -> None:
        """
        Save performance metrics.
        
        Args:
            strategy_id: Strategy identifier
            metrics: Performance metrics
            timestamp: Optional timestamp (defaults to now)
        """
        ...

    async def load_performance_history(
        self,
        strategy_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> list[dict[str, Any]]:
        """
        Load historical performance metrics.
        
        Args:
            strategy_id: Strategy identifier
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of historical metrics
        """
        ...


# Import typing for better type hints
