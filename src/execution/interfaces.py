"""
Execution Module Service Interfaces.

This module defines the service layer interfaces for the execution module,
providing clear contracts for all service operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

from src.core.types import (
    ExecutionAlgorithm,
    ExecutionResult,
    MarketData,
    OrderRequest,
    OrderStatus,
)
from src.execution.types import ExecutionInstruction


class ExecutionServiceInterface(Protocol):
    """Interface for execution service operations."""

    async def record_trade_execution(
        self,
        execution_result: ExecutionResult,
        market_data: MarketData,
        bot_id: str | None = None,
        strategy_name: str | None = None,
        pre_trade_analysis: dict[str, Any] | None = None,
        post_trade_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a completed trade execution with full audit trail."""
        ...

    async def validate_order_pre_execution(
        self,
        order: OrderRequest,
        market_data: MarketData,
        bot_id: str | None = None,
        risk_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Comprehensive order validation before execution."""
        ...

    async def get_execution_metrics(
        self,
        bot_id: str | None = None,
        symbol: str | None = None,
        time_range_hours: int = 24,
    ) -> dict[str, Any]:
        """Get comprehensive execution metrics."""
        ...


class OrderManagementServiceInterface(Protocol):
    """Interface for order management operations."""

    async def create_managed_order(
        self,
        order_request: OrderRequest,
        execution_id: str,
        timeout_minutes: int | None = None,
        callbacks: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create and manage an order."""
        ...

    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Update order status."""
        ...

    async def cancel_order(
        self,
        order_id: str,
        reason: str = "manual"
    ) -> bool:
        """Cancel a managed order."""
        ...

    async def get_order_metrics(
        self,
        symbol: str | None = None,
        time_range_hours: int = 24,
    ) -> dict[str, Any]:
        """Get order management metrics."""
        ...


class ExecutionEngineServiceInterface(Protocol):
    """Interface for execution engine operations."""

    async def execute_instruction(
        self,
        instruction: ExecutionInstruction,
        market_data: MarketData,
        bot_id: str | None = None,
        strategy_name: str | None = None,
    ) -> ExecutionResult:
        """Execute a trading instruction."""
        ...

    async def get_active_executions(self) -> dict[str, Any]:
        """Get currently active executions."""
        ...

    async def cancel_execution(
        self,
        execution_id: str
    ) -> bool:
        """Cancel an active execution."""
        ...

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get execution performance metrics."""
        ...


class RiskValidationServiceInterface(Protocol):
    """Interface for risk validation operations."""

    async def validate_order_risk(
        self,
        order: OrderRequest,
        market_data: MarketData,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate order against risk rules."""
        ...

    async def check_position_limits(
        self,
        order: OrderRequest,
        current_positions: dict[str, Any] | None = None,
    ) -> bool:
        """Check if order violates position limits."""
        ...


class ExecutionAlgorithmFactoryInterface(Protocol):
    """Interface for execution algorithm factory."""

    def create_algorithm(self, algorithm_type: ExecutionAlgorithm) -> Any:
        """Create an execution algorithm instance."""
        ...

    def get_available_algorithms(self) -> list[ExecutionAlgorithm]:
        """Get list of available algorithm types."""
        ...

    def is_algorithm_available(self, algorithm_type: ExecutionAlgorithm) -> bool:
        """Check if algorithm type is available."""
        ...


class ExecutionAlgorithmInterface(ABC):
    """Abstract base class for execution algorithms."""

    @abstractmethod
    async def execute(
        self,
        instruction: ExecutionInstruction,
        exchange_factory: Any = None,
        risk_manager: Any = None,
    ) -> ExecutionResult:
        """Execute a trading instruction."""
        ...

    @abstractmethod
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        ...

    @abstractmethod
    async def cancel(self, execution_id: str) -> dict[str, Any]:
        """Cancel an active execution (alias for cancel_execution)."""
        ...

    @abstractmethod
    async def get_status(self, execution_id: str) -> dict[str, Any]:
        """Get the status of an execution."""
        ...

    @abstractmethod
    def get_algorithm_type(self) -> ExecutionAlgorithm:
        """Get the algorithm type."""
        ...
