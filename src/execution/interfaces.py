"""
Execution Module Service Interfaces.

This module defines the service layer interfaces for the execution module,
providing clear contracts for all service operations.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Protocol

from src.core.types import (
    ExecutionAlgorithm,
    ExecutionResult,
    MarketData,
    OrderRequest,
    OrderStatus,
    Signal,
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

    async def validate_order_pre_execution_from_data(
        self,
        order_data: dict[str, Any],
        market_data: dict[str, Any],
        bot_id: str | None = None,
        risk_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate order from raw data dictionaries."""
        ...

    async def get_execution_metrics(
        self,
        bot_id: str | None = None,
        symbol: str | None = None,
        time_range_hours: int = 24,
    ) -> dict[str, Any]:
        """Get comprehensive execution metrics."""
        ...

    async def start(self) -> None:
        """Start the execution service."""
        ...

    async def stop(self) -> None:
        """Stop the execution service."""
        ...

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
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

    async def cancel_order(self, order_id: str, reason: str = "manual") -> bool:
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

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        ...

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get execution performance metrics."""
        ...


class ExecutionRepositoryServiceInterface(Protocol):
    """Interface for execution repository operations."""

    async def create_execution_record(self, execution_data: dict[str, Any]) -> dict[str, Any]:
        """Create execution record through repository."""
        ...

    async def update_execution_record(self, execution_id: str, updates: dict[str, Any]) -> bool:
        """Update execution record through repository."""
        ...

    async def get_execution_record(self, execution_id: str) -> dict[str, Any] | None:
        """Get execution record through repository."""
        ...

    async def create_order_record(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """Create order record through repository."""
        ...

    async def create_audit_log(self, audit_data: dict[str, Any]) -> dict[str, Any]:
        """Create audit log through repository."""
        ...

    async def list_orders(self, filters: dict[str, Any] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """List orders through repository."""
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


class RiskServiceInterface(Protocol):
    """Interface for risk service operations used by execution module."""

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate a trading signal."""
        ...

    async def validate_order(self, order: OrderRequest) -> bool:
        """Validate an order request."""
        ...

    async def calculate_position_size(
        self,
        signal: Signal,
        available_capital: Decimal,
        current_price: Decimal,
        method: Any = None,
    ) -> Decimal:
        """Calculate recommended position size."""
        ...

    async def calculate_risk_metrics(
        self,
        positions: list[Any],
        market_data: list[Any],
    ) -> dict[str, Any]:
        """Calculate risk metrics."""
        ...

    async def get_risk_summary(self) -> dict[str, Any]:
        """Get risk summary."""
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


class ExecutionOrchestrationServiceInterface(Protocol):
    """Interface for execution orchestration service operations."""

    async def execute_order(
        self,
        order: OrderRequest,
        market_data: MarketData,
        bot_id: str | None = None,
        strategy_name: str | None = None,
        execution_params: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute an order through the orchestration layer."""
        ...

    async def execute_order_from_data(
        self,
        order_data: dict[str, Any],
        market_data: dict[str, Any],
        bot_id: str | None = None,
        strategy_name: str | None = None,
        execution_params: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute an order from raw data dictionaries."""
        ...

    async def get_comprehensive_metrics(
        self,
        bot_id: str | None = None,
        symbol: str | None = None,
        time_range_hours: int = 24,
    ) -> dict[str, Any]:
        """Get comprehensive metrics from all execution services."""
        ...

    async def initialize(self) -> None:
        """Initialize the execution service."""
        ...

    async def cleanup(self) -> None:
        """Clean up execution service resources."""
        ...

    async def cancel_orders_by_symbol(self, symbol: str) -> None:
        """Cancel all orders for a specific symbol."""
        ...

    async def cancel_all_orders(self) -> None:
        """Cancel all active orders across all symbols."""
        ...

    async def update_order_status(
        self, order_id: str, status: str, filled_quantity: Decimal, remaining_quantity: Decimal
    ) -> None:
        """Update order status with fill information."""
        ...

    async def cancel_execution(self, execution_id: str, reason: str = "user_request") -> bool:
        """Cancel an execution through orchestration."""
        ...

    async def get_active_executions(self) -> dict[str, Any]:
        """Get all active executions."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        ...

    async def start(self) -> None:
        """Start orchestration service."""
        ...

    async def stop(self) -> None:
        """Stop orchestration service."""
        ...


class ExecutionRiskValidationServiceInterface(Protocol):
    """Interface for risk validation operations within execution module."""

    async def validate_order(self, order: OrderRequest) -> bool:
        """Validate order against risk rules."""
        ...

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate signal against risk rules."""
        ...

    async def validate_order_risk(
        self,
        order: OrderRequest,
        market_data: MarketData,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate order risk and return detailed results."""
        ...


class WebSocketServiceInterface(Protocol):
    """Interface for WebSocket connection management."""

    async def initialize_connections(self, exchanges: list[str]) -> None:
        """Initialize WebSocket connections for exchanges."""
        ...

    async def subscribe_to_order_updates(self, exchange: str, symbol: str) -> None:
        """Subscribe to order updates for a symbol on an exchange."""
        ...

    async def unsubscribe_from_order_updates(self, exchange: str, symbol: str) -> None:
        """Unsubscribe from order updates for a symbol on an exchange."""
        ...

    async def cleanup_connections(self) -> None:
        """Clean up all WebSocket connections."""
        ...

    def get_connection_status(self) -> dict[str, str]:
        """Get status of all WebSocket connections."""
        ...


class IdempotencyServiceInterface(Protocol):
    """Interface for order idempotency management."""

    async def is_duplicate_request(self, request_id: str, operation_data: dict[str, Any]) -> bool:
        """Check if request is a duplicate."""
        ...

    async def record_request(self, request_id: str, operation_data: dict[str, Any]) -> None:
        """Record a request to prevent duplicates."""
        ...

    async def cleanup_expired_requests(self) -> None:
        """Clean up expired request records."""
        ...

    async def check_position_limits(
        self,
        order: OrderRequest,
        current_positions: dict[str, Any] | None = None,
    ) -> bool:
        """Check position limits for order."""
        ...

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        ...
