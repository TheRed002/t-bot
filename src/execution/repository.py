"""
Execution Module Repository Interface.

This repository provides data access abstraction for execution operations,
following proper repository patterns without business logic.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.core.types import OrderStatus


class ExecutionRepositoryInterface(ABC):
    """Interface for execution data repository operations."""

    @abstractmethod
    async def create_execution_record(
        self,
        execution_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create execution record in storage."""
        pass

    @abstractmethod
    async def update_execution_record(
        self,
        execution_id: str,
        updates: dict[str, Any]
    ) -> bool:
        """Update execution record."""
        pass

    @abstractmethod
    async def get_execution_record(
        self,
        execution_id: str
    ) -> dict[str, Any] | None:
        """Get execution record by ID."""
        pass

    @abstractmethod
    async def get_executions_by_criteria(
        self,
        criteria: dict[str, Any],
        limit: int | None = None,
        offset: int | None = None
    ) -> list[dict[str, Any]]:
        """Get executions matching criteria."""
        pass

    @abstractmethod
    async def delete_execution_record(
        self,
        execution_id: str
    ) -> bool:
        """Delete execution record."""
        pass


class OrderRepositoryInterface(ABC):
    """Interface for order data repository operations."""

    @abstractmethod
    async def create_order_record(
        self,
        order_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create order record in storage."""
        pass

    @abstractmethod
    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        update_data: dict[str, Any] | None = None
    ) -> bool:
        """Update order status."""
        pass

    @abstractmethod
    async def get_order_record(
        self,
        order_id: str
    ) -> dict[str, Any] | None:
        """Get order record by ID."""
        pass

    @abstractmethod
    async def get_orders_by_criteria(
        self,
        criteria: dict[str, Any],
        limit: int | None = None,
        offset: int | None = None
    ) -> list[dict[str, Any]]:
        """Get orders matching criteria."""
        pass

    @abstractmethod
    async def get_active_orders(
        self,
        symbol: str | None = None,
        exchange: str | None = None
    ) -> list[dict[str, Any]]:
        """Get active orders."""
        pass


class ExecutionMetricsRepositoryInterface(ABC):
    """Interface for execution metrics repository operations."""

    @abstractmethod
    async def record_execution_metrics(
        self,
        metrics_data: dict[str, Any]
    ) -> bool:
        """Record execution metrics."""
        pass

    @abstractmethod
    async def get_execution_metrics(
        self,
        time_range_start: datetime,
        time_range_end: datetime,
        criteria: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get execution metrics for time range."""
        pass

    @abstractmethod
    async def get_aggregated_metrics(
        self,
        aggregation_type: str,
        time_range_start: datetime,
        time_range_end: datetime,
        criteria: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get aggregated execution metrics."""
        pass


class ExecutionAuditRepositoryInterface(ABC):
    """Interface for execution audit repository operations."""

    @abstractmethod
    async def create_audit_log(
        self,
        audit_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create audit log entry."""
        pass

    @abstractmethod
    async def get_audit_trail(
        self,
        execution_id: str
    ) -> list[dict[str, Any]]:
        """Get audit trail for execution."""
        pass

    @abstractmethod
    async def get_audit_logs(
        self,
        criteria: dict[str, Any],
        limit: int | None = None,
        offset: int | None = None
    ) -> list[dict[str, Any]]:
        """Get audit logs matching criteria."""
        pass


class DatabaseExecutionRepository(ExecutionRepositoryInterface):
    """Database implementation of execution repository."""

    def __init__(self, database_service):
        """Initialize with database service."""
        self.database_service = database_service

    async def create_execution_record(
        self,
        execution_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create execution record using database service."""
        # Use database service for actual storage
        return await self.database_service.create_entity_from_dict(
            "executions",
            execution_data
        )

    async def update_execution_record(
        self,
        execution_id: str,
        updates: dict[str, Any]
    ) -> bool:
        """Update execution record using database service."""
        return await self.database_service.update_entity(
            "executions",
            execution_id,
            updates
        )

    async def get_execution_record(
        self,
        execution_id: str
    ) -> dict[str, Any] | None:
        """Get execution record using database service."""
        return await self.database_service.get_entity_by_id(
            "executions",
            execution_id
        )

    async def get_executions_by_criteria(
        self,
        criteria: dict[str, Any],
        limit: int | None = None,
        offset: int | None = None
    ) -> list[dict[str, Any]]:
        """Get executions using database service."""
        return await self.database_service.query_entities(
            "executions",
            criteria,
            limit=limit,
            offset=offset
        )

    async def delete_execution_record(
        self,
        execution_id: str
    ) -> bool:
        """Delete execution record using database service."""
        return await self.database_service.delete_entity(
            "executions",
            execution_id
        )


class DatabaseOrderRepository(OrderRepositoryInterface):
    """Database implementation of order repository."""

    def __init__(self, database_service):
        """Initialize with database service."""
        self.database_service = database_service

    async def create_order_record(
        self,
        order_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Create order record using database service."""
        return await self.database_service.create_entity_from_dict(
            "orders",
            order_data
        )

    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        update_data: dict[str, Any] | None = None
    ) -> bool:
        """Update order status using database service."""
        updates = {"status": status.value}
        if update_data:
            updates.update(update_data)

        return await self.database_service.update_entity(
            "orders",
            order_id,
            updates
        )

    async def get_order_record(
        self,
        order_id: str
    ) -> dict[str, Any] | None:
        """Get order record using database service."""
        return await self.database_service.get_entity_by_id(
            "orders",
            order_id
        )

    async def get_orders_by_criteria(
        self,
        criteria: dict[str, Any],
        limit: int | None = None,
        offset: int | None = None
    ) -> list[dict[str, Any]]:
        """Get orders using database service."""
        return await self.database_service.query_entities(
            "orders",
            criteria,
            limit=limit,
            offset=offset
        )

    async def get_active_orders(
        self,
        symbol: str | None = None,
        exchange: str | None = None
    ) -> list[dict[str, Any]]:
        """Get active orders using database service."""
        criteria = {
            "status": {"in": ["PENDING", "PARTIALLY_FILLED"]}
        }

        if symbol:
            criteria["symbol"] = symbol
        if exchange:
            criteria["exchange"] = exchange

        return await self.database_service.query_entities(
            "orders",
            criteria
        )
