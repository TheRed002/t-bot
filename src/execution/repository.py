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
    async def create_execution_record(self, execution_data: dict[str, Any]) -> dict[str, Any]:
        """Create execution record in storage."""
        pass

    @abstractmethod
    async def update_execution_record(self, execution_id: str, updates: dict[str, Any]) -> bool:
        """Update execution record."""
        pass

    @abstractmethod
    async def get_execution_record(self, execution_id: str) -> dict[str, Any] | None:
        """Get execution record by ID."""
        pass

    @abstractmethod
    async def get_executions_by_criteria(
        self, criteria: dict[str, Any], limit: int | None = None, offset: int | None = None
    ) -> list[dict[str, Any]]:
        """Get executions matching criteria."""
        pass

    @abstractmethod
    async def delete_execution_record(self, execution_id: str) -> bool:
        """Delete execution record."""
        pass


class OrderRepositoryInterface(ABC):
    """Interface for order data repository operations."""

    @abstractmethod
    async def create_order_record(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """Create order record in storage."""
        pass

    @abstractmethod
    async def update_order_status(
        self, order_id: str, status: OrderStatus, update_data: dict[str, Any] | None = None
    ) -> bool:
        """Update order status."""
        pass

    @abstractmethod
    async def get_order_record(self, order_id: str) -> dict[str, Any] | None:
        """Get order record by ID."""
        pass

    @abstractmethod
    async def get_orders_by_criteria(
        self, criteria: dict[str, Any], limit: int | None = None, offset: int | None = None
    ) -> list[dict[str, Any]]:
        """Get orders matching criteria."""
        pass

    @abstractmethod
    async def get_active_orders(
        self, symbol: str | None = None, exchange: str | None = None
    ) -> list[dict[str, Any]]:
        """Get active orders."""
        pass


class ExecutionMetricsRepositoryInterface(ABC):
    """Interface for execution metrics repository operations."""

    @abstractmethod
    async def record_execution_metrics(self, metrics_data: dict[str, Any]) -> bool:
        """Record execution metrics."""
        pass

    @abstractmethod
    async def get_execution_metrics(
        self,
        time_range_start: datetime,
        time_range_end: datetime,
        criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get execution metrics for time range."""
        pass

    @abstractmethod
    async def get_aggregated_metrics(
        self,
        aggregation_type: str,
        time_range_start: datetime,
        time_range_end: datetime,
        criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get aggregated execution metrics."""
        pass


class ExecutionAuditRepositoryInterface(ABC):
    """Interface for execution audit repository operations."""

    @abstractmethod
    async def create_audit_log(self, audit_data: dict[str, Any]) -> dict[str, Any]:
        """Create audit log entry."""
        pass

    @abstractmethod
    async def get_audit_trail(self, execution_id: str) -> list[dict[str, Any]]:
        """Get audit trail for execution."""
        pass

    @abstractmethod
    async def get_audit_logs(
        self, criteria: dict[str, Any], limit: int | None = None, offset: int | None = None
    ) -> list[dict[str, Any]]:
        """Get audit logs matching criteria."""
        pass


class DatabaseExecutionRepository(ExecutionRepositoryInterface):
    """Database implementation of execution repository."""

    def __init__(self, database_service):
        """Initialize with database service."""
        if not database_service:
            raise ValueError("Database service is required")
        self.database_service = database_service

    async def create_execution_record(self, execution_data: dict[str, Any]) -> dict[str, Any]:
        """Create execution record using database service."""
        try:
            from src.database.models import ExecutionAuditLog

            # Convert dict to model instance for proper typing
            execution_record = ExecutionAuditLog(**execution_data)
            created_record = await self.database_service.create_entity(execution_record)

            # Convert back to dict for repository interface consistency
            return {
                "id": str(created_record.id),
                "execution_id": created_record.execution_id,
                "operation_type": created_record.operation_type,
                "created_at": created_record.created_at.isoformat()
                if created_record.created_at
                else None,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to create execution record: {e}") from e

    async def update_execution_record(self, execution_id: str, updates: dict[str, Any]) -> bool:
        """Update execution record using database service."""
        try:
            from src.database.models import ExecutionAuditLog

            # Get existing record
            existing = await self.database_service.get_entity_by_field(
                ExecutionAuditLog, "execution_id", execution_id
            )
            if not existing:
                return False

            # Update through database service
            result = await self.database_service.update_entity_by_id(
                ExecutionAuditLog, existing.id, updates
            )
            return result is not None
        except Exception:
            return False

    async def get_execution_record(self, execution_id: str) -> dict[str, Any] | None:
        """Get execution record using database service."""
        try:
            from src.database.models import ExecutionAuditLog

            record = await self.database_service.get_entity_by_field(
                ExecutionAuditLog, "execution_id", execution_id
            )
            if not record:
                return None

            return {
                "id": str(record.id),
                "execution_id": record.execution_id,
                "operation_type": record.operation_type,
                "created_at": record.created_at.isoformat() if record.created_at else None,
            }
        except Exception:
            return None

    async def get_executions_by_criteria(
        self, criteria: dict[str, Any], limit: int | None = None, offset: int | None = None
    ) -> list[dict[str, Any]]:
        """Get executions using database service."""
        try:
            from src.database.models import ExecutionAuditLog

            records = await self.database_service.list_entities(
                model_class=ExecutionAuditLog,
                filters=criteria,
                limit=limit,
                offset=offset,
            )

            return [
                {
                    "id": str(record.id),
                    "execution_id": record.execution_id,
                    "operation_type": record.operation_type,
                    "created_at": record.created_at.isoformat() if record.created_at else None,
                }
                for record in records
            ]
        except Exception:
            return []

    async def delete_execution_record(self, execution_id: str) -> bool:
        """Delete execution record using database service."""
        try:
            from src.database.models import ExecutionAuditLog

            # Get existing record
            existing = await self.database_service.get_entity_by_field(
                ExecutionAuditLog, "execution_id", execution_id
            )
            if not existing:
                return False

            return await self.database_service.delete_entity_by_id(ExecutionAuditLog, existing.id)
        except Exception:
            return False


class DatabaseOrderRepository(OrderRepositoryInterface):
    """Database implementation of order repository."""

    def __init__(self, database_service):
        """Initialize with database service."""
        if not database_service:
            raise ValueError("Database service is required")
        self.database_service = database_service

    async def create_order_record(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """Create order record using database service."""
        try:
            from src.database.models import Order

            # Convert dict to model instance for proper typing
            order_record = Order(**order_data)
            created_record = await self.database_service.create_entity(order_record)

            # Convert back to dict for repository interface consistency
            return {
                "id": str(created_record.id),
                "symbol": created_record.symbol,
                "side": created_record.side,
                "status": created_record.status,
                "quantity": str(created_record.quantity),
                "price": str(created_record.price) if created_record.price else None,
                "created_at": created_record.created_at.isoformat()
                if created_record.created_at
                else None,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to create order record: {e}") from e

    async def update_order_status(
        self, order_id: str, status: OrderStatus, update_data: dict[str, Any] | None = None
    ) -> bool:
        """Update order status using database service."""
        try:
            from src.database.models import Order

            updates = {"status": status.value}
            if update_data:
                updates.update(update_data)

            result = await self.database_service.update_entity_by_id(Order, order_id, updates)
            return result is not None
        except Exception:
            return False

    async def get_order_record(self, order_id: str) -> dict[str, Any] | None:
        """Get order record using database service."""
        try:
            from src.database.models import Order

            record = await self.database_service.get_entity_by_id(Order, order_id)
            if not record:
                return None

            return {
                "id": str(record.id),
                "symbol": record.symbol,
                "side": record.side,
                "status": record.status,
                "quantity": str(record.quantity),
                "price": str(record.price) if record.price else None,
                "created_at": record.created_at.isoformat() if record.created_at else None,
            }
        except Exception:
            return None

    async def get_orders_by_criteria(
        self, criteria: dict[str, Any], limit: int | None = None, offset: int | None = None
    ) -> list[dict[str, Any]]:
        """Get orders using database service."""
        try:
            from src.database.models import Order

            records = await self.database_service.list_entities(
                model_class=Order,
                filters=criteria,
                limit=limit,
                offset=offset,
            )

            return [
                {
                    "id": str(record.id),
                    "symbol": record.symbol,
                    "side": record.side,
                    "status": record.status,
                    "quantity": str(record.quantity),
                    "price": str(record.price) if record.price else None,
                    "created_at": record.created_at.isoformat() if record.created_at else None,
                }
                for record in records
            ]
        except Exception:
            return []

    async def get_active_orders(
        self, symbol: str | None = None, exchange: str | None = None
    ) -> list[dict[str, Any]]:
        """Get active orders using database service."""
        try:
            from src.database.models import Order

            criteria = {"status": ["PENDING", "PARTIALLY_FILLED"]}

            if symbol:
                criteria["symbol"] = symbol
            if exchange:
                criteria["exchange"] = exchange

            records = await self.database_service.list_entities(
                model_class=Order,
                filters=criteria,
            )

            return [
                {
                    "id": str(record.id),
                    "symbol": record.symbol,
                    "side": record.side,
                    "status": record.status,
                    "quantity": str(record.quantity),
                    "price": str(record.price) if record.price else None,
                    "created_at": record.created_at.isoformat() if record.created_at else None,
                }
                for record in records
            ]
        except Exception:
            return []
