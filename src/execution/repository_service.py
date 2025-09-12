"""
Execution Repository Service.

This service provides abstraction over repository operations for the execution module,
following proper service layer patterns and avoiding direct database access in business logic.
"""

from typing import Any

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError
from src.execution.interfaces import ExecutionRepositoryServiceInterface
from src.execution.repository import (
    ExecutionRepositoryInterface, 
    OrderRepositoryInterface,
    ExecutionAuditRepositoryInterface
)


class ExecutionRepositoryService(BaseService, ExecutionRepositoryServiceInterface):
    """Service layer for execution repository operations."""

    def __init__(
        self,
        execution_repository: ExecutionRepositoryInterface,
        order_repository: OrderRepositoryInterface,
        audit_repository: ExecutionAuditRepositoryInterface,
        correlation_id: str | None = None,
    ) -> None:
        """
        Initialize repository service.

        Args:
            execution_repository: Repository for execution records
            order_repository: Repository for order records
            audit_repository: Repository for audit logs
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="ExecutionRepositoryService",
            correlation_id=correlation_id,
        )

        if not execution_repository:
            raise ValueError("ExecutionRepository is required")
        if not order_repository:
            raise ValueError("OrderRepository is required")
        if not audit_repository:
            raise ValueError("ExecutionAuditRepository is required")

        self.execution_repository = execution_repository
        self.order_repository = order_repository
        self.audit_repository = audit_repository

    async def _do_start(self) -> None:
        """Start repository service."""
        self._logger.info("ExecutionRepositoryService started")

    async def create_execution_record(self, execution_data: dict[str, Any]) -> dict[str, Any]:
        """Create execution record through repository."""
        try:
            return await self.execution_repository.create_execution_record(execution_data)
        except Exception as e:
            self._logger.error(f"Failed to create execution record: {e}")
            raise ServiceError(f"Execution record creation failed: {e}") from e

    async def update_execution_record(self, execution_id: str, updates: dict[str, Any]) -> bool:
        """Update execution record through repository."""
        try:
            return await self.execution_repository.update_execution_record(execution_id, updates)
        except Exception as e:
            self._logger.error(f"Failed to update execution record: {e}")
            raise ServiceError(f"Execution record update failed: {e}") from e

    async def get_execution_record(self, execution_id: str) -> dict[str, Any] | None:
        """Get execution record through repository."""
        try:
            return await self.execution_repository.get_execution_record(execution_id)
        except Exception as e:
            self._logger.error(f"Failed to get execution record: {e}")
            raise ServiceError(f"Execution record retrieval failed: {e}") from e

    async def create_order_record(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """Create order record through repository."""
        try:
            return await self.order_repository.create_order_record(order_data)
        except Exception as e:
            self._logger.error(f"Failed to create order record: {e}")
            raise ServiceError(f"Order record creation failed: {e}") from e

    async def create_audit_log(self, audit_data: dict[str, Any]) -> dict[str, Any]:
        """Create audit log through repository."""
        try:
            return await self.audit_repository.create_audit_log(audit_data)
        except Exception as e:
            self._logger.error(f"Failed to create audit log: {e}")
            raise ServiceError(f"Audit log creation failed: {e}") from e

    async def list_orders(self, filters: dict[str, Any] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """List orders through repository."""
        try:
            criteria = filters or {}
            return await self.order_repository.get_orders_by_criteria(criteria, limit=limit)
        except Exception as e:
            self._logger.error(f"Failed to list orders: {e}")
            raise ServiceError(f"Order listing failed: {e}") from e

    async def get_active_orders(self, symbol: str | None = None, exchange: str | None = None) -> list[dict[str, Any]]:
        """Get active orders through repository."""
        try:
            return await self.order_repository.get_active_orders(symbol=symbol, exchange=exchange)
        except Exception as e:
            self._logger.error(f"Failed to get active orders: {e}")
            raise ServiceError(f"Active orders retrieval failed: {e}") from e