"""
Capital Management Repository Implementations.

This module provides concrete implementations of the repository protocols
required by the CapitalService, bridging the gap between the service layer
expectations and the actual database repository implementations.
"""

from typing import Any

from src.capital_management.interfaces import (
    AuditRepositoryProtocol,
    CapitalRepositoryProtocol,
)
from src.core.exceptions import ServiceError
from src.database.models.capital import CapitalAllocationDB
from src.database.repository.audit import CapitalAuditLogRepository
from src.database.repository.capital import CapitalAllocationRepository
from src.utils.decimal_utils import safe_decimal_conversion


class CapitalRepository(CapitalRepositoryProtocol):
    """
    Service-layer adapter that implements CapitalRepositoryProtocol.

    This adapter properly abstracts database operations from the service layer,
    ensuring clean separation of concerns and infrastructure independence.
    """

    def __init__(self, capital_allocation_repo: CapitalAllocationRepository):
        """Initialize with the underlying repository - dependency injection."""
        if not capital_allocation_repo:
            raise ServiceError("CapitalAllocationRepository is required")
        self._repo = capital_allocation_repo

    async def create(self, allocation_data: dict[str, Any]) -> Any:
        """Create a new capital allocation - repository layer handles infrastructure."""
        try:
            # Repository layer handles database model conversion - not service concern
            allocation = CapitalAllocationDB(
                id=allocation_data["id"],
                strategy_id=allocation_data["strategy_id"],
                exchange=allocation_data["exchange"],
                allocated_amount=safe_decimal_conversion(allocation_data["allocated_amount"]),
                utilized_amount=safe_decimal_conversion(allocation_data["utilized_amount"]),
                available_amount=safe_decimal_conversion(allocation_data["available_amount"]),
                allocation_percentage=safe_decimal_conversion(
                    allocation_data["allocation_percentage"]
                ),
                last_rebalance=allocation_data.get("last_rebalance"),
            )
            return await self._repo.create(allocation)
        except Exception as e:
            # Repository layer errors should not expose database details to service
            raise ServiceError(f"Repository operation failed: {e!s}") from e

    async def update(self, allocation_data: dict[str, Any]) -> Any:
        """Update an existing capital allocation - repository handles infrastructure."""
        try:
            allocation_id = allocation_data["id"]

            # Get existing allocation
            existing = await self._repo.get(allocation_id)
            if not existing:
                raise ServiceError(f"Allocation {allocation_id} not found")

            # Update fields - repository layer responsibility
            existing.strategy_id = allocation_data.get("strategy_id", existing.strategy_id)
            existing.exchange = allocation_data.get("exchange", existing.exchange)
            existing.allocated_amount = safe_decimal_conversion(
                allocation_data.get("allocated_amount", existing.allocated_amount)
            )
            existing.utilized_amount = safe_decimal_conversion(
                allocation_data.get("utilized_amount", existing.utilized_amount)
            )
            existing.available_amount = safe_decimal_conversion(
                allocation_data.get("available_amount", existing.available_amount)
            )
            existing.allocation_percentage = safe_decimal_conversion(
                allocation_data.get("allocation_percentage", existing.allocation_percentage)
            )
            existing.last_rebalance = allocation_data.get("last_rebalance", existing.last_rebalance)

            return await self._repo.update(existing)
        except Exception as e:
            # Abstract database errors from service layer
            raise ServiceError(f"Repository update operation failed: {e!s}") from e

    async def delete(self, allocation_id: str) -> bool:
        """Delete a capital allocation - proper error abstraction."""
        try:
            await self._repo.delete(allocation_id)
            return True
        except Exception as e:
            # Don't expose database details to service layer
            raise ServiceError(f"Repository delete operation failed: {e!s}") from e

    async def get_by_strategy_exchange(self, strategy_id: str, exchange: str) -> Any | None:
        """Get allocation by strategy and exchange - proper error handling."""
        try:
            return await self._repo.find_by_strategy_exchange(strategy_id, exchange)
        except Exception as e:
            # Repository errors should be abstracted from service layer
            raise ServiceError(f"Repository query failed: {e!s}") from e

    async def get_by_strategy(self, strategy_id: str) -> list[Any]:
        """Get all allocations for a strategy - proper error handling."""
        try:
            return await self._repo.get_by_strategy(strategy_id)
        except Exception as e:
            raise ServiceError(f"Repository query failed: {e!s}") from e

    async def get_all(self, limit: int | None = None) -> list[Any]:
        """Get all allocations with optional limit - proper error handling."""
        try:
            return await self._repo.get_all(limit=limit)
        except Exception as e:
            raise ServiceError(f"Repository query failed: {e!s}") from e


class AuditRepository(AuditRepositoryProtocol):
    """
    Service-layer adapter for audit operations with proper infrastructure abstraction.

    This adapter ensures audit operations are properly abstracted from infrastructure details.
    """

    def __init__(self, audit_repo: CapitalAuditLogRepository):
        """Initialize with the underlying audit repository - dependency injection."""
        if not audit_repo:
            raise ServiceError("CapitalAuditLogRepository is required")
        self._repo = audit_repo

    async def create(self, audit_data: dict[str, Any]) -> Any:
        """Create a new audit log entry - repository handles infrastructure."""
        try:
            from datetime import datetime, timezone

            from src.database.models.audit import CapitalAuditLog

            # Repository layer handles database model conversion
            audit_log = CapitalAuditLog(
                id=audit_data["id"],
                operation_id=audit_data["operation_id"],
                operation_type=audit_data["operation_type"],
                strategy_id=audit_data.get("strategy_id"),
                exchange=audit_data.get("exchange"),
                bot_id=audit_data.get("bot_id"),
                operation_description=audit_data.get("operation_description", ""),
                amount=safe_decimal_conversion(audit_data["amount"])
                if audit_data.get("amount")
                else None,
                previous_amount=safe_decimal_conversion(audit_data["previous_amount"])
                if audit_data.get("previous_amount")
                else None,
                new_amount=safe_decimal_conversion(audit_data["new_amount"])
                if audit_data.get("new_amount")
                else None,
                operation_context=audit_data.get("operation_context", {}),
                operation_status=audit_data.get("operation_status", "completed"),
                success=audit_data.get("success", True),
                error_message=audit_data.get("error_message"),
                authorized_by=audit_data.get("authorized_by"),
                requested_at=(
                    (
                        datetime.fromisoformat(audit_data["requested_at"])
                        if isinstance(audit_data["requested_at"], str)
                        else audit_data["requested_at"]
                    )
                    if audit_data.get("requested_at")
                    else datetime.now(timezone.utc)
                ),
                executed_at=(
                    (
                        datetime.fromisoformat(audit_data["executed_at"])
                        if isinstance(audit_data["executed_at"], str)
                        else audit_data["executed_at"]
                    )
                    if audit_data.get("executed_at")
                    else None
                ),
                source_component=audit_data.get("source_component", "CapitalService"),
                correlation_id=audit_data.get("correlation_id"),
            )

            return await self._repo.create(audit_log)
        except Exception as e:
            # Abstract database errors from service layer
            raise ServiceError(f"Audit repository operation failed: {e!s}") from e
