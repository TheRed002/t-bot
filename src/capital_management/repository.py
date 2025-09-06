"""
Capital Management Repository Implementations.

This module provides concrete implementations of the repository protocols
required by the CapitalService, bridging the gap between the service layer
expectations and the actual database repository implementations.
"""

from decimal import Decimal
from typing import Any

from src.capital_management.interfaces import (
    AuditRepositoryProtocol,
    CapitalRepositoryProtocol,
)
from src.core.exceptions import ServiceError
from src.database.models.capital import CapitalAllocationDB
from src.database.repository.audit import CapitalAuditLogRepository
from src.database.repository.capital import CapitalAllocationRepository


class CapitalRepository(CapitalRepositoryProtocol):
    """
    Adapter that implements CapitalRepositoryProtocol using CapitalAllocationRepository.

    This adapter bridges the service layer expectations with the actual database
    repository implementation, handling data transformation and API translation.
    """

    def __init__(self, capital_allocation_repo: CapitalAllocationRepository):
        """Initialize with the underlying repository."""
        self._repo = capital_allocation_repo

    async def create(self, allocation_data: dict[str, Any]) -> Any:
        """Create a new capital allocation."""
        # Convert dict to database model
        allocation = CapitalAllocationDB(
            id=allocation_data["id"],
            strategy_id=allocation_data["strategy_id"],
            exchange=allocation_data["exchange"],
            allocated_amount=Decimal(str(allocation_data["allocated_amount"])),
            utilized_amount=Decimal(str(allocation_data["utilized_amount"])),
            available_amount=Decimal(str(allocation_data["available_amount"])),
            allocation_percentage=float(allocation_data["allocation_percentage"]),
            last_rebalance=allocation_data.get("last_rebalance"),
        )
        return await self._repo.create(allocation)

    async def update(self, allocation_data: dict[str, Any]) -> Any:
        """Update an existing capital allocation."""
        allocation_id = allocation_data["id"]

        # Get existing allocation
        existing = await self._repo.get(allocation_id)
        if not existing:
            raise ServiceError(f"Allocation {allocation_id} not found")

        # Update fields
        existing.strategy_id = allocation_data.get("strategy_id", existing.strategy_id)
        existing.exchange = allocation_data.get("exchange", existing.exchange)
        existing.allocated_amount = Decimal(
            str(allocation_data.get("allocated_amount", existing.allocated_amount))
        )
        existing.utilized_amount = Decimal(
            str(allocation_data.get("utilized_amount", existing.utilized_amount))
        )
        existing.available_amount = Decimal(
            str(allocation_data.get("available_amount", existing.available_amount))
        )
        existing.allocation_percentage = float(
            allocation_data.get("allocation_percentage", existing.allocation_percentage)
        )
        existing.last_rebalance = allocation_data.get("last_rebalance", existing.last_rebalance)

        return await self._repo.update(existing)

    async def delete(self, allocation_id: str) -> bool:
        """Delete a capital allocation."""
        try:
            await self._repo.delete(allocation_id)
            return True
        except Exception:
            return False

    async def get_by_strategy_exchange(self, strategy_id: str, exchange: str) -> Any | None:
        """Get allocation by strategy and exchange."""
        return await self._repo.find_by_strategy_exchange(strategy_id, exchange)

    async def get_by_strategy(self, strategy_id: str) -> list[Any]:
        """Get all allocations for a strategy."""
        return await self._repo.get_by_strategy(strategy_id)

    async def get_all(self, limit: int | None = None) -> list[Any]:
        """Get all allocations with optional limit."""
        return await self._repo.get_all(limit=limit)


class AuditRepository(AuditRepositoryProtocol):
    """
    Adapter that implements AuditRepositoryProtocol using CapitalAuditLogRepository.

    This adapter provides audit logging functionality for capital management operations.
    """

    def __init__(self, audit_repo: CapitalAuditLogRepository):
        """Initialize with the underlying audit repository."""
        self._repo = audit_repo

    async def create(self, audit_data: dict[str, Any]) -> Any:
        """Create a new audit log entry."""
        from datetime import datetime, timezone

        from src.database.models.audit import CapitalAuditLog

        # Create audit log model
        audit_log = CapitalAuditLog(
            id=audit_data["id"],
            operation_id=audit_data["operation_id"],
            operation_type=audit_data["operation_type"],
            strategy_id=audit_data.get("strategy_id"),
            exchange=audit_data.get("exchange"),
            bot_id=audit_data.get("bot_id"),
            operation_description=audit_data.get("operation_description", ""),
            amount=Decimal(str(audit_data["amount"])) if audit_data.get("amount") else None,
            previous_amount=Decimal(str(audit_data["previous_amount"]))
            if audit_data.get("previous_amount")
            else None,
            new_amount=Decimal(str(audit_data["new_amount"]))
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
