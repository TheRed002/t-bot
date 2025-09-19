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


class CapitalRepository(CapitalRepositoryProtocol):
    """
    Service-layer adapter that implements CapitalRepositoryProtocol.

    This adapter properly abstracts database operations from the service layer,
    ensuring clean separation of concerns and infrastructure independence.
    """

    def __init__(self, capital_allocation_repo: Any = None):
        """Initialize the repository adapter."""
        if capital_allocation_repo is None:
            from src.core.exceptions import ServiceError
            raise ServiceError("CapitalAllocationRepository is required")
        self._repo = capital_allocation_repo

    async def create(self, allocation_data: dict[str, Any]) -> Any:
        """Create a new capital allocation."""
        if not self._repo:
            return allocation_data

        try:
            # Convert dict to CapitalAllocationDB object
            from src.database.models.capital import CapitalAllocationDB
            from decimal import Decimal

            # Convert string amounts to Decimal for financial precision
            converted_data = allocation_data.copy()
            for field in ["allocated_amount", "utilized_amount", "available_amount", "reserved_amount"]:
                if field in converted_data and converted_data[field] is not None:
                    if isinstance(converted_data[field], str):
                        converted_data[field] = Decimal(converted_data[field])

            allocation_obj = CapitalAllocationDB(**converted_data)
            return await self._repo.create(allocation_obj)
        except Exception as e:
            from src.core.exceptions import ServiceError
            raise ServiceError(f"Repository operation failed: {e}") from e

    async def update(self, allocation_data: dict[str, Any]) -> Any:
        """Update an existing capital allocation."""
        if not self._repo:
            return allocation_data

        try:
            # Get existing allocation
            allocation_id = allocation_data.get("id")
            existing_allocation = await self._repo.get(allocation_id)

            if not existing_allocation:
                from src.core.exceptions import ServiceError
                raise ServiceError(f"Allocation {allocation_id} not found")

            # Update existing allocation fields with proper type conversion
            from decimal import Decimal
            for key, value in allocation_data.items():
                if hasattr(existing_allocation, key):
                    # Convert string amounts to Decimal for financial precision
                    if key in ["allocated_amount", "utilized_amount", "available_amount", "reserved_amount"]:
                        if value is not None and isinstance(value, str):
                            value = Decimal(value)
                    setattr(existing_allocation, key, value)

            return await self._repo.update(existing_allocation)
        except Exception as e:
            from src.core.exceptions import ServiceError
            raise ServiceError(f"Repository update operation failed: {e}") from e

    async def delete(self, allocation_id: str) -> bool:
        """Delete a capital allocation."""
        if not self._repo:
            return True
        try:
            await self._repo.delete(allocation_id)
            return True  # Successful deletion
        except Exception as e:
            from src.core.exceptions import ServiceError
            raise ServiceError(f"Repository delete operation failed: {e}") from e

    async def get_by_strategy_exchange(self, strategy_id: str, exchange: str) -> Any | None:
        """Get allocation by strategy and exchange."""
        if not self._repo:
            return None
        try:
            return await self._repo.get_by_strategy_exchange(strategy_id, exchange)
        except Exception as e:
            from src.core.exceptions import ServiceError
            raise ServiceError(f"Repository operation failed: {e}") from e

    async def get_by_strategy(self, strategy_id: str) -> list[Any]:
        """Get all allocations for a strategy."""
        if not self._repo:
            return []
        try:
            return await self._repo.get_by_strategy(strategy_id)
        except Exception as e:
            from src.core.exceptions import ServiceError
            raise ServiceError(f"Repository operation failed: {e}") from e

    async def get_all(self, limit: int | None = None) -> list[Any]:
        """Get all allocations with optional limit."""
        if not self._repo:
            return []
        try:
            return await self._repo.get_all(limit=limit)
        except Exception as e:
            from src.core.exceptions import ServiceError
            raise ServiceError(f"Repository operation failed: {e}") from e


class AuditRepository(AuditRepositoryProtocol):
    """
    Service-layer adapter for audit operations with proper infrastructure abstraction.

    This adapter ensures audit operations are properly abstracted from infrastructure details.
    """

    def __init__(self, audit_repo: Any = None):
        """Initialize the audit repository adapter."""
        if audit_repo is None:
            from src.core.exceptions import ServiceError
            raise ServiceError("CapitalAuditLogRepository is required")
        self._repo = audit_repo

    async def create(self, audit_data: dict[str, Any]) -> Any:
        """Create a new audit log entry."""
        if not self._repo:
            return audit_data

        try:
            # Convert dict to CapitalAuditLog object
            from src.database.models.audit import CapitalAuditLog
            from decimal import Decimal
            from datetime import datetime, timezone

            # Convert string amounts to Decimal for financial precision
            converted_data = audit_data.copy()
            for field in ["amount", "previous_amount", "new_amount"]:
                if field in converted_data and converted_data[field] is not None:
                    if isinstance(converted_data[field], str):
                        converted_data[field] = Decimal(converted_data[field])

            # Convert string datetimes to datetime objects
            from dateutil.parser import parse as parse_datetime
            for field in ["requested_at", "executed_at"]:
                if field in converted_data and converted_data[field] is not None:
                    if isinstance(converted_data[field], str):
                        converted_data[field] = parse_datetime(converted_data[field])

            # Set defaults for required fields if not provided
            if "operation_description" not in converted_data:
                converted_data["operation_description"] = ""
            if "source_component" not in converted_data:
                converted_data["source_component"] = "CapitalService"
            if "requested_at" not in converted_data:
                converted_data["requested_at"] = datetime.now(timezone.utc)
            if "operation_status" not in converted_data:
                converted_data["operation_status"] = "completed"
            if "success" not in converted_data:
                converted_data["success"] = True

            audit_obj = CapitalAuditLog(**converted_data)
            return await self._repo.create(audit_obj)
        except Exception as e:
            from src.core.exceptions import ServiceError
            raise ServiceError(f"Repository operation failed: {e}") from e
