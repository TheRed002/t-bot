"""Audit repositories implementation."""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.audit import (
    CapitalAuditLog,
    ExecutionAuditLog,
    PerformanceAuditLog,
    RiskAuditLog,
)
from src.database.repository.core_compliant_base import DatabaseRepository


class CapitalAuditLogRepository(DatabaseRepository[CapitalAuditLog, str]):
    """Repository for CapitalAuditLog entities."""

    def __init__(self, session: AsyncSession):
        """Initialize capital audit log repository."""
        super().__init__(
            session=session,
            model=CapitalAuditLog,
            entity_type=CapitalAuditLog,
            key_type=str,
            name="CapitalAuditLogRepository",
        )

    async def get_by_strategy(self, strategy_id: str) -> list[CapitalAuditLog]:
        """Get audit logs by strategy."""
        return await self.get_all(filters={"strategy_id": strategy_id}, order_by="-timestamp")

    async def get_by_exchange(self, exchange: str) -> list[CapitalAuditLog]:
        """Get audit logs by exchange."""
        return await self.get_all(filters={"exchange": exchange}, order_by="-timestamp")

    async def get_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[CapitalAuditLog]:
        """Get audit logs by date range."""
        return await self.get_all(
            filters={"timestamp": {"gte": start_date, "lte": end_date}}, order_by="-timestamp"
        )


class ExecutionAuditLogRepository(DatabaseRepository[ExecutionAuditLog, str]):
    """Repository for ExecutionAuditLog entities."""

    def __init__(self, session: AsyncSession):
        """Initialize execution audit log repository."""
        super().__init__(
            session=session,
            model=ExecutionAuditLog,
            entity_type=ExecutionAuditLog,
            key_type=str,
            name="ExecutionAuditLogRepository",
        )

    async def get_by_order(self, order_id: str) -> list[ExecutionAuditLog]:
        """Get audit logs by order."""
        return await self.get_all(filters={"order_id": order_id}, order_by="-timestamp")

    async def get_by_execution_status(self, status: str) -> list[ExecutionAuditLog]:
        """Get audit logs by execution status."""
        return await self.get_all(filters={"execution_status": status}, order_by="-timestamp")

    async def get_failed_executions(self) -> list[ExecutionAuditLog]:
        """Get failed execution logs."""
        return await self.get_all(filters={"execution_status": "failed"}, order_by="-timestamp")


class RiskAuditLogRepository(DatabaseRepository[RiskAuditLog, str]):
    """Repository for RiskAuditLog entities."""

    def __init__(self, session: AsyncSession):
        """Initialize risk audit log repository."""
        super().__init__(
            session=session,
            model=RiskAuditLog,
            entity_type=RiskAuditLog,
            key_type=str,
            name="RiskAuditLogRepository",
        )

    async def get_by_risk_type(self, risk_type: str) -> list[RiskAuditLog]:
        """Get audit logs by risk type."""
        return await self.get_all(filters={"risk_type": risk_type}, order_by="-timestamp")

    async def get_high_severity_risks(self) -> list[RiskAuditLog]:
        """Get high severity risk logs."""
        return await self.get_all(filters={"severity": "high"}, order_by="-timestamp")

    async def get_critical_risks(self) -> list[RiskAuditLog]:
        """Get critical risk logs."""
        return await self.get_all(filters={"severity": "critical"}, order_by="-timestamp")


class PerformanceAuditLogRepository(DatabaseRepository[PerformanceAuditLog, str]):
    """Repository for PerformanceAuditLog entities."""

    def __init__(self, session: AsyncSession):
        """Initialize performance audit log repository."""
        super().__init__(
            session=session,
            model=PerformanceAuditLog,
            entity_type=PerformanceAuditLog,
            key_type=str,
            name="PerformanceAuditLogRepository",
        )

    async def get_by_strategy(self, strategy_id: str) -> list[PerformanceAuditLog]:
        """Get performance audit logs by strategy."""
        return await self.get_all(filters={"strategy_id": strategy_id}, order_by="-timestamp")

    async def get_by_metric_type(self, metric_type: str) -> list[PerformanceAuditLog]:
        """Get performance audit logs by metric type."""
        return await self.get_all(filters={"metric_type": metric_type}, order_by="-timestamp")
