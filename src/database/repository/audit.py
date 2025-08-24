"""Audit repositories implementation."""

from datetime import datetime

from sqlalchemy.orm import Session

from src.database.models.audit import (
    CapitalAuditLog,
    ExecutionAuditLog,
    PerformanceAuditLog,
    RiskAuditLog,
)
from src.database.repository.base import BaseRepository


class CapitalAuditLogRepository(BaseRepository[CapitalAuditLog]):
    """Repository for CapitalAuditLog entities."""

    def __init__(self, session: Session):
        """Initialize capital audit log repository."""
        super().__init__(session, CapitalAuditLog)

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


class ExecutionAuditLogRepository(BaseRepository[ExecutionAuditLog]):
    """Repository for ExecutionAuditLog entities."""

    def __init__(self, session: Session):
        """Initialize execution audit log repository."""
        super().__init__(session, ExecutionAuditLog)

    async def get_by_order(self, order_id: str) -> list[ExecutionAuditLog]:
        """Get audit logs by order."""
        return await self.get_all(filters={"order_id": order_id}, order_by="-timestamp")

    async def get_by_execution_status(self, status: str) -> list[ExecutionAuditLog]:
        """Get audit logs by execution status."""
        return await self.get_all(filters={"execution_status": status}, order_by="-timestamp")

    async def get_failed_executions(self) -> list[ExecutionAuditLog]:
        """Get failed execution logs."""
        return await self.get_all(filters={"execution_status": "failed"}, order_by="-timestamp")


class RiskAuditLogRepository(BaseRepository[RiskAuditLog]):
    """Repository for RiskAuditLog entities."""

    def __init__(self, session: Session):
        """Initialize risk audit log repository."""
        super().__init__(session, RiskAuditLog)

    async def get_by_risk_type(self, risk_type: str) -> list[RiskAuditLog]:
        """Get audit logs by risk type."""
        return await self.get_all(filters={"risk_type": risk_type}, order_by="-timestamp")

    async def get_high_severity_risks(self) -> list[RiskAuditLog]:
        """Get high severity risk logs."""
        return await self.get_all(filters={"severity": "high"}, order_by="-timestamp")

    async def get_critical_risks(self) -> list[RiskAuditLog]:
        """Get critical risk logs."""
        return await self.get_all(filters={"severity": "critical"}, order_by="-timestamp")


class PerformanceAuditLogRepository(BaseRepository[PerformanceAuditLog]):
    """Repository for PerformanceAuditLog entities."""

    def __init__(self, session: Session):
        """Initialize performance audit log repository."""
        super().__init__(session, PerformanceAuditLog)

    async def get_by_strategy(self, strategy_id: str) -> list[PerformanceAuditLog]:
        """Get performance audit logs by strategy."""
        return await self.get_all(filters={"strategy_id": strategy_id}, order_by="-timestamp")

    async def get_by_metric_type(self, metric_type: str) -> list[PerformanceAuditLog]:
        """Get performance audit logs by metric type."""
        return await self.get_all(filters={"metric_type": metric_type}, order_by="-timestamp")
