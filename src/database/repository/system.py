"""System repositories implementation."""

from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.system import (
    Alert,
    AuditLog,
    BalanceSnapshot,
    PerformanceMetrics,
)
from src.database.repository.base import DatabaseRepository
from src.database.repository.utils import RepositoryUtils


class AlertRepository(DatabaseRepository):
    """Repository for Alert entities."""

    def __init__(self, session: AsyncSession):
        """Initialize alert repository."""

        super().__init__(
            session=session, model=Alert, entity_type=Alert, key_type=str, name="AlertRepository"
        )

    async def get_by_user(self, user_id: str) -> list[Alert]:
        """Get alerts by user."""
        return await RepositoryUtils.get_entities_by_field(self, "user_id", user_id, "-timestamp")

    async def get_unread_alerts(self, user_id: str) -> list[Alert]:
        """Get unread alerts for user."""
        return await self.get_all(
            filters={"user_id": user_id, "is_read": False}, order_by="-timestamp"
        )

    async def get_by_severity(self, severity: str) -> list[Alert]:
        """Get alerts by severity."""
        return await RepositoryUtils.get_entities_by_field(self, "severity", severity, "-timestamp")

    async def get_critical_alerts(self) -> list[Alert]:
        """Get critical alerts."""
        return await self.get_by_severity("critical")

    async def get_by_type(self, alert_type: str) -> list[Alert]:
        """Get alerts by type."""
        return await RepositoryUtils.get_entities_by_field(
            self, "alert_type", alert_type, "-timestamp"
        )

    async def mark_as_read(self, alert_id: str) -> bool:
        """Mark alert as read."""
        return await RepositoryUtils.mark_entity_field(self, alert_id, "is_read", True, "Alert")

    async def mark_all_read(self, user_id: str) -> int:
        """Mark all alerts as read for user."""
        alerts = await self.get_unread_alerts(user_id)
        return await RepositoryUtils.bulk_mark_entities(self, alerts, "is_read", True, "Alert")


class AuditLogRepository(DatabaseRepository):
    """Repository for AuditLog entities."""

    def __init__(self, session: AsyncSession):
        """Initialize audit log repository."""

        super().__init__(
            session=session,
            model=AuditLog,
            entity_type=AuditLog,
            key_type=str,
            name="AuditLogRepository",
        )

    async def get_by_user(self, user_id: str) -> list[AuditLog]:
        """Get audit logs by user."""
        return await RepositoryUtils.get_entities_by_field(self, "user_id", user_id, "-timestamp")

    async def get_by_action(self, action: str) -> list[AuditLog]:
        """Get audit logs by action."""
        return await RepositoryUtils.get_entities_by_field(self, "action", action, "-timestamp")

    async def get_by_resource_type(self, resource_type: str) -> list[AuditLog]:
        """Get audit logs by resource type."""
        return await RepositoryUtils.get_entities_by_field(
            self, "resource_type", resource_type, "-timestamp"
        )

    async def get_by_resource(self, resource_type: str, resource_id: str) -> list[AuditLog]:
        """Get audit logs for specific resource."""
        filters = {"resource_type": resource_type, "resource_id": resource_id}
        return await RepositoryUtils.get_entities_by_multiple_fields(self, filters, "-timestamp")

    async def get_recent_logs(self, hours: int = 24) -> list[AuditLog]:
        """Get recent audit logs."""
        return await RepositoryUtils.get_recent_entities(self, hours, None, "-timestamp")


class PerformanceMetricsRepository(DatabaseRepository):
    """Repository for PerformanceMetrics entities."""

    def __init__(self, session: AsyncSession):
        """Initialize performance metrics repository."""

        super().__init__(
            session=session,
            model=PerformanceMetrics,
            entity_type=PerformanceMetrics,
            key_type=str,
            name="PerformanceMetricsRepository",
        )

    async def get_by_bot(self, bot_id: str) -> list[PerformanceMetrics]:
        """Get performance metrics by bot."""
        return await self.get_all(filters={"bot_id": bot_id}, order_by="-timestamp")

    async def get_latest_metrics(self, bot_id: str) -> PerformanceMetrics | None:
        """Get latest performance metrics for bot."""
        metrics = await self.get_all(filters={"bot_id": bot_id}, order_by="-timestamp", limit=1)
        return metrics[0] if metrics else None

    async def get_metrics_by_date_range(
        self, bot_id: str, start_date: datetime, end_date: datetime
    ) -> list[PerformanceMetrics]:
        """Get metrics by date range."""
        return await self.get_all(
            filters={"bot_id": bot_id, "timestamp": {"gte": start_date, "lte": end_date}},
            order_by="-timestamp",
        )

    async def get_top_performing_bots(self, limit: int = 10) -> list[PerformanceMetrics]:
        """Get top performing bots by total P&L."""
        # This would need custom SQL for aggregation
        # For now, return recent metrics ordered by total_pnl
        return await self.get_all(order_by="-total_pnl", limit=limit)


class BalanceSnapshotRepository(DatabaseRepository):
    """Repository for BalanceSnapshot entities."""

    def __init__(self, session: AsyncSession):
        """Initialize balance snapshot repository."""

        super().__init__(
            session=session,
            model=BalanceSnapshot,
            entity_type=BalanceSnapshot,
            key_type=str,
            name="BalanceSnapshotRepository",
        )

    async def get_by_user(self, user_id: str) -> list[BalanceSnapshot]:
        """Get balance snapshots by user."""
        return await self.get_all(filters={"user_id": user_id}, order_by="-timestamp")

    async def get_by_exchange(self, exchange: str) -> list[BalanceSnapshot]:
        """Get balance snapshots by exchange."""
        return await self.get_all(filters={"exchange": exchange}, order_by="-timestamp")

    async def get_by_currency(self, currency: str) -> list[BalanceSnapshot]:
        """Get balance snapshots by currency."""
        return await self.get_all(filters={"currency": currency}, order_by="-timestamp")

    async def get_latest_snapshot(
        self, user_id: str, exchange: str, currency: str
    ) -> BalanceSnapshot | None:
        """Get latest balance snapshot."""
        snapshots = await self.get_all(
            filters={"user_id": user_id, "exchange": exchange, "currency": currency},
            order_by="-timestamp",
            limit=1,
        )
        return snapshots[0] if snapshots else None

    async def get_balance_history(
        self, user_id: str, exchange: str, currency: str, days: int = 30
    ) -> list[BalanceSnapshot]:
        """Get balance history for specified period."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        return await self.get_all(
            filters={
                "user_id": user_id,
                "exchange": exchange,
                "currency": currency,
                "timestamp": {"gte": start_date, "lte": end_date},
            },
            order_by="-timestamp",
        )
