"""
Bot Metrics Repository - Data access layer for bot metrics and performance data.

This repository handles all database operations for bot metrics, health checks,
and performance analytics, providing centralized data access for bot monitoring.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.database.models.system import PerformanceMetrics
from src.database.repository.base import DatabaseRepository

logger = get_logger(__name__)


class BotMetricsRepository(DatabaseRepository):
    """Repository for bot metrics and performance data operations."""

    def __init__(self, session: AsyncSession):
        """
        Initialize bot metrics repository.

        Args:
            session: Database session
        """
        super().__init__(
            session=session,
            model=PerformanceMetrics,
            entity_type=PerformanceMetrics,
            key_type=str,
            name="BotMetricsRepository",
        )

    async def store_bot_metrics(self, bot_id: str, metrics: dict[str, Any]) -> None:
        """
        Store bot metrics.

        Args:
            bot_id: Bot identifier
            metrics: Metrics data dictionary
        """
        try:
            # Create metrics record
            metrics_record = PerformanceMetrics(
                entity_id=bot_id,
                entity_type="bot",
                metric_type="bot_performance",
                metrics_data=metrics,
                timestamp=datetime.now(timezone.utc),
            )

            self.session.add(metrics_record)
            await self.session.commit()

            logger.debug(f"Stored metrics for bot {bot_id}")

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to store bot metrics: {e}")
            raise

    async def get_bot_metrics(self, bot_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get bot metrics history.

        Args:
            bot_id: Bot identifier
            limit: Maximum number of records to return

        Returns:
            List of metrics records
        """
        try:
            result = await self.session.execute(
                select(PerformanceMetrics)
                .where(
                    and_(
                        PerformanceMetrics.entity_id == bot_id,
                        PerformanceMetrics.entity_type == "bot",
                    )
                )
                .order_by(desc(PerformanceMetrics.timestamp))
                .limit(limit)
            )

            metrics = result.scalars().all()

            return [
                {
                    "bot_id": m.entity_id,
                    "timestamp": m.timestamp,
                    "metrics": m.metrics_data,
                    "metric_type": m.metric_type,
                }
                for m in metrics
            ]

        except Exception as e:
            logger.error(f"Failed to get bot metrics: {e}")
            raise

    async def get_latest_metrics(self, bot_id: str) -> dict[str, Any] | None:
        """
        Get latest metrics for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Latest metrics record or None
        """
        metrics = await self.get_bot_metrics(bot_id, limit=1)
        return metrics[0] if metrics else None

    async def get_bot_health_checks(self, bot_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get bot health check history.

        Args:
            bot_id: Bot identifier
            limit: Maximum number of records

        Returns:
            List of health check records
        """
        try:
            result = await self.session.execute(
                select(PerformanceMetrics)
                .where(
                    and_(
                        PerformanceMetrics.entity_id == bot_id,
                        PerformanceMetrics.entity_type == "bot",
                        PerformanceMetrics.metric_type == "health_check",
                    )
                )
                .order_by(desc(PerformanceMetrics.timestamp))
                .limit(limit)
            )

            health_checks = result.scalars().all()

            return [
                {
                    "bot_id": h.entity_id,
                    "timestamp": h.timestamp,
                    "health_data": h.metrics_data,
                    "status": h.metrics_data.get("status", "unknown"),
                }
                for h in health_checks
            ]

        except Exception as e:
            logger.error(f"Failed to get health checks: {e}")
            raise

    async def store_bot_health_analysis(self, bot_id: str, analysis: dict[str, Any]) -> None:
        """
        Store bot health analysis.

        Args:
            bot_id: Bot identifier
            analysis: Health analysis data
        """
        try:
            health_record = PerformanceMetrics(
                entity_id=bot_id,
                entity_type="bot",
                metric_type="health_analysis",
                metrics_data=analysis,
                timestamp=datetime.now(timezone.utc),
            )

            self.session.add(health_record)
            await self.session.commit()

            logger.debug(f"Stored health analysis for bot {bot_id}")

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to store health analysis: {e}")
            raise

    async def get_bot_health_analyses(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]:
        """
        Get bot health analyses for a time period.

        Args:
            bot_id: Bot identifier
            hours: Number of hours to look back

        Returns:
            List of health analysis records
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            result = await self.session.execute(
                select(PerformanceMetrics)
                .where(
                    and_(
                        PerformanceMetrics.entity_id == bot_id,
                        PerformanceMetrics.entity_type == "bot",
                        PerformanceMetrics.metric_type == "health_analysis",
                        PerformanceMetrics.timestamp >= cutoff_time,
                    )
                )
                .order_by(desc(PerformanceMetrics.timestamp))
            )

            analyses = result.scalars().all()

            return [
                {"bot_id": a.entity_id, "timestamp": a.timestamp, "analysis": a.metrics_data}
                for a in analyses
            ]

        except Exception as e:
            logger.error(f"Failed to get health analyses: {e}")
            raise

    async def get_recent_health_analyses(self, hours: int = 1) -> list[dict[str, Any]]:
        """
        Get recent health analyses for all bots.

        Args:
            hours: Number of hours to look back

        Returns:
            List of recent health analyses
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            result = await self.session.execute(
                select(PerformanceMetrics)
                .where(
                    and_(
                        PerformanceMetrics.entity_type == "bot",
                        PerformanceMetrics.metric_type == "health_analysis",
                        PerformanceMetrics.timestamp >= cutoff_time,
                    )
                )
                .order_by(desc(PerformanceMetrics.timestamp))
            )

            analyses = result.scalars().all()

            return [
                {"bot_id": a.entity_id, "timestamp": a.timestamp, "analysis": a.metrics_data}
                for a in analyses
            ]

        except Exception as e:
            logger.error(f"Failed to get recent health analyses: {e}")
            raise

    async def archive_bot_record(self, bot_id: str) -> None:
        """
        Archive bot metrics records.

        Args:
            bot_id: Bot identifier
        """
        try:
            # Mark old metrics as archived
            result = await self.session.execute(
                select(PerformanceMetrics).where(
                    and_(
                        PerformanceMetrics.entity_id == bot_id,
                        PerformanceMetrics.entity_type == "bot",
                    )
                )
            )

            metrics = result.scalars().all()
            for metric in metrics:
                if "archived" not in metric.metrics_data:
                    metric.metrics_data["archived"] = True
                    metric.metrics_data["archived_at"] = datetime.now(timezone.utc).isoformat()

            await self.session.commit()

            logger.info(f"Archived {len(metrics)} records for bot {bot_id}")

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to archive bot records: {e}")
            raise

    async def get_active_bots(self) -> list[str]:
        """
        Get list of active bot IDs from recent metrics.

        Returns:
            List of active bot IDs
        """
        try:
            # Get bots with metrics in last hour
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)

            result = await self.session.execute(
                select(PerformanceMetrics.entity_id)
                .where(
                    and_(
                        PerformanceMetrics.entity_type == "bot",
                        PerformanceMetrics.timestamp >= cutoff_time,
                    )
                )
                .distinct()
            )

            bot_ids = result.scalars().all()
            return list(bot_ids)

        except Exception as e:
            logger.error(f"Failed to get active bots: {e}")
            raise

    async def get_bot_performance_summary(self, bot_id: str, hours: int = 24) -> dict[str, Any]:
        """
        Get performance summary for a bot.

        Args:
            bot_id: Bot identifier
            hours: Number of hours to analyze

        Returns:
            Performance summary dictionary
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            result = await self.session.execute(
                select(PerformanceMetrics)
                .where(
                    and_(
                        PerformanceMetrics.entity_id == bot_id,
                        PerformanceMetrics.entity_type == "bot",
                        PerformanceMetrics.metric_type == "bot_performance",
                        PerformanceMetrics.timestamp >= cutoff_time,
                    )
                )
                .order_by(PerformanceMetrics.timestamp)
            )

            metrics = result.scalars().all()

            if not metrics:
                return {"bot_id": bot_id, "period_hours": hours, "data_points": 0, "summary": {}}

            # Calculate summary statistics
            total_pnl = Decimal("0")
            win_count = 0
            loss_count = 0
            total_trades = 0

            for m in metrics:
                data = m.metrics_data
                if "total_pnl" in data:
                    total_pnl = Decimal(str(data["total_pnl"]))
                if "successful_trades" in data:
                    win_count = int(data["successful_trades"])
                if "failed_trades" in data:
                    loss_count = int(data["failed_trades"])
                if "total_trades" in data:
                    total_trades = int(data["total_trades"])

            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

            return {
                "bot_id": bot_id,
                "period_hours": hours,
                "data_points": len(metrics),
                "summary": {
                    "total_pnl": str(total_pnl),
                    "total_trades": total_trades,
                    "win_count": win_count,
                    "loss_count": loss_count,
                    "win_rate": win_rate,
                    "first_metric": metrics[0].timestamp.isoformat(),
                    "last_metric": metrics[-1].timestamp.isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            raise
