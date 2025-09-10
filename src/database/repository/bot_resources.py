"""
Bot Resources Repository - Data access layer for bot resource management.

This repository handles all database operations for bot resource allocations,
reservations, and usage tracking.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.database.models.system import ResourceAllocation, ResourceUsage
from src.database.repository.base import DatabaseRepository

logger = get_logger(__name__)


class BotResourcesRepository(DatabaseRepository):
    """Repository for bot resource management operations."""

    def __init__(self, session: AsyncSession):
        """
        Initialize bot resources repository.

        Args:
            session: Database session
        """
        super().__init__(
            session=session,
            model=ResourceAllocation,
            entity_type=ResourceAllocation,
            key_type=str,
            name="BotResourcesRepository",
        )

    async def store_resource_allocation(self, allocation: dict[str, Any]) -> None:
        """
        Store resource allocation record.

        Args:
            allocation: Resource allocation data
        """
        try:
            allocation_record = ResourceAllocation(
                entity_id=allocation["bot_id"],
                entity_type="bot",
                resource_type=allocation["resource_type"],
                allocated_amount=Decimal(str(allocation.get("allocated_amount", 0))),
                max_amount=Decimal(str(allocation.get("max_amount", 0))),
                metadata_json=allocation.get("metadata", {}),
                status="active",
                created_at=datetime.now(timezone.utc),
            )

            self.session.add(allocation_record)
            await self.session.commit()

            logger.debug(f"Stored resource allocation for bot {allocation['bot_id']}")

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to store resource allocation: {e}")
            raise

    async def update_resource_allocation_status(self, bot_id: str, status: str) -> None:
        """
        Update resource allocation status.

        Args:
            bot_id: Bot identifier
            status: New status
        """
        try:
            await self.session.execute(
                update(ResourceAllocation)
                .where(
                    and_(
                        ResourceAllocation.entity_id == bot_id,
                        ResourceAllocation.entity_type == "bot",
                    )
                )
                .values(status=status, updated_at=datetime.now(timezone.utc))
            )

            await self.session.commit()

            logger.debug(f"Updated resource allocation status for bot {bot_id} to {status}")

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to update resource allocation status: {e}")
            raise

    async def store_resource_usage(self, usage: dict[str, Any]) -> None:
        """
        Store resource usage record.

        Args:
            usage: Resource usage data
        """
        try:
            usage_record = ResourceUsage(
                entity_id=usage["bot_id"],
                entity_type="bot",
                resource_type=usage["resource_type"],
                usage_amount=Decimal(str(usage.get("usage_amount", 0))),
                usage_percentage=Decimal(str(usage.get("usage_percentage", 0))),
                metadata_json=usage.get("metadata", {}),
                timestamp=datetime.now(timezone.utc),
            )

            self.session.add(usage_record)
            await self.session.commit()

            logger.debug(f"Stored resource usage for bot {usage['bot_id']}")

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to store resource usage: {e}")
            raise

    async def store_resource_reservation(self, reservation: dict[str, Any]) -> None:
        """
        Store resource reservation.

        Args:
            reservation: Reservation data
        """
        try:
            # Store as allocation with reservation status
            reservation_record = ResourceAllocation(
                entity_id=reservation["bot_id"],
                entity_type="bot",
                resource_type=reservation["resource_type"],
                allocated_amount=Decimal(str(reservation.get("amount", 0))),
                max_amount=Decimal(str(reservation.get("max_amount", 0))),
                metadata_json={
                    "reservation_id": reservation.get("reservation_id"),
                    "expires_at": reservation.get("expires_at"),
                    **reservation.get("metadata", {}),
                },
                status="reserved",
                created_at=datetime.now(timezone.utc),
            )

            self.session.add(reservation_record)
            await self.session.commit()

            logger.debug(f"Stored resource reservation for bot {reservation['bot_id']}")

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to store resource reservation: {e}")
            raise

    async def update_resource_reservation_status(self, reservation_id: str, status: str) -> None:
        """
        Update resource reservation status.

        Args:
            reservation_id: Reservation identifier
            status: New status
        """
        try:
            # Find reservation by metadata
            result = await self.session.execute(
                select(ResourceAllocation).where(
                    and_(
                        ResourceAllocation.status == "reserved",
                        ResourceAllocation.metadata_json.op("->>")("reservation_id")
                        == reservation_id,
                    )
                )
            )

            reservation = result.scalar_one_or_none()
            if reservation:
                reservation.status = status
                reservation.updated_at = datetime.now(timezone.utc)
                await self.session.commit()

                logger.debug(f"Updated reservation {reservation_id} status to {status}")

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to update reservation status: {e}")
            raise

    async def store_resource_usage_history(self, usage_entry: dict[str, Any]) -> None:
        """
        Store resource usage history entry.

        Args:
            usage_entry: Usage history data
        """
        await self.store_resource_usage(usage_entry)

    async def store_optimization_suggestion(self, suggestion: dict[str, Any]) -> None:
        """
        Store resource optimization suggestion.

        Args:
            suggestion: Optimization suggestion data
        """
        try:
            # Store as metadata in resource usage
            optimization_record = ResourceUsage(
                entity_id=suggestion.get("bot_id", "system"),
                entity_type="optimization",
                resource_type=suggestion.get("resource_type", "all"),
                usage_amount=Decimal("0"),
                usage_percentage=Decimal("0"),
                metadata_json={
                    "suggestion_type": suggestion.get("type"),
                    "description": suggestion.get("description"),
                    "potential_savings": suggestion.get("potential_savings"),
                    "priority": suggestion.get("priority"),
                    **suggestion.get("metadata", {}),
                },
                timestamp=datetime.now(timezone.utc),
            )

            self.session.add(optimization_record)
            await self.session.commit()

            logger.debug("Stored optimization suggestion")

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to store optimization suggestion: {e}")
            raise

    async def get_resource_allocations(self, bot_id: str) -> list[dict[str, Any]]:
        """
        Get resource allocations for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            List of resource allocations
        """
        try:
            result = await self.session.execute(
                select(ResourceAllocation).where(
                    and_(
                        ResourceAllocation.entity_id == bot_id,
                        ResourceAllocation.entity_type == "bot",
                        ResourceAllocation.status == "active",
                    )
                )
            )

            allocations = result.scalars().all()

            return [
                {
                    "bot_id": a.entity_id,
                    "resource_type": a.resource_type,
                    "allocated_amount": str(a.allocated_amount),
                    "max_amount": str(a.max_amount),
                    "status": a.status,
                    "metadata": a.metadata_json,
                }
                for a in allocations
            ]

        except Exception as e:
            logger.error(f"Failed to get resource allocations: {e}")
            raise

    async def get_resource_usage_history(
        self, bot_id: str, hours: int = 24
    ) -> list[dict[str, Any]]:
        """
        Get resource usage history for a bot.

        Args:
            bot_id: Bot identifier
            hours: Number of hours to look back

        Returns:
            List of usage records
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            result = await self.session.execute(
                select(ResourceUsage)
                .where(
                    and_(
                        ResourceUsage.entity_id == bot_id,
                        ResourceUsage.entity_type == "bot",
                        ResourceUsage.timestamp >= cutoff_time,
                    )
                )
                .order_by(desc(ResourceUsage.timestamp))
            )

            usage_records = result.scalars().all()

            return [
                {
                    "bot_id": u.entity_id,
                    "resource_type": u.resource_type,
                    "usage_amount": str(u.usage_amount),
                    "usage_percentage": u.usage_percentage,
                    "timestamp": u.timestamp.isoformat(),
                    "metadata": u.metadata_json,
                }
                for u in usage_records
            ]

        except Exception as e:
            logger.error(f"Failed to get resource usage history: {e}")
            raise

    async def cleanup_expired_reservations(self) -> int:
        """
        Clean up expired resource reservations.

        Returns:
            Number of reservations cleaned up
        """
        try:
            now = datetime.now(timezone.utc)

            # Find expired reservations
            result = await self.session.execute(
                select(ResourceAllocation).where(
                    and_(
                        ResourceAllocation.status == "reserved",
                        ResourceAllocation.metadata_json.op("->>")("expires_at") < now.isoformat(),
                    )
                )
            )

            expired = result.scalars().all()

            for reservation in expired:
                reservation.status = "expired"
                reservation.updated_at = now

            await self.session.commit()

            logger.info(f"Cleaned up {len(expired)} expired reservations")
            return len(expired)

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to cleanup expired reservations: {e}")
            raise
