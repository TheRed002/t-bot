"""Bot service layer implementing business logic for bot operations."""

from typing import Any

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.database.interfaces import BotMetricsServiceInterface
from src.database.repository.bot import BotRepository

logger = get_logger(__name__)


class BotService(BaseService, BotMetricsServiceInterface):
    """Service layer for bot operations with business logic."""

    def __init__(
        self,
        bot_repo: BotRepository | None = None,
    ):
        """Initialize with injected repositories."""
        super().__init__(name="BotService")
        self.bot_repo = bot_repo

    async def get_active_bots(self) -> list[dict[str, Any]]:
        """
        Get all active bots with business logic.

        Returns:
            List of active bot records
        """
        try:
            if not self.bot_repo:
                raise ServiceError("Bot repository not available")
                
            # Get active bots through repository
            active_bots = await self.bot_repo.get_all(
                filters={"status": "ACTIVE"},
                order_by="-created_at"
            )

            # Convert to dict format with business logic
            bot_records = []
            for bot in active_bots:
                bot_records.append({
                    "bot_id": bot.id,
                    "name": bot.name,
                    "status": bot.status,
                    "created_at": bot.created_at,
                    "is_healthy": self._assess_bot_health(bot),
                    "uptime_hours": self._calculate_uptime_hours(bot),
                })

            logger.info(f"Retrieved {len(bot_records)} active bots")
            return bot_records

        except Exception as e:
            logger.error(f"Failed to get active bots: {e}")
            raise ServiceError(f"Get active bots failed: {e}") from e

    async def archive_bot_record(self, bot_id: str) -> bool:
        """
        Archive bot record with business logic validation.

        Args:
            bot_id: Bot identifier

        Returns:
            True if archived successfully
        """
        try:
            if not bot_id or not bot_id.strip():
                raise ValidationError("Bot ID is required")
                
            if not self.bot_repo:
                raise ServiceError("Bot repository not available")

            # Check if bot exists
            bot = await self.bot_repo.get_by_id(bot_id)
            if not bot:
                raise ValidationError(f"Bot {bot_id} not found")

            # Business logic: Check if bot can be archived
            if not self._can_archive_bot(bot):
                raise ValidationError(f"Bot {bot_id} cannot be archived in current state")

            # Archive the bot through repository
            success = await self.bot_repo.soft_delete(bot_id, deleted_by="system")
            
            if success:
                logger.info(f"Bot {bot_id} archived successfully")
                
            return success

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to archive bot {bot_id}: {e}")
            raise ServiceError(f"Bot archival failed: {e}") from e

    async def get_bot_metrics(self, bot_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get bot metrics with business logic.

        Args:
            bot_id: Bot identifier
            limit: Maximum number of metrics to return

        Returns:
            List of bot metrics
        """
        try:
            if not bot_id or not bot_id.strip():
                raise ValidationError("Bot ID is required")
                
            if limit <= 0:
                raise ValidationError("Limit must be positive")

            # Business logic: Get bot metrics
            # This would typically come from a metrics repository
            # For now, return placeholder data
            metrics = [
                {
                    "bot_id": bot_id,
                    "metric_type": "performance",
                    "value": 95.5,
                    "timestamp": "2023-01-01T00:00:00Z",
                },
                {
                    "bot_id": bot_id,
                    "metric_type": "uptime",
                    "value": 99.9,
                    "timestamp": "2023-01-01T00:00:00Z",
                },
            ]

            logger.debug(f"Retrieved {len(metrics)} metrics for bot {bot_id}")
            return metrics[:limit]

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to get bot metrics for {bot_id}: {e}")
            raise ServiceError(f"Get bot metrics failed: {e}") from e

    async def store_bot_metrics(self, metrics_record: dict[str, Any]) -> bool:
        """
        Store bot metrics with business logic validation.

        Args:
            metrics_record: Metrics data to store

        Returns:
            True if stored successfully
        """
        try:
            # Validate required fields
            if "bot_id" not in metrics_record:
                raise ValidationError("bot_id is required in metrics record")
                
            if "metric_type" not in metrics_record:
                raise ValidationError("metric_type is required in metrics record")
                
            if "value" not in metrics_record:
                raise ValidationError("value is required in metrics record")

            # Business logic: Validate metric value ranges
            metric_type = metrics_record["metric_type"]
            value = metrics_record["value"]
            
            if metric_type in ["performance", "uptime"] and (value < 0 or value > 100):
                raise ValidationError(f"Invalid value for {metric_type}: must be 0-100")

            # Store metrics (would typically go through metrics repository)
            logger.info(f"Storing {metric_type} metrics for bot {metrics_record['bot_id']}")
            return True

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to store bot metrics: {e}")
            raise ServiceError(f"Store bot metrics failed: {e}") from e

    def _assess_bot_health(self, bot) -> bool:
        """Business logic to assess bot health."""
        # Placeholder logic - would typically check various health indicators
        return bot.status == "ACTIVE"
        
    def _calculate_uptime_hours(self, bot) -> float:
        """Business logic to calculate bot uptime in hours."""
        # Placeholder logic - would calculate based on created_at and status changes
        from datetime import datetime, timezone
        
        if bot.created_at:
            delta = datetime.now(timezone.utc) - bot.created_at
            return delta.total_seconds() / 3600
        return 0.0
        
    def _can_archive_bot(self, bot) -> bool:
        """Business logic to determine if bot can be archived."""
        # Don't archive active bots or bots with pending operations
        return bot.status not in ["ACTIVE", "STARTING", "STOPPING"]