"""Database service interfaces for dependency injection."""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypeVar

from src.core.base.interfaces import HealthStatus

if TYPE_CHECKING:
    from src.database.models import Base

T = TypeVar("T", bound="Base")
K = TypeVar("K")


class DatabaseServiceInterface(ABC):
    """Interface for database service operations."""

    @abstractmethod
    async def start(self) -> None:
        """Start the database service."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the database service."""
        pass

    @abstractmethod
    async def create_entity(self, entity: T) -> T:
        """Create a new entity."""
        pass

    @abstractmethod
    async def get_entity_by_id(self, model_class: type[T], entity_id: K) -> T | None:
        """Get entity by ID."""
        pass

    @abstractmethod
    async def update_entity(self, entity: T) -> T:
        """Update existing entity."""
        pass

    @abstractmethod
    async def delete_entity(self, model_class: type[T], entity_id: K) -> bool:
        """Delete entity by ID."""
        pass

    @abstractmethod
    async def list_entities(
        self,
        model_class: type[T],
        limit: int | None = None,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        order_desc: bool = False,
        include_relations: list[str] | None = None,
    ) -> list[T]:
        """List entities with filtering and pagination."""
        pass

    @abstractmethod
    async def count_entities(self, model_class: type[T], filters: dict[str, Any] | None = None) -> int:
        """Count entities matching filters."""
        pass

    @abstractmethod
    async def bulk_create(self, entities: list[T]) -> list[T]:
        """Create multiple entities in bulk."""
        pass

    @abstractmethod
    async def get_health_status(self) -> HealthStatus:
        """Get service health status."""
        pass

    @abstractmethod
    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        pass


class TradingDataServiceInterface(ABC):
    """Interface for trading-specific data operations."""

    @abstractmethod
    async def get_trades_by_bot(
        self,
        bot_id: str,
        limit: int | None = None,
        offset: int = 0,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Any]:
        """Get trades for a specific bot."""
        pass

    @abstractmethod
    async def get_positions_by_bot(self, bot_id: str) -> list[Any]:
        """Get positions for a specific bot."""
        pass

    @abstractmethod
    async def calculate_total_pnl(
        self,
        bot_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> Decimal:
        """Calculate total P&L for a bot."""
        pass


class BotMetricsServiceInterface(ABC):
    """Interface for bot metrics operations."""

    @abstractmethod
    async def get_bot_metrics(self, bot_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get bot metrics."""
        pass

    @abstractmethod
    async def store_bot_metrics(self, metrics_record: dict[str, Any]) -> bool:
        """Store bot metrics."""
        pass

    @abstractmethod
    async def get_active_bots(self) -> list[dict[str, Any]]:
        """Get active bots."""
        pass

    @abstractmethod
    async def archive_bot_record(self, bot_id: str) -> bool:
        """Archive bot record."""
        pass


class HealthAnalyticsServiceInterface(ABC):
    """Interface for health analytics operations."""

    @abstractmethod
    async def store_bot_health_analysis(self, health_analysis: dict[str, Any]) -> bool:
        """Store bot health analysis."""
        pass

    @abstractmethod
    async def get_bot_health_analyses(self, bot_id: str, hours: int = 24) -> list[dict[str, Any]]:
        """Get bot health analyses."""
        pass

    @abstractmethod
    async def get_recent_health_analyses(self, hours: int = 1) -> list[dict[str, Any]]:
        """Get recent health analyses."""
        pass


class ResourceManagementServiceInterface(ABC):
    """Interface for resource management operations."""

    @abstractmethod
    async def store_resource_allocation(self, allocation_record: dict[str, Any]) -> bool:
        """Store resource allocation."""
        pass

    @abstractmethod
    async def store_resource_usage(self, usage_record: dict[str, Any]) -> bool:
        """Store resource usage."""
        pass

    @abstractmethod
    async def store_resource_reservation(self, reservation: dict[str, Any]) -> bool:
        """Store resource reservation."""
        pass

    @abstractmethod
    async def update_resource_allocation_status(self, bot_id: str, status: str) -> bool:
        """Update resource allocation status."""
        pass


class RepositoryFactoryInterface(ABC):
    """Interface for repository factory operations."""

    @abstractmethod
    def create_repository(self, repository_class: type[T], session: Any) -> T:
        """Create repository instance using dependency injection."""
        pass

    @abstractmethod
    def register_repository(self, name: str, repository_class: type[T]) -> None:
        """Register repository class for factory creation."""
        pass

    @abstractmethod
    def is_repository_registered(self, name: str) -> bool:
        """Check if repository is registered."""
        pass


class MLServiceInterface(ABC):
    """Interface for ML service operations."""

    @abstractmethod
    async def get_model_performance_summary(self, model_name: str, days: int = 30) -> dict[str, Any]:
        """Get comprehensive performance summary for a model."""
        pass

    @abstractmethod
    async def validate_model_deployment(self, model_name: str, version: int) -> bool:
        """Validate if a model version is ready for deployment."""
        pass

    @abstractmethod
    async def get_model_recommendations(self, symbol: str, limit: int = 5) -> list[dict[str, Any]]:
        """Get ML model recommendations for a trading symbol."""
        pass


class UnitOfWorkFactoryInterface(ABC):
    """Interface for Unit of Work factory operations."""

    @abstractmethod
    def create(self) -> Any:
        """Create new Unit of Work instance."""
        pass

    @abstractmethod
    def create_async(self) -> Any:
        """Create new async Unit of Work instance."""
        pass

    @abstractmethod
    def configure_dependencies(self, dependency_injector: Any) -> None:
        """Configure dependency injection for created UoW instances."""
        pass
