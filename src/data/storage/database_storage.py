"""
Database Storage Implementation

Concrete implementation of DataStorageInterface for database operations.
"""

from typing import TYPE_CHECKING, Any

from src.core import BaseComponent, Config, DataError
from src.core.exceptions import DatabaseError
from src.data.interfaces import DataStorageInterface
from src.data.types import DataRequest
from src.database.models import MarketDataRecord

if TYPE_CHECKING:
    from src.database.interfaces import DatabaseServiceInterface


class DatabaseStorage(BaseComponent, DataStorageInterface):
    """Database storage implementation."""

    def __init__(self, config: Config, database_service: "DatabaseServiceInterface | None" = None):
        """
        Initialize database storage.

        Args:
            config: Configuration
            database_service: Optional database service for DI
        """
        super().__init__()
        self.config = config
        self.database_service = database_service
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database storage."""
        if not self._initialized:
            self.logger.info("Database storage initialized")
            self._initialized = True

    async def store_records(self, records: list[MarketDataRecord]) -> bool:
        """Store market data records to database."""
        try:
            if not self.database_service:
                raise DataError("Database service not available - must be injected")

            # Use database service bulk_create method instead of direct repository access
            await self.database_service.bulk_create(records)

            self.logger.debug(f"Stored {len(records)} records to database via repository")
            return True

        except DatabaseError as e:
            self.logger.error(f"Database storage failed: {e}")
            raise DataError(f"Database storage failed: {e!s}") from e
        except Exception as e:
            self.logger.error(f"Database storage failed: {e}")
            raise DataError(f"Database storage failed: {e}") from e

    async def retrieve_records(self, request: DataRequest) -> list[MarketDataRecord]:
        """Retrieve market data records from database."""
        try:
            if not self.database_service:
                raise DataError("Database service not available - must be injected")

            # Use database service list_entities method instead of direct repository access

            # Build filters for the query
            filters = {}
            if request.symbol:
                filters["symbol"] = request.symbol
            if request.exchange:
                filters["exchange"] = request.exchange
            if request.start_time or request.end_time:
                time_filter = {}
                if request.start_time:
                    time_filter["gte"] = request.start_time
                if request.end_time:
                    time_filter["lte"] = request.end_time
                filters["data_timestamp"] = time_filter

            # Get records using database service
            return await self.database_service.list_entities(
                model_class=MarketDataRecord,
                filters=filters,
                order_by="data_timestamp",
                order_desc=True,
                limit=request.limit,
            )

        except DatabaseError as e:
            self.logger.error(f"Database retrieval failed: {e}")
            raise DataError(f"Database retrieval failed: {e!s}") from e
        except Exception as e:
            self.logger.error(f"Database retrieval failed: {e}")
            raise DataError(f"Database retrieval failed: {e}") from e

    async def get_record_count(self, symbol: str, exchange: str) -> int:
        """Get count of records for symbol and exchange."""
        try:
            if not self.database_service:
                raise DataError("Database service not available - must be injected")

            # Use database service count_entities method instead of direct repository access

            # Get count using database service
            filters = {"symbol": symbol, "exchange": exchange}
            return await self.database_service.count_entities(
                model_class=MarketDataRecord, filters=filters
            )

        except DatabaseError as e:
            self.logger.error(f"Database record count retrieval failed: {e}")
            return 0
        except Exception as e:
            self.logger.error(f"Record count retrieval failed: {e}")
            return 0

    async def health_check(self) -> dict[str, Any]:
        """Perform storage health check."""
        try:
            if not self.database_service:
                return {
                    "status": "unhealthy",
                    "component": "database_storage",
                    "error": "Database service not available - must be injected",
                }

            # Use proper service layer method for health check
            health_result = await self.database_service.get_health_status()
            return {
                "status": "healthy" if health_result.name == "HEALTHY" else "unhealthy",
                "component": "database_storage",
            }

        except Exception as e:
            return {"status": "unhealthy", "component": "database_storage", "error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup storage resources."""
        self._initialized = False
        self.logger.info("Database storage cleanup completed")
