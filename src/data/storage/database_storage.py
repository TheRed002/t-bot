"""
Database Storage Implementation

Concrete implementation of DataStorageInterface for database operations.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.base.component import BaseComponent
from src.core.config import Config
from src.core.exceptions import DataError
from src.data.interfaces import DataStorageInterface
from src.data.types import DataRequest
from src.database import get_async_session
from src.database.models import MarketDataRecord
from src.database.queries import DatabaseQueries


class DatabaseStorage(BaseComponent, DataStorageInterface):
    """Database storage implementation."""

    def __init__(self, config: Config):
        """Initialize database storage."""
        super().__init__()
        self.config = config
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database storage."""
        if not self._initialized:
            self.logger.info("Database storage initialized")
            self._initialized = True

    async def store_records(self, records: list[MarketDataRecord]) -> bool:
        """Store market data records to database."""
        session = None
        try:
            session = get_async_session()
            async with session:
                db_queries = DatabaseQueries(session, config=None)
                await db_queries.bulk_create(records)
                await session.commit()
                self.logger.debug(f"Stored {len(records)} records to database")
                return True

        except Exception as e:
            self.logger.error(f"Database storage failed: {e}")
            raise DataError(f"Database storage failed: {e}") from e
        finally:
            if session:
                await session.close()

    async def retrieve_records(self, request: DataRequest) -> list[MarketDataRecord]:
        """Retrieve market data records from database."""
        session = None
        try:
            session = get_async_session()
            async with session:
                db_queries = DatabaseQueries(session, config=None)
                
                return await db_queries.get_market_data_records(
                    symbol=request.symbol,
                    exchange=request.exchange,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    limit=request.limit,
                )
        except Exception as e:
            self.logger.error(f"Database retrieval failed: {e}")
            raise DataError(f"Database retrieval failed: {e}") from e
        finally:
            if session:
                await session.close()

    async def get_record_count(self, symbol: str, exchange: str) -> int:
        """Get count of records for symbol and exchange."""
        session = None
        try:
            session = get_async_session()
            async with session:
                db_queries = DatabaseQueries(session, config=None)
                
                records = await db_queries.get_market_data_records(
                    symbol=symbol, exchange=exchange, limit=None
                )
                return len(records)
        except Exception as e:
            self.logger.error(f"Record count retrieval failed: {e}")
            return 0
        finally:
            if session:
                await session.close()

    async def health_check(self) -> dict[str, Any]:
        """Perform storage health check."""
        session = None
        try:
            session = get_async_session()
            async with session:
                from sqlalchemy import text
                await session.execute(text("SELECT 1"))
            return {"status": "healthy", "component": "database_storage"}
        except Exception as e:
            return {"status": "unhealthy", "component": "database_storage", "error": str(e)}
        finally:
            if session:
                await session.close()

    async def cleanup(self) -> None:
        """Cleanup storage resources."""
        self._initialized = False
        self.logger.info("Database storage cleanup completed")