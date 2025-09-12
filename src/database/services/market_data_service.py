"""Market data service layer implementing business logic for market data operations."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError
from src.core.logging import get_logger
from src.database.interfaces import MarketDataServiceInterface
from src.database.repository.market_data import MarketDataRepository

logger = get_logger(__name__)


class MarketDataService(BaseService, MarketDataServiceInterface):
    """Service layer for market data operations with business logic."""

    def __init__(self, market_data_repo: MarketDataRepository):
        """Initialize with injected repository."""
        super().__init__(name="MarketDataService")
        self.market_data_repo = market_data_repo

    async def get_latest_price(self, symbol: str) -> Decimal | None:
        """Get latest price for symbol with business logic."""
        try:
            # Get latest market data record
            filters = {"symbol": symbol}
            market_data_list = await self.market_data_repo.list(
                filters=filters,
                limit=1,
                order_by="timestamp",
                order_desc=True
            )

            if market_data_list:
                latest_data = market_data_list[0]
                # Return close price if available, otherwise open price
                return latest_data.close_price or latest_data.open_price

            return None

        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            raise ServiceError(f"Latest price retrieval failed: {e}") from e

    async def get_historical_data(
        self, symbol: str, start_time: datetime, end_time: datetime, timeframe: str = "1m"
    ) -> list[dict[str, Any]]:
        """Get historical market data with business logic filtering."""
        try:
            # Build filters with business logic
            filters = {
                "symbol": symbol,
                "timestamp": {"gte": start_time, "lte": end_time},
                "timeframe": timeframe,
            }

            # Get market data from repository
            market_data_list = await self.market_data_repo.list(
                filters=filters,
                order_by="timestamp",
                order_desc=False
            )

            # Convert to dict format for API responses with business logic formatting
            return [
                {
                    "id": md.id,
                    "symbol": md.symbol,
                    "timestamp": md.timestamp,
                    "open": str(md.open_price) if md.open_price else None,
                    "high": str(md.high_price) if md.high_price else None,
                    "low": str(md.low_price) if md.low_price else None,
                    "close": str(md.close_price) if md.close_price else None,
                    "volume": str(md.volume) if md.volume else None,
                    "timeframe": md.timeframe,
                }
                for md in market_data_list
            ]

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise ServiceError(f"Historical data retrieval failed: {e}") from e
