"""
Backtesting Repository - Data persistence layer.

This repository handles all data persistence operations for backtesting,
including results storage, caching, and historical data access.
"""

from datetime import datetime
from typing import (
    Any,
    Any as DatabaseServiceInterface,
)  # Using Any to avoid interface mismatch during mypy checking

from src.core.base.component import BaseComponent
from src.core.exceptions import ServiceError
from src.core.logging import get_logger
from src.utils.decimal_utils import to_decimal

logger = get_logger(__name__)


class BacktestRepository(BaseComponent):
    """
    Repository for backtesting data operations.

    Handles:
    - Backtest result persistence
    - Historical data retrieval
    - Performance metrics storage
    - Cache management
    """

    def __init__(self, db_manager: DatabaseServiceInterface):
        """
        Initialize repository with database manager.

        Args:
            db_manager: Database manager for data operations
        """
        super().__init__(name="BacktestRepository")
        self.db_manager = db_manager
        logger.info("BacktestRepository initialized")

    async def save_backtest_result(
        self, result_data: dict[str, Any], request_data: dict[str, Any]
    ) -> str:
        """
        Save backtest result to database.

        Args:
            result_data: Backtest result data
            request_data: Original request data

        Returns:
            Saved result ID
        """
        try:
            async with self.db_manager.get_session() as session:
                # Create backtest result record
                from src.database.models.backtesting import BacktestResult

                result = BacktestResult(
                    total_return=to_decimal(result_data.get("total_return", 0)),
                    annual_return=to_decimal(result_data.get("annual_return", 0)),
                    sharpe_ratio=result_data.get("sharpe_ratio", 0.0),
                    max_drawdown=to_decimal(result_data.get("max_drawdown", 0)),
                    total_trades=result_data.get("total_trades", 0),
                    win_rate=result_data.get("win_rate", 0.0),
                    metadata=result_data.get("metadata", {}),
                    request_config=request_data,
                    created_at=datetime.utcnow(),
                )

                session.add(result)
                await session.commit()

                logger.info(f"Backtest result saved with ID: {result.id}")
                return str(result.id)

        except Exception as e:
            logger.error(f"Failed to save backtest result: {e}")
            raise ServiceError(f"Database error saving result: {e}", error_code="REPO_001") from e

    async def get_backtest_result(self, result_id: str) -> dict[str, Any] | None:
        """
        Retrieve backtest result by ID.

        Args:
            result_id: Result ID to retrieve

        Returns:
            Result data or None if not found
        """
        try:
            async with self.db_manager.get_session() as session:
                from src.database.models.backtesting import BacktestResult

                result = await session.get(BacktestResult, result_id)
                if not result:
                    return None

                return {
                    "id": str(result.id),
                    "total_return": float(result.total_return),
                    "annual_return": float(result.annual_return),
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": float(result.max_drawdown),
                    "total_trades": result.total_trades,
                    "win_rate": result.win_rate,
                    "metadata": result.metadata,
                    "request_config": result.request_config,
                    "created_at": result.created_at,
                }

        except Exception as e:
            logger.error(f"Failed to get backtest result {result_id}: {e}")
            raise ServiceError(
                f"Database error retrieving result: {e}", error_code="REPO_002"
            ) from e

    async def list_backtest_results(
        self, limit: int = 50, offset: int = 0, strategy_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        List backtest results with pagination.

        Args:
            limit: Maximum results to return
            offset: Results offset for pagination
            strategy_type: Optional filter by strategy type

        Returns:
            List of result summaries
        """
        try:
            async with self.db_manager.get_session() as session:
                from sqlalchemy import select

                from src.database.models.backtesting import BacktestResult

                query = select(BacktestResult).order_by(BacktestResult.created_at.desc())

                if strategy_type:
                    # Filter by strategy type in metadata
                    query = query.where(
                        BacktestResult.metadata.op("->>")("strategy_type") == strategy_type
                    )

                query = query.offset(offset).limit(limit)
                results = await session.execute(query)

                return [
                    {
                        "id": str(result.id),
                        "total_return": float(result.total_return),
                        "sharpe_ratio": result.sharpe_ratio,
                        "max_drawdown": float(result.max_drawdown),
                        "total_trades": result.total_trades,
                        "strategy_type": result.metadata.get("strategy_type", "unknown"),
                        "created_at": result.created_at,
                    }
                    for result in results.scalars().all()
                ]

        except Exception as e:
            logger.error(f"Failed to list backtest results: {e}")
            raise ServiceError(f"Database error listing results: {e}", error_code="REPO_003") from e

    async def delete_backtest_result(self, result_id: str) -> bool:
        """
        Delete backtest result by ID.

        Args:
            result_id: Result ID to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            async with self.db_manager.get_session() as session:
                from src.database.models.backtesting import BacktestResult

                result = await session.get(BacktestResult, result_id)
                if not result:
                    return False

                await session.delete(result)
                await session.commit()

                logger.info(f"Backtest result {result_id} deleted")
                return True

        except Exception as e:
            logger.error(f"Failed to delete backtest result {result_id}: {e}")
            raise ServiceError(f"Database error deleting result: {e}", error_code="REPO_004") from e

    async def save_trade_history(self, backtest_id: str, trades: list[dict[str, Any]]) -> None:
        """
        Save trade history for a backtest.

        Args:
            backtest_id: Backtest result ID
            trades: List of trade records
        """
        try:
            async with self.db_manager.get_session() as session:
                from src.database.models.backtesting import BacktestTrade

                trade_records = []
                for trade in trades:
                    record = BacktestTrade(
                        backtest_id=backtest_id,
                        symbol=trade.get("symbol"),
                        entry_time=trade.get("entry_time"),
                        exit_time=trade.get("exit_time"),
                        entry_price=to_decimal(trade.get("entry_price", 0)),
                        exit_price=to_decimal(trade.get("exit_price", 0)),
                        quantity=to_decimal(trade.get("size", 0)),
                        pnl=to_decimal(trade.get("pnl", 0)),
                        side=trade.get("side", "BUY"),
                    )
                    trade_records.append(record)

                session.add_all(trade_records)
                await session.commit()

                logger.info(f"Saved {len(trade_records)} trade records for backtest {backtest_id}")

        except Exception as e:
            logger.error(f"Failed to save trade history: {e}")
            raise ServiceError(f"Database error saving trades: {e}", error_code="REPO_005") from e

    async def get_trade_history(self, backtest_id: str) -> list[dict[str, Any]]:
        """
        Get trade history for a backtest.

        Args:
            backtest_id: Backtest result ID

        Returns:
            List of trade records
        """
        try:
            async with self.db_manager.get_session() as session:
                from sqlalchemy import select

                from src.database.models.backtesting import BacktestTrade

                query = select(BacktestTrade).where(BacktestTrade.backtest_id == backtest_id)
                results = await session.execute(query)

                return [
                    {
                        "symbol": trade.symbol,
                        "entry_time": trade.entry_time,
                        "exit_time": trade.exit_time,
                        "entry_price": float(trade.entry_price),
                        "exit_price": float(trade.exit_price),
                        "quantity": float(trade.quantity),
                        "pnl": float(trade.pnl),
                        "side": trade.side,
                    }
                    for trade in results.scalars().all()
                ]

        except Exception as e:
            logger.error(f"Failed to get trade history for {backtest_id}: {e}")
            raise ServiceError(
                f"Database error retrieving trades: {e}", error_code="REPO_006"
            ) from e

    async def cleanup_old_results(self, days_old: int = 30) -> int:
        """
        Clean up old backtest results.

        Args:
            days_old: Number of days old to consider for cleanup

        Returns:
            Number of results cleaned up
        """
        try:
            async with self.db_manager.get_session() as session:
                from datetime import datetime, timedelta

                from sqlalchemy import delete, select

                from src.database.models.backtesting import BacktestResult

                cutoff_date = datetime.utcnow() - timedelta(days=days_old)

                # Count results to be deleted
                count_query = select(BacktestResult).where(BacktestResult.created_at < cutoff_date)
                result = await session.execute(count_query)
                count = len(result.scalars().all())

                # Delete old results
                delete_query = delete(BacktestResult).where(BacktestResult.created_at < cutoff_date)
                await session.execute(delete_query)
                await session.commit()

                logger.info(f"Cleaned up {count} old backtest results")
                return count

        except Exception as e:
            logger.error(f"Failed to cleanup old results: {e}")
            raise ServiceError(f"Database error during cleanup: {e}", error_code="REPO_007") from e
