"""Trading-specific repository implementations."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.database.models.trading import Order, OrderFill, Position, Trade
from src.database.repository.base import DatabaseRepository
from src.database.repository.utils import RepositoryUtils

logger = get_logger(__name__)


class OrderRepository(DatabaseRepository):
    """Repository for Order entities."""

    def __init__(self, session: AsyncSession):
        """Initialize with injected session."""
        super().__init__(
            session=session, model=Order, entity_type=Order, key_type=str, name="OrderRepository"
        )

    async def get_active_orders(
        self, bot_id: str | None = None, symbol: str | None = None
    ) -> list[Order]:
        """Get active orders."""
        filters: dict[str, Any] = {"status": ["PENDING", "OPEN", "PARTIALLY_FILLED"]}

        if bot_id:
            filters["bot_id"] = [bot_id]
        if symbol:
            filters["symbol"] = [symbol]

        return await self.get_all(filters=filters, order_by="-created_at")

    async def get_by_exchange_id(self, exchange: str, exchange_order_id: str) -> Order | None:
        """Get order by exchange order ID."""
        return await self.get_by(exchange=exchange, exchange_order_id=exchange_order_id)

    async def update_order_status(self, order_id: str, status: str) -> bool:
        """Update order status - data access only."""
        return await RepositoryUtils.update_entity_status(self, order_id, status, "Order")

    async def get_orders_by_position(self, position_id: str) -> list[Order]:
        """Get all orders for a position."""
        return await RepositoryUtils.get_entities_by_field(self, "position_id", position_id)

    async def get_recent_orders(self, hours: int = 24, bot_id: str | None = None) -> list[Order]:
        """Get recent orders."""
        additional_filters = {"bot_id": bot_id} if bot_id else None
        return await RepositoryUtils.get_recent_entities(self, hours, additional_filters)


class PositionRepository(DatabaseRepository):
    """Repository for Position entities."""

    def __init__(self, session: AsyncSession):
        """Initialize with injected session."""
        super().__init__(
            session=session,
            model=Position,
            entity_type=Position,
            key_type=str,
            name="PositionRepository",
        )

    async def get_open_positions(
        self, bot_id: str | None = None, symbol: str | None = None
    ) -> list[Position]:
        """Get open positions."""
        filters = {"status": "OPEN"}

        if bot_id:
            filters["bot_id"] = bot_id
        if symbol:
            filters["symbol"] = symbol

        return await self.get_all(filters=filters, order_by="-created_at")

    async def get_position_by_symbol(self, bot_id: str, symbol: str, side: str) -> Position | None:
        """Get position by symbol and side."""
        return await self.get_by(bot_id=bot_id, symbol=symbol, side=side, status="OPEN")

    async def update_position_status(self, position_id: str, status: str, **fields) -> bool:
        """Update position status and related fields - data access only."""
        return await RepositoryUtils.update_entity_status(
            self, position_id, status, "Position", **fields
        )

    async def update_position_fields(self, position_id: str, **fields) -> bool:
        """Update position fields - data access only."""
        return await RepositoryUtils.update_entity_fields(self, position_id, "Position", **fields)

    # Business logic methods like get_total_exposure have been moved to TradingService
    # Repository should only handle data access operations


class TradeRepository(DatabaseRepository):
    """Repository for Trade entities."""

    def __init__(self, session: AsyncSession):
        """Initialize with injected session."""
        super().__init__(
            session=session, model=Trade, entity_type=Trade, key_type=str, name="TradeRepository"
        )

    async def get_profitable_trades(self, bot_id: str | None = None) -> list[Trade]:
        """Get profitable trades."""
        filters: dict[str, Any] = {"pnl": {"gt": 0}}

        if bot_id:
            filters["bot_id"] = {"eq": bot_id}

        return await self.get_all(filters=filters, order_by="-created_at")

    async def get_trades_by_symbol(self, symbol: str, bot_id: str | None = None) -> list[Trade]:
        """Get trades for a symbol."""
        filters = {"symbol": symbol}
        if bot_id:
            filters["bot_id"] = bot_id
        return await RepositoryUtils.get_entities_by_multiple_fields(self, filters)

    async def get_trades_by_bot_and_date(
        self, bot_id: str, since: datetime | None = None
    ) -> list[Trade]:
        """Get trades by bot and date - data access only."""
        from sqlalchemy import select

        stmt = select(Trade).where(Trade.bot_id == bot_id)

        if since:
            stmt = stmt.where(Trade.created_at >= since)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create_from_position(self, position: Position, exit_order: Order) -> Trade:
        """Create trade from closed position - data access only."""
        # Basic data validation only - business logic should be in service
        if not position or not exit_order:
            from src.core.exceptions import ValidationError

            raise ValidationError("Position and exit order are required")

        try:
            # Data access - create entity with provided data
            trade = Trade(
                exchange=position.exchange,
                symbol=position.symbol,
                side=position.side,
                position_id=position.id,
                exit_order_id=exit_order.id,
                quantity=position.quantity,
                entry_price=position.entry_price,
                exit_price=position.exit_price or exit_order.price,
                pnl=position.realized_pnl or Decimal("0"),
                bot_id=position.bot_id,
                strategy_id=position.strategy_id,
            )

            return await self.create(trade)

        except Exception as e:
            from src.core.exceptions import RepositoryError

            raise RepositoryError(f"Failed to create trade from position: {e}") from e

    # Business logic methods like get_trade_statistics have been moved to TradingService
    # Repository should only handle data access operations


class OrderFillRepository(DatabaseRepository):
    """Repository for OrderFill entities."""

    def __init__(self, session: AsyncSession):
        """Initialize with injected session."""
        super().__init__(
            session=session,
            model=OrderFill,
            entity_type=OrderFill,
            key_type=str,
            name="OrderFillRepository",
        )

    async def get_fills_by_order(self, order_id: str) -> list[OrderFill]:
        """Get all fills for an order."""
        return await RepositoryUtils.get_entities_by_field(self, "order_id", order_id, "created_at")

    async def get_total_filled(self, order_id: str) -> dict[str, Decimal]:
        """Get total filled summary for an order."""
        fills = await self.get_fills_by_order(order_id)

        if not fills:
            return {
                "quantity": Decimal("0"),
                "average_price": Decimal("0"),
                "total_fees": Decimal("0"),
            }

        total_quantity = Decimal("0")
        total_value = Decimal("0")
        total_fees = Decimal("0")

        for fill in fills:
            if fill.quantity:
                total_quantity += Decimal(str(fill.quantity))
            if fill.price and fill.quantity:
                total_value += Decimal(str(fill.price)) * Decimal(str(fill.quantity))
            if fill.fee:
                total_fees += Decimal(str(fill.fee))

        average_price = total_value / total_quantity if total_quantity > 0 else Decimal("0")

        return {
            "quantity": total_quantity,
            "average_price": average_price,
            "total_fees": total_fees,
        }

    async def create_fill(
        self,
        order_id: str,
        price: Decimal,
        quantity: Decimal,
        fee: Decimal = Decimal("0"),
        fee_currency: str | None = None,
        exchange_fill_id: str | None = None,
    ) -> OrderFill:
        """Create a new fill."""
        fill = OrderFill(
            order_id=order_id,
            price=price,
            quantity=quantity,
            fee=fee,
            fee_currency=fee_currency,
            exchange_fill_id=exchange_fill_id,
        )

        return await self.create(fill)
