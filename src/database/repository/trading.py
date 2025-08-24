"""Trading-specific repository implementations."""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.database.models.trading import Order, OrderFill, Position, Trade
from src.database.repository.base import BaseRepository

logger = get_logger(__name__)


class OrderRepository(BaseRepository[Order]):
    """Repository for Order entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Order)

    async def get_active_orders(
        self, bot_id: str | None = None, symbol: str | None = None
    ) -> list[Order]:
        """Get active orders."""
        filters = {"status": ["PENDING", "OPEN", "PARTIALLY_FILLED"]}

        if bot_id:
            filters["bot_id"] = bot_id
        if symbol:
            filters["symbol"] = symbol

        return await self.get_all(filters=filters, order_by="-created_at")

    async def get_by_exchange_id(self, exchange: str, exchange_order_id: str) -> Order | None:
        """Get order by exchange order ID."""
        return await self.get_by(exchange=exchange, exchange_order_id=exchange_order_id)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = await self.get(order_id)
        if order and order.is_active:
            order.status = "CANCELLED"
            await self.update(order)
            return True
        return False

    async def get_orders_by_position(self, position_id: str) -> list[Order]:
        """Get all orders for a position."""
        return await self.get_all(filters={"position_id": position_id})

    async def get_recent_orders(self, hours: int = 24, bot_id: str | None = None) -> list[Order]:
        """Get recent orders."""
        from sqlalchemy import select
        
        since = datetime.utcnow() - timedelta(hours=hours)
        stmt = select(Order).where(Order.created_at >= since)
        
        if bot_id:
            stmt = stmt.where(Order.bot_id == bot_id)
            
        stmt = stmt.order_by(Order.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class PositionRepository(BaseRepository[Position]):
    """Repository for Position entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Position)

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

    async def close_position(self, position_id: str, exit_price: float) -> bool:
        """Close a position."""
        position = await self.get(position_id)
        if position and position.is_open:
            position.status = "CLOSED"
            position.exit_price = exit_price
            position.realized_pnl = position.calculate_pnl(exit_price)
            await self.update(position)
            return True
        return False

    async def update_position_price(self, position_id: str, current_price: float) -> bool:
        """Update position's current price."""
        position = await self.get(position_id)
        if position:
            position.current_price = current_price
            position.unrealized_pnl = position.calculate_pnl(current_price)
            await self.update(position)
            return True
        return False

    async def get_total_exposure(self, bot_id: str) -> dict[str, float]:
        """Get total exposure by bot."""
        positions = await self.get_open_positions(bot_id=bot_id)

        total_long = sum(p.value for p in positions if p.side == "LONG")
        total_short = sum(p.value for p in positions if p.side == "SHORT")

        return {
            "long": total_long,
            "short": total_short,
            "net": total_long - total_short,
            "gross": total_long + total_short,
        }


class TradeRepository(BaseRepository[Trade]):
    """Repository for Trade entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Trade)

    async def get_profitable_trades(self, bot_id: str | None = None) -> list[Trade]:
        """Get profitable trades."""
        filters = {"pnl": {"gt": 0}}

        if bot_id:
            filters["bot_id"] = bot_id

        return await self.get_all(filters=filters, order_by="-created_at")

    async def get_trades_by_symbol(self, symbol: str, bot_id: str | None = None) -> list[Trade]:
        """Get trades for a symbol."""
        filters = {"symbol": symbol}

        if bot_id:
            filters["bot_id"] = bot_id

        return await self.get_all(filters=filters, order_by="-created_at")

    async def get_trade_statistics(
        self, bot_id: str, since: datetime | None = None
    ) -> dict[str, Any]:
        """Get trade statistics."""
        from sqlalchemy import select
        
        stmt = select(Trade).where(Trade.bot_id == bot_id)
        
        if since:
            stmt = stmt.where(Trade.created_at >= since)
            
        result = await self.session.execute(stmt)
        trades = list(result.scalars().all())

        if not trades:
            return {
                "total_trades": 0,
                "profitable_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0,
                "average_pnl": 0,
                "win_rate": 0,
                "largest_win": 0,
                "largest_loss": 0,
            }

        profitable = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        return {
            "total_trades": len(trades),
            "profitable_trades": len(profitable),
            "losing_trades": len(losing),
            "total_pnl": sum(t.pnl for t in trades),
            "average_pnl": sum(t.pnl for t in trades) / len(trades),
            "win_rate": (len(profitable) / len(trades)) * 100 if trades else 0,
            "largest_win": max((t.pnl for t in profitable), default=0),
            "largest_loss": min((t.pnl for t in losing), default=0),
        }

    async def create_from_position(self, position: Position, exit_order: Order) -> Trade:
        """Create trade from closed position."""
        trade = Trade(
            exchange=position.exchange,
            symbol=position.symbol,
            side=position.side,
            position_id=position.id,
            exit_order_id=exit_order.id,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=position.exit_price or exit_order.average_fill_price,
            pnl=position.realized_pnl,
            bot_id=position.bot_id,
            strategy_id=position.strategy_id,
        )

        # Calculate percentage
        if trade.entry_price > 0:
            trade.pnl_percentage = (trade.pnl / (trade.entry_price * trade.quantity)) * 100

        return await self.create(trade)


class OrderFillRepository(BaseRepository[OrderFill]):
    """Repository for OrderFill entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, OrderFill)

    async def get_fills_by_order(self, order_id: str) -> list[OrderFill]:
        """Get all fills for an order."""
        return await self.get_all(filters={"order_id": order_id}, order_by="created_at")

    async def get_total_filled(self, order_id: str) -> dict[str, float]:
        """Get total filled quantity and average price."""
        fills = await self.get_fills_by_order(order_id)

        if not fills:
            return {"quantity": 0, "average_price": 0, "total_fees": 0}

        total_quantity = sum(f.quantity for f in fills)
        total_value = sum(f.quantity * f.price for f in fills)
        total_fees = sum(f.fee or 0 for f in fills)

        return {
            "quantity": total_quantity,
            "average_price": total_value / total_quantity if total_quantity > 0 else 0,
            "total_fees": total_fees,
        }

    async def create_fill(
        self,
        order_id: str,
        price: float,
        quantity: float,
        fee: float = 0,
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
