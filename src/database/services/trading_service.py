"""Trading service layer implementing business logic for trading operations."""

from datetime import datetime
from decimal import Decimal

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.database.interfaces import TradingDataServiceInterface
from src.database.models.trading import Order, Position, Trade
from src.database.repository.trading import (
    OrderRepository,
    PositionRepository,
    TradeRepository,
)

logger = get_logger(__name__)


class TradingService(BaseService, TradingDataServiceInterface):
    """Service layer for trading operations with business logic."""

    def __init__(
        self,
        database_service=None,  # DatabaseService - injected dependency
        order_repo: OrderRepository | None = None,
        position_repo: PositionRepository | None = None,
        trade_repo: TradeRepository | None = None,
    ):
        """Initialize with injected dependencies."""
        super().__init__(name="TradingService")
        self.database_service = database_service
        self.order_repo = order_repo
        self.position_repo = position_repo
        self.trade_repo = trade_repo

    async def cancel_order(self, order_id: str, reason: str = "User requested") -> bool:
        """Cancel order with business logic validation."""
        try:
            # Get order through database service
            from src.database.models.trading import Order
            
            order = await self.database_service.get_entity_by_id(Order, order_id)
            if not order:
                raise ValidationError(f"Order {order_id} not found")

            if not self._can_cancel_order(order):
                raise ValidationError(f"Order {order_id} cannot be cancelled in status {order.status}")

            # Business logic for cancellation - update status
            order.status = "CANCELLED"
            
            # Update through database service
            updated_order = await self.database_service.update_entity(order)
            success = updated_order is not None
            
            if success:
                logger.info(f"Order {order_id} cancelled: {reason}")
                # Add audit log entry for cancellation
                await self._log_order_cancellation(order_id, reason)

            return success

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise ServiceError(f"Order cancellation failed: {e}") from e

    async def close_position(self, position_id: str, close_price: Decimal) -> bool:
        """Close position with business logic."""
        try:
            # Get position through database service
            from src.database.models.trading import Position
            
            position = await self.database_service.get_entity_by_id(Position, position_id)
            if not position:
                raise ValidationError(f"Position {position_id} not found")

            if position.status != "OPEN":
                raise ValidationError(f"Position {position_id} is not open")

            # Calculate realized P&L using business logic
            realized_pnl = self._calculate_realized_pnl(position, close_price)

            # Update position with business logic values
            position.status = "CLOSED"
            position.exit_price = close_price
            position.closed_at = datetime.utcnow()
            position.realized_pnl = realized_pnl

            # Update through database service
            updated_position = await self.database_service.update_entity(position)
            success = updated_position is not None

            if success:
                logger.info(f"Position {position_id} closed at {close_price} with P&L {realized_pnl}")

            return success

        except Exception as e:
            logger.error(f"Failed to close position {position_id}: {e}")
            raise ServiceError(f"Position close failed: {e}") from e

    async def get_trades_by_bot(
        self,
        bot_id: str,
        limit: int | None = None,
        offset: int = 0,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Trade]:
        """Get trades for bot - delegates to database service."""
        try:
            from src.database.models.trading import Trade
            
            filters = {"bot_id": bot_id}

            if start_time:
                filters["created_at"] = {"gte": start_time}
            if end_time:
                if "created_at" not in filters:
                    filters["created_at"] = {}
                filters["created_at"]["lte"] = end_time

            return await self.database_service.list_entities(
                model_class=Trade,
                filters=filters, 
                limit=limit, 
                offset=offset, 
                order_by="created_at",
                order_desc=True
            )

        except Exception as e:
            logger.error(f"Failed to get trades for bot {bot_id}: {e}")
            raise ServiceError(f"Failed to get trades: {e}") from e

    async def get_positions_by_bot(self, bot_id: str) -> list[Position]:
        """Get positions for bot - delegates to database service."""
        try:
            from src.database.models.trading import Position
            
            return await self.database_service.list_entities(
                model_class=Position,
                filters={"bot_id": bot_id}, 
                order_by="created_at",
                order_desc=True
            )
        except Exception as e:
            logger.error(f"Failed to get positions for bot {bot_id}: {e}")
            raise ServiceError(f"Failed to get positions: {e}") from e

    async def calculate_total_pnl(
        self,
        bot_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> Decimal:
        """Calculate total P&L with business logic."""
        try:
            trades = await self.get_trades_by_bot(bot_id, start_time=start_time, end_time=end_time)

            total_pnl = Decimal("0")
            for trade in trades:
                if trade.pnl:
                    total_pnl += trade.pnl

            return total_pnl

        except Exception as e:
            logger.error(f"Failed to calculate P&L for bot {bot_id}: {e}")
            raise ServiceError(f"P&L calculation failed: {e}") from e

    # Private helper methods for business logic
    def _can_cancel_order(self, order: Order) -> bool:
        """Check if order can be cancelled."""
        cancellable_statuses = ["PENDING", "OPEN", "PARTIALLY_FILLED"]
        return order.status in cancellable_statuses

    def _calculate_realized_pnl(self, position: Position, close_price: Decimal) -> Decimal:
        """Calculate realized P&L for position."""
        if not position.entry_price or not position.quantity:
            return Decimal("0")

        price_diff = close_price - position.entry_price
        if position.side == "SHORT":
            price_diff = -price_diff

        return price_diff * position.quantity

    def _calculate_unrealized_pnl(self, position: Position, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L for position."""
        if not position.entry_price or not position.quantity:
            return Decimal("0")

        price_diff = current_price - position.entry_price
        if position.side == "SHORT":
            price_diff = -price_diff

        return price_diff * position.quantity

    async def update_position_price(self, position_id: str, current_price: Decimal) -> bool:
        """Update position's current price with business logic."""
        try:
            # Get position through database service
            from src.database.models.trading import Position
            
            position = await self.database_service.get_entity_by_id(Position, position_id)
            if not position:
                raise ValidationError(f"Position {position_id} not found")

            if position.status != "OPEN":
                logger.debug(f"Skipping price update for closed position {position_id}")
                return False

            # Calculate unrealized P&L using business logic
            unrealized_pnl = self._calculate_unrealized_pnl(position, current_price)

            # Update fields with business logic values
            position.current_price = current_price
            position.unrealized_pnl = unrealized_pnl

            # Update through database service
            updated_position = await self.database_service.update_entity(position)
            return updated_position is not None

        except Exception as e:
            logger.error(f"Failed to update position price for {position_id}: {e}")
            raise ServiceError(f"Position price update failed: {e}") from e

    async def _log_order_cancellation(self, order_id: str, reason: str) -> None:
        """Log order cancellation for audit."""
        logger.info(f"Order {order_id} cancellation logged: {reason}")
        
    async def create_trade(self, trade_data: dict) -> dict:
        """Create a new trade with business logic validation."""
        try:
            # Validate trade data with business rules
            if not trade_data.get('symbol'):
                raise ValidationError("Symbol is required")
            if not trade_data.get('side') in ['BUY', 'SELL', 'LONG', 'SHORT']:
                raise ValidationError("Valid side is required")
            if not trade_data.get('quantity') or trade_data['quantity'] <= 0:
                raise ValidationError("Positive quantity is required")
                
            # Create trade entity with business logic
            from src.database.models.trading import Trade
            
            trade = Trade(
                symbol=trade_data.get("symbol"),
                side=trade_data.get("side"),
                quantity=trade_data.get("quantity"),
                entry_price=trade_data.get("entry_price"),
                exit_price=trade_data.get("exit_price"),
                pnl=trade_data.get("pnl"),
                bot_id=trade_data.get("bot_id"),
                strategy_id=trade_data.get("strategy_id"),
                exchange=trade_data.get("exchange"),
            )
            
            # Save through database service
            saved_trade = await self.database_service.create_entity(trade)
            
            # Return as dict for API responses
            return {
                "id": saved_trade.id,
                "symbol": saved_trade.symbol,
                "side": saved_trade.side,
                "quantity": str(saved_trade.quantity),
                "price": str(saved_trade.entry_price) if saved_trade.entry_price else None,
                "pnl": str(saved_trade.pnl) if saved_trade.pnl else None,
                "timestamp": saved_trade.created_at,
            }
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to create trade: {e}")
            raise ServiceError(f"Trade creation failed: {e}") from e
            
    async def get_positions(self, strategy_id: str | None = None, symbol: str | None = None) -> list[dict]:
        """Get positions with business logic filtering."""
        try:
            # Build filters with business logic
            from src.database.models.trading import Position
            
            filters = {}
            if strategy_id:
                filters["strategy_id"] = strategy_id
            if symbol:
                filters["symbol"] = symbol
                
            # Get positions from database service
            positions = await self.database_service.list_entities(
                model_class=Position,
                filters=filters, 
                order_by="created_at",
                order_desc=True
            )
            
            # Convert to dict format for API responses
            return [
                {
                    "id": position.id,
                    "symbol": position.symbol,
                    "side": position.side,
                    "quantity": str(position.quantity),
                    "entry_price": str(position.entry_price),
                    "current_price": str(position.current_price) if position.current_price else None,
                    "unrealized_pnl": str(position.unrealized_pnl) if position.unrealized_pnl else None,
                    "status": position.status,
                    "created_at": position.created_at,
                }
                for position in positions
            ]
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise ServiceError(f"Get positions failed: {e}") from e

    async def get_trade_statistics(self, bot_id: str, since: datetime | None = None) -> dict[str, Any]:
        """Get trade statistics with business logic."""
        try:
            from src.database.models.trading import Trade
            
            # Build filters for trades
            filters = {"bot_id": bot_id}
            if since:
                filters["created_at"] = {"gte": since}
            
            # Get trades through database service
            trades = await self.database_service.list_entities(
                model_class=Trade,
                filters=filters,
                order_by="created_at"
            )

            if not trades:
                return {
                    "total_trades": 0,
                    "profitable_trades": 0,
                    "losing_trades": 0,
                    "total_pnl": Decimal("0"),
                    "average_pnl": Decimal("0"),
                    "win_rate": Decimal("0"),
                    "largest_win": Decimal("0"),
                    "largest_loss": Decimal("0"),
                }

            # Business logic for calculating statistics
            profitable = [t for t in trades if t.pnl and t.pnl > 0]
            losing = [t for t in trades if t.pnl and t.pnl <= 0]
            total_pnl = sum(t.pnl or Decimal("0") for t in trades)

            return {
                "total_trades": len(trades),
                "profitable_trades": len(profitable),
                "losing_trades": len(losing),
                "total_pnl": total_pnl,
                "average_pnl": total_pnl / Decimal(str(len(trades))) if trades else Decimal("0"),
                "win_rate": (
                    (Decimal(str(len(profitable))) / Decimal(str(len(trades)))) * Decimal("100") 
                    if trades else Decimal("0")
                ),
                "largest_win": max((t.pnl for t in profitable), default=Decimal("0")),
                "largest_loss": min((t.pnl for t in losing), default=Decimal("0")),
            }

        except Exception as e:
            logger.error(f"Failed to get trade statistics for bot {bot_id}: {e}")
            raise ServiceError(f"Trade statistics calculation failed: {e}") from e

    async def get_total_exposure(self, bot_id: str) -> dict[str, Decimal]:
        """Get total exposure by bot with business logic."""
        try:
            from src.database.models.trading import Position
            
            # Get open positions
            positions = await self.database_service.list_entities(
                model_class=Position,
                filters={"bot_id": bot_id, "status": "OPEN"}
            )

            # Business logic for exposure calculation
            total_long = Decimal("0")
            total_short = Decimal("0")
            
            for position in positions:
                if hasattr(position, 'value') and position.value:
                    if position.side == "LONG":
                        total_long += position.value
                    elif position.side == "SHORT":
                        total_short += position.value
                elif position.quantity and position.entry_price:
                    # Calculate value if not stored
                    value = position.quantity * position.entry_price
                    if position.side == "LONG":
                        total_long += value
                    elif position.side == "SHORT":
                        total_short += value

            return {
                "long": total_long,
                "short": total_short,
                "net": total_long - total_short,
                "gross": total_long + total_short,
            }

        except Exception as e:
            logger.error(f"Failed to get total exposure for bot {bot_id}: {e}")
            raise ServiceError(f"Exposure calculation failed: {e}") from e

    async def get_order_fill_summary(self, order_id: str) -> dict[str, Decimal]:
        """Get order fill summary with business logic."""
        try:
            from src.database.models.trading import OrderFill
            
            # Get fills for the order
            fills = await self.database_service.list_entities(
                model_class=OrderFill,
                filters={"order_id": order_id},
                order_by="created_at"
            )

            if not fills:
                return {
                    "quantity": Decimal("0"),
                    "average_price": Decimal("0"),
                    "total_fees": Decimal("0"),
                }

            # Business logic for fill calculations
            total_quantity = sum(f.quantity for f in fills)
            total_value = sum(f.quantity * f.price for f in fills)
            total_fees = sum(f.fee or Decimal("0") for f in fills)

            return {
                "quantity": total_quantity,
                "average_price": (
                    (total_value / total_quantity) 
                    if total_quantity and total_quantity > Decimal("0") 
                    else Decimal("0")
                ),
                "total_fees": total_fees,
            }

        except Exception as e:
            logger.error(f"Failed to get order fill summary for {order_id}: {e}")
            raise ServiceError(f"Order fill summary calculation failed: {e}") from e
