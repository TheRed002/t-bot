"""Order management component for exchanges."""

from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime
import uuid

from src.core.logging import get_logger
from src.exchanges.components.connection import ConnectionManager

logger = get_logger(__name__)


class OrderManager:
    """
    Handles all order operations for exchanges.
    
    This component manages order placement, cancellation, modification,
    and tracking.
    """
    
    def __init__(self, connection: ConnectionManager):
        """
        Initialize order manager.
        
        Args:
            connection: Connection manager instance
        """
        self.connection = connection
        self._orders: Dict[str, Dict] = {}  # Local order cache
        self._logger = logger
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT/STOP)
            quantity: Order quantity
            price: Order price (for limit orders)
            **kwargs: Additional order parameters
            
        Returns:
            Order response with order ID
        """
        # Generate client order ID for tracking
        client_order_id = f"tbot_{uuid.uuid4().hex[:8]}"
        
        # Prepare order data
        order_data = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': str(quantity),
            'clientOrderId': client_order_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if price is not None:
            order_data['price'] = str(price)
        
        # Add any additional parameters
        order_data.update(kwargs)
        
        # Send order to exchange
        try:
            response = await self.connection.request(
                method='POST',
                endpoint='/api/v3/order',
                data=order_data,
                signed=True
            )
            
            # Cache order locally
            order_id = response.get('orderId', client_order_id)
            self._orders[order_id] = {
                **order_data,
                'orderId': order_id,
                'status': response.get('status', 'NEW'),
                'response': response
            }
            
            self._logger.info(
                f"Order placed: {order_id} - {side} {quantity} {symbol} @ "
                f"{price if price else 'MARKET'}"
            )
            
            return response
            
        except Exception as e:
            self._logger.error(f"Failed to place order: {e}")
            raise
    
    async def cancel_order(
        self,
        order_id: str,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol (required by some exchanges)
            
        Returns:
            Cancellation response
        """
        # Prepare cancellation data
        params = {'orderId': order_id}
        if symbol:
            params['symbol'] = symbol
        elif order_id in self._orders:
            params['symbol'] = self._orders[order_id].get('symbol')
        
        try:
            response = await self.connection.request(
                method='DELETE',
                endpoint='/api/v3/order',
                params=params,
                signed=True
            )
            
            # Update local cache
            if order_id in self._orders:
                self._orders[order_id]['status'] = 'CANCELED'
            
            self._logger.info(f"Order canceled: {order_id}")
            return response
            
        except Exception as e:
            self._logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    async def get_order(
        self,
        order_id: str,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get order details.
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            
        Returns:
            Order details
        """
        # Check local cache first
        if order_id in self._orders:
            cached = self._orders[order_id]
            # If order is terminal, return cached version
            if cached.get('status') in ['FILLED', 'CANCELED', 'REJECTED']:
                return cached
        
        # Fetch from exchange
        params = {'orderId': order_id}
        if symbol:
            params['symbol'] = symbol
        elif order_id in self._orders:
            params['symbol'] = self._orders[order_id].get('symbol')
        
        try:
            response = await self.connection.request(
                method='GET',
                endpoint='/api/v3/order',
                params=params,
                signed=True
            )
            
            # Update cache
            self._orders[order_id] = response
            
            return response
            
        except Exception as e:
            self._logger.error(f"Failed to get order {order_id}: {e}")
            raise
    
    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all open orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        try:
            response = await self.connection.request(
                method='GET',
                endpoint='/api/v3/openOrders',
                params=params,
                signed=True
            )
            
            # Update local cache
            for order in response:
                order_id = order.get('orderId')
                if order_id:
                    self._orders[order_id] = order
            
            return response
            
        except Exception as e:
            self._logger.error(f"Failed to get open orders: {e}")
            raise
    
    async def get_order_history(
        self,
        symbol: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get order history.
        
        Args:
            symbol: Trading symbol
            limit: Number of orders to retrieve
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of historical orders
        """
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        try:
            response = await self.connection.request(
                method='GET',
                endpoint='/api/v3/allOrders',
                params=params,
                signed=True
            )
            
            return response
            
        except Exception as e:
            self._logger.error(f"Failed to get order history: {e}")
            raise
    
    async def modify_order(
        self,
        order_id: str,
        symbol: str,
        quantity: Optional[Decimal] = None,
        price: Optional[Decimal] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Modify an existing order.
        
        Note: Not all exchanges support order modification.
        This may cancel and replace the order.
        
        Args:
            order_id: Order ID to modify
            symbol: Trading symbol
            quantity: New quantity
            price: New price
            **kwargs: Additional parameters
            
        Returns:
            Modified order response
        """
        # Many exchanges don't support direct modification
        # So we cancel and replace
        try:
            # Get current order
            current_order = await self.get_order(order_id, symbol)
            
            # Cancel existing order
            await self.cancel_order(order_id, symbol)
            
            # Place new order with modifications
            new_quantity = quantity or Decimal(current_order.get('origQty', 0))
            new_price = price or (
                Decimal(current_order.get('price', 0)) 
                if current_order.get('price') else None
            )
            
            return await self.place_order(
                symbol=symbol,
                side=current_order.get('side'),
                order_type=current_order.get('type'),
                quantity=new_quantity,
                price=new_price,
                **kwargs
            )
            
        except Exception as e:
            self._logger.error(f"Failed to modify order {order_id}: {e}")
            raise
    
    def get_cached_orders(self) -> Dict[str, Dict]:
        """Get all cached orders."""
        return self._orders.copy()
    
    def clear_cache(self) -> None:
        """Clear the order cache."""
        self._orders.clear()
        self._logger.debug("Order cache cleared")