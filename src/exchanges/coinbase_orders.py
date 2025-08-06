"""
Coinbase Order Manager (P-006)

This module implements the Coinbase-specific order management functionality,
including order placement, cancellation, and status tracking.

CRITICAL: This integrates with P-001 (core types, exceptions, config), P-002A (error handling),
and P-003 (base exchange interface) components.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime, timezone

# MANDATORY: Import from P-001
from src.core.types import (
    OrderRequest, OrderResponse, OrderSide, OrderType, OrderStatus
)
from src.core.exceptions import (
    ExchangeError, ExchangeConnectionError, ExchangeRateLimitError,
    ExchangeInsufficientFundsError, ValidationError, ExecutionError
)
from src.core.config import Config
from src.core.logging import get_logger

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# Coinbase-specific imports
from coinbase.rest import RESTClient
# Note: Using generic Exception handling for REST API as no specific exceptions are documented

logger = get_logger(__name__)


class CoinbaseOrderManager:
    """
    Coinbase order manager for handling order operations.
    
    Provides comprehensive order management functionality including:
    - Order placement with validation
    - Order cancellation and modification
    - Order status tracking and monitoring
    - Fee calculation and reporting
    - Order history and analytics
    """
    
    def __init__(self, config: Config, exchange_name: str = "coinbase"):
        """
        Initialize Coinbase order manager.
        
        Args:
            config: Application configuration
            exchange_name: Exchange name (default: "coinbase")
        """
        self.config = config
        self.exchange_name = exchange_name
        self.error_handler = ErrorHandler(config.error_handling)
        
        # Coinbase-specific configuration
        self.api_key = config.exchanges.coinbase_api_key
        self.api_secret = config.exchanges.coinbase_api_secret
        self.sandbox = config.exchanges.coinbase_sandbox
        
        # REST client
        self.client: Optional[RESTClient] = None
        
        # Order tracking
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.order_status_cache: Dict[str, OrderStatus] = {}
        self.order_history: List[OrderResponse] = []
        
        # Fee tracking
        self.total_fees: Dict[str, Decimal] = {}
        self.fee_currency = "USD"
        
        # Initialize rate limiter
        from src.exchanges.rate_limiter import RateLimiter
        self.rate_limiter = RateLimiter(config, exchange_name)
        
        logger.info(f"Initialized {exchange_name} order manager")
    
    async def initialize(self) -> bool:
        """
        Initialize the order manager.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize REST client with sandbox support
            base_url = "api-public.sandbox.exchange.coinbase.com" if self.sandbox else "api.coinbase.com"
            self.client = RESTClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                base_url=base_url
            )
            
            # Test connection
            await self._test_connection()
            
            logger.info(f"Successfully initialized {self.exchange_name} order manager")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_name} order manager: {str(e)}")
            return False
    
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place an order on Coinbase exchange.
        
        Args:
            order: Order request with all necessary details
            
        Returns:
            OrderResponse: Order response with execution details
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Validate order
            if not await self._validate_order(order):
                raise ValidationError("Order validation failed")
            
            # Apply rate limiting
            await self.rate_limiter.acquire("orders_per_second", 1)
            
            # Convert order to Coinbase format
            coinbase_order = self._convert_order_to_coinbase(order)
            
            # Place order
            result = await self.client.create_order(**coinbase_order)
            
            # Convert response to unified format
            order_response = self._convert_coinbase_order_to_response(result)
            
            # Track order
            self.pending_orders[order_response.id] = order
            self.order_status_cache[order_response.id] = OrderStatus.PENDING
            self.order_history.append(order_response)
            
            logger.info(f"Placed order {order_response.id} for {order.symbol}")
            return order_response
            
        except ExchangeConnectionError:
            # Re-raise connection errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to place order: {str(e)}")
            raise ExecutionError(f"Failed to place order: {str(e)}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order on Coinbase exchange.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            bool: True if cancellation successful, False otherwise
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Cancel order
            result = await self.client.cancel_orders([order_id])
            
            # Update tracking
            if order_id in self.order_status_cache:
                self.order_status_cache[order_id] = OrderStatus.CANCELLED
            
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
            
            logger.info(f"Cancelled order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get the status of an order on Coinbase exchange.
        
        Args:
            order_id: ID of the order to check
            
        Returns:
            OrderStatus: Current status of the order
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Get order details
            order = await self.client.get_order(order_id)
            
            # Convert status
            status = self._convert_coinbase_status_to_order_status(order['status'])
            
            # Update cache
            self.order_status_cache[order_id] = status
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            return OrderStatus.UNKNOWN
    
    async def get_order_details(self, order_id: str) -> Optional[OrderResponse]:
        """
        Get detailed information about an order.
        
        Args:
            order_id: ID of the order to retrieve
            
        Returns:
            Optional[OrderResponse]: Order details if found, None otherwise
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Get order details
            order = await self.client.get_order(order_id)
            
            # Convert to unified format
            order_response = self._convert_coinbase_order_to_response(order)
            
            return order_response
            
        except Exception as e:
            logger.error(f"Failed to get order details for {order_id}: {str(e)}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """
        Get all open orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List[OrderResponse]: List of open orders
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Get open orders
            orders = await self.client.list_orders(
                product_id=symbol,
                order_status="OPEN"
            )
            
            # Convert to unified format
            order_responses = []
            for order in orders:
                order_response = self._convert_coinbase_order_to_response(order)
                order_responses.append(order_response)
            
            return order_responses
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {str(e)}")
            return []
    
    async def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[OrderResponse]:
        """
        Get order history.
        
        Args:
            symbol: Optional symbol filter
            limit: Maximum number of orders to retrieve
            
        Returns:
            List[OrderResponse]: List of historical orders
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Get order history
            orders = await self.client.list_orders(
                product_id=symbol,
                limit=limit
            )
            
            # Convert to unified format
            order_responses = []
            for order in orders:
                order_response = self._convert_coinbase_order_to_response(order)
                order_responses.append(order_response)
            
            return order_responses
            
        except Exception as e:
            logger.error(f"Failed to get order history: {str(e)}")
            return []
    
    async def get_fills(self, order_id: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get fill information for orders.
        
        Args:
            order_id: Optional order ID filter
            symbol: Optional symbol filter
            
        Returns:
            List[Dict]: List of fill information
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")
            
            # Get fills
            fills = await self.client.list_fills(
                order_id=order_id,
                product_id=symbol
            )
            
            return fills
            
        except Exception as e:
            logger.error(f"Failed to get fills: {str(e)}")
            return []
    
    async def calculate_fees(self, order: OrderRequest) -> Dict[str, Decimal]:
        """
        Calculate estimated fees for an order.
        
        Args:
            order: Order request to calculate fees for
            
        Returns:
            Dict[str, Decimal]: Fee breakdown
        """
        try:
            # Get product information for fee calculation
            product = await self.client.get_product(order.symbol)
            
            # Calculate fees based on order type and size
            if order.order_type == OrderType.MARKET:
                # Market orders typically have higher fees
                fee_rate = Decimal("0.006")  # 0.6% for market orders
            else:
                # Limit orders have lower fees
                fee_rate = Decimal("0.004")  # 0.4% for limit orders
            
            # Calculate fee amount
            fee_amount = order.quantity * fee_rate
            
            return {
                "fee_rate": fee_rate,
                "fee_amount": fee_amount,
                "fee_currency": self.fee_currency
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate fees: {str(e)}")
            return {
                "fee_rate": Decimal("0"),
                "fee_amount": Decimal("0"),
                "fee_currency": self.fee_currency
            }
    
    def get_total_fees(self) -> Dict[str, Decimal]:
        """
        Get total fees paid.
        
        Returns:
            Dict[str, Decimal]: Total fees by currency
        """
        return self.total_fees.copy()
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """
        Get order statistics.
        
        Returns:
            Dict[str, Any]: Order statistics
        """
        total_orders = len(self.order_history)
        filled_orders = len([o for o in self.order_history if o.status == "FILLED"])
        cancelled_orders = len([o for o in self.order_history if o.status == "CANCELLED"])
        
        return {
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "cancelled_orders": cancelled_orders,
            "fill_rate": filled_orders / total_orders if total_orders > 0 else 0,
            "total_fees": self.total_fees
        }
    
    # Helper methods
    
    async def _test_connection(self) -> None:
        """Test connection to Coinbase API."""
        try:
            # Test connection by getting products (this should always work)
            await self.client.get_products()
        except Exception as e:
            raise ExchangeConnectionError(f"Failed to connect to Coinbase: {str(e)}")
    
    async def _validate_order(self, order: OrderRequest) -> bool:
        """
        Validate order before placement.
        
        Args:
            order: Order to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required fields
            if not order.symbol or not order.side or not order.order_type or not order.quantity:
                logger.error("Missing required order fields")
                return False
            
            # Check quantity
            if order.quantity <= 0:
                logger.error("Order quantity must be positive")
                return False
            
            # Check price for limit orders
            if order.order_type == OrderType.LIMIT and (not order.price or order.price <= 0):
                logger.error("Limit orders must have a positive price")
                return False
            
            # Check symbol format (Coinbase uses format like "BTC-USD")
            if "-" not in order.symbol:
                logger.error("Invalid symbol format for Coinbase")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Order validation failed: {str(e)}")
            return False
    
    def _convert_order_to_coinbase(self, order: OrderRequest) -> Dict[str, Any]:
        """
        Convert unified order to Coinbase format.
        
        Args:
            order: Unified order request
            
        Returns:
            Dict[str, Any]: Coinbase order format
        """
        coinbase_order = {
            "product_id": order.symbol,
            "side": order.side.value.upper(),
            "order_configuration": {}
        }
        
        # Configure order based on type
        if order.order_type == OrderType.MARKET:
            coinbase_order["order_configuration"] = {
                "market_market_ioc": {
                    "quote_size": str(order.quantity)
                }
            }
        elif order.order_type == OrderType.LIMIT:
            coinbase_order["order_configuration"] = {
                "limit_limit_gtc": {
                    "base_size": str(order.quantity),
                    "limit_price": str(order.price)
                }
            }
        elif order.order_type == OrderType.STOP_LOSS:
            coinbase_order["order_configuration"] = {
                "stop_limit_stop_limit_gtc": {
                    "base_size": str(order.quantity),
                    "limit_price": str(order.price),
                    "stop_price": str(order.stop_price)
                }
            }
        
        # Add client order ID if provided
        if order.client_order_id:
            coinbase_order["client_order_id"] = order.client_order_id
        
        return coinbase_order
    
    def _convert_coinbase_order_to_response(self, result: Dict) -> OrderResponse:
        """
        Convert Coinbase order response to unified format.
        
        Args:
            result: Coinbase order response
            
        Returns:
            OrderResponse: Unified order response
        """
        # Extract order configuration
        order_config = result.get('order_configuration', {})
        
        # Determine order type
        order_type = OrderType.LIMIT  # Default
        if 'market_market_ioc' in order_config:
            order_type = OrderType.MARKET
        elif 'stop_limit_stop_limit_gtc' in order_config:
            order_type = OrderType.STOP_LOSS
        
        # Extract quantity and price
        quantity = Decimal("0")
        price = None
        
        if order_type == OrderType.MARKET:
            market_config = order_config.get('market_market_ioc', {})
            quantity = Decimal(str(market_config.get('quote_size', '0')))
        else:
            limit_config = order_config.get('limit_limit_gtc', {})
            quantity = Decimal(str(limit_config.get('base_size', '0')))
            price = Decimal(str(limit_config.get('limit_price', '0')))
        
        return OrderResponse(
            id=result['order_id'],
            client_order_id=result.get('client_order_id'),
            symbol=result['product_id'],
            side=OrderSide.BUY if result['side'] == 'BUY' else OrderSide.SELL,
            order_type=order_type,
            quantity=quantity,
            price=price,
            filled_quantity=Decimal(str(result.get('filled_size', '0'))),
            status=result['status'],
            timestamp=datetime.fromisoformat(result['created_time'].replace('Z', '+00:00'))
        )
    
    def _convert_coinbase_status_to_order_status(self, status: str) -> OrderStatus:
        """
        Convert Coinbase order status to unified OrderStatus.
        
        Args:
            status: Coinbase order status
            
        Returns:
            OrderStatus: Unified order status
        """
        status_mapping = {
            'OPEN': OrderStatus.PENDING,
            'FILLED': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELLED,
            'EXPIRED': OrderStatus.EXPIRED,
            'REJECTED': OrderStatus.REJECTED,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED
        }
        return status_mapping.get(status, OrderStatus.UNKNOWN)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed
        pass 