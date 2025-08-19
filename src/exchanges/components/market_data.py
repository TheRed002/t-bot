"""Market data provider component for exchanges."""

from typing import Dict, List, Optional, Any
from datetime import datetime
from cachetools import TTLCache

from src.core.logging import get_logger
from src.exchanges.components.connection import ConnectionManager

logger = get_logger(__name__)


class MarketDataProvider:
    """
    Handles market data operations for exchanges.
    
    This component manages ticker data, order books, trades,
    and other market information with caching.
    """
    
    def __init__(
        self,
        connection: ConnectionManager,
        cache_ttl: int = 60,
        cache_size: int = 100
    ):
        """
        Initialize market data provider.
        
        Args:
            connection: Connection manager instance
            cache_ttl: Cache time-to-live in seconds
            cache_size: Maximum cache size
        """
        self.connection = connection
        self._ticker_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self._orderbook_cache = TTLCache(maxsize=cache_size, ttl=5)  # Shorter TTL for order books
        self._trades_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self._logger = logger
    
    async def get_ticker(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get ticker data for a symbol.
        
        Args:
            symbol: Trading symbol
            use_cache: Whether to use cached data
            
        Returns:
            Ticker data
        """
        cache_key = f"ticker_{symbol}"
        
        # Check cache
        if use_cache and cache_key in self._ticker_cache:
            self._logger.debug(f"Ticker cache hit for {symbol}")
            return self._ticker_cache[cache_key]
        
        # Fetch from exchange
        try:
            response = await self.connection.request(
                method='GET',
                endpoint='/api/v3/ticker/24hr',
                params={'symbol': symbol},
                signed=False
            )
            
            # Standardize response
            ticker = self._standardize_ticker(response)
            
            # Cache result
            self._ticker_cache[cache_key] = ticker
            
            return ticker
            
        except Exception as e:
            self._logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise
    
    async def get_all_tickers(self) -> List[Dict[str, Any]]:
        """
        Get ticker data for all symbols.
        
        Returns:
            List of ticker data
        """
        try:
            response = await self.connection.request(
                method='GET',
                endpoint='/api/v3/ticker/24hr',
                signed=False
            )
            
            # Standardize and cache each ticker
            tickers = []
            for ticker_data in response:
                ticker = self._standardize_ticker(ticker_data)
                symbol = ticker.get('symbol')
                if symbol:
                    cache_key = f"ticker_{symbol}"
                    self._ticker_cache[cache_key] = ticker
                tickers.append(ticker)
            
            return tickers
            
        except Exception as e:
            self._logger.error(f"Failed to get all tickers: {e}")
            raise
    
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 100,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of levels to retrieve
            use_cache: Whether to use cached data
            
        Returns:
            Order book data
        """
        cache_key = f"orderbook_{symbol}_{limit}"
        
        # Check cache
        if use_cache and cache_key in self._orderbook_cache:
            self._logger.debug(f"Order book cache hit for {symbol}")
            return self._orderbook_cache[cache_key]
        
        # Fetch from exchange
        try:
            response = await self.connection.request(
                method='GET',
                endpoint='/api/v3/depth',
                params={'symbol': symbol, 'limit': limit},
                signed=False
            )
            
            # Standardize response
            orderbook = self._standardize_orderbook(response)
            
            # Cache result
            self._orderbook_cache[cache_key] = orderbook
            
            return orderbook
            
        except Exception as e:
            self._logger.error(f"Failed to get order book for {symbol}: {e}")
            raise
    
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of trades to retrieve
            
        Returns:
            List of recent trades
        """
        cache_key = f"trades_{symbol}"
        
        # Check cache
        if cache_key in self._trades_cache:
            cached = self._trades_cache[cache_key]
            if len(cached) >= limit:
                self._logger.debug(f"Trades cache hit for {symbol}")
                return cached[:limit]
        
        # Fetch from exchange
        try:
            response = await self.connection.request(
                method='GET',
                endpoint='/api/v3/trades',
                params={'symbol': symbol, 'limit': limit},
                signed=False
            )
            
            # Standardize response
            trades = [self._standardize_trade(trade) for trade in response]
            
            # Cache result
            self._trades_cache[cache_key] = trades
            
            return trades
            
        except Exception as e:
            self._logger.error(f"Failed to get trades for {symbol}: {e}")
            raise
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get kline/candlestick data.
        
        Args:
            symbol: Trading symbol
            interval: Kline interval (1m, 5m, 1h, etc.)
            limit: Number of klines to retrieve
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of kline data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        try:
            response = await self.connection.request(
                method='GET',
                endpoint='/api/v3/klines',
                params=params,
                signed=False
            )
            
            # Standardize response
            klines = [self._standardize_kline(kline) for kline in response]
            
            return klines
            
        except Exception as e:
            self._logger.error(f"Failed to get klines for {symbol}: {e}")
            raise
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange trading rules and symbol information.
        
        Returns:
            Exchange information
        """
        try:
            response = await self.connection.request(
                method='GET',
                endpoint='/api/v3/exchangeInfo',
                signed=False
            )
            
            return response
            
        except Exception as e:
            self._logger.error(f"Failed to get exchange info: {e}")
            raise
    
    def _standardize_ticker(self, data: Dict) -> Dict[str, Any]:
        """Standardize ticker data format."""
        return {
            'symbol': data.get('symbol'),
            'price': float(data.get('lastPrice', 0)),
            'bid': float(data.get('bidPrice', 0)),
            'ask': float(data.get('askPrice', 0)),
            'volume': float(data.get('volume', 0)),
            'quote_volume': float(data.get('quoteVolume', 0)),
            'change_24h': float(data.get('priceChange', 0)),
            'change_percent_24h': float(data.get('priceChangePercent', 0)),
            'high_24h': float(data.get('highPrice', 0)),
            'low_24h': float(data.get('lowPrice', 0)),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _standardize_orderbook(self, data: Dict) -> Dict[str, Any]:
        """Standardize order book data format."""
        return {
            'bids': [
                {'price': float(price), 'quantity': float(qty)}
                for price, qty in data.get('bids', [])
            ],
            'asks': [
                {'price': float(price), 'quantity': float(qty)}
                for price, qty in data.get('asks', [])
            ],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _standardize_trade(self, data: Dict) -> Dict[str, Any]:
        """Standardize trade data format."""
        return {
            'id': data.get('id'),
            'price': float(data.get('price', 0)),
            'quantity': float(data.get('qty', 0)),
            'time': data.get('time'),
            'is_buyer_maker': data.get('isBuyerMaker', False),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _standardize_kline(self, data: List) -> Dict[str, Any]:
        """Standardize kline data format."""
        if isinstance(data, list) and len(data) >= 6:
            return {
                'open_time': data[0],
                'open': float(data[1]),
                'high': float(data[2]),
                'low': float(data[3]),
                'close': float(data[4]),
                'volume': float(data[5]),
                'close_time': data[6] if len(data) > 6 else None,
                'quote_volume': float(data[7]) if len(data) > 7 else 0,
                'trades': int(data[8]) if len(data) > 8 else 0,
                'timestamp': datetime.utcnow().isoformat()
            }
        return {}
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._ticker_cache.clear()
        self._orderbook_cache.clear()
        self._trades_cache.clear()
        self._logger.debug("Market data cache cleared")