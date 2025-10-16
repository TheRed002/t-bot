"""
Mock SDK Factory for Exchange Integration Tests

Provides mock exchange SDK clients when real SDKs are not available.
This allows tests to run without requiring actual SDK installations,
while still validating the exchange adapter logic.

These mocks simulate SDK behavior but DO NOT mock internal services.
"""

from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock
from datetime import datetime


class MockBinanceClient:
    """Mock Binance SDK client that simulates API responses."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._connected = False

    async def ping(self) -> Dict[str, Any]:
        """Mock ping response."""
        return {}

    async def get_account(self) -> Dict[str, Any]:
        """Mock account info response."""
        return {
            "balances": [
                {"asset": "BTC", "free": "1.50000000", "locked": "0.00000000"},
                {"asset": "USDT", "free": "10000.00000000", "locked": "0.00000000"},
                {"asset": "ETH", "free": "5.00000000", "locked": "0.00000000"},
            ]
        }

    async def get_exchange_info(self) -> Dict[str, Any]:
        """Mock exchange info response."""
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "1000000.00"},
                        {"filterType": "LOT_SIZE", "minQty": "0.00001", "maxQty": "9000.00"},
                    ],
                },
                {
                    "symbol": "ETHUSDT",
                    "status": "TRADING",
                    "baseAsset": "ETH",
                    "quoteAsset": "USDT",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "100000.00"},
                        {"filterType": "LOT_SIZE", "minQty": "0.0001", "maxQty": "10000.00"},
                    ],
                },
            ]
        }

    async def get_symbol_ticker(self, symbol: str) -> Dict[str, Any]:
        """Mock ticker response."""
        if symbol == "INVALIDSYMBOL":
            raise ValueError(f"Invalid symbol: {symbol}")

        if symbol == "BTCUSDT":
            return {
                "symbol": "BTCUSDT",
                "price": "50000.00",
                "bidPrice": "49995.00",
                "askPrice": "50005.00",
                "volume": "1000.00",
            }
        elif symbol == "ETHUSDT":
            return {
                "symbol": "ETHUSDT",
                "price": "3000.00",
                "bidPrice": "2998.00",
                "askPrice": "3002.00",
                "volume": "5000.00",
            }
        return {"symbol": symbol, "price": "100.00", "bidPrice": "99.50", "askPrice": "100.50", "volume": "100.00"}

    async def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """Mock order book response."""
        if symbol == "INVALIDSYMBOL":
            raise ValueError(f"Invalid symbol: {symbol}")

        bids = [[str(50000.00 - i * 10), str(0.1 * (i + 1))] for i in range(limit)]
        asks = [[str(50000.00 + i * 10), str(0.1 * (i + 1))] for i in range(limit)]

        return {"lastUpdateId": 1234567890, "bids": bids, "asks": asks}

    async def get_recent_trades(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Mock recent trades response."""
        if symbol == "INVALIDSYMBOL":
            raise ValueError(f"Invalid symbol: {symbol}")

        trades = []
        for i in range(limit):
            trades.append(
                {
                    "id": 1000000 + i,
                    "price": str(50000.00 + i * 5),
                    "qty": str(0.1 + i * 0.01),
                    "time": int(datetime.utcnow().timestamp() * 1000) - i * 1000,
                    "isBuyerMaker": i % 2 == 0,
                }
            )
        return trades


class MockCoinbaseClient:
    """Mock Coinbase SDK client that simulates API responses."""

    def __init__(self, api_key: str, api_secret: str, passphrase: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self._connected = False

    async def get_time(self) -> Dict[str, Any]:
        """Mock server time response (used for ping)."""
        return {"iso": datetime.utcnow().isoformat(), "epoch": datetime.utcnow().timestamp()}

    async def get_accounts(self) -> List[Dict[str, Any]]:
        """Mock accounts response."""
        return [
            {"id": "btc-account", "currency": "BTC", "balance": "1.50000000", "available": "1.50000000"},
            {"id": "usd-account", "currency": "USD", "balance": "10000.00", "available": "10000.00"},
            {"id": "eth-account", "currency": "ETH", "balance": "5.00000000", "available": "5.00000000"},
        ]

    async def get_products(self) -> List[Dict[str, Any]]:
        """Mock products (trading pairs) response."""
        return [
            {
                "id": "BTC-USD",
                "base_currency": "BTC",
                "quote_currency": "USD",
                "status": "online",
                "min_market_funds": "10",
                "base_min_size": "0.001",
                "base_max_size": "10000",
            },
            {
                "id": "ETH-USD",
                "base_currency": "ETH",
                "quote_currency": "USD",
                "status": "online",
                "min_market_funds": "10",
                "base_min_size": "0.01",
                "base_max_size": "100000",
            },
        ]

    async def get_product_ticker(self, product_id: str) -> Dict[str, Any]:
        """Mock ticker response."""
        if product_id == "INVALIDSYMBOL":
            raise ValueError(f"Invalid product: {product_id}")

        if product_id == "BTC-USD":
            return {"trade_id": 123456, "price": "50000.00", "bid": "49995.00", "ask": "50005.00", "volume": "1000.00"}
        elif product_id == "ETH-USD":
            return {"trade_id": 123457, "price": "3000.00", "bid": "2998.00", "ask": "3002.00", "volume": "5000.00"}
        return {"trade_id": 123458, "price": "100.00", "bid": "99.50", "ask": "100.50", "volume": "100.00"}

    async def get_product_order_book(self, product_id: str, level: int = 2) -> Dict[str, Any]:
        """Mock order book response."""
        if product_id == "INVALIDSYMBOL":
            raise ValueError(f"Invalid product: {product_id}")

        bids = [[str(50000.00 - i * 10), str(0.1 * (i + 1)), 1] for i in range(10)]
        asks = [[str(50000.00 + i * 10), str(0.1 * (i + 1)), 1] for i in range(10)]

        return {"sequence": 1234567890, "bids": bids, "asks": asks}

    async def get_product_trades(self, product_id: str) -> List[Dict[str, Any]]:
        """Mock recent trades response."""
        if product_id == "INVALIDSYMBOL":
            raise ValueError(f"Invalid product: {product_id}")

        trades = []
        for i in range(10):
            trades.append(
                {
                    "trade_id": 1000000 + i,
                    "price": str(50000.00 + i * 5),
                    "size": str(0.1 + i * 0.01),
                    "time": datetime.utcnow().isoformat(),
                    "side": "buy" if i % 2 == 0 else "sell",
                }
            )
        return trades


class MockOKXClient:
    """Mock OKX SDK client that simulates API responses."""

    def __init__(self, api_key: str, api_secret: str, passphrase: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self._connected = False

    async def get_server_time(self) -> Dict[str, Any]:
        """Mock server time response (used for ping)."""
        return {"code": "0", "msg": "", "data": [{"ts": str(int(datetime.utcnow().timestamp() * 1000))}]}

    async def get_account_balance(self) -> Dict[str, Any]:
        """Mock account balance response."""
        return {
            "code": "0",
            "msg": "",
            "data": [
                {"details": [{"ccy": "BTC", "availBal": "1.5", "frozenBal": "0"}, {"ccy": "USDT", "availBal": "10000", "frozenBal": "0"}]}
            ],
        }

    async def get_instruments(self, inst_type: str = "SPOT") -> Dict[str, Any]:
        """Mock instruments response."""
        return {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "instId": "BTC-USDT",
                    "instType": "SPOT",
                    "baseCcy": "BTC",
                    "quoteCcy": "USDT",
                    "minSz": "0.00001",
                    "maxSz": "9000",
                    "tickSz": "0.1",
                },
                {
                    "instId": "ETH-USDT",
                    "instType": "SPOT",
                    "baseCcy": "ETH",
                    "quoteCcy": "USDT",
                    "minSz": "0.001",
                    "maxSz": "10000",
                    "tickSz": "0.01",
                },
            ],
        }

    async def get_ticker(self, inst_id: str) -> Dict[str, Any]:
        """Mock ticker response."""
        if inst_id == "INVALIDSYMBOL":
            return {"code": "51001", "msg": "Invalid instrument ID", "data": []}

        if inst_id == "BTC-USDT":
            return {
                "code": "0",
                "msg": "",
                "data": [
                    {"instId": "BTC-USDT", "last": "50000", "bidPx": "49995", "askPx": "50005", "vol24h": "1000"}
                ],
            }
        elif inst_id == "ETH-USDT":
            return {
                "code": "0",
                "msg": "",
                "data": [{"instId": "ETH-USDT", "last": "3000", "bidPx": "2998", "askPx": "3002", "vol24h": "5000"}],
            }
        return {
            "code": "0",
            "msg": "",
            "data": [{"instId": inst_id, "last": "100", "bidPx": "99.5", "askPx": "100.5", "vol24h": "100"}],
        }

    async def get_order_book(self, inst_id: str, sz: int = 10) -> Dict[str, Any]:
        """Mock order book response."""
        if inst_id == "INVALIDSYMBOL":
            return {"code": "51001", "msg": "Invalid instrument ID", "data": []}

        bids = [[str(50000.00 - i * 10), str(0.1 * (i + 1)), "0", "1"] for i in range(sz)]
        asks = [[str(50000.00 + i * 10), str(0.1 * (i + 1)), "0", "1"] for i in range(sz)]

        return {"code": "0", "msg": "", "data": [{"asks": asks, "bids": bids, "ts": str(int(datetime.utcnow().timestamp() * 1000))}]}

    async def get_trades(self, inst_id: str, limit: int = 10) -> Dict[str, Any]:
        """Mock recent trades response."""
        if inst_id == "INVALIDSYMBOL":
            return {"code": "51001", "msg": "Invalid instrument ID", "data": []}

        trades = []
        for i in range(limit):
            trades.append(
                {
                    "tradeId": str(1000000 + i),
                    "px": str(50000.00 + i * 5),
                    "sz": str(0.1 + i * 0.01),
                    "ts": str(int(datetime.utcnow().timestamp() * 1000) - i * 1000),
                    "side": "buy" if i % 2 == 0 else "sell",
                }
            )
        return {"code": "0", "msg": "", "data": trades}


def create_mock_binance_client(api_key: str, api_secret: str, testnet: bool = True) -> MockBinanceClient:
    """Create a mock Binance client."""
    return MockBinanceClient(api_key, api_secret, testnet)


def create_mock_coinbase_client(api_key: str, api_secret: str, passphrase: str = None) -> MockCoinbaseClient:
    """Create a mock Coinbase client."""
    return MockCoinbaseClient(api_key, api_secret, passphrase)


def create_mock_okx_client(api_key: str, api_secret: str, passphrase: str) -> MockOKXClient:
    """Create a mock OKX client."""
    return MockOKXClient(api_key, api_secret, passphrase)
