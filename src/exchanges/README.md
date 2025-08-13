# Exchanges Module

The Exchanges Module provides a unified, production-grade abstraction for multiple cryptocurrency exchanges (Binance, OKX, Coinbase Pro). It standardizes order placement, market data retrieval, websocket streaming, and rate limiting behind a consistent interface, enabling higher layers (execution, risk management, strategies) to operate independently of exchange-specific details.

## What this module does
- Defines a unified exchange interface (`BaseExchange`) and implements it for Binance, OKX, and Coinbase Pro
- Provides advanced and exchange-specific rate limiting and global coordination
- Manages REST and WebSocket connections, including pooling and health monitoring
- Normalizes exchange responses to core types (`OrderRequest`, `OrderResponse`, `OrderBook`, `Ticker`, `Trade`, etc.)
- Offers Order Managers that encapsulate order workflows per exchange

## What this module does NOT do
- Risk management (position limits, VaR, circuit breakers) — see `src/risk_management`
- Strategy generation or signal logic — see `src/strategies`
- Persistent storage or data pipelines — see `src/database` and `src/data`

---

## Submodules and summaries
- `base.py`: Abstract base class defining the unified exchange interface
- `binance.py`: Binance implementation (REST + WebSocket + rate limits)
- `okx.py`: OKX implementation (REST + WebSocket + rate limits)
- `coinbase.py`: Coinbase Pro implementation (REST + WebSocket + rate limits)
- `binance_orders.py`: Binance Order Manager (placement, status, open orders, fee calc)
- `okx_orders.py`: OKX Order Manager (placement, status, fills, fee calc)
- `coinbase_orders.py`: Coinbase Order Manager (placement, status, open orders)
- `rate_limiter.py`: Token bucket and basic per-exchange rate limiter
- `advanced_rate_limiter.py`: Cross-exchange coordination and enforcement
- `connection_manager.py`: REST/WS connection and resource management
- `websocket_pool.py`: Pooling, lifecycle, metrics, and limits for WebSocket connections
- `binance_websocket.py`, `okx_websocket.py`, `coinbase_websocket.py`: WebSocket handlers and message routing
- `factory.py`: `ExchangeFactory` for creating configured exchange instances
- `global_coordinator.py`: Global request/connection coordination and limits
- `health_monitor.py`: Connection health monitoring and recovery hooks

---

## File reference: key classes and functions

### base.py
- `class BaseExchange(ABC)`
  - `async def connect(self) -> bool`
  - `async def disconnect(self) -> None`
  - `async def get_account_balance(self) -> dict[str, Decimal]`
  - `async def place_order(self, order: OrderRequest) -> OrderResponse`
  - `async def cancel_order(self, order_id: str) -> bool`
  - `async def get_order_status(self, order_id: str) -> OrderStatus`
  - `async def get_market_data(self, symbol: str, timeframe: str = "1m") -> MarketData`
  - `async def subscribe_to_stream(self, symbol: str, callback: Callable) -> None`
  - `async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook`
  - `async def get_trade_history(self, symbol: str, limit: int = 100) -> list[Trade]`
  - Optional helpers (used by risk/emergency controls):
    - `async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]`
    - `async def get_positions(self) -> list[Position]`

### binance.py (selected)
- `class BinanceExchange(BaseExchange)` implements all abstract methods
  - Public methods include: `connect`, `disconnect`, `get_account_balance`, `place_order`, `cancel_order`, `get_order_status`, `get_market_data`, `subscribe_to_stream`, `get_order_book`, `get_trade_history`, `get_exchange_info`, `get_ticker`
  - Optional helpers implemented:
    - `async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]`
    - `async def get_positions(self) -> list[Position]` (spot returns empty)
  - Internal conversion helpers: `_convert_order_to_binance`, `_convert_binance_order_to_response`, `_convert_binance_status_to_order_status`

### coinbase.py (selected)
- `class CoinbaseExchange(BaseExchange)` implements all abstract methods
  - Public methods include: `connect`, `disconnect`, `get_account_balance`, `place_order`, `cancel_order`, `get_order_status`, `get_market_data`, `subscribe_to_stream`, `get_order_book`, `get_trade_history`, `get_exchange_info`, `get_ticker`
  - Optional helpers implemented:
    - `async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]`
    - `async def get_positions(self) -> list[Position]` (spot returns empty)
  - Internal conversion helpers: `_convert_order_to_coinbase`, `_convert_coinbase_order_to_response`, `_convert_coinbase_status_to_order_status`, `_convert_timeframe_to_granularity`

### okx.py (selected)
- `class OKXExchange(BaseExchange)` implements all abstract methods
  - Public methods include: `connect`, `disconnect`, `get_account_balance`, `place_order`, `cancel_order`, `get_order_status`, `get_market_data`, `subscribe_to_stream`, `get_order_book`, `get_trade_history`, `get_exchange_info`, `get_ticker`
  - Optional helpers implemented:
    - `async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` (best-effort mapping)
    - `async def get_positions(self) -> list[Position]` (spot returns empty)
  - Internal conversion helpers: multiple `_convert_*` functions per OKX payloads

### *_orders.py
- Each Order Manager provides:
  - `async def place_*_order(order: OrderRequest) -> OrderResponse`
  - `async def cancel_order(order_id: str, ...) -> bool`
  - `async def get_order_status(order_id: str, ...) -> OrderStatus`
  - `async def get_open_orders(symbol: str | None = None) -> list[OrderResponse]`
  - Additional helpers (fees, history, tracking)

### rate_limiter.py / advanced_rate_limiter.py
- `TokenBucket`, `RateLimiter`, `AdvancedRateLimiter`, and decorators to enforce exchange-specific and global limits with async support

### connection_manager.py / websocket_pool.py / *_websocket.py
- REST/WS connection lifecycle, pooling, subscriptions, message dispatch, health checks

---

## Import relationships
### Modules that should import Exchanges
- `src.execution` (order routing), `src.risk_management` (emergency actions), `src.strategies` (direct stream subscriptions when needed), `tests`

### Local module dependencies (only local modules)
- `src.core` (types, config, exceptions, logging)
- `src.utils` (decorators, constants, validators)
- `src.error_handling` (ErrorHandler, connection resilience)

---

## Notes
- Optional `get_open_orders` and `get_positions` are implemented with safe defaults and are used by `src.risk_management.emergency_controls` to avoid duplicating exchange logic in the risk layer.
- WebSocket components abstract streaming details; higher layers receive normalized data through callbacks.
- Rate limiting is enforced both per-exchange and globally to align with the specifications.


