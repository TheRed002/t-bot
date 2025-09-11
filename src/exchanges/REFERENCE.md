# EXCHANGES Module Reference

## INTEGRATION
**Dependencies**: core, data, error_handling, monitoring, state, utils
**Used By**: None
**Provides**: BinanceOrderManager, CoinbaseOrderManager, ConnectionManager, ExchangeService, IConnectionManager, ISandboxConnectionManager, IStateService, ITradeLifecycleManager, OKXOrderManager, OKXWebSocketManager
**Patterns**: Async Operations, Circuit Breaker, Component Architecture, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Financial data handling
**Security**:
- Authentication
**Performance**:
- Parallel execution
- Retry mechanisms
- Retry mechanisms
**Architecture**:
- BaseExchange inherits from base architecture
- ConnectionManager inherits from base architecture
- ExchangeFactory inherits from base architecture

## MODULE OVERVIEW
**Files**: 17 Python files
**Classes**: 49
**Functions**: 2

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `ExchangeEvent` ✅

**Purpose**: Base class for exchange events
**Status**: Complete

**Implemented Methods:**

### Implementation: `OrderPlacedEvent` ✅

**Inherits**: ExchangeEvent
**Purpose**: Event emitted when order is placed
**Status**: Complete

### Implementation: `OrderFilledEvent` ✅

**Inherits**: ExchangeEvent
**Purpose**: Event emitted when order is filled
**Status**: Complete

### Implementation: `OrderCancelledEvent` ✅

**Inherits**: ExchangeEvent
**Purpose**: Event emitted when order is cancelled
**Status**: Complete

### Implementation: `BaseExchange` 🔧

**Inherits**: BaseService
**Purpose**: Base class for all exchange implementations following service layer pattern
**Status**: Abstract Base Class

**Implemented Methods:**
- `connected(self) -> bool` - Line 152
- `last_heartbeat(self) -> Optional[datetime]` - Line 158
- `is_connected(self) -> bool` - Line 162
- `async health_check(self) -> HealthCheckResult` - Line 290
- `async connect(self) -> None` - Line 326
- `async disconnect(self) -> None` - Line 331
- `async ping(self) -> bool` - Line 336
- `async load_exchange_info(self) -> ExchangeInfo` - Line 341
- `async get_ticker(self, symbol: str) -> Ticker` - Line 350
- `async get_order_book(self, symbol: str, limit: int = 100) -> OrderBook` - Line 357
- `async get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 364
- `async place_order(self, order_request: OrderRequest) -> OrderResponse` - Line 373
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 393
- `async get_order_status(self, symbol: str, order_id: str) -> OrderResponse` - Line 400
- `async get_open_orders(self, symbol: Optional[str] = None) -> list[OrderResponse]` - Line 407
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 416
- `async get_positions(self) -> list[Position]` - Line 428
- `get_exchange_info(self) -> Optional[ExchangeInfo]` - Line 614
- `get_trading_symbols(self) -> Optional[list[str]]` - Line 618
- `is_symbol_supported(self, symbol: str) -> bool` - Line 622

### Implementation: `MockExchangeError` ✅

**Inherits**: Exception
**Purpose**: Exception for mock exchange testing
**Status**: Complete

### Implementation: `BaseMockExchange` ✅

**Inherits**: BaseExchange
**Purpose**: Mock exchange implementation for testing
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> None` - Line 675
- `async disconnect(self) -> None` - Line 680
- `async ping(self) -> bool` - Line 684
- `async load_exchange_info(self) -> ExchangeInfo` - Line 691
- `async get_ticker(self, symbol: str) -> Ticker` - Line 709
- `async get_order_book(self, symbol: str, limit: int = 100) -> OrderBook` - Line 730
- `async get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 745
- `async place_order(self, order_request: OrderRequest) -> OrderResponse` - Line 765
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 791
- `async get_order_status(self, symbol: str, order_id: str) -> OrderResponse` - Line 802
- `async get_open_orders(self, symbol: Optional[str] = None) -> list[OrderResponse]` - Line 810
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 819
- `async get_positions(self) -> list[Position]` - Line 825

### Implementation: `BinanceExchange` ✅

**Inherits**: BaseExchange
**Purpose**: Binance exchange implementation following service layer pattern
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> None` - Line 122
- `async disconnect(self) -> None` - Line 156
- `async ping(self) -> bool` - Line 198
- `async load_exchange_info(self) -> ExchangeInfo` - Line 208
- `async get_ticker(self, symbol: str) -> Ticker` - Line 263
- `async get_order_book(self, symbol: str, limit: int = 100) -> OrderBook` - Line 308
- `async get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 351
- `async place_order(self, order_request: OrderRequest) -> OrderResponse` - Line 388
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 730
- `async get_order_status(self, symbol: str, order_id: str) -> OrderResponse` - Line 763
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 806
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 859
- `async get_positions(self) -> list[Position]` - Line 888

### Implementation: `BinanceOrderManager` ✅

**Purpose**: Binance order management for specialized order handling
**Status**: Complete

**Implemented Methods:**
- `async place_market_order(self, order: OrderRequest) -> OrderResponse` - Line 115
- `async place_limit_order(self, order: OrderRequest) -> OrderResponse` - Line 168
- `async place_stop_loss_order(self, order: OrderRequest) -> OrderResponse` - Line 223
- `async place_oco_order(self, order: OrderRequest) -> OrderResponse` - Line 277
- `async cancel_order(self, order_id: str, symbol: str) -> bool` - Line 335
- `async get_order_status(self, order_id: str, symbol: str) -> OrderStatus` - Line 367
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 399
- `async get_order_history(self, symbol: str, limit: int = 500) -> list[OrderResponse]` - Line 431
- `calculate_fees(self, order: OrderRequest, fill_price: Decimal) -> Decimal` - Line 461
- `get_tracked_orders(self) -> dict[str, dict]` - Line 687
- `clear_tracked_orders(self) -> None` - Line 695

### Implementation: `BinanceWebSocketHandler` ✅

**Purpose**: Binance WebSocket handler for real-time data streaming
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> bool` - Line 112
- `async disconnect(self) -> None` - Line 157
- `async subscribe_to_ticker_stream(self, symbol: str, callback: Callable) -> None` - Line 231
- `async subscribe_to_orderbook_stream(self, symbol: str, depth: str = '20', callback: Callable | None = None) -> None` - Line 262
- `async subscribe_to_trade_stream(self, symbol: str, callback: Callable) -> None` - Line 297
- `async subscribe_to_user_data_stream(self, callback: Callable) -> None` - Line 328
- `async unsubscribe_from_stream(self, stream_name: str) -> bool` - Line 358
- `get_active_streams(self) -> list[str]` - Line 826
- `is_connected(self) -> bool` - Line 830
- `async health_check(self) -> bool` - Line 834
- `async get_connection_metrics(self) -> dict[str, Any]` - Line 877
- `async get_stream_health(self) -> dict[str, dict[str, Any]]` - Line 917

### Implementation: `CoinbaseExchange` ✅

**Inherits**: BaseExchange
**Purpose**: Coinbase exchange implementation following BaseService pattern
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> None` - Line 114
- `async disconnect(self) -> None` - Line 153
- `async ping(self) -> bool` - Line 189
- `async load_exchange_info(self) -> ExchangeInfo` - Line 203
- `async get_ticker(self, symbol: str) -> Ticker` - Line 233
- `async get_order_book(self, symbol: str, limit: int = 100) -> OrderBook` - Line 259
- `async get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 285
- `async place_order(self, order_request: OrderRequest) -> OrderResponse` - Line 316
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 345
- `async get_order_status(self, order_id: str) -> OrderResponse` - Line 385
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 438
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 476
- `async get_balance(self, asset: str | None = None) -> dict[str, Any]` - Line 498
- `async get_positions(self) -> list[Position]` - Line 553

### Implementation: `CoinbaseOrderManager` ✅

**Purpose**: Coinbase order manager for handling order operations
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> bool` - Line 122
- `async place_order(self, order: OrderRequest) -> OrderResponse` - Line 148
- `async cancel_order(self, order_id: str, symbol: str) -> bool` - Line 197
- `async get_order_status(self, order_id: str) -> OrderStatus` - Line 228
- `async get_order_details(self, order_id: str) -> OrderResponse | None` - Line 257
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 283
- `async get_order_history(self, symbol: str | None = None, limit: int = 100) -> list[OrderResponse]` - Line 312
- `async get_fills(self, order_id: str | None = None, symbol: str | None = None) -> list[dict]` - Line 344
- `async calculate_fees(self, order: OrderRequest) -> dict[str, Decimal]` - Line 368
- `get_total_fees(self) -> dict[str, Decimal]` - Line 409
- `get_order_statistics(self) -> dict[str, Any]` - Line 418

### Implementation: `CoinbaseWebSocketHandler` ✅

**Purpose**: Coinbase WebSocket handler for real-time data streaming
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> bool` - Line 114
- `async disconnect(self) -> None` - Line 170
- `async subscribe_to_ticker(self, symbol: str, callback: Callable) -> None` - Line 214
- `async subscribe_to_orderbook(self, symbol: str, callback: Callable) -> None` - Line 252
- `async subscribe_to_trades(self, symbol: str, callback: Callable) -> None` - Line 290
- `async subscribe_to_user_data(self, callback: Callable) -> None` - Line 328
- `async unsubscribe_from_stream(self, stream_key: str) -> bool` - Line 365
- `async unsubscribe_all(self) -> None` - Line 413
- `async handle_ticker_message(self, message: dict) -> None` - Line 427
- `async handle_orderbook_message(self, message: dict) -> None` - Line 461
- `async handle_trade_message(self, message: dict) -> None` - Line 499
- `async handle_user_message(self, message: dict) -> None` - Line 536
- `async health_check(self) -> bool` - Line 587
- `get_connection_metrics(self) -> dict[str, Any]` - Line 620
- `is_connected(self) -> bool` - Line 659
- `get_active_streams(self) -> dict[str, Any]` - Line 668

### Implementation: `WebSocketConnection` ✅

**Purpose**: WebSocket connection wrapper with automatic reconnection
**Status**: Complete

**Implemented Methods:**
- `is_connected(self) -> bool` - Line 93
- `is_connecting(self) -> bool` - Line 97
- `is_disconnected(self) -> bool` - Line 101
- `async connect(self) -> bool` - Line 118
- `async disconnect(self) -> None` - Line 167
- `async send_message(self, message: dict[str, Any]) -> bool` - Line 192
- `async subscribe(self, channel: str, symbol: str | None = None) -> bool` - Line 222
- `async unsubscribe(self, channel: str, symbol: str | None = None) -> bool` - Line 241
- `async heartbeat(self) -> bool` - Line 260
- `is_healthy(self) -> bool` - Line 284
- `async process_queued_messages(self) -> int` - Line 301

### Implementation: `ConnectionManager` ✅

**Inherits**: BaseService
**Purpose**: Connection manager for exchange APIs and WebSocket streams
**Status**: Complete

**Implemented Methods:**
- `async get_rest_connection(self, endpoint: str = 'default') -> Any | None` - Line 406
- `async create_websocket_connection(self, url: str, connection_id: str = 'default') -> WebSocketConnection` - Line 424
- `async get_connection(self, exchange: str, stream_type: str) -> Any | None` - Line 452
- `async release_connection(self, exchange: str, connection: Any) -> None` - Line 506
- `async handle_connection_failure(self, exchange: str, connection: Any) -> None` - Line 538
- `async get_websocket_connection(self, connection_id: str = 'default') -> WebSocketConnection | None` - Line 579
- `async remove_websocket_connection(self, connection_id: str) -> bool` - Line 593
- `async health_check_all(self) -> dict[str, bool]` - Line 624
- `async check_network_health(self) -> dict[str, Any]` - Line 653
- `async reconnect_all(self) -> dict[str, bool]` - Line 705
- `get_connection_stats(self) -> dict[str, Any]` - Line 736
- `async disconnect_all(self) -> None` - Line 754

### Implementation: `ExchangeFactory` ✅

**Inherits**: BaseService, IExchangeFactory
**Purpose**: Simple factory for creating exchange instances
**Status**: Complete

**Implemented Methods:**
- `register_exchange(self, exchange_name: str, exchange_class: type[BaseExchange]) -> None` - Line 51
- `get_supported_exchanges(self) -> list[str]` - Line 69
- `is_exchange_supported(self, exchange_name: str) -> bool` - Line 78
- `async create_exchange(self, exchange_name: str) -> BaseExchange` - Line 90
- `get_available_exchanges(self) -> list[str]` - Line 126
- `async get_exchange(self, ...) -> IExchange | None` - Line 136
- `async remove_exchange(self, exchange_name: str) -> bool` - Line 170
- `async health_check_all(self) -> dict[str, Any]` - Line 185
- `async disconnect_all(self) -> None` - Line 200
- `register_default_exchanges(self) -> None` - Line 209

### Implementation: `TradeEvent` ✅

**Inherits**: str, Enum
**Purpose**: Trade event enumeration (mirror of state module)
**Status**: Complete

### Implementation: `IStateService` ✅

**Inherits**: Protocol
**Purpose**: Interface for StateService used by exchanges
**Status**: Complete

**Implemented Methods:**
- `async set_state(self, ...) -> bool` - Line 42
- `async get_state(self, state_type: StateType, state_id: str) -> dict[str, Any] | None` - Line 54

### Implementation: `ITradeLifecycleManager` ✅

**Inherits**: Protocol
**Purpose**: Interface for TradeLifecycleManager used by exchanges
**Status**: Complete

**Implemented Methods:**
- `async update_trade_event(self, trade_id: str, event: TradeEvent, event_data: dict[str, Any]) -> None` - Line 62

### Implementation: `IExchange` ✅

**Inherits**: Protocol
**Purpose**: Interface contract for exchange implementations
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> bool` - Line 78
- `async disconnect(self) -> None` - Line 82
- `async health_check(self) -> bool` - Line 86
- `is_connected(self) -> bool` - Line 90
- `async place_order(self, order: OrderRequest) -> OrderResponse` - Line 95
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 99
- `async get_order_status(self, symbol: str, order_id: str) -> OrderResponse` - Line 103
- `async get_market_data(self, symbol: str, timeframe: str = '1m') -> MarketData` - Line 108
- `async get_order_book(self, symbol: str, depth: int = 10) -> OrderBook` - Line 112
- `async get_ticker(self, symbol: str) -> Ticker` - Line 116
- `async get_trade_history(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 120
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 125
- `async get_positions(self) -> list[Position]` - Line 129
- `async get_exchange_info(self) -> ExchangeInfo` - Line 134
- `async subscribe_to_stream(self, symbol: str, callback: Any) -> None` - Line 139
- `exchange_name(self) -> str` - Line 145

### Implementation: `IConnectionManager` ✅

**Inherits**: Protocol
**Purpose**: Interface for connection management implementations
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> bool` - Line 153
- `async disconnect(self) -> None` - Line 157
- `is_connected(self) -> bool` - Line 161
- `async request(self, ...) -> dict[str, Any]` - Line 165

### Implementation: `IRateLimiter` ✅

**Inherits**: Protocol
**Purpose**: Interface for rate limiting implementations
**Status**: Complete

**Implemented Methods:**
- `async acquire(self, weight: int = 1) -> bool` - Line 180
- `async release(self, weight: int = 1) -> None` - Line 184
- `reset(self) -> None` - Line 188
- `get_statistics(self) -> dict[str, Any]` - Line 192

### Implementation: `IHealthMonitor` ✅

**Inherits**: Protocol
**Purpose**: Interface for health monitoring implementations
**Status**: Complete

**Implemented Methods:**
- `record_success(self) -> None` - Line 200
- `record_failure(self) -> None` - Line 204
- `record_latency(self, latency_ms: float) -> None` - Line 208
- `get_health_status(self) -> dict[str, Any]` - Line 212
- `async check_health(self) -> bool` - Line 216

### Implementation: `IExchangeAdapter` ✅

**Inherits**: Protocol
**Purpose**: Interface for exchange adapter implementations
**Status**: Complete

**Implemented Methods:**
- `async place_order(self, **kwargs) -> dict[str, Any]` - Line 224
- `async cancel_order(self, order_id: str, **kwargs) -> dict[str, Any]` - Line 228
- `async get_order(self, order_id: str, **kwargs) -> dict[str, Any]` - Line 232
- `async get_balance(self, asset: str | None = None) -> dict[str, Any]` - Line 236
- `async get_ticker(self, symbol: str) -> dict[str, Any]` - Line 240

### Implementation: `IExchangeFactory` ✅

**Inherits**: Protocol
**Purpose**: Interface for exchange factory implementations
**Status**: Complete

**Implemented Methods:**
- `get_supported_exchanges(self) -> list[str]` - Line 253
- `get_available_exchanges(self) -> list[str]` - Line 257
- `is_exchange_supported(self, exchange_name: str) -> bool` - Line 261
- `async get_exchange(self, ...) -> IExchange | None` - Line 265
- `async create_exchange(self, exchange_name: str) -> IExchange` - Line 271
- `async remove_exchange(self, exchange_name: str) -> bool` - Line 275
- `async health_check_all(self) -> dict[str, Any]` - Line 279
- `async disconnect_all(self) -> None` - Line 283

### Implementation: `SandboxMode` ✅

**Inherits**: str, Enum
**Purpose**: Sandbox operation modes
**Status**: Complete

### Implementation: `ISandboxConnectionManager` ✅

**Inherits**: Protocol
**Purpose**: Interface for sandbox connection management
**Status**: Complete

**Implemented Methods:**
- `async connect_to_sandbox(self) -> bool` - Line 310
- `async connect_to_production(self) -> bool` - Line 314
- `async switch_environment(self, mode: SandboxMode) -> bool` - Line 318
- `get_current_endpoints(self) -> dict[str, str]` - Line 322
- `async validate_environment(self) -> dict[str, Any]` - Line 326

### Implementation: `ISandboxAdapter` ✅

**Inherits**: Protocol
**Purpose**: Interface for sandbox-specific exchange adapters
**Status**: Complete

**Implemented Methods:**
- `sandbox_mode(self) -> SandboxMode` - Line 340
- `async validate_sandbox_credentials(self) -> bool` - Line 344
- `async reset_sandbox_account(self) -> bool` - Line 348
- `async get_sandbox_balance(self, asset: str | None = None) -> dict[str, Any]` - Line 352
- `async place_sandbox_order(self, order: OrderRequest) -> OrderResponse` - Line 356
- `async simulate_order_fill(self, order_id: str, fill_percentage: float = 1.0) -> bool` - Line 360
- `async inject_market_data(self, symbol: str, data: dict[str, Any]) -> bool` - Line 364
- `async simulate_network_error(self, error_type: str, duration_seconds: int = 5) -> None` - Line 368

### Implementation: `BaseMockExchange` ✅

**Inherits**: BaseExchange
**Purpose**: Base mock exchange implementation
**Status**: Complete

**Implemented Methods:**

### Implementation: `MockExchange` ✅

**Inherits**: BaseMockExchange
**Purpose**: Mock exchange implementation for development and testing
**Status**: Complete

**Implemented Methods:**
- `orders(self) -> dict[str, OrderResponse]` - Line 82
- `async connect(self) -> None` - Line 86
- `async disconnect(self) -> None` - Line 134
- `async load_exchange_info(self) -> ExchangeInfo` - Line 141
- `async get_ticker(self, symbol: str) -> dict[str, Any]` - Line 166
- `async get_order_book(self, symbol: str, limit: int = 100) -> OrderBook` - Line 180
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 298
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 304
- `async get_order_status(self, symbol: str, order_id: str) -> OrderResponse` - Line 320
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 327
- `async get_positions(self) -> list[Position]` - Line 335
- `async get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 340
- `async ping(self) -> bool` - Line 366
- `configure(self, ...) -> None` - Line 375
- `set_balance(self, balances: dict[str, Decimal]) -> None` - Line 382
- `set_price(self, symbol: str, price: Decimal) -> None` - Line 386
- `set_order_book(self, symbol: str, order_book: dict) -> None` - Line 390
- `async get_balance(self) -> dict[str, Decimal]` - Line 395
- `async get_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 399
- `async place_order_dict(self, ...) -> dict[str, Any]` - Line 405
- `async get_ticker_dict(self, symbol: str) -> dict[str, Any]` - Line 446
- `async place_order(self, *args, **kwargs)` - Line 461
- `async get_ticker_fallback(self, symbol: str)` - Line 470

### Implementation: `OKXExchange` ✅

**Inherits**: BaseExchange
**Purpose**: OKX exchange implementation following BaseService pattern
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> None` - Line 101
- `async disconnect(self) -> None` - Line 155
- `async ping(self) -> bool` - Line 181
- `async load_exchange_info(self) -> ExchangeInfo` - Line 195
- `async get_ticker(self, symbol: str) -> Ticker` - Line 229
- `async get_order_book(self, symbol: str, limit: int = 100) -> OrderBook` - Line 300
- `async get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 347
- `async place_order(self, order_request: OrderRequest) -> OrderResponse` - Line 409
- `async cancel_order(self, order_id: str, symbol: str) -> OrderResponse` - Line 477
- `async get_order_status(self, order_id: str, symbol: str | None = None) -> OrderStatus` - Line 520
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 554
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 613
- `async get_balance(self, asset: str | None = None) -> dict[str, Any]` - Line 648
- `async get_positions(self) -> list[Position]` - Line 698

### Implementation: `OKXOrderManager` ✅

**Purpose**: OKX order manager for specialized order handling
**Status**: Complete

**Implemented Methods:**
- `async place_order(self, order: OrderRequest) -> OrderResponse` - Line 94
- `async cancel_order(self, order_id: str, symbol: str) -> bool` - Line 156
- `async get_order_status(self, order_id: str) -> OrderStatus` - Line 187
- `async get_order_fills(self, order_id: str) -> list[OrderResponse]` - Line 217
- `async place_stop_loss_order(self, order: OrderRequest, stop_price: Decimal) -> OrderResponse` - Line 266
- `async place_take_profit_order(self, order: OrderRequest, take_profit_price: Decimal) -> OrderResponse` - Line 297
- `async place_oco_order(self, order: OrderRequest, stop_price: Decimal, take_profit_price: Decimal) -> list[OrderResponse]` - Line 328
- `calculate_fee(self, order: OrderRequest, is_maker: bool = False) -> Decimal` - Line 361
- `get_active_orders(self) -> dict[str, dict]` - Line 540
- `get_order_history(self, symbol: str | None = None) -> list[dict]` - Line 549
- `clear_order_history(self) -> None` - Line 566
- `get_fee_rates(self) -> dict[str, Decimal]` - Line 571
- `update_fee_rates(self, maker_rate: Decimal, taker_rate: Decimal) -> None` - Line 580

### Implementation: `OKXWebSocketManager` ✅

**Purpose**: OKX WebSocket manager for real-time data streaming
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> bool` - Line 130
- `async disconnect(self) -> None` - Line 170
- `async subscribe_to_ticker(self, symbol: str, callback: Callable) -> None` - Line 244
- `async subscribe_to_orderbook(self, symbol: str, callback: Callable) -> None` - Line 275
- `async subscribe_to_trades(self, symbol: str, callback: Callable) -> None` - Line 304
- `async subscribe_to_account(self, callback: Callable) -> None` - Line 333
- `is_connected(self) -> bool` - Line 1012
- `async health_check(self) -> bool` - Line 1021
- `get_connection_metrics(self) -> dict[str, Any]` - Line 1062
- `get_status(self) -> str` - Line 1103

### Implementation: `ExchangeService` ✅

**Inherits**: BaseService
**Purpose**: Service layer for exchange operations
**Status**: Complete

**Implemented Methods:**
- `async get_exchange(self, exchange_name: str) -> IExchange` - Line 122
- `async place_order(self, exchange_name: str, order: OrderRequest) -> OrderResponse` - Line 220
- `async cancel_order(self, exchange_name: str, order_id: str, symbol: str | None = None) -> bool` - Line 293
- `async get_order_status(self, exchange_name: str, order_id: str, symbol: str | None = None) -> OrderStatus` - Line 357
- `async get_market_data(self, exchange_name: str, symbol: str, timeframe: str = '1m') -> MarketData` - Line 409
- `async get_order_book(self, exchange_name: str, symbol: str, depth: int = 10) -> OrderBook` - Line 442
- `async get_ticker(self, exchange_name: str, symbol: str) -> Ticker` - Line 466
- `async get_account_balance(self, exchange_name: str) -> dict[str, Decimal]` - Line 489
- `async get_positions(self, exchange_name: str, symbol: str | None = None) -> list[Position]` - Line 507
- `async get_exchange_info(self, exchange_name: str) -> ExchangeInfo` - Line 533
- `get_supported_exchanges(self) -> list[str]` - Line 548
- `get_available_exchanges(self) -> list[str]` - Line 552
- `async get_exchange_status(self, exchange_name: str) -> dict[str, Any]` - Line 556
- `async get_service_health(self) -> dict[str, Any]` - Line 601
- `async disconnect_all_exchanges(self) -> None` - Line 627
- `async get_best_price(self, symbol: str, side: str, exchanges: list[str] | None = None) -> dict[str, Any]` - Line 663
- `async subscribe_to_stream(self, exchange_name: str, symbol: str, callback: Any) -> None` - Line 730

### Implementation: `ExchangeTypes` ✅

**Purpose**: Exchange-specific type definitions and utilities
**Status**: Complete

**Implemented Methods:**
- `validate_symbol(symbol: str) -> bool` - Line 33

### Implementation: `ExchangeCapability` ✅

**Inherits**: Enum
**Purpose**: Exchange capabilities enumeration
**Status**: Complete

### Implementation: `ExchangeTradingPair` ✅

**Inherits**: BaseModel
**Purpose**: Trading pair information
**Status**: Complete

### Implementation: `ExchangeFee` ✅

**Inherits**: BaseModel
**Purpose**: Exchange fee structure
**Status**: Complete

### Implementation: `ExchangeRateLimit` ✅

**Inherits**: BaseModel
**Purpose**: Exchange rate limit configuration
**Status**: Complete

### Implementation: `ExchangeConnectionConfig` ✅

**Inherits**: BaseModel
**Purpose**: Exchange connection configuration
**Status**: Complete

### Implementation: `ExchangeOrderBookLevel` ✅

**Inherits**: BaseModel
**Purpose**: Order book level information
**Status**: Complete

### Implementation: `ExchangeOrderBookSnapshot` ✅

**Inherits**: BaseModel
**Purpose**: Order book snapshot
**Status**: Complete

### Implementation: `ExchangeTrade` ✅

**Inherits**: BaseModel
**Purpose**: Exchange trade information
**Status**: Complete

### Implementation: `ExchangeBalance` ✅

**Inherits**: BaseModel
**Purpose**: Exchange balance information
**Status**: Complete

### Implementation: `ExchangePosition` ✅

**Inherits**: BaseModel
**Purpose**: Exchange position information
**Status**: Complete

### Implementation: `ExchangeOrder` ✅

**Inherits**: BaseModel
**Purpose**: Exchange order information
**Status**: Complete

### Implementation: `ExchangeWebSocketMessage` ✅

**Inherits**: BaseModel
**Purpose**: WebSocket message structure
**Status**: Complete

### Implementation: `ExchangeErrorResponse` ✅

**Inherits**: BaseModel
**Purpose**: Exchange error response structure
**Status**: Complete

### Implementation: `ExchangeHealthStatus` ✅

**Inherits**: BaseModel
**Purpose**: Exchange health status
**Status**: Complete

## COMPLETE API REFERENCE

### File: base.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.core.base.interfaces import HealthCheckResult`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.exceptions import ExchangeConnectionError`
- `from src.core.exceptions import OrderRejectionError`

#### Class: `ExchangeEvent`

**Purpose**: Base class for exchange events

```python
class ExchangeEvent:
    def __init__(self)  # Line 60
```

#### Class: `OrderPlacedEvent`

**Inherits**: ExchangeEvent
**Purpose**: Event emitted when order is placed

```python
class OrderPlacedEvent(ExchangeEvent):
```

#### Class: `OrderFilledEvent`

**Inherits**: ExchangeEvent
**Purpose**: Event emitted when order is filled

```python
class OrderFilledEvent(ExchangeEvent):
```

#### Class: `OrderCancelledEvent`

**Inherits**: ExchangeEvent
**Purpose**: Event emitted when order is cancelled

```python
class OrderCancelledEvent(ExchangeEvent):
```

#### Class: `BaseExchange`

**Inherits**: BaseService
**Purpose**: Base class for all exchange implementations following service layer pattern

```python
class BaseExchange(BaseService):
    def __init__(self, name: str, config: dict[str, Any])  # Line 91
    def connected(self) -> bool  # Line 152
    def last_heartbeat(self) -> Optional[datetime]  # Line 158
    def is_connected(self) -> bool  # Line 162
    async def _update_connection_state(self, connected: bool) -> None  # Line 166
    async def _do_start(self) -> None  # Line 185
    async def _do_stop(self) -> None  # Line 280
    async def health_check(self) -> HealthCheckResult  # Line 290
    async def connect(self) -> None  # Line 326
    async def disconnect(self) -> None  # Line 331
    async def ping(self) -> bool  # Line 336
    async def load_exchange_info(self) -> ExchangeInfo  # Line 341
    async def get_ticker(self, symbol: str) -> Ticker  # Line 350
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook  # Line 357
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 364
    async def place_order(self, order_request: OrderRequest) -> OrderResponse  # Line 373
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 393
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse  # Line 400
    async def get_open_orders(self, symbol: Optional[str] = None) -> list[OrderResponse]  # Line 407
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 416
    async def get_positions(self) -> list[Position]  # Line 428
    async def _validate_order(self, order_request: OrderRequest) -> None  # Line 434
    async def _validate_market_data(self, data: Any) -> bool  # Line 491
    async def _process_market_data(self, data: Any) -> Any  # Line 495
    async def _persist_order(self, order_response: OrderResponse) -> None  # Line 516
    async def _persist_market_data(self, ticker: Ticker) -> None  # Line 541
    async def _persist_position(self, position: Position) -> None  # Line 556
    async def _track_analytics(self, event_type: str, data: dict) -> None  # Line 579
    def get_exchange_info(self) -> Optional[ExchangeInfo]  # Line 614
    def get_trading_symbols(self) -> Optional[list[str]]  # Line 618
    def is_symbol_supported(self, symbol: str) -> bool  # Line 622
    def _validate_symbol(self, symbol: str) -> None  # Line 630
    def _validate_price(self, price: Decimal) -> None  # Line 637
    def _validate_quantity(self, quantity: Decimal) -> None  # Line 644
```

#### Class: `MockExchangeError`

**Inherits**: Exception
**Purpose**: Exception for mock exchange testing

```python
class MockExchangeError(Exception):
```

#### Class: `BaseMockExchange`

**Inherits**: BaseExchange
**Purpose**: Mock exchange implementation for testing

```python
class BaseMockExchange(BaseExchange):
    def __init__(self, name: str = 'mock', config: Optional[dict[str, Any]] = None)  # Line 666
    async def connect(self) -> None  # Line 675
    async def disconnect(self) -> None  # Line 680
    async def ping(self) -> bool  # Line 684
    async def load_exchange_info(self) -> ExchangeInfo  # Line 691
    async def get_ticker(self, symbol: str) -> Ticker  # Line 709
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook  # Line 730
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 745
    async def place_order(self, order_request: OrderRequest) -> OrderResponse  # Line 765
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 791
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse  # Line 802
    async def get_open_orders(self, symbol: Optional[str] = None) -> list[OrderResponse]  # Line 810
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 819
    async def get_positions(self) -> list[Position]  # Line 825
```

### File: binance.py

**Key Imports:**
- `from src.core.exceptions import CapitalAllocationError`
- `from src.core.exceptions import ExchangeConnectionError`
- `from src.core.exceptions import ExchangeError`
- `from src.core.exceptions import ExecutionError`
- `from src.core.exceptions import OrderRejectionError`

#### Class: `BinanceExchange`

**Inherits**: BaseExchange
**Purpose**: Binance exchange implementation following service layer pattern

```python
class BinanceExchange(BaseExchange):
    def __init__(self, config: dict[str, Any])  # Line 72
    def _validate_service_config(self, config: dict[str, Any]) -> bool  # Line 103
    async def connect(self) -> None  # Line 122
    async def disconnect(self) -> None  # Line 156
    async def ping(self) -> bool  # Line 198
    async def load_exchange_info(self) -> ExchangeInfo  # Line 208
    async def get_ticker(self, symbol: str) -> Ticker  # Line 263
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook  # Line 308
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 351
    async def place_order(self, order_request: OrderRequest) -> OrderResponse  # Line 388
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 730
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse  # Line 763
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 806
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 859
    async def get_positions(self) -> list[Position]  # Line 888
    def _map_order_type(self, binance_type: str) -> OrderType  # Line 894
    def _map_to_binance_order_type(self, order_type: OrderType) -> str  # Line 906
```

### File: binance_orders.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import ExchangeError`
- `from src.core.exceptions import ExecutionError`
- `from src.core.exceptions import OrderRejectionError`
- `from src.core.exceptions import ValidationError`

#### Class: `BinanceOrderManager`

**Purpose**: Binance order management for specialized order handling

```python
class BinanceOrderManager:
    def __init__(self, ...)  # Line 63
    def _get_asset_precision(self, symbol: str, precision_type: str = 'quantity') -> int  # Line 98
    async def _handle_order_error(self, error: Exception, operation: str, order: OrderRequest = None) -> None  # Line 102
    async def place_market_order(self, order: OrderRequest) -> OrderResponse  # Line 115
    async def place_limit_order(self, order: OrderRequest) -> OrderResponse  # Line 168
    async def place_stop_loss_order(self, order: OrderRequest) -> OrderResponse  # Line 223
    async def place_oco_order(self, order: OrderRequest) -> OrderResponse  # Line 277
    async def cancel_order(self, order_id: str, symbol: str) -> bool  # Line 335
    async def get_order_status(self, order_id: str, symbol: str) -> OrderStatus  # Line 367
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 399
    async def get_order_history(self, symbol: str, limit: int = 500) -> list[OrderResponse]  # Line 431
    def calculate_fees(self, order: OrderRequest, fill_price: Decimal) -> Decimal  # Line 461
    def _validate_market_order(self, order: OrderRequest) -> None  # Line 487
    def _validate_limit_order(self, order: OrderRequest) -> None  # Line 500
    def _validate_stop_loss_order(self, order: OrderRequest) -> None  # Line 515
    def _validate_oco_order(self, order: OrderRequest) -> None  # Line 530
    def _convert_market_order_to_binance(self, order: OrderRequest) -> dict[str, Any]  # Line 552
    def _convert_limit_order_to_binance(self, order: OrderRequest) -> dict[str, Any]  # Line 566
    def _convert_stop_loss_order_to_binance(self, order: OrderRequest) -> dict[str, Any]  # Line 583
    def _convert_oco_order_to_binance(self, order: OrderRequest) -> dict[str, Any]  # Line 599
    def _convert_binance_order_to_response(self, result: dict) -> OrderResponse  # Line 618
    def _convert_binance_oco_order_to_response(self, result: dict) -> OrderResponse  # Line 642
    def _convert_binance_type_to_order_type(self, binance_type: str) -> OrderType  # Line 653
    def _convert_binance_status_to_order_status(self, status: str) -> OrderStatus  # Line 665
    def _track_order(self, order: OrderResponse) -> None  # Line 671
    def get_tracked_orders(self) -> dict[str, dict]  # Line 687
    def clear_tracked_orders(self) -> None  # Line 695
```

### File: binance_websocket.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import ExchangeError`
- `from src.core.logging import get_logger`
- `from src.core.types.market import OrderBook`
- `from src.core.types.market import OrderBookLevel`

#### Class: `BinanceWebSocketHandler`

**Purpose**: Binance WebSocket handler for real-time data streaming

```python
class BinanceWebSocketHandler:
    def __init__(self, ...)  # Line 46
    async def connect(self) -> bool  # Line 112
    async def disconnect(self) -> None  # Line 157
    async def subscribe_to_ticker_stream(self, symbol: str, callback: Callable) -> None  # Line 231
    async def subscribe_to_orderbook_stream(self, symbol: str, depth: str = '20', callback: Callable | None = None) -> None  # Line 262
    async def subscribe_to_trade_stream(self, symbol: str, callback: Callable) -> None  # Line 297
    async def subscribe_to_user_data_stream(self, callback: Callable) -> None  # Line 328
    async def unsubscribe_from_stream(self, stream_name: str) -> bool  # Line 358
    async def _handle_ticker_stream(self, stream_name: str, stream) -> None  # Line 400
    async def _handle_orderbook_stream(self, stream_name: str, stream) -> None  # Line 446
    async def _handle_trade_stream(self, stream_name: str, stream) -> None  # Line 482
    async def _handle_user_data_stream(self, stream_name: str, stream) -> None  # Line 518
    def _convert_ticker_message(self, msg: dict) -> Ticker  # Line 564
    def _convert_orderbook_message(self, msg: dict) -> OrderBook  # Line 585
    def _convert_trade_message(self, msg: dict) -> Trade  # Line 604
    async def _handle_account_position_update(self, msg: dict) -> None  # Line 619
    async def _handle_execution_report(self, msg: dict) -> None  # Line 637
    async def _handle_balance_update(self, msg: dict) -> None  # Line 651
    async def _close_stream(self, stream_name: str) -> None  # Line 664
    async def _handle_stream_error(self, stream_name: str) -> None  # Line 703
    async def _schedule_reconnect(self) -> None  # Line 718
    async def _reconnect_after_delay(self, delay: float) -> None  # Line 745
    async def _health_monitor(self) -> None  # Line 795
    def get_active_streams(self) -> list[str]  # Line 826
    def is_connected(self) -> bool  # Line 830
    async def health_check(self) -> bool  # Line 834
    async def get_connection_metrics(self) -> dict[str, Any]  # Line 877
    async def get_stream_health(self) -> dict[str, dict[str, Any]]  # Line 917
```

### File: coinbase.py

**Key Imports:**
- `from src.core.exceptions import ExchangeConnectionError`
- `from src.core.exceptions import ExecutionError`
- `from src.core.exceptions import OrderRejectionError`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`

#### Class: `CoinbaseExchange`

**Inherits**: BaseExchange
**Purpose**: Coinbase exchange implementation following BaseService pattern

```python
class CoinbaseExchange(BaseExchange):
    def __init__(self, config: dict[str, Any])  # Line 83
    async def connect(self) -> None  # Line 114
    async def disconnect(self) -> None  # Line 153
    async def ping(self) -> bool  # Line 189
    async def load_exchange_info(self) -> ExchangeInfo  # Line 203
    async def get_ticker(self, symbol: str) -> Ticker  # Line 233
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook  # Line 259
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 285
    async def place_order(self, order_request: OrderRequest) -> OrderResponse  # Line 316
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 345
    async def get_order_status(self, order_id: str) -> OrderResponse  # Line 385
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 438
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 476
    async def get_balance(self, asset: str | None = None) -> dict[str, Any]  # Line 498
    async def get_positions(self) -> list[Position]  # Line 553
    async def _test_coinbase_connection(self) -> None  # Line 565
    async def _validate_coinbase_order(self, order: OrderRequest) -> None  # Line 594
    async def _place_order_advanced_api(self, order: OrderRequest) -> OrderResponse  # Line 618
    async def _place_order_pro_api(self, order: OrderRequest) -> OrderResponse  # Line 676
    def _extract_quantity_from_response(self, response, order: OrderRequest) -> Decimal  # Line 700
    def _extract_price_from_response(self, response, order: OrderRequest) -> Decimal | None  # Line 728
    def _extract_order_type_from_response(self, response, order: OrderRequest) -> OrderType  # Line 750
    def _convert_to_coinbase_symbol(self, symbol: str) -> str  # Line 774
```

### File: coinbase_orders.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import ExchangeConnectionError`
- `from src.core.exceptions import ExecutionError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `CoinbaseOrderManager`

**Purpose**: Coinbase order manager for handling order operations

```python
class CoinbaseOrderManager:
    def __init__(self, ...)  # Line 55
    def _get_asset_precision(self, symbol: str, precision_type: str = 'quantity') -> int  # Line 96
    def _create_rate_limiter(self, config: Config) -> Any  # Line 100
    async def initialize(self) -> bool  # Line 122
    async def place_order(self, order: OrderRequest) -> OrderResponse  # Line 148
    async def cancel_order(self, order_id: str, symbol: str) -> bool  # Line 197
    async def get_order_status(self, order_id: str) -> OrderStatus  # Line 228
    async def get_order_details(self, order_id: str) -> OrderResponse | None  # Line 257
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 283
    async def get_order_history(self, symbol: str | None = None, limit: int = 100) -> list[OrderResponse]  # Line 312
    async def get_fills(self, order_id: str | None = None, symbol: str | None = None) -> list[dict]  # Line 344
    async def calculate_fees(self, order: OrderRequest) -> dict[str, Decimal]  # Line 368
    def get_total_fees(self) -> dict[str, Decimal]  # Line 409
    def get_order_statistics(self) -> dict[str, Any]  # Line 418
    async def _test_connection(self) -> None  # Line 439
    async def _validate_order(self, order: OrderRequest) -> bool  # Line 447
    def _convert_order_to_coinbase(self, order: OrderRequest) -> dict[str, Any]  # Line 481
    def _convert_coinbase_order_to_response(self, result: dict) -> OrderResponse  # Line 527
    def _convert_coinbase_status_to_order_status(self, status: str) -> OrderStatus  # Line 573
    async def __aenter__(self)  # Line 577
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 582
```

### File: coinbase_websocket.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import ExchangeConnectionError`
- `from src.core.exceptions import ExchangeError`
- `from src.core.logging import get_logger`
- `from src.core.types.market import OrderBook`

#### Class: `CoinbaseWebSocketHandler`

**Purpose**: Coinbase WebSocket handler for real-time data streaming

```python
class CoinbaseWebSocketHandler:
    def __init__(self, ...)  # Line 46
    async def connect(self) -> bool  # Line 114
    async def disconnect(self) -> None  # Line 170
    async def subscribe_to_ticker(self, symbol: str, callback: Callable) -> None  # Line 214
    async def subscribe_to_orderbook(self, symbol: str, callback: Callable) -> None  # Line 252
    async def subscribe_to_trades(self, symbol: str, callback: Callable) -> None  # Line 290
    async def subscribe_to_user_data(self, callback: Callable) -> None  # Line 328
    async def unsubscribe_from_stream(self, stream_key: str) -> bool  # Line 365
    async def unsubscribe_all(self) -> None  # Line 413
    async def handle_ticker_message(self, message: dict) -> None  # Line 427
    async def handle_orderbook_message(self, message: dict) -> None  # Line 461
    async def handle_trade_message(self, message: dict) -> None  # Line 499
    async def handle_user_message(self, message: dict) -> None  # Line 536
    async def _handle_order_update(self, message: dict) -> None  # Line 565
    async def _handle_account_update(self, message: dict) -> None  # Line 576
    async def health_check(self) -> bool  # Line 587
    def get_connection_metrics(self) -> dict[str, Any]  # Line 620
    def is_connected(self) -> bool  # Line 659
    def get_active_streams(self) -> dict[str, Any]  # Line 668
    def _create_auth_headers(self) -> dict[str, str]  # Line 677
    async def _listen_messages(self) -> None  # Line 700
    async def _handle_message(self, message: str) -> None  # Line 726
    async def _handle_ticker_message(self, data: dict) -> None  # Line 762
    async def _handle_orderbook_message(self, data: dict) -> None  # Line 801
    async def _handle_trade_message(self, data: dict) -> None  # Line 849
    async def _handle_user_message(self, data: dict) -> None  # Line 889
    async def _handle_disconnect(self) -> None  # Line 913
    async def _schedule_reconnect(self) -> None  # Line 923
    async def _reconnect_after_delay(self, delay: float) -> None  # Line 952
    async def _health_monitor(self) -> None  # Line 1008
    async def __aenter__(self)  # Line 1039
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 1044
```

### File: connection_manager.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.config import Config`
- `from src.core.exceptions import ExchangeConnectionError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `WebSocketConnection`

**Purpose**: WebSocket connection wrapper with automatic reconnection

```python
class WebSocketConnection:
    def __init__(self, ...)  # Line 51
    def is_connected(self) -> bool  # Line 93
    def is_connecting(self) -> bool  # Line 97
    def is_disconnected(self) -> bool  # Line 101
    def __str__(self) -> str  # Line 105
    def __repr__(self) -> str  # Line 109
    async def connect(self) -> bool  # Line 118
    async def disconnect(self) -> None  # Line 167
    async def send_message(self, message: dict[str, Any]) -> bool  # Line 192
    async def subscribe(self, channel: str, symbol: str | None = None) -> bool  # Line 222
    async def unsubscribe(self, channel: str, symbol: str | None = None) -> bool  # Line 241
    async def heartbeat(self) -> bool  # Line 260
    def is_healthy(self) -> bool  # Line 284
    async def process_queued_messages(self) -> int  # Line 301
```

#### Class: `ConnectionManager`

**Inherits**: BaseService
**Purpose**: Connection manager for exchange APIs and WebSocket streams

```python
class ConnectionManager(BaseService):
    def __init__(self, ...)  # Line 332
    async def _handle_connection_error(self, error: Exception, operation: str, connection_id: str | None = None) -> None  # Line 372
    async def get_rest_connection(self, endpoint: str = 'default') -> Any | None  # Line 406
    async def create_websocket_connection(self, url: str, connection_id: str = 'default') -> WebSocketConnection  # Line 424
    async def get_connection(self, exchange: str, stream_type: str) -> Any | None  # Line 452
    async def release_connection(self, exchange: str, connection: Any) -> None  # Line 506
    async def handle_connection_failure(self, exchange: str, connection: Any) -> None  # Line 538
    async def get_websocket_connection(self, connection_id: str = 'default') -> WebSocketConnection | None  # Line 579
    async def remove_websocket_connection(self, connection_id: str) -> bool  # Line 593
    async def health_check_all(self) -> dict[str, bool]  # Line 624
    async def check_network_health(self) -> dict[str, Any]  # Line 653
    async def reconnect_all(self) -> dict[str, bool]  # Line 705
    def get_connection_stats(self) -> dict[str, Any]  # Line 736
    async def disconnect_all(self) -> None  # Line 754
    async def __aenter__(self)  # Line 784
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 788
```

### File: di_registration.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.dependency_injection import DependencyContainer`
- `from src.core.logging import get_logger`
- `from src.exchanges.binance import BinanceExchange`
- `from src.exchanges.coinbase import CoinbaseExchange`

#### Functions:

```python
def register_exchange_dependencies(container: DependencyContainer, config: Config) -> None  # Line 27
def setup_exchange_services(config: Config) -> DependencyContainer  # Line 69
```

### File: factory.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.core.config import Config`
- `from src.core.dependency_injection import DependencyContainer`
- `from src.core.exceptions import ExchangeError`
- `from src.core.exceptions import ValidationError`

#### Class: `ExchangeFactory`

**Inherits**: BaseService, IExchangeFactory
**Purpose**: Simple factory for creating exchange instances

```python
class ExchangeFactory(BaseService, IExchangeFactory):
    def __init__(self, config: Config, container: DependencyContainer | None = None) -> None  # Line 26
    async def _do_start(self) -> None  # Line 43
    async def _do_stop(self) -> None  # Line 47
    def register_exchange(self, exchange_name: str, exchange_class: type[BaseExchange]) -> None  # Line 51
    def get_supported_exchanges(self) -> list[str]  # Line 69
    def is_exchange_supported(self, exchange_name: str) -> bool  # Line 78
    async def create_exchange(self, exchange_name: str) -> BaseExchange  # Line 90
    def get_available_exchanges(self) -> list[str]  # Line 126
    async def get_exchange(self, ...) -> IExchange | None  # Line 136
    async def remove_exchange(self, exchange_name: str) -> bool  # Line 170
    async def health_check_all(self) -> dict[str, Any]  # Line 185
    async def disconnect_all(self) -> None  # Line 200
    def register_default_exchanges(self) -> None  # Line 209
```

### File: interfaces.py

**Key Imports:**
- `from src.core.types import ExchangeInfo`
- `from src.core.types import MarketData`
- `from src.core.types import OrderBook`
- `from src.core.types import OrderRequest`
- `from src.core.types import OrderResponse`

#### Class: `TradeEvent`

**Inherits**: str, Enum
**Purpose**: Trade event enumeration (mirror of state module)

```python
class TradeEvent(str, Enum):
```

#### Class: `IStateService`

**Inherits**: Protocol
**Purpose**: Interface for StateService used by exchanges

```python
class IStateService(Protocol):
    async def set_state(self, ...) -> bool  # Line 42
    async def get_state(self, state_type: StateType, state_id: str) -> dict[str, Any] | None  # Line 54
```

#### Class: `ITradeLifecycleManager`

**Inherits**: Protocol
**Purpose**: Interface for TradeLifecycleManager used by exchanges

```python
class ITradeLifecycleManager(Protocol):
    async def update_trade_event(self, trade_id: str, event: TradeEvent, event_data: dict[str, Any]) -> None  # Line 62
```

#### Class: `IExchange`

**Inherits**: Protocol
**Purpose**: Interface contract for exchange implementations

```python
class IExchange(Protocol):
    async def connect(self) -> bool  # Line 78
    async def disconnect(self) -> None  # Line 82
    async def health_check(self) -> bool  # Line 86
    def is_connected(self) -> bool  # Line 90
    async def place_order(self, order: OrderRequest) -> OrderResponse  # Line 95
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 99
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse  # Line 103
    async def get_market_data(self, symbol: str, timeframe: str = '1m') -> MarketData  # Line 108
    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook  # Line 112
    async def get_ticker(self, symbol: str) -> Ticker  # Line 116
    async def get_trade_history(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 120
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 125
    async def get_positions(self) -> list[Position]  # Line 129
    async def get_exchange_info(self) -> ExchangeInfo  # Line 134
    async def subscribe_to_stream(self, symbol: str, callback: Any) -> None  # Line 139
    def exchange_name(self) -> str  # Line 145
```

#### Class: `IConnectionManager`

**Inherits**: Protocol
**Purpose**: Interface for connection management implementations

```python
class IConnectionManager(Protocol):
    async def connect(self) -> bool  # Line 153
    async def disconnect(self) -> None  # Line 157
    def is_connected(self) -> bool  # Line 161
    async def request(self, ...) -> dict[str, Any]  # Line 165
```

#### Class: `IRateLimiter`

**Inherits**: Protocol
**Purpose**: Interface for rate limiting implementations

```python
class IRateLimiter(Protocol):
    async def acquire(self, weight: int = 1) -> bool  # Line 180
    async def release(self, weight: int = 1) -> None  # Line 184
    def reset(self) -> None  # Line 188
    def get_statistics(self) -> dict[str, Any]  # Line 192
```

#### Class: `IHealthMonitor`

**Inherits**: Protocol
**Purpose**: Interface for health monitoring implementations

```python
class IHealthMonitor(Protocol):
    def record_success(self) -> None  # Line 200
    def record_failure(self) -> None  # Line 204
    def record_latency(self, latency_ms: float) -> None  # Line 208
    def get_health_status(self) -> dict[str, Any]  # Line 212
    async def check_health(self) -> bool  # Line 216
```

#### Class: `IExchangeAdapter`

**Inherits**: Protocol
**Purpose**: Interface for exchange adapter implementations

```python
class IExchangeAdapter(Protocol):
    async def place_order(self, **kwargs) -> dict[str, Any]  # Line 224
    async def cancel_order(self, order_id: str, **kwargs) -> dict[str, Any]  # Line 228
    async def get_order(self, order_id: str, **kwargs) -> dict[str, Any]  # Line 232
    async def get_balance(self, asset: str | None = None) -> dict[str, Any]  # Line 236
    async def get_ticker(self, symbol: str) -> dict[str, Any]  # Line 240
```

#### Class: `IExchangeFactory`

**Inherits**: Protocol
**Purpose**: Interface for exchange factory implementations

```python
class IExchangeFactory(Protocol):
    def get_supported_exchanges(self) -> list[str]  # Line 253
    def get_available_exchanges(self) -> list[str]  # Line 257
    def is_exchange_supported(self, exchange_name: str) -> bool  # Line 261
    async def get_exchange(self, ...) -> IExchange | None  # Line 265
    async def create_exchange(self, exchange_name: str) -> IExchange  # Line 271
    async def remove_exchange(self, exchange_name: str) -> bool  # Line 275
    async def health_check_all(self) -> dict[str, Any]  # Line 279
    async def disconnect_all(self) -> None  # Line 283
```

#### Class: `SandboxMode`

**Inherits**: str, Enum
**Purpose**: Sandbox operation modes

```python
class SandboxMode(str, Enum):
```

#### Class: `ISandboxConnectionManager`

**Inherits**: Protocol
**Purpose**: Interface for sandbox connection management

```python
class ISandboxConnectionManager(Protocol):
    async def connect_to_sandbox(self) -> bool  # Line 310
    async def connect_to_production(self) -> bool  # Line 314
    async def switch_environment(self, mode: SandboxMode) -> bool  # Line 318
    def get_current_endpoints(self) -> dict[str, str]  # Line 322
    async def validate_environment(self) -> dict[str, Any]  # Line 326
```

#### Class: `ISandboxAdapter`

**Inherits**: Protocol
**Purpose**: Interface for sandbox-specific exchange adapters

```python
class ISandboxAdapter(Protocol):
    def sandbox_mode(self) -> SandboxMode  # Line 340
    async def validate_sandbox_credentials(self) -> bool  # Line 344
    async def reset_sandbox_account(self) -> bool  # Line 348
    async def get_sandbox_balance(self, asset: str | None = None) -> dict[str, Any]  # Line 352
    async def place_sandbox_order(self, order: OrderRequest) -> OrderResponse  # Line 356
    async def simulate_order_fill(self, order_id: str, fill_percentage: float = 1.0) -> bool  # Line 360
    async def inject_market_data(self, symbol: str, data: dict[str, Any]) -> bool  # Line 364
    async def simulate_network_error(self, error_type: str, duration_seconds: int = 5) -> None  # Line 368
```

### File: mock_exchange.py

**Key Imports:**
- `from src.core.exceptions import ExchangeConnectionError`
- `from src.core.exceptions import OrderRejectionError`
- `from src.core.types import ExchangeInfo`
- `from src.core.types import OrderBook`
- `from src.core.types import OrderBookLevel`

#### Class: `BaseMockExchange`

**Inherits**: BaseExchange
**Purpose**: Base mock exchange implementation

```python
class BaseMockExchange(BaseExchange):
    def __init__(self, name: str = 'mock', config: dict[str, Any] | None = None)  # Line 31
```

#### Class: `MockExchange`

**Inherits**: BaseMockExchange
**Purpose**: Mock exchange implementation for development and testing

```python
class MockExchange(BaseMockExchange):
    def __init__(self, config: dict[str, Any] | None = None)  # Line 50
    def orders(self) -> dict[str, OrderResponse]  # Line 82
    async def connect(self) -> None  # Line 86
    async def _do_start(self) -> None  # Line 102
    async def disconnect(self) -> None  # Line 134
    async def load_exchange_info(self) -> ExchangeInfo  # Line 141
    async def get_ticker(self, symbol: str) -> dict[str, Any]  # Line 166
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook  # Line 180
    async def _place_order_impl(self, order_request: OrderRequest) -> OrderResponse  # Line 209
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 298
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 304
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse  # Line 320
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 327
    async def get_positions(self) -> list[Position]  # Line 335
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 340
    async def ping(self) -> bool  # Line 366
    def configure(self, ...) -> None  # Line 375
    def set_balance(self, balances: dict[str, Decimal]) -> None  # Line 382
    def set_price(self, symbol: str, price: Decimal) -> None  # Line 386
    def set_order_book(self, symbol: str, order_book: dict) -> None  # Line 390
    async def get_balance(self) -> dict[str, Decimal]  # Line 395
    async def get_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 399
    async def place_order_dict(self, ...) -> dict[str, Any]  # Line 405
    async def get_ticker_dict(self, symbol: str) -> dict[str, Any]  # Line 446
    async def place_order(self, *args, **kwargs)  # Line 461
    async def get_ticker_fallback(self, symbol: str)  # Line 470
    async def _get_ticker_impl(self, symbol: str) -> Ticker  # Line 476
    def _get_current_time(self)  # Line 509
    def _update_balances_for_filled_order(self, order: OrderResponse) -> None  # Line 515
    def _ensure_connected(self) -> None  # Line 546
```

### File: okx.py

**Key Imports:**
- `from src.core.exceptions import ExchangeConnectionError`
- `from src.core.exceptions import ExchangeRateLimitError`
- `from src.core.exceptions import OrderRejectionError`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`

#### Class: `OKXExchange`

**Inherits**: BaseExchange
**Purpose**: OKX exchange implementation following BaseService pattern

```python
class OKXExchange(BaseExchange):
    def __init__(self, config: dict[str, Any])  # Line 66
    async def connect(self) -> None  # Line 101
    async def disconnect(self) -> None  # Line 155
    async def ping(self) -> bool  # Line 181
    async def load_exchange_info(self) -> ExchangeInfo  # Line 195
    async def get_ticker(self, symbol: str) -> Ticker  # Line 229
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook  # Line 300
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 347
    async def place_order(self, order_request: OrderRequest) -> OrderResponse  # Line 409
    async def cancel_order(self, order_id: str, symbol: str) -> OrderResponse  # Line 477
    async def get_order_status(self, order_id: str, symbol: str | None = None) -> OrderStatus  # Line 520
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 554
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 613
    async def get_balance(self, asset: str | None = None) -> dict[str, Any]  # Line 648
    async def get_positions(self) -> list[Position]  # Line 698
    async def _test_okx_connection(self) -> None  # Line 715
    async def _validate_okx_order(self, order: OrderRequest) -> None  # Line 742
    def _convert_order_to_okx(self, order: OrderRequest) -> dict[str, Any]  # Line 776
    def _convert_okx_order_to_response(self, result: dict[str, Any]) -> OrderResponse  # Line 796
    def _convert_okx_status_to_order_status(self, status: str) -> OrderStatus  # Line 823
    def _convert_order_type_to_okx(self, order_type: OrderType) -> str  # Line 834
    def _convert_okx_order_type_to_unified(self, okx_type: str) -> OrderType  # Line 844
    def _convert_symbol_to_okx_format(self, symbol: str) -> str  # Line 853
    def _convert_symbol_from_okx_format(self, okx_symbol: str) -> str  # Line 895
```

### File: okx_orders.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import ExchangeError`
- `from src.core.exceptions import ExchangeInsufficientFundsError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `OKXOrderManager`

**Purpose**: OKX order manager for specialized order handling

```python
class OKXOrderManager:
    def __init__(self, ...)  # Line 60
    def _get_asset_precision(self, symbol: str, precision_type: str = 'quantity') -> int  # Line 90
    async def place_order(self, order: OrderRequest) -> OrderResponse  # Line 94
    async def cancel_order(self, order_id: str, symbol: str) -> bool  # Line 156
    async def get_order_status(self, order_id: str) -> OrderStatus  # Line 187
    async def get_order_fills(self, order_id: str) -> list[OrderResponse]  # Line 217
    async def place_stop_loss_order(self, order: OrderRequest, stop_price: Decimal) -> OrderResponse  # Line 266
    async def place_take_profit_order(self, order: OrderRequest, take_profit_price: Decimal) -> OrderResponse  # Line 297
    async def place_oco_order(self, order: OrderRequest, stop_price: Decimal, take_profit_price: Decimal) -> list[OrderResponse]  # Line 328
    def calculate_fee(self, order: OrderRequest, is_maker: bool = False) -> Decimal  # Line 361
    def _validate_order_request(self, order: OrderRequest) -> None  # Line 385
    def _convert_order_to_okx(self, order: OrderRequest) -> dict[str, Any]  # Line 422
    def _convert_okx_order_to_response(self, result: dict) -> OrderResponse  # Line 461
    def _convert_okx_status_to_order_status(self, status: str) -> OrderStatus  # Line 496
    def _convert_order_type_to_okx(self, order_type: OrderType) -> str  # Line 500
    def _convert_okx_order_type_to_unified(self, okx_type: str) -> OrderType  # Line 519
    def get_active_orders(self) -> dict[str, dict]  # Line 540
    def get_order_history(self, symbol: str | None = None) -> list[dict]  # Line 549
    def clear_order_history(self) -> None  # Line 566
    def get_fee_rates(self) -> dict[str, Decimal]  # Line 571
    def update_fee_rates(self, maker_rate: Decimal, taker_rate: Decimal) -> None  # Line 580
```

### File: okx_websocket.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import ExchangeConnectionError`
- `from src.core.exceptions import ExchangeError`
- `from src.core.logging import get_logger`
- `from src.core.types.market import OrderBook`

#### Class: `OKXWebSocketManager`

**Purpose**: OKX WebSocket manager for real-time data streaming

```python
class OKXWebSocketManager:
    def __init__(self, ...)  # Line 54
    async def connect(self) -> bool  # Line 130
    async def disconnect(self) -> None  # Line 170
    async def subscribe_to_ticker(self, symbol: str, callback: Callable) -> None  # Line 244
    async def subscribe_to_orderbook(self, symbol: str, callback: Callable) -> None  # Line 275
    async def subscribe_to_trades(self, symbol: str, callback: Callable) -> None  # Line 304
    async def subscribe_to_account(self, callback: Callable) -> None  # Line 333
    async def _connect_public_websocket(self) -> None  # Line 361
    async def _connect_private_websocket(self) -> None  # Line 391
    async def _authenticate_private_websocket(self) -> None  # Line 424
    async def _listen_public_messages(self) -> None  # Line 476
    async def _listen_private_messages(self) -> None  # Line 506
    async def _handle_public_message(self, message: str) -> None  # Line 536
    async def _handle_private_message(self, message: str) -> None  # Line 561
    async def _handle_event_message(self, data: dict) -> None  # Line 586
    async def _handle_private_event_message(self, data: dict) -> None  # Line 602
    async def _handle_data_message(self, data: dict) -> None  # Line 620
    async def _handle_private_data_message(self, data: dict) -> None  # Line 649
    async def _handle_ticker_data(self, symbol: str, data: list[dict]) -> None  # Line 673
    async def _handle_orderbook_data(self, symbol: str, data: list[dict]) -> None  # Line 723
    async def _handle_trades_data(self, symbol: str, data: list[dict]) -> None  # Line 777
    async def _handle_account_data(self, data: list[dict]) -> None  # Line 820
    async def _send_public_message(self, message: dict) -> None  # Line 844
    async def _send_private_message(self, message: dict) -> None  # Line 861
    async def _handle_disconnect(self) -> None  # Line 878
    async def _schedule_reconnect(self) -> None  # Line 888
    async def _reconnect_after_delay(self, delay: float) -> None  # Line 917
    async def _health_monitor(self) -> None  # Line 981
    def is_connected(self) -> bool  # Line 1012
    async def health_check(self) -> bool  # Line 1021
    def get_connection_metrics(self) -> dict[str, Any]  # Line 1062
    def get_status(self) -> str  # Line 1103
```

### File: service.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.config import Config`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`

#### Class: `ExchangeService`

**Inherits**: BaseService
**Purpose**: Service layer for exchange operations

```python
class ExchangeService(BaseService):
    def __init__(self, ...)  # Line 56
    async def _do_start(self) -> None  # Line 89
    async def _do_stop(self) -> None  # Line 102
    async def get_exchange(self, exchange_name: str) -> IExchange  # Line 122
    async def _is_exchange_healthy(self, exchange: IExchange) -> bool  # Line 175
    async def _remove_exchange(self, exchange_name: str) -> None  # Line 190
    async def place_order(self, exchange_name: str, order: OrderRequest) -> OrderResponse  # Line 220
    async def _validate_order_request(self, order: OrderRequest) -> None  # Line 257
    async def _process_order_response(self, exchange_name: str, order_response: OrderResponse) -> None  # Line 271
    async def cancel_order(self, exchange_name: str, order_id: str, symbol: str | None = None) -> bool  # Line 293
    async def get_order_status(self, exchange_name: str, order_id: str, symbol: str | None = None) -> OrderStatus  # Line 357
    async def get_market_data(self, exchange_name: str, symbol: str, timeframe: str = '1m') -> MarketData  # Line 409
    async def get_order_book(self, exchange_name: str, symbol: str, depth: int = 10) -> OrderBook  # Line 442
    async def get_ticker(self, exchange_name: str, symbol: str) -> Ticker  # Line 466
    async def get_account_balance(self, exchange_name: str) -> dict[str, Decimal]  # Line 489
    async def get_positions(self, exchange_name: str, symbol: str | None = None) -> list[Position]  # Line 507
    async def get_exchange_info(self, exchange_name: str) -> ExchangeInfo  # Line 533
    def get_supported_exchanges(self) -> list[str]  # Line 548
    def get_available_exchanges(self) -> list[str]  # Line 552
    async def get_exchange_status(self, exchange_name: str) -> dict[str, Any]  # Line 556
    async def get_service_health(self) -> dict[str, Any]  # Line 601
    async def disconnect_all_exchanges(self) -> None  # Line 627
    async def get_best_price(self, symbol: str, side: str, exchanges: list[str] | None = None) -> dict[str, Any]  # Line 663
    async def _get_ticker_safe(self, exchange_name: str, symbol: str) -> Ticker | None  # Line 720
    async def subscribe_to_stream(self, exchange_name: str, symbol: str, callback: Any) -> None  # Line 730
    async def _service_health_check(self) -> HealthStatus  # Line 742
```

### File: types.py

**Key Imports:**
- `from src.utils import ValidationFramework`

#### Class: `ExchangeTypes`

**Purpose**: Exchange-specific type definitions and utilities

```python
class ExchangeTypes:
    def validate_symbol(symbol: str) -> bool  # Line 33
```

#### Class: `ExchangeCapability`

**Inherits**: Enum
**Purpose**: Exchange capabilities enumeration

```python
class ExchangeCapability(Enum):
```

#### Class: `ExchangeTradingPair`

**Inherits**: BaseModel
**Purpose**: Trading pair information

```python
class ExchangeTradingPair(BaseModel):
```

#### Class: `ExchangeFee`

**Inherits**: BaseModel
**Purpose**: Exchange fee structure

```python
class ExchangeFee(BaseModel):
```

#### Class: `ExchangeRateLimit`

**Inherits**: BaseModel
**Purpose**: Exchange rate limit configuration

```python
class ExchangeRateLimit(BaseModel):
```

#### Class: `ExchangeConnectionConfig`

**Inherits**: BaseModel
**Purpose**: Exchange connection configuration

```python
class ExchangeConnectionConfig(BaseModel):
```

#### Class: `ExchangeOrderBookLevel`

**Inherits**: BaseModel
**Purpose**: Order book level information

```python
class ExchangeOrderBookLevel(BaseModel):
```

#### Class: `ExchangeOrderBookSnapshot`

**Inherits**: BaseModel
**Purpose**: Order book snapshot

```python
class ExchangeOrderBookSnapshot(BaseModel):
```

#### Class: `ExchangeTrade`

**Inherits**: BaseModel
**Purpose**: Exchange trade information

```python
class ExchangeTrade(BaseModel):
```

#### Class: `ExchangeBalance`

**Inherits**: BaseModel
**Purpose**: Exchange balance information

```python
class ExchangeBalance(BaseModel):
```

#### Class: `ExchangePosition`

**Inherits**: BaseModel
**Purpose**: Exchange position information

```python
class ExchangePosition(BaseModel):
```

#### Class: `ExchangeOrder`

**Inherits**: BaseModel
**Purpose**: Exchange order information

```python
class ExchangeOrder(BaseModel):
```

#### Class: `ExchangeWebSocketMessage`

**Inherits**: BaseModel
**Purpose**: WebSocket message structure

```python
class ExchangeWebSocketMessage(BaseModel):
```

#### Class: `ExchangeErrorResponse`

**Inherits**: BaseModel
**Purpose**: Exchange error response structure

```python
class ExchangeErrorResponse(BaseModel):
```

#### Class: `ExchangeHealthStatus`

**Inherits**: BaseModel
**Purpose**: Exchange health status

```python
class ExchangeHealthStatus(BaseModel):
```

---
**Generated**: Complete reference for exchanges module
**Total Classes**: 49
**Total Functions**: 2