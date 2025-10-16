# EXCHANGES Module Reference

## INTEGRATION
**Dependencies**: core, data, error_handling, monitoring, state, utils
**Used By**: error_handling
**Provides**: BinanceOrderManager, CoinbaseOrderManager, ConnectionManager, ExchangeService, IConnectionManager, IExchangeService, ISandboxConnectionManager, IStateService, ITradeLifecycleManager, OKXOrderManager, OKXWebSocketManager
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
- Parallel execution
**Architecture**:
- BaseExchange inherits from base architecture
- ConnectionManager inherits from base architecture
- ExchangeFactory inherits from base architecture

## MODULE OVERVIEW
**Files**: 18 Python files
**Classes**: 56
**Functions**: 7

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
- `last_heartbeat(self) -> datetime | None` - Line 158
- `is_connected(self) -> bool` - Line 162
- `async health_check(self) -> HealthCheckResult` - Line 292
- `async connect(self) -> None` - Line 328
- `async disconnect(self) -> None` - Line 333
- `async ping(self) -> bool` - Line 338
- `async load_exchange_info(self) -> ExchangeInfo` - Line 343
- `async get_ticker(self, symbol: str) -> Ticker` - Line 352
- `async get_order_book(self, symbol: str, limit: int = 100) -> OrderBook` - Line 359
- `async get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 366
- `async place_order(self, order_request: OrderRequest) -> OrderResponse` - Line 375
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 395
- `async get_order_status(self, symbol: str, order_id: str) -> OrderResponse` - Line 402
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 409
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 418
- `async get_positions(self) -> list[Position]` - Line 430
- `get_exchange_info(self) -> ExchangeInfo | None` - Line 693
- `get_trading_symbols(self) -> list[str] | None` - Line 697
- `is_symbol_supported(self, symbol: str) -> bool` - Line 701

### Implementation: `MockExchangeError` ✅

**Inherits**: Exception
**Purpose**: Exception for mock exchange testing
**Status**: Complete

### Implementation: `BaseMockExchange` ✅

**Inherits**: BaseExchange
**Purpose**: Mock exchange implementation for testing
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> None` - Line 771
- `async disconnect(self) -> None` - Line 776
- `async ping(self) -> bool` - Line 780
- `async load_exchange_info(self) -> ExchangeInfo` - Line 787
- `async get_ticker(self, symbol: str) -> Ticker` - Line 805
- `async get_order_book(self, symbol: str, limit: int = 100) -> OrderBook` - Line 826
- `async get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 841
- `async place_order(self, order_request: OrderRequest) -> OrderResponse` - Line 861
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 887
- `async get_order_status(self, symbol: str, order_id: str) -> OrderResponse` - Line 898
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 906
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 915
- `async get_positions(self) -> list[Position]` - Line 921

### Implementation: `BinanceExchange` ✅

**Inherits**: BaseExchange
**Purpose**: Binance exchange implementation following service layer pattern
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> None` - Line 138
- `async disconnect(self) -> None` - Line 172
- `async ping(self) -> bool` - Line 214
- `async load_exchange_info(self) -> ExchangeInfo` - Line 224
- `async get_ticker(self, symbol: str) -> Ticker` - Line 279
- `async get_order_book(self, symbol: str, limit: int = 100) -> OrderBook` - Line 343
- `async get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 386
- `async place_order(self, order_request: OrderRequest) -> OrderResponse` - Line 423
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 768
- `async get_order_status(self, symbol: str, order_id: str) -> OrderResponse` - Line 801
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 844
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 897
- `async get_positions(self) -> list[Position]` - Line 926

### Implementation: `BinanceOrderManager` ✅

**Purpose**: Binance order management for specialized order handling
**Status**: Complete

**Implemented Methods:**
- `async place_market_order(self, order: OrderRequest) -> OrderResponse` - Line 115
- `async place_limit_order(self, order: OrderRequest) -> OrderResponse` - Line 168
- `async place_stop_loss_order(self, order: OrderRequest) -> OrderResponse` - Line 223
- `async place_oco_order(self, order: OrderRequest) -> OrderResponse` - Line 277
- `async cancel_order(self, symbol: str, order_id: str) -> bool` - Line 335
- `async get_order_status(self, symbol: str, order_id: str) -> OrderStatus` - Line 367
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
- `async connect(self) -> None` - Line 129
- `async disconnect(self) -> None` - Line 171
- `async ping(self) -> bool` - Line 207
- `async load_exchange_info(self) -> ExchangeInfo` - Line 221
- `async get_ticker(self, symbol: str) -> Ticker` - Line 251
- `async get_order_book(self, symbol: str, limit: int = 100) -> OrderBook` - Line 277
- `async get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 303
- `async place_order(self, order_request: OrderRequest) -> OrderResponse` - Line 334
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 361
- `async get_order_status(self, symbol: str, order_id: str) -> OrderResponse` - Line 401
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 454
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 492
- `async get_balance(self, asset: str | None = None) -> dict[str, Any]` - Line 514
- `async get_positions(self) -> list[Position]` - Line 569

### Implementation: `CoinbaseOrderManager` ✅

**Purpose**: Coinbase order manager for handling order operations
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> bool` - Line 124
- `async place_order(self, order: OrderRequest) -> OrderResponse` - Line 150
- `async cancel_order(self, symbol: str, order_id: str) -> bool` - Line 199
- `async get_order_status(self, symbol: str, order_id: str) -> OrderStatus` - Line 231
- `async get_order_details(self, order_id: str) -> OrderResponse | None` - Line 261
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 287
- `async get_order_history(self, symbol: str | None = None, limit: int = 100) -> list[OrderResponse]` - Line 316
- `async get_fills(self, order_id: str | None = None, symbol: str | None = None) -> list[dict]` - Line 348
- `async calculate_fees(self, order: OrderRequest) -> dict[str, Decimal]` - Line 372
- `get_total_fees(self) -> dict[str, Decimal]` - Line 413
- `get_order_statistics(self) -> dict[str, Any]` - Line 422

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

### Implementation: `ExchangeDataTransformer` ✅

**Inherits**: Protocol
**Purpose**: Protocol for exchange data transformers
**Status**: Complete

**Implemented Methods:**
- `transform_symbol_to_exchange(self, symbol: str) -> str` - Line 39
- `transform_symbol_from_exchange(self, exchange_symbol: str) -> str` - Line 43
- `transform_order_to_exchange(self, order: OrderRequest) -> dict[str, Any]` - Line 47
- `transform_order_from_exchange(self, exchange_data: dict[str, Any]) -> OrderResponse` - Line 51
- `transform_ticker_from_exchange(self, exchange_data: dict[str, Any]) -> Ticker` - Line 55

### Implementation: `BaseExchangeTransformer` 🔧

**Inherits**: ABC
**Purpose**: Base transformer with common patterns
**Status**: Abstract Base Class

**Implemented Methods:**
- `validate_decimal_precision(self, value: Decimal, symbol: str) -> Decimal` - Line 67
- `validate_symbol_format(self, symbol: str) -> str` - Line 75
- `apply_messaging_pattern(self, data: dict[str, Any], operation_type: str) -> dict[str, Any]` - Line 132
- `get_symbol_mappings(self) -> dict[str, str]` - Line 138
- `get_order_type_mappings(self) -> dict[OrderType, str]` - Line 143
- `get_order_status_mappings(self) -> dict[str, OrderStatus]` - Line 148
- `transform_for_batch_processing(cls, ...) -> dict[str, Any]` - Line 153

### Implementation: `BinanceTransformer` ✅

**Inherits**: BaseExchangeTransformer
**Purpose**: Binance data transformer
**Status**: Complete

**Implemented Methods:**
- `get_symbol_mappings(self) -> dict[str, str]` - Line 211
- `get_order_type_mappings(self) -> dict[OrderType, str]` - Line 221
- `get_order_status_mappings(self) -> dict[str, OrderStatus]` - Line 230
- `transform_symbol_to_exchange(self, symbol: str) -> str` - Line 241
- `transform_symbol_from_exchange(self, exchange_symbol: str) -> str` - Line 256

### Implementation: `CoinbaseTransformer` ✅

**Inherits**: BaseExchangeTransformer
**Purpose**: Coinbase data transformer
**Status**: Complete

**Implemented Methods:**
- `get_symbol_mappings(self) -> dict[str, str]` - Line 289
- `get_order_type_mappings(self) -> dict[OrderType, str]` - Line 300
- `get_order_status_mappings(self) -> dict[str, OrderStatus]` - Line 309
- `transform_symbol_to_exchange(self, symbol: str) -> str` - Line 319
- `transform_symbol_from_exchange(self, exchange_symbol: str) -> str` - Line 338

### Implementation: `OKXTransformer` ✅

**Inherits**: BaseExchangeTransformer
**Purpose**: OKX data transformer
**Status**: Complete

**Implemented Methods:**
- `get_symbol_mappings(self) -> dict[str, str]` - Line 349
- `get_order_type_mappings(self) -> dict[OrderType, str]` - Line 360
- `get_order_status_mappings(self) -> dict[str, OrderStatus]` - Line 369
- `transform_symbol_to_exchange(self, symbol: str) -> str` - Line 379
- `transform_symbol_from_exchange(self, exchange_symbol: str) -> str` - Line 398

### Implementation: `TransformerFactory` ✅

**Purpose**: Factory for creating exchange transformers
**Status**: Complete

**Implemented Methods:**
- `create_transformer(cls, exchange_name: str) -> BaseExchangeTransformer` - Line 414
- `get_supported_exchanges(cls) -> list[str]` - Line 423

### Implementation: `ExchangeFactory` ✅

**Inherits**: BaseService, IExchangeFactory
**Purpose**: Simple factory for creating exchange instances
**Status**: Complete

**Implemented Methods:**
- `register_exchange(self, exchange_name: str, exchange_class: type[BaseExchange]) -> None` - Line 54
- `get_supported_exchanges(self) -> list[str]` - Line 72
- `is_exchange_supported(self, exchange_name: str) -> bool` - Line 81
- `async create_exchange(self, exchange_name: str) -> BaseExchange` - Line 94
- `get_available_exchanges(self) -> list[str]` - Line 128
- `async get_exchange(self, ...) -> IExchange | None` - Line 138
- `async remove_exchange(self, exchange_name: str) -> bool` - Line 172
- `async health_check_all(self) -> dict[str, Any]` - Line 187
- `async disconnect_all(self) -> None` - Line 202
- `register_default_exchanges(self) -> None` - Line 211

### Implementation: `TradeEvent` ✅

**Inherits**: str, Enum
**Purpose**: Trade event enumeration (mirror of state module)
**Status**: Complete

### Implementation: `IStateService` ✅

**Inherits**: Protocol
**Purpose**: Interface for StateService used by exchanges
**Status**: Complete

**Implemented Methods:**
- `async set_state(self, ...) -> bool` - Line 47
- `async get_state(self, state_type: StateType, state_id: str) -> dict[str, Any] | None` - Line 59

### Implementation: `ITradeLifecycleManager` ✅

**Inherits**: Protocol
**Purpose**: Interface for TradeLifecycleManager used by exchanges
**Status**: Complete

**Implemented Methods:**
- `async update_trade_event(self, trade_id: str, event: TradeEvent, event_data: dict[str, Any]) -> None` - Line 67

### Implementation: `IExchange` ✅

**Inherits**: Protocol
**Purpose**: Interface contract for exchange implementations
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> bool` - Line 83
- `async disconnect(self) -> None` - Line 87
- `async health_check(self) -> bool` - Line 91
- `is_connected(self) -> bool` - Line 95
- `async place_order(self, order: OrderRequest) -> OrderResponse` - Line 100
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 104
- `async get_order_status(self, symbol: str, order_id: str) -> OrderResponse` - Line 108
- `async get_market_data(self, symbol: str, timeframe: str = '1m') -> MarketData` - Line 113
- `async get_order_book(self, symbol: str, depth: int = 10) -> OrderBook` - Line 117
- `async get_ticker(self, symbol: str) -> Ticker` - Line 121
- `async get_trade_history(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 125
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 130
- `async get_positions(self) -> list[Position]` - Line 134
- `async get_exchange_info(self) -> ExchangeInfo` - Line 139
- `async subscribe_to_stream(self, symbol: str, callback: Any) -> None` - Line 144
- `exchange_name(self) -> str` - Line 150

### Implementation: `IConnectionManager` ✅

**Inherits**: Protocol
**Purpose**: Interface for connection management implementations
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> bool` - Line 158
- `async disconnect(self) -> None` - Line 162
- `is_connected(self) -> bool` - Line 166
- `async request(self, ...) -> dict[str, Any]` - Line 170

### Implementation: `IRateLimiter` ✅

**Inherits**: Protocol
**Purpose**: Interface for rate limiting implementations
**Status**: Complete

**Implemented Methods:**
- `async acquire(self, weight: int = 1) -> bool` - Line 185
- `async release(self, weight: int = 1) -> None` - Line 189
- `reset(self) -> None` - Line 193
- `get_statistics(self) -> dict[str, Any]` - Line 197

### Implementation: `IHealthMonitor` ✅

**Inherits**: Protocol
**Purpose**: Interface for health monitoring implementations
**Status**: Complete

**Implemented Methods:**
- `record_success(self) -> None` - Line 205
- `record_failure(self) -> None` - Line 209
- `record_latency(self, latency_ms: float) -> None` - Line 213
- `get_health_status(self) -> dict[str, Any]` - Line 217
- `async check_health(self) -> bool` - Line 221

### Implementation: `IExchangeAdapter` ✅

**Inherits**: Protocol
**Purpose**: Interface for exchange adapter implementations
**Status**: Complete

**Implemented Methods:**
- `async place_order(self, **kwargs) -> dict[str, Any]` - Line 229
- `async cancel_order(self, order_id: str, **kwargs) -> dict[str, Any]` - Line 233
- `async get_order(self, order_id: str, **kwargs) -> dict[str, Any]` - Line 237
- `async get_balance(self, asset: str | None = None) -> dict[str, Any]` - Line 241
- `async get_ticker(self, symbol: str) -> dict[str, Any]` - Line 245

### Implementation: `IExchangeFactory` ✅

**Inherits**: Protocol
**Purpose**: Interface for exchange factory implementations
**Status**: Complete

**Implemented Methods:**
- `get_supported_exchanges(self) -> list[str]` - Line 258
- `get_available_exchanges(self) -> list[str]` - Line 262
- `is_exchange_supported(self, exchange_name: str) -> bool` - Line 266
- `async get_exchange(self, ...) -> IExchange | None` - Line 270
- `async create_exchange(self, exchange_name: str) -> IExchange` - Line 276
- `async remove_exchange(self, exchange_name: str) -> bool` - Line 280
- `async health_check_all(self) -> dict[str, Any]` - Line 284
- `async disconnect_all(self) -> None` - Line 288

### Implementation: `SandboxMode` ✅

**Inherits**: str, Enum
**Purpose**: Sandbox operation modes
**Status**: Complete

### Implementation: `ISandboxConnectionManager` ✅

**Inherits**: Protocol
**Purpose**: Interface for sandbox connection management
**Status**: Complete

**Implemented Methods:**
- `async connect_to_sandbox(self) -> bool` - Line 315
- `async connect_to_production(self) -> bool` - Line 319
- `async switch_environment(self, mode: SandboxMode) -> bool` - Line 323
- `get_current_endpoints(self) -> dict[str, str]` - Line 327
- `async validate_environment(self) -> dict[str, Any]` - Line 331

### Implementation: `ISandboxAdapter` ✅

**Inherits**: Protocol
**Purpose**: Interface for sandbox-specific exchange adapters
**Status**: Complete

**Implemented Methods:**
- `sandbox_mode(self) -> SandboxMode` - Line 345
- `async validate_sandbox_credentials(self) -> bool` - Line 349
- `async reset_sandbox_account(self) -> bool` - Line 353
- `async get_sandbox_balance(self, asset: str | None = None) -> dict[str, Any]` - Line 357
- `async place_sandbox_order(self, order: OrderRequest) -> OrderResponse` - Line 361
- `async simulate_order_fill(self, order_id: str, fill_percentage: float = 1.0) -> bool` - Line 365
- `async inject_market_data(self, symbol: str, data: dict[str, Any]) -> bool` - Line 369
- `async simulate_network_error(self, error_type: str, duration_seconds: int = 5) -> None` - Line 373

### Implementation: `IExchangeService` ✅

**Inherits**: Protocol
**Purpose**: Interface for exchange service layer implementations
**Status**: Complete

**Implemented Methods:**
- `async get_exchange(self, exchange_name: str) -> 'IExchange'` - Line 381
- `async place_order(self, exchange_name: str, order: OrderRequest) -> OrderResponse` - Line 385
- `async cancel_order(self, exchange_name: str, order_id: str, symbol: str | None = None) -> bool` - Line 389
- `async get_order_status(self, exchange_name: str, order_id: str, symbol: str | None = None) -> 'OrderStatus'` - Line 395
- `async get_market_data(self, exchange_name: str, symbol: str, timeframe: str = '1m') -> MarketData` - Line 401
- `async get_order_book(self, exchange_name: str, symbol: str, depth: int = 10) -> OrderBook` - Line 407
- `async get_ticker(self, exchange_name: str, symbol: str) -> Ticker` - Line 413
- `async get_account_balance(self, exchange_name: str) -> dict[str, Decimal]` - Line 417
- `async get_positions(self, exchange_name: str, symbol: str | None = None) -> list[Position]` - Line 421
- `async get_exchange_info(self, exchange_name: str) -> ExchangeInfo` - Line 427
- `get_supported_exchanges(self) -> list[str]` - Line 431
- `get_available_exchanges(self) -> list[str]` - Line 435
- `async get_exchange_status(self, exchange_name: str) -> dict[str, Any]` - Line 439
- `async get_service_health(self) -> dict[str, Any]` - Line 443
- `async disconnect_all_exchanges(self) -> None` - Line 447
- `async get_best_price(self, symbol: str, side: str, exchanges: list[str] | None = None) -> dict[str, Any]` - Line 451
- `async subscribe_to_stream(self, exchange_name: str, symbol: str, callback: Any) -> None` - Line 457
- `get_supported_exchanges(self) -> list[str]` - Line 463
- `get_available_exchanges(self) -> list[str]` - Line 467
- `async get_exchange_status(self, exchange_name: str) -> dict[str, Any]` - Line 471
- `async get_service_health(self) -> dict[str, Any]` - Line 475
- `async disconnect_all_exchanges(self) -> None` - Line 479
- `async get_best_price(self, symbol: str, side: str, exchanges: list[str] | None = None) -> dict[str, Any]` - Line 483

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
- `orders(self) -> dict[str, OrderResponse]` - Line 86
- `async connect(self) -> None` - Line 91
- `async disconnect(self) -> None` - Line 139
- `async load_exchange_info(self) -> ExchangeInfo` - Line 147
- `async get_ticker(self, symbol: str) -> Ticker` - Line 173
- `async get_order_book(self, symbol: str, limit: int = 100) -> OrderBook` - Line 178
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 296
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 302
- `async get_order_status(self, symbol: str, order_id: str) -> OrderResponse` - Line 318
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 325
- `async get_positions(self) -> list[Position]` - Line 333
- `async get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 338
- `async ping(self) -> bool` - Line 364
- `configure(self, ...) -> None` - Line 373
- `set_balance(self, balances: dict[str, Decimal]) -> None` - Line 380
- `set_price(self, symbol: str, price: Decimal) -> None` - Line 384
- `set_order_book(self, symbol: str, order_book: dict) -> None` - Line 388
- `async get_balance(self) -> dict[str, Decimal]` - Line 393
- `async get_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 397
- `async place_order_dict(self, ...) -> dict[str, Any]` - Line 403
- `async get_ticker_dict(self, symbol: str) -> dict[str, Any]` - Line 444
- `async place_order(self, *args, **kwargs)` - Line 459
- `async get_ticker_fallback(self, symbol: str)` - Line 468

### Implementation: `OKXExchange` ✅

**Inherits**: BaseExchange
**Purpose**: OKX exchange implementation following BaseService pattern
**Status**: Complete

**Implemented Methods:**
- `async connect(self) -> None` - Line 104
- `async disconnect(self) -> None` - Line 158
- `async ping(self) -> bool` - Line 184
- `async load_exchange_info(self) -> ExchangeInfo` - Line 198
- `async get_ticker(self, symbol: str) -> Ticker` - Line 232
- `async get_order_book(self, symbol: str, limit: int = 100) -> OrderBook` - Line 303
- `async get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]` - Line 350
- `async place_order(self, order_request: OrderRequest) -> OrderResponse` - Line 412
- `async cancel_order(self, symbol: str, order_id: str) -> OrderResponse` - Line 478
- `async get_order_status(self, symbol: str, order_id: str) -> OrderResponse` - Line 521
- `async get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]` - Line 554
- `async get_account_balance(self) -> dict[str, Decimal]` - Line 613
- `async get_balance(self, asset: str | None = None) -> dict[str, Any]` - Line 648
- `async get_positions(self) -> list[Position]` - Line 698

### Implementation: `OKXOrderManager` ✅

**Purpose**: OKX order manager for specialized order handling
**Status**: Complete

**Implemented Methods:**
- `async place_order(self, order: OrderRequest) -> OrderResponse` - Line 96
- `async cancel_order(self, symbol: str, order_id: str) -> bool` - Line 158
- `async get_order_status(self, order_id: str) -> OrderStatus` - Line 190
- `async get_order_fills(self, order_id: str) -> list[OrderResponse]` - Line 220
- `async place_stop_loss_order(self, order: OrderRequest, stop_price: Decimal) -> OrderResponse` - Line 269
- `async place_take_profit_order(self, order: OrderRequest, take_profit_price: Decimal) -> OrderResponse` - Line 300
- `async place_oco_order(self, order: OrderRequest, stop_price: Decimal, take_profit_price: Decimal) -> list[OrderResponse]` - Line 331
- `calculate_fee(self, order: OrderRequest, is_maker: bool = False) -> Decimal` - Line 364
- `get_active_orders(self) -> dict[str, dict]` - Line 544
- `get_order_history(self, symbol: str | None = None) -> list[dict]` - Line 553
- `clear_order_history(self) -> None` - Line 570
- `get_fee_rates(self) -> dict[str, Decimal]` - Line 575
- `update_fee_rates(self, maker_rate: Decimal, taker_rate: Decimal) -> None` - Line 584

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
- `is_connected(self) -> bool` - Line 1022
- `async health_check(self) -> bool` - Line 1031
- `get_connection_metrics(self) -> dict[str, Any]` - Line 1072
- `get_status(self) -> str` - Line 1113

### Implementation: `ExchangeService` ✅

**Inherits**: BaseService, IExchangeService
**Purpose**: Service layer for exchange operations
**Status**: Complete

**Implemented Methods:**
- `async get_exchange(self, exchange_name: str) -> IExchange` - Line 122
- `async place_order(self, exchange_name: str, order: OrderRequest) -> OrderResponse` - Line 220
- `async cancel_order(self, exchange_name: str, order_id: str, symbol: str | None = None) -> bool` - Line 297
- `async get_order_status(self, exchange_name: str, order_id: str, symbol: str | None = None) -> OrderStatus` - Line 361
- `async get_market_data(self, exchange_name: str, symbol: str, timeframe: str = '1m') -> MarketData` - Line 413
- `async get_order_book(self, exchange_name: str, symbol: str, depth: int = 10) -> OrderBook` - Line 446
- `async get_ticker(self, exchange_name: str, symbol: str) -> Ticker` - Line 470
- `async get_account_balance(self, exchange_name: str) -> dict[str, Decimal]` - Line 493
- `async get_positions(self, exchange_name: str, symbol: str | None = None) -> list[Position]` - Line 511
- `async get_exchange_info(self, exchange_name: str) -> ExchangeInfo` - Line 537
- `get_supported_exchanges(self) -> list[str]` - Line 552
- `get_available_exchanges(self) -> list[str]` - Line 556
- `async get_exchange_status(self, exchange_name: str) -> dict[str, Any]` - Line 560
- `async get_service_health(self) -> dict[str, Any]` - Line 605
- `async disconnect_all_exchanges(self) -> None` - Line 631
- `async get_best_price(self, symbol: str, side: str, exchanges: list[str] | None = None) -> dict[str, Any]` - Line 667
- `async subscribe_to_stream(self, exchange_name: str, symbol: str, callback: Any) -> None` - Line 734

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
    def last_heartbeat(self) -> datetime | None  # Line 158
    def is_connected(self) -> bool  # Line 162
    async def _update_connection_state(self, connected: bool) -> None  # Line 166
    async def _do_start(self) -> None  # Line 185
    async def _do_stop(self) -> None  # Line 282
    async def health_check(self) -> HealthCheckResult  # Line 292
    async def connect(self) -> None  # Line 328
    async def disconnect(self) -> None  # Line 333
    async def ping(self) -> bool  # Line 338
    async def load_exchange_info(self) -> ExchangeInfo  # Line 343
    async def get_ticker(self, symbol: str) -> Ticker  # Line 352
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook  # Line 359
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 366
    async def place_order(self, order_request: OrderRequest) -> OrderResponse  # Line 375
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 395
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse  # Line 402
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 409
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 418
    async def get_positions(self) -> list[Position]  # Line 430
    async def _validate_order(self, order_request: OrderRequest) -> None  # Line 436
    async def _validate_market_data(self, data: Any) -> bool  # Line 500
    async def _process_market_data(self, data: Any) -> Any  # Line 504
    async def _persist_order(self, order_response: OrderResponse) -> None  # Line 525
    async def _persist_market_data(self, ticker: Ticker) -> None  # Line 550
    async def _persist_position(self, position: Position) -> None  # Line 565
    def _validate_data_at_boundary(self, data: dict[str, Any], source_operation: str) -> dict[str, Any]  # Line 588
    def _propagate_error_consistently(self, error: Exception, context: str, operation_type: str = 'exchange') -> None  # Line 609
    def _apply_messaging_pattern(self, data: dict[str, Any], operation_type: str) -> dict[str, Any]  # Line 628
    async def _track_analytics(self, event_type: str, data: dict) -> None  # Line 655
    def get_exchange_info(self) -> ExchangeInfo | None  # Line 693
    def get_trading_symbols(self) -> list[str] | None  # Line 697
    def is_symbol_supported(self, symbol: str) -> bool  # Line 701
    def _validate_symbol(self, symbol: str) -> None  # Line 709
    def _validate_price(self, price: Decimal, max_price: Decimal | None = None) -> None  # Line 720
    def _validate_quantity(self, quantity: Decimal) -> None  # Line 736
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
    def __init__(self, name: str = 'mock', config: dict[str, Any] | None = None)  # Line 762
    async def connect(self) -> None  # Line 771
    async def disconnect(self) -> None  # Line 776
    async def ping(self) -> bool  # Line 780
    async def load_exchange_info(self) -> ExchangeInfo  # Line 787
    async def get_ticker(self, symbol: str) -> Ticker  # Line 805
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook  # Line 826
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 841
    async def place_order(self, order_request: OrderRequest) -> OrderResponse  # Line 861
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 887
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse  # Line 898
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 906
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 915
    async def get_positions(self) -> list[Position]  # Line 921
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
    def __init__(self, config: dict[str, Any])  # Line 82
    def _validate_service_config(self, config: dict[str, Any]) -> bool  # Line 119
    async def connect(self) -> None  # Line 138
    async def disconnect(self) -> None  # Line 172
    async def ping(self) -> bool  # Line 214
    async def load_exchange_info(self) -> ExchangeInfo  # Line 224
    async def get_ticker(self, symbol: str) -> Ticker  # Line 279
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook  # Line 343
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 386
    async def place_order(self, order_request: OrderRequest) -> OrderResponse  # Line 423
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 768
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse  # Line 801
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 844
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 897
    async def get_positions(self) -> list[Position]  # Line 926
    def _map_order_type(self, binance_type: str) -> OrderType  # Line 932
    def _map_to_binance_order_type(self, order_type: OrderType) -> str  # Line 946
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
    async def cancel_order(self, symbol: str, order_id: str) -> bool  # Line 335
    async def get_order_status(self, symbol: str, order_id: str) -> OrderStatus  # Line 367
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
- `from src.core.types import ExchangeInfo`

#### Class: `CoinbaseExchange`

**Inherits**: BaseExchange
**Purpose**: Coinbase exchange implementation following BaseService pattern

```python
class CoinbaseExchange(BaseExchange):
    def __init__(self, config: dict[str, Any])  # Line 98
    async def connect(self) -> None  # Line 129
    async def disconnect(self) -> None  # Line 171
    async def ping(self) -> bool  # Line 207
    async def load_exchange_info(self) -> ExchangeInfo  # Line 221
    async def get_ticker(self, symbol: str) -> Ticker  # Line 251
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook  # Line 277
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 303
    async def place_order(self, order_request: OrderRequest) -> OrderResponse  # Line 334
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 361
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse  # Line 401
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 454
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 492
    async def get_balance(self, asset: str | None = None) -> dict[str, Any]  # Line 514
    async def get_positions(self) -> list[Position]  # Line 569
    async def _test_coinbase_connection(self) -> None  # Line 581
    async def _validate_coinbase_order(self, order: OrderRequest) -> None  # Line 610
    async def _place_order_advanced_api(self, order: OrderRequest) -> OrderResponse  # Line 623
    async def _place_order_pro_api(self, order: OrderRequest) -> OrderResponse  # Line 681
    def _extract_quantity_from_response(self, response, order: OrderRequest) -> Decimal  # Line 705
    def _extract_price_from_response(self, response, order: OrderRequest) -> Decimal | None  # Line 733
    def _extract_order_type_from_response(self, response, order: OrderRequest) -> OrderType  # Line 755
    def _convert_to_coinbase_symbol(self, symbol: str) -> str  # Line 779
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
    def __init__(self, ...)  # Line 57
    def _get_asset_precision(self, symbol: str, precision_type: str = 'quantity') -> int  # Line 98
    def _create_rate_limiter(self, config: Config) -> Any  # Line 102
    async def initialize(self) -> bool  # Line 124
    async def place_order(self, order: OrderRequest) -> OrderResponse  # Line 150
    async def cancel_order(self, symbol: str, order_id: str) -> bool  # Line 199
    async def get_order_status(self, symbol: str, order_id: str) -> OrderStatus  # Line 231
    async def get_order_details(self, order_id: str) -> OrderResponse | None  # Line 261
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 287
    async def get_order_history(self, symbol: str | None = None, limit: int = 100) -> list[OrderResponse]  # Line 316
    async def get_fills(self, order_id: str | None = None, symbol: str | None = None) -> list[dict]  # Line 348
    async def calculate_fees(self, order: OrderRequest) -> dict[str, Decimal]  # Line 372
    def get_total_fees(self) -> dict[str, Decimal]  # Line 413
    def get_order_statistics(self) -> dict[str, Any]  # Line 422
    async def _test_connection(self) -> None  # Line 443
    async def _validate_order(self, order: OrderRequest) -> bool  # Line 451
    def _convert_order_to_coinbase(self, order: OrderRequest) -> dict[str, Any]  # Line 485
    def _convert_coinbase_order_to_response(self, result: dict) -> OrderResponse  # Line 531
    def _convert_coinbase_status_to_order_status(self, status: str) -> OrderStatus  # Line 577
    async def __aenter__(self)  # Line 581
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 586
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

### File: data_transformer.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import OrderRequest`
- `from src.core.types import OrderResponse`
- `from src.core.types import OrderStatus`

#### Class: `ExchangeDataTransformer`

**Inherits**: Protocol
**Purpose**: Protocol for exchange data transformers

```python
class ExchangeDataTransformer(Protocol):
    def transform_symbol_to_exchange(self, symbol: str) -> str  # Line 39
    def transform_symbol_from_exchange(self, exchange_symbol: str) -> str  # Line 43
    def transform_order_to_exchange(self, order: OrderRequest) -> dict[str, Any]  # Line 47
    def transform_order_from_exchange(self, exchange_data: dict[str, Any]) -> OrderResponse  # Line 51
    def transform_ticker_from_exchange(self, exchange_data: dict[str, Any]) -> Ticker  # Line 55
```

#### Class: `BaseExchangeTransformer`

**Inherits**: ABC
**Purpose**: Base transformer with common patterns

```python
class BaseExchangeTransformer(ABC):
    def __init__(self, exchange_name: str)  # Line 63
    def validate_decimal_precision(self, value: Decimal, symbol: str) -> Decimal  # Line 67
    def validate_symbol_format(self, symbol: str) -> str  # Line 75
    def _apply_data_transformation(self, data: dict[str, Any], operation_type: str = 'stream') -> dict[str, Any]  # Line 86
    def apply_messaging_pattern(self, data: dict[str, Any], operation_type: str) -> dict[str, Any]  # Line 132
    def get_symbol_mappings(self) -> dict[str, str]  # Line 138
    def get_order_type_mappings(self) -> dict[OrderType, str]  # Line 143
    def get_order_status_mappings(self) -> dict[str, OrderStatus]  # Line 148
    def transform_for_batch_processing(cls, ...) -> dict[str, Any]  # Line 153
```

#### Class: `BinanceTransformer`

**Inherits**: BaseExchangeTransformer
**Purpose**: Binance data transformer

```python
class BinanceTransformer(BaseExchangeTransformer):
    def __init__(self)  # Line 208
    def get_symbol_mappings(self) -> dict[str, str]  # Line 211
    def get_order_type_mappings(self) -> dict[OrderType, str]  # Line 221
    def get_order_status_mappings(self) -> dict[str, OrderStatus]  # Line 230
    def transform_symbol_to_exchange(self, symbol: str) -> str  # Line 241
    def transform_symbol_from_exchange(self, exchange_symbol: str) -> str  # Line 256
```

#### Class: `CoinbaseTransformer`

**Inherits**: BaseExchangeTransformer
**Purpose**: Coinbase data transformer

```python
class CoinbaseTransformer(BaseExchangeTransformer):
    def __init__(self)  # Line 286
    def get_symbol_mappings(self) -> dict[str, str]  # Line 289
    def get_order_type_mappings(self) -> dict[OrderType, str]  # Line 300
    def get_order_status_mappings(self) -> dict[str, OrderStatus]  # Line 309
    def transform_symbol_to_exchange(self, symbol: str) -> str  # Line 319
    def transform_symbol_from_exchange(self, exchange_symbol: str) -> str  # Line 338
```

#### Class: `OKXTransformer`

**Inherits**: BaseExchangeTransformer
**Purpose**: OKX data transformer

```python
class OKXTransformer(BaseExchangeTransformer):
    def __init__(self)  # Line 346
    def get_symbol_mappings(self) -> dict[str, str]  # Line 349
    def get_order_type_mappings(self) -> dict[OrderType, str]  # Line 360
    def get_order_status_mappings(self) -> dict[str, OrderStatus]  # Line 369
    def transform_symbol_to_exchange(self, symbol: str) -> str  # Line 379
    def transform_symbol_from_exchange(self, exchange_symbol: str) -> str  # Line 398
```

#### Class: `TransformerFactory`

**Purpose**: Factory for creating exchange transformers

```python
class TransformerFactory:
    def create_transformer(cls, exchange_name: str) -> BaseExchangeTransformer  # Line 414
    def get_supported_exchanges(cls) -> list[str]  # Line 423
```

#### Functions:

```python
def standardize_decimal_precision(value: Any, symbol: str = '') -> Decimal  # Line 429
def validate_market_data_structure(data: dict[str, Any], exchange: str) -> dict[str, Any]  # Line 448
def apply_cross_module_validation(...) -> dict[str, Any]  # Line 491
def ensure_boundary_fields(data: dict[str, Any], source: str = 'exchanges') -> dict[str, Any]  # Line 567
def transform_error_to_event_data(...) -> dict[str, Any]  # Line 607
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
def register_exchange_services(container: DependencyContainer) -> None  # Line 27
def setup_exchange_services(config: Config) -> DependencyContainer  # Line 78
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
    def __init__(self, config: Config, container: DependencyContainer | None = None) -> None  # Line 29
    async def _do_start(self) -> None  # Line 46
    async def _do_stop(self) -> None  # Line 50
    def register_exchange(self, exchange_name: str, exchange_class: type[BaseExchange]) -> None  # Line 54
    def get_supported_exchanges(self) -> list[str]  # Line 72
    def is_exchange_supported(self, exchange_name: str) -> bool  # Line 81
    async def create_exchange(self, exchange_name: str) -> BaseExchange  # Line 94
    def get_available_exchanges(self) -> list[str]  # Line 128
    async def get_exchange(self, ...) -> IExchange | None  # Line 138
    async def remove_exchange(self, exchange_name: str) -> bool  # Line 172
    async def health_check_all(self) -> dict[str, Any]  # Line 187
    async def disconnect_all(self) -> None  # Line 202
    def register_default_exchanges(self) -> None  # Line 211
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
    async def set_state(self, ...) -> bool  # Line 47
    async def get_state(self, state_type: StateType, state_id: str) -> dict[str, Any] | None  # Line 59
```

#### Class: `ITradeLifecycleManager`

**Inherits**: Protocol
**Purpose**: Interface for TradeLifecycleManager used by exchanges

```python
class ITradeLifecycleManager(Protocol):
    async def update_trade_event(self, trade_id: str, event: TradeEvent, event_data: dict[str, Any]) -> None  # Line 67
```

#### Class: `IExchange`

**Inherits**: Protocol
**Purpose**: Interface contract for exchange implementations

```python
class IExchange(Protocol):
    async def connect(self) -> bool  # Line 83
    async def disconnect(self) -> None  # Line 87
    async def health_check(self) -> bool  # Line 91
    def is_connected(self) -> bool  # Line 95
    async def place_order(self, order: OrderRequest) -> OrderResponse  # Line 100
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 104
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse  # Line 108
    async def get_market_data(self, symbol: str, timeframe: str = '1m') -> MarketData  # Line 113
    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook  # Line 117
    async def get_ticker(self, symbol: str) -> Ticker  # Line 121
    async def get_trade_history(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 125
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 130
    async def get_positions(self) -> list[Position]  # Line 134
    async def get_exchange_info(self) -> ExchangeInfo  # Line 139
    async def subscribe_to_stream(self, symbol: str, callback: Any) -> None  # Line 144
    def exchange_name(self) -> str  # Line 150
```

#### Class: `IConnectionManager`

**Inherits**: Protocol
**Purpose**: Interface for connection management implementations

```python
class IConnectionManager(Protocol):
    async def connect(self) -> bool  # Line 158
    async def disconnect(self) -> None  # Line 162
    def is_connected(self) -> bool  # Line 166
    async def request(self, ...) -> dict[str, Any]  # Line 170
```

#### Class: `IRateLimiter`

**Inherits**: Protocol
**Purpose**: Interface for rate limiting implementations

```python
class IRateLimiter(Protocol):
    async def acquire(self, weight: int = 1) -> bool  # Line 185
    async def release(self, weight: int = 1) -> None  # Line 189
    def reset(self) -> None  # Line 193
    def get_statistics(self) -> dict[str, Any]  # Line 197
```

#### Class: `IHealthMonitor`

**Inherits**: Protocol
**Purpose**: Interface for health monitoring implementations

```python
class IHealthMonitor(Protocol):
    def record_success(self) -> None  # Line 205
    def record_failure(self) -> None  # Line 209
    def record_latency(self, latency_ms: float) -> None  # Line 213
    def get_health_status(self) -> dict[str, Any]  # Line 217
    async def check_health(self) -> bool  # Line 221
```

#### Class: `IExchangeAdapter`

**Inherits**: Protocol
**Purpose**: Interface for exchange adapter implementations

```python
class IExchangeAdapter(Protocol):
    async def place_order(self, **kwargs) -> dict[str, Any]  # Line 229
    async def cancel_order(self, order_id: str, **kwargs) -> dict[str, Any]  # Line 233
    async def get_order(self, order_id: str, **kwargs) -> dict[str, Any]  # Line 237
    async def get_balance(self, asset: str | None = None) -> dict[str, Any]  # Line 241
    async def get_ticker(self, symbol: str) -> dict[str, Any]  # Line 245
```

#### Class: `IExchangeFactory`

**Inherits**: Protocol
**Purpose**: Interface for exchange factory implementations

```python
class IExchangeFactory(Protocol):
    def get_supported_exchanges(self) -> list[str]  # Line 258
    def get_available_exchanges(self) -> list[str]  # Line 262
    def is_exchange_supported(self, exchange_name: str) -> bool  # Line 266
    async def get_exchange(self, ...) -> IExchange | None  # Line 270
    async def create_exchange(self, exchange_name: str) -> IExchange  # Line 276
    async def remove_exchange(self, exchange_name: str) -> bool  # Line 280
    async def health_check_all(self) -> dict[str, Any]  # Line 284
    async def disconnect_all(self) -> None  # Line 288
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
    async def connect_to_sandbox(self) -> bool  # Line 315
    async def connect_to_production(self) -> bool  # Line 319
    async def switch_environment(self, mode: SandboxMode) -> bool  # Line 323
    def get_current_endpoints(self) -> dict[str, str]  # Line 327
    async def validate_environment(self) -> dict[str, Any]  # Line 331
```

#### Class: `ISandboxAdapter`

**Inherits**: Protocol
**Purpose**: Interface for sandbox-specific exchange adapters

```python
class ISandboxAdapter(Protocol):
    def sandbox_mode(self) -> SandboxMode  # Line 345
    async def validate_sandbox_credentials(self) -> bool  # Line 349
    async def reset_sandbox_account(self) -> bool  # Line 353
    async def get_sandbox_balance(self, asset: str | None = None) -> dict[str, Any]  # Line 357
    async def place_sandbox_order(self, order: OrderRequest) -> OrderResponse  # Line 361
    async def simulate_order_fill(self, order_id: str, fill_percentage: float = 1.0) -> bool  # Line 365
    async def inject_market_data(self, symbol: str, data: dict[str, Any]) -> bool  # Line 369
    async def simulate_network_error(self, error_type: str, duration_seconds: int = 5) -> None  # Line 373
```

#### Class: `IExchangeService`

**Inherits**: Protocol
**Purpose**: Interface for exchange service layer implementations

```python
class IExchangeService(Protocol):
    async def get_exchange(self, exchange_name: str) -> 'IExchange'  # Line 381
    async def place_order(self, exchange_name: str, order: OrderRequest) -> OrderResponse  # Line 385
    async def cancel_order(self, exchange_name: str, order_id: str, symbol: str | None = None) -> bool  # Line 389
    async def get_order_status(self, exchange_name: str, order_id: str, symbol: str | None = None) -> 'OrderStatus'  # Line 395
    async def get_market_data(self, exchange_name: str, symbol: str, timeframe: str = '1m') -> MarketData  # Line 401
    async def get_order_book(self, exchange_name: str, symbol: str, depth: int = 10) -> OrderBook  # Line 407
    async def get_ticker(self, exchange_name: str, symbol: str) -> Ticker  # Line 413
    async def get_account_balance(self, exchange_name: str) -> dict[str, Decimal]  # Line 417
    async def get_positions(self, exchange_name: str, symbol: str | None = None) -> list[Position]  # Line 421
    async def get_exchange_info(self, exchange_name: str) -> ExchangeInfo  # Line 427
    def get_supported_exchanges(self) -> list[str]  # Line 431
    def get_available_exchanges(self) -> list[str]  # Line 435
    async def get_exchange_status(self, exchange_name: str) -> dict[str, Any]  # Line 439
    async def get_service_health(self) -> dict[str, Any]  # Line 443
    async def disconnect_all_exchanges(self) -> None  # Line 447
    async def get_best_price(self, symbol: str, side: str, exchanges: list[str] | None = None) -> dict[str, Any]  # Line 451
    async def subscribe_to_stream(self, exchange_name: str, symbol: str, callback: Any) -> None  # Line 457
    def get_supported_exchanges(self) -> list[str]  # Line 463
    def get_available_exchanges(self) -> list[str]  # Line 467
    async def get_exchange_status(self, exchange_name: str) -> dict[str, Any]  # Line 471
    async def get_service_health(self) -> dict[str, Any]  # Line 475
    async def disconnect_all_exchanges(self) -> None  # Line 479
    async def get_best_price(self, symbol: str, side: str, exchanges: list[str] | None = None) -> dict[str, Any]  # Line 483
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
    def __init__(self, name: str = 'mock', config: dict[str, Any] | None = None)  # Line 34
```

#### Class: `MockExchange`

**Inherits**: BaseMockExchange
**Purpose**: Mock exchange implementation for development and testing

```python
class MockExchange(BaseMockExchange):
    def __init__(self, config: dict[str, Any] | None = None, name: str = 'mock_exchange')  # Line 53
    def orders(self) -> dict[str, OrderResponse]  # Line 86
    async def connect(self) -> None  # Line 91
    async def _do_start(self) -> None  # Line 107
    async def disconnect(self) -> None  # Line 139
    async def load_exchange_info(self) -> ExchangeInfo  # Line 147
    async def get_ticker(self, symbol: str) -> Ticker  # Line 173
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook  # Line 178
    async def _place_order_impl(self, order_request: OrderRequest) -> OrderResponse  # Line 207
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 296
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 302
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse  # Line 318
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 325
    async def get_positions(self) -> list[Position]  # Line 333
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 338
    async def ping(self) -> bool  # Line 364
    def configure(self, ...) -> None  # Line 373
    def set_balance(self, balances: dict[str, Decimal]) -> None  # Line 380
    def set_price(self, symbol: str, price: Decimal) -> None  # Line 384
    def set_order_book(self, symbol: str, order_book: dict) -> None  # Line 388
    async def get_balance(self) -> dict[str, Decimal]  # Line 393
    async def get_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 397
    async def place_order_dict(self, ...) -> dict[str, Any]  # Line 403
    async def get_ticker_dict(self, symbol: str) -> dict[str, Any]  # Line 444
    async def place_order(self, *args, **kwargs)  # Line 459
    async def get_ticker_fallback(self, symbol: str)  # Line 468
    async def _get_ticker_impl(self, symbol: str) -> Ticker  # Line 474
    def _get_current_time(self)  # Line 507
    def _update_balances_for_filled_order(self, order: OrderResponse) -> None  # Line 513
    def _ensure_connected(self) -> None  # Line 544
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
    def __init__(self, config: dict[str, Any])  # Line 69
    async def connect(self) -> None  # Line 104
    async def disconnect(self) -> None  # Line 158
    async def ping(self) -> bool  # Line 184
    async def load_exchange_info(self) -> ExchangeInfo  # Line 198
    async def get_ticker(self, symbol: str) -> Ticker  # Line 232
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook  # Line 303
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]  # Line 350
    async def place_order(self, order_request: OrderRequest) -> OrderResponse  # Line 412
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse  # Line 478
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse  # Line 521
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]  # Line 554
    async def get_account_balance(self) -> dict[str, Decimal]  # Line 613
    async def get_balance(self, asset: str | None = None) -> dict[str, Any]  # Line 648
    async def get_positions(self) -> list[Position]  # Line 698
    async def _test_okx_connection(self) -> None  # Line 715
    async def _validate_okx_order(self, order: OrderRequest) -> None  # Line 742
    def _convert_order_to_okx(self, order: OrderRequest) -> dict[str, Any]  # Line 767
    def _convert_okx_order_to_response(self, result: dict[str, Any]) -> OrderResponse  # Line 787
    def _convert_okx_status_to_order_status(self, status: str) -> OrderStatus  # Line 814
    def _convert_order_type_to_okx(self, order_type: OrderType) -> str  # Line 825
    def _convert_okx_order_type_to_unified(self, okx_type: str) -> OrderType  # Line 835
    def _convert_symbol_to_okx_format(self, symbol: str) -> str  # Line 844
    def _convert_symbol_from_okx_format(self, okx_symbol: str) -> str  # Line 886
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
    def __init__(self, ...)  # Line 62
    def _get_asset_precision(self, symbol: str, precision_type: str = 'quantity') -> int  # Line 92
    async def place_order(self, order: OrderRequest) -> OrderResponse  # Line 96
    async def cancel_order(self, symbol: str, order_id: str) -> bool  # Line 158
    async def get_order_status(self, order_id: str) -> OrderStatus  # Line 190
    async def get_order_fills(self, order_id: str) -> list[OrderResponse]  # Line 220
    async def place_stop_loss_order(self, order: OrderRequest, stop_price: Decimal) -> OrderResponse  # Line 269
    async def place_take_profit_order(self, order: OrderRequest, take_profit_price: Decimal) -> OrderResponse  # Line 300
    async def place_oco_order(self, order: OrderRequest, stop_price: Decimal, take_profit_price: Decimal) -> list[OrderResponse]  # Line 331
    def calculate_fee(self, order: OrderRequest, is_maker: bool = False) -> Decimal  # Line 364
    def _validate_order_request(self, order: OrderRequest) -> None  # Line 388
    def _convert_order_to_okx(self, order: OrderRequest) -> dict[str, Any]  # Line 426
    def _convert_okx_order_to_response(self, result: dict) -> OrderResponse  # Line 465
    def _convert_okx_status_to_order_status(self, status: str) -> OrderStatus  # Line 500
    def _convert_order_type_to_okx(self, order_type: OrderType) -> str  # Line 504
    def _convert_okx_order_type_to_unified(self, okx_type: str) -> OrderType  # Line 523
    def get_active_orders(self) -> dict[str, dict]  # Line 544
    def get_order_history(self, symbol: str | None = None) -> list[dict]  # Line 553
    def clear_order_history(self) -> None  # Line 570
    def get_fee_rates(self) -> dict[str, Decimal]  # Line 575
    def update_fee_rates(self, maker_rate: Decimal, taker_rate: Decimal) -> None  # Line 584
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
    async def _handle_orderbook_data(self, symbol: str, data: list[dict]) -> None  # Line 733
    async def _handle_trades_data(self, symbol: str, data: list[dict]) -> None  # Line 787
    async def _handle_account_data(self, data: list[dict]) -> None  # Line 830
    async def _send_public_message(self, message: dict) -> None  # Line 854
    async def _send_private_message(self, message: dict) -> None  # Line 871
    async def _handle_disconnect(self) -> None  # Line 888
    async def _schedule_reconnect(self) -> None  # Line 898
    async def _reconnect_after_delay(self, delay: float) -> None  # Line 927
    async def _health_monitor(self) -> None  # Line 991
    def is_connected(self) -> bool  # Line 1022
    async def health_check(self) -> bool  # Line 1031
    def get_connection_metrics(self) -> dict[str, Any]  # Line 1072
    def get_status(self) -> str  # Line 1113
```

### File: service.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.core.base.interfaces import HealthStatus`
- `from src.core.config import Config`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`

#### Class: `ExchangeService`

**Inherits**: BaseService, IExchangeService
**Purpose**: Service layer for exchange operations

```python
class ExchangeService(BaseService, IExchangeService):
    def __init__(self, ...)  # Line 56
    async def _do_start(self) -> None  # Line 89
    async def _do_stop(self) -> None  # Line 102
    async def get_exchange(self, exchange_name: str) -> IExchange  # Line 122
    async def _is_exchange_healthy(self, exchange: IExchange) -> bool  # Line 175
    async def _remove_exchange(self, exchange_name: str) -> None  # Line 190
    async def place_order(self, exchange_name: str, order: OrderRequest) -> OrderResponse  # Line 220
    async def _validate_order_request(self, order: OrderRequest) -> None  # Line 257
    async def _process_order_response(self, exchange_name: str, order_response: OrderResponse) -> None  # Line 275
    async def cancel_order(self, exchange_name: str, order_id: str, symbol: str | None = None) -> bool  # Line 297
    async def get_order_status(self, exchange_name: str, order_id: str, symbol: str | None = None) -> OrderStatus  # Line 361
    async def get_market_data(self, exchange_name: str, symbol: str, timeframe: str = '1m') -> MarketData  # Line 413
    async def get_order_book(self, exchange_name: str, symbol: str, depth: int = 10) -> OrderBook  # Line 446
    async def get_ticker(self, exchange_name: str, symbol: str) -> Ticker  # Line 470
    async def get_account_balance(self, exchange_name: str) -> dict[str, Decimal]  # Line 493
    async def get_positions(self, exchange_name: str, symbol: str | None = None) -> list[Position]  # Line 511
    async def get_exchange_info(self, exchange_name: str) -> ExchangeInfo  # Line 537
    def get_supported_exchanges(self) -> list[str]  # Line 552
    def get_available_exchanges(self) -> list[str]  # Line 556
    async def get_exchange_status(self, exchange_name: str) -> dict[str, Any]  # Line 560
    async def get_service_health(self) -> dict[str, Any]  # Line 605
    async def disconnect_all_exchanges(self) -> None  # Line 631
    async def get_best_price(self, symbol: str, side: str, exchanges: list[str] | None = None) -> dict[str, Any]  # Line 667
    async def _get_ticker_safe(self, exchange_name: str, symbol: str) -> Ticker | None  # Line 724
    async def subscribe_to_stream(self, exchange_name: str, symbol: str, callback: Any) -> None  # Line 734
    async def _service_health_check(self) -> HealthStatus  # Line 746
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
**Total Classes**: 56
**Total Functions**: 7