# WEB_INTERFACE Module Reference

## INTEGRATION
**Dependencies**: analytics, bot_management, capital_management, core, data, database, error_handling, exchanges, execution, monitoring, optimization, risk_management, state, utils
**Used By**: None
**Provides**: AuthManager, BotStatusManager, ConnectionManager, ConnectionPoolManager, MockAnalyticsService, MockCapitalService, MockMarketDataService, PortfolioManager, SocketIOManager, UnifiedWebSocketManager, VersionManager, WebAnalyticsService, WebAuthService, WebBotService, WebCapitalService, WebDataService, WebExchangeService, WebMonitoringService, WebPortfolioService, WebRiskService, WebStrategyService, WebTradingService
**Patterns**: Async Operations, Component Architecture, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Financial data handling
- Decimal precision arithmetic
**Security**:
- Credential management
- Authentication
- Authentication
**Performance**:
- Parallel execution
- Caching
- Parallel execution
**Architecture**:
- WebInterfaceFactory inherits from base architecture
- SocketIOManager inherits from base architecture
- ConnectionHealthMonitor inherits from base architecture

## MODULE OVERVIEW
**Files**: 61 Python files
**Classes**: 198
**Functions**: 386

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `PortfolioMetricsResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for portfolio metrics
**Status**: Complete

### Implementation: `RiskMetricsResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for risk metrics
**Status**: Complete

### Implementation: `StrategyMetricsResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for strategy metrics
**Status**: Complete

### Implementation: `VaRRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for VaR calculation
**Status**: Complete

### Implementation: `StressTestRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for stress testing
**Status**: Complete

### Implementation: `GenerateReportRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for report generation
**Status**: Complete

### Implementation: `AlertAcknowledgeRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for alert acknowledgment
**Status**: Complete

### Implementation: `RefreshTokenRequest` ✅

**Inherits**: BaseModel
**Purpose**: Refresh token request model
**Status**: Complete

### Implementation: `CreateUserRequest` ✅

**Inherits**: BaseModel
**Purpose**: Create user request model
**Status**: Complete

### Implementation: `ChangePasswordRequest` ✅

**Inherits**: BaseModel
**Purpose**: Change password request model
**Status**: Complete

### Implementation: `AuthResponse` ✅

**Inherits**: BaseModel
**Purpose**: Authentication response model
**Status**: Complete

### Implementation: `CreateBotRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for creating a new bot
**Status**: Complete

### Implementation: `UpdateBotRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for updating bot configuration
**Status**: Complete

### Implementation: `BotResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for bot information
**Status**: Complete

### Implementation: `BotSummaryResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for bot summary
**Status**: Complete

### Implementation: `BotListResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for bot listing
**Status**: Complete

### Implementation: `CapitalAllocationRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for capital allocation
**Status**: Complete

### Implementation: `CapitalReleaseRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for capital release
**Status**: Complete

### Implementation: `CapitalUtilizationUpdate` ✅

**Inherits**: BaseModel
**Purpose**: Request model for utilization update
**Status**: Complete

### Implementation: `CurrencyHedgeRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for currency hedging
**Status**: Complete

### Implementation: `FundFlowRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for recording fund flows
**Status**: Complete

### Implementation: `CapitalLimitsUpdate` ✅

**Inherits**: BaseModel
**Purpose**: Request model for updating capital limits
**Status**: Complete

### Implementation: `CapitalAllocationResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for capital allocation
**Status**: Complete

### Implementation: `CapitalMetricsResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for capital metrics
**Status**: Complete

### Implementation: `PipelineControlRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for pipeline control
**Status**: Complete

### Implementation: `DataValidationRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for data validation
**Status**: Complete

### Implementation: `FeatureComputeRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for feature computation
**Status**: Complete

### Implementation: `DataSourceConfigRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for data source configuration
**Status**: Complete

### Implementation: `PipelineStatusResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for pipeline status
**Status**: Complete

### Implementation: `DataQualityMetricsResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for data quality metrics
**Status**: Complete

### Implementation: `ExchangeConnectionRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for exchange connection
**Status**: Complete

### Implementation: `ExchangeConfigRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for exchange configuration
**Status**: Complete

### Implementation: `RateLimitConfigRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for rate limit configuration
**Status**: Complete

### Implementation: `ExchangeStatusResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for exchange status
**Status**: Complete

### Implementation: `ExchangeHealthResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for exchange health
**Status**: Complete

### Implementation: `ConnectionHealthMonitor` ✅

**Inherits**: BaseComponent
**Purpose**: Mock connection health monitor for exchanges
**Status**: Complete

**Implemented Methods:**
- `async get_health_status(self) -> dict` - Line 42

### Implementation: `HealthStatus` ✅

**Inherits**: BaseModel
**Purpose**: Health check response model
**Status**: Complete

### Implementation: `ComponentHealth` ✅

**Inherits**: BaseModel
**Purpose**: Individual component health model
**Status**: Complete

### Implementation: `ModelResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for ML model information
**Status**: Complete

### Implementation: `CreateModelRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for creating a new ML model
**Status**: Complete

### Implementation: `TrainModelRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for training a model
**Status**: Complete

### Implementation: `PredictionRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for model predictions
**Status**: Complete

### Implementation: `PredictionResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for model predictions
**Status**: Complete

### Implementation: `TrainingJobResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for training job status
**Status**: Complete

### Implementation: `ModelPerformanceResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for model performance metrics
**Status**: Complete

### Implementation: `DeploymentRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for model deployment
**Status**: Complete

### Implementation: `FeatureEngineeringRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for feature engineering
**Status**: Complete

### Implementation: `FeatureSelectionRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for feature selection
**Status**: Complete

### Implementation: `ABTestRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for A/B test creation
**Status**: Complete

### Implementation: `HyperparameterOptimizationRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for hyperparameter optimization
**Status**: Complete

### Implementation: `HealthCheckResponse` ✅

**Inherits**: BaseModel
**Purpose**: Health check response model
**Status**: Complete

### Implementation: `MetricsResponse` ✅

**Inherits**: BaseModel
**Purpose**: Metrics response model
**Status**: Complete

### Implementation: `AlertRuleRequest` ✅

**Inherits**: BaseModel
**Purpose**: Alert rule creation request
**Status**: Complete

### Implementation: `AlertResponse` ✅

**Inherits**: BaseModel
**Purpose**: Alert response model
**Status**: Complete

### Implementation: `PerformanceStatsResponse` ✅

**Inherits**: BaseModel
**Purpose**: Performance statistics response
**Status**: Complete

### Implementation: `MemoryReportResponse` ✅

**Inherits**: BaseModel
**Purpose**: Memory usage report response
**Status**: Complete

### Implementation: `TimeInterval` ✅

**Inherits**: Enum
**Purpose**: Time interval enumeration for data intervals
**Status**: Complete

### Implementation: `OptimizationParameterRange` ✅

**Inherits**: BaseModel
**Purpose**: Model for parameter optimization range
**Status**: Complete

### Implementation: `OptimizationRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for optimization job
**Status**: Complete

**Implemented Methods:**
- `validate_parameter_ranges(cls, v)` - Line 135
- `validate_end_date(cls, v, info)` - Line 142

### Implementation: `OptimizationResult` ✅

**Inherits**: BaseModel
**Purpose**: Model for individual optimization result
**Status**: Complete

### Implementation: `OptimizationJobResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for optimization job
**Status**: Complete

### Implementation: `OptimizationStatusResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for optimization job status
**Status**: Complete

### Implementation: `OptimizationResultsResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for optimization results
**Status**: Complete

### Implementation: `TimeInterval` ✅

**Inherits**: Enum
**Purpose**: Time interval enumeration for data intervals
**Status**: Complete

### Implementation: `PlaygroundConfigurationRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for playground configuration
**Status**: Complete

### Implementation: `PlaygroundSessionResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for playground session
**Status**: Complete

### Implementation: `PlaygroundStatusResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for playground status
**Status**: Complete

### Implementation: `PlaygroundResultResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for playground results
**Status**: Complete

### Implementation: `PlaygroundLogEntry` ✅

**Inherits**: BaseModel
**Purpose**: Model for playground log entry
**Status**: Complete

### Implementation: `PlaygroundConfigurationModel` ✅

**Inherits**: BaseModel
**Purpose**: Playground configuration model
**Status**: Complete

### Implementation: `ABTestModel` ✅

**Inherits**: BaseModel
**Purpose**: A/B test model
**Status**: Complete

### Implementation: `BatchOptimizationModel` ✅

**Inherits**: BaseModel
**Purpose**: Batch optimization model
**Status**: Complete

### Implementation: `PositionResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for position information
**Status**: Complete

### Implementation: `BalanceResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for balance information
**Status**: Complete

### Implementation: `PnLResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for P&L information
**Status**: Complete

### Implementation: `PortfolioSummaryResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for portfolio summary
**Status**: Complete

### Implementation: `AssetAllocationResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for asset allocation
**Status**: Complete

### Implementation: `RiskMetricsResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for risk metrics
**Status**: Complete

### Implementation: `RiskLimitsResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for risk limits
**Status**: Complete

### Implementation: `UpdateRiskLimitsRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for updating risk limits
**Status**: Complete

### Implementation: `RiskAlertResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for risk alerts
**Status**: Complete

### Implementation: `PositionRiskResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for individual position risk
**Status**: Complete

### Implementation: `StressTestRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for stress testing
**Status**: Complete

### Implementation: `StressTestResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for stress test results
**Status**: Complete

### Implementation: `StateSnapshotRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for creating state snapshots
**Status**: Complete

### Implementation: `StateSnapshotResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for state snapshots
**Status**: Complete

### Implementation: `CheckpointRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for checkpoint operations
**Status**: Complete

### Implementation: `CheckpointResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for checkpoint operations
**Status**: Complete

### Implementation: `RecoveryRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for recovery operations
**Status**: Complete

### Implementation: `TradeValidationRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for trade validation
**Status**: Complete

### Implementation: `TradeValidationResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for trade validation
**Status**: Complete

### Implementation: `PostTradeAnalysisRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for post-trade analysis
**Status**: Complete

### Implementation: `PostTradeAnalysisResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for post-trade analysis
**Status**: Complete

### Implementation: `SyncStatusResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for sync status
**Status**: Complete

### Implementation: `StrategyResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for strategy information
**Status**: Complete

### Implementation: `StrategyConfigRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for strategy configuration
**Status**: Complete

### Implementation: `StrategyPerformanceResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for strategy performance
**Status**: Complete

### Implementation: `BacktestRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for strategy backtesting
**Status**: Complete

### Implementation: `BacktestResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for backtest results
**Status**: Complete

### Implementation: `PlaceOrderRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for placing an order
**Status**: Complete

### Implementation: `CancelOrderRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for cancelling an order
**Status**: Complete

### Implementation: `OrderResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for order information
**Status**: Complete

### Implementation: `TradeResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for trade information
**Status**: Complete

### Implementation: `OrderBookResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for order book data
**Status**: Complete

### Implementation: `MarketDataResponse` ✅

**Inherits**: BaseModel
**Purpose**: Response model for market data
**Status**: Complete

### Implementation: `LazyApp` ✅

**Purpose**: Lazy app wrapper that creates app on first access
**Status**: Complete

**Implemented Methods:**

### Implementation: `AuthManager` ✅

**Inherits**: BaseComponent
**Purpose**: Unified authentication manager
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector)` - Line 42
- `async authenticate(self, credentials: dict[str, Any], provider_type: str | None = None) -> tuple[User, AuthToken] | None` - Line 130
- `async validate_token(self, token_value: str, provider_type: str | None = None) -> User | None` - Line 180
- `async revoke_token(self, token_value: str, provider_type: str | None = None) -> bool` - Line 211
- `async refresh_token(self, refresh_token_value: str) -> AuthToken | None` - Line 235
- `async create_api_key(self, user: User) -> AuthToken | None` - Line 255
- `get_user(self, user_id: str) -> User | None` - Line 275
- `get_user_by_username(self, username: str) -> User | None` - Line 279
- `async create_user(self, user_data: dict[str, Any]) -> User | None` - Line 286
- `async update_user(self, user_id: str, updates: dict[str, Any]) -> bool` - Line 329
- `async change_password(self, user_id: str, old_password: str, new_password: str) -> bool` - Line 365
- `get_user_stats(self) -> dict[str, Any]` - Line 392

### Implementation: `AuthMiddleware` ✅

**Inherits**: BaseHTTPMiddleware, BaseComponent
**Purpose**: Middleware to handle authentication for all requests
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 31

### Implementation: `PermissionType` ✅

**Inherits**: Enum
**Purpose**: Permission types for role-based access control
**Status**: Complete

### Implementation: `Permission` ✅

**Purpose**: Represents a system permission
**Status**: Complete

**Implemented Methods:**

### Implementation: `Role` ✅

**Purpose**: Represents a user role with associated permissions
**Status**: Complete

**Implemented Methods:**
- `add_permission(self, permission: Permission) -> None` - Line 99
- `remove_permission(self, permission: Permission) -> None` - Line 103
- `has_permission(self, permission_type: PermissionType, resource: str | None = None) -> bool` - Line 107
- `get_permissions_by_type(self, permission_type: PermissionType) -> list[Permission]` - Line 115

### Implementation: `UserStatus` ✅

**Inherits**: Enum
**Purpose**: User account status
**Status**: Complete

### Implementation: `User` ✅

**Purpose**: Represents a system user
**Status**: Complete

**Implemented Methods:**
- `add_role(self, role: Role) -> None` - Line 152
- `remove_role(self, role: Role) -> None` - Line 156
- `has_role(self, role_name: str) -> bool` - Line 160
- `has_permission(self, permission_type: PermissionType, resource: str | None = None) -> bool` - Line 164
- `get_all_permissions(self) -> set[Permission]` - Line 171
- `is_locked(self) -> bool` - Line 178
- `is_active(self) -> bool` - Line 186
- `lock_account(self, duration_minutes: int = 30) -> None` - Line 190
- `unlock_account(self) -> None` - Line 195
- `increment_login_attempts(self) -> None` - Line 202
- `reset_login_attempts(self) -> None` - Line 208

### Implementation: `TokenType` ✅

**Inherits**: Enum
**Purpose**: Authentication token types
**Status**: Complete

### Implementation: `AuthToken` ✅

**Purpose**: Represents an authentication token
**Status**: Complete

**Implemented Methods:**
- `is_expired(self) -> bool` - Line 241
- `is_valid(self) -> bool` - Line 247
- `revoke(self) -> None` - Line 251
- `touch(self) -> None` - Line 255

### Implementation: `AuthProvider` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for authentication providers
**Status**: Abstract Base Class

**Implemented Methods:**
- `async authenticate(self, credentials: dict[str, Any]) -> User | None` - Line 24
- `async create_token(self, user: User) -> AuthToken` - Line 29
- `async validate_token(self, token_value: str) -> User | None` - Line 34
- `async revoke_token(self, token_value: str) -> bool` - Line 39

### Implementation: `JWTAuthProvider` ✅

**Inherits**: AuthProvider, BaseComponent
**Purpose**: JWT-based authentication provider
**Status**: Complete

**Implemented Methods:**
- `async authenticate(self, credentials: dict[str, Any]) -> User | None` - Line 64
- `async create_token(self, user: User) -> AuthToken` - Line 104
- `async create_refresh_token(self, user: User) -> AuthToken` - Line 134
- `async validate_token(self, token_value: str) -> User | None` - Line 162
- `async revoke_token(self, token_value: str) -> bool` - Line 217
- `async refresh_access_token(self, refresh_token_value: str) -> AuthToken | None` - Line 234

### Implementation: `SessionAuthProvider` ✅

**Inherits**: AuthProvider, BaseComponent
**Purpose**: Session-based authentication provider
**Status**: Complete

**Implemented Methods:**
- `async authenticate(self, credentials: dict[str, Any]) -> User | None` - Line 288
- `async create_token(self, user: User) -> AuthToken` - Line 331
- `async validate_token(self, token_value: str) -> User | None` - Line 357
- `async revoke_token(self, token_value: str) -> bool` - Line 391
- `async cleanup_expired_sessions(self) -> int` - Line 398

### Implementation: `APIKeyAuthProvider` ✅

**Inherits**: AuthProvider, BaseComponent
**Purpose**: API key-based authentication provider
**Status**: Complete

**Implemented Methods:**
- `async authenticate(self, credentials: dict[str, Any]) -> User | None` - Line 421
- `async create_token(self, user: User) -> AuthToken` - Line 429
- `async validate_token(self, token_value: str) -> User | None` - Line 450
- `async validate_api_key(self, api_key: str) -> User | None` - Line 454
- `async revoke_token(self, token_value: str) -> bool` - Line 480

### Implementation: `WebInterfaceErrorPropagator` ✅

**Purpose**: Error propagation patterns aligned with risk_management module
**Status**: Complete

**Implemented Methods:**
- `propagate_to_risk_management(error: Exception, context: str) -> None` - Line 272
- `propagate_validation_error(error: Exception, context: str) -> None` - Line 282
- `propagate_to_monitoring(error: Exception, context: str) -> None` - Line 290

### Implementation: `WebInterfaceDataTransformer` ✅

**Purpose**: Data transformer for backward compatibility, aligned with core standards
**Status**: Complete

**Implemented Methods:**
- `format_portfolio_composition(data: Any) -> Any` - Line 303
- `format_stress_test_results(data: Any) -> Any` - Line 314
- `format_operational_metrics(data: Any) -> Any` - Line 325
- `transform_risk_data_to_event_data(data: Any, **kwargs) -> Any` - Line 336
- `validate_financial_precision(data: Any) -> Any` - Line 354

### Implementation: `APIFacade` ✅

**Inherits**: BaseComponent
**Purpose**: Unified API facade for T-Bot Trading System
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector)` - Line 56
- `async initialize(self) -> None` - Line 76
- `async cleanup(self) -> None` - Line 97
- `async place_order(self, ...) -> str` - Line 119
- `async cancel_order(self, order_id: str) -> bool` - Line 141
- `async get_positions(self) -> list[Position]` - Line 149
- `async create_bot(self, config: BotConfiguration) -> str` - Line 160
- `async start_bot(self, bot_id: str) -> bool` - Line 166
- `async stop_bot(self, bot_id: str) -> bool` - Line 172
- `async get_bot_status(self, bot_id: str) -> dict[str, Any]` - Line 178
- `async list_bots(self) -> list[dict[str, Any]]` - Line 184
- `async get_balance(self) -> dict[str, Decimal]` - Line 191
- `async get_portfolio_summary(self) -> dict[str, Any]` - Line 198
- `async get_pnl_report(self, start_date: datetime, end_date: datetime) -> dict[str, Any]` - Line 204
- `async validate_order(self, ...) -> bool` - Line 212
- `async get_risk_summary(self) -> dict[str, Any]` - Line 223
- `async get_risk_metrics(self) -> dict[str, Any]` - Line 229
- `async calculate_position_size(self, ...) -> Decimal` - Line 233
- `async list_strategies(self) -> list[dict[str, Any]]` - Line 251
- `async get_strategy_config(self, strategy_name: str) -> dict[str, Any]` - Line 257
- `async validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool` - Line 263
- `async health_check(self) -> dict[str, Any]` - Line 272
- `get_service_status(self, service_name: str) -> dict[str, Any]` - Line 288
- `async delete_bot(self, bot_id: str, force: bool = False) -> bool` - Line 309

### Implementation: `ServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Base interface for all services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 20
- `async cleanup(self) -> None` - Line 25

### Implementation: `ServiceRegistry` ✅

**Inherits**: BaseComponent
**Purpose**: Central registry for all system services
**Status**: Complete

**Implemented Methods:**
- `register_service(self, name: str, service: Any, interface: type | None = None) -> None` - Line 39
- `get_service(self, name: str) -> Any` - Line 58
- `has_service(self, name: str) -> bool` - Line 75
- `async initialize_all(self) -> None` - Line 79
- `async cleanup_all(self) -> None` - Line 92
- `list_services(self) -> dict[str, str]` - Line 103
- `get_all_service_names(self) -> list[str]` - Line 107

### Implementation: `WebInterfaceFactory` ✅

**Inherits**: BaseComponent
**Purpose**: Factory for creating web interface components
**Status**: Complete

**Implemented Methods:**
- `create_service_registry(self) -> ServiceRegistry` - Line 60
- `create_jwt_handler(self, config: dict[str, Any] | None = None) -> JWTHandler` - Line 77
- `create_auth_manager(self, config: dict[str, Any] | None = None) -> AuthManager` - Line 101
- `create_trading_service(self) -> WebTradingServiceInterface` - Line 124
- `create_bot_management_service(self) -> WebBotServiceInterface` - Line 143
- `create_portfolio_service(self) -> WebPortfolioServiceInterface` - Line 162
- `create_risk_service(self) -> WebRiskServiceInterface` - Line 181
- `create_strategy_service(self) -> WebStrategyServiceInterface` - Line 200
- `create_market_data_service(self)` - Line 220
- `create_api_facade(self) -> APIFacade` - Line 264
- `create_websocket_manager(self) -> UnifiedWebSocketManager` - Line 331
- `create_complete_web_stack(self, config: dict[str, Any] | None = None) -> dict[str, Any]` - Line 360

### Implementation: `WebPortfolioServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web portfolio service operations
**Status**: Complete

**Implemented Methods:**
- `async get_portfolio_summary_data(self) -> dict[str, Any]` - Line 16
- `async calculate_pnl_periods(self, total_pnl: Decimal, total_trades: int, win_rate: float) -> dict[str, dict[str, Any]]` - Line 20
- `async get_processed_positions(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]` - Line 26
- `async calculate_pnl_metrics(self, period: str) -> dict[str, Any]` - Line 32
- `generate_mock_balances(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]` - Line 36
- `calculate_asset_allocation(self) -> list[dict[str, Any]]` - Line 40
- `generate_performance_chart_data(self, period: str, resolution: str) -> dict[str, Any]` - Line 44

### Implementation: `WebTradingServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web trading service operations
**Status**: Complete

**Implemented Methods:**
- `async validate_order_request(self, ...) -> dict[str, Any]` - Line 52
- `async format_order_response(self, order_result: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]` - Line 63
- `async get_formatted_orders(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]` - Line 69
- `async get_formatted_trades(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]` - Line 75
- `async get_market_data_with_context(self, symbol: str, exchange: str = 'binance') -> dict[str, Any]` - Line 81
- `async generate_order_book_data(self, symbol: str, exchange: str, depth: int) -> dict[str, Any]` - Line 87
- `async place_order_through_service(self, ...) -> dict[str, Any]` - Line 93
- `async cancel_order_through_service(self, order_id: str) -> bool` - Line 104
- `async get_order_details(self, order_id: str, exchange: str) -> dict[str, Any]` - Line 108
- `async get_service_health(self) -> dict[str, Any]` - Line 112

### Implementation: `WebBotServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web bot service operations
**Status**: Complete

**Implemented Methods:**
- `async validate_bot_configuration(self, config_data: dict[str, Any]) -> dict[str, Any]` - Line 120
- `async format_bot_response(self, bot_data: dict[str, Any]) -> dict[str, Any]` - Line 124
- `async get_formatted_bot_list(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]` - Line 128
- `async calculate_bot_metrics(self, bot_id: str) -> dict[str, Any]` - Line 134
- `async validate_bot_operation(self, bot_id: str, operation: str) -> dict[str, Any]` - Line 138
- `async create_bot_configuration(self, request_data: dict[str, Any], user_id: str) -> Any` - Line 142
- `async pause_bot_through_service(self, bot_id: str) -> bool` - Line 146
- `async resume_bot_through_service(self, bot_id: str) -> bool` - Line 150
- `async create_bot_through_service(self, bot_config: Any) -> str` - Line 154
- `async get_bot_status_through_service(self, bot_id: str) -> dict[str, Any]` - Line 158
- `async start_bot_through_service(self, bot_id: str) -> bool` - Line 162
- `async stop_bot_through_service(self, bot_id: str) -> bool` - Line 166
- `async delete_bot_through_service(self, bot_id: str, force: bool = False) -> bool` - Line 170
- `async list_bots_through_service(self) -> list[dict[str, Any]]` - Line 174
- `async update_bot_configuration(self, bot_id: str, update_data: dict[str, Any], user_id: str) -> dict[str, Any]` - Line 178
- `async start_bot_with_execution_integration(self, bot_id: str) -> bool` - Line 184
- `async stop_bot_with_execution_integration(self, bot_id: str) -> bool` - Line 188
- `get_controller_health_check(self) -> dict[str, Any]` - Line 192

### Implementation: `WebMonitoringServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web monitoring service operations
**Status**: Complete

**Implemented Methods:**
- `async get_system_health_summary(self) -> dict[str, Any]` - Line 200
- `async get_performance_metrics(self, component: str | None = None) -> dict[str, Any]` - Line 204
- `async get_error_summary(self, time_range: str = '24h') -> dict[str, Any]` - Line 208
- `async get_alert_dashboard_data(self) -> dict[str, Any]` - Line 212

### Implementation: `WebRiskServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web risk service operations
**Status**: Complete

**Implemented Methods:**
- `async get_risk_dashboard_data(self) -> dict[str, Any]` - Line 220
- `async validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]` - Line 224
- `async calculate_position_risk(self, symbol: str, quantity: Decimal, price: Decimal) -> dict[str, Any]` - Line 228
- `async get_portfolio_risk_breakdown(self) -> dict[str, Any]` - Line 234
- `async get_current_risk_limits(self) -> dict[str, Any]` - Line 238

### Implementation: `WebStrategyServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web strategy service operations
**Status**: Complete

**Implemented Methods:**
- `async get_formatted_strategies(self) -> list[dict[str, Any]]` - Line 246
- `async validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> dict[str, Any]` - Line 250
- `async get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]` - Line 256
- `async format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]` - Line 260

### Implementation: `WebDataServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web data service operations
**Status**: Complete

**Implemented Methods:**
- `async get_market_overview(self, exchange: str = 'binance') -> dict[str, Any]` - Line 268
- `async get_symbol_analytics(self, symbol: str) -> dict[str, Any]` - Line 272
- `async get_historical_chart_data(self, symbol: str, timeframe: str, period: str) -> dict[str, Any]` - Line 276
- `async get_real_time_feed_status(self) -> dict[str, Any]` - Line 282

### Implementation: `WebServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Base interface for all web services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 292
- `async cleanup(self) -> None` - Line 297
- `health_check(self) -> dict[str, Any]` - Line 302
- `get_service_info(self) -> dict[str, Any]` - Line 307

### Implementation: `WebStrategyServiceExtendedInterface` ✅

**Inherits**: Protocol
**Purpose**: Extended interface for web strategy service operations
**Status**: Complete

**Implemented Methods:**
- `async get_formatted_strategies(self) -> list[dict[str, Any]]` - Line 315
- `async validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> dict[str, Any]` - Line 319
- `async get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]` - Line 325
- `async format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]` - Line 329
- `health_check(self) -> dict[str, Any]` - Line 333
- `get_service_info(self) -> dict[str, Any]` - Line 337

### Implementation: `WebAuthServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web authentication service operations
**Status**: Complete

**Implemented Methods:**
- `async get_user_by_username(self, username: str) -> Any | None` - Line 345
- `async authenticate_user(self, username: str, password: str) -> Any | None` - Line 349
- `async create_user(self, ...) -> Any` - Line 353
- `async get_auth_summary(self) -> dict[str, Any]` - Line 364
- `get_user_roles(self, current_user: Any) -> list[str]` - Line 368
- `check_permission(self, current_user: Any, required_roles: list[str]) -> bool` - Line 372
- `require_permission(self, current_user: Any, required_roles: list[str]) -> None` - Line 376
- `require_admin(self, current_user: Any) -> None` - Line 380
- `require_trading_permission(self, current_user: Any) -> None` - Line 384
- `require_risk_manager_permission(self, current_user: Any) -> None` - Line 388
- `health_check(self) -> dict[str, Any]` - Line 392
- `get_service_info(self) -> dict[str, Any]` - Line 396

### Implementation: `AuthMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Authentication middleware for request processing
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 55

### Implementation: `CacheEntry` ✅

**Purpose**: Represents a cached response entry
**Status**: Complete

**Implemented Methods:**
- `is_expired(self) -> bool` - Line 35
- `touch(self)` - Line 39

### Implementation: `CacheMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Intelligent caching middleware for API responses
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 144
- `get_cache_stats(self) -> dict[str, Any]` - Line 474
- `clear_cache(self, pattern: str | None = None) -> None` - Line 511

### Implementation: `PoolAsyncUnitOfWork` ✅

**Purpose**: Simplified async Unit of Work for connection pool usage
**Status**: Complete

**Implemented Methods:**
- `async commit(self)` - Line 64
- `async rollback(self)` - Line 69
- `async close(self)` - Line 74

### Implementation: `ConnectionPoolManager` ✅

**Purpose**: Manages connection pools for database and Redis connections
**Status**: Complete

**Implemented Methods:**
- `async initialize(self)` - Line 120
- `async get_db_connection(self)` - Line 253
- `async get_uow(self)` - Line 276
- `async get_redis_connection(self)` - Line 294
- `async health_check(self) -> dict[str, Any]` - Line 310
- `async get_pool_stats(self) -> dict[str, Any]` - Line 386
- `async close(self)` - Line 452

### Implementation: `ConnectionPoolMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to provide connection pool access to requests
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next)` - Line 504
- `async startup(self)` - Line 532
- `async shutdown(self)` - Line 536

### Implementation: `ConnectionHealthMonitor` ✅

**Purpose**: Monitor connection pool health and performance
**Status**: Complete

**Implemented Methods:**
- `async start_monitoring(self)` - Line 629
- `async stop_monitoring(self)` - Line 638

### Implementation: `CorrelationMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to handle correlation IDs for request tracking
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 22

### Implementation: `DecimalPrecisionMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to handle Decimal precision in API requests and responses
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next)` - Line 134

### Implementation: `DecimalValidationMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Additional middleware for strict decimal validation in critical trading operations
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next)` - Line 427

### Implementation: `ErrorHandlerMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Comprehensive error handling middleware
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 136
- `get_error_stats(self) -> dict` - Line 583

### Implementation: `FinancialValidationMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware for validating financial inputs on trading endpoints
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 105

### Implementation: `DecimalEnforcementMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to enforce Decimal usage in responses
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 315

### Implementation: `CurrencyValidationMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to validate currency codes and formats
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 385

### Implementation: `RateLimitMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Advanced rate limiting middleware
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 98
- `get_rate_limit_stats(self) -> dict` - Line 331

### Implementation: `SecurityMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Comprehensive security middleware for web application protection
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next)` - Line 126
- `get_security_stats(self) -> dict[str, Any]` - Line 374
- `unblock_ip(self, ip_address: str) -> bool` - Line 400
- `clear_blocked_ips(self)` - Line 417

### Implementation: `InputSanitizer` ✅

**Purpose**: Input sanitization utilities for trading data
**Status**: Complete

**Implemented Methods:**
- `sanitize_symbol(symbol: str) -> str` - Line 434
- `sanitize_decimal_string(value: str) -> str` - Line 454
- `validate_exchange_name(exchange: str) -> bool` - Line 491
- `validate_order_side(side: str) -> bool` - Line 515

### Implementation: `UserInDB` ✅

**Inherits**: BaseModel
**Purpose**: User model for database storage
**Status**: Complete

### Implementation: `User` ✅

**Inherits**: BaseModel
**Purpose**: User model for API responses
**Status**: Complete

### Implementation: `Token` ✅

**Inherits**: BaseModel
**Purpose**: Token response model
**Status**: Complete

### Implementation: `LoginRequest` ✅

**Inherits**: BaseModel
**Purpose**: Login request model
**Status**: Complete

### Implementation: `TokenData` ✅

**Inherits**: BaseModel
**Purpose**: Token data structure
**Status**: Complete

### Implementation: `JWTHandler` ✅

**Inherits**: BaseComponent
**Purpose**: Advanced JWT token handler with security features
**Status**: Complete

**Implemented Methods:**
- `hash_password(self, password: str) -> str` - Line 190
- `verify_password(self, plain_password: str, hashed_password: str) -> bool` - Line 194
- `create_access_token(self, ...) -> str` - Line 202
- `create_refresh_token(self, user_id: str, username: str) -> str` - Line 254
- `validate_token(self, token: str) -> TokenData` - Line 289
- `validate_scopes(self, token_data: TokenData, required_scopes: list[str]) -> bool` - Line 336
- `refresh_access_token(self, refresh_token: str) -> tuple[str, str]` - Line 357
- `revoke_token(self, token: str) -> bool` - Line 408
- `is_token_blacklisted(self, token: str) -> bool` - Line 451
- `get_security_summary(self) -> dict[str, Any]` - Line 485
- `cleanup_expired_blacklist(self) -> int` - Line 497

### Implementation: `WebAnalyticsService` ✅

**Inherits**: BaseService
**Purpose**: Web interface service for analytics operations
**Status**: Complete

**Implemented Methods:**
- `async get_portfolio_metrics(self) -> dict[str, Any]` - Line 80
- `async get_portfolio_composition(self) -> dict[str, Any]` - Line 104
- `async get_correlation_matrix(self) -> Any` - Line 118
- `async export_portfolio_data(self, format: str = 'json', include_metadata: bool = True) -> str` - Line 130
- `async get_risk_metrics(self) -> dict[str, Any]` - Line 148
- `async calculate_var(self, confidence_level: float, time_horizon: int, method: str) -> dict[str, Any]` - Line 171
- `async run_stress_test(self, scenario_name: str, scenario_params: dict[str, Any]) -> dict[str, Any]` - Line 194
- `async get_risk_exposure(self) -> dict[str, Any]` - Line 212
- `async get_strategy_metrics(self, strategy_id: str) -> dict[str, Any] | None` - Line 237
- `async get_strategy_performance(self, strategy_id: str, days: int) -> dict[str, Any]` - Line 268
- `async compare_strategies(self, strategy_ids: list[str]) -> dict[str, Any]` - Line 288
- `async get_operational_metrics(self) -> dict[str, Any]` - Line 311
- `async get_system_errors(self, hours: int) -> list[dict[str, Any]]` - Line 330
- `async get_operational_events(self, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]` - Line 343
- `async generate_report(self, ...) -> dict[str, Any]` - Line 359
- `async get_report(self, report_id: str) -> dict[str, Any] | None` - Line 390
- `async list_reports(self, user_id: str, limit: int) -> list[dict[str, Any]]` - Line 395
- `async schedule_report(self, report_type: str, schedule: str, recipients: list[str], created_by: str) -> dict[str, Any]` - Line 400
- `async get_active_alerts(self) -> list[dict[str, Any]]` - Line 419
- `async acknowledge_alert(self, alert_id: str, acknowledged_by: str, notes: str | None = None) -> bool` - Line 444
- `async get_alert_history(self, days: int, severity: str | None = None) -> list[dict[str, Any]]` - Line 459
- `async configure_alerts(self, config: dict[str, Any], configured_by: str) -> dict[str, Any]` - Line 466

### Implementation: `WebAuthService` ✅

**Inherits**: BaseService
**Purpose**: Service handling authentication business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async get_user_by_username(self, username: str) -> UserInDB | None` - Line 33
- `async authenticate_user(self, username: str, password: str) -> User | None` - Line 62
- `async create_user(self, ...) -> User` - Line 89
- `async get_auth_summary(self) -> dict[str, Any]` - Line 139
- `health_check(self) -> dict[str, Any]` - Line 221
- `get_user_roles(self, current_user: Any) -> list[str]` - Line 230
- `check_permission(self, current_user: Any, required_roles: list[str]) -> bool` - Line 245
- `require_permission(self, current_user: Any, required_roles: list[str]) -> None` - Line 250
- `require_admin(self, current_user: Any) -> None` - Line 258
- `require_trading_permission(self, current_user: Any) -> None` - Line 262
- `require_risk_manager_permission(self, current_user: Any) -> None` - Line 266
- `require_admin_or_developer_permission(self, current_user: Any) -> None` - Line 270
- `require_management_permission(self, current_user: Any) -> None` - Line 274
- `require_treasurer_permission(self, current_user: Any) -> None` - Line 278
- `require_operator_permission(self, current_user: Any) -> None` - Line 282
- `get_service_info(self) -> dict[str, Any]` - Line 286

### Implementation: `WebBotService` ✅

**Inherits**: BaseComponent
**Purpose**: Service handling bot management business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 28
- `async cleanup(self) -> None` - Line 32
- `async validate_bot_configuration(self, config_data: dict[str, Any]) -> dict[str, Any]` - Line 36
- `async format_bot_response(self, bot_data: dict[str, Any]) -> dict[str, Any]` - Line 95
- `async get_formatted_bot_list(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 121
- `async calculate_bot_metrics(self, bot_id: str) -> dict[str, Any]` - Line 200
- `async validate_bot_operation(self, bot_id: str, operation: str) -> dict[str, Any]` - Line 279
- `async create_bot_configuration(self, request_data: dict[str, Any], user_id: str) -> BotConfiguration` - Line 320
- `async update_bot_configuration(self, bot_id: str, update_data: dict[str, Any], user_id: str) -> dict[str, Any]` - Line 357
- `health_check(self) -> dict[str, Any]` - Line 469
- `get_service_info(self) -> dict[str, Any]` - Line 478
- `async create_bot_through_service(self, bot_config) -> str` - Line 493
- `async get_bot_status_through_service(self, bot_id: str) -> dict[str, Any]` - Line 523
- `async start_bot_through_service(self, bot_id: str) -> bool` - Line 544
- `async stop_bot_through_service(self, bot_id: str) -> bool` - Line 558
- `async delete_bot_through_service(self, bot_id: str, force: bool = False) -> bool` - Line 572
- `async list_bots_through_service(self) -> list[dict[str, Any]]` - Line 593
- `get_controller_health_check(self) -> dict[str, Any]` - Line 619
- `async start_bot_with_execution_integration(self, bot_id: str) -> bool` - Line 651
- `async stop_bot_with_execution_integration(self, bot_id: str) -> bool` - Line 679
- `async pause_bot_through_service(self, bot_id: str) -> bool` - Line 699
- `async resume_bot_through_service(self, bot_id: str) -> bool` - Line 713

### Implementation: `WebCapitalService` ✅

**Inherits**: BaseService
**Purpose**: Web interface service for capital management operations
**Status**: Complete

**Implemented Methods:**
- `async allocate_capital(self, ...) -> dict[str, Any]` - Line 57
- `async release_capital(self, ...) -> bool` - Line 112
- `async update_utilization(self, ...) -> bool` - Line 158
- `async get_allocations(self, ...) -> list[dict[str, Any]]` - Line 197
- `async get_strategy_allocation(self, strategy_id: str, exchange: str | None = None) -> dict[str, Any] | None` - Line 228
- `async get_capital_metrics(self) -> dict[str, Any]` - Line 243
- `async get_utilization_breakdown(self, by: str = 'strategy') -> dict[str, Any]` - Line 263
- `async get_available_capital(self, strategy_id: str | None = None, exchange: str | None = None) -> dict[str, Any]` - Line 323
- `async get_capital_exposure(self) -> dict[str, Any]` - Line 347
- `async get_currency_exposure(self) -> dict[str, Any]` - Line 389
- `async create_currency_hedge(self, ...) -> dict[str, Any]` - Line 439
- `async get_currency_rates(self, base_currency: str = 'USD') -> dict[str, float]` - Line 476
- `async get_fund_flows(self, days: int = 30, flow_type: str | None = None) -> list[dict[str, Any]]` - Line 498
- `async record_fund_flow(self, ...) -> dict[str, Any]` - Line 531
- `async generate_fund_flow_report(self, start_date: datetime | None, end_date: datetime | None, format: str) -> dict[str, Any]` - Line 581
- `async get_allocation_limits(self, limit_type: str | None = None) -> list[dict[str, Any]]` - Line 615
- `async set_allocation_limits(self, **kwargs) -> dict[str, Any]` - Line 619
- `async get_capital_limits(self, limit_type: str | None = None) -> list[dict[str, Any]]` - Line 630
- `async update_capital_limits(self, ...) -> bool` - Line 661
- `async get_limit_breaches(self, hours: int = 24, severity: str | None = None) -> list[dict[str, Any]]` - Line 681
- `async convert_currency(self, from_currency: str, to_currency: str, amount: Decimal) -> dict[str, Any]` - Line 734
- `async rebalance_portfolio(self, target_allocations: dict[str, Any] | None = None, dry_run: bool = False) -> dict[str, Any]` - Line 762
- `async calculate_optimal_allocation(self, risk_tolerance: str, optimization_method: str) -> dict[str, Any]` - Line 785
- `async get_capital_efficiency(self) -> dict[str, Any]` - Line 810
- `async reserve_capital(self, amount: Decimal, currency: str, purpose: str, duration_minutes: int = 60) -> dict[str, Any]` - Line 837
- `async get_reserved_capital(self) -> dict[str, Any]` - Line 862

### Implementation: `WebDataService` ✅

**Inherits**: BaseService
**Purpose**: Web interface service for data management operations
**Status**: Complete

**Implemented Methods:**
- `async get_pipeline_status(self, pipeline_id: str | None = None) -> Any` - Line 66
- `async control_pipeline(self, ...) -> dict[str, Any]` - Line 79
- `async get_pipeline_metrics(self, hours: int) -> dict[str, Any]` - Line 115
- `async get_data_quality_metrics(self, data_source: str | None = None, symbol: str | None = None) -> dict[str, Any]` - Line 138
- `async get_validation_report(self, days: int) -> dict[str, Any]` - Line 165
- `async validate_data(self, ...) -> dict[str, Any]` - Line 194
- `async get_data_anomalies(self, hours: int = 24, severity: str | None = None) -> list[dict[str, Any]]` - Line 231
- `async list_features(self, category: str | None = None, active_only: bool = True) -> list[dict[str, Any]]` - Line 270
- `async get_feature_details(self, feature_id: str) -> dict[str, Any] | None` - Line 314
- `async compute_features(self, ...) -> dict[str, Any]` - Line 339
- `async get_feature_metadata(self) -> dict[str, Any]` - Line 376
- `async list_data_sources(self, source_type: str | None = None, enabled_only: bool = True) -> list[dict[str, Any]]` - Line 397
- `async configure_data_source(self, ...) -> str` - Line 449
- `async update_data_source(self, source_id: str, config: dict[str, Any], updated_by: str) -> bool` - Line 479
- `async delete_data_source(self, source_id: str, deleted_by: str) -> bool` - Line 495
- `async get_data_health(self) -> dict[str, Any]` - Line 511
- `async get_data_latency(self, source: str | None = None, hours: int = 1) -> dict[str, Any]` - Line 533
- `async get_data_throughput(self, source: str | None = None, hours: int = 1) -> dict[str, Any]` - Line 550
- `async clear_data_cache(self, cache_type: str | None = None) -> int` - Line 570
- `async get_cache_statistics(self) -> dict[str, Any]` - Line 600

### Implementation: `WebExchangeService` ✅

**Inherits**: BaseService
**Purpose**: Web interface service for exchange management operations
**Status**: Complete

**Implemented Methods:**
- `async get_connections(self) -> list[dict[str, Any]]` - Line 61
- `async connect_exchange(self, ...) -> dict[str, Any]` - Line 97
- `async disconnect_exchange(self, exchange: str, disconnected_by: str) -> bool` - Line 152
- `async get_exchange_status(self, exchange: str) -> dict[str, Any] | None` - Line 170
- `async get_exchange_config(self, exchange: str) -> dict[str, Any] | None` - Line 204
- `async validate_exchange_config(self, exchange: str, config: dict[str, Any]) -> dict[str, Any]` - Line 229
- `async update_exchange_config(self, exchange: str, config: dict[str, Any], updated_by: str) -> bool` - Line 252
- `async get_exchange_symbols(self, exchange: str, active_only: bool = True) -> list[str]` - Line 268
- `async get_exchange_fees(self, exchange: str, symbol: str | None = None) -> dict[str, Any]` - Line 293
- `async get_rate_limits(self, exchange: str) -> dict[str, Any]` - Line 316
- `async get_rate_usage(self, exchange: str) -> dict[str, Any]` - Line 332
- `async update_rate_config(self, ...) -> bool` - Line 349
- `async get_all_exchanges_health(self) -> dict[str, Any]` - Line 369
- `async get_exchange_health(self, exchange: str) -> dict[str, Any] | None` - Line 393
- `async get_exchange_latency(self, exchange: str, hours: int) -> dict[str, Any]` - Line 416
- `async get_exchange_errors(self, exchange: str, hours: int, error_type: str | None = None) -> list[dict[str, Any]]` - Line 433
- `async get_orderbook(self, exchange: str, symbol: str, limit: int) -> dict[str, Any] | None` - Line 466
- `async get_ticker(self, exchange: str, symbol: str) -> dict[str, Any] | None` - Line 497
- `async get_exchange_balance(self, exchange: str, user_id: str) -> dict[str, dict[str, Decimal]]` - Line 527
- `async subscribe_websocket(self, exchange: str, channel: str, symbols: list[str], subscriber: str) -> str` - Line 555
- `async unsubscribe_websocket(self, exchange: str, subscription_id: str, subscriber: str) -> bool` - Line 578

### Implementation: `WebMonitoringService` ✅

**Inherits**: BaseComponent
**Purpose**: Service handling monitoring business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 23
- `async cleanup(self) -> None` - Line 27
- `async get_system_health_summary(self) -> dict[str, Any]` - Line 31
- `async get_performance_metrics(self, component: str | None = None) -> dict[str, Any]` - Line 91
- `async get_error_summary(self, time_range: str = '24h') -> dict[str, Any]` - Line 168
- `async get_alert_dashboard_data(self) -> dict[str, Any]` - Line 244
- `health_check(self) -> dict[str, Any]` - Line 505
- `get_service_info(self) -> dict[str, Any]` - Line 514

### Implementation: `WebPortfolioService` ✅

**Inherits**: BaseComponent
**Purpose**: Service handling portfolio business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 27
- `async cleanup(self) -> None` - Line 31
- `async get_portfolio_summary_data(self) -> dict[str, Any]` - Line 35
- `async calculate_pnl_periods(self, total_pnl: Decimal, total_trades: int, win_rate: float) -> dict[str, dict[str, Any]]` - Line 86
- `async get_processed_positions(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]` - Line 152
- `async calculate_pnl_metrics(self, period: str) -> dict[str, Any]` - Line 231
- `generate_mock_balances(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]` - Line 308
- `calculate_asset_allocation(self) -> list[dict[str, Any]]` - Line 358
- `generate_performance_chart_data(self, period: str, resolution: str) -> dict[str, Any]` - Line 392

### Implementation: `WebRiskService` ✅

**Inherits**: BaseComponent
**Purpose**: Service handling risk management business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 37
- `async cleanup(self) -> None` - Line 41
- `async get_risk_dashboard_data(self) -> dict[str, Any]` - Line 45
- `async validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]` - Line 159
- `async calculate_position_risk(self, symbol: str, quantity: Decimal, price: Decimal) -> dict[str, Any]` - Line 230
- `async get_portfolio_risk_breakdown(self) -> dict[str, Any]` - Line 388
- `health_check(self) -> dict[str, Any]` - Line 710
- `generate_mock_risk_alerts(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 719
- `generate_mock_position_risks(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 796
- `generate_mock_correlation_matrix(self, symbols: list[str], period: str) -> dict[str, Any]` - Line 870
- `generate_mock_stress_test_results(self, test_request: dict[str, Any]) -> dict[str, Any]` - Line 919
- `async validate_risk_parameters_v2(self, parameters: dict[str, Any]) -> dict[str, Any]` - Line 982
- `async get_current_risk_limits(self) -> dict[str, Any]` - Line 1066
- `get_service_info(self) -> dict[str, Any]` - Line 1092

### Implementation: `WebStrategyService` ✅

**Inherits**: BaseComponent
**Purpose**: Service handling strategy business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 27
- `async cleanup(self) -> None` - Line 31
- `async get_formatted_strategies(self) -> list[dict[str, Any]]` - Line 35
- `async validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> dict[str, Any]` - Line 154
- `async get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]` - Line 203
- `async format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]` - Line 303
- `async health_check(self) -> 'HealthCheckResult'` - Line 343
- `get_service_info(self) -> dict[str, Any]` - Line 352
- `async get_strategy_config_through_service(self, strategy_name: str) -> dict[str, Any]` - Line 366
- `async validate_strategy_config_through_service(self, strategy_name: str, parameters: dict[str, Any]) -> bool` - Line 404

### Implementation: `WebTradingService` ✅

**Inherits**: BaseComponent
**Purpose**: Service handling trading business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 28
- `async cleanup(self) -> None` - Line 32
- `async place_order_through_service(self, ...) -> dict[str, Any]` - Line 36
- `async cancel_order_through_service(self, order_id: str) -> bool` - Line 89
- `async get_service_health(self) -> dict[str, Any]` - Line 103
- `async get_order_details(self, order_id: str, exchange: str) -> dict[str, Any]` - Line 124
- `async validate_order_request(self, ...) -> dict[str, Any]` - Line 170
- `async format_order_response(self, order_result: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]` - Line 259
- `async get_formatted_orders(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 294
- `async get_formatted_trades(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 369
- `async get_market_data_with_context(self, symbol: str, exchange: str = 'binance') -> dict[str, Any]` - Line 432
- `async generate_order_book_data(self, symbol: str, exchange: str, depth: int) -> dict[str, Any]` - Line 474
- `health_check(self) -> dict[str, Any]` - Line 513
- `get_service_info(self) -> dict[str, Any]` - Line 522

### Implementation: `TradingNamespace` ✅

**Inherits**: AsyncNamespace
**Purpose**: Main namespace for trading-related Socket
**Status**: Complete

**Implemented Methods:**
- `async emit_standardized(self, ...) -> None` - Line 67
- `async on_connect(self, ...)` - Line 103
- `async on_disconnect(self, sid: str)` - Line 231
- `async on_authenticate(self, sid: str, data: dict[str, Any])` - Line 280
- `async on_subscribe(self, sid: str, data: dict[str, Any])` - Line 302
- `async on_unsubscribe(self, sid: str, data: dict[str, Any])` - Line 324
- `async on_ping(self, sid: str, data: dict[str, Any] | None = None)` - Line 338
- `async on_execute_order(self, sid: str, data: dict[str, Any])` - Line 351
- `async on_get_portfolio(self, sid: str, data: dict[str, Any])` - Line 442

### Implementation: `SocketIOManager` ✅

**Inherits**: BaseComponent
**Purpose**: Manager for Socket
**Status**: Complete

**Implemented Methods:**
- `create_server(self, cors_allowed_origins: list[str] | None = None) -> AsyncServer` - Line 541
- `create_app(self)` - Line 570
- `async start_background_tasks(self)` - Line 578
- `async stop_background_tasks(self)` - Line 599
- `async emit_to_user(self, user_id: str, event: str, data: Any)` - Line 750
- `async broadcast(self, event: str, data: Any, room: str | None = None)` - Line 757

### Implementation: `VersioningMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to handle API versioning
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 29

### Implementation: `VersionRoutingMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to handle version-specific routing
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 178

### Implementation: `VersionStatus` ✅

**Inherits**: Enum
**Purpose**: API version status
**Status**: Complete

### Implementation: `APIVersion` ✅

**Purpose**: API version configuration
**Status**: Complete

**Implemented Methods:**
- `is_compatible_with(self, other: 'APIVersion') -> bool` - Line 60
- `is_deprecated(self) -> bool` - Line 73
- `is_sunset(self) -> bool` - Line 77

### Implementation: `VersionManager` ✅

**Inherits**: BaseComponent
**Purpose**: Manager for API versioning and compatibility
**Status**: Complete

**Implemented Methods:**
- `register_version(self, version: APIVersion) -> None` - Line 156
- `parse_version(self, version_string: str) -> APIVersion | None` - Line 161
- `get_version(self, version: str) -> APIVersion | None` - Line 173
- `get_latest_version(self) -> APIVersion | None` - Line 177
- `get_default_version(self) -> APIVersion | None` - Line 183
- `list_versions(self, include_deprecated: bool = True) -> list[APIVersion]` - Line 189
- `resolve_version(self, requested_version: str | None = None) -> APIVersion` - Line 196
- `check_feature_availability(self, version: str, feature: str) -> bool` - Line 239
- `get_deprecation_info(self, version: str) -> dict[str, Any] | None` - Line 246
- `deprecate_version(self, version: str, sunset_date: datetime | None = None) -> bool` - Line 263
- `sunset_version(self, version: str) -> bool` - Line 277
- `get_version_migration_guide(self, from_version: str, to_version: str) -> dict[str, Any]` - Line 289

### Implementation: `BotStatusManager` ✅

**Purpose**: Manages WebSocket connections for bot status updates
**Status**: Complete

**Implemented Methods:**
- `async connect(self, websocket: WebSocket, user_id: str)` - Line 32
- `disconnect(self, user_id: str)` - Line 47
- `subscribe_to_bot(self, user_id: str, bot_id: str)` - Line 64
- `unsubscribe_from_bot(self, user_id: str, bot_id: str)` - Line 75
- `subscribe_to_all_bots(self, user_id: str)` - Line 87
- `async send_to_user(self, user_id: str, message: dict)` - Line 96
- `async broadcast_to_bot_subscribers(self, bot_id: str, message: dict)` - Line 114

### Implementation: `BotStatusMessage` ✅

**Inherits**: BaseModel
**Purpose**: Base model for bot status messages
**Status**: Complete

### Implementation: `BotSubscriptionMessage` ✅

**Inherits**: BaseModel
**Purpose**: Model for bot subscription messages
**Status**: Complete

### Implementation: `ConnectionManager` ✅

**Purpose**: Manages WebSocket connections for market data streaming
**Status**: Complete

**Implemented Methods:**
- `async connect(self, websocket: WebSocket, user_id: str)` - Line 31
- `disconnect(self, user_id: str)` - Line 46
- `subscribe_to_symbol(self, user_id: str, symbol: str)` - Line 63
- `unsubscribe_from_symbol(self, user_id: str, symbol: str)` - Line 74
- `async send_to_user(self, user_id: str, message: dict)` - Line 86
- `async broadcast_to_symbol_subscribers(self, symbol: str, message: dict)` - Line 105

### Implementation: `MarketDataMessage` ✅

**Inherits**: BaseModel
**Purpose**: Base model for market data messages
**Status**: Complete

### Implementation: `SubscriptionMessage` ✅

**Inherits**: BaseModel
**Purpose**: Model for subscription messages
**Status**: Complete

### Implementation: `PortfolioManager` ✅

**Purpose**: Manages WebSocket connections for portfolio updates
**Status**: Complete

**Implemented Methods:**
- `async connect(self, websocket: WebSocket, user_id: str)` - Line 35
- `disconnect(self, user_id: str)` - Line 59
- `subscribe_to_updates(self, user_id: str, update_types: list[str])` - Line 68
- `unsubscribe_from_updates(self, user_id: str, update_types: list[str])` - Line 76
- `async send_to_user(self, user_id: str, message: dict)` - Line 87
- `async broadcast_to_all(self, message: dict, update_type: str)` - Line 106

### Implementation: `PortfolioMessage` ✅

**Inherits**: BaseModel
**Purpose**: Base model for portfolio messages
**Status**: Complete

### Implementation: `PortfolioSubscriptionMessage` ✅

**Inherits**: BaseModel
**Purpose**: Model for portfolio subscription messages
**Status**: Complete

### Implementation: `ChannelType` ✅

**Inherits**: Enum
**Purpose**: WebSocket channel types
**Status**: Complete

### Implementation: `SubscriptionLevel` ✅

**Inherits**: Enum
**Purpose**: Subscription permission levels
**Status**: Complete

### Implementation: `WebSocketEventHandler` ✅

**Purpose**: Base class for WebSocket event handlers
**Status**: Complete

**Implemented Methods:**
- `async subscribe(self, session_id: str) -> bool` - Line 61
- `async unsubscribe(self, session_id: str) -> bool` - Line 66
- `async start(self) -> None` - Line 71
- `async stop(self) -> None` - Line 75
- `async emit_to_channel(self, sio: AsyncServer, event: str, data: Any) -> None` - Line 99

### Implementation: `MarketDataHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for market data WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 130

### Implementation: `BotStatusHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for bot status WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 177

### Implementation: `PortfolioHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for portfolio WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 226

### Implementation: `TradesHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for trade execution WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 276

### Implementation: `OrdersHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for order status WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 327

### Implementation: `AlertsHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for alerts and notifications WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 380

### Implementation: `LogsHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for system logs WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 432

### Implementation: `RiskMetricsHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for risk metrics WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 485

### Implementation: `UnifiedWebSocketNamespace` ✅

**Inherits**: AsyncNamespace
**Purpose**: Unified namespace for all WebSocket communications
**Status**: Complete

**Implemented Methods:**
- `async on_connect(self, ...)` - Line 578
- `async on_disconnect(self, sid: str)` - Line 615
- `async on_authenticate(self, sid: str, data: dict[str, Any])` - Line 644
- `async on_subscribe(self, sid: str, data: dict[str, Any])` - Line 710
- `async on_unsubscribe(self, sid: str, data: dict[str, Any])` - Line 742
- `async on_ping(self, sid: str, data: dict[str, Any] | None = None)` - Line 824
- `async start_handlers(self)` - Line 837
- `async stop_handlers(self)` - Line 842

### Implementation: `UnifiedWebSocketManager` ✅

**Inherits**: BaseComponent
**Purpose**: Unified manager for all WebSocket communications
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector)` - Line 871
- `create_server(self, cors_allowed_origins: list[str] | None = None) -> AsyncServer` - Line 881
- `async start(self)` - Line 911
- `async stop(self)` - Line 934
- `get_connection_stats(self) -> dict[str, Any]` - Line 953

## COMPLETE API REFERENCE

### File: analytics.py

**Key Imports:**
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.decorators import monitored`
- `from src.web_interface.auth.middleware import get_current_user`

#### Class: `PortfolioMetricsResponse`

**Inherits**: BaseModel
**Purpose**: Response model for portfolio metrics

```python
class PortfolioMetricsResponse(BaseModel):
```

#### Class: `RiskMetricsResponse`

**Inherits**: BaseModel
**Purpose**: Response model for risk metrics

```python
class RiskMetricsResponse(BaseModel):
```

#### Class: `StrategyMetricsResponse`

**Inherits**: BaseModel
**Purpose**: Response model for strategy metrics

```python
class StrategyMetricsResponse(BaseModel):
```

#### Class: `VaRRequest`

**Inherits**: BaseModel
**Purpose**: Request model for VaR calculation

```python
class VaRRequest(BaseModel):
```

#### Class: `StressTestRequest`

**Inherits**: BaseModel
**Purpose**: Request model for stress testing

```python
class StressTestRequest(BaseModel):
```

#### Class: `GenerateReportRequest`

**Inherits**: BaseModel
**Purpose**: Request model for report generation

```python
class GenerateReportRequest(BaseModel):
```

#### Class: `AlertAcknowledgeRequest`

**Inherits**: BaseModel
**Purpose**: Request model for alert acknowledgment

```python
class AlertAcknowledgeRequest(BaseModel):
```

#### Functions:

```python
async def get_portfolio_metrics(current_user: dict = Any, web_analytics_service = Any)  # Line 108
async def get_portfolio_composition(current_user: dict = Any, web_analytics_service = Any)  # Line 133
async def get_correlation_matrix(...)  # Line 148
async def export_portfolio_data(...)  # Line 167
async def get_risk_metrics(current_user: dict = Any, web_analytics_service = Any)  # Line 187
async def calculate_var(request: VaRRequest, current_user: dict = Any, web_analytics_service = Any)  # Line 202
async def run_stress_test(...)  # Line 222
async def get_risk_exposure(current_user: dict = Any, web_analytics_service = Any)  # Line 258
async def get_single_strategy_metrics(strategy_id: str, current_user: dict = Any, web_analytics_service = Any)  # Line 274
async def get_strategy_metrics(...)  # Line 294
async def get_strategy_performance(...)  # Line 313
async def compare_strategies(...)  # Line 332
async def get_operational_metrics(current_user: dict = Any, web_analytics_service = Any)  # Line 356
async def get_system_errors(...)  # Line 371
async def get_operational_events(...)  # Line 399
async def generate_report(...)  # Line 419
async def get_report(report_id: str, current_user: dict = Any, web_analytics_service = Any)  # Line 441
async def list_reports(limit: int = Any, current_user: dict = Any, web_analytics_service = Any)  # Line 461
async def schedule_report(...)  # Line 479
async def get_active_alerts(current_user: dict = Any, web_analytics_service = Any)  # Line 515
async def get_alerts(...)  # Line 530
async def acknowledge_alert(...)  # Line 550
async def get_alert_history(...)  # Line 575
async def configure_alerts(...)  # Line 592
```

### File: auth.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.web_interface.security.auth import LoginRequest`
- `from src.web_interface.security.auth import Token`
- `from src.web_interface.security.auth import User`
- `from src.web_interface.security.auth import authenticate_user`

#### Class: `RefreshTokenRequest`

**Inherits**: BaseModel
**Purpose**: Refresh token request model

```python
class RefreshTokenRequest(BaseModel):
```

#### Class: `CreateUserRequest`

**Inherits**: BaseModel
**Purpose**: Create user request model

```python
class CreateUserRequest(BaseModel):
```

#### Class: `ChangePasswordRequest`

**Inherits**: BaseModel
**Purpose**: Change password request model

```python
class ChangePasswordRequest(BaseModel):
```

#### Class: `AuthResponse`

**Inherits**: BaseModel
**Purpose**: Authentication response model

```python
class AuthResponse(BaseModel):
```

#### Functions:

```python
async def login(login_request: LoginRequest)  # Line 63
async def refresh_token(refresh_request: RefreshTokenRequest)  # Line 113
async def logout(current_user: User = Any)  # Line 157
async def get_current_user_info(current_user: User = Any)  # Line 183
async def change_password(change_request: ChangePasswordRequest, current_user: User = Any)  # Line 197
async def create_new_user(user_request: CreateUserRequest, admin_user: User = Any)  # Line 253
async def list_users(admin_user: User = Any)  # Line 306
async def get_demo_credentials()  # Line 342
async def get_auth_status()  # Line 380
```

### File: bot_management.py

**Key Imports:**
- `from src.core.caching import CacheKeys`
- `from src.core.caching import cached`
- `from src.core.events import BotEvent`
- `from src.core.events import BotEventType`
- `from src.core.events import get_event_publisher`

#### Class: `CreateBotRequest`

**Inherits**: BaseModel
**Purpose**: Request model for creating a new bot

```python
class CreateBotRequest(BaseModel):
```

#### Class: `UpdateBotRequest`

**Inherits**: BaseModel
**Purpose**: Request model for updating bot configuration

```python
class UpdateBotRequest(BaseModel):
```

#### Class: `BotResponse`

**Inherits**: BaseModel
**Purpose**: Response model for bot information

```python
class BotResponse(BaseModel):
```

#### Class: `BotSummaryResponse`

**Inherits**: BaseModel
**Purpose**: Response model for bot summary

```python
class BotSummaryResponse(BaseModel):
```

#### Class: `BotListResponse`

**Inherits**: BaseModel
**Purpose**: Response model for bot listing

```python
class BotListResponse(BaseModel):
```

#### Functions:

```python
def get_bot_service()  # Line 35
def get_web_bot_service_instance()  # Line 45
def get_execution_service() -> ExecutionServiceInterface | None  # Line 50
def set_bot_service(service)  # Line 65
def set_bot_orchestrator(orchestrator)  # Line 70
async def create_bot(bot_request: CreateBotRequest, current_user: User = Any)  # Line 145
async def list_bots(status_filter: BotStatus | None = Any, current_user: User = Any)  # Line 270
async def get_bot(bot_id: str, current_user: User = Any)  # Line 335
async def update_bot(bot_id: str, update_request: UpdateBotRequest, current_user: User = Any)  # Line 397
async def start_bot(bot_id: str, current_user: User = Any)  # Line 486
async def stop_bot(bot_id: str, current_user: User = Any)  # Line 568
async def pause_bot(bot_id: str, current_user: User = Any)  # Line 647
async def resume_bot(bot_id: str, current_user: User = Any)  # Line 712
async def delete_bot(bot_id: str, force: bool = Any, current_user: User = Any)  # Line 777
async def get_orchestrator_status(current_user: User = Any)  # Line 846
async def get_status_alias(current_user: User = Any)  # Line 895
async def get_list_alias(...)  # Line 901
async def create_alias(request: dict[str, Any], trading_user: User = Any)  # Line 912
async def get_config_alias(current_user: User = Any)  # Line 921
async def get_logs_alias(current_user: User = Any)  # Line 948
```

### File: capital.py

**Key Imports:**
- `from src.core.exceptions import CapitalAllocationError`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.decorators import monitored`

#### Class: `CapitalAllocationRequest`

**Inherits**: BaseModel
**Purpose**: Request model for capital allocation

```python
class CapitalAllocationRequest(BaseModel):
```

#### Class: `CapitalReleaseRequest`

**Inherits**: BaseModel
**Purpose**: Request model for capital release

```python
class CapitalReleaseRequest(BaseModel):
```

#### Class: `CapitalUtilizationUpdate`

**Inherits**: BaseModel
**Purpose**: Request model for utilization update

```python
class CapitalUtilizationUpdate(BaseModel):
```

#### Class: `CurrencyHedgeRequest`

**Inherits**: BaseModel
**Purpose**: Request model for currency hedging

```python
class CurrencyHedgeRequest(BaseModel):
```

#### Class: `FundFlowRequest`

**Inherits**: BaseModel
**Purpose**: Request model for recording fund flows

```python
class FundFlowRequest(BaseModel):
```

#### Class: `CapitalLimitsUpdate`

**Inherits**: BaseModel
**Purpose**: Request model for updating capital limits

```python
class CapitalLimitsUpdate(BaseModel):
```

#### Class: `CapitalAllocationResponse`

**Inherits**: BaseModel
**Purpose**: Response model for capital allocation

```python
class CapitalAllocationResponse(BaseModel):
```

#### Class: `CapitalMetricsResponse`

**Inherits**: BaseModel
**Purpose**: Response model for capital metrics

```python
class CapitalMetricsResponse(BaseModel):
```

#### Functions:

```python
async def allocate_capital(...)  # Line 139
async def release_capital(...)  # Line 182
async def update_utilization(...)  # Line 211
async def get_allocations(...)  # Line 233
async def get_strategy_allocation(...)  # Line 250
async def get_capital_metrics(current_user: dict = Any, web_capital_service = Any)  # Line 277
async def get_utilization(by: str = Any, current_user: dict = Any, web_capital_service = Any)  # Line 293
async def get_available_capital(...)  # Line 310
async def get_capital_exposure(current_user: dict = Any, web_capital_service = Any)  # Line 335
async def get_currency_exposure(current_user: dict = Any, web_capital_service = Any)  # Line 352
async def create_currency_hedge(...)  # Line 368
async def get_exchange_rates(base_currency: str = Any, current_user: dict = Any, web_capital_service = Any)  # Line 399
async def convert_currency(request: dict, current_user: dict = Any, web_capital_service = Any)  # Line 416
async def get_fund_flows(...)  # Line 438
async def record_fund_flow(...)  # Line 462
async def get_fund_flow_report(...)  # Line 495
async def get_allocation_limits(current_user: dict = Any, web_capital_service = Any)  # Line 518
async def set_allocation_limits(...)  # Line 534
async def update_capital_limits(...)  # Line 557
async def get_limit_breaches(...)  # Line 592
async def rebalance_portfolio(...)  # Line 617
async def calculate_optimal_allocation(...)  # Line 644
async def get_capital_efficiency(current_user: dict = Any, web_capital_service = Any)  # Line 664
async def reserve_capital(request: dict, current_user: dict = Any, web_capital_service = Any)  # Line 681
async def get_reserved_capital(current_user: dict = Any, web_capital_service = Any)  # Line 703
```

### File: data.py

**Key Imports:**
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.decorators import monitored`
- `from src.web_interface.auth.middleware import get_current_user`

#### Class: `PipelineControlRequest`

**Inherits**: BaseModel
**Purpose**: Request model for pipeline control

```python
class PipelineControlRequest(BaseModel):
```

#### Class: `DataValidationRequest`

**Inherits**: BaseModel
**Purpose**: Request model for data validation

```python
class DataValidationRequest(BaseModel):
```

#### Class: `FeatureComputeRequest`

**Inherits**: BaseModel
**Purpose**: Request model for feature computation

```python
class FeatureComputeRequest(BaseModel):
```

#### Class: `DataSourceConfigRequest`

**Inherits**: BaseModel
**Purpose**: Request model for data source configuration

```python
class DataSourceConfigRequest(BaseModel):
```

#### Class: `PipelineStatusResponse`

**Inherits**: BaseModel
**Purpose**: Response model for pipeline status

```python
class PipelineStatusResponse(BaseModel):
```

#### Class: `DataQualityMetricsResponse`

**Inherits**: BaseModel
**Purpose**: Response model for data quality metrics

```python
class DataQualityMetricsResponse(BaseModel):
```

#### Functions:

```python
async def get_pipeline_status(pipeline_id: str | None = None, current_user: dict = Any, web_data_service = Any)  # Line 99
async def control_pipeline(...)  # Line 122
async def get_pipeline_metrics(hours: int = Any, current_user: dict = Any, web_data_service = Any)  # Line 160
async def get_data_quality_metrics(...)  # Line 178
async def get_validation_report(days: int = Any, current_user: dict = Any, web_data_service = Any)  # Line 198
async def validate_data(request: DataValidationRequest, current_user: dict = Any, web_data_service = Any)  # Line 215
async def get_data_anomalies(...)  # Line 246
async def list_features(...)  # Line 271
async def get_feature_details(feature_id: str, current_user: dict = Any, web_data_service = Any)  # Line 290
async def compute_features(request: FeatureComputeRequest, current_user: dict = Any, web_data_service = Any)  # Line 313
async def get_feature_metadata(current_user: dict = Any, web_data_service = Any)  # Line 345
async def list_data_sources(...)  # Line 362
async def configure_data_source(...)  # Line 383
async def update_data_source(...)  # Line 426
async def delete_data_source(...)  # Line 461
async def get_data_health(current_user: dict = Any, web_data_service = Any)  # Line 496
async def get_data_latency(...)  # Line 519
async def get_data_throughput(...)  # Line 542
async def clear_data_cache(...)  # Line 566
async def get_cache_statistics(current_user: dict = Any, web_data_service = Any)  # Line 599
```

### File: exchanges.py

**Key Imports:**
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.decorators import monitored`
- `from src.web_interface.auth.middleware import get_current_user`

#### Class: `ExchangeConnectionRequest`

**Inherits**: BaseModel
**Purpose**: Request model for exchange connection

```python
class ExchangeConnectionRequest(BaseModel):
```

#### Class: `ExchangeConfigRequest`

**Inherits**: BaseModel
**Purpose**: Request model for exchange configuration

```python
class ExchangeConfigRequest(BaseModel):
```

#### Class: `RateLimitConfigRequest`

**Inherits**: BaseModel
**Purpose**: Request model for rate limit configuration

```python
class RateLimitConfigRequest(BaseModel):
```

#### Class: `ExchangeStatusResponse`

**Inherits**: BaseModel
**Purpose**: Response model for exchange status

```python
class ExchangeStatusResponse(BaseModel):
```

#### Class: `ExchangeHealthResponse`

**Inherits**: BaseModel
**Purpose**: Response model for exchange health

```python
class ExchangeHealthResponse(BaseModel):
```

#### Functions:

```python
async def get_exchange_connections(current_user: dict = Any, web_exchange_service = Any)  # Line 81
async def connect_exchange(...)  # Line 97
async def disconnect_exchange(...)  # Line 151
async def get_exchange_status(exchange: str, current_user: dict = Any, web_exchange_service = Any)  # Line 185
async def get_exchange_config(exchange: str, current_user: dict = Any, web_exchange_service = Any)  # Line 209
async def update_exchange_config(...)  # Line 238
async def get_exchange_symbols(...)  # Line 286
async def get_exchange_fees(...)  # Line 312
async def get_rate_limits(exchange: str, current_user: dict = Any, web_exchange_service = Any)  # Line 332
async def get_rate_usage(exchange: str, current_user: dict = Any, web_exchange_service = Any)  # Line 350
async def update_rate_config(...)  # Line 372
async def get_exchanges_health(current_user: dict = Any, web_exchange_service = Any)  # Line 413
async def get_exchange_health(exchange: str, current_user: dict = Any, web_exchange_service = Any)  # Line 436
async def get_exchange_latency(...)  # Line 459
async def get_exchange_errors(...)  # Line 482
async def get_orderbook(...)  # Line 511
async def get_ticker(exchange: str, symbol: str, current_user: dict = Any, web_exchange_service = Any)  # Line 544
async def get_exchange_balance(...)  # Line 574
async def subscribe_websocket(...)  # Line 620
async def unsubscribe_websocket(...)  # Line 654
```

### File: health.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.config import Config`

#### Class: `ConnectionHealthMonitor`

**Inherits**: BaseComponent
**Purpose**: Mock connection health monitor for exchanges

```python
class ConnectionHealthMonitor(BaseComponent):
    def __init__(self, exchange)  # Line 39
    async def get_health_status(self) -> dict  # Line 42
```

#### Class: `HealthStatus`

**Inherits**: BaseModel
**Purpose**: Health check response model

```python
class HealthStatus(BaseModel):
```

#### Class: `ComponentHealth`

**Inherits**: BaseModel
**Purpose**: Individual component health model

```python
class ComponentHealth(BaseModel):
```

#### Functions:

```python
def _get_logger()  # Line 21
def get_local_logger()  # Line 31
def get_config_dependency() -> Config  # Line 75
async def check_database_health(config: Config) -> ComponentHealth  # Line 100
async def check_redis_health(config: Config) -> ComponentHealth  # Line 139
async def check_exchanges_health(config: Config) -> ComponentHealth  # Line 179
async def check_ml_models_health(config: Config) -> ComponentHealth  # Line 245
async def basic_health_check()  # Line 283
async def detailed_health_check(config: Config = Any)  # Line 302
async def readiness_check()  # Line 386
async def liveness_check()  # Line 398
async def startup_check()  # Line 408
```

### File: ml_models.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.web_interface.security.auth import User`
- `from src.web_interface.security.auth import get_admin_user`
- `from src.web_interface.security.auth import get_current_user`

#### Class: `ModelResponse`

**Inherits**: BaseModel
**Purpose**: Response model for ML model information

```python
class ModelResponse(BaseModel):
```

#### Class: `CreateModelRequest`

**Inherits**: BaseModel
**Purpose**: Request model for creating a new ML model

```python
class CreateModelRequest(BaseModel):
```

#### Class: `TrainModelRequest`

**Inherits**: BaseModel
**Purpose**: Request model for training a model

```python
class TrainModelRequest(BaseModel):
```

#### Class: `PredictionRequest`

**Inherits**: BaseModel
**Purpose**: Request model for model predictions

```python
class PredictionRequest(BaseModel):
```

#### Class: `PredictionResponse`

**Inherits**: BaseModel
**Purpose**: Response model for model predictions

```python
class PredictionResponse(BaseModel):
```

#### Class: `TrainingJobResponse`

**Inherits**: BaseModel
**Purpose**: Response model for training job status

```python
class TrainingJobResponse(BaseModel):
```

#### Class: `ModelPerformanceResponse`

**Inherits**: BaseModel
**Purpose**: Response model for model performance metrics

```python
class ModelPerformanceResponse(BaseModel):
```

#### Class: `DeploymentRequest`

**Inherits**: BaseModel
**Purpose**: Request model for model deployment

```python
class DeploymentRequest(BaseModel):
```

#### Class: `FeatureEngineeringRequest`

**Inherits**: BaseModel
**Purpose**: Request model for feature engineering

```python
class FeatureEngineeringRequest(BaseModel):
```

#### Class: `FeatureSelectionRequest`

**Inherits**: BaseModel
**Purpose**: Request model for feature selection

```python
class FeatureSelectionRequest(BaseModel):
```

#### Class: `ABTestRequest`

**Inherits**: BaseModel
**Purpose**: Request model for A/B test creation

```python
class ABTestRequest(BaseModel):
```

#### Class: `HyperparameterOptimizationRequest`

**Inherits**: BaseModel
**Purpose**: Request model for hyperparameter optimization

```python
class HyperparameterOptimizationRequest(BaseModel):
```

#### Functions:

```python
def set_dependencies(manager)  # Line 25
async def list_models(...)  # Line 164
async def create_model(model_request: CreateModelRequest, current_user: User = Any)  # Line 303
async def get_model(model_id: str, current_user: User = Any)  # Line 358
async def train_model(model_id: str, training_request: TrainModelRequest, current_user: User = Any)  # Line 412
async def get_training_jobs(model_id: str, limit: int = Any, current_user: User = Any)  # Line 484
async def predict(model_id: str, prediction_request: PredictionRequest, current_user: User = Any)  # Line 552
async def get_model_performance(model_id: str, period: str = Any, current_user: User = Any)  # Line 634
async def deploy_model(model_id: str, deployment_request: DeploymentRequest, current_user: User = Any)  # Line 702
async def retire_model(model_id: str, reason: str = Any, current_user: User = Any)  # Line 755
async def engineer_features(request: FeatureEngineeringRequest, current_user: User = Any)  # Line 828
async def select_features(request: FeatureSelectionRequest, current_user: User = Any)  # Line 875
async def create_ab_test(request: ABTestRequest, current_user: User = Any)  # Line 943
async def get_ab_test_results(test_id: str, current_user: User = Any)  # Line 992
async def promote_ab_test_winner(test_id: str, promote_treatment: bool = Any, current_user: User = Any)  # Line 1051
async def optimize_hyperparameters(...)  # Line 1125
async def export_model(...)  # Line 1184
async def compare_models(...)  # Line 1240
async def get_ml_system_health(current_user: User = Any)  # Line 1312
```

### File: monitoring.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.utils.web_interface_utils import handle_api_error`
- `from src.web_interface.di_registration import get_web_monitoring_service`
- `from src.web_interface.security.auth import get_current_user`
- `from src.web_interface.security.auth import require_permissions`

#### Class: `HealthCheckResponse`

**Inherits**: BaseModel
**Purpose**: Health check response model

```python
class HealthCheckResponse(BaseModel):
```

#### Class: `MetricsResponse`

**Inherits**: BaseModel
**Purpose**: Metrics response model

```python
class MetricsResponse(BaseModel):
```

#### Class: `AlertRuleRequest`

**Inherits**: BaseModel
**Purpose**: Alert rule creation request

```python
class AlertRuleRequest(BaseModel):
```

#### Class: `AlertResponse`

**Inherits**: BaseModel
**Purpose**: Alert response model

```python
class AlertResponse(BaseModel):
```

#### Class: `PerformanceStatsResponse`

**Inherits**: BaseModel
**Purpose**: Performance statistics response

```python
class PerformanceStatsResponse(BaseModel):
```

#### Class: `MemoryReportResponse`

**Inherits**: BaseModel
**Purpose**: Memory usage report response

```python
class MemoryReportResponse(BaseModel):
```

#### Functions:

```python
def get_monitoring_service() -> 'WebMonitoringService'  # Line 112
async def health_check()  # Line 119
async def system_status()  # Line 141
async def prometheus_metrics()  # Line 156
async def metrics_json(timeframe: int = Any)  # Line 177
async def performance_stats(timeframe: int = Any)  # Line 200
async def memory_report(user = Any)  # Line 227
async def reset_performance_metrics(user = Any)  # Line 246
async def database_query_stats(user = Any)  # Line 266
async def get_alerts(severity: str | None = Any, status: str | None = Any, limit: int = Any)  # Line 287
async def create_alert_rule(rule_request: dict, user = Any)  # Line 320
async def delete_alert_rule(rule_name: str, user = Any)  # Line 343
async def acknowledge_alert(fingerprint: str, user = Any)  # Line 370
async def alert_stats()  # Line 396
async def get_monitoring_config(user = Any)  # Line 409
async def start_monitoring(user = Any)  # Line 445
async def stop_monitoring(user = Any)  # Line 471
```

### File: optimization.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.optimization.di_registration import get_optimization_service`
- `from src.optimization.interfaces import IOptimizationService`

#### Class: `TimeInterval`

**Inherits**: Enum
**Purpose**: Time interval enumeration for data intervals

```python
class TimeInterval(Enum):
```

#### Class: `OptimizationParameterRange`

**Inherits**: BaseModel
**Purpose**: Model for parameter optimization range

```python
class OptimizationParameterRange(BaseModel):
```

#### Class: `OptimizationRequest`

**Inherits**: BaseModel
**Purpose**: Request model for optimization job

```python
class OptimizationRequest(BaseModel):
    def validate_parameter_ranges(cls, v)  # Line 135
    def validate_end_date(cls, v, info)  # Line 142
```

#### Class: `OptimizationResult`

**Inherits**: BaseModel
**Purpose**: Model for individual optimization result

```python
class OptimizationResult(BaseModel):
```

#### Class: `OptimizationJobResponse`

**Inherits**: BaseModel
**Purpose**: Response model for optimization job

```python
class OptimizationJobResponse(BaseModel):
```

#### Class: `OptimizationStatusResponse`

**Inherits**: BaseModel
**Purpose**: Response model for optimization job status

```python
class OptimizationStatusResponse(BaseModel):
```

#### Class: `OptimizationResultsResponse`

**Inherits**: BaseModel
**Purpose**: Response model for optimization results

```python
class OptimizationResultsResponse(BaseModel):
```

#### Functions:

```python
def set_dependencies(backtest_engine, strat_factory, dependency_injector = None)  # Line 52
async def create_optimization_job(...)  # Line 230
async def start_optimization_job(job_id: str, background_tasks: BackgroundTasks, user: User = Any)  # Line 330
async def get_optimization_job_status(job_id: str, user: User = Any)  # Line 396
async def get_optimization_job_results(job_id: str, top_n: int = Any, user: User = Any)  # Line 459
async def stop_optimization_job(job_id: str, user: User = Any)  # Line 571
async def delete_optimization_job(job_id: str, user: User = Any)  # Line 616
async def list_optimization_jobs(user: User = Any, status_filter: str | None = Any, limit: int = Any)  # Line 662
async def _run_optimization_job(job_id: str, optimization_service: IOptimizationService)  # Line 732
def _build_parameter_space_from_request(request_data: dict[str, Any]) -> 'ParameterSpace'  # Line 828
def _generate_parameter_combinations(request_data: dict[str, Any]) -> list[dict[str, Any]]  # Line 861
async def _run_parameter_combination(job_id: str, parameters: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]  # Line 912
```

### File: playground.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types import BotType`
- `from src.utils.web_interface_utils import handle_not_found`
- `from src.web_interface.security.auth import User`
- `from src.web_interface.security.auth import get_current_user`

#### Class: `TimeInterval`

**Inherits**: Enum
**Purpose**: Time interval enumeration for data intervals

```python
class TimeInterval(Enum):
```

#### Class: `PlaygroundConfigurationRequest`

**Inherits**: BaseModel
**Purpose**: Request model for playground configuration

```python
class PlaygroundConfigurationRequest(BaseModel):
```

#### Class: `PlaygroundSessionResponse`

**Inherits**: BaseModel
**Purpose**: Response model for playground session

```python
class PlaygroundSessionResponse(BaseModel):
```

#### Class: `PlaygroundStatusResponse`

**Inherits**: BaseModel
**Purpose**: Response model for playground status

```python
class PlaygroundStatusResponse(BaseModel):
```

#### Class: `PlaygroundResultResponse`

**Inherits**: BaseModel
**Purpose**: Response model for playground results

```python
class PlaygroundResultResponse(BaseModel):
```

#### Class: `PlaygroundLogEntry`

**Inherits**: BaseModel
**Purpose**: Model for playground log entry

```python
class PlaygroundLogEntry(BaseModel):
```

#### Class: `PlaygroundConfigurationModel`

**Inherits**: BaseModel
**Purpose**: Playground configuration model

```python
class PlaygroundConfigurationModel(BaseModel):
```

#### Class: `ABTestModel`

**Inherits**: BaseModel
**Purpose**: A/B test model

```python
class ABTestModel(BaseModel):
```

#### Class: `BatchOptimizationModel`

**Inherits**: BaseModel
**Purpose**: Batch optimization model

```python
class BatchOptimizationModel(BaseModel):
```

#### Functions:

```python
def get_dependencies()  # Line 44
async def create_playground_session(...)  # Line 185
async def start_playground_session(session_id: str, background_tasks: BackgroundTasks, user: User = Any)  # Line 277
async def get_playground_session_status(session_id: str, user: User = Any)  # Line 331
async def get_playground_session_results(session_id: str, user: User = Any)  # Line 384
async def get_playground_session_logs(session_id: str, limit: int = Any, level: str | None = Any, user: User = Any)  # Line 444
async def stop_playground_session(session_id: str, user: User = Any)  # Line 500
async def delete_playground_session(session_id: str, user: User = Any)  # Line 552
async def list_playground_sessions(user: User = Any, status_filter: str | None = Any, limit: int = Any)  # Line 608
async def _run_playground_session(session_id: str)  # Line 652
async def _run_backtest_session(session_id: str, config: dict[str, Any])  # Line 686
async def _run_sandbox_session(session_id: str, config: dict[str, Any])  # Line 781
def _add_session_log(session_id: str, level: str, message: str, context: dict[str, Any] | None = None)  # Line 838
async def get_configurations(user: User = Any, page: int = Any, limit: int = Any)  # Line 920
async def save_configuration(config: PlaygroundConfigurationModel, user: User = Any)  # Line 943
async def delete_configuration(config_id: str, user: User = Any)  # Line 967
async def control_execution(execution_id: str, action: str, user: User = Any)  # Line 992
async def get_executions(...)  # Line 1045
async def create_ab_test(ab_test: ABTestModel, user: User = Any)  # Line 1094
async def run_ab_test(ab_test_id: str, background_tasks: BackgroundTasks, user: User = Any)  # Line 1115
async def start_batch_optimization(...)  # Line 1143
async def get_batch_progress(batch_id: str, user: User = Any)  # Line 1169
async def _run_ab_test(ab_test_id: str)  # Line 1213
async def _run_batch_optimization(batch_id: str)  # Line 1240
```

### File: portfolio.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.web_interface.di_registration import get_web_portfolio_service`
- `from src.web_interface.security.auth import User`
- `from src.web_interface.security.auth import get_current_user`
- `from src.web_interface.security.auth import get_trading_user`

#### Class: `PositionResponse`

**Inherits**: BaseModel
**Purpose**: Response model for position information

```python
class PositionResponse(BaseModel):
```

#### Class: `BalanceResponse`

**Inherits**: BaseModel
**Purpose**: Response model for balance information

```python
class BalanceResponse(BaseModel):
```

#### Class: `PnLResponse`

**Inherits**: BaseModel
**Purpose**: Response model for P&L information

```python
class PnLResponse(BaseModel):
```

#### Class: `PortfolioSummaryResponse`

**Inherits**: BaseModel
**Purpose**: Response model for portfolio summary

```python
class PortfolioSummaryResponse(BaseModel):
```

#### Class: `AssetAllocationResponse`

**Inherits**: BaseModel
**Purpose**: Response model for asset allocation

```python
class AssetAllocationResponse(BaseModel):
```

#### Functions:

```python
def get_portfolio_service()  # Line 26
def get_web_portfolio_service_instance() -> 'WebPortfolioService'  # Line 33
def set_dependencies(orchestrator, engine)  # Line 39
async def get_portfolio_summary(current_user: User = Any)  # Line 118
async def get_positions(...)  # Line 178
async def get_balances(exchange: str | None = Any, currency: str | None = Any, current_user: User = Any)  # Line 231
async def get_pnl(period: str = Any, bot_id: str | None = Any, current_user: User = Any)  # Line 288
async def get_asset_allocation(current_user: User = Any)  # Line 323
async def get_performance_chart(period: str = Any, resolution: str = Any, current_user: User = Any)  # Line 359
async def get_portfolio_history(period: str = Any, current_user: User = Any)  # Line 395
async def get_portfolio_performance(current_user: User = Any)  # Line 435
async def rebalance_portfolio(request: dict[str, Any], trading_user: User = Any)  # Line 474
async def get_portfolio_holdings(current_user: User = Any)  # Line 517
```

### File: risk.py

**Key Imports:**
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.web_interface.di_registration import get_web_risk_service`
- `from src.web_interface.security.auth import User`

#### Class: `RiskMetricsResponse`

**Inherits**: BaseModel
**Purpose**: Response model for risk metrics

```python
class RiskMetricsResponse(BaseModel):
```

#### Class: `RiskLimitsResponse`

**Inherits**: BaseModel
**Purpose**: Response model for risk limits

```python
class RiskLimitsResponse(BaseModel):
```

#### Class: `UpdateRiskLimitsRequest`

**Inherits**: BaseModel
**Purpose**: Request model for updating risk limits

```python
class UpdateRiskLimitsRequest(BaseModel):
```

#### Class: `RiskAlertResponse`

**Inherits**: BaseModel
**Purpose**: Response model for risk alerts

```python
class RiskAlertResponse(BaseModel):
```

#### Class: `PositionRiskResponse`

**Inherits**: BaseModel
**Purpose**: Response model for individual position risk

```python
class PositionRiskResponse(BaseModel):
```

#### Class: `StressTestRequest`

**Inherits**: BaseModel
**Purpose**: Request model for stress testing

```python
class StressTestRequest(BaseModel):
```

#### Class: `StressTestResponse`

**Inherits**: BaseModel
**Purpose**: Response model for stress test results

```python
class StressTestResponse(BaseModel):
```

#### Functions:

```python
def get_web_risk_service_instance()  # Line 24
async def get_risk_metrics(current_user: User = Any)  # Line 138
async def get_risk_limits(current_user: User = Any)  # Line 197
async def update_risk_limits(limits_request: UpdateRiskLimitsRequest, current_user: User = Any)  # Line 227
async def get_risk_alerts(...)  # Line 297
async def get_position_risks(exchange: str | None = Any, symbol: str | None = Any, current_user: User = Any)  # Line 358
async def run_stress_test(stress_test_request: StressTestRequest, current_user: User = Any)  # Line 406
async def get_correlation_matrix(symbols: list[str] = Any, period: str = Any, current_user: User = Any)  # Line 458
```

### File: state_management.py

**Key Imports:**
- `from src.core.exceptions import StateError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import BotState`
- `from src.core.types import MarketData`

#### Class: `StateSnapshotRequest`

**Inherits**: BaseModel
**Purpose**: Request model for creating state snapshots

```python
class StateSnapshotRequest(BaseModel):
```

#### Class: `StateSnapshotResponse`

**Inherits**: BaseModel
**Purpose**: Response model for state snapshots

```python
class StateSnapshotResponse(BaseModel):
```

#### Class: `CheckpointRequest`

**Inherits**: BaseModel
**Purpose**: Request model for checkpoint operations

```python
class CheckpointRequest(BaseModel):
```

#### Class: `CheckpointResponse`

**Inherits**: BaseModel
**Purpose**: Response model for checkpoint operations

```python
class CheckpointResponse(BaseModel):
```

#### Class: `RecoveryRequest`

**Inherits**: BaseModel
**Purpose**: Request model for recovery operations

```python
class RecoveryRequest(BaseModel):
```

#### Class: `TradeValidationRequest`

**Inherits**: BaseModel
**Purpose**: Request model for trade validation

```python
class TradeValidationRequest(BaseModel):
```

#### Class: `TradeValidationResponse`

**Inherits**: BaseModel
**Purpose**: Response model for trade validation

```python
class TradeValidationResponse(BaseModel):
```

#### Class: `PostTradeAnalysisRequest`

**Inherits**: BaseModel
**Purpose**: Request model for post-trade analysis

```python
class PostTradeAnalysisRequest(BaseModel):
```

#### Class: `PostTradeAnalysisResponse`

**Inherits**: BaseModel
**Purpose**: Response model for post-trade analysis

```python
class PostTradeAnalysisResponse(BaseModel):
```

#### Class: `SyncStatusResponse`

**Inherits**: BaseModel
**Purpose**: Response model for sync status

```python
class SyncStatusResponse(BaseModel):
```

#### Functions:

```python
async def get_state_service() -> StateService  # Line 145
async def get_checkpoint_manager() -> CheckpointManager  # Line 162
async def get_lifecycle_manager() -> TradeLifecycleManager  # Line 179
async def get_quality_controller() -> QualityController  # Line 196
async def get_bot_state(...)  # Line 221
async def save_bot_state(...)  # Line 265
async def get_state_metrics(...)  # Line 332
async def create_checkpoint(...)  # Line 378
async def list_checkpoints(...)  # Line 443
async def restore_from_checkpoint(...)  # Line 473
async def get_checkpoint_stats(current_user: dict = Any, checkpoint_manager: CheckpointManager = Any)  # Line 530
async def get_trade_lifecycle(...)  # Line 556
async def get_trade_history(...)  # Line 616
async def get_trade_performance(...)  # Line 668
async def validate_pre_trade(...)  # Line 753
async def analyze_post_trade(...)  # Line 802
async def get_quality_summary(...)  # Line 861
async def get_sync_status(...)  # Line 915
async def force_sync(...)  # Line 962
async def get_sync_metrics(current_user: dict = Any, state_service: StateService = Any)  # Line 998
async def get_sync_conflicts(...)  # Line 1037
```

### File: strategies.py

**Key Imports:**
- `from src.core.exceptions import ServiceError`
- `from src.core.logging import get_logger`
- `from src.web_interface.security.auth import User`
- `from src.web_interface.security.auth import get_admin_user`
- `from src.web_interface.security.auth import get_current_user`

#### Class: `StrategyResponse`

**Inherits**: BaseModel
**Purpose**: Response model for strategy information

```python
class StrategyResponse(BaseModel):
```

#### Class: `StrategyConfigRequest`

**Inherits**: BaseModel
**Purpose**: Request model for strategy configuration

```python
class StrategyConfigRequest(BaseModel):
```

#### Class: `StrategyPerformanceResponse`

**Inherits**: BaseModel
**Purpose**: Response model for strategy performance

```python
class StrategyPerformanceResponse(BaseModel):
```

#### Class: `BacktestRequest`

**Inherits**: BaseModel
**Purpose**: Request model for strategy backtesting

```python
class BacktestRequest(BaseModel):
```

#### Class: `BacktestResponse`

**Inherits**: BaseModel
**Purpose**: Response model for backtest results

```python
class BacktestResponse(BaseModel):
```

#### Functions:

```python
def get_strategy_service()  # Line 26
def get_web_strategy_service_instance() -> 'WebStrategyService'  # Line 32
def set_dependencies(factory)  # Line 59
async def list_strategies(...)  # Line 153
async def get_strategy(strategy_name: str, current_user: User = Any)  # Line 226
async def configure_strategy(...)  # Line 328
async def get_strategy_performance(strategy_name: str, period: str = Any, current_user: User = Any)  # Line 390
async def start_backtest(strategy_name: str, backtest_request: BacktestRequest, current_user: User = Any)  # Line 456
async def list_strategy_categories(current_user: User = Any)  # Line 538
async def optimize_strategy(...)  # Line 597
```

### File: trading.py

**Key Imports:**
- `from src.core.data_transformer import CoreDataTransformer`
- `from src.core.exceptions import ErrorSeverity`
- `from src.core.exceptions import NetworkError`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import TimeoutError`

#### Class: `PlaceOrderRequest`

**Inherits**: BaseModel
**Purpose**: Request model for placing an order

```python
class PlaceOrderRequest(BaseModel):
```

#### Class: `CancelOrderRequest`

**Inherits**: BaseModel
**Purpose**: Request model for cancelling an order

```python
class CancelOrderRequest(BaseModel):
```

#### Class: `OrderResponse`

**Inherits**: BaseModel
**Purpose**: Response model for order information

```python
class OrderResponse(BaseModel):
```

#### Class: `TradeResponse`

**Inherits**: BaseModel
**Purpose**: Response model for trade information

```python
class TradeResponse(BaseModel):
```

#### Class: `OrderBookResponse`

**Inherits**: BaseModel
**Purpose**: Response model for order book data

```python
class OrderBookResponse(BaseModel):
```

#### Class: `MarketDataResponse`

**Inherits**: BaseModel
**Purpose**: Response model for market data

```python
class MarketDataResponse(BaseModel):
```

#### Functions:

```python
def get_trading_service() -> Any  # Line 36
def get_web_trading_service_instance()  # Line 43
def set_dependencies(engine: Any, orchestrator: Any) -> None  # Line 49
async def place_order(order_request: PlaceOrderRequest, current_user: User = Any)  # Line 155
async def cancel_order(order_id: str, exchange: str = Any, current_user: User = Any)  # Line 344
async def get_orders(...)  # Line 416
async def get_order(order_id: str, exchange: str = Any, current_user: User = Any)  # Line 469
async def get_trades(...)  # Line 505
async def get_market_data(symbol: str, exchange: str = Any, current_user: User = Any)  # Line 561
async def get_order_book(symbol: str, exchange: str = Any, depth: int = Any, current_user: User = Any)  # Line 610
async def get_execution_status(current_user: User = Any)  # Line 654
async def get_active_orders(current_user: User = Any)  # Line 693
async def get_trading_positions(current_user: User = Any)  # Line 732
async def get_trading_balance(current_user: User = Any)  # Line 775
```

### File: app.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import ConfigurationError`
- `from src.core.logging import correlation_context`
- `from src.core.logging import get_logger`
- `from src.monitoring import AlertManager`

#### Class: `LazyApp`

**Purpose**: Lazy app wrapper that creates app on first access

```python
class LazyApp:
    def __getattr__(self, name)  # Line 773
    def __call__(self, *args, **kwargs)  # Line 777
```

#### Functions:

```python
async def _initialize_services()  # Line 71
async def _connect_api_endpoints_to_services(registry)  # Line 152
async def lifespan(app: FastAPI)  # Line 172
def create_app(...) -> Any  # Line 297
def _register_routes(app: FastAPI) -> None  # Line 460
def _setup_monitoring(fastapi_app: FastAPI, config: Config) -> None  # Line 626
def get_app()  # Line 717
def _get_app_lazy()  # Line 751
def get_asgi_app()  # Line 760
```

### File: auth_manager.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.logging import get_logger`

#### Class: `AuthManager`

**Inherits**: BaseComponent
**Purpose**: Unified authentication manager

```python
class AuthManager(BaseComponent):
    def __init__(self, jwt_handler = None, config: dict[str, Any] | None = None)  # Line 23
    def configure_dependencies(self, injector)  # Line 42
    def _initialize_providers(self) -> None  # Line 51
    def _create_default_users(self) -> None  # Line 83
    def _start_cleanup_tasks(self) -> None  # Line 125
    async def authenticate(self, credentials: dict[str, Any], provider_type: str | None = None) -> tuple[User, AuthToken] | None  # Line 130
    async def validate_token(self, token_value: str, provider_type: str | None = None) -> User | None  # Line 180
    async def revoke_token(self, token_value: str, provider_type: str | None = None) -> bool  # Line 211
    async def refresh_token(self, refresh_token_value: str) -> AuthToken | None  # Line 235
    async def create_api_key(self, user: User) -> AuthToken | None  # Line 255
    def get_user(self, user_id: str) -> User | None  # Line 275
    def get_user_by_username(self, username: str) -> User | None  # Line 279
    async def create_user(self, user_data: dict[str, Any]) -> User | None  # Line 286
    async def update_user(self, user_id: str, updates: dict[str, Any]) -> bool  # Line 329
    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool  # Line 365
    def get_user_stats(self) -> dict[str, Any]  # Line 392
    async def _cleanup_expired_sessions(self) -> None  # Line 405
```

#### Functions:

```python
def get_auth_manager(injector = None, config: dict[str, Any] | None = None) -> AuthManager  # Line 430
def initialize_auth_manager(config: dict[str, Any]) -> AuthManager  # Line 469
```

### File: decorators.py

#### Functions:

```python
async def get_current_user(request: Request) -> User | None  # Line 16
async def get_current_active_user(current_user: User = Any) -> User  # Line 36
async def get_trading_user(current_user: User = Any) -> User  # Line 53
async def get_admin_user(current_user: User = Any) -> User  # Line 62
def require_auth(func)  # Line 71
def require_permission(permission: PermissionType, resource: str | None = None)  # Line 81
def require_role(role_name: str)  # Line 99
```

### File: middleware.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.database.models.user import User`

#### Class: `AuthMiddleware`

**Inherits**: BaseHTTPMiddleware, BaseComponent
**Purpose**: Middleware to handle authentication for all requests

```python
class AuthMiddleware(BaseHTTPMiddleware, BaseComponent):
    def __init__(self, app)  # Line 26
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 31
```

#### Functions:

```python
async def get_current_user(credentials: HTTPAuthorizationCredentials = Any) -> User  # Line 69
```

### File: models.py

#### Class: `PermissionType`

**Inherits**: Enum
**Purpose**: Permission types for role-based access control

```python
class PermissionType(Enum):
```

#### Class: `Permission`

**Purpose**: Represents a system permission

```python
class Permission:
    def __str__(self) -> str  # Line 68
    def __hash__(self) -> int  # Line 75
```

#### Class: `Role`

**Purpose**: Represents a user role with associated permissions

```python
class Role:
    def __hash__(self) -> int  # Line 89
    def __eq__(self, other) -> bool  # Line 93
    def add_permission(self, permission: Permission) -> None  # Line 99
    def remove_permission(self, permission: Permission) -> None  # Line 103
    def has_permission(self, permission_type: PermissionType, resource: str | None = None) -> bool  # Line 107
    def get_permissions_by_type(self, permission_type: PermissionType) -> list[Permission]  # Line 115
```

#### Class: `UserStatus`

**Inherits**: Enum
**Purpose**: User account status

```python
class UserStatus(Enum):
```

#### Class: `User`

**Purpose**: Represents a system user

```python
class User:
    def add_role(self, role: Role) -> None  # Line 152
    def remove_role(self, role: Role) -> None  # Line 156
    def has_role(self, role_name: str) -> bool  # Line 160
    def has_permission(self, permission_type: PermissionType, resource: str | None = None) -> bool  # Line 164
    def get_all_permissions(self) -> set[Permission]  # Line 171
    def is_locked(self) -> bool  # Line 178
    def is_active(self) -> bool  # Line 186
    def lock_account(self, duration_minutes: int = 30) -> None  # Line 190
    def unlock_account(self) -> None  # Line 195
    def increment_login_attempts(self) -> None  # Line 202
    def reset_login_attempts(self) -> None  # Line 208
```

#### Class: `TokenType`

**Inherits**: Enum
**Purpose**: Authentication token types

```python
class TokenType(Enum):
```

#### Class: `AuthToken`

**Purpose**: Represents an authentication token

```python
class AuthToken:
    def is_expired(self) -> bool  # Line 241
    def is_valid(self) -> bool  # Line 247
    def revoke(self) -> None  # Line 251
    def touch(self) -> None  # Line 255
```

#### Functions:

```python
def create_system_roles() -> dict[str, Role]  # Line 261
```

### File: providers.py

**Key Imports:**
- `from src.core.base import BaseComponent`

#### Class: `AuthProvider`

**Inherits**: ABC
**Purpose**: Abstract base class for authentication providers

```python
class AuthProvider(ABC):
    async def authenticate(self, credentials: dict[str, Any]) -> User | None  # Line 24
    async def create_token(self, user: User) -> AuthToken  # Line 29
    async def validate_token(self, token_value: str) -> User | None  # Line 34
    async def revoke_token(self, token_value: str) -> bool  # Line 39
```

#### Class: `JWTAuthProvider`

**Inherits**: AuthProvider, BaseComponent
**Purpose**: JWT-based authentication provider

```python
class JWTAuthProvider(AuthProvider, BaseComponent):
    def __init__(self, ...)  # Line 47
    async def authenticate(self, credentials: dict[str, Any]) -> User | None  # Line 64
    async def create_token(self, user: User) -> AuthToken  # Line 104
    async def create_refresh_token(self, user: User) -> AuthToken  # Line 134
    async def validate_token(self, token_value: str) -> User | None  # Line 162
    async def revoke_token(self, token_value: str) -> bool  # Line 217
    async def refresh_access_token(self, refresh_token_value: str) -> AuthToken | None  # Line 234
```

#### Class: `SessionAuthProvider`

**Inherits**: AuthProvider, BaseComponent
**Purpose**: Session-based authentication provider

```python
class SessionAuthProvider(AuthProvider, BaseComponent):
    def __init__(self, session_timeout_minutes: int = 60)  # Line 283
    async def authenticate(self, credentials: dict[str, Any]) -> User | None  # Line 288
    async def _check_credentials(self, credentials: dict[str, Any]) -> User | None  # Line 294
    async def create_token(self, user: User) -> AuthToken  # Line 331
    async def validate_token(self, token_value: str) -> User | None  # Line 357
    async def revoke_token(self, token_value: str) -> bool  # Line 391
    async def cleanup_expired_sessions(self) -> int  # Line 398
```

#### Class: `APIKeyAuthProvider`

**Inherits**: AuthProvider, BaseComponent
**Purpose**: API key-based authentication provider

```python
class APIKeyAuthProvider(AuthProvider, BaseComponent):
    def __init__(self)  # Line 416
    async def authenticate(self, credentials: dict[str, Any]) -> User | None  # Line 421
    async def create_token(self, user: User) -> AuthToken  # Line 429
    async def validate_token(self, token_value: str) -> User | None  # Line 450
    async def validate_api_key(self, api_key: str) -> User | None  # Line 454
    async def revoke_token(self, token_value: str) -> bool  # Line 480
    def _generate_api_key(self) -> str  # Line 487
```

### File: data_transformer.py

**Key Imports:**
- `from src.core.data_transformer import CoreDataTransformer`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.decimal_utils import to_decimal`

#### Class: `WebInterfaceErrorPropagator`

**Purpose**: Error propagation patterns aligned with risk_management module

```python
class WebInterfaceErrorPropagator:
    def propagate_to_risk_management(error: Exception, context: str) -> None  # Line 272
    def propagate_validation_error(error: Exception, context: str) -> None  # Line 282
    def propagate_to_monitoring(error: Exception, context: str) -> None  # Line 290
```

#### Class: `WebInterfaceDataTransformer`

**Purpose**: Data transformer for backward compatibility, aligned with core standards

```python
class WebInterfaceDataTransformer:
    def format_portfolio_composition(data: Any) -> Any  # Line 303
    def format_stress_test_results(data: Any) -> Any  # Line 314
    def format_operational_metrics(data: Any) -> Any  # Line 325
    def transform_risk_data_to_event_data(data: Any, **kwargs) -> Any  # Line 336
    def validate_financial_precision(data: Any) -> Any  # Line 354
```

#### Functions:

```python
def format_decimal(value: Any) -> str  # Line 19
def add_timestamp(data: dict[str, Any]) -> dict[str, Any]  # Line 28
def ensure_decimal_strings(data: dict[str, Any]) -> dict[str, Any]  # Line 34
def transform_for_web_response(data: Any, event_type: str = 'web_response', processing_mode: str = 'stream') -> dict[str, Any]  # Line 57
def transform_for_api_response(data: Any, endpoint: str) -> dict[str, Any]  # Line 113
def align_with_risk_management_patterns(data: dict[str, Any], operation_type: str) -> dict[str, Any]  # Line 143
def validate_web_to_risk_management_boundary(data: dict[str, Any]) -> None  # Line 182
def validate_bot_management_to_web_boundary(data: dict[str, Any]) -> None  # Line 222
def align_error_propagation_patterns() -> None  # Line 252
def align_processing_paradigms_with_risk_management(data: dict[str, Any], operation_type: str, target_mode: str | None = None) -> dict[str, Any]  # Line 366
def handle_batch_to_stream_conversion_for_risk_management(batch_data: list[dict[str, Any]], operation_context: str = 'web_interface_batch') -> list[dict[str, Any]]  # Line 419
def validate_paradigm_alignment_for_risk_management(data: dict[str, Any]) -> None  # Line 460
```

### File: dependencies.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.dependency_injection import get_global_injector`
- `from src.core.logging import get_logger`

#### Functions:

```python
def get_web_auth_service() -> 'WebAuthService'  # Line 28
def get_web_analytics_service() -> 'WebAnalyticsService'  # Line 40
def get_web_capital_service() -> 'WebCapitalService'  # Line 52
def get_web_data_service() -> 'WebDataService'  # Line 64
def get_web_exchange_service() -> 'WebExchangeService'  # Line 76
def get_web_portfolio_service() -> 'WebPortfolioService'  # Line 89
def get_web_trading_service() -> 'WebTradingService'  # Line 101
def get_web_bot_service() -> 'WebBotService'  # Line 113
def get_web_monitoring_service() -> 'WebMonitoringService'  # Line 125
def get_web_risk_service() -> 'WebRiskService'  # Line 137
def get_service_registry() -> Any  # Line 150
def get_api_facade(injector: DependencyInjector = None) -> Any  # Line 162
def get_websocket_manager(injector: DependencyInjector = None) -> Any  # Line 175
def get_jwt_handler(injector: DependencyInjector = None) -> Any  # Line 189
def get_auth_manager(injector: DependencyInjector = None) -> Any  # Line 202
def get_web_auth_service_instance() -> 'WebAuthService'  # Line 216
def get_web_analytics_service_instance() -> 'WebAnalyticsService'  # Line 221
def get_web_capital_service_instance() -> 'WebCapitalService'  # Line 226
def get_web_data_service_instance() -> 'WebDataService'  # Line 231
def get_web_exchange_service_instance() -> 'WebExchangeService'  # Line 236
def get_web_portfolio_service_instance() -> 'WebPortfolioService'  # Line 241
def get_web_trading_service_instance() -> 'WebTradingService'  # Line 246
def get_web_bot_service_instance() -> 'WebBotService'  # Line 251
def get_web_monitoring_service_instance() -> 'WebMonitoringService'  # Line 256
def get_web_risk_service_instance() -> 'WebRiskService'  # Line 261
def ensure_all_services_registered(injector: DependencyInjector = None) -> None  # Line 267
def get_all_web_services(injector: DependencyInjector = None) -> dict[str, Any]  # Line 295
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`

#### Functions:

```python
def _register_core_services(injector: DependencyInjector, factory: 'WebInterfaceFactory') -> None  # Line 19
def _register_web_business_services(injector: DependencyInjector, factory: 'WebInterfaceFactory') -> None  # Line 63
def register_web_interface_services(injector: DependencyInjector) -> None  # Line 142
def _create_mock_analytics_service()  # Line 184
def _resolve_analytics_dependencies(injector: DependencyInjector) -> dict  # Line 238
def _register_analytics_services(injector: DependencyInjector) -> None  # Line 274
def _register_utility_services(injector: DependencyInjector, factory: 'WebInterfaceFactory') -> None  # Line 363
def get_web_portfolio_service(injector: DependencyInjector = None) -> 'WebPortfolioService'  # Line 401
def get_web_trading_service(injector: DependencyInjector = None) -> 'WebTradingService'  # Line 415
def get_web_bot_service(injector: DependencyInjector = None) -> 'WebBotService'  # Line 428
def get_web_monitoring_service(injector: DependencyInjector = None) -> 'WebMonitoringService'  # Line 441
def get_web_risk_service(injector: DependencyInjector = None) -> 'WebRiskService'  # Line 454
def get_api_facade_service(injector: DependencyInjector = None)  # Line 467
def get_web_interface_factory(injector: DependencyInjector = None) -> 'WebInterfaceFactory'  # Line 480
```

### File: api_facade.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ServiceError`
- `from src.utils.decimal_utils import to_decimal`
- `from src.core.types import BotConfiguration`
- `from src.core.types import OrderSide`

#### Class: `APIFacade`

**Inherits**: BaseComponent
**Purpose**: Unified API facade for T-Bot Trading System

```python
class APIFacade(BaseComponent):
    def __init__(self, ...)  # Line 36
    def configure_dependencies(self, injector)  # Line 56
    async def initialize(self) -> None  # Line 76
    async def cleanup(self) -> None  # Line 97
    async def place_order(self, ...) -> str  # Line 119
    async def cancel_order(self, order_id: str) -> bool  # Line 141
    async def get_positions(self) -> list[Position]  # Line 149
    async def create_bot(self, config: BotConfiguration) -> str  # Line 160
    async def start_bot(self, bot_id: str) -> bool  # Line 166
    async def stop_bot(self, bot_id: str) -> bool  # Line 172
    async def get_bot_status(self, bot_id: str) -> dict[str, Any]  # Line 178
    async def list_bots(self) -> list[dict[str, Any]]  # Line 184
    async def get_balance(self) -> dict[str, Decimal]  # Line 191
    async def get_portfolio_summary(self) -> dict[str, Any]  # Line 198
    async def get_pnl_report(self, start_date: datetime, end_date: datetime) -> dict[str, Any]  # Line 204
    async def validate_order(self, ...) -> bool  # Line 212
    async def get_risk_summary(self) -> dict[str, Any]  # Line 223
    async def get_risk_metrics(self) -> dict[str, Any]  # Line 229
    async def calculate_position_size(self, ...) -> Decimal  # Line 233
    async def list_strategies(self) -> list[dict[str, Any]]  # Line 251
    async def get_strategy_config(self, strategy_name: str) -> dict[str, Any]  # Line 257
    async def validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool  # Line 263
    async def health_check(self) -> dict[str, Any]  # Line 272
    def get_service_status(self, service_name: str) -> dict[str, Any]  # Line 288
    async def delete_bot(self, bot_id: str, force: bool = False) -> bool  # Line 309
```

#### Functions:

```python
def get_api_facade(...) -> APIFacade  # Line 320
```

### File: service_registry.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ValidationError`

#### Class: `ServiceInterface`

**Inherits**: ABC
**Purpose**: Base interface for all services

```python
class ServiceInterface(ABC):
    async def initialize(self) -> None  # Line 20
    async def cleanup(self) -> None  # Line 25
```

#### Class: `ServiceRegistry`

**Inherits**: BaseComponent
**Purpose**: Central registry for all system services

```python
class ServiceRegistry(BaseComponent):
    def __init__(self)  # Line 33
    def register_service(self, name: str, service: Any, interface: type | None = None) -> None  # Line 39
    def get_service(self, name: str) -> Any  # Line 58
    def has_service(self, name: str) -> bool  # Line 75
    async def initialize_all(self) -> None  # Line 79
    async def cleanup_all(self) -> None  # Line 92
    def list_services(self) -> dict[str, str]  # Line 103
    def get_all_service_names(self) -> list[str]  # Line 107
    def _implements_interface(self, service: Any, interface: type) -> bool  # Line 111
```

#### Functions:

```python
def get_service_registry(injector = None) -> ServiceRegistry  # Line 131
def register_service(name: str, service: Any, interface: type | None = None) -> None  # Line 139
def get_service(name: str) -> Any  # Line 145
```

### File: factory.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`
- `from src.web_interface.interfaces import WebBotServiceInterface`
- `from src.web_interface.interfaces import WebPortfolioServiceInterface`

#### Class: `WebInterfaceFactory`

**Inherits**: BaseComponent
**Purpose**: Factory for creating web interface components

```python
class WebInterfaceFactory(BaseComponent):
    def __init__(self, injector: DependencyInjector | None = None) -> None  # Line 46
    def create_service_registry(self) -> ServiceRegistry  # Line 60
    def create_jwt_handler(self, config: dict[str, Any] | None = None) -> JWTHandler  # Line 77
    def create_auth_manager(self, config: dict[str, Any] | None = None) -> AuthManager  # Line 101
    def create_trading_service(self) -> WebTradingServiceInterface  # Line 124
    def create_bot_management_service(self) -> WebBotServiceInterface  # Line 143
    def create_portfolio_service(self) -> WebPortfolioServiceInterface  # Line 162
    def create_risk_service(self) -> WebRiskServiceInterface  # Line 181
    def create_strategy_service(self) -> WebStrategyServiceInterface  # Line 200
    def create_market_data_service(self)  # Line 220
    def create_api_facade(self) -> APIFacade  # Line 264
    def create_websocket_manager(self) -> UnifiedWebSocketManager  # Line 331
    def create_complete_web_stack(self, config: dict[str, Any] | None = None) -> dict[str, Any]  # Line 360
```

#### Functions:

```python
def create_web_interface_service(...) -> WebServiceInterface | Any  # Line 393
def create_web_interface_stack(injector: DependencyInjector | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]  # Line 466
```

### File: interfaces.py

#### Class: `WebPortfolioServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web portfolio service operations

```python
class WebPortfolioServiceInterface(Protocol):
    async def get_portfolio_summary_data(self) -> dict[str, Any]  # Line 16
    async def calculate_pnl_periods(self, total_pnl: Decimal, total_trades: int, win_rate: float) -> dict[str, dict[str, Any]]  # Line 20
    async def get_processed_positions(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]  # Line 26
    async def calculate_pnl_metrics(self, period: str) -> dict[str, Any]  # Line 32
    def generate_mock_balances(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]  # Line 36
    def calculate_asset_allocation(self) -> list[dict[str, Any]]  # Line 40
    def generate_performance_chart_data(self, period: str, resolution: str) -> dict[str, Any]  # Line 44
```

#### Class: `WebTradingServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web trading service operations

```python
class WebTradingServiceInterface(Protocol):
    async def validate_order_request(self, ...) -> dict[str, Any]  # Line 52
    async def format_order_response(self, order_result: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]  # Line 63
    async def get_formatted_orders(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]  # Line 69
    async def get_formatted_trades(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]  # Line 75
    async def get_market_data_with_context(self, symbol: str, exchange: str = 'binance') -> dict[str, Any]  # Line 81
    async def generate_order_book_data(self, symbol: str, exchange: str, depth: int) -> dict[str, Any]  # Line 87
    async def place_order_through_service(self, ...) -> dict[str, Any]  # Line 93
    async def cancel_order_through_service(self, order_id: str) -> bool  # Line 104
    async def get_order_details(self, order_id: str, exchange: str) -> dict[str, Any]  # Line 108
    async def get_service_health(self) -> dict[str, Any]  # Line 112
```

#### Class: `WebBotServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web bot service operations

```python
class WebBotServiceInterface(Protocol):
    async def validate_bot_configuration(self, config_data: dict[str, Any]) -> dict[str, Any]  # Line 120
    async def format_bot_response(self, bot_data: dict[str, Any]) -> dict[str, Any]  # Line 124
    async def get_formatted_bot_list(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]  # Line 128
    async def calculate_bot_metrics(self, bot_id: str) -> dict[str, Any]  # Line 134
    async def validate_bot_operation(self, bot_id: str, operation: str) -> dict[str, Any]  # Line 138
    async def create_bot_configuration(self, request_data: dict[str, Any], user_id: str) -> Any  # Line 142
    async def pause_bot_through_service(self, bot_id: str) -> bool  # Line 146
    async def resume_bot_through_service(self, bot_id: str) -> bool  # Line 150
    async def create_bot_through_service(self, bot_config: Any) -> str  # Line 154
    async def get_bot_status_through_service(self, bot_id: str) -> dict[str, Any]  # Line 158
    async def start_bot_through_service(self, bot_id: str) -> bool  # Line 162
    async def stop_bot_through_service(self, bot_id: str) -> bool  # Line 166
    async def delete_bot_through_service(self, bot_id: str, force: bool = False) -> bool  # Line 170
    async def list_bots_through_service(self) -> list[dict[str, Any]]  # Line 174
    async def update_bot_configuration(self, bot_id: str, update_data: dict[str, Any], user_id: str) -> dict[str, Any]  # Line 178
    async def start_bot_with_execution_integration(self, bot_id: str) -> bool  # Line 184
    async def stop_bot_with_execution_integration(self, bot_id: str) -> bool  # Line 188
    def get_controller_health_check(self) -> dict[str, Any]  # Line 192
```

#### Class: `WebMonitoringServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web monitoring service operations

```python
class WebMonitoringServiceInterface(Protocol):
    async def get_system_health_summary(self) -> dict[str, Any]  # Line 200
    async def get_performance_metrics(self, component: str | None = None) -> dict[str, Any]  # Line 204
    async def get_error_summary(self, time_range: str = '24h') -> dict[str, Any]  # Line 208
    async def get_alert_dashboard_data(self) -> dict[str, Any]  # Line 212
```

#### Class: `WebRiskServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web risk service operations

```python
class WebRiskServiceInterface(Protocol):
    async def get_risk_dashboard_data(self) -> dict[str, Any]  # Line 220
    async def validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]  # Line 224
    async def calculate_position_risk(self, symbol: str, quantity: Decimal, price: Decimal) -> dict[str, Any]  # Line 228
    async def get_portfolio_risk_breakdown(self) -> dict[str, Any]  # Line 234
    async def get_current_risk_limits(self) -> dict[str, Any]  # Line 238
```

#### Class: `WebStrategyServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web strategy service operations

```python
class WebStrategyServiceInterface(Protocol):
    async def get_formatted_strategies(self) -> list[dict[str, Any]]  # Line 246
    async def validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> dict[str, Any]  # Line 250
    async def get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]  # Line 256
    async def format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]  # Line 260
```

#### Class: `WebDataServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web data service operations

```python
class WebDataServiceInterface(Protocol):
    async def get_market_overview(self, exchange: str = 'binance') -> dict[str, Any]  # Line 268
    async def get_symbol_analytics(self, symbol: str) -> dict[str, Any]  # Line 272
    async def get_historical_chart_data(self, symbol: str, timeframe: str, period: str) -> dict[str, Any]  # Line 276
    async def get_real_time_feed_status(self) -> dict[str, Any]  # Line 282
```

#### Class: `WebServiceInterface`

**Inherits**: ABC
**Purpose**: Base interface for all web services

```python
class WebServiceInterface(ABC):
    async def initialize(self) -> None  # Line 292
    async def cleanup(self) -> None  # Line 297
    def health_check(self) -> dict[str, Any]  # Line 302
    def get_service_info(self) -> dict[str, Any]  # Line 307
```

#### Class: `WebStrategyServiceExtendedInterface`

**Inherits**: Protocol
**Purpose**: Extended interface for web strategy service operations

```python
class WebStrategyServiceExtendedInterface(Protocol):
    async def get_formatted_strategies(self) -> list[dict[str, Any]]  # Line 315
    async def validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> dict[str, Any]  # Line 319
    async def get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]  # Line 325
    async def format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]  # Line 329
    def health_check(self) -> dict[str, Any]  # Line 333
    def get_service_info(self) -> dict[str, Any]  # Line 337
```

#### Class: `WebAuthServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web authentication service operations

```python
class WebAuthServiceInterface(Protocol):
    async def get_user_by_username(self, username: str) -> Any | None  # Line 345
    async def authenticate_user(self, username: str, password: str) -> Any | None  # Line 349
    async def create_user(self, ...) -> Any  # Line 353
    async def get_auth_summary(self) -> dict[str, Any]  # Line 364
    def get_user_roles(self, current_user: Any) -> list[str]  # Line 368
    def check_permission(self, current_user: Any, required_roles: list[str]) -> bool  # Line 372
    def require_permission(self, current_user: Any, required_roles: list[str]) -> None  # Line 376
    def require_admin(self, current_user: Any) -> None  # Line 380
    def require_trading_permission(self, current_user: Any) -> None  # Line 384
    def require_risk_manager_permission(self, current_user: Any) -> None  # Line 388
    def health_check(self) -> dict[str, Any]  # Line 392
    def get_service_info(self) -> dict[str, Any]  # Line 396
```

### File: auth.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.web_interface.security.jwt_handler import JWTHandler`

#### Class: `AuthMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Authentication middleware for request processing

```python
class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, jwt_handler: JWTHandler)  # Line 31
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 55
    def _extract_token(self, auth_header: str) -> str | None  # Line 132
    def _add_timing_header(self, response: Response, start_time: float) -> None  # Line 152
    def _add_security_headers(self, response: Response) -> None  # Line 163
    def _log_request_completion(self, request: Request, response: Response, start_time: float) -> None  # Line 180
```

### File: cache.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `CacheEntry`

**Purpose**: Represents a cached response entry

```python
class CacheEntry:
    def __init__(self, data: Any, headers: dict[str, str], expires_at: float, path: str = '')  # Line 26
    def is_expired(self) -> bool  # Line 35
    def touch(self)  # Line 39
```

#### Class: `CacheMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Intelligent caching middleware for API responses

```python
class CacheMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_cache_size: int = 1000, default_ttl: int = 60)  # Line 57
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 144
    def _should_bypass_cache(self, request: Request) -> bool  # Line 190
    def _should_cache_response(self, request: Request, response: Response) -> bool  # Line 220
    def _get_endpoint_config(self, path: str) -> dict[str, Any]  # Line 239
    def _generate_cache_key(self, request: Request) -> str  # Line 264
    def _get_cached_response(self, cache_key: str) -> Response | None  # Line 291
    async def _cache_response(self, cache_key: str, request: Request, response: Response)  # Line 320
    def _evict_lru_entries(self)  # Line 380
    def _handle_cache_invalidation(self, request: Request)  # Line 407
    def _matches_trigger_pattern(self, operation_key: str, pattern: str) -> bool  # Line 428
    def _invalidate_cache_patterns(self, patterns: list[str])  # Line 435
    def _add_cache_headers(self, response: Response, hit: bool = False)  # Line 458
    def get_cache_stats(self) -> dict[str, Any]  # Line 474
    def clear_cache(self, pattern: str | None = None) -> None  # Line 511
```

### File: connection_pool.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.logging import get_logger`
- `from src.database import RedisClient`

#### Class: `PoolAsyncUnitOfWork`

**Purpose**: Simplified async Unit of Work for connection pool usage

```python
class PoolAsyncUnitOfWork:
    def __init__(self, async_session_factory)  # Line 41
    async def __aenter__(self)  # Line 46
    async def __aexit__(self, exc_type, exc_val, exc_tb)  # Line 51
    async def commit(self)  # Line 64
    async def rollback(self)  # Line 69
    async def close(self)  # Line 74
```

#### Class: `ConnectionPoolManager`

**Purpose**: Manages connection pools for database and Redis connections

```python
class ConnectionPoolManager:
    def __init__(self, config: Config)  # Line 89
    async def initialize(self)  # Line 120
    async def _initialize_database_pool(self)  # Line 141
    async def _initialize_redis_pool(self)  # Line 229
    async def get_db_connection(self)  # Line 253
    async def get_uow(self)  # Line 276
    async def get_redis_connection(self)  # Line 294
    async def health_check(self) -> dict[str, Any]  # Line 310
    async def get_pool_stats(self) -> dict[str, Any]  # Line 386
    async def close(self)  # Line 452
```

#### Class: `ConnectionPoolMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to provide connection pool access to requests

```python
class ConnectionPoolMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: Config)  # Line 492
    async def dispatch(self, request: Request, call_next)  # Line 504
    async def startup(self)  # Line 532
    async def shutdown(self)  # Line 536
```

#### Class: `ConnectionHealthMonitor`

**Purpose**: Monitor connection pool health and performance

```python
class ConnectionHealthMonitor:
    def __init__(self, pool_manager: ConnectionPoolManager)  # Line 609
    async def start_monitoring(self)  # Line 629
    async def stop_monitoring(self)  # Line 638
    async def _monitoring_loop(self)  # Line 654
    async def _perform_health_checks(self)  # Line 666
    async def _check_pool_utilization(self, pool_stats: dict[str, Any])  # Line 693
    async def _alert_connection_issue(self, pool_type: str, details: dict[str, Any])  # Line 715
    async def _alert_high_utilization(self, pool_type: str, utilization: float)  # Line 724
```

#### Functions:

```python
def get_global_pool_manager() -> ConnectionPoolManager | None  # Line 545
def set_global_pool_manager(pool_manager: ConnectionPoolManager)  # Line 550
async def get_db_connection()  # Line 556
async def get_redis_connection()  # Line 571
async def get_uow()  # Line 586
```

### File: correlation.py

**Key Imports:**
- `from src.core.logging import correlation_context`
- `from src.core.logging import get_logger`

#### Class: `CorrelationMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to handle correlation IDs for request tracking

```python
class CorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 22
```

### File: decimal_precision.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `DecimalPrecisionMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to handle Decimal precision in API requests and responses

```python
class DecimalPrecisionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, financial_fields: list[str] | None = None)  # Line 33
    async def dispatch(self, request: Request, call_next)  # Line 134
    async def _process_request_body(self, request: Request) -> Request  # Line 157
    async def _process_response_body(self, response: Response) -> Response  # Line 208
    def _convert_to_decimal(self, data: Any) -> Any  # Line 256
    def _is_financial_field(self, field_name: str) -> bool  # Line 344
    def _validate_decimal_precision(self, value: Decimal, field_name: str) -> bool  # Line 367
```

#### Class: `DecimalValidationMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Additional middleware for strict decimal validation in critical trading operations

```python
class DecimalValidationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, critical_endpoints: list[str] | None = None)  # Line 410
    async def dispatch(self, request: Request, call_next)  # Line 427
    async def _validate_request_precision(self, request: Request) -> dict[str, Any]  # Line 454
    def _validate_data_precision(self, data: Any, errors: list[str], path: str)  # Line 484
    def _has_precision_issues(self, value: float, field_name: str) -> bool  # Line 511
```

### File: error_handler.py

**Key Imports:**
- `from src.core.exceptions import AuthenticationError`
- `from src.core.exceptions import ConfigurationError`
- `from src.core.exceptions import ExecutionError`
- `from src.core.exceptions import NetworkError`
- `from src.core.exceptions import TradingBotError`

#### Class: `ErrorHandlerMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Comprehensive error handling middleware

```python
class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, debug: bool = False)  # Line 46
    def _validate_request_boundary(self, request: Request) -> None  # Line 85
    def _validate_response_boundary(self, request: Request, response: Response) -> None  # Line 110
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 136
    async def _handle_tbot_exception(self, request: Request, exception: TradingBotError) -> JSONResponse  # Line 218
    async def _handle_unexpected_exception(self, request: Request, exception: Exception) -> JSONResponse  # Line 293
    def _log_http_exception(self, request: Request, exception: HTTPException) -> None  # Line 367
    def _log_tbot_exception(self, request: Request, exception: TradingBotError, status_code: int) -> None  # Line 394
    def _log_unexpected_exception(self, request: Request, exception: Exception) -> None  # Line 430
    async def _handle_recovered_error(self, request: Request, exception: Exception, error_context: ErrorContext) -> JSONResponse  # Line 465
    def _get_request_id(self, request: Request) -> str  # Line 550
    def _get_current_timestamp(self) -> str  # Line 572
    def get_error_stats(self) -> dict  # Line 583
```

### File: financial_validation.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `FinancialValidationMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware for validating financial inputs on trading endpoints

```python
class FinancialValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 105
    def _is_financial_endpoint(self, path: str) -> bool  # Line 135
    async def _validate_request_body(self, request: Request) -> None  # Line 139
    async def _validate_response_body(self, response: Response) -> None  # Line 163
    def _validate_financial_data(self, data: dict[str, Any] | list[Any], context: str) -> None  # Line 178
    def _validate_financial_value(self, field_name: str, value: Any, context: str) -> None  # Line 201
    def _validate_decimal_precision(self, field_name: str, value: Decimal, context: str) -> None  # Line 245
    def _validate_value_range(self, field_name: str, value: Decimal, context: str) -> None  # Line 264
    def _get_field_type(self, field_name: str) -> str  # Line 288
```

#### Class: `DecimalEnforcementMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to enforce Decimal usage in responses

```python
class DecimalEnforcementMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 315
    def _is_financial_endpoint(self, path: str) -> bool  # Line 329
```

#### Class: `CurrencyValidationMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to validate currency codes and formats

```python
class CurrencyValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 385
    async def _validate_currency_codes(self, request: Request) -> None  # Line 406
    def _check_currency_fields(self, data: Any) -> None  # Line 424
    def _is_financial_endpoint(self, path: str) -> bool  # Line 437
```

### File: rate_limit.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.logging import get_logger`

#### Class: `RateLimitMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Advanced rate limiting middleware

```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: Config)  # Line 33
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 98
    def _get_rate_limit_info(self, request: Request) -> tuple[str, int]  # Line 163
    def _get_endpoint_limit(self, path: str) -> int | None  # Line 202
    def _check_rate_limit(self, key: str, limit: int, current_time: float) -> bool  # Line 223
    def _record_request(self, key: str, current_time: float) -> None  # Line 246
    def _calculate_retry_after(self, key: str, current_time: float) -> int  # Line 256
    def _add_rate_limit_headers(self, response: Response, key: str, limit: int) -> None  # Line 277
    async def _cleanup_old_entries(self) -> None  # Line 303
    def get_rate_limit_stats(self) -> dict  # Line 331
```

### File: security.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `SecurityMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Comprehensive security middleware for web application protection

```python
class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, ...)  # Line 33
    async def dispatch(self, request: Request, call_next)  # Line 126
    def _get_client_ip(self, request: Request) -> str  # Line 168
    async def _validate_request_size(self, request: Request) -> bool  # Line 190
    async def _validate_input(self, request: Request) -> dict[str, Any]  # Line 215
    def _detect_threat_in_value(self, value: str) -> str | None  # Line 255
    def _detect_trading_specific_threats(self, value: str) -> bool  # Line 290
    async def _handle_suspicious_activity(self, client_ip: str, threat_type: str)  # Line 332
    def _add_security_headers(self, response: Response)  # Line 358
    def get_security_stats(self) -> dict[str, Any]  # Line 374
    def unblock_ip(self, ip_address: str) -> bool  # Line 400
    def clear_blocked_ips(self)  # Line 417
```

#### Class: `InputSanitizer`

**Purpose**: Input sanitization utilities for trading data

```python
class InputSanitizer:
    def sanitize_symbol(symbol: str) -> str  # Line 434
    def sanitize_decimal_string(value: str) -> str  # Line 454
    def validate_exchange_name(exchange: str) -> bool  # Line 491
    def validate_order_side(side: str) -> bool  # Line 515
```

### File: auth.py

**Key Imports:**
- `from src.core.config import Config`
- `from src.core.exceptions import AuthenticationError`
- `from src.core.exceptions import ServiceError`
- `from src.core.logging import get_logger`
- `from src.database.models.user import User`

#### Class: `UserInDB`

**Inherits**: BaseModel
**Purpose**: User model for database storage

```python
class UserInDB(BaseModel):
```

#### Class: `User`

**Inherits**: BaseModel
**Purpose**: User model for API responses

```python
class User(BaseModel):
```

#### Class: `Token`

**Inherits**: BaseModel
**Purpose**: Token response model

```python
class Token(BaseModel):
```

#### Class: `LoginRequest`

**Inherits**: BaseModel
**Purpose**: Login request model

```python
class LoginRequest(BaseModel):
```

#### Functions:

```python
def get_auth_service()  # Line 26
def init_auth(config: Config) -> None  # Line 92
def _convert_db_user_to_user_in_db(db_user: DBUser) -> UserInDB  # Line 113
async def get_user(username: str, database_service = None) -> UserInDB | None  # Line 129
async def authenticate_user(username: str, password: str, database_service = None) -> User | None  # Line 149
def create_access_token(user: UserInDB, expires_delta: timedelta | None = None) -> Token  # Line 172
async def get_current_user(credentials: HTTPAuthorizationCredentials = Any) -> User  # Line 213
async def get_current_user_with_scopes(required_scopes: list[str])  # Line 269
async def get_admin_user(current_user: User = Any) -> User  # Line 304
async def get_trading_user(current_user: User = Any) -> User  # Line 322
async def create_user(...) -> User  # Line 342
async def get_auth_summary(database_service = None) -> dict  # Line 374
def require_permissions(required_permissions: list[str])  # Line 414
```

### File: jwt_handler.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.config import Config`
- `from src.core.exceptions import AuthenticationError`
- `from src.core.exceptions import ValidationError`
- `from src.database.redis_client import RedisClient`

#### Class: `TokenData`

**Inherits**: BaseModel
**Purpose**: Token data structure

```python
class TokenData(BaseModel):
```

#### Class: `JWTHandler`

**Inherits**: BaseComponent
**Purpose**: Advanced JWT token handler with security features

```python
class JWTHandler(BaseComponent):
    def __init__(self, config: Config)  # Line 46
    def _init_redis(self)  # Line 93
    def _redis_sync(self, coro)  # Line 133
    def _get_secret_key(self, config: Config) -> str  # Line 160
    def hash_password(self, password: str) -> str  # Line 190
    def verify_password(self, plain_password: str, hashed_password: str) -> bool  # Line 194
    def create_access_token(self, ...) -> str  # Line 202
    def create_refresh_token(self, user_id: str, username: str) -> str  # Line 254
    def validate_token(self, token: str) -> TokenData  # Line 289
    def validate_scopes(self, token_data: TokenData, required_scopes: list[str]) -> bool  # Line 336
    def refresh_access_token(self, refresh_token: str) -> tuple[str, str]  # Line 357
    def revoke_token(self, token: str) -> bool  # Line 408
    def is_token_blacklisted(self, token: str) -> bool  # Line 451
    def get_security_summary(self) -> dict[str, Any]  # Line 485
    def cleanup_expired_blacklist(self) -> int  # Line 497
```

### File: analytics_service.py

**Key Imports:**
- `from src.analytics.interfaces import AlertServiceProtocol`
- `from src.analytics.interfaces import AnalyticsServiceProtocol`
- `from src.analytics.interfaces import ExportServiceProtocol`
- `from src.analytics.interfaces import OperationalServiceProtocol`
- `from src.analytics.interfaces import PortfolioServiceProtocol`

#### Class: `WebAnalyticsService`

**Inherits**: BaseService
**Purpose**: Web interface service for analytics operations

```python
class WebAnalyticsService(BaseService):
    def __init__(self, ...)  # Line 39
    async def _do_start(self) -> None  # Line 65
    async def _do_stop(self) -> None  # Line 71
    async def get_portfolio_metrics(self) -> dict[str, Any]  # Line 80
    async def get_portfolio_composition(self) -> dict[str, Any]  # Line 104
    async def get_correlation_matrix(self) -> Any  # Line 118
    async def export_portfolio_data(self, format: str = 'json', include_metadata: bool = True) -> str  # Line 130
    async def get_risk_metrics(self) -> dict[str, Any]  # Line 148
    async def calculate_var(self, confidence_level: float, time_horizon: int, method: str) -> dict[str, Any]  # Line 171
    async def run_stress_test(self, scenario_name: str, scenario_params: dict[str, Any]) -> dict[str, Any]  # Line 194
    async def get_risk_exposure(self) -> dict[str, Any]  # Line 212
    async def get_strategy_metrics(self, strategy_id: str) -> dict[str, Any] | None  # Line 237
    async def get_strategy_performance(self, strategy_id: str, days: int) -> dict[str, Any]  # Line 268
    async def compare_strategies(self, strategy_ids: list[str]) -> dict[str, Any]  # Line 288
    async def get_operational_metrics(self) -> dict[str, Any]  # Line 311
    async def get_system_errors(self, hours: int) -> list[dict[str, Any]]  # Line 330
    async def get_operational_events(self, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]  # Line 343
    async def generate_report(self, ...) -> dict[str, Any]  # Line 359
    async def get_report(self, report_id: str) -> dict[str, Any] | None  # Line 390
    async def list_reports(self, user_id: str, limit: int) -> list[dict[str, Any]]  # Line 395
    async def schedule_report(self, report_type: str, schedule: str, recipients: list[str], created_by: str) -> dict[str, Any]  # Line 400
    async def get_active_alerts(self) -> list[dict[str, Any]]  # Line 419
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, notes: str | None = None) -> bool  # Line 444
    async def get_alert_history(self, days: int, severity: str | None = None) -> list[dict[str, Any]]  # Line 459
    async def configure_alerts(self, config: dict[str, Any], configured_by: str) -> dict[str, Any]  # Line 466
    def _get_empty_portfolio_metrics(self) -> dict[str, Any]  # Line 477
    def _calculate_strategy_comparison(self, strategies: dict[str, dict[str, Any]]) -> dict[str, Any]  # Line 491
```

### File: auth_service.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.core.exceptions import AuthenticationError`
- `from src.core.exceptions import ServiceError`
- `from src.database.models.user import User`
- `from src.web_interface.interfaces import WebAuthServiceInterface`

#### Class: `WebAuthService`

**Inherits**: BaseService
**Purpose**: Service handling authentication business logic for web interface

```python
class WebAuthService(BaseService):
    def __init__(self, user_repository = None)  # Line 21
    async def _do_start(self) -> None  # Line 25
    async def _do_stop(self) -> None  # Line 29
    async def get_user_by_username(self, username: str) -> UserInDB | None  # Line 33
    async def authenticate_user(self, username: str, password: str) -> User | None  # Line 62
    async def create_user(self, ...) -> User  # Line 89
    async def get_auth_summary(self) -> dict[str, Any]  # Line 139
    def _verify_password(self, password: str, password_hash: str) -> bool  # Line 163
    def _hash_password(self, password: str) -> str  # Line 168
    def _get_user_scopes(self, username: str) -> list[str]  # Line 173
    async def _update_last_login(self, username: str) -> None  # Line 184
    def _convert_db_user_to_user_in_db(self, db_user: Any) -> UserInDB  # Line 195
    def _convert_db_user_to_user(self, db_user: Any) -> User  # Line 210
    def health_check(self) -> dict[str, Any]  # Line 221
    def get_user_roles(self, current_user: Any) -> list[str]  # Line 230
    def check_permission(self, current_user: Any, required_roles: list[str]) -> bool  # Line 245
    def require_permission(self, current_user: Any, required_roles: list[str]) -> None  # Line 250
    def require_admin(self, current_user: Any) -> None  # Line 258
    def require_trading_permission(self, current_user: Any) -> None  # Line 262
    def require_risk_manager_permission(self, current_user: Any) -> None  # Line 266
    def require_admin_or_developer_permission(self, current_user: Any) -> None  # Line 270
    def require_management_permission(self, current_user: Any) -> None  # Line 274
    def require_treasurer_permission(self, current_user: Any) -> None  # Line 278
    def require_operator_permission(self, current_user: Any) -> None  # Line 282
    def get_service_info(self) -> dict[str, Any]  # Line 286
```

### File: bot_service.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import BotConfiguration`
- `from src.core.types import BotPriority`

#### Class: `WebBotService`

**Inherits**: BaseComponent
**Purpose**: Service handling bot management business logic for web interface

```python
class WebBotService(BaseComponent):
    def __init__(self, bot_facade = None)  # Line 24
    async def initialize(self) -> None  # Line 28
    async def cleanup(self) -> None  # Line 32
    async def validate_bot_configuration(self, config_data: dict[str, Any]) -> dict[str, Any]  # Line 36
    async def format_bot_response(self, bot_data: dict[str, Any]) -> dict[str, Any]  # Line 95
    async def get_formatted_bot_list(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 121
    async def calculate_bot_metrics(self, bot_id: str) -> dict[str, Any]  # Line 200
    async def validate_bot_operation(self, bot_id: str, operation: str) -> dict[str, Any]  # Line 279
    async def create_bot_configuration(self, request_data: dict[str, Any], user_id: str) -> BotConfiguration  # Line 320
    async def update_bot_configuration(self, bot_id: str, update_data: dict[str, Any], user_id: str) -> dict[str, Any]  # Line 357
    def _calculate_health_score(self, metrics: dict[str, Any]) -> float  # Line 442
    def health_check(self) -> dict[str, Any]  # Line 469
    def get_service_info(self) -> dict[str, Any]  # Line 478
    async def create_bot_through_service(self, bot_config) -> str  # Line 493
    async def get_bot_status_through_service(self, bot_id: str) -> dict[str, Any]  # Line 523
    async def start_bot_through_service(self, bot_id: str) -> bool  # Line 544
    async def stop_bot_through_service(self, bot_id: str) -> bool  # Line 558
    async def delete_bot_through_service(self, bot_id: str, force: bool = False) -> bool  # Line 572
    async def list_bots_through_service(self) -> list[dict[str, Any]]  # Line 593
    def get_controller_health_check(self) -> dict[str, Any]  # Line 619
    def _get_execution_service(self)  # Line 639
    async def start_bot_with_execution_integration(self, bot_id: str) -> bool  # Line 651
    async def stop_bot_with_execution_integration(self, bot_id: str) -> bool  # Line 679
    async def pause_bot_through_service(self, bot_id: str) -> bool  # Line 699
    async def resume_bot_through_service(self, bot_id: str) -> bool  # Line 713
```

### File: capital_service.py

**Key Imports:**
- `from src.capital_management.interfaces import AbstractCurrencyManagementService`
- `from src.capital_management.interfaces import AbstractFundFlowManagementService`
- `from src.capital_management.interfaces import CapitalServiceProtocol`
- `from src.core.base import BaseService`
- `from src.core.exceptions import ServiceError`

#### Class: `WebCapitalService`

**Inherits**: BaseService
**Purpose**: Web interface service for capital management operations

```python
class WebCapitalService(BaseService):
    def __init__(self, ...)  # Line 34
    async def _do_start(self) -> None  # Line 47
    async def _do_stop(self) -> None  # Line 51
    async def allocate_capital(self, ...) -> dict[str, Any]  # Line 57
    async def release_capital(self, ...) -> bool  # Line 112
    async def update_utilization(self, ...) -> bool  # Line 158
    async def get_allocations(self, ...) -> list[dict[str, Any]]  # Line 197
    async def get_strategy_allocation(self, strategy_id: str, exchange: str | None = None) -> dict[str, Any] | None  # Line 228
    async def get_capital_metrics(self) -> dict[str, Any]  # Line 243
    async def get_utilization_breakdown(self, by: str = 'strategy') -> dict[str, Any]  # Line 263
    async def get_available_capital(self, strategy_id: str | None = None, exchange: str | None = None) -> dict[str, Any]  # Line 323
    async def get_capital_exposure(self) -> dict[str, Any]  # Line 347
    async def get_currency_exposure(self) -> dict[str, Any]  # Line 389
    async def create_currency_hedge(self, ...) -> dict[str, Any]  # Line 439
    async def get_currency_rates(self, base_currency: str = 'USD') -> dict[str, float]  # Line 476
    async def get_fund_flows(self, days: int = 30, flow_type: str | None = None) -> list[dict[str, Any]]  # Line 498
    async def record_fund_flow(self, ...) -> dict[str, Any]  # Line 531
    async def generate_fund_flow_report(self, start_date: datetime | None, end_date: datetime | None, format: str) -> dict[str, Any]  # Line 581
    async def get_allocation_limits(self, limit_type: str | None = None) -> list[dict[str, Any]]  # Line 615
    async def set_allocation_limits(self, **kwargs) -> dict[str, Any]  # Line 619
    async def get_capital_limits(self, limit_type: str | None = None) -> list[dict[str, Any]]  # Line 630
    async def update_capital_limits(self, ...) -> bool  # Line 661
    async def get_limit_breaches(self, hours: int = 24, severity: str | None = None) -> list[dict[str, Any]]  # Line 681
    def _format_allocation(self, allocation: CapitalAllocation) -> dict[str, Any]  # Line 695
    async def _get_available_capital_amount(self) -> Decimal  # Line 710
    def _calculate_concentration_risk(self, exposure_by_strategy: dict[str, Decimal]) -> float  # Line 715
    async def convert_currency(self, from_currency: str, to_currency: str, amount: Decimal) -> dict[str, Any]  # Line 734
    async def rebalance_portfolio(self, target_allocations: dict[str, Any] | None = None, dry_run: bool = False) -> dict[str, Any]  # Line 762
    async def calculate_optimal_allocation(self, risk_tolerance: str, optimization_method: str) -> dict[str, Any]  # Line 785
    async def get_capital_efficiency(self) -> dict[str, Any]  # Line 810
    async def reserve_capital(self, amount: Decimal, currency: str, purpose: str, duration_minutes: int = 60) -> dict[str, Any]  # Line 837
    async def get_reserved_capital(self) -> dict[str, Any]  # Line 862
```

### File: data_service.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.data.interfaces import DataServiceInterface`

#### Class: `WebDataService`

**Inherits**: BaseService
**Purpose**: Web interface service for data management operations

```python
class WebDataService(BaseService):
    def __init__(self, ...)  # Line 28
    async def _do_start(self) -> None  # Line 52
    async def _do_stop(self) -> None  # Line 58
    async def get_pipeline_status(self, pipeline_id: str | None = None) -> Any  # Line 66
    async def control_pipeline(self, ...) -> dict[str, Any]  # Line 79
    async def get_pipeline_metrics(self, hours: int) -> dict[str, Any]  # Line 115
    async def get_data_quality_metrics(self, data_source: str | None = None, symbol: str | None = None) -> dict[str, Any]  # Line 138
    async def get_validation_report(self, days: int) -> dict[str, Any]  # Line 165
    async def validate_data(self, ...) -> dict[str, Any]  # Line 194
    async def get_data_anomalies(self, hours: int = 24, severity: str | None = None) -> list[dict[str, Any]]  # Line 231
    async def list_features(self, category: str | None = None, active_only: bool = True) -> list[dict[str, Any]]  # Line 270
    async def get_feature_details(self, feature_id: str) -> dict[str, Any] | None  # Line 314
    async def compute_features(self, ...) -> dict[str, Any]  # Line 339
    async def get_feature_metadata(self) -> dict[str, Any]  # Line 376
    async def list_data_sources(self, source_type: str | None = None, enabled_only: bool = True) -> list[dict[str, Any]]  # Line 397
    async def configure_data_source(self, ...) -> str  # Line 449
    async def update_data_source(self, source_id: str, config: dict[str, Any], updated_by: str) -> bool  # Line 479
    async def delete_data_source(self, source_id: str, deleted_by: str) -> bool  # Line 495
    async def get_data_health(self) -> dict[str, Any]  # Line 511
    async def get_data_latency(self, source: str | None = None, hours: int = 1) -> dict[str, Any]  # Line 533
    async def get_data_throughput(self, source: str | None = None, hours: int = 1) -> dict[str, Any]  # Line 550
    async def clear_data_cache(self, cache_type: str | None = None) -> int  # Line 570
    async def get_cache_statistics(self) -> dict[str, Any]  # Line 600
    def _get_mock_pipeline_status(self, pipeline_id: str | None) -> Any  # Line 622
    def _validate_source_config(self, source_type: str, config: dict[str, Any]) -> None  # Line 663
```

### File: exchange_service.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.exchanges.interfaces import IExchangeFactory`

#### Class: `WebExchangeService`

**Inherits**: BaseService
**Purpose**: Web interface service for exchange management operations

```python
class WebExchangeService(BaseService):
    def __init__(self, ...)  # Line 30
    async def _do_start(self) -> None  # Line 47
    async def _do_stop(self) -> None  # Line 53
    async def get_connections(self) -> list[dict[str, Any]]  # Line 61
    async def connect_exchange(self, ...) -> dict[str, Any]  # Line 97
    async def disconnect_exchange(self, exchange: str, disconnected_by: str) -> bool  # Line 152
    async def get_exchange_status(self, exchange: str) -> dict[str, Any] | None  # Line 170
    async def get_exchange_config(self, exchange: str) -> dict[str, Any] | None  # Line 204
    async def validate_exchange_config(self, exchange: str, config: dict[str, Any]) -> dict[str, Any]  # Line 229
    async def update_exchange_config(self, exchange: str, config: dict[str, Any], updated_by: str) -> bool  # Line 252
    async def get_exchange_symbols(self, exchange: str, active_only: bool = True) -> list[str]  # Line 268
    async def get_exchange_fees(self, exchange: str, symbol: str | None = None) -> dict[str, Any]  # Line 293
    async def get_rate_limits(self, exchange: str) -> dict[str, Any]  # Line 316
    async def get_rate_usage(self, exchange: str) -> dict[str, Any]  # Line 332
    async def update_rate_config(self, ...) -> bool  # Line 349
    async def get_all_exchanges_health(self) -> dict[str, Any]  # Line 369
    async def get_exchange_health(self, exchange: str) -> dict[str, Any] | None  # Line 393
    async def get_exchange_latency(self, exchange: str, hours: int) -> dict[str, Any]  # Line 416
    async def get_exchange_errors(self, exchange: str, hours: int, error_type: str | None = None) -> list[dict[str, Any]]  # Line 433
    async def get_orderbook(self, exchange: str, symbol: str, limit: int) -> dict[str, Any] | None  # Line 466
    async def get_ticker(self, exchange: str, symbol: str) -> dict[str, Any] | None  # Line 497
    async def get_exchange_balance(self, exchange: str, user_id: str) -> dict[str, dict[str, Decimal]]  # Line 527
    async def subscribe_websocket(self, exchange: str, channel: str, symbols: list[str], subscriber: str) -> str  # Line 555
    async def unsubscribe_websocket(self, exchange: str, subscription_id: str, subscriber: str) -> bool  # Line 578
    def _validate_api_key(self, api_key: str) -> bool  # Line 591
```

### File: monitoring_service.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ServiceError`
- `from src.web_interface.interfaces import WebMonitoringServiceInterface`

#### Class: `WebMonitoringService`

**Inherits**: BaseComponent
**Purpose**: Service handling monitoring business logic for web interface

```python
class WebMonitoringService(BaseComponent):
    def __init__(self, monitoring_facade = None)  # Line 19
    async def initialize(self) -> None  # Line 23
    async def cleanup(self) -> None  # Line 27
    async def get_system_health_summary(self) -> dict[str, Any]  # Line 31
    async def get_performance_metrics(self, component: str | None = None) -> dict[str, Any]  # Line 91
    async def get_error_summary(self, time_range: str = '24h') -> dict[str, Any]  # Line 168
    async def get_alert_dashboard_data(self) -> dict[str, Any]  # Line 244
    def _calculate_overall_health_score(self, health_data: dict[str, Any]) -> float  # Line 327
    def _format_component_health(self, components: dict[str, Any]) -> dict[str, Any]  # Line 357
    def _format_uptime(self, uptime_seconds: int) -> str  # Line 368
    def _calculate_performance_score(self, component: str, metrics: dict[str, Any]) -> float  # Line 383
    def _parse_time_range(self, time_range: str) -> int  # Line 425
    def _analyze_error_patterns(self, error_data: dict[str, Any]) -> dict[str, Any]  # Line 437
    def _categorize_alerts(self, alerts: list[dict[str, Any]]) -> dict[str, Any]  # Line 462
    def _get_severity_priority(self, severity: str) -> int  # Line 495
    def _get_status_icon(self, status: str) -> str  # Line 500
    def health_check(self) -> dict[str, Any]  # Line 505
    def get_service_info(self) -> dict[str, Any]  # Line 514
```

### File: portfolio_service.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ServiceError`
- `from src.core.logging import get_logger`
- `from src.web_interface.interfaces import WebPortfolioServiceInterface`

#### Class: `WebPortfolioService`

**Inherits**: BaseComponent
**Purpose**: Service handling portfolio business logic for web interface

```python
class WebPortfolioService(BaseComponent):
    def __init__(self, portfolio_facade = None)  # Line 23
    async def initialize(self) -> None  # Line 27
    async def cleanup(self) -> None  # Line 31
    async def get_portfolio_summary_data(self) -> dict[str, Any]  # Line 35
    async def calculate_pnl_periods(self, total_pnl: Decimal, total_trades: int, win_rate: float) -> dict[str, dict[str, Any]]  # Line 86
    async def get_processed_positions(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]  # Line 152
    async def calculate_pnl_metrics(self, period: str) -> dict[str, Any]  # Line 231
    def generate_mock_balances(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]  # Line 308
    def calculate_asset_allocation(self) -> list[dict[str, Any]]  # Line 358
    def generate_performance_chart_data(self, period: str, resolution: str) -> dict[str, Any]  # Line 392
```

### File: risk_service.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.risk_management.interfaces import RiskServiceInterface`

#### Class: `WebRiskService`

**Inherits**: BaseComponent
**Purpose**: Service handling risk management business logic for web interface

```python
class WebRiskService(BaseComponent):
    def __init__(self, risk_service: RiskServiceInterface = None)  # Line 33
    async def initialize(self) -> None  # Line 37
    async def cleanup(self) -> None  # Line 41
    async def get_risk_dashboard_data(self) -> dict[str, Any]  # Line 45
    async def validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]  # Line 159
    async def calculate_position_risk(self, symbol: str, quantity: Decimal, price: Decimal) -> dict[str, Any]  # Line 230
    async def get_portfolio_risk_breakdown(self) -> dict[str, Any]  # Line 388
    def _analyze_risk_levels(self, risk_data: dict[str, Any]) -> dict[str, Any]  # Line 522
    def _calculate_exposure_percentage(self, exposure: Decimal, portfolio_value: Decimal) -> Decimal  # Line 574
    def _calculate_drawdown_percentage(self, drawdown: Decimal, portfolio_value: Decimal) -> Decimal  # Line 582
    def _assess_position_risk_level(self, position_percentage: Decimal, volatility: Decimal) -> str  # Line 590
    def _generate_risk_recommendations(self, parameters: dict[str, Any]) -> list[str]  # Line 601
    def _generate_position_warnings(self, position_percentage: Decimal, volatility: Decimal) -> list[str]  # Line 615
    def _generate_position_recommendations(self, symbol: str, position_percentage: Decimal) -> list[str]  # Line 633
    def _analyze_portfolio_risk_distribution(self, positions: list[dict[str, Any]]) -> dict[str, Any]  # Line 651
    def _get_risk_level_by_volatility(self, volatility: float) -> str  # Line 699
    def health_check(self) -> dict[str, Any]  # Line 710
    def generate_mock_risk_alerts(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 719
    def generate_mock_position_risks(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 796
    def generate_mock_correlation_matrix(self, symbols: list[str], period: str) -> dict[str, Any]  # Line 870
    def generate_mock_stress_test_results(self, test_request: dict[str, Any]) -> dict[str, Any]  # Line 919
    async def validate_risk_parameters_v2(self, parameters: dict[str, Any]) -> dict[str, Any]  # Line 982
    async def get_current_risk_limits(self) -> dict[str, Any]  # Line 1066
    def get_service_info(self) -> dict[str, Any]  # Line 1092
```

### File: strategy_service.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ServiceError`
- `from src.web_interface.interfaces import WebStrategyServiceInterface`

#### Class: `WebStrategyService`

**Inherits**: BaseComponent
**Purpose**: Service handling strategy business logic for web interface

```python
class WebStrategyService(BaseComponent):
    def __init__(self, strategy_service = None)  # Line 23
    async def initialize(self) -> None  # Line 27
    async def cleanup(self) -> None  # Line 31
    async def get_formatted_strategies(self) -> list[dict[str, Any]]  # Line 35
    async def validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> dict[str, Any]  # Line 154
    async def get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]  # Line 203
    async def format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]  # Line 303
    async def health_check(self) -> 'HealthCheckResult'  # Line 343
    def get_service_info(self) -> dict[str, Any]  # Line 352
    async def get_strategy_config_through_service(self, strategy_name: str) -> dict[str, Any]  # Line 366
    async def validate_strategy_config_through_service(self, strategy_name: str, parameters: dict[str, Any]) -> bool  # Line 404
```

### File: trading_service.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import OrderSide`
- `from src.core.types import OrderType`

#### Class: `WebTradingService`

**Inherits**: BaseComponent
**Purpose**: Service handling trading business logic for web interface

```python
class WebTradingService(BaseComponent):
    def __init__(self, trading_facade = None)  # Line 24
    async def initialize(self) -> None  # Line 28
    async def cleanup(self) -> None  # Line 32
    async def place_order_through_service(self, ...) -> dict[str, Any]  # Line 36
    async def cancel_order_through_service(self, order_id: str) -> bool  # Line 89
    async def get_service_health(self) -> dict[str, Any]  # Line 103
    async def get_order_details(self, order_id: str, exchange: str) -> dict[str, Any]  # Line 124
    async def validate_order_request(self, ...) -> dict[str, Any]  # Line 170
    async def format_order_response(self, order_result: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]  # Line 259
    async def get_formatted_orders(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 294
    async def get_formatted_trades(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 369
    async def get_market_data_with_context(self, symbol: str, exchange: str = 'binance') -> dict[str, Any]  # Line 432
    async def generate_order_book_data(self, symbol: str, exchange: str, depth: int) -> dict[str, Any]  # Line 474
    def health_check(self) -> dict[str, Any]  # Line 513
    def get_service_info(self) -> dict[str, Any]  # Line 522
    def _get_current_timestamp(self) -> str  # Line 538
```

### File: socketio_manager.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.data_transformer import CoreDataTransformer`
- `from src.core.events import BotEvent`
- `from src.core.events import BotEventType`
- `from src.core.exceptions import AuthenticationError`

#### Class: `TradingNamespace`

**Inherits**: AsyncNamespace
**Purpose**: Main namespace for trading-related Socket

```python
class TradingNamespace(AsyncNamespace):
    def __init__(self, namespace: str = '/', jwt_handler = None)  # Line 43
    async def emit_standardized(self, ...) -> None  # Line 67
    async def on_connect(self, ...)  # Line 103
    async def on_disconnect(self, sid: str)  # Line 231
    async def on_authenticate(self, sid: str, data: dict[str, Any])  # Line 280
    async def on_subscribe(self, sid: str, data: dict[str, Any])  # Line 302
    async def on_unsubscribe(self, sid: str, data: dict[str, Any])  # Line 324
    async def on_ping(self, sid: str, data: dict[str, Any] | None = None)  # Line 338
    async def on_execute_order(self, sid: str, data: dict[str, Any])  # Line 351
    async def on_get_portfolio(self, sid: str, data: dict[str, Any])  # Line 442
    async def _validate_token(self, token: str) -> dict[str, Any] | None  # Line 466
    def _require_scope(self, sid: str, required_scope: str) -> bool  # Line 514
```

#### Class: `SocketIOManager`

**Inherits**: BaseComponent
**Purpose**: Manager for Socket

```python
class SocketIOManager(BaseComponent):
    def __init__(self)  # Line 535
    def create_server(self, cors_allowed_origins: list[str] | None = None) -> AsyncServer  # Line 541
    def create_app(self)  # Line 570
    async def start_background_tasks(self)  # Line 578
    async def stop_background_tasks(self)  # Line 599
    async def _broadcast_market_data(self)  # Line 640
    async def _broadcast_bot_status(self)  # Line 683
    async def _broadcast_portfolio_updates(self)  # Line 720
    async def emit_to_user(self, user_id: str, event: str, data: Any)  # Line 750
    async def broadcast(self, event: str, data: Any, room: str | None = None)  # Line 757
```

### File: decorators.py

#### Functions:

```python
def versioned_endpoint(...)  # Line 17
def deprecated(version: str, removal_version: str | None = None, message: str | None = None)  # Line 138
def feature_flag(feature_name: str, default: bool = True)  # Line 198
def version_specific(versions: list[str])  # Line 247
```

### File: middleware.py

#### Class: `VersioningMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to handle API versioning

```python
class VersioningMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, version_header: str = 'X-API-Version')  # Line 21
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 29
    def _extract_version(self, request: Request) -> str  # Line 77
    def _add_version_headers(self, response: Response, request: Request) -> None  # Line 110
```

#### Class: `VersionRoutingMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to handle version-specific routing

```python
class VersionRoutingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app)  # Line 174
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 178
```

### File: version_manager.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ValidationError`

#### Class: `VersionStatus`

**Inherits**: Enum
**Purpose**: API version status

```python
class VersionStatus(Enum):
```

#### Class: `APIVersion`

**Purpose**: API version configuration

```python
class APIVersion:
    def __post_init__(self)  # Line 41
    def __str__(self) -> str  # Line 47
    def __lt__(self, other: 'APIVersion') -> bool  # Line 50
    def __eq__(self, other: object) -> bool  # Line 54
    def is_compatible_with(self, other: 'APIVersion') -> bool  # Line 60
    def is_deprecated(self) -> bool  # Line 73
    def is_sunset(self) -> bool  # Line 77
```

#### Class: `VersionManager`

**Inherits**: BaseComponent
**Purpose**: Manager for API versioning and compatibility

```python
class VersionManager(BaseComponent):
    def __init__(self)  # Line 85
    def _initialize_default_versions(self) -> None  # Line 95
    def register_version(self, version: APIVersion) -> None  # Line 156
    def parse_version(self, version_string: str) -> APIVersion | None  # Line 161
    def get_version(self, version: str) -> APIVersion | None  # Line 173
    def get_latest_version(self) -> APIVersion | None  # Line 177
    def get_default_version(self) -> APIVersion | None  # Line 183
    def list_versions(self, include_deprecated: bool = True) -> list[APIVersion]  # Line 189
    def resolve_version(self, requested_version: str | None = None) -> APIVersion  # Line 196
    def check_feature_availability(self, version: str, feature: str) -> bool  # Line 239
    def get_deprecation_info(self, version: str) -> dict[str, Any] | None  # Line 246
    def deprecate_version(self, version: str, sunset_date: datetime | None = None) -> bool  # Line 263
    def sunset_version(self, version: str) -> bool  # Line 277
    def get_version_migration_guide(self, from_version: str, to_version: str) -> dict[str, Any]  # Line 289
```

#### Functions:

```python
def get_version_manager() -> VersionManager  # Line 331
```

### File: bot_status.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `BotStatusManager`

**Purpose**: Manages WebSocket connections for bot status updates

```python
class BotStatusManager:
    def __init__(self)  # Line 27
    async def connect(self, websocket: WebSocket, user_id: str)  # Line 32
    def disconnect(self, user_id: str)  # Line 47
    def subscribe_to_bot(self, user_id: str, bot_id: str)  # Line 64
    def unsubscribe_from_bot(self, user_id: str, bot_id: str)  # Line 75
    def subscribe_to_all_bots(self, user_id: str)  # Line 87
    async def send_to_user(self, user_id: str, message: dict)  # Line 96
    async def broadcast_to_bot_subscribers(self, bot_id: str, message: dict)  # Line 114
    async def _send_with_timeout(self, websocket: WebSocket, message: dict, user_id: str)  # Line 162
    async def _send_with_timeout_broadcast(self, websocket: WebSocket, message_json: str, user_id: str)  # Line 171
```

#### Class: `BotStatusMessage`

**Inherits**: BaseModel
**Purpose**: Base model for bot status messages

```python
class BotStatusMessage(BaseModel):
```

#### Class: `BotSubscriptionMessage`

**Inherits**: BaseModel
**Purpose**: Model for bot subscription messages

```python
class BotSubscriptionMessage(BaseModel):
```

#### Functions:

```python
async def bot_status_websocket(websocket: WebSocket)  # Line 204
async def send_initial_bot_status(user_id: str, bot_id: str, update_types: list[str])  # Line 329
async def send_all_bot_status(user_id: str)  # Line 398
async def bot_status_simulator()  # Line 407
async def get_bot_status_connections()  # Line 504
```

### File: market_data.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.web_interface.security.auth import User`

#### Class: `ConnectionManager`

**Purpose**: Manages WebSocket connections for market data streaming

```python
class ConnectionManager:
    def __init__(self)  # Line 26
    async def connect(self, websocket: WebSocket, user_id: str)  # Line 31
    def disconnect(self, user_id: str)  # Line 46
    def subscribe_to_symbol(self, user_id: str, symbol: str)  # Line 63
    def unsubscribe_from_symbol(self, user_id: str, symbol: str)  # Line 74
    async def send_to_user(self, user_id: str, message: dict)  # Line 86
    async def broadcast_to_symbol_subscribers(self, symbol: str, message: dict)  # Line 105
    async def _send_with_timeout(self, websocket: WebSocket, message: dict, user_id: str)  # Line 153
    async def _send_with_timeout_broadcast(self, websocket: WebSocket, message_json: str, user_id: str)  # Line 162
```

#### Class: `MarketDataMessage`

**Inherits**: BaseModel
**Purpose**: Base model for market data messages

```python
class MarketDataMessage(BaseModel):
```

#### Class: `SubscriptionMessage`

**Inherits**: BaseModel
**Purpose**: Model for subscription messages

```python
class SubscriptionMessage(BaseModel):
```

#### Functions:

```python
async def authenticate_websocket(websocket: WebSocket) -> User | None  # Line 194
async def market_data_websocket(websocket: WebSocket)  # Line 265
async def send_initial_market_data(user_id: str, symbol: str, data_types: list[str])  # Line 378
async def market_data_simulator()  # Line 453
async def get_market_data_status()  # Line 496
```

### File: portfolio.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `PortfolioManager`

**Purpose**: Manages WebSocket connections for portfolio updates

```python
class PortfolioManager:
    def __init__(self)  # Line 28
    async def connect(self, websocket: WebSocket, user_id: str)  # Line 35
    def disconnect(self, user_id: str)  # Line 59
    def subscribe_to_updates(self, user_id: str, update_types: list[str])  # Line 68
    def unsubscribe_from_updates(self, user_id: str, update_types: list[str])  # Line 76
    async def send_to_user(self, user_id: str, message: dict)  # Line 87
    async def broadcast_to_all(self, message: dict, update_type: str)  # Line 106
    async def _send_with_timeout(self, websocket: WebSocket, message: dict, user_id: str)  # Line 158
    async def _send_with_timeout_broadcast(self, websocket: WebSocket, message_json: str, user_id: str)  # Line 167
```

#### Class: `PortfolioMessage`

**Inherits**: BaseModel
**Purpose**: Base model for portfolio messages

```python
class PortfolioMessage(BaseModel):
```

#### Class: `PortfolioSubscriptionMessage`

**Inherits**: BaseModel
**Purpose**: Model for portfolio subscription messages

```python
class PortfolioSubscriptionMessage(BaseModel):
```

#### Functions:

```python
async def portfolio_websocket(websocket: WebSocket)  # Line 198
async def send_initial_portfolio_data(user_id: str, update_types: list[str])  # Line 304
async def portfolio_simulator()  # Line 424
async def get_portfolio_connections()  # Line 536
```

### File: public.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Functions:

```python
async def public_websocket(websocket: WebSocket, token: str | None = Any)  # Line 22
```

### File: unified_manager.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.logging import correlation_context`

#### Class: `ChannelType`

**Inherits**: Enum
**Purpose**: WebSocket channel types

```python
class ChannelType(Enum):
```

#### Class: `SubscriptionLevel`

**Inherits**: Enum
**Purpose**: Subscription permission levels

```python
class SubscriptionLevel(Enum):
```

#### Class: `WebSocketEventHandler`

**Purpose**: Base class for WebSocket event handlers

```python
class WebSocketEventHandler:
    def __init__(self, ...)  # Line 51
    async def subscribe(self, session_id: str) -> bool  # Line 61
    async def unsubscribe(self, session_id: str) -> bool  # Line 66
    async def start(self) -> None  # Line 71
    async def stop(self) -> None  # Line 75
    async def emit_to_channel(self, sio: AsyncServer, event: str, data: Any) -> None  # Line 99
```

#### Class: `MarketDataHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for market data WebSocket events

```python
class MarketDataHandler(WebSocketEventHandler):
    def __init__(self)  # Line 126
    async def start(self) -> None  # Line 130
    async def _broadcast_loop(self) -> None  # Line 135
    async def _emit_data(self, event: str, data: Any) -> None  # Line 165
```

#### Class: `BotStatusHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for bot status WebSocket events

```python
class BotStatusHandler(WebSocketEventHandler):
    def __init__(self)  # Line 173
    async def start(self) -> None  # Line 177
    async def _broadcast_loop(self) -> None  # Line 182
    async def _emit_data(self, event: str, data: Any) -> None  # Line 214
```

#### Class: `PortfolioHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for portfolio WebSocket events

```python
class PortfolioHandler(WebSocketEventHandler):
    def __init__(self)  # Line 222
    async def start(self) -> None  # Line 226
    async def _broadcast_loop(self) -> None  # Line 231
    async def _emit_data(self, event: str, data: Any) -> None  # Line 264
```

#### Class: `TradesHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for trade execution WebSocket events

```python
class TradesHandler(WebSocketEventHandler):
    def __init__(self)  # Line 272
    async def start(self) -> None  # Line 276
    async def _broadcast_loop(self) -> None  # Line 281
    async def _emit_data(self, event: str, data: Any) -> None  # Line 315
```

#### Class: `OrdersHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for order status WebSocket events

```python
class OrdersHandler(WebSocketEventHandler):
    def __init__(self)  # Line 323
    async def start(self) -> None  # Line 327
    async def _broadcast_loop(self) -> None  # Line 332
    async def _emit_data(self, event: str, data: Any) -> None  # Line 368
```

#### Class: `AlertsHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for alerts and notifications WebSocket events

```python
class AlertsHandler(WebSocketEventHandler):
    def __init__(self)  # Line 376
    async def start(self) -> None  # Line 380
    async def _broadcast_loop(self) -> None  # Line 385
    async def _emit_data(self, event: str, data: Any) -> None  # Line 420
```

#### Class: `LogsHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for system logs WebSocket events

```python
class LogsHandler(WebSocketEventHandler):
    def __init__(self)  # Line 428
    async def start(self) -> None  # Line 432
    async def _broadcast_loop(self) -> None  # Line 437
    async def _emit_data(self, event: str, data: Any) -> None  # Line 473
```

#### Class: `RiskMetricsHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for risk metrics WebSocket events

```python
class RiskMetricsHandler(WebSocketEventHandler):
    def __init__(self)  # Line 481
    async def start(self) -> None  # Line 485
    async def _broadcast_loop(self) -> None  # Line 490
    async def _emit_data(self, event: str, data: Any) -> None  # Line 538
```

#### Class: `UnifiedWebSocketNamespace`

**Inherits**: AsyncNamespace
**Purpose**: Unified namespace for all WebSocket communications

```python
class UnifiedWebSocketNamespace(AsyncNamespace):
    def __init__(self, namespace: str = '/')  # Line 546
    def _create_emit_function(self, channel: ChannelType) -> Callable  # Line 569
    async def on_connect(self, ...)  # Line 578
    async def on_disconnect(self, sid: str)  # Line 615
    async def on_authenticate(self, sid: str, data: dict[str, Any])  # Line 644
    async def _authenticate_session(self, sid: str, token: str) -> bool  # Line 653
    async def on_subscribe(self, sid: str, data: dict[str, Any])  # Line 710
    async def on_unsubscribe(self, sid: str, data: dict[str, Any])  # Line 742
    async def _subscribe_to_channel(self, sid: str, channel: ChannelType) -> bool  # Line 759
    async def _unsubscribe_from_channel(self, sid: str, channel: ChannelType) -> bool  # Line 780
    def _is_authenticated(self, sid: str) -> bool  # Line 797
    def _has_channel_permission(self, sid: str, channel: ChannelType) -> bool  # Line 801
    async def on_ping(self, sid: str, data: dict[str, Any] | None = None)  # Line 824
    async def start_handlers(self)  # Line 837
    async def stop_handlers(self)  # Line 842
```

#### Class: `UnifiedWebSocketManager`

**Inherits**: BaseComponent
**Purpose**: Unified manager for all WebSocket communications

```python
class UnifiedWebSocketManager(BaseComponent):
    def __init__(self, api_facade = None)  # Line 864
    def configure_dependencies(self, injector)  # Line 871
    def create_server(self, cors_allowed_origins: list[str] | None = None) -> AsyncServer  # Line 881
    async def start(self)  # Line 911
    async def stop(self)  # Line 934
    def get_connection_stats(self) -> dict[str, Any]  # Line 953
```

#### Functions:

```python
def get_unified_websocket_manager(api_facade: 'APIFacade' = None) -> UnifiedWebSocketManager  # Line 973
```

---
**Generated**: Complete reference for web_interface module
**Total Classes**: 198
**Total Functions**: 386