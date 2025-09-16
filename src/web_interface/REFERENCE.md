# WEB_INTERFACE Module Reference

## INTEGRATION
**Dependencies**: analytics, capital_management, core, data, database, error_handling, exchanges, execution, monitoring, optimization, state, utils
**Used By**: None
**Provides**: AuthManager, BotStatusManager, ConnectionManager, ConnectionPoolManager, PortfolioManager, SocketIOManager, UnifiedWebSocketManager, VersionManager, WebAnalyticsService, WebBotService, WebCapitalService, WebDataService, WebExchangeService, WebMonitoringService, WebPortfolioService, WebRiskService, WebStrategyService, WebTradingService
**Patterns**: Async Operations, Component Architecture, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Decimal precision arithmetic
- Database decimal columns
**Security**:
- Authentication
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
**Files**: 60 Python files
**Classes**: 201
**Functions**: 362

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

**Implemented Methods:**
- `validate_amount(cls, v)` - Line 39

### Implementation: `CapitalReleaseRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for capital release
**Status**: Complete

**Implemented Methods:**
- `validate_amount(cls, v)` - Line 60

### Implementation: `CapitalUtilizationUpdate` ✅

**Inherits**: BaseModel
**Purpose**: Request model for utilization update
**Status**: Complete

**Implemented Methods:**
- `validate_amount(cls, v)` - Line 80

### Implementation: `CurrencyHedgeRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for currency hedging
**Status**: Complete

### Implementation: `FundFlowRequest` ✅

**Inherits**: BaseModel
**Purpose**: Request model for recording fund flows
**Status**: Complete

**Implemented Methods:**
- `validate_amount(cls, v)` - Line 113

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
- `async get_health_status(self) -> dict` - Line 31

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
- `validate_parameter_ranges(cls, v)` - Line 134
- `validate_end_date(cls, v, info)` - Line 141

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
**Purpose**: Enhanced playground configuration model
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
- `async authenticate(self, credentials: dict[str, Any], provider_type: str | None = None) -> tuple[User, AuthToken] | None` - Line 121
- `async validate_token(self, token_value: str, provider_type: str | None = None) -> User | None` - Line 171
- `async revoke_token(self, token_value: str, provider_type: str | None = None) -> bool` - Line 202
- `async refresh_token(self, refresh_token_value: str) -> AuthToken | None` - Line 226
- `async create_api_key(self, user: User) -> AuthToken | None` - Line 246
- `get_user(self, user_id: str) -> User | None` - Line 266
- `get_user_by_username(self, username: str) -> User | None` - Line 270
- `async create_user(self, user_data: dict[str, Any]) -> User | None` - Line 277
- `async update_user(self, user_id: str, updates: dict[str, Any]) -> bool` - Line 320
- `async change_password(self, user_id: str, old_password: str, new_password: str) -> bool` - Line 356
- `get_user_stats(self) -> dict[str, Any]` - Line 383

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

### Implementation: `WebInterfaceDataTransformer` ✅

**Purpose**: Handles consistent data transformation for web_interface module
**Status**: Complete

**Implemented Methods:**
- `transform_order_request_to_event_data(symbol, ...) -> dict[str, Any]` - Line 19
- `transform_risk_data_to_event_data(risk_data: dict[str, Any], metadata: dict[str, Any] = None) -> dict[str, Any]` - Line 55
- `transform_portfolio_data_to_event_data(portfolio_data: dict[str, Any], metadata: dict[str, Any] = None) -> dict[str, Any]` - Line 79
- `transform_error_to_event_data(error, ...) -> dict[str, Any]` - Line 103
- `validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]` - Line 131
- `ensure_boundary_fields(data: dict[str, Any], source: str = 'web_interface') -> dict[str, Any]` - Line 161
- `transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] = None) -> dict[str, Any]` - Line 195
- `transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str = None) -> dict[str, Any]` - Line 247
- `align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]` - Line 275

### Implementation: `APIFacade` ✅

**Inherits**: BaseComponent
**Purpose**: Unified API facade for T-Bot Trading System
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 62
- `async cleanup(self) -> None` - Line 72
- `async place_order(self, ...) -> str` - Line 82
- `async cancel_order(self, order_id: str) -> bool` - Line 126
- `async get_positions(self) -> list[Position]` - Line 136
- `async create_bot(self, config: BotConfiguration) -> str` - Line 142
- `async start_bot(self, bot_id: str) -> bool` - Line 147
- `async stop_bot(self, bot_id: str) -> bool` - Line 152
- `async get_bot_status(self, bot_id: str) -> dict[str, Any]` - Line 157
- `async list_bots(self) -> list[dict[str, Any]]` - Line 162
- `async get_ticker(self, symbol: str) -> MarketData` - Line 168
- `async subscribe_to_ticker(self, symbol: str, callback: Callable) -> None` - Line 173
- `async unsubscribe_from_ticker(self, symbol: str) -> None` - Line 178
- `async get_balance(self) -> dict[str, Decimal]` - Line 184
- `async get_portfolio_summary(self) -> dict[str, Any]` - Line 189
- `async get_pnl_report(self, start_date: datetime, end_date: datetime) -> dict[str, Any]` - Line 194
- `async validate_order(self, ...) -> bool` - Line 200
- `async get_risk_summary(self) -> dict[str, Any]` - Line 217
- `async get_risk_metrics(self) -> dict[str, Any]` - Line 222
- `async calculate_position_size(self, ...) -> Decimal` - Line 227
- `get_current_risk_level(self)` - Line 243
- `is_emergency_stop_active(self) -> bool` - Line 248
- `async list_strategies(self) -> list[dict[str, Any]]` - Line 254
- `async get_strategy_config(self, strategy_name: str) -> dict[str, Any]` - Line 259
- `async validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool` - Line 264
- `async health_check(self) -> dict[str, Any]` - Line 270
- `get_service_status(self, service_name: str) -> dict[str, Any]` - Line 290
- `async delete_bot(self, bot_id: str, force: bool = False) -> bool` - Line 303

### Implementation: `ServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Base interface for all services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 19
- `async cleanup(self) -> None` - Line 24

### Implementation: `ServiceRegistry` ✅

**Inherits**: BaseComponent
**Purpose**: Central registry for all system services
**Status**: Complete

**Implemented Methods:**
- `register_service(self, name: str, service: Any, interface: type | None = None) -> None` - Line 38
- `get_service(self, name: str) -> Any` - Line 57
- `has_service(self, name: str) -> bool` - Line 74
- `async initialize_all(self) -> None` - Line 78
- `async cleanup_all(self) -> None` - Line 91
- `list_services(self) -> dict[str, str]` - Line 102
- `get_all_service_names(self) -> list[str]` - Line 106

### Implementation: `WebInterfaceFactory` ✅

**Inherits**: BaseComponent
**Purpose**: Factory for creating web interface components
**Status**: Complete

**Implemented Methods:**
- `create_service_registry(self) -> ServiceRegistry` - Line 53
- `create_jwt_handler(self, config: dict[str, Any] | None = None) -> JWTHandler` - Line 60
- `create_auth_manager(self, config: dict[str, Any] | None = None) -> AuthManager` - Line 84
- `create_trading_service(self) -> TradingServiceImpl` - Line 107
- `create_bot_management_service(self) -> BotManagementServiceImpl` - Line 115
- `create_market_data_service(self) -> MarketDataServiceImpl` - Line 123
- `create_portfolio_service(self) -> PortfolioServiceImpl` - Line 131
- `create_risk_service(self) -> RiskServiceImpl` - Line 139
- `create_strategy_service(self) -> StrategyServiceImpl` - Line 147
- `create_api_facade(self) -> APIFacade` - Line 155
- `create_websocket_manager(self) -> UnifiedWebSocketManager` - Line 179
- `create_complete_web_stack(self, config: dict[str, Any] | None = None) -> dict[str, Any]` - Line 196

### Implementation: `WebPortfolioServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web portfolio service operations
**Status**: Complete

**Implemented Methods:**
- `async get_portfolio_summary_data(self) -> dict[str, Any]` - Line 16
- `async calculate_pnl_periods(self, total_pnl: Decimal, total_trades: int, win_rate: float) -> dict[str, dict[str, Any]]` - Line 20
- `async get_processed_positions(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 26
- `async calculate_pnl_metrics(self, period: str) -> dict[str, Any]` - Line 32
- `generate_mock_balances(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 36
- `calculate_asset_allocation(self) -> list[dict[str, Any]]` - Line 40
- `generate_performance_chart_data(self, period: str, resolution: str) -> dict[str, Any]` - Line 44

### Implementation: `WebTradingServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web trading service operations
**Status**: Complete

**Implemented Methods:**
- `async validate_order_request(self, ...) -> dict[str, Any]` - Line 54
- `async format_order_response(self, order_result: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]` - Line 65
- `async get_formatted_orders(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 71
- `async get_formatted_trades(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 77
- `async get_market_data_with_context(self, symbol: str, exchange: str = 'binance') -> dict[str, Any]` - Line 83
- `async generate_order_book_data(self, symbol: str, exchange: str, depth: int) -> dict[str, Any]` - Line 89

### Implementation: `WebBotServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web bot service operations
**Status**: Complete

**Implemented Methods:**
- `async validate_bot_configuration(self, config_data: dict[str, Any]) -> dict[str, Any]` - Line 99
- `async format_bot_response(self, bot_data: dict[str, Any]) -> dict[str, Any]` - Line 105
- `async get_formatted_bot_list(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 111
- `async calculate_bot_metrics(self, bot_id: str) -> dict[str, Any]` - Line 117
- `async validate_bot_operation(self, bot_id: str, operation: str) -> dict[str, Any]` - Line 123
- `async create_bot_configuration(self, request_data: dict[str, Any], user_id: str) -> Any` - Line 129

### Implementation: `WebMonitoringServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web monitoring service operations
**Status**: Complete

**Implemented Methods:**
- `async get_system_health_summary(self) -> dict[str, Any]` - Line 139
- `async get_performance_metrics(self, component: str | None = None) -> dict[str, Any]` - Line 143
- `async get_error_summary(self, time_range: str = '24h') -> dict[str, Any]` - Line 149
- `async get_alert_dashboard_data(self) -> dict[str, Any]` - Line 155

### Implementation: `WebRiskServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web risk service operations
**Status**: Complete

**Implemented Methods:**
- `async get_risk_dashboard_data(self) -> dict[str, Any]` - Line 163
- `async validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]` - Line 167
- `async calculate_position_risk(self, symbol: str, quantity: Decimal, price: Decimal) -> dict[str, Any]` - Line 173
- `async get_portfolio_risk_breakdown(self) -> dict[str, Any]` - Line 179

### Implementation: `WebStrategyServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web strategy service operations
**Status**: Complete

**Implemented Methods:**
- `async get_formatted_strategies(self) -> list[dict[str, Any]]` - Line 187
- `async validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> dict[str, Any]` - Line 191
- `async get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]` - Line 197
- `async format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]` - Line 203

### Implementation: `WebDataServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web data service operations
**Status**: Complete

**Implemented Methods:**
- `async get_market_overview(self, exchange: str = 'binance') -> dict[str, Any]` - Line 213
- `async get_symbol_analytics(self, symbol: str) -> dict[str, Any]` - Line 217
- `async get_historical_chart_data(self, symbol: str, timeframe: str, period: str) -> dict[str, Any]` - Line 221
- `async get_real_time_feed_status(self) -> dict[str, Any]` - Line 227

### Implementation: `WebServiceInterface` 🔧

**Inherits**: ABC
**Purpose**: Base interface for all web services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize(self) -> None` - Line 237
- `async cleanup(self) -> None` - Line 242
- `health_check(self) -> dict[str, Any]` - Line 247
- `get_service_info(self) -> dict[str, Any]` - Line 252

### Implementation: `WebStrategyServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for web strategy service operations
**Status**: Complete

**Implemented Methods:**
- `async get_formatted_strategies(self) -> list[dict[str, Any]]` - Line 260
- `async validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> dict[str, Any]` - Line 264
- `async get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]` - Line 270
- `async format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]` - Line 276
- `health_check(self) -> dict[str, Any]` - Line 282
- `get_service_info(self) -> dict[str, Any]` - Line 286

### Implementation: `AuthMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Authentication middleware for request processing
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 56

### Implementation: `CacheEntry` ✅

**Purpose**: Represents a cached response entry
**Status**: Complete

**Implemented Methods:**
- `is_expired(self) -> bool` - Line 34
- `touch(self)` - Line 38

### Implementation: `CacheMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Intelligent caching middleware for API responses
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next)` - Line 143
- `get_cache_stats(self) -> dict[str, Any]` - Line 463
- `clear_cache(self, pattern: str | None = None)` - Line 500

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
- `async get_uow(self)` - Line 275
- `async get_redis_connection(self)` - Line 293
- `async health_check(self) -> dict[str, Any]` - Line 309
- `async get_pool_stats(self) -> dict[str, Any]` - Line 385
- `async close(self)` - Line 451

### Implementation: `ConnectionPoolMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to provide connection pool access to requests
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next)` - Line 503
- `async startup(self)` - Line 531
- `async shutdown(self)` - Line 535

### Implementation: `ConnectionHealthMonitor` ✅

**Purpose**: Monitor connection pool health and performance
**Status**: Complete

**Implemented Methods:**
- `async start_monitoring(self)` - Line 628
- `async stop_monitoring(self)` - Line 637

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
- `async dispatch(self, request: Request, call_next)` - Line 423

### Implementation: `ErrorHandlerMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Comprehensive error handling middleware
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 85
- `get_error_stats(self) -> dict` - Line 525

### Implementation: `FinancialValidationMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware for validating financial inputs on trading endpoints
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next) -> Response` - Line 74

### Implementation: `DecimalEnforcementMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to enforce Decimal usage in responses
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next) -> Response` - Line 280

### Implementation: `CurrencyValidationMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to validate currency codes and formats
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next) -> Response` - Line 317

### Implementation: `RateLimitMiddleware` ✅

**Inherits**: BaseHTTPMiddleware
**Purpose**: Advanced rate limiting middleware
**Status**: Complete

**Implemented Methods:**
- `async dispatch(self, request: Request, call_next: Callable) -> Response` - Line 99
- `get_rate_limit_stats(self) -> dict` - Line 332

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
- `hash_password(self, password: str) -> str` - Line 186
- `verify_password(self, plain_password: str, hashed_password: str) -> bool` - Line 190
- `create_access_token(self, ...) -> str` - Line 198
- `create_refresh_token(self, user_id: str, username: str) -> str` - Line 250
- `validate_token(self, token: str) -> TokenData` - Line 285
- `validate_scopes(self, token_data: TokenData, required_scopes: list[str]) -> bool` - Line 332
- `refresh_access_token(self, refresh_token: str) -> tuple[str, str]` - Line 353
- `revoke_token(self, token: str) -> bool` - Line 404
- `is_token_blacklisted(self, token: str) -> bool` - Line 447
- `get_security_summary(self) -> dict[str, Any]` - Line 481
- `cleanup_expired_blacklist(self) -> int` - Line 493

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

### Implementation: `WebBotService` ✅

**Inherits**: BaseComponent, WebBotServiceInterface
**Purpose**: Service handling bot management business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 28
- `async cleanup(self) -> None` - Line 32
- `async validate_bot_configuration(self, config_data: dict[str, Any]) -> dict[str, Any]` - Line 36
- `async format_bot_response(self, bot_data: dict[str, Any]) -> dict[str, Any]` - Line 89
- `async get_formatted_bot_list(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 115
- `async calculate_bot_metrics(self, bot_id: str) -> dict[str, Any]` - Line 195
- `async validate_bot_operation(self, bot_id: str, operation: str) -> dict[str, Any]` - Line 273
- `async create_bot_configuration(self, request_data: dict[str, Any], user_id: str) -> BotConfiguration` - Line 314
- `health_check(self) -> dict[str, Any]` - Line 377
- `get_service_info(self) -> dict[str, Any]` - Line 386
- `async create_bot_through_service(self, bot_config) -> str` - Line 401
- `async get_bot_status_through_service(self, bot_id: str) -> dict[str, Any]` - Line 415
- `async start_bot_through_service(self, bot_id: str) -> bool` - Line 435
- `async stop_bot_through_service(self, bot_id: str) -> bool` - Line 448
- `async delete_bot_through_service(self, bot_id: str, force: bool = False) -> bool` - Line 461
- `async list_bots_through_service(self) -> list[dict[str, Any]]` - Line 479
- `get_facade_health_check(self) -> dict[str, Any]` - Line 504

### Implementation: `WebCapitalService` ✅

**Inherits**: BaseService
**Purpose**: Web interface service for capital management operations
**Status**: Complete

**Implemented Methods:**
- `async allocate_capital(self, ...) -> dict[str, Any]` - Line 46
- `async release_capital(self, ...) -> bool` - Line 101
- `async update_utilization(self, ...) -> bool` - Line 149
- `async get_allocations(self, ...) -> list[dict[str, Any]]` - Line 188
- `async get_strategy_allocation(self, strategy_id: str, exchange: str | None = None) -> dict[str, Any] | None` - Line 211
- `async get_capital_metrics(self) -> dict[str, Any]` - Line 228
- `async get_utilization_breakdown(self, by: str = 'strategy') -> dict[str, Any]` - Line 248
- `async get_available_capital(self, strategy_id: str | None = None, exchange: str | None = None) -> dict[str, Any]` - Line 310
- `async get_capital_exposure(self) -> dict[str, Any]` - Line 334
- `async get_currency_exposure(self) -> dict[str, Any]` - Line 376
- `async create_currency_hedge(self, ...) -> dict[str, Any]` - Line 399
- `async get_currency_rates(self, base_currency: str = 'USD') -> dict[str, float]` - Line 429
- `async get_fund_flows(self, days: int = 30, flow_type: str | None = None) -> list[dict[str, Any]]` - Line 447
- `async record_fund_flow(self, ...) -> dict[str, Any]` - Line 460
- `async generate_fund_flow_report(self, start_date: datetime | None, end_date: datetime | None, format: str) -> dict[str, Any]` - Line 493
- `async get_capital_limits(self, limit_type: str | None = None) -> list[dict[str, Any]]` - Line 517
- `async update_capital_limits(self, ...) -> bool` - Line 548
- `async get_limit_breaches(self, hours: int = 24, severity: str | None = None) -> list[dict[str, Any]]` - Line 570

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
- `async get_data_throughput(self, source: str | None = None, hours: int = 1) -> dict[str, Any]` - Line 552
- `async clear_data_cache(self, cache_type: str | None = None) -> int` - Line 572
- `async get_cache_statistics(self) -> dict[str, Any]` - Line 602

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
- `async get_exchange_fees(self, exchange: str, symbol: str | None = None) -> dict[str, Any]` - Line 289
- `async get_rate_limits(self, exchange: str) -> dict[str, Any]` - Line 314
- `async get_rate_usage(self, exchange: str) -> dict[str, Any]` - Line 330
- `async update_rate_config(self, ...) -> bool` - Line 347
- `async get_all_exchanges_health(self) -> dict[str, Any]` - Line 367
- `async get_exchange_health(self, exchange: str) -> dict[str, Any] | None` - Line 391
- `async get_exchange_latency(self, exchange: str, hours: int) -> dict[str, Any]` - Line 414
- `async get_exchange_errors(self, exchange: str, hours: int, error_type: str | None = None) -> list[dict[str, Any]]` - Line 431
- `async get_orderbook(self, exchange: str, symbol: str, limit: int) -> dict[str, Any] | None` - Line 464
- `async get_ticker(self, exchange: str, symbol: str) -> dict[str, Any] | None` - Line 497
- `async get_exchange_balance(self, exchange: str, user_id: str) -> dict[str, dict[str, Decimal]]` - Line 527
- `async subscribe_websocket(self, exchange: str, channel: str, symbols: list[str], subscriber: str) -> str` - Line 555
- `async unsubscribe_websocket(self, exchange: str, subscription_id: str, subscriber: str) -> bool` - Line 578

### Implementation: `WebMonitoringService` ✅

**Inherits**: BaseComponent, WebMonitoringServiceInterface
**Purpose**: Service handling monitoring business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 23
- `async cleanup(self) -> None` - Line 27
- `async get_system_health_summary(self) -> dict[str, Any]` - Line 31
- `async get_performance_metrics(self, component: str | None = None) -> dict[str, Any]` - Line 91
- `async get_error_summary(self, time_range: str = '24h') -> dict[str, Any]` - Line 159
- `async get_alert_dashboard_data(self) -> dict[str, Any]` - Line 233
- `health_check(self) -> dict[str, Any]` - Line 475
- `get_service_info(self) -> dict[str, Any]` - Line 484

### Implementation: `WebPortfolioService` ✅

**Inherits**: BaseComponent, WebPortfolioServiceInterface
**Purpose**: Service handling portfolio business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 27
- `async cleanup(self) -> None` - Line 31
- `async get_portfolio_summary_data(self) -> dict[str, Any]` - Line 35
- `async calculate_pnl_periods(self, total_pnl: Decimal, total_trades: int, win_rate: float) -> dict[str, dict[str, Any]]` - Line 86
- `async get_processed_positions(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 152
- `async calculate_pnl_metrics(self, period: str) -> dict[str, Any]` - Line 231
- `generate_mock_balances(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 308
- `calculate_asset_allocation(self) -> list[dict[str, Any]]` - Line 358
- `generate_performance_chart_data(self, period: str, resolution: str) -> dict[str, Any]` - Line 392

### Implementation: `WebRiskService` ✅

**Inherits**: BaseComponent, WebRiskServiceInterface
**Purpose**: Service handling risk management business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 35
- `async cleanup(self) -> None` - Line 39
- `async get_risk_dashboard_data(self) -> dict[str, Any]` - Line 43
- `async validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]` - Line 154
- `async calculate_position_risk(self, symbol: str, quantity: Decimal, price: Decimal) -> dict[str, Any]` - Line 227
- `async get_portfolio_risk_breakdown(self) -> dict[str, Any]` - Line 372
- `health_check(self) -> dict[str, Any]` - Line 673
- `generate_mock_risk_alerts(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 682
- `generate_mock_position_risks(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 759
- `generate_mock_correlation_matrix(self, symbols: list[str], period: str) -> dict[str, Any]` - Line 833
- `generate_mock_stress_test_results(self, test_request: dict[str, Any]) -> dict[str, Any]` - Line 876
- `get_service_info(self) -> dict[str, Any]` - Line 939

### Implementation: `TradingServiceImpl` ✅

**Inherits**: TradingServiceInterface, BaseComponent
**Purpose**: Implementation of trading service
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector)` - Line 42
- `async initialize(self) -> None` - Line 53
- `async cleanup(self) -> None` - Line 57
- `async place_order(self, ...) -> str` - Line 61
- `async cancel_order(self, order_id: str) -> bool` - Line 90
- `async get_positions(self) -> list[Position]` - Line 105

### Implementation: `BotManagementServiceImpl` ✅

**Inherits**: BotManagementServiceInterface, BaseComponent
**Purpose**: Implementation of bot management service
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector)` - Line 153
- `async initialize(self) -> None` - Line 164
- `async cleanup(self) -> None` - Line 168
- `async create_bot(self, config: BotConfiguration) -> str` - Line 172
- `async start_bot(self, bot_id: str) -> bool` - Line 188
- `async stop_bot(self, bot_id: str) -> bool` - Line 204
- `async get_bot_status(self, bot_id: str) -> dict[str, Any]` - Line 220
- `async list_bots(self) -> list[dict[str, Any]]` - Line 240
- `async get_all_bots_status(self) -> dict[str, Any]` - Line 285
- `async delete_bot(self, bot_id: str, force: bool = False) -> bool` - Line 319

### Implementation: `MarketDataServiceImpl` ✅

**Inherits**: MarketDataServiceInterface, BaseComponent
**Purpose**: Implementation of market data service
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector)` - Line 355
- `async initialize(self) -> None` - Line 366
- `async cleanup(self) -> None` - Line 370
- `async get_ticker(self, symbol: str) -> MarketData` - Line 374
- `async subscribe_to_ticker(self, symbol: str, callback: Callable) -> None` - Line 428
- `async unsubscribe_from_ticker(self, symbol: str) -> None` - Line 438

### Implementation: `PortfolioServiceImpl` ✅

**Inherits**: PortfolioServiceInterface, BaseComponent
**Purpose**: Implementation of portfolio service
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector)` - Line 455
- `async initialize(self) -> None` - Line 466
- `async cleanup(self) -> None` - Line 470
- `async get_balance(self) -> dict[str, Decimal]` - Line 474
- `async get_portfolio_summary(self) -> dict[str, Any]` - Line 488
- `async get_pnl_report(self, start_date: datetime, end_date: datetime) -> dict[str, Any]` - Line 517

### Implementation: `RiskServiceImpl` ✅

**Inherits**: RiskServiceInterface, BaseComponent
**Purpose**: Implementation of risk management service
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector)` - Line 547
- `async initialize(self) -> None` - Line 558
- `async cleanup(self) -> None` - Line 562
- `async validate_order(self, ...) -> dict[str, Any]` - Line 566
- `async get_risk_metrics(self) -> dict[str, Any]` - Line 590
- `async update_risk_limits(self, limits: dict[str, Any]) -> bool` - Line 611

### Implementation: `StrategyServiceImpl` ✅

**Inherits**: StrategyServiceInterface, BaseComponent
**Purpose**: Implementation of strategy service
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector)` - Line 637
- `async initialize(self) -> None` - Line 657
- `async cleanup(self) -> None` - Line 661
- `async list_strategies(self) -> list[dict[str, Any]]` - Line 665
- `async get_strategy_config(self, strategy_name: str) -> dict[str, Any]` - Line 718
- `async validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool` - Line 751

### Implementation: `WebStrategyService` ✅

**Inherits**: BaseComponent, WebStrategyServiceInterface
**Purpose**: Service handling strategy business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 24
- `async cleanup(self) -> None` - Line 28
- `async get_formatted_strategies(self) -> list[dict[str, Any]]` - Line 32
- `async validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> dict[str, Any]` - Line 151
- `async get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]` - Line 198
- `async format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]` - Line 297
- `health_check(self) -> dict[str, Any]` - Line 337
- `get_service_info(self) -> dict[str, Any]` - Line 346

### Implementation: `WebTradingService` ✅

**Inherits**: BaseComponent, WebTradingServiceInterface
**Purpose**: Service handling trading business logic for web interface
**Status**: Complete

**Implemented Methods:**
- `async initialize(self) -> None` - Line 28
- `async cleanup(self) -> None` - Line 32
- `async place_order_through_service(self, ...) -> dict[str, Any]` - Line 36
- `async cancel_order_through_service(self, order_id: str) -> bool` - Line 70
- `async get_service_health(self) -> dict[str, Any]` - Line 84
- `async get_order_details(self, order_id: str, exchange: str) -> dict[str, Any]` - Line 105
- `async validate_order_request(self, ...) -> dict[str, Any]` - Line 149
- `async format_order_response(self, order_result: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]` - Line 256
- `async get_formatted_orders(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 294
- `async get_formatted_trades(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]` - Line 383
- `async get_market_data_with_context(self, symbol: str, exchange: str = 'binance') -> dict[str, Any]` - Line 450
- `async generate_order_book_data(self, symbol: str, exchange: str, depth: int) -> dict[str, Any]` - Line 491
- `health_check(self) -> dict[str, Any]` - Line 530
- `get_service_info(self) -> dict[str, Any]` - Line 539

### Implementation: `TradingNamespace` ✅

**Inherits**: AsyncNamespace
**Purpose**: Main namespace for trading-related Socket
**Status**: Complete

**Implemented Methods:**
- `async on_connect(self, sid: str, environ: Optional[dict[str, Any], Any] = None)` - Line 58
- `async on_disconnect(self, sid: str)` - Line 151
- `async on_authenticate(self, sid: str, data: dict[str, Any])` - Line 188
- `async on_subscribe(self, sid: str, data: dict[str, Any])` - Line 210
- `async on_unsubscribe(self, sid: str, data: dict[str, Any])` - Line 232
- `async on_ping(self, sid: str, data: dict[str, Any] | None = None)` - Line 246
- `async on_execute_order(self, sid: str, data: dict[str, Any])` - Line 259
- `async on_get_portfolio(self, sid: str, data: dict[str, Any])` - Line 289

### Implementation: `SocketIOManager` ✅

**Inherits**: BaseComponent
**Purpose**: Manager for Socket
**Status**: Complete

**Implemented Methods:**
- `create_server(self, cors_allowed_origins: list[str] | None = None) -> AsyncServer` - Line 374
- `create_app(self)` - Line 403
- `async start_background_tasks(self)` - Line 411
- `async stop_background_tasks(self)` - Line 432
- `async emit_to_user(self, user_id: str, event: str, data: Any)` - Line 559
- `async broadcast(self, event: str, data: Any, room: str | None = None)` - Line 566

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
- `is_compatible_with(self, other: 'APIVersion') -> bool` - Line 59
- `is_deprecated(self) -> bool` - Line 72
- `is_sunset(self) -> bool` - Line 76

### Implementation: `VersionManager` ✅

**Inherits**: BaseComponent
**Purpose**: Manager for API versioning and compatibility
**Status**: Complete

**Implemented Methods:**
- `register_version(self, version: APIVersion) -> None` - Line 155
- `parse_version(self, version_string: str) -> APIVersion | None` - Line 160
- `get_version(self, version: str) -> APIVersion | None` - Line 172
- `get_latest_version(self) -> APIVersion | None` - Line 176
- `get_default_version(self) -> APIVersion | None` - Line 182
- `list_versions(self, include_deprecated: bool = True) -> list[APIVersion]` - Line 188
- `resolve_version(self, requested_version: str | None = None) -> APIVersion` - Line 195
- `check_feature_availability(self, version: str, feature: str) -> bool` - Line 235
- `get_deprecation_info(self, version: str) -> dict[str, Any] | None` - Line 242
- `deprecate_version(self, version: str, sunset_date: datetime | None = None) -> bool` - Line 259
- `sunset_version(self, version: str) -> bool` - Line 273
- `get_version_migration_guide(self, from_version: str, to_version: str) -> dict[str, Any]` - Line 285

### Implementation: `BotStatusManager` ✅

**Purpose**: Manages WebSocket connections for bot status updates
**Status**: Complete

**Implemented Methods:**
- `async connect(self, websocket: WebSocket, user_id: str)` - Line 33
- `disconnect(self, user_id: str)` - Line 44
- `subscribe_to_bot(self, user_id: str, bot_id: str)` - Line 61
- `unsubscribe_from_bot(self, user_id: str, bot_id: str)` - Line 72
- `subscribe_to_all_bots(self, user_id: str)` - Line 84
- `async send_to_user(self, user_id: str, message: dict)` - Line 93
- `async broadcast_to_bot_subscribers(self, bot_id: str, message: dict)` - Line 109

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
- `async connect(self, websocket: WebSocket, user_id: str)` - Line 32
- `disconnect(self, user_id: str)` - Line 43
- `subscribe_to_symbol(self, user_id: str, symbol: str)` - Line 60
- `unsubscribe_from_symbol(self, user_id: str, symbol: str)` - Line 71
- `async send_to_user(self, user_id: str, message: dict)` - Line 83
- `async broadcast_to_symbol_subscribers(self, symbol: str, message: dict)` - Line 98

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
- `disconnect(self, user_id: str)` - Line 55
- `subscribe_to_updates(self, user_id: str, update_types: list[str])` - Line 64
- `unsubscribe_from_updates(self, user_id: str, update_types: list[str])` - Line 72
- `async send_to_user(self, user_id: str, message: dict)` - Line 83
- `async broadcast_to_all(self, message: dict, update_type: str)` - Line 99

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
- `async emit_to_channel(self, sio: AsyncServer, event: str, data: Any) -> None` - Line 89

### Implementation: `MarketDataHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for market data WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 107

### Implementation: `BotStatusHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for bot status WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 154

### Implementation: `PortfolioHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for portfolio WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 203

### Implementation: `TradesHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for trade execution WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 253

### Implementation: `OrdersHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for order status WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 304

### Implementation: `AlertsHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for alerts and notifications WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 357

### Implementation: `LogsHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for system logs WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 409

### Implementation: `RiskMetricsHandler` ✅

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for risk metrics WebSocket events
**Status**: Complete

**Implemented Methods:**
- `async start(self) -> None` - Line 462

### Implementation: `UnifiedWebSocketNamespace` ✅

**Inherits**: AsyncNamespace
**Purpose**: Unified namespace for all WebSocket communications
**Status**: Complete

**Implemented Methods:**
- `async on_connect(self, ...)` - Line 555
- `async on_disconnect(self, sid: str)` - Line 592
- `async on_authenticate(self, sid: str, data: dict[str, Any])` - Line 608
- `async on_subscribe(self, sid: str, data: dict[str, Any])` - Line 645
- `async on_unsubscribe(self, sid: str, data: dict[str, Any])` - Line 677
- `async on_ping(self, sid: str, data: dict[str, Any] | None = None)` - Line 759
- `async start_handlers(self)` - Line 772
- `async stop_handlers(self)` - Line 777

### Implementation: `UnifiedWebSocketManager` ✅

**Inherits**: BaseComponent
**Purpose**: Unified manager for all WebSocket communications
**Status**: Complete

**Implemented Methods:**
- `configure_dependencies(self, injector)` - Line 795
- `create_server(self, cors_allowed_origins: list[str] | None = None) -> AsyncServer` - Line 804
- `async start(self)` - Line 834
- `async stop(self)` - Line 846
- `get_connection_stats(self) -> dict[str, Any]` - Line 858

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
async def get_portfolio_metrics(current_user: dict = Any, web_analytics_service = Any)  # Line 107
async def get_portfolio_composition(current_user: dict = Any, web_analytics_service = Any)  # Line 132
async def get_correlation_matrix(...)  # Line 147
async def export_portfolio_data(...)  # Line 166
async def get_risk_metrics(current_user: dict = Any, web_analytics_service = Any)  # Line 188
async def calculate_var(request: VaRRequest, current_user: dict = Any, web_analytics_service = Any)  # Line 203
async def run_stress_test(...)  # Line 223
async def get_risk_exposure(current_user: dict = Any, web_analytics_service = Any)  # Line 259
async def get_single_strategy_metrics(strategy_id: str, current_user: dict = Any, web_analytics_service = Any)  # Line 275
async def get_strategy_metrics(...)  # Line 295
async def get_strategy_performance(...)  # Line 314
async def compare_strategies(...)  # Line 333
async def get_operational_metrics(current_user: dict = Any, web_analytics_service = Any)  # Line 355
async def get_system_errors(hours: int = Any, current_user: dict = Any, web_analytics_service = Any)  # Line 370
async def get_operational_events(...)  # Line 392
async def generate_report(...)  # Line 412
async def get_report(report_id: str, current_user: dict = Any, web_analytics_service = Any)  # Line 434
async def list_reports(limit: int = Any, current_user: dict = Any, web_analytics_service = Any)  # Line 454
async def schedule_report(...)  # Line 472
async def get_active_alerts(current_user: dict = Any, web_analytics_service = Any)  # Line 502
async def get_alerts(...)  # Line 517
async def acknowledge_alert(...)  # Line 537
async def get_alert_history(...)  # Line 562
async def configure_alerts(...)  # Line 579
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
async def login(login_request: LoginRequest)  # Line 64
async def refresh_token(refresh_request: RefreshTokenRequest)  # Line 113
async def logout(current_user: User = Any)  # Line 157
async def get_current_user_info(current_user: User = Any)  # Line 183
async def change_password(change_request: ChangePasswordRequest, current_user: User = Any)  # Line 197
async def create_new_user(user_request: CreateUserRequest, admin_user: User = Any)  # Line 253
async def list_users(admin_user: User = Any)  # Line 305
async def get_demo_credentials()  # Line 340
async def get_auth_status()  # Line 378
```

### File: bot_management.py

**Key Imports:**
- `from src.core.caching import CacheKeys`
- `from src.core.caching import cached`
- `from src.core.exceptions import EntityNotFoundError`
- `from src.core.exceptions import ExecutionError`
- `from src.core.exceptions import ServiceError`

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
def get_bot_service()  # Line 33
def get_web_bot_service_instance()  # Line 43
def get_execution_service() -> ExecutionServiceInterface | None  # Line 48
def set_bot_service(service)  # Line 61
def set_bot_orchestrator(orchestrator)  # Line 66
async def create_bot(bot_request: CreateBotRequest, current_user: User = Any)  # Line 141
async def list_bots(status_filter: BotStatus | None = Any, current_user: User = Any)  # Line 247
async def get_bot(bot_id: str, current_user: User = Any)  # Line 312
async def update_bot(bot_id: str, update_request: UpdateBotRequest, current_user: User = Any)  # Line 374
async def start_bot(bot_id: str, current_user: User = Any)  # Line 485
async def stop_bot(bot_id: str, current_user: User = Any)  # Line 561
async def pause_bot(bot_id: str, current_user: User = Any)  # Line 632
async def resume_bot(bot_id: str, current_user: User = Any)  # Line 688
async def delete_bot(bot_id: str, force: bool = Any, current_user: User = Any)  # Line 744
async def get_orchestrator_status(current_user: User = Any)  # Line 813
async def get_status_alias(current_user: User = Any)  # Line 862
async def get_list_alias(...)  # Line 868
async def create_alias(request: dict[str, Any], trading_user: User = Any)  # Line 879
async def get_config_alias(current_user: User = Any)  # Line 888
async def get_logs_alias(current_user: User = Any)  # Line 919
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
    def validate_amount(cls, v)  # Line 39
```

#### Class: `CapitalReleaseRequest`

**Inherits**: BaseModel
**Purpose**: Request model for capital release

```python
class CapitalReleaseRequest(BaseModel):
    def validate_amount(cls, v)  # Line 60
```

#### Class: `CapitalUtilizationUpdate`

**Inherits**: BaseModel
**Purpose**: Request model for utilization update

```python
class CapitalUtilizationUpdate(BaseModel):
    def validate_amount(cls, v)  # Line 80
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
    def validate_amount(cls, v)  # Line 113
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
async def allocate_capital(...)  # Line 167
async def release_capital(...)  # Line 208
async def update_utilization(...)  # Line 243
async def get_allocations(...)  # Line 271
async def get_strategy_allocation(...)  # Line 293
async def get_capital_metrics(current_user: dict = Any, web_capital_service = Any)  # Line 320
async def get_utilization(by: str = Any, current_user: dict = Any, web_capital_service = Any)  # Line 336
async def get_available_capital(...)  # Line 353
async def get_capital_exposure(current_user: dict = Any, web_capital_service = Any)  # Line 378
async def get_currency_exposure(current_user: dict = Any, web_capital_service = Any)  # Line 395
async def create_currency_hedge(...)  # Line 411
async def get_exchange_rates(base_currency: str = Any, current_user: dict = Any, web_capital_service = Any)  # Line 442
async def convert_currency(request: dict, current_user: dict = Any, web_capital_service = Any)  # Line 459
async def get_fund_flows(...)  # Line 481
async def record_fund_flow(request: FundFlowRequest, current_user: dict = Any, web_capital_service = Any)  # Line 505
async def get_fund_flow_report(...)  # Line 538
async def get_allocation_limits(current_user: dict = Any, web_capital_service = Any)  # Line 561
async def set_allocation_limits(request: dict, current_user: dict = Any, web_capital_service = Any)  # Line 577
async def update_capital_limits(...)  # Line 601
async def get_limit_breaches(...)  # Line 636
async def rebalance_portfolio(request: dict, current_user: dict = Any, web_capital_service = Any)  # Line 663
async def calculate_optimal_allocation(...)  # Line 691
async def get_capital_efficiency(current_user: dict = Any, web_capital_service = Any)  # Line 712
async def reserve_capital(request: dict, current_user: dict = Any, web_capital_service = Any)  # Line 729
async def get_reserved_capital(current_user: dict = Any, web_capital_service = Any)  # Line 751
```

### File: data.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.decorators import monitored`
- `from src.web_interface.auth.middleware import get_current_user`
- `from src.web_interface.dependencies import get_web_data_service`

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
async def get_pipeline_metrics(hours: int = Any, current_user: dict = Any, web_data_service = Any)  # Line 154
async def get_data_quality_metrics(...)  # Line 172
async def get_validation_report(days: int = Any, current_user: dict = Any, web_data_service = Any)  # Line 192
async def validate_data(request: DataValidationRequest, current_user: dict = Any, web_data_service = Any)  # Line 209
async def get_data_anomalies(...)  # Line 240
async def list_features(...)  # Line 267
async def get_feature_details(feature_id: str, current_user: dict = Any, web_data_service = Any)  # Line 288
async def compute_features(request: FeatureComputeRequest, current_user: dict = Any, web_data_service = Any)  # Line 311
async def get_feature_metadata(current_user: dict = Any, web_data_service = Any)  # Line 343
async def list_data_sources(...)  # Line 360
async def configure_data_source(...)  # Line 381
async def update_data_source(...)  # Line 418
async def delete_data_source(source_id: str, current_user: dict = Any, web_data_service = Any)  # Line 447
async def get_data_health(current_user: dict = Any, web_data_service = Any)  # Line 476
async def get_data_latency(...)  # Line 499
async def get_data_throughput(...)  # Line 522
async def clear_data_cache(cache_type: str | None = Any, current_user: dict = Any, web_data_service = Any)  # Line 548
async def get_cache_statistics(current_user: dict = Any, web_data_service = Any)  # Line 575
```

### File: exchanges.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.utils.decorators import monitored`
- `from src.web_interface.auth.middleware import get_current_user`
- `from src.web_interface.dependencies import get_web_exchange_service`

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
async def disconnect_exchange(exchange: str, current_user: dict = Any, web_exchange_service = Any)  # Line 142
async def get_exchange_status(exchange: str, current_user: dict = Any, web_exchange_service = Any)  # Line 170
async def get_exchange_config(exchange: str, current_user: dict = Any, web_exchange_service = Any)  # Line 194
async def update_exchange_config(...)  # Line 223
async def get_exchange_symbols(...)  # Line 265
async def get_exchange_fees(...)  # Line 291
async def get_rate_limits(exchange: str, current_user: dict = Any, web_exchange_service = Any)  # Line 311
async def get_rate_usage(exchange: str, current_user: dict = Any, web_exchange_service = Any)  # Line 329
async def update_rate_config(...)  # Line 351
async def get_exchanges_health(current_user: dict = Any, web_exchange_service = Any)  # Line 386
async def get_exchange_health(exchange: str, current_user: dict = Any, web_exchange_service = Any)  # Line 409
async def get_exchange_latency(...)  # Line 432
async def get_exchange_errors(...)  # Line 457
async def get_orderbook(...)  # Line 486
async def get_ticker(exchange: str, symbol: str, current_user: dict = Any, web_exchange_service = Any)  # Line 519
async def get_exchange_balance(exchange: str, current_user: dict = Any, web_exchange_service = Any)  # Line 549
async def subscribe_websocket(...)  # Line 590
async def unsubscribe_websocket(...)  # Line 624
```

### File: health.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.config import Config`
- `from src.core.logging import get_logger`

#### Class: `ConnectionHealthMonitor`

**Inherits**: BaseComponent
**Purpose**: Mock connection health monitor for exchanges

```python
class ConnectionHealthMonitor(BaseComponent):
    def __init__(self, exchange)  # Line 28
    async def get_health_status(self) -> dict  # Line 31
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
def get_config_dependency() -> Config  # Line 64
async def check_database_health(config: Config) -> ComponentHealth  # Line 89
async def check_redis_health(config: Config) -> ComponentHealth  # Line 130
async def check_exchanges_health(config: Config) -> ComponentHealth  # Line 172
async def check_ml_models_health(config: Config) -> ComponentHealth  # Line 235
async def basic_health_check()  # Line 273
async def detailed_health_check(config: Config = Any)  # Line 292
async def readiness_check()  # Line 376
async def liveness_check()  # Line 388
async def startup_check()  # Line 398
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
async def engineer_features(request: FeatureEngineeringRequest, current_user: User = Any)  # Line 825
async def select_features(request: FeatureSelectionRequest, current_user: User = Any)  # Line 870
async def create_ab_test(request: ABTestRequest, current_user: User = Any)  # Line 935
async def get_ab_test_results(test_id: str, current_user: User = Any)  # Line 986
async def promote_ab_test_winner(test_id: str, promote_treatment: bool = Any, current_user: User = Any)  # Line 1047
async def optimize_hyperparameters(...)  # Line 1109
async def export_model(...)  # Line 1163
async def compare_models(...)  # Line 1215
async def get_ml_system_health(current_user: User = Any)  # Line 1280
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
def get_monitoring_service() -> WebMonitoringService  # Line 110
async def health_check()  # Line 117
async def system_status()  # Line 139
async def prometheus_metrics()  # Line 154
async def metrics_json(timeframe: int = Any)  # Line 175
async def performance_stats(timeframe: int = Any)  # Line 200
async def memory_report(user = Any)  # Line 229
async def reset_performance_metrics(user = Any)  # Line 250
async def database_query_stats(user = Any)  # Line 272
async def get_alerts(severity: str | None = Any, status: str | None = Any, limit: int = Any)  # Line 295
async def create_alert_rule(rule_request: dict, user = Any)  # Line 328
async def delete_alert_rule(rule_name: str, user = Any)  # Line 351
async def acknowledge_alert(fingerprint: str, user = Any)  # Line 378
async def alert_stats()  # Line 404
async def get_monitoring_config(user = Any)  # Line 417
async def start_monitoring(user = Any)  # Line 451
async def stop_monitoring(user = Any)  # Line 477
```

### File: optimization.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`
- `from src.optimization.di_registration import get_optimization_service`
- `from src.optimization.interfaces import IOptimizationService`
- `from src.optimization.parameter_space import ParameterSpaceBuilder`

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
    def validate_parameter_ranges(cls, v)  # Line 134
    def validate_end_date(cls, v, info)  # Line 141
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
def set_dependencies(backtest_engine, strat_factory, dependency_injector = None)  # Line 51
async def create_optimization_job(...)  # Line 229
async def start_optimization_job(job_id: str, background_tasks: BackgroundTasks, user: User = Any)  # Line 331
async def get_optimization_job_status(job_id: str, user: User = Any)  # Line 397
async def get_optimization_job_results(job_id: str, top_n: int = Any, user: User = Any)  # Line 460
async def stop_optimization_job(job_id: str, user: User = Any)  # Line 572
async def delete_optimization_job(job_id: str, user: User = Any)  # Line 617
async def list_optimization_jobs(user: User = Any, status_filter: str | None = Any, limit: int = Any)  # Line 663
async def _run_optimization_job(job_id: str, optimization_service: IOptimizationService)  # Line 733
def _build_parameter_space_from_request(request_data: dict[str, Any]) -> 'ParameterSpace'  # Line 816
def _generate_parameter_combinations(request_data: dict[str, Any]) -> list[dict[str, Any]]  # Line 849
async def _run_parameter_combination(job_id: str, parameters: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]  # Line 900
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
**Purpose**: Enhanced playground configuration model

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
def set_dependencies(orchestrator, backtest_engine, strat_factory)  # Line 48
async def create_playground_session(...)  # Line 171
async def start_playground_session(session_id: str, background_tasks: BackgroundTasks, user: User = Any)  # Line 259
async def get_playground_session_status(session_id: str, user: User = Any)  # Line 313
async def get_playground_session_results(session_id: str, user: User = Any)  # Line 366
async def get_playground_session_logs(session_id: str, limit: int = Any, level: str | None = Any, user: User = Any)  # Line 426
async def stop_playground_session(session_id: str, user: User = Any)  # Line 482
async def delete_playground_session(session_id: str, user: User = Any)  # Line 532
async def list_playground_sessions(user: User = Any, status_filter: str | None = Any, limit: int = Any)  # Line 586
async def _run_playground_session(session_id: str)  # Line 630
async def _run_backtest_session(session_id: str, config: dict[str, Any])  # Line 664
async def _run_sandbox_session(session_id: str, config: dict[str, Any])  # Line 757
def _add_session_log(session_id: str, level: str, message: str, context: dict[str, Any] | None = None)  # Line 814
async def get_configurations(user: User = Any, page: int = Any, limit: int = Any)  # Line 896
async def save_configuration(config: PlaygroundConfigurationModel, user: User = Any)  # Line 919
async def delete_configuration(config_id: str, user: User = Any)  # Line 943
async def control_execution(execution_id: str, action: str, user: User = Any)  # Line 968
async def get_executions(...)  # Line 1021
async def create_ab_test(ab_test: ABTestModel, user: User = Any)  # Line 1070
async def run_ab_test(ab_test_id: str, background_tasks: BackgroundTasks, user: User = Any)  # Line 1091
async def start_batch_optimization(...)  # Line 1119
async def get_batch_progress(batch_id: str, user: User = Any)  # Line 1145
async def _run_ab_test(ab_test_id: str)  # Line 1189
async def _run_batch_optimization(batch_id: str)  # Line 1216
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
def get_portfolio_service()  # Line 24
def get_web_portfolio_service_instance() -> WebPortfolioService  # Line 31
def set_dependencies(orchestrator, engine)  # Line 37
async def get_portfolio_summary(current_user: User = Any)  # Line 116
async def get_positions(...)  # Line 176
async def get_balances(exchange: str | None = Any, currency: str | None = Any, current_user: User = Any)  # Line 229
async def get_pnl(period: str = Any, bot_id: str | None = Any, current_user: User = Any)  # Line 286
async def get_asset_allocation(current_user: User = Any)  # Line 321
async def get_performance_chart(period: str = Any, resolution: str = Any, current_user: User = Any)  # Line 357
async def get_portfolio_history(period: str = Any, current_user: User = Any)  # Line 393
async def get_portfolio_performance(current_user: User = Any)  # Line 431
async def rebalance_portfolio(request: dict[str, Any], trading_user: User = Any)  # Line 470
async def get_portfolio_holdings(current_user: User = Any)  # Line 516
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
def get_web_risk_service_instance()  # Line 23
async def get_risk_metrics(current_user: User = Any)  # Line 137
async def get_risk_limits(current_user: User = Any)  # Line 196
async def update_risk_limits(limits_request: UpdateRiskLimitsRequest, current_user: User = Any)  # Line 234
async def get_risk_alerts(...)  # Line 307
async def get_position_risks(exchange: str | None = Any, symbol: str | None = Any, current_user: User = Any)  # Line 368
async def run_stress_test(stress_test_request: StressTestRequest, current_user: User = Any)  # Line 416
async def get_correlation_matrix(symbols: list[str] = Any, period: str = Any, current_user: User = Any)  # Line 468
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
def get_strategy_service()  # Line 24
def get_web_strategy_service_instance() -> WebStrategyService  # Line 30
def set_dependencies(factory)  # Line 41
async def list_strategies(...)  # Line 135
async def get_strategy(strategy_name: str, current_user: User = Any)  # Line 321
async def configure_strategy(...)  # Line 421
async def get_strategy_performance(strategy_name: str, period: str = Any, current_user: User = Any)  # Line 483
async def start_backtest(strategy_name: str, backtest_request: BacktestRequest, current_user: User = Any)  # Line 549
async def list_strategy_categories(current_user: User = Any)  # Line 631
async def optimize_strategy(...)  # Line 690
```

### File: trading.py

**Key Imports:**
- `from src.core.exceptions import ErrorSeverity`
- `from src.core.exceptions import NetworkError`
- `from src.core.exceptions import TimeoutError`
- `from src.core.logging import get_logger`
- `from src.core.types import OrderSide`

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
def get_trading_service() -> Any  # Line 32
def get_web_trading_service_instance()  # Line 39
def set_dependencies(engine: Any, orchestrator: Any) -> None  # Line 45
async def place_order(order_request: PlaceOrderRequest, current_user: User = Any)  # Line 151
async def cancel_order(order_id: str, exchange: str = Any, current_user: User = Any)  # Line 335
async def get_orders(...)  # Line 405
async def get_order(order_id: str, exchange: str = Any, current_user: User = Any)  # Line 458
async def get_trades(...)  # Line 494
async def get_market_data(symbol: str, exchange: str = Any, current_user: User = Any)  # Line 550
async def get_order_book(symbol: str, exchange: str = Any, depth: int = Any, current_user: User = Any)  # Line 599
async def get_execution_status(current_user: User = Any)  # Line 641
async def get_active_orders(current_user: User = Any)  # Line 671
async def get_trading_positions(current_user: User = Any)  # Line 711
async def get_trading_balance(current_user: User = Any)  # Line 755
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
    def __getattr__(self, name)  # Line 756
    def __call__(self, *args, **kwargs)  # Line 760
```

#### Functions:

```python
async def _initialize_services()  # Line 62
async def _connect_api_endpoints_to_services(registry)  # Line 136
async def lifespan(app: FastAPI)  # Line 156
def create_app(...) -> Any  # Line 281
def _register_routes(app: FastAPI) -> None  # Line 444
def _setup_monitoring(fastapi_app: FastAPI, config: Config) -> None  # Line 605
def get_app()  # Line 696
def _get_app_lazy()  # Line 734
def get_asgi_app()  # Line 743
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
    def _create_default_users(self) -> None  # Line 74
    def _start_cleanup_tasks(self) -> None  # Line 116
    async def authenticate(self, credentials: dict[str, Any], provider_type: str | None = None) -> tuple[User, AuthToken] | None  # Line 121
    async def validate_token(self, token_value: str, provider_type: str | None = None) -> User | None  # Line 171
    async def revoke_token(self, token_value: str, provider_type: str | None = None) -> bool  # Line 202
    async def refresh_token(self, refresh_token_value: str) -> AuthToken | None  # Line 226
    async def create_api_key(self, user: User) -> AuthToken | None  # Line 246
    def get_user(self, user_id: str) -> User | None  # Line 266
    def get_user_by_username(self, username: str) -> User | None  # Line 270
    async def create_user(self, user_data: dict[str, Any]) -> User | None  # Line 277
    async def update_user(self, user_id: str, updates: dict[str, Any]) -> bool  # Line 320
    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool  # Line 356
    def get_user_stats(self) -> dict[str, Any]  # Line 383
    async def _cleanup_expired_sessions(self) -> None  # Line 396
```

#### Functions:

```python
def get_auth_manager(injector = None, config: dict[str, Any] | None = None) -> AuthManager  # Line 421
def initialize_auth_manager(config: dict[str, Any]) -> AuthManager  # Line 457
```

### File: decorators.py

#### Functions:

```python
async def get_current_user(request: Request) -> User | None  # Line 17
async def get_current_active_user(current_user: User = Any) -> User  # Line 37
async def get_trading_user(current_user: User = Any) -> User  # Line 54
async def get_admin_user(current_user: User = Any) -> User  # Line 63
def require_auth(func)  # Line 72
def require_permission(permission: PermissionType, resource: str | None = None)  # Line 82
def require_role(role_name: str)  # Line 100
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
- `from src.utils.decimal_utils import to_decimal`

#### Class: `WebInterfaceDataTransformer`

**Purpose**: Handles consistent data transformation for web_interface module

```python
class WebInterfaceDataTransformer:
    def transform_order_request_to_event_data(symbol, ...) -> dict[str, Any]  # Line 19
    def transform_risk_data_to_event_data(risk_data: dict[str, Any], metadata: dict[str, Any] = None) -> dict[str, Any]  # Line 55
    def transform_portfolio_data_to_event_data(portfolio_data: dict[str, Any], metadata: dict[str, Any] = None) -> dict[str, Any]  # Line 79
    def transform_error_to_event_data(error, ...) -> dict[str, Any]  # Line 103
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]  # Line 131
    def ensure_boundary_fields(data: dict[str, Any], source: str = 'web_interface') -> dict[str, Any]  # Line 161
    def transform_for_pub_sub(cls, event_type: str, data: Any, metadata: dict[str, Any] = None) -> dict[str, Any]  # Line 195
    def transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str = None) -> dict[str, Any]  # Line 247
    def align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'stream') -> dict[str, Any]  # Line 275
```

### File: dependencies.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.dependency_injection import get_global_injector`
- `from src.core.logging import get_logger`
- `from src.web_interface.di_registration import register_web_interface_services`
- `from src.web_interface.services.analytics_service import WebAnalyticsService`

#### Functions:

```python
def get_web_analytics_service() -> WebAnalyticsService  # Line 26
def get_web_capital_service() -> WebCapitalService  # Line 36
def get_web_data_service() -> WebDataService  # Line 46
def get_web_exchange_service() -> WebExchangeService  # Line 56
def get_web_portfolio_service() -> WebPortfolioService  # Line 67
def get_web_trading_service() -> WebTradingService  # Line 77
def get_web_bot_service() -> WebBotService  # Line 87
def get_web_monitoring_service() -> WebMonitoringService  # Line 97
def get_web_risk_service() -> WebRiskService  # Line 107
def get_service_registry() -> Any  # Line 118
def get_api_facade(injector: DependencyInjector = None) -> Any  # Line 128
def get_websocket_manager(injector: DependencyInjector = None) -> Any  # Line 139
def get_jwt_handler(injector: DependencyInjector = None) -> Any  # Line 151
def get_auth_manager(injector: DependencyInjector = None) -> Any  # Line 162
def get_web_analytics_service_instance() -> WebAnalyticsService  # Line 174
def get_web_capital_service_instance() -> WebCapitalService  # Line 179
def get_web_data_service_instance() -> WebDataService  # Line 184
def get_web_exchange_service_instance() -> WebExchangeService  # Line 189
def get_web_portfolio_service_instance() -> WebPortfolioService  # Line 194
def get_web_trading_service_instance() -> WebTradingService  # Line 199
def get_web_bot_service_instance() -> WebBotService  # Line 204
def get_web_monitoring_service_instance() -> WebMonitoringService  # Line 209
def get_web_risk_service_instance() -> WebRiskService  # Line 214
def ensure_all_services_registered(injector: DependencyInjector = None) -> None  # Line 220
def get_all_web_services(injector: DependencyInjector = None) -> dict[str, Any]  # Line 239
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`
- `from src.web_interface.factory import WebInterfaceFactory`
- `from src.web_interface.services.analytics_service import WebAnalyticsService`
- `from src.web_interface.services.bot_service import WebBotService`

#### Functions:

```python
def register_web_interface_services(injector: DependencyInjector) -> None  # Line 19
def get_web_portfolio_service(injector: DependencyInjector = None) -> WebPortfolioService  # Line 142
def get_web_trading_service(injector: DependencyInjector = None) -> WebTradingService  # Line 154
def get_web_bot_service(injector: DependencyInjector = None) -> WebBotService  # Line 166
def get_web_monitoring_service(injector: DependencyInjector = None) -> WebMonitoringService  # Line 178
def get_web_risk_service(injector: DependencyInjector = None) -> WebRiskService  # Line 190
def get_api_facade_service(injector: DependencyInjector = None)  # Line 202
```

### File: api_facade.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.base.interfaces import BotManagementServiceInterface`
- `from src.core.base.interfaces import MarketDataServiceInterface`
- `from src.core.base.interfaces import PortfolioServiceInterface`
- `from src.core.base.interfaces import RiskServiceInterface`

#### Class: `APIFacade`

**Inherits**: BaseComponent
**Purpose**: Unified API facade for T-Bot Trading System

```python
class APIFacade(BaseComponent):
    def __init__(self, ...)  # Line 54
    async def initialize(self) -> None  # Line 62
    async def cleanup(self) -> None  # Line 72
    async def place_order(self, ...) -> str  # Line 82
    async def cancel_order(self, order_id: str) -> bool  # Line 126
    async def get_positions(self) -> list[Position]  # Line 136
    async def create_bot(self, config: BotConfiguration) -> str  # Line 142
    async def start_bot(self, bot_id: str) -> bool  # Line 147
    async def stop_bot(self, bot_id: str) -> bool  # Line 152
    async def get_bot_status(self, bot_id: str) -> dict[str, Any]  # Line 157
    async def list_bots(self) -> list[dict[str, Any]]  # Line 162
    async def get_ticker(self, symbol: str) -> MarketData  # Line 168
    async def subscribe_to_ticker(self, symbol: str, callback: Callable) -> None  # Line 173
    async def unsubscribe_from_ticker(self, symbol: str) -> None  # Line 178
    async def get_balance(self) -> dict[str, Decimal]  # Line 184
    async def get_portfolio_summary(self) -> dict[str, Any]  # Line 189
    async def get_pnl_report(self, start_date: datetime, end_date: datetime) -> dict[str, Any]  # Line 194
    async def validate_order(self, ...) -> bool  # Line 200
    async def get_risk_summary(self) -> dict[str, Any]  # Line 217
    async def get_risk_metrics(self) -> dict[str, Any]  # Line 222
    async def calculate_position_size(self, ...) -> Decimal  # Line 227
    def get_current_risk_level(self)  # Line 243
    def is_emergency_stop_active(self) -> bool  # Line 248
    async def list_strategies(self) -> list[dict[str, Any]]  # Line 254
    async def get_strategy_config(self, strategy_name: str) -> dict[str, Any]  # Line 259
    async def validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool  # Line 264
    async def health_check(self) -> dict[str, Any]  # Line 270
    def get_service_status(self, service_name: str) -> dict[str, Any]  # Line 290
    async def delete_bot(self, bot_id: str, force: bool = False) -> bool  # Line 303
```

#### Functions:

```python
def get_api_facade(injector = None) -> APIFacade  # Line 313
```

### File: service_registry.py

**Key Imports:**
- `from src.core.base import BaseComponent`

#### Class: `ServiceInterface`

**Inherits**: ABC
**Purpose**: Base interface for all services

```python
class ServiceInterface(ABC):
    async def initialize(self) -> None  # Line 19
    async def cleanup(self) -> None  # Line 24
```

#### Class: `ServiceRegistry`

**Inherits**: BaseComponent
**Purpose**: Central registry for all system services

```python
class ServiceRegistry(BaseComponent):
    def __init__(self)  # Line 32
    def register_service(self, name: str, service: Any, interface: type | None = None) -> None  # Line 38
    def get_service(self, name: str) -> Any  # Line 57
    def has_service(self, name: str) -> bool  # Line 74
    async def initialize_all(self) -> None  # Line 78
    async def cleanup_all(self) -> None  # Line 91
    def list_services(self) -> dict[str, str]  # Line 102
    def get_all_service_names(self) -> list[str]  # Line 106
    def _implements_interface(self, service: Any, interface: type) -> bool  # Line 110
```

#### Functions:

```python
def get_service_registry(injector = None) -> ServiceRegistry  # Line 130
def register_service(name: str, service: Any, interface: type | None = None) -> None  # Line 158
def get_service(name: str) -> Any  # Line 164
```

### File: factory.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`

#### Class: `WebInterfaceFactory`

**Inherits**: BaseComponent
**Purpose**: Factory for creating web interface components

```python
class WebInterfaceFactory(BaseComponent):
    def __init__(self, injector: DependencyInjector | None = None) -> None  # Line 39
    def create_service_registry(self) -> ServiceRegistry  # Line 53
    def create_jwt_handler(self, config: dict[str, Any] | None = None) -> JWTHandler  # Line 60
    def create_auth_manager(self, config: dict[str, Any] | None = None) -> AuthManager  # Line 84
    def create_trading_service(self) -> TradingServiceImpl  # Line 107
    def create_bot_management_service(self) -> BotManagementServiceImpl  # Line 115
    def create_market_data_service(self) -> MarketDataServiceImpl  # Line 123
    def create_portfolio_service(self) -> PortfolioServiceImpl  # Line 131
    def create_risk_service(self) -> RiskServiceImpl  # Line 139
    def create_strategy_service(self) -> StrategyServiceImpl  # Line 147
    def create_api_facade(self) -> APIFacade  # Line 155
    def create_websocket_manager(self) -> UnifiedWebSocketManager  # Line 179
    def create_complete_web_stack(self, config: dict[str, Any] | None = None) -> dict[str, Any]  # Line 196
```

#### Functions:

```python
def create_web_interface_service(...) -> Any  # Line 229
def create_web_interface_stack(injector: DependencyInjector | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]  # Line 271
```

### File: interfaces.py

#### Class: `WebPortfolioServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web portfolio service operations

```python
class WebPortfolioServiceInterface(Protocol):
    async def get_portfolio_summary_data(self) -> dict[str, Any]  # Line 16
    async def calculate_pnl_periods(self, total_pnl: Decimal, total_trades: int, win_rate: float) -> dict[str, dict[str, Any]]  # Line 20
    async def get_processed_positions(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 26
    async def calculate_pnl_metrics(self, period: str) -> dict[str, Any]  # Line 32
    def generate_mock_balances(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 36
    def calculate_asset_allocation(self) -> list[dict[str, Any]]  # Line 40
    def generate_performance_chart_data(self, period: str, resolution: str) -> dict[str, Any]  # Line 44
```

#### Class: `WebTradingServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web trading service operations

```python
class WebTradingServiceInterface(Protocol):
    async def validate_order_request(self, ...) -> dict[str, Any]  # Line 54
    async def format_order_response(self, order_result: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]  # Line 65
    async def get_formatted_orders(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 71
    async def get_formatted_trades(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 77
    async def get_market_data_with_context(self, symbol: str, exchange: str = 'binance') -> dict[str, Any]  # Line 83
    async def generate_order_book_data(self, symbol: str, exchange: str, depth: int) -> dict[str, Any]  # Line 89
```

#### Class: `WebBotServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web bot service operations

```python
class WebBotServiceInterface(Protocol):
    async def validate_bot_configuration(self, config_data: dict[str, Any]) -> dict[str, Any]  # Line 99
    async def format_bot_response(self, bot_data: dict[str, Any]) -> dict[str, Any]  # Line 105
    async def get_formatted_bot_list(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 111
    async def calculate_bot_metrics(self, bot_id: str) -> dict[str, Any]  # Line 117
    async def validate_bot_operation(self, bot_id: str, operation: str) -> dict[str, Any]  # Line 123
    async def create_bot_configuration(self, request_data: dict[str, Any], user_id: str) -> Any  # Line 129
```

#### Class: `WebMonitoringServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web monitoring service operations

```python
class WebMonitoringServiceInterface(Protocol):
    async def get_system_health_summary(self) -> dict[str, Any]  # Line 139
    async def get_performance_metrics(self, component: str | None = None) -> dict[str, Any]  # Line 143
    async def get_error_summary(self, time_range: str = '24h') -> dict[str, Any]  # Line 149
    async def get_alert_dashboard_data(self) -> dict[str, Any]  # Line 155
```

#### Class: `WebRiskServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web risk service operations

```python
class WebRiskServiceInterface(Protocol):
    async def get_risk_dashboard_data(self) -> dict[str, Any]  # Line 163
    async def validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]  # Line 167
    async def calculate_position_risk(self, symbol: str, quantity: Decimal, price: Decimal) -> dict[str, Any]  # Line 173
    async def get_portfolio_risk_breakdown(self) -> dict[str, Any]  # Line 179
```

#### Class: `WebStrategyServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web strategy service operations

```python
class WebStrategyServiceInterface(Protocol):
    async def get_formatted_strategies(self) -> list[dict[str, Any]]  # Line 187
    async def validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> dict[str, Any]  # Line 191
    async def get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]  # Line 197
    async def format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]  # Line 203
```

#### Class: `WebDataServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web data service operations

```python
class WebDataServiceInterface(Protocol):
    async def get_market_overview(self, exchange: str = 'binance') -> dict[str, Any]  # Line 213
    async def get_symbol_analytics(self, symbol: str) -> dict[str, Any]  # Line 217
    async def get_historical_chart_data(self, symbol: str, timeframe: str, period: str) -> dict[str, Any]  # Line 221
    async def get_real_time_feed_status(self) -> dict[str, Any]  # Line 227
```

#### Class: `WebServiceInterface`

**Inherits**: ABC
**Purpose**: Base interface for all web services

```python
class WebServiceInterface(ABC):
    async def initialize(self) -> None  # Line 237
    async def cleanup(self) -> None  # Line 242
    def health_check(self) -> dict[str, Any]  # Line 247
    def get_service_info(self) -> dict[str, Any]  # Line 252
```

#### Class: `WebStrategyServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for web strategy service operations

```python
class WebStrategyServiceInterface(Protocol):
    async def get_formatted_strategies(self) -> list[dict[str, Any]]  # Line 260
    async def validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> dict[str, Any]  # Line 264
    async def get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]  # Line 270
    async def format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]  # Line 276
    def health_check(self) -> dict[str, Any]  # Line 282
    def get_service_info(self) -> dict[str, Any]  # Line 286
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
    def __init__(self, app, jwt_handler: JWTHandler)  # Line 32
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 56
    def _extract_token(self, auth_header: str) -> str | None  # Line 133
    def _add_timing_header(self, response: Response, start_time: float) -> None  # Line 153
    def _add_security_headers(self, response: Response) -> None  # Line 164
    def _log_request_completion(self, request: Request, response: Response, start_time: float) -> None  # Line 181
```

### File: cache.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `CacheEntry`

**Purpose**: Represents a cached response entry

```python
class CacheEntry:
    def __init__(self, data: Any, headers: dict[str, str], expires_at: float, path: str = '')  # Line 25
    def is_expired(self) -> bool  # Line 34
    def touch(self)  # Line 38
```

#### Class: `CacheMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Intelligent caching middleware for API responses

```python
class CacheMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_cache_size: int = 1000, default_ttl: int = 60)  # Line 56
    async def dispatch(self, request: Request, call_next)  # Line 143
    def _should_bypass_cache(self, request: Request) -> bool  # Line 189
    def _should_cache_response(self, request: Request, response: Response) -> bool  # Line 219
    def _get_endpoint_config(self, path: str) -> dict[str, Any]  # Line 238
    def _generate_cache_key(self, request: Request) -> str  # Line 263
    def _get_cached_response(self, cache_key: str) -> Response | None  # Line 290
    async def _cache_response(self, cache_key: str, request: Request, response: Response)  # Line 319
    def _evict_lru_entries(self)  # Line 369
    def _handle_cache_invalidation(self, request: Request)  # Line 396
    def _matches_trigger_pattern(self, operation_key: str, pattern: str) -> bool  # Line 417
    def _invalidate_cache_patterns(self, patterns: list[str])  # Line 424
    def _add_cache_headers(self, response: Response, hit: bool = False)  # Line 447
    def get_cache_stats(self) -> dict[str, Any]  # Line 463
    def clear_cache(self, pattern: str | None = None)  # Line 500
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
    async def get_uow(self)  # Line 275
    async def get_redis_connection(self)  # Line 293
    async def health_check(self) -> dict[str, Any]  # Line 309
    async def get_pool_stats(self) -> dict[str, Any]  # Line 385
    async def close(self)  # Line 451
```

#### Class: `ConnectionPoolMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to provide connection pool access to requests

```python
class ConnectionPoolMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: Config)  # Line 491
    async def dispatch(self, request: Request, call_next)  # Line 503
    async def startup(self)  # Line 531
    async def shutdown(self)  # Line 535
```

#### Class: `ConnectionHealthMonitor`

**Purpose**: Monitor connection pool health and performance

```python
class ConnectionHealthMonitor:
    def __init__(self, pool_manager: ConnectionPoolManager)  # Line 608
    async def start_monitoring(self)  # Line 628
    async def stop_monitoring(self)  # Line 637
    async def _monitoring_loop(self)  # Line 653
    async def _perform_health_checks(self)  # Line 665
    async def _check_pool_utilization(self, pool_stats: dict[str, Any])  # Line 692
    async def _alert_connection_issue(self, pool_type: str, details: dict[str, Any])  # Line 714
    async def _alert_high_utilization(self, pool_type: str, utilization: float)  # Line 723
```

#### Functions:

```python
def get_global_pool_manager() -> ConnectionPoolManager | None  # Line 544
def set_global_pool_manager(pool_manager: ConnectionPoolManager)  # Line 549
async def get_db_connection()  # Line 555
async def get_redis_connection()  # Line 570
async def get_uow()  # Line 585
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
    def _is_financial_field(self, field_name: str) -> bool  # Line 340
    def _validate_decimal_precision(self, value: Decimal, field_name: str) -> bool  # Line 363
```

#### Class: `DecimalValidationMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Additional middleware for strict decimal validation in critical trading operations

```python
class DecimalValidationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, critical_endpoints: list[str] | None = None)  # Line 406
    async def dispatch(self, request: Request, call_next)  # Line 423
    async def _validate_request_precision(self, request: Request) -> dict[str, Any]  # Line 450
    def _validate_data_precision(self, data: Any, errors: list[str], path: str)  # Line 480
    def _has_precision_issues(self, value: float, field_name: str) -> bool  # Line 507
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
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 85
    async def _handle_tbot_exception(self, request: Request, exception: TradingBotError) -> JSONResponse  # Line 161
    async def _handle_unexpected_exception(self, request: Request, exception: Exception) -> JSONResponse  # Line 237
    def _log_http_exception(self, request: Request, exception: HTTPException) -> None  # Line 310
    def _log_tbot_exception(self, request: Request, exception: TradingBotError, status_code: int) -> None  # Line 337
    def _log_unexpected_exception(self, request: Request, exception: Exception) -> None  # Line 373
    async def _handle_recovered_error(self, request: Request, exception: Exception, error_context: ErrorContext) -> JSONResponse  # Line 408
    def _get_request_id(self, request: Request) -> str  # Line 492
    def _get_current_timestamp(self) -> str  # Line 514
    def get_error_stats(self) -> dict  # Line 525
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
    async def dispatch(self, request: Request, call_next) -> Response  # Line 74
    def _is_financial_endpoint(self, path: str) -> bool  # Line 103
    async def _validate_request_body(self, request: Request) -> None  # Line 107
    async def _validate_response_body(self, response: Response) -> None  # Line 128
    def _validate_financial_data(self, data: dict[str, Any], context: str) -> None  # Line 143
    def _validate_financial_value(self, field_name: str, value: Any, context: str) -> None  # Line 166
    def _validate_decimal_precision(self, field_name: str, value: Decimal, context: str) -> None  # Line 211
    def _validate_value_range(self, field_name: str, value: Decimal, context: str) -> None  # Line 230
    def _get_field_type(self, field_name: str) -> str  # Line 256
```

#### Class: `DecimalEnforcementMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to enforce Decimal usage in responses

```python
class DecimalEnforcementMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response  # Line 280
    def _is_financial_endpoint(self, path: str) -> bool  # Line 294
```

#### Class: `CurrencyValidationMiddleware`

**Inherits**: BaseHTTPMiddleware
**Purpose**: Middleware to validate currency codes and formats

```python
class CurrencyValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response  # Line 317
    async def _validate_currency_codes(self, request: Request) -> None  # Line 337
    def _check_currency_fields(self, data: Any) -> None  # Line 352
    def _is_financial_endpoint(self, path: str) -> bool  # Line 365
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
    def __init__(self, app, config: Config)  # Line 34
    async def dispatch(self, request: Request, call_next: Callable) -> Response  # Line 99
    def _get_rate_limit_info(self, request: Request) -> tuple[str, int]  # Line 164
    def _get_endpoint_limit(self, path: str) -> int | None  # Line 203
    def _check_rate_limit(self, key: str, limit: int, current_time: float) -> bool  # Line 224
    def _record_request(self, key: str, current_time: float) -> None  # Line 247
    def _calculate_retry_after(self, key: str, current_time: float) -> int  # Line 257
    def _add_rate_limit_headers(self, response: Response, key: str, limit: int) -> None  # Line 278
    async def _cleanup_old_entries(self) -> None  # Line 304
    def get_rate_limit_stats(self) -> dict  # Line 332
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
- `from src.core.logging import get_logger`
- `from src.database.models.user import User`
- `from src.database.service import DatabaseService`

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
def init_auth(config: Config) -> None  # Line 74
def _convert_db_user_to_user_in_db(db_user: DBUser) -> UserInDB  # Line 95
async def get_user(username: str, database_service: DatabaseService = None) -> UserInDB | None  # Line 109
async def authenticate_user(username: str, password: str, database_service: DatabaseService = None) -> UserInDB | None  # Line 153
def create_access_token(user: UserInDB, expires_delta: timedelta | None = None) -> Token  # Line 236
async def get_current_user(credentials: HTTPAuthorizationCredentials = Any) -> User  # Line 277
async def get_current_user_with_scopes(required_scopes: list[str])  # Line 332
async def get_admin_user(current_user: User = Any) -> User  # Line 367
async def get_trading_user(current_user: User = Any) -> User  # Line 385
async def create_user(...) -> UserInDB  # Line 405
async def get_auth_summary(database_service: DatabaseService = None) -> dict  # Line 471
def require_permissions(required_permissions: list[str])  # Line 518
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
    def _redis_sync(self, coro)  # Line 131
    def _get_secret_key(self, config: Config) -> str  # Line 156
    def hash_password(self, password: str) -> str  # Line 186
    def verify_password(self, plain_password: str, hashed_password: str) -> bool  # Line 190
    def create_access_token(self, ...) -> str  # Line 198
    def create_refresh_token(self, user_id: str, username: str) -> str  # Line 250
    def validate_token(self, token: str) -> TokenData  # Line 285
    def validate_scopes(self, token_data: TokenData, required_scopes: list[str]) -> bool  # Line 332
    def refresh_access_token(self, refresh_token: str) -> tuple[str, str]  # Line 353
    def revoke_token(self, token: str) -> bool  # Line 404
    def is_token_blacklisted(self, token: str) -> bool  # Line 447
    def get_security_summary(self) -> dict[str, Any]  # Line 481
    def cleanup_expired_blacklist(self) -> int  # Line 493
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
    def _get_empty_portfolio_metrics(self) -> dict[str, Any]  # Line 479
    def _calculate_strategy_comparison(self, strategies: dict[str, dict[str, Any]]) -> dict[str, Any]  # Line 493
```

### File: bot_service.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import BotConfiguration`
- `from src.core.types import BotPriority`

#### Class: `WebBotService`

**Inherits**: BaseComponent, WebBotServiceInterface
**Purpose**: Service handling bot management business logic for web interface

```python
class WebBotService(BaseComponent, WebBotServiceInterface):
    def __init__(self, bot_facade = None)  # Line 24
    async def initialize(self) -> None  # Line 28
    async def cleanup(self) -> None  # Line 32
    async def validate_bot_configuration(self, config_data: dict[str, Any]) -> dict[str, Any]  # Line 36
    async def format_bot_response(self, bot_data: dict[str, Any]) -> dict[str, Any]  # Line 89
    async def get_formatted_bot_list(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 115
    async def calculate_bot_metrics(self, bot_id: str) -> dict[str, Any]  # Line 195
    async def validate_bot_operation(self, bot_id: str, operation: str) -> dict[str, Any]  # Line 273
    async def create_bot_configuration(self, request_data: dict[str, Any], user_id: str) -> BotConfiguration  # Line 314
    def _calculate_health_score(self, metrics: dict[str, Any]) -> float  # Line 351
    def health_check(self) -> dict[str, Any]  # Line 377
    def get_service_info(self) -> dict[str, Any]  # Line 386
    async def create_bot_through_service(self, bot_config) -> str  # Line 401
    async def get_bot_status_through_service(self, bot_id: str) -> dict[str, Any]  # Line 415
    async def start_bot_through_service(self, bot_id: str) -> bool  # Line 435
    async def stop_bot_through_service(self, bot_id: str) -> bool  # Line 448
    async def delete_bot_through_service(self, bot_id: str, force: bool = False) -> bool  # Line 461
    async def list_bots_through_service(self) -> list[dict[str, Any]]  # Line 479
    def get_facade_health_check(self) -> dict[str, Any]  # Line 504
```

### File: capital_service.py

**Key Imports:**
- `from src.capital_management.interfaces import CapitalServiceProtocol`
- `from src.core.base import BaseService`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`

#### Class: `WebCapitalService`

**Inherits**: BaseService
**Purpose**: Web interface service for capital management operations

```python
class WebCapitalService(BaseService):
    def __init__(self, capital_service: CapitalServiceProtocol)  # Line 30
    async def _do_start(self) -> None  # Line 36
    async def _do_stop(self) -> None  # Line 40
    async def allocate_capital(self, ...) -> dict[str, Any]  # Line 46
    async def release_capital(self, ...) -> bool  # Line 101
    async def update_utilization(self, ...) -> bool  # Line 149
    async def get_allocations(self, ...) -> list[dict[str, Any]]  # Line 188
    async def get_strategy_allocation(self, strategy_id: str, exchange: str | None = None) -> dict[str, Any] | None  # Line 211
    async def get_capital_metrics(self) -> dict[str, Any]  # Line 228
    async def get_utilization_breakdown(self, by: str = 'strategy') -> dict[str, Any]  # Line 248
    async def get_available_capital(self, strategy_id: str | None = None, exchange: str | None = None) -> dict[str, Any]  # Line 310
    async def get_capital_exposure(self) -> dict[str, Any]  # Line 334
    async def get_currency_exposure(self) -> dict[str, Any]  # Line 376
    async def create_currency_hedge(self, ...) -> dict[str, Any]  # Line 399
    async def get_currency_rates(self, base_currency: str = 'USD') -> dict[str, float]  # Line 429
    async def get_fund_flows(self, days: int = 30, flow_type: str | None = None) -> list[dict[str, Any]]  # Line 447
    async def record_fund_flow(self, ...) -> dict[str, Any]  # Line 460
    async def generate_fund_flow_report(self, start_date: datetime | None, end_date: datetime | None, format: str) -> dict[str, Any]  # Line 493
    async def get_capital_limits(self, limit_type: str | None = None) -> list[dict[str, Any]]  # Line 517
    async def update_capital_limits(self, ...) -> bool  # Line 548
    async def get_limit_breaches(self, hours: int = 24, severity: str | None = None) -> list[dict[str, Any]]  # Line 570
    def _format_allocation(self, allocation: CapitalAllocation) -> dict[str, Any]  # Line 584
    async def _get_available_capital_amount(self) -> Decimal  # Line 599
    def _calculate_concentration_risk(self, exposure_by_strategy: dict[str, Decimal]) -> float  # Line 604
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
    async def get_data_throughput(self, source: str | None = None, hours: int = 1) -> dict[str, Any]  # Line 552
    async def clear_data_cache(self, cache_type: str | None = None) -> int  # Line 572
    async def get_cache_statistics(self) -> dict[str, Any]  # Line 602
    def _get_mock_pipeline_status(self, pipeline_id: str | None) -> Any  # Line 624
    def _validate_source_config(self, source_type: str, config: dict[str, Any]) -> None  # Line 665
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
    async def get_exchange_fees(self, exchange: str, symbol: str | None = None) -> dict[str, Any]  # Line 289
    async def get_rate_limits(self, exchange: str) -> dict[str, Any]  # Line 314
    async def get_rate_usage(self, exchange: str) -> dict[str, Any]  # Line 330
    async def update_rate_config(self, ...) -> bool  # Line 347
    async def get_all_exchanges_health(self) -> dict[str, Any]  # Line 367
    async def get_exchange_health(self, exchange: str) -> dict[str, Any] | None  # Line 391
    async def get_exchange_latency(self, exchange: str, hours: int) -> dict[str, Any]  # Line 414
    async def get_exchange_errors(self, exchange: str, hours: int, error_type: str | None = None) -> list[dict[str, Any]]  # Line 431
    async def get_orderbook(self, exchange: str, symbol: str, limit: int) -> dict[str, Any] | None  # Line 464
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

**Inherits**: BaseComponent, WebMonitoringServiceInterface
**Purpose**: Service handling monitoring business logic for web interface

```python
class WebMonitoringService(BaseComponent, WebMonitoringServiceInterface):
    def __init__(self, monitoring_facade = None)  # Line 19
    async def initialize(self) -> None  # Line 23
    async def cleanup(self) -> None  # Line 27
    async def get_system_health_summary(self) -> dict[str, Any]  # Line 31
    async def get_performance_metrics(self, component: str | None = None) -> dict[str, Any]  # Line 91
    async def get_error_summary(self, time_range: str = '24h') -> dict[str, Any]  # Line 159
    async def get_alert_dashboard_data(self) -> dict[str, Any]  # Line 233
    def _calculate_overall_health_score(self, health_data: dict[str, Any]) -> float  # Line 304
    def _format_component_health(self, components: dict[str, Any]) -> dict[str, Any]  # Line 333
    def _format_uptime(self, uptime_seconds: int) -> str  # Line 344
    def _calculate_performance_score(self, component: str, metrics: dict[str, Any]) -> float  # Line 359
    def _parse_time_range(self, time_range: str) -> int  # Line 400
    def _analyze_error_patterns(self, error_data: dict[str, Any]) -> dict[str, Any]  # Line 412
    def _categorize_alerts(self, alerts: list[dict[str, Any]]) -> dict[str, Any]  # Line 433
    def _get_severity_priority(self, severity: str) -> int  # Line 460
    def _get_status_icon(self, status: str) -> str  # Line 465
    def health_check(self) -> dict[str, Any]  # Line 475
    def get_service_info(self) -> dict[str, Any]  # Line 484
```

### File: portfolio_service.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ServiceError`
- `from src.core.logging import get_logger`
- `from src.web_interface.interfaces import WebPortfolioServiceInterface`

#### Class: `WebPortfolioService`

**Inherits**: BaseComponent, WebPortfolioServiceInterface
**Purpose**: Service handling portfolio business logic for web interface

```python
class WebPortfolioService(BaseComponent, WebPortfolioServiceInterface):
    def __init__(self, portfolio_facade = None)  # Line 23
    async def initialize(self) -> None  # Line 27
    async def cleanup(self) -> None  # Line 31
    async def get_portfolio_summary_data(self) -> dict[str, Any]  # Line 35
    async def calculate_pnl_periods(self, total_pnl: Decimal, total_trades: int, win_rate: float) -> dict[str, dict[str, Any]]  # Line 86
    async def get_processed_positions(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 152
    async def calculate_pnl_metrics(self, period: str) -> dict[str, Any]  # Line 231
    def generate_mock_balances(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 308
    def calculate_asset_allocation(self) -> list[dict[str, Any]]  # Line 358
    def generate_performance_chart_data(self, period: str, resolution: str) -> dict[str, Any]  # Line 392
```

### File: risk_service.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import RiskManagementError`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.web_interface.data_transformer import WebInterfaceDataTransformer`

#### Class: `WebRiskService`

**Inherits**: BaseComponent, WebRiskServiceInterface
**Purpose**: Service handling risk management business logic for web interface

```python
class WebRiskService(BaseComponent, WebRiskServiceInterface):
    def __init__(self, risk_facade = None)  # Line 31
    async def initialize(self) -> None  # Line 35
    async def cleanup(self) -> None  # Line 39
    async def get_risk_dashboard_data(self) -> dict[str, Any]  # Line 43
    async def validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]  # Line 154
    async def calculate_position_risk(self, symbol: str, quantity: Decimal, price: Decimal) -> dict[str, Any]  # Line 227
    async def get_portfolio_risk_breakdown(self) -> dict[str, Any]  # Line 372
    def _analyze_risk_levels(self, risk_data: dict[str, Any]) -> dict[str, Any]  # Line 497
    def _calculate_exposure_percentage(self, exposure: Decimal, portfolio_value: Decimal) -> Decimal  # Line 549
    def _calculate_drawdown_percentage(self, drawdown: Decimal, portfolio_value: Decimal) -> Decimal  # Line 555
    def _assess_position_risk_level(self, position_percentage: Decimal, volatility: Decimal) -> str  # Line 561
    def _generate_risk_recommendations(self, parameters: dict[str, Any]) -> list[str]  # Line 572
    def _generate_position_warnings(self, position_percentage: Decimal, volatility: Decimal) -> list[str]  # Line 586
    def _generate_position_recommendations(self, symbol: str, position_percentage: Decimal) -> list[str]  # Line 602
    def _analyze_portfolio_risk_distribution(self, positions: list[dict[str, Any]]) -> dict[str, Any]  # Line 616
    def _get_risk_level_by_volatility(self, volatility: float) -> str  # Line 662
    def health_check(self) -> dict[str, Any]  # Line 673
    def generate_mock_risk_alerts(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 682
    def generate_mock_position_risks(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 759
    def generate_mock_correlation_matrix(self, symbols: list[str], period: str) -> dict[str, Any]  # Line 833
    def generate_mock_stress_test_results(self, test_request: dict[str, Any]) -> dict[str, Any]  # Line 876
    def get_service_info(self) -> dict[str, Any]  # Line 939
```

### File: service_implementations.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.base.interfaces import BotManagementServiceInterface`
- `from src.core.base.interfaces import MarketDataServiceInterface`
- `from src.core.base.interfaces import PortfolioServiceInterface`
- `from src.core.base.interfaces import RiskServiceInterface`

#### Class: `TradingServiceImpl`

**Inherits**: TradingServiceInterface, BaseComponent
**Purpose**: Implementation of trading service

```python
class TradingServiceImpl(TradingServiceInterface, BaseComponent):
    def __init__(self, execution_engine = None)  # Line 38
    def configure_dependencies(self, injector)  # Line 42
    async def initialize(self) -> None  # Line 53
    async def cleanup(self) -> None  # Line 57
    async def place_order(self, ...) -> str  # Line 61
    async def cancel_order(self, order_id: str) -> bool  # Line 90
    async def get_positions(self) -> list[Position]  # Line 105
```

#### Class: `BotManagementServiceImpl`

**Inherits**: BotManagementServiceInterface, BaseComponent
**Purpose**: Implementation of bot management service

```python
class BotManagementServiceImpl(BotManagementServiceInterface, BaseComponent):
    def __init__(self, bot_orchestrator = None)  # Line 149
    def configure_dependencies(self, injector)  # Line 153
    async def initialize(self) -> None  # Line 164
    async def cleanup(self) -> None  # Line 168
    async def create_bot(self, config: BotConfiguration) -> str  # Line 172
    async def start_bot(self, bot_id: str) -> bool  # Line 188
    async def stop_bot(self, bot_id: str) -> bool  # Line 204
    async def get_bot_status(self, bot_id: str) -> dict[str, Any]  # Line 220
    async def list_bots(self) -> list[dict[str, Any]]  # Line 240
    async def get_all_bots_status(self) -> dict[str, Any]  # Line 285
    async def delete_bot(self, bot_id: str, force: bool = False) -> bool  # Line 319
```

#### Class: `MarketDataServiceImpl`

**Inherits**: MarketDataServiceInterface, BaseComponent
**Purpose**: Implementation of market data service

```python
class MarketDataServiceImpl(MarketDataServiceInterface, BaseComponent):
    def __init__(self, data_service = None)  # Line 348
    def configure_dependencies(self, injector)  # Line 355
    async def initialize(self) -> None  # Line 366
    async def cleanup(self) -> None  # Line 370
    async def get_ticker(self, symbol: str) -> MarketData  # Line 374
    async def subscribe_to_ticker(self, symbol: str, callback: Callable) -> None  # Line 428
    async def unsubscribe_from_ticker(self, symbol: str) -> None  # Line 438
```

#### Class: `PortfolioServiceImpl`

**Inherits**: PortfolioServiceInterface, BaseComponent
**Purpose**: Implementation of portfolio service

```python
class PortfolioServiceImpl(PortfolioServiceInterface, BaseComponent):
    def __init__(self, portfolio_manager = None)  # Line 451
    def configure_dependencies(self, injector)  # Line 455
    async def initialize(self) -> None  # Line 466
    async def cleanup(self) -> None  # Line 470
    async def get_balance(self) -> dict[str, Decimal]  # Line 474
    async def get_portfolio_summary(self) -> dict[str, Any]  # Line 488
    async def get_pnl_report(self, start_date: datetime, end_date: datetime) -> dict[str, Any]  # Line 517
```

#### Class: `RiskServiceImpl`

**Inherits**: RiskServiceInterface, BaseComponent
**Purpose**: Implementation of risk management service

```python
class RiskServiceImpl(RiskServiceInterface, BaseComponent):
    def __init__(self, risk_manager = None)  # Line 543
    def configure_dependencies(self, injector)  # Line 547
    async def initialize(self) -> None  # Line 558
    async def cleanup(self) -> None  # Line 562
    async def validate_order(self, ...) -> dict[str, Any]  # Line 566
    async def get_risk_metrics(self) -> dict[str, Any]  # Line 590
    async def update_risk_limits(self, limits: dict[str, Any]) -> bool  # Line 611
```

#### Class: `StrategyServiceImpl`

**Inherits**: StrategyServiceInterface, BaseComponent
**Purpose**: Implementation of strategy service

```python
class StrategyServiceImpl(StrategyServiceInterface, BaseComponent):
    def __init__(self, strategy_service = None, strategy_factory = None)  # Line 630
    def configure_dependencies(self, injector)  # Line 637
    async def initialize(self) -> None  # Line 657
    async def cleanup(self) -> None  # Line 661
    async def list_strategies(self) -> list[dict[str, Any]]  # Line 665
    async def get_strategy_config(self, strategy_name: str) -> dict[str, Any]  # Line 718
    async def validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool  # Line 751
```

### File: strategy_service.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ServiceError`
- `from src.web_interface.interfaces import WebStrategyServiceInterface`

#### Class: `WebStrategyService`

**Inherits**: BaseComponent, WebStrategyServiceInterface
**Purpose**: Service handling strategy business logic for web interface

```python
class WebStrategyService(BaseComponent, WebStrategyServiceInterface):
    def __init__(self, strategy_facade = None)  # Line 20
    async def initialize(self) -> None  # Line 24
    async def cleanup(self) -> None  # Line 28
    async def get_formatted_strategies(self) -> list[dict[str, Any]]  # Line 32
    async def validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> dict[str, Any]  # Line 151
    async def get_strategy_performance_data(self, strategy_name: str) -> dict[str, Any]  # Line 198
    async def format_backtest_results(self, backtest_data: dict[str, Any]) -> dict[str, Any]  # Line 297
    def health_check(self) -> dict[str, Any]  # Line 337
    def get_service_info(self) -> dict[str, Any]  # Line 346
```

### File: trading_service.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import OrderSide`
- `from src.core.types import OrderType`

#### Class: `WebTradingService`

**Inherits**: BaseComponent, WebTradingServiceInterface
**Purpose**: Service handling trading business logic for web interface

```python
class WebTradingService(BaseComponent, WebTradingServiceInterface):
    def __init__(self, trading_facade = None)  # Line 24
    async def initialize(self) -> None  # Line 28
    async def cleanup(self) -> None  # Line 32
    async def place_order_through_service(self, ...) -> dict[str, Any]  # Line 36
    async def cancel_order_through_service(self, order_id: str) -> bool  # Line 70
    async def get_service_health(self) -> dict[str, Any]  # Line 84
    async def get_order_details(self, order_id: str, exchange: str) -> dict[str, Any]  # Line 105
    async def validate_order_request(self, ...) -> dict[str, Any]  # Line 149
    async def format_order_response(self, order_result: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]  # Line 256
    async def get_formatted_orders(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 294
    async def get_formatted_trades(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]  # Line 383
    async def get_market_data_with_context(self, symbol: str, exchange: str = 'binance') -> dict[str, Any]  # Line 450
    async def generate_order_book_data(self, symbol: str, exchange: str, depth: int) -> dict[str, Any]  # Line 491
    def health_check(self) -> dict[str, Any]  # Line 530
    def get_service_info(self) -> dict[str, Any]  # Line 539
    def _get_current_timestamp(self) -> str  # Line 555
```

### File: socketio_manager.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import AuthenticationError`
- `from src.core.logging import correlation_context`
- `from src.error_handling import ConnectionManager`
- `from src.error_handling import ConnectionState`

#### Class: `TradingNamespace`

**Inherits**: AsyncNamespace
**Purpose**: Main namespace for trading-related Socket

```python
class TradingNamespace(AsyncNamespace):
    def __init__(self, namespace: str = '/', jwt_handler = None)  # Line 34
    async def on_connect(self, sid: str, environ: Optional[dict[str, Any], Any] = None)  # Line 58
    async def on_disconnect(self, sid: str)  # Line 151
    async def on_authenticate(self, sid: str, data: dict[str, Any])  # Line 188
    async def on_subscribe(self, sid: str, data: dict[str, Any])  # Line 210
    async def on_unsubscribe(self, sid: str, data: dict[str, Any])  # Line 232
    async def on_ping(self, sid: str, data: dict[str, Any] | None = None)  # Line 246
    async def on_execute_order(self, sid: str, data: dict[str, Any])  # Line 259
    async def on_get_portfolio(self, sid: str, data: dict[str, Any])  # Line 289
    async def _validate_token(self, token: str) -> dict[str, Any] | None  # Line 313
    def _require_scope(self, sid: str, required_scope: str) -> bool  # Line 347
```

#### Class: `SocketIOManager`

**Inherits**: BaseComponent
**Purpose**: Manager for Socket

```python
class SocketIOManager(BaseComponent):
    def __init__(self)  # Line 368
    def create_server(self, cors_allowed_origins: list[str] | None = None) -> AsyncServer  # Line 374
    def create_app(self)  # Line 403
    async def start_background_tasks(self)  # Line 411
    async def stop_background_tasks(self)  # Line 432
    async def _broadcast_market_data(self)  # Line 455
    async def _broadcast_bot_status(self)  # Line 492
    async def _broadcast_portfolio_updates(self)  # Line 529
    async def emit_to_user(self, user_id: str, event: str, data: Any)  # Line 559
    async def broadcast(self, event: str, data: Any, room: str | None = None)  # Line 566
```

### File: decorators.py

#### Functions:

```python
def versioned_endpoint(...)  # Line 18
def deprecated(version: str, removal_version: str | None = None, message: str | None = None)  # Line 139
def feature_flag(feature_name: str, default: bool = True)  # Line 199
def version_specific(versions: list[str])  # Line 248
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
    def __post_init__(self)  # Line 40
    def __str__(self) -> str  # Line 46
    def __lt__(self, other: 'APIVersion') -> bool  # Line 49
    def __eq__(self, other: object) -> bool  # Line 53
    def is_compatible_with(self, other: 'APIVersion') -> bool  # Line 59
    def is_deprecated(self) -> bool  # Line 72
    def is_sunset(self) -> bool  # Line 76
```

#### Class: `VersionManager`

**Inherits**: BaseComponent
**Purpose**: Manager for API versioning and compatibility

```python
class VersionManager(BaseComponent):
    def __init__(self)  # Line 84
    def _initialize_default_versions(self) -> None  # Line 94
    def register_version(self, version: APIVersion) -> None  # Line 155
    def parse_version(self, version_string: str) -> APIVersion | None  # Line 160
    def get_version(self, version: str) -> APIVersion | None  # Line 172
    def get_latest_version(self) -> APIVersion | None  # Line 176
    def get_default_version(self) -> APIVersion | None  # Line 182
    def list_versions(self, include_deprecated: bool = True) -> list[APIVersion]  # Line 188
    def resolve_version(self, requested_version: str | None = None) -> APIVersion  # Line 195
    def check_feature_availability(self, version: str, feature: str) -> bool  # Line 235
    def get_deprecation_info(self, version: str) -> dict[str, Any] | None  # Line 242
    def deprecate_version(self, version: str, sunset_date: datetime | None = None) -> bool  # Line 259
    def sunset_version(self, version: str) -> bool  # Line 273
    def get_version_migration_guide(self, from_version: str, to_version: str) -> dict[str, Any]  # Line 285
```

#### Functions:

```python
def get_version_manager() -> VersionManager  # Line 323
```

### File: bot_status.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Class: `BotStatusManager`

**Purpose**: Manages WebSocket connections for bot status updates

```python
class BotStatusManager:
    def __init__(self)  # Line 28
    async def connect(self, websocket: WebSocket, user_id: str)  # Line 33
    def disconnect(self, user_id: str)  # Line 44
    def subscribe_to_bot(self, user_id: str, bot_id: str)  # Line 61
    def unsubscribe_from_bot(self, user_id: str, bot_id: str)  # Line 72
    def subscribe_to_all_bots(self, user_id: str)  # Line 84
    async def send_to_user(self, user_id: str, message: dict)  # Line 93
    async def broadcast_to_bot_subscribers(self, bot_id: str, message: dict)  # Line 109
    async def _send_with_timeout(self, websocket: WebSocket, message: dict, user_id: str)  # Line 133
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
async def bot_status_websocket(websocket: WebSocket)  # Line 164
async def send_initial_bot_status(user_id: str, bot_id: str, update_types: list[str])  # Line 289
async def send_all_bot_status(user_id: str)  # Line 358
async def bot_status_simulator()  # Line 367
async def get_bot_status_connections()  # Line 464
```

### File: market_data.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.web_interface.security.auth import User`

#### Class: `ConnectionManager`

**Purpose**: Manages WebSocket connections for market data streaming

```python
class ConnectionManager:
    def __init__(self)  # Line 27
    async def connect(self, websocket: WebSocket, user_id: str)  # Line 32
    def disconnect(self, user_id: str)  # Line 43
    def subscribe_to_symbol(self, user_id: str, symbol: str)  # Line 60
    def unsubscribe_from_symbol(self, user_id: str, symbol: str)  # Line 71
    async def send_to_user(self, user_id: str, message: dict)  # Line 83
    async def broadcast_to_symbol_subscribers(self, symbol: str, message: dict)  # Line 98
    async def _send_with_timeout(self, websocket: WebSocket, message: dict, user_id: str)  # Line 122
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
async def authenticate_websocket(websocket: WebSocket) -> User | None  # Line 152
async def market_data_websocket(websocket: WebSocket)  # Line 203
async def send_initial_market_data(user_id: str, symbol: str, data_types: list[str])  # Line 316
async def market_data_simulator()  # Line 391
async def get_market_data_status()  # Line 435
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
    def disconnect(self, user_id: str)  # Line 55
    def subscribe_to_updates(self, user_id: str, update_types: list[str])  # Line 64
    def unsubscribe_from_updates(self, user_id: str, update_types: list[str])  # Line 72
    async def send_to_user(self, user_id: str, message: dict)  # Line 83
    async def broadcast_to_all(self, message: dict, update_type: str)  # Line 99
    async def _send_with_timeout(self, websocket: WebSocket, message: dict, user_id: str)  # Line 129
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
async def portfolio_websocket(websocket: WebSocket)  # Line 158
async def send_initial_portfolio_data(user_id: str, update_types: list[str])  # Line 264
async def portfolio_simulator()  # Line 384
async def get_portfolio_connections()  # Line 496
```

### File: public.py

**Key Imports:**
- `from src.core.logging import get_logger`

#### Functions:

```python
async def public_websocket(websocket: WebSocket, token: str | None = Any)  # Line 23
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
    async def emit_to_channel(self, sio: AsyncServer, event: str, data: Any) -> None  # Line 89
```

#### Class: `MarketDataHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for market data WebSocket events

```python
class MarketDataHandler(WebSocketEventHandler):
    def __init__(self)  # Line 103
    async def start(self) -> None  # Line 107
    async def _broadcast_loop(self) -> None  # Line 112
    async def _emit_data(self, event: str, data: Any) -> None  # Line 142
```

#### Class: `BotStatusHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for bot status WebSocket events

```python
class BotStatusHandler(WebSocketEventHandler):
    def __init__(self)  # Line 150
    async def start(self) -> None  # Line 154
    async def _broadcast_loop(self) -> None  # Line 159
    async def _emit_data(self, event: str, data: Any) -> None  # Line 191
```

#### Class: `PortfolioHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for portfolio WebSocket events

```python
class PortfolioHandler(WebSocketEventHandler):
    def __init__(self)  # Line 199
    async def start(self) -> None  # Line 203
    async def _broadcast_loop(self) -> None  # Line 208
    async def _emit_data(self, event: str, data: Any) -> None  # Line 241
```

#### Class: `TradesHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for trade execution WebSocket events

```python
class TradesHandler(WebSocketEventHandler):
    def __init__(self)  # Line 249
    async def start(self) -> None  # Line 253
    async def _broadcast_loop(self) -> None  # Line 258
    async def _emit_data(self, event: str, data: Any) -> None  # Line 292
```

#### Class: `OrdersHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for order status WebSocket events

```python
class OrdersHandler(WebSocketEventHandler):
    def __init__(self)  # Line 300
    async def start(self) -> None  # Line 304
    async def _broadcast_loop(self) -> None  # Line 309
    async def _emit_data(self, event: str, data: Any) -> None  # Line 345
```

#### Class: `AlertsHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for alerts and notifications WebSocket events

```python
class AlertsHandler(WebSocketEventHandler):
    def __init__(self)  # Line 353
    async def start(self) -> None  # Line 357
    async def _broadcast_loop(self) -> None  # Line 362
    async def _emit_data(self, event: str, data: Any) -> None  # Line 397
```

#### Class: `LogsHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for system logs WebSocket events

```python
class LogsHandler(WebSocketEventHandler):
    def __init__(self)  # Line 405
    async def start(self) -> None  # Line 409
    async def _broadcast_loop(self) -> None  # Line 414
    async def _emit_data(self, event: str, data: Any) -> None  # Line 450
```

#### Class: `RiskMetricsHandler`

**Inherits**: WebSocketEventHandler
**Purpose**: Handler for risk metrics WebSocket events

```python
class RiskMetricsHandler(WebSocketEventHandler):
    def __init__(self)  # Line 458
    async def start(self) -> None  # Line 462
    async def _broadcast_loop(self) -> None  # Line 467
    async def _emit_data(self, event: str, data: Any) -> None  # Line 515
```

#### Class: `UnifiedWebSocketNamespace`

**Inherits**: AsyncNamespace
**Purpose**: Unified namespace for all WebSocket communications

```python
class UnifiedWebSocketNamespace(AsyncNamespace):
    def __init__(self, namespace: str = '/')  # Line 523
    def _create_emit_function(self, channel: ChannelType) -> Callable  # Line 546
    async def on_connect(self, ...)  # Line 555
    async def on_disconnect(self, sid: str)  # Line 592
    async def on_authenticate(self, sid: str, data: dict[str, Any])  # Line 608
    async def _authenticate_session(self, sid: str, token: str) -> bool  # Line 617
    async def on_subscribe(self, sid: str, data: dict[str, Any])  # Line 645
    async def on_unsubscribe(self, sid: str, data: dict[str, Any])  # Line 677
    async def _subscribe_to_channel(self, sid: str, channel: ChannelType) -> bool  # Line 694
    async def _unsubscribe_from_channel(self, sid: str, channel: ChannelType) -> bool  # Line 715
    def _is_authenticated(self, sid: str) -> bool  # Line 732
    def _has_channel_permission(self, sid: str, channel: ChannelType) -> bool  # Line 736
    async def on_ping(self, sid: str, data: dict[str, Any] | None = None)  # Line 759
    async def start_handlers(self)  # Line 772
    async def stop_handlers(self)  # Line 777
```

#### Class: `UnifiedWebSocketManager`

**Inherits**: BaseComponent
**Purpose**: Unified manager for all WebSocket communications

```python
class UnifiedWebSocketManager(BaseComponent):
    def __init__(self, api_facade = None)  # Line 788
    def configure_dependencies(self, injector)  # Line 795
    def create_server(self, cors_allowed_origins: list[str] | None = None) -> AsyncServer  # Line 804
    async def start(self)  # Line 834
    async def stop(self)  # Line 846
    def get_connection_stats(self) -> dict[str, Any]  # Line 858
```

#### Functions:

```python
def get_unified_websocket_manager(api_facade: 'APIFacade' = None) -> UnifiedWebSocketManager  # Line 878
```

---
**Generated**: Complete reference for web_interface module
**Total Classes**: 201
**Total Functions**: 362