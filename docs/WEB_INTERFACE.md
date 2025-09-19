# Web Interface API Documentation

## Overview
This document outlines all API endpoints that should be exposed through the web interface module. These endpoints leverage the existing service layer implementations to provide a comprehensive REST API for the trading bot system.

## API Endpoints

### 1. Authentication & Authorization
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | User authentication with JWT |
| POST | `/api/auth/logout` | Logout & token invalidation |
| POST | `/api/auth/refresh` | Refresh access token |
| GET | `/api/auth/profile` | Get current user profile |
| POST | `/api/auth/2fa/enable` | Enable 2FA |
| POST | `/api/auth/2fa/verify` | Verify 2FA code |

### 2. Dashboard & Real-time Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dashboard/overview` | Portfolio overview, P&L, active bots |
| GET | `/api/dashboard/metrics` | Key performance metrics |
| WS | `/ws/dashboard` | Real-time dashboard updates |
| GET | `/api/dashboard/alerts` | Recent alerts and notifications |
| GET | `/api/dashboard/activity` | Recent trading activity |

### 3. Bot Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/bots` | List all bots with status |
| GET | `/api/bots/{bot_id}` | Get bot details |
| POST | `/api/bots` | Create new bot |
| PUT | `/api/bots/{bot_id}` | Update bot configuration |
| DELETE | `/api/bots/{bot_id}` | Delete bot |
| POST | `/api/bots/{bot_id}/start` | Start bot |
| POST | `/api/bots/{bot_id}/stop` | Stop bot |
| POST | `/api/bots/{bot_id}/pause` | Pause bot |
| GET | `/api/bots/{bot_id}/metrics` | Bot performance metrics |
| GET | `/api/bots/{bot_id}/logs` | Bot execution logs |
| WS | `/ws/bots/{bot_id}` | Real-time bot status updates |

### 4. Strategy Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/strategies` | List available strategies |
| GET | `/api/strategies/{strategy_id}` | Get strategy details |
| POST | `/api/strategies` | Create custom strategy |
| PUT | `/api/strategies/{strategy_id}` | Update strategy |
| DELETE | `/api/strategies/{strategy_id}` | Delete strategy |
| GET | `/api/strategies/{strategy_id}/signals` | Get strategy signals |
| GET | `/api/strategies/{strategy_id}/performance` | Strategy performance metrics |
| POST | `/api/strategies/{strategy_id}/backtest` | Run backtest |
| GET | `/api/strategies/templates` | Get strategy templates |

### 5. Trading & Execution
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/orders` | List orders (with filters) |
| GET | `/api/orders/{order_id}` | Get order details |
| POST | `/api/orders` | Place manual order |
| PUT | `/api/orders/{order_id}` | Modify order |
| DELETE | `/api/orders/{order_id}` | Cancel order |
| GET | `/api/trades` | List executed trades |
| GET | `/api/trades/{trade_id}` | Trade details |
| GET | `/api/positions` | Current open positions |
| GET | `/api/positions/{position_id}` | Position details |
| POST | `/api/positions/{position_id}/close` | Close position |
| WS | `/ws/orders` | Real-time order updates |
| WS | `/ws/trades` | Real-time trade execution updates |

### 6. Market Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/market/symbols` | Available trading pairs |
| GET | `/api/market/ticker/{symbol}` | Current price ticker |
| GET | `/api/market/orderbook/{symbol}` | Order book depth |
| GET | `/api/market/candles/{symbol}` | Historical OHLCV data |
| GET | `/api/market/trades/{symbol}` | Recent trades |
| WS | `/ws/market/{symbol}` | Real-time market data stream |
| GET | `/api/market/stats/{symbol}` | 24h statistics |

### 7. Portfolio & Capital Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/portfolio/overview` | Portfolio summary |
| GET | `/api/portfolio/balances` | Asset balances |
| GET | `/api/portfolio/allocation` | Capital allocation |
| GET | `/api/portfolio/history` | Historical portfolio value |
| POST | `/api/portfolio/rebalance` | Trigger rebalancing |
| GET | `/api/portfolio/performance` | Performance metrics |
| GET | `/api/portfolio/pnl` | P&L analysis |
| POST | `/api/capital/allocate` | Allocate capital to bot/strategy |
| PUT | `/api/capital/limits` | Update capital limits |

### 8. Risk Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/risk/metrics` | Current risk metrics |
| GET | `/api/risk/limits` | Risk limits configuration |
| PUT | `/api/risk/limits` | Update risk limits |
| GET | `/api/risk/var` | Value at Risk calculations |
| GET | `/api/risk/exposure` | Current exposure analysis |
| GET | `/api/risk/alerts` | Risk alerts |
| POST | `/api/risk/stop-loss` | Set stop-loss rules |
| GET | `/api/risk/correlation` | Asset correlation matrix |
| GET | `/api/risk/stress-test` | Stress test results |

### 9. Analytics & Reporting
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/analytics/performance` | Performance analytics |
| GET | `/api/analytics/attribution` | Performance attribution |
| GET | `/api/analytics/sharpe` | Sharpe ratio & risk metrics |
| GET | `/api/analytics/drawdown` | Drawdown analysis |
| GET | `/api/analytics/reports` | Generated reports |
| POST | `/api/analytics/reports` | Generate custom report |
| GET | `/api/analytics/export` | Export data (CSV/JSON) |
| GET | `/api/analytics/benchmark` | Benchmark comparison |

### 10. Backtesting
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/backtest` | Run backtest |
| GET | `/api/backtest/{backtest_id}` | Get backtest results |
| GET | `/api/backtest/history` | Backtest history |
| DELETE | `/api/backtest/{backtest_id}` | Delete backtest |
| GET | `/api/backtest/{backtest_id}/metrics` | Detailed metrics |
| GET | `/api/backtest/{backtest_id}/trades` | Simulated trades |
| POST | `/api/backtest/monte-carlo` | Monte Carlo simulation |
| POST | `/api/backtest/walk-forward` | Walk-forward analysis |

### 11. ML Models
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/ml/models` | List ML models |
| GET | `/api/ml/models/{model_id}` | Model details |
| POST | `/api/ml/models/train` | Train new model |
| POST | `/api/ml/models/{model_id}/predict` | Get predictions |
| GET | `/api/ml/models/{model_id}/performance` | Model performance |
| PUT | `/api/ml/models/{model_id}` | Update model |
| DELETE | `/api/ml/models/{model_id}` | Delete model |
| GET | `/api/ml/features` | Feature importance |

### 12. Exchange Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/exchanges` | List configured exchanges |
| GET | `/api/exchanges/{exchange_id}` | Exchange details |
| POST | `/api/exchanges` | Add exchange connection |
| PUT | `/api/exchanges/{exchange_id}` | Update exchange config |
| DELETE | `/api/exchanges/{exchange_id}` | Remove exchange |
| GET | `/api/exchanges/{exchange_id}/status` | Connection status |
| POST | `/api/exchanges/{exchange_id}/test` | Test connection |
| GET | `/api/exchanges/{exchange_id}/limits` | Rate limits |

### 13. System & Monitoring
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health check |
| GET | `/api/health/detailed` | Detailed health metrics |
| GET | `/api/monitoring/metrics` | System metrics |
| GET | `/api/monitoring/logs` | System logs |
| GET | `/api/monitoring/events` | System events |
| GET | `/api/monitoring/performance` | Performance metrics |
| WS | `/ws/monitoring` | Real-time monitoring stream |
| GET | `/api/system/config` | System configuration |
| PUT | `/api/system/config` | Update configuration |

### 14. State Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/state` | Current system state |
| GET | `/api/state/history` | State history |
| POST | `/api/state/checkpoint` | Create checkpoint |
| POST | `/api/state/restore` | Restore from checkpoint |
| GET | `/api/state/consistency` | Consistency check |

### 15. Error Handling & Recovery
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/errors` | Recent errors |
| GET | `/api/errors/{error_id}` | Error details |
| POST | `/api/errors/{error_id}/resolve` | Mark error resolved |
| GET | `/api/circuit-breakers` | Circuit breaker status |
| POST | `/api/circuit-breakers/{breaker_id}/reset` | Reset circuit breaker |

### 16. Data Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/data/sources` | Data sources |
| POST | `/api/data/validate` | Validate data quality |
| GET | `/api/data/pipeline/status` | Pipeline status |
| POST | `/api/data/cache/clear` | Clear cache |
| GET | `/api/data/features` | Feature store |

### 17. Optimization
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/optimization/parameters` | Optimize parameters |
| GET | `/api/optimization/{job_id}` | Optimization results |
| POST | `/api/optimization/bayesian` | Bayesian optimization |
| GET | `/api/optimization/history` | Optimization history |

## Implementation Guidelines

### Service Layer Integration
All endpoints must delegate to the appropriate service layer:
- **BotService** - Bot lifecycle and management
- **StrategyService** - Strategy operations
- **ExecutionService** - Order and trade execution
- **RiskService** - Risk management and monitoring
- **CapitalService** - Capital allocation
- **AnalyticsService** - Analytics and reporting
- **BacktestService** - Backtesting operations
- **MLService** - Machine learning operations
- **ExchangeService** - Exchange connectivity
- **DataService** - Data management
- **StateService** - State management
- **MonitoringService** - System monitoring

### Authentication & Authorization
- All endpoints except `/api/auth/login` require JWT authentication
- Implement role-based access control (RBAC)
- Support API key authentication for programmatic access
- Implement rate limiting per user/API key

### WebSocket Connections
Real-time updates via WebSocket for:
- Dashboard metrics
- Bot status updates
- Market data streaming
- Order and trade updates
- System monitoring

### Response Format
```json
{
  "success": true,
  "data": {},
  "timestamp": "2024-01-01T00:00:00Z",
  "correlation_id": "uuid"
}
```

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message",
    "details": {}
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "correlation_id": "uuid"
}
```

### Pagination
List endpoints support pagination:
```
GET /api/endpoint?page=1&limit=20&sort=created_at&order=desc
```

### Filtering
List endpoints support filtering:
```
GET /api/orders?status=open&symbol=BTC/USDT&start_date=2024-01-01
```

### Caching Strategy
- Use Redis for frequently accessed data
- Cache market data with short TTL (5-30 seconds)
- Cache dashboard metrics with medium TTL (1-5 minutes)
- Cache historical data with long TTL (1-24 hours)
- Implement cache invalidation on data updates

### Performance Requirements
- API response time < 200ms for cached data
- API response time < 1s for database queries
- WebSocket latency < 50ms
- Support 1000+ concurrent connections
- Handle 10,000+ requests per minute

### Security Requirements
- HTTPS only
- CORS configuration for frontend
- Input validation on all endpoints
- SQL injection prevention
- XSS protection
- Rate limiting
- Request size limits
- Audit logging for sensitive operations

### Monitoring & Observability
- Request/response logging
- Performance metrics (latency, throughput)
- Error tracking
- Business metrics (trades, P&L)
- Health checks
- Distributed tracing
- Alert configuration

### API Versioning
Consider implementing versioning:
- URL versioning: `/api/v1/endpoint`
- Header versioning: `API-Version: 1.0`
- Support backward compatibility

### Documentation
- OpenAPI/Swagger specification
- Interactive API documentation
- Code examples in multiple languages
- WebSocket protocol documentation
- Authentication guide
- Rate limiting documentation
- Error code reference

## Development Priority

### Phase 1 - Core Functionality
1. Authentication endpoints
2. Bot management endpoints
3. Basic dashboard endpoints
4. Market data endpoints
5. Order/trade endpoints

### Phase 2 - Advanced Features
1. Strategy management
2. Risk management
3. Portfolio management
4. Basic analytics
5. WebSocket connections

### Phase 3 - Enterprise Features
1. ML model management
2. Advanced analytics
3. Backtesting
4. Optimization
5. Advanced monitoring

### Phase 4 - Polish & Scale
1. Performance optimization
2. Advanced caching
3. Horizontal scaling
4. Enhanced security
5. Comprehensive documentation