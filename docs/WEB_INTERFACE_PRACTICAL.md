# T-Bot Web Interface API - Practical Implementation Guide

## Overview
This document provides a practical, implementable API specification for the T-Bot trading system. These endpoints are directly supported by existing service implementations and focus on the core goal: **creating and managing profitable trading bots**.

## Core Use Cases
1. **Sandbox Testing**: Run bots with fake money to test strategies
2. **Backtesting**: Test strategies on historical data before risking capital
3. **Live Trading**: Deploy profitable strategies with real money
4. **Multi-Bot Management**: Run multiple strategies simultaneously
5. **Performance Monitoring**: Track P&L and risk in real-time

---

## Phase 1: Essential Trading Operations (Implement First)

### 1. Environment Management
**Critical for switching between sandbox/production modes**

| Method | Endpoint | Description | Service |
|--------|----------|-------------|---------|
| GET | `/api/system/environment` | Get current environment (sandbox/production) | Config |
| POST | `/api/system/environment` | Switch environment mode | Config |
| GET | `/api/system/config` | Get system configuration | Config |

**Request Example:**
```json
POST /api/system/environment
{
  "mode": "sandbox",  // or "production"
  "exchange": "binance"
}
```

### 2. Bot Management
**Core functionality for bot lifecycle**

| Method | Endpoint | Description | Service |
|--------|----------|-------------|---------|
| POST | `/api/bots` | Create new bot | BotService.create_bot() |
| GET | `/api/bots` | List all bots | BotService.list_bots() |
| GET | `/api/bots/{bot_id}` | Get bot details | BotService.get_bot() |
| POST | `/api/bots/{bot_id}/start` | Start bot | BotService.start_bot() |
| POST | `/api/bots/{bot_id}/stop` | Stop bot | BotService.stop_bot() |
| DELETE | `/api/bots/{bot_id}` | Delete bot | BotService.delete_bot() |
| GET | `/api/bots/{bot_id}/metrics` | Get bot performance | BotService.get_bot_metrics() |

**Create Bot Request:**
```json
POST /api/bots
{
  "name": "BTC Mean Reversion Bot",
  "strategy_id": "mean_reversion",
  "exchange": "binance",
  "symbols": ["BTC/USDT"],
  "capital_allocation": "1000.00",
  "risk_config": {
    "max_position_size_pct": 0.1,
    "stop_loss_pct": 0.02
  }
}
```

### 3. Trading Operations
**Essential for order execution and position management**

| Method | Endpoint | Description | Service |
|--------|----------|-------------|---------|
| POST | `/api/orders` | Place manual order | ExecutionService.place_order() |
| GET | `/api/orders` | List orders | ExecutionService.get_orders() |
| GET | `/api/orders/{order_id}` | Get order details | ExecutionService.get_order() |
| DELETE | `/api/orders/{order_id}` | Cancel order | ExecutionService.cancel_order() |
| GET | `/api/positions` | Get open positions | ExecutionService.get_positions() |
| POST | `/api/positions/{position_id}/close` | Close position | ExecutionService.close_position() |

### 4. Market Data
**Minimal market data for trading decisions**

| Method | Endpoint | Description | Service |
|--------|----------|-------------|---------|
| GET | `/api/market/symbols` | Available trading pairs | ExchangeService.get_symbols() |
| GET | `/api/market/ticker/{symbol}` | Current price | ExchangeService.get_ticker() |
| GET | `/api/market/candles/{symbol}` | OHLCV data | DataService.get_market_data() |

### 5. System Health
**Monitor system status**

| Method | Endpoint | Description | Service |
|--------|----------|-------------|---------|
| GET | `/api/health` | Basic health check | All services health() |
| GET | `/api/health/detailed` | Detailed component status | All services health_detailed() |

---

## Phase 2: Strategy & Backtesting (2-3 weeks)

### 6. Strategy Management
**Configure and manage trading strategies**

| Method | Endpoint | Description | Service |
|--------|----------|-------------|---------|
| GET | `/api/strategies` | List available strategies | StrategyService.list_strategies() |
| GET | `/api/strategies/{strategy_id}` | Get strategy details | StrategyService.get_strategy() |
| POST | `/api/strategies/validate` | Validate strategy config | StrategyService.validate_strategy() |
| GET | `/api/strategies/{strategy_id}/signals` | Get current signals | StrategyService.get_signals() |

### 7. Backtesting
**Test strategies on historical data**

| Method | Endpoint | Description | Service |
|--------|----------|-------------|---------|
| POST | `/api/backtest` | Run backtest | BacktestService.run_backtest() |
| GET | `/api/backtest/{backtest_id}` | Get results | BacktestService.get_result() |
| GET | `/api/backtest/{backtest_id}/metrics` | Detailed metrics | BacktestService.get_metrics() |
| POST | `/api/backtest/quick` | Quick backtest (last 30 days) | BacktestService.quick_test() |

**Backtest Request:**
```json
POST /api/backtest
{
  "strategy_config": {
    "type": "mean_reversion",
    "parameters": {"lookback": 20, "threshold": 2.0}
  },
  "symbols": ["BTC/USDT"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "initial_capital": "10000.00"
}
```

### 8. Portfolio & Capital
**Monitor portfolio and manage capital**

| Method | Endpoint | Description | Service |
|--------|----------|-------------|---------|
| GET | `/api/portfolio/overview` | Portfolio summary | CapitalService.get_portfolio_overview() |
| GET | `/api/portfolio/balances` | Asset balances | CapitalService.get_balances() |
| GET | `/api/portfolio/pnl` | P&L analysis | AnalyticsService.calculate_pnl() |
| POST | `/api/capital/allocate` | Allocate capital to bot | CapitalService.allocate_capital() |

### 9. Risk Management
**Monitor and control risk**

| Method | Endpoint | Description | Service |
|--------|----------|-------------|---------|
| GET | `/api/risk/metrics` | Current risk metrics | RiskService.get_risk_metrics() |
| PUT | `/api/risk/limits` | Update risk limits | RiskService.update_limits() |
| GET | `/api/risk/exposure` | Position exposure | RiskService.calculate_exposure() |

---

## Phase 3: Real-time & Analytics (3-4 weeks)

### 10. WebSocket Connections
**Real-time updates for UI**

| Protocol | Endpoint | Description | Service |
|----------|----------|-------------|---------|
| WS | `/ws/bots/{bot_id}` | Bot status updates | BotService events |
| WS | `/ws/portfolio` | Portfolio updates | CapitalService events |
| WS | `/ws/trades` | Trade execution updates | ExecutionService events |

**WebSocket Message Format:**
```json
{
  "type": "bot_status",
  "bot_id": "bot_123",
  "data": {
    "status": "running",
    "pnl": "125.50",
    "open_positions": 2
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 11. Analytics & Reporting
**Performance analysis and reporting**

| Method | Endpoint | Description | Service |
|--------|----------|-------------|---------|
| GET | `/api/analytics/performance` | Performance metrics | AnalyticsService.get_performance() |
| GET | `/api/analytics/trades` | Trade analysis | AnalyticsService.analyze_trades() |
| POST | `/api/analytics/report` | Generate report | AnalyticsService.generate_report() |
| GET | `/api/analytics/export` | Export data (CSV/JSON) | AnalyticsService.export_data() |

---

## Implementation Examples

### Bot Creation Flow
```python
# 1. Validate strategy
POST /api/strategies/validate
{
  "type": "mean_reversion",
  "parameters": {"lookback": 20}
}

# 2. Run quick backtest
POST /api/backtest/quick
{
  "strategy_config": {...},
  "symbol": "BTC/USDT"
}

# 3. Create bot if profitable
POST /api/bots
{
  "name": "BTC Bot",
  "strategy_id": "mean_reversion",
  "capital_allocation": "1000.00"
}

# 4. Start bot
POST /api/bots/{bot_id}/start
```

### Sandbox Testing Flow
```python
# 1. Switch to sandbox
POST /api/system/environment
{"mode": "sandbox"}

# 2. Create test bot
POST /api/bots
{
  "name": "Test Bot",
  "strategy_id": "test_strategy"
}

# 3. Run bot with fake money
POST /api/bots/{bot_id}/start

# 4. Monitor performance
GET /api/bots/{bot_id}/metrics
```

---

## Service Layer Mapping

Each endpoint maps directly to existing service methods:

```python
# BotService
- create_bot(bot_config)
- start_bot(bot_id)
- stop_bot(bot_id)
- delete_bot(bot_id)
- get_bot_metrics(bot_id)
- list_bots()

# ExecutionService
- place_order(order_request)
- cancel_order(order_id)
- get_orders(filters)
- get_positions()

# BacktestService
- run_backtest(request)
- get_result(backtest_id)
- get_metrics(backtest_id)

# RiskService
- get_risk_metrics()
- calculate_position_size(signal)
- update_limits(limits)

# StrategyService
- list_strategies()
- validate_strategy(config)
- get_signals(strategy_id)

# AnalyticsService
- calculate_pnl(bot_id)
- get_performance(bot_id)
- generate_report(params)
```

---

## Authentication (Simple Implementation)

### Basic JWT Authentication
```python
# Login
POST /api/auth/login
{
  "username": "trader",
  "password": "password"
}
Response: {"token": "jwt_token"}

# All other requests
Headers: {
  "Authorization": "Bearer jwt_token"
}
```

---

## Error Handling

### Standard Error Response
```json
{
  "success": false,
  "error": {
    "code": "INSUFFICIENT_CAPITAL",
    "message": "Not enough capital to create bot",
    "details": {
      "required": "1000.00",
      "available": "500.00"
    }
  }
}
```

### Common Error Codes
- `INVALID_CONFIG` - Invalid bot/strategy configuration
- `INSUFFICIENT_CAPITAL` - Not enough funds
- `RISK_LIMIT_EXCEEDED` - Risk limits violated
- `EXCHANGE_ERROR` - Exchange API error
- `BACKTEST_FAILED` - Backtesting error

---

## Development Priority

### Week 1-2: Core Trading
1. Environment switching (sandbox/production)
2. Bot CRUD operations
3. Basic health checks
4. Manual order placement

### Week 3-4: Strategy & Backtesting
1. Strategy validation
2. Backtesting engine
3. Portfolio overview
4. Risk metrics

### Week 5-6: Real-time & Polish
1. WebSocket connections
2. Performance analytics
3. Report generation
4. UI integration

---

## Testing Strategy

### Sandbox Testing Checklist
- [ ] Create bot in sandbox mode
- [ ] Execute trades with fake money
- [ ] Monitor bot performance
- [ ] Test stop-loss triggers
- [ ] Validate risk limits

### Production Readiness
- [ ] All endpoints return < 200ms
- [ ] Error handling for all edge cases
- [ ] Rate limiting implemented
- [ ] Authentication working
- [ ] Logging and monitoring active

---

## Success Metrics

### Phase 1 Success (2 weeks)
- Can create and run a bot in sandbox
- Can monitor bot status and P&L
- Can switch between environments
- Basic order execution working

### Phase 2 Success (4 weeks)
- Full backtesting workflow
- Multiple bots running simultaneously
- Risk limits enforced
- Portfolio tracking accurate

### Phase 3 Success (6 weeks)
- Real-time updates via WebSocket
- Comprehensive analytics
- Professional trading dashboard
- Production-ready system

---

## Notes

1. **Start Simple**: Focus on getting one bot running profitably before adding complexity
2. **Sandbox First**: Always test in sandbox before production
3. **Service Layer**: Use existing services, don't recreate logic
4. **Incremental**: Build and test incrementally
5. **User-Focused**: Prioritize features that help make money