# CLAUDE_v2.md - T-Bot Trading System Reference

## 🎯 Quick Context
**Project**: Cryptocurrency trading bot system with ML capabilities
**Tech Stack**: Python 3.12, FastAPI, React, Redux, PostgreSQL, Redis, InfluxDB
**Architecture**: Microservices with WebSocket real-time updates
**Current State**: Frontend bot creation wizard complete, 15+ strategies implemented

## 📁 Critical File Locations

### Backend Entry Points
- `src/main.py` - System initialization
- `src/web_interface/app.py` - FastAPI application
- `src/bot_management/bot_instance.py` - Bot lifecycle
- `src/strategies/base.py` - Strategy interface
- `src/risk_management/base.py` - Risk management
- `src/exchanges/base.py` - Exchange interface

### Frontend Entry Points
- `frontend/src/App.tsx` - React application
- `frontend/src/components/Bots/BotCreationWizardShadcn.tsx` - Bot wizard (1400+ lines)
- `frontend/src/pages/BotManagementPage.tsx` - Bot management
- `frontend/src/store/slices/botSlice.ts` - Bot state management
- `frontend/src/services/api/` - API services

## 🏗️ System Architecture

### Service Layer Pattern
```python
# All services follow this pattern:
class ServiceName(BaseComponent):
    def __init__(self, dependencies...):
        # Dependency injection
    async def start(self):
        # Initialization
    async def stop(self):
        # Cleanup
```

### API Structure
```
/api/
├── /auth/          # Authentication
├── /bots/          # Bot management
├── /portfolio/     # Portfolio operations
├── /strategies/    # Strategy operations
├── /risk/          # Risk management
├── /trading/       # Order execution
└── /ml/           # ML models
```

## 🤖 Bot Management

### Bot Lifecycle States
```python
CREATED → STARTING → RUNNING → PAUSED/STOPPING → STOPPED
                    ↓
                  ERROR (requires intervention)
```

### Bot Configuration Structure
```python
{
    "bot_name": str,
    "strategy_name": StrategyType,  # 15+ types
    "exchanges": List[str],
    "symbols": List[str],
    "allocated_capital": float,
    "risk_percentage": float,
    "strategy_config": {...}
}
```

## 📊 Trading Strategies

### Available Strategy Types (15+)
```python
# Static Strategies
MEAN_REVERSION = "mean_reversion"
MOMENTUM = "momentum"
ARBITRAGE = "arbitrage"
MARKET_MAKING = "market_making"
TREND_FOLLOWING = "trend_following"
PAIRS_TRADING = "pairs_trading"
STATISTICAL_ARBITRAGE = "statistical_arbitrage"
BREAKOUT = "breakout"
CROSS_EXCHANGE_ARBITRAGE = "cross_exchange_arbitrage"
TRIANGULAR_ARBITRAGE = "triangular_arbitrage"

# Dynamic Strategies
VOLATILITY_BREAKOUT = "volatility_breakout"

# Hybrid Strategies
ENSEMBLE = "ensemble"
FALLBACK = "fallback"
RULE_BASED_AI = "rule_based_ai"

# Evolutionary
GENETIC = "genetic"
```

### Strategy Implementation Pattern
```python
class StrategyName(BaseStrategy):
    async def generate_signals(self, data: MarketData) -> TradingSignal:
        # Signal generation logic
        return TradingSignal(...)
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        # Validation logic
        return True/False
```

## 🛡️ Risk Management

### Circuit Breakers
- **DailyLossLimit**: Stops at 5% daily loss
- **DrawdownLimit**: Stops at 15% drawdown
- **VolatilitySpike**: Market volatility protection
- **ModelConfidence**: ML degradation detection
- **SystemErrorRate**: System stability
- **CorrelationSpike**: Portfolio correlation risk

### Risk Controls
```python
position_size = min(
    kelly_criterion_size,
    max_position_limit,
    available_capital * risk_percentage
)
```

## 🔗 Exchange Integration

### Supported Exchanges
- **Binance** (`src/exchanges/binance.py`)
- **OKX** (`src/exchanges/okx.py`)
- **Coinbase** (`src/exchanges/coinbase.py`)

### Rate Limiting
```python
# Advanced rate limiter with burst capacity
rate_limiter.check_and_wait(endpoint, weight)
```

## 🎨 Frontend Architecture

### Redux State Structure
```typescript
{
  auth: { user, token, isAuthenticated },
  bots: { list, loading, error },
  portfolio: { positions, balance, history },
  websocket: { connected, messages },
  strategies: { available, templates }
}
```

### Key Components
- `BotCreationWizardShadcn`: 7-step wizard with strategy configuration
- `TradingPage`: Live trading interface
- `RiskDashboard`: Risk monitoring
- `PlaygroundPage`: Strategy testing

## 🔄 WebSocket Events

### Client → Server
```javascript
socket.emit('subscribe_market_data', { symbols })
socket.emit('bot_command', { bot_id, command })
```

### Server → Client
```javascript
socket.on('market_data_update', (data) => {})
socket.on('bot_status_change', (status) => {})
socket.on('portfolio_update', (portfolio) => {})
socket.on('risk_alert', (alert) => {})
```

## 📦 Database Schema

### Key Tables
- `bots` - Bot configurations
- `trades` - Executed trades
- `positions` - Active positions
- `strategies` - Strategy configurations
- `risk_metrics` - Risk calculations
- `ml_models` - ML model metadata

## 🚀 Common Operations

### Create Bot
```python
POST /api/bots
{
    "bot_name": "My Bot",
    "strategy_name": "mean_reversion",
    "exchanges": ["binance"],
    "symbols": ["BTC/USDT"],
    "allocated_capital": 1000
}
```

### Start/Stop Bot
```python
POST /api/bots/{bot_id}/start
POST /api/bots/{bot_id}/stop
```

### Get Portfolio
```python
GET /api/portfolio
Response: {
    "total_value": 10000,
    "positions": [...],
    "pnl": {...}
}
```

## 🔧 Development Commands

### Backend
```bash
# Activate environment
source ~/.venv/bin/activate

# Run backend
cd /mnt/e/Work/P-41\ Trading/code/t-bot
python -m src.main

# Run tests
pytest tests/ -v

# Format code
ruff format src/
black src/ --line-length 100
```

### Frontend
```bash
# Install dependencies
cd frontend && npm install

# Start development server
npm start  # Runs on port 3000

# Build production
npm run build

# Run tests
npm test
```

## 🐛 Common Issues & Solutions

### WebSocket Connection Failed
- Check backend is running on port 8000
- Verify CORS settings in `app.py`

### Strategy Not Loading
- Ensure strategy is registered in `COMPREHENSIVE_STRATEGIES`
- Check strategy type enum matches backend

### Bot Creation Fails
- Verify all required fields in request
- Check capital allocation limits
- Ensure exchange credentials configured

## 📝 Code Patterns

### Async Pattern
```python
async def operation():
    async with self.lock:
        result = await async_operation()
    return result
```

### Error Handling
```python
@with_error_handling
async def risky_operation():
    # Automatic retry and logging
    pass
```

### Repository Pattern
```python
repository = BotRepository(db_session)
bot = await repository.get_by_id(bot_id)
await repository.update(bot)
```

## 🔑 Important Constants

### Risk Limits
- Max position size: 10% of capital
- Daily loss limit: 5%
- Max drawdown: 15%
- Min confidence: 0.6

### Performance Targets
- Order execution: < 50ms
- Market data processing: < 10ms
- WebSocket latency: < 25ms
- API response time: < 200ms

## 📊 Monitoring

### Key Metrics
- Bot P&L and Sharpe ratio
- System latency and throughput
- Risk metric violations
- ML model performance

### Alert Channels
- WebSocket real-time alerts
- Email notifications
- Slack/Discord webhooks
- Dashboard notifications

## 🚦 Status Codes

### Bot Status
- `RUNNING`: Active trading
- `PAUSED`: Temporarily stopped
- `ERROR`: Requires intervention
- `STOPPED`: Fully stopped

### Circuit Breaker States
- `CLOSED`: Normal operation
- `OPEN`: Trading halted
- `HALF_OPEN`: Testing recovery

---

## 🎯 Quick Reference for Common Tasks

### Add New Strategy
1. Create strategy file in `src/strategies/static/`
2. Inherit from `BaseStrategy`
3. Implement `generate_signals()` and `validate_signal()`
4. Add to `COMPREHENSIVE_STRATEGIES` in frontend
5. Register in strategy factory

### Add API Endpoint
1. Create route in `src/web_interface/api/`
2. Add to API facade if needed
3. Update frontend API service
4. Add TypeScript types
5. Update Redux slice if stateful

### Debug Bot Issue
1. Check bot status: `GET /api/bots/{bot_id}`
2. Review logs: `docker logs t-bot-backend`
3. Check risk metrics: `GET /api/risk/metrics`
4. Verify exchange connection
5. Review recent trades

---

**Last Updated**: 2024-12-25
**Version**: 2.0.0
**Maintainer**: LLM-Assisted Development