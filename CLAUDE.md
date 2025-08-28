# T-Bot Trading System - Claude Configuration v2

## Project Goal
Build a production-ready cryptocurrency trading bot with institutional-grade reliability, comprehensive risk management, and ML-powered strategies.

## Environment
- **Python**: 3.10.12 with venv at `~/.venv/bin/activate`
- **Directory**: `/mnt/e/Work/P-41 Trading/code/t-bot`
- **Node**: 18.19.0 at `$HOME/.nvm/versions/node/v18.19.0/bin`

## Critical Standards
- **FOLLOW**: `CODING_STANDARDS.md` and `COMMON_PATTERNS.md` exactly
- **Financial Precision**: ALWAYS use `Decimal`, NEVER `float` for money
- **Database**: Use `DECIMAL(20,8)` for crypto, relationships need `back_populates`
- **Services**: Controllers→Services→Repositories (never skip layers)

## Module Hierarchy (Process in Order)
1. **core** - Base classes, exceptions, types, config, DI container
2. **utils** - Validators, formatters, decorators 
3. **error_handling** - Decorators, handlers, circuit breakers
4. **database** - Models, repositories, UoW, migrations
5. **monitoring, state, data** - Infrastructure services
6. **exchanges, risk_management** - Trading components
7. **execution, ml, strategies** - Business logic
8. **analytics, backtesting, capital/bot_management, web_interface** - Applications

## Core Module Usage
```python
# MANDATORY imports for ALL modules
from src.core.base import BaseComponent, BaseService
from src.core.exceptions import ValidationError, ServiceError  # Use specific
from src.core.logging import get_logger
from src.core.types import *  # Trading types
from src.utils.decorators import retry, with_circuit_breaker
```

## Required Agents (Use Task tool for ALL)
### Code Quality & Testing
- `code-guardian-enforcer` - Prevent violations & service bypasses
- `integration-test-architect` - Create comprehensive integration tests
- `module-integration-validator` - Verify module dependencies work
- `financial-qa-engineer` - Test edge cases & chaos scenarios
- `quality-control-enforcer` - Orchestrate deployment readiness

### Trading & Financial
- `algo-trading-specialist` - Trading algorithms & backtesting
- `trading-system-architect` - HFT systems & order routing
- `trading-engine-developer` - Execution systems & smart routing
- `risk-management-expert` - VaR, position sizing, risk controls
- `quantitative-strategy-researcher` - Strategy validation & alpha
- `quant-ml-strategist` - ML models for trading
- `technical-indicator-specialist` - Implement & optimize indicators
- `portfolio-optimization-expert` - MPT & allocation strategies
- `backtest-simulation-expert` - Historical testing frameworks

### Data & Infrastructure
- `market-data-expert` - Real-time feeds & exchange APIs
- `data-pipeline-maestro` - Stream processing & data integrity
- `realtime-pipeline-architect` - High-throughput data pipelines
- `postgres-timescale-architect` - Time-series DB optimization
- `timeseries-db-optimizer` - Query performance & retention
- `redis-cache-optimizer` - Cache warming & TTL strategies
- `message-queue-architect` - RabbitMQ/Redis Streams setup
- `celery-task-orchestrator` - Distributed task queues

### APIs & Integration
- `financial-api-architect` - High-performance trading APIs
- `exchange-integration-specialist` - Exchange connections & FIX
- `rate-limit-architect` - Token bucket & throttling
- `auth-security-architect` - JWT, OAuth2, API keys

### Security & Operations
- `financial-security-expert` - Threat assessment & security
- `cybersecurity-guardian` - Vulnerability analysis
- `infrastructure-wizard` - Deployment & monitoring
- `trading-ops-monitor` - Alerting & compliance
- `config-management-expert` - Feature flags & hot-reload

### UI & Documentation
- `realtime-dashboard-architect` - WebSocket/React dashboards
- `trading-ui-specialist` - Trading interfaces & order books
- `financial-docs-writer` - Technical documentation
- `mcp__playwright__*` - Browser automation testing

### Project Coordination
- `strategic-project-coordinator` - High-level orchestration
- `tactical-task-coordinator` - Daily task management
- `pipeline-execution-orchestrator` - Multi-agent workflows
- `knowledge-synthesis-orchestrator` - Capture patterns & learnings
- `automation-orchestrator` - Self-healing systems

### Architecture & Design
- `system-design-architect` - API contracts, data models
- `integration-architect` - Find existing services first
- `portfolio-simulation-architect` - Virtual portfolios & matching

## Quality Commands (Run After Changes)
```bash
# Format & Lint
ruff check src/ --fix && ruff format src/
black src/ --line-length 120

# Type Check
mypy src/ --ignore-missing-imports

# Test
pytest tests/ -v --tb=short
```

## Critical Files
- **Config**: `src/core/config.py` - All settings
- **Types**: `src/core/types/*.py` - Domain models  
- **Models**: `src/database/models/*.py` - DB schema
- **Services**: `src/*/service.py` - Business logic
- **Main**: `src/main.py` - Entry point

## Trading Validations
- Position size ≤ 2% of capital per trade
- Stop-loss required for all positions
- Order prices within market bounds (±10%)
- Total exposure ≤ 100% of portfolio

## Security Rules
- NEVER log API keys or passwords
- Validate ALL external inputs
- Use parameterized queries
- Rate limit exchange APIs

## Testing Requirements
- Unit tests: `tests/unit/test_MODULE/`
- Coverage: >70% (90% for trading/risk)
- Financial calculations: 100% coverage

## Agent Usage Patterns
**New Feature**: `integration-architect` → `system-design-architect` → `code-guardian-enforcer`
**Bug Fix**: `financial-qa-engineer` → fix → `integration-test-architect`
**Performance**: `performance-optimization-specialist` → `redis-cache-optimizer`
**Trading Strategy**: `algo-trading-specialist` → `backtest-simulation-expert` → `risk-management-expert`
**Database**: `postgres-timescale-architect` → `timeseries-db-optimizer`
**API**: `financial-api-architect` → `rate-limit-architect` → `auth-security-architect`

## Key Patterns
- **Service Pattern**: Inherit from `BaseService`
- **Repository Pattern**: Standard CRUD interface
- **Dependency Injection**: Use `DependencyContainer`
- **Error Handling**: Use decorators, not try-catch
- **Async Operations**: Always `await`, use `asyncio.gather()`

## Save Non-Production Files
Use `.claude_experiments/` for drafts, logs, experiments

---

## MANDATORY Agent Checks (Run Frequently)
After EVERY code change, run these agents:
1. `code-guardian-enforcer` - Catch violations immediately
2. `module-integration-validator` - Verify dependencies
3. `financial-qa-engineer` - Test financial accuracy
4. `integration-test-architect` - Ensure integration tests exist
5. `performance-optimization-specialist` - Check for bottlenecks

**Remember**: This is a FINANCIAL system - accuracy, testing, and risk controls are MANDATORY. Use agents liberally for precision!