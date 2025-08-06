# Integration Points and Reverse Integration Guide

## Overview
This document defines how components integrate with each other and critical reverse integration requirements to ensure sequential prompt execution results in a complete, working application.

## Core Integration Patterns

### 1. Exception Hierarchy (P-001 → All Prompts)
**Forward Integration:**
- All prompts must import and use exceptions from `src.core.exceptions`
- Never create duplicate exception classes

**Reverse Integration Required:**
- **P-002A**: Add new specific exceptions to P-001's exception hierarchy
- **P-003+**: Use existing `ExchangeError` from P-001, don't create new ones
- **P-008+**: Use existing `RiskManagementError` from P-001
- **P-017+**: Use existing `ModelError` from P-001

### 2. Type Definitions (P-001 → All Prompts)
**Forward Integration:**
- All prompts must use types from `src.core.types`
- Import `Signal`, `MarketData`, `Position`, `OrderRequest`, `OrderResponse`

**Reverse Integration Required:**
- **P-003**: Add exchange-specific types to P-001's types.py, not create separate files
- **P-008**: Add risk-specific types to P-001's types.py
- **P-011**: Add strategy-specific types to P-001's types.py
- **P-017**: Add ML-specific types to P-001's types.py

### 3. Configuration Management (P-001 → All Components)
**Forward Integration:**
- All configuration classes inherit from P-001's base config
- Use P-001's environment variable patterns

**Reverse Integration Required:**
- **P-002**: Add database config classes to P-001's config.py
- **P-003**: Add exchange config classes to P-001's config.py
- **P-008**: Add risk config classes to P-001's config.py
- **P-026**: Add web interface config to P-001's config.py

### 4. Database Models (P-002 → Data Storage)
**Forward Integration:**
- All prompts storing data must use P-002's models
- Import models, don't recreate them

**Reverse Integration Required:**
- **P-008**: Update P-002's models to add risk-related fields
- **P-011**: Update P-002's models to add strategy-specific fields  
- **P-017**: Update P-002's models to add ML model metadata fields
- **P-021**: Update P-002's models to add bot instance fields

### 5. Error Handling Framework (P-002A → All Components)
**Forward Integration:**
- All prompts must use P-002A's error handling patterns
- Apply retry decorators and circuit breakers

**Reverse Integration Required:**
- **P-003+**: Integrate specific recovery scenarios from P-002A
- **P-008+**: Use P-002A's validation patterns for risk checks
- **P-020**: Implement P-002A's partial fill recovery in execution engine
- **P-026+**: Use P-002A's input validation patterns

### 6. Utility Framework (P-007A → All Components) 
**Forward Integration:**
- All prompts must use decorators and validators from P-007A
- Apply performance monitoring and validation decorators

**Reverse Integration Required:**
- **P-003+**: Add @time_execution decorators to all exchange API calls
- **P-008+**: Use validation utilities for all risk parameter validation
- **P-011+**: Apply @retry and @circuit_breaker decorators to strategy methods
- **P-017+**: Use @cache_result decorators for model inference
- **P-026+**: Use formatters for all API responses

## Critical Shared Components

### BaseStrategy Interface (P-011)
**Who Must Use:**
- P-012: Static strategies inherit from BaseStrategy
- P-013: Dynamic strategies inherit from BaseStrategy  
- P-013A-E: All strategy types inherit from BaseStrategy
- P-019: AI strategies inherit from BaseStrategy

**Reverse Integration:**
- **P-013A-E**: Update P-011's BaseStrategy to add methods needed by advanced strategies
- **P-019**: Update P-011's BaseStrategy to add AI-specific abstract methods

### BaseExchange Interface (P-003)
**Who Must Use:**
- P-004: BinanceExchange(BaseExchange)
- P-005: OKXExchange(BaseExchange)
- P-006: CoinbaseExchange(BaseExchange)

**Reverse Integration:**
- **P-004-006**: Update P-003's BaseExchange if new methods are needed
- **P-007**: Update P-003's rate limiting interface based on exchange-specific needs

### Risk Management Integration (P-008 → Execution)
**Forward Integration:**
- P-020: Execution engine must validate all orders through risk management
- P-021: Bot instances must enforce risk limits

**Reverse Integration:**
- **P-010A**: Update P-008's position sizing to use capital management
- **P-020**: Update P-008's risk validation to handle execution-specific scenarios

## Database Schema Evolution

### Model Updates Required
```python
# P-002 must be updated by subsequent prompts:

# P-008 adds risk fields:
class Position:
    risk_score: float
    risk_adjusted_size: Decimal
    
# P-011 adds strategy fields:
class BotInstance:
    strategy_version: str
    strategy_parameters: JSON
    
# P-017 adds ML fields:
class MLModel:
    confidence_threshold: float
    fallback_strategy: str
```

## API Evolution Patterns

### REST API Endpoints (P-027)
**Reverse Integration Required:**
- **P-013A**: Add arbitrage-specific endpoints to P-027
- **P-013B**: Add market making endpoints to P-027
- **P-017**: Add model management endpoints to P-027 (already added)
- **P-021**: Add bot pause/resume endpoints to P-027 (already added)

### WebSocket Streams (P-028)
**Reverse Integration Required:**
- **P-013A**: Add arbitrage opportunity streams to P-028
- **P-013B**: Add market making status streams to P-028
- **P-017**: Add model prediction streams to P-028

## Configuration File Evolution

### Master Configuration (P-001)
Must be updated to include:
```yaml
# Added by P-008
risk_management:
  position_sizing: kelly_criterion
  max_drawdown: 0.15
  
# Added by P-010A  
capital_management:
  base_currency: USDT
  allocation_strategy: dynamic
  
# Added by P-013A
arbitrage:
  min_profit_threshold: 0.001
  max_execution_time: 500
```

## Critical Integration Checkpoints

### After P-001: Core Foundation
- [ ] Exception hierarchy complete
- [ ] Type definitions comprehensive
- [ ] Configuration framework extensible

### After P-002A: Error Handling
- [ ] All components use consistent error handling
- [ ] Recovery scenarios documented
- [ ] Circuit breakers integrated

### After P-011: Strategy Framework
- [ ] BaseStrategy interface complete
- [ ] Strategy factory functional
- [ ] Configuration integration working

### After P-020: Execution Engine
- [ ] Risk integration complete
- [ ] Error handling integrated
- [ ] Performance monitoring active

### After P-026: Web Interface
- [ ] All API endpoints functional
- [ ] Security properly integrated
- [ ] Configuration management working

## Integration Validation Commands

```bash
# After each major milestone, run:
python -m pytest tests/integration/
python -m mypy src/
python -c "from src.core.config import Config; Config()"
python -c "from src.strategies.factory import StrategyFactory; StrategyFactory.list_strategies()"
``` 