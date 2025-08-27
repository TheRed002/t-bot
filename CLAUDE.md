# Claude Code Project Configuration - T-Bot Trading System

## Project Overview
T-Bot is a cryptocurrency trading bot with multiple exchange integrations, risk management, and automated trading strategies.

## Environment Setup
- **Python Version**: 3.10.12
- **Virtual Environment**: WSL Ubuntu at `~/.venv/bin/activate`
- **Working Directory**: `/mnt/e/Work/P-41 Trading/code/t-bot`
- **Setup tool**: `make`, `export PATH="$HOME/.nvm/versions/node/v18.19.0/bin:$PATH`

## Critical Commands
These commands should be run automatically when making code changes:

### Linting & Formatting
```bash
wsl -e bash -c "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && ruff check src/ --fix"
wsl -e bash -c "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && ruff format src/"
wsl -e bash -c "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && black src/ --line-length 100"
```

### Type Checking
```bash
wsl -e bash -c "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && mypy src/ --ignore-missing-imports"
```

### Testing
```bash
# Run all tests
wsl -e bash -c "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && pytest tests/ -v"

# Run with coverage
wsl -e bash -c "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && pytest tests/ --cov=src --cov-report=html --cov-report=term"

# Run specific test categories
wsl -e bash -c "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && pytest tests/unit/ -v"
wsl -e bash -c "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && pytest tests/integration/ -v"
```

## Code Agents

### 1. Formatting Agent
**Purpose**: Automatically format code according to project standards
**Batch Processing**: Process multiple files in parallel
```bash
# Format all Python files in batches
find src/ -name "*.py" | xargs -P 4 -n 10 bash -c 'for file; do ruff format "$file"; done' _
```

### 2. Logical Error Detection Agent
**Purpose**: Detect logical inconsistencies, race conditions, and architectural issues
**Focus Areas**:
- Async/await consistency
- Resource management (connections, locks)
- Error handling completeness
- State management in trading logic

### 3. Test Coverage Agent
**Purpose**: Ensure comprehensive test coverage
**Metrics**:
- Line coverage > 80%
- Branch coverage > 70%
- Critical paths 100% covered (trading operations, risk management)

### 4. Calculation Accuracy Agent
**Purpose**: Verify numerical calculations in trading logic
**Critical Areas**:
- Price calculations
- Position sizing
- Risk metrics
- Portfolio value calculations
- Fee calculations

### 5. Code Duplication Agent
**Purpose**: Identify and refactor duplicate code
**Tools**:
```bash
wsl -e bash -c "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && pylint src/ --disable=all --enable=duplicate-code"
```
**MENDATORY**: Always use specialized agents when available for a specific task. 

### 5. Frontend inspector Agent via Playwright
**Purpose**: Identify and fix UI related issues by inspecting UI and console logs


## Batch Processing Strategy

### Cost-Effective Analysis Pipeline
1. **File Grouping**: Process files by module to maintain context
2. **Parallel Execution**: Run independent checks simultaneously
3. **Incremental Processing**: Only analyze changed files
4. **Cache Results**: Store analysis results to avoid re-processing

### Priority Levels
- **P0 (Critical)**: Trading logic, order execution, risk management
- **P1 (High)**: Exchange connections, data processing, error handling
- **P2 (Medium)**: Strategies, indicators, utilities
- **P3 (Low)**: Documentation, tests, configuration

## Automated Workflow

### Pre-Commit Checks
```bash
# Run all quality checks before committing
wsl -e bash -c "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && ./scripts/pre-commit-checks.sh"
```

### Continuous Monitoring
```bash
# Watch for file changes and run appropriate checks
wsl -e bash -c "cd '/mnt/e/Work/P-41 Trading/code/t-bot' && source ~/.venv/bin/activate && ./scripts/watch-and-check.sh"
```

## Module Dependencies
- **P-001**: Core types, exceptions, config
- **P-002A**: Error handling components
- **P-007**: Advanced rate limiting
- **P-007A**: Utility functions

## Security Considerations
- Never log sensitive data (API keys, secrets)
- Validate all external inputs
- Use parameterized queries for database operations
- Implement rate limiting for all exchange APIs
- Secure WebSocket connections with proper authentication

## Performance Guidelines
- Use async/await for I/O operations
- Implement connection pooling
- Cache frequently accessed data
- Use batch operations for database writes
- Monitor memory usage in long-running processes

## Trading-Specific Validations
- Order price must be within market bounds
- Position size must respect account limits
- Stop-loss must be set for all positions
- Risk per trade must not exceed configured percentage
- Total exposure must stay within portfolio limits

## Directories
- If you want to generate any image, code, logs, summary or anything which is not related to production project, please save them in .claude_experiments directory at project root so that we don't clutter the actual code. 

