# Claude Code Project Configuration - T-Bot Trading System

## Project Overview
T-Bot is a cryptocurrency trading bot with multiple exchange integrations, risk management, and automated trading strategies.

## Environment Setup
- **Python Version**: 3.10.12
- **Virtual Environment**: WSL Ubuntu at `~/.venv/bin/activate`
- **Working Directory**: `/mnt/e/Work/P-41 Trading/code/t-bot`

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

## Code Quality Agents

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


Current task:
 We need to complete work according to the specifications. Only end chat when all work is done. project manager agent should assign work to fastapi-performance-expert agent for writing web backend, assign react-ui-architect for all the frontend related stuff. You need to used python-ml-optimization-expert for writing any AI/ML related code and other python code which AI/ML will use. Then you need to ask python-qa-test-engineer to ensure Q/A and in the end you need to ask fintech-trading-reviewer agent to review code done so far. Grab 1 prompt at a time, work on it using agents and once all issues are fixed, run all tests to make sure everything is working, then push the code to git using default branch main with appropriate git message, and move to the next task. Think carefully at each step. This is a financial app and we cannot afford any mistakes. If you think some work can be done in parallel, you can opt for that as well but be extra careful. Make sure you write tests and verify they are all passing. Let's also create a help page on UI which will have all the documentation related to the project. Also add a README.md and SETUP.md files to setup project locally.
 While the system is feature-complete, the fintech review identified some critical improvements needed before live trading:
1. Fix Kelly Criterion implementation with half-Kelly and bounds
2. Add order idempotency for duplicate prevention
3. Maintain Decimal precision throughout calculations
4. Implement correlation-based circuit breakers
The whole code should be audited thorougly and must have atleast 90% test coverage. Make sure there are no mistakes. The web module should be complete as well. 
2 more features that I want to add:
1- Add a Playground UI using which, we can see different steps/stages of order, i.e. position size selector, side selector, symbol selector, risk settings, SL/TP settings, portfolio settings, strategy selection, etc. We will either select these various settings using selector or input values whereever required and start the bot and it should run the bot (either on historical data or live data). It should also have sandbox/live options. We also need the backtesting in this playground. The idea is this: We select various settings, the model to use and different strategic options and then we run the bot, the bot will run in the background and we can see the status of bot along with logs and results in bot page.s 
2- Add a brute force option either in playground or some other screen (decide where it should be) to run N number of bots with all possible settings, strategy selection, model selection etc. The goal is to come up with a specific parameters and model selection which will yeild most returns. We need to make sure that data is never overfitting or underfitting. Be extra careful with this and think thorougly before doing it. You might want to create additional modules to have this functionality. Make sure you re-use code where ever possible.
After all this, do a thorough code review from all aspects either formatting, testing, or logical. This is a financial app, ultra think about everything and make sure there are no issues and this system will eventually yield profits when connected with live exchanges. 