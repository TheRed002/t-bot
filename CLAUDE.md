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


Current Task:
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /home/bbc/.venv/lib/python3.10/site-packages (from rich->keras>=3.10.0->tensorflow[and-cuda]) (2.19.2)
Requirement already satisfied: mdurl~=0.1 in /home/bbc/.venv/lib/python3.10/site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.10.0->tensorflow[and-cuda]) (0.1.2)
Downloading tensorflow-2.20.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (620.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 620.4/620.4 MB 5.7 MB/s  0:01:13
Downloading tensorboard-2.20.0-py3-none-any.whl (5.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.5/5.5 MB 8.1 MB/s  0:00:00
Installing collected packages: tensorboard, tensorflow
  Attempting uninstall: tensorboard
    Found existing installation: tensorboard 2.19.0
    Uninstalling tensorboard-2.19.0:
      Successfully uninstalled tensorboard-2.19.0
  Attempting uninstall: tensorflow
    Found existing installation: tensorflow 2.19.0
    Uninstalling tensorflow-2.19.0:
      Successfully uninstalled tensorflow-2.19.0
Successfully installed tensorboard-2.20.0 tensorflow-2.20.0
Requirement already satisfied: cupy-cuda12x in /home/bbc/.venv/lib/python3.10/site-packages (13.5.1)
Requirement already satisfied: numpy<2.6,>=1.22 in /home/bbc/.venv/lib/python3.10/site-packages (from cupy-cuda12x) (2.1.3)
Requirement already satisfied: fastrlock>=0.5 in /home/bbc/.venv/lib/python3.10/site-packages (from cupy-cuda12x) (0.8.3)
ERROR: Could not find a version that satisfies the requirement rapids-cuda12 (from versions: none)
ERROR: No matching distribution found for rapids-cuda12
make[1]: *** [Makefile:138: install-gpu-deps] Error 1
make[1]: Leaving directory '/mnt/e/Work/P-41 Trading/code/t-bot'
make: *** [Makefile:98: setup] Error 2