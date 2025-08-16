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
everything seems all mixed up when I run make setup command. Why following error? 
"make[1]: Entering directory '/mnt/e/Work/P-41 Trading/code/t-bot'"

(.venv) bbc@DESKTOP-G4N9FG8:/mnt/e/Work/P-41 Trading/code/t-bot$ make setup
🔧 Complete T-Bot Setup...
📋 Running pre-installation checks...
[INFO] Running pre-installation checks...
[INFO] Running on WSL
[INFO] Checking Python installation...
[SUCCESS] Python 3.10.12 found
[SUCCESS] Python 3.10 detected (recommended)
[SUCCESS] Virtual environment active: /home/bbc/.venv
[INFO] Checking system dependencies...
[SUCCESS] All build tools are installed
[INFO] Setting up TA-Lib directories...
[WARNING] /usr/local is not writable, will need sudo for installation
[SUCCESS] Directories prepared
[SUCCESS] Pre-installation checks completed!
[INFO] You can now proceed with: make setup
make[1]: Entering directory '/mnt/e/Work/P-41 Trading/code/t-bot'
🐍 Setting up Python virtual environment...
✅ Virtual environment created at ~/.venv
Requirement already satisfied: pip in /home/bbc/.venv/lib/python3.10/site-packages (25.2)
Requirement already satisfied: setuptools in /home/bbc/.venv/lib/python3.10/site-packages (80.9.0)
Requirement already satisfied: wheel in /home/bbc/.venv/lib/python3.10/site-packages (0.45.1)
✅ Virtual environment ready!
make[1]: Leaving directory '/mnt/e/Work/P-41 Trading/code/t-bot'
make[1]: Entering directory '/mnt/e/Work/P-41 Trading/code/t-bot'
📦 Installing external libraries...
[INFO] Setting up environment for all external libraries...
[SUCCESS] Master environment setup completed
[INFO] Installing all external libraries...
[INFO] Installing talib...
[SUCCESS] talib is already properly installed - skipping
[INFO] Installing cuda...
[SUCCESS] cuda is already properly installed - skipping
[INFO] Installing cudnn...
[SUCCESS] cudnn is already properly installed - skipping
[INFO] Installing lightgbm...
[SUCCESS] lightgbm is already properly installed - skipping
[SUCCESS] All external libraries are ready (4 already installed, 4 total)
✅ External libraries installed!
make[1]: Leaving directory '/mnt/e/Work/P-41 Trading/code/t-bot'
make[1]: Entering directory '/mnt/e/Work/P-41 Trading/code/t-bot'
📦 Installing Python dependencies...
[INFO] Starting T-Bot requirements installation...
[SUCCESS] Virtual environment is active: /home/bbc/.venv
[INFO] Checking TA-Lib C library...
[WARNING] TA-Lib C library not found, installing...
[INFO] Installing TA-Lib 0.6.4...
[INFO] Setting up environment for TA-Lib...
[SUCCESS] TA-Lib environment setup completed
[INFO] Checking TA-Lib installation...
[SUCCESS] TA-Lib is installed and working
[SUCCESS] TA-Lib is already installed and working
[SUCCESS] TA-Lib C library installed successfully
[WARNING] TA-Lib C library installed but not in ldconfig cache
ℹ You may need to run: sudo ldconfig
[INFO] Installing Python requirements...
[INFO] Upgrading pip, setuptools, and wheel...
Requirement already satisfied: pip in /home/bbc/.venv/lib/python3.10/site-packages (25.2)
Requirement already satisfied: setuptools in /home/bbc/.venv/lib/python3.10/site-packages (80.9.0)
Requirement already satisfied: wheel in /home/bbc/.venv/lib/python3.10/site-packages (0.45.1)
[INFO] Installing numpy (required for TA-Lib)...
Requirement already satisfied: numpy>=1.26.4 in /home/bbc/.venv/lib/python3.10/site-packages (2.1.3)
[INFO] Installing TA-Lib Python package...
Requirement already satisfied: TA-Lib in /home/bbc/.venv/lib/python3.10/site-packages (0.6.5)
Requirement already satisfied: build in /home/bbc/.venv/lib/python3.10/site-packages (from TA-Lib) (1.3.0)
Requirement already satisfied: numpy in /home/bbc/.venv/lib/python3.10/site-packages (from TA-Lib) (2.1.3)
Requirement already satisfied: pip in /home/bbc/.venv/lib/python3.10/site-packages (from TA-Lib) (25.2)
Requirement already satisfied: packaging>=19.1 in /home/bbc/.venv/lib/python3.10/site-packages (from build->TA-Lib) (25.0)
Requirement already satisfied: pyproject_hooks in /home/bbc/.venv/lib/python3.10/site-packages (from build->TA-Lib) (1.2.0)
Requirement already satisfied: tomli>=1.1.0 in /home/bbc/.venv/lib/python3.10/site-packages (from build->TA-Lib) (2.2.1)
[SUCCESS] TA-Lib Python package installed successfully
[INFO] Installing remaining requirements...
ERROR: Could not open requirements file: [Errno 2] No such file or directory: '../../requirements.txt'
[SUCCESS] All Python requirements installed
[INFO] Verifying installation...
TA-Lib version: 0.6.5
[SUCCESS] TA-Lib Python package is working