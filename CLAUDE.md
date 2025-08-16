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
I see following error when I run make setup command
Building wheels for collected packages: TA-Lib
  Building wheel for TA-Lib (pyproject.toml) ... error
  error: subprocess-exited-with-error

  × Building wheel for TA-Lib (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [37 lines of output]
      <string>:75: UserWarning: Cannot find ta-lib library, installation may fail.
      /tmp/pip-build-env-4enf1akj/overlay/lib/python3.10/site-packages/setuptools/dist.py:759: SetuptoolsDeprecationWarning: License classifiers are deprecated.
      !!

              ********************************************************************************
              Please consider removing the following classifiers in favor of a SPDX license expression:

              License :: OSI Approved :: BSD License

              See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license for details.
              ********************************************************************************

      !!
        self._finalize_license_expression()
      running bdist_wheel
      running build
      running build_py
      creating build/lib.linux-x86_64-cpython-310/talib
      copying talib/abstract.py -> build/lib.linux-x86_64-cpython-310/talib
      copying talib/__init__.py -> build/lib.linux-x86_64-cpython-310/talib
      copying talib/stream.py -> build/lib.linux-x86_64-cpython-310/talib
      copying talib/deprecated.py -> build/lib.linux-x86_64-cpython-310/talib
      running build_ext
      building 'talib._ta_lib' extension
      creating build/temp.linux-x86_64-cpython-310/talib
      x86_64-linux-gnu-gcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -fPIC -I/usr/include -I/usr/local/include -I/opt/include -I/opt/local/include -I/opt/homebrew/include -I/opt/homebrew/opt/ta-lib/include -I/tmp/pip-build-env-4enf1akj/overlay/lib/python3.10/site-packages/numpy/core/include -I/home/bbc/.venv/include -I/usr/include/python3.10 -c talib/_ta_lib.c -o build/temp.linux-x86_64-cpython-310/talib/_ta_lib.o
      In file included from /tmp/pip-build-env-4enf1akj/overlay/lib/python3.10/site-packages/numpy/core/include/numpy/ndarraytypes.h:1929,
                       from /tmp/pip-build-env-4enf1akj/overlay/lib/python3.10/site-packages/numpy/core/include/numpy/ndarrayobject.h:12,
                       from /tmp/pip-build-env-4enf1akj/overlay/lib/python3.10/site-packages/numpy/core/include/numpy/arrayobject.h:5,
                       from talib/_ta_lib.c:1235:
      /tmp/pip-build-env-4enf1akj/overlay/lib/python3.10/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:17:2: warning: #warning "Using deprecated NumPy API, disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
         17 | #warning "Using deprecated NumPy API, disable it with " \
            |  ^~~~~~~
      x86_64-linux-gnu-gcc -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -g -fwrapv -O2 build/temp.linux-x86_64-cpython-310/talib/_ta_lib.o -L/usr/lib -L/usr/local/lib -L/usr/lib64 -L/usr/local/lib64 -L/opt/lib -L/opt/local/lib -L/opt/homebrew/lib -L/opt/homebrew/opt/ta-lib/lib -L/usr/lib/x86_64-linux-gnu -Wl,--enable-new-dtags,-rpath,/usr/lib -Wl,--enable-new-dtags,-rpath,/usr/local/lib -Wl,--enable-new-dtags,-rpath,/usr/lib64 -Wl,--enable-new-dtags,-rpath,/usr/local/lib64 -Wl,--enable-new-dtags,-rpath,/opt/lib -Wl,--enable-new-dtags,-rpath,/opt/local/lib -Wl,--enable-new-dtags,-rpath,/opt/homebrew/lib -Wl,--enable-new-dtags,-rpath,/opt/homebrew/opt/ta-lib/lib -lta_lib -o build/lib.linux-x86_64-cpython-310/talib/_ta_lib.cpython-310-x86_64-linux-gnu.so
      /usr/bin/ld: cannot find -lta_lib: No such file or directory
      collect2: error: ld returned 1 exit status
      error: command '/usr/bin/x86_64-linux-gnu-gcc' failed with exit code 1
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for TA-Lib
Failed to build TA-Lib
error: failed-wheel-build-for-install

× Failed to build installable wheels for some pyproject.toml based projects
╰─> TA-Lib
make[1]: *** [Makefile:128: install-deps] Error 1
make[1]: Leaving directory '/mnt/e/Work/P-41 Trading/code/t-bot'
make: *** [Makefile:94: setup] Error 2