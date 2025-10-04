# T-Bot - Cryptocurrency Trading Bot

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Status: Beta](https://img.shields.io/badge/status-beta-yellow.svg)

## ⚠️ Disclaimer

**READ THIS BEFORE USING**

- 🚫 **NOT FINANCIAL ADVICE** - This is for educational and research purposes only
- ⚠️ **USE AT YOUR OWN RISK** - Cryptocurrency trading carries substantial risk
- 📜 **NO WARRANTY** - Provided "AS IS" without warranty of any kind
- 🏛️ **REGULATORY COMPLIANCE** - Ensure compliance with your local laws
- 📝 **TEST FIRST** - Always use testnet/sandbox mode before live trading

**The developers are not responsible for any financial losses.**

---

## Overview

T-Bot is a professional cryptocurrency trading bot supporting automated trading across multiple exchanges. Built with Python 3.10+ and React, it features institutional-grade risk management, machine learning integration, and real-time monitoring.

## Features

### Exchanges
- **Binance** - Spot and futures trading via REST API and WebSocket
- **Coinbase** - Advanced trading interface with full API support
- **OKX** - Complete exchange integration with WebSocket streams
- **Mock Mode** - Test without API keys or real funds

### Trading Strategies
**Static**: Mean reversion, trend following, breakout, market making, cross-exchange arbitrage, triangular arbitrage
**Dynamic**: Adaptive momentum, volatility breakout
**Hybrid**: Rule-based AI, fallback strategies, ensemble methods

### Risk Management
- Position sizing with Kelly Criterion
- Stop-loss and take-profit automation
- Portfolio exposure limits
- Circuit breakers (correlation and threshold-based)
- Order idempotency (duplicate prevention)
- Financial-grade decimal precision

### Machine Learning
- Feature engineering (100+ indicators via TA-Lib)
- Model training (TensorFlow, scikit-learn, XGBoost, LightGBM)
- Hyperparameter optimization (Optuna)
- Model validation and drift detection

### Analytics
- Backtesting engine with walk-forward analysis
- Parameter optimization (brute force and Bayesian)
- Real-time performance monitoring
- Portfolio analytics and reporting

### Web Interface
- FastAPI backend with JWT authentication
- React frontend with Material-UI
- Real-time WebSocket updates
- Interactive dashboards

## Technology Stack

**Backend**: Python 3.10+, FastAPI, SQLAlchemy, asyncio
**Database**: PostgreSQL 15, Redis 7, InfluxDB 2.7
**ML**: TensorFlow, scikit-learn, XGBoost, LightGBM, TA-Lib
**Frontend**: React 18, TypeScript, Webpack, Material-UI
**Infrastructure**: Docker, Docker Compose
**Testing**: pytest, pytest-asyncio
**Code Quality**: ruff, black, mypy

## Installation

### Prerequisites
- Python 3.10.12 or higher
- Docker and Docker Compose
- Node.js 18+ (for frontend)
- Git

### Quick Setup

```bash
# Clone repository
git clone https://github.com/TheRed002/t-bot.git
cd t-bot

# Complete setup (Python venv, dependencies, Docker services)
make setup

# Or step-by-step:
make setup-venv      # Create Python virtual environment
make install-deps    # Install Python dependencies
make setup-external  # Install TA-Lib and external libraries
make services-up     # Start PostgreSQL, Redis, InfluxDB
make migrate         # Run database migrations
```

### Configuration

1. Copy environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your settings:
   - Database credentials
   - Exchange API keys (or leave blank for mock mode)
   - JWT secret key
   - Other configuration

3. Configure strategies in `config/strategies/`

## Running

### Full Application
```bash
make run-all        # Backend + Web API + Frontend
```

### Individual Components
```bash
make run            # Trading bot backend only
make run-mock       # Mock mode (no API keys needed)
make run-web        # Web API only (port 8000)
make run-frontend   # React frontend only (port 3000)
```

### Services
```bash
make services-up    # Start PostgreSQL, Redis, InfluxDB
make services-down  # Stop services
make services-logs  # View service logs
make status         # Check status of all components
```

### Stop All
```bash
make kill          # Stop all T-Bot processes
```

## Testing

```bash
make test          # Run all tests
make test-unit     # Unit tests only
make test-integration  # Integration tests only
make test-mock     # Tests in mock mode
make coverage      # Generate coverage report
```

## Development

```bash
make lint          # Check code quality
make format        # Auto-format code (ruff + black)
make typecheck     # Run mypy type checking
make check-all     # Lint + typecheck + test
make clean         # Remove temporary files
```

## Security Best Practices

1. **Never commit secrets** - Use `.env` file (already gitignored)
2. **Start with testnet** - Use `BINANCE_TESTNET=true`, `OKX_SANDBOX=true`
3. **Use mock mode** - Test strategies with `make run-mock`
4. **Limit API permissions** - Read-only for testing, restrict withdrawal
5. **Monitor actively** - Watch logs and set up alerts
6. **Regular backups** - Database and configuration backups

## Project Structure

```
t-bot/
├── src/
│   ├── analytics/          # Performance analytics
│   ├── backtesting/        # Backtesting engine
│   ├── bot_management/     # Bot lifecycle
│   ├── capital_management/ # Portfolio allocation
│   ├── core/              # Base classes, config, DI
│   ├── data/              # Market data pipeline
│   ├── database/          # Models, repositories
│   ├── error_handling/    # Error recovery
│   ├── exchanges/         # Exchange integrations
│   ├── execution/         # Order execution
│   ├── ml/                # Machine learning
│   ├── monitoring/        # Telemetry
│   ├── optimization/      # Parameter optimization
│   ├── risk_management/   # Risk controls
│   ├── state/             # State management
│   ├── strategies/        # Trading strategies
│   ├── utils/             # Shared utilities
│   └── web_interface/     # Web API + frontend
├── tests/                 # Unit and integration tests
├── config/                # Strategy configs
├── docker/                # Docker configuration
├── frontend/              # React frontend
├── scripts/               # Setup and deployment scripts
├── Makefile              # Build automation
└── README.md             # This file
```

## Documentation

- API Documentation: http://localhost:8000/docs (when running)
- Code Standards: `CODING_STANDARDS.md`
- Common Patterns: `COMMON_PATTERNS.md`
- Claude Instructions: `CLAUDE.md`

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Follow coding standards (`make format` and `make lint`)
4. Add tests for new features
5. Submit a pull request

## License

MIT License - see [LICENSE.txt.md](LICENSE.txt.md)

Copyright (c) 2025 Muhammad Bilal Farooq

## Acknowledgments

- Exchange APIs: Binance, Coinbase, OKX
- Technical Analysis: TA-Lib
- ML Frameworks: TensorFlow, scikit-learn, XGBoost
- Web Framework: FastAPI, React
- Database: PostgreSQL, Redis, InfluxDB

---

**Version**: 0.9.0 (Beta)
**Author**: Muhammad Bilal Farooq
**Repository**: https://github.com/TheRed002/t-bot
