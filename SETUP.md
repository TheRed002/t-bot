# T-Bot Trading System - Setup Guide

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Activate Python virtual environment (WSL/Linux)
source ~/.venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start external services (PostgreSQL, Redis, InfluxDB)
make services-up

# 4. Run in mock mode - NO API KEYS NEEDED!
make run-mock

# That's it! The system is running in mock trading mode.
```

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Development Setup](#development-setup)
4. [Production Setup](#production-setup)
5. [Configuration](#configuration)
6. [Database Setup](#database-setup)
7. [Exchange Configuration](#exchange-configuration)
8. [Running the System](#running-the-system)
9. [Verification](#verification)
10. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **OS**: Ubuntu 20.04+, macOS 12+, Windows 10+ (WSL2)
- **Python**: 3.10+
- **Node.js**: 18+
- **Docker**: 20.10+

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **Storage**: 100+ GB NVMe SSD
- **Network**: 100 Mbps+ stable connection
- **OS**: Ubuntu 22.04 LTS

## Installation Methods

### Method 1: Hybrid Development Setup (Recommended)
This approach runs external services in Docker while the application runs locally for easier development and debugging.

#### Step 1: Install Docker
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose

# macOS
brew install docker docker-compose

# Windows
# Install Docker Desktop from https://docker.com
```

#### Step 2: Clone Repository
```bash
git clone https://github.com/your-org/t-bot.git
cd t-bot
```

#### Step 3: Configure Environment
```bash
# The .env file is already configured for development
# Review and adjust if needed
nano .env

# Key settings already configured:
# - MOCK_MODE=true (no API keys required)
# - Database: localhost:5432
# - Redis: localhost:6379
# - InfluxDB: localhost:8086
```

#### Step 4: Start External Services Only
```bash
# Start PostgreSQL, Redis, and InfluxDB
make services-up

# Or manually:
docker-compose -f docker-compose.services.yml up -d

# Check status
docker-compose -f docker-compose.services.yml ps
```

#### Step 5: Run Application Locally
```bash
# Run in mock mode (no API keys needed)
make run-mock

# Or run with real connections (requires API keys)
make run

# Start web interface
make web
```

### Method 2: Full Docker Setup (Production)
```bash
# For production deployment with everything in containers
docker-compose -f docker-compose.yml up -d
```

### Method 2: Native Installation (Development)

#### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y \
    python3.10 python3.10-venv python3-pip \
    postgresql-14 postgresql-client-14 \
    redis-server \
    nodejs npm \
    build-essential libssl-dev libffi-dev \
    python3-dev libpq-dev
```

**macOS:**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.10 postgresql@14 redis node
```

**Windows (WSL2):**
```bash
# In WSL2 Ubuntu terminal
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip \
    postgresql redis-server nodejs npm
```

#### Step 2: Python Environment Setup
```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

#### Step 3: Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Build frontend
npm run build

# Return to root
cd ..
```

#### Step 4: Database Setup
```bash
# Start PostgreSQL
sudo systemctl start postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE USER tbot WITH PASSWORD 'your_secure_password';
CREATE DATABASE tbot_db OWNER tbot;
GRANT ALL PRIVILEGES ON DATABASE tbot_db TO tbot;
EOF

# Run migrations
alembic upgrade head

# Start Redis
sudo systemctl start redis-server
```

## Development Setup

### IDE Configuration

#### VS Code
```bash
# Install recommended extensions
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension dbaeumer.vscode-eslint
code --install-extension esbenp.prettier-vscode
```

Create `.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true
}
```

#### PyCharm
1. Open project in PyCharm
2. Configure Python interpreter: `File > Settings > Project > Python Interpreter`
3. Select virtual environment: `venv/bin/python`
4. Enable pytest: `File > Settings > Tools > Python Integrated Tools`

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Production Setup

### System Configuration

#### Ubuntu Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
    nginx \
    certbot python3-certbot-nginx \
    ufw \
    supervisor \
    htop iotop

# Configure firewall
sudo ufw allow 22/tcp     # SSH
sudo ufw allow 80/tcp     # HTTP
sudo ufw allow 443/tcp    # HTTPS
sudo ufw allow 8000/tcp   # API
sudo ufw allow 3000/tcp   # Frontend (development)
sudo ufw enable
```

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/tbot
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/tbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Setup SSL
sudo certbot --nginx -d your-domain.com
```

#### Supervisor Configuration
```ini
# /etc/supervisor/conf.d/tbot.conf
[program:tbot-api]
command=/opt/tbot/venv/bin/python -m uvicorn src.web_interface.app:app --host 0.0.0.0 --port 8000
directory=/opt/tbot
user=tbot
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/tbot/api.log
environment=PATH="/opt/tbot/venv/bin",PYTHONPATH="/opt/tbot"

[program:tbot-bot]
command=/opt/tbot/venv/bin/python -m src.main
directory=/opt/tbot
user=tbot
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/tbot/bot.log
environment=PATH="/opt/tbot/venv/bin",PYTHONPATH="/opt/tbot"
```

```bash
# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start all
```

## Configuration

### Mock Mode (Development Without API Keys)
The system includes a fully functional mock exchange that simulates real trading without requiring API keys:

```bash
# Enable mock mode in .env
MOCK_MODE=true

# Run with mock exchange
make run-mock
```

Features of Mock Mode:
- Simulated order execution with realistic delays
- Mock balance management (starts with 10,000 USDT)
- Real-time price simulation based on market data
- Support for all order types (market, limit, stop-loss)
- Perfect for testing strategies without risk

### Main Configuration File
Create `config/config.yaml`:

```yaml
# System Configuration
system:
  environment: production  # development, staging, production
  debug: false
  log_level: INFO
  timezone: UTC

# Database Configuration
database:
  postgresql:
    host: localhost
    port: 5432
    database: tbot_db
    user: tbot
    password: ${DB_PASSWORD}  # Use environment variable
    pool_size: 20
    max_overflow: 10
  
  redis:
    host: localhost
    port: 6379
    db: 0
    password: ${REDIS_PASSWORD}
    max_connections: 50
  
  influxdb:
    host: localhost
    port: 8086
    database: tbot_metrics
    username: tbot
    password: ${INFLUX_PASSWORD}

# Exchange Configuration
exchanges:
  binance:
    enabled: true
    api_key: ${BINANCE_API_KEY}
    api_secret: ${BINANCE_API_SECRET}
    testnet: false
    rate_limit: 1200  # requests per minute
  
  coinbase:
    enabled: true
    api_key: ${COINBASE_API_KEY}
    api_secret: ${COINBASE_API_SECRET}
    passphrase: ${COINBASE_PASSPHRASE}
    sandbox: false
  
  okx:
    enabled: true
    api_key: ${OKX_API_KEY}
    api_secret: ${OKX_API_SECRET}
    passphrase: ${OKX_PASSPHRASE}
    testnet: false

# Risk Management
risk_management:
  max_position_size: 0.25  # 25% of portfolio
  max_positions: 10
  max_daily_loss: 0.05  # 5% daily loss limit
  use_stop_loss: true
  default_stop_loss: 0.02  # 2%
  use_take_profit: true
  default_take_profit: 0.05  # 5%
  
  circuit_breakers:
    enabled: true
    max_consecutive_losses: 5
    max_drawdown: 0.10  # 10%
    correlation_threshold: 0.80
    cooldown_period: 3600  # seconds

# Trading Configuration
trading:
  base_currency: USDT
  min_order_size: 10.0
  fee_rate: 0.001  # 0.1%
  slippage_tolerance: 0.002  # 0.2%
  
  execution:
    use_limit_orders: true
    order_timeout: 30  # seconds
    max_retries: 3
    smart_routing: true

# ML Configuration
ml:
  models_path: ./models
  feature_engineering:
    lookback_periods: [20, 50, 100, 200]
    technical_indicators: true
    volume_indicators: true
    volatility_indicators: true
  
  training:
    validation_split: 0.2
    test_split: 0.1
    cross_validation_folds: 5
    max_epochs: 100
    early_stopping_patience: 10

# Web Interface
web:
  host: 0.0.0.0
  port: 8000
  cors_origins: ["http://localhost:3000"]
  jwt_secret: ${JWT_SECRET}
  jwt_expiry: 86400  # 24 hours
  
  rate_limiting:
    enabled: true
    default_limit: "100/minute"
    burst_limit: "10/second"

# Monitoring
monitoring:
  metrics:
    enabled: true
    export_interval: 60  # seconds
    retention_days: 30
  
  logging:
    level: INFO
    format: json
    file: /var/log/tbot/app.log
    max_size: 100  # MB
    backup_count: 10
  
  alerting:
    enabled: true
    email:
      smtp_host: smtp.gmail.com
      smtp_port: 587
      from_email: alerts@your-domain.com
      to_emails: [admin@your-domain.com]
    
    slack:
      webhook_url: ${SLACK_WEBHOOK}
      channel: "#trading-alerts"
```

### Environment Variables
Create `.env` file:

```bash
# Database
DB_PASSWORD=your_secure_db_password
REDIS_PASSWORD=your_redis_password
INFLUX_PASSWORD=your_influx_password

# Exchange API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

COINBASE_API_KEY=your_coinbase_api_key
COINBASE_API_SECRET=your_coinbase_api_secret
COINBASE_PASSPHRASE=your_coinbase_passphrase

OKX_API_KEY=your_okx_api_key
OKX_API_SECRET=your_okx_api_secret
OKX_PASSPHRASE=your_okx_passphrase

# Security
JWT_SECRET=your_jwt_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here

# Monitoring
SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Application
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO
```

## Database Setup

### PostgreSQL Schema
```sql
-- Create tables
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bots (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    name VARCHAR(100) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'stopped',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    bot_id INTEGER REFERENCES bots(id),
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    fee DECIMAL(20, 8),
    pnl DECIMAL(20, 8),
    status VARCHAR(20) NOT NULL,
    order_id VARCHAR(100),
    executed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_trades_bot_id ON trades(bot_id);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_executed_at ON trades(executed_at);
CREATE INDEX idx_bots_user_id ON bots(user_id);
CREATE INDEX idx_bots_status ON bots(status);
```

### Run Migrations
```bash
# Initialize Alembic
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

### Redis Configuration
```bash
# Set Redis password
redis-cli CONFIG SET requirepass "your_redis_password"

# Save configuration
redis-cli CONFIG REWRITE

# Test connection
redis-cli -a your_redis_password ping
```

## Exchange Configuration

### Binance
1. Log in to [Binance](https://www.binance.com)
2. Go to API Management
3. Create new API key with trading permissions
4. Enable spot and futures trading
5. Add IP whitelist for security
6. Save API key and secret

### Coinbase
1. Log in to [Coinbase Pro](https://pro.coinbase.com)
2. Go to API settings
3. Create new API key
4. Select permissions: View, Trade
5. Add IP whitelist
6. Save API key, secret, and passphrase

### OKX
1. Log in to [OKX](https://www.okx.com)
2. Go to API settings
3. Create new API key
4. Set permissions: Read, Trade
5. Add IP whitelist
6. Save API key, secret, and passphrase

## Running the System

### Quick Start (Development with Mock Mode)

```bash
# 1. Start external services (PostgreSQL, Redis, InfluxDB)
make services-up

# 2. Run in mock mode (no API keys required!)
make run-mock

# 3. Optional: Start web interface
make web
```

### Development Mode Options

```bash
# Run with mock exchange (no API keys needed)
make run-mock

# Run with real exchanges (requires API keys in .env)
make run

# Start only external services
make services-up

# Stop external services
make services-down

# View service logs
make services-logs

# Clean up services and volumes
make services-clean
```

### Testing Commands

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration

# Run with coverage report
make coverage

# Format and lint code
make format
make lint
```

### Production Mode

```bash
# Using Docker Compose (full stack)
docker-compose -f docker-compose.yml up -d

# Using Supervisor (for production servers)
sudo supervisorctl start all

# Check status
sudo supervisorctl status
docker-compose ps
```

### WSL-Specific Commands (Windows)

```bash
# Run tests in WSL
make wsl-test

# Run unit tests in WSL
make wsl-test-unit

# Run coverage in WSL
make wsl-coverage
```

## Verification

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check database connection
python -c "from src.database.connection import get_db; print(get_db())"

# Check Redis connection
redis-cli -a your_redis_password ping

# Check exchange connections
python scripts/check_exchanges.py
```

### Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# With coverage
pytest --cov=src --cov-report=html

# Frontend tests
cd frontend && npm test
```

### Monitor Logs

```bash
# Docker logs
docker-compose logs -f

# System logs
tail -f /var/log/tbot/app.log

# Supervisor logs
tail -f /var/log/supervisor/tbot-*.log
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Error
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
psql -h localhost -U tbot -d tbot_db

# Reset password if needed
sudo -u postgres psql
ALTER USER tbot WITH PASSWORD 'new_password';
```

#### 2. Redis Connection Error
```bash
# Check Redis status
sudo systemctl status redis-server

# Test connection
redis-cli -a your_redis_password ping

# Check Redis logs
tail -f /var/log/redis/redis-server.log
```

#### 3. Exchange API Errors
```bash
# Test exchange connection
python scripts/test_exchange.py --exchange binance

# Check API permissions
python scripts/check_api_permissions.py
```

#### 4. Frontend Build Issues
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
npm run build
```

#### 5. Port Already in Use
```bash
# Find process using port
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>
```

### Performance Tuning

#### PostgreSQL
```sql
-- Optimize for trading workload
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;

-- Reload configuration
SELECT pg_reload_conf();
```

#### Redis
```bash
# Edit /etc/redis/redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
tcp-keepalive 60
timeout 300

# Restart Redis
sudo systemctl restart redis-server
```

### Backup and Recovery

#### Database Backup
```bash
# Manual backup
pg_dump -U tbot -h localhost tbot_db > backup_$(date +%Y%m%d).sql

# Automated backup (cron)
0 2 * * * pg_dump -U tbot -h localhost tbot_db > /backups/tbot_$(date +\%Y\%m\%d).sql
```

#### Restore from Backup
```bash
# Restore database
psql -U tbot -h localhost tbot_db < backup_20240101.sql

# Restore Redis
redis-cli -a your_redis_password --rdb /path/to/dump.rdb
```

## Security Checklist

- [ ] Change all default passwords
- [ ] Enable firewall with minimal open ports
- [ ] Configure SSL/TLS for web interface
- [ ] Set up API key rotation schedule
- [ ] Enable 2FA for exchange accounts
- [ ] Implement IP whitelisting
- [ ] Regular security updates
- [ ] Audit logs enabled
- [ ] Backup encryption enabled
- [ ] Rate limiting configured

## Next Steps

1. **Configure Strategies**: Edit `config/strategies/` files
2. **Set Risk Parameters**: Adjust risk management settings
3. **Create First Bot**: Use web interface or API
4. **Run Backtests**: Test strategies on historical data
5. **Start Paper Trading**: Test with live data, no real money
6. **Monitor Performance**: Check dashboards and logs
7. **Go Live**: Enable production trading with small amounts

## Support

For issues or questions:
- Check [Documentation](docs/)
- Review [FAQ](docs/FAQ.md)
- Create GitHub Issue
- Contact support team

---

**Remember**: Always test thoroughly in paper trading mode before using real funds!