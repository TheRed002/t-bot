# T-Bot Trading System - Docker Deployment Guide

This guide covers the complete Docker containerization setup for the T-Bot Trading System, including both development and production environments.

## 🏗️ Architecture Overview

The T-Bot system uses a microservices architecture with the following components:

- **Backend**: FastAPI application with Python 3.10
- **Frontend**: React application served with Nginx
- **PostgreSQL**: Primary database for persistent storage
- **Redis**: Caching and session storage
- **InfluxDB**: Time-series data for metrics and market data

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 8GB RAM
- 20GB available disk space

## 🚀 Quick Start

### Development Environment

1. **Clone the repository and navigate to the project directory**:
   ```bash
   git clone <repository-url>
   cd t-bot
   ```

2. **Run the development setup script**:
   ```bash
   ./scripts/docker-dev.sh
   ```

3. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Production Environment

1. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your production values
   ```

2. **Set up secrets**:
   ```bash
   # Create secret files in the secrets/ directory
   echo "your_secure_password" > secrets/db_password.txt
   echo "your_jwt_secret_key" > secrets/jwt_secret.txt
   # Add all other required secrets
   chmod 600 secrets/*.txt
   ```

3. **Run the production setup script**:
   ```bash
   sudo ./scripts/docker-prod.sh
   ```

## 📁 Directory Structure

```
t-bot/
├── Dockerfile                     # Multi-stage backend container
├── docker-compose.yml            # Development environment
├── docker-compose.prod.yml       # Production environment
├── .dockerignore                 # Backend Docker ignore
├── .env.example                  # Environment template
├── .env.development             # Development config
├── frontend/
│   ├── Dockerfile               # Frontend container
│   ├── nginx.conf              # Nginx configuration
│   └── .dockerignore           # Frontend Docker ignore
├── scripts/
│   ├── docker-dev.sh           # Development startup
│   ├── docker-prod.sh          # Production startup
│   └── backup-db.sh            # Database backup
└── secrets/                     # Production secrets
    ├── .gitignore
    └── README.example
```

## 🔧 Configuration

### Environment Variables

Key environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `false` |
| `DATABASE_URL` | PostgreSQL connection string | - |
| `REDIS_URL` | Redis connection string | - |
| `INFLUXDB_URL` | InfluxDB connection string | - |
| `JWT_SECRET_KEY` | JWT signing key | - |
| `CORS_ORIGINS` | Allowed CORS origins | `http://localhost:3000` |

### Exchange API Configuration

Configure your exchange API credentials:

```bash
# Binance
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
BINANCE_TESTNET=false

# Coinbase
COINBASE_API_KEY=your_api_key
COINBASE_SECRET_KEY=your_secret_key
COINBASE_SANDBOX=false

# OKX
OKX_API_KEY=your_api_key
OKX_SECRET_KEY=your_secret_key
OKX_PASSPHRASE=your_passphrase
OKX_SANDBOX=false
```

## 🏗️ Multi-Stage Docker Build

The backend Dockerfile uses multi-stage builds for optimization:

- **Base Stage**: Common dependencies and TA-Lib installation
- **Builder Stage**: Python dependency installation in virtual environment
- **Development Stage**: Development with hot-reload
- **Production Stage**: Optimized production build

Build specific stages:
```bash
# Development
docker build --target development -t tbot-backend:dev .

# Production
docker build --target production -t tbot-backend:prod .
```

## 🐳 Docker Compose Services

### Development Services

- `backend`: FastAPI with hot-reload
- `frontend`: React with development server
- `postgresql`: PostgreSQL 15 with development data
- `redis`: Redis 7 with development settings
- `influxdb`: InfluxDB 2.7 for metrics

Optional development tools:
```bash
# Start with database management tools
docker-compose --profile tools up -d
```

### Production Services

All development services plus:
- Resource limits and reservations
- Health checks with appropriate timeouts
- Restart policies
- Logging configuration
- Secrets management

Optional production services:
```bash
# Start with monitoring
docker-compose -f docker-compose.prod.yml --profile monitoring up -d

# Start with Nginx reverse proxy
docker-compose -f docker-compose.prod.yml --profile nginx up -d
```

## 🔒 Security Best Practices

### Container Security

1. **Non-root user**: All services run as non-root users
2. **Minimal base images**: Using Alpine images where possible
3. **Multi-stage builds**: Reduced attack surface
4. **Resource limits**: Prevents resource exhaustion
5. **Health checks**: Ensures service reliability

### Secrets Management

Production secrets are managed via Docker secrets:

```bash
# Create secret files
echo "secret_value" | docker secret create db_password -
```

### Network Security

- Isolated Docker networks
- Nginx reverse proxy with security headers
- Rate limiting on API endpoints
- CORS configuration

## 📊 Monitoring and Logging

### Logging

Structured logging with JSON format:
- Application logs: `/app/logs/application/`
- System logs: JSON driver with rotation
- Log retention: 30 days with compression

### Health Checks

All services include health checks:
- Backend: HTTP health endpoint
- Frontend: Nginx status
- Databases: Native health checks

### Monitoring Stack (Optional)

Enable with `--profile monitoring`:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **InfluxDB**: Time-series metrics storage

## 🛠️ Useful Commands

### Development Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Execute commands in container
docker-compose exec backend bash

# Rebuild specific service
docker-compose build --no-cache backend

# Stop all services
docker-compose down
```

### Production Commands

```bash
# Start production stack
docker-compose -f docker-compose.prod.yml up -d

# Scale backend service
docker-compose -f docker-compose.prod.yml up -d --scale backend=3

# Update service
docker-compose -f docker-compose.prod.yml pull backend
docker-compose -f docker-compose.prod.yml up -d backend

# Backup database
./scripts/backup-db.sh --prod
```

### Maintenance Commands

```bash
# Database migration
docker-compose exec backend alembic upgrade head

# Clean up unused resources
docker system prune -f

# View resource usage
docker stats

# Container shell access
docker-compose exec backend sh
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy T-Bot
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to production
        run: |
          docker-compose -f docker-compose.prod.yml pull
          docker-compose -f docker-compose.prod.yml up -d
```

### Docker Registry

Build and push images:
```bash
# Build and tag
docker build -t your-registry/tbot-backend:latest .
docker build -t your-registry/tbot-frontend:latest ./frontend

# Push to registry
docker push your-registry/tbot-backend:latest
docker push your-registry/tbot-frontend:latest
```

## 🚨 Troubleshooting

### Common Issues

1. **Services not starting**:
   ```bash
   # Check service logs
   docker-compose logs service_name
   
   # Check service status
   docker-compose ps
   ```

2. **Database connection issues**:
   ```bash
   # Test database connectivity
   docker-compose exec backend python -c "
   import asyncpg
   import asyncio
   asyncio.run(asyncpg.connect('postgresql://tbot:password@postgresql:5432/tbot_dev'))
   "
   ```

3. **Permission issues**:
   ```bash
   # Fix file permissions
   sudo chown -R $(id -u):$(id -g) logs/ data/ models/
   ```

4. **Port conflicts**:
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Stop conflicting services
   sudo systemctl stop service_name
   ```

### Performance Tuning

1. **Resource allocation**:
   - Increase container memory limits
   - Adjust worker processes
   - Tune database connections

2. **Database optimization**:
   ```bash
   # PostgreSQL tuning
   docker-compose exec postgresql psql -U tbot -c "
   ALTER SYSTEM SET shared_buffers = '256MB';
   ALTER SYSTEM SET effective_cache_size = '1GB';
   SELECT pg_reload_conf();
   "
   ```

## 📞 Support

For issues and questions:
1. Check the logs: `docker-compose logs -f`
2. Verify service health: `docker-compose ps`
3. Review this documentation
4. Check GitHub issues

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.