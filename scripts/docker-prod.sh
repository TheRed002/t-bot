#!/bin/bash
# Production Docker startup script for T-Bot Trading System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting T-Bot Trading System in Production Mode${NC}"

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check if running as root for production deployment
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Not running as root. Some production features may not work properly.${NC}"
    echo -e "${YELLOW}   Consider running with sudo for production deployment.${NC}"
fi

# Verify production environment file exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ Production environment file (.env) not found.${NC}"
    echo -e "${YELLOW}📋 Please copy .env.example to .env and configure it for production.${NC}"
    exit 1
fi

# Verify secrets directory exists and is properly configured
if [ ! -d secrets ]; then
    echo -e "${RED}❌ Secrets directory not found.${NC}"
    echo -e "${YELLOW}📋 Please create the secrets directory and configure production secrets.${NC}"
    exit 1
fi

# Check for required secret files
required_secrets=(
    "db_password.txt"
    "jwt_secret.txt"
    "binance_api_key.txt"
    "binance_secret_key.txt"
    "coinbase_api_key.txt"
    "coinbase_secret_key.txt"
    "okx_api_key.txt"
    "okx_secret_key.txt"
    "okx_passphrase.txt"
)

missing_secrets=()
for secret in "${required_secrets[@]}"; do
    if [ ! -f "secrets/${secret}" ]; then
        missing_secrets+=("${secret}")
    fi
done

if [ ${#missing_secrets[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing required secret files:${NC}"
    for secret in "${missing_secrets[@]}"; do
        echo -e "  - secrets/${secret}"
    done
    echo -e "${YELLOW}📋 Please create these files in the secrets/ directory.${NC}"
    exit 1
fi

# Verify secret file permissions
echo -e "${YELLOW}🔐 Checking secret file permissions...${NC}"
for secret in "${required_secrets[@]}"; do
    if [ -f "secrets/${secret}" ]; then
        perms=$(stat -c "%a" "secrets/${secret}")
        if [ "$perms" != "600" ]; then
            echo -e "${YELLOW}⚠️  Fixing permissions for secrets/${secret}${NC}"
            chmod 600 "secrets/${secret}"
        fi
    fi
done

# Create necessary directories with proper permissions
echo -e "${YELLOW}📁 Creating necessary directories...${NC}"
mkdir -p logs/{application,system,exchange,database,ml}
mkdir -p data/{raw,processed,features,cache}
mkdir -p models/{trained_models,model_registry}
mkdir -p state/{sessions,recovery,temp}
mkdir -p backups/{database,config}
mkdir -p reports/{performance,strategy,risk}

# Set proper ownership if running as root
if [ "$EUID" -eq 0 ]; then
    chown -R 1001:1001 logs/ data/ models/ state/ backups/ reports/
fi

# Stop any existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose -f docker-compose.prod.yml down --remove-orphans

# Pull latest images
echo -e "${YELLOW}📥 Pulling latest images...${NC}"
docker-compose -f docker-compose.prod.yml pull

# Build and start services
echo -e "${YELLOW}🏗️  Building Docker images...${NC}"
docker-compose -f docker-compose.prod.yml build --parallel --no-cache

echo -e "${YELLOW}🚢 Starting services...${NC}"
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 30

# Check service health
services=("postgresql" "redis" "influxdb")
for service in "${services[@]}"; do
    echo -e "${YELLOW}🔍 Checking ${service} health...${NC}"
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose -f docker-compose.prod.yml ps | grep -q "${service}.*healthy"; then
            echo -e "${GREEN}✅ ${service} is healthy${NC}"
            break
        fi
        sleep 2
        ((timeout--))
    done
    
    if [ $timeout -eq 0 ]; then
        echo -e "${RED}❌ ${service} failed to become healthy${NC}"
        docker-compose -f docker-compose.prod.yml logs "${service}"
        exit 1
    fi
done

# Run database migrations
echo -e "${YELLOW}📊 Running database migrations...${NC}"
docker-compose -f docker-compose.prod.yml exec backend alembic upgrade head || {
    echo -e "${RED}❌ Database migration failed${NC}"
    exit 1
}

# Verify all services are running
echo -e "${YELLOW}🔍 Verifying all services...${NC}"
services_status=$(docker-compose -f docker-compose.prod.yml ps --services --filter "status=running")
expected_services=("backend" "frontend" "postgresql" "redis" "influxdb")

for service in "${expected_services[@]}"; do
    if echo "$services_status" | grep -q "^${service}$"; then
        echo -e "${GREEN}✅ ${service} is running${NC}"
    else
        echo -e "${RED}❌ ${service} is not running${NC}"
        exit 1
    fi
done

# Setup log rotation (if running as root)
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}📝 Setting up log rotation...${NC}"
    cat > /etc/logrotate.d/tbot << 'EOF'
/mnt/e/Work/P-41 Trading/code/t-bot/logs/*/*.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
    create 644 1001 1001
}
EOF
fi

echo -e "${GREEN}🎉 T-Bot Trading System is now running in production mode!${NC}"
echo ""
echo -e "${BLUE}📋 Service URLs:${NC}"
echo -e "  🌐 Frontend:     http://localhost:3000"
echo -e "  🚀 Backend API:  http://localhost:8000"
echo -e "  📚 API Docs:     http://localhost:8000/docs"
echo -e "  📊 Monitoring:   http://localhost:3001 (Grafana - if enabled)"
echo ""
echo -e "${BLUE}🔧 Management Commands:${NC}"
echo -e "  📋 View logs:        docker-compose -f docker-compose.prod.yml logs -f [service]"
echo -e "  🛑 Stop all:         docker-compose -f docker-compose.prod.yml down"
echo -e "  🔄 Restart service:  docker-compose -f docker-compose.prod.yml restart [service]"
echo -e "  📊 Service status:   docker-compose -f docker-compose.prod.yml ps"
echo -e "  💾 Database backup:  ./scripts/backup-db.sh"
echo ""
echo -e "${YELLOW}📊 Monitor your system:${NC}"
echo -e "  - Check service health regularly"
echo -e "  - Monitor logs for errors"
echo -e "  - Keep an eye on resource usage"
echo -e "  - Backup your data regularly"
echo ""
echo -e "${GREEN}Production deployment complete! 📈${NC}"