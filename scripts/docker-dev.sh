#!/bin/bash
# Development Docker startup script for T-Bot Trading System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting T-Bot Trading System in Development Mode${NC}"

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}📁 Creating necessary directories...${NC}"
mkdir -p logs/{application,system,exchange,database,ml}
mkdir -p data/{raw,processed,features,cache}
mkdir -p models/{trained_models,model_registry}
mkdir -p state/{sessions,recovery,temp}
mkdir -p backups/{database,config}
mkdir -p reports/{performance,strategy,risk}

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}📋 Copying development environment file...${NC}"
    cp .env.development .env
    echo -e "${GREEN}✅ Environment file created. You may want to customize it.${NC}"
fi

# Stop any existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose down --remove-orphans

# Build and start services
echo -e "${YELLOW}🏗️  Building Docker images...${NC}"
docker-compose build --parallel

echo -e "${YELLOW}🚢 Starting services...${NC}"
docker-compose up -d

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 10

# Check service health
services=("postgresql" "redis" "influxdb")
for service in "${services[@]}"; do
    echo -e "${YELLOW}🔍 Checking ${service} health...${NC}"
    timeout=30
    while [ $timeout -gt 0 ]; do
        if docker-compose ps | grep -q "${service}.*healthy"; then
            echo -e "${GREEN}✅ ${service} is healthy${NC}"
            break
        fi
        sleep 1
        ((timeout--))
    done
    
    if [ $timeout -eq 0 ]; then
        echo -e "${RED}❌ ${service} failed to become healthy${NC}"
        docker-compose logs "${service}"
        exit 1
    fi
done

# Run database migrations
echo -e "${YELLOW}📊 Running database migrations...${NC}"
docker-compose exec backend alembic upgrade head || {
    echo -e "${RED}❌ Database migration failed${NC}"
    exit 1
}

echo -e "${GREEN}🎉 T-Bot Trading System is now running in development mode!${NC}"
echo ""
echo -e "${BLUE}📋 Service URLs:${NC}"
echo -e "  🌐 Frontend:  http://localhost:3000"
echo -e "  🚀 Backend:   http://localhost:8000"
echo -e "  📚 API Docs:  http://localhost:8000/docs"
echo -e "  🐘 PgAdmin:   http://localhost:5050 (start with: docker-compose --profile tools up -d)"
echo -e "  📈 Redis UI:  http://localhost:8081 (start with: docker-compose --profile tools up -d)"
echo ""
echo -e "${BLUE}🔧 Useful commands:${NC}"
echo -e "  📋 View logs:     docker-compose logs -f [service]"
echo -e "  🛑 Stop all:      docker-compose down"
echo -e "  🧹 Clean up:      docker-compose down -v --remove-orphans"
echo -e "  🔧 Rebuild:       docker-compose build --no-cache"
echo -e "  🔍 Service info:  docker-compose ps"
echo ""
echo -e "${GREEN}Happy trading! 📈${NC}"