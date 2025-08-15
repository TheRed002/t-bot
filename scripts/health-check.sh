#!/bin/bash
# Health check script for T-Bot Trading System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏥 T-Bot Health Check${NC}"

# Configuration
COMPOSE_FILE="docker-compose.yml"
if [ "$1" = "--prod" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
    echo -e "${BLUE}🏭 Checking production deployment${NC}"
else
    echo -e "${BLUE}🔧 Checking development deployment${NC}"
fi

# Check if Docker Compose is running
if ! docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
    echo -e "${RED}❌ No services are running${NC}"
    exit 1
fi

# Define services to check
if [ "$1" = "--prod" ]; then
    SERVICES=("backend" "frontend" "postgresql" "redis" "influxdb")
else
    SERVICES=("backend" "frontend" "postgresql" "redis" "influxdb")
fi

# Check each service
ALL_HEALTHY=true

for service in "${SERVICES[@]}"; do
    echo -e "${YELLOW}🔍 Checking ${service}...${NC}"
    
    # Check if container is running
    if docker-compose -f "$COMPOSE_FILE" ps | grep "${service}" | grep -q "Up"; then
        echo -e "${GREEN}  ✅ ${service} is running${NC}"
        
        # Service-specific health checks
        case $service in
            "backend")
                if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
                    echo -e "${GREEN}  ✅ Backend health check passed${NC}"
                else
                    echo -e "${RED}  ❌ Backend health check failed${NC}"
                    ALL_HEALTHY=false
                fi
                ;;
            "frontend")
                if curl -f -s http://localhost:3000/health > /dev/null 2>&1; then
                    echo -e "${GREEN}  ✅ Frontend health check passed${NC}"
                else
                    echo -e "${RED}  ❌ Frontend health check failed${NC}"
                    ALL_HEALTHY=false
                fi
                ;;
            "postgresql")
                if docker-compose -f "$COMPOSE_FILE" exec -T postgresql pg_isready -U tbot > /dev/null 2>&1; then
                    echo -e "${GREEN}  ✅ PostgreSQL is ready${NC}"
                else
                    echo -e "${RED}  ❌ PostgreSQL is not ready${NC}"
                    ALL_HEALTHY=false
                fi
                ;;
            "redis")
                if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping | grep -q "PONG"; then
                    echo -e "${GREEN}  ✅ Redis is responding${NC}"
                else
                    echo -e "${RED}  ❌ Redis is not responding${NC}"
                    ALL_HEALTHY=false
                fi
                ;;
            "influxdb")
                if docker-compose -f "$COMPOSE_FILE" exec -T influxdb influx ping > /dev/null 2>&1; then
                    echo -e "${GREEN}  ✅ InfluxDB is responding${NC}"
                else
                    echo -e "${RED}  ❌ InfluxDB is not responding${NC}"
                    ALL_HEALTHY=false
                fi
                ;;
        esac
    else
        echo -e "${RED}  ❌ ${service} is not running${NC}"
        ALL_HEALTHY=false
    fi
done

echo ""
if [ "$ALL_HEALTHY" = true ]; then
    echo -e "${GREEN}🎉 All services are healthy!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some services are unhealthy${NC}"
    echo -e "${YELLOW}💡 Try running: docker-compose -f $COMPOSE_FILE logs [service-name]${NC}"
    exit 1
fi