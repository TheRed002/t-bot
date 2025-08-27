#!/bin/bash

# T-Bot Trading System Deployment Script
# This script handles deployment to different environments with health checks and rollback capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-staging}"
VERSION="${2:-latest}"
REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_PREFIX="${IMAGE_PREFIX:-tbot}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Utility functions
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if required tools are installed
    local required_tools=("docker" "docker-compose" "curl" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check if environment configuration exists
    local env_file="${PROJECT_ROOT}/.env.${ENVIRONMENT}"
    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file '$env_file' not found"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

load_environment() {
    log_info "Loading environment configuration for: $ENVIRONMENT"
    
    local env_file="${PROJECT_ROOT}/.env.${ENVIRONMENT}"
    set -a  # automatically export all variables
    source "$env_file"
    set +a
    
    # Validate required environment variables
    local required_vars=("DATABASE_URL" "REDIS_URL" "SECRET_KEY" "JWT_SECRET")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable '$var' is not set"
            exit 1
        fi
    done
    
    log_success "Environment configuration loaded"
}

backup_database() {
    if [[ "$BACKUP_ENABLED" != "true" ]]; then
        log_info "Database backup disabled"
        return 0
    fi
    
    log_info "Creating database backup..."
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="${PROJECT_ROOT}/backups/db_backup_${ENVIRONMENT}_${timestamp}.sql"
    
    # Create backups directory if it doesn't exist
    mkdir -p "${PROJECT_ROOT}/backups"
    
    # Extract database connection details
    local db_url="${DATABASE_URL}"
    
    # Create backup using pg_dump
    if docker run --rm --network host \
        -e PGPASSWORD="${DB_PASSWORD:-}" \
        postgres:15-alpine \
        pg_dump -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" \
        -U "${DB_USER:-tbot_user}" -d "${DB_NAME:-tbot}" \
        > "$backup_file"; then
        log_success "Database backup created: $backup_file"
        
        # Keep only last 5 backups
        find "${PROJECT_ROOT}/backups" -name "db_backup_${ENVIRONMENT}_*.sql" -type f | \
            sort -r | tail -n +6 | xargs -r rm
    else
        log_error "Database backup failed"
        if [[ "$ENVIRONMENT" == "production" ]]; then
            exit 1
        fi
    fi
}

pull_images() {
    log_info "Pulling container images..."
    
    local backend_image="${REGISTRY}/${IMAGE_PREFIX}-backend:${VERSION}"
    local frontend_image="${REGISTRY}/${IMAGE_PREFIX}-frontend:${VERSION}"
    
    if docker pull "$backend_image" && docker pull "$frontend_image"; then
        log_success "Container images pulled successfully"
    else
        log_error "Failed to pull container images"
        exit 1
    fi
}

generate_compose_file() {
    log_info "Generating docker-compose configuration..."
    
    local compose_file="${PROJECT_ROOT}/docker-compose.${ENVIRONMENT}.yml"
    local backend_image="${REGISTRY}/${IMAGE_PREFIX}-backend:${VERSION}"
    local frontend_image="${REGISTRY}/${IMAGE_PREFIX}-frontend:${VERSION}"
    
    cat > "$compose_file" << EOF
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: \${DB_NAME:-tbot}
      POSTGRES_USER: \${DB_USER:-tbot_user}
      POSTGRES_PASSWORD: \${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/configs/postgresql.conf:/etc/postgresql/postgresql.conf
      - ./docker/configs/pg_hba.conf:/etc/postgresql/pg_hba.conf
    ports:
      - "\${DB_PORT:-5432}:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U \${DB_USER:-tbot_user} -d \${DB_NAME:-tbot}"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  redis:
    image: redis:7-alpine
    command: redis-server /usr/local/etc/redis/redis.conf
    volumes:
      - redis_data:/data
      - ./docker/configs/redis.conf:/usr/local/etc/redis/redis.conf
    ports:
      - "\${REDIS_PORT:-6379}:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

  backend:
    image: ${backend_image}
    environment:
      - ENV=${ENVIRONMENT}
      - SECRET_KEY=\${SECRET_KEY}
      - JWT_SECRET=\${JWT_SECRET}
      - DATABASE_URL=\${DATABASE_URL}
      - REDIS_URL=\${REDIS_URL}
      - LOG_LEVEL=\${LOG_LEVEL:-INFO}
    ports:
      - "\${BACKEND_PORT:-8000}:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      replicas: \${BACKEND_REPLICAS:-2}
      resources:
        limits:
          memory: \${BACKEND_MEMORY_LIMIT:-1G}
          cpus: '\${BACKEND_CPU_LIMIT:-0.5}'
        reservations:
          memory: \${BACKEND_MEMORY_RESERVE:-512M}
          cpus: '\${BACKEND_CPU_RESERVE:-0.25}'

  frontend:
    image: ${frontend_image}
    environment:
      - REACT_APP_API_URL=\${BACKEND_URL:-http://localhost:8000}
      - REACT_APP_WS_URL=\${WEBSOCKET_URL:-ws://localhost:8000}
    ports:
      - "\${FRONTEND_PORT:-3000}:3000"
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    deploy:
      replicas: \${FRONTEND_REPLICAS:-1}
      resources:
        limits:
          memory: \${FRONTEND_MEMORY_LIMIT:-512M}
          cpus: '\${FRONTEND_CPU_LIMIT:-0.25}'

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    driver: bridge
EOF

    log_success "Docker Compose configuration generated"
}

run_migrations() {
    log_info "Running database migrations..."
    
    local backend_image="${REGISTRY}/${IMAGE_PREFIX}-backend:${VERSION}"
    
    # Run migrations in a temporary container
    if docker run --rm --network host \
        -e DATABASE_URL="${DATABASE_URL}" \
        "$backend_image" \
        alembic upgrade head; then
        log_success "Database migrations completed"
    else
        log_error "Database migrations failed"
        exit 1
    fi
}

deploy_services() {
    log_info "Deploying services..."
    
    local compose_file="${PROJECT_ROOT}/docker-compose.${ENVIRONMENT}.yml"
    
    # Stop existing services gracefully
    if docker-compose -f "$compose_file" ps -q | grep -q .; then
        log_info "Stopping existing services..."
        docker-compose -f "$compose_file" down --timeout 30
    fi
    
    # Start new services
    if docker-compose -f "$compose_file" up -d; then
        log_success "Services deployed successfully"
    else
        log_error "Service deployment failed"
        exit 1
    fi
}

wait_for_health() {
    log_info "Waiting for services to become healthy..."
    
    local backend_url="${BACKEND_URL:-http://localhost:8000}"
    local frontend_url="${FRONTEND_URL:-http://localhost:3000}"
    local timeout=$HEALTH_CHECK_TIMEOUT
    local interval=5
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        local backend_healthy=false
        local frontend_healthy=false
        
        # Check backend health
        if curl -s -f "${backend_url}/health" > /dev/null 2>&1; then
            backend_healthy=true
        fi
        
        # Check frontend health
        if curl -s -f "${frontend_url}" > /dev/null 2>&1; then
            frontend_healthy=true
        fi
        
        if [ "$backend_healthy" = true ] && [ "$frontend_healthy" = true ]; then
            log_success "All services are healthy"
            return 0
        fi
        
        log_info "Waiting for services... (${elapsed}s/${timeout}s)"
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    
    log_error "Services failed to become healthy within ${timeout}s"
    return 1
}

run_smoke_tests() {
    log_info "Running smoke tests..."
    
    local backend_url="${BACKEND_URL:-http://localhost:8000}"
    local tests_passed=true
    
    # Test backend health endpoint
    if ! curl -s -f "${backend_url}/health" | jq -e '.status == "healthy"' > /dev/null; then
        log_error "Backend health check failed"
        tests_passed=false
    fi
    
    # Test backend API documentation
    if ! curl -s -f "${backend_url}/docs" > /dev/null; then
        log_error "Backend API documentation not accessible"
        tests_passed=false
    fi
    
    # Test OpenAPI specification
    if ! curl -s -f "${backend_url}/openapi.json" > /dev/null; then
        log_error "OpenAPI specification not accessible"
        tests_passed=false
    fi
    
    if [ "$tests_passed" = true ]; then
        log_success "Smoke tests passed"
        return 0
    else
        log_error "Smoke tests failed"
        return 1
    fi
}

rollback() {
    if [[ "$ROLLBACK_ON_FAILURE" != "true" ]]; then
        log_warning "Rollback disabled"
        return 0
    fi
    
    log_warning "Rolling back deployment..."
    
    local compose_file="${PROJECT_ROOT}/docker-compose.${ENVIRONMENT}.yml"
    
    # Stop current deployment
    docker-compose -f "$compose_file" down --timeout 30
    
    # Here you would typically:
    # 1. Restore previous container versions
    # 2. Restore database backup if needed
    # 3. Restart with previous configuration
    
    log_warning "Rollback completed - manual intervention may be required"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Remove temporary compose file if it exists
    local compose_file="${PROJECT_ROOT}/docker-compose.${ENVIRONMENT}.yml"
    if [[ -f "$compose_file" ]]; then
        rm "$compose_file"
    fi
    
    # Clean up old docker images
    docker image prune -f > /dev/null 2>&1 || true
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting T-Bot deployment to $ENVIRONMENT environment"
    log_info "Version: $VERSION"
    
    # Set trap for cleanup on exit
    trap cleanup EXIT
    
    # Deployment steps
    check_prerequisites
    load_environment
    backup_database
    pull_images
    generate_compose_file
    run_migrations
    deploy_services
    
    if wait_for_health && run_smoke_tests; then
        log_success "Deployment completed successfully!"
        
        # Send success notification
        log_info "Deployment Summary:"
        log_info "  Environment: $ENVIRONMENT"
        log_info "  Version: $VERSION"
        log_info "  Backend URL: ${BACKEND_URL:-http://localhost:8000}"
        log_info "  Frontend URL: ${FRONTEND_URL:-http://localhost:3000}"
    else
        log_error "Deployment failed health checks"
        rollback
        exit 1
    fi
}

# Help function
show_help() {
    cat << EOF
T-Bot Trading System Deployment Script

Usage: $0 [ENVIRONMENT] [VERSION]

ENVIRONMENT: Target environment (staging, production) [default: staging]
VERSION:     Container image version tag [default: latest]

Environment Variables:
  REGISTRY:                Container registry [default: ghcr.io]
  IMAGE_PREFIX:           Image name prefix [default: tbot]
  BACKUP_ENABLED:         Enable database backup [default: true]
  HEALTH_CHECK_TIMEOUT:   Health check timeout in seconds [default: 300]
  ROLLBACK_ON_FAILURE:    Enable automatic rollback [default: true]

Examples:
  $0 staging latest
  $0 production v1.2.3
  BACKUP_ENABLED=false $0 staging

EOF
}

# Script entry point
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

main "$@"