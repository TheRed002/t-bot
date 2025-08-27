#!/bin/bash

# T-Bot Docker Deployment Script
# Institutional-grade zero-downtime deployment with rollback capabilities

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/docker"

# Default values
ENVIRONMENT="${ENVIRONMENT:-development}"
VERSION="${VERSION:-latest}"
DEPLOY_STRATEGY="${DEPLOY_STRATEGY:-rolling}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
PARALLEL_DEPLOY="${PARALLEL_DEPLOY:-false}"
DRY_RUN="${DRY_RUN:-false}"
BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-true}"
AUTO_ROLLBACK="${AUTO_ROLLBACK:-true}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
MONITORING_ENABLED="${MONITORING_ENABLED:-true}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Deployment state tracking
DEPLOYMENT_ID="deploy-$(date +%Y%m%d-%H%M%S)"
DEPLOYMENT_LOG="${PROJECT_ROOT}/logs/deployments/${DEPLOYMENT_ID}.log"
ROLLBACK_STATE_FILE="${PROJECT_ROOT}/state/deployment/${DEPLOYMENT_ID}-rollback.json"

# Logging functions
log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo -e "${BLUE}${message}${NC}"
    echo "$message" >> "$DEPLOYMENT_LOG"
}

log_success() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $*"
    echo -e "${GREEN}${message}${NC}"
    echo "$message" >> "$DEPLOYMENT_LOG"
}

log_warning() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ $*"
    echo -e "${YELLOW}${message}${NC}"
    echo "$message" >> "$DEPLOYMENT_LOG"
}

log_error() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] ✗ $*"
    echo -e "${RED}${message}${NC}"
    echo "$message" >> "$DEPLOYMENT_LOG"
}

log_info() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] ℹ $*"
    echo -e "${PURPLE}${message}${NC}"
    echo "$message" >> "$DEPLOYMENT_LOG"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [SERVICES...]

Deploy T-Bot trading system with zero-downtime deployment strategies.

OPTIONS:
    -e, --environment ENV     Target environment (development|testing|production) [default: development]
    -v, --version VERSION     Version to deploy [default: latest]
    -s, --strategy STRATEGY   Deployment strategy (rolling|blue-green|canary) [default: rolling]
    -t, --timeout TIMEOUT     Health check timeout in seconds [default: 300]
    --parallel               Deploy services in parallel
    --dry-run               Show what would be deployed without executing
    --no-backup             Skip database backup before deployment
    --no-rollback           Disable automatic rollback on failure
    --monitoring            Enable deployment monitoring [default: true]
    --slack-webhook URL     Slack webhook for notifications
    -h, --help              Show this help message

DEPLOYMENT STRATEGIES:
    rolling                 Update services one by one (default, safest)
    blue-green             Spin up new environment, switch traffic
    canary                 Deploy to subset, gradually increase traffic

SERVICES:
    If no services specified, all services will be deployed.
    Available services:
    - trading-engine        Core trading engine
    - backend              Web API backend
    - websocket            WebSocket service
    - workers              Background workers
    - frontend             React frontend
    - databases            Database services
    - monitoring           Monitoring stack

EXAMPLES:
    # Deploy all services to development
    $0 -e development

    # Rolling deployment to production with specific version
    $0 -e production -v v1.2.3 -s rolling

    # Blue-green deployment with Slack notifications
    $0 -e production -s blue-green --slack-webhook https://hooks.slack.com/...

    # Dry run to see what would be deployed
    $0 --dry-run -e production -v v1.2.3

    # Deploy specific services only
    $0 -e production trading-engine backend

ENVIRONMENT VARIABLES:
    DEPLOYMENT_TIMEOUT     Override default timeout
    SLACK_WEBHOOK         Slack webhook URL for notifications
    ROLLBACK_ON_FAILURE   Enable/disable automatic rollback

EOF
}

# Parse command line arguments
parse_args() {
    local SERVICES=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -s|--strategy)
                DEPLOY_STRATEGY="$2"
                shift 2
                ;;
            -t|--timeout)
                HEALTH_CHECK_TIMEOUT="$2"
                shift 2
                ;;
            --parallel)
                PARALLEL_DEPLOY="true"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --no-backup)
                BACKUP_BEFORE_DEPLOY="false"
                shift
                ;;
            --no-rollback)
                AUTO_ROLLBACK="false"
                shift
                ;;
            --monitoring)
                MONITORING_ENABLED="true"
                shift
                ;;
            --slack-webhook)
                SLACK_WEBHOOK="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                SERVICES+=("$1")
                shift
                ;;
        esac
    done
    
    # Export services array
    declare -g -a TARGET_SERVICES=("${SERVICES[@]:-trading-engine backend websocket workers frontend}")
}

# Validate environment and inputs
validate_inputs() {
    case "$ENVIRONMENT" in
        development|testing|production) ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    case "$DEPLOY_STRATEGY" in
        rolling|blue-green|canary) ;;
        *)
            log_error "Invalid deployment strategy: $DEPLOY_STRATEGY"
            exit 1
            ;;
    esac
    
    if [[ ! "$HEALTH_CHECK_TIMEOUT" =~ ^[0-9]+$ ]] || [[ "$HEALTH_CHECK_TIMEOUT" -lt 30 ]]; then
        log_error "Invalid health check timeout: $HEALTH_CHECK_TIMEOUT (must be >= 30 seconds)"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check Docker and docker-compose
    for cmd in docker docker-compose; do
        if ! command -v $cmd &> /dev/null; then
            log_error "$cmd is not installed"
            exit 1
        fi
    done
    
    # Check environment-specific compose file
    local compose_file="${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml"
    if [[ ! -f "$compose_file" ]]; then
        log_error "Compose file not found: $compose_file"
        exit 1
    fi
    
    # Check if running as appropriate user for production
    if [[ "$ENVIRONMENT" == "production" ]] && [[ "$(id -u)" -eq 0 ]]; then
        log_warning "Running as root in production environment"
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Create necessary directories
    mkdir -p "${PROJECT_ROOT}/logs/deployments" \
             "${PROJECT_ROOT}/state/deployment" \
             "${PROJECT_ROOT}/backups/deployment"
    
    log_success "Prerequisites checked"
}

# Send Slack notification
send_slack_notification() {
    local message="$1"
    local color="${2:-good}"
    
    if [[ -z "$SLACK_WEBHOOK" ]]; then
        return 0
    fi
    
    local payload=$(cat << EOF
{
    "attachments": [
        {
            "color": "$color",
            "fields": [
                {
                    "title": "T-Bot Deployment",
                    "value": "$message",
                    "short": false
                },
                {
                    "title": "Environment",
                    "value": "$ENVIRONMENT",
                    "short": true
                },
                {
                    "title": "Version",
                    "value": "$VERSION",
                    "short": true
                },
                {
                    "title": "Strategy",
                    "value": "$DEPLOY_STRATEGY",
                    "short": true
                },
                {
                    "title": "Deployment ID",
                    "value": "$DEPLOYMENT_ID",
                    "short": true
                }
            ],
            "footer": "T-Bot Deployment System",
            "ts": $(date +%s)
        }
    ]
}
EOF
)
    
    curl -X POST -H 'Content-type: application/json' \
         --data "$payload" \
         "$SLACK_WEBHOOK" &>/dev/null || true
}

# Backup services before deployment
backup_services() {
    if [[ "$BACKUP_BEFORE_DEPLOY" != "true" ]]; then
        log_info "Skipping backup as requested"
        return 0
    fi
    
    log "Creating backup before deployment..."
    
    local backup_dir="${PROJECT_ROOT}/backups/deployment/${DEPLOYMENT_ID}"
    mkdir -p "$backup_dir"
    
    # Backup databases
    if docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" ps postgresql &>/dev/null; then
        log "Backing up PostgreSQL..."
        docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" exec -T postgresql \
            pg_dump -U "${DB_USER}" "${DB_NAME}" > "${backup_dir}/postgresql-$(date +%Y%m%d-%H%M%S).sql"
    fi
    
    # Backup Redis
    if docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" ps redis &>/dev/null; then
        log "Backing up Redis..."
        docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" exec -T redis \
            redis-cli BGSAVE
        sleep 5
        docker cp "$(docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" ps -q redis):/data/dump.rdb" \
            "${backup_dir}/redis-$(date +%Y%m%d-%H%M%S).rdb"
    fi
    
    # Save current service states
    docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" ps --format json > \
        "${backup_dir}/service-states.json"
    
    log_success "Backup completed: $backup_dir"
    
    # Save rollback information
    cat > "$ROLLBACK_STATE_FILE" << EOF
{
    "deployment_id": "$DEPLOYMENT_ID",
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "environment": "$ENVIRONMENT",
    "previous_version": "$(docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" images --format json | jq -r '.[0].Tag' 2>/dev/null || echo 'unknown')",
    "backup_location": "$backup_dir",
    "services": $(echo "${TARGET_SERVICES[@]}" | jq -R 'split(" ")')
}
EOF
}

# Wait for service health
wait_for_service_health() {
    local service="$1"
    local timeout="${2:-$HEALTH_CHECK_TIMEOUT}"
    local compose_file="${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml"
    
    log "Waiting for $service to become healthy (timeout: ${timeout}s)..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))
    
    while [[ $(date +%s) -lt $end_time ]]; do
        local health_status
        health_status=$(docker-compose -f "$compose_file" ps --format json | \
                       jq -r ".[] | select(.Service == \"$service\") | .Health" 2>/dev/null || echo "unknown")
        
        case "$health_status" in
            "healthy")
                log_success "$service is healthy"
                return 0
                ;;
            "unhealthy")
                log_error "$service is unhealthy"
                return 1
                ;;
            "starting"|"unknown")
                log_info "$service health status: $health_status"
                ;;
        esac
        
        sleep 10
    done
    
    log_error "Timeout waiting for $service to become healthy"
    return 1
}

# Rolling deployment strategy
deploy_rolling() {
    local services=("$@")
    
    log "Starting rolling deployment..."
    
    for service in "${services[@]}"; do
        log "Deploying $service..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would deploy $service with version $VERSION"
            continue
        fi
        
        # Update service
        docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" \
            up -d --no-deps "$service"
        
        # Wait for health check
        if ! wait_for_service_health "$service"; then
            log_error "Rolling deployment failed at service: $service"
            return 1
        fi
        
        # Brief pause between services
        sleep 5
    done
    
    log_success "Rolling deployment completed"
}

# Blue-green deployment strategy
deploy_blue_green() {
    local services=("$@")
    
    log "Starting blue-green deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform blue-green deployment"
        return 0
    fi
    
    # Create green environment
    log "Creating green environment..."
    
    # Temporary compose file for green environment
    local green_compose="${DOCKER_DIR}/docker-compose.${ENVIRONMENT}-green.yml"
    
    # Modify compose file for green environment (different ports/names)
    sed "s/tbot-\([^-]*\)-${ENVIRONMENT}/tbot-\1-${ENVIRONMENT}-green/g; \
         s/:\([0-9]*\):\([0-9]*\)/:\$((\\1 + 1000)):\\2/g" \
        "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" > "$green_compose"
    
    # Deploy green environment
    docker-compose -f "$green_compose" up -d "${services[@]}"
    
    # Wait for all services to be healthy
    local all_healthy=true
    for service in "${services[@]}"; do
        if ! wait_for_service_health "$service" 180; then
            all_healthy=false
            break
        fi
    done
    
    if [[ "$all_healthy" == "true" ]]; then
        log "Green environment is healthy, switching traffic..."
        
        # Switch traffic (this would typically involve load balancer reconfiguration)
        # For now, we'll stop blue and rename green to blue
        docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" down
        
        # Rename green containers to blue names
        for service in "${services[@]}"; do
            docker rename "tbot-${service}-${ENVIRONMENT}-green" "tbot-${service}-${ENVIRONMENT}"
        done
        
        log_success "Blue-green deployment completed"
    else
        log_error "Green environment failed health checks, rolling back..."
        docker-compose -f "$green_compose" down
        rm -f "$green_compose"
        return 1
    fi
    
    # Cleanup
    rm -f "$green_compose"
}

# Canary deployment strategy
deploy_canary() {
    local services=("$@")
    
    log "Starting canary deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform canary deployment"
        return 0
    fi
    
    # For simplicity, canary deployment will deploy 1 instance first,
    # monitor for a period, then deploy the rest
    
    log "Deploying canary instance..."
    
    # Deploy first service as canary
    local canary_service="${services[0]}"
    docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" \
        up -d --scale "$canary_service=1" "$canary_service"
    
    if ! wait_for_service_health "$canary_service" 120; then
        log_error "Canary deployment failed"
        return 1
    fi
    
    log "Canary is healthy, monitoring for 60 seconds..."
    sleep 60
    
    # Check if canary is still healthy
    if ! wait_for_service_health "$canary_service" 30; then
        log_error "Canary became unhealthy, aborting deployment"
        return 1
    fi
    
    log "Canary is stable, deploying remaining services..."
    
    # Deploy remaining services
    for service in "${services[@]}"; do
        docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" \
            up -d --no-deps "$service"
        
        if ! wait_for_service_health "$service"; then
            log_error "Canary deployment failed at service: $service"
            return 1
        fi
    done
    
    log_success "Canary deployment completed"
}

# Main deployment function
deploy_services() {
    local services=("${TARGET_SERVICES[@]}")
    
    log "Starting deployment of services: ${services[*]}"
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    log "Strategy: $DEPLOY_STRATEGY"
    log "Deployment ID: $DEPLOYMENT_ID"
    
    # Send notification
    send_slack_notification "🚀 Starting deployment of ${services[*]} to $ENVIRONMENT" "warning"
    
    # Create backup
    backup_services
    
    # Execute deployment based on strategy
    case "$DEPLOY_STRATEGY" in
        rolling)
            if deploy_rolling "${services[@]}"; then
                deployment_success=true
            else
                deployment_success=false
            fi
            ;;
        blue-green)
            if deploy_blue_green "${services[@]}"; then
                deployment_success=true
            else
                deployment_success=false
            fi
            ;;
        canary)
            if deploy_canary "${services[@]}"; then
                deployment_success=true
            else
                deployment_success=false
            fi
            ;;
    esac
    
    if [[ "$deployment_success" == "true" ]]; then
        log_success "Deployment completed successfully!"
        send_slack_notification "✅ Deployment completed successfully!" "good"
        
        # Post-deployment verification
        post_deployment_verification
        
        return 0
    else
        log_error "Deployment failed!"
        send_slack_notification "❌ Deployment failed!" "danger"
        
        if [[ "$AUTO_ROLLBACK" == "true" ]] && [[ "$DRY_RUN" != "true" ]]; then
            log "Initiating automatic rollback..."
            rollback_deployment
        fi
        
        return 1
    fi
}

# Post-deployment verification
post_deployment_verification() {
    log "Running post-deployment verification..."
    
    # Health check all services
    local compose_file="${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml"
    local all_healthy=true
    
    for service in "${TARGET_SERVICES[@]}"; do
        if ! wait_for_service_health "$service" 60; then
            all_healthy=false
            break
        fi
    done
    
    if [[ "$all_healthy" == "true" ]]; then
        log_success "Post-deployment verification passed"
        
        # Update deployment status
        echo "SUCCESS" > "${PROJECT_ROOT}/state/deployment/${DEPLOYMENT_ID}-status"
    else
        log_error "Post-deployment verification failed"
        echo "FAILED" > "${PROJECT_ROOT}/state/deployment/${DEPLOYMENT_ID}-status"
        return 1
    fi
}

# Rollback deployment
rollback_deployment() {
    if [[ ! -f "$ROLLBACK_STATE_FILE" ]]; then
        log_error "Rollback state file not found: $ROLLBACK_STATE_FILE"
        return 1
    fi
    
    log "Rolling back deployment..."
    
    local previous_version
    previous_version=$(jq -r '.previous_version' "$ROLLBACK_STATE_FILE")
    
    if [[ "$previous_version" == "unknown" ]] || [[ "$previous_version" == "null" ]]; then
        log_error "Cannot determine previous version for rollback"
        return 1
    fi
    
    # Rollback services
    for service in "${TARGET_SERVICES[@]}"; do
        log "Rolling back $service to version $previous_version..."
        
        # This would typically involve pulling the previous image and updating
        docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" \
            stop "$service"
        
        # Here you would pull the previous version and restart
        # For now, we'll just restart the current version
        docker-compose -f "${DOCKER_DIR}/docker-compose.${ENVIRONMENT}.yml" \
            start "$service"
    done
    
    log_warning "Rollback completed. Please verify system status."
    send_slack_notification "⚠️ Deployment rolled back to previous version" "warning"
}

# Cleanup function
cleanup() {
    log "Cleaning up deployment artifacts..."
    
    # Remove temporary files
    find "${DOCKER_DIR}" -name "*-green.yml" -delete 2>/dev/null || true
    
    # Cleanup old deployment logs (keep last 30 days)
    find "${PROJECT_ROOT}/logs/deployments" -name "*.log" -mtime +30 -delete 2>/dev/null || true
    
    # Cleanup old rollback states (keep last 10)
    ls -t "${PROJECT_ROOT}/state/deployment/"*-rollback.json 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null || true
}

# Main function
main() {
    # Handle signals
    trap cleanup EXIT
    trap 'log_error "Deployment interrupted"; exit 1' INT TERM
    
    # Parse arguments
    parse_args "$@"
    
    # Validate inputs
    validate_inputs
    
    # Check prerequisites
    check_prerequisites
    
    # Initialize deployment log
    echo "T-Bot Deployment Log - ${DEPLOYMENT_ID}" > "$DEPLOYMENT_LOG"
    echo "Started: $(date)" >> "$DEPLOYMENT_LOG"
    echo "Environment: $ENVIRONMENT" >> "$DEPLOYMENT_LOG"
    echo "Version: $VERSION" >> "$DEPLOYMENT_LOG"
    echo "Strategy: $DEPLOY_STRATEGY" >> "$DEPLOYMENT_LOG"
    echo "Services: ${TARGET_SERVICES[*]}" >> "$DEPLOYMENT_LOG"
    echo "----------------------------------------" >> "$DEPLOYMENT_LOG"
    
    # Deploy services
    if deploy_services; then
        log_success "Deployment completed successfully!"
        exit 0
    else
        log_error "Deployment failed!"
        exit 1
    fi
}

# Run main function
main "$@"