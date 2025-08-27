#!/bin/bash

# T-Bot Trading System Rollback Script
# This script handles rolling back to a previous deployment version

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-staging}"
TARGET_VERSION="${2:-}"
REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_PREFIX="${IMAGE_PREFIX:-tbot}"

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

# Get current deployment info
get_current_deployment() {
    log_info "Getting current deployment information..."
    
    local compose_file="${PROJECT_ROOT}/docker-compose.${ENVIRONMENT}.yml"
    if [[ ! -f "$compose_file" ]]; then
        log_error "No active deployment found for environment: $ENVIRONMENT"
        exit 1
    fi
    
    # Extract current image versions from compose file
    local current_backend=$(grep "image:" "$compose_file" | grep backend | awk -F':' '{print $NF}' | tr -d ' ')
    local current_frontend=$(grep "image:" "$compose_file" | grep frontend | awk -F':' '{print $NF}' | tr -d ' ')
    
    log_info "Current backend version: $current_backend"
    log_info "Current frontend version: $current_frontend"
    
    echo "$current_backend"
}

# List available versions
list_available_versions() {
    log_info "Listing available versions for rollback..."
    
    # This would typically query the container registry
    # For now, we'll create a simple version history
    local version_history=(
        "v1.2.3"
        "v1.2.2" 
        "v1.2.1"
        "v1.2.0"
        "v1.1.9"
    )
    
    echo "Available versions for rollback:"
    for i in "${!version_history[@]}"; do
        echo "  $((i+1)). ${version_history[i]}"
    done
    
    if [[ -z "$TARGET_VERSION" ]]; then
        read -p "Select version number (1-${#version_history[@]}): " selection
        if [[ "$selection" -ge 1 && "$selection" -le "${#version_history[@]}" ]]; then
            TARGET_VERSION="${version_history[$((selection-1))]}"
        else
            log_error "Invalid selection"
            exit 1
        fi
    fi
    
    log_info "Selected rollback version: $TARGET_VERSION"
}

# Backup current state
backup_current_state() {
    log_info "Creating backup of current state..."
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="${PROJECT_ROOT}/backups/rollback_${ENVIRONMENT}_${timestamp}"
    
    mkdir -p "$backup_dir"
    
    # Backup current compose file
    cp "${PROJECT_ROOT}/docker-compose.${ENVIRONMENT}.yml" "$backup_dir/"
    
    # Backup database if enabled
    if [[ "${BACKUP_ENABLED:-true}" == "true" ]]; then
        local backup_file="${backup_dir}/db_backup_${timestamp}.sql"
        
        # Load environment
        local env_file="${PROJECT_ROOT}/.env.${ENVIRONMENT}"
        if [[ -f "$env_file" ]]; then
            set -a
            source "$env_file"
            set +a
        fi
        
        # Create database backup
        docker run --rm --network host \
            -e PGPASSWORD="${DB_PASSWORD:-}" \
            postgres:15-alpine \
            pg_dump -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" \
            -U "${DB_USER:-tbot_user}" -d "${DB_NAME:-tbot}" \
            > "$backup_file" || log_warning "Database backup failed"
    fi
    
    log_success "Current state backed up to: $backup_dir"
}

# Perform rollback
perform_rollback() {
    log_info "Performing rollback to version: $TARGET_VERSION"
    
    # Load environment
    local env_file="${PROJECT_ROOT}/.env.${ENVIRONMENT}"
    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file not found: $env_file"
        exit 1
    fi
    
    set -a
    source "$env_file"
    set +a
    
    # Generate rollback compose file
    local compose_file="${PROJECT_ROOT}/docker-compose.${ENVIRONMENT}.yml"
    local backend_image="${REGISTRY}/${IMAGE_PREFIX}-backend:${TARGET_VERSION}"
    local frontend_image="${REGISTRY}/${IMAGE_PREFIX}-frontend:${TARGET_VERSION}"
    
    # Pull rollback images
    log_info "Pulling rollback images..."
    if ! docker pull "$backend_image" || ! docker pull "$frontend_image"; then
        log_error "Failed to pull rollback images"
        exit 1
    fi
    
    # Update compose file with rollback versions
    sed -i "s|${REGISTRY}/${IMAGE_PREFIX}-backend:.*|${backend_image}|g" "$compose_file"
    sed -i "s|${REGISTRY}/${IMAGE_PREFIX}-frontend:.*|${frontend_image}|g" "$compose_file"
    
    # Stop current services
    log_info "Stopping current services..."
    docker-compose -f "$compose_file" down --timeout 30
    
    # Start rollback services
    log_info "Starting rollback services..."
    docker-compose -f "$compose_file" up -d
    
    # Wait for services to be healthy
    local backend_url="${BACKEND_URL:-http://localhost:8000}"
    local timeout=180
    local interval=5
    local elapsed=0
    
    log_info "Waiting for services to become healthy..."
    while [ $elapsed -lt $timeout ]; do
        if curl -s -f "${backend_url}/health" > /dev/null 2>&1; then
            log_success "Rollback services are healthy"
            return 0
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        log_info "Waiting... (${elapsed}s/${timeout}s)"
    done
    
    log_error "Rollback services failed to become healthy"
    return 1
}

# Verify rollback
verify_rollback() {
    log_info "Verifying rollback deployment..."
    
    local backend_url="${BACKEND_URL:-http://localhost:8000}"
    local frontend_url="${FRONTEND_URL:-http://localhost:3000}"
    
    # Test backend
    if ! curl -s -f "${backend_url}/health" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
        log_error "Backend health check failed"
        return 1
    fi
    
    # Test frontend
    if ! curl -s -f "${frontend_url}" > /dev/null 2>&1; then
        log_error "Frontend health check failed"
        return 1
    fi
    
    # Test API endpoints
    if ! curl -s -f "${backend_url}/docs" > /dev/null 2>&1; then
        log_error "API documentation not accessible"
        return 1
    fi
    
    log_success "Rollback verification passed"
    return 0
}

# Emergency rollback (faster, less safe)
emergency_rollback() {
    log_warning "Performing EMERGENCY rollback - this may cause data loss!"
    
    read -p "Are you sure you want to continue? (yes/no): " confirmation
    if [[ "$confirmation" != "yes" ]]; then
        log_info "Emergency rollback cancelled"
        exit 0
    fi
    
    # Skip backups and safety checks for speed
    BACKUP_ENABLED=false
    
    perform_rollback
    
    if verify_rollback; then
        log_success "Emergency rollback completed successfully"
    else
        log_error "Emergency rollback failed - manual intervention required"
        exit 1
    fi
}

# Show rollback status
show_status() {
    log_info "Rollback Status for $ENVIRONMENT environment:"
    
    local compose_file="${PROJECT_ROOT}/docker-compose.${ENVIRONMENT}.yml"
    if [[ ! -f "$compose_file" ]]; then
        log_error "No active deployment found"
        exit 1
    fi
    
    # Show current versions
    log_info "Current deployment versions:"
    grep "image:" "$compose_file" | while read -r line; do
        echo "  $line"
    done
    
    # Show service status
    log_info "Service status:"
    docker-compose -f "$compose_file" ps
    
    # Show recent backups
    log_info "Recent backups:"
    if [[ -d "${PROJECT_ROOT}/backups" ]]; then
        ls -la "${PROJECT_ROOT}/backups" | grep "rollback_${ENVIRONMENT}" | head -5
    else
        echo "  No backups found"
    fi
}

# Help function
show_help() {
    cat << EOF
T-Bot Trading System Rollback Script

Usage: $0 [ENVIRONMENT] [VERSION] [OPTIONS]

Arguments:
  ENVIRONMENT: Target environment (staging, production) [default: staging]
  VERSION:     Target version to rollback to (optional - interactive selection)

Options:
  --emergency: Perform emergency rollback (faster, skips safety checks)
  --status:    Show current rollback status
  --list:      List available versions for rollback

Examples:
  $0 staging                    # Interactive rollback to staging
  $0 production v1.2.1          # Rollback production to specific version
  $0 staging --emergency        # Emergency rollback
  $0 staging --status           # Show rollback status

Environment Variables:
  REGISTRY:      Container registry [default: ghcr.io]
  IMAGE_PREFIX:  Image name prefix [default: tbot]
  BACKUP_ENABLED: Enable backups [default: true]

EOF
}

# Main function
main() {
    case "${1:-}" in
        "--help"|"-h")
            show_help
            exit 0
            ;;
        "--status")
            show_status
            exit 0
            ;;
        "--emergency")
            emergency_rollback
            exit 0
            ;;
        "--list")
            list_available_versions
            exit 0
            ;;
    esac
    
    log_info "Starting T-Bot rollback for $ENVIRONMENT environment"
    
    # Get current deployment info
    local current_version
    current_version=$(get_current_deployment)
    
    # List and select target version if not provided
    if [[ -z "$TARGET_VERSION" ]]; then
        list_available_versions
    fi
    
    # Confirm rollback
    log_warning "This will rollback from current version to: $TARGET_VERSION"
    read -p "Continue with rollback? (yes/no): " confirmation
    if [[ "$confirmation" != "yes" ]]; then
        log_info "Rollback cancelled"
        exit 0
    fi
    
    # Perform rollback
    backup_current_state
    
    if perform_rollback && verify_rollback; then
        log_success "Rollback completed successfully!"
        log_info "Rollback Summary:"
        log_info "  Environment: $ENVIRONMENT"
        log_info "  Previous Version: $current_version"
        log_info "  Current Version: $TARGET_VERSION"
    else
        log_error "Rollback failed - check logs and perform manual recovery"
        exit 1
    fi
}

# Script entry point
main "$@"