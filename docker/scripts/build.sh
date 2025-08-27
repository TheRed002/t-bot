#!/bin/bash

# T-Bot Docker Build Script
# Institutional-grade automated container building with security scanning

set -euo pipefail

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/docker"

# Default values
ENVIRONMENT="${ENVIRONMENT:-development}"
VERSION="${VERSION:-latest}"
BUILD_DATE="${BUILD_DATE:-$(date -u +"%Y-%m-%dT%H:%M:%SZ")}"
VCS_REF="${VCS_REF:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"
BUILD_ARGS=""
PUSH_IMAGES="${PUSH_IMAGES:-false}"
REGISTRY="${REGISTRY:-tbot}"
PARALLEL_BUILD="${PARALLEL_BUILD:-true}"
SECURITY_SCAN="${SECURITY_SCAN:-true}"
CACHE_FROM="${CACHE_FROM:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $*${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $*${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ $*${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ $*${NC}"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [SERVICES...]

Build Docker containers for T-Bot trading system with institutional-grade security.

OPTIONS:
    -e, --environment ENV     Build environment (development|testing|production) [default: development]
    -v, --version VERSION     Version tag for images [default: latest]
    -r, --registry REGISTRY   Docker registry prefix [default: tbot]
    -p, --push               Push images to registry after build
    -s, --scan               Enable security scanning [default: true]
    -c, --cache              Use build cache [default: true]
    --parallel               Build services in parallel [default: true]
    --no-parallel           Build services sequentially
    --no-cache              Build without cache
    --no-scan               Disable security scanning
    -h, --help              Show this help message

SERVICES:
    If no services specified, all services will be built.
    Available services:
    - trading-engine         Core trading engine
    - backend               Web API backend
    - websocket             WebSocket service
    - workers               Background workers
    - frontend              React frontend

EXAMPLES:
    # Build all services for development
    $0 -e development

    # Build specific services for production with push
    $0 -e production -v v1.2.3 --push trading-engine backend

    # Build with custom registry
    $0 -r my-registry.com/tbot -v latest

    # Build without security scanning (not recommended for production)
    $0 --no-scan

ENVIRONMENT VARIABLES:
    DOCKER_BUILDKIT=1       Enable Docker BuildKit (recommended)
    BUILDKIT_PROGRESS       Set to 'plain' for verbose output
    BUILD_DATE             Build timestamp (auto-generated)
    VCS_REF                Git commit hash (auto-generated)

EOF
}

# Parse command line arguments
parse_args() {
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
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -p|--push)
                PUSH_IMAGES="true"
                shift
                ;;
            -s|--scan)
                SECURITY_SCAN="true"
                shift
                ;;
            --no-scan)
                SECURITY_SCAN="false"
                shift
                ;;
            -c|--cache)
                CACHE_FROM="true"
                shift
                ;;
            --no-cache)
                CACHE_FROM="false"
                shift
                ;;
            --parallel)
                PARALLEL_BUILD="true"
                shift
                ;;
            --no-parallel)
                PARALLEL_BUILD="false"
                shift
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
}

# Validate environment
validate_environment() {
    case "$ENVIRONMENT" in
        development|testing|production) ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: development, testing, production"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker BuildKit
    if [[ "${DOCKER_BUILDKIT:-}" != "1" ]]; then
        log_warning "DOCKER_BUILDKIT not enabled. Enabling..."
        export DOCKER_BUILDKIT=1
    fi
    
    # Check docker-compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed"
        exit 1
    fi
    
    # Check security scanning tools if enabled
    if [[ "$SECURITY_SCAN" == "true" ]]; then
        if ! command -v trivy &> /dev/null; then
            log_warning "Trivy not found. Installing..."
            install_trivy
        fi
    fi
    
    # Check Git for VCS_REF
    if command -v git &> /dev/null && git rev-parse --git-dir > /dev/null 2>&1; then
        VCS_REF=$(git rev-parse --short HEAD)
    else
        log_warning "Git not available or not in git repository. Using 'unknown' for VCS_REF"
    fi
    
    log_success "Prerequisites checked"
}

# Install security scanning tools
install_trivy() {
    log "Installing Trivy security scanner..."
    
    case "$(uname -s)" in
        Linux*)
            wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
            echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
            sudo apt-get update && sudo apt-get install trivy
            ;;
        Darwin*)
            brew install aquasecurity/trivy/trivy
            ;;
        *)
            log_error "Unsupported operating system for automatic Trivy installation"
            log_error "Please install Trivy manually: https://aquasecurity.github.io/trivy/"
            exit 1
            ;;
    esac
    
    log_success "Trivy installed"
}

# Build common args
build_common_args() {
    BUILD_ARGS="--build-arg BUILD_DATE='${BUILD_DATE}'"
    BUILD_ARGS+=" --build-arg VCS_REF='${VCS_REF}'"
    BUILD_ARGS+=" --build-arg VERSION='${VERSION}'"
    
    if [[ "$CACHE_FROM" == "true" ]]; then
        BUILD_ARGS+=" --build-arg BUILDKIT_INLINE_CACHE=1"
    fi
    
    if [[ "$CACHE_FROM" == "false" ]]; then
        BUILD_ARGS+=" --no-cache"
    fi
}

# Security scan image
security_scan() {
    local image_name="$1"
    local service_name="$2"
    
    if [[ "$SECURITY_SCAN" != "true" ]]; then
        return 0
    fi
    
    log "Running security scan for ${service_name}..."
    
    # Create scan results directory
    mkdir -p "${PROJECT_ROOT}/reports/security/${service_name}"
    
    # Run Trivy scan
    local scan_output="${PROJECT_ROOT}/reports/security/${service_name}/trivy-$(date +%Y%m%d-%H%M%S).json"
    
    if trivy image --format json --output "${scan_output}" "${image_name}"; then
        # Check for HIGH and CRITICAL vulnerabilities
        local critical_count=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL") | .VulnerabilityID' "${scan_output}" | wc -l)
        local high_count=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH") | .VulnerabilityID' "${scan_output}" | wc -l)
        
        if [[ $critical_count -gt 0 ]]; then
            log_error "Found $critical_count CRITICAL vulnerabilities in ${service_name}"
            if [[ "$ENVIRONMENT" == "production" ]]; then
                log_error "Production builds cannot have CRITICAL vulnerabilities"
                return 1
            fi
        fi
        
        if [[ $high_count -gt 0 ]]; then
            log_warning "Found $high_count HIGH vulnerabilities in ${service_name}"
            if [[ "$ENVIRONMENT" == "production" && $high_count -gt 5 ]]; then
                log_error "Production builds cannot have more than 5 HIGH vulnerabilities"
                return 1
            fi
        fi
        
        log_success "Security scan completed for ${service_name}"
        log "Scan results: ${scan_output}"
    else
        log_error "Security scan failed for ${service_name}"
        return 1
    fi
}

# Build single service
build_service() {
    local service="$1"
    local dockerfile=""
    local context="${PROJECT_ROOT}"
    local target="$ENVIRONMENT"
    local image_name=""
    
    case "$service" in
        trading-engine)
            dockerfile="${DOCKER_DIR}/Dockerfile.trading-engine"
            image_name="${REGISTRY}/trading-engine:${VERSION}"
            ;;
        backend)
            dockerfile="${DOCKER_DIR}/Dockerfile.backend"
            image_name="${REGISTRY}/backend:${VERSION}"
            ;;
        websocket)
            dockerfile="${DOCKER_DIR}/Dockerfile.websocket"
            image_name="${REGISTRY}/websocket:${VERSION}"
            ;;
        workers)
            dockerfile="${DOCKER_DIR}/Dockerfile.workers"
            image_name="${REGISTRY}/workers:${VERSION}"
            ;;
        frontend)
            dockerfile="${DOCKER_DIR}/Dockerfile.frontend"
            image_name="${REGISTRY}/frontend:${VERSION}"
            ;;
        *)
            log_error "Unknown service: $service"
            return 1
            ;;
    esac
    
    log "Building ${service} (target: ${target})..."
    
    # Build command
    local build_cmd="docker build"
    build_cmd+=" --file '${dockerfile}'"
    build_cmd+=" --target '${target}'"
    build_cmd+=" --tag '${image_name}'"
    build_cmd+=" --tag '${REGISTRY}/${service}:latest'"
    build_cmd+=" ${BUILD_ARGS}"
    build_cmd+=" '${context}'"
    
    # Execute build
    if eval "${build_cmd}"; then
        log_success "Built ${service} successfully"
        
        # Security scan
        if ! security_scan "${image_name}" "${service}"; then
            return 1
        fi
        
        # Push if requested
        if [[ "$PUSH_IMAGES" == "true" ]]; then
            log "Pushing ${image_name}..."
            if docker push "${image_name}" && docker push "${REGISTRY}/${service}:latest"; then
                log_success "Pushed ${service} successfully"
            else
                log_error "Failed to push ${service}"
                return 1
            fi
        fi
    else
        log_error "Failed to build ${service}"
        return 1
    fi
}

# Build all services
build_services() {
    local services_to_build=("${SERVICES[@]:-trading-engine backend websocket workers frontend}")
    local build_pids=()
    local failed_services=()
    
    build_common_args
    
    log "Building services for environment: ${ENVIRONMENT}"
    log "Version: ${VERSION}"
    log "Registry: ${REGISTRY}"
    log "Services: ${services_to_build[*]}"
    
    if [[ "$PARALLEL_BUILD" == "true" ]] && [[ ${#services_to_build[@]} -gt 1 ]]; then
        log "Building services in parallel..."
        
        # Start builds in background
        for service in "${services_to_build[@]}"; do
            (
                if build_service "$service"; then
                    echo "SUCCESS:$service"
                else
                    echo "FAILED:$service"
                    exit 1
                fi
            ) &
            build_pids+=($!)
        done
        
        # Wait for all builds to complete
        for pid in "${build_pids[@]}"; do
            if ! wait "$pid"; then
                failed_services+=("PID:$pid")
            fi
        done
    else
        log "Building services sequentially..."
        
        for service in "${services_to_build[@]}"; do
            if ! build_service "$service"; then
                failed_services+=("$service")
            fi
        done
    fi
    
    # Report results
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log_success "All services built successfully!"
        
        # Generate build report
        generate_build_report "${services_to_build[@]}"
        
        return 0
    else
        log_error "Failed to build services: ${failed_services[*]}"
        return 1
    fi
}

# Generate build report
generate_build_report() {
    local services=("$@")
    local report_file="${PROJECT_ROOT}/reports/build/build-report-$(date +%Y%m%d-%H%M%S).json"
    
    mkdir -p "$(dirname "$report_file")"
    
    cat > "$report_file" << EOF
{
  "build_info": {
    "timestamp": "${BUILD_DATE}",
    "environment": "${ENVIRONMENT}",
    "version": "${VERSION}",
    "vcs_ref": "${VCS_REF}",
    "registry": "${REGISTRY}",
    "builder": "$(whoami)",
    "host": "$(hostname)"
  },
  "services": [
$(for service in "${services[@]}"; do
    echo "    {"
    echo "      \"name\": \"$service\","
    echo "      \"image\": \"${REGISTRY}/${service}:${VERSION}\","
    echo "      \"size\": \"$(docker image inspect "${REGISTRY}/${service}:${VERSION}" --format='{{.Size}}' 2>/dev/null || echo 'unknown')\","
    echo "      \"layers\": \"$(docker image inspect "${REGISTRY}/${service}:${VERSION}" --format='{{len .RootFS.Layers}}' 2>/dev/null || echo 'unknown')\""
    echo "    }$([ "$service" != "${services[-1]}" ] && echo ",")"
done)
  ],
  "security": {
    "scan_enabled": ${SECURITY_SCAN},
    "scan_results_dir": "reports/security/"
  }
}
EOF
    
    log_success "Build report generated: $report_file"
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    # Remove dangling images
    if docker images -f "dangling=true" -q | grep -q .; then
        log "Removing dangling images..."
        docker rmi $(docker images -f "dangling=true" -q) 2>/dev/null || true
    fi
    
    # Prune build cache if requested
    if [[ "${CLEANUP_CACHE:-false}" == "true" ]]; then
        log "Pruning build cache..."
        docker builder prune -f
    fi
}

# Main function
main() {
    local SERVICES=()
    
    # Handle signals
    trap cleanup EXIT
    trap 'log_error "Build interrupted"; exit 1' INT TERM
    
    # Parse arguments
    parse_args "$@"
    
    # Validate inputs
    validate_environment
    
    # Check prerequisites
    check_prerequisites
    
    # Create reports directory
    mkdir -p "${PROJECT_ROOT}/reports"/{build,security}
    
    # Build services
    if build_services; then
        log_success "Build completed successfully!"
        exit 0
    else
        log_error "Build failed!"
        exit 1
    fi
}

# Run main function
main "$@"