#!/bin/bash
# Base template for external library installation scripts
# All external library scripts should source this file

# Common variables
LIBRARY_NAME=""
LIBRARY_VERSION=""
DOWNLOAD_URL=""
INSTALL_DIR="$HOME/.nemobotter-external"
VENV_PATH="$HOME/.venv"

# Debug flag - can be set via environment variable or command line
DEBUG_FLAG="${DEBUG:-false}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global flag to track if apt update has been run
APT_UPDATED=false

# Debug-aware print functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    if [[ "$DEBUG_FLAG" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Debug-aware command execution
run_command() {
    local cmd="$1"
    local description="$2"
    
    print_debug "Executing: $cmd"
    
    if [[ "$DEBUG_FLAG" == "true" ]]; then
        # Show full output in debug mode
        eval "$cmd"
        local exit_code=$?
    else
        # Suppress output in non-debug mode
        eval "$cmd" >/dev/null 2>&1
        local exit_code=$?
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "$description completed"
    else
        print_error "$description failed (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Debug-aware apt-get wrapper
apt_install() {
    local packages="$1"
    local description="${2:-Installing packages: $packages}"
    
    print_debug "Installing packages: $packages"
    
    if [[ "$DEBUG_FLAG" == "true" ]]; then
        sudo apt-get -y install $packages
        local exit_code=$?
    else
        sudo apt-get -y install $packages >/dev/null 2>&1
        local exit_code=$?
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "$description completed"
    else
        print_error "$description failed (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Debug-aware apt-get update wrapper
apt_update() {
    print_debug "Updating package list"
    
    if [[ "$DEBUG_FLAG" == "true" ]]; then
        sudo apt-get update
        local exit_code=$?
    else
        sudo apt-get update -qq
        local exit_code=$?
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "Package list updated"
    else
        print_error "Package list update failed (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Base methods that should be overridden by each library
check_installed() {
    print_error "check_installed() method not implemented for $LIBRARY_NAME"
    return 1
}

download() {
    print_error "download() method not implemented for $LIBRARY_NAME"
    return 1
}

install() {
    print_error "install() method not implemented for $LIBRARY_NAME"
    return 1
}

# Common utility methods
setup_environment() {
    print_status "Setting up environment for $LIBRARY_NAME..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Only run apt update once per session
    if [[ "$APT_UPDATED" == false ]]; then
        apt_update
        APT_UPDATED=true
    fi
    
    # Set environment variables
    export CUDA_HOME=/usr/local/cuda
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    
    print_success "Environment setup completed"
}

cleanup() {
    print_status "Cleaning up temporary files..."
    # Override in specific scripts if needed
}

# Main installation method
install_library() {
    print_status "Installing $LIBRARY_NAME $LIBRARY_VERSION..."
    
    # Setup environment
    setup_environment
    
    # Check if already installed
    if check_installed; then
        print_success "$LIBRARY_NAME is already installed and working"
        return 0
    fi
    
    # Download if needed
    if ! download; then
        print_error "Failed to download $LIBRARY_NAME"
        return 1
    fi
    
    # Install
    if ! install; then
        print_error "Failed to install $LIBRARY_NAME"
        cleanup
        return 1
    fi
    
    # Verify installation
    if check_installed; then
        print_success "$LIBRARY_NAME installed successfully"
        cleanup
        return 0
    else
        print_error "$LIBRARY_NAME installation verification failed"
        cleanup
        return 1
    fi
}

# Usage function
show_usage() {
    echo "Usage: $0 [check|download|install|cleanup|test] [--debug]"
    echo "  check    - Check if library is installed"
    echo "  download - Download library files"
    echo "  install  - Install library (includes download and install)"
    echo "  cleanup  - Clean up temporary files"
    echo "  test     - Run comprehensive tests on the library"
    echo "  --debug  - Enable debug mode for verbose output"
    echo ""
    echo "Environment variables:"
    echo "  DEBUG=true - Enable debug mode"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --debug)
                DEBUG_FLAG="true"
                shift
                ;;
            *)
                break
                ;;
        esac
    done
}

# Main execution
main() {
    # Parse debug flag first
    parse_args "$@"
    
    # Remove --debug from arguments
    local args=()
    for arg in "$@"; do
        if [[ "$arg" != "--debug" ]]; then
            args+=("$arg")
        fi
    done
    
    case "${args[0]:-install}" in
        "check")
            check_installed
            ;;
        "download")
            setup_environment
            download
            ;;
        "install")
            install_library
            ;;
        "cleanup")
            cleanup
            ;;
        "test")
            test_library
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown command: ${args[0]}"
            show_usage
            exit 1
            ;;
    esac
}

# Default test function (can be overridden by individual scripts)
test_library() {
    print_status "Testing $LIBRARY_NAME..."
    print_warning "No test implementation found for $LIBRARY_NAME"
    print_status "Individual scripts should override this function to implement specific tests"
    return 1
}

# Export functions for sourcing
export -f print_status print_success print_warning print_error print_debug
export -f run_command apt_install apt_update
export -f setup_environment cleanup install_library
export -f check_installed download install test_library main parse_args 