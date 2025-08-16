#!/bin/bash

# Pre-installation script to ensure system dependencies are ready
# This should be run before attempting to install TA-Lib

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_status() {
    echo -e "${CYAN}[INFO]${NC} $1"
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

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

# Check if running on WSL
check_wsl() {
    if grep -q microsoft /proc/version 2>/dev/null; then
        print_status "Running on WSL"
        return 0
    else
        print_status "Not running on WSL"
        return 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Checking system dependencies..."
    
    # Check if we have necessary build tools
    local missing_tools=()
    
    for tool in gcc g++ make wget curl; do
        if ! command -v $tool &> /dev/null; then
            missing_tools+=($tool)
        fi
    done
    
    if [[ ${#missing_tools[@]} -eq 0 ]]; then
        print_success "All build tools are installed"
        return 0
    fi
    
    print_warning "Missing tools: ${missing_tools[*]}"
    print_status "Installing build dependencies..."
    
    # Try to install with sudo if available
    if command -v sudo &> /dev/null; then
        print_info "This will require your sudo password..."
        sudo apt-get update
        sudo apt-get install -y build-essential wget curl
        print_success "Build dependencies installed"
    else
        print_error "Cannot install dependencies without sudo"
        print_info "Please install manually:"
        print_info "  apt-get update"
        print_info "  apt-get install -y build-essential wget curl"
        return 1
    fi
}

# Setup TA-Lib directories
setup_talib_dirs() {
    print_status "Setting up TA-Lib directories..."
    
    # Create installation directory
    mkdir -p ~/.nemobotter-external/talib
    
    # Ensure /usr/local directories exist
    if [[ -w /usr/local ]]; then
        mkdir -p /usr/local/lib
        mkdir -p /usr/local/include
    else
        print_warning "/usr/local is not writable, will need sudo for installation"
    fi
    
    print_success "Directories prepared"
}

# Check Python version
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        print_success "Python $python_version found"
        
        # Check if it's 3.10
        if [[ "$python_version" == 3.10* ]]; then
            print_success "Python 3.10 detected (recommended)"
        else
            print_warning "Python $python_version detected (3.10 recommended)"
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
    
    # Check virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_success "Virtual environment active: $VIRTUAL_ENV"
    elif [[ -d "$HOME/.venv" ]]; then
        print_warning "Virtual environment exists but not active"
        print_info "Activate with: source ~/.venv/bin/activate"
    else
        print_warning "No virtual environment found"
        print_info "Create with: python3 -m venv ~/.venv"
    fi
}

# Main function
main() {
    print_status "Running pre-installation checks..."
    
    # Check environment
    check_wsl || true
    
    # Check Python
    if ! check_python; then
        print_error "Python setup issues detected"
        exit 1
    fi
    
    # Install system dependencies
    if ! install_system_deps; then
        print_error "Failed to install system dependencies"
        exit 1
    fi
    
    # Setup directories
    setup_talib_dirs
    
    print_success "Pre-installation checks completed!"
    print_info "You can now proceed with: make setup"
}

# Run main function
main "$@"