#!/bin/bash

# T-Bot Requirements Installation Script
# Ensures all dependencies are installed in the correct order

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    echo -e "${CYAN}ℹ${NC} $1"
}

# Check if virtual environment is activated
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "Virtual environment not activated"
        if [[ -d "$HOME/.venv" ]]; then
            print_status "Activating virtual environment at ~/.venv"
            source ~/.venv/bin/activate
        else
            print_error "Virtual environment not found at ~/.venv"
            print_info "Please create it first: python3.10 -m venv ~/.venv"
            exit 1
        fi
    else
        print_success "Virtual environment is active: $VIRTUAL_ENV"
    fi
}

# Install TA-Lib C library if needed
install_talib_c() {
    print_status "Checking TA-Lib C library..."
    
    # Check if TA-Lib is already installed
    if ldconfig -p 2>/dev/null | grep -q libta_lib; then
        print_success "TA-Lib C library is already installed"
        return 0
    fi
    
    print_warning "TA-Lib C library not found, installing..."
    
    # Run the TA-Lib installation script
    if bash "$SCRIPT_DIR/talib.sh" install; then
        print_success "TA-Lib C library installed successfully"
        
        # Ensure library is in path
        export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
        
        # Verify installation
        if ldconfig -p 2>/dev/null | grep -q libta_lib; then
            print_success "TA-Lib C library verified"
            return 0
        else
            print_warning "TA-Lib C library installed but not in ldconfig cache"
            print_info "You may need to run: sudo ldconfig"
            # Don't fail here, as the library might still work
            return 0
        fi
    else
        print_error "Failed to install TA-Lib C library"
        print_info "Please install manually:"
        print_info "  1. cd ~/.nemobotter-external/talib/ta-lib-0.6.4"
        print_info "  2. sudo make install"
        print_info "  3. sudo ldconfig"
        return 1
    fi
}

# Install Python requirements
install_python_requirements() {
    print_status "Installing Python requirements..."
    
    # Upgrade pip first
    print_status "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel
    
    # Install numpy first (required by TA-Lib)
    print_status "Installing numpy (required for TA-Lib)..."
    pip install "numpy>=1.26.4"
    
    # Install TA-Lib Python package
    print_status "Installing TA-Lib Python package..."
    
    # Set environment variables for TA-Lib
    export TA_LIBRARY_PATH=/usr/local/lib
    export TA_INCLUDE_PATH=/usr/local/include
    
    if pip install --no-cache-dir TA-Lib; then
        print_success "TA-Lib Python package installed successfully"
    else
        print_error "Failed to install TA-Lib Python package"
        print_info "Trying alternative installation method..."
        
        # Try with explicit paths
        if pip install --no-cache-dir --global-option=build_ext --global-option="-L/usr/local/lib" --global-option="-I/usr/local/include" TA-Lib; then
            print_success "TA-Lib Python package installed with explicit paths"
        else
            print_error "Failed to install TA-Lib Python package"
            return 1
        fi
    fi
    
    # Install other requirements
    print_status "Installing remaining requirements..."
    
    # We should be in the project root when called from Makefile
    if [[ ! -f "requirements.txt" ]]; then
        print_error "Cannot find requirements.txt in current directory: $(pwd)"
        print_info "Please run this script from the project root directory"
        return 1
    fi
    
    print_status "Installing from requirements.txt..."
    
    # Install requirements without TA-Lib first (to avoid conflicts)
    grep -v "TA-Lib" requirements.txt > /tmp/requirements_no_talib.txt 2>/dev/null || true
    
    if [[ -f "/tmp/requirements_no_talib.txt" ]] && [[ -s "/tmp/requirements_no_talib.txt" ]]; then
        pip install -r /tmp/requirements_no_talib.txt
        rm /tmp/requirements_no_talib.txt
    else
        # If grep didn't work or file is empty, install the full requirements
        pip install -r requirements.txt
    fi
    
    print_success "All Python requirements installed"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test TA-Lib import
    if python -c "import talib; print(f'TA-Lib version: {talib.__version__}')" 2>/dev/null; then
        print_success "TA-Lib Python package is working"
    else
        print_error "TA-Lib Python package import failed"
        return 1
    fi
    
    # Test other critical imports
    local critical_packages=("pandas" "numpy" "ccxt" "fastapi" "sqlalchemy")
    local all_good=true
    
    for package in "${critical_packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            print_success "$package is installed"
        else
            print_error "$package import failed"
            all_good=false
        fi
    done
    
    if [[ "$all_good" == "true" ]]; then
        print_success "All critical packages verified"
        return 0
    else
        print_error "Some packages failed verification"
        return 1
    fi
}

# Main execution
main() {
    print_status "Starting T-Bot requirements installation..."
    
    # Check virtual environment
    check_venv
    
    # Install TA-Lib C library
    if ! install_talib_c; then
        print_error "Failed to install TA-Lib C library"
        exit 1
    fi
    
    # Install Python requirements
    if ! install_python_requirements; then
        print_error "Failed to install Python requirements"
        exit 1
    fi
    
    # Verify installation
    if verify_installation; then
        print_success "All requirements installed successfully!"
        print_info "You can now run: make run-mock"
    else
        print_warning "Installation completed with some issues"
        print_info "Please check the errors above"
    fi
}

# Run main function
main "$@"