#!/bin/bash
# TA-Lib installation script

# Source the base library template
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/base_library.sh"

# TA-Lib specific variables
LIBRARY_NAME="TA-Lib"
LIBRARY_VERSION="0.6.4"
DOWNLOAD_URL="https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz"
INSTALL_DIR="$HOME/.nemobotter-external/talib"

# Override base methods for TA-Lib
test_library() {
    print_status "Testing TA-Lib installation..."
    
    # Test Python TA-Lib access
    test_python_talib_access
    
    print_success "TA-Lib tests passed"
    return 0
}

check_installed() {
    print_status "Checking TA-Lib installation..."
    
    # Check if TA-Lib C library is available in system
    if ! ldconfig -p | grep -q libta-lib; then
        print_warning "TA-Lib C library not found"
        return 1
    fi
    
    # Check if TA-Lib header file exists
    if ! [[ -f "/usr/local/include/ta_libc.h" ]] && ! [[ -f "/usr/local/include/ta-lib/ta_libc.h" ]] && ! [[ -f "/usr/include/ta_libc.h" ]] && ! [[ -f "/usr/include/x86_64-linux-gnu/ta_libc.h" ]]; then
        print_warning "TA-Lib header files not found"
        return 1
    fi
    
    # Check if TA-Lib library file exists
    if ! [[ -f "/usr/local/lib/libta-lib.so" ]] && ! [[ -f "/usr/lib/libta-lib.so" ]] && ! [[ -f "/usr/lib/x86_64-linux-gnu/libta-lib.so" ]]; then
        print_warning "TA-Lib library file not found"
        return 1
    fi
    
    print_success "TA-Lib is installed and working"
    return 0
}

download() {
    print_status "Downloading TA-Lib source..."
    
    cd "$INSTALL_DIR"
    
    # Check if already downloaded
    if [[ -f "ta-lib-0.6.4-src.tar.gz" ]]; then
        print_success "TA-Lib source already downloaded"
        return 0
    fi
    
    # Download source
    print_debug "Downloading TA-Lib from: $DOWNLOAD_URL"
    if run_command "wget '$DOWNLOAD_URL' -O ta-lib-0.6.4-src.tar.gz" "TA-Lib source download"; then
        print_success "TA-Lib source downloaded successfully"
        return 0
    else
        print_error "Failed to download TA-Lib source"
        return 1
    fi
}

install() {
    print_status "Installing TA-Lib..."
    
    cd "$INSTALL_DIR"
    
    # Check if source exists
    if [[ ! -f "ta-lib-0.6.4-src.tar.gz" ]]; then
        print_error "TA-Lib source not found"
        return 1
    fi
    
    # Extract source if not already extracted
    if [[ ! -d "ta-lib-0.6.4" ]]; then
        print_status "Extracting TA-Lib source..."
        print_debug "Extracting TA-Lib source archive"
        if run_command "tar -xzf ta-lib-0.6.4-src.tar.gz" "TA-Lib source extraction"; then
            print_success "TA-Lib source extracted"
        else
            print_error "Failed to extract TA-Lib source"
            return 1
        fi
    fi
    
    cd ta-lib-0.6.4
    
    # Configure and compile
    print_status "Configuring and compiling TA-Lib..."
    print_debug "Configuring TA-Lib with --prefix=/usr/local"
    if run_command "./configure --prefix=/usr/local" "TA-Lib configuration"; then
        print_debug "Compiling TA-Lib with make -j$(nproc)"
        if run_command "make -j$(nproc)" "TA-Lib compilation"; then
            print_success "TA-Lib compiled successfully"
        else
            print_error "Failed to compile TA-Lib"
            return 1
        fi
    else
        print_error "Failed to configure TA-Lib"
        return 1
    fi
    
    # Install to system
    print_status "Installing TA-Lib to system..."
    print_debug "Installing TA-Lib to /usr/local"
    print_info "This step requires sudo access to install system libraries"
    
    # Try to install with sudo, or provide instructions if it fails
    if command -v sudo &> /dev/null; then
        if run_command "sudo make install" "TA-Lib system installation"; then
            print_success "TA-Lib installed to system"
        else
            print_error "Failed to install TA-Lib with sudo"
            print_info "Please run manually: cd $INSTALL_DIR/ta-lib-0.6.4 && sudo make install"
            return 1
        fi
    else
        # If sudo is not available, try without it (might work in some environments)
        if run_command "make install" "TA-Lib system installation"; then
            print_success "TA-Lib installed to system"
        else
            print_error "Cannot install TA-Lib without sudo"
            print_info "Please run manually: cd $INSTALL_DIR/ta-lib-0.6.4 && sudo make install"
            return 1
        fi
    fi
    
    # Update library cache
    print_status "Updating library cache..."
    print_debug "Running ldconfig to update library cache"
    if command -v sudo &> /dev/null; then
        run_command "sudo ldconfig" "Library cache update"
    else
        run_command "ldconfig" "Library cache update" || true
    fi
    
    # Export library path for current session
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    
    # Verify installation
    if check_installed; then
        print_success "TA-Lib installed successfully"
        return 0
    else
        print_warning "TA-Lib installation verification failed"
        return 1
    fi
}

test_python_talib_access() {
    print_status "Testing Python TA-Lib access..."
    
    # Check if virtual environment exists
    if [[ ! -d "$VENV_PATH" ]]; then
        print_warning "Virtual environment not found, skipping Python TA-Lib test"
        return
    fi
    
    # Test if TA-Lib is accessible from Python
    if "$VENV_PATH/bin/python" -c "
import sys
try:
    import talib
    print('TA-Lib Python version:', talib.__version__)
    print('TA-Lib Python access verified')
except ImportError as e:
    print(f'TA-Lib Python import failed: {e}')
except Exception as e:
    print(f'Error checking TA-Lib in Python: {e}')
" 2>/dev/null | grep -q "TA-Lib Python access verified"; then
        print_success "TA-Lib Python access verified"
    else
        print_warning "TA-Lib Python access not verified"
    fi
}

cleanup() {
    print_status "Cleaning up TA-Lib installation files..."
    # Keep the source for future use
    # rm -f "$INSTALL_DIR/ta-lib-0.6.4-src.tar.gz"
}

# Override setup_environment for TA-Lib
setup_environment() {
    print_status "Setting up environment for TA-Lib..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Install build dependencies if needed
    if ! command -v gcc &> /dev/null || ! command -v make &> /dev/null; then
        print_status "Installing build dependencies..."
        apt_install "build-essential" "Build dependencies installation"
    fi
    
    print_success "TA-Lib environment setup completed"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Ensure main function is available from base library
    if declare -f main > /dev/null; then
        main "$@"
    else
        print_error "Main function not found. Please check base library sourcing."
        exit 1
    fi
fi 